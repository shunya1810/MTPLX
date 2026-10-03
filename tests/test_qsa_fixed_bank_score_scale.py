"""The fixed-bank QSA selector, compiled against eager, at a near tie.

The compiled verifier records ``QSAIndexer._select_eager`` for the fixed QSA
bank of a promoted Flash-Next cache (``_call_rows`` in
mtplx/models/qwen4_exp.py): a block's score is the sum over indexer heads of
the positive query-key dots, divided by float32(sqrt(head_dim)), and the
query keeps the top ``block_topk`` blocks. At head_dim 128 that divisor has
to reach every compiled graph exactly. MLX 0.32.2 writes the scalar
constants of a fused kernel into its source with 7 significant digits, so
sqrt(128) = 11.3137083 would read back as 11.31371, and the distinct scores
1.5000001 / sqrt(128) and 1.5000002 / sqrt(128) would tie. The 1e-12 per-id
nudge rounds away at that magnitude, so the tie is broken by block id, and a
tie at the top-k cutoff selects a different block, which changes the
attention and the committed state.

The case below is built at the Flash-Next indexer geometry (4 heads of 128,
a 512-block budget, compress ratio 4), with bfloat16 queries and pooled keys
as served and 1,024 visible blocks: 511 blocks score well above the cut, two
blocks straddle it with exactly those pre-scale scores, the rest score below.
Whatever order a tie would take, one of the two placements changes the
selection.
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

from mtplx.models.qwen4_exp import QSAIndexer, TextArgs

HEADS = 4
HEAD_DIM = 128
RATIO = 4
BUDGET = 2048  # block_topk 512
VISIBLE = 1024
CAPACITY_BLOCKS = 1040
ROWS = 4  # a verify window
POS_START = VISIBLE * RATIO  # every row sees the 1,024 complete blocks

# The two scores at the cut, before the division: 1.5 + 2**-23 and
# 1.5 + 2**-22, exact sums of bfloat16 products.
LOW_TAIL = 2.0**-11
HIGH_TAIL = 2.0**-10


def _args() -> TextArgs:
    return TextArgs.from_dict(
        {
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "vocab_size": 128,
            "layer_types": ["full_attention"] * 2,
            "rope_parameters": {
                "partial_rotary_factor": 0.25,
                "rope_theta": 10000000,
                "rope_type": "default",
            },
            "indexer_n_heads": HEADS,
            "indexer_kv_heads": 1,
            "indexer_head_dim": HEAD_DIM,
            "indexer_budget": BUDGET,
            "indexer_compress_ratio": RATIO,
        }
    )


class _FixedBank:
    """What ``_select_eager`` reads from a promoted TensorOffsetQSACache."""

    fixed_capacity = True

    def __init__(self, pooled_t: mx.array, rows_gather: bool) -> None:
        self._pooled_t = pooled_t
        self.fixed_rows_gather = rows_gather
        self.raw_keys = mx.zeros((1, CAPACITY_BLOCKS * RATIO, 1), dtype=mx.bfloat16)

    def pooled_f32_view(self, nb: int) -> mx.array:
        return self._pooled_t[..., :nb]


def _near_tie(low_block: int, high_block: int):
    """Queries and pooled keys for the case above, bfloat16 as served."""

    q = np.zeros((1, ROWS, HEADS, HEAD_DIM), dtype=np.float32)
    q[:, :, 0, 0] = 1.5
    q[:, :, 0, 1] = 2.0**-12
    pooled = np.zeros((1, CAPACITY_BLOCKS, HEAD_DIM), dtype=np.float32)
    pooled[0, :VISIBLE, 0] = 0.5  # score 0.75: below the cut
    above = [b for b in range(VISIBLE) if b not in (low_block, high_block)][: 511]
    pooled[0, above, 0] = 2.0  # score 3.0: above it
    pooled[0, [low_block, high_block], 0] = 1.0
    pooled[0, low_block, 1] = LOW_TAIL
    pooled[0, high_block, 1] = HIGH_TAIL
    q = mx.array(q).astype(mx.bfloat16)
    pooled = mx.array(pooled).astype(mx.bfloat16)
    pooled_t = mx.swapaxes(pooled.astype(mx.float32), 1, 2)[:, None]
    mx.eval(q, pooled, pooled_t)
    return q, pooled, pooled_t


def _selected_blocks(mask: mx.array) -> list[set[int]]:
    tokens = np.array(mask[0, 0])[:, : VISIBLE * RATIO]
    per_block = tokens.reshape(ROWS, VISIBLE, RATIO).all(axis=-1)
    return [set(np.flatnonzero(row).tolist()) for row in per_block]


def test_the_two_scores_tie_only_under_a_seven_digit_divisor():
    exact = np.float32(math.sqrt(HEAD_DIM))
    printed = np.float32(float(f"{float(exact):.7g}"))
    assert printed != exact
    low = np.float32(1.5) + np.float32(2.0**-23)
    high = np.float32(1.5) + np.float32(2.0**-22)
    assert low / exact < high / exact
    assert low / printed == high / printed
    # The per-id nudge cannot separate them at this magnitude.
    nudged = high / printed - np.float32(VISIBLE) * np.float32(1e-12)
    assert nudged == high / printed


def _leaves(value) -> list[mx.array]:
    if isinstance(value, mx.array):
        return [value]
    if isinstance(value, (tuple, list)):
        return [leaf for item in value for leaf in _leaves(item)]
    return []


@pytest.mark.skipif(not mx.metal.is_available(), reason="the compiled verifier runs on the GPU")
@pytest.mark.parametrize("rows_gather", [False, True], ids=["dense-mask", "rows-gather"])
@pytest.mark.parametrize(("low_block", "high_block"), [(700, 300), (300, 700)])
def test_compiled_selection_matches_eager_at_the_cut(low_block, high_block, rows_gather):
    indexer = QSAIndexer(_args())
    assert indexer.block_topk == 512 and indexer.head_dim == HEAD_DIM
    q, pooled, pooled_t = _near_tie(low_block, high_block)
    pos_start = mx.array(POS_START, dtype=mx.int32)
    total = pos_start + ROWS

    def select(q, pooled, pooled_t, pos_start, total):
        bank = _FixedBank(pooled_t, rows_gather)
        return tuple(_leaves(indexer._select_eager(q, pos_start, bank, pooled, total)))

    with mx.stream(mx.gpu):
        eager = select(q, pooled, pooled_t, pos_start, total)
        compiled = mx.compile(select)(q, pooled, pooled_t, pos_start, total)
        mx.eval(eager, compiled)

    if not rows_gather:
        # The eager selector keeps the higher of the two scores.
        for blocks in _selected_blocks(eager[0]):
            assert len(blocks) == 512
            assert high_block in blocks and low_block not in blocks
    assert len(compiled) == len(eager) > 0
    for got, want in zip(compiled, eager):
        assert got.dtype == want.dtype and got.shape == want.shape
        assert mx.array_equal(got, want).item()


@pytest.mark.skipif(not mx.metal.is_available(), reason="fused kernels are the GPU's")
def test_the_divisor_stays_exact_where_a_division_fuses():
    # Whatever MLX fuses around the division in _select_eager, the divisor
    # must hold its value inside a fused kernel: joined to a fused
    # elementwise op, it still separates the pair. The Python float and a
    # one-element evaluated array do not (both are inlined as constants).
    indexer = QSAIndexer(_args())
    pair = mx.array([1.5 + 2.0**-23, 1.5 + 2.0**-22], dtype=mx.float32)

    def fused(scores):
        return mx.maximum(scores / indexer._score_divisor(), 0.0)

    with mx.stream(mx.gpu):
        eager = fused(pair)
        compiled = mx.compile(fused)(pair)
        mx.eval(eager, compiled)
    assert mx.array_equal(compiled, eager).item()
    assert compiled[0].item() < compiled[1].item()
