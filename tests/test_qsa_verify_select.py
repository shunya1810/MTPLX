"""The fixed bank's QSA selection in two kernels, against the compiled stock selector.

A promoted Flash-Next cache hands its compiled verifier a fixed QSA bank whose
offset is a graph tensor, so the verifier records ``QSAIndexer._select_eager``:
the score GEMM and ``argpartition`` with a chain of small kernels around them.
mtplx/kernels/qsa_verify_select.py runs that chain as two kernels. Every case
here builds the indexer at Flash-Next's geometry (4 heads of 128, a 512-block
budget, compress ratio 4), traces the stock selector and the fused one inside a
compiled step body, and compares every output bit: the dense mask below the
rows-gather floor, the per-row token lists above it.
"""

from __future__ import annotations

import collections
import tempfile
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from mtplx.compile_state import compiled_step_body
from mtplx.kernels import qsa_verify_select as sel
from mtplx.models.qwen4_exp import QSAIndexer, TextArgs

pytestmark = pytest.mark.skipif(
    not mx.metal.is_available(), reason="the kernels and their compiled parent run on the GPU"
)

HEADS = 4
HEAD_DIM = 128
RATIO = 4
BUDGET = 2048  # block_topk 512
TOPK = BUDGET // RATIO


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


class _Bank:
    """What ``_select_eager`` reads from a promoted TensorOffsetQSACache."""

    fixed_capacity = True

    def __init__(self, pooled: mx.array, raw_keys: mx.array, rows_gather: bool) -> None:
        self.pooled = pooled
        self.raw_keys = raw_keys
        self.fixed_rows_gather = rows_gather

    def pooled_f32_view(self, nb: int) -> mx.array:
        return mx.swapaxes(self.pooled.astype(mx.float32), 1, 2)[:, None][..., :nb]


class _StockBank(_Bank):
    """A stock QSACache's view: never the fixed bank's lane."""

    fixed_capacity = False


@pytest.fixture(scope="module")
def indexer() -> QSAIndexer:
    built = QSAIndexer(_args())
    assert built.block_topk == TOPK and built.head_dim == HEAD_DIM
    return built


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    monkeypatch.delenv(sel.ENV, raising=False)
    sel.reset_for_tests()
    yield
    sel.reset_for_tests()


def _model(indexer: QSAIndexer) -> SimpleNamespace:
    return SimpleNamespace(
        layers=[
            SimpleNamespace(linear_attn=object()),
            SimpleNamespace(self_attn=SimpleNamespace(indexer=indexer)),
        ]
    )


def _install(indexer: QSAIndexer, rows=(4,)) -> dict:
    report = sel.install(_model(indexer), rows=rows)
    assert report["installed"], report["disabled_reason"]
    return report


@pytest.fixture(scope="module")
def proven(indexer):
    """One install for widths 1 to 8 (its probe compiles every case twice)."""

    sel.reset_for_tests()
    report = sel.install(_model(indexer), rows=tuple(range(1, 9)))
    assert report["installed"], report["disabled_reason"]
    verdict = dict(sel._STATE)
    sel.reset_for_tests()
    return verdict


@pytest.fixture
def installed(proven):
    sel._STATE.update(proven)
    yield
    sel.reset_for_tests()


def _inputs(blocks: int, rows: int, seed: int, *, ties: bool = False):
    k1, k2 = mx.random.split(mx.random.key(seed), 2)
    pooled = mx.random.normal((1, blocks, HEAD_DIM), key=k1).astype(mx.bfloat16)
    q = mx.random.normal((1, rows, HEADS, HEAD_DIM), key=k2).astype(mx.bfloat16)
    if ties:
        # Half the blocks score exactly 0.0 (every head-dot negative), so the
        # top-k cut falls inside a run of ties that the block ids break.
        half = blocks // 2
        pooled = mx.concatenate([-mx.abs(pooled[:, :half]), mx.abs(pooled[:, half:])], axis=1)
        q = mx.abs(q)
    raw = mx.zeros((1, blocks * RATIO, 1), dtype=mx.bfloat16)
    mx.eval(pooled, q, raw)
    return q, pooled, raw


def _runner(indexer: QSAIndexer, rows_gather: bool, bank=_Bank):
    def run(q, pooled, raw, pos):
        out = indexer._select_eager(q, pos, bank(pooled, raw, rows_gather), pooled, pos + q.shape[1])
        return tuple(out[1:]) if isinstance(out, tuple) else (out,)

    return run


def _compiled(indexer, rows_gather, q, pooled, raw, pos):
    with compiled_step_body():
        outs = mx.compile(_runner(indexer, rows_gather))(q, pooled, raw, pos)
    mx.eval(*outs)
    return outs


def _stock(indexer, rows_gather, q, pooled, raw, pos, monkeypatch):
    with monkeypatch.context() as patch:
        patch.setattr(sel, "serves", lambda *a, **k: False)
        return _compiled(indexer, rows_gather, q, pooled, raw, pos)


def _assert_same(want, got):
    assert len(want) == len(got)
    for a, b in zip(want, got):
        assert a.shape == b.shape and a.dtype == b.dtype
        assert np.array_equal(np.array(a), np.array(b))


def _primitives(indexer, rows_gather, q, pooled, raw, pos, *, compiled: bool) -> collections.Counter:
    run = _runner(indexer, rows_gather)
    if compiled:
        with compiled_step_body():
            outs = mx.compile(run)(q, pooled, raw, pos)
    else:
        outs = run(q, pooled, raw, pos)
    path = tempfile.mktemp(suffix=".dot")
    mx.export_to_dot(path, *outs)
    names = [line.split('label ="')[1].split('"')[0] for line in open(path) if "shape=rectangle" in line]
    views = {"Reshape", "Broadcast", "Slice", "Transpose", "ExpandDims", "Squeeze", "Flatten", "Unflatten", "AsStrided"}
    return collections.Counter(name for name in names if name not in views)


CASES = [
    # (blocks, position of row 0, relu ties)
    pytest.param(300, 300 * RATIO - 5, False, id="every-block-kept"),
    pytest.param(TOPK + 37, (TOPK + 30) * RATIO + 1, True, id="cut-inside-ties"),
    pytest.param(4 * TOPK + 64, (4 * TOPK + 60) * RATIO - 1, False, id="long-bank"),
]


@pytest.mark.parametrize("rows_gather", [False, True], ids=["dense-mask", "rows-gather"])
@pytest.mark.parametrize(("blocks", "position", "ties"), CASES)
@pytest.mark.parametrize("rows", [1, 2, 3, 4, 5, 6, 7, 8])
def test_every_width_matches_the_compiled_stock_selector(
    indexer, installed, monkeypatch, rows, blocks, position, ties, rows_gather
):
    q, pooled, raw = _inputs(blocks, rows, seed=blocks + rows, ties=ties)
    pos = mx.array(position, dtype=mx.int32)
    want = _stock(indexer, rows_gather, q, pooled, raw, pos, monkeypatch)
    got = _compiled(indexer, rows_gather, q, pooled, raw, pos)
    _assert_same(want, got)
    engaged = sel.engagement()["engaged"]
    lane = "rows-gather" if rows_gather and rows > 1 else "dense"
    assert (rows, lane) in engaged


def test_inside_a_compiled_body_the_selection_is_two_kernels_around_the_gemm_and_argpartition(
    indexer, installed
):
    q, pooled, raw = _inputs(1088, 4, seed=3)
    pos = mx.array(4000, dtype=mx.int32)
    for rows_gather in (False, True):
        fused = _primitives(indexer, rows_gather, q, pooled, raw, pos, compiled=True)
        assert fused == collections.Counter(
            {"AsType": 2, "Matmul": 1, "ArgPartition": 1, "CustomKernel": 2}
        ), fused


def test_the_stock_chain_is_the_long_one(indexer, monkeypatch):
    q, pooled, raw = _inputs(1088, 4, seed=3)
    pos = mx.array(4000, dtype=mx.int32)
    for rows_gather in (False, True):
        stock = _primitives(indexer, rows_gather, q, pooled, raw, pos, compiled=True)
        assert "CustomKernel" not in stock
        assert sum(stock.values()) >= 14, stock


def test_outside_a_compiled_body_the_stock_selector_runs(indexer, installed):
    q, pooled, raw = _inputs(1088, 4, seed=4)
    pos = mx.array(4000, dtype=mx.int32)
    eager = _primitives(indexer, False, q, pooled, raw, pos, compiled=False)
    assert "CustomKernel" not in eager
    assert sel.engagement()["traces"] == 0


def test_a_width_the_probe_did_not_prove_keeps_the_stock_selector(indexer, monkeypatch):
    _install(indexer, rows=(4,))
    q, pooled, raw = _inputs(1088, 3, seed=5)
    pos = mx.array(4000, dtype=mx.int32)
    got = _compiled(indexer, False, q, pooled, raw, pos)
    assert sel.engagement()["traces"] == 0
    _assert_same(_stock(indexer, False, q, pooled, raw, pos, monkeypatch), got)


def test_a_stock_qsa_cache_never_takes_the_fused_selection(indexer, installed, monkeypatch):
    def must_not_run(*args, **kwargs):
        raise AssertionError("a stock QSA cache reached the fused selection")

    monkeypatch.setattr(QSAIndexer, "_verify_select_fused", must_not_run)
    # 1,088 complete blocks: the last row sits at position 4,351.
    q, pooled, raw = _inputs(1088, 4, seed=6)
    with compiled_step_body():
        out = indexer._select_eager(q, 4348, _StockBank(pooled, raw, False), pooled, 4352)
    mx.eval(out)
    assert sel.engagement()["traces"] == 0


def test_the_scores_take_the_compiled_spelling(indexer, installed):
    report = sel.engagement()
    assert report["divide"] in sel.DIVISIONS and report["fma"] in (False, True)
    rows, blocks = 4, 2048
    k1, k2 = mx.random.split(mx.random.key(11), 2)
    raw = mx.random.normal((1, rows, HEADS, blocks), key=k1) * mx.exp(
        3.0 * mx.random.normal((1, rows, HEADS, blocks), key=k2)
    )
    pos = mx.array(blocks * RATIO - 2, dtype=mx.int32)

    def stock(raw, pos):
        divisor = indexer._score_divisor()  # inside the trace, as the selector takes it
        qpos = pos + mx.arange(rows, dtype=mx.int32)
        valid = mx.arange(blocks, dtype=mx.int32)[None, :] < ((qpos + 1) // RATIO)[:, None]
        scores = (mx.maximum(raw, 0.0).sum(axis=2) / divisor)[0]
        masked = mx.where(valid, scores, mx.array(-mx.inf, dtype=mx.float32))
        return masked - mx.arange(blocks, dtype=mx.int32).astype(mx.float32)[None, :] * 1e-12

    want = np.array(mx.compile(stock)(raw, pos)).view(np.uint32)
    got = np.array(sel.block_scores(raw, pos, indexer._score_divisor(), ratio=RATIO, topk=TOPK)).view(np.uint32)
    assert np.array_equal(want, got)


def test_the_probe_turns_the_selection_off_on_any_difference(indexer, monkeypatch, capsys):
    real = sel.dense_mask

    def one_bit_off(part, pos_start, *, ratio, topk):
        mask = real(part, pos_start, ratio=ratio, topk=topk)
        flat = mask.reshape(-1)
        return mx.concatenate([mx.logical_not(flat[:1]), flat[1:]]).reshape(mask.shape)

    monkeypatch.setattr(sel, "dense_mask", one_bit_off)
    report = sel.install(_model(indexer), rows=(4,))
    assert not report["installed"]
    assert "dense mask differ" in report["disabled_reason"]
    assert report["probe_failures"] == 1
    assert "QSA verify selection off" in capsys.readouterr().out
    assert not sel.serves(indexer, 4, 1088, dense=True)


def test_a_kernel_that_fails_to_build_leaves_the_stock_selector(indexer, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("Unable to build the Metal library")

    monkeypatch.setattr(sel, "_scores_kernel", broken)
    report = sel.install(_model(indexer), rows=(4,))
    assert not report["installed"]
    assert "Unable to build" in report["disabled_reason"]


def test_the_switch_keeps_the_stock_selector(indexer, monkeypatch):
    monkeypatch.setenv(sel.ENV, "0")
    report = sel.install(_model(indexer), rows=(4,))
    assert not report["installed"]
    assert sel.ENV in report["disabled_reason"]


def test_a_dense_mask_past_the_bitmap_keeps_the_stock_selector(indexer, installed):
    limit = 2048 * 32
    assert sel.serves(indexer, 4, limit, dense=True)
    assert not sel.serves(indexer, 4, limit + 1, dense=True)
    assert sel.serves(indexer, 4, limit + 1, dense=False)


def test_another_indexer_geometry_keeps_the_stock_selector(indexer, installed):
    other = QSAIndexer(TextArgs.from_dict({**_args().__dict__, "indexer_budget": 1024}))
    assert other.block_topk != indexer.block_topk
    assert not sel.serves(other, 4, 1088, dense=True)
