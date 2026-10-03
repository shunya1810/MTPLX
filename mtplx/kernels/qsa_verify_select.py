"""The QSA indexer's selection at verify widths inside a fixed bank, in two kernels.

Inside a compiled verify body a fixed-capacity QSA bank records the stock
eager selector (``QSAIndexer._select_eager``): its offset is a graph tensor, so
none of the host-planned selectors can run. Around the two library steps that
do the work, the score GEMM and ``argpartition``, that selector is a chain of
small kernels per layer and per verify: the query positions, the visible block
counts, the block ids and their validity, the relu, the head sum, the scale,
the invalid-block fill and the tie-break, and then either the dense mask (the
scatter of the chosen blocks, the validity AND, the repeat to tokens, the tail
and the causal bounds) or, on the rows-gather lane, the per-row token lists
(the block validity gather, the token ids, the repeat, the tail tokens and
their bounds, two concatenations and the zeroing of invalid slots). About 15
dispatches on the dense lane and 20 on the rows-gather lane, in each of
Flash-Next's 12 QSA layers, for every verify.

This module runs them as two kernels between the library steps:

1. ``block_scores``: from the GEMM's ``[1, S, H, blocks]`` float32 output,
   ``where(valid, sum_h max(score, 0) / divisor, -inf) - block * 1e-12``.
2. ``dense_mask`` or ``token_lists``: from ``argpartition``'s ``[S, blocks]``
   output, the ``[1, 1, S, capacity]`` boolean mask, or the ``[S, K * ratio +
   ratio]`` token ids and validity.

EXACTNESS. The integer and boolean work is exact by construction. The float
work repeats the stock arithmetic in its order: the head sum starts from +0
and adds the heads in order (MLX's small column reduce), the relu is MLX's
``Maximum`` (a NaN passes through), and the division and the tie-break are
spelled as the fused elementwise kernels of a compiled graph evaluate them.
``install`` compares the whole selection with the stock selector compiled on
this GPU, on both lanes and at every width it enables, before the first verify
trace; any difference or build failure leaves the stock selector serving with
a printed reason.

Rollback: ``MTPLX_QWEN4_QSA_VERIFY_SELECT=0`` keeps the stock selector.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Iterable, Optional

import mlx.core as mx
import numpy as np

ENV = "MTPLX_QWEN4_QSA_VERIFY_SELECT"

#: Verify windows the selection serves (the fixed bank runs verify rows only).
MIN_ROWS = 1
MAX_ROWS = 8

#: Threads per threadgroup (issue #400: at most 256).
_THREADS = 256

#: The dense-mask kernel keeps one bit per block in threadgroup memory: 2,048
#: words cover 65,536 blocks (a 262,144-token bank at ratio 4).
_MASK_WORDS = 2048

#: The division spellings the scores kernel can take, by the code its header
#: selects: the IEEE quotient, the fast quotient, and the product with the
#: IEEE or the fast reciprocal of the divisor (a compiled graph's fused kernel
#: may evaluate a division by a broadcast operand either way).
DIVISIONS = {
    0: "IEEE quotient",
    1: "fast quotient",
    2: "times the IEEE reciprocal",
    3: "times the fast reciprocal",
}

_STATE: dict[str, Any] = {
    "installed": False,
    "disabled_reason": None,
    "rows": (),
    "geometry": None,
    # The division spelling the probe matched (a DIVISIONS key), and whether
    # the tie-break is a fused multiply-add.
    "divide": None,
    "fma": None,
    # True while install's probe traces the selector.
    "probing": False,
}
_COUNTS: dict[str, int] = {"installs": 0, "probe_cases": 0, "probe_failures": 0, "traces": 0}
_ENGAGED: set[tuple[int, str]] = set()


_SCORES_SOURCE = r"""
    // raw: [1, S, H, NB] float32, pos: [1] int32 (KV index of row 0),
    // divisor: [1] float32. out: [S, NB] float32.
    uint b = thread_position_in_grid.x;
    uint s = thread_position_in_grid.y;
    const uint nb = raw_shape[3];
    if (b >= nb) {
        return;
    }
    const device float* column = raw + (size_t(s) * HEADS) * nb + b;
    // MLX's small column reduce: +0, then the heads in order.
    float total = 0.0f;
    for (uint h = 0; h < HEADS; ++h) {
        float value = column[size_t(h) * nb];
        // MLX's Maximum for floats: a NaN passes through.
        float relu = metal::isnan(value) ? value : (value > 0.0f ? value : 0.0f);
        total = relu + total;
    }
#if QSA_DIVIDE == 1
    float score = metal::fast::divide(total, divisor[0]);
#elif QSA_DIVIDE == 2
    float score = total * metal::precise::divide(1.0f, divisor[0]);
#elif QSA_DIVIDE == 3
    float score = total * metal::fast::divide(1.0f, divisor[0]);
#else
    float score = metal::precise::divide(total, divisor[0]);
#endif
    int qpos = pos[0] + int(s);
    int visible = (qpos + 1) / RATIO;
    float kept = (int(b) < visible) ? score : -INFINITY;
    float block = float(int(b));
#if QSA_FMA_TIE
    out[size_t(s) * nb + b] = metal::fma(-block, TIE, kept);
#else
    float tie = block * TIE;
    out[size_t(s) * nb + b] = kept - tie;
#endif
"""

_MASK_SOURCE = r"""
    // part: [S, NB] uint32 (argpartition), pos: [1] int32.
    // mask: [1, 1, S, NB * RATIO] bool.
    threadgroup atomic_uint chosen[MASK_WORDS];
    const uint s = threadgroup_position_in_grid.y;
    const uint tid = thread_position_in_threadgroup.x;
    const uint lanes = threads_per_threadgroup.x;
    const uint nb = part_shape[1];
    const uint k = nb < TOPK ? nb : TOPK;
    const uint words = (nb + 31) / 32;
    for (uint w = tid; w < words; w += lanes) {
        atomic_store_explicit(&chosen[w], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int qpos = pos[0] + int(s);
    const int visible = (qpos + 1) / RATIO;
    const device uint32_t* top = part + size_t(s) * nb + (nb - k);
    for (uint j = tid; j < k; j += lanes) {
        uint block = top[j];
        if (int(block) < visible) {
            atomic_fetch_or_explicit(
                &chosen[block / 32], 1u << (block % 32), memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int tail_start = visible * RATIO;
    const uint width = nb * RATIO;
    device bool* row = mask + size_t(s) * width;
    for (uint t = tid; t < width; t += lanes) {
        uint block = t / RATIO;
        uint word = atomic_load_explicit(&chosen[block / 32], memory_order_relaxed);
        bool selected = ((word >> (block % 32)) & 1u) != 0u;
        bool tail = int(t) >= tail_start;
        bool causal = int(t) <= qpos;
        row[t] = (selected || tail) && causal;
    }
"""

#: Tokens of one row that one threadgroup of the chunked dense mask writes.
_MASK_CHUNK = 1024

_MASK_CHUNKED_SOURCE = r"""
    // part: [S, NB] uint32 (argpartition), pos: [1] int32.
    // mask: [1, 1, S, NB * RATIO] bool; threadgroup (x, s) writes tokens
    // [x * CHUNK, (x + 1) * CHUNK) of row s from the chosen blocks in its range.
    threadgroup atomic_uint chosen[CHUNK / RATIO / 32];
    const uint s = threadgroup_position_in_grid.y;
    const uint first_token = threadgroup_position_in_grid.x * CHUNK;
    const uint first_block = first_token / RATIO;
    const uint tid = thread_position_in_threadgroup.x;
    const uint lanes = threads_per_threadgroup.x;
    const uint nb = part_shape[1];
    const uint k = nb < TOPK ? nb : TOPK;
    for (uint w = tid; w < CHUNK / RATIO / 32; w += lanes) {
        atomic_store_explicit(&chosen[w], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int qpos = pos[0] + int(s);
    const int visible = (qpos + 1) / RATIO;
    const device uint32_t* top = part + size_t(s) * nb + (nb - k);
    for (uint j = tid; j < k; j += lanes) {
        uint block = top[j];
        if (block >= first_block && block < first_block + CHUNK / RATIO && int(block) < visible) {
            uint local = block - first_block;
            atomic_fetch_or_explicit(&chosen[local / 32], 1u << (local % 32), memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int tail_start = visible * RATIO;
    const uint width = nb * RATIO;
    const uint last_token = min(width, first_token + CHUNK);
    device bool* row = mask + size_t(s) * width;
    for (uint t = first_token + tid; t < last_token; t += lanes) {
        uint local = t / RATIO - first_block;
        uint word = atomic_load_explicit(&chosen[local / 32], memory_order_relaxed);
        bool selected = ((word >> (local % 32)) & 1u) != 0u;
        bool tail = int(t) >= tail_start;
        bool causal = int(t) <= qpos;
        row[t] = (selected || tail) && causal;
    }
"""

_TOKENS_SOURCE = r"""
    // part: [S, NB] uint32 (argpartition), pos: [1] int32.
    // token_idx: [S, K * RATIO + RATIO] int32, token_ok: same shape, bool.
    const uint c = thread_position_in_grid.x;
    const uint s = thread_position_in_grid.y;
    const uint nb = part_shape[1];
    const uint k = nb < TOPK ? nb : TOPK;
    const uint width = k * RATIO + RATIO;
    if (c >= width) {
        return;
    }
    const int qpos = pos[0] + int(s);
    const int visible = (qpos + 1) / RATIO;
    int token;
    bool ok;
    if (c < k * RATIO) {
        int block = int(part[size_t(s) * nb + (nb - k) + c / RATIO]);
        ok = block < visible;
        token = block * RATIO + int(c % RATIO);
    } else {
        token = visible * RATIO + int(c - k * RATIO);
        ok = token <= qpos;
    }
    token_idx[size_t(s) * width + c] = ok ? token : 0;
    token_ok[size_t(s) * width + c] = ok;
"""


def _header(heads: int, ratio: int, topk: int, divide: int, fma_tie: bool) -> str:
    return (
        f"#define HEADS {int(heads)}u\n"
        f"#define RATIO {int(ratio)}\n"
        f"#define TOPK {int(topk)}u\n"
        f"#define MASK_WORDS {_MASK_WORDS}\n"
        "#define TIE 1e-12f\n"
        f"#define QSA_DIVIDE {int(divide)}\n"
        f"#define QSA_FMA_TIE {1 if fma_tie else 0}\n"
    )


@lru_cache(maxsize=None)
def _scores_kernel(heads: int, ratio: int, topk: int, divide: int, fma_tie: bool):
    return mx.fast.metal_kernel(
        name=f"mtplx_qsa_verify_scores_h{heads}_r{ratio}_d{divide}{'_fma' if fma_tie else ''}",
        input_names=["raw", "pos", "divisor"],
        output_names=["out"],
        header=_header(heads, ratio, topk, divide, fma_tie),
        source=_SCORES_SOURCE,
        ensure_row_contiguous=True,
    )


@lru_cache(maxsize=None)
def _mask_kernel(ratio: int, topk: int):
    return mx.fast.metal_kernel(
        name=f"mtplx_qsa_verify_mask_r{ratio}_k{topk}",
        input_names=["part", "pos"],
        output_names=["mask"],
        header=_header(1, ratio, topk, 0, False),
        source=_MASK_SOURCE,
        ensure_row_contiguous=True,
    )


@lru_cache(maxsize=None)
def _mask_chunked_kernel(ratio: int, topk: int):
    return mx.fast.metal_kernel(
        name=f"mtplx_qsa_verify_mask_chunked_r{ratio}_k{topk}",
        input_names=["part", "pos"],
        output_names=["mask"],
        header=_header(1, ratio, topk, 0, False) + f"#define CHUNK {_MASK_CHUNK}u\n",
        source=_MASK_CHUNKED_SOURCE,
        ensure_row_contiguous=True,
    )


@lru_cache(maxsize=None)
def _tokens_kernel(ratio: int, topk: int):
    return mx.fast.metal_kernel(
        name=f"mtplx_qsa_verify_tokens_r{ratio}_k{topk}",
        input_names=["part", "pos"],
        output_names=["token_idx", "token_ok"],
        header=_header(1, ratio, topk, 0, False),
        source=_TOKENS_SOURCE,
        ensure_row_contiguous=True,
    )


def _as_scalar_position(pos_start: Any) -> mx.array:
    pos = pos_start if isinstance(pos_start, mx.array) else mx.array(int(pos_start), dtype=mx.int32)
    if pos.dtype != mx.int32:
        pos = pos.astype(mx.int32)
    return pos.reshape((1,))


def block_scores(
    raw: mx.array,
    pos_start: Any,
    divisor: mx.array,
    *,
    ratio: int,
    topk: int,
    divide: Optional[int] = None,
    fma_tie: Optional[bool] = None,
) -> mx.array:
    """``[S, blocks]`` masked block scores from the score GEMM's output.

    ``divide`` picks the division's spelling (see ``DIVISIONS``); by default
    the one ``install`` matched to the compiled stock chain on this GPU.
    """

    if divide is None:
        divide = int(_STATE["divide"] or 0)
    if fma_tie is None:
        fma_tie = bool(_STATE["fma"])
    _, rows, heads, blocks = (int(d) for d in raw.shape)
    kernel = _scores_kernel(heads, int(ratio), int(topk), int(divide), bool(fma_tie))
    (out,) = kernel(
        inputs=[raw, _as_scalar_position(pos_start), divisor],
        grid=(blocks, rows, 1),
        threadgroup=(min(_THREADS, blocks), 1, 1),
        output_shapes=[(rows, blocks)],
        output_dtypes=[mx.float32],
    )
    return out


def dense_mask(part: mx.array, pos_start: Any, *, ratio: int, topk: int) -> mx.array:
    """The ``[1, 1, S, blocks * ratio]`` selection mask from argpartition's output."""

    rows, blocks = (int(d) for d in part.shape)
    (mask,) = _mask_kernel(int(ratio), int(topk))(
        inputs=[part, _as_scalar_position(pos_start)],
        grid=(_THREADS, rows, 1),
        threadgroup=(_THREADS, 1, 1),
        output_shapes=[(1, 1, rows, blocks * int(ratio))],
        output_dtypes=[mx.bool_],
    )
    return mask


def dense_mask_chunked(part: mx.array, pos_start: Any, *, ratio: int, topk: int) -> mx.array:
    """``dense_mask`` with one threadgroup per 1,024 tokens of a row."""

    rows, blocks = (int(d) for d in part.shape)
    width = blocks * int(ratio)
    (mask,) = _mask_chunked_kernel(int(ratio), int(topk))(
        inputs=[part, _as_scalar_position(pos_start)],
        grid=(((width + _MASK_CHUNK - 1) // _MASK_CHUNK) * _THREADS, rows, 1),
        threadgroup=(_THREADS, 1, 1),
        output_shapes=[(1, 1, rows, width)],
        output_dtypes=[mx.bool_],
    )
    return mask


def token_lists(part: mx.array, pos_start: Any, *, ratio: int, topk: int) -> tuple[mx.array, mx.array]:
    """Per-row token ids and validity (the rows-gather lane) from argpartition's output."""

    rows, blocks = (int(d) for d in part.shape)
    k = min(int(topk), blocks)
    width = k * int(ratio) + int(ratio)
    token_idx, token_ok = _tokens_kernel(int(ratio), int(topk))(
        inputs=[part, _as_scalar_position(pos_start)],
        grid=(width, rows, 1),
        threadgroup=(min(_THREADS, width), 1, 1),
        output_shapes=[(rows, width), (rows, width)],
        output_dtypes=[mx.int32, mx.bool_],
    )
    return token_idx, token_ok


def mask_fits(blocks: int) -> bool:
    """The dense-mask kernel's threadgroup bitmap holds this many blocks."""

    return 0 < int(blocks) <= _MASK_WORDS * 32


def switched_on() -> bool:
    raw = (os.environ.get(ENV) or "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def serves(indexer: Any, rows: int, blocks: int, *, dense: bool) -> bool:
    """True when ``install`` proved the selection for this indexer and width.

    Called while a compiled verify body is traced, once per QSA layer (the
    caller also requires the compiled step body: the kernels reproduce the
    compiled graph's fused lowering, not the eager kernels).
    """

    if not _STATE["installed"] or int(rows) not in _STATE["rows"]:
        return False
    if _geometry(indexer) != _STATE["geometry"]:
        return False
    return mask_fits(blocks) if dense else int(blocks) > 0


def note_engaged(rows: int, lane: str) -> None:
    """Counts traced selections; the first per (width, lane) leaves one log
    line. The install probe's own traces are not engagements."""

    if _STATE["probing"]:
        return
    _COUNTS["traces"] += 1
    key = (int(rows), lane)
    if key not in _ENGAGED:
        _ENGAGED.add(key)
        print(
            "[mtplx] QSA verify selection engaged in a compiled verify trace at "
            f"{rows} rows ({lane} lane, 2 dispatches around the score GEMM and argpartition)",
            flush=True,
        )


def _geometry(indexer: Any) -> Optional[tuple[int, int, int, int]]:
    try:
        return (
            int(indexer.n_heads),
            int(indexer.head_dim),
            int(indexer.ratio),
            int(indexer.block_topk),
        )
    except (AttributeError, TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Install: the first-use canary
# ---------------------------------------------------------------------------


class _ProbeBank:
    """The fixed bank's selector surface, over given arrays."""

    fixed_capacity = True

    def __init__(self, pooled: mx.array, raw_keys: mx.array, rows_gather: bool) -> None:
        self.pooled = pooled
        self.raw_keys = raw_keys
        self.fixed_rows_gather = bool(rows_gather)

    def pooled_f32_view(self, nb: int) -> mx.array:
        return mx.swapaxes(self.pooled.astype(mx.float32), 1, 2)[:, None][..., :nb]


def _probe_inputs(indexer: Any, rows: int, blocks: int, seed: int, ties: bool):
    key = mx.random.key(seed)
    k1, k2 = mx.random.split(key, 2)
    dim = int(indexer.head_dim)
    pooled = mx.random.normal((1, blocks, dim), key=k1).astype(mx.bfloat16)
    q = mx.random.normal((1, rows, int(indexer.n_heads), dim), key=k2).astype(mx.bfloat16)
    if ties:
        # Blocks whose every head-dot is negative relu-score exactly 0.0; a
        # run of them makes the top-k cut fall inside a tie.
        pooled = mx.concatenate([-mx.abs(pooled[:, : blocks // 2]) , pooled[:, blocks // 2 :]], axis=1)
        q = mx.abs(q)
    raw = mx.zeros((1, blocks * int(indexer.ratio), dim), dtype=mx.bfloat16)
    mx.eval(pooled, q, raw)
    return q, pooled, raw


def _probe_case(indexer: Any, rows: int, blocks: int, position: int, rows_gather: bool, seed: int, ties: bool) -> Optional[str]:
    q, pooled, raw = _probe_inputs(indexer, rows, blocks, seed, ties)
    pos = mx.array(int(position), dtype=mx.int32)

    def selector():
        # A new function object per arm: MLX caches a compiled trace by the
        # function it wraps, and the two arms must be traced separately.
        def run(q, pooled, raw, pos):
            bank = _ProbeBank(pooled, raw, rows_gather)
            out = indexer._select_eager(q, pos, bank, pooled, pos + q.shape[1])
            if isinstance(out, tuple):
                return tuple(out[1:])
            return (out,)

        return run

    from mtplx.compile_state import compiled_step_body

    stock_run, fused_run = selector(), selector()
    _STATE["installed"] = False
    with compiled_step_body():
        stock = mx.compile(stock_run)(q, pooled, raw, pos)
    mx.eval(*stock)
    _STATE["installed"] = True
    try:
        with compiled_step_body():
            fused = mx.compile(fused_run)(q, pooled, raw, pos)
        mx.eval(*fused)
    finally:
        _STATE["installed"] = False
    if len(stock) != len(fused):
        return "the fused selection returned a different lane"
    for want, got in zip(stock, fused):
        if want.shape != got.shape or want.dtype != got.dtype:
            return f"shape or dtype differs: {want.shape} {want.dtype} vs {got.shape} {got.dtype}"
        # Integer and boolean outputs, compared as their stored bytes.
        if not np.array_equal(np.array(want).view(np.uint8), np.array(got).view(np.uint8)):
            return (
                f"{'token lists' if rows_gather else 'dense mask'} differ at {rows} rows, "
                f"{blocks} blocks, position {position}"
            )
    return None


def _match_scores_spelling(indexer: Any) -> Optional[str]:
    """Pick the division and tie-break spellings that equal the stock chain.

    The fused elementwise kernels of a compiled graph build with the fast math
    mode, where a division and a multiply-subtract may lower differently from
    the IEEE spelling. Try the spellings on a score sheet with a wide exponent
    range and keep the first that matches every bit.
    """

    rows, heads, blocks = 8, int(indexer.n_heads), 4096
    key = mx.random.key(7)
    k1, k2 = mx.random.split(key, 2)
    raw = mx.random.normal((1, rows, heads, blocks), key=k1) * mx.exp(
        4.0 * mx.random.normal((1, rows, heads, blocks), key=k2)
    )
    raw = mx.where(mx.arange(blocks) % 97 == 0, mx.array(0.0), raw)
    # Scores small enough that the tie-break term is not lost in their last
    # bit, where a fused multiply-add and a multiply then subtract part ways.
    raw = mx.where(mx.arange(blocks) % 7 == 0, raw * 1e-8, raw)
    pos = mx.array(blocks * int(indexer.ratio) - rows, dtype=mx.int32)
    mx.eval(raw, pos)

    def stock(raw, pos):
        # The divisor is taken inside the trace, as _select_eager takes it: a
        # one-element view of a two-element buffer, which the fused kernel
        # reads from memory. Evaluated outside the trace and captured, it
        # would be written into the fused source with 7 significant digits.
        divisor = indexer._score_divisor()
        qpos = pos + mx.arange(rows, dtype=mx.int32)
        nb_q = (qpos + 1) // indexer.ratio
        blk = mx.arange(blocks, dtype=mx.int32)
        valid = blk[None, :] < nb_q[:, None]
        scores = mx.maximum(raw, 0.0).sum(axis=2) / divisor
        scores = scores[0]
        masked = mx.where(valid, scores, mx.array(-mx.inf, dtype=mx.float32))
        return masked - blk.astype(mx.float32)[None, :] * 1e-12

    want = np.array(mx.compile(stock)(raw, pos)).view(np.uint32)
    for divide in DIVISIONS:
        for fma_tie in (False, True):
            got = block_scores(
                raw, pos, indexer._score_divisor(), ratio=int(indexer.ratio),
                topk=int(indexer.block_topk), divide=divide, fma_tie=fma_tie,
            )
            if np.array_equal(want, np.array(got).view(np.uint32)):
                _STATE["divide"] = divide
                _STATE["fma"] = fma_tie
                return None
    return "no division and tie-break spelling reproduces the stock block scores"


def install(model: Any, *, rows: Iterable[int] = (4,)) -> dict:
    """Prove the selection on this GPU for each width in ``rows``, then enable it.

    ``model`` is the text model; the first QSA layer's indexer supplies the
    geometry. Runs at model build, outside any ``mx.compile`` trace.
    """

    reset_state()
    _COUNTS["installs"] += 1
    widths = tuple(sorted({int(r) for r in rows}))
    report: dict[str, Any] = {"rows": widths}

    def _off(reason: str, *, failure: bool = False) -> dict:
        _STATE["disabled_reason"] = reason
        if failure:
            _COUNTS["probe_failures"] += 1
        print(
            f"[mtplx] QSA verify selection off: {reason}; the stock selector serves",
            flush=True,
        )
        report.update(engagement())
        return report

    if not switched_on():
        return _off(f"{ENV}=0")
    if not widths or widths[0] < MIN_ROWS or widths[-1] > MAX_ROWS:
        return _off(f"widths {widths} outside {MIN_ROWS}..{MAX_ROWS}")
    if not mx.metal.is_available():
        return _off("no Metal GPU")
    indexer = None
    for layer in list(getattr(model, "layers", ()) or ()):
        attention = getattr(layer, "self_attn", None)
        candidate = getattr(attention, "indexer", None)
        if candidate is not None:
            indexer = candidate
            break
    if indexer is None:
        return _off("the model has no QSA indexer")
    geometry = _geometry(indexer)
    if geometry is None or geometry[2] < 1 or geometry[3] < 1:
        return _off("unreadable QSA indexer geometry")

    cases = 1
    _COUNTS["probe_cases"] += 1
    try:
        reason = _match_scores_spelling(indexer)
    except Exception as exc:  # a build failure or a refused dispatch
        reason = f"{type(exc).__name__}: {exc}"
    if reason is not None:
        return _off(reason, failure=True)

    _STATE.update(rows=widths, geometry=geometry)
    ratio, topk = geometry[2], geometry[3]
    seed = 20260929
    # Below the top-k (every block kept), just past it, and a long bank; at a
    # block boundary and inside a block; with and without relu ties.
    shapes = (
        (max(8, topk // 2), topk // 2 * ratio - 3, False),
        (topk + 37, (topk + 30) * ratio + 1, True),
        (4 * topk + 64, (4 * topk + 60) * ratio - 1, False),
    )
    for width in widths:
        for blocks, position, ties in shapes:
            for rows_gather in (False, True):
                cases += 1
                _COUNTS["probe_cases"] += 1
                seed += 1
                _STATE["probing"] = True
                try:
                    reason = _probe_case(indexer, width, blocks, position, rows_gather, seed, ties)
                except Exception as exc:  # a build failure or a refused dispatch
                    reason = f"{type(exc).__name__}: {exc}"
                finally:
                    _STATE["probing"] = False
                if reason is not None:
                    _STATE.update(rows=(), geometry=None)
                    return _off(reason, failure=True)

    _STATE.update(installed=True, disabled_reason=None)
    print(
        f"[mtplx] QSA verify selection on: rows={widths}, {cases} probe cases "
        "bit-equal to the compiled stock selector "
        f"(division {DIVISIONS[_STATE['divide']]}, tie-break {'fma' if _STATE['fma'] else 'multiply then subtract'})",
        flush=True,
    )
    report.update(engagement())
    return report


def engagement() -> dict:
    return {
        "installed": bool(_STATE["installed"]),
        "disabled_reason": _STATE["disabled_reason"],
        "rows": tuple(_STATE["rows"]),
        "geometry": _STATE["geometry"],
        "divide": _STATE["divide"],
        "fma": _STATE["fma"],
        "engaged": tuple(sorted(_ENGAGED)),
        **dict(_COUNTS),
    }


def reset_state() -> None:
    _STATE.update(
        installed=False, disabled_reason=None, rows=(), geometry=None, divide=None, fma=None, probing=False
    )
    _ENGAGED.clear()


def reset_for_tests() -> None:
    reset_state()
    for key in _COUNTS:
        _COUNTS[key] = 0


__all__ = [
    "ENV",
    "MAX_ROWS",
    "MIN_ROWS",
    "block_scores",
    "dense_mask",
    "engagement",
    "install",
    "mask_fits",
    "note_engaged",
    "reset_for_tests",
    "reset_state",
    "serves",
    "switched_on",
    "token_lists",
]
