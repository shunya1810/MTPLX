"""Flash-Next's routed gate/up gather on the tensor units, reading token rows in place.

A prefill-width MoE sorts its routed rows by expert and multiplies them with
``mx.gather_qmm(..., sorted_indices=True)``.  The sort first copies every token
row once per expert it is routed to (``x[order // top_k]``: at a 4,096-token
Flash-Next chunk, 40,960 rows of 2,560 values, 210 MB written and read back in
every layer), and MLX's kernel then streams that copy.

This kernel skips the copy.  Each lane reads the token rows its fragments need
through the sort's row map, so the ten routed copies of a token row are one
row in memory and come from cache.  Threadgroups own 64 sorted rows of ONE
expert: each expert's run is cut into its own tiles, and a threadgroup finds
its expert by binary search over a small prefix table of tiles per expert.
Streaming rows that are already sorted (the down projection) is not faster
this way (0.88x to 0.94x at 4,096 and 8,192-token chunks), so that stays on
MLX's kernel.

With ``swiglu=True`` the kernel writes ``silu(gate) * up`` directly, for
Flash-Next's fused ``[gate | up]`` weight or for separate gate and up weights
(mlx-lm's ``SwitchGLU``, as the Qwen3.5 and 3.6 MoE families use it): each
threadgroup holds 32 gate rows and the 32 matching up rows, so a lane's gate
and up values meet in one threadgroup, and the full gate/up output, the split
and the elementwise passes over it are never materialized.  The epilogue
rounds the accumulators to the activation dtype and applies MLX's own
``Sigmoid`` and ``Multiply`` functors (read from the installed headers) with
the same rounding after each op as ``nn.silu(gate) * up`` and mlx-lm's
compiled ``swiglu``, which a probe over all 65,536 bf16 and fp16 inputs
matched exactly.

Measured on an M5 Max (MLX 0.32.2, 2026-09-29; E 512, top-10, 4-bit gate/up
2,560 -> 1,280, median of 15 synchronized calls) against the chain it
replaces (row copy, stock gather, split, ``nn.silu(gate) * up``): 1.47x /
1.31x / 1.23x at 2,048 / 4,096 / 8,192-token chunks with group 32 and 1.43x /
1.30x / 1.23x with group 64; 10.0 ms -> 7.65 ms per layer at 4,096 tokens.

The arithmetic of every output row is MLX's: the kernel is compiled from the
installed MLX's own tensor-unit tile, multiply-accumulate and quantized
weight-loader templates (read from the installed package's headers at first
use; nothing of MLX is copied into this tree), with the tile shape, the 32-wide
inner steps and the float accumulation of MLX's sorted kernel, so the result is
bit-identical to ``mx.gather_qmm(tokens[row_map], ..., sorted_indices=True)``.
Row positions are 32-bit throughout, so there is no 32,767-row bound
(``mtplx.moe_sorted_gather``).  A first-use canary per compiled instantiation
and per MLX row regime (under and from 64 rows per expert) runs that whole
first call both ways, the kernel and the stock chain behind the row guard, and
compares the outputs bit for bit (so a sign of zero counts); a mismatch or a
compile failure turns the kernel off for that instantiation and regime for the
rest of the process with a printed reason and a counter, and the caller runs
the stock op (for ``swiglu=True``: the stock gather, split and SwiGLU).  An
install whose kernel headers are missing or unreadable keeps the stock path
too, with the reason printed once.

Only on tensor-unit GPUs (``nax_detect.nax_available()``: the M1 to M4
rehearsal switch keeps the stock path), for affine weights, bf16 or fp16
activations, 4 or 8 bits, K and N multiples of 64, and at least
:func:`min_rows` routed rows (decode and verify widths, and the short chunks
where MLX's own kernel is as fast, keep MLX's kernel).
``MTPLX_MOE_SORTED_GATHER_KERNEL=0`` keeps the stock path.
"""

from __future__ import annotations

import os
import re
import sys
from functools import lru_cache
from pathlib import Path

import mlx.core as mx

from mtplx import nax_detect

__all__ = ["applies", "available", "gather_rows_qmm", "min_rows", "stats"]

BM = 64
BN = 64
BK = 64
THREADS = 128
#: Routed rows below :func:`min_rows` keep MLX's kernel.  MLX before 0.32.3
#: tiles sorted rows 64 at a time across expert boundaries, and ours is faster
#: from about 410 tokens at top-10 (1.07x to 1.33x at 410 to 1,024 tokens on an
#: M5 Max).  MLX 0.32.3 tiles each expert's run on its own too and is faster
#: than ours on short chunks (ours 0.82x to 0.98x at 410 to 1,024 tokens), so
#: with it ours starts at 16,384 rows, where it wins on both (1.12x to 1.25x
#: from 2,048 tokens).
MIN_ROWS = 4096
MIN_ROWS_SEGMENTED_MLX = 16384
_BITS = (4, 8)
_GROUPS = (32, 64, 128)
_DTYPES = (mx.bfloat16, mx.float16)
_ENV = "MTPLX_MOE_SORTED_GATHER_KERNEL"

# The set MLX's own JIT concatenates for its sorted tensor-unit gather.
_KERNEL_HEADERS = (
    "mlx/backend/metal/kernels/steel/gemm/gemm_nax.h",
    "mlx/backend/metal/kernels/quantized_utils.h",
    "mlx/backend/metal/kernels/quantized_nax.h",
)
# MLX prepends utils.h (and what it includes) to every custom kernel.
_PREAMBLE_HEADER = "mlx/backend/metal/kernels/utils.h"
_INCLUDE = re.compile(r'^\s*#\s*include\s+"([^"]+)"\s*$')
_PRAGMA_ONCE = re.compile(r"^\s*#\s*pragma\s+once\s*$")

_STATS: dict[str, int] = {
    "calls": 0, "fallbacks": 0, "canaries": 0, "canary_failures": 0, "header_failures": 0,
}
_CANARY: dict[tuple, bool] = {}


def stats() -> dict[str, int]:
    return dict(_STATS)


def _include_root(package: Path | None = None) -> Path | None:
    """MLX's installed kernel headers (under ``package``, default the installed
    ``mlx`` package), or None when any of them is missing."""

    root = (package if package is not None else Path(mx.__file__).resolve().parent) / "include"
    for rel in _KERNEL_HEADERS + (_PREAMBLE_HEADER,):
        if not (root / rel).is_file():
            return None
    return root


def _closure(root: Path, rel: str, seen: set[str]) -> None:
    if rel in seen:
        return
    seen.add(rel)
    for line in (root / rel).read_text().splitlines():
        match = _INCLUDE.match(line)
        if match:
            _closure(root, match.group(1), seen)


def _inline(root: Path, rel: str, seen: set[str], out: list[str]) -> None:
    """Append ``rel`` with its quoted includes expanded once (in include order)."""

    if rel in seen:
        return
    seen.add(rel)
    for line in (root / rel).read_text().splitlines():
        match = _INCLUDE.match(line)
        if match:
            _inline(root, match.group(1), seen, out)
        elif not _PRAGMA_ONCE.match(line):
            out.append(line)


def _functor(root: Path, rel: str, name: str) -> str | None:
    """One elementwise functor struct from an installed MLX header, verbatim
    from the installed package (so the epilogue computes what MLX's own
    elementwise kernels compute)."""

    text = (root / rel).read_text()
    start = text.find(f"struct {name} {{")
    if start < 0:
        return None
    end = text.find("\n};", start)
    return None if end < 0 else text[start : end + 3]


def _headers_unavailable(reason: str) -> None:
    _STATS["header_failures"] += 1
    print(
        f"[moe-sorted-gather] MLX kernel headers unusable ({reason}); "
        "the stock sorted gather runs instead",
        file=sys.stderr,
        flush=True,
    )


@lru_cache(maxsize=1)
def _mlx_headers() -> str | None:
    """The installed MLX's tile, MMA and quantized-loader templates and its
    Sigmoid and Multiply functors as one header string, or None (the stock
    path runs) when this MLX install ships no kernel headers or any of them
    cannot be read; the reason is printed once and counted."""

    try:
        root = _include_root()
        if root is None:
            return None
        seen: set[str] = set()
        _closure(root, _PREAMBLE_HEADER, seen)  # already in every custom kernel
        out: list[str] = []
        for rel in _KERNEL_HEADERS:
            _inline(root, rel, seen, out)
        sigmoid = _functor(root, "mlx/backend/metal/kernels/unary_ops.h", "Sigmoid")
        multiply = _functor(root, "mlx/backend/metal/kernels/binary_ops.h", "Multiply")
    except Exception as exc:  # unreadable, missing or unexpected headers: the stock path runs
        _headers_unavailable(f"{type(exc).__name__}: {exc}")
        return None
    if sigmoid is None or multiply is None:
        _headers_unavailable("no Sigmoid or Multiply functor in unary_ops.h / binary_ops.h")
        return None
    out += [sigmoid, multiply]
    return "\n".join(out) + "\nusing namespace mlx::steel;\n"


# Our kernel body.  Template constants: T (activation dtype), GS, BITS, KD (the
# reduction width), ND (rows per expert of w), OUTC (output columns), ED
# (experts) and SWIGLU: the kernel then writes silu(gate) * up, each
# threadgroup holding 32 gate rows of w and the 32 matching up rows of wu (UD
# rows per expert, starting at UOFF; wu is w itself at UOFF = ND / 2 for a
# fused [gate | up] weight).
_SOURCE = r"""
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 64;
    constexpr int WN = 2;
    constexpr short SM = 32;
    constexpr short SN = 32;
    constexpr short SK = 32;
    constexpr short TM = SM / 16;
    constexpr short TN = SN / 16;
    constexpr short TK = SK / 16;
    constexpr int PACK = get_pack_factor<BITS, 8>();
    constexpr int PACK_BYTES = get_bytes_per_pack<BITS>();
    constexpr int BK_PAD = BK + 16 / sizeof(T);
    constexpr int ROW_BYTES = KD * PACK_BYTES / PACK;
    constexpr int ROW_GROUPS = KD / GS;
    constexpr int OUT_COLS = OUTC;
    // SWIGLU: the tile's first 32 weight rows are gate rows, the last 32 the
    // matching up rows, each half loaded by 64 threads from its own row range.
    using WeightLoader = metal::conditional_t<
        SWIGLU,
        QuantizedBlockLoader<T, BN / 2, BK, BK_PAD, 1, 64, GS, BITS>,
        QuantizedBlockLoader<T, BN, BK, BK_PAD, 1, 128, GS, BITS>>;

    threadgroup T w_tile[BN * BK_PAD];

    // The tile this threadgroup owns, and the one expert whose rows it holds:
    // the last expert whose first tile is at or before it.
    const int tile = int(threadgroup_position_in_grid.y);
    if (tile >= tile_start[ED]) {
        return;
    }
    int lo = 0;
    int hi = ED - 1;
    while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (tile_start[mid] <= tile) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    const int expert = lo;
    const int row0 = row_start[expert] + (tile - tile_start[expert]) * BM;
    const int tile_rows = min(BM, row_start[expert + 1] - row0);
    const int col0 = int(threadgroup_position_in_grid.x) * BN;

    const ushort sg = ushort(simdgroup_index_in_threadgroup);
    const ushort lane = ushort(thread_index_in_simdgroup);
    const short tm = SM * short(sg / WN);
    const short tn = SN * short(sg % WN);
    // Rows of this tile that fall in this simdgroup's 32-row half (0 to 32).
    const short rows = short(clamp(tile_rows - int(tm), 0, int(SM)));

    ushort load_sg = sg;
    const device uint8_t* w_src = (const device uint8_t*)w;
    const device T* s_src = scales;
    const device T* b_src = biases;
    size_t w_row = size_t(expert) * ND + size_t(col0);
    threadgroup T* w_dst = w_tile;
    if (SWIGLU) {
        // Output columns [col0 / 2, col0 / 2 + 32): the matching gate rows
        // of w and up rows of wu (the same array at an offset when the
        // weight is a fused [gate | up]).
        const int up_half = int(load_sg / 2);
        const size_t c = size_t(col0 / 2);
        if (up_half) {
            w_src = (const device uint8_t*)wu;
            s_src = scales_u;
            b_src = biases_u;
            w_row = size_t(expert) * UD + size_t(UOFF) + c;
        } else {
            w_row = size_t(expert) * ND + c;
        }
        w_dst = w_tile + up_half * (BN / 2) * BK_PAD;
        load_sg = load_sg % 2;
    }
    thread WeightLoader loader(
        w_src + w_row * ROW_BYTES,
        s_src + w_row * ROW_GROUPS,
        b_src + w_row * ROW_GROUPS,
        KD,
        w_dst,
        load_sg,
        lane);

    NAXTile<float, TM, TN> acc;
    acc.clear();

    // A lane holds rows (y, y + 8) and four consecutive columns from x of every
    // 16x16 fragment (MLX's fragment coordinates).  Its four token rows are
    // fixed for the whole K loop; rows past the tile read as zero, as MLX's
    // bounded load does.
    const short2 coord = BaseNAXFrag::get_coord();
    const device T* lane_rows[TM][2];
    bool lane_live[TM][2];
    for (short i = 0; i < TM; i++) {
        for (short r = 0; r < 2; r++) {
            const short local = i * 16 + coord.y + r * 8;
            lane_live[i][r] = local < rows;
            const int token = lane_live[i][r] ? int(row_map[row0 + tm + local]) : 0;
            lane_rows[i][r] = x + size_t(token) * KD + coord.x;
        }
    }

    // The K loop, compiled twice: for a simdgroup whose 32 rows are all live
    // (no bounds) and for a partial or empty half.
    dispatch_bool(rows == SM, [&](auto full) {
        for (int k = 0; k < KD / BK; k++) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            loader.load_unsafe();
            threadgroup_barrier(mem_flags::mem_threadgroup);

            STEEL_PRAGMA_NO_UNROLL
            for (short kk = 0; kk < BK; kk += SK) {
                if (decltype(full)::value || rows > 0) {
                    NAXTile<T, TM, TK> a;
                    NAXTile<T, TN, TK> b;
                    volatile int keep_order;
                    const int col = k * BK + kk;
                    STEEL_PRAGMA_UNROLL
                    for (short i = 0; i < TM; i++) {
                        STEEL_PRAGMA_UNROLL
                        for (short j = 0; j < TK; j++) {
                            thread auto& frag = a.frag_at(i, j);
                            STEEL_PRAGMA_UNROLL
                            for (short r = 0; r < 2; r++) {
                                const device T* src = lane_rows[i][r] + col + j * 16;
                                STEEL_PRAGMA_UNROLL
                                for (short c = 0; c < 4; c++) {
                                    if constexpr (decltype(full)::value) {
                                        frag[r * 4 + c] = src[c];
                                    } else {
                                        frag[r * 4 + c] = lane_live[i][r] ? src[c] : T(0);
                                    }
                                }
                            }
                        }
                    }
                    b.template load<T, BK_PAD, 1>(w_tile + tn * BK_PAD + kk);
                    tile_matmad_nax(
                        acc,
                        a,
                        metal::bool_constant<false>{},
                        b,
                        metal::bool_constant<true>{});
                    (void)keep_order;
                }
            }
            loader.next();
        }
    });

    if constexpr (SWIGLU) {
        // The up simdgroup of each 32-row half hands its values, rounded to T
        // as the stock gather's output is, to the gate simdgroup of the same
        // rows through the (now free) weight tile, lane to lane.  The gate
        // simdgroup writes silu(gate) * up with MLX's own functors, rounding to
        // T after each op as the stock chain does.
        constexpr short VALS = TM * TN * 8;
        static_assert(2 * 32 * VALS <= BN * BK_PAD, "the hand-off must fit the weight tile");
        threadgroup T* handoff = w_tile + (sg / 2) * (32 * VALS) + lane * VALS;
        const thread float* vals = acc.elems();
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tn != 0) {
            for (short e = 0; e < VALS; e++) {
                handoff[e] = static_cast<T>(vals[e]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tn == 0 && rows > 0) {
            device T* out = y + size_t(row0 + tm) * OUT_COLS + size_t(col0 / 2);
            for (short i = 0; i < TM; i++) {
                for (short j = 0; j < TN; j++) {
                    for (short r = 0; r < 2; r++) {
                        const short row = i * 16 + coord.y + r * 8;
                        if (row >= rows) {
                            continue;
                        }
                        for (short c = 0; c < 4; c++) {
                            const short e = (i * TN + j) * 8 + r * 4 + c;
                            const T gate = static_cast<T>(vals[e]);
                            const T act = Multiply()(gate, Sigmoid()(gate));
                            out[size_t(row) * OUT_COLS + j * 16 + coord.x + c] =
                                Multiply()(act, handoff[e]);
                        }
                    }
                }
            }
        }
    } else if (rows > 0) {
        device T* out = y + size_t(row0 + tm) * ND + size_t(col0 + tn);
        if (rows == SM) {
            acc.store(out, ND);
        } else {
            acc.store_safe(out, ND, short2(SN, rows));
        }
    }
"""


@lru_cache(maxsize=1)
def _kernel():
    headers = _mlx_headers()
    if headers is None:
        return None
    return mx.fast.metal_kernel(
        name="mtplx_moe_gather_rows",
        input_names=[
            "x", "w", "scales", "biases", "wu", "scales_u", "biases_u",
            "tile_start", "row_start", "row_map",
        ],
        output_names=["y"],
        source=_SOURCE,
        header=headers,
        ensure_row_contiguous=True,
    )


@lru_cache(maxsize=None)
def min_rows(version: str | None = None) -> int:
    """The fewest routed rows the kernel takes under this (or the given) MLX."""

    from mtplx.moe_sorted_gather import mlx_release_affected

    tiles_across_experts = mlx_release_affected(mx.__version__ if version is None else version)
    return MIN_ROWS if tiles_across_experts else MIN_ROWS_SEGMENTED_MLX


def _enabled() -> bool:
    raw = (os.environ.get(_ENV) or "").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def applies(
    tokens: mx.array,
    row_map: mx.array,
    w: mx.array,
    scales: mx.array,
    biases: mx.array | None,
    rhs_indices: mx.array,
    *,
    group_size,
    bits,
    mode,
    up: tuple | None = None,
) -> bool:
    """Whether this gather can take the kernel (shape, dtype and device).

    ``tokens`` is ``[n_tokens, 1, K]``; ``row_map`` and ``rhs_indices`` are the
    ``[rows]`` token index and expert of every sorted row.  ``up`` is a
    separate up projection's ``(weight, scales, biases)`` for the SwiGLU
    epilogue (``w`` then holds the gate rows only).
    """

    if up is not None:
        wu, su, bu = up
        if tuple(wu.shape) != tuple(w.shape) or wu.dtype != w.dtype:
            return False
        if tuple(su.shape) != tuple(scales.shape) or su.dtype != scales.dtype:
            return False
        if bu is None or biases is None or tuple(bu.shape) != tuple(biases.shape) or bu.dtype != biases.dtype:
            return False

    if mode != "affine" or bits not in _BITS or group_size not in _GROUPS:
        return False
    if tokens.dtype not in _DTYPES or tokens.ndim != 3 or tokens.shape[1] != 1:
        return False
    if scales.dtype != tokens.dtype or biases is None or biases.dtype != tokens.dtype:
        return False
    if rhs_indices is None or rhs_indices.ndim != 1 or row_map is None or row_map.ndim != 1:
        return False
    rows = int(rhs_indices.shape[0])
    if rows < min_rows() or int(row_map.shape[0]) != rows or row_map.dtype != mx.uint32:
        return False
    if w.ndim != 3 or w.dtype != mx.uint32:
        return False
    n = int(w.shape[1])
    k = int(w.shape[2]) * 32 // int(bits)
    if n % BN or k % BK or int(tokens.shape[-1]) != k or k % group_size:
        return False
    if not _enabled() or not nax_detect.nax_available():
        return False
    return _mlx_headers() is not None


def _schedule(rhs_indices: mx.array, experts: int) -> tuple[mx.array, mx.array]:
    """Exclusive prefix sums of 64-row tiles and of rows per expert, [E + 1] each."""

    counts = mx.zeros((experts,), dtype=mx.int32).at[rhs_indices].add(1)
    zero = mx.zeros((1,), dtype=mx.int32)
    row_start = mx.concatenate([zero, mx.cumsum(counts)])
    tiles = (counts + (BM - 1)) // BM
    tile_start = mx.concatenate([zero, mx.cumsum(tiles)])
    return tile_start, row_start


def _launch(tokens, row_map, w, scales, biases, rhs_indices, *, group_size, bits, swiglu=False, up=None):
    rows = int(rhs_indices.shape[0])
    experts = int(w.shape[0])
    n = int(w.shape[1])
    k = int(w.shape[2]) * 32 // int(bits)
    if not swiglu:
        wu, su, bu, up_rows, up_offset, out_cols, cols = w, scales, biases, n, 0, n, n
    elif up is None:  # fused [gate | up] rows in w
        wu, su, bu, up_rows, up_offset, out_cols, cols = w, scales, biases, n, n // 2, n // 2, n
    else:  # gate rows in w, up rows in up[0]
        wu, su, bu = up
        up_rows, up_offset, out_cols, cols = int(wu.shape[1]), 0, n, 2 * n
    tile_start, row_start = _schedule(rhs_indices, experts)
    (y,) = _kernel()(
        inputs=[tokens.reshape(-1, k), w, scales, biases, wu, su, bu, tile_start, row_start, row_map],
        template=[
            ("T", tokens.dtype),
            ("GS", int(group_size)),
            ("BITS", int(bits)),
            ("KD", k),
            ("ND", n),
            ("UD", up_rows),
            ("UOFF", up_offset),
            ("OUTC", out_cols),
            ("ED", experts),
            ("SWIGLU", bool(swiglu)),
        ],
        # Every expert adds at most one partial tile.
        grid=((cols // BN) * THREADS, (rows + BM - 1) // BM + experts, 1),
        threadgroup=(THREADS, 1, 1),
        output_shapes=[(rows, out_cols)],
        output_dtypes=[tokens.dtype],
    )
    return y.reshape(rows, 1, out_cols)


def _fail(key: tuple, reason: str) -> None:
    _CANARY[key] = False
    _STATS["canary_failures"] += 1
    print(
        f"[moe-sorted-gather] tensor-unit kernel off for {key}: {reason}; "
        "the stock sorted gather runs instead",
        file=sys.stderr,
        flush=True,
    )


def _stock(tokens, row_map, w, scales, biases, idx, *, group_size, bits, swiglu, up=None, act=None):
    """What the kernel replaces: the row copy and the stock sorted gather
    (behind the row guard, so it is correct at every width) and, for
    ``swiglu``, the gate and up halves (split from a fused weight, or ``up``'s
    own gather) through ``act(up, gate)`` (default ``nn.silu(gate) * up``)."""

    from mtplx.moe_sorted_gather import gather_qmm as guarded_gather_qmm

    def gather(wq, sq, bq):
        return guarded_gather_qmm(
            tokens[row_map], wq, sq, bq, rhs_indices=idx, transpose=True,
            group_size=group_size, bits=bits, sorted_indices=True,
        )

    y = gather(w, scales, biases)
    if not swiglu:
        return y
    if up is None:
        gate, upv = mx.split(y, 2, axis=-1)
    else:
        gate, upv = y, gather(*up)
    if act is not None:
        return act(upv, gate)
    import mlx.nn as nn

    return nn.silu(gate) * upv


def _same_bits(a: mx.array, b: mx.array) -> bool:
    """Equal bit for bit (so +0.0 and -0.0, or two NaN payloads, differ)."""

    if tuple(a.shape) != tuple(b.shape) or a.dtype != b.dtype:
        return False
    return bool(mx.array_equal(a.view(mx.uint16), b.view(mx.uint16)).item())


def _regime(rows: int, experts: int) -> str:
    """MLX picks its sorted kernel's row tile from rows per expert (32 rows
    under 64 per expert, 64 from there); each regime gets its own check."""

    return "wide" if rows >= 64 * experts else "narrow"


def _canary(
    key, tokens, row_map, w, scales, biases, rhs_indices, *, group_size, bits, swiglu, up=None, act=None
) -> tuple[bool, mx.array | None]:
    """The first call of an instantiation in each row regime runs both ways:
    the kernel and the stock chain on the whole call, compared bit for bit.
    Returns whether the kernel may run and, on that first call, its checked
    output."""

    passed = _CANARY.get(key)
    if passed is not None:
        return passed, None
    _STATS["canaries"] += 1
    kw = dict(group_size=group_size, bits=bits, swiglu=swiglu, up=up)
    try:
        ours = _launch(tokens, row_map, w, scales, biases, rhs_indices, **kw)
        stock = _stock(tokens, row_map, w, scales, biases, rhs_indices, act=act, **kw)
        mx.eval(ours, stock)
        same = _same_bits(ours, stock)
    except Exception as exc:  # a compile or dispatch failure on this GPU
        _fail(key, f"{type(exc).__name__}: {exc}")
        return False, None
    if not same:
        _fail(key, "output differs from the stock sorted gather, bit for bit, on its first call")
        return False, None
    _CANARY[key] = True
    print(
        f"[moe-sorted-gather] tensor-unit kernel on for {key}: its first call equals the stock "
        "sorted gather bit for bit",
        file=sys.stderr,
        flush=True,
    )
    return True, ours


def gather_rows_qmm(
    tokens: mx.array,
    row_map: mx.array,
    w: mx.array,
    scales: mx.array,
    biases: mx.array,
    rhs_indices: mx.array,
    *,
    group_size: int,
    bits: int,
    swiglu: bool = False,
    up: tuple | None = None,
    act=None,
) -> mx.array | None:
    """``gather_qmm(tokens[row_map], w, ..., sorted_indices=True)`` as
    ``[rows, 1, N]`` without the copy; with ``swiglu``, ``silu(gate) * up``
    as ``[rows, 1, I]``, where ``w`` is a fused ``[gate | up]`` weight
    (``up`` None, I = N / 2) or the gate weight with ``up`` the up
    projection's ``(weight, scales, biases)`` (I = N).  ``act(up, gate)`` is
    the chain the first-use check compares against (default
    ``nn.silu(gate) * up``).  None when the stock path must run."""

    key = (
        tokens.dtype, int(group_size), int(bits), tuple(w.shape), bool(swiglu),
        up is not None, type(act).__name__, _regime(int(rhs_indices.shape[0]), int(w.shape[0])),
    )
    kw = dict(group_size=int(group_size), bits=int(bits), swiglu=bool(swiglu), up=up)
    passed, checked = _canary(key, tokens, row_map, w, scales, biases, rhs_indices, act=act, **kw)
    if not passed:
        _STATS["fallbacks"] += 1
        return None
    _STATS["calls"] += 1
    if checked is not None:
        return checked
    return _launch(tokens, row_map, w, scales, biases, rhs_indices, **kw)


def available() -> bool:
    """Whether the kernel can run in this process at all (device, headers,
    switch), before any per-call shape check."""

    return _enabled() and nax_detect.nax_available() and _mlx_headers() is not None
