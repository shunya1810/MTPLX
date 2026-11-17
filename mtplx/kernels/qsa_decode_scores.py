"""Masked block scores of 1-8 query rows in one dispatch over the float16 pooled keys.

The eager selector of a verify forward is a chain of separate MLX kernels:
the float32 GEMM against the fp32 mirror of the pooled keys, relu, head sum,
division, causal mask and the index tie-break.  For one to eight rows each of
those is a dispatch of its own; this kernel reads the float16 pooled keys once
and does all of it: simdgroup-matrix multiply (half x half -> float
accumulate, so every product is exact and only the summation order differs
from the float32 GEMM) with relu, head sum, division, mask and tie-break in
the registers of the same threads, spelled the way the eager chain (and, with
``divide``/``fma_tie``, a compiled graph) evaluates them.

The scores equal the stock ones to ~1e-7; the top-k sets differ only for a
block that sits within that of the cut-off.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

import mlx.core as mx

from mtplx.kernels import qsa_verify_select as V

ENV = "MTPLX_QSA_DECODE_SCORES"
_NF = 4
_SGS = 4
MAX_ROWS = 8

_SOURCE = r"""
    // q: [8, HEADS, D] half (rows beyond SROWS are zero), p: [N, D] half,
    // pos: [1] int32 (KV index of row 0), divisor: [1] float32. out: [SROWS, N].
    uint sg = simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    uint col0 = (threadgroup_position_in_grid.x * SGS + sg) * (NF * 8);
    const uint n = p_shape[0];
    if (col0 >= n) {
        return;
    }
    // A last fragment that runs past N reads its rows through this scratch
    // (zero beyond N) instead of past the end of the pooled keys.
    threadgroup half tail[SGS][64];
    simdgroup_float8x8 acc[HEADS][NF];
    #pragma clang loop unroll(full)
    for (int h = 0; h < HEADS; ++h) {
        #pragma clang loop unroll(full)
        for (int j = 0; j < NF; ++j) acc[h][j] = simdgroup_float8x8(0.0f);
    }
    for (int k = 0; k < HEAD_DIM; k += 8) {
        simdgroup_half8x8 a[HEADS];
        #pragma clang loop unroll(full)
        for (int h = 0; h < HEADS; ++h) {
            simdgroup_load(a[h], q + ulong(h) * HEAD_DIM + k, ulong(HEADS) * HEAD_DIM);
        }
        #pragma clang loop unroll(full)
        for (int j = 0; j < NF; ++j) {
            if (col0 + uint(j) * 8 < n) {
                simdgroup_half8x8 b;
                if (col0 + uint(j) * 8 + 8 <= n) {
                    simdgroup_load(b, p + ulong(col0 + j * 8) * HEAD_DIM + k, ulong(HEAD_DIM), ulong2(0, 0), true);
                } else {
                    const uint r = lane / 4;
                    const uint c = (lane % 4) * 2;
                    const uint row = col0 + uint(j) * 8 + r;
                    tail[sg][r * 8 + c] = row < n ? p[ulong(row) * HEAD_DIM + k + c] : half(0);
                    tail[sg][r * 8 + c + 1] = row < n ? p[ulong(row) * HEAD_DIM + k + c + 1] : half(0);
                    simdgroup_barrier(mem_flags::mem_threadgroup);
                    simdgroup_load(b, &tail[sg][0], ulong(8), ulong2(0, 0), true);
                    simdgroup_barrier(mem_flags::mem_threadgroup);
                }
                #pragma clang loop unroll(full)
                for (int h = 0; h < HEADS; ++h) simdgroup_multiply_accumulate(acc[h][j], a[h], b, acc[h][j]);
            }
        }
    }
    // Fragment element coordinates (MLX steel BaseMMAFrag::get_coord).
    const short qid = short(lane) / 4;
    const short fm = (qid & 4) + ((short(lane) / 2) % 4);
    const short fn = (qid & 2) * 2 + (short(lane) % 2) * 2;
    if (fm >= SROWS) {
        return;
    }
    const int qpos = pos[0] + int(fm);
    const int visible = (qpos + 1) / RATIO;
    #pragma clang loop unroll(full)
    for (int j = 0; j < NF; ++j) {
        if (col0 + uint(j) * 8 < n) {
            #pragma clang loop unroll(full)
            for (int e = 0; e < 2; ++e) {
                float total = 0.0f;
                #pragma clang loop unroll(full)
                for (int h = 0; h < HEADS; ++h) {
                    float value = acc[h][j].thread_elements()[e];
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
                const uint col = col0 + uint(j) * 8 + uint(fn) + uint(e);
                if (col >= n) {
                    continue;
                }
                float kept = (int(col) < visible) ? score : -INFINITY;
                float block = float(int(col));
#if QSA_FMA_TIE
                out[size_t(fm) * n + col] = metal::fma(-block, TIE, kept);
#else
                float tie = block * TIE;
                out[size_t(fm) * n + col] = kept - tie;
#endif
            }
        }
    }
"""


def enabled() -> bool:
    raw = (os.environ.get(ENV) or "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def supported(q: mx.array, pooled: mx.array, heads: int, head_dim: int) -> bool:
    """float16 ``q [1, S<=8, H, D]`` against float16 ``pooled [1, N, D]``."""

    if not enabled() or not hasattr(mx, "metal") or not mx.metal.is_available():
        return False
    if q.dtype != mx.float16 or pooled.dtype != mx.float16:
        return False
    if q.ndim != 4 or int(q.shape[0]) != 1 or not (1 <= int(q.shape[1]) <= MAX_ROWS):
        return False
    if int(q.shape[2]) != int(heads) or int(q.shape[3]) != int(head_dim) or int(head_dim) % 8:
        return False
    if pooled.ndim != 3 or int(pooled.shape[0]) != 1 or int(pooled.shape[2]) != int(head_dim):
        return False
    return int(pooled.shape[1]) > 0


@lru_cache(maxsize=None)
def _kernel(heads: int, head_dim: int, ratio: int, rows: int, divide: int, fma_tie: bool):
    header = (
        V._header(heads, ratio, 0, divide, fma_tie)
        + f"#define HEAD_DIM {int(head_dim)}\n#define NF {_NF}\n#define SGS {_SGS}\n#define SROWS {int(rows)}\n"
    )
    return mx.fast.metal_kernel(
        name=f"mtplx_qsa_decode_scores_h{heads}_d{head_dim}_r{ratio}_s{rows}_v{divide}{'_fma' if fma_tie else ''}",
        input_names=["q", "p", "pos", "divisor"],
        output_names=["out"],
        header=header,
        source=_SOURCE,
        ensure_row_contiguous=True,
    )


def masked_scores(
    q: mx.array,
    pooled: mx.array,
    pos_start: Any,
    divisor: mx.array,
    *,
    ratio: int,
    divide: int = 0,
    fma_tie: bool = False,
) -> mx.array:
    """``[S, N]`` masked block scores (eager spelling: IEEE quotient, multiply then subtract)."""

    _, rows, heads, head_dim = (int(d) for d in q.shape)
    blocks = int(pooled.shape[1])
    q8 = q[0]
    if rows < MAX_ROWS:
        q8 = mx.concatenate([q8, mx.zeros((MAX_ROWS - rows, heads, head_dim), dtype=q8.dtype)], axis=0)
    span = _NF * 8 * _SGS
    (out,) = _kernel(heads, head_dim, int(ratio), rows, int(divide), bool(fma_tie))(
        inputs=[q8, pooled[0], V._as_scalar_position(pos_start), divisor.reshape((1,)).astype(mx.float32)],
        grid=(((blocks + span - 1) // span) * 32 * _SGS, 1, 1),
        threadgroup=(32 * _SGS, 1, 1),
        output_shapes=[(rows, blocks)],
        output_dtypes=[mx.float32],
    )
    return out
