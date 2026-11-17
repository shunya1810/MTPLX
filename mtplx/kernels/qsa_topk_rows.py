"""Exact top-k of a few float32 rows in one threadgroup each (radix select).

OPT-IN (MTPLX_QSA_TOPK_KERNEL=1), default off: 12-layer microbenchmarks win (0.8 vs
1.3 ms), but inside a real 128K verify forward it is 1-2 ms slower than
argpartition (fn-s8/LOG.md 17).

Decode and verify select the 512 best of up to 65,536 block scores for 1-8 rows.
``mx.argpartition`` does that with a general sort network whose launch and
passes cost 0.2-0.4 ms per layer at these widths; four threadgroups of 1,024
threads find the cut-off key in four 8-bit histogram passes (the data stays in
L2: 130 KB a row) and compact the winners in index order.

Contract (the same set ``mx.argpartition`` returns for the last ``k`` entries,
whenever the scores differ):
  * ``top[s, :]`` holds the indices of the ``k`` largest ``masked[s, :]``;
  * equal scores at the cut-off are taken lowest index first (torch.topk);
  * the order inside a row is "greater than the cut-off by index", then "equal
    to the cut-off by index" - not sorted by score.

``qsa_verify_select.dense_mask`` and ``token_lists`` read a row's indices only
as a set per row (a bitmap) or as a list the attention gathers; see
``mtplx.kernels.qsa_verify_select`` for the order-insensitive consumers.
"""

from __future__ import annotations

import os
from functools import lru_cache

import mlx.core as mx

ENV = "MTPLX_QSA_TOPK_KERNEL"
_THREADS = 1024

_SOURCE = r"""
    threadgroup atomic_uint hist[256];
    threadgroup uint chosen_bin;
    threadgroup uint chosen_above;
    threadgroup uint simd_gt[32];
    threadgroup uint simd_eq[32];

    const uint s = threadgroup_position_in_grid.y;
    const uint tid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint nb = masked_shape[1];
    const uint K = nb < KTOP ? nb : KTOP;
    auto row = masked + size_t(s) * nb;
    const uint per = (nb + LANES - 1) / LANES;
    const uint lo = min(nb, tid * per);
    const uint hi = min(nb, lo + per);

    uint prefix = 0;
    uint need = K;
    for (uint pass = 0; pass < 4; ++pass) {
        const uint shift = 24 - 8 * pass;
        if (tid < 256) {
            atomic_store_explicit(&hist[tid], 0u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = lo; i < hi; ++i) {
            uint bits = as_type<uint>(row[i]);
            uint key = (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
            if (pass == 0 || (key >> (shift + 8)) == prefix) {
                atomic_fetch_add_explicit(&hist[(key >> shift) & 255u], 1u, memory_order_relaxed);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            uint acc = 0;
            uint bin = 0;
            for (int b = 255; b >= 0; --b) {
                uint c = atomic_load_explicit(&hist[b], memory_order_relaxed);
                if (acc + c >= need) {
                    bin = uint(b);
                    break;
                }
                acc += c;
            }
            chosen_bin = bin;
            chosen_above = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        prefix = (prefix << 8) | chosen_bin;
        need -= chosen_above;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // prefix is the K-th largest key; `need` of the keys equal to it are taken.
    uint gt = 0;
    uint eq = 0;
    for (uint i = lo; i < hi; ++i) {
        uint bits = as_type<uint>(row[i]);
        uint key = (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
        gt += key > prefix ? 1u : 0u;
        eq += key == prefix ? 1u : 0u;
    }
    uint gt_in = simd_prefix_exclusive_sum(gt);
    uint eq_in = simd_prefix_exclusive_sum(eq);
    if (lane == 31) {
        simd_gt[sg] = gt_in + gt;
        simd_eq[sg] = eq_in + eq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint gt_base = 0;
    uint eq_base = 0;
    for (uint g = 0; g < sg; ++g) {
        gt_base += simd_gt[g];
        eq_base += simd_eq[g];
    }
    uint gt_at = gt_base + gt_in;
    uint eq_at = eq_base + eq_in;
    device uint* out_row = top + size_t(s) * K;
    const uint eq_slot0 = K - need;
    for (uint i = lo; i < hi; ++i) {
        uint bits = as_type<uint>(row[i]);
        uint key = (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
        if (key > prefix) {
            out_row[gt_at++] = i;
        } else if (key == prefix) {
            if (eq_at < need) {
                out_row[eq_slot0 + eq_at] = i;
            }
            eq_at++;
        }
    }
"""


def enabled() -> bool:
    raw = (os.environ.get(ENV) or "0").strip().lower()
    return raw in {"1", "true", "yes", "on"}


@lru_cache(maxsize=None)
def _kernel(topk: int):
    return mx.fast.metal_kernel(
        name=f"mtplx_qsa_topk_rows_k{topk}",
        input_names=["masked"],
        output_names=["top"],
        header=f"#define KTOP {int(topk)}u\n#define LANES {_THREADS}u\n",
        source=_SOURCE,
        ensure_row_contiguous=True,
    )


def topk_rows(masked: mx.array, topk: int) -> mx.array:
    """``[rows, min(topk, blocks)]`` uint32 indices of each row's largest scores."""

    if masked.ndim != 2 or masked.dtype != mx.float32:
        raise ValueError(f"masked must be float32 [rows, blocks], got {masked.shape} {masked.dtype}")
    rows, blocks = (int(d) for d in masked.shape)
    k = min(int(topk), blocks)
    if rows <= 0 or k <= 0:
        raise ValueError(f"empty selection: rows={rows} blocks={blocks} topk={topk}")
    (top,) = _kernel(int(topk))(
        inputs=[masked],
        grid=(_THREADS, rows, 1),
        threadgroup=(_THREADS, 1, 1),
        output_shapes=[(rows, k)],
        output_dtypes=[mx.uint32],
    )
    return top
