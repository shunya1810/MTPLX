"""Expert-segmented NVFP4 / MXFP8 GEMM for Inkling 2 prefill experts.

``mx.gather_qmm`` with sorted indices tiles the sorted rows by BM and, when a
tile spans several experts, runs the whole tile once per expert; on M1 Max it
reaches ~5.4 TFLOPS on the routed experts where the dense fp4 matmul reaches
~7.0. Here the rows of each expert are cut into their own BM-row tiles (a
tile table built on the GPU from the sorted expert ids), so every tile runs
one expert once, through MLX's own steel BlockMMA and QuantizedBlockLoader
(the headers are inlined from the installed package; the tile body is
fp_qmm_t_impl with by-value sizes).
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import mlx.core as mx

import os

# Tile rows: 64 when the experts average at least _BM64_MIN_ROWS routed rows
# (half the weight dequantization per row), else 32 (less padding in each
# expert's last tile). Real Inkling prefill, same prompt, A/B interleaved:
# 4,096-row chunks (256 rows an expert) 7.50 -> 7.43 s at 4K, 31.2 -> 30.7 s at
# 16K, 66.7 -> 66.0 s at 32K; 2,048-row (128 an expert) 3.85 -> 3.91 s and
# 1,024-row (64) 2.04 -> 2.19 s, slower. Identical logits either way.
# Server prefill (one server, interleaved 32 / chosen, 4 runs each): 16K
# 488.5-496.8 -> 501.1-501.3 tok/s, 30K 467.5-469.2 -> 473.1-473.8 tok/s.
# MTPLX_INKLING2_SEG_BM fixes the rows.
_BM_FIXED = (os.environ.get("MTPLX_INKLING2_SEG_BM") or "").strip()
_BM64_MIN_ROWS = int(os.environ.get("MTPLX_INKLING2_SEG_BM64_MIN_ROWS", "256"))
_BN = int(os.environ.get("MTPLX_INKLING2_SEG_BN", "64"))
_BK = 32


def tile_rows(pairs: int, experts: int) -> int:
    """BM for ``pairs`` routed rows over ``experts`` experts."""
    if _BM_FIXED:
        return int(_BM_FIXED)
    return 64 if pairs >= _BM64_MIN_ROWS * max(1, experts) else 32


def _include_root() -> Path:
    return Path(mx.__file__).resolve().parent / "include"


def _inline(rel: str, seen: set) -> str:
    if rel in seen:
        return ""
    seen.add(rel)
    path = _include_root() / rel
    out = []
    for line in path.read_text().splitlines():
        m = re.match(r'\s*#include\s+"([^"]+)"', line)
        if m:
            out.append(_inline(m.group(1), seen))
            continue
        if line.strip() == "#pragma once":
            continue
        out.append(line)
    return "\n".join(out)


@lru_cache(maxsize=None)
def _header() -> str:
    # utils.h (and what it pulls in) is already in every custom kernel's prelude
    seen = {"mlx/backend/metal/kernels/utils.h", "mlx/backend/metal/kernels/bf16.h",
            "mlx/backend/metal/kernels/bf16_math.h", "mlx/backend/metal/kernels/complex.h"}
    parts = [
        _inline("mlx/backend/metal/kernels/steel/gemm/mma.h", seen),
        _inline("mlx/backend/metal/kernels/steel/gemm/loader.h", seen),
        _inline("mlx/backend/metal/kernels/fp_quantized.h", seen),
    ]
    return "\n".join(parts) + r"""

// fp_qmm_t_impl with sizes by value and the tile's first row given
template <typename T, const int group_size, const int bits, const int BM, const int BK, const int BN>
METAL_FUNC void ink_qmm_t_tile(
    const device uint32_t* w, const device uint8_t* scales, const device T* x, device T* y,
    threadgroup T* Xs, threadgroup T* Ws, const int K, const int N, const int M,
    const int y_col, uint simd_gid, uint simd_lid) {
  constexpr int WM = 2;
  constexpr int WN = 2;
  constexpr int pack_factor = get_pack_factor<8, bits>();
  constexpr int bytes_per_pack = get_bytes_per_pack();
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  using mma_t = mlx::steel::BlockMMA<T, T, BM, BN, BK, WM, WN, false, true, BK_padded, BK_padded>;
  using loader_x_t = mlx::steel::BlockLoader<T, BM, BK, BK_padded, 1, WM * WN * SIMD_SIZE>;
  using loader_w_t = QuantizedBlockLoader<T, BN, BK, BK_padded, 1, WM * WN * SIMD_SIZE, group_size, bits>;
  const int K_w = K * bytes_per_pack / pack_factor;
  const int K_g = K / group_size;
  auto wl = (const device uint8_t*)w;
  wl += y_col * K_w;
  scales += y_col * K_g;
  y += y_col;
  const short num_els = min(BM, M);
  loader_x_t loader_x(x, K, Xs, simd_gid, simd_lid);
  loader_w_t loader_w(wl, scales, K, Ws, simd_gid, simd_lid);
  mma_t mma_op(simd_gid, simd_lid);
  if (num_els < BM) {
    for (int k = 0; k < K; k += BK) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      loader_x.load_safe(short2(BK, num_els));
      loader_w.load_unsafe();
      threadgroup_barrier(mem_flags::mem_threadgroup);
      mma_op.mma(Xs, Ws);
      loader_x.next();
      loader_w.next();
    }
  } else {
    for (int k = 0; k < K; k += BK) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      loader_x.load_unsafe();
      loader_w.load_unsafe();
      threadgroup_barrier(mem_flags::mem_threadgroup);
      mma_op.mma(Xs, Ws);
      loader_x.next();
      loader_w.next();
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (num_els < BM) {
    mma_op.store_result_safe(y, N, short2(BN, num_els));
  } else {
    mma_op.store_result(y, N);
  }
}
"""


# One threadgroup: thread e finds expert e's run in the sorted ids, the tile
# counts are scanned, and each expert writes its tiles (expert, first row,
# rows); the rest of the table gets rows = 0.
_TABLE_SRC = r"""
    const uint e = thread_position_in_threadgroup.x;
    threadgroup int cnt[E];
    threadgroup int start[E];
    int lo = 0, hi = 0, ntile = 0;
    if (e < (uint)E) {
        int a = 0, b = P;
        while (a < b) { int m = (a + b) / 2; if ((int)sorted[m] < (int)e) a = m + 1; else b = m; }
        lo = a;
        a = lo; b = P;
        while (a < b) { int m = (a + b) / 2; if ((int)sorted[m] <= (int)e) a = m + 1; else b = m; }
        hi = a;
        ntile = (hi - lo + BM - 1) / BM;
        cnt[e] = ntile;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (e == 0) {
        int acc = 0;
        for (int i = 0; i < E; ++i) { start[i] = acc; acc += cnt[i]; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (e < (uint)E) {
        const int base = start[e];
        for (int j = 0; j < ntile; ++j) {
            table[(base + j) * 4 + 0] = (int)e;
            table[(base + j) * 4 + 1] = lo + j * BM;
            table[(base + j) * 4 + 2] = min(BM, hi - lo - j * BM);
            table[(base + j) * 4 + 3] = 0;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int total = start[E - 1] + cnt[E - 1];
    for (int t = total + (int)e; t < MAXT; t += TGS) {
        table[t * 4 + 0] = 0; table[t * 4 + 1] = 0; table[t * 4 + 2] = 0; table[t * 4 + 3] = 0;
    }
"""

_GEMM_SRC = r"""
    constexpr int BK_padded = (BK + 16 / sizeof(T));
    threadgroup T Xs[BM * BK_padded];
    threadgroup T Ws[BN * BK_padded];
    const uint t = threadgroup_position_in_grid.y;
    const int rows = table[t * 4 + 2];
    if (rows == 0) return;
    const int e = table[t * 4 + 0];
    const int row0 = table[t * 4 + 1];
    constexpr int pack_factor = get_pack_factor<8, BITS>();
    constexpr int bytes_per_pack = get_bytes_per_pack();
    const ulong w_stride = (ulong)N * (K * bytes_per_pack / pack_factor) / 4;   // uint32 words an expert
    const ulong s_stride = (ulong)N * (K / GS);
    ink_qmm_t_tile<T, GS, BITS, BM, BK, BN>(
        w + e * w_stride, scales + e * s_stride, x + (ulong)row0 * K, y + (ulong)row0 * N,
        Xs, Ws, K, N, rows, threadgroup_position_in_grid.x * BN,
        simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
"""


@lru_cache(maxsize=None)
def _table_kernel():
    return mx.fast.metal_kernel(
        name="mtplx_inkling2_seg_table", input_names=["sorted"], output_names=["table"], source=_TABLE_SRC,
    )


@lru_cache(maxsize=None)
def _gemm_kernel():
    return mx.fast.metal_kernel(
        name="mtplx_inkling2_seg_gemm", input_names=["x", "w", "scales", "table"], output_names=["y"],
        header=_header(), source=_GEMM_SRC,
    )


def seg_gemm_ok(n: int, k: int) -> bool:
    return n % _BN == 0 and k % _BK == 0


def max_tiles(pairs: int, experts: int) -> int:
    return -(-pairs // tile_rows(pairs, experts)) + experts


def tile_table(sorted_experts: mx.array, num_experts: int) -> mx.array:
    """[max_tiles, 4] int32 rows (expert, first row, rows, 0); rows 0 = unused."""
    P = int(sorted_experts.shape[0])
    mt = max_tiles(P, num_experts)
    tgs = max(32, -(-num_experts // 32) * 32)
    (table,) = _table_kernel()(
        inputs=[sorted_experts],
        template=[("P", P), ("E", num_experts), ("BM", tile_rows(P, num_experts)), ("MAXT", mt), ("TGS", tgs)],
        grid=(tgs, 1, 1),
        threadgroup=(tgs, 1, 1),
        output_shapes=[(mt * 4,)],
        output_dtypes=[mx.int32],
    )
    return table


def seg_gemm(x: mx.array, w: mx.array, scales: mx.array, table: mx.array, *, bits: int, group_size: int) -> mx.array:
    """x [P, K] (rows sorted by expert); w uint32 [E, N, K * bits / 32];
    scales uint8 [E, N, K / group_size]. Returns y [P, N] = x_p @ dequant(w_e(p)).T."""
    P, K = int(x.shape[0]), int(x.shape[1])
    N = int(w.shape[1])
    if N % _BN or K % _BK:
        raise ValueError("seg_gemm: N and K must be multiples of 32")
    mt = int(table.shape[0]) // 4
    (y,) = _gemm_kernel()(
        inputs=[x, w, scales, table],
        template=[("T", x.dtype), ("K", K), ("N", N), ("GS", group_size), ("BITS", bits),
                  ("BM", tile_rows(P, int(w.shape[0]))), ("BN", _BN), ("BK", _BK)],
        grid=((N // _BN) * 128, mt, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(P, N)],
        output_dtypes=[x.dtype],
    )
    return y
