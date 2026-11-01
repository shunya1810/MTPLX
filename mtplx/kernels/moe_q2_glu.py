"""Routed-expert GLU for 2-bit g64 experts at decode and verify widths (M = 1..8).

The ISTA Q2_0 Flash-Next pack loads its experts as MLX affine 2-bit g64
(qwen4_exp._q2_0_affine_from_gguf). MLX's gather_qmm serves that layout at
~70 GB/s for one token and re-reads every expert per (token, slot) pair at
verify widths. These two kernels run a layer's routed experts in two
dispatches:

  A) h[p, j] = silu(gate_e(p) . x[t(p)]) * (up_e(p) . x[t(p)])  for each
     (token, slot) pair p and expert row j, from the sanitize-fused gate+up
     pack [E, 2N, K/16].
  B) y[t, d] = sum_slot w[t, slot] * (down_e[d, :] . h[t, slot, :]).

On M1 a 2-bit dot is ALU-bound before it is bandwidth-bound, so the decode
follows MLX's qmv: the activation is staged in threadgroup memory as float4
pre-scaled by (1, 1/4, 1/16, 1/64), and each code byte is masked with
(3, 12, 48, 192) in place, no shifts. Eight lanes share a row and read eight
consecutive 16-weight words per step (coalesced). A word's dot is
scale * sum(q * x) + bias * sum(x) with the word's x sum precomputed, so the
affine contract (w = scale * q + bias) holds for any bias. Accumulation is
float32; h is rounded to the activation dtype like the stock chain's GLU.
"""

from functools import lru_cache

import mlx.core as mx

_HEADER = """
#include <metal_stdlib>
using namespace metal;

// Dot of one word (sixteen 2-bit codes, element k in bits 2k) with its 16
// pre-scaled activations (four float4, element k scaled by 4^-(k % 4)).
inline float q2_word_dot(uint word, const threadgroup float4* xv) {
    const uint4 mask = uint4(3u, 12u, 48u, 192u);
    const uchar4 b = as_type<uchar4>(word);
    float acc = dot(float4(uint4(b.x) & mask), xv[0]);
    acc += dot(float4(uint4(b.y) & mask), xv[1]);
    acc += dot(float4(uint4(b.z) & mask), xv[2]);
    acc += dot(float4(uint4(b.w) & mask), xv[3]);
    return acc;
}
"""

# Stage NVALS values of src as pre-scaled float4 (xs) and their 16-value sums (wsum).
_STAGE = """
    for (uint w = tid; w < (uint)(NVALS / 16); w += 256) {
        float sum = 0.0f;
        for (int q = 0; q < 4; ++q) {
            const uint i = 4 * w + q;
            const float4 v = float4(src[4 * i], src[4 * i + 1], src[4 * i + 2], src[4 * i + 3]);
            xs[i] = v * float4(1.0f, 0.25f, 0.0625f, 0.015625f);
            sum += v.x + v.y + v.z + v.w;
        }
        wsum[w] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
"""

# Kernel A. Threadgroup = one (token, slot) pair x 32 expert rows: 8
# simdgroups, each owning 4 rows with 8 lanes a row.
_SRC_A = """
    constexpr int K = K_C;
    constexpr int NVALS = K;
    constexpr int NG = K / 64;
    constexpr int NW = K / 16;
    threadgroup float4 xs[K / 4];
    threadgroup float wsum[NW];

    const uint tid = thread_position_in_threadgroup.x;
    const uint pair = threadgroup_position_in_grid.y;
    const int n_inter = n_inter_c;
    const int topk = topk_c;
    const int t = pair / topk;
    const uint e = experts[pair];
    const device T* src = x + (size_t)t * K;
""" + _STAGE + """
    const uint sg = tid / 32;
    const uint lane = tid % 32;
    const uint l8 = lane % 8;
    constexpr int RPT = ROWS_A;         // rows per threadgroup (multiple of 32)
    for (int r0 = 0; r0 < RPT; r0 += 32) {
    const int j = threadgroup_position_in_grid.x * RPT + r0 + sg * 4 + lane / 8;
    if (j >= n_inter) break;

    float acc_g = 0.0f;
    float acc_u = 0.0f;
    for (int hh = 0; hh < 2; ++hh) {
        const size_t row = (size_t)e * 2 * n_inter + hh * n_inter + j;
        const device uint* wrow = gw + row * NW;
        const device T* srow = gs + row * NG;
        const device T* brow = gb + row * NG;
        // Loads are issued CHUNK words ahead of their use (the full row
        // unrolled costs occupancy: M1 Max, 148 vs 85 us a layer).
        constexpr int CHUNK = CHUNK_A;
        float acc = 0.0f;
        for (int i0 = 0; i0 < NW / 8; i0 += CHUNK) {
            uint words[CHUNK];
            float sc[CHUNK];
            float bi[CHUNK];
#pragma unroll
            for (int i = 0; i < CHUNK; ++i) {
                const int w = l8 + 8 * (i0 + i);
                words[i] = wrow[w];
                sc[i] = (float)srow[w / 4];
                bi[i] = (float)brow[w / 4];
            }
#pragma unroll
            for (int i = 0; i < CHUNK; ++i) {
                const int w = l8 + 8 * (i0 + i);
                acc += sc[i] * q2_word_dot(words[i], xs + 4 * w) + bi[i] * wsum[w];
            }
        }
        acc += simd_shuffle_xor(acc, 1);
        acc += simd_shuffle_xor(acc, 2);
        acc += simd_shuffle_xor(acc, 4);
        if (hh == 0) acc_g = acc; else acc_u = acc;
    }
    if (l8 == 0) {
        const float gv = (float)((T)acc_g);
        const float uv = (float)((T)acc_u);
        const float sw = gv / (1.0f + metal::exp(-gv));   // silu
        h[(size_t)pair * n_inter + j] = (T)((float)((T)sw) * uv);
    }
    }
"""

# Kernel B. Threadgroup = one token x 32 output dims: 8 simdgroups, 4 dims
# each, 8 lanes a dim; a lane walks the slots, taking every 8th word of the
# slot's row, over the token's h rows staged in threadgroup memory.
_SRC_B = """
    constexpr int N = N_C;
    constexpr int TOPK = TOPK_C;
    constexpr int NVALS = TOPK * N;
    constexpr int NG = N / 64;
    constexpr int NW = N / 16;
    threadgroup float4 xs[NVALS / 4];
    threadgroup float wsum[NVALS / 16];

    const uint tid = thread_position_in_threadgroup.x;
    const int t = threadgroup_position_in_grid.y;
    const int dmodel = dmodel_c;
    const device T* src = h + (size_t)t * NVALS;
""" + _STAGE + """
    const uint sg = tid / 32;
    const uint lane = tid % 32;
    const uint l8 = lane % 8;
    const int d = threadgroup_position_in_grid.x * 32 + sg * 4 + lane / 8;
    if (d >= dmodel) return;

    float acc = 0.0f;
    // One slot at a time with its NW / 8 words of this lane loaded up front.
    for (int slot = 0; slot < TOPK; ++slot) {
        const uint e = experts[t * TOPK + slot];
        const size_t row = (size_t)e * dmodel + d;
        uint words[NW / 8];
        float sc[NW / 8];
        float bi[NW / 8];
#pragma unroll
        for (int i = 0; i < NW / 8; ++i) {
            const int w = l8 + 8 * i;
            words[i] = dw[row * NW + w];
            sc[i] = (float)ds[row * NG + w / 4];
            bi[i] = (float)db[row * NG + w / 4];
        }
        float dot_ = 0.0f;
#pragma unroll
        for (int i = 0; i < NW / 8; ++i) {
            const int c = slot * NW + l8 + 8 * i;
            dot_ += sc[i] * q2_word_dot(words[i], xs + 4 * c) + bi[i] * wsum[c];
        }
        acc = metal::fma(rw[t * TOPK + slot], dot_, acc);
    }
    acc += simd_shuffle_xor(acc, 1);
    acc += simd_shuffle_xor(acc, 2);
    acc += simd_shuffle_xor(acc, 4);
    if (l8 == 0) {
        y[(size_t)t * dmodel + d] = (T)acc;
    }
"""


@lru_cache(maxsize=4)
def _kernel_a():
    return mx.fast.metal_kernel(
        name="mtplx_moe_q2_glu_h",
        input_names=["x", "gw", "gs", "gb", "experts"],
        output_names=["h"],
        header=_HEADER,
        source=_SRC_A,
    )


@lru_cache(maxsize=4)
def _kernel_b():
    return mx.fast.metal_kernel(
        name="mtplx_moe_q2_down_y",
        input_names=["h", "dw", "ds", "db", "experts", "rw"],
        output_names=["y"],
        header=_HEADER,
        source=_SRC_B,
    )


def _chunk_a(k: int) -> int:
    import os

    words = k // 128  # words per lane per row
    c = int(os.environ.get("MTPLX_Q2_MOE_CHUNK_A", "4"))
    return c if c > 0 and words % c == 0 else 1


def _rows_a() -> int:
    import os

    r = int(os.environ.get("MTPLX_Q2_MOE_ROWS_A", "32"))
    return r if r > 0 and r % 32 == 0 else 32


def moe_q2_glu(x2, gu_w, gu_s, gu_b, dn_w, dn_s, dn_b, experts, route_w):
    """x2 [M, K]; gate+up pack [E, 2N, K/16] uint32 (2-bit g64) with scales and
    biases [E, 2N, K/64]; down pack [E, D, N/16] with [E, D, N/64];
    experts uint32 [M * topk] (token-major); route_w float32 [M * topk].
    Returns y [M, D] in x2's dtype."""
    m_rows, k = int(x2.shape[0]), int(x2.shape[1])
    n_inter = int(gu_w.shape[1]) // 2
    dmodel = int(dn_w.shape[1])
    topk = int(experts.shape[0]) // m_rows
    if k % 64 or n_inter % 64 or gu_w.shape[2] * 16 != k or dn_w.shape[2] * 16 != n_inter:
        raise ValueError("moe_q2_glu needs 2-bit g64 packs whose dot lengths are multiples of 64")
    pairs = m_rows * topk
    rows_a = _rows_a()
    (h,) = _kernel_a()(
        inputs=[x2, gu_w, gu_s, gu_b, experts],
        template=[("T", x2.dtype), ("K_C", k), ("n_inter_c", n_inter), ("topk_c", topk), ("CHUNK_A", _chunk_a(k)), ("ROWS_A", rows_a)],
        grid=(((n_inter + rows_a - 1) // rows_a) * 256, pairs, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(pairs, n_inter)],
        output_dtypes=[x2.dtype],
    )
    (y,) = _kernel_b()(
        inputs=[h, dn_w, dn_s, dn_b, experts, route_w],
        template=[("T", x2.dtype), ("N_C", n_inter), ("TOPK_C", topk), ("dmodel_c", dmodel)],
        grid=(((dmodel + 31) // 32) * 256, m_rows, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(m_rows, dmodel)],
        output_dtypes=[x2.dtype],
    )
    return y
