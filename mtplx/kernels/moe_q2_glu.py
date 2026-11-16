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
        if (SHARED) {
            // combine tail: routed sum + shared expert * sigmoid(shared gate)
            const float sg_ = (float)sgate[t];
            const float gate = (float)((T)(1.0f / (1.0f + metal::exp(-sg_))));
            const float sh = (float)((T)((float)shared[(size_t)t * dmodel + d] * gate));
            y[(size_t)t * dmodel + d] = (T)((float)((T)acc) + sh);
        } else {
            y[(size_t)t * dmodel + d] = (T)acc;
        }
    }
"""

# Router finalize. One threadgroup per token row: float32 softmax over the E
# router logits, the TOPK largest probabilities picked by TOPK rounds of a
# threadgroup argmax (ties go to the lower expert index), and their weights,
# renormalized over the picks when NORM. Replaces softmax, argpartition, the
# gather, the renormalization and two casts (six dependent dispatches).
_SRC_ROUTE = """
    constexpr int E = E_C;
    constexpr int TOPK = TOPK_C;
    constexpr int PER = (E + 255) / 256;
    threadgroup float red_v[8];
    threadgroup int red_i[8];
    threadgroup float tg_max;
    threadgroup float tg_sum;
    const uint tid = thread_position_in_threadgroup.x;
    const uint lane = tid % 32;
    const uint sg = tid / 32;
    const uint row = threadgroup_position_in_grid.y;
    float v[PER];
    float mx_ = -INFINITY;
    for (int k = 0; k < PER; ++k) {
        const int e = tid + 256 * k;
        v[k] = e < E ? (float)logits[(size_t)row * E + e] : -INFINITY;
        mx_ = max(mx_, v[k]);
    }
    mx_ = simd_max(mx_);
    if (lane == 0) red_v[sg] = mx_;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float m = red_v[0];
        for (int i = 1; i < 8; ++i) m = max(m, red_v[i]);
        tg_max = m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float m = tg_max;
    float ssum = 0.0f;
    for (int k = 0; k < PER; ++k) {
        v[k] = (tid + 256 * k) < (uint)E ? metal::precise::exp(v[k] - m) : 0.0f;
        ssum += v[k];
    }
    ssum = simd_sum(ssum);
    if (lane == 0) red_v[sg] = ssum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float a = 0.0f;
        for (int i = 0; i < 8; ++i) a += red_v[i];
        tg_sum = a;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float denom = tg_sum;
    float picked[TOPK];
    int picked_i[TOPK];
    for (int r = 0; r < TOPK; ++r) {
        float bv = -1.0f;
        int bi = E;
        for (int k = 0; k < PER; ++k) {
            const int e = tid + 256 * k;
            if (e < E && (v[k] > bv || (v[k] == bv && e < bi))) { bv = v[k]; bi = e; }
        }
        for (int off = 16; off > 0; off /= 2) {
            const float ov = simd_shuffle_down(bv, off);
            const int oi = simd_shuffle_down(bi, off);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        if (lane == 0) { red_v[sg] = bv; red_i[sg] = bi; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float gv = red_v[0];
        int gi = red_i[0];
        for (int i = 1; i < 8; ++i) {
            if (red_v[i] > gv || (red_v[i] == gv && red_i[i] < gi)) { gv = red_v[i]; gi = red_i[i]; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        picked[r] = gv / denom;
        picked_i[r] = gi;
        for (int k = 0; k < PER; ++k) {
            if ((int)(tid + 256 * k) == gi) v[k] = -2.0f;   // out of the next rounds
        }
    }
    if (tid == 0) {
        float tot = 0.0f;
        for (int r = 0; r < TOPK; ++r) tot += picked[r];
        for (int r = 0; r < TOPK; ++r) {
            experts[row * TOPK + r] = (uint)picked_i[r];
            weights[row * TOPK + r] = NORM ? picked[r] / tot : picked[r];
        }
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
        input_names=["h", "dw", "ds", "db", "experts", "rw", "shared", "sgate"],
        output_names=["y"],
        header=_HEADER,
        source=_SRC_B,
    )


@lru_cache(maxsize=4)
def _kernel_route():
    return mx.fast.metal_kernel(
        name="mtplx_moe_route_topk",
        input_names=["logits"],
        output_names=["experts", "weights"],
        header=_HEADER,
        source=_SRC_ROUTE,
    )


def moe_route_topk(logits: mx.array, top_k: int, *, normalize: bool):
    """[M, E] router logits -> (experts uint32 [M * top_k], weights float32
    [M * top_k]): softmax, top-k, optional renormalization over the picks."""
    m_rows, n_exp = int(logits.shape[0]), int(logits.shape[1])
    experts, weights = _kernel_route()(
        inputs=[logits],
        template=[("E_C", n_exp), ("TOPK_C", int(top_k)), ("NORM", int(bool(normalize)))],
        grid=(256, m_rows, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(m_rows * top_k,), (m_rows * top_k,)],
        output_dtypes=[mx.uint32, mx.float32],
    )
    return experts, weights


def _chunk_a(k: int) -> int:
    import os

    words = k // 128  # words per lane per row
    c = int(os.environ.get("MTPLX_Q2_MOE_CHUNK_A", "4"))
    return c if c > 0 and words % c == 0 else 1


def _rows_a() -> int:
    import os

    r = int(os.environ.get("MTPLX_Q2_MOE_ROWS_A", "32"))
    return r if r > 0 and r % 32 == 0 else 32


# GGUF Q2_0 blocks kept as they are (MTPLX_GGUF_Q2_0_BLOCKS, the ISTA pack
# on qwen4_exp.GGUFMixedSwitchGLU): the payload [E, R, K/16] and the block
# scales d [E, R, K/64] are the affine pack's weight and scales word for word,
# and w = d (q - 1) is the affine contract with bias = -scale, so the same two
# kernels run with the bias read replaced by -scale and gate and up as two
# arrays instead of the sanitize-fused [E, 2N] pack.
_SRC_A_BLOCKS = (
    _SRC_A.replace(
        "const size_t row = (size_t)e * 2 * n_inter + hh * n_inter + j;\n"
        "        const device uint* wrow = gw + row * NW;\n"
        "        const device T* srow = gs + row * NG;\n"
        "        const device T* brow = gb + row * NG;",
        "const size_t row = (size_t)e * n_inter + j;\n"
        "        const device uint* wrow = (hh == 0 ? gw : uw) + row * NW;\n"
        "        const device half* srow = (hh == 0 ? gs : us) + row * NG;",
    ).replace("bi[i] = (float)brow[w / 4];", "bi[i] = -sc[i];")
)
_SRC_B_BLOCKS = _SRC_B.replace("bi[i] = (float)db[row * NG + w / 4];", "bi[i] = -sc[i];").replace(
    "sc[i] = (float)ds[row * NG + w / 4];", "sc[i] = (float)ds[row * NG + w / 4];"
)
assert _SRC_A_BLOCKS != _SRC_A and "brow" not in _SRC_A_BLOCKS and "db[" not in _SRC_B_BLOCKS


@lru_cache(maxsize=4)
def _kernel_a_blocks():
    return mx.fast.metal_kernel(
        name="mtplx_moe_q2_blocks_glu_h",
        input_names=["x", "gw", "gs", "uw", "us", "experts"],
        output_names=["h"],
        header=_HEADER,
        source=_SRC_A_BLOCKS,
    )


@lru_cache(maxsize=4)
def _kernel_b_blocks():
    return mx.fast.metal_kernel(
        name="mtplx_moe_q2_blocks_down_y",
        input_names=["h", "dw", "ds", "experts", "rw", "shared", "sgate"],
        output_names=["y"],
        header=_HEADER,
        source=_SRC_B_BLOCKS,
    )


def moe_q2_blocks_glu(x2, sw, experts, route_w, shared=None, shared_gate=None):
    """moe_q2_glu for a GGUFMixedSwitchGLU whose gate, up and down are all
    Q2_0 blocks (``<proj>_q`` uint32 [E, R, K/16], ``<proj>_d`` fp16 [E, R, K/64])."""
    m_rows, k = int(x2.shape[0]), int(x2.shape[1])
    n_inter = int(sw.n_inter)
    dmodel = int(sw.hidden)
    topk = int(experts.shape[0]) // m_rows
    if k % 64 or n_inter % 64:
        raise ValueError("moe_q2_blocks_glu needs dot lengths that are multiples of 64")
    pairs = m_rows * topk
    rows_a = _rows_a()
    (h,) = _kernel_a_blocks()(
        inputs=[x2, sw.gate_q, sw.gate_d, sw.up_q, sw.up_d, experts],
        template=[("T", x2.dtype), ("K_C", k), ("n_inter_c", n_inter), ("topk_c", topk), ("CHUNK_A", _chunk_a(k)), ("ROWS_A", rows_a)],
        grid=(((n_inter + rows_a - 1) // rows_a) * 256, pairs, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(pairs, n_inter)],
        output_dtypes=[x2.dtype],
    )
    with_shared = shared is not None
    if not with_shared:
        shared = shared_gate = x2  # unused
    (y,) = _kernel_b_blocks()(
        inputs=[h, sw.down_q, sw.down_d, experts, route_w, shared, shared_gate],
        template=[("T", x2.dtype), ("N_C", n_inter), ("TOPK_C", topk), ("dmodel_c", dmodel),
                  ("SHARED", int(with_shared))],
        grid=(((dmodel + 31) // 32) * 256, m_rows, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(m_rows, dmodel)],
        output_dtypes=[x2.dtype],
    )
    return y


def moe_q2_glu(x2, gu_w, gu_s, gu_b, dn_w, dn_s, dn_b, experts, route_w, shared=None, shared_gate=None):
    """x2 [M, K]; gate+up pack [E, 2N, K/16] uint32 (2-bit g64) with scales and
    biases [E, 2N, K/64]; down pack [E, D, N/16] with [E, D, N/64];
    experts uint32 [M * topk] (token-major); route_w float32 [M * topk].
    With ``shared`` [M, D] and ``shared_gate`` [M] (logits) the down pass also
    adds shared * sigmoid(shared_gate). Returns y [M, D] in x2's dtype."""
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
    with_shared = shared is not None
    if not with_shared:
        shared = shared_gate = x2  # unused
    (y,) = _kernel_b()(
        inputs=[h, dn_w, dn_s, dn_b, experts, route_w, shared, shared_gate],
        template=[("T", x2.dtype), ("N_C", n_inter), ("TOPK_C", topk), ("dmodel_c", dmodel),
                  ("SHARED", int(with_shared))],
        grid=(((dmodel + 31) // 32) * 256, m_rows, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(m_rows, dmodel)],
        output_dtypes=[x2.dtype],
    )
    return y
