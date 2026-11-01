"""Routed-expert GLU for 2-bit g64 experts at verify widths (M = 2..8) on
simdgroup matrices, one pass per distinct expert.

moe_q2_glu's scalar kernels decode every (token, slot) pair's expert on its
own: at M = 4 a layer decodes 40 experts' rows for ~28 distinct experts, and
each weight costs a mask, a convert and an FMA per pair. Here a weight is
dequantized once into an 8x8 half tile held in registers (the simdgroup
matrix thread layout: lane l owns row (l / 4 & 4) + (l / 2) % 4, columns
(l / 4 & 2) * 2 + (l % 2) * 2 and the next one) and multiplied against all
M tokens (padded to 8) with simdgroup_multiply_accumulate:

  A) for each distinct expert e and 64 expert rows j: h[p, j] =
     silu(gate_e . x[t]) * (up_e . x[t]) for every pair p = (t, slot) that
     routes to e. A threadgroup whose pair is not the first one naming its
     expert returns at once; the first one serves every token.
  B) for each distinct expert e and 64 output dims d: z[p, d] = down_e[d] .
     h[p] for the pairs of e, h staged per token in threadgroup memory.
  C) y[t, d] = sum_slot w[t, slot] z[(t, slot), d] (+ shared expert *
     sigmoid(shared gate)).

Weights are rounded to fp16 once dequantized (as mx.dequantize writes them);
accumulation is float32.
"""

from functools import lru_cache

import mlx.core as mx

_HEADER = """
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// (row, col) of the first of the two elements lane `l` owns in an 8x8
// simdgroup matrix; the second is (row, col + 1).
#define FRAG_H(m) (reinterpret_cast<thread vec<half, 2>&>((m).thread_elements()))
#define FRAG_F(m) (reinterpret_cast<thread vec<float, 2>&>((m).thread_elements()))

inline short2 frag_rc(ushort l) {
    const short qid = l / 4;
    return short2((qid & 4) + ((l / 2) % 4), (qid & 2) * 2 + (l % 2) * 2);
}

// Dequantized pair of consecutive 2-bit weights (row r, columns c, c + 1).
template <typename T>
inline vec<half, 2> deq_pair(const device uint* wrow, const device T* srow, const device T* brow, int c) {
    const uint word = wrow[c / 16];
    const uint q = (word >> (2 * (c % 16))) & 15u;
    const float s = (float)srow[c / 64];
    const float b = (float)brow[c / 64];
    return vec<half, 2>((half)(s * (float)(q & 3u) + b), (half)(s * (float)(q >> 2) + b));
}
"""

# A: grid (N / 64 threadgroups, pairs); 8 simdgroups x 8 rows (gate and up).
_SRC_A = """
    constexpr int K = K_C;
    const int n_inter = n_inter_c;
    const int topk = topk_c;
    const int m_rows = m_rows_c;
    const uint p = threadgroup_position_in_grid.y;
    const uint e = experts[p];
    for (uint q = 0; q < p; ++q) {
        if (experts[q] == e) return;           // the first pair of e serves it
    }
    const ushort lane = thread_index_in_simdgroup;
    const ushort sg = simdgroup_index_in_threadgroup;
    const int j0 = threadgroup_position_in_grid.x * 64 + sg * 8;
    const short2 rc = frag_rc(lane);
    const int r = rc.x, c = rc.y;
    const size_t grow = (size_t)e * 2 * n_inter + j0 + r;       // gate row
    const size_t urow = grow + n_inter;                          // up row
    const device uint* gwr = gw + grow * (K / 16);
    const device uint* uwr = gw + urow * (K / 16);
    const device T* gsr = gs + grow * (K / 64);
    const device T* gbr = gb + grow * (K / 64);
    const device T* usr = gs + urow * (K / 64);
    const device T* ubr = gb + urow * (K / 64);

    constexpr int KC = 512;                     // x columns staged per pass
    threadgroup half xs[KC * 8];                // [k][token]
    const uint tid = thread_position_in_threadgroup.x;
    simdgroup_float8x8 cg = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 cu = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_half8x8 ag, au, bx;
    for (int kc = 0; kc < K; kc += KC) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < (uint)(KC * 8); i += 256) {
            const uint t = i / KC, k = i % KC;          // coalesced read of x[t][kc + k]
            xs[k * 8 + t] = xpad[(size_t)t * K + kc + k];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int g = kc / 64; g < (kc + KC) / 64; ++g) {
            const half2 gsc = half2((half)gsr[g]), gbi = half2((half)gbr[g]);
            const half2 usc = half2((half)usr[g]), ubi = half2((half)ubr[g]);
            for (int wi = 0; wi < 4; ++wi) {
                const uint gword = gwr[g * 4 + wi];
                const uint uword = uwr[g * 4 + wi];
                for (int hf = 0; hf < 2; ++hf) {
                    const uint sh = 2 * (hf * 8 + c);
                    const uint gq = (gword >> sh) & 15u;
                    const uint uq = (uword >> sh) & 15u;
                    FRAG_H(ag) = fma(gsc, half2(ushort2(gq & 3u, gq >> 2)), gbi);
                    FRAG_H(au) = fma(usc, half2(ushort2(uq & 3u, uq >> 2)), ubi);
                    const int kk = g * 64 + wi * 16 + hf * 8 - kc;
                    simdgroup_load(bx, xs + kk * 8, 8);
                    simdgroup_multiply_accumulate(cg, ag, bx, cg);
                    simdgroup_multiply_accumulate(cu, au, bx, cu);
                }
            }
        }
    }
    const vec<float, 2> g2 = FRAG_F(cg);
    const vec<float, 2> u2 = FRAG_F(cu);
    for (int i = 0; i < 2; ++i) {
        const int t = c + i;                    // token column
        if (t >= m_rows) continue;
        int slot = -1;
        for (int s = 0; s < topk; ++s) if (experts[t * topk + s] == e) slot = s;
        if (slot < 0) continue;
        const float gv = (float)((T)g2[i]);
        const float uv = (float)((T)u2[i]);
        const float sw = gv / (1.0f + metal::exp(-gv));
        h[((size_t)t * topk + slot) * n_inter + j0 + r] = (T)((float)((T)sw) * uv);
    }
"""

# B: grid (D / 64 threadgroups, pairs); h of e's tokens staged as [N][8].
_SRC_B = """
    constexpr int N = N_C;
    const int dmodel = dmodel_c;
    const int topk = topk_c;
    const int m_rows = m_rows_c;
    threadgroup half hs[N * 8];
    threadgroup int slot_of[8];
    const uint p = threadgroup_position_in_grid.y;
    const uint e = experts[p];
    for (uint q = 0; q < p; ++q) {
        if (experts[q] == e) return;
    }
    const uint tid = thread_position_in_threadgroup.x;
    if (tid < 8) {
        int slot = -1;
        if ((int)tid < m_rows) {
            for (int s = 0; s < topk; ++s) if (experts[tid * topk + s] == e) slot = s;
        }
        slot_of[tid] = slot;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < (uint)(N * 8); i += 256) {
        const uint k = i / 8, t = i % 8;
        const int slot = slot_of[t];
        hs[i] = slot < 0 ? (half)0.0h : (half)h[((size_t)t * topk + slot) * N + k];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const ushort lane = thread_index_in_simdgroup;
    const ushort sg = simdgroup_index_in_threadgroup;
    const int d0 = threadgroup_position_in_grid.x * 64 + sg * 8;
    const short2 rc = frag_rc(lane);
    const int r = rc.x, c = rc.y;
    const size_t row = (size_t)e * dmodel + d0 + r;
    const device uint* wr = dw + row * (N / 16);
    const device T* sr = ds + row * (N / 64);
    const device T* br = db + row * (N / 64);
    simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_half8x8 a, b;
    for (int g = 0; g < N / 64; ++g) {
        const half2 sc = half2((half)sr[g]), bi = half2((half)br[g]);
        for (int wi = 0; wi < 4; ++wi) {
            const uint word = wr[g * 4 + wi];
            for (int hf = 0; hf < 2; ++hf) {
                const uint q = (word >> (2 * (hf * 8 + c))) & 15u;
                FRAG_H(a) = fma(sc, half2(ushort2(q & 3u, q >> 2)), bi);
                simdgroup_load(b, hs + (g * 64 + wi * 16 + hf * 8) * 8, 8);
                simdgroup_multiply_accumulate(acc, a, b, acc);
            }
        }
    }
    const vec<float, 2> z2 = FRAG_F(acc);
    for (int i = 0; i < 2; ++i) {
        const int t = c + i;
        const int slot = t < 8 ? slot_of[t] : -1;
        if (slot < 0) continue;
        z[((size_t)t * topk + slot) * dmodel + d0 + r] = (T)z2[i];
    }
"""

# C: one thread per (token, dim).
_SRC_C = """
    const int dmodel = dmodel_c;
    const int topk = topk_c;
    const uint i = thread_position_in_grid.x;
    if (i >= (uint)(m_rows_c * dmodel)) return;
    const uint t = i / dmodel, d = i % dmodel;
    float acc = 0.0f;
    for (int s = 0; s < topk; ++s) acc = metal::fma(rw[t * topk + s], (float)z[((size_t)t * topk + s) * dmodel + d], acc);
    if (SHARED) {
        const float gate = (float)((T)(1.0f / (1.0f + metal::exp(-(float)sgate[t]))));
        const float sh = (float)((T)((float)shared[(size_t)t * dmodel + d] * gate));
        y[i] = (T)((float)((T)acc) + sh);
    } else {
        y[i] = (T)acc;
    }
"""


@lru_cache(maxsize=4)
def _kernel(name, inputs, outputs, src):
    return mx.fast.metal_kernel(
        name=name, input_names=list(inputs), output_names=list(outputs), header=_HEADER, source=src
    )


def moe_q2_mma(x2, gu_w, gu_s, gu_b, dn_w, dn_s, dn_b, experts, route_w, shared=None, shared_gate=None):
    """Same contract as moe_q2_glu.moe_q2_glu, for 2 <= M <= 8 rows."""
    m_rows, k = int(x2.shape[0]), int(x2.shape[1])
    n_inter = int(gu_w.shape[1]) // 2
    dmodel = int(dn_w.shape[1])
    topk = int(experts.shape[0]) // m_rows
    if not 1 <= m_rows <= 8 or k % 64 or n_inter % 64 or dmodel % 64:
        raise ValueError("moe_q2_mma serves 1..8 rows of 2-bit g64 packs with 64-multiple widths")
    pairs = m_rows * topk
    dt = x2.dtype
    xpad = x2.astype(mx.float16)
    if m_rows < 8:
        xpad = mx.concatenate([xpad, mx.zeros((8 - m_rows, k), mx.float16)], axis=0)
    (h,) = _kernel("mtplx_moe_q2_mma_h", ("xpad", "gw", "gs", "gb", "experts"), ("h",), _SRC_A)(
        inputs=[xpad, gu_w, gu_s, gu_b, experts],
        template=[("T", dt), ("K_C", k), ("n_inter_c", n_inter), ("topk_c", topk), ("m_rows_c", m_rows)],
        grid=((n_inter // 64) * 256, pairs, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(pairs, n_inter)],
        output_dtypes=[dt],
        init_value=0,
    )
    (z,) = _kernel("mtplx_moe_q2_mma_z", ("h", "dw", "ds", "db", "experts"), ("z",), _SRC_B)(
        inputs=[h, dn_w, dn_s, dn_b, experts],
        template=[("T", dt), ("N_C", n_inter), ("dmodel_c", dmodel), ("topk_c", topk), ("m_rows_c", m_rows)],
        grid=((dmodel // 64) * 256, pairs, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(pairs, dmodel)],
        output_dtypes=[dt],
    )
    with_shared = shared is not None
    if not with_shared:
        shared = shared_gate = x2
    (y,) = _kernel("mtplx_moe_q2_mma_y", ("z", "rw", "shared", "sgate"), ("y",), _SRC_C)(
        inputs=[z, route_w, shared, shared_gate],
        template=[("T", dt), ("dmodel_c", dmodel), ("topk_c", topk), ("m_rows_c", m_rows),
                  ("SHARED", int(with_shared))],
        grid=(m_rows * dmodel, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(m_rows, dmodel)],
        output_dtypes=[dt],
    )
    return y
