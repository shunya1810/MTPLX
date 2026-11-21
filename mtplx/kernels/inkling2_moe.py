"""Routed NVFP4 experts of Inkling 2 Mini at decode / verify widths.

Two kernels replace the gather_qmm pair and its glue (dtype casts, the outer
scale multiplies, the gate/up de-interleave, the softcapped SwiGLU, the
routing weights and the sum over experts):

* mid:  h[pair, j] = swiglu_softcap(fac_e * (gate_j . x), fac_e * (up_j . x))
        rows 2j / 2j+1 of the gate_up weight are gate / up (checkpoint order).
* down: y[t, d] = sum_slot w[t, slot] * fac_e * (down_d . h[t, slot]).

Weights stay as shipped: E2M1 codes two per byte (even element in the low
nibble), one E4M3 scale per 16 values, a float32 outer scale ``fac_e``
(amax / denominator) per expert. Both decode through threadgroup tables
(byte -> two E2M1 values, byte -> E4M3 value). Accumulation is float32, so
the raw product never meets float16's range (the reason the gather_qmm path
runs this layer in bfloat16).
"""

from __future__ import annotations

import os
from functools import lru_cache

import mlx.core as mx

__all__ = ["moe_nvfp4", "nvfp4_mid", "nvfp4_down"]

_HEADER = r"""
inline void ink_build_tables(threadgroup half2* c_lut, threadgroup float* s_lut, uint tid) {
    // byte -> (E2M1 low nibble, E2M1 high nibble)
    constexpr float e2m1[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const uint lo = tid & 15u, hi = tid >> 4;
    const float vlo = (lo & 8u) ? -e2m1[lo & 7u] : e2m1[lo & 7u];
    const float vhi = (hi & 8u) ? -e2m1[hi & 7u] : e2m1[hi & 7u];
    c_lut[tid] = half2(vlo, vhi);
    // byte -> E4M3 (fn) value
    const uint s = tid >> 7, e = (tid >> 3) & 15u, m = tid & 7u;
    float v = (e == 0u) ? metal::ldexp((float)m * 0.125f, -6)
                        : metal::ldexp(1.0f + (float)m * 0.125f, (int)e - 7);
    if (e == 15u && m == 7u) v = 0.0f;
    s_lut[tid] = s ? -v : v;
}

inline float ink_dot8(uint word, float4 a, float4 b, threadgroup const half2* c_lut) {
    const half2 v0 = c_lut[word & 0xFFu], v1 = c_lut[(word >> 8) & 0xFFu];
    const half2 v2 = c_lut[(word >> 16) & 0xFFu], v3 = c_lut[word >> 24];
    return dot(float4(v0.x, v0.y, v1.x, v1.y), a) + dot(float4(v2.x, v2.y, v3.x, v3.y), b);
}
"""

# grid.x = blocks of ROWS output j's, grid.y = (token, slot) pairs; 256
# threads = 8 simdgroups x 4 lane groups of 8 lanes; a lane group owns NR
# consecutive j's (2 NR weight rows) and its lanes stride the 32-value chunks.
_MID_SRC = r"""
    constexpr int CH = K / 32;
    constexpr int LPR = 8;
    constexpr int ROWS = 32 * NR;
    threadgroup half2 c_lut[256];
    threadgroup float s_lut[256];
    const uint tid = thread_position_in_threadgroup.x;
    ink_build_tables(c_lut, s_lut, tid);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint pair = threadgroup_position_in_grid.y;
    const uint t = pair / TOPK;
    const uint e = experts[pair];
    const float fac = escale[e];
    const uint sg = tid / 32, lane = tid % 32;
    const uint grp = lane / LPR, l8 = lane % LPR;
    const uint j0 = threadgroup_position_in_grid.x * ROWS + (sg * 4 + grp) * NR;
    const device vec<T, 4>* xd = (const device vec<T, 4>*)(x + (ulong)t * K);

    float acc[2 * NR];
    for (int i = 0; i < 2 * NR; ++i) acc[i] = 0.0f;
    for (uint c = l8; c < (uint)CH; c += LPR) {
        float4 xv[8];
        for (int i = 0; i < 8; ++i) xv[i] = float4(xd[c * 8 + i]);
        for (int rp = 0; rp < 2 * NR; ++rp) {
            const uint j = j0 + rp / 2;
            if (j >= (uint)N) break;
            const ulong row = (ulong)e * (2 * N) + 2 * j + (rp & 1);
            const uint4 q = ((const device uint4*)(w + row * (K / 8)))[c];
            const device uchar* sc = scales + row * (K / 16) + 2 * c;
            const float s0 = s_lut[sc[0]], s1 = s_lut[sc[1]];
            const float p0 = ink_dot8(q.x, xv[0], xv[1], c_lut) + ink_dot8(q.y, xv[2], xv[3], c_lut);
            const float p1 = ink_dot8(q.z, xv[4], xv[5], c_lut) + ink_dot8(q.w, xv[6], xv[7], c_lut);
            acc[rp] += s0 * p0 + s1 * p1;
        }
    }
    for (int rp = 0; rp < 2 * NR; ++rp) {
        float v = acc[rp];
        v += simd_shuffle_xor(v, 1);
        v += simd_shuffle_xor(v, 2);
        v += simd_shuffle_xor(v, 4);
        acc[rp] = v;
    }
    if (l8 == 0) {
        for (int r = 0; r < NR; ++r) {
            const uint j = j0 + r;
            if (j >= (uint)N) break;
            const float g = (float)((T)(acc[2 * r] * fac));
            const float u = (float)((T)(acc[2 * r + 1] * fac));
            const float gs = (float)GSOFT, us = (float)USOFT;
            const float gc = metal::precise::tanh(g / gs) * gs;
            const float uc = metal::precise::tanh(u / us) * us;
            const float sig = 1.0f / (1.0f + metal::precise::exp(-g));
            h[(ulong)pair * N + j] = (T)(gc * sig * uc);
        }
    }
"""

# grid.x = blocks of 256 / LPD output dims, grid.y = tokens; 256 threads,
# LPD lanes an output dim; the token's TOPK h rows are staged as half4.
_DOWN_SRC = r"""
    constexpr int CH = N / 32;
    constexpr int NV4 = TOPK * N / 4;
    threadgroup half2 c_lut[256];
    threadgroup float s_lut[256];
    threadgroup half4 hs[NV4];
    const uint tid = thread_position_in_threadgroup.x;
    ink_build_tables(c_lut, s_lut, tid);
    const uint t = threadgroup_position_in_grid.y;
    const device vec<T, 4>* src = (const device vec<T, 4>*)(h + (ulong)t * TOPK * N);
    for (uint i = tid; i < (uint)NV4; i += 256) hs[i] = half4(src[i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint sg = tid / 32, lane = tid % 32, l4 = lane % LPD;
    const uint dim = threadgroup_position_in_grid.x * (256 / LPD) + sg * (32 / LPD) + lane / LPD;
    if (dim >= (uint)DM) return;
    float acc = 0.0f;
    for (uint slot = 0; slot < (uint)TOPK; ++slot) {
        const uint e = experts[t * TOPK + slot];
        const ulong row = (ulong)e * DM + dim;
        const device uint4* qrow = (const device uint4*)(w + row * (N / 8));
        const device uchar* srow = scales + row * (N / 16);
        float d = 0.0f;
        for (uint c = l4; c < (uint)CH; c += LPD) {
            const uint4 q = qrow[c];
            const float s0 = s_lut[srow[2 * c]], s1 = s_lut[srow[2 * c + 1]];
            const uint b = (slot * N + c * 32) / 4;
            const float p0 = ink_dot8(q.x, float4(hs[b]), float4(hs[b + 1]), c_lut)
                           + ink_dot8(q.y, float4(hs[b + 2]), float4(hs[b + 3]), c_lut);
            const float p1 = ink_dot8(q.z, float4(hs[b + 4]), float4(hs[b + 5]), c_lut)
                           + ink_dot8(q.w, float4(hs[b + 6]), float4(hs[b + 7]), c_lut);
            d += s0 * p0 + s1 * p1;
        }
        acc = metal::fma(rw[t * TOPK + slot] * escale[e], d, acc);
    }
    for (uint o = 1; o < (uint)LPD; o *= 2) acc += simd_shuffle_xor(acc, o);
    if (l4 == 0) y[(ulong)t * DM + dim] = (T)acc;
"""


@lru_cache(maxsize=None)
def _mid_kernel():
    return mx.fast.metal_kernel(
        name="mtplx_inkling2_nvfp4_mid",
        input_names=["x", "w", "scales", "escale", "experts"],
        output_names=["h"],
        header=_HEADER,
        source=_MID_SRC,
    )


@lru_cache(maxsize=None)
def _down_kernel():
    return mx.fast.metal_kernel(
        name="mtplx_inkling2_nvfp4_down",
        input_names=["h", "w", "scales", "escale", "experts", "rw"],
        output_names=["y"],
        header=_HEADER,
        source=_DOWN_SRC,
    )


def _env_int(name: str, default: int, allowed: tuple) -> int:
    try:
        v = int(os.environ.get(name, default))
    except ValueError:
        return default
    return v if v in allowed else default


def _int_softcap(v: float) -> int:
    if float(v) != int(v):
        raise ValueError(f"softcap {v} is not integral (kernel template takes ints)")
    return int(v)


def nvfp4_mid(x2, w, scales, escale, experts, *, gate_softcap: float, up_softcap: float):
    """x2 [M, K]; w uint32 [E, 2N, K/8]; scales uint8 [E, 2N, K/16]; escale
    float32 [E]; experts uint32 [M * topk]. Returns h [M * topk, N]."""
    m_rows, k = int(x2.shape[0]), int(x2.shape[1])
    n = int(w.shape[1]) // 2
    pairs = int(experts.shape[0])
    nr = _env_int("MTPLX_INKLING2_MOE_MID_NR", 2, (1, 2, 4))
    rows = 32 * nr
    (h,) = _mid_kernel()(
        inputs=[x2, w, scales, escale, experts],
        template=[
            ("T", x2.dtype),
            ("K", k),
            ("N", n),
            ("TOPK", pairs // m_rows),
            ("NR", nr),
            ("GSOFT", _int_softcap(gate_softcap)),
            ("USOFT", _int_softcap(up_softcap)),
        ],
        grid=(((n + rows - 1) // rows) * 256, pairs, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(pairs, n)],
        output_dtypes=[x2.dtype],
    )
    return h


def nvfp4_down(h, w, scales, escale, experts, route_w, m_rows: int):
    """h [M * topk, N]; w uint32 [E, D, N/8]; route_w float32 [M * topk].
    Returns y [M, D] = sum over slots of route_w * escale * (down . h)."""
    n = int(h.shape[1])
    dmodel = int(w.shape[1])
    topk = int(experts.shape[0]) // m_rows
    lpd = _env_int("MTPLX_INKLING2_MOE_DOWN_LPD", 4, (2, 4, 8))
    dpt = 256 // lpd
    (y,) = _down_kernel()(
        inputs=[h, w, scales, escale, experts, route_w],
        template=[("T", h.dtype), ("N", n), ("DM", dmodel), ("TOPK", topk), ("LPD", lpd)],
        grid=(((dmodel + dpt - 1) // dpt) * 256, m_rows, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(m_rows, dmodel)],
        output_dtypes=[h.dtype],
    )
    return y


def moe_nvfp4(x2, gate_up, down, experts, route_w, *, gate_softcap: float, up_softcap: float):
    """Routed experts for ExpertLinear pairs (nvfp4 with outer scales)."""
    h = nvfp4_mid(
        x2,
        gate_up["weight"],
        gate_up["scales"],
        gate_up["expert_scale"],
        experts,
        gate_softcap=gate_softcap,
        up_softcap=up_softcap,
    )
    return nvfp4_down(
        h,
        down["weight"],
        down["scales"],
        down["expert_scale"],
        experts,
        route_w,
        int(x2.shape[0]),
    )
