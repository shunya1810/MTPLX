"""Hyper-connection (GatedResidual) read for 1..4 decode/verify rows in two
dispatches over a kernel-private 8-bit pack of the mixer weights.

mtplx/kernels/hc_read_rows.py reads the fp16 mixes in four dependent
dispatches (norm, down, finalize, up). At Flash-Next's width (4 x 2560, low
rank 320) on an M1 Max a 1-row read costs ~100 us: ~40 of it the three
dispatch boundaries, the rest two fp16 GEMVs (13 MB). Here:

  D1  per (stream g, 32 down/inject rows): the pending residual write
      (hyper + block_out * inject, written out by the first row block) and
      x * gamma staged once in threadgroup memory, the stream's sum of
      squares, and the projections onto the folded [R + HC, HC * D]
      down+inject pack, unscaled: the grouped RMS scale of stream g
      multiplies the partial in D2.
  D2  per 32 hidden positions: the RMS scales, mix = silu(down / hc) and
      inject = 2 sigmoid(inject_logits / hc) from the partials (the first
      threadgroup writes inject), then each stream's up row (8-bit),
      sigmoid, x normed, and the mean over streams.

The pack is mx.quantize affine 8-bit g64 of the module's fp16 weights, made
once per module (the fp16 weights stay for prefill). 8-bit mixes measured
KLD-clean upstream (0.0003, hyper_connection_v3); Splash serves Flash-Next's
mixes as Q8_0. Rounding points follow the eager chain where they are cheap
(projection outputs, /hc, silu, sigmoid, the product, the mean); the norm is
applied after the dot, so results differ from the eager chain by a few ulp.
"""

from __future__ import annotations

import os
from functools import lru_cache

import mlx.core as mx

MAX_ROWS = 4
_NSG = 8          # simdgroups per threadgroup
_NR = int(os.environ.get("MTPLX_HC_Q8_NR", "4"))   # D1 output rows per simdgroup
_ND = int(os.environ.get("MTPLX_HC_Q8_ND", "4"))   # D2 hidden positions per simdgroup

_D1_SRC = r"""
    // Threadgroup = one stream g x NSG * NR output rows. The stream's rows
    // (pending write applied, written out by the first row block) are staged
    // once as x * gamma with their 16-element sums; simdgroups then dot their
    // NR pack rows against them.
    constexpr uint NOUT = R + HC;
    constexpr uint HCD = HC * D;
    constexpr uint NV = D / 16;                // 16-element vectors per stream row
    constexpr uint RPT = NSG * NR;             // output rows per threadgroup
    threadgroup T xg[S * D];
    threadgroup float xsum[S * NV];
    threadgroup float ssp[S * NSG];
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint t = thread_index_in_threadgroup;
    const uint g = threadgroup_position_in_grid.y;
    const uint blk = threadgroup_position_in_grid.x;

    float ss[S];
    for (int s = 0; s < S; ++s) ss[s] = 0.0f;
    for (uint u = t; u < (uint)(S * NV); u += NSG * 32) {
        const uint s = u / NV;
        const uint e0 = (u % NV) * 16;
        const uint col = g * D + e0;
        const device vec<T, 4>* xv4 = (const device vec<T, 4>*)(x + (size_t)s * HCD + col);
        const device vec<T, 4>* gv4 = (const device vec<T, 4>*)(gamma + col);
        const device vec<T, 4>* bv4 = (const device vec<T, 4>*)(bo + (size_t)s * D + e0);
        const float ij = WRITE ? (float)inj[s * HC + g] : 0.0f;
        float sum = 0.0f;
        float sq = 0.0f;
        for (int q = 0; q < 4; ++q) {
            vec<T, 4> v = xv4[q];
            if (WRITE) {
                const vec<T, 4> prod = vec<T, 4>(float4(bv4[q]) * ij);
                v = vec<T, 4>(float4(v) + float4(prod));
                if (blk == 0) ((device vec<T, 4>*)(hyper_out + (size_t)s * HCD + col))[q] = v;
            }
            const float4 f = float4(v);
            sq += dot(f, f);
            const vec<T, 4> gv = vec<T, 4>(f * float4(gv4[q]));
            ((threadgroup vec<T, 4>*)(xg + s * D + e0))[q] = gv;
            const float4 gf = float4(gv);
            sum += gf.x + gf.y + gf.z + gf.w;
        }
        for (int s2 = 0; s2 < S; ++s2) if (s2 == (int)s) ss[s2] += sq;
        xsum[u] = sum;
    }
    if (blk == 0) {
        for (int s = 0; s < S; ++s) {
            const float a = simd_sum(ss[s]);
            if (lane == 0) ssp[s * NSG + sg] = a;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (blk == 0 && t < (uint)S) {
        float a = 0.0f;
        for (uint i = 0; i < (uint)NSG; ++i) a += ssp[t * NSG + i];
        sumsq[g * S + t] = a;
    }

    const uint row0 = blk * RPT + sg * NR;
    float acc[NR][S];
    for (int r = 0; r < NR; ++r)
        for (int s = 0; s < S; ++s) acc[r][s] = 0.0f;
    for (uint v = lane; v < NV; v += 32) {
        const uint col = g * D + v * 16;
        for (int r = 0; r < NR; ++r) {
            const uint n = row0 + r;
            if (n >= NOUT) break;
            const uint4 wv = ((device const uint4*)(wq + (size_t)n * (HCD / 4)))[col / 16];
            const float sc = (float)wsc[(size_t)n * (HCD / 64) + col / 64];
            const float bi = (float)wbi[(size_t)n * (HCD / 64) + col / 64];
            const thread uchar* q = (const thread uchar*)&wv;
            for (int s = 0; s < S; ++s) {
                const threadgroup T* xv = xg + s * D + v * 16;
                float d_ = 0.0f;
                for (int k = 0; k < 16; ++k) d_ = metal::fma((float)q[k], (float)xv[k], d_);
                acc[r][s] += sc * d_ + bi * xsum[s * NV + v];
            }
        }
    }
    for (int r = 0; r < NR; ++r) {
        const uint n = row0 + r;
        if (n >= NOUT) break;
        for (int s = 0; s < S; ++s) {
            const float a = simd_sum(acc[r][s]);
            if (lane == 0) partial[((size_t)g * S + s) * NOUT + n] = a;
        }
    }
"""

_D2_SRC = r"""
    constexpr uint NOUT = R + HC;
    constexpr uint HCD = HC * D;
    constexpr uint RV = R / 8;                 // 8-byte up-weight vectors per row
    constexpr uint VPL = RV / 8;               // per lane (8 lanes a stream)
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint t = thread_index_in_threadgroup;
    threadgroup float scale_s[HC * S];
    threadgroup float mixs[S * R];
    if (t < (uint)(HC * S)) {
        scale_s[t] = metal::rsqrt(sumsq[t] / (float)D + eps_arr[0]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = t; i < (uint)(S * NOUT); i += NSG * 32) {
        const uint s = i / NOUT;
        const uint n = i % NOUT;
        float a = 0.0f;
        for (uint g = 0; g < (uint)HC; ++g) a += scale_s[g * S + s] * partial[((size_t)g * S + s) * NOUT + n];
        const float lin = (float)((T)a);
        const float div = (float)((T)(lin / (float)HC));
        const float sig = 1.0f / (1.0f + metal::exp(-div));
        if (n < (uint)R) {
            mixs[s * R + n] = (float)((T)(div * sig));
        } else if (threadgroup_position_in_grid.x == 0) {
            inject[(size_t)s * HC + (n - R)] = (T)(2.0f * (float)((T)sig));
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint g = lane / 8;                   // stream of this lane
    const uint l8 = lane % 8;
    const uint d0 = (threadgroup_position_in_grid.x * NSG + sg) * ND;
    for (int nd = 0; nd < ND; ++nd) {
        const uint d = d0 + nd;
        if (d >= (uint)D) return;
        const size_t row = (size_t)g * D + d;
        float acc[S];
        for (int s = 0; s < S; ++s) acc[s] = 0.0f;
        for (uint i = 0; i < VPL; ++i) {
            const uint v = l8 + 8 * i;
            const uint2 wv = ((device const uint2*)(uq + row * (R / 4)))[v];
            const float sc = (float)usc[row * (R / 64) + (v * 8) / 64];
            const float bi = (float)ubi[row * (R / 64) + (v * 8) / 64];
            const thread uchar* q = (const thread uchar*)&wv;
            for (int s = 0; s < S; ++s) {
                threadgroup const float* m = mixs + s * R + v * 8;
                float qd = 0.0f, ms = 0.0f;
                for (int k = 0; k < 8; ++k) {
                    qd = metal::fma((float)q[k], m[k], qd);
                    ms += m[k];
                }
                acc[s] += sc * qd + bi * ms;
            }
        }
        float out[S];
        for (int s = 0; s < S; ++s) {
            float a = acc[s];
            a += simd_shuffle_xor(a, 1);
            a += simd_shuffle_xor(a, 2);
            a += simd_shuffle_xor(a, 4);
            const float lin = (float)((T)a);
            const float w = (float)((T)(1.0f / (1.0f + metal::exp(-lin))));
            T v = WRITE ? hyper_out[(size_t)s * HCD + row] : x[(size_t)s * HCD + row];
            const T nv = (T)((float)v * scale_s[g * S + s]);
            const T normed = (T)((float)nv * (float)gamma[row]);
            float p = (float)((T)(w * (float)normed));
            p += simd_shuffle_xor(p, 8);
            p += simd_shuffle_xor(p, 16);
            out[s] = p;
        }
        if (lane == 0) {
            for (int s = 0; s < S; ++s) mixed[(size_t)s * D + d] = (T)(out[s] / (float)HC);
        }
    }
"""


@lru_cache(maxsize=8)
def _eps(eps: float) -> mx.array:
    return mx.array([eps], dtype=mx.float32)


@lru_cache(maxsize=None)
def _d1_kernel(write: bool):
    return mx.fast.metal_kernel(
        name=f"mtplx_hc_q8_d1_{'w' if write else 'r'}",
        input_names=["x", "gamma", "bo", "inj", "wq", "wsc", "wbi"],
        output_names=["partial", "sumsq", "hyper_out"],
        source=_D1_SRC,
    )


@lru_cache(maxsize=None)
def _d2_kernel(write: bool):
    return mx.fast.metal_kernel(
        name=f"mtplx_hc_q8_d2_{'w' if write else 'r'}",
        input_names=["partial", "sumsq", "x", "hyper_out", "gamma", "uq", "usc", "ubi", "eps_arr"],
        output_names=["mixed", "inject"],
        source=_D2_SRC,
    )


def enabled() -> bool:
    """``MTPLX_HC_READ_Q8``: default on (M1 family through hc_read_rows' gate); 0 opts out."""
    raw = (os.environ.get("MTPLX_HC_READ_Q8") or "").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def supports(module, hyper_input: mx.array) -> bool:
    """The geometry the two kernels compile: rows 1..4, stream width a multiple
    of 512 (whole 16-byte vectors per lane), low rank a multiple of 64."""
    if "block_inject_weight" not in module:
        return False
    rows = 1
    for s in hyper_input.shape[:-1]:
        rows *= int(s)
    if not 1 <= rows <= MAX_ROWS:
        return False
    dt = hyper_input.dtype
    if dt not in (mx.float16, mx.bfloat16):
        return False
    hc, d = int(module.hc_count), int(module.hidden_size)
    if hc != 4:  # D2 gives each stream 8 lanes of a simdgroup
        return False
    wd = module.input_mix_weight_down
    for lin in (wd, module.input_mix_weight_up, module.block_inject_weight):
        if hasattr(lin, "scales") or "bias" in lin:
            return False
    r = int(wd.weight.shape[0])
    return hyper_input.shape[-1] == hc * d and d % 512 == 0 and r % 64 == 0 and r <= 768


def _pack(module):
    pack = getattr(module, "_hc_q8_pack", None)
    if pack is None:
        down = mx.concatenate(
            [module.input_mix_weight_down.weight, module.block_inject_weight.weight], axis=0
        )
        dq = mx.quantize(down, group_size=64, bits=8)
        uq = mx.quantize(module.input_mix_weight_up.weight, group_size=64, bits=8)
        pack = tuple(dq) + tuple(uq)
        mx.eval(*pack)
        object.__setattr__(module, "_hc_q8_pack", pack)
    return pack


def hc_read_q8(module, hyper_input: mx.array, pending=None):
    """(mixed, written hyper, inject) for 1..MAX_ROWS rows; ``pending`` = (block_out, inject)."""
    lead = hyper_input.shape[:-1]
    rows = 1
    for s in lead:
        rows *= int(s)
    hc, d = int(module.hc_count), int(module.hidden_size)
    hcd = hc * d
    dt = hyper_input.dtype
    wq, wsc, wbi, uq, usc, ubi = _pack(module)
    r = int(module.input_mix_weight_down.weight.shape[0])
    n_out = r + hc
    x2 = hyper_input.reshape(rows, hcd)
    write = pending is not None
    if write:
        bo2 = pending[0].reshape(rows, d).astype(dt)
        inj2 = pending[1].reshape(rows, hc).astype(dt)
    else:
        bo2 = inj2 = x2  # unused
    gamma = module.hc_norm.weight
    rpt = _NSG * _NR
    partial, sumsq, hyper_out = _d1_kernel(write)(
        inputs=[x2, gamma, bo2, inj2, wq, wsc, wbi],
        template=[("T", dt), ("HC", hc), ("D", d), ("R", r), ("S", rows), ("NSG", _NSG),
                  ("NR", _NR), ("WRITE", int(write))],
        grid=(32 * _NSG * ((n_out + rpt - 1) // rpt), hc, 1),
        threadgroup=(32 * _NSG, 1, 1),
        output_shapes=[(hc * rows * n_out,), (hc * rows,), (rows, hcd) if write else (1,)],
        output_dtypes=[mx.float32, mx.float32, dt],
    )
    per_tg = _NSG * _ND
    mixed, inject = _d2_kernel(write)(
        inputs=[partial, sumsq, x2, hyper_out, gamma, uq, usc, ubi, _eps(float(module.hc_norm.eps))],
        template=[("T", dt), ("HC", hc), ("D", d), ("R", r), ("S", rows), ("NSG", _NSG),
                  ("ND", _ND), ("WRITE", int(write))],
        grid=(32 * _NSG * ((d + per_tg - 1) // per_tg), 1, 1),
        threadgroup=(32 * _NSG, 1, 1),
        output_shapes=[(rows, d), (rows, hc)],
        output_dtypes=[dt, dt],
    )
    written = hyper_out.reshape(hyper_input.shape) if write else hyper_input
    return mixed.reshape(*lead, d), written, inject.reshape(*lead, hc)
