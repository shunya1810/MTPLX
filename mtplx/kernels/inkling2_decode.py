"""One-token (decode / small verify) kernels for the Inkling 2 trunk.

Every kernel here replaces a chain of small MLX ops that each cost a
dependent dispatch (~7.6 us on M1 Max) for a few microseconds of work.

* message gates (``message_gates``): G gates at once, two kernels
  (down projection with the two unweighted RMS norms folded in; then the
  forget / message projections, tanh and the mix).
* short convolution step (``conv_step``): window = (state, x), depthwise
  conv in the weight dtype, residual, scale, and the new state, one kernel.
"""

from __future__ import annotations

from functools import lru_cache

import mlx.core as mx

__all__ = ["message_gates", "conv_step"]


# --------------------------------------------------------------------------
# Message gates
#
# MessageGate(h, m): s = Wd . fp16(concat(rms(h), rms(m)))   (fp16 Linear)
#                    out = (1 - tanh(Wf . s)) * h + tanh(Wg . s) * m
# rms is the unweighted RMS norm (eps 1e-6), rounded to h's dtype for the h
# half and to float16 for both halves before the projection (the reference
# concatenates in float32 and casts to the down weight's float16).

_GATE_DOWN_SRC = r"""
    // grid: (RB-row blocks of R) x G ; threadgroup 256
    const uint tid = thread_position_in_threadgroup.x;
    const uint g = threadgroup_position_in_grid.y;
    const uint r0 = threadgroup_position_in_grid.x * RB;
    const uint lane = tid % 32, sg = tid / 32;
    const device TH* hg = h + (ulong)g * D;
    const device float* mg = m + (ulong)g * D;
    threadgroup float red[2][8];
    float sh = 0.0f, sm = 0.0f;
    for (uint i = tid; i < (uint)D; i += 256) {
        const float a = (float)hg[i];
        const float b = mg[i];
        sh += a * a;
        sm += b * b;
    }
    sh = simd_sum(sh);
    sm = simd_sum(sm);
    if (lane == 0) { red[0][sg] = sh; red[1][sg] = sm; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sh = 0.0f; sm = 0.0f;
    for (uint i = 0; i < 8; ++i) { sh += red[0][i]; sm += red[1][i]; }
    const float eps_f = (float)EPS * 1e-9f;
    const float ih = metal::precise::rsqrt(sh / (float)D + eps_f);
    const float im = metal::precise::rsqrt(sm / (float)D + eps_f);
    // each simdgroup owns RB / 8 rows
    for (uint rr = sg; rr < (uint)RB; rr += 8) {
        const uint r = r0 + rr;
        if (r >= (uint)R) break;
        const device half* w = wd + ((ulong)g * R + r) * (2 * D);
        float acc = 0.0f;
        for (uint i = lane; i < (uint)D; i += 32) {
            const half xh = (half)(TH)((float)hg[i] * ih);
            const half xm = (half)(mg[i] * im);
            acc += (float)w[i] * (float)xh + (float)w[D + i] * (float)xm;
        }
        acc = simd_sum(acc);
        if (lane == 0) s[(ulong)g * R + r] = (float)(half)acc;
    }
"""

_GATE_MIX_SRC = r"""
    // grid: (DB-dim blocks of D) x G ; threadgroup 256 = 8 simdgroups, each
    // simdgroup owns DB / 8 dims; lanes split the R = 64 inner products.
    const uint tid = thread_position_in_threadgroup.x;
    const uint g = threadgroup_position_in_grid.y;
    const uint lane = tid % 32, sg = tid / 32;
    const device float* sg_s = s + (ulong)g * R;
    float sv[R / 32];
    for (uint k = 0; k < (uint)(R / 32); ++k) sv[k] = sg_s[lane + 32 * k];
    const uint d0 = threadgroup_position_in_grid.x * DB;
    for (uint dd = sg; dd < (uint)DB; dd += 8) {
        const uint d = d0 + dd;
        if (d >= (uint)D) break;
        const device float* wfr = wf + ((ulong)g * D + d) * R;
        const device float* wgr = wg + ((ulong)g * D + d) * R;
        float f = 0.0f, q = 0.0f;
        for (uint k = 0; k < (uint)(R / 32); ++k) {
            f += wfr[lane + 32 * k] * sv[k];
            q += wgr[lane + 32 * k] * sv[k];
        }
        f = simd_sum(f);
        q = simd_sum(q);
        if (lane == 0) {
            const float hv = (float)h[(ulong)g * D + d];
            const float mv = m[(ulong)g * D + d];
            out[(ulong)g * D + d] = (1.0f - metal::precise::tanh(f)) * hv + metal::precise::tanh(q) * mv;
        }
    }
"""


@lru_cache(maxsize=None)
def _gate_down_kernel():
    return mx.fast.metal_kernel(
        name="mtplx_inkling2_gate_down",
        input_names=["h", "m", "wd"],
        output_names=["s"],
        source=_GATE_DOWN_SRC,
    )


@lru_cache(maxsize=None)
def _gate_mix_kernel():
    return mx.fast.metal_kernel(
        name="mtplx_inkling2_gate_mix",
        input_names=["s", "wf", "wg", "h", "m"],
        output_names=["out"],
        source=_GATE_MIX_SRC,
    )


def message_gates(h: mx.array, m: mx.array, wd: mx.array, wf: mx.array, wg: mx.array, *, eps: float = 1e-6) -> mx.array:
    """h [G, D] (float16 / float32), m [G, D] float32, wd [G, R, 2D] float16,
    wf / wg [G, D, R] float32. Returns [G, D] float32."""
    G, D = int(h.shape[0]), int(h.shape[1])
    R = int(wd.shape[1])
    if R % 32 or wf.dtype != mx.float32 or wg.dtype != mx.float32 or wd.dtype != mx.float16:
        raise ValueError("message_gates: unsupported weight layout")
    m = m.astype(mx.float32)
    RB = 8
    (s,) = _gate_down_kernel()(
        inputs=[h, m, wd],
        template=[("TH", h.dtype), ("D", D), ("R", R), ("RB", RB), ("EPS", _f(eps))],
        grid=(((R + RB - 1) // RB) * 256, G, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(G, R)],
        output_dtypes=[mx.float32],
    )
    DB = 32
    (out,) = _gate_mix_kernel()(
        inputs=[s, wf, wg, h, m],
        template=[("D", D), ("R", R), ("DB", DB)],
        grid=(((D + DB - 1) // DB) * 256, G, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(G, D)],
        output_dtypes=[mx.float32],
    )
    return out


def _f(value: float) -> int:
    # metal_kernel templates take int / bool / dtype: eps travels as an
    # integer count of 1e-9 (the kernel scales it back).
    return int(round(value * 1e9))


# --------------------------------------------------------------------------
# Short convolution step

_CONV_SRC = r"""
    // one thread per channel: window = (state[0..2], x); depthwise conv in
    // the weight dtype; out = (conv + x) / sqrt(K + 1); new state = window[1..3].
    const uint c = thread_position_in_grid.x;
    if (c >= (uint)C) return;
    float win[4];
    for (int k = 0; k < 3; ++k) win[k] = state[(ulong)k * C + c];
    win[3] = (float)x[c];
    float acc = 0.0f;
    for (int k = 0; k < 4; ++k) acc += (float)w[(ulong)c * 4 + k] * (float)(WT)win[k];
    const float conv = (float)(WT)acc;
    out[c] = (TX)((conv + win[3]) * 0.4472135954999579f);
    for (int k = 0; k < 3; ++k) new_state[(ulong)k * C + c] = win[k + 1];
"""


@lru_cache(maxsize=None)
def _conv_kernel():
    return mx.fast.metal_kernel(
        name="mtplx_inkling2_conv_step",
        input_names=["x", "state", "w"],
        output_names=["out", "new_state"],
        source=_CONV_SRC,
    )


def conv_step(x: mx.array, state: mx.array, weight: mx.array) -> tuple[mx.array, mx.array]:
    """x [C] / [1,1,C]; state [1,3,C] float32; weight [C, 4, 1] (depthwise
    conv1d). Returns (out shaped like x, new state [1,3,C] float32)."""
    C = int(x.shape[-1])
    w = weight.reshape(C, 4)
    out, new_state = _conv_kernel()(
        inputs=[x.reshape(C), state.reshape(3, C), w],
        template=[("TX", x.dtype), ("WT", weight.dtype), ("C", C)],
        grid=(((C + 255) // 256) * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(C,), (3, C)],
        output_dtypes=[x.dtype, mx.float32],
    )
    return out.reshape(x.shape), new_state.reshape(1, 3, C)


# --------------------------------------------------------------------------
# Attention prologue (one token): everything between the projections and the
# KV cache in one kernel.
#   q: scaled L2 clip, then the log-scaling tau          -> (HQ, D)
#   k: [short conv step] then L2 clip                     -> (HKV, D)
#   v: [short conv step] then unweighted RMS norm         -> (HKV, D)
#   r: scaled L2 clip, tau, then r @ rel_proj             -> bias (HQ, R)
# Threadgroups (128 threads): [0, HQ) q heads, [HQ, HQ+HKV) k heads,
# [HQ+HKV, HQ+2HKV) v heads, then HQ bias rows. params = (qk_th, rel_th,
# eps, tau). Rounds where the MLX reference rounds (each op returns the
# input dtype T).

_PREP_SRC = r"""
    constexpr float RSQ5 = 0.4472135954999579f;
    const uint tg = threadgroup_position_in_grid.x;
    const uint tid = thread_position_in_threadgroup.x;
    const uint lane = tid % 32, sg = tid / 32;
    threadgroup float red[4];
    const float qk_th = params[0], rel_th = params[1], eps = params[2], tau = params[3];
    auto tg_sum = [&](float v) -> float {
        v = simd_sum(v);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0) red[sg] = v;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        return red[0] + red[1] + red[2] + red[3];
    };
    if (tg < (uint)HQ) {
        const uint h = tg, d = tid;               // D == 128 == threads
        const float s = (float)q[h * D + d] * (float)qscale[h * D + d];
        const float n = metal::precise::sqrt(tg_sum(s * s));
        float o = (float)(T)(s * metal::min(qk_th / (n + 1e-6f), 1.0f));
        if (TAU) o = (float)(T)(o * tau);
        q_out[h * D + d] = (T)o;
    } else if (tg < (uint)(HQ + HKV)) {
        const uint h = tg - HQ, d = tid, c = h * D + d;
        float x = (float)k[c];
        if (CONV) {
            float win[4];
            for (int j = 0; j < 3; ++j) win[j] = kstate[(ulong)j * (HKV * D) + c];
            win[3] = x;
            float acc = 0.0f;
            for (int j = 0; j < 4; ++j) acc += (float)kw[(ulong)c * 4 + j] * (float)(T)win[j];
            x = (float)(T)(((float)(T)acc + win[3]) * RSQ5);
            for (int j = 0; j < 3; ++j) kstate_new[(ulong)j * (HKV * D) + c] = win[j + 1];
        }
        const float n = metal::precise::sqrt(tg_sum(x * x));
        k_out[c] = (T)(x * metal::min(qk_th / (n + 1e-6f), 1.0f));
    } else if (tg < (uint)(HQ + 2 * HKV)) {
        const uint h = tg - HQ - HKV, d = tid, c = h * D + d;
        float x = (float)v[c];
        if (CONV) {
            float win[4];
            for (int j = 0; j < 3; ++j) win[j] = vstate[(ulong)j * (HKV * D) + c];
            win[3] = x;
            float acc = 0.0f;
            for (int j = 0; j < 4; ++j) acc += (float)vw[(ulong)c * 4 + j] * (float)(T)win[j];
            x = (float)(T)(((float)(T)acc + win[3]) * RSQ5);
            for (int j = 0; j < 3; ++j) vstate_new[(ulong)j * (HKV * D) + c] = win[j + 1];
        }
        const float ms = tg_sum(x * x) / (float)D;
        v_out[c] = (T)(x * metal::precise::rsqrt(ms + eps));
    } else {
        const uint h = tg - HQ - 2 * HKV;
        // clipped relative row (DR = 16 values), every thread keeps a copy
        float rr[DR];
        float ss = 0.0f;
        for (int i = 0; i < DR; ++i) {
            rr[i] = (float)r[h * DR + i] * (float)rscale[h * DR + i];
            ss += rr[i] * rr[i];
        }
        const float f = metal::min(rel_th / (metal::precise::sqrt(ss) + 1e-6f), 1.0f);
        for (int i = 0; i < DR; ++i) {
            float o = (float)(T)(rr[i] * f);
            if (TAU) o = (float)(T)(o * tau);
            rr[i] = o;
        }
        for (uint j = tid; j < (uint)R; j += 128) {
            float acc = 0.0f;
            for (int i = 0; i < DR; ++i) acc += rr[i] * (float)relp[(ulong)i * R + j];
            pb[(ulong)h * R + j] = (T)acc;
        }
    }
"""


@lru_cache(maxsize=None)
def _prep_kernel():
    return mx.fast.metal_kernel(
        name="mtplx_inkling2_attn_prep",
        input_names=["q", "k", "v", "r", "qscale", "rscale", "relp", "params", "kstate", "vstate", "kw", "vw"],
        output_names=["q_out", "k_out", "v_out", "pb", "kstate_new", "vstate_new"],
        source=_PREP_SRC,
    )


def attn_prep(
    q, k, v, r, *, qscale, rscale, relp, qk_th, rel_th, eps, tau=None,
    kstate=None, vstate=None, kw=None, vw=None,
):
    """One token. q (HQ*D,), k / v (HKV*D,), r (HQ*DR,) in dtype T; qscale
    (HQ, D), rscale (HQ, DR), relp (DR, R). With kstate/vstate/kw/vw the k/v
    short conv step runs first (sliding layers). Returns q (HQ, D), k (HKV, D),
    v (HKV, D), bias (HQ, R), and the new conv states (or None)."""
    HQ, D = int(qscale.shape[0]), int(qscale.shape[1])
    DR, R = int(relp.shape[0]), int(relp.shape[1])
    HKV = int(k.size) // D
    if D != 128:
        raise ValueError("attn_prep expects head_dim 128")
    T = q.dtype
    conv = kstate is not None
    tau_v = 1.0 if tau is None else tau
    params = mx.array([qk_th, rel_th, eps, 0.0], dtype=mx.float32)
    if tau is not None:
        params = mx.concatenate([params[:3], mx.reshape(mx.array(tau_v, mx.float32) if not isinstance(tau_v, mx.array) else tau_v.astype(mx.float32), (1,))])
    dummy = mx.zeros((1,), mx.float32)
    dummy_w = mx.zeros((1,), T)
    outs = _prep_kernel()(
        inputs=[
            q.reshape(-1), k.reshape(-1), v.reshape(-1), r.reshape(-1), qscale, rscale, relp, params,
            kstate.reshape(-1) if conv else dummy,
            vstate.reshape(-1) if conv else dummy,
            kw.reshape(-1) if conv else dummy_w,
            vw.reshape(-1) if conv else dummy_w,
        ],
        template=[("T", T), ("HQ", HQ), ("HKV", HKV), ("D", D), ("DR", DR), ("R", R),
                  ("CONV", int(conv)), ("TAU", int(tau is not None))],
        grid=((2 * HQ + 2 * HKV) * 128, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(HQ, D), (HKV, D), (HKV, D), (HQ, R), (3, HKV * D) if conv else (1,), (3, HKV * D) if conv else (1,)],
        output_dtypes=[T, T, T, T, mx.float32, mx.float32],
    )
    q_out, k_out, v_out, pb, ks, vs = outs
    return q_out, k_out, v_out, pb, (ks if conv else None), (vs if conv else None)
