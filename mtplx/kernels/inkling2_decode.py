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
