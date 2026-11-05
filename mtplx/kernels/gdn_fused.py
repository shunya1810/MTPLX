"""One kernel for the A3B GDN layer between its two projections (2-row verify).

The stock verify chain after the fused in-projection is: four contiguous
split copies, the conv-state concatenate, the depthwise conv, SiLU, the
inline-g recurrence, the gated RMSNorm and the SiLU(z) gate - eight dependent
dispatches per GDN layer. On the M1 lane a dependent dispatch costs ~4.5 us
whatever its work, and these eight do almost none.

Here one threadgroup owns one value head: it runs the conv + SiLU for the
head's q/k/v channels straight off the fused projection output, the q/k
RMSNorm, the recurrence for all 128 value rows (two tokens), and the gated
RMSNorm, which only needs the head's own 128 outputs. Every value keeps the
stock arithmetic: the conv accumulates its four taps in float in tap order,
SiLU and sigmoid use MLX's unary formulas in the stock dtypes, the recurrence
is the inline-g kernel's per-row code, and the gated norm reduces each row
exactly like MLX's rms_single_row (lane j squares elements 4j..4j+3 in order,
one simd_sum).

Two numerics/layout modes, each bit-identical to the chain it replaces:

- states mode (``gdn_fused_m2``): per-token states, g with fast ``exp`` - the
  A3B postconv kernels (inline_g / headquarter) of the target-prefix route.
- tape mode (``gdn_fused_m2_tape``): final state plus the replay tape, g with
  ``precise::exp`` like MLX's Exp op in mlx_lm's compute_g - the generic
  linear-gdn-from-conv-tape capture that the graphbank / resident verify runs.
"""

from __future__ import annotations

import os
from functools import lru_cache

import mlx.core as mx

_ENV = "MTPLX_A3B_GDN_FUSED"
_NSG = 16  # simdgroups per threadgroup; 128 / _NSG value rows each

_HEADER = """
    template <typename T>
    inline T mlx_sigmoid(T x) {
        auto y = 1 / (1 + metal::exp(metal::abs(x)));
        return (x < 0) ? y : 1 - y;
    }
"""

_SOURCE = """
    constexpr int Dk = 128;
    constexpr int Dv = 128;
    constexpr int Hk = 16;
    constexpr int Hv = 32;
    constexpr int KeyDim = 2048;
    constexpr int ConvDim = 8192;
    constexpr int PW = 12352;
    constexpr int Z_OFF = 8192;
    constexpr int B_OFF = 12288;
    constexpr int A_OFF = 12320;
    constexpr int T = 2;
    constexpr int KW = 4;
    constexpr int KEEP = 3;
    constexpr int n_per_t = Dk / 32;
    constexpr int NT = 32 * NSG;
    constexpr int DVS = Dv / NSG;

    const uint lane = thread_position_in_threadgroup.x;
    const uint sg = thread_position_in_threadgroup.y;
    const uint tid = sg * 32 + lane;
    const uint hv = threadgroup_position_in_grid.z;
    const uint hk = hv / (Hv / Hk);

    threadgroup InT qc[T][Dk];
    threadgroup InT kc[T][Dk];
    threadgroup InT vc[T][Dv];
    threadgroup float qn[T][Dk];
    threadgroup float kn[T][Dk];
    threadgroup float g_sh[T];
    threadgroup float beta_sh[T];
    threadgroup InT yraw[T][Dv];

    // conv + SiLU for the head's q, k (shared with the paired value head) and v channels
    for (uint c = tid; c < 3 * 128; c += NT) {
        uint part = c / 128;
        uint i = c % 128;
        uint ch = part == 0 ? hk * Dk + i : (part == 1 ? KeyDim + hk * Dk + i : 2 * KeyDim + hv * Dv + i);
        InT in[KEEP + T];
        for (int j = 0; j < KEEP; ++j) in[j] = conv_state[j * ConvDim + ch];
        for (int t = 0; t < T; ++t) in[KEEP + t] = proj[t * PW + ch];
        for (int t = 0; t < T; ++t) {
            float acc = 0.0f;
            for (int k = 0; k < KW; ++k) acc += float(in[t + k]) * float(conv_w[ch * KW + k]);
            InT cv = static_cast<InT>(acc);
            InT o = cv * mlx_sigmoid(cv);
            if (part == 0) qc[t][i] = o; else if (part == 1) kc[t][i] = o; else vc[t][i] = o;
        }
        if (part == 2 || (hv % (Hv / Hk)) == 0) {
            for (int t = 0; t < T; ++t)
                for (int j = 0; j < KEEP; ++j)
                    conv_states[(t * KEEP + j) * ConvDim + ch] = in[t + 1 + j];
            TAPE_CONV
        }
    }
    if (tid < T) {
        uint t = tid;
        InT b_val = proj[t * PW + B_OFF + hv];
        auto beta_y = 1 / (1 + metal::exp(metal::abs(b_val)));
        InT beta_val = (b_val < InT(0)) ? beta_y : 1 - beta_y;
        InT a_val = proj[t * PW + A_OFF + hv] + dt_bias[hv];
        constexpr InT inf = metal::numeric_limits<InT>::infinity();
        InT maxval = metal::max(a_val, InT(0));
        InT minval = metal::min(a_val, InT(0));
        InT softplus_val = (minval == -inf || maxval == inf)
          ? maxval
          : (maxval + log1p(metal::exp(minval - maxval)));
        float decay_a = EXPF(float(A_log[hv]));
        beta_sh[t] = static_cast<float>(beta_val);
        g_sh[t] = EXPF(-decay_a * float(softplus_val));
        TAPE_G
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float inv_scale = 1.0f / metal::sqrt(float(Dk));
    float q_scale = inv_scale * inv_scale;
    float k_scale = static_cast<float>(static_cast<InT>(inv_scale));
    if (sg < T) {
        uint t = sg;
        float q_sum = 0.0f;
        float k_sum = 0.0f;
        float q_raw[n_per_t];
        float k_raw[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * lane + i;
            q_raw[i] = static_cast<float>(qc[t][s_idx]);
            k_raw[i] = static_cast<float>(kc[t][s_idx]);
            q_sum += q_raw[i] * q_raw[i];
            k_sum += k_raw[i] * k_raw[i];
        }
        q_sum = simd_sum(q_sum);
        k_sum = simd_sum(k_sum);
        float q_inv = metal::precise::rsqrt(q_sum / float(Dk) + 1.0e-6f);
        float k_inv = metal::precise::rsqrt(k_sum / float(Dk) + 1.0e-6f);
        for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * lane + i;
            auto q_norm = static_cast<InT>(q_raw[i] * q_inv);
            auto k_norm = static_cast<InT>(k_raw[i] * k_inv);
            qn[t][s_idx] = static_cast<float>(static_cast<InT>(static_cast<float>(q_norm) * q_scale));
            kn[t][s_idx] = static_cast<float>(static_cast<InT>(static_cast<float>(k_norm) * k_scale));
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int r = 0; r < DVS; ++r) {
        uint dv_idx = sg * DVS + r;
        const device StT* parent_state = state_in + (hv * Dv + dv_idx) * Dk;
        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) state[i] = static_cast<float>(parent_state[n_per_t * lane + i]);
        for (int t = 0; t < T; ++t) {
            float kv_mem = 0.0f;
            for (int i = 0; i < n_per_t; ++i) {
                auto s_idx = n_per_t * lane + i;
                state[i] = state[i] * g_sh[t];
                kv_mem += state[i] * kn[t][s_idx];
            }
            kv_mem = simd_sum(kv_mem);
            auto delta = (static_cast<float>(vc[t][dv_idx]) - kv_mem) * beta_sh[t];
            float out = 0.0f;
            for (int i = 0; i < n_per_t; ++i) {
                auto s_idx = n_per_t * lane + i;
                state[i] = state[i] + kn[t][s_idx] * delta;
                out += state[i] * qn[t][s_idx];
            }
            out = simd_sum(out);
            if (lane == 0) yraw[t][dv_idx] = static_cast<InT>(out);
            STEP_OUT
        }
        FINAL_OUT
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // gated RMSNorm: rms_single_row order, then _precise_swiglu in float
    if (sg < T) {
        uint t = sg;
        float acc = 0.0f;
        InT yv[4];
        for (int i = 0; i < 4; ++i) {
            yv[i] = yraw[t][4 * lane + i];
            float f = static_cast<float>(yv[i]);
            acc += f * f;
        }
        acc = simd_sum(acc);
        float inv = metal::precise::rsqrt(acc / float(Dv) + eps);
        for (int i = 0; i < 4; ++i) {
            uint idx = 4 * lane + i;
            InT normed = norm_w[idx] * static_cast<InT>(static_cast<float>(yv[i]) * inv);
            float gz = static_cast<float>(proj[t * PW + Z_OFF + hv * Dv + idx]);
            float gate = gz * mlx_sigmoid(gz);
            y[(t * Hv + hv) * Dv + idx] = static_cast<InT>(gate * static_cast<float>(normed));
        }
    }
"""


_STATUS = {"state": "off", "reason": None}


def enabled() -> bool:
    return (os.environ.get(_ENV) or "").strip().lower() in {"1", "true", "on", "yes"}


def status() -> dict:
    return dict(_STATUS)


def generic_route_enabled() -> bool:
    """The generic capture verify (graphbank / resident decode) takes the fused
    layer once the A3B install has checked the tape-mode kernel bit for bit."""

    return _STATUS.get("tape") == "ok" and enabled()


def generic_layer(gdn, inputs: mx.array, cache):
    hub = getattr(gdn.in_proj_qkv, "_hub", None)
    if hub is None:
        return None
    proj = hub.fused(inputs)
    if not eligible(gdn, proj, cache[0], cache[1]):
        return None
    return gdn_fused_m2_tape(gdn, proj, cache[0], cache[1])


_STATES_PARTS = {
    # Per-token states, g with fast exp: the A3B postconv kernels' arithmetic.
    "EXPF": "metal::exp",
    "TAPE_G": "",
    "TAPE_CONV": "",
    "STEP_OUT": """auto state_t = states + ((t * Hv + hv) * Dv + dv_idx) * Dk;
            for (int i = 0; i < n_per_t; ++i) state_t[n_per_t * lane + i] = static_cast<StT>(state[i]);""",
    "FINAL_OUT": "",
}
_TAPE_PARTS = {
    # Final state + replay tape, g through MLX's precise exp (mlx_lm compute_g):
    # the generic linear-gdn-from-conv-tape capture's arithmetic and layout.
    "EXPF": "metal::precise::exp",
    "TAPE_G": "g_out[t * Hv + hv] = g_sh[t];",
    "TAPE_CONV": """for (int t = 0; t < T; ++t)
                conv_out[t * ConvDim + ch] = part == 0 ? qc[t][i] : (part == 1 ? kc[t][i] : vc[t][i]);""",
    "STEP_OUT": "if (lane == 0) tape[(t * Hv + hv) * Dv + dv_idx] = delta;",
    "FINAL_OUT": """auto fin = final_state + (hv * Dv + dv_idx) * Dk;
        for (int i = 0; i < n_per_t; ++i) fin[n_per_t * lane + i] = static_cast<StT>(state[i]);""",
}


def _source(parts: dict) -> str:
    src = _SOURCE
    for key, value in parts.items():
        assert key in src, key
        src = src.replace(key, value)
    return src


_INPUTS = ["proj", "conv_state", "conv_w", "A_log", "dt_bias", "state_in", "norm_w", "eps"]


@lru_cache(maxsize=None)
def _kernel():
    return mx.fast.metal_kernel(
        name="mtplx_a3b_gdn_fused_m2",
        input_names=_INPUTS,
        output_names=["y", "states", "conv_states"],
        header=_HEADER,
        source=_source(_STATES_PARTS),
    )


@lru_cache(maxsize=None)
def _tape_kernel():
    return mx.fast.metal_kernel(
        name="mtplx_a3b_gdn_fused_m2_tape",
        input_names=_INPUTS,
        output_names=["y", "final_state", "conv_states", "conv_out", "g_out", "tape"],
        header=_HEADER,
        source=_source(_TAPE_PARTS),
    )


def eligible(gdn, proj: mx.array, conv_state: mx.array, state: mx.array) -> bool:
    return (
        tuple(proj.shape) == (1, 2, 12352)
        and proj.dtype == mx.float16
        and tuple(conv_state.shape) == (1, 3, 8192)
        and tuple(state.shape) == (1, 32, 128, 128)
        and state.dtype == mx.float32
        and tuple(gdn.conv1d.weight.shape) == (8192, 4, 1)
        and tuple(gdn.norm.weight.shape) == (128,)
    )


def gdn_fused_m2(gdn, proj: mx.array, conv_state: mx.array, state: mx.array):
    """``proj`` is the fused (qkv | z | b | a) projection of two tokens. Returns
    the gated-normed output ``[1, 2, 32, 128]``, the per-token recurrent states
    ``[1, 2, 32, 128, 128]`` and conv states ``[1, 2, 3, 8192]``."""

    y, states, conv_states = _kernel()(
        inputs=[proj, conv_state, gdn.conv1d.weight, gdn.A_log, gdn.dt_bias, state, gdn.norm.weight,
                float(gdn.norm.eps)],
        template=[("InT", mx.float16), ("StT", mx.float32), ("NSG", _NSG)],
        grid=(32, _NSG, 32),
        threadgroup=(32, _NSG, 1),
        output_shapes=[(1, 2, 32, 128), (1, 2, 32, 128, 128), (1, 2, 3, 8192)],
        output_dtypes=[mx.float16, mx.float32, mx.float16],
    )
    return y, states, conv_states


class FusedM2Layer:
    """One layer's M2 recurrence (calling it runs the plain inline-g kernel)
    that also carries the whole fused layer body for the A3B verify trace."""

    __slots__ = ("_impl", "_stock_layer")

    def __init__(self, impl, stock_layer):
        self._impl = impl
        self._stock_layer = stock_layer

    def __call__(self, *args, **kwargs):
        return self._impl(*args, **kwargs)

    def layer(self, gdn, inputs: mx.array, cache):
        B, S, _ = inputs.shape
        proj = gdn.in_proj_qkv._hub.fused(inputs)
        if not eligible(gdn, proj, cache[0], cache[1]):
            return self._stock_layer(gdn, inputs, cache, self._impl)
        out, states, conv_states = gdn_fused_m2(gdn, proj, cache[0], cache[1])
        cache[0] = mx.contiguous(conv_states[:, -1, :, :])
        cache[1] = states[:, -1, :, :, :]
        out = gdn.out_proj(out.reshape(B, S, -1))
        return out, {"conv_states": conv_states, "states": states}


def _selfcheck(gdn, impl, stock_conv) -> bool:
    """Fused kernel vs the stock chain on random inputs with the layer's weights."""

    k = mx.random.split(mx.random.key(7), 3)
    proj = (mx.random.normal((1, 2, 12352), key=k[0]) * 0.8).astype(mx.float16)
    conv_state = (mx.random.normal((1, 3, 8192), key=k[1]) * 0.8).astype(mx.float16)
    state = mx.random.normal((1, 32, 128, 128), key=k[2]) * 0.05
    qkv, z, b, a = mx.split(proj, [8192, 12288, 12320], axis=-1)
    conv_out, conv_states = stock_conv(qkv, conv_state, gdn)
    out, states = impl(conv_out, a, b, state)
    y_ref = gdn.norm(out, z.reshape(1, 2, 32, 128))
    y, st, cs = gdn_fused_m2(gdn, proj, conv_state, state)
    return bool(
        mx.array_equal(y, y_ref).item()
        and mx.array_equal(st, states).item()
        and mx.array_equal(cs, conv_states).item()
    )


def _selfcheck_tape(gdn, stock_conv, stock_tape) -> bool:
    """Tape-mode kernel vs the generic linear-gdn-from-conv-tape chain."""

    from mlx_lm.models.gated_delta import compute_g

    k = mx.random.split(mx.random.key(8), 3)
    proj = (mx.random.normal((1, 2, 12352), key=k[0]) * 0.8).astype(mx.float16)
    conv_state = (mx.random.normal((1, 3, 8192), key=k[1]) * 0.8).astype(mx.float16)
    state = mx.random.normal((1, 32, 128, 128), key=k[2]) * 0.05
    qkv, z, b, a = mx.split(proj, [8192, 12288, 12320], axis=-1)
    conv_out, conv_states = stock_conv(qkv, conv_state, gdn)
    g = compute_g(gdn.A_log, a, gdn.dt_bias)
    out, final_state, tape = stock_tape(conv_out, g, mx.sigmoid(b), state, gdn)
    y_ref = gdn.norm(out, z.reshape(1, 2, 32, 128))
    got = gdn_fused_m2_tape(gdn, proj, conv_state, state)
    ref = (y_ref, final_state, conv_states, conv_out, g.astype(mx.float32), tape)
    return all(mx.array_equal(x, y).item() for x, y in zip(got, ref))


def install_m2(gdns, implementations, *, stock_layer, stock_conv, stock_tape=None):
    """Wrap each layer's M2 implementation, or return them unchanged when the
    projections are not fused or the kernel does not reproduce the stock bits.
    With ``stock_tape`` the tape-mode kernel is checked too (``status()["tape"]``)."""

    _STATUS["tape"] = None
    try:
        hubs_ok = all(getattr(g.in_proj_qkv, "_hub", None) is not None for g in gdns)
        if not hubs_ok:
            _STATUS.update(state="skipped", reason="GDN in-projections are not fused")
            print("[gdn-fused] skipped: GDN in-projections are not fused", flush=True)
            return implementations
        if not _selfcheck(gdns[0], implementations[0], stock_conv):
            _STATUS.update(state="skipped", reason="selfcheck mismatch")
            print("[gdn-fused] skipped: selfcheck mismatch", flush=True)
            return implementations
    except Exception as exc:  # pragma: no cover - device/compile failures
        _STATUS.update(state="skipped", reason=f"{type(exc).__name__}: {exc}")
        print(f"[gdn-fused] skipped: {type(exc).__name__}: {exc}", flush=True)
        return implementations
    _STATUS.update(state="installed", reason=None)
    if stock_tape is not None:
        try:
            _STATUS["tape"] = "ok" if _selfcheck_tape(gdns[0], stock_conv, stock_tape) else "mismatch"
        except Exception as exc:  # pragma: no cover - device/compile failures
            _STATUS["tape"] = f"{type(exc).__name__}: {exc}"
    print(f"[gdn-fused] installed on {len(implementations)} GDN layers (tape mode: {_STATUS['tape']})", flush=True)
    return tuple(FusedM2Layer(impl, stock_layer) for impl in implementations)


def gdn_fused_m2_tape(gdn, proj: mx.array, conv_state: mx.array, state: mx.array):
    """Same layer in the tape capture's numerics and layout. Returns the
    gated-normed output, the final recurrent state ``[1, 32, 128, 128]``, conv
    states ``[1, 2, 3, 8192]``, conv outputs ``[1, 2, 8192]``, g ``[1, 2, 32]``
    and the delta tape ``[1, 2, 32, 128]``."""

    return _tape_kernel()(
        inputs=[proj, conv_state, gdn.conv1d.weight, gdn.A_log, gdn.dt_bias, state, gdn.norm.weight,
                float(gdn.norm.eps)],
        template=[("InT", mx.float16), ("StT", mx.float32), ("NSG", _NSG)],
        grid=(32, _NSG, 32),
        threadgroup=(32, _NSG, 1),
        output_shapes=[(1, 2, 32, 128), (1, 32, 128, 128), (1, 2, 3, 8192), (1, 2, 8192), (1, 2, 32),
                       (1, 2, 32, 128)],
        output_dtypes=[mx.float16, mx.float32, mx.float16, mx.float16, mx.float32, mx.float32],
    )
