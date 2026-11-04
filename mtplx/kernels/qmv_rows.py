"""Two-row affine int4 qmv for M1 decode verify (A3B depth 1 verifies 2 rows).

Stock ``mx.quantized_matmul`` serves a 2-row ``x`` with ``qmv_fast`` once per
row (``tid.x`` is the row), so each row re-extracts every weight nibble: the
2-row call costs 1.4-1.6x the 1-row one on M1 Max. Here one simdgroup walks
the weights once and feeds both rows from the same masked nibbles, with four
simdgroups per threadgroup (stock uses two).

``exact=True`` keeps each row's arithmetic identical to ``qmv_fast_impl`` (the
pre-scaled float ``x_thread``, the half partial ``sum``, ``scale * accum +
sum * bias`` per 512-value block, ``simd_sum``), so the output is bit-identical
to stock. ``exact=False`` holds the pre-scaled ``x`` in half registers
(accumulation stays float): ~7e-4 max relative drift, for sampled verifies.
"""

from __future__ import annotations

import os
from functools import lru_cache

import mlx.core as mx

_BLOCK = 512  # values_per_thread (16) x 32 lanes
_RPS = 4  # output rows per simdgroup
_NSG = 4  # simdgroups per threadgroup
_ROWS_PER_TG = _RPS * _NSG
# Below ~2K outputs the stock per-row grid is as fast (router 256, shared-expert
# gates); in the A3B verify forward N >= 2048 covers the GDN in/out, attention
# qkv/o and lm_head projections.
_MIN_N = 2048


@lru_cache(maxsize=None)
def _kernel(group_size: int, dtype: mx.Dtype, exact: bool):
    dtype_tag = {mx.float16: "fp16", mx.bfloat16: "bf16"}[dtype]
    xreg = "float" if exact else "half"
    source = f"""
        constexpr int GS = {group_size};
        constexpr int VPT = 16;
        constexpr int BLOCK = VPT * 32;
        constexpr int RPS = {_RPS};
        constexpr int NSG = {_NSG};
        constexpr int SCALE_STEP = GS / VPT;

        const int K = int(K_size);
        const int N = int(N_size);
        const uint lane = thread_index_in_simdgroup;
        const int out_row = (int(threadgroup_position_in_grid.x) * NSG
            + int(simdgroup_index_in_threadgroup)) * RPS;
        if (out_row >= N) {{
            return;
        }}
        const int in_vec_size_w = K / 2;
        const int in_vec_size_g = K / GS;
        const device uint8_t* ws = (const device uint8_t*)w_q
            + out_row * in_vec_size_w + lane * 8;
        const device T* sc = scales + out_row * in_vec_size_g + lane / SCALE_STEP;
        const device T* bi = biases + out_row * in_vec_size_g + lane / SCALE_STEP;
        const device T* x0 = x + lane * VPT;
        const device T* x1 = x + K + lane * VPT;

        float result0[RPS];
        float result1[RPS];
        for (int r = 0; r < RPS; ++r) {{
            result0[r] = 0.0f;
            result1[r] = 0.0f;
        }}
        for (int k = 0; k < K; k += BLOCK) {{
            {xreg} a[VPT];
            {xreg} b[VPT];
            float sum0 = 0.0f;
            float sum1 = 0.0f;
            for (int i = 0; i < VPT; i += 4) {{
                sum0 += x0[i] + x0[i + 1] + x0[i + 2] + x0[i + 3];
                sum1 += x1[i] + x1[i + 1] + x1[i + 2] + x1[i + 3];
                a[i] = x0[i];
                a[i + 1] = x0[i + 1] / 16.0f;
                a[i + 2] = x0[i + 2] / 256.0f;
                a[i + 3] = x0[i + 3] / 4096.0f;
                b[i] = x1[i];
                b[i + 1] = x1[i + 1] / 16.0f;
                b[i + 2] = x1[i + 2] / 256.0f;
                b[i + 3] = x1[i + 3] / 4096.0f;
            }}
            for (int r = 0; r < RPS; ++r) {{
                const device uint16_t* wl =
                    (const device uint16_t*)(ws + r * in_vec_size_w);
                const float s = sc[r * in_vec_size_g];
                const float bias = bi[r * in_vec_size_g];
                float accum0 = 0.0f;
                float accum1 = 0.0f;
                _Pragma("unroll")
                for (int i = 0; i < VPT / 4; ++i) {{
                    const uint16_t w = wl[i];
                    const float w0 = float(w & 0x000f);
                    const float w1 = float(w & 0x00f0);
                    const float w2 = float(w & 0x0f00);
                    const float w3 = float(w & 0xf000);
                    accum0 += (float(a[4 * i]) * w0 + float(a[4 * i + 1]) * w1
                        + float(a[4 * i + 2]) * w2 + float(a[4 * i + 3]) * w3);
                    accum1 += (float(b[4 * i]) * w0 + float(b[4 * i + 1]) * w1
                        + float(b[4 * i + 2]) * w2 + float(b[4 * i + 3]) * w3);
                }}
                result0[r] += s * accum0 + sum0 * bias;
                result1[r] += s * accum1 + sum1 * bias;
            }}
            ws += BLOCK / 2;
            sc += BLOCK / GS;
            bi += BLOCK / GS;
            x0 += BLOCK;
            x1 += BLOCK;
        }}
        for (int r = 0; r < RPS; ++r) {{
            const float v0 = simd_sum(result0[r]);
            const float v1 = simd_sum(result1[r]);
            if (lane == 0) {{
                y[out_row + r] = static_cast<T>(v0);
                y[N + out_row + r] = static_cast<T>(v1);
            }}
        }}
    """
    return mx.fast.metal_kernel(
        name=f"mtplx_qmv_rows2_gs{group_size}_{dtype_tag}_{'exact' if exact else 'halfx'}",
        input_names=["x", "w_q", "scales", "biases", "K_size", "N_size"],
        output_names=["y"],
        source=source,
    )


def qmv_rows_eligible(m: int, k: int, n: int, bits: int, group_size: int, dtype) -> bool:
    """2-row int4 shapes the kernel tiles exactly (N % 16, K % 512)."""

    return (
        int(m) == 2
        and int(bits) == 4
        and int(group_size) in (32, 64, 128)
        and int(k) % _BLOCK == 0
        and int(n) % _ROWS_PER_TG == 0
        and dtype in (mx.float16, mx.bfloat16)
    )


def qmv_rows(
    x: mx.array,
    w_q: mx.array,
    scales: mx.array,
    biases: mx.array,
    *,
    group_size: int,
    exact: bool = True,
) -> mx.array:
    """``x @ dequant(W).T`` for a contiguous ``(2, K)`` x; ``(2, N)`` out."""

    k = int(x.shape[1])
    n = int(w_q.shape[0])
    kernel = _kernel(int(group_size), x.dtype, bool(exact))
    return kernel(
        inputs=[x, w_q, scales, biases, k, n],
        template=[("T", x.dtype)],
        grid=(32 * _NSG * (n // _ROWS_PER_TG), 1, 1),
        threadgroup=(32 * _NSG, 1, 1),
        output_shapes=[(2, n)],
        output_dtypes=[x.dtype],
    )[0]


def _lane_enabled() -> bool:
    return (os.environ.get("MTPLX_M1_QMV_ROWS") or "").strip().lower() in {"1", "true", "on", "yes"}


def sampled_route_active() -> bool:
    """Whether a 2-row verify traced now takes the half-x kernel: the lane is
    on and the request samples (t > 0). Greedy verifies keep stock so MTP
    argmax matches AR argmax; compiled-step caches key on this value."""

    if not _lane_enabled():
        return False
    from ..attention_context import current_attention_phase, greedy_verify_scope

    return current_attention_phase() != "prefill" and not greedy_verify_scope()


def maybe_qmv_rows(
    x: mx.array,
    w_q: mx.array,
    scales: mx.array,
    biases: mx.array | None,
    *,
    group_size: int,
    bits: int,
    mode: str = "affine",
) -> mx.array | None:
    """The half-x 2-row kernel for an eligible sampled verify matmul, else None."""

    if biases is None or str(mode) != "affine" or x.ndim < 2:
        return None
    m = 1
    for d in x.shape[:-1]:
        m *= int(d)
    k = int(x.shape[-1])
    n = int(w_q.shape[0])
    if n < _MIN_N or not qmv_rows_eligible(m, k, n, bits, group_size, x.dtype):
        return None
    if not sampled_route_active():
        return None
    y = qmv_rows(x.reshape(2, k), w_q, scales, biases, group_size=int(group_size), exact=False)
    return y.reshape(*x.shape[:-1], n)
