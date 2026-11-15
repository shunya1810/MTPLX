"""Prefill HC consumers with native projections and BF16 op boundaries.

The grouped norm follows Apple's four-value RMS reduction. The mix consumer
uses a table made by the installed MLX sigmoid, avoiding JIT/metallib
transcendental differences. Neither projection nor the injection reduction
is replaced. Decode and verify retain their existing routes.

Every GPU generation: both kernels are plain SIMD (float arithmetic,
``simd_sum``, 640-thread norm groups, 256-thread mix groups) with no tensor
units, so they run on M1 to M5, but only an M5 has measured them. The
load-time self-check (``mtplx.kernel_selfcheck``, lane
``qwen4_hc_prefill_read``) compares their bits with the stock chain on this
GPU and turns the read off for the process on any difference, build failure
or refused thread count; the model's call site then keeps the eager chain.

Adapted from ddalcu/mlx-serve commit ea540c5560c2dbf4db36a649edb0a17a4ecab34a,
src/kernels/hc_prefill_{norm,mix}.metal, and Apple's rms_single_row.

Copyright (c) 2026 David Dalcu
Copyright (c) 2024 Apple Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

from __future__ import annotations

import os
from functools import lru_cache

ENV = "MTPLX_QWEN4_HC_PREFILL_READ"
LANE = "qwen4_hc_prefill_read"

_HC = 4
_HIDDEN = 2560
_WIDTH = _HC * _HIDDEN
_MAX_ROWS = 8192

_NORM_SOURCE = r"""
    const uint row = threadgroup_position_in_grid.x;
    const uint h = row % HC;
    const uint t = thread_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    threadgroup float partial[32];
    threadgroup float inverse;
    float values[4], total = 0;
    for (uint i = 0; i < 4; ++i) {
        values[i] = float(x[size_t(row) * H + t * 4 + i]);
        total += values[i] * values[i];
    }
    total = simd_sum(total);
    if (sg == 0) partial[lane] = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) partial[sg] = total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        total = simd_sum(partial[lane]);
        if (lane == 0) inverse = metal::precise::rsqrt(total / float(H) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = 0; i < 4; ++i) {
        // rms_norm(None) materializes BF16 before the grouped weight multiply.
        const T normalized = T(values[i] * inverse);
        out[size_t(row) * H + t * 4 + i] =
            T(float(normalized) * float(w[h * H + t * 4 + i]));
    }
"""

_MIX_SOURCE = r"""
    const uint i = thread_position_in_grid.x;
    if (i >= uint(rows) * H) return;
    const uint row = i / H, column = i % H;
    T total = T(0);
    for (uint h = 0; h < HC; ++h) {
        const uint offset = (row * HC + h) * H + column;
        const T gate = sigmoid_table[as_type<ushort>(up[offset])];
        const T product = T(float(gate) * float(normed[offset]));
        // MLX col_reduce_small uses one y-lane per HC stream: each starts
        // from +0, then the lanes are added in ascending stream order.
        // Preserve the initial add as well (observable for signed zero).
        const T partial = T(float(product) + 0.0f);
        total = h == 0 ? partial : T(float(partial) + float(total));
    }
    // mean() materializes its reciprocal in the input dtype before multiply.
    out[i] = T(float(total) * float(T(1.0f / float(HC))));
"""


def switched_on() -> bool:
    """The user's switch alone (default on)."""

    raw = (os.environ.get(ENV) or "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def enabled() -> bool:
    """Switched on, and not turned off by the load-time self-check on this GPU.

    :func:`hc_prefill_read` has no fallback of its own, so the model checks
    this first and a lane the self-check turned off never reaches the kernels.
    """

    # Absolute and lazy: the CPU contract test loads this file on its own.
    from mtplx.kernel_selfcheck import lane_disabled

    return switched_on() and not lane_disabled(LANE)


def _geometry(shape, hc_count, hidden_size):
    """Admit actual prefill sequence widths, never wide batches of decode."""
    if len(shape) != 3 or hc_count != _HC or hidden_size != _HIDDEN:
        return None
    batch, sequence, width = shape
    if batch not in (1, 2) or sequence < 32 or width != _WIDTH:
        return None
    rows = batch * sequence
    return rows if rows <= _MAX_ROWS else None


@lru_cache(maxsize=1)
def _kernels():
    import mlx.core as mx

    norm = mx.fast.metal_kernel(
        name="mtplx_hc_prefill_norm",
        input_names=["x", "w", "eps"],
        output_names=["out"],
        source=_NORM_SOURCE,
        ensure_row_contiguous=True,
    )
    mix = mx.fast.metal_kernel(
        name="mtplx_hc_prefill_mix",
        input_names=["up", "normed", "sigmoid_table", "rows"],
        output_names=["out"],
        source=_MIX_SOURCE,
        ensure_row_contiguous=True,
    )
    return norm, mix


@lru_cache(maxsize=None)
def _sigmoid_table(dtype=None):
    import mlx.core as mx

    # One shared 128 KiB array per dtype. Lazy native evaluation uses this MLX
    # build's sigmoid for every 16-bit pattern (bf16, or fp16 for the M1 packs
    # that carry the model in float16), including infinities and NaNs.
    dtype = mx.bfloat16 if dtype is None else dtype
    bits = mx.arange(65536, dtype=mx.uint32).astype(mx.uint16)
    return mx.sigmoid(bits.view(dtype), stream=mx.gpu)


def _normalize(x, weight, eps, *, hc=_HC, hidden=_HIDDEN):
    import mlx.core as mx

    groups = x.size // hidden
    return _kernels()[0](
        inputs=[x, weight, eps],
        template=[("T", x.dtype), ("HC", hc), ("H", hidden)],
        grid=(groups * (hidden // 4), 1, 1),
        threadgroup=(hidden // 4, 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
    )[0]


def _mix(up, normed, *, hc=_HC, hidden=_HIDDEN):
    import mlx.core as mx

    rows = up.size // (hc * hidden)
    return _kernels()[1](
        inputs=[up, normed, _sigmoid_table(up.dtype), rows],
        template=[("T", up.dtype), ("HC", hc), ("H", hidden)],
        grid=(rows * hidden, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(*up.shape[:-1], hidden)],
        output_dtypes=[up.dtype],
    )[0]


def hc_prefill_read(owner, hyper_input):
    """Return the HC read result, or None to retain the existing eager chain.

    Bounds and dtypes are metadata-only checks. No weight conversion, host
    synchronization, previous-write deferral or exception fallback is added.
    """
    if _geometry(hyper_input.shape, owner.hc_count, owner.hidden_size) is None:
        return None
    import mlx.core as mx
    import mlx.nn as nn

    if (
        hyper_input.dtype not in (mx.bfloat16, mx.float16)
        or not mx.metal.is_available()
        or mx.default_device() != mx.gpu
        or owner.hc_norm.weight.dtype != hyper_input.dtype
        or owner.hc_norm.weight.shape != (_WIDTH,)
        or owner.hc_norm.group_size != _HIDDEN
    ):
        return None
    projections = [
        (owner.input_mix_weight_down, (320, _WIDTH)),
        (owner.input_mix_weight_up, (_WIDTH, 320)),
    ]
    combine = "block_inject_weight" in owner
    if combine:
        projections.append((owner.block_inject_weight, (_HC, _WIDTH)))
    if any(
        hasattr(proj, "scales")
        or proj.weight.dtype != hyper_input.dtype
        or proj.weight.shape != shape
        or getattr(proj, "bias", None) is not None
        for proj, shape in projections
    ):
        return None
    normed = _normalize(hyper_input, owner.hc_norm.weight, owner.hc_norm.eps)
    mix = nn.silu(owner.input_mix_weight_down(normed) / owner.hc_count)
    up = owner.input_mix_weight_up(mix)
    mixed_input = _mix(up, normed)
    if not combine:
        return mixed_input
    inject = 2.0 * mx.sigmoid(owner.block_inject_weight(normed) / owner.hc_count)
    return mixed_input, hyper_input, inject
