# Copyright © 2026 Apple Inc. (MIT), ported from ml-explore/mlx-lm PR #2086
# (mlx_lm/models/inkling2/{config,layers,model,checkpoint}.py), which the
# pinned mlx-lm does not ship yet.
"""Inkling 2 Mini (``inkling2_loop``) trunk for MTPLX.

A 24-layer MoE decoder whose middle layers (``loop_config.core_start`` to
``core_end``) run ``num_passes`` times with shared weights: 48 layer visits per
token, each visit with its own KV and convolution cache. Attention has no
RoPE: a learned relative bias (``d_rel`` -> ``rel_extent``) plus a sink logit
per head and token. Layers read their inputs through depth attention over the
block residuals, and the looped layers mix in messages from the previous pass.

Changes against the mlx-lm port:

* Text only. The vision tower's weights are dropped at load.
* The pinned MLX has no ``global_scale`` on ``gather_qmm``: the per-expert
  NVFP4 factor ``amax / outer_scale_denominator`` multiplies the expert
  outputs instead (``ExpertLinear``), which is the same product.
* The attention, gate, output and LM-head projections ship in BF16. They are
  read once per layer visit (48 times per token), so by default they load as
  8-bit affine (``MTPLX_INKLING2_PROJ_BITS``, ``MTPLX_INKLING2_HEAD_BITS``;
  0 keeps them unquantized), the precision the Unsloth Q4_K_XL GGUF uses.
* The relative attention kernel also takes float16 (M1 runs the trunk in fp16).
* ``mtp.safetensors`` (the DSpark drafter) is not loaded here.
"""

from __future__ import annotations

import math
import os
import threading
from dataclasses import dataclass
from functools import partial
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten

from mlx_lm.models.base import BaseModelArgs
from mlx_lm.models.cache import (
    ArraysCache,
    BatchRotatingKVCache,
    CacheList,
    KVCache,
    RotatingKVCache,
)
from mlx_lm.models.switch_layers import SwitchLinear, _gather_sort, _scatter_unsort

from ..kernels.inkling2_kernels import depth_attention, expert_routing, relative_attention

__all__ = ["Model", "ModelArgs", "TextArgs"]


_ROUTER_SCORE_EPS = 1e-20
_NVFP4_E2M1_MAX = 6.0
_NVFP4_E4M3_MAX = 448.0


def _env_int(name: str, default: int) -> int:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


# float16 activations through NVFP4 experts: the raw product is the output
# divided by the outer scale (4e-5..2.4e-4 in the official checkpoint), far
# past float16's 65,504. "scaled" (default) folds outer scale x 2^8 into the
# input per (token, expert) pair, runs the product in float16 and takes the
# 2^8 back out (measured |gate_up| <= 21, |down| <= 31 over en/ja/code/agent
# prefill, so the scaled output keeps 8x headroom); "bf16" / "f32" run the
# product in a wider dtype instead (M1 Max, 2,048-token sorted gate_up:
# float16 32.3 ms, bfloat16 45.1, float32 37.4).
_EXPERT_ACT = (os.environ.get("MTPLX_INKLING2_EXPERT_ACT") or "scaled").strip().lower()
_EXPERT_WIDE_DTYPE = mx.float32 if _EXPERT_ACT in {"f32", "float32"} else mx.bfloat16
_EXPERT_SCALE_SHIFT = 256.0


# Decode (one token per call) runs each layer visit as two compiled halves
# around the KV cache. MTPLX_INKLING2_DECODE_COMPILE=0 keeps the eager path.
_DECODE_COMPILE = (os.environ.get("MTPLX_INKLING2_DECODE_COMPILE") or "1").strip() not in {
    "0",
    "false",
    "off",
}


# Fused NVFP4 expert kernels up to this many (token, expert) pairs (decode is
# 8, a 4-row verify 32); 0 keeps gather_qmm everywhere.
_FUSED_MOE_MAX_PAIRS = _env_int("MTPLX_INKLING2_FUSED_MOE_PAIRS", 64)


def _use_kernels() -> bool:
    return mx.default_device() == mx.gpu and mx.metal.is_available()


# --------------------------------------------------------------------------
# Config


@dataclass
class TextArgs(BaseModelArgs):
    hidden_size: int
    num_hidden_layers: int
    vocab_size: int
    unpadded_vocab_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    swa_num_attention_heads: int
    swa_num_key_value_heads: int
    swa_head_dim: int
    sliding_window_size: int
    local_layer_ids: list
    d_rel: int
    rel_extent: int
    log_scaling_n_floor: int
    log_scaling_alpha: float
    rms_norm_eps: float
    qk_l2_clip_threshold: float
    rel_l2_clip_threshold: float
    sconv_kernel_size: int
    n_layers_per_attn_res_block: int
    n_routed_experts: int
    num_experts_per_tok: int
    intermediate_size: int
    mlp_gate_softcap: float
    mlp_up_softcap: float
    route_scale: float
    final_logit_softcapping: float
    loop_config: dict | None = None

    def __post_init__(self):
        loop = self.loop_config or {}
        self.loop_core = range(loop.get("core_start", 0), loop.get("core_end", 0))


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    text_config: dict
    vision_config: dict | None = None
    image_token_id: int = 200054

    def __post_init__(self):
        self.text = TextArgs.from_dict(self.text_config)


# --------------------------------------------------------------------------
# Small ops


def unweighted_rms_norm(x: mx.array, eps: float = 1e-6) -> mx.array:
    return mx.fast.rms_norm(x, None, eps)


@partial(mx.compile, shapeless=True)
def _l2_clip(hidden_states: mx.array, scale: mx.array, threshold: float) -> mx.array:
    states = hidden_states.astype(mx.float32) * scale
    norm = mx.sqrt((states * states).sum(axis=-1, keepdims=True))
    scaled = states * mx.minimum(threshold / (norm + 1e-6), 1.0)
    return scaled.astype(hidden_states.dtype)


@partial(mx.compile, shapeless=True)
def _softcapped_swiglu(
    gate: mx.array, up: mx.array, gate_softcap: float, up_softcap: float
) -> mx.array:
    input_dtype = gate.dtype
    gate, up = gate.astype(mx.float32), up.astype(mx.float32)
    gate_capped = mx.tanh(gate / gate_softcap) * gate_softcap
    up_capped = mx.tanh(up / up_softcap) * up_softcap
    return (gate_capped * mx.sigmoid(gate) * up_capped).astype(input_dtype)


class L2Clip(nn.Module):
    def __init__(self, threshold: float):
        super().__init__()
        self.threshold = threshold

    def __call__(self, hidden_states: mx.array) -> mx.array:
        return _l2_clip(hidden_states, 1.0, self.threshold)


class ScaledL2Clip(nn.Module):
    def __init__(self, num_heads: int, head_dim: int, threshold: float):
        super().__init__()
        self.threshold = threshold
        self.scale = mx.ones((num_heads, head_dim))

    def __call__(self, hidden_states: mx.array) -> mx.array:
        return _l2_clip(hidden_states, self.scale, self.threshold)


class RelativeLogits(nn.Module):
    def __init__(self, d_rel: int, rel_extent: int):
        super().__init__()
        self.proj = mx.zeros((d_rel, rel_extent))

    def __call__(self, relative_states: mx.array) -> mx.array:
        return relative_states @ self.proj.astype(relative_states.dtype)


def _conv_input(
    hidden_states: mx.array,
    conv_state: mx.array | None,
    conv_mask: mx.array | None,
    kernel_size: int,
) -> mx.array:
    hidden_states = hidden_states.astype(mx.float32)
    if conv_mask is not None:
        hidden_states = mx.where(conv_mask[..., None], hidden_states, 0)
    if conv_state is None:
        shape = (hidden_states.shape[0], kernel_size - 1, hidden_states.shape[-1])
        conv_state = mx.zeros(shape, mx.float32)
    return mx.concatenate([conv_state, hidden_states], axis=1)


def _conv_state(window: mx.array, lengths: mx.array | None, n: int) -> mx.array:
    if lengths is None:
        pos = mx.arange(window.shape[1] - n, window.shape[1])[None, :, None]
    else:
        ends = mx.clip(lengths, 0, window.shape[1] - n)
        pos = (ends[:, None] + mx.arange(n))[..., None]
    # A gather copies, so the state does not keep the input in memory.
    return mx.take_along_axis(window, pos, axis=1)


class ShortConvolution(nn.Module):
    def __init__(self, dims: int, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv1d = nn.Conv1d(dims, dims, kernel_size, groups=dims, bias=False)

    def __call__(
        self,
        hidden_states: mx.array,
        conv_state: mx.array | None,
        conv_mask: mx.array | None,
    ) -> tuple[mx.array, mx.array]:
        K = self.kernel_size
        residual = hidden_states.astype(mx.float32)
        window = _conv_input(hidden_states, conv_state, conv_mask, K)
        # As the reference, the conv runs in the weight dtype.
        out = self.conv1d(window.astype(self.conv1d.weight.dtype))
        out = (out.astype(mx.float32) + residual) / math.sqrt(K + 1)
        return out.astype(hidden_states.dtype), window


class DynamicShortConvolution(nn.Module):
    def __init__(self, dims: int, input_dims: int, num_heads: int, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.num_heads = num_heads
        self.static_weight = mx.zeros((kernel_size, dims))
        self.weight_proj = nn.Linear(input_dims, num_heads * kernel_size, bias=False)

    def __call__(
        self,
        hidden_states: mx.array,
        dynamic_input: mx.array,
        conv_state: mx.array | None,
        conv_mask: mx.array | None,
    ) -> tuple[mx.array, mx.array]:
        B, L, C = hidden_states.shape
        K, H = self.kernel_size, self.num_heads
        dynamic_weights = self.weight_proj(dynamic_input).astype(mx.float32)
        dynamic_weights = dynamic_weights.reshape(B, L, H, 1, K)
        static_weights = self.static_weight.astype(mx.float32).T.reshape(H, -1, K)
        weights = static_weights + dynamic_weights

        window = _conv_input(hidden_states, conv_state, conv_mask, K)
        windows = mx.stack(
            [window[:, K - 1 - lag : K - 1 - lag + L] for lag in range(K)], axis=-1
        )
        windows = windows.reshape(B, L, H, -1, K)
        out = (windows * weights).sum(axis=-1).reshape(B, L, C)
        out = (out + hidden_states.astype(mx.float32)) / math.sqrt(2 * K + 1)
        return out.astype(hidden_states.dtype), window


# --------------------------------------------------------------------------
# Attention


def _positions(offset, length: int) -> mx.array:
    pos = mx.arange(length)
    if isinstance(offset, mx.array):
        return offset[:, None] + pos
    return pos + offset


def _key_positions(cache: Any, num_keys: int, num_queries: int, end) -> mx.array:
    k = mx.arange(num_keys)
    if num_queries == 1 and isinstance(cache, (RotatingKVCache, BatchRotatingKVCache)):
        # A decode step writes in place, so the keys can be rotated.
        k = (k - cache._idx) % num_keys
    if isinstance(end, mx.array):
        return (end - num_keys)[:, None] + k
    return k + (end - num_keys)


def _attention_reference(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    position_bias: mx.array,
    sinks: mx.array,
    *,
    scale: float,
    window: int,
    query_positions: mx.array,
    key_positions: mx.array,
) -> mx.array:
    B, Hq, L, D = query.shape
    Hkv, S = key.shape[1], key.shape[2]
    R = position_bias.shape[-1]
    distance = query_positions[..., :, None] - key_positions[..., None, :]
    visible = (distance >= 0) & (key_positions[..., None, :] >= 0)
    if window:
        visible = visible & (distance < window)
    if distance.ndim == 3:
        distance, visible = distance[:, None], visible[:, None]
    bias = mx.take_along_axis(
        position_bias.transpose(0, 2, 1, 3),
        mx.broadcast_to(mx.clip(distance, 0, R - 1), (B, Hq, L, S)),
        axis=-1,
    )
    bias = mx.where(distance < R, bias.astype(mx.float32), 0.0)
    attention_mask = mx.where(visible, bias, -mx.inf).reshape(B, Hkv, -1, L, S)

    query = query.astype(mx.float32).reshape(B, Hkv, -1, L, D)
    key = key.astype(mx.float32)[:, :, None]
    attn_weights = (query @ key.swapaxes(-1, -2)) * scale + attention_mask
    sinks = sinks.astype(mx.float32).transpose(0, 2, 1).reshape(B, Hkv, -1, L, 1)
    combined_logits = mx.concatenate([attn_weights, sinks], axis=-1)
    attn_weights = mx.softmax(combined_logits, axis=-1)[..., :S]
    attn_output = attn_weights.astype(value.dtype) @ value[:, :, None]
    return attn_output.reshape(B, Hq, L, D).transpose(0, 2, 1, 3)


# Decode-step key geometry shared by the caches of one kind (they advance in
# lockstep): keyed by (kind, keys, ring index, end, window, R, dtype).
_STEP_GEOMETRY = threading.local()


def _reset_step_geometry() -> None:
    _STEP_GEOMETRY.table = {}


def _decode_geometry(cache, num_keys: int, end, window: int, R: int, dtype):
    ring = isinstance(cache, (RotatingKVCache, BatchRotatingKVCache))
    idx = int(cache._idx) if ring else -1
    key = (ring, num_keys, idx, int(end), window, R, dtype)
    table = getattr(_STEP_GEOMETRY, "table", None)
    if table is None:
        table = _STEP_GEOMETRY.table = {}
    hit = table.get(key)
    if hit is not None:
        return hit
    key_positions = _key_positions(cache, num_keys, 1, end).reshape(num_keys)
    distance = (end - 1) - key_positions
    visible = (distance >= 0) & (key_positions >= 0)
    if window:
        visible = visible & (distance < window)
    dclip = mx.clip(distance, 0, R - 1).astype(mx.int32)
    use_bias = visible & (distance < R)
    base = mx.where(visible, mx.array(0.0, dtype), mx.array(-mx.inf, dtype))
    hit = (dclip, use_bias, base)
    if len(table) > 16:
        table.clear()
    table[key] = hit
    return hit


@partial(mx.compile, shapeless=True)
def _decode_mask(bias_rows: mx.array, dclip: mx.array, use_bias: mx.array, base: mx.array) -> mx.array:
    """(H, R) bias rows of one query -> (1, H, 1, S) additive mask."""
    bias = mx.take(bias_rows, dclip, axis=-1)
    return mx.where(use_bias, bias, base)[None, :, None, :]


class Attention(nn.Module):
    # __call__ takes (q, k, v hidden states, cache, conv states) and the
    # relative bias / sinks are part of the attention: the runtime's generic
    # dense-SDPA hooks (attention_split) and q|k|v projection fusion
    # (proj_fusion: q/k/v inputs differ on global layers and message passes)
    # must leave this class alone.
    _mtplx_generic_sdpa_rewrites_unsupported = True
    _mtplx_proj_fusion_unsupported = True

    def __init__(self, args: TextArgs, *, sliding: bool):
        super().__init__()
        if sliding:
            self.num_heads = args.swa_num_attention_heads
            self.num_key_value_heads = args.swa_num_key_value_heads
            self.head_dim = args.swa_head_dim
            self.sliding_window = args.sliding_window_size
            rel_extent = args.sliding_window_size
        else:
            self.num_heads = args.num_attention_heads
            self.num_key_value_heads = args.num_key_value_heads
            self.head_dim = args.head_dim
            self.sliding_window = 0
            rel_extent = args.rel_extent
        self.scaling = self.head_dim**-0.5

        dim = args.hidden_size
        q_dim = self.num_heads * self.head_dim
        kv_dim = self.num_key_value_heads * self.head_dim
        self.q_proj = nn.Linear(dim, q_dim, bias=False)
        self.k_proj = nn.Linear(dim, kv_dim, bias=False)
        self.v_proj = nn.Linear(dim, kv_dim, bias=False)
        self.r_proj = nn.Linear(dim, self.num_heads * args.d_rel, bias=False)
        self.o_proj = nn.Linear(q_dim, dim, bias=False)
        self.gate_proj = nn.Linear(dim, q_dim, bias=False)
        self.sink_proj = nn.Linear(dim, self.num_heads, bias=False)
        self.rel_logits_proj = RelativeLogits(args.d_rel, rel_extent)
        self.q_l2_clip = ScaledL2Clip(
            self.num_heads, self.head_dim, args.qk_l2_clip_threshold
        )
        self.k_l2_clip = L2Clip(args.qk_l2_clip_threshold)
        self.rel_l2_clip = ScaledL2Clip(
            self.num_heads, args.d_rel, args.rel_l2_clip_threshold
        )
        self.eps = args.rms_norm_eps

    def _prep(
        self,
        hidden_states: mx.array,
        query_states: mx.array,
        key_states: mx.array,
        value_states: mx.array,
        relative_states: mx.array,
        tau: mx.array | None = None,
    ) -> tuple:
        """Everything before the KV cache: gate, sinks, clipped q/k, unit-RMS
        v, the relative bias rows; q/k/v come back (B, H, L, D)."""
        B, L, _ = hidden_states.shape
        gate = self.gate_proj(hidden_states)
        sinks = self.sink_proj(hidden_states)

        query_states = query_states.reshape(B, L, self.num_heads, -1)
        key_states = key_states.reshape(B, L, self.num_key_value_heads, -1)
        value_states = value_states.reshape(B, L, self.num_key_value_heads, -1)
        relative_states = relative_states.reshape(B, L, self.num_heads, -1)
        query_states = self.q_l2_clip(query_states)
        key_states = self.k_l2_clip(key_states)
        value_states = unweighted_rms_norm(
            value_states.astype(mx.float32), self.eps
        ).astype(value_states.dtype)
        relative_states = self.rel_l2_clip(relative_states)
        if tau is not None:
            query_states = (query_states.astype(mx.float32) * tau).astype(
                query_states.dtype
            )
            relative_states = (relative_states.astype(mx.float32) * tau).astype(
                relative_states.dtype
            )
        position_bias = self.rel_logits_proj(relative_states)

        query_states = query_states.transpose(0, 2, 1, 3)
        key_states = key_states.transpose(0, 2, 1, 3)
        value_states = value_states.transpose(0, 2, 1, 3)
        return query_states, key_states, value_states, position_bias, sinks, gate

    def _finish(self, attn_output: mx.array, gate: mx.array) -> mx.array:
        B, L = gate.shape[:2]
        attn_output = attn_output.reshape(B, L, -1).astype(mx.float32)
        attn_output = attn_output * mx.sigmoid(gate.astype(mx.float32))
        return self.o_proj(attn_output.astype(gate.dtype))

    def _attend(
        self,
        hidden_states: mx.array,
        query_states: mx.array,
        key_states: mx.array,
        value_states: mx.array,
        relative_states: mx.array,
        cache: Any | None,
        tau: mx.array | None = None,
    ) -> mx.array:
        query_states, key_states, value_states, position_bias, sinks, gate = self._prep(
            hidden_states, query_states, key_states, value_states, relative_states, tau
        )
        if cache is not None:
            key_states, value_states = cache.update_and_fetch(key_states, value_states)
        attn_output = self._attention(
            query_states, key_states, value_states, position_bias, sinks, cache
        )
        return self._finish(attn_output, gate)

    def _attention(self, query, key, value, position_bias, sinks, cache) -> mx.array:
        B, _, L, _ = query.shape
        S = key.shape[2]
        end = cache.offset if cache is not None else L
        half = query.dtype == key.dtype == value.dtype and query.dtype in (
            mx.bfloat16,
            mx.float16,
        )
        if L > 1 and _use_kernels() and half:
            key_start = mx.broadcast_to(S - end, (B,))
            return relative_attention(
                query,
                key,
                value,
                position_bias,
                sinks,
                scale=self.scaling,
                window=self.sliding_window,
                key_start=key_start,
            )

        if L == 1 and B == 1:
            # The fused attention takes the sinks of one row. The key geometry
            # (distance, visibility) is the same for every cache of a kind in
            # a decode step, so it is built once per step and shared.
            R = position_bias.shape[-1]
            dclip, use_bias, base = _decode_geometry(
                cache, S, end, self.sliding_window, R, query.dtype
            )
            mask = _decode_mask(position_bias[0, 0], dclip, use_bias, base)
            attn_output = mx.fast.scaled_dot_product_attention(
                query,
                key,
                value,
                scale=self.scaling,
                mask=mask,
                sinks=sinks[0, 0].astype(query.dtype),
            )
            return attn_output.transpose(0, 2, 1, 3)

        query_positions = _positions(end - L, L)
        key_positions = _key_positions(cache, S, L, end)
        return _attention_reference(
            query,
            key,
            value,
            position_bias,
            sinks,
            scale=self.scaling,
            window=self.sliding_window,
            query_positions=query_positions,
            key_positions=key_positions,
        )


class SlidingAttention(Attention):
    def __init__(self, args: TextArgs):
        super().__init__(args, sliding=True)
        dims = self.num_key_value_heads * self.head_dim
        self.k_sconv = ShortConvolution(dims, args.sconv_kernel_size)
        self.v_sconv = ShortConvolution(dims, args.sconv_kernel_size)

    def _project(self, hidden_states, key_hidden_states, value_hidden_states, conv_states, conv_mask):
        key_states, conv_states[0] = self.k_sconv(
            self.k_proj(key_hidden_states), conv_states[0], conv_mask
        )
        value_states, conv_states[1] = self.v_sconv(
            self.v_proj(value_hidden_states), conv_states[1], conv_mask
        )
        return (
            self.q_proj(hidden_states),
            key_states,
            value_states,
            self.r_proj(hidden_states),
        )

    def __call__(
        self,
        hidden_states: mx.array,
        key_hidden_states: mx.array,
        value_hidden_states: mx.array,
        *,
        cache: Any | None,
        conv_states: list,
        conv_mask: mx.array | None,
    ) -> mx.array:
        kv_cache = cache[0] if cache is not None else None
        q, k, v, r = self._project(
            hidden_states, key_hidden_states, value_hidden_states, conv_states, conv_mask
        )
        return self._attend(hidden_states, q, k, v, r, kv_cache)


class FullAttention(Attention):
    def __init__(self, args: TextArgs):
        super().__init__(args, sliding=False)
        dims = self.num_key_value_heads * self.head_dim
        conv_args = (args.hidden_size, self.num_key_value_heads, args.sconv_kernel_size)
        self.k_sconv = DynamicShortConvolution(dims, *conv_args)
        self.v_sconv = DynamicShortConvolution(dims, *conv_args)
        self.log_scaling_n_floor = args.log_scaling_n_floor
        self.log_scaling_alpha = args.log_scaling_alpha

    def _project(self, hidden_states, key_hidden_states, value_hidden_states, conv_states, conv_mask):
        key_states, conv_states[0] = self.k_sconv(
            self.k_proj(key_hidden_states),
            key_hidden_states,
            conv_states[0],
            conv_mask,
        )
        value_states, conv_states[1] = self.v_sconv(
            self.v_proj(value_hidden_states),
            value_hidden_states,
            conv_states[1],
            conv_mask,
        )
        return (
            self.q_proj(hidden_states),
            key_states,
            value_states,
            self.r_proj(hidden_states),
        )

    def _tau(self, offset, length: int) -> mx.array:
        position_ids = _positions(offset, length).astype(mx.float32)
        tau = 1.0 + self.log_scaling_alpha * mx.log(
            mx.maximum((position_ids + 1) / self.log_scaling_n_floor, 1.0)
        )
        return tau[..., None, None]

    def tau_scalar(self, offset: int) -> float:
        """The log scaling of one query at position ``offset`` (decode)."""
        return 1.0 + self.log_scaling_alpha * math.log(
            max((offset + 1) / self.log_scaling_n_floor, 1.0)
        )

    def __call__(
        self,
        hidden_states: mx.array,
        key_hidden_states: mx.array,
        value_hidden_states: mx.array,
        *,
        cache: Any | None,
        conv_states: list,
        conv_mask: mx.array | None,
    ) -> mx.array:
        kv_cache = cache[0] if cache is not None else None
        q, k, v, r = self._project(
            hidden_states, key_hidden_states, value_hidden_states, conv_states, conv_mask
        )
        offset = kv_cache.offset if kv_cache is not None else 0
        tau = self._tau(offset, hidden_states.shape[1])
        return self._attend(hidden_states, q, k, v, r, kv_cache, tau=tau)


# --------------------------------------------------------------------------
# Mixture of experts


class ExpertLinear(nn.Module):
    """Experts in a microscaled MLX mode (nvfp4 / mxfp4 / mxfp8), with an
    optional float32 factor per expert multiplied into the outputs (the NVFP4
    outer scale ``amax / outer_scale_denominator``)."""

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        num_experts: int,
        *,
        group_size: int,
        bits: int,
        mode: str,
        expert_scale: bool,
    ):
        super().__init__()
        self.group_size = group_size
        self.bits = bits
        self.mode = mode
        self.weight = mx.zeros((num_experts, output_dims, input_dims * bits // 32), mx.uint32)
        self.scales = mx.zeros((num_experts, output_dims, input_dims // group_size), mx.uint8)
        if expert_scale:
            self.expert_scale = mx.ones((num_experts,), mx.float32)

    def __call__(self, x: mx.array, indices: mx.array, sorted_indices: bool = False) -> mx.array:
        out_dtype = x.dtype
        has_scale = "expert_scale" in self
        if has_scale and x.dtype == mx.float16 and _EXPERT_ACT == "scaled":
            factor = self["expert_scale"][indices].reshape(indices.shape + (1, 1))
            x = (x.astype(mx.float32) * (factor * _EXPERT_SCALE_SHIFT)).astype(mx.float16)
            out = mx.gather_qmm(
                x,
                self["weight"],
                self["scales"],
                None,
                rhs_indices=indices,
                transpose=True,
                group_size=self.group_size,
                bits=self.bits,
                mode=self.mode,
                sorted_indices=sorted_indices,
            )
            return out * mx.array(1.0 / _EXPERT_SCALE_SHIFT, mx.float16)
        if has_scale and x.dtype == mx.float16:
            x = x.astype(_EXPERT_WIDE_DTYPE)
        out = mx.gather_qmm(
            x,
            self["weight"],
            self["scales"],
            None,
            rhs_indices=indices,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
            sorted_indices=sorted_indices,
        )
        if "expert_scale" in self:
            factor = self["expert_scale"][indices].reshape(indices.shape + (1, 1))
            out = (out.astype(mx.float32) * factor).astype(out_dtype)
        return out


class SoftcappedSwiGLU(nn.Module):
    def __init__(self, args: TextArgs):
        super().__init__()
        self.gate_softcap = args.mlp_gate_softcap
        self.up_softcap = args.mlp_up_softcap

    def __call__(self, gate: mx.array, up: mx.array) -> mx.array:
        return _softcapped_swiglu(gate, up, self.gate_softcap, self.up_softcap)


class Experts(nn.Module):
    def __init__(self, args: TextArgs):
        super().__init__()
        dims, hidden = args.hidden_size, args.intermediate_size
        num_experts = args.n_routed_experts
        # The rows alternate between gate and up, as in the checkpoint.
        self.gate_up_proj = SwitchLinear(dims, 2 * hidden, num_experts, bias=False)
        self.down_proj = SwitchLinear(hidden, dims, num_experts, bias=False)
        self.act_fn = SoftcappedSwiGLU(args)

    def __call__(self, x: mx.array, indices: mx.array) -> mx.array:
        x = mx.expand_dims(x, (-2, -3))
        indices = mx.stop_gradient(indices)
        do_sort = indices.size >= 64
        idx, inv_order = indices, None
        if do_sort:
            x, idx, inv_order = _gather_sort(x, indices)
        gate_up = self.gate_up_proj(x, idx, sorted_indices=do_sort)
        x = self.act_fn(gate_up[..., 0::2], gate_up[..., 1::2])
        x = self.down_proj(x, idx, sorted_indices=do_sort)
        if do_sort:
            x = _scatter_unsort(x, inv_order, indices.shape)
        return x.squeeze(-2)


class TopKRouter(nn.Module):
    def __init__(self, args: TextArgs):
        super().__init__()
        self.top_k = args.num_experts_per_tok
        self.routed_scaling_factor = args.route_scale
        num_experts, dims = args.n_routed_experts, args.hidden_size
        self.weight = mx.zeros((num_experts, dims), dtype=mx.float32)
        self.e_score_correction_bias = mx.zeros((num_experts,), dtype=mx.float32)

    def __call__(self, hidden_states: mx.array) -> tuple[mx.array, mx.array]:
        logits = hidden_states.astype(mx.float32) @ self.weight.T
        if _use_kernels():
            return expert_routing(
                logits,
                self.e_score_correction_bias,
                top_k=self.top_k,
                multiplier=1.0,
                score_eps=_ROUTER_SCORE_EPS,
                route_scale=self.routed_scaling_factor,
            )
        scores = mx.sigmoid(logits)
        biased = scores + self.e_score_correction_bias
        k = self.top_k
        indices = mx.argpartition(-biased, kth=k - 1, axis=-1)[..., :k]
        weights = mx.take_along_axis(scores, indices, axis=-1)
        weights = weights / (weights.sum(axis=-1, keepdims=True) + _ROUTER_SCORE_EPS)
        return indices, weights * self.routed_scaling_factor


class SparseMoeBlock(nn.Module):
    def __init__(self, args: TextArgs):
        super().__init__()
        self.gate = TopKRouter(args)
        self.experts = Experts(args)

    def _fused_ok(self, num_pairs: int, dtype) -> bool:
        experts = self.experts
        gu, dn = experts.gate_up_proj, experts.down_proj
        return (
            _FUSED_MOE_MAX_PAIRS > 0
            and dtype in (mx.float16, mx.bfloat16)
            and num_pairs <= _FUSED_MOE_MAX_PAIRS
            and isinstance(gu, ExpertLinear)
            and isinstance(dn, ExpertLinear)
            and gu.mode == dn.mode == "nvfp4"
            and "expert_scale" in gu
            and "expert_scale" in dn
            and _use_kernels()
        )

    def __call__(self, hidden_states: mx.array) -> mx.array:
        selected_experts, routing_weights = self.gate(hidden_states)
        if self._fused_ok(selected_experts.size, hidden_states.dtype):
            # Decode / verify widths: two kernels (gate/up + softcapped
            # SwiGLU, down + routing weights + sum over experts).
            from ..kernels.inkling2_moe import moe_nvfp4

            dims = hidden_states.shape[-1]
            experts = self.experts
            act = experts.act_fn
            y = moe_nvfp4(
                hidden_states.reshape(-1, dims),
                experts.gate_up_proj,
                experts.down_proj,
                selected_experts.reshape(-1).astype(mx.uint32),
                routing_weights.reshape(-1).astype(mx.float32),
                gate_softcap=act.gate_softcap,
                up_softcap=act.up_softcap,
            )
            return y.reshape(hidden_states.shape)
        expert_states = self.experts(hidden_states, selected_experts)
        expert_states = expert_states.astype(mx.float32) * routing_weights[..., None]
        final_hidden_states = expert_states.sum(axis=-2)
        return final_hidden_states.astype(hidden_states.dtype)


# --------------------------------------------------------------------------
# Depth attention and the decoder layers


def _depth_attention(
    block_states: mx.array, partial_block_states: mx.array, query: mx.array
) -> mx.array:
    query = query.astype(mx.float32)
    scale = partial_block_states.shape[-1] ** -0.5
    if _use_kernels():
        return depth_attention(block_states, partial_block_states, query, scale=scale)
    stacked_states = mx.concatenate([block_states, partial_block_states[None]])
    stacked_states = mx.moveaxis(stacked_states, 0, -2)
    weights = mx.softmax((stacked_states @ query.T) * scale, axis=-2, precise=True)
    return weights.swapaxes(-1, -2) @ stacked_states


class BlockAttentionResidual(nn.Module):
    def __init__(self, dims: int, num_queries: int = 1):
        super().__init__()
        shape = (dims,) if num_queries == 1 else (num_queries, dims)
        self.query = mx.zeros(shape, dtype=mx.float32)

    def __call__(
        self, block_states: mx.array, partial_block_states: mx.array
    ) -> mx.array:
        dims = partial_block_states.shape[-1]
        return _depth_attention(
            block_states, partial_block_states, self.query.reshape(-1, dims)
        )


class MessageGate(nn.Module):
    def __init__(self, dims: int, rank: int):
        super().__init__()
        self.down_proj = nn.Linear(2 * dims, rank, bias=False)
        self.forget_gate_proj = nn.Linear(rank, dims, bias=False)
        self.message_gate_proj = nn.Linear(rank, dims, bias=False)

    def __call__(self, hidden_states: mx.array, message: mx.array) -> mx.array:
        states = mx.concatenate(
            [unweighted_rms_norm(hidden_states), unweighted_rms_norm(message)],
            axis=-1,
        )
        states = self.down_proj(states.astype(self.down_proj.weight.dtype))
        states = states.astype(mx.float32)
        forget_gate = mx.tanh(self.forget_gate_proj(states))
        message_gate = mx.tanh(self.message_gate_proj(states))
        return (1 - forget_gate) * hidden_states + message_gate * message


class MessageInput(nn.Module):
    def __init__(self, dims: int, num_looped_layers: int):
        super().__init__()
        # The query, key, value and mlp queries of each looped layer.
        self.query = mx.zeros((num_looped_layers, 4, dims), mx.float32)

    def __call__(self, bank: tuple, loop_layer_idx: int) -> mx.array:
        return _depth_attention(*bank, self.query[loop_layer_idx])


class DecoderLayer(nn.Module):
    def __init__(self, args: TextArgs, layer_idx: int, *, num_queries: int):
        super().__init__()
        dims = args.hidden_size
        self.is_block_boundary = layer_idx % args.n_layers_per_attn_res_block == 0
        self.attention_input = (
            BlockAttentionResidual(dims, num_queries) if layer_idx > 0 else None
        )
        self.mlp_input = BlockAttentionResidual(dims)
        self.mlp = SparseMoeBlock(args)
        self.post_attention_layernorm = nn.RMSNorm(dims, eps=args.rms_norm_eps)
        kernel_size = args.sconv_kernel_size
        self.post_attention_convolution = ShortConvolution(dims, kernel_size)
        self.post_mlp_convolution = ShortConvolution(dims, kernel_size)
        if layer_idx in args.loop_core:
            rank = args.loop_config["gate_rank"]
            self.q_message_gate = MessageGate(dims, rank)
            self.k_message_gate = MessageGate(dims, rank)
            self.v_message_gate = MessageGate(dims, rank)
            self.mlp_message_gate = MessageGate(dims, rank)

    def _mix_messages(self, states: tuple, messages: mx.array) -> tuple:
        q_states, k_states, v_states = states
        return (
            self.q_message_gate(q_states, messages[..., 0, :]),
            self.k_message_gate(k_states, messages[..., 1, :]),
            self.v_message_gate(v_states, messages[..., 2, :]),
        )

    def __call__(
        self,
        hidden_states: mx.array,
        block_states: mx.array | None,
        *,
        cache: Any | None = None,
        conv_mask: mx.array | None = None,
        messages: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        if (
            _DECODE_COMPILE
            and cache is not None
            and hidden_states.shape[1] == 1
            and conv_mask is None
            and cache[1].lengths is None
            and all(c is not None for c in cache[1].cache)
        ):
            return self._decode_step(hidden_states, block_states, cache, messages)
        dtype = hidden_states.dtype
        conv = cache[1] if cache is not None else None
        conv_states = list(conv.cache) if conv is not None else [None] * 4
        block_residual = hidden_states
        normed_residual = unweighted_rms_norm(block_residual.astype(mx.float32))

        attention_inputs = None
        if self.attention_input is not None:
            attention_inputs = self.attention_input(block_states, normed_residual)
        hidden_states = self._self_attention(
            hidden_states, attention_inputs, cache, conv_states, conv_mask, messages
        )
        hidden_states, conv_states[2] = self.post_attention_convolution(
            hidden_states, conv_states[2], conv_mask
        )

        residual = hidden_states = hidden_states.astype(mx.float32)
        if self.is_block_boundary:
            block = normed_residual[None]
            if block_states is None:
                block_states = block
            else:
                block_states = mx.concatenate([block_states, block])
        else:
            residual = hidden_states = block_residual.astype(mx.float32) + hidden_states

        normed_states = unweighted_rms_norm(hidden_states)
        hidden_states = self.mlp_input(block_states, normed_states)[..., 0, :]
        if messages is not None:
            hidden_states = self.mlp_message_gate(hidden_states, messages[..., 3, :])
        hidden_states = self.mlp(
            self.post_attention_layernorm(hidden_states).astype(dtype)
        )
        hidden_states, conv_states[3] = self.post_mlp_convolution(
            hidden_states, conv_states[3], conv_mask
        )

        if conv is not None:
            n = self.post_mlp_convolution.kernel_size - 1
            for i, window in enumerate(conv_states):
                conv[i] = _conv_state(window, conv.lengths, n)
            conv.advance(block_residual.shape[1])
        hidden_states = residual + hidden_states.astype(mx.float32)
        return hidden_states.astype(dtype), block_states


    # ---- decode (one token): two compiled halves around the KV cache

    def _decode_fns(self, key: tuple):
        fns = self.__dict__.get("_compiled_decode")
        if fns is None:
            fns = {}
            object.__setattr__(self, "_compiled_decode", fns)
        if key not in fns:
            has_block, has_msg, has_tau = key
            attn = self.self_attn

            def pre(h, c0, c1, *rest):
                rest = list(rest)
                bs = rest.pop(0) if has_block else None
                msg = rest.pop(0) if has_msg else None
                tau = rest.pop(0) if has_tau else None
                normed_residual = unweighted_rms_norm(h.astype(mx.float32))
                attention_inputs = None
                if self.attention_input is not None:
                    attention_inputs = self.attention_input(bs, normed_residual)
                hq, hk, hv = self._attn_inputs(h, attention_inputs, msg)
                conv_states = [c0, c1]
                q, k, v, r = attn._project(hq, hk, hv, conv_states, None)
                q, k, v, pb, sinks, gate = attn._prep(hq, q, k, v, r, tau)
                return (
                    q,
                    k,
                    v,
                    pb,
                    sinks,
                    gate,
                    conv_states[0][:, 1:],
                    conv_states[1][:, 1:],
                    normed_residual,
                )

            def post(attn_out, gate, h, normed_residual, c2, c3, *rest):
                rest = list(rest)
                bs = rest.pop(0) if has_block else None
                msg = rest.pop(0) if has_msg else None
                dtype = h.dtype
                x = attn._finish(attn_out, gate)
                x, w2 = self.post_attention_convolution(x, c2, None)
                residual = x = x.astype(mx.float32)
                if self.is_block_boundary:
                    block = normed_residual[None]
                    bs = block if bs is None else mx.concatenate([bs, block])
                else:
                    residual = x = h.astype(mx.float32) + x
                x = self.mlp_input(bs, unweighted_rms_norm(x))[..., 0, :]
                if msg is not None:
                    x = self.mlp_message_gate(x, msg[..., 3, :])
                x = self.mlp(self.post_attention_layernorm(x).astype(dtype))
                x, w3 = self.post_mlp_convolution(x, c3, None)
                out = (residual + x.astype(mx.float32)).astype(dtype)
                return out, bs, w2[:, 1:], w3[:, 1:]

            fns[key] = (mx.compile(pre), mx.compile(post))
        return fns[key]

    def _decode_step(self, hidden_states, block_states, cache, messages):
        kv_cache, conv = cache[0], cache[1]
        c0, c1, c2, c3 = conv.cache
        attn = self.self_attn
        tau = None
        if isinstance(attn, FullAttention):
            tau = mx.array(attn.tau_scalar(kv_cache.offset), dtype=mx.float32)
        key = (block_states is not None, messages is not None, tau is not None)
        pre, post = self._decode_fns(key)
        extra = [a for a in (block_states, messages, tau) if a is not None]
        q, k, v, pb, sinks, gate, c0, c1, normed_residual = pre(
            hidden_states, c0, c1, *extra
        )
        k, v = kv_cache.update_and_fetch(k, v)
        attn_out = attn._attention(q, k, v, pb, sinks, kv_cache)
        extra = [a for a in (block_states, messages) if a is not None]
        out, block_states, c2, c3 = post(
            attn_out, gate, hidden_states, normed_residual, c2, c3, *extra
        )
        conv[0], conv[1], conv[2], conv[3] = c0, c1, c2, c3
        conv.advance(1)
        return out, block_states


class SlidingDecoderLayer(DecoderLayer):
    def __init__(self, args: TextArgs, layer_idx: int):
        super().__init__(args, layer_idx, num_queries=1)
        self.self_attn = SlidingAttention(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def _attn_inputs(self, hidden_states, attention_inputs, messages):
        dtype = hidden_states.dtype
        if attention_inputs is not None:
            hidden_states = attention_inputs[..., 0, :]
        if messages is None:
            return (self.input_layernorm(hidden_states).astype(dtype),) * 3
        states = self._mix_messages((hidden_states,) * 3, messages)
        return tuple(self.input_layernorm(s).astype(dtype) for s in states)

    def _self_attention(
        self, hidden_states, attention_inputs, cache, conv_states, conv_mask, messages
    ):
        states = self._attn_inputs(hidden_states, attention_inputs, messages)
        return self.self_attn(
            *states,
            cache=cache,
            conv_states=conv_states,
            conv_mask=conv_mask,
        )


class FullDecoderLayer(DecoderLayer):
    def __init__(self, args: TextArgs, layer_idx: int):
        super().__init__(args, layer_idx, num_queries=3)
        self.self_attn = FullAttention(args)
        eps = args.rms_norm_eps
        self.q_input_layernorm = nn.RMSNorm(args.hidden_size, eps=eps)
        self.k_input_layernorm = nn.RMSNorm(args.hidden_size, eps=eps)
        self.v_input_layernorm = nn.RMSNorm(args.hidden_size, eps=eps)

    def _attn_inputs(self, hidden_states, attention_inputs, messages):
        dtype = hidden_states.dtype
        if attention_inputs is None:
            query_states = key_states = value_states = hidden_states
        else:
            query_states, key_states, value_states = (
                attention_inputs[..., i, :] for i in range(3)
            )
        if messages is not None:
            query_states, key_states, value_states = self._mix_messages(
                (query_states, key_states, value_states), messages
            )
        return (
            self.q_input_layernorm(query_states).astype(dtype),
            self.k_input_layernorm(key_states).astype(dtype),
            self.v_input_layernorm(value_states).astype(dtype),
        )

    def _self_attention(
        self, hidden_states, attention_inputs, cache, conv_states, conv_mask, messages
    ):
        states = self._attn_inputs(hidden_states, attention_inputs, messages)
        return self.self_attn(
            *states,
            cache=cache,
            conv_states=conv_states,
            conv_mask=conv_mask,
        )


# --------------------------------------------------------------------------
# Trunk


class TextModel(nn.Module):
    def __init__(self, args: TextArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        # Not in embed_tokens as in transformers, so that it can be quantized.
        self.embed_norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.layers = [
            (
                SlidingDecoderLayer(args, i)
                if i in args.local_layer_ids
                else FullDecoderLayer(args, i)
            )
            for i in range(args.num_hidden_layers)
        ]
        self.final_input = BlockAttentionResidual(args.hidden_size)
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        if args.loop_config is not None:
            self.message_input = MessageInput(args.hidden_size, len(args.loop_core))

    def embed(self, inputs: mx.array) -> mx.array:
        return self.embed_norm(self.embed_tokens(inputs))

    @property
    def layer_visits(self) -> list:
        if self.args.loop_config is None:
            return list(self.layers)
        core = self.args.loop_core
        passes = self.args.loop_config["num_passes"]
        layers = self.layers
        return (
            layers[: core.start]
            + layers[core.start : core.stop] * passes
            + layers[core.stop :]
        )

    def _run_loop(self, hidden_states, cache, conv_mask):
        core = self.args.loop_core
        caches = iter(cache)

        def run(layers, hidden_states, block_states, bank=None):
            for i, layer in enumerate(layers):
                messages = None if bank is None else self.message_input(bank, i)
                hidden_states, block_states = layer(
                    hidden_states,
                    block_states,
                    cache=next(caches),
                    conv_mask=conv_mask,
                    messages=messages,
                )
            return hidden_states, block_states

        prelude = run(self.layers[: core.start], hidden_states, None)
        hidden_states, block_states = prelude
        for loop_pass in range(self.args.loop_config["num_passes"]):
            bank = None
            if loop_pass > 0:
                bank = (
                    block_states[prelude[1].shape[0] + 1 :],
                    unweighted_rms_norm(hidden_states.astype(mx.float32)),
                )
                hidden_states, block_states = prelude
            hidden_states, block_states = run(
                self.layers[core.start : core.stop], hidden_states, block_states, bank
            )
        return run(self.layers[core.stop :], hidden_states, block_states)

    def __call__(
        self,
        inputs: mx.array,
        cache: list | None = None,
        input_embeddings: mx.array | None = None,
    ) -> mx.array:
        if input_embeddings is None:
            input_embeddings = self.embed(inputs)
        hidden_states = input_embeddings
        dtype = hidden_states.dtype

        if cache is None:
            cache = [None] * len(self.layer_visits)
        _reset_step_geometry()
        conv_mask = None
        if cache[0] is not None:
            conv_mask = cache[0][1].make_mask(hidden_states.shape[1])

        if self.args.loop_config is None:
            block_states = None
            for layer, c in zip(self.layers, cache):
                hidden_states, block_states = layer(
                    hidden_states, block_states, cache=c, conv_mask=conv_mask
                )
        else:
            hidden_states, block_states = self._run_loop(
                hidden_states, cache, conv_mask
            )

        normed_states = unweighted_rms_norm(hidden_states.astype(mx.float32))
        hidden_states = self.final_input(block_states, normed_states)[..., 0, :]
        return self.norm(hidden_states).astype(dtype)


class Inkling2Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.language_model = TextModel(args.text)

    @property
    def layers(self) -> list:
        return self.language_model.layers


# --------------------------------------------------------------------------
# Checkpoint names


_RENAMES = [
    (r"^model\.llm\.unembed\.", "lm_head."),
    (r"^model\.llm\.embed\.", "model.language_model.embed_tokens."),
    (r"^model\.llm\.", "model.language_model."),
    (r"\.attn_res\.mlp_queries\.(\d+)$", r".layers.\1.mlp_input.query"),
    (r"\.attn_res\.output_query$", ".final_input.query"),
    (r"\.attn\.wq_du\.", ".self_attn.q_proj."),
    (r"\.attn\.wk_dv\.", ".self_attn.k_proj."),
    (r"\.attn\.wv_dv\.", ".self_attn.v_proj."),
    (r"\.attn\.wr_du\.", ".self_attn.r_proj."),
    (r"\.attn\.wo_ud\.", ".self_attn.o_proj."),
    (r"\.attn\.gate_attn\.", ".self_attn.gate_proj."),
    (r"\.attn\.attn_sink_proj\.", ".self_attn.sink_proj."),
    (r"\.controllers\.attn\.", ".q_message_gate."),
    (r"\.controllers\.(k|v|mlp)\.", r".\1_message_gate."),
    (r"_message_gate\.down_weight$", "_message_gate.down_proj.weight"),
    (r"_message_gate\.forget_weight$", "_message_gate.forget_gate_proj.weight"),
    (r"_message_gate\.gain_weight$", "_message_gate.message_gate_proj.weight"),
    (r"\.attn\.", ".self_attn."),
    (r"\.attn_sconv\.", ".post_attention_convolution."),
    (r"\.mlp_sconv\.", ".post_mlp_convolution."),
    (r"sconv\.weight$", "sconv.conv1d.weight"),
    (r"convolution\.weight$", "convolution.conv1d.weight"),
    (r"sconv\.static_w$", "sconv.static_weight"),
    (r"sconv\.weight_proj$", "sconv.weight_proj.weight"),
    (r"\.mlp_norm\.", ".post_attention_layernorm."),
    (r"\.attn_norm\.", ".input_layernorm."),
    (r"\.(q|k|v)_input_norm\.", r".\1_input_layernorm."),
    (r"\.mlp\.gate\.bias$", ".mlp.gate.e_score_correction_bias"),
]


def _rename(key: str) -> str:
    import re

    if m := re.match(r"model\.llm\.attn_res\.attn_queries\.(\d+)$", key):
        # Layer 0 has no attention input, so query i is for layer i + 1.
        layer = int(m.group(1)) + 1
        return f"model.language_model.layers.{layer}.attention_input.query"
    for pattern, replacement in _RENAMES:
        key = re.sub(pattern, replacement, key)
    return key


# The projections read once per layer visit. Quantized to 8 bits by default.
_PROJ_SUFFIXES = (
    ".self_attn.q_proj",
    ".self_attn.k_proj",
    ".self_attn.v_proj",
    ".self_attn.r_proj",
    ".self_attn.o_proj",
    ".self_attn.gate_proj",
)
_PROJ_GROUP_SIZE = 64


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Inkling2Model(args)
        text = args.text
        self.lm_head = nn.Linear(text.hidden_size, text.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache: list | None = None,
        input_embeddings: mx.array | None = None,
    ) -> mx.array:
        hidden_states = self.model.language_model(
            inputs, cache=cache, input_embeddings=input_embeddings
        )
        logits = self.lm_head(hidden_states)
        text = self.args.text
        softcap = text.final_logit_softcapping
        logits = logits.astype(mx.float32)
        logits = mx.tanh(logits / softcap) * softcap
        return logits[..., : text.unpadded_vocab_size]

    # ---- loading

    def sanitize(self, weights: dict) -> dict:
        import re

        experts = re.compile(r"^(.+\.mlp\.experts)\.(w13|w2)_weight(\w*)$")
        amax, denominator = {}, {}
        message_queries = {}
        sanitized = {}
        for k, v in weights.items():
            if k.startswith("model.visual."):
                continue
            if k.startswith("model.llm."):
                k = _rename(k)
                if k.endswith("conv1d.weight"):
                    v = v.swapaxes(1, 2)
            if k.endswith(("cross_pass_queries", "cross_pass_kv_queries")):
                message_queries["kv" if "_kv_" in k else "q"] = v
                continue
            if k.endswith(
                (
                    ".mlp.gate.weight",
                    ".forget_gate_proj.weight",
                    ".message_gate_proj.weight",
                )
            ):
                v = v.astype(mx.float32)
            elif m := experts.match(k):
                prefix, which, suffix = m.groups()
                name = f"{prefix}.{'gate_up_proj' if which == 'w13' else 'down_proj'}"
                if suffix == "_amax":
                    amax[name] = v
                elif suffix == "_outer_scale_denominator":
                    denominator[name] = v
                elif suffix == "_scale":
                    sanitized[f"{name}.scales"] = v.view(mx.uint8)
                else:
                    sanitized[f"{name}.weight"] = v.view(mx.uint32)
                continue
            sanitized[k] = v

        if message_queries:
            # The checkpoint has [query, mlp] and [key, value] rows per looped layer.
            dims = message_queries["q"].shape[-1]
            q = message_queries["q"].reshape(-1, 2, dims)
            kv = message_queries["kv"].reshape(-1, 2, dims)
            query = mx.concatenate([q[:, :1], kv, q[:, 1:]], axis=1)
            sanitized["model.language_model.message_input.query"] = query.astype(mx.float32)

        for name in amax.keys() & denominator.keys():
            factor = amax[name].astype(mx.float32) / denominator[name].astype(mx.float32)
            sanitized[f"{name}.expert_scale"] = factor

        # Experts: swap each SwitchLinear for the microscaled module its scales
        # describe (nvfp4: 4 bits per 16, mxfp4: 4 per 32, mxfp8: 8 per 32).
        modes = {(4, 16): "nvfp4", (4, 32): "mxfp4", (8, 32): "mxfp8"}
        swapped = []
        for path, module in self.named_modules():
            if not isinstance(module, SwitchLinear):
                continue
            scales = sanitized.get(f"{path}.scales")
            if scales is None or scales.dtype != mx.uint8:
                continue
            num_experts, out_dims, in_dims = module.weight.shape
            group_size = in_dims // scales.shape[-1]
            bits = sanitized[f"{path}.weight"].shape[-1] * 32 // in_dims
            mode = modes.get((bits, group_size))
            if mode is None:
                raise ValueError(f"{path}: unsupported expert format bits={bits} group={group_size}")
            swapped.append(
                (
                    path,
                    ExpertLinear(
                        in_dims,
                        out_dims,
                        num_experts,
                        group_size=group_size,
                        bits=bits,
                        mode=mode,
                        expert_scale=f"{path}.expert_scale" in sanitized,
                    ),
                )
            )
        if swapped:
            self.update_modules(tree_unflatten(swapped))

        self._quantize_projections(sanitized)
        return sanitized

    def _quantize_projections(self, sanitized: dict) -> None:
        """Requantize the BF16 projections (8-bit affine by default)."""

        proj_bits = _env_int("MTPLX_INKLING2_PROJ_BITS", 8)
        head_bits = _env_int("MTPLX_INKLING2_HEAD_BITS", 8)
        targets: dict[str, int] = {}
        for path, module in self.named_modules():
            if not isinstance(module, (nn.Linear, nn.Embedding)):
                continue
            if path.endswith(_PROJ_SUFFIXES):
                bits = proj_bits
            elif path in ("lm_head", "model.language_model.embed_tokens"):
                bits = head_bits
            else:
                continue
            if bits and f"{path}.weight" in sanitized and f"{path}.scales" not in sanitized:
                targets[path] = bits
        if not targets:
            return
        for path, bits in targets.items():
            w = sanitized.pop(f"{path}.weight")
            wq, scales, biases = mx.quantize(w, group_size=_PROJ_GROUP_SIZE, bits=bits)
            sanitized[f"{path}.weight"] = wq
            sanitized[f"{path}.scales"] = scales
            sanitized[f"{path}.biases"] = biases

        def predicate(path: str, module: nn.Module):
            bits = targets.get(path)
            if bits is None:
                return False
            return {"group_size": _PROJ_GROUP_SIZE, "bits": bits, "mode": "affine"}

        nn.quantize(self, class_predicate=predicate)

    @property
    def cast_predicate(self):
        def predicate(k: str) -> bool:
            return not k.endswith(
                (
                    ".query",
                    ".mlp.gate.weight",
                    ".mlp.gate.e_score_correction_bias",
                    ".forget_gate_proj.weight",
                    ".message_gate_proj.weight",
                    ".expert_scale",
                )
            )

        return predicate

    @property
    def layers(self) -> list:
        return self.model.language_model.layers

    def make_cache(self) -> list:
        window = self.args.text.sliding_window_size
        return [
            CacheList(
                (
                    RotatingKVCache(max_size=window)
                    if isinstance(layer, SlidingDecoderLayer)
                    else KVCache()
                ),
                ArraysCache(4),
            )
            for layer in self.model.language_model.layer_visits
        ]
