"""Synchronized target-verify phase profiler for Qwen3.5/Qwen3.6."""

from __future__ import annotations

import json
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mtplx.benchmarks.schema import encode_prompt_case, load_prompt_suite
from mtplx.runtime import load


@dataclass
class _ProfileAccumulator:
    totals: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    layers: list[dict[str, Any]] = field(default_factory=list)

    def add(self, key: str, elapsed: float) -> None:
        self.totals[key] += elapsed

    def time(self, key: str, fn: Callable[[], Any]) -> Any:
        started = time.perf_counter()
        value = fn()
        if isinstance(value, tuple):
            mx.eval(*value)
        else:
            mx.eval(value)
        self.add(key, time.perf_counter() - started)
        return value


def _candidate_tokens(tokenizer: Any, min_len: int) -> list[int]:
    text = (
        "    result = a + b\n"
        "    return result\n"
        "\n"
        "def subtract(a, b):\n"
        "    return a - b\n"
        "\n"
        "def multiply(a, b):\n"
        "    return a * b\n"
    )
    ids = tokenizer.encode(text)
    while len(ids) < min_len:
        ids.extend(ids)
    return [int(x) for x in ids[:min_len]]


def _gdn_forward_profile(gdn: Any, inputs: mx.array, mask: Any, cache: Any, acc: _ProfileAccumulator) -> mx.array:
    from mlx_lm.models.gated_delta import compute_g, gated_delta_kernel, gated_delta_ops

    B, S, _ = inputs.shape

    qkv, z, b, a = acc.time(
        "gdn_projections_s",
        lambda: (
            gdn.in_proj_qkv(inputs),
            gdn.in_proj_z(inputs).reshape(B, S, gdn.num_v_heads, gdn.head_v_dim),
            gdn.in_proj_b(inputs),
            gdn.in_proj_a(inputs),
        ),
    )

    if cache is not None and cache[0] is not None:
        conv_state = cache[0]
    else:
        conv_state = mx.zeros(
            (B, gdn.conv_kernel_size - 1, gdn.conv_dim),
            dtype=inputs.dtype,
        )

    def run_conv():
        qkv_masked = mx.where(mask[..., None], qkv, 0) if mask is not None else qkv
        conv_input = mx.concatenate([conv_state, qkv_masked], axis=1)
        if cache is not None:
            n_keep = gdn.conv_kernel_size - 1
            if cache.lengths is not None:
                ends = mx.clip(cache.lengths, 0, S)
                positions = (ends[:, None] + mx.arange(n_keep))[..., None]
                cache[0] = mx.take_along_axis(conv_input, positions, axis=1)
            else:
                cache[0] = mx.contiguous(conv_input[:, -n_keep:, :])
        return nn.silu(gdn.conv1d(conv_input))

    conv_out = acc.time("gdn_conv_s", run_conv)

    def prepare():
        q, k, v = [
            t.reshape(B, S, h, d)
            for t, h, d in zip(
                mx.split(conv_out, [gdn.key_dim, 2 * gdn.key_dim], -1),
                [gdn.num_k_heads, gdn.num_k_heads, gdn.num_v_heads],
                [gdn.head_k_dim, gdn.head_k_dim, gdn.head_v_dim],
            )
        ]
        inv_scale = k.shape[-1] ** -0.5
        q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
        k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)
        return q, k, v

    q, k, v = acc.time("gdn_prepare_s", prepare)
    g, beta = acc.time(
        "gdn_gates_s",
        lambda: (compute_g(gdn.A_log, a, gdn.dt_bias), mx.sigmoid(b)),
    )

    state = cache[1] if cache else None
    if state is None:
        state = mx.zeros(
            (B, gdn.num_v_heads, gdn.head_v_dim, gdn.head_k_dim),
            dtype=mx.float32,
        )
    use_kernel = (
        not getattr(gdn, "training", False)
        and mx.default_device() == mx.gpu
        and mx.metal.is_available()
    )
    out, state = acc.time(
        "gdn_recurrent_s",
        lambda: (
            gated_delta_kernel(q, k, v, g, beta, state, mask)
            if use_kernel
            else gated_delta_ops(q, k, v, g, beta, state, mask)
        ),
    )

    if cache is not None:
        cache[1] = state
        cache.advance(S)

    out = acc.time("gdn_norm_s", lambda: gdn.norm(out, z))
    return acc.time("gdn_out_proj_s", lambda: gdn.out_proj(out.reshape(B, S, -1)))


def _attention_forward_profile(attn: Any, x: mx.array, mask: Any, cache: Any, acc: _ProfileAccumulator) -> mx.array:
    from mlx_lm.models.base import scaled_dot_product_attention

    B, L, _ = x.shape

    q_proj_output, keys, values = acc.time(
        "attn_projections_s",
        lambda: (attn.q_proj(x), attn.k_proj(x), attn.v_proj(x)),
    )

    queries, gate = mx.split(
        q_proj_output.reshape(B, L, attn.num_attention_heads, -1),
        2,
        axis=-1,
    )
    gate = gate.reshape(B, L, -1)

    # The former single "attn_qkv_norm_rope_cache_s" section is split so a
    # context-dependent cost can be pinned to norm, RoPE or the cache update
    # (their sum equals the old section).
    def norm_qkv():
        q = attn.q_norm(queries).transpose(0, 2, 1, 3)
        k = attn.k_norm(keys.reshape(B, L, attn.num_key_value_heads, -1)).transpose(0, 2, 1, 3)
        v = values.reshape(B, L, attn.num_key_value_heads, -1).transpose(0, 2, 1, 3)
        return q, k, v

    queries, keys, values = acc.time("attn_qk_norm_s", norm_qkv)

    def rope_qk():
        if cache is not None:
            return attn.rope(queries, offset=cache.offset), attn.rope(keys, offset=cache.offset)
        return attn.rope(queries), attn.rope(keys)

    queries, keys = acc.time("attn_rope_s", rope_qk)
    if cache is not None:
        capacity_before = None if getattr(cache, "keys", None) is None else int(cache.keys.shape[2])
        keys, values = acc.time(
            "attn_cache_update_s", lambda: cache.update_and_fetch(keys, values)
        )
        if capacity_before is not None and int(cache.keys.shape[2]) != capacity_before:
            acc.add("attn_cache_growth_events", 1.0)
    output = acc.time(
        "attn_sdpa_s",
        lambda: scaled_dot_product_attention(
            queries,
            keys,
            values,
            cache=cache,
            scale=attn.scale,
            mask=mask,
        ),
    )
    output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
    return acc.time("attn_out_proj_s", lambda: attn.o_proj(output * mx.sigmoid(gate)))


def _profiled_forward(model: Any, inputs: mx.array, cache: Any) -> tuple[mx.array, dict[str, Any]]:
    from mlx_lm.models.base import create_attention_mask, create_ssm_mask

    text_model = getattr(model, "language_model", model)
    inner = text_model.model
    acc = _ProfileAccumulator()

    hidden_states = acc.time("embed_s", lambda: inner.embed_tokens(inputs))
    if cache is None:
        cache = [None] * len(inner.layers)

    fa_mask = create_attention_mask(hidden_states, cache[inner.fa_idx])
    ssm_mask = create_ssm_mask(hidden_states, cache[inner.ssm_idx])

    for layer_idx, (layer, layer_cache) in enumerate(zip(inner.layers, cache)):
        layer_started = time.perf_counter()
        layer_kind = "gdn" if layer.is_linear else "attention"
        normed = acc.time("input_norm_s", lambda layer=layer: layer.input_layernorm(hidden_states))
        if layer.is_linear:
            r = _gdn_forward_profile(layer.linear_attn, normed, ssm_mask, layer_cache, acc)
        else:
            r = _attention_forward_profile(layer.self_attn, normed, fa_mask, layer_cache, acc)
        h = acc.time("residual_s", lambda: hidden_states + r)
        mlp_input = acc.time("post_attention_norm_s", lambda layer=layer: layer.post_attention_layernorm(h))
        mlp_out = acc.time("mlp_s", lambda layer=layer: layer.mlp(mlp_input))
        hidden_states = acc.time("mlp_residual_s", lambda: h + mlp_out)
        acc.layers.append(
            {
                "layer": layer_idx,
                "kind": layer_kind,
                "elapsed_s": time.perf_counter() - layer_started,
            }
        )

    post_norm = acc.time("final_norm_s", lambda: inner.norm(hidden_states))
    logits = acc.time(
        "lm_head_s",
        lambda: (
            inner.embed_tokens.as_linear(post_norm)
            if text_model.args.tie_word_embeddings
            else text_model.lm_head(post_norm)
        ),
    )
    return logits, {"sections": dict(acc.totals), "layers": acc.layers}


def _prefill(rt: Any, prompt_ids: list[int], *, chunk_tokens: int = 2048) -> Any:
    cache = rt.make_cache()
    for start in range(0, len(prompt_ids), chunk_tokens):
        chunk = prompt_ids[start : start + chunk_tokens]
        logits = rt.forward_ar(mx.array([chunk]), cache=cache, return_hidden=False)
        mx.eval(logits)
    return cache


def _max_abs_diff(a: mx.array, b: mx.array) -> float:
    diff = mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32)))
    mx.eval(diff)
    return float(np.asarray(diff).item())


def _mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def run_verify_profile(
    model_path: Path | str,
    prompt_suite: Path | str,
    *,
    lengths: list[int],
    repeats: int = 2,
    warmup: int = 1,
    prompt_index: int = 0,
    enable_thinking: bool | None = None,
    context_tokens: int | None = None,
    diagnostic_only: bool = False,
) -> dict[str, Any]:
    rt = load(model_path, mtp=True)
    prompts = load_prompt_suite(prompt_suite)
    case = prompts[prompt_index]
    prompt_ids = encode_prompt_case(
        rt.tokenizer,
        case,
        chat_template=True,
        enable_thinking=enable_thinking,
    )
    candidates = _candidate_tokens(rt.tokenizer, max(lengths))

    if context_tokens is not None:
        if context_tokens < 1:
            raise ValueError("context_tokens must be positive")
        context_ids = list(prompt_ids)
        filler = _candidate_tokens(
            rt.tokenizer,
            max(context_tokens - len(context_ids), 1),
        )
        context_ids = (context_ids + filler)[:context_tokens]
    else:
        context_ids = prompt_ids

    rows = []
    for length in lengths:
        input_ids = mx.array([candidates[:length]])
        stock_elapsed = None
        if not diagnostic_only:
            stock_cache = _prefill(rt, context_ids)
            stock_started = time.perf_counter()
            stock_logits = rt.forward_ar(input_ids, cache=stock_cache, return_hidden=False)
            mx.eval(stock_logits)
            stock_elapsed = time.perf_counter() - stock_started

        manual_cache = _prefill(rt, context_ids)
        active_memory_before = (
            int(mx.metal.get_active_memory()) if mx.metal.is_available() else None
        )
        if mx.metal.is_available():
            mx.metal.reset_peak_memory()
        manual_started = time.perf_counter()
        manual_logits, first_profile = _profiled_forward(rt.model, input_ids, manual_cache)
        mx.eval(manual_logits)
        first_elapsed = time.perf_counter() - manual_started
        max_abs_diff = (
            _max_abs_diff(stock_logits, manual_logits)
            if not diagnostic_only
            else None
        )

        repeat_profiles = []
        for repeat in range(warmup + repeats):
            cache = _prefill(rt, context_ids)
            started = time.perf_counter()
            logits, profile = _profiled_forward(rt.model, input_ids, cache)
            mx.eval(logits)
            elapsed = time.perf_counter() - started
            if repeat >= warmup:
                repeat_profiles.append(
                    {
                        "elapsed_s": elapsed,
                        "sections": profile["sections"],
                        "layers": profile["layers"],
                    }
                )

        section_keys = sorted(
            {
                key
                for profile in repeat_profiles
                for key in profile["sections"].keys()
            }
        )
        section_means = {
            key: _mean([profile["sections"].get(key, 0.0) for profile in repeat_profiles])
            for key in section_keys
        }
        layer_kind_means = {
            "gdn_layers_s": _mean(
                [
                    sum(layer["elapsed_s"] for layer in profile["layers"] if layer["kind"] == "gdn")
                    for profile in repeat_profiles
                ]
            ),
            "attention_layers_s": _mean(
                [
                    sum(layer["elapsed_s"] for layer in profile["layers"] if layer["kind"] == "attention")
                    for profile in repeat_profiles
                ]
            ),
        }
        elapsed_mean = _mean([profile["elapsed_s"] for profile in repeat_profiles])
        active_memory_after = (
            int(mx.metal.get_active_memory()) if mx.metal.is_available() else None
        )
        peak_memory = (
            int(mx.metal.get_peak_memory()) if mx.metal.is_available() else None
        )
        rows.append(
            {
                "tokens": length,
                "context_tokens": len(context_ids),
                "stock_elapsed_s": stock_elapsed,
                "manual_first_elapsed_s": first_elapsed,
                "manual_elapsed_mean_s": elapsed_mean,
                "manual_over_stock_ratio": elapsed_mean / stock_elapsed if stock_elapsed else None,
                "logit_max_abs_diff": max_abs_diff,
                "active_memory_before_verify_bytes": active_memory_before,
                "active_memory_after_verify_bytes": active_memory_after,
                "peak_memory_bytes": peak_memory,
                "section_means_s": section_means,
                "layer_kind_means_s": layer_kind_means,
                "first_profile_sections_s": first_profile["sections"],
                "repeats": repeats,
                "warmup": warmup,
            }
        )

    return {
        "model_path": str(model_path),
        "prompt_suite": str(prompt_suite),
        "prompt_id": case.id,
        "prompt_category": case.category,
        "prompt_sha256": case.prompt_sha256,
        "prompt_tokens": len(prompt_ids),
        "context_tokens": len(context_ids),
        "diagnostic_only": diagnostic_only,
        "enable_thinking": enable_thinking,
        "lengths": lengths,
        "repeats": repeats,
        "warmup": warmup,
        "rows": rows,
        "note": "Synchronized section timing inflates absolute latency; use for attribution, not headline tok/s.",
    }


def write_verify_profile(path: Path | str, result: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True))
