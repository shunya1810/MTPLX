"""Plain paged-KV quantization helpers.

This is intentionally separate from TurboQuant.  TurboQuant depends on the
external vLLM-Metal encode/attention kernels; this module provides an in-tree
q8/q4 storage mode that can always fall back through MLX SDPA after dequant.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any


MODES = {"q8", "int8", "q4", "int4"}


@dataclass(frozen=True)
class PagedKVQuantConfig:
    mode: str = "q8"

    @property
    def normalized_mode(self) -> str:
        raw = self.mode.strip().lower().replace("-", "_")
        if raw in {"int8", "q8_0"}:
            return "q8"
        if raw in {"int4", "q4_0"}:
            return "q4"
        return raw

    @property
    def bits(self) -> int:
        return 4 if self.normalized_mode == "q4" else 8


def env_enabled(name: str, *, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


AUTO_THRESHOLD_ENV = "MTPLX_PAGED_KV_QUANT_AUTO_THRESHOLD"
DEFAULT_AUTO_THRESHOLD_TOKENS = 131072


def paged_kv_quant_setting_from_env() -> str:
    """The configured mode, canonicalized: ``off``/``q8``/``q4``/``auto``.

    Launchers and reports read this; the KV layout readers use
    :func:`paged_kv_quant_mode_from_env`, which resolves ``auto`` per request.
    """

    from mtplx.runtime_options import normalize_paged_kv_quantization

    raw = (
        os.environ.get("MTPLX_VLLM_METAL_PAGED_KV_QUANT")
        or os.environ.get("MTPLX_PAGED_KV_QUANT")
        or ""
    )
    return str(normalize_paged_kv_quantization(raw))


def auto_threshold_tokens() -> int:
    raw = os.environ.get(AUTO_THRESHOLD_ENV, "").strip()
    try:
        return max(1, int(raw)) if raw else DEFAULT_AUTO_THRESHOLD_TOKENS
    except ValueError:
        return DEFAULT_AUTO_THRESHOLD_TOKENS


def resolve_auto_mode(context_tokens: int) -> str:
    """``auto``: q8 from the threshold up, off below it.

    M1 Max, Qwen3.8-27B, published bench (docs/benchmarks/m1max-longctx): at
    128K q8 decodes 18.9 tok/s vs 16.7 for fp16 with a 34.1 vs 40.7 GB peak;
    at 256K fp16's 56.7 GB peak passes the 55.7 GB recommended working set
    while q8 stays at 41.7 GB. Below 64K fp16 is as fast or faster and keeps
    the unquantized output.
    """

    return "q8" if int(context_tokens) >= auto_threshold_tokens() else "off"


def paged_kv_quant_mode_from_env() -> str:
    """The single parse of the paged-KV-quant env pair, canonicalized.

    Returns one of ``off``/``q8``/``q4``. Every reader goes through
    :func:`~mtplx.runtime_options.normalize_paged_kv_quantization`, so
    spellings like ``8``/``8bit``/``uint8`` cannot normalize in one reader,
    raise in a second, and fall through to the wrong KV layout in a third.
    ``auto`` resolves against the current request's prompt length
    (``MTPLX_CURRENT_PREFILL_CONTEXT_TOKENS``, set before its prefill), so
    every reader within one request sees the same mode.
    """

    mode = paged_kv_quant_setting_from_env()
    if mode == "auto":
        try:
            context = int(os.environ.get("MTPLX_CURRENT_PREFILL_CONTEXT_TOKENS") or 0)
        except ValueError:
            context = 0
        return resolve_auto_mode(context)
    return mode


def config_from_env() -> PagedKVQuantConfig | None:
    mode = paged_kv_quant_mode_from_env()
    if mode == "off":
        return None
    return PagedKVQuantConfig(mode=mode)


def packed_dim(head_dim: int, bits: int) -> int:
    head_dim = int(head_dim)
    bits = int(bits)
    if bits == 8:
        return head_dim
    if bits == 4:
        if head_dim % 2:
            raise ValueError(f"q4 paged KV quantization requires even head_dim, got {head_dim}")
        return head_dim // 2
    raise ValueError(f"unsupported paged KV quantization bits={bits}")


def quantize_symmetric(x: Any, *, bits: int) -> tuple[Any, Any]:
    import mlx.core as mx

    bits = int(bits)
    qmax = 127 if bits == 8 else 7
    max_abs = mx.max(mx.abs(x.astype(mx.float32)), axis=-1, keepdims=True)
    scale = mx.maximum(max_abs / float(qmax), mx.array(1.0e-6, dtype=mx.float32))
    q = mx.round(x.astype(mx.float32) / scale)
    q = mx.clip(q, -float(qmax), float(qmax))
    # Scales stay fp32 end-to-end: they are computed here in fp32, stored
    # fp32 (cache_state scale caches), and consumed in fp32 by
    # dequantize_symmetric and the paged q8 kernel. The former fp16
    # round-trip added avoidable error on top of the int quantization for a
    # ~1.5% (q8) memory saving on the scale sidecar only.
    if bits == 8:
        return q.astype(mx.int8), scale
    if bits == 4:
        unsigned = (q + 8).astype(mx.uint8)
        even = unsigned[..., 0::2]
        odd = unsigned[..., 1::2]
        packed = mx.bitwise_or(even, mx.left_shift(odd, 4)).astype(mx.uint8)
        return packed, scale
    raise ValueError(f"unsupported paged KV quantization bits={bits}")


def dequantize_symmetric(q: Any, scale: Any, *, bits: int, head_dim: int) -> Any:
    import mlx.core as mx

    bits = int(bits)
    if bits == 8:
        return q.astype(mx.float32) * scale.astype(mx.float32)
    if bits == 4:
        low = mx.bitwise_and(q, 0x0F)
        high = mx.bitwise_and(mx.right_shift(q, 4), 0x0F)
        stacked = mx.stack([low, high], axis=-1).reshape(*q.shape[:-1], int(head_dim))
        signed = stacked.astype(mx.int16) - 8
        return signed.astype(mx.float32) * scale.astype(mx.float32)
    raise ValueError(f"unsupported paged KV quantization bits={bits}")


def compression_ratio(*, head_dim: int, bits: int) -> float:
    head_dim = int(head_dim)
    bits = int(bits)
    # Two fp16 tensors, key + value.
    fp16_bytes = 2 * head_dim * 2
    # Two quantized tensors plus one fp32 scale for K and one for V.
    quant_bytes = 2 * packed_dim(head_dim, bits) + 2 * 4
    return float(fp16_bytes) / float(quant_bytes)
