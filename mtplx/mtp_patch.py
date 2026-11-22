"""Runtime MTP injection for Qwen3.6/Qwen3.5 MLX models."""

from __future__ import annotations

import inspect
import json
import logging
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from .artifacts import expected_mtp_file, is_mtp_key, normalize_mtp_key, text_config
from .constants import (
    EXPECTED_ALL_PREQUANTIZED_MTP_KEYS,
    EXPECTED_PREQUANTIZED_MTP_KEYS,
    EXPECTED_QWEN_MOE_PREQUANTIZED_MTP_KEYS,
    EXPECTED_QWEN_MOE_SWITCH_MLP_PREQUANTIZED_MTP_KEYS,
    expand_mtp_layer_keys,
)
from .expert_layout import (
    num_experts_from_config,
    split_fused_experts,
    stack_numbered_experts,
)

logger = logging.getLogger(__name__)

_MTP_QUANT_POLICY_ALIASES = {
    "prequantized-int4": "cyankiwi",
}



def _prefill_layer_eval_every() -> int:
    """Layers between evals in an eager prefill forward (``MTPLX_PREFILL_LAYER_EVAL_EVERY``, 0 = off).

    With large Metal command buffers (MLX_MAX_OPS_PER_BUFFER / MLX_MAX_MB_PER_BUFFER)
    a prefill chunk keeps many layers' transients alive until its buffer commits;
    an eval every few layers bounds that without touching decode, whose verify is a
    compiled graph (the phase gate keeps this out of every non-prefill forward).
    """
    raw = (os.environ.get("MTPLX_PREFILL_LAYER_EVAL_EVERY") or "").strip()
    if not raw:
        return 0
    try:
        every = int(raw)
    except ValueError:
        return 0
    if every <= 0:
        return 0
    from .attention_context import current_attention_phase

    return every if current_attention_phase() == "prefill" else 0

def _canonical_mtp_quant_policy(policy: str | None) -> str | None:
    if policy is None:
        return None
    normalized = str(policy).strip()
    return _MTP_QUANT_POLICY_ALIASES.get(normalized, normalized)


_RMSNORM_SUFFIXES = (
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "q_norm.weight",
    "k_norm.weight",
    "pre_fc_norm_hidden.weight",
    "pre_fc_norm_embedding.weight",
    "norm.weight",
)


@dataclass(frozen=True)
class MTPContract:
    base_hidden_variant: str = "post_norm"
    hidden_variant: str = "post_norm"
    concat_order: str = "embedding_hidden"
    mtp_position_mode: str = "cache"
    mtp_quant_bits: int | None = None
    mtp_quant_group_size: int = 64
    mtp_quant_mode: str = "affine"
    mtp_quant_policy: str | None = None
    mtp_prequantized: bool = False
    mtp_prequantized_modules: tuple[str, ...] = ()
    mtp_prequantized_module_specs: dict[str, dict[str, Any]] = field(default_factory=dict)

    def validate(self) -> None:
        if self.base_hidden_variant not in {"pre_norm", "post_norm"}:
            raise ValueError("base_hidden_variant must be 'pre_norm' or 'post_norm'")
        if not _valid_mtp_hidden_variant(self.hidden_variant):
            raise ValueError(
                "hidden_variant must be 'fc', 'pre_norm', 'post_norm', "
                "'embedding', 'prev', or mix:<left>:<right>:<alpha>"
            )
        if self.concat_order not in {"embedding_hidden", "hidden_embedding"}:
            raise ValueError("concat_order must be 'embedding_hidden' or 'hidden_embedding'")
        if self.mtp_position_mode not in {"cache", "local", "absolute"}:
            raise ValueError("mtp_position_mode must be 'cache', 'local', or 'absolute'")
        if self.mtp_quant_bits is not None and self.mtp_quant_bits <= 0:
            raise ValueError("mtp_quant_bits must be positive when set")
        if self.mtp_quant_group_size <= 0:
            raise ValueError("mtp_quant_group_size must be positive")
        if _canonical_mtp_quant_policy(self.mtp_quant_policy) not in {None, "all", "cyankiwi"}:
            raise ValueError("mtp_quant_policy must be None, 'all', 'cyankiwi', or 'prequantized-int4'")
        for module, spec in self.mtp_prequantized_module_specs.items():
            bits = spec.get("bits")
            group_size = spec.get("group_size")
            if bits is None or int(bits) <= 0:
                raise ValueError(f"prequantized module {module} has invalid bits")
            if group_size is None or int(group_size) <= 0:
                raise ValueError(f"prequantized module {module} has invalid group_size")

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "base_hidden_variant": self.base_hidden_variant,
            "hidden_variant": self.hidden_variant,
            "concat_order": self.concat_order,
            "mtp_position_mode": self.mtp_position_mode,
        }
        if self.mtp_quant_bits is not None:
            out["mtp_quant_bits"] = int(self.mtp_quant_bits)
        out["mtp_quant_group_size"] = int(self.mtp_quant_group_size)
        out["mtp_quant_mode"] = self.mtp_quant_mode
        if self.mtp_quant_policy is not None:
            out["mtp_quant_policy"] = self.mtp_quant_policy
        if self.mtp_prequantized:
            out["mtp_prequantized"] = True
        if self.mtp_prequantized_modules:
            out["mtp_prequantized_modules"] = list(self.mtp_prequantized_modules)
        if self.mtp_prequantized_module_specs:
            out["mtp_prequantized_module_specs"] = {
                str(module): {
                    "bits": int(spec["bits"]),
                    "group_size": int(spec["group_size"]),
                    "mode": str(spec.get("mode") or self.mtp_quant_mode),
                }
                for module, spec in sorted(self.mtp_prequantized_module_specs.items())
            }
        return out

    def with_config_defaults(self, config: dict[str, Any]) -> "MTPContract":
        mtp_quant = config.get("mtplx_mtp_quantization")
        contract_data = config.get("mtplx_mtp_contract")
        contract = self.with_metadata(contract_data, preserve_explicit=True)
        if not isinstance(mtp_quant, dict) or not mtp_quant:
            return contract
        prequantized = bool(mtp_quant.get("prequantized"))
        updates: dict[str, Any] = {}
        if (prequantized or contract.mtp_quant_bits is None) and mtp_quant.get("bits") is not None:
            updates["mtp_quant_bits"] = int(mtp_quant["bits"])
        if (
            prequantized or contract.mtp_quant_group_size == 64
        ) and mtp_quant.get("group_size") is not None:
            updates["mtp_quant_group_size"] = int(mtp_quant["group_size"])
        if (prequantized or contract.mtp_quant_mode == "affine") and mtp_quant.get("mode") is not None:
            updates["mtp_quant_mode"] = str(mtp_quant["mode"])
        if (prequantized or contract.mtp_quant_policy is None) and mtp_quant.get("policy") is not None:
            updates["mtp_quant_policy"] = str(mtp_quant["policy"])
        if not contract.mtp_prequantized and mtp_quant.get("prequantized") is not None:
            updates["mtp_prequantized"] = bool(mtp_quant["prequantized"])
        return replace(contract, **updates) if updates else contract

    def with_metadata(
        self,
        metadata: dict[str, Any] | None,
        *,
        preserve_explicit: bool = True,
    ) -> "MTPContract":
        if not isinstance(metadata, dict):
            return self
        defaults = MTPContract()
        field_map = {
            "base_hidden_variant": "base_hidden_variant",
            "hidden_variant": "hidden_variant",
            "mtp_hidden_variant": "hidden_variant",
            "concat_order": "concat_order",
            "mtp_position_mode": "mtp_position_mode",
            "position_mode": "mtp_position_mode",
            "mtp_quant_bits": "mtp_quant_bits",
            "mtp_quant_group_size": "mtp_quant_group_size",
            "mtp_quant_mode": "mtp_quant_mode",
            "mtp_quant_policy": "mtp_quant_policy",
            "mtp_prequantized": "mtp_prequantized",
        }
        updates: dict[str, Any] = {}
        for source_key, field_name in field_map.items():
            if source_key not in metadata:
                continue
            if preserve_explicit and getattr(self, field_name) != getattr(defaults, field_name):
                continue
            value = metadata[source_key]
            if field_name in {"mtp_quant_bits", "mtp_quant_group_size"} and value is not None:
                value = int(value)
            elif field_name == "mtp_prequantized":
                value = bool(value)
            elif value is not None:
                value = str(value)
            updates[field_name] = value
        modules = metadata.get("mtp_prequantized_modules")
        if (
            isinstance(modules, list | tuple)
            and (not preserve_explicit or not self.mtp_prequantized_modules)
        ):
            updates["mtp_prequantized_modules"] = tuple(str(item) for item in modules)
        module_specs = _normalize_prequantized_module_specs(
            metadata.get("mtp_prequantized_module_specs")
        )
        if module_specs and (
            not preserve_explicit or not self.mtp_prequantized_module_specs
        ):
            updates["mtp_prequantized_module_specs"] = module_specs
        updated = replace(self, **updates) if updates else self
        updated.validate()
        return updated

    def with_runtime_metadata(
        self,
        metadata: dict[str, Any] | None,
        *,
        preserve_explicit: bool = True,
    ) -> "MTPContract":
        if not isinstance(metadata, dict):
            return self
        contract_data = metadata.get("mtp_contract")
        if isinstance(contract_data, dict):
            return self.with_metadata(contract_data, preserve_explicit=preserve_explicit)
        return self.with_metadata(metadata, preserve_explicit=preserve_explicit)


def _valid_mtp_hidden_variant(value: str) -> bool:
    if value in {"fc", "pre_norm", "post_norm", "embedding", "prev"}:
        return True
    if not value.startswith("mix:"):
        return False
    parts = value.split(":")
    if len(parts) != 4:
        return False
    sources = {"fc", "pre_norm", "post_norm", "embedding", "prev"}
    if parts[1] not in sources or parts[2] not in sources:
        return False
    try:
        alpha = float(parts[3].replace("p", "."))
    except ValueError:
        return False
    return 0.0 <= alpha <= 1.0


def _normalize_prequantized_module_specs(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, dict):
        return {}
    normalized: dict[str, dict[str, Any]] = {}
    for module, raw_spec in value.items():
        if not isinstance(raw_spec, dict):
            continue
        bits = raw_spec.get("bits")
        group_size = raw_spec.get("group_size")
        if bits is None or group_size is None:
            continue
        try:
            bit_value = int(bits)
            group_value = int(group_size)
        except (TypeError, ValueError):
            continue
        if bit_value <= 0 or group_value <= 0:
            continue
        normalized[str(module)] = {
            "bits": bit_value,
            "group_size": group_value,
            "mode": str(raw_spec.get("mode") or "affine"),
        }
    return normalized


def _num_mtp_layers(config: dict[str, Any]) -> int:
    tcfg = text_config(config)
    return int(
        tcfg.get("mtp_num_hidden_layers")
        or tcfg.get("num_nextn_predict_layers")
        or config.get("num_nextn_predict_layers")
        or 0
    )


def _text_model(model: Any) -> Any:
    return getattr(model, "language_model", model)


def _qwen36_a3b_text_geometry(tcfg: dict) -> bool:
    """Qwen3.6-35B-A3B's exact text geometry (the M1 A3B lane's kernels' shape)."""

    try:
        return (
            str(tcfg.get("model_type") or "").lower().startswith("qwen3_5_moe")
            and int(tcfg.get("num_experts") or 0) == 256
            and int(tcfg.get("num_experts_per_tok") or 0) == 8
            and int(tcfg.get("hidden_size") or 0) == 2048
            and int(tcfg.get("num_hidden_layers") or 0) == 40
        )
    except (TypeError, ValueError):
        return False


def _quantize_mtp_module(mtp: Any, contract: MTPContract) -> None:
    import mlx.nn as nn

    module_specs = contract.mtp_prequantized_module_specs
    if contract.mtp_quant_bits is None and not module_specs:
        return

    if contract.mtp_prequantized and contract.mtp_prequantized_modules:
        modules = set(contract.mtp_prequantized_modules)

        def prequantized_predicate(path: str, module: Any):
            if path in modules and hasattr(module, "to_quantized"):
                spec = module_specs.get(path, {})
                bits = spec.get("bits", contract.mtp_quant_bits)
                group_size = spec.get("group_size", contract.mtp_quant_group_size)
                if bits is None:
                    return False
                return {
                    "group_size": int(group_size),
                    "bits": int(bits),
                    "mode": str(spec.get("mode") or contract.mtp_quant_mode),
                }
            return False

        nn.quantize(mtp, class_predicate=prequantized_predicate)
        return

    policy = _canonical_mtp_quant_policy(contract.mtp_quant_policy) or "all"
    _scope = os.environ.get("MTPLX_MTP_QUANT_SCOPE", "").strip()
    if policy == "all" and _scope:
        # Experiment switch: quantize only modules whose path contains one of
        # the comma-separated substrings (e.g. "mlp").
        _parts = [x for x in _scope.split(",") if x]
        nn.quantize(
            mtp,
            group_size=contract.mtp_quant_group_size,
            bits=contract.mtp_quant_bits,
            mode=contract.mtp_quant_mode,
            class_predicate=lambda path, module: hasattr(module, "to_quantized")
            and any(x in path for x in _parts),
        )
        return
    if policy == "all":
        nn.quantize(
            mtp,
            group_size=contract.mtp_quant_group_size,
            bits=contract.mtp_quant_bits,
            mode=contract.mtp_quant_mode,
        )
        return

    if policy != "cyankiwi":
        raise ValueError(f"Unsupported MTP quantization policy: {policy}")

    def predicate(path: str, module: Any):
        if path == "fc" or path.startswith("pre_fc_norm") or path == "norm":
            return False
        if path.startswith("layers.") and hasattr(module, "to_quantized"):
            return {
                "group_size": contract.mtp_quant_group_size,
                "bits": contract.mtp_quant_bits,
                "mode": contract.mtp_quant_mode,
            }
        return False

    nn.quantize(mtp, class_predicate=predicate)


def _stack_mtp_moe_experts(
    weights: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Stack numbered or fused MTP experts into mlx-lm's switch_mlp layout."""
    num_experts = num_experts_from_config(config)
    if num_experts <= 0:
        return weights
    hidden_size = text_config(config).get("hidden_size")
    if hidden_size:
        weights = split_fused_experts(weights, hidden_size=int(hidden_size))
    return stack_numbered_experts(weights, num_experts=num_experts, strict=False)


def _prequantized_module_prefixes(weights: dict[str, Any]) -> tuple[str, ...]:
    modules: set[str] = set()
    for key in weights:
        if not key.endswith(".scales"):
            continue
        prefix = key.removesuffix(".scales")
        if f"{prefix}.weight" in weights and f"{prefix}.biases" in weights:
            modules.add(prefix)
    return tuple(sorted(modules))


def _mtp_norms_are_delta_encoded(config: dict[str, Any]) -> bool:
    mtp_quant = config.get("mtplx_mtp_quantization")
    values = [
        config.get("mtplx_mtp_norm_encoding"),
        config.get("mtp_norm_encoding"),
    ]
    if isinstance(mtp_quant, dict):
        values.extend(
            [
                mtp_quant.get("norm_encoding"),
                mtp_quant.get("norm_weight_encoding"),
            ]
        )
    return any(
        str(value).strip().lower() in {"delta", "delta_plus_one", "mlx_delta"}
        for value in values
        if value is not None
    )


def _restore_delta_encoded_mtp_norms(
    weights: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    from .compressed_tensors import LAST_MTP_NORM_REPAIRS

    LAST_MTP_NORM_REPAIRS.clear()
    if not _mtp_norms_are_delta_encoded(config):
        return _heal_raw_delta_mtp_norms(weights)
    restored = dict(weights)
    for key, value in list(restored.items()):
        if value.ndim == 1 and any(key.endswith(suffix) for suffix in _RMSNORM_SUFFIXES):
            restored[key] = value + 1.0
    return restored


# Norm-suffix sets and the delta-detection thresholds live in
# compressed_tensors (#301): mtp_sidecar_norms_are_delta /
# mtp_norms_are_delta_encoded — one predicate for this heal path and
# the forge's set-level shift decision.


def _heal_raw_delta_mtp_norms(weights: dict[str, Any]) -> dict[str, Any]:
    """Detect and repair a sidecar whose norms were never +1-restored.

    Raw Qwen3.5 exports store MTP RMSNorm weights zero-centered (delta
    convention); mlx-lm's trunk sanitize restores +1.0 but the MTP tensors are
    loaded separately and must be restored here. The shipped 4B artifact
    (#176) carries raw norms with no declared encoding, which poisons every
    draft. Convention detection and the shift itself live in
    compressed_tensors.shift_delta_mtp_norms (shared with forge, #301).
    """

    from .compressed_tensors import shift_delta_mtp_norms

    return shift_delta_mtp_norms(weights)


def _infer_prequantized_group_size(weights: dict[str, Any], bits: int | None) -> int | None:
    if bits is None or bits <= 0:
        return None
    bits_int = int(bits)
    values_per_word = max(1, 32 // bits_int)
    inferred: set[int] = set()
    for key, weight in weights.items():
        if not key.endswith(".weight"):
            continue
        prefix = key.removesuffix(".weight")
        scales = weights.get(f"{prefix}.scales")
        biases = weights.get(f"{prefix}.biases")
        if scales is None or biases is None:
            continue
        weight_shape = getattr(weight, "shape", ())
        scales_shape = getattr(scales, "shape", ())
        biases_shape = getattr(biases, "shape", ())
        if not weight_shape or not scales_shape or scales_shape != biases_shape:
            continue
        packed_cols = int(weight_shape[-1])
        scale_groups = int(scales_shape[-1])
        if packed_cols <= 0 or scale_groups <= 0:
            continue
        if 32 % bits_int == 0:
            expanded_cols = packed_cols * values_per_word
        else:
            total_bits = packed_cols * 32
            if total_bits % bits_int != 0:
                continue
            expanded_cols = total_bits // bits_int
        if expanded_cols % scale_groups == 0:
            inferred.add(expanded_cols // scale_groups)
    if len(inferred) == 1:
        return inferred.pop()
    return None


def _contract_with_prequantized_tensor_geometry(
    contract: MTPContract,
    weights: dict[str, Any],
) -> MTPContract:
    if not contract.mtp_prequantized:
        return contract
    inferred_group_size = _infer_prequantized_group_size(weights, contract.mtp_quant_bits)
    if inferred_group_size is None or inferred_group_size == contract.mtp_quant_group_size:
        return contract
    return replace(contract, mtp_quant_group_size=inferred_group_size)


def _contract_with_prequantized_module_specs(
    contract: MTPContract,
    weights: dict[str, Any],
    config: dict[str, Any],
) -> MTPContract:
    if not contract.mtp_prequantized:
        return contract
    modules = _prequantized_module_prefixes(weights)
    if not modules:
        return contract
    module_specs = {
        module: spec
        for module in modules
        if (
            spec := _config_mtp_module_quant_spec(
                config,
                module,
                fallback_mode=contract.mtp_quant_mode,
            )
        )
    }
    updates: dict[str, Any] = {"mtp_prequantized_modules": modules}
    if module_specs:
        updates["mtp_prequantized_module_specs"] = module_specs
    return replace(contract, **updates)


def _config_mtp_module_quant_spec(
    config: dict[str, Any],
    module: str,
    *,
    fallback_mode: str,
) -> dict[str, Any] | None:
    candidates = (
        module,
        f"mtp.{module}",
        f"model.mtp.{module}",
        f"language_model.mtp.{module}",
        f"language_model.model.mtp.{module}",
    )
    for source in _quant_config_sources(config):
        containers = [source]
        nested = source.get("modules")
        if isinstance(nested, dict):
            containers.append(nested)
        for container in containers:
            for key in candidates:
                raw = container.get(key)
                spec = _module_quant_spec_from_mapping(raw, fallback_mode=fallback_mode)
                if spec:
                    return spec
    return None


def _quant_config_sources(config: dict[str, Any]) -> list[dict[str, Any]]:
    tcfg = text_config(config)
    sources: list[dict[str, Any]] = []
    for owner in (config, tcfg):
        for key in (
            "mtplx_mtp_quantization",
            "quantization",
            "quantization_config",
        ):
            value = owner.get(key)
            if isinstance(value, dict):
                sources.append(value)
    return sources


def _module_quant_spec_from_mapping(
    raw: Any,
    *,
    fallback_mode: str,
) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    bits = raw.get("bits")
    group_size = raw.get("group_size")
    if bits is None or group_size is None:
        return None
    try:
        bit_value = int(bits)
        group_value = int(group_size)
    except (TypeError, ValueError):
        return None
    if bit_value <= 0 or group_value <= 0:
        return None
    return {
        "bits": bit_value,
        "group_size": group_value,
        "mode": str(raw.get("mode") or fallback_mode),
    }


def _finalize_mtp_weights(
    raw_mtp: dict[str, Any],
    config: dict[str, Any],
    *,
    prequantized: bool = False,
) -> dict[str, Any]:
    try:
        import mlx.core as mx
    except Exception:
        mx = None

    tcfg = text_config(config)
    quant_config = tcfg.get("quantization") or tcfg.get("quantization_config") or {}
    if not quant_config:
        quant_config = config.get("quantization") or config.get("quantization_config") or {}
    bits = int(quant_config.get("bits", 4)) if quant_config else 4
    group_size = int(quant_config.get("group_size", 64)) if quant_config else 64

    if prequantized:
        return _stack_mtp_moe_experts(
            _restore_delta_encoded_mtp_norms(dict(raw_mtp), config),
            config,
        )

    weights: dict[str, Any] = {}
    processed: set[str] = set()
    for key in sorted(raw_mtp):
        if key in processed or key.endswith((".scales", ".biases")):
            continue
        scales_key = key.replace(".weight", ".scales")
        biases_key = key.replace(".weight", ".biases")
        if scales_key != key and scales_key in raw_mtp and biases_key in raw_mtp:
            if mx is None:
                raise RuntimeError("MLX is required to dequantize embedded MTP weights")
            weights[key] = mx.dequantize(
                raw_mtp[key],
                raw_mtp[scales_key],
                raw_mtp[biases_key],
                group_size=group_size,
                bits=bits,
            )
            processed.update({key, scales_key, biases_key})
        else:
            weights[key] = raw_mtp[key]
            processed.add(key)

    return _stack_mtp_moe_experts(_restore_delta_encoded_mtp_norms(weights, config), config)


def _strip_mtp_namespace(key: str) -> str:
    return normalize_mtp_key(key).removeprefix("mtp.")


def _mtp_contract_for_weight_keys(
    contract: MTPContract,
    keys: tuple[str, ...],
    config: dict[str, Any],
) -> MTPContract:
    normalized = {normalize_mtp_key(key) for key in keys}
    if contract.mtp_prequantized:
        return contract
    # The named key sets are depth-1 templates; expand per the declared layer
    # count so N-layer sidecars are recognized identically.
    n_layers = max(_num_mtp_layers(config), 1)
    if normalized == expand_mtp_layer_keys(EXPECTED_ALL_PREQUANTIZED_MTP_KEYS, n_layers):
        policy = "all"
    elif normalized == expand_mtp_layer_keys(
        EXPECTED_QWEN_MOE_SWITCH_MLP_PREQUANTIZED_MTP_KEYS, n_layers
    ):
        policy = "all"
    elif normalized == expand_mtp_layer_keys(
        EXPECTED_QWEN_MOE_PREQUANTIZED_MTP_KEYS, n_layers
    ):
        policy = "cyankiwi"
    elif normalized == expand_mtp_layer_keys(EXPECTED_PREQUANTIZED_MTP_KEYS, n_layers):
        policy = "cyankiwi"
    else:
        return contract

    tcfg = text_config(config)
    quant_config = tcfg.get("quantization") or tcfg.get("quantization_config") or {}
    if not quant_config:
        quant_config = config.get("quantization") or config.get("quantization_config") or {}
    updates: dict[str, Any] = {
        "mtp_prequantized": True,
        "mtp_quant_policy": contract.mtp_quant_policy or policy,
    }
    if contract.mtp_quant_bits is None:
        updates["mtp_quant_bits"] = int((quant_config or {}).get("bits", 4))
    if contract.mtp_quant_group_size == 64:
        updates["mtp_quant_group_size"] = int((quant_config or {}).get("group_size", 64))
    if contract.mtp_quant_mode == "affine":
        updates["mtp_quant_mode"] = str((quant_config or {}).get("mode", "affine"))
    return replace(contract, **updates)


def _load_mtp_weights(
    mtp_file: Path,
    config: dict[str, Any],
    *,
    prequantized: bool = False,
) -> dict[str, Any]:
    import mlx.core as mx

    raw = mx.load(str(mtp_file))
    raw_mtp = {_strip_mtp_namespace(k): v for k, v in raw.items() if is_mtp_key(k)}
    del raw
    return _finalize_mtp_weights(raw_mtp, config, prequantized=prequantized)


def _safetensors_runtime_framework() -> str:
    try:
        import mlx.core  # noqa: F401 - registers mlx.core for safetensors.

        return "mlx"
    except Exception:
        return "np"


def _mtp_file_keys(mtp_file: Path) -> tuple[str, ...]:
    try:
        from safetensors import safe_open

        with safe_open(str(mtp_file), framework=_safetensors_runtime_framework()) as handle:
            return tuple(sorted(str(key) for key in handle.keys()))
    except Exception:
        try:
            import mlx.core as mx

            raw = mx.load(str(mtp_file))
            keys = tuple(sorted(str(key) for key in raw.keys()))
            del raw
            return keys
        except Exception:
            return ()


def _embedded_mtp_weight_map(model_path: Path) -> dict[str, Path]:
    index_path = model_path / "model.safetensors.index.json"
    if index_path.exists():
        try:
            weight_map = json.loads(index_path.read_text(encoding="utf-8")).get("weight_map", {})
        except Exception:
            weight_map = {}
        return {
            str(key): model_path / str(rel)
            for key, rel in weight_map.items()
            if is_mtp_key(str(key))
        }

    from safetensors import safe_open

    result: dict[str, Path] = {}
    framework = _safetensors_runtime_framework()
    for shard in sorted(model_path.glob("model*.safetensors")):
        with safe_open(str(shard), framework=framework) as handle:
            for key in handle.offset_keys():
                if is_mtp_key(str(key)):
                    result[str(key)] = shard
    return result


def _load_embedded_mtp_weights(
    model_path: Path,
    config: dict[str, Any],
    *,
    prequantized: bool = False,
) -> dict[str, Any]:
    key_to_file = _embedded_mtp_weight_map(model_path)
    if not key_to_file:
        return {}

    raw_mtp: dict[str, Any] = {}
    files: dict[Path, list[str]] = {}
    for key, shard in key_to_file.items():
        files.setdefault(shard, []).append(key)

    for shard, keys in files.items():
        try:
            import mlx.core as mx

            shard_tensors = mx.load(str(shard))
            for key in sorted(keys):
                raw_mtp[_strip_mtp_namespace(key)] = shard_tensors[key]
            del shard_tensors
        except Exception:
            from safetensors import safe_open

            with safe_open(str(shard), framework=_safetensors_runtime_framework()) as handle:
                for key in sorted(keys):
                    raw_mtp[_strip_mtp_namespace(key)] = handle.get_tensor(key)

    return _finalize_mtp_weights(raw_mtp, config, prequantized=prequantized)


def inject_mtp_support(
    model: Any,
    model_path: Path | str,
    config: dict[str, Any],
    contract: MTPContract | None = None,
) -> bool:
    """Attach Qwen native MTP support to a loaded mlx-lm model instance."""
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models.base import create_attention_mask, create_ssm_mask, scaled_dot_product_attention
    from mlx_lm.models.cache import KVCache
    from mlx_lm.models.qwen3_5 import DecoderLayer, TextModelArgs

    contract = contract or MTPContract()
    contract.validate()
    n_layers = _num_mtp_layers(config)
    if n_layers <= 0:
        logger.info("[MTP inject] Model config has no MTP layers")
        return False

    model_path = Path(model_path)
    mtp_file = expected_mtp_file(model_path, config)
    mtp_weights: dict[str, Any] | None = None
    mtp_source = mtp_file
    if mtp_file.exists():
        contract = _mtp_contract_for_weight_keys(contract, _mtp_file_keys(mtp_file), config)
        mtp_weights = _load_mtp_weights(
            mtp_file,
            config,
            prequantized=contract.mtp_prequantized,
        )
    else:
        embedded_weight_map = _embedded_mtp_weight_map(model_path)
        contract = _mtp_contract_for_weight_keys(
            contract,
            tuple(embedded_weight_map),
            config,
        )
        mtp_weights = _load_embedded_mtp_weights(
            model_path,
            config,
            prequantized=contract.mtp_prequantized,
        )
        mtp_source = model_path / "model*.safetensors::embedded-mtp"
        if not mtp_weights:
            logger.warning("[MTP inject] MTP weights not found: %s", mtp_file)
            return False
    if contract.mtp_prequantized:
        contract = _contract_with_prequantized_tensor_geometry(contract, mtp_weights)
        contract = _contract_with_prequantized_module_specs(
            contract,
            mtp_weights,
            config,
        )

    tcfg = text_config(config)
    text_model = _text_model(model)
    args = getattr(text_model, "args", None)
    if not isinstance(args, TextModelArgs):
        args = TextModelArgs.from_dict(tcfg)

    fa_idx = args.full_attention_interval - 1

    class _MTPModule(nn.Module):
        def __init__(self, args: TextModelArgs, n_layers: int):
            super().__init__()
            self.pre_fc_norm_hidden = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
            self.pre_fc_norm_embedding = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
            self.fc = nn.Linear(args.hidden_size * 2, args.hidden_size, bias=False)
            self.layers = [DecoderLayer(args, layer_idx=fa_idx) for _ in range(n_layers)]
            self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    mtp = _MTPModule(args, n_layers)
    _env_mtp_bits = os.environ.get("MTPLX_MTP_QUANT_BITS", "").strip()
    _a3b_experts_q4g32 = False
    if not _env_mtp_bits:
        # M1 family default (2026-10-13): an fp16 sidecar's layer is 0.85 GB a draft step reads;
        # q8 halves it (draft -3 ms/round, decode +0.9% at 2K, +1.9% at 32K on an M1 Max).
        from .cache_state import m1_long_context_defaults

        if m1_long_context_defaults():
            _env_mtp_bits = "8"
            if (
                not contract.mtp_prequantized
                and contract.mtp_quant_bits is None
                and _qwen36_a3b_text_geometry(tcfg)
            ):
                # A Qwen3.6-35B-A3B fine-tune forged with an fp16 MTP sidecar (Ornith-1.5):
                # quantize only the routed experts to 4-bit g32, the layout the official
                # Optimized-Speed pack ships, so the M1 A3B lane's MTP whole-MoE binding
                # (mtp_dense_q4g32_dense) accepts it; router, shared expert and attention
                # stay fp16 as in that pack.
                _a3b_experts_q4g32 = True
    if _env_mtp_bits in {"0", "16", "off"}:
        _env_mtp_bits = ""
    if _env_mtp_bits and not contract.mtp_prequantized and contract.mtp_quant_bits is None:
        # Experiment switch: quantize an fp16/bf16 MTP sidecar at load. The draft
        # only proposes; speculative acceptance keeps the target distribution.
        contract = replace(contract, mtp_quant_bits=int(_env_mtp_bits))
    if contract.mtp_prequantized:
        _quantize_mtp_module(mtp, contract)
    mtp.load_weights(list(mtp_weights.items()), strict=False)
    if _a3b_experts_q4g32:
        nn.quantize(
            mtp,
            group_size=32,
            bits=4,
            mode="affine",
            class_predicate=lambda path, module: hasattr(module, "to_quantized")
            and ".switch_mlp." in f".{path}.",
        )
    elif not contract.mtp_prequantized:
        _quantize_mtp_module(mtp, contract)
    mx.eval(mtp.parameters())

    text_model.mtp = mtp
    text_model._mtplx_hidden_variant = contract.hidden_variant
    text_model._mtplx_concat_order = contract.concat_order
    text_model._mtplx_mtp_quant_policy = contract.mtp_quant_policy
    # Which draft-head gains the load restored (compressed_tensors.
    # unshifted_mtp_norms): reported in /health and every request's log row.
    from .compressed_tensors import LAST_MTP_NORM_REPAIRS

    text_model._mtplx_draft_head_norm_repairs = [dict(r) for r in LAST_MTP_NORM_REPAIRS]

    original_text_class = text_model.__class__

    class _MTPLXTextModel(original_text_class):
        def __call__(
            self,
            inputs,
            cache=None,
            return_hidden: bool = False,
            input_embeddings=None,
            hidden_variant: str | None = None,
            emit_logits: bool = True,
            logits_keep: int | None = None,
            **kwargs,
        ):
            inner = self.model
            hidden_states = input_embeddings if input_embeddings is not None else inner.embed_tokens(inputs)
            if cache is None:
                cache = [None] * len(inner.layers)

            fa_mask = create_attention_mask(hidden_states, cache[inner.fa_idx])
            ssm_mask = create_ssm_mask(hidden_states, cache[inner.ssm_idx])
            eval_every = _prefill_layer_eval_every()
            from .ane_prefill import pipelined_layers as _ane_pipelined_layers

            piped = _ane_pipelined_layers(inner, hidden_states, cache, fa_mask, ssm_mask)
            if piped is None:
                from .ane_prefill_moe import pipelined_layers as _ane_moe_pipelined_layers

                piped = _ane_moe_pipelined_layers(inner, hidden_states, cache, fa_mask, ssm_mask)
            if piped is not None:
                hidden_states = piped
            for layer_index, (layer, layer_cache) in enumerate(zip(inner.layers, cache)):
                if piped is not None:
                    break
                mask = ssm_mask if layer.is_linear else fa_mask
                hidden_states = layer(hidden_states, mask=mask, cache=layer_cache)
                if eval_every and (layer_index + 1) % eval_every == 0:
                    # Close the command buffer here so this span's transients
                    # are freed before the next layers allocate theirs.
                    mx.eval(hidden_states)

            pre_norm = hidden_states
            variant = hidden_variant or getattr(self, "_mtplx_hidden_variant", "post_norm")
            needs_post_norm = emit_logits or (return_hidden and variant != "pre_norm")
            post_norm = inner.norm(hidden_states) if needs_post_norm else None
            logits = None
            if emit_logits:
                logits_source = post_norm
                if logits_keep is not None:
                    keep = max(1, int(logits_keep))
                    logits_source = logits_source[:, -keep:, :]
                logits = (
                    inner.embed_tokens.as_linear(logits_source)
                    if self.args.tie_word_embeddings
                    else self.lm_head(logits_source)
                )
            if not return_hidden:
                return logits
            hidden = pre_norm if variant == "pre_norm" else post_norm
            return logits, hidden

        def _mixed_hidden(self, variant: str, *, previous, fc_hidden, pre_norm, post_norm, input_embeds):
            aliases = {
                "fc": fc_hidden,
                "pre_norm": pre_norm,
                "post_norm": post_norm,
                "embedding": input_embeds,
                "prev": previous,
            }
            if variant in aliases:
                return aliases[variant]

            # Experimental hidden repair syntax:
            #   mix:<left>:<right>:<alpha>
            # returns alpha * left + (1 - alpha) * right.
            # Alpha accepts decimal points or "p" as the decimal separator,
            # e.g. mix:pre_norm:prev:0p75.
            if variant.startswith("mix:"):
                parts = variant.split(":")
                if len(parts) != 4:
                    raise ValueError("mix variant must be mix:<left>:<right>:<alpha>")
                left_name, right_name, alpha_raw = parts[1], parts[2], parts[3]
                if left_name not in aliases or right_name not in aliases:
                    raise ValueError(
                        "mix variant sources must be one of "
                        "'fc', 'pre_norm', 'post_norm', 'embedding', or 'prev'"
                    )
                alpha = float(alpha_raw.replace("p", "."))
                if not 0.0 <= alpha <= 1.0:
                    raise ValueError("mix variant alpha must be in [0, 1]")
                return aliases[left_name] * alpha + aliases[right_name] * (1.0 - alpha)

            raise ValueError(
                "mtp_hidden_variant must be 'fc', 'pre_norm', 'post_norm', "
                "'embedding', 'prev', or mix:<left>:<right>:<alpha>"
            )

        def _mtp_full_attention_layer(self, layer, x, *, mask=None, cache=None, position_offset: int | None = None):
            if position_offset is None:
                return layer(x, mask=mask, cache=cache)
            if layer.is_linear:
                raise ValueError("explicit MTP position offsets require a full-attention MTP layer")

            attn = layer.self_attn
            normed = layer.input_layernorm(x)
            B, L, _ = normed.shape

            q_proj_output = attn.q_proj(normed)
            queries, gate = mx.split(
                q_proj_output.reshape(B, L, attn.num_attention_heads, -1),
                2,
                axis=-1,
            )
            gate = gate.reshape(B, L, -1)

            keys, values = attn.k_proj(normed), attn.v_proj(normed)
            queries = attn.q_norm(queries).transpose(0, 2, 1, 3)
            keys = attn.k_norm(keys.reshape(B, L, attn.num_key_value_heads, -1)).transpose(
                0,
                2,
                1,
                3,
            )
            values = values.reshape(B, L, attn.num_key_value_heads, -1).transpose(
                0,
                2,
                1,
                3,
            )

            queries = attn.rope(queries, offset=int(position_offset))
            keys = attn.rope(keys, offset=int(position_offset))
            paged_mtp_enabled = (
                os.environ.get("MTPLX_VLLM_METAL_PAGED_MTP_ATTN", "")
                .strip()
                .lower()
                in {"1", "true", "yes", "on"}
            )
            use_paged_mtp = bool(
                paged_mtp_enabled
                and cache is not None
                and int(L) == 1
                and hasattr(cache, "update_without_fetch")
                and hasattr(cache, "paged_attention")
            )
            if use_paged_mtp:
                # The paged primitive is causal-safe only for single-token MTP
                # draft/update calls. Multi-token committed-history appends keep
                # the stock SDPA path so each query cannot see future keys from
                # the same append chunk.
                cache.update_without_fetch(keys, values)
                output = cache.paged_attention(queries, scale=attn.scale)
                if output is None:
                    keys, values = cache.state
                    output = scaled_dot_product_attention(
                        queries,
                        keys,
                        values,
                        cache=cache,
                        scale=attn.scale,
                        mask=mask,
                    )
            else:
                if cache is not None:
                    keys, values = cache.update_and_fetch(keys, values)
                output = scaled_dot_product_attention(
                    queries,
                    keys,
                    values,
                    cache=cache,
                    scale=attn.scale,
                    mask=mask,
                )
            output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
            h = x + attn.o_proj(output * mx.sigmoid(gate))
            return h + layer.mlp(layer.post_attention_layernorm(h))

        def _mtp_core(
            self,
            hidden_states,
            next_token_ids,
            mtp_cache=None,
            concat_order=None,
            mtp_hidden_variant: str = "post_norm",
            position_offset: int | None = None,
            emit_logits: bool = True,
            input_embeddings=None,
        ):
            # Vision prompts: the caller passes the same spliced embedding
            # rows the trunk prefill consumed, so the MTP history sees the
            # image content instead of raw image-pad embeddings (#103).
            input_embeds = (
                input_embeddings
                if input_embeddings is not None
                else self.model.embed_tokens(next_token_ids)
            )
            e = self.mtp.pre_fc_norm_embedding(input_embeds)
            h = self.mtp.pre_fc_norm_hidden(hidden_states)
            order = concat_order or getattr(self, "_mtplx_concat_order", "embedding_hidden")
            parts = [e, h] if order == "embedding_hidden" else [h, e]
            x = self.mtp.fc(mx.concatenate(parts, axis=-1))
            fc_hidden = x
            num_draft_layers = len(self.mtp.layers)
            if mtp_cache:
                if len(mtp_cache) < num_draft_layers:
                    raise ValueError(
                        "MTP cache carries "
                        f"{len(mtp_cache)} entries for {num_draft_layers} draft "
                        "layers; rebuild it with make_mtp_cache()"
                    )
                layer_caches = mtp_cache
            else:
                layer_caches = [None] * num_draft_layers
            # All draft-layer caches advance in lockstep, so the mask derived
            # from the first layer's cache offset is valid for every layer.
            mask = create_attention_mask(x, layer_caches[0])
            for mtp_layer, layer_cache in zip(self.mtp.layers, layer_caches):
                x = self._mtp_full_attention_layer(
                    mtp_layer,
                    x,
                    mask=mask,
                    cache=layer_cache,
                    position_offset=position_offset,
                )
            pre_norm = x
            post_norm = self.mtp.norm(x)
            hidden = self._mixed_hidden(
                mtp_hidden_variant,
                previous=hidden_states,
                fc_hidden=fc_hidden,
                pre_norm=pre_norm,
                post_norm=post_norm,
                input_embeds=input_embeds,
            )
            if not emit_logits:
                return None, hidden
            draft_lm_head = getattr(self, "_mtplx_draft_lm_head", None)
            logits = (
                draft_lm_head(post_norm)
                if draft_lm_head is not None
                else (
                    self.model.embed_tokens.as_linear(post_norm)
                    if self.args.tie_word_embeddings
                    else self.lm_head(post_norm)
                )
            )
            return logits, hidden

        def mtp_forward(
            self,
            hidden_states,
            next_token_ids,
            cache=None,
            mtp_cache=None,
            concat_order=None,
            return_hidden: bool = False,
            mtp_hidden_variant: str = "post_norm",
            position_offset: int | None = None,
        ):
            logits, hidden = self._mtp_core(
                hidden_states,
                next_token_ids,
                mtp_cache=mtp_cache,
                concat_order=concat_order,
                mtp_hidden_variant=mtp_hidden_variant,
                position_offset=position_offset,
                emit_logits=True,
            )
            if not return_hidden:
                return logits
            return logits, hidden

        def mtp_update_cache(
            self,
            hidden_states,
            next_token_ids,
            mtp_cache=None,
            concat_order=None,
            mtp_hidden_variant: str | None = None,
            position_offset: int | None = None,
            input_embeddings=None,
        ):
            _logits, hidden = self._mtp_core(
                hidden_states,
                next_token_ids,
                mtp_cache=mtp_cache,
                concat_order=concat_order,
                mtp_hidden_variant=mtp_hidden_variant
                or getattr(self, "_mtplx_hidden_variant", "post_norm"),
                position_offset=position_offset,
                emit_logits=False,
                input_embeddings=input_embeddings,
            )
            return hidden

        def make_mtp_cache(self):
            return [KVCache() for _ in self.mtp.layers]

    text_model.__class__ = _MTPLXTextModel

    if hasattr(model, "language_model") and model.language_model is text_model:
        model.mtp = mtp
        original_outer_class = model.__class__

        class _MTPLXOuterModel(original_outer_class):
            def __call__(
                self,
                inputs,
                cache=None,
                return_hidden: bool = False,
                input_embeddings=None,
                hidden_variant: str | None = None,
                emit_logits: bool = True,
                logits_keep: int | None = None,
                **kwargs,
            ):
                return self.language_model(
                    inputs,
                    cache=cache,
                    return_hidden=return_hidden,
                    input_embeddings=input_embeddings,
                    hidden_variant=hidden_variant,
                    emit_logits=emit_logits,
                    logits_keep=logits_keep,
                    **kwargs,
                )

            def mtp_forward(self, *args, **kwargs):
                return self.language_model.mtp_forward(*args, **kwargs)

            def mtp_update_cache(self, *args, **kwargs):
                return self.language_model.mtp_update_cache(*args, **kwargs)

            def make_mtp_cache(self):
                return self.language_model.make_mtp_cache()

        model.__class__ = _MTPLXOuterModel

    logger.info("[MTP inject] Loaded %d tensors from %s", len(mtp_weights), mtp_source)
    return True


def validate_mtp_support(model: Any) -> bool:
    text_model = _text_model(model)
    if getattr(text_model, "mtp", None) is None:
        return False
    if not getattr(text_model.mtp, "layers", None):
        return False
    try:
        call_sig = inspect.signature(type(text_model).__call__)
    except Exception:
        return False
    return (
        "return_hidden" in call_sig.parameters
        and callable(getattr(text_model, "mtp_forward", None))
        and callable(getattr(text_model, "make_mtp_cache", None))
    )
