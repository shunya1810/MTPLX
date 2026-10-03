from __future__ import annotations

import pytest

from mtplx.glm_mtp_patch import _rewrite_glm_mtp_weights
from mtplx.mtp_patch import (
    MTPContract,
    _contract_with_prequantized_module_specs,
    _contract_with_prequantized_tensor_geometry,
    _finalize_mtp_weights,
    _mtp_contract_for_weight_keys,
    _prequantized_module_prefixes,
    _stack_mtp_moe_experts,
)
from mtplx.constants import EXPECTED_QWEN_MOE_SWITCH_MLP_PREQUANTIZED_MTP_KEYS


class _GLMRewriteArgs:
    n_routed_experts = 0


def test_glm_mtp_rewrite_rejects_config_only_lm_head_payload() -> None:
    mapped = _rewrite_glm_mtp_weights(
        {"lm_head.weight": object()},
        args=_GLMRewriteArgs(),
        start_layer=47,
        num_mtp_layers=1,
        rewrite_mla_kv_b=False,
    )

    assert mapped == {}


def test_glm_mtp_rewrite_shares_lm_head_only_with_real_mtp_layer_payload() -> None:
    mapped = _rewrite_glm_mtp_weights(
        {
            "lm_head.weight": "head-weight",
            "lm_head.scales": "head-scales",
            "lm_head.biases": "head-biases",
            "model.layers.47.enorm.weight": "enorm",
            "model.layers.47.hnorm.weight": "hnorm",
            "model.layers.47.eh_proj.weight": "eh-proj",
            "model.layers.47.self_attn.q_proj.weight": "mtp-block",
        },
        args=_GLMRewriteArgs(),
        start_layer=47,
        num_mtp_layers=1,
        rewrite_mla_kv_b=False,
    )

    assert mapped["layers.0.shared_head_head.weight"] == "head-weight"
    assert mapped["layers.0.shared_head_head.scales"] == "head-scales"
    assert mapped["layers.0.shared_head_head.biases"] == "head-biases"
    assert mapped["layers.0.mtp_block.self_attn.q_proj.weight"] == "mtp-block"


def test_mtp_contract_reads_config_quant_defaults() -> None:
    contract = MTPContract().with_config_defaults(
        {
            "mtplx_mtp_quantization": {
                "policy": "cyankiwi",
                "bits": 4,
                "group_size": 32,
                "mode": "affine",
                "prequantized": True,
            }
        }
    )

    assert contract.mtp_quant_policy == "cyankiwi"
    assert contract.mtp_quant_bits == 4
    assert contract.mtp_quant_group_size == 32
    assert contract.mtp_quant_mode == "affine"
    assert contract.mtp_prequantized is True


def test_mtp_contract_accepts_neutral_prequantized_int4_policy() -> None:
    contract = MTPContract().with_config_defaults(
        {
            "mtplx_mtp_quantization": {
                "policy": "prequantized-int4",
                "bits": 4,
                "group_size": 32,
                "mode": "affine",
                "prequantized": True,
            }
        }
    )

    contract.validate()

    assert contract.mtp_quant_policy == "prequantized-int4"
    assert contract.mtp_quant_bits == 4
    assert contract.mtp_quant_group_size == 32
    assert contract.mtp_prequantized is True


def test_mtp_contract_cli_bits_override_config_bits() -> None:
    contract = MTPContract(mtp_quant_bits=8).with_config_defaults(
        {
            "mtplx_mtp_quantization": {
                "policy": "cyankiwi",
                "bits": 4,
                "group_size": 32,
                "mode": "affine",
            }
        }
    )

    assert contract.mtp_quant_bits == 8
    assert contract.mtp_quant_group_size == 32
    assert contract.mtp_quant_policy == "cyankiwi"


def test_prequantized_mtp_config_overrides_stale_runtime_contract() -> None:
    contract = MTPContract(mtp_quant_group_size=64).with_config_defaults(
        {
            "mtplx_mtp_contract": {
                "mtp_quant_group_size": 64,
                "mtp_quant_mode": "affine",
            },
            "mtplx_mtp_quantization": {
                "policy": "cyankiwi",
                "bits": 4,
                "group_size": 32,
                "mode": "affine",
                "prequantized": True,
            },
        }
    )

    assert contract.mtp_quant_bits == 4
    assert contract.mtp_quant_group_size == 32
    assert contract.mtp_quant_policy == "cyankiwi"
    assert contract.mtp_prequantized is True


def test_prequantized_mtp_tensor_geometry_corrects_group_size() -> None:
    class FakeArray:
        def __init__(self, shape: tuple[int, ...]):
            self.shape = shape

    contract = MTPContract(
        mtp_quant_bits=4,
        mtp_quant_group_size=64,
        mtp_prequantized=True,
    )
    weights = {
        "layers.0.mlp.switch_mlp.gate_proj.weight": FakeArray((256, 512, 256)),
        "layers.0.mlp.switch_mlp.gate_proj.scales": FakeArray((256, 512, 64)),
        "layers.0.mlp.switch_mlp.gate_proj.biases": FakeArray((256, 512, 64)),
    }

    corrected = _contract_with_prequantized_tensor_geometry(contract, weights)

    assert corrected.mtp_quant_group_size == 32


def test_prequantized_mtp_module_specs_read_mixed_quant_config() -> None:
    class FakeArray:
        def __init__(self, shape: tuple[int, ...]):
            self.shape = shape

    contract = MTPContract(
        mtp_quant_bits=4,
        mtp_quant_group_size=64,
        mtp_prequantized=True,
    )
    weights = {
        "layers.0.self_attn.q_proj.weight": FakeArray((8192, 320)),
        "layers.0.self_attn.q_proj.scales": FakeArray((8192, 16)),
        "layers.0.self_attn.q_proj.biases": FakeArray((8192, 16)),
        "layers.0.mlp.switch_mlp.gate_proj.weight": FakeArray((256, 512, 256)),
        "layers.0.mlp.switch_mlp.gate_proj.scales": FakeArray((256, 512, 32)),
        "layers.0.mlp.switch_mlp.gate_proj.biases": FakeArray((256, 512, 32)),
    }
    config = {
        "quantization": {
            "bits": 4,
            "group_size": 64,
            "mode": "affine",
            "language_model.mtp.layers.0.self_attn.q_proj": {
                "bits": 5,
                "group_size": 128,
                "mode": "affine",
            },
            "language_model.mtp.layers.0.mlp.switch_mlp.gate_proj": {
                "bits": 4,
                "group_size": 64,
                "mode": "affine",
            },
        }
    }

    updated = _contract_with_prequantized_module_specs(contract, weights, config)

    assert updated.mtp_prequantized_modules == (
        "layers.0.mlp.switch_mlp.gate_proj",
        "layers.0.self_attn.q_proj",
    )
    assert updated.mtp_prequantized_module_specs["layers.0.self_attn.q_proj"] == {
        "bits": 5,
        "group_size": 128,
        "mode": "affine",
    }
    assert updated.to_dict()["mtp_prequantized_module_specs"][
        "layers.0.self_attn.q_proj"
    ]["bits"] == 5


def test_low_mtp_norm_weights_are_not_shifted_without_delta_contract() -> None:
    mx = pytest.importorskip("mlx.core")
    weights = {
        "pre_fc_norm_hidden.weight": mx.array([0.25, 0.5], dtype=mx.float32),
        "pre_fc_norm_embedding.weight": mx.array([0.125, 0.375], dtype=mx.float32),
    }

    finalized = _finalize_mtp_weights(weights, {}, prequantized=True)

    assert float(finalized["pre_fc_norm_hidden.weight"][0].item()) == pytest.approx(0.25)
    assert float(finalized["pre_fc_norm_embedding.weight"][0].item()) == pytest.approx(0.125)


def test_raw_delta_qwen_sidecar_norms_are_healed() -> None:
    # The shipped 4B (#176) stores raw zero-centered norms with no declared
    # encoding: q/k means ~0.75 plus sub-0.5 low-set norms is the two-signal
    # raw-delta fingerprint, and the loader must restore the +1 convention.
    mx = pytest.importorskip("mlx.core")
    weights = {
        "layers.0.input_layernorm.weight": mx.full((8,), 0.30, dtype=mx.float32),
        "layers.0.post_attention_layernorm.weight": mx.full((8,), 0.39, dtype=mx.float32),
        "layers.0.self_attn.q_norm.weight": mx.full((8,), 0.75, dtype=mx.float32),
        "layers.0.self_attn.k_norm.weight": mx.full((8,), 0.74, dtype=mx.float32),
        "norm.weight": mx.full((8,), 2.58, dtype=mx.float32),
        "pre_fc_norm_embedding.weight": mx.full((8,), -0.41, dtype=mx.float32),
        "pre_fc_norm_hidden.weight": mx.full((8,), -0.25, dtype=mx.float32),
        "fc.weight": mx.full((4, 8), 0.007, dtype=mx.float32),
    }

    finalized = _finalize_mtp_weights(weights, {}, prequantized=True)

    assert float(finalized["layers.0.self_attn.q_norm.weight"][0].item()) == pytest.approx(1.75)
    assert float(finalized["layers.0.input_layernorm.weight"][0].item()) == pytest.approx(1.30)
    assert float(finalized["norm.weight"][0].item()) == pytest.approx(3.58)
    assert float(finalized["pre_fc_norm_embedding.weight"][0].item()) == pytest.approx(0.59)
    assert float(finalized["pre_fc_norm_hidden.weight"][0].item()) == pytest.approx(0.75)
    assert float(finalized["fc.weight"][0, 0].item()) == pytest.approx(0.007)


def test_healthy_final_convention_sidecar_is_untouched() -> None:
    mx = pytest.importorskip("mlx.core")
    weights = {
        "layers.0.input_layernorm.weight": mx.full((8,), 1.10, dtype=mx.float32),
        "layers.0.post_attention_layernorm.weight": mx.full((8,), 1.24, dtype=mx.float32),
        "layers.0.self_attn.q_norm.weight": mx.full((8,), 1.75, dtype=mx.float32),
        "layers.0.self_attn.k_norm.weight": mx.full((8,), 1.74, dtype=mx.float32),
        "norm.weight": mx.full((8,), 2.43, dtype=mx.float32),
        "pre_fc_norm_embedding.weight": mx.full((8,), 0.52, dtype=mx.float32),
        "pre_fc_norm_hidden.weight": mx.full((8,), 0.77, dtype=mx.float32),
    }

    finalized = _finalize_mtp_weights(weights, {}, prequantized=True)

    for key, value in weights.items():
        assert float(finalized[key][0].item()) == pytest.approx(float(value[0].item())), key


def test_single_signal_low_norms_are_not_healed() -> None:
    # q/k near raw levels but a healthy low set: one signal is not enough to
    # rewrite weights (guards against overeager shifting of unusual models).
    mx = pytest.importorskip("mlx.core")
    weights = {
        "layers.0.self_attn.q_norm.weight": mx.full((8,), 0.75, dtype=mx.float32),
        "layers.0.self_attn.k_norm.weight": mx.full((8,), 0.74, dtype=mx.float32),
        "layers.0.input_layernorm.weight": mx.full((8,), 0.90, dtype=mx.float32),
        "pre_fc_norm_hidden.weight": mx.full((8,), 0.77, dtype=mx.float32),
    }

    finalized = _finalize_mtp_weights(weights, {}, prequantized=True)

    assert float(finalized["layers.0.self_attn.q_norm.weight"][0].item()) == pytest.approx(0.75)
    assert float(finalized["layers.0.input_layernorm.weight"][0].item()) == pytest.approx(0.90)


def test_delta_encoded_mtp_norm_weights_are_restored_by_contract() -> None:
    mx = pytest.importorskip("mlx.core")
    weights = {
        "pre_fc_norm_hidden.weight": mx.array([0.25, 0.5], dtype=mx.float32),
    }

    finalized = _finalize_mtp_weights(
        weights,
        {"mtplx_mtp_norm_encoding": "delta_plus_one"},
        prequantized=True,
    )

    assert float(finalized["pre_fc_norm_hidden.weight"][0].item()) == pytest.approx(1.25)


def test_mtp_contract_rejects_unknown_quant_policy() -> None:
    with pytest.raises(ValueError, match="mtp_quant_policy"):
        MTPContract(mtp_quant_policy="mystery").validate()


def test_mtp_contract_accepts_forge_calibrated_hidden_variants() -> None:
    contract = MTPContract().with_metadata(
        {
            "base_hidden_variant": "post_norm",
            "hidden_variant": "fc",
            "concat_order": "hidden_embedding",
        },
        preserve_explicit=False,
    )

    assert contract.hidden_variant == "fc"
    assert contract.concat_order == "hidden_embedding"
    assert contract.to_dict()["base_hidden_variant"] == "post_norm"


def test_mtp_contract_rejects_unknown_hidden_variant() -> None:
    with pytest.raises(ValueError, match="hidden_variant"):
        MTPContract(hidden_variant="mystery").validate()


def test_stack_mtp_moe_experts_stacks_quantized_numbered_experts() -> None:
    mx = pytest.importorskip("mlx.core")
    num_experts = 4
    weights = {"layers.0.input_layernorm.weight": mx.ones((8,))}
    for expert_index in range(num_experts):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            prefix = f"layers.0.mlp.experts.{expert_index}.{proj}"
            weights[f"{prefix}.weight"] = mx.full((2, 2), float(expert_index))
            weights[f"{prefix}.scales"] = mx.full((2, 1), float(expert_index + 1))
            weights[f"{prefix}.biases"] = mx.full((2, 1), float(-expert_index))

    stacked = _stack_mtp_moe_experts(
        weights,
        {"text_config": {"n_routed_experts": num_experts, "mtp_num_hidden_layers": 1}},
    )

    assert not any(".experts." in key for key in stacked)
    gate = stacked["layers.0.mlp.switch_mlp.gate_proj.weight"]
    assert tuple(gate.shape) == (num_experts, 2, 2)
    assert float(gate[3, 0, 0].item()) == 3.0
    assert tuple(stacked["layers.0.mlp.switch_mlp.down_proj.scales"].shape) == (
        num_experts,
        2,
        1,
    )


def _fused_qwen_moe_mtp_weights(mx, *, num_experts: int, hidden: int, inter: int, bmm: bool):
    """Official Qwen3.5/3.6 MoE MTP payload: fused experts plus the plain leaves."""
    from mtplx.constants import EXPECTED_QWEN_MOE_MTP_KEYS

    gate = mx.arange(num_experts * inter * hidden, dtype=mx.float32).reshape(num_experts, inter, hidden)
    up = -gate - 1.0
    down = mx.arange(num_experts * hidden * inter, dtype=mx.float32).reshape(num_experts, hidden, inter) + 0.5
    weights = {}
    for key in EXPECTED_QWEN_MOE_MTP_KEYS:
        stripped = key.removeprefix("mtp.")
        if stripped.endswith("mlp.experts.gate_up_proj"):
            fused = mx.concatenate([gate, up], axis=1)  # [E, 2*inter, hidden]
            weights[stripped] = fused.swapaxes(1, 2) if bmm else fused
        elif stripped.endswith("mlp.experts.down_proj"):
            weights[stripped] = down.swapaxes(1, 2) if bmm else down  # [E, hidden, inter]
        else:
            weights[stripped] = mx.ones((hidden,))
    return weights, gate, up, down


@pytest.mark.parametrize("bmm", [False, True])
def test_finalize_maps_fused_qwen_moe_mtp_experts_onto_switch_mlp(bmm: bool) -> None:
    """Fused experts must reach switch_mlp, or load_weights(strict=False) drops them."""
    mx = pytest.importorskip("mlx.core")
    from mtplx.constants import EXPECTED_QWEN_MOE_SWITCH_MLP_MTP_KEYS

    num_experts, hidden, inter = 4, 8, 3
    weights, gate, up, down = _fused_qwen_moe_mtp_weights(
        mx, num_experts=num_experts, hidden=hidden, inter=inter, bmm=bmm
    )
    config = {"text_config": {"num_experts": num_experts, "hidden_size": hidden}}

    finalized = _finalize_mtp_weights(weights, config)

    assert set(finalized) == {k.removeprefix("mtp.") for k in EXPECTED_QWEN_MOE_SWITCH_MLP_MTP_KEYS}
    switch = "layers.0.mlp.switch_mlp"
    assert mx.array_equal(finalized[f"{switch}.gate_proj.weight"], gate).item()
    assert mx.array_equal(finalized[f"{switch}.up_proj.weight"], up).item()
    assert mx.array_equal(finalized[f"{switch}.down_proj.weight"], down).item()


def test_fused_and_numbered_mtp_experts_finalize_identically() -> None:
    mx = pytest.importorskip("mlx.core")
    num_experts, hidden, inter = 4, 8, 3
    fused, gate, up, down = _fused_qwen_moe_mtp_weights(
        mx, num_experts=num_experts, hidden=hidden, inter=inter, bmm=False
    )
    numbered = {k: v for k, v in fused.items() if ".mlp.experts." not in k}
    for e in range(num_experts):
        numbered[f"layers.0.mlp.experts.{e}.gate_proj.weight"] = gate[e]
        numbered[f"layers.0.mlp.experts.{e}.up_proj.weight"] = up[e]
        numbered[f"layers.0.mlp.experts.{e}.down_proj.weight"] = down[e]
    config = {"text_config": {"num_experts": num_experts, "hidden_size": hidden}}

    from_fused = _finalize_mtp_weights(fused, config)
    from_numbered = _finalize_mtp_weights(numbered, config)

    assert set(from_fused) == set(from_numbered)
    for key in from_fused:
        assert mx.array_equal(from_fused[key], from_numbered[key]).item(), key


def test_prequantized_module_prefixes_uses_only_complete_quantized_triples() -> None:
    mx = pytest.importorskip("mlx.core")
    weights = {
        "layers.0.mlp.switch_mlp.gate_proj.weight": mx.ones((2, 2, 2)),
        "layers.0.mlp.switch_mlp.gate_proj.scales": mx.ones((2, 2, 1)),
        "layers.0.mlp.switch_mlp.gate_proj.biases": mx.ones((2, 2, 1)),
        "layers.0.self_attn.q_proj.weight": mx.ones((2, 2)),
    }

    assert _prequantized_module_prefixes(weights) == (
        "layers.0.mlp.switch_mlp.gate_proj",
    )


def test_mtp_contract_detects_prequantized_switch_moe_sidecar() -> None:
    contract = _mtp_contract_for_weight_keys(
        MTPContract(),
        EXPECTED_QWEN_MOE_SWITCH_MLP_PREQUANTIZED_MTP_KEYS,
        {
            "text_config": {
                "model_type": "qwen3_5_moe_text",
                "num_experts": 256,
                "mtp_num_hidden_layers": 1,
                "quantization": {"bits": 4, "group_size": 64, "mode": "affine"},
            }
        },
    )

    assert contract.mtp_prequantized is True
    assert contract.mtp_quant_policy == "all"
    assert contract.mtp_quant_bits == 4


class _GeomArray:
    def __init__(self, shape: tuple[int, ...]):
        self.shape = shape


def _geometry_weights(*, packed_cols: int, scale_groups: int) -> dict:
    return {
        "layers.0.mtp.fc.weight": _GeomArray((4096, packed_cols)),
        "layers.0.mtp.fc.scales": _GeomArray((4096, scale_groups)),
        "layers.0.mtp.fc.biases": _GeomArray((4096, scale_groups)),
    }


def _geometry_contract(bits: int, group_size: int) -> MTPContract:
    return MTPContract(
        mtp_quant_bits=bits,
        mtp_quant_group_size=group_size,
        mtp_prequantized=True,
    )


def test_prequantized_5bit_geometry_preserves_correct_declared_group_size() -> None:
    # Issue #182 / PR #183 (Jonathangadeaharder): 640 uint32 words hold
    # 640 * 32 / 5 = 4096 five-bit values. Floor division (32 // 5 = 6)
    # undercounted to 3840 and overwrote the correct declared group size 64
    # with an invalid 60, making the artifact unloadable.
    contract = _geometry_contract(5, 64)
    weights = _geometry_weights(packed_cols=640, scale_groups=64)
    corrected = _contract_with_prequantized_tensor_geometry(contract, weights)
    assert corrected.mtp_quant_group_size == 64


def test_prequantized_5bit_geometry_heals_wrong_declared_group_size() -> None:
    contract = _geometry_contract(5, 32)
    weights = _geometry_weights(packed_cols=640, scale_groups=64)
    corrected = _contract_with_prequantized_tensor_geometry(contract, weights)
    assert corrected.mtp_quant_group_size == 64


def test_prequantized_3bit_geometry_infers_exact_group_size() -> None:
    # 384 words * 32 bits / 3 = 4096 values; groups of 64 -> 64 scale groups.
    contract = _geometry_contract(3, 32)
    weights = _geometry_weights(packed_cols=384, scale_groups=64)
    corrected = _contract_with_prequantized_tensor_geometry(contract, weights)
    assert corrected.mtp_quant_group_size == 64


def test_prequantized_6bit_geometry_infers_exact_group_size() -> None:
    # 768 words * 32 bits / 6 = 4096 values; groups of 64 -> 64 scale groups.
    contract = _geometry_contract(6, 32)
    weights = _geometry_weights(packed_cols=768, scale_groups=64)
    corrected = _contract_with_prequantized_tensor_geometry(contract, weights)
    assert corrected.mtp_quant_group_size == 64


def test_prequantized_8bit_divisor_geometry_unchanged() -> None:
    # Divisor widths keep the original arithmetic: 1024 words * 4/word = 4096.
    contract = _geometry_contract(8, 32)
    weights = _geometry_weights(packed_cols=1024, scale_groups=64)
    corrected = _contract_with_prequantized_tensor_geometry(contract, weights)
    assert corrected.mtp_quant_group_size == 64


def test_prequantized_nondivisor_padded_geometry_keeps_declared_contract() -> None:
    # 641 words * 32 = 20512 bits does not divide by 5: padded/ambiguous rows
    # must be skipped rather than inferred, leaving the declared size alone.
    contract = _geometry_contract(5, 64)
    weights = _geometry_weights(packed_cols=641, scale_groups=64)
    corrected = _contract_with_prequantized_tensor_geometry(contract, weights)
    assert corrected.mtp_quant_group_size == 64
