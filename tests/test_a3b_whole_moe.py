"""Correct-by-construction whole-MoE routing for exact A3B small rows."""

from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import pytest

import mtplx.a3b_whole_moe as whole_moe_module
from mtplx import a3b_compiled_target_prefix as compiled_target_module
from mtplx import generation as generation_module
from mtplx import runtime as runtime_module
from mtplx.a3b_whole_moe import (
    A3BWholeMoeConfigError,
    install_a3b_whole_moe,
    prepare_a3b_whole_moe,
    run_a3b_whole_moe_selfcheck,
    validate_a3b_whole_moe_request,
)
from mtplx.kernels import a3b_whole_moe as kernel_module
from mtplx.moe_packed_projections import (
    PackedGateUpMLP,
    PackedSwitchGLU,
    _PackedDenseProjection,
    _PackedQuantizedProjection,
)


_LAYER_TYPES = tuple(
    "linear_attention" if index % 4 != 3 else "full_attention"
    for index in range(40)
)


class _ArraySpec:
    def __init__(self, shape, dtype=mx.bfloat16) -> None:
        self.shape = tuple(shape)
        self.ndim = len(self.shape)
        self.dtype = dtype

    def reshape(self, *shape):
        return _ArraySpec(shape, self.dtype)


class _ResultSpec(_ArraySpec):
    def __init__(self, label: str, shape) -> None:
        super().__init__(shape)
        self.label = label

    def reshape(self, *shape):
        return _ResultSpec(self.label, shape)


class _FakeSparseBlock:
    def __init__(self, **attributes) -> None:
        vars(self).update(attributes)
        self.stock_calls = []

    def __call__(self, value):
        self.stock_calls.append(value)
        return _ResultSpec("stock", value.shape)


class _Projection:
    def __init__(
        self,
        weight_shape,
        *,
        bits: int | None = None,
        group_size: int | None = None,
        scales_shape=None,
        biases_shape=None,
    ) -> None:
        self.weight = _ArraySpec(
            weight_shape, mx.uint32 if bits is not None else mx.bfloat16
        )
        if bits is not None:
            self.bits = bits
            self.group_size = group_size
            self.mode = "affine"
            self.scales = _ArraySpec(scales_shape)
            self.biases = _ArraySpec(biases_shape)

    def __contains__(self, name: str) -> bool:
        return hasattr(self, name)

    def __getitem__(self, name: str):
        return getattr(self, name)


def _target_block():
    return _sparse_block(
        gate=_Projection(
            (256, 512),
            bits=8,
            group_size=64,
            scales_shape=(256, 32),
            biases_shape=(256, 32),
        ),
        routed_group_size=64,
        shared_quantized=True,
        scalar_gate=_Projection(
            (1, 512),
            bits=8,
            group_size=64,
            scales_shape=(1, 32),
            biases_shape=(1, 32),
        ),
    )


def _mtp_block():
    return _sparse_block(
        gate=_Projection((256, 2048)),
        routed_group_size=32,
        shared_quantized=False,
        scalar_gate=_Projection((1, 2048)),
    )


def _quantized_routed_projection(output: int, *, group_size: int):
    return _Projection(
        (256, output, 2048 * 4 // 32),
        bits=4,
        group_size=group_size,
        scales_shape=(256, output, 2048 // group_size),
        biases_shape=(256, output, 2048 // group_size),
    )


def _packed_quantized_projection(
    weight_shape,
    *,
    group_size: int,
    scales_shape,
):
    return _PackedQuantizedProjection(
        _ArraySpec(weight_shape, mx.uint32),
        _ArraySpec(scales_shape),
        _ArraySpec(scales_shape),
        group_size=group_size,
        bits=4,
        mode="affine",
    )


def _sparse_block(*, gate, routed_group_size: int, shared_quantized: bool, scalar_gate):
    routed_gate_up = _packed_quantized_projection(
        (256, 1024, 2048 * 4 // 32),
        group_size=routed_group_size,
        scales_shape=(256, 1024, 2048 // routed_group_size),
    )
    routed_down = _Projection(
        (256, 2048, 512 * 4 // 32),
        bits=4,
        group_size=routed_group_size,
        scales_shape=(256, 2048, 512 // routed_group_size),
        biases_shape=(256, 2048, 512 // routed_group_size),
    )
    switch_mlp = PackedSwitchGLU(routed_gate_up, routed_down, object(), 512)
    if shared_quantized:
        shared_gate_up = _packed_quantized_projection(
            (1024, 256),
            group_size=64,
            scales_shape=(1024, 32),
        )
        shared_down = _Projection(
            (2048, 64),
            bits=4,
            group_size=64,
            scales_shape=(2048, 8),
            biases_shape=(2048, 8),
        )
    else:
        shared_gate_up = _PackedDenseProjection(_ArraySpec((1024, 2048)))
        shared_down = _Projection((2048, 512))
    shared_expert = PackedGateUpMLP(shared_gate_up, shared_down, 512)
    return _FakeSparseBlock(
        gate=gate,
        switch_mlp=switch_mlp,
        shared_expert=shared_expert,
        shared_expert_gate=scalar_gate,
        num_experts=256,
        top_k=8,
        norm_topk_prob=True,
        sharding_group=None,
    )


def _exact_config():
    return {
        "model_type": "qwen3_5_moe",
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "text_config": {
            "model_type": "qwen3_5_moe_text",
            "hidden_size": 2048,
            "num_hidden_layers": 40,
            "layer_types": list(_LAYER_TYPES),
            "num_experts": 256,
            "num_experts_per_tok": 8,
            "norm_topk_prob": True,
            "moe_intermediate_size": 512,
            "shared_expert_intermediate_size": 512,
            "mtp_num_hidden_layers": 1,
        },
    }


def _exact_model():
    target_blocks = [_target_block() for _ in range(40)]
    target_layers = []
    for index, block in enumerate(target_blocks):
        is_linear = _LAYER_TYPES[index] == "linear_attention"
        layer = SimpleNamespace(
            is_linear=is_linear,
            mlp=block,
            post_attention_layernorm=SimpleNamespace(
                weight=_ArraySpec((2048,))
            ),
        )
        if is_linear:
            layer.linear_attn = object()
        else:
            layer.self_attn = object()
        target_layers.append(layer)
    mtp_block = _mtp_block()
    mtp_layer = SimpleNamespace(
        mlp=mtp_block,
        self_attn=object(),
        post_attention_layernorm=SimpleNamespace(weight=_ArraySpec((2048,))),
    )
    model = SimpleNamespace(
        language_model=SimpleNamespace(model=SimpleNamespace(layers=target_layers)),
        mtp=SimpleNamespace(layers=[mtp_layer]),
    )
    return model, target_blocks, [mtp_block]


def test_flag_off_returns_no_plan_and_preserves_all_block_classes(monkeypatch):
    monkeypatch.delenv("MTPLX_A3B_WHOLE_MOE_FUSION", raising=False)
    model, targets, mtp = _exact_model()
    original_classes = tuple(type(block) for block in (*targets, *mtp))

    assert prepare_a3b_whole_moe(model, config=_exact_config()) is None
    assert tuple(type(block) for block in (*targets, *mtp)) == original_classes


def test_reset_clears_accepted_mtp_installation_status():
    whole_moe_module._STATS["accepted_mtp_blocks"] = 1

    whole_moe_module._reset_a3b_whole_moe_for_tests()

    assert whole_moe_module.a3b_whole_moe_stats()["accepted_mtp_blocks"] == 0


def test_exact_checkpoint_builds_40_target_and_one_mtp_binding(monkeypatch):
    monkeypatch.setenv("MTPLX_A3B_WHOLE_MOE_FUSION", "1")
    model, _, _ = _exact_model()

    plan = prepare_a3b_whole_moe(model, config=_exact_config())

    assert len(plan.target_bindings) == 40
    assert len(plan.mtp_bindings) == 1
    assert {binding.variant for binding in plan.target_bindings} == {
        "target_q8g64_q4g64"
    }
    assert plan.mtp_bindings[0].variant == "mtp_dense_q4g32_dense"
    assert plan.target_bindings[0].routed_gate_up.weight.shape == (
        256,
        1024,
        256,
    )
    assert plan.target_bindings[0].shared_gate_up.weight.shape == (1024, 256)
    assert plan.mtp_bindings[0].shared_gate_up.weight.shape == (1024, 2048)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda block: setattr(
                block,
                "switch_mlp",
                SimpleNamespace(
                    gate_up_proj=block.switch_mlp.gate_up_proj,
                    down_proj=block.switch_mlp.down_proj,
                    _split_at=512,
                ),
            ),
            "PackedSwitchGLU",
        ),
        (
            lambda block: setattr(block.switch_mlp, "_split_at", 256),
            "split 512",
        ),
        (
            lambda block: setattr(block.shared_expert, "_split_at", 256),
            "split 512",
        ),
        (
            lambda block: setattr(block.switch_mlp.down_proj, "bias", _ArraySpec((2048,))),
            "additive bias",
        ),
    ],
)
def test_exact_packed_ownership_is_required_at_construction(monkeypatch, mutate, match):
    monkeypatch.setenv("MTPLX_A3B_WHOLE_MOE_FUSION", "1")
    model, targets, _ = _exact_model()
    mutate(targets[0])

    with pytest.raises(A3BWholeMoeConfigError, match=match):
        prepare_a3b_whole_moe(model, config=_exact_config())


def test_existing_installed_or_row_owned_route_conflict_is_rejected(monkeypatch):
    monkeypatch.setenv("MTPLX_A3B_WHOLE_MOE_FUSION", "1")
    for marker in ("_mtplx_a3b_whole_moe_route", "_mtplx_a3b_router_route"):
        model, targets, _ = _exact_model()
        setattr(type(targets[0]), marker, object())
        try:
            with pytest.raises(A3BWholeMoeConfigError, match="route conflict"):
                prepare_a3b_whole_moe(model, config=_exact_config())
        finally:
            delattr(type(targets[0]), marker)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda config, model: config.update(model_type="qwen3_5"),
            "model architecture",
        ),
        (
            lambda config, model: config["text_config"].update(hidden_size=4096),
            "topology",
        ),
        (
            lambda config, model: model.language_model.model.layers.pop(),
            "40 target",
        ),
        (
            lambda config, model: model.mtp.layers.clear(),
            "one MTP",
        ),
        (
            lambda config, model: setattr(
                model.language_model.model.layers[0].mlp, "top_k", 4
            ),
            "top-k 8",
        ),
        (
            lambda config, model: setattr(
                model.language_model.model.layers[0].mlp,
                "norm_topk_prob",
                False,
            ),
            "normalization",
        ),
        (
            lambda config, model: setattr(
                model.language_model.model.layers[0].mlp,
                "sharding_group",
                object(),
            ),
            "sharding",
        ),
        (
            lambda config, model: setattr(
                model.language_model.model.layers[0].post_attention_layernorm.weight,
                "dtype",
                mx.float32,
            ),
            "BF16 hidden",
        ),
        (
            lambda config, model: setattr(
                model.language_model.model.layers[0].mlp.gate,
                "group_size",
                32,
            ),
            "target router",
        ),
        (
            lambda config, model: setattr(
                model.language_model.model.layers[0].mlp.switch_mlp.gate_up_proj.weight,
                "shape",
                (256, 1024, 128),
            ),
            "target routed gate/up",
        ),
        (
            lambda config, model: setattr(
                model.mtp.layers[0].mlp.switch_mlp.down_proj,
                "group_size",
                64,
            ),
            "MTP routed down",
        ),
        (
            lambda config, model: setattr(
                model.mtp.layers[0].mlp.shared_expert.gate_up_proj.weight,
                "dtype",
                mx.float32,
            ),
            "MTP shared gate/up",
        ),
    ],
)
def test_external_contract_mismatch_fails_before_install(
    monkeypatch,
    mutate,
    match,
):
    monkeypatch.setenv("MTPLX_A3B_WHOLE_MOE_FUSION", "1")
    config = _exact_config()
    model, targets, mtp = _exact_model()
    original_classes = tuple(type(block) for block in (*targets, *mtp))
    mutate(config, model)

    with pytest.raises(A3BWholeMoeConfigError, match=match):
        prepare_a3b_whole_moe(model, config=config)

    assert tuple(type(block) for block in (*targets, *mtp)) == original_classes


def test_fixed_kernel_sources_encode_exact_geometry_without_hot_validation():
    sources = kernel_module.all_whole_moe_sources()
    # 12 shipped K1 kernels + 4 k=2 (m3) kernels: stage1 projection/finalizer,
    # stage2, stage3 recompiled at ROWS=3; + the row-split M2 stage3.
    assert len(sources) == 17
    for source in sources.values():
        assert "constexpr uint HIDDEN = 2048" in source
        assert "constexpr uint EXPERTS = 256" in source
        assert "constexpr uint TOP_K = 8" in source
        assert "constexpr uint INTERMEDIATE = 512" in source
        for forbidden in (
            "getenv",
            "dtype",
            "shape",
            "eligible",
            "fallback",
            "lane_disabled",
            "record_",
            "counter",
        ):
            assert forbidden not in source
    for name, source in sources.items():
        if "stage1" not in name:
            assert "threadgroup_barrier" not in source


class _CapturedKernel:
    def __init__(self) -> None:
        self.call = None

    def __call__(self, **kwargs):
        self.call = kwargs
        return tuple(
            _ArraySpec(shape, dtype)
            for shape, dtype in zip(
                kwargs["output_shapes"], kwargs["output_dtypes"]
            )
        )


def test_target_m2_stage1_runs_fixed_projection_then_exact_finalizer(monkeypatch):
    monkeypatch.setenv("MTPLX_A3B_WHOLE_MOE_FUSION", "1")
    projection = _CapturedKernel()
    finalizer = _CapturedKernel()
    monkeypatch.setattr(
        kernel_module,
        "_build_target_m2_stage1_projection_kernel",
        lambda: projection,
    )
    monkeypatch.setattr(
        kernel_module,
        "_build_target_m2_stage1_finalizer_kernel",
        lambda: finalizer,
    )
    model, _, _ = _exact_model()
    binding = prepare_a3b_whole_moe(
        model,
        config=_exact_config(),
    ).target_bindings[0]
    value = _ArraySpec((2, 2048))

    expert_ids, route_scores, shared_gate = kernel_module.target_m2_stage1(
        value, binding
    )

    assert projection.call["grid"] == (8 * 256, 1, 1)
    assert projection.call["threadgroup"] == (256, 1, 1)
    assert projection.call["output_shapes"] == [(2, 256), (2, 1)]
    assert projection.call["output_dtypes"] == [mx.bfloat16, mx.bfloat16]
    assert finalizer.call["grid"] == (256, 1, 1)
    assert finalizer.call["threadgroup"] == (256, 1, 1)
    assert finalizer.call["output_shapes"] == [(2, 8), (2, 8)]
    assert finalizer.call["output_dtypes"] == [mx.uint32, mx.bfloat16]
    assert finalizer.call["inputs"][0].shape == (2, 256)
    assert expert_ids.shape == (2, 8)
    assert route_scores.shape == (2, 8)
    assert shared_gate.shape == (2, 1)


def test_target_m2_stage2_is_row_paired_with_fixed_288_threadgroups(monkeypatch):
    monkeypatch.setenv("MTPLX_A3B_WHOLE_MOE_FUSION", "1")
    kernel = _CapturedKernel()
    monkeypatch.setattr(
        kernel_module,
        "_build_target_m2_stage2_kernel",
        lambda: kernel,
    )
    model, _, _ = _exact_model()
    binding = prepare_a3b_whole_moe(
        model,
        config=_exact_config(),
    ).target_bindings[0]

    output = kernel_module.target_m2_stage2(
        _ArraySpec((2, 2048)),
        _ArraySpec((2, 8), mx.uint32),
        binding,
    )

    assert output.shape == (2, 9, 512)
    assert kernel.call["grid"] == (288 * 128, 1, 1)
    assert kernel.call["threadgroup"] == (128, 1, 1)
    assert kernel.call["output_shapes"] == [(2, 9, 512)]
    assert kernel.call["output_dtypes"] == [mx.bfloat16]


def test_fixed_entrypoint_launch_table(monkeypatch):
    expected = {
        "target_m1_stage1": ((256, 1, 1), (256, 1, 1)),
        "target_m2_stage1_projection": ((8 * 256, 1, 1), (256, 1, 1)),
        "target_m2_stage1_finalizer": ((256, 1, 1), (256, 1, 1)),
        "mtp_m1_stage1": ((256, 1, 1), (256, 1, 1)),
        "target_m1_stage2": ((288 * 128, 1, 1), (128, 1, 1)),
        "target_m2_stage2": ((288 * 128, 1, 1), (128, 1, 1)),
        "mtp_m1_stage2": ((288 * 128, 1, 1), (128, 1, 1)),
        "target_m1_stage3": ((128 * 128, 1, 1), (128, 1, 1)),
        "target_m2_stage3": ((128 * 128, 1, 1), (128, 1, 1)),
        "mtp_m1_stage3": ((128 * 128, 1, 1), (128, 1, 1)),
        "target_m2r1_stage2": ((288 * 128, 1, 1), (128, 1, 1)),
        "target_m2r1_stage3": ((128 * 128, 1, 1), (128, 1, 1)),
        "target_m3_stage1_projection": ((8 * 256, 1, 1), (256, 1, 1)),
        "target_m3_stage1_finalizer": ((256, 1, 1), (256, 1, 1)),
        "target_m3_stage2": ((288 * 128, 1, 1), (128, 1, 1)),
        "target_m3_stage3": ((128 * 128, 1, 1), (128, 1, 1)),
    }
    assert kernel_module.whole_moe_launch_table() == expected


def test_stage1_sources_encode_router_softmax_top8_and_score_rounding():
    sources = kernel_module.all_whole_moe_sources()
    source = sources["target_m1_stage1"]
    assert "qdot8_affine" in source
    assert "constexpr uint ROUTER_GROUP = 64" in source
    assert "threadgroup bfloat router_logits[ROWS * EXPERTS]" in source
    assert "metal::exp" in source
    assert "simd_max" in source
    assert "bfloat rounded_denominator = bfloat(0.0f)" in source
    assert "candidate_probability == winner_probability" in source

    projection = sources["target_m2_stage1_projection"]
    assert "qdot8_affine" in projection
    assert "constexpr uint ROUTER_GROUP = 64" in projection
    assert "metal::exp" not in projection
    assert "simd_max" not in projection

    finalizer = sources["target_m2_stage1_finalizer"]
    assert "qdot8_affine" not in finalizer
    assert "metal::exp" in finalizer
    assert "simd_max" in finalizer
    assert "bfloat rounded_denominator = bfloat(0.0f)" in finalizer
    assert "candidate_probability == winner_probability" in finalizer
    source = sources["mtp_m1_stage1"]
    assert "dense_bf16_dot" in source
    assert "threadgroup bfloat router_logits[ROWS * EXPERTS]" in source
    assert "metal::exp" in source
    assert "bfloat rounded_denominator = bfloat(0.0f)" in source


def test_target_m2_stage1_projection_tiles_experts_without_duplicate_row_tiles():
    source = kernel_module.all_whole_moe_sources()[
        "target_m2_stage1_projection"
    ]

    assert "constexpr uint OUTPUT_EXPERTS_PER_GROUP = 8 * 4" in source
    assert kernel_module.STAGE1_PROJECTION_THREADGROUPS == 8
    assert "uint expert_tile = threadgroup_position_in_grid.x" in source
    assert "uint expert_base = expert_tile * OUTPUT_EXPERTS_PER_GROUP" in source
    assert "for (uint subtile = 0; subtile < 8; ++subtile)" not in source
    assert "uint row = threadgroup_position_in_grid.x" not in source
    assert "float router_result[ROWS][4]" in source
    assert "for (uint row = 0; row < ROWS; ++row)" in source
    assert source.count("uchar packed_weight = weights[weight_base + item]") == 1
    assert source.count("float scale = float(router_scales[metadata_index])") == 1
    assert source.count("float bias = float(router_biases[metadata_index])") == 1
    assert "router_logits[row * EXPERTS + expert] = bfloat(reduced)" in source


def test_target_m2_stage1_projection_loads_shared_gate_once_for_both_rows():
    source = kernel_module.all_whole_moe_sources()[
        "target_m2_stage1_projection"
    ]

    assert "if (expert_tile == 0 && simd_gid == 0)" in source
    assert "float shared_partial[ROWS]" in source
    assert source.count("uchar packed_weight = weights[k_lane + item]") == 1
    assert source.count("float scale = float(shared_gate_scales[metadata_index])") == 1
    assert source.count("float bias = float(shared_gate_biases[metadata_index])") == 1
    assert "shared_gate[row] = bfloat(shared_reduced)" in source


def test_target_m2_stage1_finalizer_preserves_exact_two_row_arithmetic():
    source = kernel_module.all_whole_moe_sources()[
        "target_m2_stage1_finalizer"
    ]

    assert "for (uint row = 0; row < ROWS; ++row)" in source
    assert "float local_logit = float(router_logits[row * EXPERTS + tid])" in source
    assert "probabilities[row * EXPERTS + tid] = bfloat(" in source
    assert "candidate_probability == winner_probability" in source
    assert "probability1 == probability0 && index1 > index0" in source
    assert "bfloat rounded_denominator = bfloat(0.0f)" in source
    assert "merged_probabilities[TOP_K - 1 - output_rank]" in source
    assert source.count("threadgroup_barrier(mem_flags::mem_threadgroup)") >= 5
    # Each simdgroup reduces the eight partials itself: no broadcast through a
    # slot other simdgroups may still be reading (the 2026-10-14 M1 race).
    assert "simd_values[0] =" not in source
    assert "simd_sum_values[simd_gid] = local_sum" in source
    assert "probability / row_sum" in source


def test_stage1_softmax_has_no_partial_slot_broadcast():
    for name, source in kernel_module.all_whole_moe_sources().items():
        if "local_maximum" not in source:
            continue
        assert "simd_values[0] =" not in source, name
        assert "simd_max_values[simd_gid] = local_maximum" in source, name
        assert "simd_sum_values[simd_gid] = local_sum" in source, name


def test_stage2_sources_encode_selected_q4_and_exact_bf16_swiglu():
    sources = kernel_module.all_whole_moe_sources()
    for name in ("target_m1_stage2", "target_m2_stage2"):
        source = sources[name]
        assert "constexpr uint ROUTED_GROUP = 64" in source
        assert "qdot4_affine" in source
        assert "sigmoid_mlx_exact" in source
        assert "bfloat gate_value = bfloat(gate_sum)" in source
        assert "bfloat up_value = bfloat(up_sum)" in source
        assert "activations[output_index] = bfloat(silu * up_value)" in source
        assert "routed_gate_up_weight" in source
        assert "shared_gate_up_weight" in source
        assert "routed_gate_weight" not in source
        assert "routed_up_weight" not in source
    source = sources["mtp_m1_stage2"]
    assert "constexpr uint ROUTED_GROUP = 32" in source
    assert "qdot4_affine" in source
    assert "dense_shared_dot" in source
    assert "sigmoid_mlx_exact" in source
    assert "routed_gate_up_weight" in source
    assert "shared_gate_up_weight" in source
    assert "routed_gate_weight" not in source
    assert "routed_up_weight" not in source


def test_target_m2_stage2_reuses_fixed_shared_expert_weights_across_rows():
    source = kernel_module.all_whole_moe_sources()["target_m2_stage2"]

    assert "// row-specific selected expert path" in source
    assert "expert_ids[row * TOP_K + slot]" in source
    assert "// row-paired fixed shared expert path" in source
    assert "float shared_gate_result[ROWS][4]" in source
    assert "float shared_up_result[ROWS][4]" in source
    assert source.count(
        "ushort shared_gate_bits = shared_gate_packed[piece]"
    ) == 1
    assert source.count("ushort shared_up_bits = shared_up_packed[piece]") == 1
    assert source.count("float gate_scale = float(") == 1
    assert source.count("float gate_bias = float(") == 1
    assert source.count("float up_scale = float(") == 1
    assert source.count("float up_bias = float(") == 1
    weight_load = source.index(
        "ushort shared_gate_bits = shared_gate_packed[piece]"
    )
    both_row_consumers = source.index(
        "for (uint row = 0; row < ROWS; ++row)", weight_load
    )
    assert weight_load < both_row_consumers
    assert "activations[output_index] = bfloat(silu * up_value)" in source


def test_m2_stage3_rowsplit_source_is_the_m2r1_chain_with_the_grid_row():
    source = kernel_module.all_whole_moe_sources()["target_m2_stage3_rowsplit"]
    assert "uint split_row = threadgroup_position_in_grid.y;" in source
    assert "expert_ids[split_row * TOP_K + slot]" in source
    assert "route_scores[split_row * TOP_K + slot]" in source
    assert "shared_gate[split_row]" in source
    assert "output[split_row * HIDDEN + output_column] = bfloat(" in source
    assert "routed_accumulator1" not in source
    for needle in ("expert_ids[0 *", "route_scores[0 *", "(0 * ACTIVATION_SLOTS", "shared_gate[0]"):
        assert needle not in source


def test_stage3_sources_encode_down_reduction_shared_gate_and_only_final_store():
    sources = kernel_module.all_whole_moe_sources()
    for name in ("target_m1_stage3", "target_m2_stage3"):
        source = sources[name]
        assert "constexpr uint ROUTED_GROUP = 64" in source
        assert "qdot4_affine" in source
        assert "bfloat down_value = bfloat(down_sum)" in source
        assert "bfloat route_product = bfloat(" in source
        if name == "target_m2_stage3":
            assert "routed_accumulator0[result_index] = bfloat(" in source
            assert "routed_accumulator1[result_index] = bfloat(" in source
        else:
            assert "routed_accumulator[result_index] = bfloat(" in source
        assert "sigmoid_mlx_exact" in source
        if name == "target_m2_stage3":
            assert "output[output_column] = bfloat(" in source
            assert "output[HIDDEN + output_column] = bfloat(" in source
        else:
            assert "output[output_index] = bfloat(" in source
        assert "routed_outputs" not in source
        assert "shared_output" not in source
    source = sources["mtp_m1_stage3"]
    assert "constexpr uint ROUTED_GROUP = 32" in source
    assert "qdot4_affine" in source
    assert "dense_shared_down" in source
    assert "output[output_index] = bfloat(" in source


def test_stage3_sources_use_exact_m1_and_row_paired_m2_ownership():
    sources = kernel_module.all_whole_moe_sources()
    for name in ("target_m1_stage3", "mtp_m1_stage3"):
        source = sources[name]
        assert "constexpr uint OUTPUT_TILES = HIDDEN / 16" in source
        assert "uint group = threadgroup_position_in_grid.x" in source
        assert "uint row = group / OUTPUT_TILES" in source
        assert "uint tile = group - row * OUTPUT_TILES" in source
        assert "for (uint row = 0; row < ROWS; ++row)" not in source

    source = sources["target_m2_stage3"]
    assert "constexpr uint OUTPUT_TILES = HIDDEN / 16" in source
    assert "uint tile = threadgroup_position_in_grid.x" in source
    assert "uint group =" not in source
    assert "uint row =" not in source
    assert "threadgroup bfloat shared_inputs[ROWS * 4 * INTERMEDIATE]" in source
    assert "routed_accumulator0" in source
    assert "routed_accumulator1" in source


def test_target_m2_stage3_loads_each_shared_down_word_once_for_both_rows():
    source = kernel_module.all_whole_moe_sources()["target_m2_stage3"]

    assert source.count("ushort packed = shared_down_packed[piece]") == 1
    assert "shared_quantized_dot0" in source
    assert "shared_quantized_dot1" in source
    assert "shared_inputs[shared_input_base0 + item]" in source
    assert "shared_inputs[shared_input_base1 + item]" in source
    assert "shared_inputs[shared_input_base0 + item + 1] = x01" in source
    assert "shared_inputs[shared_input_base1 + item + 1] = x11" in source
    assert "float(shared_inputs[shared_input_base0 + item + 1]) / 16.0f" in source
    assert "float(shared_inputs[shared_input_base1 + item + 1]) / 16.0f" in source
    assert "bfloat(float(x" not in source


def test_all_fixed_entrypoints_launch_directly_without_runtime_validation(monkeypatch):
    monkeypatch.setenv("MTPLX_A3B_WHOLE_MOE_FUSION", "1")
    model, _, _ = _exact_model()
    plan = prepare_a3b_whole_moe(model, config=_exact_config())
    target = plan.target_bindings[0]
    mtp = plan.mtp_bindings[0]
    for name, (grid, threadgroup) in kernel_module.whole_moe_launch_table().items():
        if name.startswith(("target_m2_stage1_", "target_m3_stage1_")):
            # Split projection/finalizer stage1 kernels have no per-split
            # entrypoint; the combined target_m2_stage1 / target_m3_stage1 are
            # exercised separately.
            continue
        rows = (
            1
            if "m2r1" in name
            else (3 if "m3" in name else (2 if "m2" in name else 1))
        )
        binding = mtp if name.startswith("mtp") else target
        captured = _CapturedKernel()
        monkeypatch.setattr(
            kernel_module,
            f"_build_{name}_kernel",
            lambda captured=captured: captured,
        )
        entrypoint = getattr(kernel_module, name)
        value = _ArraySpec((rows, 2048))
        if name.endswith("stage1"):
            result = entrypoint(value, binding)
            assert tuple(item.shape for item in result) == (
                (rows, 8),
                (rows, 8),
                (rows, 1),
            )
            if binding is target:
                expected_inputs = [
                    value,
                    binding.router.weight,
                    binding.router.scales,
                    binding.router.biases,
                    binding.shared_scalar_gate.weight,
                    binding.shared_scalar_gate.scales,
                    binding.shared_scalar_gate.biases,
                ]
            else:
                expected_inputs = [
                    value,
                    binding.router.weight,
                    binding.shared_scalar_gate.weight,
                ]
            expected_shapes = [(rows, 8), (rows, 8), (rows, 1)]
            expected_dtypes = [mx.uint32, mx.bfloat16, mx.bfloat16]
        elif name.endswith("stage2"):
            expert_ids = _ArraySpec((rows, 8), mx.uint32)
            result = entrypoint(value, expert_ids, binding)
            assert result.shape == (rows, 9, 512)
            expected_inputs = [
                value,
                expert_ids,
                binding.routed_gate_up.weight,
                binding.routed_gate_up.scales,
                binding.routed_gate_up.biases,
                binding.shared_gate_up.weight,
            ]
            if binding is target:
                expected_inputs.extend(
                    [
                        binding.shared_gate_up.scales,
                        binding.shared_gate_up.biases,
                    ]
                )
            expected_shapes = [(rows, 9, 512)]
            expected_dtypes = [mx.bfloat16]
        else:
            activations = _ArraySpec((rows, 9, 512))
            expert_ids = _ArraySpec((rows, 8), mx.uint32)
            route_scores = _ArraySpec((rows, 8))
            shared_gate = _ArraySpec((rows, 1))
            result = entrypoint(
                activations,
                expert_ids,
                route_scores,
                shared_gate,
                binding,
            )
            assert result.shape == (rows, 2048)
            expected_inputs = [
                activations,
                expert_ids,
                route_scores,
                shared_gate,
                binding.routed_down.weight,
                binding.routed_down.scales,
                binding.routed_down.biases,
                binding.shared_down.weight,
            ]
            if binding is target:
                expected_inputs.extend(
                    [binding.shared_down.scales, binding.shared_down.biases]
                )
            expected_shapes = [(rows, 2048)]
            expected_dtypes = [mx.bfloat16]
        assert len(captured.call["inputs"]) == len(expected_inputs)
        assert all(
            actual is expected
            for actual, expected in zip(captured.call["inputs"], expected_inputs)
        )
        assert captured.call["output_shapes"] == expected_shapes
        assert captured.call["output_dtypes"] == expected_dtypes
        assert captured.call["grid"] == grid
        assert captured.call["threadgroup"] == threadgroup
        source = inspect.getsource(entrypoint)
        for forbidden in (
            "os.environ",
            "metal.is_available",
            ".dtype",
            ".shape",
            "eligible",
            "fallback",
            "lane_disabled",
            "try:",
            "except",
            "raise ",
        ):
            assert forbidden not in source


def test_installed_routes_prebind_kernel_objects_before_hot_call():
    for name in (
        "bind_target_m1_stage3",
        "bind_target_m2_stage3",
        "bind_mtp_m1_stage3",
    ):
        binder = getattr(kernel_module, name)
        source = inspect.getsource(binder)
        call_start = source.index("def call(")
        assert "_build_" in source[:call_start]
        assert "_build_" not in source[call_start:]
        assert "_KERNELS" not in source[call_start:]


def _exact_selfcheck_report():
    return {
        "lanes": {
            "a3b_whole_moe_target_m1": "ok",
            "a3b_whole_moe_target_m2": "ok",
            "a3b_whole_moe_target_m1_m2_row_parity": "ok",
            "a3b_whole_moe_mtp_m1": "ok",
        },
        "dmax": {
            "a3b_whole_moe_target_m1": 0.125,
            "a3b_whole_moe_target_m2": 0.25,
            "a3b_whole_moe_target_m1_m2_row_parity": 0.0,
            "a3b_whole_moe_mtp_m1": 0.125,
        },
        "a3b_whole_moe_components": {
            "a3b_whole_moe_target_m1": {
                "expert_ids": 0.0,
                "route_scores": 0.001,
                "shared_gate": 0.01,
                "activations": 0.0625,
                "stage3_output": 0.125,
                "output": 0.125,
            },
            "a3b_whole_moe_target_m2": {
                "expert_ids": 0.0,
                "route_scores": 0.001,
                "shared_gate": 0.01,
                "activations": 0.0625,
                "stage3_output": 0.25,
                "output": 0.25,
            },
            "a3b_whole_moe_mtp_m1": {
                "expert_ids": 0.0,
                "route_scores": 0.001,
                "shared_gate": 0.01,
                "activations": 0.0625,
                "stage3_output": 0.125,
                "output": 0.125,
            },
        },
    }


def _passing_full_graph_preflight():
    return {
        "a3b_whole_moe_target_prefix_full_graph_m1": "ok",
        "a3b_whole_moe_target_prefix_full_graph_m2": "ok",
    }


def test_model_bound_selfcheck_runs_only_changed_target_m2_geometry(monkeypatch):
    monkeypatch.setenv("MTPLX_A3B_WHOLE_MOE_FUSION", "1")
    model, _, _ = _exact_model()
    plan = prepare_a3b_whole_moe(model, config=_exact_config())
    calls = []

    def check(binding, *, rows):
        calls.append((binding.variant, rows))
        return {
            "expert_ids": 0.0,
            "route_scores": 0.001,
            "shared_gate": 0.01,
            "activations": float(rows) / 128.0,
            "stage3_output": float(rows) / 64.0,
            "output": float(rows) / 64.0,
        }

    monkeypatch.setattr(whole_moe_module, "_check_whole_moe_lane", check)
    parity_calls = []

    def parity(binding):
        parity_calls.append(binding.variant)
        return 0.0

    monkeypatch.setattr(
        whole_moe_module, "_check_whole_moe_m1_m2_row_parity", parity
    )
    report = run_a3b_whole_moe_selfcheck(
        plan,
        {"lanes": {"existing_lane": "ok"}, "dmax": {"existing_lane": 0.0}},
    )

    assert calls == [
        *(("target_q8g64_q4g64", 2),) * 40,
    ]
    assert parity_calls == ["target_q8g64_q4g64"] * 40
    assert report["lanes"] == {
        "existing_lane": "ok",
        "a3b_whole_moe_target_m2": "ok",
        "a3b_whole_moe_target_m1_m2_row_parity": "ok",
    }
    assert report["dmax"]["a3b_whole_moe_target_m2"] == 2.0 / 64.0
    assert report["dmax"]["a3b_whole_moe_target_m1_m2_row_parity"] == 0.0
    assert report["a3b_whole_moe_components"][
        "a3b_whole_moe_target_m2"
    ]["activations"] == 2.0 / 128.0


def test_selfcheck_applies_component_specific_limits() -> None:
    assert whole_moe_module._SELFCHECK_LIMITS == {
        "expert_ids": 0.0,
        "route_scores": 0.0078125,
        "shared_gate": 0.0625,
        "activations": 0.125,
        "stage3_output": 0.5,
        "output": 0.5,
    }


def test_whole_moe_flag_requires_load_time_selfcheck(monkeypatch):
    from mtplx.kernel_selfcheck import selfcheck_enabled

    monkeypatch.delenv("MTPLX_KERNEL_SELFCHECK", raising=False)
    monkeypatch.delenv("MTPLX_QWEN_ROW_OWNED_ROUTER", raising=False)
    monkeypatch.delenv("MTPLX_FUSE_GDN_POST_CONV", raising=False)
    monkeypatch.setenv("MTPLX_A3B_WHOLE_MOE_FUSION", "1")

    assert selfcheck_enabled() is True


def test_model_bound_selfcheck_is_deterministic_and_compilation_gated():
    source = inspect.getsource(whole_moe_module._check_whole_moe_lane)

    assert "mx.arange" in source
    assert "mx.random" not in source
    assert "mx.compile(route)" in source
    assert "mx.compile(lambda current: binding.block(current))" in source
    assert "kernel_module.target_m2_stage1(value, binding)" in source
    assert "kernel_module.target_m2_stage2(" in source
    assert "_packed_stage12_unchecked(value, binding, rows=rows)" in source
    assert "stage3(" in source
    assert "reference_activations" in source
    assert '"stage3_output"' in source
    assert "mx.array_equal(expert_ids, reference_ids)" in source


def _patch_whole_moe_stages(monkeypatch):
    monkeypatch.setattr(
        whole_moe_module,
        "_target_m2_route",
        lambda binding: lambda value: _ResultSpec("target_m2", value.shape),
    )
    monkeypatch.setattr(
        whole_moe_module,
        "_target_m1_route",
        lambda binding: lambda value: _ResultSpec("target_m1", value.shape),
    )
    monkeypatch.setattr(
        whole_moe_module,
        "_target_m3_route",
        lambda binding: lambda value: _ResultSpec("target_m3", value.shape),
    )


def test_installed_partition_prebinds_all_four_exact_target_m2_kernels():
    stage1_source = inspect.getsource(whole_moe_module._row_owned_stage1_unchecked)
    assert "mx.softmax" in stage1_source
    assert "precise=True" in stage1_source
    assert "_qwen_row_owned_route_unchecked" in stage1_source
    stage2_source = inspect.getsource(whole_moe_module._packed_stage2_unchecked)
    assert "gate_up_proj.gather" in stage2_source
    assert "shared_expert.gate_up_proj" in stage2_source
    assert "swiglu" in stage2_source
    stage12_source = inspect.getsource(whole_moe_module._packed_stage12_unchecked)
    assert "_row_owned_stage1_unchecked" in stage12_source
    assert "_packed_stage2_unchecked" in stage12_source

    source = inspect.getsource(whole_moe_module._target_m2_route)
    assert "bind_target_m2(binding)" in source
    assert "_packed_stage12_unchecked" not in source

    fixed_source = inspect.getsource(kernel_module.bind_target_m2)
    assert "_build_target_m2_stage1_projection_kernel()" in fixed_source
    assert "_build_target_m2_stage1_finalizer_kernel()" in fixed_source
    assert "_build_target_m2_stage2_kernel()" in fixed_source
    # Stage 3 runs row-split (one threadgroup per row, bit-identical to the
    # paired kernel; the install self-check's M1/M2 row parity holds it at 0).
    assert "_build_target_m2_stage3_rowsplit_kernel()" in fixed_source
    assert "_launch_target_m2_stage1(" in fixed_source
    assert "_launch_target_stage1(" not in fixed_source
    assert "_row_owned_stage1_unchecked" not in fixed_source
    assert "_packed_stage12_unchecked" not in fixed_source
    assert "_launch_target_stage2(" in fixed_source
    assert "_launch_target_m2_stage3_rowsplit(" in fixed_source
    assert "output.reshape(1, 2, 2048)" in fixed_source
    call_start = fixed_source.index("def call(")
    for forbidden in (
        "os.environ",
        "selfcheck",
        "installed",
        "eligible",
        "fallback",
        "lane_disabled",
        "try:",
        "except",
        "value.shape",
    ):
        assert forbidden not in fixed_source[call_start:]


def test_successful_selfcheck_installs_only_40_target_m2_overrides(monkeypatch):
    monkeypatch.setenv("MTPLX_A3B_WHOLE_MOE_FUSION", "1")
    model, targets, mtp = _exact_model()
    original_classes = tuple(type(block) for block in (*targets, *mtp))
    plan = prepare_a3b_whole_moe(model, config=_exact_config())
    assert tuple(type(block) for block in (*targets, *mtp)) == original_classes

    report = install_a3b_whole_moe(
        plan,
        _exact_selfcheck_report(),
        compiled_preflight=_passing_full_graph_preflight,
    )

    assert report["installation_status"] == "installed"
    assert report["target_blocks"] == 40
    assert report["mtp_blocks"] == 0
    assert report["accepted_mtp_blocks"] == 1
    assert report["validated_contract"]["target"]["routed_gate_up"] == (
        "affine_q4_group64_[256,1024,256]"
    )
    assert report["validated_contract"]["mtp"]["shared_gate_up"] == (
        "dense_bf16_[1024,2048]"
    )
    assert report["validated_contract"]["target"]["routes"] == {
        "M1": "m2_row_arithmetic_at_rows1_stage23_row_parity",
        "M2": "fixed_tiled_stage1_stage23_one_read_row_paired",
    }
    assert report["validated_contract"]["mtp"]["routes"] == {
        "M1": "accepted_row_owned_router_combine"
    }
    assert report["selfcheck_lanes"] == {
        "a3b_whole_moe_target_m2": "ok",
        "a3b_whole_moe_target_m1_m2_row_parity": "ok",
        **_passing_full_graph_preflight(),
    }
    assert all(
        type(block).__call__ is whole_moe_module._target_a3b_whole_moe_call
        for block in targets
    )
    assert type(mtp[0]) is original_classes[-1]


def test_selfcheck_failure_prevents_every_installation(monkeypatch):
    monkeypatch.setenv("MTPLX_A3B_WHOLE_MOE_FUSION", "1")
    model, targets, mtp = _exact_model()
    original_classes = tuple(type(block) for block in (*targets, *mtp))
    plan = prepare_a3b_whole_moe(model, config=_exact_config())

    with pytest.raises(A3BWholeMoeConfigError, match="self-check"):
        install_a3b_whole_moe(
            plan,
            {"lanes": {"a3b_whole_moe_target_m2": "fallback"}},
            compiled_preflight=_passing_full_graph_preflight,
        )

    assert tuple(type(block) for block in (*targets, *mtp)) == original_classes


def test_installed_route_delegates_everything_except_target_m2(monkeypatch):
    monkeypatch.setenv("MTPLX_A3B_WHOLE_MOE_FUSION", "1")
    _patch_whole_moe_stages(monkeypatch)
    phase = {"value": "prefill"}
    monkeypatch.setattr(
        whole_moe_module,
        "current_attention_phase",
        lambda: phase["value"],
    )
    model, targets, mtp = _exact_model()
    plan = prepare_a3b_whole_moe(model, config=_exact_config())

    class AcceptedTarget(type(targets[0])):
        def __call__(self, value):
            self.stock_calls.append(value)
            return _ResultSpec("accepted_target", value.shape)

    class AcceptedMTP(type(mtp[0])):
        def __call__(self, value):
            self.stock_calls.append(value)
            return _ResultSpec("accepted_mtp", value.shape)

    for target in targets:
        target.__class__ = AcceptedTarget
    mtp[0].__class__ = AcceptedMTP
    install_a3b_whole_moe(
        plan,
        _exact_selfcheck_report(),
        compiled_preflight=_passing_full_graph_preflight,
    )

    assert targets[0](_ArraySpec((1, 64, 2048))).label == "accepted_target"
    phase["value"] = "ar_decode"
    # Single decode rows now run the fused M1 route: the consistent model
    # function is what makes greedy K1 byte-comparable to greedy generate_ar.
    assert targets[0](_ArraySpec((1, 1, 2048))).label == "target_m1"
    assert mtp[0](_ArraySpec((1, 1, 2048))).label == "accepted_mtp"
    phase["value"] = "unknown"
    assert targets[0](_ArraySpec((1, 1, 2048))).label == "accepted_target"
    assert mtp[0](_ArraySpec((1, 1, 2048))).label == "accepted_mtp"
    phase["value"] = "decode_verify"
    assert targets[0](_ArraySpec((1, 2, 2048))).label == "target_m2"
    assert targets[0](_ArraySpec((1, 1, 2048))).label == "target_m1"
    # 3-row verify [primary, d1, d2] now routes to the k=2 fused M3 kernel
    # (byte-exact per row to the M1 route) instead of delegating to stock.
    assert targets[0](_ArraySpec((1, 3, 2048))).label == "target_m3"
    assert targets[0](_ArraySpec((1, 4, 2048))).label == "accepted_target"
    assert len(targets[0].stock_calls) == 3
    assert len(mtp[0].stock_calls) == 2


def test_compiled_full_graph_failure_rolls_back_every_class(monkeypatch):
    monkeypatch.setenv("MTPLX_A3B_WHOLE_MOE_FUSION", "1")
    _patch_whole_moe_stages(monkeypatch)
    model, targets, mtp = _exact_model()
    original_target_classes = tuple(type(block) for block in targets)
    original_mtp_class = type(mtp[0])
    plan = prepare_a3b_whole_moe(model, config=_exact_config())

    def fail_after_swap():
        assert all(
            type(block) is not original
            for block, original in zip(targets, original_target_classes)
        )
        assert type(mtp[0]) is original_mtp_class
        raise RuntimeError("full graph compile failed")

    with pytest.raises(A3BWholeMoeConfigError, match="full compiled target-prefix"):
        install_a3b_whole_moe(
            plan,
            _exact_selfcheck_report(),
            compiled_preflight=fail_after_swap,
        )

    assert tuple(type(block) for block in targets) == original_target_classes
    assert type(mtp[0]) is original_mtp_class


def test_installation_keeps_no_global_model_or_block_references():
    source = inspect.getsource(whole_moe_module)

    assert "_INSTALLED_BLOCKS" not in source
    assert "extend(changed)" not in source


def test_installed_hot_call_only_routes_on_phase_and_logical_m():
    source = inspect.getsource(whole_moe_module._target_a3b_whole_moe_call)
    assert "current_attention_phase" in source
    assert "value.shape" in source
    assert "route.accepted_call(self, value)" in source
    assert "_mtp_a3b_whole_moe_call" not in inspect.getsource(whole_moe_module)
    for forbidden in (
        "os.environ",
        ".dtype",
        "bits",
        "group_size",
        "eligible",
        "selfcheck",
        "installed",
        "installation_status",
        "_STATS",
        "fallback",
        "lane_disabled",
        "try:",
        "except",
        "switch_mlp",
        "shared_expert",
        "mx.softmax",
        "m2_call is not None",
        "stock_call",
    ):
        assert forbidden not in source


def test_runtime_constructs_one_whole_block_owner_after_packing() -> None:
    source = inspect.getsource(runtime_module.load)

    adapter_guard = source.index("validate_a3b_whole_moe_load_options(")
    packing = source.index("configure_moe_packed_projections(model)")
    prepare_whole = source.index("prepare_a3b_whole_moe(model, config=config)")
    prepare_router = source.index("prepare_qwen_row_owned_routers(")
    selfcheck = source.index("maybe_run_model_selfcheck(model)")
    whole_selfcheck = source.index("run_a3b_whole_moe_selfcheck(")
    compiled_factory = source.index("prepare_a3b_compiled_target_prefix(")
    # PR #195 made construction class-indirect (LagunaARRuntime | MTPLXRuntime);
    # the ordering invariant this test guards is unchanged.
    runtime_construction = source.index("runtime = runtime_class(")
    install_whole = source.index("install_a3b_whole_moe(")
    compiled_preflight = source.index("preflight_a3b_k1_target_prefix_load_graph(")
    install_router = source.index("install_qwen_row_owned_routers(")

    assert adapter_guard < packing < prepare_whole < prepare_router < selfcheck
    assert selfcheck < install_router < whole_selfcheck
    assert whole_selfcheck < compiled_factory < runtime_construction
    assert runtime_construction < install_whole < compiled_preflight
    assert prepare_router < selfcheck < install_router < compiled_factory
    assert "if whole_moe_plan is not None" in source
    assert "elif router_plan is not None" not in source


def test_full_graph_preflight_executes_both_compiled_target_geometries():
    source = inspect.getsource(
        compiled_target_module.preflight_a3b_k1_target_prefix_full_graph
    )

    assert "cache: list[Any]" in source
    assert "prompt_tokens: int" in source
    assert "max_tokens: int" in source
    assert "hidden_variant: str | None" in source
    assert "runtime.make_cache" not in source
    assert "runtime.forward_ar" not in source
    assert "install_a3b_k1_target_prefix_route(" in source
    assert "route.compiled_m2(" in source
    assert "route.compiled_m1(" in source
    assert "route.verify_m2(" not in source
    assert "route.repair_m1(" not in source
    assert "len(m2_outputs) != 182" in source
    assert "len(m1_outputs) != 92" in source
    assert source.count("mx.eval(") >= 2
    assert "mx.random" not in source


def test_load_graph_preflight_is_only_the_minimum_installation_compatibility_probe():
    source = inspect.getsource(
        compiled_target_module.preflight_a3b_k1_target_prefix_load_graph
    )

    assert "_preflight_a3b_k1_target_prefix_request_geometry(" in source
    assert "prompt_tokens=1" in source
    assert "max_tokens=2" in source


def test_request_preflight_synthesizes_exact_geometry_and_memoizes_by_shape(
    monkeypatch,
):
    factory = object()
    calls = []
    runtime = SimpleNamespace(
        a3b_whole_moe_installed=True,
        a3b_compiled_target_prefix_factory=factory,
        _a3b_whole_moe_request_preflights={},
        _a3b_whole_moe_request_geometry_keys={},
    )

    def fake_fresh_geometry(
        rt,
        selected_factory,
        *,
        prompt_tokens: int,
        max_tokens: int,
        hidden_variant: str | None,
        cache_factory,
        prefill_layout: str,
    ):
        calls.append(
            (
                "fresh_geometry",
                rt,
                selected_factory,
                prompt_tokens,
                max_tokens,
                hidden_variant,
                prefill_layout,
            )
        )
        return {
            "canonical_key": "a" * 64,
            "full_attention_key_shape": [1, 2, 256, 256],
            "full_attention_value_shape": [1, 2, 256, 256],
            "hidden_variant": hidden_variant,
            "lanes": {
                "a3b_whole_moe_request_full_graph_m1": "ok",
                "a3b_whole_moe_request_full_graph_m2": "ok",
            },
        }

    monkeypatch.setattr(
        compiled_target_module,
        "_preflight_a3b_k1_target_prefix_request_geometry",
        fake_fresh_geometry,
    )

    first = compiled_target_module.ensure_a3b_whole_moe_request_preflight(
        runtime,
        factory,
        prompt_tokens=181,
        max_tokens=64,
        hidden_variant="post_norm",
        cache_factory=lambda: None,
        prefill_layout="contiguous_dense_decode",
    )
    second = compiled_target_module.ensure_a3b_whole_moe_request_preflight(
        runtime,
        factory,
        prompt_tokens=180,
        max_tokens=65,
        hidden_variant="post_norm",
        cache_factory=lambda: None,
        prefill_layout="contiguous_dense_decode",
    )

    assert calls == [
        (
            "fresh_geometry",
            runtime,
            factory,
            181,
            64,
            "post_norm",
            "contiguous_dense_decode",
        ),
    ]
    assert first["status"] == second["status"] == "ok"
    assert (first["prompt_tokens"], first["max_tokens"]) == (181, 64)
    assert (second["prompt_tokens"], second["max_tokens"]) == (180, 65)
    assert first["full_attention_key_shape"] == second[
        "full_attention_key_shape"
    ]


def test_fresh_request_preflight_builds_shape_state_without_duplicate_prompt_prefill():
    source = inspect.getsource(
        compiled_target_module._preflight_a3b_k1_target_prefix_request_geometry
    )

    assert "cache_factory()" in source
    assert "runtime.make_cache()" not in source
    assert "mx.array([[0]])" in source
    assert "entry.offset = int(prompt_tokens)" in source
    assert "preflight_a3b_k1_target_prefix_full_graph(" in source
    assert "prompt_ids" not in source
    assert "restore_or_prefill_prompt_state" not in source
    assert "_prefill(" not in source


def test_disabled_whole_moe_request_preflight_constructs_nothing(monkeypatch):
    runtime = SimpleNamespace(a3b_whole_moe_installed=False)

    def unexpected(*args, **kwargs):
        raise AssertionError((args, kwargs))

    monkeypatch.setattr(
        generation_module,
        "_ensure_a3b_whole_moe_request_preflight",
        unexpected,
    )

    assert generation_module.ensure_a3b_whole_moe_request_preflight(
        runtime,
        [1],
        max_tokens=8,
        base_hidden_variant="post_norm",
    ) == {"status": "disabled"}


def test_compiled_target_cache_does_not_pin_the_runtime():
    for helper in (
        compiled_target_module._shared_m1_step,
        compiled_target_module._shared_m2_step,
    ):
        source = inspect.getsource(helper)
        assert '"runtime": runtime' not in source
        assert '"runtime_ref": weakref.ref(runtime)' in source
    for builder in (
        compiled_target_module._make_a3b_k1_target_prefix_m1_step,
        compiled_target_module._make_a3b_k1_target_prefix_m2_step,
    ):
        assert 'host["runtime_ref"]()' in inspect.getsource(builder)


def test_non_k1_generation_entrypoints_reject_installed_whole_moe_before_prefill():
    for entrypoint in (generation_module.generate_ar, generation_module.generate_mtp1):
        source = inspect.getsource(entrypoint)
        rejection = source.index("reject_non_k1_a3b_whole_moe_request(")
        prefill = min(
            position
            for marker in ("_prefill(", "restore_or_prefill_prompt_state(")
            if (position := source.find(marker)) >= 0
        )
        assert rejection < prefill


def test_server_shared_prefix_is_explicitly_constructed_as_prefill():
    server_source = (
        Path(__file__).parents[1] / "mtplx" / "server" / "openai.py"
    ).read_text()

    assert 'with attention_phase("ar_batch_shared_prefill")' not in server_source
    assert 'with attention_phase("prefill")' in server_source


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"verify_strategy": "capture_commit"}, "target-prefix"),
        ({"requested_speculative_depth": 2}, "K1"),
        ({"speculative_depth": 0}, "K1"),
        ({"verify_core": "linear-gdn-from-conv-tape"}, "stock capture"),
        ({"draft_core": "device-d2"}, "stock draft"),
        ({"compiled_target_prefix": False}, "compiled target-prefix"),
        ({"session_bank_present": True}, "cold prompt"),
        ({"vision_splice_present": True}, "vision"),
        ({"prefill_layout": "contiguous_then_repage"}, "contiguous dense"),
    ],
)
def test_request_mismatch_fails_before_generation(override, match):
    request = {
        "verify_strategy": "target_prefix",
        "requested_speculative_depth": 1,
        "speculative_depth": 1,
        "verify_core": "stock",
        "draft_core": "stock",
        "compiled_target_prefix": True,
        "session_bank_present": False,
        "vision_splice_present": False,
        "prefill_layout": "contiguous_dense_decode",
    }
    request.update(override)

    with pytest.raises(A3BWholeMoeConfigError, match=match):
        validate_a3b_whole_moe_request(**request)


@pytest.mark.parametrize(
    ("mtp_adapter", "merge_mtp_adapter"),
    [(Path("adapter"), False), (None, True)],
)
def test_whole_moe_rejects_mtp_adapter_configuration_at_load_boundary(
    monkeypatch,
    mtp_adapter,
    merge_mtp_adapter,
):
    monkeypatch.setenv("MTPLX_A3B_WHOLE_MOE_FUSION", "1")

    with pytest.raises(A3BWholeMoeConfigError, match="MTP adapters"):
        whole_moe_module.validate_a3b_whole_moe_load_options(
            mtp_adapter=mtp_adapter,
            merge_mtp_adapter=merge_mtp_adapter,
        )


def test_generation_validates_whole_moe_request_before_prefill():
    source = inspect.getsource(generation_module.generate_mtpk)

    validation = source.index("validate_a3b_whole_moe_request(")
    request_preflight = source.index("ensure_a3b_whole_moe_request_preflight(")
    counter_start = source.index("counter_start = _runtime_counter_snapshot(rt)")
    prefill = source.index("restore_or_prefill_prompt_state(")
    assert validation < request_preflight < counter_start < prefill
    assert 'getattr(rt, "a3b_whole_moe_installed", False)' in source


def test_actual_request_route_requires_the_preflighted_leaf_signature_before_decode():
    install_source = inspect.getsource(
        compiled_target_module.install_a3b_k1_target_prefix_route
    )
    generation_source = inspect.getsource(generation_module.generate_mtpk)

    assert "_route_compile_specialization_key(" in install_source
    assert "runtime._a3b_whole_moe_request_preflights" in install_source
    assert 'route.request_preflight_status = "matched"' in install_source
    route_install = generation_source.index("install_a3b_k1_target_prefix_route(")
    decode_loop = generation_source.index("while len(tokens) < max_tokens:")
    assert route_install < decode_loop
    assert '"request_preflight_key": self.request_preflight_key' in inspect.getsource(
        compiled_target_module.A3BK1TargetPrefixRoute.final_report
    )


def test_mtp_history_phase_is_constructed_by_prompt_and_decode_callers():
    helper = inspect.getsource(generation_module._append_mtp_history)
    assert 'phase: Literal["prefill", "ar_decode"]' in helper
    assert "with attention_phase(phase)" in helper
    assert 'attention_phase("prefill")' not in helper

    for prompt_owner in (
        generation_module._prefill_restored_prompt_suffix,
        generation_module.restore_or_prefill_prompt_state,
        generation_module._prefill_committed_mtp_history_streaming,
    ):
        source = inspect.getsource(prompt_owner)
        call = source.index("_append_mtp_history(")
        next_call = source.find("_append_mtp_history(", call + 1)
        owned = source[call : None if next_call < 0 else next_call]
        assert 'phase="prefill"' in owned

    generation = inspect.getsource(generation_module.generate_mtpk)
    nested_start = generation.index("def append_mtp_history(")
    nested_end = generation.index("def maybe_eval_state_roots(", nested_start)
    nested = generation[nested_start:nested_end]
    assert "_append_mtp_history(" in nested
    assert 'phase="ar_decode"' in nested


def test_runtime_contract_propagates_only_the_whole_moe_enable_flag() -> None:
    from mtplx.profiles import normalize_runtime_env_overrides

    assert normalize_runtime_env_overrides(
        {"MTPLX_A3B_WHOLE_MOE_FUSION": True}
    ) == {"MTPLX_A3B_WHOLE_MOE_FUSION": "1"}
