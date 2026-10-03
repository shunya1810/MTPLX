"""Fused routed experts in a Qwen3.5/3.6 MoE MTP head (#574).

Official checkpoints store the head's routed experts as two fused tensors,
in the hub Linear layout or the transformers bmm layout. Both MTP loaders
(the generic injector and the dedicated ``qwen3_5_mtp`` one) map them onto the
``switch_mlp`` leaves the module loads, and the head then computes exactly
what the same head saved as numbered experts computes: the same leaves, draft
logits, hidden state and cache, over a prompt append and the next step.

The tiny config is the square case (hidden 128, expert width 64, so a Linear
gate_up ``[E, 128, 128]`` has the shape of a bmm one); the block's down
tensor decides it.
"""

from __future__ import annotations

import json

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten

from mtplx.expert_layout import split_fused_experts
from mtplx.mtp_patch import inject_mtp_support
from mtplx.qwen3_5_mtp_patch import _make_qwen3_5_mtp_module, inject_qwen3_5_mtp_support
from test_qwen3_5_mtp_object_level import _tiny_text_config

_SWITCH = "layers.0.mlp.switch_mlp"


def _bits_equal(a, b) -> bool:
    view = {2: mx.uint16, 4: mx.uint32}[a.dtype.size]
    return a.shape == b.shape and bool(mx.array_equal(a.view(view), b.view(view)).item())


def _write_head(root, config, tensors) -> None:
    root.mkdir()
    (root / "config.json").write_text(json.dumps(config))
    mx.save_safetensors(str(root / "mtp.safetensors"), {"mtp." + k: v for k, v in tensors.items()})


def _heads(tmp_path, *, layout, inter, model_type):
    """A trunk plus one random head saved fused (``layout``) and numbered."""
    from mlx_lm.models.qwen3_5 import TextModel, TextModelArgs

    mx.random.seed(574)
    text = _tiny_text_config()
    text["moe_intermediate_size"] = inter
    args = TextModelArgs.from_dict(text)
    config = {"model_type": model_type, "text_config": text, "num_nextn_predict_layers": 1}
    head = dict(tree_flatten(_make_qwen3_5_mtp_module(args).parameters()))
    mx.eval(list(head.values()))

    fused = {k: v for k, v in head.items() if not k.startswith(_SWITCH + ".")}
    gate_up = mx.concatenate(
        [head[f"{_SWITCH}.gate_proj.weight"], head[f"{_SWITCH}.up_proj.weight"]], axis=1
    )
    down = head[f"{_SWITCH}.down_proj.weight"]
    if layout == "bmm":
        gate_up, down = gate_up.swapaxes(1, 2), down.swapaxes(1, 2)
    fused["layers.0.mlp.experts.gate_up_proj"] = mx.contiguous(gate_up)
    fused["layers.0.mlp.experts.down_proj"] = mx.contiguous(down)
    _write_head(tmp_path / "fused", config, fused)

    numbered = {k: v for k, v in head.items() if not k.startswith(_SWITCH + ".")}
    for name in ("gate_proj", "up_proj", "down_proj"):
        stacked = head[f"{_SWITCH}.{name}.weight"]
        for expert in range(args.num_experts):
            numbered[f"layers.0.mlp.experts.{expert}.{name}.weight"] = stacked[expert]
    _write_head(tmp_path / "numbered", config, numbered)

    trunk = TextModel(args)
    twin = TextModel(args)
    twin.load_weights(tree_flatten(trunk.parameters()))
    return trunk, twin, config, head, args


@pytest.mark.parametrize("model_type", ["qwen3_5_moe", "qwen3_5_mtp"])
@pytest.mark.parametrize("layout", ["linear", "bmm"])
def test_fused_head_computes_what_the_numbered_head_computes(tmp_path, layout, model_type):
    inject = inject_qwen3_5_mtp_support if model_type == "qwen3_5_mtp" else inject_mtp_support
    fused_model, numbered_model, config, head, args = _heads(
        tmp_path, layout=layout, inter=64, model_type=model_type
    )
    assert inject(fused_model, tmp_path / "fused", config)
    assert inject(numbered_model, tmp_path / "numbered", config)

    loaded = dict(tree_flatten(fused_model.mtp.parameters()))
    for key, value in head.items():
        assert _bits_equal(loaded[key], value), key

    fused_cache = fused_model.make_mtp_cache()
    numbered_cache = numbered_model.make_mtp_cache()
    for ids in ([[1, 2, 3]], [[4]]):
        hidden = mx.random.normal((1, len(ids[0]), args.hidden_size))
        tokens = mx.array(ids)
        logits_a, hidden_a = fused_model.mtp_forward(
            hidden, tokens, mtp_cache=fused_cache, return_hidden=True
        )
        logits_b, hidden_b = numbered_model.mtp_forward(
            hidden, tokens, mtp_cache=numbered_cache, return_hidden=True
        )
        mx.eval(logits_a, logits_b, hidden_a, hidden_b)
        assert _bits_equal(logits_a, logits_b)
        assert _bits_equal(hidden_a, hidden_b)
        for a, b in zip(fused_cache[0].state, numbered_cache[0].state):
            assert _bits_equal(a, b)


def _fused(gate_up_shape, down_shape):
    weights = {}
    if gate_up_shape is not None:
        weights["layers.0.mlp.experts.gate_up_proj"] = mx.zeros(gate_up_shape)
    if down_shape is not None:
        weights["layers.0.mlp.experts.down_proj"] = mx.zeros(down_shape)
    return weights


@pytest.mark.parametrize("layout", ["linear", "bmm"])
@pytest.mark.parametrize("inter", [64, 128])  # square gate_up, square down
def test_a_square_tensor_takes_its_blocks_layout(layout, inter):
    mx.random.seed(inter)
    gate, up = mx.random.normal((4, inter, 128)), mx.random.normal((4, inter, 128))
    down = mx.random.normal((4, 128, inter))
    gate_up = mx.concatenate([gate, up], axis=1)
    if layout == "bmm":
        gate_up, down_fused = gate_up.swapaxes(1, 2), down.swapaxes(1, 2)
    else:
        down_fused = down
    weights = {
        "layers.0.mlp.experts.gate_up_proj": mx.contiguous(gate_up),
        "layers.0.mlp.experts.down_proj": mx.contiguous(down_fused),
    }
    mapped = split_fused_experts(weights, hidden_size=128)
    assert _bits_equal(mapped[f"{_SWITCH}.gate_proj.weight"], gate)
    assert _bits_equal(mapped[f"{_SWITCH}.up_proj.weight"], up)
    assert _bits_equal(mapped[f"{_SWITCH}.down_proj.weight"], down)


@pytest.mark.parametrize(
    "gate_up_shape, down_shape",
    [
        ((4, 128, 128), None),  # square and alone: either layout fits
        ((4, 256, 128), (4, 64, 128)),  # Linear gate_up beside a bmm down
        ((4, 96, 96), (4, 96, 48)),  # no dimension is the hidden size
    ],
)
def test_a_layout_the_block_cannot_name_is_refused(gate_up_shape, down_shape):
    with pytest.raises(ValueError, match="cannot tell the fused expert layout"):
        split_fused_experts(_fused(gate_up_shape, down_shape), hidden_size=128)
