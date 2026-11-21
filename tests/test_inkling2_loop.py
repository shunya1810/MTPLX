"""Inkling 2 Mini (inkling2_loop) trunk: tiny random model checks.

* the cached decode matches the full forward (loop passes keep their caches),
* an official-layout checkpoint (NVFP4 experts with an outer scale, MXFP8
  experts, BF16 projections) loads through ``sanitize`` and matches a model
  holding the dequantized expert weights,
* float16 activations survive the raw NVFP4 product (it runs wider).
"""

import os
import re
import unittest

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from mtplx.models import inkling2_loop as ink

CONFIG = {
    "model_type": "inkling2_loop",
    "text_config": {
        "hidden_size": 64,
        "num_hidden_layers": 8,
        "vocab_size": 128,
        "unpadded_vocab_size": 120,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "swa_num_attention_heads": 4,
        "swa_num_key_value_heads": 2,
        "swa_head_dim": 16,
        "sliding_window_size": 8,
        "local_layer_ids": [0, 1, 2, 3, 4, 6],
        "d_rel": 4,
        "rel_extent": 8,
        "log_scaling_n_floor": 4,
        "log_scaling_alpha": 0.1,
        "rms_norm_eps": 1e-6,
        "qk_l2_clip_threshold": 3.0,
        "rel_l2_clip_threshold": 2.0,
        "sconv_kernel_size": 4,
        "n_layers_per_attn_res_block": 3,
        "n_routed_experts": 4,
        "num_experts_per_tok": 2,
        "intermediate_size": 64,
        "mlp_gate_softcap": 4.0,
        "mlp_up_softcap": 25.0,
        "route_scale": 2.0,
        "final_logit_softcapping": 30.0,
        "loop_config": {
            "core_start": 3,
            "core_end": 6,
            "num_passes": 3,
            "gate_rank": 8,
            "is_varloop": False,
        },
    },
}

_NAMES = [
    (r"^lm_head\.", "model.llm.unembed."),
    (r"^model\.language_model\.embed_tokens\.", "model.llm.embed."),
    (r"^model\.language_model\.", "model.llm."),
    (r"\.final_input\.query$", ".attn_res.output_query"),
    (r"\.layers\.(\d+)\.mlp_input\.query$", r".attn_res.mlp_queries.\1"),
    (r"\.self_attn\.q_proj\.", ".attn.wq_du."),
    (r"\.self_attn\.k_proj\.", ".attn.wk_dv."),
    (r"\.self_attn\.v_proj\.", ".attn.wv_dv."),
    (r"\.self_attn\.r_proj\.", ".attn.wr_du."),
    (r"\.self_attn\.o_proj\.", ".attn.wo_ud."),
    (r"\.self_attn\.gate_proj\.", ".attn.gate_attn."),
    (r"\.self_attn\.sink_proj\.", ".attn.attn_sink_proj."),
    (r"\.self_attn\.", ".attn."),
    (r"\.q_message_gate\.", ".controllers.attn."),
    (r"\.(k|v|mlp)_message_gate\.", r".controllers.\1."),
    (r"(\.controllers\.\w+)\.down_proj\.weight$", r"\1.down_weight"),
    (r"(\.controllers\.\w+)\.forget_gate_proj\.weight$", r"\1.forget_weight"),
    (r"(\.controllers\.\w+)\.message_gate_proj\.weight$", r"\1.gain_weight"),
    (r"\.post_attention_convolution\.", ".attn_sconv."),
    (r"\.post_mlp_convolution\.", ".mlp_sconv."),
    (r"\.conv1d\.weight$", ".weight"),
    (r"sconv\.static_weight$", "sconv.static_w"),
    (r"sconv\.weight_proj\.weight$", "sconv.weight_proj"),
    (r"\.post_attention_layernorm\.", ".mlp_norm."),
    (r"\.input_layernorm\.", ".attn_norm."),
    (r"\.(q|k|v)_input_layernorm\.", r".\1_input_norm."),
    (r"\.mlp\.gate\.e_score_correction_bias$", ".mlp.gate.bias"),
]


def _model(seed=0):
    mx.random.seed(seed)
    model = ink.Model(ink.ModelArgs.from_dict(CONFIG))
    weights = []
    for k, v in tree_flatten(model.parameters()):
        if k.endswith(("norm.weight", "l2_clip.scale")):
            v = 1 + 0.1 * mx.random.normal(v.shape)
        else:
            fan_in = v.shape[-1] if v.ndim > 1 else 16
            v = mx.random.normal(v.shape) * fan_in**-0.5
        weights.append((k, v))
    model.update(tree_unflatten(weights))
    mx.eval(model.parameters())
    return model


def _checkpoint(model):
    """Official layout: experts of layer 7 MXFP8, the rest NVFP4 with an outer
    scale; returns (checkpoint, dequantized expert weights by model path)."""

    ckpt, dequant = {}, {}
    denominator = mx.array([2688.0])
    for k, v in tree_flatten(model.parameters()):
        if k.endswith(".message_input.query"):
            q, kv = mx.concatenate([v[:, :1], v[:, 3:]], axis=1), v[:, 1:3]
            ckpt["model.llm.middle_cycle_cross_pass_queries"] = q.flatten(0, 1)
            ckpt["model.llm.middle_cycle_cross_pass_kv_queries"] = kv.flatten(0, 1)
            continue
        if m := re.match(r"(.+\.mlp\.experts)\.(gate_up|down)_proj\.weight$", k):
            prefix, name = m.groups()
            w = prefix.replace("model.language_model.", "model.llm.")
            w = f"{w}.{'w13' if name == 'gate_up' else 'w2'}_weight"
            if ".layers.7." in k:
                wq, s = mx.quantize(v, group_size=32, bits=8, mode="mxfp8")[:2]
                dequant[k] = mx.dequantize(wq, s, group_size=32, bits=8, mode="mxfp8")
                ckpt[w] = wq.view(mx.uint8)
                ckpt[f"{w}_scale"] = s
            else:
                amax = mx.abs(v).max(axis=(1, 2))
                fac = amax / 2688.0
                qs = [mx.quantize(v[e] / fac[e], mode="nvfp4")[:2] for e in range(v.shape[0])]
                wq = mx.stack([q for q, _ in qs])
                s = mx.stack([s for _, s in qs])
                dequant[k] = mx.stack(
                    [mx.dequantize(q, s, mode="nvfp4") * fac[e] for e, (q, s) in enumerate(qs)]
                )
                ckpt[w] = wq.view(mx.uint8)
                ckpt[f"{w}_scale"] = s
                ckpt[f"{w}_amax"] = amax
                ckpt[f"{w}_outer_scale_denominator"] = denominator
            continue
        if m := re.match(r".+\.layers\.(\d+)\.attention_input\.query$", k):
            ckpt[f"model.llm.attn_res.attn_queries.{int(m.group(1)) - 1}"] = v
            continue
        for pattern, replacement in _NAMES:
            k = re.sub(pattern, replacement, k)
        if k.endswith("sconv.weight"):
            v = v.swapaxes(1, 2)
        ckpt[k] = v
    return ckpt, dequant


class TestInkling2Loop(unittest.TestCase):
    def test_cached_decode_matches_forward(self):
        model = _model()
        tokens = mx.array([[5, 9, 17, 3, 44, 2, 101, 7, 19, 23, 61, 8]])
        full = model(tokens)
        cache = model.make_cache()
        self.assertEqual(len(cache), 3 + 3 * 3 + 2)
        steps = [model(tokens[:, :5], cache=cache)]
        for i in range(5, tokens.shape[1]):
            steps.append(model(tokens[:, i : i + 1], cache=cache))
        cached = mx.concatenate(steps, axis=1)
        self.assertTrue(mx.allclose(cached, full, rtol=1e-4, atol=1e-4).item())

    def _load(self, ckpt):
        loaded = ink.Model(ink.ModelArgs.from_dict(CONFIG))
        weights = loaded.sanitize(dict(ckpt))
        loaded.load_weights(list(weights.items()), strict=True)
        mx.eval(loaded.parameters())
        return loaded

    def test_official_checkpoint_loads(self):
        model = _model(1)
        ckpt, dequant = _checkpoint(model)
        old = {k: os.environ.get(k) for k in ("MTPLX_INKLING2_PROJ_BITS", "MTPLX_INKLING2_HEAD_BITS")}
        os.environ["MTPLX_INKLING2_PROJ_BITS"] = "0"
        os.environ["MTPLX_INKLING2_HEAD_BITS"] = "0"
        try:
            loaded = self._load(ckpt)
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        experts = loaded.model.language_model.layers[0].mlp.experts.gate_up_proj
        self.assertIsInstance(experts, ink.ExpertLinear)
        self.assertEqual(experts.mode, "nvfp4")
        self.assertIn("expert_scale", experts)
        last = loaded.model.language_model.layers[7].mlp.experts.down_proj
        self.assertEqual(last.mode, "mxfp8")
        self.assertNotIn("expert_scale", last)

        model.update(tree_unflatten(list(dequant.items())))
        tokens = mx.array([[5, 9, 17, 3, 44, 2, 101, 7]])
        self.assertTrue(mx.allclose(loaded(tokens), model(tokens), rtol=1e-3, atol=1e-3).item())

        # float16 activations: the raw NVFP4 product exceeds 65,504 here
        # (outer scale ~1e-4) unless it runs wider.
        fp16 = loaded.model.language_model.layers[0].mlp.experts.gate_up_proj
        x = mx.ones((1, 1, 1, 1, 64), mx.float16) * 30
        idx = mx.array([[[0, 1]]], dtype=mx.uint32)
        y = fp16(x, idx)
        self.assertEqual(y.dtype, mx.float16)
        self.assertTrue(mx.isfinite(y).all().item())

    def test_default_load_quantizes_projections(self):
        model = _model(2)
        ckpt, _ = _checkpoint(model)
        loaded = self._load(ckpt)
        attn = loaded.model.language_model.layers[0].self_attn
        self.assertIsInstance(attn.q_proj, nn.QuantizedLinear)
        self.assertEqual(attn.q_proj.bits, 8)
        self.assertIsInstance(loaded.lm_head, nn.QuantizedLinear)
        self.assertIsInstance(loaded.model.language_model.embed_tokens, nn.QuantizedEmbedding)
        # Small projections the loop reads per pass stay as shipped.
        self.assertIsInstance(attn.sink_proj, nn.Linear)
        tokens = mx.array([[5, 9, 17, 3]])
        self.assertTrue(mx.isfinite(loaded(tokens)).all().item())


if __name__ == "__main__":
    unittest.main()
