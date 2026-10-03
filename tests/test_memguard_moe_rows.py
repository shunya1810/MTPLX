"""MoE packs are priced by their routed dimensions (item c of the review of
9c96dd9c): ``_forward_row_bytes`` read only ``intermediate_size``, a field an
MoE pack's expert layers do not use, so a row's routed intermediates (every
selected expert's gate, up and product, their outputs, the router's scores
over every expert, the shared experts) went unpriced."""

from __future__ import annotations

from types import SimpleNamespace

import mtplx.server.openai as srv


def _row(**fields):
    return srv._forward_row_bytes(SimpleNamespace(**fields))


def test_a_qwen3_next_style_moe_row():
    # hidden 2,048, 512 experts, 10 per token at 512, one 512 shared expert.
    fields = dict(
        hidden_size=2048,
        num_attention_heads=16,
        num_key_value_heads=2,
        head_dim=256,
        intermediate_size=5120,
        num_experts=512,
        num_experts_per_tok=10,
        moe_intermediate_size=512,
        shared_expert_intermediate_size=512,
    )
    routed = 2 * (2048 + 10 * (3 * 512 + 2048) + 3 * 512) + 4 * 3 * 512
    assert routed == 84_992
    assert _row(**fields) == routed
    # The dense MLP term alone said 34,816.
    assert routed > 2 * (2048 + 3 * 5120)


def test_deepseek_names_and_shared_experts_by_count():
    fields = dict(
        hidden_size=7168,
        num_attention_heads=128,
        num_key_value_heads=128,
        head_dim=128,
        n_routed_experts=256,
        num_experts_per_tok=8,
        moe_intermediate_size=2048,
        n_shared_experts=1,
    )
    routed = 2 * (7168 + 8 * (3 * 2048 + 7168) + 3 * 2048) + 4 * 3 * 256
    assert _row(**fields) == routed == 242_688


def test_a_dense_config_is_unchanged():
    from tests.test_memguard_admission import _q27_text_args

    assert srv._forward_row_bytes(_q27_text_args()) == 139_264
