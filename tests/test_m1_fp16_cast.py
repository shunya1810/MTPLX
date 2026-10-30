"""MTPLX_M1_FP16_CAST: which loads cast a bf16 pack to fp16, and what the cast touches."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest

from mtplx import cache_state, runtime

QWEN4 = {"model_type": "qwen4_exp", "text_config": {"model_type": "qwen4_exp_text"}}
QWEN38 = {"model_type": "qwen3_5", "text_config": {"model_type": "qwen3_5_text"}}


@pytest.fixture()
def m1(monkeypatch):
    monkeypatch.delenv("MTPLX_M1_FP16_CAST", raising=False)
    monkeypatch.setattr(cache_state, "m1_long_context_defaults", lambda: True)


def test_qwen4_family_on_m1_casts_by_default(m1):
    assert runtime._m1_fp16_cast_requested(QWEN4)
    assert runtime._m1_fp16_cast_requested({"model_type": "qwen4_exp_text"})


def test_other_families_stay_opt_in(m1):
    assert not runtime._m1_fp16_cast_requested(QWEN38)
    assert not runtime._m1_fp16_cast_requested(None)


def test_off_the_m1_family_nothing_casts(monkeypatch):
    monkeypatch.delenv("MTPLX_M1_FP16_CAST", raising=False)
    monkeypatch.setattr(cache_state, "m1_long_context_defaults", lambda: False)
    assert not runtime._m1_fp16_cast_requested(QWEN4)


def test_the_env_wins_both_ways(m1, monkeypatch):
    monkeypatch.setenv("MTPLX_M1_FP16_CAST", "0")
    assert not runtime._m1_fp16_cast_requested(QWEN4)
    monkeypatch.setenv("MTPLX_M1_FP16_CAST", "1")
    assert runtime._m1_fp16_cast_requested(QWEN38)


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.QuantizedLinear(64, 32, group_size=32, bits=4)
        self.norm = nn.RMSNorm(64)
        self.A_log = mx.zeros((4,))
        self.ngram_table = mx.zeros((8, 4))

    @property
    def cast_predicate(self):
        return lambda path: "ngram" not in path and not path.endswith("A_log")


def test_the_cast_skips_what_the_model_keeps():
    model = _Tiny()
    model.update(
        {
            "proj": {
                "scales": model.proj.scales.astype(mx.bfloat16),
                "biases": model.proj.biases.astype(mx.bfloat16),
            },
            "norm": {"weight": model.norm.weight.astype(mx.bfloat16)},
            "A_log": model.A_log.astype(mx.bfloat16),
            "ngram_table": model.ngram_table.astype(mx.bfloat16),
        }
    )
    report = runtime._cast_bf16_parameters_to_fp16(model)
    assert report["casted"] == 3 and report["kept_bf16"] == 2
    assert model.proj.scales.dtype == mx.float16
    assert model.proj.biases.dtype == mx.float16
    assert model.norm.weight.dtype == mx.float16
    assert model.proj.weight.dtype == mx.uint32
    assert model.A_log.dtype == mx.bfloat16
    assert model.ngram_table.dtype == mx.bfloat16
