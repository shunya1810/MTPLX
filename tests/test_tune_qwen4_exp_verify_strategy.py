"""The tune candidate runner reads the pack config to pick the verify lane.

_depth_sweep_native60 hardcoded verify_strategy="capture_commit" and the
linear-gdn-from-conv-tape capture backend for every candidate. Those are
qwen3-next structure lanes: their capture stack introspects the qwen3-next
DecoderLayer layout (input_layernorm et al.) and raises AttributeError on
Flash-Next hyper-connection layers. The server already coerces qwen4_exp to
'batched' at boot; the candidate runner now applies the same predicate.
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest

from mtplx.commands.public import _model_config_is_qwen4_exp


def _write(tmp_path, config):
    (tmp_path / "config.json").write_text(json.dumps(config))
    return str(tmp_path)


def test_flash_next_pack_is_recognised_from_its_text_config(tmp_path):
    path = _write(tmp_path, {"model_type": "qwen4_exp", "text_config": {"model_type": "qwen4_exp_text"}})
    assert _model_config_is_qwen4_exp(path) is True


def test_top_level_model_type_alone_is_enough(tmp_path):
    path = _write(tmp_path, {"model_type": "qwen4_exp_text"})
    assert _model_config_is_qwen4_exp(path) is True


def test_dense_qwen3_5_pack_keeps_the_capture_commit_lane(tmp_path):
    path = _write(tmp_path, {"model_type": "qwen3_5", "text_config": {"model_type": "qwen3_5_text"}})
    assert _model_config_is_qwen4_exp(path) is False


def test_missing_or_unreadable_config_is_not_flash_next(tmp_path):
    assert _model_config_is_qwen4_exp(str(tmp_path)) is False
    (tmp_path / "config.json").write_text("{not json")
    assert _model_config_is_qwen4_exp(str(tmp_path)) is False


@pytest.mark.parametrize("exports", [
    {},
    {"MTPLX_QWEN4_FIXED_M4_VERIFY": "0", "MTPLX_FRSPEC_DRAFT": "0"},
    {"MTPLX_COMPILED_GDN": "0", "MTPLX_SKIP_VERIFY_SNAPSHOT": "1"},
])
def test_tune_uses_serve_contract_without_manual_family_exports(tmp_path, monkeypatch, exports):
    from mtplx.commands import public
    from mtplx.profiles import apply_profile_env, get_profile
    from mtplx.server import openai

    for key in list(os.environ):
        if key.startswith("MTPLX_"):
            monkeypatch.delenv(key)
    for key, value in exports.items():
        monkeypatch.setenv(key, value)
    model = _write(tmp_path, {"model_type": "qwen4_exp", "text_config": {
        "model_type": "qwen4_exp_text", "hidden_size": 2560,
        "num_hidden_layers": 48, "hc_count": 4, "hc_lowrank": 320,
        "indexer_compress_ratio": 4, "linear_num_key_heads": 16,
        "linear_num_value_heads": 48, "linear_key_head_dim": 128,
        "linear_value_head_dim": 128, "ple_layer_ids": [2], "ngram_size": 3,
        "ngram_vocab_size_base": 20000000, "heads_per_ngram": 8,
        "ple_embed_dim": 2560, "ngram_sidecar": True, "num_experts": 512,
        "num_experts_per_tok": 10, "moe_intermediate_size": 640, "vocab_size": 248320,
    }})
    # This pack-level override must survive profile defaults too.
    pack = {"MTPLX_QSA_GATHER_MAX_ROWS": "24"}
    monkeypatch.setattr(openai, "load_runtime_contract", lambda _: (
        SimpleNamespace(runtime_env_overrides=pack), None))
    profile = get_profile("turbo")
    before = dict(os.environ)
    resolved = public._runtime_env_with_model_contract_overrides(
        profile.env_dict(), {}, profile, model=model)
    assert dict(os.environ) == before
    expected = dict(os.environ)
    family = openai._server_runtime_env_overrides(
        SimpleNamespace(model=model, generation_mode="mtp", verify_strategy="batched"), pack)
    apply_profile_env("turbo", environ=expected, runtime_env_overrides=family)
    for key, value in resolved.items():
        assert value == expected.get(key), key
    assert resolved["MTPLX_FAMILY_CAPTURE_COMMIT"] == "1"
    assert resolved["MTPLX_SKIP_VERIFY_SNAPSHOT"] == "0"
    from mtplx.cache_state import m1_long_context_defaults

    # The M1 family keeps the profile's NAX/vk verify lanes for this family;
    # elsewhere the server neutralizes them.
    assert resolved["MTPLX_NAX_VERIFY"] == ("1" if m1_long_context_defaults() else "0")
    assert resolved["MTPLX_QSA_GATHER_MAX_ROWS"] == "24"
    if not exports:
        assert resolved["MTPLX_QWEN4_FIXED_M4_VERIFY"] == "1"
        assert resolved["MTPLX_BATCH_TARGET_ARRAYS"] == "1"
        assert resolved["MTPLX_LAZY_TARGET_DISTRIBUTIONS"] == "0"


def test_family_environment_restored_if_memory_preflight_refuses(tmp_path, monkeypatch):
    from mtplx.commands import public
    from mtplx.server import openai

    key = "MTPLX_QWEN4_FIXED_M4_VERIFY"
    monkeypatch.delenv(key, raising=False)
    before = dict(os.environ)
    def refuse(**kwargs):
        assert os.environ[key] == "1"
        raise RuntimeError("memory preflight refused")
    monkeypatch.setattr(openai, "apply_memory_caps_preflight", refuse)
    with pytest.raises(RuntimeError, match="memory preflight refused"):
        public._depth_sweep_native60(model=str(tmp_path), prompt_suite="unused",
            max_tokens=1, limit=1, seed=0, runtime_env={key: "1"})
    assert dict(os.environ) == before
