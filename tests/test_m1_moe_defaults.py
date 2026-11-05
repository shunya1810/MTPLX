"""M1 family: the 35B-A3B MoE defaults to capture_commit; dense models are untouched."""

import os
from types import SimpleNamespace

import pytest

from mtplx.commands import public as p


@pytest.fixture(autouse=True)
def _private_environ(monkeypatch):
    env = dict(os.environ)
    for key in (*p._M1_MOE_ENV, *p._M1_MOE_GDN_ENV, *p._M1_A3B_ENV):
        env.pop(key, None)
    env.pop("MTPLX_M1_MOE_LANE", None)
    monkeypatch.setattr(os, "environ", env)
    return env


def _args(**kw):
    return SimpleNamespace(_cli_flags=set(kw.pop("cli", ())), _injected_default_flags=set(), depth=3,
                           verify_strategy=None, draft_temperature=None, draft_top_p=None, draft_top_k=None, **kw)


def _moe_id():
    return p.QWEN36_35B_OPTIMIZED_SPEED_FP16_PUBLIC_MODEL_ID


def test_m1_moe_uses_capture_commit(monkeypatch):
    monkeypatch.setenv("MTPLX_M1_LONG_CONTEXT", "1")
    monkeypatch.delenv("MTPLX_M1_MOE_VERIFY_STRATEGY", raising=False)
    a = _args()
    p._apply_qwen36_35b_optimized_speed_defaults(a, _moe_id())
    assert a.verify_strategy == "capture_commit" and a.depth == 1


def test_off_m1_and_override_keep_target_prefix(monkeypatch):
    monkeypatch.setenv("MTPLX_M1_LONG_CONTEXT", "0")
    a = _args()
    p._apply_qwen36_35b_optimized_speed_defaults(a, _moe_id())
    assert a.verify_strategy == "target_prefix"
    monkeypatch.setenv("MTPLX_M1_LONG_CONTEXT", "1")
    monkeypatch.setenv("MTPLX_M1_MOE_VERIFY_STRATEGY", "target_prefix")
    a = _args()
    p._apply_qwen36_35b_optimized_speed_defaults(a, _moe_id())
    assert a.verify_strategy == "target_prefix"


def test_explicit_flag_and_dense_models_are_untouched(monkeypatch):
    monkeypatch.setenv("MTPLX_M1_LONG_CONTEXT", "1")
    a = _args(cli={"verify-strategy"}); a.verify_strategy = "trim_commit"
    p._apply_qwen36_35b_optimized_speed_defaults(a, _moe_id())
    assert a.verify_strategy == "trim_commit"
    a = _args()
    p._apply_qwen36_35b_optimized_speed_defaults(a, "mtplx-qwen38-27b-optimized-speed-fp16")
    assert a.verify_strategy is None and a.depth == 3


def _model_dir(tmp_path, name, config):
    import json

    d = tmp_path / name
    d.mkdir()
    (d / "config.json").write_text(json.dumps(config))
    return d


_GDN = {"recommended_backend": "qwen3_next"}


def test_a3b_gets_every_lane(_private_environ, tmp_path):
    env = _private_environ
    env["MTPLX_M1_LONG_CONTEXT"] = "1"
    env["MTPLX_QWEN_MOE_PACK_GATE_UP"] = "0"
    a = _args()
    p._apply_qwen36_35b_optimized_speed_defaults(a, _moe_id())
    p._apply_m1_moe_defaults(a, _GDN, _moe_id(), _model_dir(tmp_path, "a3b", {"num_experts": 256}))
    child = a._child_env_defaults
    for key in (*p._M1_MOE_ENV, *p._M1_MOE_GDN_ENV, "MTPLX_A3B_WHOLE_MOE_FUSION"):
        assert key in child, key
    assert child["MTPLX_M1_QMV_ROWS"] == "1"  # sampled-verify 2-row projections
    assert child["MTPLX_M1_CHUNK_TOPK"] == "1"  # sampled-round top-k candidates
    assert child["MTPLX_A3B_FUSED_ADD_NORM"] == "1"  # residual add inside the next norm
    assert child["MTPLX_A3B_GDN_FUSED"] == "1"  # one kernel per GDN layer
    assert "MTPLX_QWEN_MOE_PACK_GATE_UP" not in child  # an explicit env wins
    assert "MTPLX_LINEAR_GDN_FROM_CONV_TGY" not in child
    assert "MTPLX_A3B_WHOLE_MOE_FUSION" not in env  # never this process's env


def test_other_moe_gets_the_generic_lane_only(_private_environ, tmp_path):
    env = _private_environ
    env["MTPLX_M1_LONG_CONTEXT"] = "1"
    a = _args()
    moe = _model_dir(tmp_path, "moe", {"text_config": {"num_experts": 128}})
    p._apply_m1_moe_defaults(a, {"recommended_backend": "mixtral"}, "some-moe", moe)
    assert set(a._child_env_defaults) == set(p._M1_MOE_ENV)
    b = _args()
    p._apply_m1_moe_defaults(b, _GDN, "some-qwen3-next-moe", moe)
    assert set(b._child_env_defaults) == set(p._M1_MOE_ENV) | set(p._M1_MOE_GDN_ENV)


@pytest.mark.parametrize(
    "setup",
    [
        {"MTPLX_M1_LONG_CONTEXT": "0"},
        {"MTPLX_M1_LONG_CONTEXT": "1", "MTPLX_M1_MOE_LANE": "0"},
    ],
)
def test_m1_moe_lane_env_stays_off(_private_environ, tmp_path, setup):
    env = _private_environ
    env.update(setup)
    a = _args()
    p._apply_qwen36_35b_optimized_speed_defaults(a, _moe_id())
    p._apply_m1_moe_defaults(a, _GDN, _moe_id(), _model_dir(tmp_path, "a3b", {"num_experts": 256}))
    assert not a._child_env_defaults


def test_dense_models_get_nothing(_private_environ, tmp_path):
    env = _private_environ
    env["MTPLX_M1_LONG_CONTEXT"] = "1"
    a = _args()
    dense = _model_dir(tmp_path, "dense", {"num_experts": 0, "text_config": {"hidden_size": 5120}})
    p._apply_m1_moe_defaults(a, _GDN, "mtplx-qwen38-27b-optimized-speed-fp16", dense)
    assert a._child_env_defaults == {}


def test_a3b_target_prefix_keeps_only_the_generic_lane(_private_environ, tmp_path):
    env = _private_environ
    env.update({"MTPLX_M1_LONG_CONTEXT": "1", "MTPLX_M1_MOE_VERIFY_STRATEGY": "target_prefix"})
    a = _args()
    p._apply_qwen36_35b_optimized_speed_defaults(a, _moe_id())
    p._apply_m1_moe_defaults(a, _GDN, _moe_id(), _model_dir(tmp_path, "a3b", {"num_experts": 256}))
    assert "MTPLX_A3B_WHOLE_MOE_FUSION" not in a._child_env_defaults
    assert "MTPLX_M1_MOE_DEFERRED_DRAFT" in a._child_env_defaults


def test_ane_prefill_skips_moe_models():
    from types import SimpleNamespace as NS

    from mtplx import ane_prefill as ap

    class Qwen3NextSparseMoeBlock:  # noqa: N801 - mirrors the class name it detects
        pass

    moe = NS(named_modules=lambda: [("layers.0.mlp", Qwen3NextSparseMoeBlock())])
    dense = NS(named_modules=lambda: [("layers.0.mlp", object())])
    assert ap._is_moe(moe) and not ap._is_moe(dense) and not ap._is_moe(None)


def test_qwen4_moe_keeps_the_2048_chunk_and_a_wide_rung(_private_environ, monkeypatch):
    monkeypatch.setenv("MTPLX_M1_LONG_CONTEXT", "1")
    env = p._m1_moe_lane_env(moe=True, gdn=False, a3b_capture_commit=False, qwen4=True)
    assert "MTPLX_PREFILL_CHUNK_SIZE_DENSE" not in env
    assert "MTPLX_PREFILL_CHUNK_SIZE_REPAGE" not in env
    assert env["MTPLX_QWEN4_PREFILL_WIDE_CHUNK"] == "4096"
    assert env["MTPLX_GDN_BLOCKED_PREFILL"] == "1"
    assert env["MTPLX_QSA_PREFILL_DIRECT_MIN_CONTEXT"] == env["MTPLX_QSA_PREFILL_WIDE_MIN_CONTEXT"] == "8192"
    other = p._m1_moe_lane_env(moe=True, gdn=False, a3b_capture_commit=False)
    assert other["MTPLX_PREFILL_CHUNK_SIZE_DENSE"] == "4096"
