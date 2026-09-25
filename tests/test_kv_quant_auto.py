"""auto KV quantization: q8 from the prompt-length threshold up, off below."""

from mtplx import kv_quant
from mtplx.runtime_options import normalize_paged_kv_quantization


def test_normalize_accepts_auto():
    assert normalize_paged_kv_quantization("auto") == "auto"
    assert normalize_paged_kv_quantization(" AUTO ") == "auto"


def test_auto_resolves_per_request_prompt_length(monkeypatch):
    monkeypatch.setenv("MTPLX_PAGED_KV_QUANT", "auto")
    monkeypatch.delenv("MTPLX_VLLM_METAL_PAGED_KV_QUANT", raising=False)
    monkeypatch.delenv(kv_quant.AUTO_THRESHOLD_ENV, raising=False)
    assert kv_quant.paged_kv_quant_setting_from_env() == "auto"
    monkeypatch.setenv("MTPLX_CURRENT_PREFILL_CONTEXT_TOKENS", "131071")
    assert kv_quant.paged_kv_quant_mode_from_env() == "off"
    assert kv_quant.config_from_env() is None
    monkeypatch.setenv("MTPLX_CURRENT_PREFILL_CONTEXT_TOKENS", "131072")
    assert kv_quant.paged_kv_quant_mode_from_env() == "q8"
    assert kv_quant.config_from_env().bits == 8


def test_auto_threshold_env_and_unset_context(monkeypatch):
    monkeypatch.setenv("MTPLX_PAGED_KV_QUANT", "auto")
    monkeypatch.delenv("MTPLX_VLLM_METAL_PAGED_KV_QUANT", raising=False)
    monkeypatch.setenv(kv_quant.AUTO_THRESHOLD_ENV, "65536")
    monkeypatch.setenv("MTPLX_CURRENT_PREFILL_CONTEXT_TOKENS", "70000")
    assert kv_quant.paged_kv_quant_mode_from_env() == "q8"
    monkeypatch.delenv("MTPLX_CURRENT_PREFILL_CONTEXT_TOKENS")
    assert kv_quant.paged_kv_quant_mode_from_env() == "off"


def test_explicit_modes_are_not_resolved(monkeypatch):
    monkeypatch.delenv("MTPLX_VLLM_METAL_PAGED_KV_QUANT", raising=False)
    monkeypatch.setenv("MTPLX_CURRENT_PREFILL_CONTEXT_TOKENS", "262144")
    for mode in ("off", "q8", "q4"):
        monkeypatch.setenv("MTPLX_PAGED_KV_QUANT", mode)
        assert kv_quant.paged_kv_quant_mode_from_env() == mode
