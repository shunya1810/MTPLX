"""Real admission arithmetic for scoring, with only machine readings replaced."""

from types import SimpleNamespace

import mlx.core as mx
import pytest
from fastapi.testclient import TestClient
from test_memguard_admission import (
    GIB, Q27_KV, Q27_WEIGHTS, _Machine, _install, _manager, _q27_text_args,
    _served_profile,  # noqa: F401 -- served memory policy fixture
)
from test_prompt_scoring_parent import TinyScoringRuntime
from test_server_openai import CaptureTokenizer, _fake_state

from mtplx.server import openai as srv


@pytest.fixture
def scoring_admission(monkeypatch):
    state = _fake_state()
    with mx.stream(mx.cpu):
        state.runtime = TinyScoringRuntime()
    # Admission uses the real 27B geometry; execution uses a tiny target.
    state.runtime.text_args = _q27_text_args
    state.runtime.tokenizer = CaptureTokenizer()
    state.args.prefill_chunk_tokens = 2048
    state.args.generation_mode = "ar"
    state.context_window = 16384
    state.begin_foreground = lambda: None
    state.end_foreground = lambda: None
    state.requests_completed = 0
    state.last_request_at = 0.0
    state.sessions = _manager()
    state.memory_plan = SimpleNamespace(
        kv_bytes_per_token=Q27_KV, kv_bytes_per_token_effective=Q27_KV,
        aux_bytes_per_token=0, mtp_history_bytes_per_token=4096,
        prefill_transient_bytes_per_token=0, runtime_transients_bytes=3 * GIB,
        model_weights_bytes=Q27_WEIGHTS,
    )
    state.metal_memory_caps = {"memory_limit_bytes": 24 * GIB, "total_ram_bytes": 32 * GIB}
    state.allow_swap = False
    machine = _Machine(state.sessions.bank, base_gib=21.1, cache_gib=0, host_gib=0)
    _install(monkeypatch, machine)
    monkeypatch.setenv("MTPLX_PREFILL_ADMISSION_SHED", "1")
    # Endpoint runs on a worker: explicitly keep tiny attention on the CPU.
    forward = state.runtime.forward_ar

    def cpu_forward(*args, **kwargs):
        with mx.stream(mx.cpu):
            return forward(*args, **kwargs)

    state.runtime.forward_ar = cpu_forward
    return state, TestClient(srv.create_app(state))


def test_default_generic_scoring_admits_the_27b_request_that_fits(scoring_admission):
    state, client = scoring_admission
    prompt = [7] * 8192
    # The generation bill at serving width reproduced the review's refusal
    # (24.63 GiB on the 24 GiB limit at 2,048 rows, 2.12.1). With the bill
    # measured on the 27B (2026-10-02: 1.52 GiB at 1,024 rows, 2.55 at
    # 2,048) it runs at 1,024 rows, under the limit.
    priced = {}
    previous = srv._prefill_admission_shed(
        state, prompt_ids=prompt, session_bank=None, session_id=None,
        max_new_tokens=0, mtp_depth=0, prefill_chunk_tokens=2048, pricing=priced,
    )
    assert not previous.get("refused")
    assert previous["prefill_chunk_requested"] == 2048
    assert previous["prefill_chunk_tokens"] == 1024
    assert 23.28 * GIB < previous["projected_bytes_after"] < 24 * GIB

    response = client.post("/v1/completions", json={
        "prompt": prompt, "echo": True, "logprobs": 2, "max_tokens": 0,
    })

    assert response.status_code == 200, response.text
    assert state.runtime.widths == [256] * 32
    assert len(response.json()["choices"][0]["logprobs"]["token_logprobs"]) == 8192
    assert response.json()["mtplx_stats"]["prefill_chunk_tokens"] == 256
    scoring_price = {}
    admitted = srv._prefill_admission_shed(
        state, prompt_ids=prompt, session_bank=None, session_id=None,
        prefill_chunk_tokens=256, pricing=scoring_price, prompt_scoring=True,
    )
    assert not (admitted or {}).get("refused")
    projected = 21.1 + scoring_price["growth"]["growth_bytes"] / GIB
    assert projected < 24
    print({"old_projected_gib": 24.63125, "scoring_projected_gib": projected,
           "scoring_growth": scoring_price["growth"]})


@pytest.mark.parametrize("layout", ["contiguous_dense_decode", "contiguous_then_repage"])
def test_scoring_bill_has_only_target_cache_and_bounded_logits(
    scoring_admission, monkeypatch, layout
):
    state, client = scoring_admission
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL_LAYOUT", layout)
    captured = []
    original = srv.make_prefill_system_guard

    def capture(*args, **kwargs):
        guard = original(*args, **kwargs)
        captured.append((kwargs["priced"], guard))
        return guard

    # Spy on the real guard, leaving admission and reservation arithmetic intact.
    monkeypatch.setattr("mtplx.server.prefill_safety.make_prefill_system_guard", capture)
    response = client.post("/v1/completions", json={
        "prompt": [7] * 512, "echo": True, "logprobs": 2, "max_tokens": 0,
    })
    assert response.status_code == 200, response.text
    bill, guard = captured[0]
    assert bill["scratch_rows"] == 256
    assert bill["live_prefill_bytes"] == 512 * Q27_KV
    assert bill["repage_bytes"] == bill["decode_start_bytes"] == 0
    assert bill["output_reserve_bytes"] == bill["publish_copy_bytes"] == 0
    assert bill["logits_rows"] == 256
    assert bill["logits_bytes"] == 256 * 248320 * 4
    assert bill["growth_bytes"] == bill["prefill_end_bytes"]
    assert guard.after_prefill_reserve_bytes == 0
    assert guard.forward_rows_bytes is None


@pytest.mark.parametrize("width", [256, 2048, None])
def test_gemma_scoring_bill_excludes_the_assistants_full_prompt_kv(monkeypatch, width):
    from test_memguard_backend_contract import (
        GEMMA_PLANNED_KV, GEMMA_RESIDENT, GEMMA_WINDOW_ROW, _gemma_state,
    )
    from mtplx.server.prefill_safety import prompt_scoring_growth

    monkeypatch.setenv("MTPLX_GEMMA4_PREFILL_CHUNK_TOKENS", "whole" if width is None else "auto")
    state = _gemma_state(_manager())
    bill = prompt_scoring_growth(state, prompt_tokens=8192, width=width)
    expected = (8192 * GEMMA_PLANNED_KV if width is None else
                8192 * GEMMA_RESIDENT + (1023 + width) * GEMMA_WINDOW_ROW)
    assert bill["live_prefill_bytes"] == expected
    assert bill["scratch_rows"] == (width or 8192)
    assert bill["logits_rows"] == 256
    assert bill["repage_bytes"] == bill["decode_start_bytes"] == 0


def test_scoring_fallback_guard_prices_its_own_width_without_admission(scoring_admission, monkeypatch):
    from mtplx.server.prefill_safety import make_prefill_system_guard, prompt_scoring_growth

    state, _client = scoring_admission
    monkeypatch.setenv("MTPLX_PREFILL_ADMISSION_SHED", "off")
    bill = prompt_scoring_growth(state, prompt_tokens=8192, width=256)
    guard = make_prefill_system_guard(
        state, prompt_tokens=8192, chunk_tokens=256, priced=None, prompt_scoring=True
    )
    assert guard.chunk_reserve_bytes == bill["chunk_bytes"]
    assert guard.after_prefill_reserve_bytes == 0
    assert not guard()
