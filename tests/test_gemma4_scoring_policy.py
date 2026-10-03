"""Scoring enters the serving prefill memory policy before any forward."""

from threading import Event
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from test_gemma4_prompt_scoring import _StubGemmaRuntime
from test_server_openai import CaptureTokenizer, _fake_state

from mtplx.server import openai


@pytest.fixture
def scoring(monkeypatch):
    state = _fake_state()
    state.runtime = _StubGemmaRuntime(tokenizer=CaptureTokenizer())
    state.args.prefill_chunk_tokens = 256
    state.context_window = 16384
    state.requests_completed = 0
    state.last_request_at = 0.0
    state.pressure_abort_event = Event()
    probe = SimpleNamespace(
        state=state, admissions=[], widths=[], foreground=[], receipt=None, shed=[]
    )
    state.begin_foreground = lambda: probe.foreground.append("begin")
    state.end_foreground = lambda: probe.foreground.append("end")

    def admit(_state, **kwargs):
        assert state.lock.locked()
        assert state.runtime.forward_calls == []
        probe.admissions.append(kwargs)
        return probe.receipt

    def reserve(_state, **kwargs):
        probe.widths.append(kwargs["chunk_tokens"])
        return 8

    monkeypatch.setenv("MTPLX_GEMMA4_PREFILL_CHUNK_TOKENS", "2048")
    monkeypatch.setattr(openai, "_prefill_admission_shed", admit)
    monkeypatch.setattr(openai, "_prefill_chunk_reserve_bytes", reserve)
    monkeypatch.setattr(openai, "_prefill_after_forward_plan", lambda *_a, **_k: {})
    monkeypatch.setattr(openai, "_mlx_memory_stats_live", lambda: {
        "ok": True, "active_memory_bytes": 1, "cache_memory_bytes": 0,
    })
    monkeypatch.setattr(openai, "_footprint_floor", lambda _s, **kw: (kw["allocator_bytes"], {}))
    monkeypatch.setattr(openai, "_shed_after_allocation_failure", lambda _s: probe.shed.append(True))
    monkeypatch.setattr(openai, "_PREFILL_SYSTEM_CHECK_INTERVAL_S", 0)
    probe.client = TestClient(openai.create_app(state))
    return probe


def _score(probe, rows):
    response = probe.client.post("/v1/completions", json={
        "prompt": "a" * rows, "echo": True, "logprobs": 2, "max_tokens": 0,
    })
    assert probe.foreground == ["begin", "end"]
    assert not probe.state.lock.locked()
    return response


def test_scoring_uses_server_prefill_width_and_admission(scoring):
    response = _score(scoring, 8192)
    assert response.status_code == 200, response.text
    widths = [len(call["tokens"]) for call in scoring.state.runtime.forward_calls]
    assert max(widths) <= 256
    assert widths == [256] * 32
    assert len(scoring.admissions) == 1
    admission = scoring.admissions[0]
    assert admission["prefill_chunk_tokens"] == 256
    assert admission["max_new_tokens"] == 0
    assert admission["mtp_depth"] == 0
    assert admission["session_bank"] is None
    assert admission["session_id"] is None
    assert scoring.widths == [256]


def test_scoring_uses_admission_narrowed_width(scoring):
    scoring.receipt = {"prefill_chunk_requested": 256, "prefill_chunk_tokens": 128}
    response = _score(scoring, 513)
    assert response.status_code == 200, response.text
    widths = [len(call["tokens"]) for call in scoring.state.runtime.forward_calls]
    assert max(widths) <= 128
    assert sum(widths) == 513
    assert min(widths) >= 2
    assert scoring.widths == [128]


def test_scoring_admission_refusal_runs_no_forward(scoring):
    scoring.receipt = {
        "refused": True, "limit_bytes": 100, "projected_bytes": 200,
        "prompt_tokens": 512, "miss_tokens": 512,
    }
    response = _score(scoring, 512)
    assert response.status_code == 507
    assert scoring.state.runtime.forward_calls == []
    assert response.json()["error"]["code"] == "insufficient_memory"


@pytest.mark.parametrize("after_first_chunk", [False, True])
def test_scoring_honors_sustained_pressure_before_each_forward(scoring, after_first_chunk):
    runtime = scoring.state.runtime
    forward = runtime.forward_target

    def trigger(*args, **kwargs):
        output = forward(*args, **kwargs)
        scoring.state.pressure_abort_event.set()
        return output

    runtime.forward_target = trigger
    if not after_first_chunk:
        scoring.state.pressure_abort_event.set()
    response = _score(scoring, 768)

    assert response.status_code == 507
    assert len(runtime.forward_calls) == int(after_first_chunk)
    assert scoring.shed == [True]


def test_scoring_checks_live_memory_before_the_next_chunk(scoring, monkeypatch):
    runtime = scoring.state.runtime
    scoring.state.metal_memory_caps = {"memory_limit_bytes": 100}
    monkeypatch.setattr(openai, "_mlx_memory_stats_live", lambda: {
        "ok": True,
        "active_memory_bytes": 200 if runtime.forward_calls else 1,
        "cache_memory_bytes": 0,
    })

    response = _score(scoring, 768)

    assert response.status_code == 507
    assert len(runtime.forward_calls) == 1
    assert response.json()["error"]["detail"]["memory"]["reason"] == "engine_limit"
    assert scoring.shed == [True]


def test_scoring_aborts_between_heads_of_one_target_forward(scoring, monkeypatch):
    scoring.state.args.prefill_chunk_tokens = 2048
    target = scoring.state.runtime.target
    head = target.logits_from_hidden

    def trigger(hidden):
        logits = head(hidden)
        scoring.state.pressure_abort_event.set()
        return logits

    monkeypatch.setattr(target, "logits_from_hidden", trigger)
    response = _score(scoring, 4096)

    assert response.status_code == 507
    assert len(scoring.state.runtime.forward_calls) == 1
    assert target.head_rows == [256]  # Stop before the other seven head chunks.
    assert scoring.shed == [True]
