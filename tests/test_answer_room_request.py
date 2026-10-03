"""The answer's room is decided before the prefill, and said out loud.

The server prices a request's longest answer before any work
(``_answer_room``, tests/test_one_copy_admission.py). This file pins what the
request then does with the answer: a request with no room for even a short
answer is refused before generation with a plain 507, a request with room
for a shorter answer runs with that limit, and an answer that reaches the
limit tells the client why it stopped instead of passing for its own
max_tokens.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")

from test_api_benchmark_contracts import _envelope_client

from mtplx.server import openai

ROOM = {
    "action": "answer_room",
    "prompt_tokens": 3,
    "requested_answer_tokens": 8,
    "answer_rows": 11,
    "need_bytes": 900,
    "room_bytes": 300,
}


def _generator(captured: dict, *, tokens: int, finish_reason: str):
    from mtplx.generation import GenerationStats

    def generate(*_args, **kwargs):
        captured["max_tokens"] = kwargs.get("max_tokens")
        stats = GenerationStats(
            mode="mtpk",
            generated_tokens=tokens,
            elapsed_s=0.01,
            tok_s=200.0,
            decode_elapsed_s=0.005,
            decode_tok_s=400.0,
            prompt_eval_time_s=0.005,
            prompt_tps=600.0,
            verify_calls=1,
            accepted_by_depth=[1],
        )
        return SimpleNamespace(
            tokens=list(range(40, 40 + tokens)),
            text="x" * tokens,
            stats=stats,
            final_state=None,
            finish_reason=finish_reason,
        )

    return generate


def _post(client):
    return client.post(
        "/v1/chat/completions",
        headers={"x-mtplx-cache-mode": "bypass"},
        json={"messages": [{"role": "user", "content": "Write."}], "max_tokens": 8},
    )


def test_a_capped_answer_that_reaches_its_limit_says_why(monkeypatch):
    captured: dict = {}
    client, _state = _envelope_client(
        monkeypatch, generator=_generator(captured, tokens=5, finish_reason="length")
    )
    monkeypatch.setattr(openai, "_answer_room", lambda *a, **k: {**ROOM, "answer_token_cap": 5})
    response = _post(client)
    assert response.status_code == 200
    assert captured["max_tokens"] == 5
    stop = response.json()["mtplx_stats"]["memory_stop"]
    assert stop["reason"] == "answer_capped_for_memory"
    assert stop["requested_tokens"] == 8 and stop["cap_tokens"] == 5
    assert "5 tokens" in stop["message"] and "memory" in stop["message"]


def test_a_capped_answer_that_ends_on_its_own_says_nothing(monkeypatch):
    captured: dict = {}
    client, _state = _envelope_client(
        monkeypatch, generator=_generator(captured, tokens=2, finish_reason="stop")
    )
    monkeypatch.setattr(openai, "_answer_room", lambda *a, **k: {**ROOM, "answer_token_cap": 5})
    response = _post(client)
    assert response.status_code == 200
    assert "memory_stop" not in response.json()["mtplx_stats"]


def test_no_room_for_a_short_answer_is_refused_before_any_generation(monkeypatch):
    captured: dict = {}
    client, _state = _envelope_client(
        monkeypatch, generator=_generator(captured, tokens=2, finish_reason="stop")
    )
    monkeypatch.setattr(openai, "_answer_room", lambda *a, **k: {**ROOM, "refused": True})
    response = _post(client)
    assert response.status_code == 507
    assert "max_tokens" not in captured
    assert "refused before prefill" in response.text


def test_an_answer_with_room_runs_untouched(monkeypatch):
    captured: dict = {}
    client, _state = _envelope_client(
        monkeypatch, generator=_generator(captured, tokens=2, finish_reason="stop")
    )
    monkeypatch.setattr(openai, "_answer_room", lambda *a, **k: None)
    response = _post(client)
    assert response.status_code == 200
    assert captured["max_tokens"] == 8
