"""Blank retries only run when a fresh seed can change the output.

A greedy decode (temperature 0) ignores the seed, so retrying a blank
greedy response replays the identical generation. The engine is faked; the
real ``_run_generation`` retry loop and envelope assembly run as in
production (same harness as tests/test_api_benchmark_contracts.py).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")

from test_api_benchmark_contracts import _envelope_client

BLANK_TOKENS = [271]


def _counting_blank_generator(calls: list[int]):
    """Engine stand-in that streams one blank token per call via the
    token callback, like the real engine does."""
    from mtplx.generation import GenerationStats

    def _generate(*_args, **kwargs):
        calls.append(1)
        callback = kwargs.get("token_callback")
        if callback is not None:
            callback(list(BLANK_TOKENS))
        stats = GenerationStats(
            mode="mtpk",
            generated_tokens=len(BLANK_TOKENS),
            elapsed_s=0.01,
            tok_s=100.0,
            decode_elapsed_s=0.005,
            decode_tok_s=200.0,
            prompt_eval_time_s=0.005,
            prompt_tps=600.0,
            verify_calls=1,
            accepted_by_depth=[1],
        )
        return SimpleNamespace(
            tokens=list(BLANK_TOKENS),
            text="\n\n",
            stats=stats,
            final_state=None,
            finish_reason="length",
        )

    return _generate


def _complete(monkeypatch, **body):
    calls: list[int] = []
    client, _state = _envelope_client(
        monkeypatch, generator=_counting_blank_generator(calls)
    )
    response = client.post(
        "/v1/completions",
        json={"prompt": "Antwoord:", "max_tokens": 1, **body},
    )
    assert response.status_code == 200, response.text
    return response.json(), calls


def test_greedy_blank_completion_is_not_retried(monkeypatch):
    payload, calls = _complete(monkeypatch, temperature=0)

    assert len(calls) == 1
    assert payload["mtplx_stats"]["server_attempts"] == 1
    assert payload["mtplx_stats"]["server_blank_retries"] == 0
    assert payload["usage"]["completion_tokens"] == 1


def test_sampled_blank_completion_still_retries_with_fresh_seeds(monkeypatch):
    payload, calls = _complete(monkeypatch, temperature=0.7)

    assert len(calls) == 4  # 1 + default --blank-retry-attempts 3
    assert payload["mtplx_stats"]["server_attempts"] == 4
    assert payload["mtplx_stats"]["server_blank_retries"] == 3


def test_retried_completion_reports_usage_of_returned_attempt(monkeypatch):
    payload, calls = _complete(monkeypatch, temperature=0.7)

    assert len(calls) == 4
    # Discarded attempts used to accumulate: 4 completion tokens for a
    # max_tokens=1 response.
    assert payload["usage"]["completion_tokens"] == 1


def test_explicit_seed_blank_completion_is_not_retried(monkeypatch):
    payload, calls = _complete(monkeypatch, temperature=0.7, seed=7)

    assert len(calls) == 1
    assert payload["mtplx_stats"]["server_attempts"] == 1
