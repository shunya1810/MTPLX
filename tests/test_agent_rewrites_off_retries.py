"""MTPLX_AGENT_REWRITES=off and the stream worker's steering retries.

The tool-fed empty retry re-generates a turn from the transcript plus an
injected user message. #282 made MTPLX_AGENT_REWRITES=off a hard passthrough
guarantee with no injected steering text, so under off it must stand down.

The stalled-promise retry used to be the second steering retry. It was
removed on 2026-09-30 (it threw away finished answers whose last line read
like a promise), so a stalled promise now finishes in one pass in every
posture; tests/test_finished_answers_are_never_regenerated.py covers the
answer shapes it used to catch.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient
from test_server_openai import (
    _fake_streaming_session_state,
    _stream_payloads,
    _tool_schema,
)

from mtplx.server import openai
from mtplx.server.openai import create_app

ORPHAN_TAIL = "</think>\n\nparameter=limit>\n180\n</parameter>\n</function>\n</tool_call>"
STALLED_PROMISE = "</think>\n\nLet me check the build output now."
ANSWER = "Implemented the HUD cleanup and verified npm build."


def _tool_fed_request() -> dict:
    return {
        "messages": [
            {"role": "user", "content": "Improve this project after reading the files."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_read",
                        "type": "function",
                        "function": {
                            "name": "read",
                            "arguments": '{"filePath":"src/Game.ts"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_read",
                "content": '{"content":"export const score = 0"}',
            },
        ],
        "tools": [_tool_schema()],
        "tool_choice": "auto",
        "stream": True,
        "max_tokens": 128,
        "enable_thinking": True,
    }


def _pass_kind(observability: dict) -> str:
    if observability.get("reasoning_completion_repair_attempted"):
        return "reasoning_completion_repair"
    if observability.get("tool_fed_empty_retry_attempted"):
        return "tool_fed_empty_retry"
    return "first"


def _run_stream(monkeypatch, texts):
    """Stream one seedless tool-fed request; return prompts, pass kinds, stats."""

    state = _fake_streaming_session_state()
    state.args.stream_interval = 1
    state.args.stats_footer = False
    client = TestClient(create_app(state))
    prompts: list[list[int]] = []
    kinds: list[str] = []

    def fake_run_generation(_state, prompt_ids, **kwargs):
        text = texts[len(prompts)]
        prompts.append(list(prompt_ids))
        observability = kwargs.get("request_observability") or {}
        kinds.append(_pass_kind(observability))
        tokens = [ord(char) for char in text]
        token_callback = kwargs.get("token_callback")
        if token_callback is not None:
            for token in tokens:
                token_callback([token])
        return {
            "text": text,
            "tokens": tokens,
            "stats": {
                **observability,
                "generation_mode": kwargs["generation_mode"],
                "mtp_depth": kwargs["depth"],
                "completion_tokens": len(tokens),
                "decode_tok_s": 20.0,
            },
            "prompt_tokens": len(prompt_ids),
            "completion_tokens": len(tokens),
            "finish_reason": "stop",
        }

    monkeypatch.setattr(openai, "_run_generation", fake_run_generation)
    with client.stream(
        "POST",
        "/v1/chat/completions",
        headers={"x-mtplx-cache-mode": "bypass", "x-mtplx-client": "pi"},
        json=_tool_fed_request(),
    ) as response:
        body = "".join(response.iter_text())
    assert response.status_code == 200
    payloads = _stream_payloads(body)
    final = [p for p in payloads if p["choices"][0]["finish_reason"]]
    return prompts, kinds, final[-1]["mtplx_stats"]


@pytest.mark.parametrize("mode", [None, "on"], ids=["default", "on"])
def test_the_tool_fed_empty_retry_runs_in_the_default_and_on_postures(
    monkeypatch, mode
):
    if mode is None:
        monkeypatch.delenv("MTPLX_AGENT_REWRITES", raising=False)
    else:
        monkeypatch.setenv("MTPLX_AGENT_REWRITES", mode)

    _prompts, kinds, _stats = _run_stream(monkeypatch, [ORPHAN_TAIL, ANSWER])

    assert kinds == ["first", "tool_fed_empty_retry"]


@pytest.mark.parametrize("mode", [None, "on", "off"], ids=["default", "on", "off"])
def test_a_stalled_promise_finishes_in_one_pass_in_every_posture(monkeypatch, mode):
    if mode is None:
        monkeypatch.delenv("MTPLX_AGENT_REWRITES", raising=False)
    else:
        monkeypatch.setenv("MTPLX_AGENT_REWRITES", mode)

    prompts, kinds, stats = _run_stream(monkeypatch, [STALLED_PROMISE, ANSWER])

    assert kinds == ["first"]
    assert len(prompts) == 1
    assert "stalled_agent_retry_attempted" not in stats


@pytest.mark.parametrize(
    "first_text",
    [ORPHAN_TAIL, STALLED_PROMISE],
    ids=["orphan_tool_markup", "stalled_promise"],
)
def test_agent_rewrites_off_disables_steering_retries(monkeypatch, first_text):
    monkeypatch.setenv("MTPLX_AGENT_REWRITES", "off")

    prompts, kinds, stats = _run_stream(monkeypatch, [first_text, ANSWER])

    assert kinds == ["first"]
    assert len(prompts) == 1
    assert "tool_fed_empty_retry_attempted" not in stats
    assert "stalled_agent_retry_attempted" not in stats


def test_agent_rewrites_off_keeps_protocol_reasoning_completion(monkeypatch):
    """Off disables injected user instructions, not the existing close repair."""
    monkeypatch.setenv("MTPLX_AGENT_REWRITES", "off")
    thinking = "The user wants the task finished. I can now summarize the result."
    prompts, kinds, stats = _run_stream(monkeypatch, [thinking, ANSWER])
    assert kinds == ["first", "reasoning_completion_repair"]
    continued = prompts[0] + [ord(char) for char in thinking]
    assert prompts[1][: len(continued)] == continued
    assert "tool_fed_empty_retry_attempted" not in stats
    assert "stalled_agent_retry_attempted" not in stats
