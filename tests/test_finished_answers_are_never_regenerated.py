"""A finished answer is delivered as generated: the server never re-runs it.

On 2026-09-29 (Pi, Flash-Next, 128 GB) the stream worker's stalled-agent
retry read three finished answers as "a tool promise without a tool call"
because their last line matched a regular expression (for example
``Say "A" or "B" and I'll run it.``). It threw each answer away, re-rendered
the transcript with an injected user turn and generated the turn again:
82 s of finished answers lost and up to 60K tokens prefilled a second time
from a prompt that had lost the committed-token repair. That retry is gone.

These tests pin the contract for the answer shapes that used to trigger it
and for the ones that never did: exactly one generation per request (so no
hidden second prefill) and the original text reaches the client. The first
assertion of each shape proves the text is one the removed retry's own
predicate matched, so the test fails on the old code.
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

COMPLETED_WITH_OFFER = (
    "The collision box now matches the sprite and the score resets on "
    "restart. I can also add a high-score table or tune the gravity next."
    "\n\nSay \"A\" or \"B\" and I'll run it."
)
CHOICE_QUESTION = (
    "Both fixes are in place and the build passes.\n\nWould you like the "
    "physics tweak or the audio fix first? Tell me which one and I will "
    "check it right away."
)
BARE_PROMISE = "Let me check the build output now."
FINAL_ANSWER = "Implemented the HUD cleanup and verified npm build."
PROMISE_BEFORE_CALL = "Let me read the file."
TOOL_CALL = "<tool_call>\n<function=session_status>\n</function>\n</tool_call>"


def _tool_fed_request(*, stream: bool = True) -> dict:
    """A turn after tool results with tools declared: the removed retry's
    precondition (tools active, tool results in the history, no seed)."""

    return {
        "messages": [
            {"role": "user", "content": "Finish the Flappy Bird fixes."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_read",
                        "type": "function",
                        "function": {
                            "name": "session_status",
                            "arguments": "{}",
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_read",
                "content": '{"status":"build passes"}',
            },
        ],
        "tools": [_tool_schema()],
        "tool_choice": "auto",
        "stream": stream,
        "max_tokens": 256,
        "enable_thinking": True,
    }


def _serve(monkeypatch, first_text: str, *, stream: bool = True):
    """One request over a fake engine that would answer a second pass with a
    different text. Returns (generation prompts, response body or json)."""

    state = _fake_streaming_session_state()
    state.args.stream_interval = 1
    state.args.stats_footer = False
    client = TestClient(create_app(state))
    prompts: list[list[int]] = []
    texts = ["</think>\n\n" + first_text, "</think>\n\nSECOND PASS"]

    def fake_run_generation(_state, prompt_ids, **kwargs):
        text = texts[min(len(prompts), 1)]
        prompts.append(list(prompt_ids))
        tokens = [ord(char) for char in text]
        token_callback = kwargs.get("token_callback")
        if token_callback is not None:
            for token in tokens:
                token_callback([token])
        observability = kwargs.get("request_observability") or {}
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
    headers = {"x-mtplx-cache-mode": "bypass", "x-mtplx-client": "pi"}
    if not stream:
        response = client.post(
            "/v1/chat/completions",
            headers=headers,
            json=_tool_fed_request(stream=False),
        )
        assert response.status_code == 200
        return prompts, response.json()
    with client.stream(
        "POST",
        "/v1/chat/completions",
        headers=headers,
        json=_tool_fed_request(),
    ) as response:
        body = "".join(response.iter_text())
    assert response.status_code == 200
    return prompts, body


def _streamed_content(body: str) -> str:
    return "".join(
        choice.get("delta", {}).get("content") or ""
        for payload in _stream_payloads(body)
        for choice in payload.get("choices", [])
    )


def _final_stats(body: str) -> dict:
    final = [
        payload
        for payload in _stream_payloads(body)
        if payload.get("choices") and payload["choices"][0].get("finish_reason")
    ]
    return final[-1].get("mtplx_stats") or {}


@pytest.fixture(params=[None, "on"], ids=["default_posture", "rewrites_on"])
def posture(request, monkeypatch):
    if request.param is None:
        monkeypatch.delenv("MTPLX_AGENT_REWRITES", raising=False)
    else:
        monkeypatch.setenv("MTPLX_AGENT_REWRITES", request.param)
    return request.param


@pytest.mark.parametrize(
    "answer",
    [COMPLETED_WITH_OFFER, CHOICE_QUESTION, BARE_PROMISE],
    ids=["completed_answer_offering_to_run", "question_offering_choices", "bare_promise"],
)
def test_answers_the_old_retry_matched_are_delivered_from_one_generation(
    monkeypatch, posture, answer
):
    # The removed retry's predicate matches this answer, so the old stream
    # worker generated the turn a second time and served "SECOND PASS".
    assert openai._looks_like_stalled_agent_tool_promise(answer) is True

    prompts, body = _serve(monkeypatch, answer)

    assert len(prompts) == 1
    content = _streamed_content(body)
    assert answer in content
    assert "SECOND PASS" not in content
    stats = _final_stats(body)
    assert "stalled_agent_retry_attempted" not in stats
    assert "stream_attempts" not in stats
    assert "retry_path" not in stats


def test_a_tool_promise_followed_by_its_call_is_one_generation(monkeypatch, posture):
    prompts, body = _serve(monkeypatch, PROMISE_BEFORE_CALL + "\n\n" + TOOL_CALL)

    assert len(prompts) == 1
    payloads = _stream_payloads(body)
    names = [
        item.get("function", {}).get("name")
        for payload in payloads
        for choice in payload.get("choices", [])
        for item in choice.get("delta", {}).get("tool_calls", [])
        if item.get("function", {}).get("name")
    ]
    assert names == ["session_status"]
    final = [p for p in payloads if p.get("choices") and p["choices"][0].get("finish_reason")]
    assert final[-1]["choices"][0]["finish_reason"] == "tool_calls"
    assert "stream_attempts" not in _final_stats(body)


def test_an_ordinary_final_answer_is_one_generation(monkeypatch, posture):
    assert openai._looks_like_stalled_agent_tool_promise(FINAL_ANSWER) is False

    prompts, body = _serve(monkeypatch, FINAL_ANSWER)

    assert len(prompts) == 1
    assert FINAL_ANSWER in _streamed_content(body)
    assert "stream_attempts" not in _final_stats(body)


def test_non_streamed_answer_offering_to_run_is_one_generation(monkeypatch, posture):
    prompts, payload = _serve(monkeypatch, COMPLETED_WITH_OFFER, stream=False)

    assert len(prompts) == 1
    message = payload["choices"][0]["message"]
    assert COMPLETED_WITH_OFFER in (message.get("content") or "")
    assert "stalled_agent_retry_attempted" not in (payload.get("mtplx_stats") or {})


def test_the_retry_is_gone_from_the_recovery_accounting():
    from mtplx.server.stream_recovery import _STREAM_RECOVERY_STAT_PREFIXES

    assert "stalled_agent_retry_" not in _STREAM_RECOVERY_STAT_PREFIXES
    assert "stalled_agent_retry_attempted" not in openai.PUBLIC_MTPLX_STATS_KEYS
