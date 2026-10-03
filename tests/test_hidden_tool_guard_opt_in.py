"""The hidden-tool stream guard is opt-in (MTPLX_STREAM_HIDDEN_TOOL_GUARD=on).

The guard cancels a stream whose tool call has buffered a token count and a
wall time outside a known parameter and sends "malformed tool_call:
unterminated stream". On 2026-09-29 it cancelled an 82 s Pi answer that was
most likely a long valid edit (the raw text was not kept), after misfiring on
a Cline write (2026-07-25) and on JSON-dialect writes (#196). A count and a
clock cannot tell a runaway from a long valid call, and a generation stop must
have zero false positives, so the default is off.

Every test here runs with the ceilings forced to 4 tokens and 0 s, far below
the stamped 2,048 tokens / 30 s: on the old code each default-posture stream
below is cancelled mid-call. The opt-in keeps the old behaviour, and a zero
ceiling is never the off switch.
"""

from __future__ import annotations

import json
import time

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient
from test_server_openai import (
    CaptureTokenizer,
    _fake_state,
    _stream_payloads,
    _tool_schema,
    _write_tool_schema,
)

from mtplx.server import openai
from mtplx.server.openai import create_app

GUARD_ERROR = "malformed tool_call: unterminated stream"
FILE_BODY = "\n".join(
    [
        "<!DOCTYPE html>",
        "<html>",
        "<body>",
        "<canvas id=\"game\" width=\"480\" height=\"640\"></canvas>",
        "<script>const gravity = 0.45; let pipes = [];</script>",
        "</body>",
        "</html>",
    ]
    * 60
)
XML_WRITE = (
    "<tool_call>\n<function=write>\n"
    "<parameter=filePath>\nflappy/index.html\n</parameter>\n"
    f"<parameter=content>\n{FILE_BODY}\n</parameter>\n"
    "</function>\n</tool_call>"
)
JSON_WRITE = (
    "<tool_call>\n<function=write>"
    + json.dumps({"filePath": "flappy/index.html", "content": FILE_BODY})
    + "</function>\n</tool_call>"
)
UNCLOSED = "<tool_call>\n<function=session_status>\n" + "x" * 64
# A shorter write for the slow-producer case (one pause per batch).
SMALL_BODY = "\n".join(["<script>let score = 0; const gap = 160;</script>"] * 40)
XML_WRITE_SMALL = (
    "<tool_call>\n<function=write>\n"
    "<parameter=filePath>\nflappy/index.html\n</parameter>\n"
    f"<parameter=content>\n{SMALL_BODY}\n</parameter>\n"
    "</function>\n</tool_call>"
)


@pytest.fixture
def tiny_ceilings(monkeypatch):
    monkeypatch.setattr(openai, "STREAM_HIDDEN_TOOL_GUARD_TOKENS", 4)
    monkeypatch.setattr(openai, "STREAM_HIDDEN_TOOL_GUARD_S", 0.0)


def _engine(
    text: str,
    *,
    chunk: int = 1,
    pause_s: float = 0.0,
    cancel_at: int | None = None,
    await_cancel_at: int | None = None,
):
    """A fake engine that streams ``text`` in ``chunk``-character batches,
    optionally pausing between batches and tripping the request's cancel
    event (a client disconnect) after ``cancel_at`` characters. With
    ``await_cancel_at`` it waits there (up to 5 s) for the stream to cancel
    it, so a consumer-side stop is observed deterministically. It records
    how far it got and whether the stream stopped it."""

    record: dict = {"sent": 0, "stopped_by": None, "completed": False}

    def fake_run_generation(_state, _prompt_ids, **kwargs):
        callback = kwargs.get("token_callback")
        cancel_event = kwargs.get("cancel_event")
        tokens = [ord(char) for char in text]
        try:
            for start in range(0, len(tokens), chunk):
                if cancel_at is not None and start >= cancel_at and cancel_event is not None:
                    cancel_event.set_origin("client_disconnect")
                if (
                    await_cancel_at is not None
                    and start == await_cancel_at
                    and cancel_event is not None
                ):
                    cancel_event.wait(5.0)
                if callback is not None:
                    callback(tokens[start : start + chunk])
                record["sent"] = start + chunk
                if pause_s:
                    time.sleep(pause_s)
        except openai._StreamCancelled:
            record["stopped_by"] = getattr(cancel_event, "origin", None)
            raise
        record["completed"] = True
        return {
            "text": text,
            "tokens": tokens,
            "stats": {
                "generation_mode": kwargs["generation_mode"],
                "mtp_depth": kwargs["depth"],
                "completion_tokens": len(tokens),
            },
            "prompt_tokens": 3,
            "completion_tokens": len(tokens),
            "finish_reason": "stop",
        }

    return fake_run_generation, record


def _stream(monkeypatch, text: str, tools: list[dict], **engine):
    # The release pacer spreads a multi-token batch over the expected gap;
    # the guard is checked once a batch has drained either way, so it is
    # switched off here to keep the batched cases fast.
    monkeypatch.setenv("MTPLX_STREAM_PACER", "0")
    state = _fake_state()
    state.runtime.tokenizer = CaptureTokenizer()
    state.args.stream_interval = 1
    state.args.stats_footer = False
    fake, record = _engine(text, **engine)
    monkeypatch.setattr(openai, "_run_generation", fake)
    client = TestClient(create_app(state))
    with client.stream(
        "POST",
        "/v1/chat/completions",
        headers={"x-mtplx-cache-mode": "bypass"},
        json={
            "messages": [{"role": "user", "content": "Write the game file."}],
            "tools": tools,
            "tool_choice": "auto",
            "stream": True,
            "max_tokens": 8192,
        },
    ) as response:
        body = "".join(response.iter_text())
    assert response.status_code == 200
    return body, record


def _tool_arguments(body: str) -> dict:
    arguments = "".join(
        item.get("function", {}).get("arguments", "")
        for payload in _stream_payloads(body)
        for choice in payload.get("choices", [])
        for item in choice.get("delta", {}).get("tool_calls", [])
    )
    return json.loads(arguments)


def _finish_reason(body: str) -> str | None:
    final = [
        payload
        for payload in _stream_payloads(body)
        if payload.get("choices") and payload["choices"][0].get("finish_reason")
    ]
    return final[-1]["choices"][0]["finish_reason"] if final else None


@pytest.fixture
def default_posture(monkeypatch):
    monkeypatch.delenv("MTPLX_STREAM_HIDDEN_TOOL_GUARD", raising=False)


@pytest.mark.parametrize(
    ("text", "body_text", "chunk", "pause_s"),
    [
        (XML_WRITE, FILE_BODY, 1, 0.0),
        (JSON_WRITE, FILE_BODY, 1, 0.0),
        # Tags split across batches at odd widths.
        (XML_WRITE, FILE_BODY, 7, 0.0),
        # A slow producer: a 20 ms gap after every 16 characters, against a
        # 10 ms ceiling (decode stalls and verify hiccups inside a call).
        (XML_WRITE_SMALL, SMALL_BODY, 16, 0.02),
    ],
    ids=["large_xml_parameter", "large_json_body", "split_tags", "pauses"],
)
def test_long_valid_tool_calls_are_never_cancelled_by_default(
    monkeypatch, tiny_ceilings, default_posture, text, body_text, chunk, pause_s
):
    if pause_s:
        monkeypatch.setattr(openai, "STREAM_HIDDEN_TOOL_GUARD_S", pause_s / 2)
    body, record = _stream(
        monkeypatch, text, [_write_tool_schema()], chunk=chunk, pause_s=pause_s
    )

    assert record["completed"] is True
    assert GUARD_ERROR not in body
    assert _finish_reason(body) == "tool_calls"
    arguments = _tool_arguments(body)
    assert arguments["content"] == body_text
    assert arguments["filePath"] == "flappy/index.html"


def test_a_malformed_ending_runs_to_the_models_own_end_by_default(
    monkeypatch, tiny_ceilings, default_posture
):
    body, record = _stream(monkeypatch, UNCLOSED, [_tool_schema()])

    # The model finished its turn; the end-of-stream tool parser, not a
    # clock, decides what the unclosed call becomes.
    assert record["completed"] is True
    assert record["stopped_by"] is None
    assert GUARD_ERROR not in body
    assert "data: [DONE]" in body


def test_a_client_cancel_mid_call_still_stops_the_generation(
    monkeypatch, tiny_ceilings, default_posture
):
    body, record = _stream(
        monkeypatch, XML_WRITE, [_write_tool_schema()], chunk=64, cancel_at=512
    )

    assert record["completed"] is False
    assert record["stopped_by"] == "client_disconnect"
    assert GUARD_ERROR not in body


def test_the_opt_in_keeps_the_runaway_backstop(monkeypatch, tiny_ceilings):
    monkeypatch.setenv("MTPLX_STREAM_HIDDEN_TOOL_GUARD", "on")

    body, record = _stream(
        monkeypatch, UNCLOSED, [_tool_schema()], await_cancel_at=48
    )

    assert GUARD_ERROR in body
    assert "data: [DONE]" in body
    assert record["completed"] is False
    assert record["stopped_by"] == "hidden_tool_guard"


@pytest.mark.parametrize("value", ["", "0", "off", "false", "no"])
def test_only_the_flag_arms_the_guard_never_the_ceilings(monkeypatch, tiny_ceilings, value):
    # Zero ceilings mean "cancel at once" for an armed guard; they are not
    # an off switch, and without the flag they never cancel anything.
    monkeypatch.setattr(openai, "STREAM_HIDDEN_TOOL_GUARD_TOKENS", 0)
    if value:
        monkeypatch.setenv("MTPLX_STREAM_HIDDEN_TOOL_GUARD", value)
    else:
        monkeypatch.delenv("MTPLX_STREAM_HIDDEN_TOOL_GUARD", raising=False)
    assert openai._stream_hidden_tool_guard_enabled() is False

    body, record = _stream(monkeypatch, UNCLOSED, [_tool_schema()])

    assert record["completed"] is True
    assert GUARD_ERROR not in body


@pytest.mark.parametrize("value", ["1", "on", "true", "yes", "ON"])
def test_the_flag_values_that_arm_it(monkeypatch, value):
    monkeypatch.setenv("MTPLX_STREAM_HIDDEN_TOOL_GUARD", value)
    assert openai._stream_hidden_tool_guard_enabled() is True
