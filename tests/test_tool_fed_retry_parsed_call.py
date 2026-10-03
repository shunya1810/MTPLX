"""A well-formed bare tool call after tool results is a call, not orphan markup.

Before the fix, a call whose arguments strip to one short token
(``count_lines`` with ``part=2``) was classified ``orphan_tool_control_markup``
and the stream worker re-ran the turn with a nudge message, re-prefilling the
history. These tests pin the exemption, its kill switch, and that genuine
orphan tails still retry, and so do calls the permissive parser only half
reads (an unclosed parameter, a complete call followed by an unfinished
one), sampled or greedy.
"""

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from mtplx.server import openai
from mtplx.server.openai import create_app
from tests.test_server_openai import _fake_streaming_session_state, _stream_payloads

BARE_CALL = (
    "\n</think>\n\n<tool_call>\n<function=count_lines>\n<parameter=part>\n2\n"
    "</parameter>\n</function>\n</tool_call>"
)
ORPHAN_TAIL = "parameter=limit>\n180\n</parameter>\n</function>\n</tool_call>"
ANSWER = "</think>\n\nPart 2 has 93 lines."
BARE_READ = (
    "\n</think>\n\n<tool_call>\n<function=read>\n<parameter=path>\n"
    "/tmp/notes.txt\n</parameter>\n</function>\n</tool_call>"
)
# The parser reads read(offset=2) and drops the unclosed path parameter.
UNCLOSED_PARAMETER = (
    "\n</think>\n\n<tool_call><function=read><parameter=offset>2</parameter>"
    "<parameter=path></function></tool_call>"
)
# The parser reads the first call and drops the unfinished second envelope.
COMPLETE_THEN_TRUNCATED = BARE_CALL + (
    "\n<tool_call>\n<function=count_lines>\n<parameter=part>\n3"
)


def _count_lines_tool():
    return {
        "type": "function",
        "function": {
            "name": "count_lines",
            "description": "Count the lines in one part of the notes.",
            "parameters": {
                "type": "object",
                "properties": {"part": {"type": "integer"}},
                "required": ["part"],
            },
        },
    }


def _read_tool():
    return {
        "type": "function",
        "function": {
            "name": "read",
            "description": "Read a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "offset": {"type": "integer"},
                },
                "required": ["path"],
            },
        },
    }


def _run(monkeypatch, texts, **request_fields):
    state = _fake_streaming_session_state()
    state.args.stream_interval = 1
    client = TestClient(create_app(state))
    calls: list[str] = []

    def fake_run_generation(_state, prompt_ids, **kwargs):
        text = texts[len(calls)]
        calls.append(text)
        tokens = [ord(char) for char in text]
        token_callback = kwargs.get("token_callback")
        if token_callback is not None:
            for token in tokens:
                token_callback([token])
        return {
            "text": text,
            "tokens": tokens,
            "stats": {
                **(kwargs.get("request_observability") or {}),
                "generation_mode": kwargs["generation_mode"],
                "mtp_depth": kwargs["depth"],
                "completion_tokens": len(tokens),
                "decode_tok_s": 22.0,
            },
            "prompt_tokens": len(prompt_ids),
            "completion_tokens": len(tokens),
            "finish_reason": "stop",
        }

    monkeypatch.setattr(openai, "_run_generation", fake_run_generation)
    with client.stream(
        "POST",
        "/v1/chat/completions",
        headers={"x-mtplx-cache-mode": "bypass"},
        json={
            "messages": [
                {"role": "user", "content": "Count the lines of each part."},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "count_lines",
                                "arguments": '{"part": 1}',
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "Part 1 has 93 lines."},
            ],
            "tools": [_count_lines_tool(), _read_tool()],
            "stream": True,
            "max_tokens": 128,
            "enable_thinking": True,
            **request_fields,
        },
    ) as response:
        body = "".join(response.iter_text())
    assert response.status_code == 200
    payloads = _stream_payloads(body)
    final = [p for p in payloads if p["choices"][0]["finish_reason"]]
    return calls, payloads, final[-1]


def _tool_call_names(payloads):
    return [
        (call.get("function") or {}).get("name")
        for payload in payloads
        for choice in payload.get("choices", [])
        for call in (choice.get("delta") or {}).get("tool_calls") or []
        if (call.get("function") or {}).get("name")
    ]


def test_heuristic_alone_flags_bare_call_as_orphan():
    # The pre-existing heuristic is unchanged; the exemption sits beside it.
    assert (
        openai._tool_fed_degenerate_completion_reason(BARE_CALL)
        == "orphan_tool_control_markup"
    )


def test_bare_well_formed_call_is_not_retried(monkeypatch):
    monkeypatch.delenv("MTPLX_TOOL_FED_RETRY_PARSED_CALL_GUARD", raising=False)
    calls, payloads, final = _run(monkeypatch, [BARE_CALL, ANSWER])

    assert calls == [BARE_CALL]
    assert _tool_call_names(payloads) == ["count_lines"]
    assert final["choices"][0]["finish_reason"] == "tool_calls"
    assert not final["mtplx_stats"].get("tool_fed_empty_retry_attempted")


def test_kill_switch_restores_retry(monkeypatch):
    monkeypatch.setenv("MTPLX_TOOL_FED_RETRY_PARSED_CALL_GUARD", "0")
    calls, _payloads, final = _run(monkeypatch, [BARE_CALL, ANSWER])

    assert calls == [BARE_CALL, ANSWER]
    assert final["mtplx_stats"]["tool_fed_empty_retry_attempted"] is True
    assert final["mtplx_stats"]["tool_fed_empty_retry_reason"] == (
        "orphan_tool_control_markup"
    )


def test_orphan_tail_without_call_still_retries(monkeypatch):
    monkeypatch.delenv("MTPLX_TOOL_FED_RETRY_PARSED_CALL_GUARD", raising=False)
    calls, _payloads, final = _run(monkeypatch, [ORPHAN_TAIL, ANSWER])

    assert calls == [ORPHAN_TAIL, ANSWER]
    assert final["mtplx_stats"]["tool_fed_empty_retry_attempted"] is True


def test_plain_single_path_read_is_not_retried(monkeypatch):
    monkeypatch.delenv("MTPLX_TOOL_FED_RETRY_PARSED_CALL_GUARD", raising=False)
    assert (
        openai._tool_fed_degenerate_completion_reason(BARE_READ)
        == "orphan_tool_control_markup"
    )
    calls, payloads, final = _run(monkeypatch, [BARE_READ, ANSWER])

    assert calls == [BARE_READ]
    assert _tool_call_names(payloads) == ["read"]
    assert final["choices"][0]["finish_reason"] == "tool_calls"
    assert not final["mtplx_stats"].get("tool_fed_empty_retry_attempted")


@pytest.mark.parametrize(
    "malformed",
    [UNCLOSED_PARAMETER, COMPLETE_THEN_TRUNCATED],
    ids=["unclosed_parameter", "complete_then_truncated"],
)
@pytest.mark.parametrize("temperature", [None, 0.0], ids=["sampled", "greedy"])
def test_partly_parsed_call_still_gets_the_repair(monkeypatch, malformed, temperature):
    """The parser recovers a call from both texts, but neither is a
    well-formed call, so the stream re-runs the turn with the repair
    message. Greedy included: the repair changes the prompt, so it is not
    the identical-prompt blank retry that greedy decodes skip."""
    monkeypatch.delenv("MTPLX_TOOL_FED_RETRY_PARSED_CALL_GUARD", raising=False)
    assert (
        openai._tool_fed_degenerate_completion_reason(malformed)
        == "orphan_tool_control_markup"
    )
    fields = {} if temperature is None else {"temperature": temperature}
    calls, payloads, final = _run(monkeypatch, [malformed, ANSWER], **fields)

    assert calls == [malformed, ANSWER]
    assert final["mtplx_stats"]["tool_fed_empty_retry_attempted"] is True
    assert final["mtplx_stats"]["tool_fed_empty_retry_reason"] == (
        "orphan_tool_control_markup"
    )
    assert _tool_call_names(payloads) == []
    assert final["choices"][0]["finish_reason"] == "stop"


def test_greedy_well_formed_call_is_not_retried(monkeypatch):
    monkeypatch.delenv("MTPLX_TOOL_FED_RETRY_PARSED_CALL_GUARD", raising=False)
    calls, payloads, final = _run(
        monkeypatch, [BARE_CALL, ANSWER], temperature=0.0
    )

    assert calls == [BARE_CALL]
    assert _tool_call_names(payloads) == ["count_lines"]
    assert not final["mtplx_stats"].get("tool_fed_empty_retry_attempted")


CALL = (
    "<tool_call>\n<function=count_lines>\n<parameter=part>\n2\n</parameter>\n"
    "</function>\n</tool_call>"
)
NO_ARGUMENTS = "<tool_call>\n<function=list_parts>\n</function>\n</tool_call>"
JSON_CALL = '<tool_call>{"name": "count_lines", "arguments": {"part": 2}}</tool_call>'
# an unclosed parameter swallowed into the next parameter's value
SWALLOWED_PARAMETER = (
    "<tool_call>\n<function=read>\n<parameter=path>\n<parameter=offset>\n2\n"
    "</parameter>\n</function>\n</tool_call>"
)
TWO_FUNCTIONS = (
    "<tool_call>\n<function=count_lines>\n<function=read>\n</function>\n"
    "</tool_call>"
)


@pytest.mark.parametrize(
    ("reasoning", "content", "call_count", "complete"),
    [
        ("\n", CALL, 1, True),
        ("\n", CALL + "\n" + CALL.replace("2", "3"), 2, True),
        ("\n", NO_ARGUMENTS, 1, True),
        ("", JSON_CALL, 1, True),
        (CALL, "", 1, True),
        ("\n", SWALLOWED_PARAMETER, 1, False),
        ("\n", CALL + "\n</parameter>", 1, False),
        ("\n", "2" + CALL, 1, False),
        ("\n", CALL + "\n" + CALL, 1, False),
        ("<parameter=path>", CALL, 1, False),
        ("\n", TWO_FUNCTIONS, 1, False),
        ("\n", "<tool_call>count_lines(part=2)</tool_call>", 1, False),
    ],
)
def test_call_markup_completeness(reasoning, content, call_count, complete):
    assert (
        openai._tool_fed_call_markup_is_complete(
            reasoning, content, call_count=call_count
        )
        is complete
    )
