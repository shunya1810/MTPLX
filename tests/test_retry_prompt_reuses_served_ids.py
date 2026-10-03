"""A server-side recovery pass reuses the prompt ids the request was served.

The tool-fed empty retry and the opt-in read-only force-answer retry used to
re-render the whole transcript plus their repair turn. That render lost the
committed-token splice (the model's own tokenization of earlier turns), the
committed reasoning substitution and the expanded image pads, so the retry
prompt diverged from the session's KV at the first seam: on 2026-09-29 at
token 16,371 of 46,861, restoring 4,096 tokens. The retry prompt is now the
served ids up to the generation prompt, then the repair turn; when no such
prompt exists the first pass stands and the stats say why.
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
from mtplx.server.retry_prompt import served_prompt_with_appended_turn

GEN = [90, 91]  # the template's generation prompt
TURN = [70, 71, 72]  # the appended user turn, as the template renders it


class TestHelper:
    def test_append_only_template_keeps_every_served_id(self):
        history = [1, 2, 3, 4, 5, 6]
        plain = history + GEN
        appended = history + TURN + GEN
        # The splice served tokens 2..4 as one id of the model's own.
        served = [1, 999, 5, 6] + GEN

        ids, receipt = served_prompt_with_appended_turn(served, plain, appended)

        assert ids == [1, 999, 5, 6] + TURN + GEN
        assert receipt["reused_served_tokens"] == 4
        assert receipt["appended_turn_tokens"] == len(TURN) + len(GEN)

    def test_a_shared_leading_token_stays_on_the_served_side(self):
        # `<|im_start|>` opens both the generation prompt and the new turn.
        history = [1, 2, 3]
        plain = history + [8, 90]
        appended = history + [8, 70, 71, 8, 90]
        served = [1, 999] + [8, 90]

        ids, receipt = served_prompt_with_appended_turn(served, plain, appended)

        assert ids == [1, 999, 8, 70, 71, 8, 90]
        assert receipt["replaced_tokens"] == 1

    def test_the_visible_working_close_stays_last(self):
        history = [1, 2, 3]
        close = [55, 56]
        served = [1, 999] + GEN + close

        ids, _receipt = served_prompt_with_appended_turn(
            served, history + GEN, history + TURN + GEN, served_suffix_ids=close
        )

        assert ids == [1, 999] + TURN + GEN + close

    def test_expanded_image_pads_before_the_turn_are_kept_as_served(self):
        # The plain render has one pad per image; the served prompt has the
        # expanded run the request's vision rows were built for.
        plain = [1, 77, 2] + GEN
        appended = [1, 77, 2] + TURN + GEN
        served = [1, 77, 77, 77, 77, 2] + GEN

        ids, _receipt = served_prompt_with_appended_turn(served, plain, appended)

        assert ids == [1, 77, 77, 77, 77, 2] + TURN + GEN

    def test_a_template_that_rerenders_history_reuses_the_served_prefix(self):
        # Appending a user turn makes this template drop the reasoning of the
        # last assistant turn ([40, 41]); the served ids end like the plain
        # render there, so everything before the drop is reused.
        plain = [1, 2, 3, 40, 41, 5] + GEN
        appended = [1, 2, 3, 5] + TURN + GEN
        served = [1, 999, 40, 41, 5] + GEN

        ids, receipt = served_prompt_with_appended_turn(served, plain, appended)

        assert ids == [1, 999, 5] + TURN + GEN
        assert receipt["reused_served_tokens"] == 2

    def test_no_exact_prompt_when_the_served_tail_differs(self):
        # The splice reached into the part the template re-renders.
        plain = [1, 2, 3, 40, 41, 5] + GEN
        appended = [1, 2, 3, 5] + TURN + GEN
        served = [1, 2, 3, 998, 5] + GEN

        ids, receipt = served_prompt_with_appended_turn(served, plain, appended)

        assert ids is None
        assert receipt["reason"] == "served_prompt_tail_differs_from_template"


ORPHAN_TAIL = "</think>\n\nparameter=limit>\n180\n</parameter>\n</function>\n</tool_call>"
ANSWER = "</think>\n\nImplemented the HUD cleanup and verified npm build."
SPLICED = 0x1F600  # an id the plain render never produces


def _tool_fed_request() -> dict:
    return {
        "messages": [
            {"role": "user", "content": "Improve this project after reading the files."},
            {
                "role": "assistant",
                "content": "I will read the game file first.",
                "tool_calls": [
                    {
                        "id": "call_read",
                        "type": "function",
                        "function": {"name": "session_status", "arguments": "{}"},
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


def _run(monkeypatch, texts, *, splice):
    """Stream one request whose served prompt carries a committed-token
    splice (``splice`` maps the raw render to the served ids)."""

    state = _fake_streaming_session_state()
    state.args.stream_interval = 1
    state.args.stats_footer = False
    client = TestClient(create_app(state))
    prompts: list[list[int]] = []

    def fake_canonicalize(_state, *, messages, prompt_ids, **_kwargs):
        return list(messages), splice(list(prompt_ids))

    def fake_run_generation(_state, prompt_ids, **kwargs):
        text = texts[len(prompts)]
        prompts.append(list(prompt_ids))
        tokens = [ord(char) for char in text]
        callback = kwargs.get("token_callback")
        if callback is not None:
            for token in tokens:
                callback([token])
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

    monkeypatch.setattr(openai, "_maybe_canonicalize_committed_reasoning", fake_canonicalize)
    monkeypatch.setattr(openai, "_run_generation", fake_run_generation)
    with client.stream(
        "POST",
        "/v1/chat/completions",
        headers={"x-mtplx-session-id": "retry-session", "x-mtplx-client": "pi"},
        json=_tool_fed_request(),
    ) as response:
        body = "".join(response.iter_text())
    assert response.status_code == 200
    final = [
        payload
        for payload in _stream_payloads(body)
        if payload.get("choices") and payload["choices"][0].get("finish_reason")
    ]
    return prompts, final[-1].get("mtplx_stats") or {}


def _splice(text: str):
    """Serve ``text`` as one id, as the committed-token splice serves the
    model's own tokenization of an earlier turn."""

    marker = [ord(char) for char in text]

    def splice(ids: list[int]) -> list[int]:
        for start in range(len(ids) - len(marker) + 1):
            if ids[start : start + len(marker)] == marker:
                return ids[:start] + [SPLICED] + ids[start + len(marker) :]
        raise AssertionError(f"{text!r} is not in the served prompt")

    return splice


# The earlier assistant turn's text (the committed-token splice's own
# ground), and the first user message (present under the forced-answer
# contract too, which rewrites the assistant turns of a read-only loop).
_splice_first_turn = _splice("I will")
_splice_first_user_message = _splice("Improve this")


def _generation_prompt_len() -> int:
    return len("assistant:")


def test_tool_fed_retry_prompt_keeps_the_served_splice(monkeypatch):
    monkeypatch.delenv("MTPLX_AGENT_REWRITES", raising=False)

    prompts, stats = _run(monkeypatch, [ORPHAN_TAIL, ANSWER], splice=_splice_first_turn)

    assert len(prompts) == 2
    served, retry = prompts
    assert SPLICED in served
    kept = len(served) - _generation_prompt_len()
    # The retry extends the served ids byte for byte up to the generation
    # prompt; the old fresh render dropped the spliced id.
    assert retry[:kept] == served[:kept]
    retry_text = "".join(chr(token) for token in retry[kept:] if token != SPLICED)
    assert "Complete the active coding task now." in retry_text
    assert retry_text.endswith("assistant:")
    assert stats["tool_fed_empty_retry_attempted"] is True
    assert stats["tool_fed_empty_retry_reused_prompt_tokens"] == kept


def test_tool_fed_retry_stands_down_when_no_exact_prompt_exists(monkeypatch):
    monkeypatch.delenv("MTPLX_AGENT_REWRITES", raising=False)

    def served_differs_at_the_end(ids: list[int]) -> list[int]:
        return ids[:-1] + [SPLICED]

    prompts, stats = _run(
        monkeypatch, [ORPHAN_TAIL, ANSWER], splice=served_differs_at_the_end
    )

    assert len(prompts) == 1
    assert stats["tool_fed_empty_retry_skipped"] == (
        "served_prompt_tail_differs_from_template"
    )
    assert "tool_fed_empty_retry_attempted" not in stats


def test_read_only_force_answer_retry_keeps_the_served_prompt(monkeypatch):
    monkeypatch.setenv("MTPLX_AGENT_REWRITES", "on")
    monkeypatch.setattr(openai, "_is_read_only_inspection_request", lambda _text: True)
    monkeypatch.setattr(
        openai,
        "_request_should_force_answer_for_read_only_inspection",
        lambda _messages: True,
    )

    prompts, stats = _run(
        monkeypatch,
        ["</think>\n\nLet me inspect another file first.", ANSWER],
        splice=_splice_first_user_message,
    )

    assert len(prompts) == 2
    served, retry = prompts
    assert SPLICED in served
    kept = len(served) - _generation_prompt_len()
    # The old retry rendered the transcript again (and without the served
    # tool contract): the spliced id was gone from its prompt.
    assert retry[:kept] == served[:kept]
    assert stats["read_only_force_answer_retry_attempted"] is True
