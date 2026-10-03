"""Stream-worker recovery passes: which prompt a repair continues, and stats.

Chain under test (seedless tool-fed request): pass 1 ends in orphan tool
markup, the tool-fed empty retry (pass 2, retry prompt with an appended
steering turn) stops inside its thinking, and the reasoning-completion repair
(pass 3) closes the thinking. Pass 2's tokens were generated under the retry
prompt, so the repair must continue that prompt; and the final stats must
account for all three passes.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient
from test_server_openai import (
    _fake_streaming_session_state,
    _stream_payloads,
    _tool_schema,
)

from mtplx.server import openai
from mtplx.server.openai import (
    PUBLIC_MTPLX_STATS_KEYS,
    _run_stream_recovery_chain,
    create_app,
)

ORPHAN_TAIL = "parameter=limit>\n180\n</parameter>\n</function>\n</tool_call>"
THINKING_ONLY = "The user wants the cleanup finished. Let me compose the summary now."
ANSWER = "Implemented the HUD cleanup and verified npm build."

PASS_STATS = (
    {"prompt_eval_time_s": 2.0, "new_prefill_tokens": 400, "ttft_s": 2.1},
    {"prompt_eval_time_s": 3.0, "new_prefill_tokens": 900, "ttft_s": 5.3},
    {"prompt_eval_time_s": 0.25, "new_prefill_tokens": 51, "ttft_s": 5.9},
)


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


def _run_chain(monkeypatch, texts, *, session_id=None):
    state = _fake_streaming_session_state()
    state.args.stream_interval = 1
    state.args.stats_footer = False
    client = TestClient(create_app(state))
    prompts: list[list[int]] = []
    batch_keys: list[str] = []

    def fake_run_generation(_state, prompt_ids, **kwargs):
        index = len(prompts)
        text = texts[index]
        prompts.append(list(prompt_ids))
        observability = kwargs.get("request_observability") or {}
        batch_keys.append(_pass_kind(observability))
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
                "decode_tok_s": 20.0,
                **PASS_STATS[index],
            },
            "prompt_tokens": len(prompt_ids),
            "completion_tokens": len(tokens),
            "finish_reason": "stop",
        }

    monkeypatch.setattr(openai, "_run_generation", fake_run_generation)
    headers = {"x-mtplx-client": "pi"}
    if session_id is None:
        headers["x-mtplx-cache-mode"] = "bypass"
    else:
        headers["x-mtplx-session-id"] = session_id
    with client.stream(
        "POST",
        "/v1/chat/completions",
        headers=headers,
        json=_tool_fed_request(),
    ) as response:
        body = "".join(response.iter_text())
    assert response.status_code == 200
    payloads = _stream_payloads(body)
    final = [p for p in payloads if p["choices"][0]["finish_reason"]]
    return prompts, batch_keys, final[-1]["mtplx_stats"]


def _is_prefix(prefix: list[int], ids: list[int]) -> bool:
    return ids[: len(prefix)] == prefix


def test_reasoning_repair_after_retry_continues_the_retry_prompt(monkeypatch):
    prompts, batch_keys, _stats = _run_chain(
        monkeypatch, [ORPHAN_TAIL, THINKING_ONLY, ANSWER]
    )

    assert batch_keys == [
        "first",
        "tool_fed_empty_retry",
        "reasoning_completion_repair",
    ]
    original, retry, repair = prompts
    assert not _is_prefix(retry, original)
    # Repair = retry prompt + pass-2 tokens + closing suffix.
    assert _is_prefix(retry + [ord(c) for c in THINKING_ONLY], repair)
    assert not _is_prefix(original + [ord(c) for c in THINKING_ONLY], repair)


def test_named_session_recovery_keeps_the_retry_context_and_totals(monkeypatch):
    prompts, kinds, stats = _run_chain(
        monkeypatch,
        [ORPHAN_TAIL, THINKING_ONLY, ANSWER],
        session_id="recovery-regression",
    )
    assert kinds == ["first", "tool_fed_empty_retry", "reasoning_completion_repair"]
    assert _is_prefix(prompts[1] + [ord(c) for c in THINKING_ONLY], prompts[2])
    assert stats["stream_attempts"] == 3
    assert stats["stream_attempts_new_prefill_tokens"] == 1351


def test_reasoning_repair_always_uses_the_prompt_that_produced_its_tokens(monkeypatch):
    monkeypatch.setenv("MTPLX_REASONING_REPAIR_FOLLOWS_RETRY_PROMPT", "0")

    prompts, _batch_keys, _stats = _run_chain(
        monkeypatch, [ORPHAN_TAIL, THINKING_ONLY, ANSWER]
    )

    _original, retry, repair = prompts
    assert _is_prefix(retry + [ord(c) for c in THINKING_ONLY], repair)


def test_reasoning_repair_without_retry_still_uses_request_prompt(monkeypatch):
    prompts, batch_keys, _stats = _run_chain(monkeypatch, [THINKING_ONLY, ANSWER])

    assert batch_keys == ["first", "reasoning_completion_repair"]
    assert _is_prefix(prompts[0] + [ord(c) for c in THINKING_ONLY], prompts[1])


def test_final_stats_account_for_every_pass(monkeypatch):
    _prompts, _batch_keys, stats = _run_chain(
        monkeypatch, [ORPHAN_TAIL, THINKING_ONLY, ANSWER]
    )

    assert stats["stream_attempts"] == 3
    assert stats["stream_attempts_prompt_eval_time_s"] == pytest.approx(5.25)
    assert stats["stream_attempts_new_prefill_tokens"] == 1351
    assert stats["stream_attempts_completion_tokens"] == (
        len(ORPHAN_TAIL) + len(THINKING_ONLY) + len(ANSWER)
    )
    assert stats["stream_attempts_first_ttft_s"] == pytest.approx(2.1)
    # Existing fields keep describing the last pass.
    assert stats["prompt_eval_time_s"] == pytest.approx(0.25)
    assert stats["new_prefill_tokens"] == 51
    # The retry's fields survive the repair that followed it.
    assert stats["tool_fed_empty_retry_attempted"] is True
    assert stats["tool_fed_empty_retry_reason"] == "orphan_tool_control_markup"
    assert stats["tool_fed_empty_retry_succeeded"] is True
    assert stats["reasoning_completion_repair_attempted"] is True


def test_single_pass_envelope_has_no_attempt_totals(monkeypatch):
    _prompts, batch_keys, stats = _run_chain(monkeypatch, [ANSWER])

    assert batch_keys == ["first"]
    assert not any(key.startswith("stream_attempts") for key in stats)


def test_attempt_totals_are_public_stats_keys():
    for key in (
        "stream_attempts",
        "stream_attempts_first_ttft_s",
        "stream_attempts_prompt_eval_time_s",
        "stream_attempts_new_prefill_tokens",
        "stream_attempts_completion_tokens",
    ):
        assert key in PUBLIC_MTPLX_STATS_KEYS


def test_recovery_chain_keeps_last_pass_values_and_updates_metrics():
    state = SimpleNamespace(last_metrics=[{"request_id": "r"}])
    first = {
        "completion_tokens": 3,
        "stats": {"prompt_eval_time_s": 1.0, "ttft_s": 1.5, "tool_fed_empty_retry_x": 1},
    }
    second = {
        "completion_tokens": 4,
        "stats": {"request_id": "r", "prompt_eval_time_s": 0.5, "new_prefill_tokens": 7, "ttft_s": 4.0},
    }

    result = _run_stream_recovery_chain(
        state, first, [lambda g: g, lambda g: second, lambda g: g]
    )

    assert result is second
    assert result["stats"]["ttft_s"] == 4.0
    assert result["stats"]["prompt_eval_time_s"] == 0.5
    assert result["stats"]["tool_fed_empty_retry_x"] == 1
    assert result["stats"]["stream_attempts"] == 2
    assert result["stats"]["stream_attempts_prompt_eval_time_s"] == pytest.approx(1.5)
    assert result["stats"]["stream_attempts_new_prefill_tokens"] == 7
    assert result["stats"]["stream_attempts_completion_tokens"] == 7
    assert state.last_metrics[-1]["stream_attempts"] == 2
    assert state.last_metrics[-1]["tool_fed_empty_retry_x"] == 1


def test_recovery_totals_do_not_overwrite_another_requests_metrics():
    state = SimpleNamespace(last_metrics=[{"request_id": "r"}, {"request_id": "other"}])
    first = {"completion_tokens": 3, "stats": {"request_id": "r"}}
    second = {"completion_tokens": 4, "stats": {"request_id": "r"}}
    _run_stream_recovery_chain(state, first, [lambda _: second])
    assert state.last_metrics[0]["stream_attempts_completion_tokens"] == 7
    assert state.last_metrics[1] == {"request_id": "other"}


def test_recovery_chain_releases_discarded_generation_state():
    """Only scalar receipts may survive into later passes, never old KV."""
    import weakref

    class CacheState:
        pass

    old_state = CacheState()
    old_ref = weakref.ref(old_state)

    def first_result():
        nonlocal old_state
        result = {"_final_state": old_state, "completion_tokens": 3, "stats": {}}
        old_state = None
        return result

    def third_pass(second):
        assert old_ref() is None, "discarded pass retains its model cache"
        return {"completion_tokens": 5, "stats": {}}

    # Match the worker: its local still references the initial result while
    # the helper runs all recovery steps.
    first = first_result()
    result = _run_stream_recovery_chain(
        SimpleNamespace(last_metrics=[]),
        first,
        [lambda first: {"completion_tokens": 4, "stats": {}}, third_pass],
    )
    assert result["stats"]["stream_attempts_completion_tokens"] == 12
