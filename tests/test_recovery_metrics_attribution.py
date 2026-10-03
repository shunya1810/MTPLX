"""An interleaved completion must not receive another request's retry flags."""

from copy import deepcopy

import pytest
from fastapi.testclient import TestClient
from test_server_openai import _fake_streaming_session_state
from test_stream_recovery_paths import ANSWER, ORPHAN_TAIL, THINKING_ONLY, _tool_fed_request

from mtplx.server import openai
from mtplx.server.stream_recovery import _STREAM_RECOVERY_STAT_PREFIXES


@pytest.mark.parametrize("keep_request_metric", [True, False], ids=["row_present", "row_evicted"])
@pytest.mark.parametrize(("kind", "first_text"), [
    ("inspection_empty_retry", ""),
    ("tool_fed_empty_retry", ORPHAN_TAIL),
    ("reasoning_completion_repair", THINKING_ONLY),
    ("read_only_force_answer_retry", "</think>\n\nLet me inspect another file first."),
])
def test_actual_retry_writers_keep_interleaved_metrics_separate(
    monkeypatch, keep_request_metric, kind, first_text
):
    monkeypatch.setenv("MTPLX_AGENT_REWRITES", "on")
    # Select the two read-only postures without making this attribution test
    # depend on the inspection classifier's phrase vocabulary.
    monkeypatch.setattr(openai, "_is_read_only_inspection_request", lambda _text:
                        kind in {"inspection_empty_retry", "read_only_force_answer_retry"})
    monkeypatch.setattr(openai, "_request_should_force_answer_for_read_only_inspection",
                        lambda _messages: kind == "read_only_force_answer_retry")
    state = _fake_streaming_session_state()
    state.args.stream_interval = 1
    state.args.stats_footer = False
    state.last_metrics = []
    other = {"request_id": "interleaved-request", "completion_tokens": 7}
    expected_other = deepcopy(other)
    attempts = []
    request_metric = {}
    retry_text = ANSWER if kind == "reasoning_completion_repair" else "</think>\n\n" + ANSWER

    def generate(_state, prompt_ids, **kwargs):
        obs = kwargs["request_observability"]
        attempt = "retry" if obs.get(f"{kind}_attempted") else "first"
        attempts.append(attempt)
        assert attempts in (["first"], ["first", "retry"])
        text = first_text if attempt == "first" else retry_text
        tokens = [ord(char) for char in text]
        if kwargs.get("token_callback"):
            kwargs["token_callback"](tokens)
        stats = {
            **obs, "generation_mode": kwargs["generation_mode"],
            "mtp_depth": kwargs["depth"], "completion_tokens": len(tokens),
            "decode_tok_s": 20.0,
        }
        if attempt == "retry":
            request_metric.update(stats)
            if keep_request_metric:
                state.last_metrics.append(request_metric)
            # B completes after A's generation publishes its metric, before
            # A's real retry function resumes and writes its recovery fields.
            state.last_metrics.append(other)
        return {
            "text": text, "tokens": tokens, "stats": stats,
            "prompt_tokens": len(prompt_ids), "completion_tokens": len(tokens),
            "finish_reason": "stop",
        }

    monkeypatch.setattr(openai, "_run_generation", generate)
    response = TestClient(openai.create_app(state)).post(
        "/v1/chat/completions",
        headers={"x-mtplx-cache-mode": "bypass", "x-mtplx-client": "pi"},
        json=_tool_fed_request(),
    )

    assert response.status_code == 200
    assert attempts == ["first", "retry"]
    assert other == expected_other
    if keep_request_metric:
        assert request_metric[f"{kind}_succeeded"] is True
        assert request_metric["stream_attempts"] == 2
        assert request_metric["stream_attempts_completion_tokens"] == len(first_text) + len(retry_text)
    assert not any(key.startswith(_STREAM_RECOVERY_STAT_PREFIXES) for key in other)
