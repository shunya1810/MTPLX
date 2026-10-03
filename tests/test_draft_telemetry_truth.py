"""Draft-sampler telemetry must equal the engine object — the 2.8 truth lane.

Server-level truth tests for the variable-draft-temperature campaign:

- F1: the MTPLX_CLIENT launch env var is an observability LABEL; control
  ownership (managed-client policy) requires real per-request evidence
  (header/body hint or user agent).
- F2: a launch without a draft sampler resolves as target_mirror — the
  engine receives the mirrored draft sampler explicitly and the stamped
  draft temperature IS what the engine drafts with, never a "none"/null
  stamp over a target-temperature draft.
- F8: AR responses carry no draft-sampler telemetry at all (absent keys,
  not null-with-value).
- F9: the OpenCode server-side sampler normalization is a launch_default
  ownership tier, not request_explicit — the family curve and greedy
  coupling still run.
- F13: the mtp_batch cohort key includes the TARGET sampling triple, so
  temp-0 and temp-1 loads never share a cohort (both directions).
- F14: MTPLX_DRAFT_TEMPERATURE_SCALE is applied by the server resolver, so
  the stamped number is the effective number (the engine never rescales a
  server-resolved sampler).
- F17: finish_reason is present in mtplx_stats (declared keys exist), and
  the KL prompt-scoring lane decodes each unique token id once.

Every generation capture asserts stats == the engine object for the same
request: a stat that disagrees with the engine object is a lie.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from starlette.testclient import TestClient

from mtplx.server import openai
from mtplx.server.openai import SamplerConfig, create_app

from test_server_openai import (  # noqa: E402 - shared fixtures
    ForegroundState,
    _fake_state,
)


def _fake_generation_output():
    from mtplx.generation import GenerationStats

    stats = GenerationStats(
        mode="mtpk",
        generated_tokens=2,
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
        tokens=[79, 75],
        text="OK",
        stats=stats,
        final_state=None,
        finish_reason="stop",
    )


def _truth_client(
    monkeypatch,
    *,
    draft_sampler: SamplerConfig | None,
    captured: list[dict],
    pinned: bool = False,
    curve=None,
) -> tuple[TestClient, SimpleNamespace]:
    """Real _run_generation (the envelope/telemetry path under test), fake
    engine generators that capture the exact engine objects they receive."""

    monkeypatch.delenv("MTPLX_CLIENT", raising=False)
    monkeypatch.delenv("MTPLX_DRAFT_TEMPERATURE_SCALE", raising=False)
    state = _fake_state()
    foreground = ForegroundState()
    state.lock = foreground.lock
    state.begin_foreground = foreground.begin_foreground
    state.end_foreground = foreground.end_foreground
    state.has_foreground = foreground.has_foreground
    state.foreground_count = foreground.foreground_count
    state.requests_completed = 0
    state.requests_cancelled = 0
    state.last_request_at = 0.0
    state.last_request_started_at = 0.0
    state.active_requests = 0
    state.draft_sampler = draft_sampler
    state.draft_sampler_pinned = pinned
    state.draft_temperature_curve = curve
    monkeypatch.setattr(
        openai, "_encode_messages", lambda *_args, **_kwargs: [1, 2, 3]
    )

    def capture_mtpk(_runtime, _prompt_ids, **kwargs):
        captured.append(
            {
                "generator": "mtpk",
                "sampler": kwargs.get("sampler"),
                "draft_sampler": kwargs.get("draft_sampler"),
            }
        )
        return _fake_generation_output()

    def capture_ar(_runtime, _prompt_ids, **kwargs):
        captured.append(
            {
                "generator": "ar",
                "sampler": kwargs.get("sampler"),
                "draft_sampler": kwargs.get("draft_sampler", None),
            }
        )
        return _fake_generation_output()

    monkeypatch.setattr(openai, "generate_mtpk", capture_mtpk)
    monkeypatch.setattr(openai, "generate_ar", capture_ar)
    return TestClient(create_app(state)), state


def _chat(client: TestClient, body: dict, headers: dict | None = None) -> dict:
    response = client.post(
        "/v1/chat/completions",
        headers={"x-mtplx-cache-mode": "bypass", **(headers or {})},
        json={
            "model": "default",
            "messages": [{"role": "user", "content": "Reply OK only."}],
            "max_tokens": 8,
            **body,
        },
    )
    assert response.status_code == 200, response.text[:300]
    return response.json()


def _assert_stats_equal_engine(stats: dict, engine: dict) -> None:
    """The exactness-law gate: telemetry equals the engine object."""
    engine_draft = engine["draft_sampler"]
    assert engine_draft is not None
    assert stats["draft_sampler_resolved_temperature"] == pytest.approx(
        float(engine_draft.temperature)
    )


# ---------------------------------------------------------------------------
# F1 — launch env var labels, never owns controls
# ---------------------------------------------------------------------------


def test_launch_env_labels_but_never_owns_controls(monkeypatch):
    """A benchmarker's temp:0 against a hermes/app-launched daemon must be
    honored; the launch env var stays visible as the telemetry label."""

    captured: list[dict] = []
    client, _state = _truth_client(
        monkeypatch, draft_sampler=None, captured=captured
    )
    monkeypatch.setenv("MTPLX_CLIENT", "hermes")

    payload = _chat(client, {"temperature": 0.0})

    stats = payload["mtplx_stats"]
    assert captured[-1]["sampler"].temperature == 0.0
    assert stats["effective_temperature"] == 0.0
    assert stats["mtplx_control_owner"] == "client"
    # Ops value kept: the label still identifies the launch surface.
    assert stats["request_client_hint"] == "hermes"


def test_per_request_managed_hint_still_owns_controls(monkeypatch):
    captured: list[dict] = []
    client, state = _truth_client(
        monkeypatch, draft_sampler=None, captured=captured
    )
    monkeypatch.setenv("MTPLX_CLIENT", "hermes")

    payload = _chat(
        client,
        {"temperature": 0.0},
        headers={"x-mtplx-client": "mtplx_app"},
    )

    stats = payload["mtplx_stats"]
    launch_default = float(state.args.temperature)
    assert captured[-1]["sampler"].temperature == pytest.approx(launch_default)
    assert stats["mtplx_control_owner"] == "server"
    assert stats["request_client_hint"] == "mtplx_app"


def test_launch_env_does_not_mask_request_ua_identity(monkeypatch):
    monkeypatch.setenv("MTPLX_CLIENT", "hermes")
    hint = openai._request_client_hint_from_headers(
        {"user-agent": "claude-cli/1.0.44 (external, cli)"}, {}
    )
    assert hint == "claude_code"
    # Headerless requests keep the launch label.
    assert openai._request_client_hint_from_headers({}, {}) == "hermes"


def test_launch_env_never_reaches_managed_classification(monkeypatch):
    monkeypatch.setenv("MTPLX_CLIENT", "hermes")
    assert openai._app_managed_client_hint({}, {}) is None
    assert openai._client_controls_allowed({}, {}) is True
    # Real per-request evidence still classifies managed.
    assert (
        openai._app_managed_client_hint({"x-mtplx-client": "mtplx_app"}, {})
        == "mtplx_app"
    )


# ---------------------------------------------------------------------------
# F2 + telemetry==engine — consecutive-request desync sequences
# ---------------------------------------------------------------------------


def test_consecutive_requests_mirror_target_without_pinning(monkeypatch):
    """target_mirror follows each request's temperature (1.0 -> 0.6 -> 0 ->
    1.0); telemetry equals the engine object every time (mirror, not pin)."""

    monkeypatch.delenv("MTPLX_GREEDY_DRAFT_COUPLING", raising=False)
    captured: list[dict] = []
    client, _state = _truth_client(
        monkeypatch, draft_sampler=None, captured=captured
    )

    for temperature in (1.0, 0.6, 0.0, 1.0):
        payload = _chat(client, {"temperature": temperature})
        stats = payload["mtplx_stats"]
        engine = captured[-1]
        assert engine["generator"] == "mtpk"
        assert engine["draft_sampler"] is not None, (
            "engine must receive the mirrored draft sampler explicitly"
        )
        assert float(engine["draft_sampler"].temperature) == pytest.approx(
            temperature
        )
        _assert_stats_equal_engine(stats, engine)
        assert stats["draft_sampler_policy"] == "target_mirror"
        assert "target_mirror" in stats["draft_sampler_policy_source"]


def test_consecutive_requests_launch_draft_desync_sequence(monkeypatch):
    """Launch draft sampler present: 1.0 -> 0.6 -> 0 (greedy coupled) -> 1.0
    with telemetry equal to the engine object each time."""

    monkeypatch.delenv("MTPLX_GREEDY_DRAFT_COUPLING", raising=False)
    captured: list[dict] = []
    launch = SamplerConfig(temperature=0.6, top_p=0.95, top_k=20)
    client, _state = _truth_client(
        monkeypatch, draft_sampler=launch, captured=captured
    )

    for target, expected_draft in (
        (1.0, 0.6),
        (0.6, 0.6),
        (0.0, 0.0),
        (1.0, 0.6),
    ):
        payload = _chat(client, {"temperature": target})
        stats = payload["mtplx_stats"]
        engine = captured[-1]
        assert float(engine["draft_sampler"].temperature) == pytest.approx(
            expected_draft
        )
        _assert_stats_equal_engine(stats, engine)
        if target == 0.0:
            assert stats["draft_sampler_policy_source"].endswith(
                "+greedy_coupled"
            )
            assert stats["draft_sampler_greedy_coupled"] is True


def test_greedy_launch_draft_arm_real_resolver_golden_shape(monkeypatch):
    """Golden-style greedy arm through the REAL resolver (no monkeypatched
    resolution): pinned key subset for a temp-0 request against a launched
    draft sampler."""

    monkeypatch.delenv("MTPLX_GREEDY_DRAFT_COUPLING", raising=False)
    captured: list[dict] = []
    launch = SamplerConfig(temperature=0.6, top_p=0.95, top_k=20)
    client, _state = _truth_client(
        monkeypatch, draft_sampler=launch, captured=captured
    )

    payload = _chat(client, {"temperature": 0.0})
    stats = payload["mtplx_stats"]

    golden_subset = {
        "effective_temperature": 0.0,
        "draft_sampler_policy": "static",
        "draft_sampler_policy_source": "family_default+greedy_coupled",
        "draft_sampler_resolved_temperature": 0.0,
        "draft_sampler_greedy_coupled": True,
    }
    assert {key: stats.get(key) for key in golden_subset} == golden_subset
    assert float(captured[-1]["draft_sampler"].temperature) == 0.0


# ---------------------------------------------------------------------------
# F8 — AR responses carry no draft-sampler telemetry
# ---------------------------------------------------------------------------


def test_ar_response_has_no_draft_sampler_telemetry(monkeypatch):
    captured: list[dict] = []
    launch = SamplerConfig(temperature=0.6, top_p=0.95, top_k=20)
    client, _state = _truth_client(
        monkeypatch, draft_sampler=launch, captured=captured
    )

    payload = _chat(client, {"generation_mode": "ar"})
    stats = payload["mtplx_stats"]

    assert captured[-1]["generator"] == "ar"
    draft_keys = [key for key in stats if key.startswith("draft_sampler")]
    assert draft_keys == [], f"AR response leaked draft telemetry: {draft_keys}"
    assert stats["draft_time_s"] == 0.0
    assert stats["generation_mode"] == "ar"


def test_mtp_response_keeps_draft_sampler_telemetry(monkeypatch):
    captured: list[dict] = []
    launch = SamplerConfig(temperature=0.6, top_p=0.95, top_k=20)
    client, _state = _truth_client(
        monkeypatch, draft_sampler=launch, captured=captured
    )

    payload = _chat(client, {})
    stats = payload["mtplx_stats"]
    assert stats["draft_sampler_resolved_temperature"] == pytest.approx(0.6)
    assert stats["draft_sampler_policy"] == "static"
    _assert_stats_equal_engine(stats, captured[-1])


# ---------------------------------------------------------------------------
# F9 — OpenCode server-side normalization is launch_default, not
# request_explicit; the curve and greedy coupling still run
# ---------------------------------------------------------------------------

_OPENCODE_HEADERS = {
    "x-mtplx-client": "opencode",
    "user-agent": "opencode/1.4.2",
}

_TOOLS_OPENAI = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]


def test_opencode_injection_is_launch_default_tier_and_runs_curve(monkeypatch):
    """The headline agent client: server-injected sampler normalization must
    not freeze the draft sampler as request_explicit — the family curve maps
    the effective target temperature and the result is stamped truthfully."""

    monkeypatch.delenv("MTPLX_GREEDY_DRAFT_COUPLING", raising=False)
    captured: list[dict] = []
    launch = SamplerConfig(temperature=1.0, top_p=0.95, top_k=20)
    curve = ((0.2, 0.1), (0.6, 0.4), (1.0, 1.0))
    client, state = _truth_client(
        monkeypatch, draft_sampler=launch, captured=captured, curve=curve
    )
    # A 0.6-family launch: the fixture's default model now boots at its own
    # 1.0, where this curve is the identity and could not show it ran.
    state.args.temperature = 0.6

    payload = _chat(
        client,
        {
            "messages": [{"role": "user", "content": "Weather in Paris?"}],
            "max_tokens": 64,
            "tools": _TOOLS_OPENAI,
            "tool_choice": "auto",
        },
        headers=_OPENCODE_HEADERS,
    )
    stats = payload["mtplx_stats"]
    engine = captured[-1]

    # Target normalized to the launched defaults (0.6 family sampler).
    assert stats["effective_temperature"] == pytest.approx(
        float(state.args.temperature)
    )
    # The tier is visible and it is NOT request_explicit.
    assert stats["draft_sampler_ownership"] == "launch_default"
    assert stats["draft_sampler_policy"] != "request_explicit"
    # The curve ran: target 0.6 -> draft 0.4, engine object matches.
    assert float(engine["draft_sampler"].temperature) == pytest.approx(0.4)
    _assert_stats_equal_engine(stats, engine)


def test_opencode_explicit_sampler_stays_server_owned_and_loud(monkeypatch):
    """OpenCode is a MANAGED surface: its body sampler params are server-
    owned by design (2.5.3 contract). The ignore must be LOUD in telemetry
    — client_sampler_fields_ignored — and the draft telemetry must still
    equal the engine object for the curated sampler actually used."""

    monkeypatch.delenv("MTPLX_GREEDY_DRAFT_COUPLING", raising=False)
    captured: list[dict] = []
    launch = SamplerConfig(temperature=0.6, top_p=0.95, top_k=20)
    client, state = _truth_client(
        monkeypatch, draft_sampler=launch, captured=captured
    )

    payload = _chat(
        client,
        {
            "messages": [{"role": "user", "content": "Weather in Paris?"}],
            "max_tokens": 64,
            "tools": _TOOLS_OPENAI,
            "tool_choice": "auto",
            "temperature": 0.0,
            "top_k": 1,
        },
        headers=_OPENCODE_HEADERS,
    )
    stats = payload["mtplx_stats"]
    engine = captured[-1]
    # Server-owned: the curated launch sampler ran, and the ignore is
    # explicit, never silent.
    assert stats["mtplx_control_owner"] == "server"
    assert "temperature" in stats["client_sampler_fields_ignored"]
    assert captured[-1]["sampler"].temperature == pytest.approx(
        float(state.args.temperature)
    )
    # The draft telemetry equals the engine object for that curated target.
    assert float(engine["draft_sampler"].temperature) == pytest.approx(0.6)
    _assert_stats_equal_engine(stats, engine)


def test_count_tokens_accepts_opencode_override_shaped_body(monkeypatch):
    captured: list[dict] = []
    launch = SamplerConfig(temperature=0.6, top_p=0.95, top_k=20)
    client, _state = _truth_client(
        monkeypatch, draft_sampler=launch, captured=captured
    )

    response = client.post(
        "/v1/messages/count_tokens",
        headers=_OPENCODE_HEADERS,
        json={
            "model": "default",
            "max_tokens": 64,
            "temperature": 0.55,
            "top_p": 1.0,
            "messages": [{"role": "user", "content": "Weather in Paris?"}],
            "tools": [
                {
                    "name": "get_weather",
                    "description": "Get weather for a city",
                    "input_schema": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                }
            ],
        },
    )
    assert response.status_code == 200, response.text[:300]
    assert response.json() == {"input_tokens": 3}
    assert captured == []  # counting never generates


# ---------------------------------------------------------------------------
# F13 — cohort key includes the target sampling triple
# ---------------------------------------------------------------------------


def test_cohort_key_separates_target_temperatures_both_directions():
    draft = SamplerConfig(temperature=0.0, top_p=0.95, top_k=20)
    lane = SimpleNamespace(route_id="route-a")
    greedy_target = SamplerConfig(temperature=0.0, top_p=1.0, top_k=1)
    sampled_target = SamplerConfig(temperature=1.0, top_p=0.95, top_k=20)

    key_greedy_first = openai._mtp_batch_compatibility_key(
        lane, False, greedy_target, draft
    )
    key_sampled_second = openai._mtp_batch_compatibility_key(
        lane, False, sampled_target, draft
    )
    assert key_greedy_first != key_sampled_second

    # Other direction: sampled load arrives first, greedy joins later.
    key_sampled_first = openai._mtp_batch_compatibility_key(
        lane, False, sampled_target, draft
    )
    key_greedy_second = openai._mtp_batch_compatibility_key(
        lane, False, greedy_target, draft
    )
    assert key_sampled_first != key_greedy_second
    assert key_sampled_first == key_sampled_second
    assert key_greedy_first == key_greedy_second


def test_cohort_key_still_separates_draft_samplers():
    lane = SimpleNamespace(route_id="route-a")
    target = SamplerConfig(temperature=0.6, top_p=0.95, top_k=20)
    draft_a = SamplerConfig(temperature=0.1, top_p=0.95, top_k=20)
    draft_b = SamplerConfig(temperature=0.6, top_p=0.95, top_k=20)
    assert openai._mtp_batch_compatibility_key(
        lane, False, target, draft_a
    ) != openai._mtp_batch_compatibility_key(lane, False, target, draft_b)


# ---------------------------------------------------------------------------
# F14 — the stamped draft temperature is the effective one under the scale
# ---------------------------------------------------------------------------


def test_scale_knob_is_applied_before_the_stamp(monkeypatch):
    captured: list[dict] = []
    launch = SamplerConfig(temperature=0.8, top_p=0.95, top_k=20)
    client, _state = _truth_client(
        monkeypatch, draft_sampler=launch, captured=captured
    )
    monkeypatch.setenv("MTPLX_DRAFT_TEMPERATURE_SCALE", "0.5")

    payload = _chat(client, {"temperature": 0.6})
    stats = payload["mtplx_stats"]
    engine = captured[-1]

    # The server hands the engine the ALREADY-scaled sampler and stamps
    # that same number: stats == engine object under the knob.
    assert float(engine["draft_sampler"].temperature) == pytest.approx(0.4)
    assert stats["draft_sampler_resolved_temperature"] == pytest.approx(0.4)
    _assert_stats_equal_engine(stats, engine)


def test_generation_no_longer_rescales_the_resolved_sampler(monkeypatch):
    """The server resolver owns the knob; the generation-side effective-
    sampler helper passes a resolved sampler through untouched and keeps
    the pure mirror fallback for direct engine callers."""

    from mtplx import generation

    monkeypatch.setenv("MTPLX_DRAFT_TEMPERATURE_SCALE", "0.5")
    sampler = generation.SamplerConfig(temperature=0.6, top_p=0.95, top_k=20)
    resolved = generation.SamplerConfig(temperature=0.4, top_p=0.95, top_k=20)
    effective = generation._effective_draft_sampler(sampler, resolved)
    assert float(effective.temperature) == pytest.approx(0.4)
    # Mirror fallback for direct callers stays — and never rescales.
    mirrored = generation._effective_draft_sampler(sampler, None)
    assert float(mirrored.temperature) == pytest.approx(0.6)


def test_scale_knob_invalid_or_nonpositive_is_ignored(monkeypatch):
    captured: list[dict] = []
    launch = SamplerConfig(temperature=0.8, top_p=0.95, top_k=20)
    client, _state = _truth_client(
        monkeypatch, draft_sampler=launch, captured=captured
    )
    monkeypatch.setenv("MTPLX_DRAFT_TEMPERATURE_SCALE", "not-a-number")

    payload = _chat(client, {"temperature": 0.6})
    assert float(captured[-1]["draft_sampler"].temperature) == pytest.approx(0.8)
    assert payload["mtplx_stats"][
        "draft_sampler_resolved_temperature"
    ] == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# F17 — declared stats exist; the KL lane decodes each unique id once
# ---------------------------------------------------------------------------


def test_finish_reason_is_present_in_mtplx_stats(monkeypatch):
    captured: list[dict] = []
    client, _state = _truth_client(
        monkeypatch, draft_sampler=None, captured=captured
    )
    payload = _chat(client, {})
    assert payload["mtplx_stats"]["finish_reason"] == "stop"


def test_prompt_scoring_decodes_each_unique_token_once(monkeypatch):
    """The KL lane decodes token ids through a memo: the number of
    tokenizer.decode calls is bounded by the number of UNIQUE ids, not by
    positions x top_k (~65k calls at long context before the fix). The
    per-token decomposition (exact offsets) is unchanged."""

    monkeypatch.delenv("MTPLX_CLIENT", raising=False)
    state = _fake_state()
    foreground = ForegroundState()
    state.lock = foreground.lock
    state.begin_foreground = foreground.begin_foreground
    state.end_foreground = foreground.end_foreground
    state.has_foreground = foreground.has_foreground
    state.foreground_count = foreground.foreground_count
    state.requests_completed = 0
    state.requests_cancelled = 0
    state.last_request_at = 0.0
    state.last_request_started_at = 0.0
    state.active_requests = 0

    decode_calls: list[list[int]] = []

    def counting_decode(tokens, **_kwargs):
        decode_calls.append(list(tokens))
        return "".join(chr(96 + (int(token) % 26) + 1) for token in tokens)

    prompt_ids = [1, 2, 1, 2, 3]
    state.runtime.tokenizer = SimpleNamespace(
        decode=counting_decode,
        encode=lambda _text, **_kwargs: list(prompt_ids),
    )

    def fake_score(_runtime, ids, *, top_k, abort_check=None, prefill_callback=None):
        n = len(ids)
        positions = []
        for i in range(n - 1):
            positions.append([(ids[i + 1], -0.1), (ids[0], -2.0)][: top_k or 2])
        return {
            "positions": positions,
            "token_logprobs": [-0.1] * (n - 1),
            "prompt_tokens": n,
            "elapsed_s": 0.01,
        }

    monkeypatch.setattr(openai, "score_prompt_logprobs", fake_score)
    client = TestClient(create_app(state))

    response = client.post(
        "/v1/completions",
        json={
            "prompt": "abcab",
            "echo": True,
            "logprobs": 2,
            "max_tokens": 0,
            "temperature": 0,
        },
    )
    assert response.status_code == 200, response.text[:300]
    logprobs = response.json()["choices"][0]["logprobs"]

    # Exactness of the per-token decomposition is unchanged.
    assert logprobs["tokens"] == ["b", "c", "b", "c", "d"]
    assert logprobs["text_offset"] == [0, 1, 2, 3, 4]
    assert response.json()["choices"][0]["text"] == "bcbcd"
    assert logprobs["token_ids"] == prompt_ids

    # Every decode call is a single token, and each unique id decodes once.
    assert all(len(call) == 1 for call in decode_calls)
    unique_ids = {1, 2, 3}
    assert len(decode_calls) == len(unique_ids), (
        f"expected one decode per unique id, saw {len(decode_calls)} calls"
    )
