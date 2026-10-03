"""Working set between turns: the MLX buffer pool is returned to macOS after
every completed request (class A: clear_cache never touches a live array)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from mtplx.server import openai


@pytest.fixture
def cleared(monkeypatch):
    calls: list[str] = []

    def fake_clear(_state, *, reason, lock_wait_s=0.0):
        calls.append(reason)
        return {"cleared": True, "reason": reason}

    monkeypatch.setattr(openai, "_clear_mlx_cache_after_request", fake_clear)
    return calls


@pytest.mark.parametrize(
    "observability, session_id",
    [
        ({"request_client_evidence": "opencode"}, "sess-1"),
        ({}, None),
        ({"request_client_evidence": "aime"}, None),
    ],
)
def test_every_completed_request_releases_the_pool_by_default(
    monkeypatch, cleared, observability, session_id
):
    monkeypatch.delenv("MTPLX_CLEAR_CACHE_AFTER_REQUEST", raising=False)

    result = openai._auto_clear_mlx_cache_after_completed_request(
        SimpleNamespace(), session_id=session_id, request_observability=observability
    )

    # On 50de43bb only a stateless AIME question cleared; an agent turn kept
    # up to 8 GiB of freed buffers resident until the next request.
    assert result == {"cleared": True, "reason": "after_request"}
    assert cleared == ["after_request"]


def test_off_keeps_the_pool_and_aime_restricts_it(monkeypatch, cleared):
    monkeypatch.setenv("MTPLX_CLEAR_CACHE_AFTER_REQUEST", "off")
    assert (
        openai._auto_clear_mlx_cache_after_completed_request(
            SimpleNamespace(), session_id="s", request_observability={}
        )
        is None
    )
    monkeypatch.setenv("MTPLX_CLEAR_CACHE_AFTER_REQUEST", "aime")
    assert (
        openai._auto_clear_mlx_cache_after_completed_request(
            SimpleNamespace(),
            session_id="s",
            request_observability={"request_client_evidence": "opencode"},
        )
        is None
    )
    assert cleared == []


def test_run_generation_releases_the_pool_and_reports_it(monkeypatch, cleared):
    from test_server_openai import _fake_streaming_session_state

    monkeypatch.delenv("MTPLX_CLEAR_CACHE_AFTER_REQUEST", raising=False)
    state = _fake_streaming_session_state()
    state.draft_sampler = None
    state.requests_completed = 0

    def fake_generate_mtpk(*_args, **_kwargs):
        return SimpleNamespace(
            tokens=[ord("O")],
            text="O",
            stats=SimpleNamespace(
                to_dict=lambda: {
                    "prompt_eval_time_s": 0.0,
                    "generated_tokens": 1,
                    "elapsed_s": 0.01,
                    "tok_s": 100.0,
                }
            ),
            final_state=None,
        )

    monkeypatch.setattr(openai, "generate_mtpk", fake_generate_mtpk)
    out = openai._run_generation(
        state,
        [1, 2, 3],
        max_tokens=1,
        temperature=None,
        top_p=None,
        top_k=None,
        seed=None,
        generation_mode="mtp",
        depth=3,
        session_id="sess-agent",
        session_bank=state.sessions.bank,
        session_template_hash=state.template_hash,
        session_draft_head_identity=state.draft_head_identity,
        session_policy_fingerprint="policy",
        request_observability={"request_client_evidence": "opencode"},
    )

    assert cleared == ["after_request"]
    assert out["stats"]["mlx_cache_cleanup"] == {
        "cleared": True,
        "reason": "after_request",
    }
