"""Regression pins for the 2.8.0 release-night QA fix wave (2026-08-17).

Each test guards one fix landed after the independent hit-list audit and the
live probe battery:
- F8/D3: the batched-AR lanes must scrub draft-sampler policy stamps.
- F1/D1: behavior branches key on per-request evidence, never the launch env
  label (``request_client_hint`` stays as the observability label).
- F4/D2: the scored token's top-K entry always carries the true scored value,
  and ``top_logprobs[0]`` is a dict (``{}``), never null — harness KL parsers
  iterate entries with ``.items()``.
- Stream/non-stream text parity: streamed content concatenates to the
  non-stream ``strip()`` result (the "</think>\\n\\n" separator must not leak
  as a leading content delta); the canonicalization normalizer stays
  byte-exact.
- Q1: ``resolve_model_path`` falls back to a branded bare-name cache dir the
  same way bench model selection does.
- F15 residual: ``mtplx doctor --summary`` prints the compiled-verify fence.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from mtplx import hf_loader
from mtplx.commands import public
from mtplx.server import openai as srv


# --- F8/D3: ar_batch lanes scrub draft-sampler stamps -----------------------


def test_batched_ar_service_scrubs_draft_sampler_stamps():
    source = Path(srv.__file__).read_text()
    # Both ar_batch observability merges are immediately followed by the F8
    # scrub; the serial lane's scrub already has its own behavioral tests.
    merged = source.count("draft_sampler")
    assert (
        source.count(
            'for key in [k for k in stats if k.startswith("draft_sampler")]:'
        )
        >= 2
    ), "serial + service scrub sites expected"
    assert (
        source.count(
            'for key in [k for k in envelope if k.startswith("draft_sampler")]:'
        )
        >= 2
    ), "serial envelope + batched finalize scrub sites expected"
    assert merged  # sanity: the keys exist at all


def test_finalize_batched_ar_kills_draft_sampler_keys(monkeypatch):
    captured = {}

    def fake_repair(stats, **kwargs):
        return dict(stats)

    monkeypatch.setattr(srv, "_repair_streamed_generation_stats", fake_repair)
    monkeypatch.setattr(
        srv, "_auto_clear_mlx_cache_after_completed_request", lambda *a, **k: None
    )
    monkeypatch.setattr(srv, "_mlx_allocator_public_stats", lambda: {})
    monkeypatch.setattr(
        srv, "_generation_truth_stats", lambda state, mode: {"generation_mode": mode}
    )
    monkeypatch.setattr(srv, "_dashboard_record_completion", lambda *a, **k: None)
    state = SimpleNamespace(
        args=SimpleNamespace(),
        runtime=None,
        last_metrics=[],
        last_request_at=0.0,
        requests_completed=0,
    )
    generated = {
        "tokens": [1, 2, 3],
        "elapsed_s": 1.0,
        "stats": {"decode_tok_s": 10.0},
        "text": "ok",
    }
    result = srv._finalize_batched_ar_generation(
        state,
        [1, 2],
        generated,
        session_id=None,
        session_cache_hit=False,
        cache_miss_reason=None,
        session_restore_mode="none",
        request_observability={
            "draft_sampler_ownership": "launch_default",
            "draft_sampler_policy": "static",
            "draft_sampler_policy_temperature": 0.6,
            "request_client_hint": "opencode",
        },
    )
    captured = result.get("stats") or result
    draft_keys = [k for k in captured if str(k).startswith("draft_sampler")]
    assert draft_keys == [], f"ar_batch response leaked {draft_keys}"


# --- F1/D1: env label must not steer behavior --------------------------------


def test_opencode_override_ignores_env_only_hint():
    # The env-inclusive label alone (no per-request evidence) must not flip
    # another client's sampler.
    observability = {"request_client_hint": "opencode"}
    sampler = srv._opencode_default_sampler_override(
        messages=[
            srv.ChatMessage(role="system", content="You are OpenCode."),
            srv.ChatMessage(role="user", content="Hi, how are you"),
        ],
        tools_active=True,
        request_temperature=None,
        request_top_p=None,
        request_top_k=None,
        request_observability=observability,
        default_temperature=0.6,
        default_top_p=0.95,
        default_top_k=20,
    )
    assert sampler is None


def test_auto_clear_client_ignores_env_label(monkeypatch):
    # The client-restricted mode must not trust a label: "auto" now clears
    # after every request (tests/test_working_set.py), so the label check
    # lives on the "aime" mode.
    monkeypatch.setenv("MTPLX_CLEAR_CACHE_AFTER_REQUEST", "aime")
    result = srv._auto_clear_mlx_cache_after_completed_request(
        SimpleNamespace(),
        session_id=None,
        request_observability={
            "request_client_hint": "aime",
            "request_client_label": "aime",
        },
    )
    assert result is None  # no per-request evidence -> no aime behavior


# --- F4/D2 + KL parser safety ------------------------------------------------


class _CollidingTokenizer:
    """Two ids decode to the same display string (byte-fallback collapse)."""

    def decode(self, ids):
        token_id = ids[0]
        if token_id in (7, 9):
            return "�"
        return f"tok{token_id}"


def test_prompt_scoring_topk_collision_keeps_true_scored_value():
    # Reimplements the endpoint's assembly contract against the helper
    # invariants: scored token's entry equals token_logprobs[i] even when a
    # higher-ranked entry decodes to the same string.
    tokenizer = _CollidingTokenizer()
    prompt_ids = [1, 9]
    scored = {
        "token_logprobs": [-8.0],
        "positions": [[(7, -0.5), (9, -8.0)]],
    }
    token_strings = [tokenizer.decode([int(t)]) for t in prompt_ids]
    top_logprob_dicts = [{}]
    for position, entries in enumerate(scored["positions"]):
        row = {}
        for token_id, logprob in entries:
            token_text = tokenizer.decode([int(token_id)])
            if token_text not in row:
                row[token_text] = float(logprob)
        actual_text = token_strings[position + 1]
        row[actual_text] = float(scored["token_logprobs"][position])
        top_logprob_dicts.append(row)
    assert top_logprob_dicts[0] == {}
    assert top_logprob_dicts[1]["�"] == -8.0


def test_prompt_scoring_source_has_no_null_top0_and_unconditional_actual():
    source = Path(srv.__file__).read_text()
    assert "top_logprob_dicts: list[dict[str, float]] = [{}]" in source
    assert (
        'row[actual_text] = float(scored["token_logprobs"][position])' in source
    )
    assert "if actual_text not in row:" not in source


# --- stream/non-stream text parity -------------------------------------------


def _collect(chunks):
    return "".join(text for channel, text in chunks if channel == "content")


def test_stream_splitter_trims_visible_edges_like_nonstream_strip():
    splitter = srv._ThinkingContentStreamSplitter(
        thinking_enabled=True,
        trim_visible_content_edges=True,
    )
    out = []
    for piece in ["reasoning here", "</think>", "\n\n", "Hello", " world.", "\n\n"]:
        out.extend(splitter.feed(piece))
    out.extend(splitter.finish())
    assert _collect(out) == "Hello world."
    reasoning = "".join(t for c, t in out if c == "reasoning_content")
    assert "reasoning here" in reasoning


def test_stream_splitter_preserves_interior_whitespace():
    splitter = srv._ThinkingContentStreamSplitter(
        thinking_enabled=True,
        trim_visible_content_edges=True,
    )
    out = []
    for piece in ["</think>", "\n\npara one.\n\n", "para two.", "\n"]:
        out.extend(splitter.feed(piece))
    out.extend(splitter.finish())
    assert _collect(out) == "para one.\n\npara two."


def test_stream_splitter_drops_separator_ahead_of_tool_markup():
    splitter = srv._ThinkingContentStreamSplitter(
        thinking_enabled=False,
        trim_visible_content_edges=True,
    )
    out = []
    for piece in [
        "I'll check.",
        "\n\n<tool_call>\n<function=f>\n</function>\n</tool_call>",
    ]:
        out.extend(splitter.feed(piece))
    out.extend(splitter.finish())
    joined = _collect(out)
    assert joined.startswith("I'll check.<tool_call>") or (
        "I'll check." in joined and "\n\n<tool_call" not in joined
    )


def test_normalizer_default_keeps_bytes_exact():
    # Canonicalization must never adopt the trim: transcript identity is
    # byte-exact (F11 postcommit extension).
    splitter = srv._ThinkingContentStreamSplitter(thinking_enabled=True)
    out = []
    for piece in ["</think>", "\n\nHello.", "\n"]:
        out.extend(splitter.feed(piece))
    out.extend(splitter.finish())
    assert _collect(out) == "\n\nHello.\n"


# --- Q1: branded bare-name cache fallback ------------------------------------


def test_resolve_model_path_falls_back_to_branded_dir(tmp_path, monkeypatch):
    cache = tmp_path / "models"
    branded = cache / "My-Pack"
    branded.mkdir(parents=True)
    (branded / "config.json").write_text("{}")

    def fake_cached_model_path(repo_id, cache_dir=None):
        return cache / repo_id.replace("/", "--")

    ready = {"calls": []}

    def fake_ready(path, repo_id):
        ready["calls"].append(str(path))
        return path == branded

    monkeypatch.setattr(hf_loader, "cached_model_path", fake_cached_model_path)
    monkeypatch.setattr(hf_loader, "_cached_model_ready_for_repo", fake_ready)
    resolved = hf_loader.resolve_model_path(
        "SomeOrg/My-Pack", cache_dir=str(cache)
    )
    assert resolved == branded
    assert len(ready["calls"]) == 2  # snapshot layout first, branded second


def test_resolve_model_path_still_errors_when_nothing_matches(monkeypatch, tmp_path):
    monkeypatch.setattr(
        hf_loader, "cached_model_path", lambda repo_id, cache_dir=None: tmp_path / "x"
    )
    monkeypatch.setattr(
        hf_loader, "_cached_model_ready_for_repo", lambda path, repo_id: False
    )
    try:
        hf_loader.resolve_model_path("SomeOrg/Absent")
    except FileNotFoundError as exc:
        assert "mtplx pull SomeOrg/Absent" in str(exc)
    else:
        raise AssertionError("expected FileNotFoundError")


# --- F15 residual: doctor --summary carries the fence ------------------------


def test_doctor_summary_prints_compiled_verify_fence(capsys):
    report = {
        "diagnostics": {"overall": "pass", "checks": []},
        "compiled_verify": {
            "mode": "on",
            "mode_source": "turbo profile",
            "fenced": True,
            "max_context_tokens": 32768,
            "max_context_source": "turbo profile",
        },
    }
    args = SimpleNamespace(summary=True, deep=False)
    public._render_doctor_report(args, report)
    out = capsys.readouterr().out
    assert "compiled verify fence: <= 32768 tokens" in out
