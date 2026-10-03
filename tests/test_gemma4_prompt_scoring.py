"""Prompt scoring (echo + logprobs + max_tokens=0) on the Gemma 4 pair.

``Gemma4AssistantRuntime`` has no ``forward_ar``, so ``score_prompt_logprobs``
raised AttributeError and the server answered HTTP 500. Scoring now runs the
same target forward as the Gemma generation prefill and applies the logits
head per chunk. No-model harness: a causal stub target (hidden row i depends
on tokens 0..i only), in the style of test_tail_gemma4_stream_holdback.
"""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

import mtplx.backends.gemma4_assistant as gemma4
from mtplx.generation import score_prompt_logprobs

VOCAB = 128
HIDDEN = 8


class _StubTarget:
    def __init__(self, weight: mx.array) -> None:
        self.weight = weight
        self.head_rows: list[int] = []

    def cache_offset(self, cache):
        return len(cache[0]["tokens"])

    def logits_from_hidden(self, hidden):
        self.head_rows.append(int(hidden.shape[1]))
        # Elementwise product + sum instead of a GPU matmul: plain float32,
        # so the numpy reference matches to rounding.
        return (hidden[..., :, None] * self.weight).sum(axis=-2)


class _StubGemmaRuntime:
    """Gemma pair double: forward_target and the logits head, no forward_ar."""

    backend_id = "gemma4_assistant"
    mtp_enabled = True

    def __init__(self, seed: int = 0, tokenizer=None) -> None:
        rng = np.random.default_rng(seed)
        self.embed = mx.array(rng.standard_normal((VOCAB, HIDDEN)).astype(np.float32))
        self.target = _StubTarget(
            mx.array(rng.standard_normal((HIDDEN, VOCAB)).astype(np.float32) * 3.0)
        )
        self.tokenizer = tokenizer
        self.forward_calls: list[dict] = []

    def make_cache(self):
        return [{"tokens": []}]

    def forward_target(self, input_ids, *, cache=None, phase="unknown", compute_logits=True):
        ids = np.asarray(input_ids).reshape(-1).tolist()
        self.forward_calls.append(
            {"tokens": ids, "phase": phase, "compute_logits": compute_logits}
        )
        cache[0]["tokens"].extend(ids)
        prefix = cache[0]["tokens"]
        emb = self.embed[mx.array(prefix)]
        steps = mx.arange(1, len(prefix) + 1, dtype=mx.float32)[:, None]
        hidden = mx.tanh(mx.cumsum(emb, axis=0) / steps)[None, -len(ids):]
        return SimpleNamespace(
            logits=self.target.logits_from_hidden(hidden) if compute_logits else None,
            hidden=hidden,
            shared_kv_states={},
            cache_offset=len(ids),
            attention_phase=phase,
            cache_counters={},
        )


def _reference_logprobs(rt: _StubGemmaRuntime, prompt: list[int]) -> np.ndarray:
    emb = np.array(rt.embed)[prompt]
    steps = np.arange(1, len(prompt) + 1, dtype=np.float32)[:, None]
    logits = np.tanh(np.cumsum(emb, axis=0) / steps) @ np.array(rt.target.weight)
    logits = logits - logits.max(axis=-1, keepdims=True)
    return logits - np.log(np.exp(logits).sum(axis=-1, keepdims=True))


def _prompt(n: int, seed: int = 1) -> list[int]:
    return np.random.default_rng(seed).integers(0, VOCAB, size=n).tolist()


def test_gemma4_runtime_has_no_forward_ar():
    """The failure this covers: the generic scoring path needs forward_ar."""

    assert not hasattr(gemma4.Gemma4AssistantRuntime, "forward_ar")


def test_gemma4_scoring_matches_reference_log_softmax():
    rt = _StubGemmaRuntime()
    prompt = _prompt(40)

    scored = score_prompt_logprobs(rt, prompt, top_k=5, chunk_size=16)

    ref = _reference_logprobs(rt, prompt)
    assert scored["prompt_tokens"] == 40
    assert len(scored["positions"]) == 39
    expected_targets = [ref[i, prompt[i + 1]] for i in range(39)]
    np.testing.assert_allclose(scored["token_logprobs"], expected_targets, atol=1e-4)
    for i, entries in enumerate(scored["positions"]):
        expected_ids = np.argsort(-ref[i])[:5].tolist()
        assert [token for token, _lp in entries] == expected_ids
        np.testing.assert_allclose(
            [lp for _token, lp in entries], ref[i, expected_ids], atol=1e-4
        )


def test_gemma4_scoring_is_one_prefill_forward_and_chunked_head():
    """One target pass like the generation prefill; logits per chunk only."""

    rt = _StubGemmaRuntime()
    prompt = _prompt(40)

    score_prompt_logprobs(rt, prompt, top_k=3, chunk_size=16)

    assert rt.forward_calls == [
        {"tokens": prompt, "phase": "prefill", "compute_logits": False}
    ]
    assert rt.target.head_rows == [16, 16, 8]


def test_gemma4_scoring_independent_of_chunk_size():
    rt = _StubGemmaRuntime()
    prompt = _prompt(37)

    small = score_prompt_logprobs(rt, prompt, top_k=4, chunk_size=16)
    whole = score_prompt_logprobs(rt, prompt, top_k=4, chunk_size=4096)

    assert small["positions"] == whole["positions"]
    assert small["token_logprobs"] == whole["token_logprobs"]


@pytest.mark.parametrize("rows", [48, 49])
def test_gemma4_scoring_respects_prefill_width(monkeypatch, rows):
    """A bounded head must not hide whole-prompt attention allocations."""
    monkeypatch.setenv("MTPLX_GEMMA4_PREFILL_CHUNK_TOKENS", "16")
    rt = _StubGemmaRuntime()
    prompt = _prompt(rows)
    scored = score_prompt_logprobs(rt, prompt, top_k=4, chunk_size=32)

    widths = [len(call["tokens"]) for call in rt.forward_calls]
    assert max(widths) <= 16
    assert min(widths) >= 2  # Never enter the sliding cache's decode path.
    assert sum(widths) == rows
    assert max(rt.target.head_rows) <= 16
    ref = _reference_logprobs(rt, prompt)
    np.testing.assert_allclose(
        scored["token_logprobs"],
        [ref[i, prompt[i + 1]] for i in range(rows - 1)],
        atol=1e-4,
    )


def test_gemma4_scored_label_matches_first_token_from_generation_prefill():
    """Scoring prompt + label gives the label the logprob the generation
    prefill of the prompt alone assigns it as first token."""

    rt = _StubGemmaRuntime()
    prompt = _prompt(20)
    label = 77

    scored = score_prompt_logprobs(rt, prompt + [label], top_k=3)
    prefill, _elapsed = gemma4._gemma4_prefill_prompt(
        rt, prompt, cache=rt.make_cache(), phase="prefill"
    )
    row = np.array(prefill.logits[0, -1].astype(mx.float32))
    row = row - row.max()
    first_token_logprob = row[label] - np.log(np.exp(row).sum())

    assert scored["token_logprobs"][-1] == pytest.approx(
        float(first_token_logprob), abs=1e-5
    )


def test_gemma4_prompt_scoring_endpoint_returns_200(monkeypatch):
    """End to end over the real score_prompt_logprobs: no 500 on Gemma."""

    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from test_server_openai import CaptureTokenizer, _fake_state

    from mtplx.server import openai
    from mtplx.server.openai import create_app

    tokenizer = CaptureTokenizer()
    state = _fake_state()
    state_model_path = state.runtime.model_path
    state.runtime = _StubGemmaRuntime(tokenizer=tokenizer)
    state.runtime.model_path = state_model_path
    state.begin_foreground = lambda: None
    state.end_foreground = lambda: None
    state.requests_completed = 0
    state.last_request_at = 0.0
    monkeypatch.setattr(openai, "score_prompt_logprobs", score_prompt_logprobs)
    client = TestClient(create_app(state))

    response = client.post(
        "/v1/completions",
        json={"prompt": "abcd", "echo": True, "logprobs": 2, "max_tokens": 0},
    )

    assert response.status_code == 200, response.text
    logprobs = response.json()["choices"][0]["logprobs"]
    assert logprobs["token_ids"] == [97, 98, 99, 100]
    assert logprobs["token_logprobs"][0] is None
    ref = _reference_logprobs(state.runtime, [97, 98, 99, 100])
    np.testing.assert_allclose(
        logprobs["token_logprobs"][1:],
        [ref[0, 98], ref[1, 99], ref[2, 100]],
        atol=1e-4,
    )
