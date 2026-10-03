"""First-token logprobs on the Gemma 4 pair backend.

``generate_ar`` and ``generate_mtpk`` hand Gemma 4 requests to their own
loops in ``mtplx.backends.gemma4_assistant`` (target-only AR and the exact
speculative assistant loop). Both loops sample the first token from the
prompt's target logits row, so both must report that row's raw
distribution when ``first_token_logprobs_top_k`` is set, and nothing when
it is not. No-model harness: a scripted target (same pattern as
test_tail_gemma4_stream_holdback).
"""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

import mtplx.backends.gemma4_assistant as gemma4
from mtplx import generation
from mtplx.sampling import SamplerConfig

VOCAB = 16
GREEDY = SamplerConfig(temperature=0.0, top_p=1.0, top_k=0)
# Distinct logits so the top-K order is unambiguous; argmax is token 6.
PROMPT_ROW = [float(i % 7) - 0.25 * i for i in range(VOCAB)]


def _expected_logprobs() -> np.ndarray:
    row = np.asarray(PROMPT_ROW, dtype=np.float64)
    return row - np.log(np.exp(row - row.max()).sum()) - row.max()


class _Tokenizer:
    def decode(self, tokens, **_kwargs):
        return "".join(f"<{int(token)}>" for token in tokens)


class _ScriptedGemmaRuntime:
    """Target double: after any token the target wants token + 1."""

    def __init__(self) -> None:
        self.backend_id = "gemma4_assistant"
        self.tokenizer = _Tokenizer()
        self.telemetry = SimpleNamespace(to_dict=dict)
        self.config = SimpleNamespace(
            draft_block_size=2,
            assistant_model_path="scripted-assistant",
            target_distribution_mode="exact",
        )
        self.distribution_compile_stats = {}

    def forward_target(self, input_ids, *, cache=None, phase=None):
        del cache, phase
        token = int(np.asarray(input_ids).reshape(-1)[-1])
        row = [0.0] * VOCAB
        row[(token + 1) % VOCAB] = 10.0
        return SimpleNamespace(
            logits=mx.array([[row]], dtype=mx.float32),
            hidden=mx.zeros((1, 1, 2), dtype=mx.float32),
            shared_kv_states={},
            cache_offset=0,
        )


def _patch_prefill(monkeypatch) -> None:
    def prefill(runtime, prompt_ids, **_kwargs):
        del runtime
        return SimpleNamespace(
            cache=[],
            logits=mx.array([PROMPT_ROW], dtype=mx.float32),
            hidden=mx.zeros((1, 1, 2), dtype=mx.float32),
            shared_kv_states={},
            kv_offset=0,
            prompt_eval_time_s=0.0,
            cached_tokens=0,
            suffix_tokens=len(prompt_ids),
            cache_hit=False,
            cache_source="none",
            cache_miss_reason=None,
            restore_mode="cold",
        )

    monkeypatch.setattr(gemma4, "_restore_or_prefill_gemma4_prompt", prefill)


def _patch_speculative_round(monkeypatch) -> None:
    def fake_round(runtime, *, primary_token_id, draft_block_size, **_kwargs):
        del runtime
        token = int(primary_token_id)
        accepted = []
        for _ in range(max(1, int(draft_block_size) - 1)):
            token = (token + 1) % VOCAB
            accepted.append(token)
        return SimpleNamespace(
            accepted_token_ids=accepted,
            accepted_count=len(accepted),
            corrected_token_id=None,
            bonus_token_id=None,
            next_primary_token_id=(token + 1) % VOCAB,
            next_hidden=mx.zeros((1, 1, 2), dtype=mx.float32),
            next_shared_kv_states={},
            next_kv_offset=0,
            metadata={},
        )

    monkeypatch.setattr(gemma4, "gemma4_exact_speculative_round", fake_round)


def _run_ar(**kwargs):
    return gemma4.generate_gemma4_ar(
        _ScriptedGemmaRuntime(),
        [0],
        sampler=GREEDY,
        seed=7,
        stop_token_ids=set(),
        **kwargs,
    )


def _run_assistant(**kwargs):
    return gemma4.generate_gemma4_assistant(
        _ScriptedGemmaRuntime(),
        [0],
        sampler=GREEDY,
        speculative_depth=2,
        seed=7,
        stop_token_ids=set(),
        **kwargs,
    )


@pytest.mark.parametrize("run", [_run_ar, _run_assistant], ids=["ar", "assistant"])
def test_gemma4_first_token_logprobs_match_prompt_row(monkeypatch, run):
    _patch_prefill(monkeypatch)
    _patch_speculative_round(monkeypatch)

    out = run(max_tokens=1, first_token_logprobs_top_k=3)

    expected = _expected_logprobs()
    first = out.first_token_logprobs
    assert out.tokens == [int(np.argmax(expected))]
    assert first.token_id == out.tokens[0]
    assert first.logprob == pytest.approx(expected[first.token_id], abs=1e-5)
    best = np.argsort(-expected)[:3]
    assert [token for token, _ in first.top] == [int(token) for token in best]
    for token, value in first.top:
        assert value == pytest.approx(expected[token], abs=1e-5)


@pytest.mark.parametrize("run", [_run_ar, _run_assistant], ids=["ar", "assistant"])
def test_gemma4_first_token_logprobs_cover_only_the_first_token(monkeypatch, run):
    _patch_prefill(monkeypatch)
    _patch_speculative_round(monkeypatch)

    out = run(max_tokens=4, first_token_logprobs_top_k=0)

    assert len(out.tokens) == 4
    assert out.first_token_logprobs.token_id == out.tokens[0]
    assert out.first_token_logprobs.top == ()


@pytest.mark.parametrize("run", [_run_ar, _run_assistant], ids=["ar", "assistant"])
def test_gemma4_without_logprobs_request_reports_none(monkeypatch, run):
    _patch_prefill(monkeypatch)
    _patch_speculative_round(monkeypatch)

    out = run(max_tokens=2)

    assert out.first_token_logprobs is None


@pytest.mark.parametrize(
    ("entrypoint", "target"),
    [
        (generation.generate_ar, "generate_gemma4_ar"),
        (
            lambda *args, **kwargs: generation.generate_mtpk(
                *args, speculative_depth=2, **kwargs
            ),
            "generate_gemma4_assistant",
        ),
    ],
    ids=["generate_ar", "generate_mtpk"],
)
def test_generation_entrypoints_forward_logprobs_top_k_to_gemma4(
    monkeypatch, entrypoint, target
):
    seen: dict = {}

    def fake(runtime, prompt_ids, **kwargs):
        del runtime, prompt_ids
        seen.update(kwargs)
        return "sentinel"

    monkeypatch.setattr(gemma4, target, fake)
    monkeypatch.setattr(
        generation, "reject_non_k1_a3b_whole_moe_request", lambda *_a, **_k: None
    )

    result = entrypoint(
        _ScriptedGemmaRuntime(),
        [0],
        max_tokens=1,
        sampler=GREEDY,
        first_token_logprobs_top_k=5,
    )

    assert result == "sentinel"
    assert seen["first_token_logprobs_top_k"] == 5
