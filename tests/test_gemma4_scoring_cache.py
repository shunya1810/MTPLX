"""Small CPU scoring checks with real Gemma attention and rotating caches."""

import numpy as np
import pytest
from test_gemma4_chunked_prefill import (
    OUTPUT_ATOL,
    _prompt,
    _spy_forwards,
    cpu,  # noqa: F401 -- pytest fixture
    tiny_pair,  # noqa: F401 -- pytest fixture
)

import mlx.core as mx
from mtplx import generation
from mtplx.backends.gemma4_assistant import Gemma4RollbackRotatingKVCache


def _capture_cache(runtime, monkeypatch):
    caches = []
    make_cache = runtime.make_cache

    def make():
        cache = make_cache()
        caches.append(cache)
        return cache

    monkeypatch.setattr(runtime, "make_cache", make)
    return caches


def test_scoring_matches_causal_reference_across_rotating_cache_rollover(
    tiny_pair, cpu, monkeypatch
):
    runtime = tiny_pair(8, seed=4)
    prompt = _prompt(67)
    reference = runtime.forward_target(
        mx.array([prompt]), cache=runtime.make_cache(), phase="prefill", compute_logits=True
    ).logits[0].astype(mx.float32)
    reference = reference - mx.logsumexp(reference, axis=-1, keepdims=True)
    expected = np.array(reference)[np.arange(66), prompt[1:]]
    caches = _capture_cache(runtime, monkeypatch)
    calls = _spy_forwards(runtime)

    with generation.prefill_chunk_size_override(16):
        scored = generation.score_prompt_logprobs(runtime, prompt, top_k=3, chunk_size=16)

    assert [rows for _phase, _offset, rows in calls] == [3, 16, 16, 16, 16]
    windows = [c for c in caches[0] if isinstance(c, Gemma4RollbackRotatingKVCache)]
    assert windows and all(c.offset == 67 for c in windows)
    assert all(int(c.keys.shape[-2]) == 8 - 1 + 16 for c in windows)
    assert all(c._record_updates and c._last_update is None for c in windows)
    np.testing.assert_allclose(scored["token_logprobs"], expected, rtol=0, atol=OUTPUT_ATOL)


def test_scoring_abort_restores_rotating_cache_update_mode(tiny_pair, cpu, monkeypatch):
    runtime = tiny_pair(8)
    caches = _capture_cache(runtime, monkeypatch)
    calls = _spy_forwards(runtime)

    with generation.prefill_chunk_size_override(16), pytest.raises(generation.PostcommitAbort):
        generation.score_prompt_logprobs(
            runtime, _prompt(67), top_k=3, abort_check=lambda: len(calls) >= 1
        )

    assert len(calls) == 1
    windows = [c for c in caches[0] if isinstance(c, Gemma4RollbackRotatingKVCache)]
    assert windows and all(c._record_updates and c._last_update is None for c in windows)


def test_scoring_repeats_exactly_and_bounds_width_rounding(tiny_pair, cpu):
    runtime = tiny_pair(8, seed=4)
    prompt = _prompt(67)
    with generation.prefill_chunk_size_override(16):
        first = generation.score_prompt_logprobs(runtime, prompt, top_k=3)
        repeated = generation.score_prompt_logprobs(runtime, prompt, top_k=3)
    with generation.prefill_chunk_size_override(32):
        wider = generation.score_prompt_logprobs(runtime, prompt, top_k=3)

    assert first["positions"] == repeated["positions"]
    assert first["token_logprobs"] == repeated["token_logprobs"]
    for scored in (first, repeated, wider):
        assert np.isfinite(scored["token_logprobs"]).all()
        assert all(np.isfinite(value) for row in scored["positions"] for _, value in row)
    # Different attention widths can round differently; this is class C,
    # never a claim of bit identity against the pre-#551 HTTP 500.
    np.testing.assert_allclose(
        first["token_logprobs"], wider["token_logprobs"], rtol=0, atol=OUTPUT_ATOL
    )
    delta = np.max(np.abs(np.array(first["token_logprobs"]) - wider["token_logprobs"]))
    print({"widths": [16, 32], "max_abs_logprob_delta": float(delta), "bound": OUTPUT_ATOL})
