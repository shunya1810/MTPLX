"""Pipelined AR decode (device-drawn token, next forward dispatched before the
host reads it): same tokens as the blocking loop, the same cache contents and
final-state logits, no forward past max_tokens, steering steps on the host."""

from pathlib import Path

import mlx.core as mx
import numpy as np

from mtplx.generation import _ar_pipeline_sampler_ok, generate_ar
from mtplx.mtp_patch import MTPContract
from mtplx.runtime import MTPLXRuntime
from mtplx.sampling import SamplerConfig

V = 16
STOP = 9


class TinyTokenizer:
    def decode(self, tokens, **_kwargs):
        return "".join(str(int(token)) for token in tokens)


class CountingModel:
    """Next token = last input + 1 (mod V), as a sharp one-hot; the cache
    records every forwarded token (lazily, so the model never syncs)."""

    def __init__(self):
        self.forwards = 0

    def make_cache(self):
        return []

    def make_mtp_cache(self):
        return []

    def sanitize(self, weights):
        return weights

    def __call__(self, input_ids, *, cache=None, return_hidden=False, hidden_variant=None,
                 emit_logits=True, logits_keep=None):
        self.forwards += 1
        ids = input_ids.reshape(1, -1).astype(mx.int32)
        if cache is not None:
            cache.append(ids)
        length = int(ids.shape[1])
        hidden = mx.zeros((1, length, 2), dtype=mx.float32)
        if not emit_logits:
            return (None, hidden) if return_hidden else None
        nxt = mx.remainder(ids + 1, V)
        logits = (mx.arange(V)[None, None, :] == nxt[..., None]).astype(mx.float32) * 30.0
        keep = length if logits_keep is None else min(length, max(1, int(logits_keep)))
        logits = logits[:, -keep:, :]
        return (logits, hidden) if return_hidden else logits


def _run(monkeypatch, pipeline: str, *, max_tokens=6, prompt=(1,), capture=True, sampler=None):
    monkeypatch.setenv("MTPLX_AR_PIPELINE", pipeline)
    for key in ("MTPLX_ASYNC_AR", "MTPLX_EVAL_AUDIT"):
        monkeypatch.delenv(key, raising=False)
    model = CountingModel()
    rt = MTPLXRuntime(model=model, tokenizer=TinyTokenizer(), model_path=Path("tiny"),
                      mtp_enabled=False, contract=MTPContract())
    out = generate_ar(
        rt,
        list(prompt),
        max_tokens=max_tokens,
        sampler=sampler or SamplerConfig(temperature=0.0, top_p=1.0, top_k=0),
        stop_token_ids={STOP},
        capture_final_state=capture,
    )
    return out, model


def _cache_tokens(state):
    return [int(t) for a in state.final_trunk_cache for t in np.array(a).reshape(-1)]


def test_pipelined_matches_blocking_through_a_stop_token(monkeypatch):
    ref, ref_model = _run(monkeypatch, "0")
    got, got_model = _run(monkeypatch, "1")
    assert ref.tokens == got.tokens == [2, 3, 4, 5, 6, 7]
    # max_tokens reached: the pipeline must not forward past the last token
    assert got_model.forwards == ref_model.forwards

    ref, _ = _run(monkeypatch, "0", max_tokens=20, prompt=(5,))
    got, _ = _run(monkeypatch, "1", max_tokens=20, prompt=(5,))
    assert ref.tokens == got.tokens == [6, 7, 8, 9]
    assert ref.final_state is not None and got.final_state is not None
    assert _cache_tokens(ref.final_state) == _cache_tokens(got.final_state) == [5, 6, 7, 8, 9]
    assert mx.array_equal(ref.final_state.final_logits, got.final_state.final_logits).item()


def test_pipelined_sampling_draws_valid_tokens(monkeypatch):
    sampler = SamplerConfig(temperature=1.0, top_p=1.0, top_k=0)
    got, _ = _run(monkeypatch, "1", max_tokens=8, prompt=(0,), sampler=sampler, capture=False)
    # logits are one-hot at +30, so sampling at T=1 follows the chain
    assert got.tokens == [1, 2, 3, 4, 5, 6, 7, 8]


def test_sampler_shapes_the_pipeline_takes():
    assert _ar_pipeline_sampler_ok(SamplerConfig(temperature=0.0, top_p=1.0, top_k=0))
    assert _ar_pipeline_sampler_ok(SamplerConfig(temperature=1.0, top_p=1.0, top_k=0))
    assert _ar_pipeline_sampler_ok(SamplerConfig(temperature=0.6, top_p=0.95, top_k=20))
    # full-vocabulary nucleus and penalties stay on the host sampler
    assert not _ar_pipeline_sampler_ok(SamplerConfig(temperature=1.0, top_p=0.9, top_k=0))
    assert not _ar_pipeline_sampler_ok(
        SamplerConfig(temperature=1.0, top_p=1.0, top_k=0, presence_penalty=0.5)
    )


def test_model_attribute_opts_in_by_default(monkeypatch):
    from mtplx.generation import _ar_pipeline_enabled

    monkeypatch.delenv("MTPLX_AR_PIPELINE", raising=False)
    monkeypatch.delenv("MTPLX_EVAL_AUDIT", raising=False)
    greedy = SamplerConfig(temperature=0.0, top_p=1.0, top_k=0)

    class Rt:
        model = CountingModel()

    assert not _ar_pipeline_enabled(Rt(), greedy, None)
    Rt.model.mtplx_ar_pipeline_default = True
    assert _ar_pipeline_enabled(Rt(), greedy, None)
    assert not _ar_pipeline_enabled(Rt(), greedy, object())  # grammar constraint
    monkeypatch.setenv("MTPLX_AR_PIPELINE", "0")
    assert not _ar_pipeline_enabled(Rt(), greedy, None)
