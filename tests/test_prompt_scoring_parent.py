"""Generic scoring preserves 50de43bb's forwards and every returned score bit."""

from pathlib import Path
import runpy

import mlx.core as mx
import numpy as np
import pytest
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.models.llama import Model, ModelArgs

from mtplx import generation


class TinyScoringRuntime:
    """Two real attention layers with fresh KV per score, no loaded weights."""

    mtp_enabled = False

    def __init__(self, *, mtp_enabled=False):
        self.mtp_enabled = mtp_enabled
        mx.random.seed(7)
        self.model = Model(ModelArgs(
            model_type="llama", hidden_size=32, num_hidden_layers=2,
            intermediate_size=64, num_attention_heads=4, num_key_value_heads=2,
            rms_norm_eps=1e-5, vocab_size=97,
        ))
        mx.eval(self.model.parameters())
        self.widths = []
        self.caches = []

    def make_cache(self):
        cache = make_prompt_cache(self.model)
        self.caches.append(cache)
        return cache

    def forward_ar(self, tokens, *, cache, **_kwargs):
        self.widths.append(int(tokens.shape[1]))
        logits = self.model(tokens, cache=cache)
        return (logits, logits[..., :1]) if self.mtp_enabled else logits


def _score_bits(result):
    values = result["token_logprobs"] + [
        value for entries in result["positions"] for _token, value in entries
    ]
    return np.asarray(values, dtype=np.float32).view(np.uint32)


@pytest.mark.parametrize("serving_width", [None, 64, 128, 2048])
@pytest.mark.parametrize("chunk_size", [None, 512])
@pytest.mark.parametrize("mtp_enabled", [False, True])
def test_generic_scoring_matches_parent_widths_and_logprob_bits(
    serving_width, chunk_size, mtp_enabled
):
    parent = runpy.run_path(
        str(Path(__file__).parent / "fixtures/prompt_scoring_50de43bb.py"),
        init_globals=dict(vars(generation)),
    )["score_prompt_logprobs"]
    prompt = np.random.default_rng(11).integers(0, 97, 529).tolist()
    kwargs = {} if chunk_size is None else {"chunk_size": chunk_size}
    with mx.stream(mx.cpu):
        runtime = TinyScoringRuntime(mtp_enabled=mtp_enabled)
        expected = parent(runtime, prompt, top_k=5, **kwargs)
        parent_widths = list(runtime.widths)
        runtime.widths.clear()
        with generation.prefill_chunk_size_override(serving_width):
            actual = generation.score_prompt_logprobs(runtime, prompt, top_k=5, **kwargs)

        assert parent_widths == ([256, 256, 17] if chunk_size is None else [512, 17])
        assert runtime.widths == parent_widths
        np.testing.assert_array_equal(_score_bits(actual), _score_bits(expected))
        assert actual["positions"] == expected["positions"]
        for old_cache, new_cache in zip(runtime.caches[0], runtime.caches[1], strict=True):
            assert old_cache.offset == new_cache.offset == len(prompt)
            for old, new in zip(old_cache.state, new_cache.state, strict=True):
                np.testing.assert_array_equal(
                    np.asarray(old).view(np.uint32), np.asarray(new).view(np.uint32)
                )
