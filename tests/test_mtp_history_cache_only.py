"""Draft-head history appends in a prefill loop evaluate the cache only.

On a tiny quantized Qwen3.5-MoE with a one-layer MoE draft head, and on the
dense Qwen3.5 layout (the 27B) with a dense draft head, both injected the
product way, these tests pin:

* a prefill-phase append evaluates
  exactly the draft cache's key and value buffers, never the hidden;
* the cache carries the same bits as with the full layer pass, chunk after
  chunk, and the first draft step after it gives the same logits and hidden;
* the same with image embeddings in the appended rows, and when the append
  lands on a cache restored from saved state (the next turn after a bank
  restore);
* on by default (``=0`` turns it off), and decode appends, other cache kinds
  and a missing cache keep evaluating the hidden.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

pytest.importorskip("mlx_lm.models.qwen3_5_moe")

from mlx_lm.models.cache import KVCache

from mtplx import generation
from mtplx.attention_context import attention_phase
from mtplx.mtp_history_cache_only import (
    ENV,
    mtp_history_cache_arrays,
    mtp_history_cache_only_enabled,
)
from tests.a3b_tiny_synth import assert_bit_equal, prompt, tiny_model_with_draft_head


class _Runtime:
    """The runtime surface ``_append_mtp_history`` uses."""

    def __init__(self, model):
        self.model = model
        self.diagnostic_counters: dict[str, int] = {}

    def update_mtp_cache(self, hidden_states, next_token_ids, *, mtp_cache, **kwargs):
        return self.model.mtp_update_cache(
            hidden_states, next_token_ids, mtp_cache=mtp_cache, **kwargs
        )


@pytest.fixture(scope="module", params=["moe", "dense"])
def model(request, tmp_path_factory):
    return tiny_model_with_draft_head(
        tmp_path_factory.mktemp(f"draft-head-{request.param}"), dense=request.param == "dense"
    )


def _trunk_hidden(model, tokens):
    with attention_phase("prefill"):
        _logits, hidden = model(mx.array([tokens]), return_hidden=True)
    mx.eval(hidden)
    return hidden


def _append_chunks(
    model, tokens, hidden, chunks, *, cache_only, monkeypatch, phase="prefill", cache=None, embeddings=None
):
    monkeypatch.setenv(ENV, "1" if cache_only else "0")
    rt = _Runtime(model)
    cache = model.make_mtp_cache() if cache is None else cache
    evaluated: list = []
    real_eval = generation._eval

    def recording_eval(*values, **kwargs):
        evaluated.append(values)
        return real_eval(*values, **kwargs)

    monkeypatch.setattr(generation, "_eval", recording_eval)
    for start, end in chunks:
        generation._append_mtp_history(
            rt,
            cache,
            hidden[:, start:end, :],
            tokens[start + 1 : end + 1],
            phase=phase,
            mtp_hidden_variant="post_norm",
            force_eval=True,
            input_embeddings=None if embeddings is None else embeddings[:, start:end, :],
        )
    monkeypatch.setattr(generation, "_eval", real_eval)
    return rt, cache, evaluated


def _draft_step(model, cache, hidden_row, token):
    with attention_phase("decode"):
        logits, draft_hidden = model.mtp_forward(
            hidden_row, mx.array([[token]]), mtp_cache=cache, return_hidden=True
        )
    mx.eval(logits, draft_hidden)
    return logits, draft_hidden


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16, mx.float32])
def test_bit_equality_compares_bit_patterns(dtype):
    """Every "same bits" assertion here and in the kernel tests goes through
    ``assert_bit_equal``: a sign-of-zero difference must fail it and the same
    NaN must pass it, in every float dtype the caches hold."""

    zero = mx.array([0.0, 1.0], dtype=dtype)
    negative_zero = mx.array([-0.0, 1.0], dtype=dtype)
    assert bool(mx.array_equal(zero, negative_zero).item())  # equal as values ...
    with pytest.raises(AssertionError):
        assert_bit_equal([zero], [negative_zero])  # ... not as bits
    nan = mx.array([float("nan"), 1.0], dtype=dtype)
    assert_bit_equal([nan], [nan])


def test_on_by_default_and_off_with_a_false_value(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    assert mtp_history_cache_only_enabled() is True
    for value in ("0", "false", "off", "no"):
        monkeypatch.setenv(ENV, value)
        assert mtp_history_cache_only_enabled() is False
    monkeypatch.setenv(ENV, "1")
    assert mtp_history_cache_only_enabled() is True


def test_cache_arrays_only_for_plain_kv_caches():
    assert mtp_history_cache_arrays(None) is None
    assert mtp_history_cache_arrays([]) is None
    empty = KVCache()
    assert mtp_history_cache_arrays([empty]) is None
    filled = KVCache()
    filled.update_and_fetch(mx.zeros((1, 1, 3, 8)), mx.ones((1, 1, 3, 8)))
    arrays = mtp_history_cache_arrays([filled])
    assert arrays is not None and arrays[0] is filled.keys and arrays[1] is filled.values

    class OtherCache(KVCache):
        pass

    other = OtherCache()
    other.update_and_fetch(mx.zeros((1, 1, 3, 8)), mx.ones((1, 1, 3, 8)))
    assert mtp_history_cache_arrays([filled, other]) is None


@pytest.mark.parametrize(
    "chunks",
    [
        [(0, 150)],
        [(0, 64), (64, 200), (200, 239)],
        [(0, 1), (1, 9), (9, 239)],
    ],
)
def test_cache_only_append_leaves_the_same_cache_and_draft(model, monkeypatch, chunks):
    tokens = prompt(241)
    hidden = _trunk_hidden(model, tokens[:240])
    _rt_full, full_cache, full_evals = _append_chunks(
        model, tokens, hidden, chunks, cache_only=False, monkeypatch=monkeypatch
    )
    rt, cache, evals = _append_chunks(
        model, tokens, hidden, chunks, cache_only=True, monkeypatch=monkeypatch
    )

    # Only the cache buffers were evaluated; the full pass evaluated the hidden.
    assert rt.diagnostic_counters["mtp_history_cache_only_appends"] == len(chunks)
    assert all(len(values) == 2 for values in evals)
    assert evals[-1][0] is cache[0].keys and evals[-1][1] is cache[0].values
    assert all(len(values) == 1 for values in full_evals)
    assert all(values[0].ndim == 3 for values in full_evals)

    end = chunks[-1][1]
    assert cache[0].offset == full_cache[0].offset == end
    assert_bit_equal(cache[0].state, full_cache[0].state)
    row = hidden[:, end - 1 : end, :]
    assert_bit_equal(
        _draft_step(model, cache, row, tokens[end]),
        _draft_step(model, full_cache, row, tokens[end]),
    )
    # The deeper draft state after that step is the same too.
    assert_bit_equal(cache[0].state, full_cache[0].state)


def test_cache_only_append_with_image_embeddings(model, monkeypatch):
    """Rows whose draft-head input is an embedding (image tokens) rather than
    a token id: the same cache bits and the same first draft either way."""

    tokens = prompt(121, seed=5)
    hidden = _trunk_hidden(model, tokens[:120])
    mx.random.seed(55)
    embeddings = (mx.random.normal(hidden.shape) * 0.5).astype(hidden.dtype)
    chunks = [(0, 48), (48, 120)]
    _rt, full_cache, _ = _append_chunks(
        model, tokens, hidden, chunks, cache_only=False, monkeypatch=monkeypatch, embeddings=embeddings
    )
    rt, cache, evals = _append_chunks(
        model, tokens, hidden, chunks, cache_only=True, monkeypatch=monkeypatch, embeddings=embeddings
    )
    assert rt.diagnostic_counters["mtp_history_cache_only_appends"] == len(chunks)
    assert all(len(values) == 2 for values in evals)
    assert_bit_equal(cache[0].state, full_cache[0].state)
    row = hidden[:, 119:120, :]
    assert_bit_equal(_draft_step(model, cache, row, tokens[120]), _draft_step(model, full_cache, row, tokens[120]))


def test_cache_only_append_onto_a_restored_cache(model, monkeypatch):
    """The next turn after a bank restore: the draft cache comes back from
    saved state (exact-length buffers), then the new turn's rows append."""

    tokens = prompt(201, seed=9)
    hidden = _trunk_hidden(model, tokens[:200])
    _rt, first_turn, _ = _append_chunks(
        model, tokens, hidden, [(0, 130)], cache_only=False, monkeypatch=monkeypatch
    )
    saved = [mx.array(leaf) for leaf in first_turn[0].state]
    mx.eval(saved)

    def restored():
        cache = model.make_mtp_cache()
        cache[0].state = tuple(saved)
        return cache

    _rt, full_cache, _ = _append_chunks(
        model, tokens, hidden, [(130, 200)], cache_only=False, monkeypatch=monkeypatch, cache=restored()
    )
    rt, cache, evals = _append_chunks(
        model, tokens, hidden, [(130, 200)], cache_only=True, monkeypatch=monkeypatch, cache=restored()
    )
    assert rt.diagnostic_counters["mtp_history_cache_only_appends"] == 1
    assert cache[0].offset == full_cache[0].offset == 200
    assert_bit_equal(cache[0].state, full_cache[0].state)
    row = hidden[:, 199:200, :]
    assert_bit_equal(_draft_step(model, cache, row, tokens[200]), _draft_step(model, full_cache, row, tokens[200]))


def test_decode_appends_keep_evaluating_the_hidden(model, monkeypatch):
    tokens = prompt(41)
    hidden = _trunk_hidden(model, tokens[:40])
    rt, _cache, evals = _append_chunks(
        model, tokens, hidden, [(0, 40)], cache_only=True, monkeypatch=monkeypatch,
        phase="ar_decode",
    )
    assert "mtp_history_cache_only_appends" not in rt.diagnostic_counters
    assert len(evals) == 1 and len(evals[0]) == 1


def test_other_cache_kinds_keep_evaluating_the_hidden(model, monkeypatch):
    class OtherCache(KVCache):
        pass

    monkeypatch.setenv(ENV, "1")
    tokens = prompt(41)
    hidden = _trunk_hidden(model, tokens[:40])
    rt = _Runtime(model)
    evaluated: list = []
    monkeypatch.setattr(generation, "_eval", lambda *values, **_kw: evaluated.append(values))
    generation._append_mtp_history(
        rt,
        [OtherCache()],
        hidden,
        tokens[1:41],
        phase="prefill",
        mtp_hidden_variant="post_norm",
        force_eval=True,
    )
    assert "mtp_history_cache_only_appends" not in rt.diagnostic_counters
    assert len(evaluated) == 1 and len(evaluated[0]) == 1


class _LoopRuntime(_Runtime):
    """The runtime surface the cold streaming prefill loop uses."""

    mtp_enabled = True

    def __init__(self, model, tmp_path):
        super().__init__(model)
        self.model_path = tmp_path

    def make_cache(self):
        return self.model.make_cache()

    def make_mtp_cache(self):
        return self.model.make_mtp_cache()

    @property
    def embed_tokens(self):
        return self.model.language_model.model.embed_tokens

    def forward_ar(
        self,
        tokens,
        *,
        cache,
        return_hidden=False,
        hidden_variant=None,
        emit_logits=True,
        logits_keep=None,
        input_embeddings=None,
    ):
        return self.model(
            tokens,
            cache=cache,
            return_hidden=return_hidden,
            hidden_variant=hidden_variant,
            emit_logits=emit_logits,
            logits_keep=logits_keep,
            input_embeddings=input_embeddings,
        )


def _cold(model, tokens, tmp_path, *, cache_only, monkeypatch):
    monkeypatch.setenv(ENV, "1" if cache_only else "0")
    rt = _LoopRuntime(model, tmp_path)
    out = generation._prefill_committed_mtp_history_streaming(rt, list(tokens))
    cache, logits, hidden, mtp_cache = out[:4]
    mx.eval(logits, hidden)
    return rt, cache, logits, hidden, mtp_cache


def test_cold_prefill_loop_leaves_the_same_draft_cache(model, monkeypatch, tmp_path):
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", "1")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE", "96")
    tokens = prompt(301, seed=3)
    full = _cold(model, tokens, tmp_path, cache_only=False, monkeypatch=monkeypatch)
    ours = _cold(model, tokens, tmp_path, cache_only=True, monkeypatch=monkeypatch)
    rt = ours[0]
    assert rt.diagnostic_counters["mtp_history_cache_only_appends"] >= 3
    assert "mtp_history_cache_only_appends" not in full[0].diagnostic_counters
    assert_bit_equal([ours[2], ours[3]], [full[2], full[3]])
    assert_bit_equal(
        [leaf for entry in ours[1] for leaf in entry.state if leaf is not None],
        [leaf for entry in full[1] for leaf in entry.state if leaf is not None],
    )
    assert ours[4][0].offset == full[4][0].offset == len(tokens) - 1
    assert_bit_equal(ours[4][0].state, full[4][0].state)
    assert_bit_equal(
        _draft_step(model, ours[4], ours[3], tokens[-1]),
        _draft_step(model, full[4], full[3], tokens[-1]),
    )
