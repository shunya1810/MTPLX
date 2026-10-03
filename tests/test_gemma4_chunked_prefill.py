"""Gemma 4's prefill runs in chunks, and a chunked prefill is the same prefill.

Gemma 4 forwarded every uncached prompt token in one call. MLX runs its
prefill attention unfused (the fused query-block kernel stops at 256-wide
heads and, in MLX 0.32, takes 256 only under a plain causal mask on a GPU with
NAX), so each full-attention layer (512-wide heads) computes its scores as an
array, prompt x prompt for a cold prompt, and each sliding layer as much under
its window mask. On the 31B (4-bit, 128 GB M5 Max,
2026-09-27) cold prompts of 6,026 / 12,026 / 24,026 tokens peaked at 25 / 43 /
92 GiB of MLX memory with time to first token 13 / 41 / 174 s, and a 32K
prompt would not fit a 128 GB Mac. The prefill now forwards the uncached rows
in chunks of the house prefill width (2,048 by default), so the block is
chunk x (cached + chunk).

These tests run the backend's own code (``Gemma4TargetAdapter``,
``Gemma4RollbackRotatingKVCache``, MLX-LM's ``KVCache``, the assistant drafter,
the session bank) on a tiny Gemma 4 built from MLX-LM's ``gemma4_text``: six
layers, sliding and full attention at the 31B's head sizes (so both layer
types take the unfused paths they take on the real pack), keys equal values
on the full layers, and a KV-shared
tail (the last layer of each type reads an earlier layer's KV). The 31B-only
config gates are lifted; nothing under test depends on the sizes.

Exactness is checked on the CPU in float32. Chunked and whole-prompt prefills
agree to rounding there (measured max abs difference: 4e-4 on hidden states
of magnitude 9, 1.2e-4 on logits, 1.3e-4 on keys and values), and every greedy and
sampled continuation matches. They are not bit-equal: attention reduces the
same terms over rows of different lengths and offsets (a whole-prompt sliding
row carries every prompt key, a chunk's row its window plus the chunk), so
MLX groups the sums differently. In bf16 on the GPU the tiny model's own
rounding dominates: the whole-prompt path itself leaves a float32 reference
within the first 13 tokens, and the chunked path does no earlier.
"""

from __future__ import annotations

import time
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
gemma4_text = pytest.importorskip("mlx_lm.models.gemma4_text")

import mtplx.backends.gemma4_assistant as gemma4
from mtplx import generation
from mtplx.sampling import SamplerConfig
from mtplx.session_bank import SessionBank

# Read by the backend; spelled out so this file does not need the new names.
CHUNK_ENV = "MTPLX_GEMMA4_PREFILL_CHUNK_TOKENS"
VOCAB = 97
LAYERS = ("sliding_attention", "sliding_attention", "full_attention") * 2
GREEDY = SamplerConfig(temperature=0.0, top_p=1.0, top_k=0)
SAMPLED = SamplerConfig(temperature=1.0, top_p=0.95, top_k=20)

# On the CPU in float32 (measured maxima in the module docstring; at least
# seven times the worst of them).
STATE_ATOL = 1e-3
OUTPUT_ATOL = 5e-3


def _text_config(window: int, *, layers=LAYERS, shared: int = 2, hidden: int = 64) -> dict:
    return {
        "model_type": "gemma4_text",
        "hidden_size": hidden,
        "num_hidden_layers": len(layers),
        "intermediate_size": 2 * hidden,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 256,
        "global_head_dim": 512,
        "num_global_key_value_heads": 1,
        "attention_k_eq_v": True,
        "sliding_window": window,
        "num_kv_shared_layers": shared,
        "use_double_wide_mlp": False,
        "hidden_size_per_layer_input": 0,
        "enable_moe_block": False,
        "vocab_size": VOCAB,
        "max_position_embeddings": 65_536,
        "final_logit_softcapping": 30.0,
        "layer_types": list(layers),
        "tie_word_embeddings": True,
    }


class _Tokenizer:
    def decode(self, token_ids, **_kwargs):
        return " ".join(str(int(token)) for token in token_ids)


@pytest.fixture
def tiny_pair(monkeypatch):
    """Build a tiny Gemma 4 target and assistant on the backend's classes."""

    monkeypatch.setattr(gemma4.Gemma4TargetAdapter, "_validate_31b_dense", lambda self: None)
    monkeypatch.setattr(gemma4.Gemma4AssistantArgs, "validate_31b_dense", lambda self: None)

    def build(window: int, *, dtype=None, seed: int = 0) -> gemma4.Gemma4AssistantRuntime:
        dtype = mx.float32 if dtype is None else dtype
        mx.random.seed(seed)
        target = gemma4_text.Model(gemma4_text.ModelArgs.from_dict(_text_config(window)))
        target.set_dtype(dtype)
        draft_class, args_class = gemma4._assistant_model_classes({})
        drafter = draft_class(
            args_class(
                backbone_hidden_size=64,
                text_config=_text_config(window, layers=LAYERS[:3], shared=3, hidden=32),
            )
        )
        drafter.set_dtype(dtype)
        mx.eval(target.parameters(), drafter.parameters())
        return gemma4.Gemma4AssistantRuntime(
            target_model=target,
            tokenizer=_Tokenizer(),
            assistant_model=drafter,
            config=gemma4.Gemma4AssistantRuntimeConfig(
                target_model_path=Path("tiny-gemma4/target"),
                assistant_model_path=Path("tiny-gemma4/assistant"),
                draft_block_size=4,
            ),
        )

    return build


@pytest.fixture
def cpu():
    with mx.stream(mx.cpu):
        yield


def _prompt(rows: int, seed: int = 0) -> list[int]:
    return [int(token) for token in np.random.default_rng(seed).integers(0, VOCAB, size=rows)]


def _spy_forwards(runtime) -> list[tuple[str, int, int]]:
    """Record (phase, cache offset before, rows) for every target forward."""

    calls: list[tuple[str, int, int]] = []
    original = runtime.forward_target

    def spy(input_ids, *, cache=None, phase="unknown", compute_logits=True):
        calls.append((phase, runtime.target.cache_offset(cache), int(input_ids.shape[1])))
        return original(input_ids, cache=cache, phase=phase, compute_logits=compute_logits)

    runtime.forward_target = spy
    return calls


def _prefill(runtime, prompt, width, monkeypatch, *, cache=None):
    monkeypatch.setenv(CHUNK_ENV, str(width))
    calls = _spy_forwards(runtime)
    cache = runtime.make_cache() if cache is None else cache
    output, _elapsed = gemma4._gemma4_prefill_prompt(
        runtime, list(prompt), cache=cache, phase="prefill"
    )
    del runtime.forward_target
    return cache, output, [rows for _phase, _offset, rows in calls]


def _np(value) -> np.ndarray:
    return np.array(value.astype(mx.float32))


def _close(got, want, atol: float, what: str) -> None:
    got, want = _np(got), _np(want)
    assert got.shape == want.shape, what
    assert np.allclose(got, want, rtol=0.0, atol=atol), (
        f"{what}: max abs diff {float(np.max(np.abs(got - want)))}"
    )


def _same_bits(got, want, what: str) -> None:
    got, want = _np(got), _np(want)
    assert got.shape == want.shape, what
    assert np.array_equal(got, want), (
        f"{what}: not bit-equal, max abs diff {float(np.max(np.abs(got - want)))}"
    )


def _span_loop_prefill(runtime, prompt, spans):
    """A chunked prefill written out: one ``forward_target`` a span on a
    fresh cache, nothing evaluated between spans, the last row's logits from
    its hidden, and the drafter's sliding KV as the first span's return
    followed by each later span's own rows. On the CPU in float32 the
    backend's chunked prefill is this bit for bit. The one forward the old
    code ran over every row differs from it in the last bits (the attention
    sums group differently), so a comparison against it fails there, at the
    numbers, rather than at a count of forwards."""

    cache = runtime.make_cache()
    # The tiny pair's sliding layers are the ones whose shared KV comes from
    # a windowed (rotating) cache.
    windowed = ("sliding_attention",)
    outputs, start = [], 0
    for rows in spans:
        ids = mx.array([list(prompt[start : start + rows])], dtype=mx.int32)
        outputs.append(
            runtime.forward_target(ids, cache=cache, phase="prefill", compute_logits=False)
        )
        start += rows
    assert start == len(prompt)
    last = outputs[-1]
    hidden = last.hidden[:, -1:, :]
    shared = dict(last.shared_kv_states)
    for kind in windowed:
        parts = [outputs[0].shared_kv_states[kind]] + [
            tuple(part[..., -rows:, :] for part in out.shared_kv_states[kind])
            for out, rows in zip(outputs[1:], spans[1:])
        ]
        shared[kind] = tuple(
            mx.concatenate([part[index] for part in parts], axis=2) for index in (0, 1)
        )
    return cache, runtime.target.logits_from_hidden(hidden), hidden, shared


def _held_rows(item) -> tuple[mx.array, mx.array]:
    """The rows a cache holds, oldest first (what its next forward reads)."""

    if isinstance(item, gemma4.Gemma4RollbackRotatingKVCache):
        return item._temporal_order(item.keys), item._temporal_order(item.values)
    return item.keys[..., : item.offset, :], item.values[..., : item.offset, :]


# ---------------------------------------------------------------------------
# The chunk plan and its width.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("rows", "width", "spans"),
    [
        (300, 512, [300]),
        (512, 512, [512]),
        (513, 512, [2, 511]),
        (514, 512, [2, 512]),
        (1024, 512, [512, 512]),
        (1100, 512, [76, 512, 512]),
        (1025, 512, [2, 511, 512]),
        (2049, 2048, [2, 2047]),
        (4097, 2048, [2, 2047, 2048]),
        (7, 3, [2, 2, 3]),
        (9, 3, [3, 3, 3]),
        # A width of two cannot split an odd span into forwards of two or
        # more rows; the narrowest chunk is three.
        (8, 2, [2, 3, 3]),
    ],
)
def test_the_chunks_end_full_at_the_last_row_and_never_run_one_row(rows, width, spans):
    """The remainder goes first so the last chunk is full (the sliding caches
    keep their window plus it, what a banked prompt boundary can be trimmed
    back through). No forward runs more rows than the width the admission
    prices, or fewer than two: a one-row remainder becomes two rows ahead of
    a chunk one row short."""

    got = gemma4.gemma4_prefill_spans(rows, width)
    assert [end - start for start, end in got] == spans
    assert got[0][0] == 0 and got[-1][1] == rows
    assert all(a[1] == b[0] for a, b in pairwise(got))
    assert all(end - start >= 2 for start, end in got) or rows == 1
    assert all(end - start <= max(gemma4.GEMMA4_MIN_PREFILL_CHUNK, width) for start, end in got)


def test_the_width_is_the_house_chunk_unless_gemma_or_the_request_says_otherwise(
    monkeypatch,
):
    monkeypatch.delenv(CHUNK_ENV, raising=False)
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE", "2048")
    assert gemma4.gemma4_prefill_chunk_tokens(30_000) == 2048
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE", "1024")
    assert gemma4.gemma4_prefill_chunk_tokens(30_000) == 1024
    monkeypatch.setenv(CHUNK_ENV, "512")
    assert gemma4.gemma4_prefill_chunk_tokens(30_000) == 512
    # The request's own width (a caller's chunk, or the admission narrowing
    # one) is the most specific setting.
    with generation.prefill_chunk_size_override(256):
        assert gemma4.gemma4_prefill_chunk_tokens(30_000) == 256
    with generation.prefill_chunk_size_override(1):
        assert gemma4.gemma4_prefill_chunk_tokens(30_000) == 3
    for whole in ("whole", "0", "off"):
        monkeypatch.setenv(CHUNK_ENV, whole)
        assert gemma4.gemma4_prefill_chunk_tokens(30_000) is None
        with generation.prefill_chunk_size_override(256):
            assert gemma4.gemma4_prefill_chunk_tokens(30_000) is None


@pytest.mark.parametrize("knob", [None, "768", "whole"])
@pytest.mark.parametrize("requested", [None, 256, 4096])
def test_the_admission_prices_the_width_the_prefill_runs(monkeypatch, knob, requested):
    """One constructor for both sides: the admission's widest width for a
    request is the width the prefill runs once the server installs that
    request's chunk (``prefill_chunk_size_override``)."""

    if knob is None:
        monkeypatch.delenv(CHUNK_ENV, raising=False)
    else:
        monkeypatch.setenv(CHUNK_ENV, knob)
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE", "2048")
    runtime = object.__new__(gemma4.Gemma4AssistantRuntime)
    widths = runtime.prefill_forward_widths(30_000, requested)
    with generation.prefill_chunk_size_override(requested):
        runs = gemma4.gemma4_prefill_chunk_tokens(30_000)
    assert widths[0] == runs
    if knob == "whole":
        assert widths == [None]
    else:
        default = 768 if knob == "768" else 2048
        # The narrowest width the admission may fall back to is the default.
        assert widths[-1] == min(default, runs)


# ---------------------------------------------------------------------------
# Exactness on the tiny pair.
# ---------------------------------------------------------------------------


EXACTNESS_CASES = [
    # Window inside the chunk; the width does not divide the prompt.
    (16, 300, 64, [44, 64, 64, 64, 64]),
    # Window wider than the chunk.
    (128, 300, 64, [44, 64, 64, 64, 64]),
    # A one-row remainder: two rows, then a chunk one row short.
    (64, 2049, 512, [2, 511, 512, 512, 512]),
    (32, 1000, 96, [40] + [96] * 10),
]


@pytest.mark.parametrize(("window", "rows", "width", "spans"), EXACTNESS_CASES)
def test_a_chunked_prefill_forwards_its_spans(
    tiny_pair, monkeypatch, cpu, window, rows, width, spans
):
    """The structure alone: one forward a span (the old code: one forward
    over every row), in the prefill and in a generation's."""

    runtime = tiny_pair(window)
    prompt = _prompt(rows)
    _cache, _output, whole_forwards = _prefill(runtime, prompt, "whole", monkeypatch)
    _cache, _output, forwards = _prefill(runtime, prompt, width, monkeypatch)
    assert whole_forwards == [rows]
    assert forwards == spans
    calls = _spy_forwards(runtime)
    gemma4.generate_gemma4_ar(
        runtime, prompt, max_tokens=2, sampler=GREEDY, seed=0, stop_token_ids=set()
    )
    del runtime.forward_target
    assert [n for phase, _offset, n in calls if phase == "prefill"] == spans


@pytest.mark.parametrize(("window", "rows", "width", "spans"), EXACTNESS_CASES)
def test_a_chunked_prefill_is_the_span_loop_bit_for_bit(
    tiny_pair, monkeypatch, cpu, window, rows, width, spans
):
    """The numbers the prefill hands on (last logits and hidden, the
    drafter's shared KV, every cache's held rows) are those of the chunked
    prefill written out (``_span_loop_prefill``), to the bit."""

    runtime = tiny_pair(window)
    prompt = _prompt(rows)
    cache, chunked, _forwards = _prefill(runtime, prompt, width, monkeypatch)
    loop_cache, logits, hidden, shared = _span_loop_prefill(runtime, prompt, spans)
    _same_bits(chunked.logits, logits, "last logits")
    _same_bits(chunked.hidden, hidden, "last hidden")
    assert set(chunked.shared_kv_states) == set(shared)
    for kind, (keys, values) in shared.items():
        _same_bits(chunked.shared_kv_states[kind][0], keys, f"{kind} shared keys")
        _same_bits(chunked.shared_kv_states[kind][1], values, f"{kind} shared values")
    for index, (item, reference) in enumerate(zip(cache, loop_cache)):
        assert item.offset == reference.offset == rows
        for got, want, what in zip(_held_rows(item), _held_rows(reference), ("keys", "values")):
            _same_bits(got, want, f"cache {index} {what}")


@pytest.mark.parametrize(("window", "rows", "width", "spans"), EXACTNESS_CASES)
def test_a_chunked_prefill_is_the_whole_prompt_prefill(
    tiny_pair, monkeypatch, cpu, window, rows, width, spans
):
    """Chunked and whole-prompt prefills agree to rounding (the module
    docstring has the measured differences)."""

    runtime = tiny_pair(window)
    prompt = _prompt(rows)
    whole_cache, whole, _whole_forwards = _prefill(runtime, prompt, "whole", monkeypatch)
    cache, chunked, _forwards = _prefill(runtime, prompt, width, monkeypatch)

    assert chunked.cache_offset == whole.cache_offset == rows
    _close(chunked.logits, whole.logits, OUTPUT_ATOL, "last logits")
    _close(chunked.hidden, whole.hidden, OUTPUT_ATOL, "last hidden")

    # The drafter reads what one forward over the prompt returns: every row
    # of the last full-attention layer and of the last sliding layer.
    assert set(chunked.shared_kv_states) == set(whole.shared_kv_states) == {
        "sliding_attention",
        "full_attention",
    }
    for kind, (keys, values) in whole.shared_kv_states.items():
        got_keys, got_values = chunked.shared_kv_states[kind]
        assert int(got_keys.shape[-2]) == rows
        _close(got_keys, keys, STATE_ATOL, f"{kind} shared keys")
        _close(got_values, values, STATE_ATOL, f"{kind} shared values")

    last = spans[-1]
    assert len(cache) == len(whole_cache) == 4
    for index, (item, reference) in enumerate(zip(cache, whole_cache)):
        assert type(item) is type(reference)
        assert item.offset == reference.offset == rows
        keys, values = _held_rows(item)
        want_keys, want_values = _held_rows(reference)
        if isinstance(item, gemma4.Gemma4RollbackRotatingKVCache):
            # The window before the last chunk plus the last chunk, where the
            # whole-prompt forward left every row.
            assert int(keys.shape[-2]) == min(rows, window - 1 + last)
            assert int(want_keys.shape[-2]) == rows
            want_keys = want_keys[..., -int(keys.shape[-2]) :, :]
            want_values = want_values[..., -int(values.shape[-2]) :, :]
        _close(keys, want_keys, STATE_ATOL, f"cache {index} keys")
        _close(values, want_values, STATE_ATOL, f"cache {index} values")


def test_only_prompts_longer_than_one_chunk_are_split(tiny_pair, monkeypatch, cpu):
    """Up to one chunk the prefill is the single forward it always was, with
    the sliding caches' rollback record as before; past it, chunks."""

    runtime = tiny_pair(16)
    prompt = _prompt(66)
    cache, single, forwards = _prefill(runtime, prompt[:64], 64, monkeypatch)
    _whole_cache, whole, _ = _prefill(runtime, prompt[:64], "whole", monkeypatch)
    assert forwards == [64]
    assert np.array_equal(_np(single.logits), _np(whole.logits))
    assert np.array_equal(_np(single.hidden), _np(whole.hidden))
    windows = [c for c in cache if isinstance(c, gemma4.Gemma4RollbackRotatingKVCache)]
    assert windows and all(c._last_update is not None for c in windows)

    # One row past the width is two forwards, neither wider than the width
    # the admission priced.
    _cache, _output, forwards = _prefill(runtime, prompt[:65], 64, monkeypatch)
    assert forwards == [2, 63]

    cache, _output, forwards = _prefill(runtime, prompt, 64, monkeypatch)
    assert forwards == [2, 64]
    windows = [c for c in cache if isinstance(c, gemma4.Gemma4RollbackRotatingKVCache)]
    # Nothing rolls a prefill chunk back: no record holds a buffer, and the
    # caches record the next (speculative) update again.
    assert all(c._last_update is None and c._record_updates for c in windows)
    assert all(int(c.keys.shape[-2]) == min(66, 16 - 1 + 64) for c in windows)


@pytest.mark.parametrize(
    ("window", "rows", "width", "spans"),
    [(16, 300, 64, [44, 64, 64, 64, 64]), (32, 1000, 96, [40] + [96] * 10)],
)
def test_decode_after_a_chunked_prefill_is_token_identical(
    tiny_pair, monkeypatch, cpu, window, rows, width, spans
):
    """Decode starts from the chunked prefill written out, to the bit; then
    40 tokens target-only (greedy) and through the assistant (greedy and
    sampled at temperature 1.0 with a seed) are the same tokens, with the
    same acceptance, after either prefill."""

    runtime = tiny_pair(window, seed=1)
    prompt = _prompt(rows, seed=1)
    monkeypatch.setenv(CHUNK_ENV, str(width))
    state = gemma4._restore_or_prefill_gemma4_prompt(runtime, prompt, require_shared_kv=True)
    _loop_cache, logits, hidden, shared = _span_loop_prefill(runtime, prompt, spans)
    _same_bits(state.logits, logits[:, -1, :], "the logits decode starts from")
    _same_bits(state.hidden, hidden, "the hidden decode starts from")
    for kind, (keys, values) in shared.items():
        _same_bits(state.shared_kv_states[kind][0], keys, f"{kind} shared keys")
        _same_bits(state.shared_kv_states[kind][1], values, f"{kind} shared values")

    def run(chunk, loop, sampler):
        monkeypatch.setenv(CHUNK_ENV, str(chunk))
        calls = _spy_forwards(runtime)
        if loop == "ar":
            out = gemma4.generate_gemma4_ar(
                runtime, prompt, max_tokens=40, sampler=sampler, seed=7, stop_token_ids=set()
            )
        else:
            out = gemma4.generate_gemma4_assistant(
                runtime,
                prompt,
                max_tokens=40,
                sampler=sampler,
                seed=7,
                stop_token_ids=set(),
                speculative_depth=4,
            )
        del runtime.forward_target
        prefill = [n for phase, _offset, n in calls if phase == "prefill"]
        return list(out.tokens), out.stats, prefill

    for loop, sampler in (("ar", GREEDY), ("mtp", GREEDY), ("mtp", SAMPLED)):
        whole_tokens, whole_stats, _whole_prefill = run("whole", loop, sampler)
        tokens, stats, _prefill_rows = run(width, loop, sampler)
        assert len(tokens) == 40
        assert tokens == whole_tokens, (loop, sampler.temperature)
        assert stats.accepted_drafts == whole_stats.accepted_drafts
        assert stats.drafted_tokens == whole_stats.drafted_tokens


def test_the_drafter_proposes_the_same_block_after_either_prefill(
    tiny_pair, monkeypatch, cpu
):
    """The drafter's first block after a chunked prefill is its block after
    the chunked prefill written out, to the bit, and proposes the tokens it
    proposes after the whole-prompt prefill."""

    runtime = tiny_pair(16, seed=2)
    prompt = _prompt(300, seed=2)

    def propose(hidden, shared_kv_states, kv_offset):
        return runtime.propose_block(
            last_token_id=int(prompt[-1]),
            hidden=hidden,
            shared_kv_states=shared_kv_states,
            kv_offset=kv_offset,
            sampler=GREEDY,
            rng=np.random.default_rng(0),
            draft_block_size=6,
        )

    blocks = {}
    for width in ("whole", 64):
        _cache, output, _forwards = _prefill(runtime, prompt, width, monkeypatch)
        blocks[width] = propose(output.hidden, output.shared_kv_states, output.cache_offset)
    _loop_cache, _logits, hidden, shared = _span_loop_prefill(
        runtime, prompt, [44, 64, 64, 64, 64]
    )
    loop_block = propose(hidden, shared, len(prompt))
    assert blocks[64].token_ids == loop_block.token_ids
    for got, want in zip(blocks[64].steps, loop_block.steps, strict=True):
        _same_bits(got.logits, want.logits, "drafter logits")
        _same_bits(got.hidden, want.hidden, "drafter hidden")
    assert blocks[64].token_ids == blocks["whole"].token_ids
    assert len(blocks[64].steps) == 5
    for got, want in zip(blocks[64].steps, blocks["whole"].steps):
        _close(got.logits, want.logits, OUTPUT_ATOL, "drafter logits")
        _close(got.hidden, want.hidden, OUTPUT_ATOL, "drafter hidden")


# ---------------------------------------------------------------------------
# The restore paths prefill in chunks too.
# ---------------------------------------------------------------------------


def _bank_prompt_boundary(runtime, bank, prompt, state) -> None:
    """What the server banks for a turn: the pre-decode clone of the prompt
    cache (``capture_final_state``) and the drafter's state beside it."""

    entry = bank.put(
        runtime=runtime,
        token_ids=list(prompt),
        cache=gemma4._clone_gemma4_prompt_cache(runtime, state.cache),
        logits=state.logits,
        hidden=state.hidden,
        hidden_variant="gemma4_pre_norm",
        session_id="tiny",
        mtp_history_policy=gemma4.GEMMA4_SESSION_STATE_POLICY,
        extra_state=gemma4._gemma4_session_extra_state(
            shared_kv_states=state.shared_kv_states, kv_offset=state.kv_offset
        ),
    )
    assert entry is not None


def _restore(runtime, prompt, bank):
    return gemma4._restore_or_prefill_gemma4_prompt(
        runtime,
        list(prompt),
        session_bank=bank,
        session_restore_mode="clone",
        require_shared_kv=True,
    )


def _same_outputs(got, want) -> None:
    assert got.kv_offset == want.kv_offset
    _close(got.logits, want.logits, OUTPUT_ATOL, "logits")
    _close(got.hidden, want.hidden, OUTPUT_ATOL, "hidden")


def _same_shared_kv(got, want) -> None:
    assert set(got.shared_kv_states) == set(want.shared_kv_states)
    for kind, (keys, values) in want.shared_kv_states.items():
        _close(got.shared_kv_states[kind][0], keys, STATE_ATOL, f"{kind} keys")
        _close(got.shared_kv_states[kind][1], values, STATE_ATOL, f"{kind} values")


def _bank_cold_prompt(runtime, prompt, monkeypatch) -> SessionBank:
    bank = SessionBank(max_entries=4, max_bytes=8 * 1024**3, per_session_max_bytes=8 * 1024**3)
    state = gemma4._restore_or_prefill_gemma4_prompt(runtime, list(prompt), require_shared_kv=True)
    assert not state.cache_hit
    _bank_prompt_boundary(runtime, bank, prompt, state)
    return bank


def test_a_restored_prefix_prefills_its_suffix_in_chunks(tiny_pair, monkeypatch, cpu):
    """An exact restore of a banked prompt forwards the new suffix in chunks
    from the restored offset: the same state as forwarding it whole after
    the same restore, and the same logits and hidden as a cold prefill (the
    drafter's sliding KV after a restore is the kept window plus the suffix,
    as it always was)."""

    runtime = tiny_pair(16, seed=3)
    first = _prompt(400, seed=3)
    second = first + _prompt(300, seed=4)
    monkeypatch.setenv(CHUNK_ENV, "64")
    bank = _bank_cold_prompt(runtime, first, monkeypatch)

    calls = _spy_forwards(runtime)
    warm = _restore(runtime, second, bank)
    del runtime.forward_target
    assert warm.cache_hit and warm.cached_tokens == 400 and warm.suffix_tokens == 300
    assert [rows for _phase, _offset, rows in calls] == [44, 64, 64, 64, 64]
    assert calls[0][1] == 400
    assert int(warm.shared_kv_states["sliding_attention"][0].shape[-2]) == 16 - 1 + 300

    monkeypatch.setenv(CHUNK_ENV, "whole")
    warm_whole = _restore(runtime, second, bank)
    assert warm_whole.cache_hit and warm_whole.cached_tokens == 400
    _same_outputs(warm, warm_whole)
    _same_shared_kv(warm, warm_whole)
    cold = gemma4._restore_or_prefill_gemma4_prompt(runtime, second, require_shared_kv=True)
    assert not cold.cache_hit
    _same_outputs(warm, cold)


def test_a_rewritten_tail_restores_within_the_last_chunk(tiny_pair, monkeypatch, cpu):
    """A prompt that diverges inside the banked prompt's last chunk restores
    the common prefix (the sliding caches keep the window plus the last
    chunk) and prefills the rest in chunks, equal to a cold prefill. Deeper
    than the last chunk the bank finds no full window and the prompt runs
    cold: the whole-prompt forward kept every row of every sliding layer (on
    the 31B, 800 KiB a token) and could be trimmed to any depth."""

    runtime = tiny_pair(16, seed=5)
    first = _prompt(1200, seed=5)
    monkeypatch.setenv(CHUNK_ENV, "256")
    bank = _bank_cold_prompt(runtime, first, monkeypatch)

    shallow = first[:1100] + _prompt(600, seed=6)
    calls = _spy_forwards(runtime)
    warm = _restore(runtime, shallow, bank)
    del runtime.forward_target
    assert warm.cache_hit and warm.restore_mode.startswith("block_prefix")
    # The bank lands one slot short of the common prefix; that seed token
    # leads the tail, which runs in chunks.
    assert warm.cached_tokens == 1099 and warm.suffix_tokens == len(shallow) - 1099
    assert [rows for _phase, _offset, rows in calls] == [89, 256, 256]
    monkeypatch.setenv(CHUNK_ENV, "whole")
    _same_outputs(
        warm,
        gemma4._restore_or_prefill_gemma4_prompt(runtime, shallow, require_shared_kv=True),
    )

    monkeypatch.setenv(CHUNK_ENV, "256")
    deep = first[:700] + _prompt(600, seed=7)
    cold = _restore(runtime, deep, bank)
    assert not cold.cache_hit and cold.cached_tokens == 0


def test_the_bank_plans_only_the_restores_it_can_land(tiny_pair, monkeypatch, cpu):
    """The admission prices a request by the bank's restore plan
    (``SessionBank.restore_plan``). A banked chunked prompt keeps its window
    plus the last chunk, so the plan offers a near-prefix restore only where
    the trim can land and prices a deeper rewrite cold, the way the restore
    runs it. The review of 808a11e2: a banked 24,026-token prompt rewritten
    after 20,000 tokens was priced as a 20,000-token restore and ran cold."""

    runtime = tiny_pair(16, seed=5)
    first = _prompt(1200, seed=5)
    monkeypatch.setenv(CHUNK_ENV, "256")
    bank = _bank_cold_prompt(runtime, first, monkeypatch)
    (entry,) = list(bank._entries.values())
    # 1,200 rows in chunks of 256 (176, then four of 256): each sliding cache
    # keeps 15 + 256 rows, and the deepest exact trim leaves one window of 16.
    floor = 1200 - (15 + 256 - 16)
    identity = {
        "model_path": str(runtime.model_path),
        "mtp_enabled": bool(runtime.mtp_enabled),
        "hidden_variant": "gemma4_pre_norm",
        "mtp_history_policy": gemma4.GEMMA4_SESSION_STATE_POLICY,
    }
    # The restore lands one slot short of the match (the seed slot), so the
    # shallowest restorable match is one past the floor.
    for matched, restores in ((1100, True), (floor + 1, True), (floor, False), (700, False)):
        prompt = first[:matched] + _prompt(600, seed=matched)
        plan = bank.restore_plan(prompt, **identity)
        state = _restore(runtime, prompt, bank)
        assert state.cache_hit is restores, matched
        if restores:
            assert (plan["mode"], plan["reuse_tokens"]) == ("near_prefix", matched)
            assert state.cached_tokens == matched - 1
        else:
            assert (plan["mode"], plan["reuse_tokens"]) == ("none", 0), matched
            assert state.cached_tokens == 0
    assert list(bank._entries.values()) == [entry]
    assert entry.restore_floor_tokens == floor


# ---------------------------------------------------------------------------
# The request's abort check and the telemetry see every chunk.
# ---------------------------------------------------------------------------


def test_the_abort_check_runs_between_chunks(tiny_pair, monkeypatch, cpu):
    runtime = tiny_pair(16)
    prompt = _prompt(300)
    monkeypatch.setenv(CHUNK_ENV, "64")
    asked: list[int] = []

    def trips_on(call: int):
        def check() -> bool:
            asked.append(1)
            return len(asked) >= call

        return check

    calls = _spy_forwards(runtime)
    state = gemma4._restore_or_prefill_gemma4_prompt(
        runtime, prompt, require_shared_kv=True, abort_check=trips_on(99)
    )
    # Before the prefill, before its first chunk, between the five chunks,
    # before decode.
    assert len(asked) == 1 + 1 + 4 + 1
    assert state.kv_offset == 300

    asked.clear()
    calls.clear()
    with pytest.raises(generation.PostcommitAbort):
        gemma4._restore_or_prefill_gemma4_prompt(
            runtime, prompt, require_shared_kv=True, abort_check=trips_on(4)
        )
    # The fourth question comes before the third chunk.
    assert [rows for _phase, _offset, rows in calls] == [44, 64]


def test_an_abort_between_chunks_leaves_the_caches_and_the_bank_sound(
    tiny_pair, monkeypatch, cpu
):
    """An abort between the chunks of a restored prompt's suffix raises the
    abort and nothing else, turns the sliding caches' rollback records back
    on (the next speculative round rolls back exactly through them), and
    leaves the banked entry it restored from as it was: a restore after the
    abort is the restore before it, to the bit."""

    runtime = tiny_pair(16, seed=3)
    first = _prompt(400, seed=3)
    second = first + _prompt(300, seed=4)
    monkeypatch.setenv(CHUNK_ENV, "64")
    bank = _bank_cold_prompt(runtime, first, monkeypatch)
    (entry,) = list(bank._entries.values())
    before = _restore(runtime, second, bank)

    made: list[list] = []
    make_cache = runtime.make_cache

    def recording_make_cache():
        cache = make_cache()
        made.append(cache)
        return cache

    monkeypatch.setattr(runtime, "make_cache", recording_make_cache)
    asked: list[int] = []

    def trips_on_fourth() -> bool:
        asked.append(1)
        return len(asked) >= 4

    with pytest.raises(generation.PostcommitAbort):
        gemma4._restore_or_prefill_gemma4_prompt(
            runtime,
            second,
            session_bank=bank,
            session_restore_mode="clone",
            require_shared_kv=True,
            abort_check=trips_on_fourth,
        )
    (cache,) = made
    windows = [c for c in cache if isinstance(c, gemma4.Gemma4RollbackRotatingKVCache)]
    # Restored 400 rows, then two chunks of the suffix (44 and 64 rows).
    assert runtime.target.cache_offset(cache) == 400 + 44 + 64
    assert windows and all(c._record_updates and c._last_update is None for c in windows)

    # The bank: the same entry, and the same restore.
    assert list(bank._entries.values()) == [entry]
    monkeypatch.setattr(runtime, "make_cache", make_cache)
    after = _restore(runtime, second, bank)
    assert after.cache_hit and after.cached_tokens == before.cached_tokens == 400
    _same_bits(after.logits, before.logits, "logits after the abort")
    _same_bits(after.hidden, before.hidden, "hidden after the abort")
    for kind, (keys, values) in before.shared_kv_states.items():
        _same_bits(after.shared_kv_states[kind][0], keys, f"{kind} shared keys")
        _same_bits(after.shared_kv_states[kind][1], values, f"{kind} shared values")

    # The next speculative round on the aborted cache: a three-token verify
    # forward records its update and rolls back exactly.
    held = [(*(np.array(t) for t in _held_rows(c)), int(c.offset)) for c in cache]
    runtime.forward_target(
        mx.array([_prompt(3, seed=9)], dtype=mx.int32),
        cache=cache,
        phase="verify",
        compute_logits=False,
    )
    assert all(c._last_update is not None for c in windows)
    for item, (keys, values, offset) in zip(cache, held, strict=True):
        assert int(item.trim(3)) == 3
        assert int(item.offset) == offset
        got_keys, got_values = _held_rows(item)
        assert np.array_equal(np.array(got_keys), keys)
        assert np.array_equal(np.array(got_values), values)


def test_each_chunk_is_its_own_cache_delta(tiny_pair, monkeypatch, cpu):
    runtime = tiny_pair(16)
    runtime.telemetry = gemma4.Gemma4RuntimeTelemetry(trace_events=True)
    guarded: list[tuple[int, int]] = []

    class _Policy(gemma4.Gemma4LongContextPolicy):
        def guard_cache_delta(self, *, phase, before, after, offset, q_len):
            guarded.append((int(offset), int(q_len)))
            return super().guard_cache_delta(
                phase=phase, before=before, after=after, offset=offset, q_len=q_len
            )

    monkeypatch.setattr(runtime, "config", runtime.config.__class__(
        target_model_path=runtime.config.target_model_path,
        assistant_model_path=runtime.config.assistant_model_path,
        draft_block_size=runtime.config.draft_block_size,
        long_context_policy=_Policy(),
    ))
    _prefill(runtime, _prompt(300), 64, monkeypatch)
    events = [
        (event["offset"], event["q_len"])
        for event in runtime.telemetry.events
        if event.get("phase") == "prefill"
    ]
    expected = [(0, 44), (44, 64), (108, 64), (172, 64), (236, 64)]
    assert events == expected
    assert guarded == expected


# ---------------------------------------------------------------------------
# Memory and time.
# ---------------------------------------------------------------------------


def _measured_prefill(runtime, prompt, width, monkeypatch) -> tuple[int, float]:
    monkeypatch.setenv(CHUNK_ENV, str(width))
    mx.clear_cache()
    mx.synchronize()
    mx.reset_peak_memory()
    base = mx.get_active_memory()
    started = time.perf_counter()
    cache = runtime.make_cache()
    output, _elapsed = gemma4._gemma4_prefill_prompt(runtime, prompt, cache=cache, phase="prefill")
    mx.eval(output.logits)
    elapsed = time.perf_counter() - started
    peak = mx.get_peak_memory() - base
    del cache, output
    mx.clear_cache()
    return peak, elapsed


def test_the_prefill_peak_follows_the_chunk_not_the_prompt(tiny_pair, monkeypatch):
    """8,192 rows in bf16 on the tiny pair: one forward materializes 8,192 x
    8,192 scores a layer (1.3 GiB at peak, measured); 1,024-row chunks build
    1,024 x (cached + 1,024), about a tenth. The old code ignores the chunk
    setting and peaks the same either way."""

    runtime = tiny_pair(128, dtype=mx.bfloat16)
    prompt = _prompt(8192)
    _measured_prefill(runtime, prompt[:1024], 1024, monkeypatch)  # warm the kernels
    whole, _ = _measured_prefill(runtime, prompt, "whole", monkeypatch)
    chunked, _ = _measured_prefill(runtime, prompt, 1024, monkeypatch)
    assert whole > 4 * chunked, (whole, chunked)
    # Twice the prompt in chunks still peaks below half of it in one forward
    # (measured 209 MiB against 618): the chunked peak grows with the KV and
    # the last chunk's rows x prompt block, the one forward's with the
    # prompt squared.
    whole_half, _ = _measured_prefill(runtime, prompt[:4096], "whole", monkeypatch)
    assert chunked < whole_half, (chunked, whole_half)
