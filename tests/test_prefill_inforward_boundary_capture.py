"""Restore boundaries recorded INSIDE the last wide prefill forward (loop half).

The tail ladder ends a forward at every boundary it wants, and on Flash-Next a
forward costs about 0.1 s whatever its width (one pass over the routed
experts).  The in-forward boundary (on by default, ``MTPLX_GDN_BOUNDARY_INFORWARD=0``
restores the ladder) keeps the plain chunk grid and asks the model to record the recurrent state at the ladder's positions inside
the forwards that contain them.

What these tests pin, on the tiny Flash-Next model and the real prefill loops:

* the boundary POSITIONS are the ladder's, for every prompt length, chunk
  width and mandatory edge, and retention keeps the same ones;
* capturing changes nothing about the prefill itself: logits, hidden states,
  the trunk cache and the draft-head history are bit-identical to the same
  chunk grid run without capture;
* every banked record has the layout of the ladder's own snapshot call
  (states per entry, meta states, one hidden row) and the values of a forward
  that ENDS at that position;
* in bfloat16 on Metal, where a narrow and a wide forward give the same rows,
  every leaf of every record is bit-identical to the ladder's.  The capture
  adds no arithmetic of its own; what can differ on the full model is the
  GEMM rounding of a narrower forward, the class of every chunk-layout change;
* the chunk eval names the recurrent states, which frees the chunk's pre-conv
  streams (3.1 GB at 4,096 rows on Flash-Next), and the rollback switch
  restores the old eval set;
* with the switch off, a family without the hooks, or an image request, the
  loops run the ladder exactly as before; a missed capture is counted.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

import mtplx.generation as generation
from mtplx import demotions
from mtplx.cache_state import CacheSnapshot, _is_trimmable, restore_cache
from mtplx.generation import (
    _prefill_boundary_plan,
    _prefill_committed_mtp_history_streaming,
    _prefill_spans_with_tail_grid,
    _iter_prefill_chunk_spans,
)
from mtplx.models.qwen4_exp import TextArgs, TextModel

INTERVAL = 256
SWITCH = "MTPLX_GDN_BOUNDARY_INFORWARD"


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

PLAN_CASES = [
    (tokens, chunk, edges)
    for chunk in (1000, 2048, 4096, 8192)
    for tokens in (1, 255, 256, 257, 300, 1023, 1499, 2047, 2049, 4060, 16349, 65501, 131038)
    for edges in ((), (217,), (tokens - 40,), (2048,))
]


@pytest.fixture(autouse=True)
def _default_layout(monkeypatch):
    for name in (
        "MTPLX_GDN_BOUNDARY_TAIL_LAYOUT",
        "MTPLX_GDN_BOUNDARY_TAIL_MIN_RUNG",
        "MTPLX_GDN_BOUNDARY_TAIL_BACKOFF",
        "MTPLX_GDN_BOUNDARY_TAIL_INTERVAL",
        "MTPLX_GDN_BOUNDARY_MAX",
        "MTPLX_GDN_BOUNDARY_CAPTURE",
        "MTPLX_PREFILL_EVAL_RECURRENT_STATE",
        "MTPLX_PREFILL_CHUNK_TRACE",
        SWITCH,
    ):
        monkeypatch.delenv(name, raising=False)
    demotions.reset()
    yield
    demotions.reset()


def _plan(tokens, chunk, edges, *, inforward):
    return _prefill_boundary_plan(
        tokens,
        capture_boundaries=True,
        inforward=inforward,
        tail_interval=INTERVAL,
        mandatory_edges=edges,
        chunk_size=chunk,
    )


@pytest.mark.parametrize("tokens,chunk,edges", PLAN_CASES)
def test_the_boundary_positions_are_the_ladders(tokens, chunk, edges):
    ladder = _prefill_spans_with_tail_grid(
        tokens, tail_interval=INTERVAL, mandatory_edges=edges, chunk_size=chunk
    )
    spans, interior = _plan(tokens, chunk, edges, inforward=True)
    assert spans == list(_iter_prefill_chunk_spans(tokens, chunk_size=chunk))
    positions = sorted({end for _start, end in spans} | set(interior))
    assert positions == sorted(end for _start, end in ladder)
    # Every recorded position sits strictly inside exactly one forward.
    for position in interior:
        owners = [span for span in spans if span[0] < position < span[1]]
        assert len(owners) == 1


@pytest.mark.parametrize("tokens,chunk,edges", PLAN_CASES)
def test_the_plan_with_the_switch_off_is_the_ladder(tokens, chunk, edges):
    spans, interior = _plan(tokens, chunk, edges, inforward=False)
    assert interior == ()
    assert spans == _prefill_spans_with_tail_grid(
        tokens, tail_interval=INTERVAL, mandatory_edges=edges, chunk_size=chunk
    )


def test_without_boundary_capture_the_plan_is_the_plain_grid():
    spans, interior = _prefill_boundary_plan(
        5000,
        capture_boundaries=False,
        inforward=True,
        tail_interval=INTERVAL,
        mandatory_edges=(217,),
        chunk_size=2048,
    )
    assert interior == ()
    assert spans == [(0, 2048), (2048, 4096), (4096, 5000)]


def test_the_forwards_saved_on_the_cells_the_founder_measures():
    """4K cold at the 4,096 chunk: four forwards become two (plus the final
    token in both); a 1,500-token warm suffix: three become two."""

    for tokens, chunk, before, after in (
        (4060, 4096, 3, 1),
        (16349, 4096, 6, 4),
        (1499, 4096, 2, 1),
        (1499, 2048, 2, 1),
    ):
        ladder, _ = _plan(tokens, chunk, (), inforward=False)
        plain, interior = _plan(tokens, chunk, (), inforward=True)
        assert (len(ladder), len(plain)) == (before, after)
        assert len(interior) == before - after


# ---------------------------------------------------------------------------
# The loops, on the tiny model
# ---------------------------------------------------------------------------


def _tiny_args() -> TextArgs:
    return TextArgs(
        hidden_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=128,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=32,
        hc_count=2,
        hc_lowrank=16,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=16,
        indexer_budget=8,
        indexer_compress_ratio=2,
        ple_layer_ids=[2],
        ngram_vocab_size_base=512,
        heads_per_ngram=2,
        ple_embed_dim=64,
    )


class _TinyRuntime:
    """The runtime surface the prefill loops use, over the real tiny model.

    ``forward_ar`` is the family's own ``TextModel.__call__``; the draft-head
    history is a running sum so the test can see every appended row without a
    second model.
    """

    def __init__(self, model: TextModel):
        self.model = model
        self.mtp_enabled = True
        self.model_path = Path("tiny-inforward")
        self.diagnostic_counters: dict[str, int] = {}
        self.forwards: list[int] = []
        self.history_rows: list[np.ndarray] = []

    def make_cache(self):
        return self.model.make_cache()

    def make_mtp_cache(self):
        return []

    @property
    def embed_tokens(self):
        return self.model.model.embed_tokens

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
        self.forwards.append(int(tokens.shape[1]))
        return self.model(
            tokens,
            cache=cache,
            input_embeddings=input_embeddings,
            return_hidden=return_hidden,
            hidden_variant=hidden_variant,
            emit_logits=emit_logits,
            logits_keep=int(logits_keep or 0),
        )

    def update_mtp_cache(self, hidden_states, token_ids, **_kwargs):
        mx.eval(hidden_states)
        self.history_rows.append(np.array(hidden_states.astype(mx.float32)))
        return hidden_states.sum()


@pytest.fixture()
def tiny(monkeypatch):
    import mlx_lm.models.cache as cache_module
    import mtplx.models.qwen4_exp as qwen4_exp

    previous_device = mx.default_device()
    previous_arrays_cache = qwen4_exp.ArraysCache
    qwen4_exp.ArraysCache = cache_module.ArraysCache
    mx.set_default_device(mx.cpu)
    mx.random.seed(0)
    model = TextModel(_tiny_args())
    mx.eval(model.parameters())
    # Small numbers so a 96-token prompt has a real ladder: chunk 48, rungs
    # of 8 on an 8-token grid, the nearest boundary 3 tokens before the end.
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", "1")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE", "48")
    monkeypatch.setenv("MTPLX_GDN_BOUNDARY_TAIL_INTERVAL", "8")
    monkeypatch.setenv("MTPLX_GDN_BOUNDARY_TAIL_MIN_RUNG", "8")
    monkeypatch.setenv("MTPLX_GDN_BOUNDARY_TAIL_BACKOFF", "3")
    yield model
    qwen4_exp.ArraysCache = previous_arrays_cache
    mx.set_default_device(previous_device)


def _prompt(tokens: int, seed: int = 7) -> list[int]:
    rng = np.random.default_rng(seed)
    return [int(token) for token in rng.integers(0, 128, size=tokens)]


def _cold(model, prompt, *, sink, inforward, monkeypatch, stable_prefix_len=None):
    monkeypatch.setenv(SWITCH, "1" if inforward else "0")
    rt = _TinyRuntime(model)
    out = _prefill_committed_mtp_history_streaming(
        rt, list(prompt), gdn_boundary_sink=sink, stable_prefix_len=stable_prefix_len
    )
    cache, logits, hidden = out[0], out[1], out[2]
    mx.eval(logits, hidden)
    return rt, cache, logits, hidden


def _leaves(value):
    if value is None:
        return []
    if isinstance(value, mx.array):
        return [np.array(value.astype(mx.float32)) if value.dtype == mx.bfloat16 else np.array(value)]
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            out.extend(_leaves(item))
        return out
    return []


def _cache_leaves(cache):
    return [_leaves(getattr(entry, "state", None)) for entry in cache]


def _assert_same_leaves(left, right):
    assert len(left) == len(right)
    for a, b in zip(left, right):
        if isinstance(a, list):
            _assert_same_leaves(a, b)
            continue
        assert a.shape == b.shape and a.dtype == b.dtype
        assert np.array_equal(a, b)


def _record_leaves(record):
    position, snapshot, hidden = record
    assert isinstance(snapshot, CacheSnapshot)
    return [_leaves(state) for state in snapshot.states], _leaves(hidden)


def test_with_the_switch_off_the_loop_runs_the_ladder(tiny, monkeypatch):
    prompt = _prompt(97)
    sink: list = []
    rt, _cache, _logits, _hidden = _cold(
        tiny, prompt, sink=sink, inforward=False, monkeypatch=monkeypatch
    )
    ladder = _prefill_spans_with_tail_grid(96, tail_interval=8, chunk_size=48)
    assert len(ladder) > 2  # the last chunk really is cut
    assert rt.forwards == [end - start for start, end in ladder] + [1]
    assert "prefill_inforward_boundary_captures" not in rt.diagnostic_counters


def test_the_loop_runs_plain_chunks_and_banks_the_ladders_positions(tiny, monkeypatch):
    prompt = _prompt(97)
    ladder_sink: list = []
    _cold(tiny, prompt, sink=ladder_sink, inforward=False, monkeypatch=monkeypatch)
    sink: list = []
    rt, _cache, _logits, _hidden = _cold(
        tiny, prompt, sink=sink, inforward=True, monkeypatch=monkeypatch
    )
    assert rt.forwards == [48, 48, 1]
    assert [int(record[0]) for record in sink] == [int(r[0]) for r in ladder_sink]
    interior = [p for p in (int(r[0]) for r in sink) if p not in (48, 96)]
    assert interior and rt.diagnostic_counters[
        "prefill_inforward_boundary_captures"
    ] == len(interior)
    assert demotions.counts()["inforward_boundary_capture_missed"] == 0
    # Same record shape as the ladder's: a snapshot per entry (None where the
    # entry rolls back by trimming) and one hidden row.
    for ours, theirs in zip(sink, ladder_sink):
        ours_states, ours_hidden = _record_leaves(ours)
        their_states, their_hidden = _record_leaves(theirs)
        assert [len(s) for s in ours_states] == [len(s) for s in their_states]
        assert [h.shape for h in ours_hidden] == [h.shape for h in their_hidden]
        assert ours[1].meta_states == theirs[1].meta_states


def test_retention_keeps_the_same_positions_as_the_ladder(tiny, monkeypatch):
    monkeypatch.setenv("MTPLX_GDN_BOUNDARY_MAX", "3")
    prompt = _prompt(97)
    ladder_sink: list = []
    _cold(tiny, prompt, sink=ladder_sink, inforward=False, monkeypatch=monkeypatch)
    sink: list = []
    _cold(tiny, prompt, sink=sink, inforward=True, monkeypatch=monkeypatch)
    assert len(sink) == 3
    assert [int(r[0]) for r in sink] == [int(r[0]) for r in ladder_sink]


def test_capturing_changes_nothing_about_the_prefill(tiny, monkeypatch):
    """Same chunk grid, capture on against capture off: every output bit."""

    prompt = _prompt(97)
    plain_rt, plain_cache, plain_logits, plain_hidden = _cold(
        tiny, prompt, sink=None, inforward=True, monkeypatch=monkeypatch
    )
    sink: list = []
    rt, cache, logits, hidden = _cold(
        tiny, prompt, sink=sink, inforward=True, monkeypatch=monkeypatch
    )
    assert plain_rt.forwards == rt.forwards == [48, 48, 1]
    assert np.array_equal(np.array(plain_logits), np.array(logits))
    assert np.array_equal(np.array(plain_hidden), np.array(hidden))
    _assert_same_leaves(_cache_leaves(plain_cache), _cache_leaves(cache))
    # The draft head saw the same rows in the same order.
    assert len(plain_rt.history_rows) == len(rt.history_rows)
    for a, b in zip(plain_rt.history_rows, rt.history_rows):
        assert np.array_equal(a, b)


def _snapshot_record_of_a_forward_ending_at(model, prompt, position, monkeypatch):
    """What the LADDER's own call banks at ``position`` when the forwards
    before it are the plain chunks: chunks up to the one holding ``position``,
    that chunk cut at ``position``, then ``_capture_gdn_boundary``."""

    rt = _TinyRuntime(model)
    cache = rt.make_cache()
    body = mx.array([prompt[:-1]])
    hidden = None
    for start, end in _iter_prefill_chunk_spans(len(prompt) - 1, chunk_size=48):
        stop = min(end, position)
        if stop <= start:
            break
        _logits, hidden = rt.forward_ar(
            body[:, start:stop], cache=cache, return_hidden=True, emit_logits=False
        )
        mx.eval(hidden)
    sink: list = []
    generation._capture_gdn_boundary(sink, position, cache, hidden_last=hidden[:, -1:, :])
    assert len(sink) == 1
    return sink[0]


def test_each_record_has_the_layout_of_the_ladders_snapshot_call(tiny, monkeypatch):
    prompt = _prompt(97)
    sink: list = []
    _cold(tiny, prompt, sink=sink, inforward=True, monkeypatch=monkeypatch)
    for record in sink:
        position = int(record[0])
        reference = _snapshot_record_of_a_forward_ending_at(
            tiny, prompt, position, monkeypatch
        )
        ours_states, ours_hidden = _record_leaves(record)
        ref_states, ref_hidden = _record_leaves(reference)
        assert len(ours_states) == len(ref_states)
        for mine, theirs in zip(ours_states, ref_states):
            assert [(x.shape, x.dtype) for x in mine] == [
                (x.shape, x.dtype) for x in theirs
            ]
            for x, y in zip(mine, theirs):
                assert np.allclose(x, y, rtol=1e-4, atol=1e-5)
        assert [h.shape for h in ours_hidden] == [h.shape for h in ref_hidden]
        assert np.allclose(ours_hidden[0], ref_hidden[0], rtol=1e-4, atol=1e-5)


# ---------------------------------------------------------------------------
# Bit identity with the ladder, in the shipped dtype, on the GPU
# ---------------------------------------------------------------------------


def _max_abs(left, right) -> float:
    worst = 0.0
    for a, b in zip(left, right):
        if isinstance(a, list):
            worst = max(worst, _max_abs(a, b))
        elif a.size:
            worst = max(
                worst,
                float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))),
            )
    return worst


@pytest.mark.skipif(not mx.metal.is_available(), reason="needs the Metal device")
def test_banked_records_are_bit_identical_to_the_ladders_in_bfloat16(monkeypatch):
    """The in-forward capture adds no arithmetic of its own.

    A record can differ from the ladder's for one reason only: the ladder
    computes the rows before the boundary in a NARROWER forward, and a GEMM
    may round a row differently at a different row count (the rounding class
    of every chunk-layout change).  So the premise is measured first: does a
    forward that ends at the boundary give the same hidden row as the wide
    forward?  Where it does (the tiny model in bfloat16 on Metal, every
    position), every leaf of every record must equal the ladder's bit for
    bit.  Where a device rounds the two forwards differently the records may
    differ by exactly that rounding and no more.
    """

    import mlx_lm.models.cache as cache_module
    import mtplx.models.qwen4_exp as qwen4_exp

    previous_device = mx.default_device()
    previous_arrays_cache = qwen4_exp.ArraysCache
    qwen4_exp.ArraysCache = cache_module.ArraysCache
    for name, value in (
        ("MTPLX_SUSTAINED_PREFILL", "1"),
        ("MTPLX_PREFILL_CHUNK_SIZE", "48"),
        ("MTPLX_GDN_BOUNDARY_TAIL_INTERVAL", "8"),
        ("MTPLX_GDN_BOUNDARY_TAIL_MIN_RUNG", "8"),
        ("MTPLX_GDN_BOUNDARY_TAIL_BACKOFF", "3"),
    ):
        monkeypatch.setenv(name, value)
    try:
        mx.set_default_device(mx.gpu)
        mx.random.seed(0)
        model = TextModel(_tiny_args())
        model.set_dtype(mx.bfloat16)
        mx.eval(model.parameters())
        prompt = _prompt(97)
        ladder_sink: list = []
        _cold(model, prompt, sink=ladder_sink, inforward=False, monkeypatch=monkeypatch)
        sink: list = []
        _cold(model, prompt, sink=sink, inforward=True, monkeypatch=monkeypatch)
        assert [int(r[0]) for r in sink] == [int(r[0]) for r in ladder_sink]
        assert len(sink) >= 4
        for ours, theirs in zip(sink, ladder_sink):
            ours_states, ours_hidden = _record_leaves(ours)
            their_states, their_hidden = _record_leaves(theirs)
            same_rows = np.array_equal(ours_hidden[0], their_hidden[0])
            if same_rows:
                _assert_same_leaves(ours_states, their_states)
            else:
                rounding = _max_abs(ours_hidden, their_hidden)
                assert _max_abs(ours_states, their_states) <= max(64 * rounding, 1e-3)
        # On this family's reference device the premise holds everywhere, so
        # the bit-identity branch is the one that ran.
        assert all(
            np.array_equal(_record_leaves(a)[1][0], _record_leaves(b)[1][0])
            for a, b in zip(sink, ladder_sink)
        ) or not _is_reference_gpu()
    finally:
        qwen4_exp.ArraysCache = previous_arrays_cache
        mx.set_default_device(previous_device)


def _is_reference_gpu() -> bool:
    """Apple GPU generation 17 (M5 class): where the receipts were taken."""

    try:
        name = str(mx.metal.device_info().get("architecture") or "")
    except Exception:
        return False
    return name.startswith("applegpu_g17")


# ---------------------------------------------------------------------------
# The chunk eval names the recurrent states (memory, not arithmetic)
# ---------------------------------------------------------------------------


def _lazy_recurrent_entry(rows: int = 4096, width: int = 1024):
    """A recurrent cache entry left the way a GDN layer leaves it: a lazy
    3-row copy whose parent is the chunk's whole pre-conv stream."""

    import mlx_lm.models.cache as cache_module

    chunk = mx.ones((1, rows, width), dtype=mx.float32) * 0.5
    mx.eval(chunk)
    parent = mx.concatenate([mx.zeros((1, 3, width), dtype=mx.float32), chunk], axis=1)
    entry = cache_module.ArraysCache(size=2)
    entry[0] = mx.contiguous(parent[:, -3:, :])
    entry[1] = mx.zeros((1, 2, 2, 2), dtype=mx.float32)
    hidden = parent.sum()
    return entry, hidden, int(parent.nbytes)


@pytest.mark.parametrize("named", (True, False))
def test_the_chunk_eval_frees_the_pre_conv_stream(named, monkeypatch):
    previous_device = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        if not named:
            monkeypatch.setenv("MTPLX_PREFILL_EVAL_RECURRENT_STATE", "0")
        mx.clear_cache()
        base = mx.get_active_memory()
        entry, hidden, parent_bytes = _lazy_recurrent_entry()
        assert not _is_trimmable(entry)
        generation._eval_prefill_chunk(None, hidden, [entry])
        held = mx.get_active_memory() - base
        if named:
            assert held < parent_bytes // 8
        else:
            # The rollback switch restores the old eval set, and with it the
            # pinned parent: this is the 3.1 GB at 4,096 rows on Flash-Next.
            assert held >= parent_bytes
        del entry, hidden
    finally:
        mx.set_default_device(previous_device)


def test_a_cache_only_chunk_still_evaluates_the_whole_cache(monkeypatch):
    previous_device = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        calls: list[int] = []
        real = generation._eval_cache_roots
        monkeypatch.setattr(
            generation, "_eval_cache_roots", lambda cache: (calls.append(1), real(cache))
        )
        entry, _hidden, _bytes = _lazy_recurrent_entry(rows=8, width=8)
        generation._eval_prefill_chunk(None, None, [entry])
        assert calls == [1]
    finally:
        mx.set_default_device(previous_device)


# ---------------------------------------------------------------------------
# A model without the hooks, an image request, a missed capture
# ---------------------------------------------------------------------------


def test_a_family_without_the_hooks_keeps_the_ladder(monkeypatch):
    monkeypatch.setenv(SWITCH, "1")

    class _NoHooks:
        model = object()

    assert generation._resolve_inforward_boundary_hooks(_NoHooks()) is None


def test_an_image_request_keeps_the_ladder(tiny, monkeypatch):
    monkeypatch.setenv(SWITCH, "1")
    rt = _TinyRuntime(tiny)
    assert generation._resolve_inforward_boundary_hooks(rt) is not None
    assert (
        generation._resolve_inforward_boundary_hooks(rt, vision_splice=object()) is None
    )


def test_the_switch_is_on_by_default(tiny):
    assert generation._resolve_inforward_boundary_hooks(_TinyRuntime(tiny)) is not None


def test_zero_restores_the_ladder(tiny, monkeypatch):
    monkeypatch.setenv(SWITCH, "0")
    assert generation._resolve_inforward_boundary_hooks(_TinyRuntime(tiny)) is None


def test_a_missed_capture_is_counted_and_named(tiny, monkeypatch):
    rt = _TinyRuntime(tiny)
    sink: list = []
    banked = generation._bank_inforward_boundaries(
        rt,
        sink,
        rt.make_cache(),
        {},
        (5, 9),
        span_start=0,
        position_base=0,
        hidden_chunk=None,
    )
    assert banked == 0 and sink == []
    assert demotions.counts()["inforward_boundary_capture_missed"] == 2
    assert "inforward_boundary_capture_missed" in demotions.KINDS


def test_a_partial_capture_never_becomes_a_boundary(tiny):
    cache = _TinyRuntime(tiny).make_cache()
    states = [None] * len(cache)  # no recurrent entry captured anything
    assert generation._boundary_snapshot_from_capture(cache, states) is None
    assert generation._boundary_snapshot_from_capture(cache, states[:-1]) is None


# ---------------------------------------------------------------------------
# The warm loop: a restored prefix, then the suffix
# ---------------------------------------------------------------------------


def _warm(model, prompt, cached, *, inforward, capture, monkeypatch):
    """Prefill ``prompt[:cached]`` cold, then the rest through the restored-
    suffix loop, the way a warm agent turn runs."""

    from types import SimpleNamespace

    monkeypatch.setenv(SWITCH, "0")
    monkeypatch.setenv("MTPLX_SMALL_SUFFIX_FUSED_MAX", "0")
    rt = _TinyRuntime(model)
    cache = rt.make_cache()
    out = rt.forward_ar(
        mx.array([prompt[:cached]]), cache=cache, return_hidden=True, emit_logits=False
    )
    mx.eval(out[1])
    rt.forwards.clear()
    restored = SimpleNamespace(
        cache=cache,
        mtp_history_cache=[],
        hidden=None,
        entry=SimpleNamespace(prefix_len=cached),
    )
    if inforward:
        monkeypatch.setenv(SWITCH, "1")
    sink: list | None = [] if capture else None
    logits, hidden, _forward_s, _history_s = generation._prefill_restored_prompt_suffix(
        rt,
        restored,
        list(prompt[cached:]),
        base_hidden_variant="post_norm",
        mtp_hidden_variant="post_norm",
        mtp_history_policy="committed",
        cached_tokens=cached,
        gdn_boundary_sink=sink,
    )
    mx.eval(logits, hidden)
    return rt, cache, logits, hidden, sink


def test_the_warm_suffix_loop_banks_the_ladders_absolute_positions(tiny, monkeypatch):
    prompt = _prompt(131, seed=11)
    cached = 60
    ladder_rt, _c, _l, _h, ladder_sink = _warm(
        tiny, prompt, cached, inforward=False, capture=True, monkeypatch=monkeypatch
    )
    rt, _c2, _l2, _h2, sink = _warm(
        tiny, prompt, cached, inforward=True, capture=True, monkeypatch=monkeypatch
    )
    assert len(ladder_rt.forwards) > len(rt.forwards)
    assert rt.forwards == [48, 22, 1]
    positions = [int(record[0]) for record in sink]
    assert positions == [int(record[0]) for record in ladder_sink]
    assert all(position > cached for position in positions)
    assert positions[-1] == len(prompt) - 1


def test_capturing_changes_nothing_about_a_warm_suffix(tiny, monkeypatch):
    prompt = _prompt(131, seed=11)
    _rt, plain_cache, plain_logits, plain_hidden, _none = _warm(
        tiny, prompt, 60, inforward=True, capture=False, monkeypatch=monkeypatch
    )
    rt, cache, logits, hidden, sink = _warm(
        tiny, prompt, 60, inforward=True, capture=True, monkeypatch=monkeypatch
    )
    assert sink and rt.diagnostic_counters["prefill_inforward_boundary_captures"] >= 1
    assert np.array_equal(np.array(plain_logits), np.array(logits))
    assert np.array_equal(np.array(plain_hidden), np.array(hidden))
    _assert_same_leaves(_cache_leaves(plain_cache), _cache_leaves(cache))


def test_only_the_recurrent_containers_are_named_in_the_chunk_eval(tiny):
    """Attention entries and unknown containers are left alone: reading an
    unknown container's ``state`` can itself be work."""

    class _Opaque:
        def is_trimmable(self):
            return False

        @property
        def state(self):  # pragma: no cover - must never be read
            raise AssertionError("an unknown container's state was read")

    cache = _TinyRuntime(tiny).make_cache()
    mx.eval(tiny.model(mx.array([_prompt(12)]), cache))
    recurrent = [entry for entry in cache if not _is_trimmable(entry)]
    leaves = generation._recurrent_state_leaves([*cache, _Opaque(), None])
    expected = [leaf for entry in recurrent for leaf in entry.cache if leaf is not None]
    assert len(leaves) == len(expected) and len(leaves) >= 2 * len(recurrent)
    assert all(a is b for a, b in zip(leaves, expected))
