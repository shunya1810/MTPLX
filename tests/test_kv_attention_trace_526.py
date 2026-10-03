"""The KV attention diagnostic at the attention call site (issue #526).

#526 reported only "non-finite logits (nan=248320)": nothing said which KV
cache, route or offset served the attention. The split-attention hook now
emits one line per call under MTPLX_KV_ATTENTION_TRACE=1, only for
non-finite outputs under MTPLX_KV_ATTENTION_TRACE=nonfinite, and in every mode
records the call's host facts in the current request's record. The error
captures that record where it is raised, so the server reports the failing
request's own attention. A failure after a compiled replay names the trace of
the graph that ran (matched by the dispatched specialization), or says that
metadata is unavailable, instead of a stale eager record or another width's
trace.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from mtplx.attention_context import (
    compiled_dispatch,
    kv_attention_failure_line,
    kv_attention_request_scope,
)
from mtplx.attention_split import configure_split_full_attention
from mtplx.cache_state import TensorOffsetQuantizedPagedKVCache, VllmMetalPagedKVCache
from mtplx.compile_state import compiled_step_body, in_compiled_step_body
from mtplx.kv_quant import PagedKVQuantConfig
from mtplx.sampling import NonFiniteLogitsError

HEADS, KV_HEADS, HEAD_DIM, IN_DIM = 2, 1, 64, 8


class _Proj:
    def __init__(self, out_dim: int, in_dim: int, seed: int) -> None:
        mx.random.seed(seed)
        self.weight = 0.2 * mx.random.normal((out_dim, in_dim))
        mx.eval(self.weight)

    def __call__(self, x):
        return x @ self.weight.T


class _Norm:
    def __init__(self) -> None:
        self.weight = mx.ones((HEAD_DIM,))

    def __call__(self, x):
        return x


class _GatedAttention:
    """Qwen3Next-shaped gated attention the split hook accepts."""

    num_attention_heads = HEADS
    num_key_value_heads = KV_HEADS
    scale = HEAD_DIM**-0.5

    def __init__(self) -> None:
        self.q_proj = _Proj(2 * HEADS * HEAD_DIM, IN_DIM, 1)
        self.k_proj = _Proj(KV_HEADS * HEAD_DIM, IN_DIM, 2)
        self.v_proj = _Proj(KV_HEADS * HEAD_DIM, IN_DIM, 3)
        self.q_norm = _Norm()
        self.k_norm = _Norm()
        self.o_proj = lambda x: x

    def rope(self, x, offset=0):
        return x

    def __call__(self, x, mask=None, cache=None):
        raise AssertionError("split hook not installed")


class _Model:
    def __init__(self) -> None:
        layer = type("Layer", (), {"is_linear": False})()
        layer.self_attn = _GatedAttention()
        self.model = type("Inner", (), {"layers": [layer]})()


@pytest.fixture
def attn(monkeypatch):
    for name in ("MTPLX_SPLIT_FULL_ATTN", "MTPLX_SDPA_2PASS", "MTPLX_BLOCKWISE_ATTN", "MTPLX_GQA_PACKED_SDPA"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_ATTN", "1")
    model = _Model()
    configure_split_full_attention(model)
    with kv_attention_request_scope():
        yield model.model.layers[0].self_attn


def _pages(mode: str = "q8", rows: int = 5) -> VllmMetalPagedKVCache:
    cache = VllmMetalPagedKVCache(block_size=16, num_blocks=4, kv_quant_config=PagedKVQuantConfig(mode))
    mx.random.seed(11)
    cache.update_without_fetch(
        mx.random.normal((1, KV_HEADS, rows, HEAD_DIM)),
        mx.random.normal((1, KV_HEADS, rows, HEAD_DIM)),
    )
    return cache


def _x(rows: int = 2):
    mx.random.seed(13)
    return mx.random.normal((1, rows, IN_DIM))


def _lines(capsys) -> list[str]:
    return [line for line in capsys.readouterr().err.splitlines() if line.startswith("mtplx_kv_attention ")]


def _error() -> NonFiniteLogitsError:
    return NonFiniteLogitsError("non-finite logits in softmax", nan_count=5, inf_count=0, vocab_size=5)


def test_trace_prints_one_line_per_call_with_every_field(attn, capsys, monkeypatch):
    monkeypatch.setenv("MTPLX_KV_ATTENTION_TRACE", "1")
    attn(_x(), mask="causal", cache=_pages())
    (line,) = _lines(capsys)
    assert line == (
        "mtplx_kv_attention layer=0 phase=unknown cache=VllmMetalPagedKVCache "
        "bits=8 route=paged_kv_quant_dequant q_dtype=float32 offset=7 capacity=64 "
        "q_len=2 mask=causal fallback=- finite=1"
    )


def test_nonfinite_mode_prints_only_the_poisoned_call_and_keeps_its_record(attn, capsys, monkeypatch):
    monkeypatch.setenv("MTPLX_KV_ATTENTION_TRACE", "nonfinite")
    attn(_x(), mask="causal", cache=_pages())
    assert _lines(capsys) == []

    poisoned = _pages()
    scales = poisoned.value_scale_cache
    scales[0, 0] = float("nan")  # one bad fp32 row scale in the pages
    poisoned.value_scale_cache = scales
    attn(_x(), mask="causal", cache=poisoned)
    (line,) = _lines(capsys)
    assert "cache=VllmMetalPagedKVCache bits=8" in line
    assert line.endswith("finite=0")
    # The failure record follows the same call. Before, only the default
    # mode kept a record, so this mode reported nothing on failure.
    assert kv_attention_failure_line() == "mtplx_kv_attention dispatch=eager " + line.removeprefix(
        "mtplx_kv_attention "
    )


def test_default_mode_records_host_facts_without_printing(attn, capsys, monkeypatch):
    monkeypatch.delenv("MTPLX_KV_ATTENTION_TRACE", raising=False)
    attn(_x(), mask="causal", cache=_pages())
    assert _lines(capsys) == []
    assert kv_attention_failure_line() == (
        "mtplx_kv_attention dispatch=eager layer=0 phase=unknown "
        "cache=VllmMetalPagedKVCache bits=8 route=paged_kv_quant_dequant "
        "q_dtype=float32 offset=7 capacity=64 q_len=2 mask=causal fallback=- "
        "finite=unchecked"
    )


def test_promoted_adapter_line_names_the_array_mask_decline(attn, capsys, monkeypatch):
    monkeypatch.setenv("MTPLX_KV_ATTENTION_TRACE", "1")
    adapter = TensorOffsetQuantizedPagedKVCache.from_paged_cache(_pages())
    attn(_x(), mask=adapter.make_mask(2), cache=adapter)
    (line,) = _lines(capsys)
    assert "cache=TensorOffsetQuantizedPagedKVCache bits=8 route=dense_state_sdpa" in line
    assert "offset=7 capacity=64 q_len=2 mask=array_bool fallback=array_mask finite=1" in line

    # Default mode: the adapter's offset lives in an array, so the host-only
    # record says so instead of reading the device.
    monkeypatch.delenv("MTPLX_KV_ATTENTION_TRACE", raising=False)
    attn(_x(), mask=adapter.make_mask(2), cache=adapter)
    assert "offset=array capacity=64" in kv_attention_failure_line()


def test_the_error_carries_the_record_of_the_request_that_raised_it(attn, caplog, monkeypatch):
    """Request A fails; request B runs attention before A's error is
    formatted. Old code: A's report was B's call, because the record was one
    process-wide dict keyed by layer."""

    monkeypatch.delenv("MTPLX_KV_ATTENTION_TRACE", raising=False)
    with kv_attention_request_scope():
        attn(_x(), mask="causal", cache=_pages("q8"))
        error_a = _error()
    with kv_attention_request_scope():
        assert kv_attention_failure_line() is None  # a new request starts clean
        attn(_x(), mask="causal", cache=_pages("q4"))
        line_b = kv_attention_failure_line()
    assert "bits=8" in error_a.kv_attention and "bits=4" in line_b

    from mtplx.server.openai import _non_finite_logits_failure

    state = SimpleNamespace(dashboard=SimpleNamespace(), sessions=None)
    with caplog.at_level(logging.ERROR, logger="mtplx.server"):
        _non_finite_logits_failure(state, error_a, request_id="rid-a")
    messages = [record.getMessage() for record in caplog.records]
    assert any(error_a.kv_attention in message for message in messages)
    assert not any("bits=4" in message for message in messages)
    (event,) = state.dashboard.memory_guard_events
    assert event["action"] == "non_finite_logits"
    assert event["kv_attention"] == error_a.kv_attention


def _compiled_step(attn, adapter):
    """A compiled step over ``adapter``'s leaves, like the verify banks'."""

    def step(x, *state):
        for slot, leaf in enumerate(state):
            adapter.cache[slot] = leaf
        with compiled_step_body():
            return attn(x, mask=adapter.make_mask(int(x.shape[1])), cache=adapter)

    return mx.compile(step)


def test_a_failure_after_a_compiled_replay_names_the_traced_call(attn, capsys, monkeypatch):
    """The Python forward runs when a compiled step is traced, never when it
    replays. A record made in the traced body is kept as a trace of the
    dispatched graph, and after the dispatch the failure line says so,
    instead of presenting the last eager call as the failing dispatch."""

    monkeypatch.delenv("MTPLX_KV_ATTENTION_TRACE", raising=False)
    attn(_x(), mask="causal", cache=_pages())  # an eager call first
    adapter = TensorOffsetQuantizedPagedKVCache.from_paged_cache(_pages())
    leaves = list(adapter.cache)
    step = _compiled_step(attn, adapter)

    x = _x()
    with compiled_dispatch((id(step), tuple(x.shape))):
        mx.eval(step(x, *leaves))
    line = kv_attention_failure_line()
    assert line.startswith(
        "mtplx_kv_attention dispatch=compiled_replay_of_trace layer=0 phase=unknown "
        "cache=TensorOffsetQuantizedPagedKVCache bits=8 route=dense_state_sdpa"
    )
    assert "offset=array" in line

    # With the trace mode on, the printed trace line says what it is.
    monkeypatch.setenv("MTPLX_KV_ATTENTION_TRACE", "1")
    fresh = TensorOffsetQuantizedPagedKVCache.from_paged_cache(_pages())
    fresh_step = _compiled_step(attn, fresh)
    with compiled_dispatch((id(fresh_step), tuple(x.shape))):
        mx.eval(fresh_step(x, *list(fresh.cache)))
    (printed,) = _lines(capsys)
    assert "offset=traced" in printed and printed.endswith("finite=traced")

    # An eager call after the replay is the dispatch a failure then names.
    monkeypatch.delenv("MTPLX_KV_ATTENTION_TRACE", raising=False)
    attn(_x(), mask="causal", cache=_pages())
    assert kv_attention_failure_line().startswith("mtplx_kv_attention dispatch=eager ")


def test_a_replay_is_matched_to_the_trace_of_its_own_specialization(attn, monkeypatch):
    """Speculative decode alternates verify widths within one request. Trace a
    four-row and then a two-row graph, replay the cached four-row one: the
    failure names the four-row trace. Before, the record held one trace per
    layer and the replay marker no identity, so it named the two-row trace."""

    monkeypatch.delenv("MTPLX_KV_ATTENTION_TRACE", raising=False)
    adapter = TensorOffsetQuantizedPagedKVCache.from_paged_cache(_pages())
    leaves = list(adapter.cache)
    step = _compiled_step(attn, adapter)
    x4, x2 = _x(4), _x(2)
    four = (id(step), tuple(x4.shape))
    two = (id(step), tuple(x2.shape))

    for identity, x in ((four, x4), (two, x2)):  # the two traces
        with compiled_dispatch(identity):
            mx.eval(step(x, *leaves))
    for identity, x, q_len in ((four, x4, 4), (two, x2, 2), (four, x4, 4)):  # replays
        with compiled_dispatch(identity):
            mx.eval(step(x, *leaves))
        line = kv_attention_failure_line()
        assert line.startswith("mtplx_kv_attention dispatch=compiled_replay_of_trace ")
        assert f" q_len={q_len} " in line, line


def test_a_replay_of_a_graph_this_request_did_not_trace_says_so(attn, monkeypatch):
    monkeypatch.delenv("MTPLX_KV_ATTENTION_TRACE", raising=False)
    adapter = TensorOffsetQuantizedPagedKVCache.from_paged_cache(_pages())
    leaves = list(adapter.cache)
    step = _compiled_step(attn, adapter)
    x = _x()
    identity = (id(step), tuple(x.shape))
    with kv_attention_request_scope():  # an earlier request traced it
        with compiled_dispatch(identity):
            mx.eval(step(x, *leaves))
    attn(_x(), mask="causal", cache=_pages())  # this request: an eager call, then the replay
    with compiled_dispatch(identity):
        mx.eval(step(x, *leaves))
    line = kv_attention_failure_line()
    assert line.startswith("mtplx_kv_attention dispatch=compiled_replay (")
    assert "metadata is unavailable" in line


def test_an_abandoned_trace_does_not_leave_later_calls_labelled_as_traces(attn, monkeypatch):
    """MLX can abandon a traced Python frame without running its exit:
    ``np.asarray`` on a tracer raises MLX's refusal as a C++ exception through
    numpy's buffer protocol, and every ``finally`` up to the compiled call is
    skipped. The bank catches the failure and falls back eager, as it should,
    but the step-body flag the traced body had set stayed set, so every later
    eager call on the thread was recorded as a trace and a failure named a
    "compiled replay". The dispatcher now restores the flag it found."""

    from mtplx.graphbank import SpecDecodeGraphBank

    monkeypatch.delenv("MTPLX_KV_ATTENTION_TRACE", raising=False)

    class _NumpyRuntime:
        def forward_ar(self, input_ids, cache=None, return_hidden=True, hidden_variant=None):
            ids = np.asarray(input_ids)  # a tracer cannot be converted
            logits = mx.zeros((1, int(ids.shape[-1]), 4))
            return (logits, logits) if return_hidden else logits

    bank = SpecDecodeGraphBank(_NumpyRuntime())
    bank.forward_ar(mx.array([[1, 2]]))
    assert bank.stats.fallback_reasons == {"compile_error:ValueError": 1}
    # Old code: True from here on.
    assert in_compiled_step_body() is False
    attn(_x(), mask="causal", cache=_pages())
    assert kv_attention_failure_line().startswith("mtplx_kv_attention dispatch=eager layer=0 ")


def test_each_model_work_item_gets_its_own_record(attn, monkeypatch):
    from mtplx.server.openai import _submit_foreground_model_work

    monkeypatch.delenv("MTPLX_KV_ATTENTION_TRACE", raising=False)

    class _Inline:
        def submit(self, fn, *args, **kwargs):
            return fn(*args, **kwargs)

    state = SimpleNamespace(model_scheduler=None, generation_executor=_Inline())

    def job(mode):
        before = kv_attention_failure_line()
        attn(_x(), mask="causal", cache=_pages(mode))
        return before, kv_attention_failure_line()

    first = _submit_foreground_model_work(state, job, "q8")
    second = _submit_foreground_model_work(state, job, "q4")
    assert first[0] is None and "bits=8" in first[1]
    assert second[0] is None and "bits=4" in second[1]
