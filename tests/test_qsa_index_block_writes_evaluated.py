"""A stock QSACache's index-block writes, and the draft history's appends, are
evaluated with the work that makes them (#544).

``QSACache.write_pooled`` stores each completed index block in ``pooled`` and
in its fp32 mirror with lazy slice updates. A forward reads those buffers only
past the selection budget (2,048 tokens on Flash-Next), and then only one of
the two; a draft-head history append reads neither. Before the fix an unread
buffer stacked one update per block into a single unevaluated graph for the
life of the cache. The graph kept every written block alive, and with it the
Metal shared event of the ``mx.async_eval`` that computed the block. The
session bank snapshots lazily, so each banked final state held one event per
index write of its request. A long-running non-stream Flash-Next server then
failed with ``[Event::Event] Failed to create Metal shared event`` after 140K
to 260K generated tokens.

The draft history has the same shape one level up: with
``MTPLX_LAZY_MTP_HISTORY_APPEND`` each append is left for the next draft
forward to evaluate, and a context-copy round appends without drafting.

Pinned here:
- after every evaluated forward the pooled buffer and its mirror hold no
  unevaluated work, below and above the budget;
- a forward whose output is dropped leaves only its own writes pending;
- the dependency changes no bit of any output or cache leaf, on the CPU in
  fp32 and on the GPU in bf16 and fp16, through the eager, fused and compiled
  indexer routes;
- the cache's byte count includes the fp32 mirror;
- the event counter sees the events that unread arrays hold, live Metal
  shared events stay flat over many rounds, and a lazy bank snapshot of the
  cache pins none of them;
- a tiny Flash-Next generation runs its compiled verify rounds and hands the
  bank a final state with nothing pending, whatever its length;
- appends that no draft reads never stack: every append of a context-copy
  streak starts from an evaluated history, and a generation that ends inside
  the streak leaves nothing pending.
"""

from __future__ import annotations

import dataclasses
import gc
import importlib.util
import io
import os
import shutil
import subprocess
from pathlib import Path

import mlx.core as mx
import mlx.utils
import pytest

import mtplx.generation as generation
import mtplx.graphbank as graphbank
import mtplx.models.qwen4_exp as qwen4_exp
from mtplx.cache_state import snapshot_cache, snapshot_cache_lazy_hybrid
from mtplx.models.qwen4_exp import Attention, QSACache, TextArgs

_GPU = mx.metal.is_available()
_NEEDS_GPU = pytest.mark.skipif(not _GPU, reason="needs the Metal GPU")
_RATIO = 2


def _tiny_args() -> TextArgs:
    # budget 8 / ratio 2: selection engages once more than 4 blocks complete,
    # so a short run covers the unread regime and the selecting one.
    return TextArgs(
        hidden_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=16,
        indexer_budget=8,
        indexer_compress_ratio=_RATIO,
    )


def _cache() -> QSACache:
    # The model gives each layer's cache the layer's compression ratio.
    return QSACache(_RATIO)


def _layer(device, dtype=mx.float32) -> Attention:
    mx.set_default_device(device)
    mx.random.seed(0)
    layer = Attention(_tiny_args())
    if dtype != mx.float32:
        layer.update(mlx.utils.tree_map(lambda p: p.astype(dtype), layer.parameters()))
    mx.eval(layer.parameters())
    return layer


@pytest.fixture()
def restore_device():
    previous = mx.default_device()
    yield
    mx.set_default_device(previous)


def _hidden(tokens: int, seed: int, dtype=mx.float32) -> mx.array:
    mx.random.seed(seed)
    return mx.random.normal((1, tokens, 64)).astype(dtype)


def _pending_ops(array: mx.array | None) -> int:
    """Primitives not yet evaluated in ``array``'s graph (0 once evaluated)."""

    if array is None:
        return 0
    out = io.StringIO()
    mx.export_to_dot(out, array)
    return out.getvalue().count("shape=rectangle")


def _pending_writes(array: mx.array | None) -> int:
    """Slice updates not yet evaluated in ``array``'s graph."""

    if array is None:
        return 0
    out = io.StringIO()
    mx.export_to_dot(out, array)
    return out.getvalue().count('label ="SliceUpdate"')


_BIT_VIEWS = {mx.float32: mx.uint32, mx.bfloat16: mx.uint16, mx.float16: mx.uint16}


def _same_bits(a: mx.array, b: mx.array) -> bool:
    """Equal shape, dtype and bit pattern (so -0.0 and NaN payloads count)."""

    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    view = _BIT_VIEWS.get(a.dtype)
    if view is None:
        return bool(mx.array_equal(a, b).item())
    return bool(mx.array_equal(a.view(view), b.view(view)).item())


def _qsa_leaves(entries) -> list[mx.array]:
    return [
        leaf
        for entry in entries
        for leaf in (
            entry.kv.keys,
            entry.kv.values,
            entry.raw_keys,
            entry.pooled,
            entry.pooled_f32_t,
        )
        if leaf is not None
    ]


def _live_metal_shared_events() -> int | None:
    """Live ``MTLSharedEvent`` objects in this process, from ``heap(1)``.

    heap(1) prints no line for a class with no live instance, so a missing
    line reads as zero; the positive control below proves the class name.
    """

    tool = shutil.which("heap")
    if tool is None:
        return None
    result = subprocess.run(
        [tool, str(os.getpid())], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines():
        if "_MTLSharedEvent" in line:
            return int(line.split()[0])
    return 0


def test_every_evaluated_forward_leaves_the_index_buffers_evaluated(restore_device):
    layer = _layer(mx.cpu)
    cache = _cache()
    mx.eval(layer(_hidden(2, seed=1), cache))
    pooled_seen = mirror_seen = False
    for step in range(30):
        out = layer(_hidden(1, seed=100 + step), cache)
        mx.eval(out)
        pooled_seen |= cache.pooled is not None
        mirror_seen |= cache.pooled_f32_t is not None
        assert _pending_ops(cache.pooled) == 0, f"pooled backlog after forward {step}"
        assert _pending_ops(cache.pooled_f32_t) == 0, f"mirror backlog after forward {step}"
    # The run crossed the budget (2 + 30 tokens, 16 blocks against a top-k
    # of 4), so both the unread regime and the selecting one were exercised.
    assert cache.offset == 32
    assert pooled_seen and mirror_seen


def test_a_dropped_forward_leaves_only_its_own_writes_pending(restore_device):
    layer = _layer(mx.cpu)
    cache = _cache()
    mx.eval(layer(_hidden(4, seed=2), cache))
    # Two rows complete one block. The output is dropped, the way a lazy
    # draft-head history append drops its own.
    layer(_hidden(2, seed=3), cache)
    one_forward = _pending_ops(cache.pooled)
    assert one_forward > 0
    mx.eval(layer(_hidden(1, seed=4), cache))
    assert _pending_ops(cache.pooled) == 0
    assert _pending_ops(cache.pooled_f32_t) == 0


def _run_sequence(layer: Attention, dtype) -> list[mx.array]:
    """Prefill, decode across the budget, a rolled-back verify, a restore."""

    produced: list[mx.array] = []
    cache = _cache()
    produced.append(layer(_hidden(6, seed=10, dtype=dtype), cache))
    for step in range(8):
        produced.append(layer(_hidden(1, seed=20 + step, dtype=dtype), cache))
        mx.eval(produced[-1])
    produced.append(layer(_hidden(4, seed=40, dtype=dtype), cache))
    mx.eval(produced[-1])
    assert cache.trim(3) == 3
    for step in range(6):
        produced.append(layer(_hidden(1, seed=50 + step, dtype=dtype), cache))
    resumed = _cache()
    resumed.state = cache.state
    for step in range(4):
        produced.append(layer(_hidden(1, seed=70 + step, dtype=dtype), resumed))
    produced.extend(leaf for leaf in resumed.state if leaf is not None)
    produced.append(resumed.pooled_f32_view(resumed.pooled_len))
    mx.eval(produced)
    return produced


_INDEXER_ROUTES = {
    "eager": {},
    "fused": {"MTPLX_FUSED_QSA_INDEXER": "1"},
    # The compiled indexer core sits behind the fused selector's switch.
    "compiled": {"MTPLX_FUSED_QSA_INDEXER": "1", "MTPLX_COMPILED_QSA_INDEXER": "1"},
}
_ROUTE_METHODS = {
    "eager": "_select_eager",
    "fused": "_select_fused",
    "compiled": "_call_rows_compiled",
}


def _arm_indexer_route(monkeypatch, route: str) -> dict[str, int]:
    """Select one indexer route and count the selections each route makes."""

    for name in ("MTPLX_FUSED_QSA_INDEXER", "MTPLX_COMPILED_QSA_INDEXER"):
        monkeypatch.delenv(name, raising=False)
    for name, value in _INDEXER_ROUTES[route].items():
        monkeypatch.setenv(name, value)
    calls = {name: 0 for name in _ROUTE_METHODS}
    for name, method in _ROUTE_METHODS.items():
        original = getattr(qwen4_exp.QSAIndexer, method)

        def counted(self, *args, _name=name, _original=original, **kwargs):
            calls[_name] += 1
            return _original(self, *args, **kwargs)

        monkeypatch.setattr(qwen4_exp.QSAIndexer, method, counted)
    return calls


_PARITY_CASES = [
    pytest.param(mx.cpu, mx.float32, "eager", id="cpu-fp32-eager"),
    *(
        pytest.param(mx.gpu, dtype, route, id=f"gpu-{label}-{route}", marks=_NEEDS_GPU)
        for label, dtype in (("bf16", mx.bfloat16), ("fp16", mx.float16))
        for route in _INDEXER_ROUTES
    ),
]


@pytest.mark.parametrize("device,dtype,route", _PARITY_CASES)
def test_the_dependency_changes_no_bit(restore_device, monkeypatch, device, dtype, route):
    calls = _arm_indexer_route(monkeypatch, route)
    layer = _layer(device, dtype)
    tied = _run_sequence(layer, dtype)
    # The selections ran on the armed route, not on a silent fallback.
    assert calls[route] > 0, calls
    if route == "eager":
        assert calls["fused"] == calls["compiled"] == 0, calls
    monkeypatch.setattr(qwen4_exp, "_after_index_block_writes", lambda rows, _cache: rows)
    untied = _run_sequence(layer, dtype)
    assert len(tied) == len(untied)
    for index, (a, b) in enumerate(zip(tied, untied)):
        assert _same_bits(a, b), f"value {index} differs"


def test_the_byte_count_includes_the_fp32_mirror(restore_device):
    layer = _layer(mx.cpu)
    cache = _cache()
    # Twelve rows complete six blocks, past the four-block budget.
    mx.eval(layer(_hidden(12, seed=6), cache))
    assert cache.pooled_f32_t is not None
    buffers = (cache.raw_keys, cache.pooled, cache.pooled_f32_t)
    assert cache.nbytes == cache.kv.nbytes + sum(buffer.nbytes for buffer in buffers)


def _hold_unread_events(count: int) -> list[mx.array]:
    """Arrays each computed by its own ``mx.async_eval`` and never read."""

    held = []
    for index in range(count):
        value = mx.full((4,), float(index + 1))
        mx.async_eval(value)
        held.append(value)
    mx.synchronize()
    return held


@pytest.mark.skipif(not _GPU, reason="Metal shared events exist only on the GPU")
def test_the_event_counter_sees_the_events_unread_arrays_hold(restore_device):
    mx.set_default_device(mx.gpu)
    before = _live_metal_shared_events()
    if before is None:
        pytest.skip("heap(1) is not available to count Metal shared events")
    held = _hold_unread_events(32)
    holding = _live_metal_shared_events()
    for value in held:
        value[0].item()
    read = _live_metal_shared_events()
    # Each asynchronous evaluation made one event; its array holds it until
    # the host reads the array.
    assert holding - before >= 32, (before, holding)
    assert holding - read >= 32, (holding, read)


@pytest.mark.skipif(not _GPU, reason="Metal shared events exist only on the GPU")
def test_metal_shared_events_stay_flat_across_rounds(restore_device):
    before = _live_metal_shared_events()
    if before is None:
        pytest.skip("heap(1) is not available to count Metal shared events")
    layer = _layer(mx.gpu, mx.bfloat16)
    cache = _cache()
    mx.eval(layer(_hidden(2, seed=5, dtype=mx.bfloat16), cache))
    for step in range(96):
        out = layer(_hidden(1, seed=200 + step, dtype=mx.bfloat16), cache)
        # The decode loop's shape: one pipelined evaluation per round, then
        # a host read of its result.
        mx.async_eval(out)
        out[0, -1, 0].item()
    live = _live_metal_shared_events()
    # The bank keeps a lazy snapshot of the final state.
    snapshots = [snapshot_cache([cache]), snapshot_cache_lazy_hybrid([cache])]
    del cache, out
    gc.collect()
    banked = _live_metal_shared_events()
    del snapshots
    gc.collect()
    # 96 rounds complete 48 index blocks; before the fix each one pinned the
    # event of the round that wrote it, live and in the snapshots.
    assert live - before <= 4, f"{live - before} Metal shared events pinned after 96 rounds"
    assert banked - before <= 4, f"{banked - before} Metal shared events pinned by the snapshots"
    # The counter reads this process's events at this point of the run.
    held = _hold_unread_events(16)
    assert _live_metal_shared_events() - banked >= 16
    del held


_SMOKE = Path(__file__).resolve().parents[1] / "scripts" / "qwen4exp_mtp_tiny_smoke.py"


@pytest.fixture(scope="module")
def tiny_pack():
    """The tiny random Flash-Next pack on the compiled fixed-M4 lane."""

    if not _GPU:
        pytest.skip("bfloat16 expert gathers need the GPU")
    import mlx_lm.models.cache as cache_module

    from mtplx.models.qwen4_exp import Model, ModelArgs, Qwen4ExpMTP
    from mtplx.mtp_patch import validate_mtp_support

    spec = importlib.util.spec_from_file_location("qwen4exp_mtp_tiny_smoke", _SMOKE)
    smoke = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(smoke)
    previous_device = mx.default_device()
    mx.set_default_device(mx.gpu)
    previous_arrays_cache = qwen4_exp.ArraysCache
    qwen4_exp.ArraysCache = cache_module.ArraysCache
    mx.random.seed(0)
    args = dataclasses.replace(
        smoke._tiny_text_args(),
        head_dim=32,
        indexer_head_dim=32,
        indexer_compress_ratio=4,
        ple_layer_ids=[1],
        rope_parameters={
            "mrope_interleaved": True,
            "mrope_section": [2, 1, 1],
            "partial_rotary_factor": 0.25,
            "rope_theta": 10000000,
            "rope_type": "default",
        },
    )
    model = Model(ModelArgs(model_type="qwen4_exp", text_config=dataclasses.asdict(args)))
    model.language_model.mtp = Qwen4ExpMTP(model.language_model.args)
    model.update(
        mlx.utils.tree_map(
            lambda p: p.astype(mx.bfloat16) if p.dtype == mx.float32 else p,
            model.parameters(),
        )
    )
    model.eval()
    mx.eval(model.parameters())
    assert validate_mtp_support(model)
    yield smoke, model
    qwen4_exp.ArraysCache = previous_arrays_cache
    mx.set_default_device(previous_device)


def _generate(tiny_pack, monkeypatch, max_tokens: int, env: dict[str, str] | None = None):
    from mtplx.qwen4_fixed_verify import install_qwen4_fixed_verify_route
    from mtplx.sampling import SamplerConfig

    smoke, model = tiny_pack
    for name in tuple(os.environ):
        if name.startswith("MTPLX_QWEN4_") or name in {
            "MTPLX_COMPILED_VERIFY",
            "MTPLX_STATE_REBASE_EVERY",
            "MTPLX_FAMILY_CAPTURE_COMMIT",
        }:
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MTPLX_COMPILED_VERIFY", "1")
    monkeypatch.setenv("MTPLX_LAZY_MTP_HISTORY_APPEND", "1")
    for name, value in (env or {}).items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(graphbank, "_compiled_verify_bits_gate_ok", lambda _rt: True)
    rt = smoke._tiny_runtime(model)
    install_qwen4_fixed_verify_route(rt)
    sampler = SamplerConfig(temperature=0.6, top_p=0.95, top_k=20)
    out = generation.generate_mtpk(
        rt,
        [3, 5, 7, 9, 11, 13] + list(range(20, 40)),
        max_tokens=max_tokens,
        sampler=sampler,
        draft_sampler=sampler,
        speculative_depth=3,
        seed=1234,
        mtp_cache_policy="persistent",
        mtp_history_policy="committed",
        verify_strategy="batched",
        stop_token_ids=set(),
        capture_final_state=True,
    )
    assert out.final_state is not None and len(out.tokens) == max_tokens
    return out


def _qsa_entries(cache) -> list[QSACache]:
    return [entry for entry in cache or () if isinstance(entry, QSACache)]


@pytest.mark.parametrize("max_tokens", [120, 360])
def test_a_generation_hands_the_bank_a_final_state_with_nothing_pending(
    tiny_pack, monkeypatch, max_tokens
):
    out = _generate(tiny_pack, monkeypatch, max_tokens)
    report = out.stats.graphbank["compiled_verify"]
    # The verify rounds ran on the compiled fixed-M4 lane, not on an eager
    # fallback that would take the fixed caches out of the picture.
    assert report["compiled_calls"] > 0 and report["fallback_calls"] == 0, report
    trunk = _qsa_entries(out.final_state.final_trunk_cache)
    head = _qsa_entries(out.final_state.final_committed_mtp_cache)
    assert trunk and head, "the tiny pack has QSA layers in the trunk and the draft head"
    # Before the fix the pooled buffers carried hundreds of slice updates
    # here, one per index write of the request, and the draft head's history
    # its last lazy appends.
    assert [_pending_writes(entry.pooled) for entry in trunk + head] == [0] * len(trunk + head)
    assert sum(_pending_ops(leaf) for leaf in _qsa_leaves(head)) == 0


def _copy_streak(tiny_pack, monkeypatch, max_tokens: int, *, lazy_history: bool = True):
    """A generation whose context-copy lane proposes a prompt block every round.

    The tiny random model rejects the blocks, so the lane runs its three
    probation rounds back to back, each appending the committed row to the
    draft history without drafting, then backs off and drafts as usual.
    Records the pending work of the draft history's keys as each append starts.
    """

    import mtplx.context_copy as context_copy

    monkeypatch.setattr(
        context_copy.NgramIndex, "find", lambda self, history, max_pos=None: (0, 64)
    )
    at_append: list[int] = []
    original = generation._append_mtp_history

    def recording(rt, mtp_cache, *args, **kwargs):
        at_append.append(
            sum(_pending_ops(entry.kv.keys) for entry in _qsa_entries(mtp_cache))
        )
        return original(rt, mtp_cache, *args, **kwargs)

    monkeypatch.setattr(generation, "_append_mtp_history", recording)
    # Copy rounds commit through the family capture-commit, the Flash-Next
    # server default.
    out = _generate(
        tiny_pack,
        monkeypatch,
        max_tokens,
        env={
            "MTPLX_FAMILY_CAPTURE_COMMIT": "1",
            "MTPLX_LAZY_MTP_HISTORY_APPEND": "1" if lazy_history else "0",
        },
    )
    return out, at_append


def test_history_appends_no_draft_reads_never_stack(tiny_pack, monkeypatch):
    out, at_append = _copy_streak(tiny_pack, monkeypatch, max_tokens=40)
    assert out.stats.context_copy_rounds == 3
    assert out.stats.drafted_tokens > 0
    # Each copy round appended on top of the previous round's unread append,
    # and the draft keys carried hundreds of pending operations by the third
    # round before the fix; now every append starts from an evaluated
    # history, so at most one append is ever pending.
    assert at_append and max(at_append) == 0, at_append


def test_a_generation_ending_in_a_copy_streak_leaves_no_pending_history(
    tiny_pack, monkeypatch
):
    out, at_append = _copy_streak(tiny_pack, monkeypatch, max_tokens=4)
    assert out.stats.context_copy_rounds == 3
    assert out.stats.drafted_tokens == 0
    assert max(at_append) == 0, at_append
    head = _qsa_entries(out.final_state.final_committed_mtp_cache)
    # The bank would have snapshotted every append of the streak as one
    # graph of several hundred operations before the fix.
    assert sum(_pending_ops(leaf) for leaf in _qsa_leaves(head)) == 0


def test_a_generation_decodes_the_same_tokens_without_the_dependency(tiny_pack, monkeypatch):
    tied = _generate(tiny_pack, monkeypatch, max_tokens=160)
    monkeypatch.setattr(qwen4_exp, "_after_index_block_writes", lambda rows, _cache: rows)
    untied = _generate(tiny_pack, monkeypatch, max_tokens=160)
    assert list(tied.tokens) == list(untied.tokens)
    assert _same_bits(tied.final_state.final_logits, untied.final_state.final_logits)


def test_evaluating_unread_appends_changes_no_token(tiny_pack, monkeypatch):
    lazy, _ = _copy_streak(tiny_pack, monkeypatch, max_tokens=40)
    # Evaluating every append where it is made is the reference order.
    eager, _ = _copy_streak(
        tiny_pack, monkeypatch, max_tokens=40, lazy_history=False
    )
    assert lazy.stats.context_copy_rounds == eager.stats.context_copy_rounds == 3
    assert list(lazy.tokens) == list(eager.tokens)
    assert _same_bits(lazy.final_state.final_logits, eager.final_state.final_logits)
    lazy_head = _qsa_leaves(_qsa_entries(lazy.final_state.final_committed_mtp_cache))
    eager_head = _qsa_leaves(_qsa_entries(eager.final_state.final_committed_mtp_cache))
    assert len(lazy_head) == len(eager_head)
    assert all(_same_bits(a, b) for a, b in zip(lazy_head, eager_head))
