"""Issue #526: a promoted paged KV cache must never write past its buffers.

The verify banks promote paged KV pages to fixed-shape tensor-offset adapters.
Before this fix nothing grew them: a window that crossed the capacity was
written by a dynamic ``mx.slice_update`` at the array offset, and MLX does not
clamp that update. On the head-major quantized banks (1, H_kv, rows, width)
the rows past the end of head h landed on the first rows of head h + 1 (its
attention sink) and the last head's rows went past the allocation, in the
payloads and in both fp32 scale planes, while the offset ran past the
capacity. Under q4/q8 KV the 27B then produced all-NaN logits.

Every write path is reserved now: an eager forward reserves in the adapter's
``make_mask`` (the mask is built once per forward from the capacity, before
any layer writes), ``SpecDecodeGraphBank`` reserves in its preflight, and
``CompiledVerifyBank`` falls back eager on a bucket that does not fit. These
tests build the failure's shape (31 or 32 rows in 32, two 16-row blocks, then
a window past the end) on each path, and compare the rounds after a growth
with a run whose buffers were large enough from the start.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

import mtplx.graphbank as graphbank_module
import mtplx.system_memory as system_memory
from mtplx.cache_state import (
    PagedKVGrowthRefused,
    TensorOffsetQuantizedPagedKVCache,
    TensorOffsetVllmMetalPagedKVCache,
    VllmMetalPagedKVCache,
    _concrete_offset,
    is_compile_trace_error,
    link_paged_window_group,
    reserve_paged_window,
)
from mtplx.gdn_capture import commit_captured_prefix
from mtplx.graphbank import CompiledVerifyBank, SpecDecodeGraphBank
from mtplx.kv_quant import PagedKVQuantConfig, quantize_symmetric

HEADS = 2
HEAD_DIM = 64
BLOCK = 16
BLOCKS = 2  # 32 rows
MODES = ["q8", "q4", "plain"]


@pytest.fixture(autouse=True)
def _growth_env(monkeypatch):
    monkeypatch.setenv("MTPLX_DYNAMIC_PAGED_KV", "1")
    for name in (
        "MTPLX_CONTEXT_WINDOW_TOKENS",
        "MTPLX_GRAPHBANK_QUANTIZED_PAGED",
        "MTPLX_GRAPHBANK_PRESERVE_PAGED_KV",
        "MTPLX_ALLOW_SWAP",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(graphbank_module, "_PREWARM_DONE", True)


def _arrays_cache_cls() -> type:
    # Resolved per use, like production (see test_graphbank_compiled_verify).
    import mlx_lm.models.cache as cache_module

    return cache_module.ArraysCache


def _config(mode: str):
    return None if mode == "plain" else PagedKVQuantConfig(mode)


def _adapter_cls(mode: str) -> type:
    if mode == "plain":
        return TensorOffsetVllmMetalPagedKVCache
    return TensorOffsetQuantizedPagedKVCache


def _window(rows: int, seed: int) -> tuple[mx.array, mx.array]:
    mx.random.seed(seed)
    keys = mx.random.normal((1, HEADS, rows, HEAD_DIM))
    values = mx.random.normal((1, HEADS, rows, HEAD_DIM))
    mx.eval(keys, values)
    return keys, values


def _promoted(mode: str, *, rows: int = 31, blocks: int = BLOCKS, seed: int = 5):
    paged = VllmMetalPagedKVCache(
        block_size=BLOCK, num_blocks=blocks, kv_quant_config=_config(mode)
    )
    paged.update_without_fetch(*_window(rows, seed=seed))
    return _adapter_cls(mode).from_paged_cache(paged)


def _rows(adapter, start: int, end: int) -> list[np.ndarray]:
    """Every buffer leaf's rows [start, end), head-major, as numpy."""

    if isinstance(adapter, TensorOffsetQuantizedPagedKVCache):
        if getattr(adapter, "layout", "bank") == "pages":
            # M1 family: token-major pages (blocks, block_size, H, width).
            out = []
            for slot in (0, 1, 3, 4):
                pages = adapter.cache[slot]
                flat = pages.reshape(-1, int(pages.shape[2]), int(pages.shape[3]))
                out.append(np.array(flat[start:end].transpose(1, 0, 2)[None, ...]))
            return out
        return [np.array(adapter.cache[slot][:, :, start:end, :]) for slot in (0, 1, 3, 4)]
    out = []
    for slot in (0, 1):
        pages = adapter.cache[slot]
        flat = pages.reshape(-1, int(pages.shape[2]), int(pages.shape[3]))
        rows = flat[start:end].transpose(1, 0, 2)[None, ...]
        out.append(np.array(rows.astype(mx.float32)))  # exact for bf16 and fp32
    return out


def _expected_rows(adapter, keys: mx.array, values: mx.array) -> list[np.ndarray]:
    """What a correct write of (keys, values) stores, in _rows() layout."""

    if isinstance(adapter, TensorOffsetQuantizedPagedKVCache):
        q_k, s_k = quantize_symmetric(keys, bits=adapter.kv_bits)
        q_v, s_v = quantize_symmetric(values, bits=adapter.kv_bits)
        return [np.array(x) for x in (q_k, q_v, s_k, s_v)]
    return [np.array(keys.astype(mx.float32)), np.array(values.astype(mx.float32))]


def _assert_same(got: list[np.ndarray], want: list[np.ndarray], what: str) -> None:
    assert len(got) == len(want)
    for leaf, (g, w) in enumerate(zip(got, want)):
        assert g.shape == w.shape, f"{what}: leaf {leaf} shape {g.shape} != {w.shape}"
        assert np.array_equal(g, w), f"{what}: leaf {leaf} differs"


# -- the adapters' own write path -------------------------------------------


@pytest.mark.parametrize("mode", MODES)
def test_eager_write_past_capacity_is_refused_before_any_row_moves(mode, monkeypatch):
    # Old code: "DID NOT RAISE". The write went through, offset 35 of 32.
    monkeypatch.delenv("MTPLX_DYNAMIC_PAGED_KV", raising=False)
    adapter = _promoted(mode)
    assert adapter.size() == 31 and adapter.capacity == 32
    before = _rows(adapter, 0, 32)

    with pytest.raises(ValueError, match="capacity exceeded"):
        adapter.update_without_fetch(*_window(4, seed=9))

    assert adapter.size() == 31
    _assert_same(_rows(adapter, 0, 32), before, "refused write")


@pytest.mark.parametrize("mode", MODES)
def test_ensure_capacity_appends_zero_blocks_and_keeps_every_row(mode):
    # Old code: AttributeError. Promoted paged adapters could not grow.
    adapter = _promoted(mode)
    before = _rows(adapter, 0, 32)

    assert adapter.ensure_capacity(35) is True
    # The eager pages' growth policy: 1.5x the blocks, at least the need.
    assert adapter.num_blocks == 3 and adapter.capacity == 48
    _assert_same(_rows(adapter, 0, 32), before, "grown buffers, old rows")
    assert all(not leaf.any() for leaf in _rows(adapter, 32, 48)), "new rows are zero"

    keys, values = _window(4, seed=9)
    adapter.update_without_fetch(keys, values)
    assert adapter.size() == 35 <= adapter.capacity
    _assert_same(_rows(adapter, 0, 31), [x[:, :, :31, :] for x in before], "rows below the window")
    _assert_same(_rows(adapter, 31, 35), _expected_rows(adapter, keys, values), "written window")


@pytest.mark.parametrize("mode", MODES)
def test_ensure_capacity_refuses_when_dynamic_growth_is_off(mode, monkeypatch):
    monkeypatch.delenv("MTPLX_DYNAMIC_PAGED_KV", raising=False)
    adapter = _promoted(mode)
    before = _rows(adapter, 0, 32)
    assert adapter.ensure_capacity(35) is False
    assert adapter.capacity == 32 and adapter.num_blocks == BLOCKS
    _assert_same(_rows(adapter, 0, 32), before, "refused growth")


@pytest.mark.parametrize("mode", MODES)
def test_reserve_paged_window_grows_every_sibling_or_refuses_before_any_row(mode, monkeypatch):
    first, second = _promoted(mode, seed=5), _promoted(mode, seed=6)
    link_paged_window_group([None, first, second])
    assert {id(m) for m in first._window_members()} == {id(first), id(second)}

    assert reserve_paged_window(first._window_members(), 4) == 2
    assert first.capacity == second.capacity == 48
    assert reserve_paged_window([first, second], 4) == 0  # fits now

    monkeypatch.delenv("MTPLX_DYNAMIC_PAGED_KV", raising=False)
    full = _promoted(mode)
    before = _rows(full, 0, 32)
    with pytest.raises(ValueError, match="refusing before any row is written"):
        reserve_paged_window([full], 4)
    assert full.capacity == 32 and full.size() == 31
    _assert_same(_rows(full, 0, 32), before, "refused reservation")


# -- the trace exemption -----------------------------------------------------


def test_trace_refusal_is_recognised_inside_mx_compile():
    # Pins MLX's refusal text: a reworded message must fail here, not turn
    # every traced write into an error.
    seen: dict[str, object] = {}

    def body(offset):
        try:
            offset.item()
        except ValueError as exc:
            seen["recognised"] = is_compile_trace_error(exc)
        seen["concrete"] = _concrete_offset(offset)
        return offset + 1

    mx.eval(mx.compile(body)(mx.array(3, dtype=mx.int32)))
    assert seen == {"recognised": True, "concrete": None}
    assert _concrete_offset(mx.array(7, dtype=mx.int32)) == 7


def test_concrete_offset_raises_every_other_value_error():
    # Old code: returned None, so a malformed offset read as "traced" and the
    # write went ahead unchecked.
    with pytest.raises(ValueError, match="length-1 arrays"):
        _concrete_offset(mx.array([1, 2], dtype=mx.int32))


# -- the offset read follows the bucket walk's batching rules ---------------


def test_reservation_reads_offsets_through_the_batched_offset_rules(monkeypatch):
    members = [_promoted("q8", rows=20, seed=s) for s in (1, 2, 3)]
    for member in members:
        member.cache[2] = member.cache[2] + 0  # pending, like after a trim
    calls: list[int] = []
    real_eval = mx.eval

    def counting_eval(*arrays):
        calls.append(len(arrays))
        return real_eval(*arrays)

    monkeypatch.setattr(mx, "eval", counting_eval)
    token = graphbank_module.set_paged_offsets_context_ok(True)
    try:
        assert reserve_paged_window(members, 4) == 0
        assert calls == [3], "one batched eval for every offset"

        calls.clear()
        graphbank_module.set_paged_offsets_context_ok(False)
        assert reserve_paged_window(members, 4) == 0
        assert calls == [], "past the long-context fence each offset syncs alone"

        calls.clear()
        graphbank_module.set_paged_offsets_context_ok(True)
        monkeypatch.setattr(graphbank_module, "_BATCH_PAGED_OFFSETS", False)
        assert reserve_paged_window(members, 4) == 0
        assert calls == [], "MTPLX_BATCH_PAGED_OFFSETS=0 opts out"
    finally:
        graphbank_module._PAGED_OFFSETS_CONTEXT_OK.reset(token)


# -- growth admission against the memory guard's reading --------------------


def _install_available(monkeypatch, available_bytes: int) -> None:
    total = 64 * 1024**3
    monkeypatch.setattr(
        system_memory,
        "_reader",
        lambda: system_memory.SystemMemory(
            available_bytes=int(available_bytes),
            total_bytes=total,
            level_percent=int(available_bytes * 100 // total),
        ),
    )


@pytest.mark.parametrize("mode", MODES)
def test_growth_that_would_cross_the_desktop_floor_is_refused_cleanly(mode, monkeypatch):
    # 64 GB Mac with 1 GiB left: under the 3.2 GiB shed floor already.
    _install_available(monkeypatch, 1 * 1024**3)
    adapter = _promoted(mode)
    before = _rows(adapter, 0, 32)
    with pytest.raises(PagedKVGrowthRefused, match="insufficient memory") as refused:
        reserve_paged_window([adapter], 4)
    assert isinstance(refused.value, MemoryError)  # the server answers 507
    assert adapter.capacity == 32 and adapter.size() == 31
    _assert_same(_rows(adapter, 0, 32), before, "refused growth")

    # The operator's --allow-swap (env form) accepts the swap, as for prompts.
    monkeypatch.setenv("MTPLX_ALLOW_SWAP", "1")
    assert reserve_paged_window([adapter], 4) == 1
    assert adapter.capacity == 48


@pytest.mark.parametrize("old_leaves", ["released", "still_referenced"])
def test_growth_peak_memory_stays_inside_the_admitted_transient(old_leaves, monkeypatch):
    """The admission bound is a true upper bound on what a growth allocates.

    Four linked q8 adapters of 2 heads x 256 dims grow from 8192 to 12288
    rows. Each adapter is evaluated as it grows, so an old leaf nobody else
    holds is released before the next adapter grows ("released"); the bound
    covers the case where every old leaf is still referenced, by a banked
    snapshot for example ("still_referenced"), and there the peak meets it.
    """

    _install_available(monkeypatch, 60 * 1024**3)
    width = 256
    members = []
    for seed in range(4):
        paged = VllmMetalPagedKVCache(
            block_size=BLOCK, num_blocks=512, kv_quant_config=PagedKVQuantConfig("q8")
        )
        mx.random.seed(seed)
        paged.update_without_fetch(
            mx.random.normal((1, HEADS, 8190, width)).astype(mx.bfloat16),
            mx.random.normal((1, HEADS, 8190, width)).astype(mx.bfloat16),
        )
        members.append(TensorOffsetQuantizedPagedKVCache.from_paged_cache(paged))
    del paged
    link_paged_window_group(members)
    mx.eval(*[leaf for m in members for leaf in m.cache if leaf is not None])
    held = (
        [m.cache[s] for m in members for s in m._leaf_slots()]
        if old_leaves == "still_referenced"
        else []
    )
    old_bytes = sum(int(m.cache[s].nbytes) for m in members for s in m._leaf_slots())
    bound = 0
    tail = 0
    for m in members:
        leaves, zero_tail = m._growth_bytes(768)  # 8192 -> 12288 rows
        bound += leaves
        tail = max(tail, zero_tail)
    bound += tail

    mx.clear_cache()
    mx.reset_peak_memory()
    active_before = mx.get_active_memory()
    assert reserve_paged_window(members, 4) == 4
    peak_delta = mx.get_peak_memory() - active_before
    assert all(m.capacity == 12288 for m in members)
    print(
        f"growth peak ({old_leaves}): old leaves {old_bytes} B, admitted bound "
        f"{bound} B, measured peak above the pre-growth active memory {peak_delta} B"
    )
    # The bound counts buffers; the only other allocations are the scalar
    # fill values of the zero tails (1 byte per int8 leaf, 4 per fp32 leaf).
    assert 0 < peak_delta <= bound + 64
    del held


# -- the forward paths ------------------------------------------------------


class TwoHeadPagedRuntime:
    """A GDN-like layer plus a TWO-KV-head attention layer over paged KV.

    Same cache-mutation pattern as the graph-bank toys: fresh-array slot
    assignment in the recurrent layer, ``update_and_fetch`` on the attention
    entry and a masked readout. Like every model forward it builds its
    attention mask from the cache (``create_attention_mask`` ->
    ``make_mask``) before any write, and reads with that mask when it is an
    array, so a mask narrower than the buffers fails loudly. Two KV heads
    make the old overflow visible as data (head 0's extra rows landing on
    head 1's first rows), not only as an offset past the capacity. The
    readout is pure MLX math: no Metal attention kernel runs; the banks,
    promotion, fallbacks and in-graph quantized writes are the real code.
    ``last_kv`` is the last EAGER forward's window: after a compiled call it
    holds that call's tracers, which must never be evaluated.
    """

    D = 4
    K = 3
    V = 5

    def __init__(self, mode: str, *, blocks: int = BLOCKS, seed: int = 7) -> None:
        self.mode = mode
        self.blocks = blocks
        mx.random.seed(seed)
        scale = 0.3
        width = HEADS * HEAD_DIM
        self.embed = mx.random.normal((self.V, self.D))
        self.w_conv = scale * mx.random.normal((self.K * self.D, self.D))
        self.w_out = scale * mx.random.normal((self.D, self.V))
        self.w_kp = 0.4 * mx.random.normal((self.D, width))
        self.w_vp = 0.4 * mx.random.normal((self.D, width))
        self.w_qp = 0.4 * mx.random.normal((self.D, width))
        self.w_ao = 0.05 * mx.random.normal((width, self.D))
        # Materialized now, as a loaded model's weights are: a lazy weight
        # captured by a compiled trace is recomputed inside the graph, and the
        # traces then disagree with the eager forward by far more than kernel
        # rounding, depending on what happened to be evaluated before them.
        mx.eval(self.embed, self.w_conv, self.w_out, self.w_kp, self.w_vp, self.w_qp, self.w_ao)
        self.last_kv: tuple[mx.array, mx.array] | None = None

    def make_cache(self) -> list:
        gdn = _arrays_cache_cls()(2)
        gdn[0] = mx.zeros((1, self.K, self.D))
        gdn[1] = mx.zeros((1, 1, self.D, self.D))
        paged = VllmMetalPagedKVCache(
            block_size=BLOCK, num_blocks=self.blocks, kv_quant_config=_config(self.mode)
        )
        return [gdn, paged]

    def _forward(self, input_ids, cache):
        from mlx_lm.models.base import create_attention_mask

        B, S = int(input_ids.shape[0]), int(input_ids.shape[1])
        gdn_entry, attn_entry = cache
        h = self.embed[input_ids]
        built_mask = create_attention_mask(h, attn_entry)

        conv = gdn_entry.cache[0]
        state = gdn_entry.cache[1]
        conv_steps, state_steps, outs = [], [], []
        for t in range(S):
            conv = mx.concatenate([conv[:, 1:, :], h[:, t : t + 1, :]], axis=1)
            mixed = mx.tanh(conv.reshape(B, -1) @ self.w_conv)
            state = mx.tanh(state + mixed[:, None, :, None] * mixed[:, None, None, :])
            conv_steps.append(conv)
            state_steps.append(state)
            outs.append(mx.sum(state, axis=-1))
        gdn_entry[0] = conv
        gdn_entry[1] = state
        gdn_entry.advance(S)
        h = h + mx.concatenate(outs, axis=1)

        def heads(w):
            return (h @ w).reshape(B, S, HEADS, HEAD_DIM).transpose(0, 2, 1, 3)

        keys, values = heads(self.w_kp), heads(self.w_vp)
        if self.mode == "plain":
            # Plain pages hold the activation dtype; the compiled paged lane
            # takes bf16/fp16 pages only (graphbank._paged_kernel_bucket_eligible).
            keys, values = keys.astype(mx.bfloat16), values.astype(mx.bfloat16)
        self.last_kv = (keys, values)
        k_buf, v_buf = attn_entry.update_and_fetch(keys, values)
        capacity = int(k_buf.shape[2])
        if isinstance(built_mask, mx.array):
            mask = built_mask.astype(mx.float32)
        else:
            offset = attn_entry.offset  # the stock cache's int, after the write
            limit = offset - S + 1 + mx.arange(S)
            mask = (mx.arange(capacity)[None, :] < limit[:, None]).astype(mx.float32)
        scores = heads(self.w_qp) @ mx.swapaxes(k_buf, 2, 3).astype(mx.float32)
        attn = (scores * mask) @ v_buf.astype(mx.float32)  # (B, H, S, HEAD_DIM)
        h = h + attn.transpose(0, 2, 1, 3).reshape(B, S, -1) @ self.w_ao
        captures = {
            0: {
                "conv_states": mx.stack(conv_steps, axis=1),
                "states": mx.stack(state_steps, axis=1),
            }
        }
        return h @ self.w_out, h, captures

    def forward_ar(self, input_ids, cache=None, return_hidden: bool = False, hidden_variant=None):
        del hidden_variant
        logits, h, _captures = self._forward(input_ids, cache)
        return (logits, h) if return_hidden else logits

    def forward_ar_capture(
        self,
        input_ids,
        cache=None,
        return_hidden: bool = False,
        hidden_variant: str | None = None,
        capture_backend: str | None = None,
    ):
        del hidden_variant, capture_backend
        logits, h, captures = self._forward(input_ids, cache)
        if return_hidden:
            return logits, h, captures
        return logits, captures


class ThreeLayerPagedRuntime(TwoHeadPagedRuntime):
    """The same toy with THREE attention layers over paged KV behind one mask.

    Like every model forward, the mask is built once, from the first
    full-attention layer's cache, and serves every attention layer; each
    layer writes its own pages. A layer missing from the first adapter's
    reservation group would reach its write unreserved and hit the backstop.
    """

    LAYERS = 3

    def __init__(self, mode: str, *, blocks: int = BLOCKS, seed: int = 7) -> None:
        super().__init__(mode, blocks=blocks, seed=seed)
        width = HEADS * HEAD_DIM
        mx.random.seed(seed + 100)
        self.layer_w = [
            tuple(0.4 * mx.random.normal((self.D, width)) for _ in range(3))
            for _ in range(self.LAYERS)
        ]
        mx.eval(*[w for ws in self.layer_w for w in ws])

    def make_cache(self) -> list:
        gdn, _paged = super().make_cache()
        pages = [
            VllmMetalPagedKVCache(
                block_size=BLOCK, num_blocks=self.blocks, kv_quant_config=_config(self.mode)
            )
            for _ in range(self.LAYERS)
        ]
        return [gdn, *pages]

    def _forward(self, input_ids, cache):
        from mlx_lm.models.base import create_attention_mask

        B, S = int(input_ids.shape[0]), int(input_ids.shape[1])
        gdn_entry, *attn_entries = cache
        h = self.embed[input_ids]
        built_mask = create_attention_mask(h, attn_entries[0])

        conv = gdn_entry.cache[0]
        state = gdn_entry.cache[1]
        conv_steps, state_steps, outs = [], [], []
        for t in range(S):
            conv = mx.concatenate([conv[:, 1:, :], h[:, t : t + 1, :]], axis=1)
            mixed = mx.tanh(conv.reshape(B, -1) @ self.w_conv)
            state = mx.tanh(state + mixed[:, None, :, None] * mixed[:, None, None, :])
            conv_steps.append(conv)
            state_steps.append(state)
            outs.append(mx.sum(state, axis=-1))
        gdn_entry[0] = conv
        gdn_entry[1] = state
        gdn_entry.advance(S)
        h = h + mx.concatenate(outs, axis=1)

        for (w_k, w_v, w_q), attn_entry in zip(self.layer_w, attn_entries):
            def heads(w):
                return (h @ w).reshape(B, S, HEADS, HEAD_DIM).transpose(0, 2, 1, 3)

            keys, values = heads(w_k), heads(w_v)
            if self.mode == "plain":
                keys, values = keys.astype(mx.bfloat16), values.astype(mx.bfloat16)
            k_buf, v_buf = attn_entry.update_and_fetch(keys, values)
            capacity = int(k_buf.shape[2])
            if isinstance(built_mask, mx.array):
                mask = built_mask.astype(mx.float32)
            else:
                offset = attn_entry.offset
                limit = offset - S + 1 + mx.arange(S)
                mask = (mx.arange(capacity)[None, :] < limit[:, None]).astype(mx.float32)
            scores = heads(w_q) @ mx.swapaxes(k_buf, 2, 3).astype(mx.float32)
            attn = (scores * mask) @ v_buf.astype(mx.float32)
            h = h + attn.transpose(0, 2, 1, 3).reshape(B, S, -1) @ self.w_ao
        captures = {
            0: {
                "conv_states": mx.stack(conv_steps, axis=1),
                "states": mx.stack(state_steps, axis=1),
            }
        }
        return h @ self.w_out, h, captures


def _prefill(rt, cache, rows: int) -> None:
    rt.forward_ar_capture(mx.array([[i % 5 for i in range(rows)]]), cache=cache, return_hidden=True)


def _verify(bank, rt, cache, ids, *, eager: bool = False):
    if eager:
        return rt.forward_ar_capture(mx.array([ids]), cache=cache, return_hidden=True)
    return bank.forward_ar_capture(mx.array([ids]), cache=cache)


@pytest.mark.parametrize("keep", [4, 2], ids=["accept", "trim_then_rewrite"])
@pytest.mark.parametrize("mode", MODES)
def test_compiled_bank_overflow_round_reserves_before_mutation(mode, keep):
    rt = TwoHeadPagedRuntime(mode)
    bank = CompiledVerifyBank(rt)
    cache = rt.make_cache()
    _prefill(rt, cache, 27)

    # Compiled round: promotes the pages and fills them to 31 of 32 rows.
    _l, _h, captures = _verify(bank, rt, cache, [1, 2, 3, 4])
    assert commit_captured_prefix(cache, captures, keep_tokens=4, verified_tokens=4)
    adapter = cache[1]
    assert type(adapter) is _adapter_cls(mode)
    assert bank.stats["compiled_calls"] == 1
    assert adapter.size() == 31 and adapter.capacity == 32
    committed = _rows(adapter, 0, 31)

    # 31 + 4 > 32: the bank falls back eager, and the forward's make_mask
    # grows the adapter before the first write.
    logits, _h, captures = _verify(bank, rt, cache, [4, 3, 2, 1])
    assert bank.stats["fallback_reasons"].get("capacity_overflow") == 1
    assert cache[1] is adapter
    # 2.12.0 fails here: head 0's rows 32..34 were written onto head 1's
    # rows 0..2 (and head 1's past the allocation).
    _assert_same(_rows(adapter, 0, 31), committed, "committed rows after the overflow round")
    # ... and here: offset 35 in 32 rows.
    assert adapter.size() == 35 <= adapter.capacity == 48
    assert adapter.grow_events == 1
    _assert_same(_rows(adapter, 31, 35), _expected_rows(adapter, *rt.last_kv), "overflow window")
    assert np.isfinite(np.array(logits)).all()

    # Accept all four rows, or reject two and let the next window rewrite them.
    assert commit_captured_prefix(cache, captures, keep_tokens=keep, verified_tokens=4)
    kept = 31 + keep
    assert adapter.size() == kept
    committed = _rows(adapter, 0, kept)
    resumed, _h, _c = _verify(bank, rt, cache, [2, 2, 2, 2])
    # Back on the compiled route, over the grown buffers.
    assert bank.stats["compiled_calls"] == 2
    assert adapter.size() == kept + 4 <= adapter.capacity
    _assert_same(_rows(adapter, 0, kept), committed, "committed rows after the next window")

    # The same rounds on buffers that were 48 rows from the start, with the
    # same route per round, give the same logits and the same rows.
    ref_rt = TwoHeadPagedRuntime(mode, blocks=3)
    ref_bank = CompiledVerifyBank(ref_rt)
    ref = ref_rt.make_cache()
    _prefill(ref_rt, ref, 27)
    _l, _h, c = _verify(ref_bank, ref_rt, ref, [1, 2, 3, 4])
    assert commit_captured_prefix(ref, c, keep_tokens=4, verified_tokens=4)
    ref_logits, _h, c = _verify(ref_bank, ref_rt, ref, [4, 3, 2, 1], eager=True)
    assert commit_captured_prefix(ref, c, keep_tokens=keep, verified_tokens=4)
    ref_resumed, _h, _c = _verify(ref_bank, ref_rt, ref, [2, 2, 2, 2])
    assert ref_bank.stats["compiled_calls"] == 2 and ref[1].capacity == 48
    assert np.array_equal(np.array(logits), np.array(ref_logits))
    assert np.array_equal(np.array(resumed), np.array(ref_resumed))
    _assert_same(_rows(adapter, 0, 48), _rows(ref[1], 0, 48), "grown run vs preallocated run")


@pytest.mark.parametrize("mode", MODES)
def test_several_growth_events_keep_every_row_and_logit(mode):
    """Three growths (2 -> 3 -> 5 -> 8 blocks: 32 -> 48 -> 80 -> 128 rows),
    each on an eager fallback round, against a run preallocated at 128 rows
    that takes the same route in every round."""

    rt = TwoHeadPagedRuntime(mode)
    bank = CompiledVerifyBank(rt)
    cache = rt.make_cache()
    _prefill(rt, cache, 27)
    ref_rt = TwoHeadPagedRuntime(mode, blocks=8)
    ref_bank = CompiledVerifyBank(ref_rt)
    ref = ref_rt.make_cache()
    _prefill(ref_rt, ref, 27)

    rounds = 0
    capacities = []
    while cache[1].capacity < 100:
        ids = [(rounds + j) % 5 for j in range(4)]
        fallbacks = bank.stats["fallback_calls"]
        logits, _h, captures = _verify(bank, rt, cache, ids)
        eager = bank.stats["fallback_calls"] > fallbacks
        ref_logits, _h, ref_captures = _verify(ref_bank, ref_rt, ref, ids, eager=eager)
        assert commit_captured_prefix(cache, captures, keep_tokens=4, verified_tokens=4)
        assert commit_captured_prefix(ref, ref_captures, keep_tokens=4, verified_tokens=4)
        adapter = cache[1]
        rows = adapter.size()
        assert rows == ref[1].size() <= adapter.capacity
        _assert_same(_rows(adapter, 0, rows), _rows(ref[1], 0, rows), f"round {rounds} rows")
        np.testing.assert_allclose(
            np.array(logits), np.array(ref_logits), rtol=1e-6, atol=1e-6,
            err_msg=f"round {rounds} logits",
        )
        capacities.append(adapter.capacity)
        rounds += 1
    assert cache[1].grow_events == 3
    assert sorted(set(capacities)) == [32, 48, 80, 128]
    assert bank.stats["fallback_reasons"].get("capacity_overflow") == 3


class _PagedCountingModel:
    """A model for ``generate_mtpk``: after token t it wants t + 1 (mod V).

    Logits are MLX math on the ids, so the compiled verify bank can trace the
    forward, and the MTP head always agrees, so every draft is accepted. One
    attention layer over two KV heads writes paged KV through the mask it
    builds from its cache, as a model forward does; its readout feeds the
    hidden state with weight zero, so the buffers are read without changing
    the tokens. Every call that runs Python (prefill, eager commits, traces)
    is recorded as (rows, cache type, offset, capacity).
    """

    V, D = 16, 8

    def __init__(self, mode: str, *, blocks: int) -> None:
        from types import SimpleNamespace

        self.mode = mode
        self.blocks = blocks
        self.mtp = SimpleNamespace(_mtplx_lora_targets=[])
        mx.random.seed(3)
        width = HEADS * HEAD_DIM
        self.embed = mx.random.normal((self.V, self.D))
        self.w_k, self.w_v, self.w_q = (0.3 * mx.random.normal((self.D, width)) for _ in range(3))
        mx.eval(self.embed, self.w_k, self.w_v, self.w_q)
        self.calls: list[tuple[int, str, int | None, int | None]] = []

    def make_cache(self) -> list:
        return [
            VllmMetalPagedKVCache(
                block_size=BLOCK, num_blocks=self.blocks, kv_quant_config=_config(self.mode)
            )
        ]

    def make_mtp_cache(self) -> list:
        return []

    def mtp_update_cache(self, hidden_states, next_token_ids, **_kwargs):
        return hidden_states

    def _logits(self, ids):
        wanted = (ids + 1) % self.V
        return 10.0 * (mx.arange(self.V)[None, None, :] == wanted[..., None]).astype(mx.float32)

    def __call__(self, input_ids, *, cache=None, return_hidden=False, hidden_variant=None,
                 emit_logits=True, logits_keep=None, input_embeddings=None):
        from mlx_lm.models.base import create_attention_mask

        entry = cache[0]
        offset = entry.offset if isinstance(entry.offset, int) else _concrete_offset(entry.cache[2])
        self.calls.append(
            (int(input_ids.shape[1]), type(entry).__name__, offset, getattr(entry, "capacity", None))
        )
        B, S = int(input_ids.shape[0]), int(input_ids.shape[1])
        h = self.embed[input_ids]
        create_attention_mask(h, entry)

        def heads(w):
            return (h @ w).reshape(B, S, HEADS, HEAD_DIM).transpose(0, 2, 1, 3)

        keys, values = heads(self.w_k), heads(self.w_v)
        if self.mode == "plain":
            keys, values = keys.astype(mx.bfloat16), values.astype(mx.bfloat16)
        k_buf, _v_buf = entry.update_and_fetch(keys, values)
        read = mx.sum(heads(self.w_q) @ mx.swapaxes(k_buf, 2, 3).astype(mx.float32))
        hidden = h + 0.0 * read
        logits = self._logits(input_ids)
        if logits_keep is not None:
            logits = logits[:, -int(logits_keep):, :]
        if not emit_logits:
            logits = None
        return (logits, hidden) if return_hidden else logits

    def mtp_forward(self, hidden_states, next_token_ids, *, mtp_cache=None, concat_order=None,
                    return_hidden=False, mtp_hidden_variant=None, position_offset=None):
        logits = self._logits(next_token_ids)
        hidden = mx.zeros((1, int(next_token_ids.shape[-1]), self.D))
        return (logits, hidden) if return_hidden else logits


def _counting_prompt(prompt_len: int) -> list[int]:
    return [(i % (_PagedCountingModel.V - 1)) + 1 for i in range(prompt_len)]


def _generate_counting(
    mode: str, *, prompt_len: int, max_tokens: int, blocks: int, lazy: bool, monkeypatch,
    token_callback=None,
):
    from pathlib import Path

    from mtplx.generation import generate_mtpk
    from mtplx.mtp_patch import MTPContract
    from mtplx.runtime import MTPLXRuntime
    from mtplx.sampling import SamplerConfig

    class _Tokenizer:
        def decode(self, tokens, **_kwargs):
            return " ".join(str(int(token)) for token in tokens)

    monkeypatch.setenv("MTPLX_GRAPHBANK_PRESERVE_PAGED_KV", "1")
    monkeypatch.setenv("MTPLX_COMPILED_VERIFY", "1")
    monkeypatch.setenv("MTPLX_COMPILED_VERIFY_PREWARM", "0")
    monkeypatch.setenv("MTPLX_CONTEXT_COPY", "0")
    if lazy:
        monkeypatch.setenv("MTPLX_LAZY_BONUS_VERIFY", "1")
        monkeypatch.setenv("MTPLX_LAZY_BONUS_VERIFY_MIN_DEPTH", "1")
    else:
        monkeypatch.delenv("MTPLX_LAZY_BONUS_VERIFY", raising=False)
    model = _PagedCountingModel(mode, blocks=blocks)
    rt = MTPLXRuntime(
        model=model, tokenizer=_Tokenizer(), model_path=Path("tiny-paged"),
        mtp_enabled=True, contract=MTPContract(),
    )
    prompt = _counting_prompt(prompt_len)
    out = generate_mtpk(
        rt, prompt, max_tokens=max_tokens,
        sampler=SamplerConfig(temperature=0.0, top_p=1.0, top_k=0),
        speculative_depth=1, seed=0, stop_token_ids=set(),
        verify_strategy="capture_commit", capture_final_state=True,
        token_callback=token_callback,
    )
    wanted = [(prompt[-1] + 1 + i) % model.V for i in range(max_tokens)]
    return model, out, wanted


@pytest.mark.parametrize("path", ["final_pending_commit", "lazy_bonus_commit"])
@pytest.mark.parametrize("mode", MODES)
def test_generation_commits_one_token_past_a_full_promoted_cache(mode, path, monkeypatch):
    """generate_mtpk's final pending-token commit and its lazy-bonus commit
    call ``rt.forward_ar`` with one token on the promoted cache, outside any
    bank. Both run here through the production branches, sized so that the
    one-row window starts on a promoted adapter whose 32 rows are exactly
    full. The forward's own mask reserves the row: the cache grows to 48, the
    turn stays safe to commit, and the result equals a run whose buffers were
    48 rows from the start. On 150fb319 the final commit raised the capacity
    error (caught: the turn became unsafe to commit, its session-bank entry
    skipped) and the lazy-bonus commit failed the request."""

    lazy = path == "lazy_bonus_commit"
    # Prefill of 10 rows then 2 rows per round puts the final commit at 32;
    # with lazy bonus verify (1-row verify, 1-row bonus commit) an 11-row
    # prompt puts a bonus commit at 32.
    shape = dict(prompt_len=11, max_tokens=30) if lazy else dict(prompt_len=10, max_tokens=23)
    model, out, wanted = _generate_counting(mode, blocks=BLOCKS, lazy=lazy, monkeypatch=monkeypatch, **shape)

    adapter_name = _adapter_cls(mode).__name__
    crossing = [
        i for i, (rows, name, offset, capacity) in enumerate(model.calls)
        if rows == 1 and name == adapter_name and offset == capacity == 32
    ]
    assert len(crossing) == 1, model.calls
    if lazy:
        lazy_rounds = [
            e for e in out.stats.events
            if isinstance(e, dict) and (e.get("lazy_bonus_verify") or {}).get("enabled")
        ]
        assert lazy_rounds and crossing[0] < len(model.calls) - 1
        # The rounds after it ran on the grown buffers.
        assert any(capacity == 48 for _r, _n, _o, capacity in model.calls[crossing[0] + 1:])
    else:
        assert crossing[0] == len(model.calls) - 1  # the last forward: the final commit
    assert out.tokens == wanted
    assert out.final_state is not None and out.final_state.safe_to_commit is True
    assert not [e for e in out.stats.events if isinstance(e, dict) and "final_state_capture_error" in e]

    ref_model, ref_out, _wanted = _generate_counting(
        mode, blocks=3, lazy=lazy, monkeypatch=monkeypatch, **shape
    )
    assert ref_out.tokens == out.tokens
    (got,) = out.final_state.final_trunk_cache
    (want,) = ref_out.final_state.final_trunk_cache
    assert got.offset == want.offset
    for leaf, (g, w) in enumerate(zip(got.state, want.state)):
        assert np.array_equal(np.array(g.astype(mx.float32)), np.array(w.astype(mx.float32))), leaf


@pytest.mark.parametrize("mode", MODES)
def test_growth_refused_at_the_final_commit_keeps_the_response_and_banks_nothing(mode, monkeypatch):
    """A refusal after the last token was emitted comes from the final
    pending-token commit, which is session-bank bookkeeping. generate_mtpk
    catches it there: the completed response stands (this refusal is not a
    507), the turn is marked unsafe to commit, and the bank stores nothing
    for it, neither a snapshot nor a live-reference lease. The next request
    then runs normally."""

    from threading import Lock
    from types import SimpleNamespace

    from mtplx.server.openai import _store_generation_final_history_snapshot

    _install_available(monkeypatch, 1 * 1024**3)  # every growth is refused
    shape = dict(prompt_len=10, max_tokens=23)
    model, out, wanted = _generate_counting(
        mode, blocks=BLOCKS, lazy=False, monkeypatch=monkeypatch, **shape
    )
    assert out.tokens == wanted
    errors = [
        e["final_state_capture_error"] for e in out.stats.events
        if isinstance(e, dict) and "final_state_capture_error" in e
    ]
    assert len(errors) == 1 and "insufficient memory to grow the paged KV cache" in errors[0]
    assert "refusing before any row is written" in errors[0]
    assert out.final_state is not None and out.final_state.safe_to_commit is False
    (final_cache,) = out.final_state.final_trunk_cache
    assert final_cache.offset == 32  # the refused row was never written

    class _Bank:
        def __init__(self) -> None:
            self.puts: list[dict] = []

        def put(self, **kwargs):
            self.puts.append(kwargs)
            return SimpleNamespace(prefix_len=len(kwargs["token_ids"]), nbytes=1, token_hash="h")

    bank = _Bank()
    state = SimpleNamespace(sessions=SimpleNamespace(bank=bank, peek=lambda _sid: None), lock=Lock())
    outcome = _store_generation_final_history_snapshot(
        state,
        session_id="pi-session",
        prompt_ids=_counting_prompt(shape["prompt_len"]),
        generated={"tokens": list(out.tokens), "_final_state": out.final_state},
        messages=[],
        assistant_content=out.text,
        thinking_enabled=False,
        policy_fingerprint="policy",
        keep_live_ref=True,
    )
    assert outcome["stored"] is False and outcome["reason"] == "generation_final_state_unsafe"
    assert bank.puts == []

    _install_available(monkeypatch, 60 * 1024**3)  # the next request, memory back
    _model, again, wanted_again = _generate_counting(
        mode, blocks=BLOCKS, lazy=False, monkeypatch=monkeypatch, **shape
    )
    assert again.tokens == wanted_again and again.final_state.safe_to_commit is True


@pytest.mark.parametrize("mode", MODES)
def test_growth_refused_mid_generation_raises_the_memory_refusal_after_the_emitted_tokens(mode, monkeypatch):
    """A refusal before the last token (here the lazy-bonus commit) leaves
    generate_mtpk as PagedKVGrowthRefused, a MemoryError; the server answers
    it with its 507 frame (next test). Every token streamed before it is
    the exact count."""

    _install_available(monkeypatch, 1 * 1024**3)
    emitted: list[int] = []
    with pytest.raises(PagedKVGrowthRefused, match="insufficient memory to grow the paged KV cache"):
        _generate_counting(
            mode, prompt_len=11, max_tokens=30, blocks=BLOCKS, lazy=True,
            monkeypatch=monkeypatch, token_callback=emitted.extend,
        )
    prompt = _counting_prompt(11)
    wanted = [(prompt[-1] + 1 + i) % _PagedCountingModel.V for i in range(30)]
    assert 0 < len(emitted) < 30 and emitted == wanted[: len(emitted)]


def test_a_growth_refusal_mid_stream_is_a_507_frame_banks_nothing_and_the_next_request_runs(monkeypatch):
    """The server side of a refusal before the last token: the stream carries
    the tokens already produced, then one error frame, insufficient_memory
    with the refusal's own words, then [DONE]; the engine sheds its caches as
    for any allocation failure, the session bank stores nothing for the
    turn, and the next request is answered."""

    import json

    from fastapi.testclient import TestClient

    import mtplx.server.openai as openai
    from test_server_openai import _fake_final_state, _fake_streaming_session_state

    state = _fake_streaming_session_state()
    shed: list[bool] = []
    monkeypatch.setattr(openai, "_shed_after_allocation_failure", lambda _state: shed.append(True))
    refusal = (
        "insufficient memory to grow the paged KV cache: growing 16 caches from 16512 to "
        "24768 rows for a 4-row window at offset 16510 needs 1.21 GiB while this Mac has "
        "0.90 GiB available and keeps 3.20 GiB free for the desktop; refusing before any "
        "row is written"
    )
    calls = {"n": 0}

    def fake_run_generation(_state, prompt_ids, **kwargs):
        calls["n"] += 1
        token_callback = kwargs.get("token_callback")
        tokens = [ord("O"), ord("K")]
        if calls["n"] == 1:
            if token_callback is not None:
                token_callback(tokens[:1])
            raise PagedKVGrowthRefused(refusal)
        if token_callback is not None:
            token_callback(tokens)
        return {
            "text": "OK",
            "tokens": tokens,
            "stats": {"generation_mode": kwargs["generation_mode"], "mtp_depth": kwargs["depth"]},
            "prompt_tokens": len(prompt_ids),
            "completion_tokens": 2,
            "finish_reason": "stop",
            "_final_state": _fake_final_state(tokens),
        }

    monkeypatch.setattr(openai, "_run_generation", fake_run_generation)
    request = {
        "messages": [{"role": "user", "content": "Count"}],
        "enable_thinking": False,
        "max_tokens": 8,
    }
    with TestClient(openai.create_app(state)) as client:
        with client.stream(
            "POST", "/v1/chat/completions",
            headers={"x-mtplx-session-id": "pi-session"},
            json={**request, "stream": True},
        ) as response:
            status = response.status_code
            body = response.read().decode()
        second = client.post(
            "/v1/chat/completions",
            headers={"x-mtplx-session-id": "pi-session"},
            json=request,
        )

    assert status == 200  # the stream had started
    payloads = [
        json.loads(line.removeprefix("data: "))
        for line in body.splitlines()
        if line.startswith("data: {")
    ]
    content = "".join(
        (choice.get("delta") or {}).get("content") or ""
        for payload in payloads
        for choice in payload.get("choices", [])
    )
    assert content == "O"
    (error,) = [payload["error"] for payload in payloads if "error" in payload]
    assert error["code"] == "insufficient_memory"
    assert refusal in error["message"]
    assert body.rstrip().endswith("data: [DONE]")
    assert shed == [True]
    assert state.sessions.bank.puts == []  # nothing banked for the refused turn
    assert second.status_code == 200
    assert second.json()["choices"][0]["message"]["content"] == "OK"


@pytest.mark.parametrize("mode", MODES)
def test_direct_forward_with_growth_off_refuses_before_any_row(mode, monkeypatch):
    rt = TwoHeadPagedRuntime(mode)
    bank = CompiledVerifyBank(rt)
    cache = rt.make_cache()
    _prefill(rt, cache, 28)
    _l, _h, captures = _verify(bank, rt, cache, [1, 2, 3, 4])
    assert commit_captured_prefix(cache, captures, keep_tokens=4, verified_tokens=4)
    adapter = cache[1]
    committed = _rows(adapter, 0, 32)
    monkeypatch.delenv("MTPLX_DYNAMIC_PAGED_KV", raising=False)
    with pytest.raises(ValueError, match="refusing before any row is written"):
        rt.forward_ar(mx.array([[3]]), cache=cache, return_hidden=True)
    assert adapter.size() == 32 and adapter.capacity == 32
    _assert_same(_rows(adapter, 0, 32), committed, "refused forward")


def _spec_decode_bank(rt, cache, promoted_by: str) -> SpecDecodeGraphBank:
    if promoted_by == "this_bank":
        return SpecDecodeGraphBank(rt, max_verify_len=4)
    # A caller that hands the bank adapters it promoted itself.
    graphbank_module.promote_kv_cache_offsets(cache, reserve_tokens=4, preserve_paged=True)
    return SpecDecodeGraphBank(rt, max_verify_len=4, promote_tensor_offsets=False)


@pytest.mark.parametrize("promoted_by", ["this_bank", "caller"])
@pytest.mark.parametrize("mode", MODES)
def test_spec_decode_graph_bank_reserves_before_its_compiled_replay(mode, promoted_by, monkeypatch):
    """SpecDecodeGraphBank (graphbank selection with preserved paged KV)
    replays a traced graph that writes the adapters at their traced offset;
    no Python runs on that path. Old code: the second call replayed the
    31-row graph into 32 rows and head 0's rows landed on head 1's. The
    reservation used to sit inside the bank's own promotion, so adapters a
    caller promoted (promote_tensor_offsets=False) were not reserved at all."""

    monkeypatch.setenv("MTPLX_GRAPHBANK_PRESERVE_PAGED_KV", "1")
    rt = TwoHeadPagedRuntime(mode)
    cache = rt.make_cache()
    _prefill(rt, cache, 27)
    bank = _spec_decode_bank(rt, cache, promoted_by)

    bank.forward_ar(mx.array([[1, 2, 3, 4]]), cache=cache)
    adapter = cache[1]
    assert type(adapter) is _adapter_cls(mode)
    assert bank.stats.compiled_calls == 1
    assert adapter.size() == 31 and adapter.capacity == 32
    committed = _rows(adapter, 0, 31)

    logits, _hidden = bank.forward_ar(mx.array([[4, 3, 2, 1]]), cache=cache)
    mx.eval(logits)
    assert bank.stats.compiled_calls == 2 and bank.stats.fallback_calls == 0
    _assert_same(_rows(adapter, 0, 31), committed, "committed rows after the replay")
    assert adapter.size() == 35 <= adapter.capacity == 48
    assert np.isfinite(np.array(logits)).all()

    # The same two calls on buffers that were 48 rows from the start.
    ref_rt = TwoHeadPagedRuntime(mode, blocks=3)
    ref = ref_rt.make_cache()
    _prefill(ref_rt, ref, 27)
    ref_bank = _spec_decode_bank(ref_rt, ref, promoted_by)
    ref_bank.forward_ar(mx.array([[1, 2, 3, 4]]), cache=ref)
    ref_logits, _hidden = ref_bank.forward_ar(mx.array([[4, 3, 2, 1]]), cache=ref)
    assert ref_bank.stats.compiled_calls == 2
    assert np.array_equal(np.array(logits), np.array(ref_logits))
    _assert_same(_rows(adapter, 0, 48), _rows(ref[1], 0, 48), "grown run vs preallocated run")


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
@pytest.mark.parametrize("bits", [8, 4])
def test_packed_quant_kernel_refuses_an_eager_array_offset_past_its_buffers(bits):
    # The integer offset branch always bailed past the buffers; the array
    # branch (the promoted adapter's) passed any value to a walk that reads
    # offset rows of every leaf. Old code: "DID NOT RAISE".
    from mtplx.kernels.sdpa_gqa_packed_quant import sdpa_gqa_packed_tail_quant

    head_dim, capacity = 256, 40
    mx.random.seed(bits)
    queries = mx.random.normal((1, 4, 2, head_dim)).astype(mx.bfloat16)
    k_q, k_s = quantize_symmetric(mx.random.normal((1, 2, capacity, head_dim)), bits=bits)
    v_q, v_s = quantize_symmetric(mx.random.normal((1, 2, capacity, head_dim)), bits=bits)
    args = dict(
        queries=queries, k_q=k_q, k_scale=k_s, v_q=v_q, v_scale=v_s,
        scale=head_dim**-0.5, bits=bits,
    )

    with pytest.raises(ValueError, match="past the KV buffers"):
        sdpa_gqa_packed_tail_quant(offset=mx.array(capacity + 3, dtype=mx.int32), **args)

    # An in-range array offset still runs and matches the integer branch.
    by_array = sdpa_gqa_packed_tail_quant(offset=mx.array(33, dtype=mx.int32), **args)
    by_int = sdpa_gqa_packed_tail_quant(offset=33, **args)
    assert by_array is not None and by_int is not None
    assert np.array_equal(
        np.array(by_array.astype(mx.float32)), np.array(by_int.astype(mx.float32))
    )


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
@pytest.mark.parametrize("bits", [8, 4])
def test_packed_quant_kernel_raises_on_an_eager_nan_offset(bits):
    # int(nan) raises ValueError, which the kernel's offset probe used to read
    # as "traced": the range check was skipped and the kernel dispatched with
    # a NaN cast to int32. Old code: "DID NOT RAISE".
    from mtplx.kernels.sdpa_gqa_packed_quant import sdpa_gqa_packed_tail_quant

    head_dim, capacity = 256, 40
    mx.random.seed(bits)
    queries = mx.random.normal((1, 4, 2, head_dim)).astype(mx.bfloat16)
    k_q, k_s = quantize_symmetric(mx.random.normal((1, 2, capacity, head_dim)), bits=bits)
    v_q, v_s = quantize_symmetric(mx.random.normal((1, 2, capacity, head_dim)), bits=bits)
    with pytest.raises(ValueError, match="NaN"):
        sdpa_gqa_packed_tail_quant(
            queries=queries, k_q=k_q, k_scale=k_s, v_q=v_q, v_scale=v_s,
            offset=mx.array(float("nan")), scale=head_dim**-0.5, bits=bits,
        )


def test_packed_quant_offset_probe_exempts_only_the_trace_refusal():
    from mtplx.kernels.sdpa_gqa_packed_quant import _concrete_offset as kernel_offset

    class _FailingOffset:
        def item(self):
            raise ValueError("an unrelated evaluation failure")

    # Old code: None, so the caller skipped its range check.
    with pytest.raises(ValueError, match="unrelated evaluation failure"):
        kernel_offset(_FailingOffset())

    seen = {}

    def body(offset):
        seen["traced"] = kernel_offset(offset)
        return offset + 1

    mx.eval(mx.compile(body)(mx.array(3, dtype=mx.int32)))
    assert seen == {"traced": None}
    assert kernel_offset(mx.array(7, dtype=mx.int32)) == 7


def _assert_one_group(cache) -> list:
    adapters = [entry for entry in cache if isinstance(entry, TensorOffsetVllmMetalPagedKVCache)]
    for adapter in adapters:
        assert [id(m) for m in adapter._window_members()] == [id(a) for a in adapters]
    return adapters


def _round(bank, rt, cache, ids):
    logits, _h, captures = _verify(bank, rt, cache, ids)
    assert commit_captured_prefix(cache, captures, keep_tokens=len(ids), verified_tokens=len(ids))
    return logits


@pytest.mark.parametrize("mode", MODES)
def test_one_mask_reserves_every_layer_through_demotion_restore_and_repromotion(mode):
    """Three attention layers cross the capacity through the one mask the
    forward builds from the first layer's cache; then the bank demotes the
    list, a snapshot of it is restored into a fresh list (a session-bank
    restore), the fresh list is promoted again and crosses the next capacity.
    At every stage the reservation group is exactly the list's adapters, every
    layer grows together, and rows and logits match a run preallocated at 128
    rows through the same stages."""

    from mtplx.cache_state import restore_cache, snapshot_cache

    rt = ThreeLayerPagedRuntime(mode)
    ref_rt = ThreeLayerPagedRuntime(mode, blocks=8)
    runs = []
    for runtime in (rt, ref_rt):
        bank = CompiledVerifyBank(runtime)
        cache = runtime.make_cache()
        _prefill(runtime, cache, 27)
        runs.append([runtime, bank, cache])

    def both(ids):
        outs = [_round(bank, runtime, cache, ids) for runtime, bank, cache in runs]
        np.testing.assert_allclose(np.array(outs[0]), np.array(outs[1]), rtol=1e-6, atol=1e-6)
        for layer in range(1, 4):
            got, want = runs[0][2][layer], runs[1][2][layer]
            assert got.size() == want.size()
            _assert_same(_rows(got, 0, got.size()), _rows(want, 0, want.size()), f"layer {layer}")

    both([1, 2, 3, 4])  # compiled, 27 -> 31: promotes all three layers
    adapters = _assert_one_group(runs[0][2])
    assert len(adapters) == 3 and {a.capacity for a in adapters} == {32}
    both([4, 3, 2, 1])  # 31 + 4 > 32: one mask reserves every layer
    assert [a.capacity for a in _assert_one_group(runs[0][2])] == [48, 48, 48]

    for run in runs:  # demote, snapshot, restore into a fresh list
        runtime, bank, cache = run
        assert bank.demote(cache) == 3
        assert all(type(entry) is VllmMetalPagedKVCache for entry in cache[1:])
        fresh = runtime.make_cache()
        restore_cache(fresh, snapshot_cache(cache))
        run[1] = CompiledVerifyBank(runtime)
        run[2] = fresh

    both([2, 4, 1, 3])  # 35 -> 39, re-promoted
    first = _assert_one_group(runs[0][2])
    assert [a.capacity for a in first] == [48, 48, 48]
    both([3, 1, 4, 2])  # 43
    both([1, 1, 2, 2])  # 47
    both([2, 2, 1, 1])  # 47 + 4 > 48: every layer grows again
    grown = _assert_one_group(runs[0][2])
    assert [a.capacity for a in grown] == [80, 80, 80]
    assert all(a.size() == 51 for a in grown)


def test_a_list_rebuilt_from_promoted_adapters_is_relinked_as_one_group(monkeypatch):
    """Linking used to happen only when a pass promoted something new, so a
    list assembled from adapters promoted elsewhere kept their old groups
    and the first adapter's mask reserved only part of the list."""

    monkeypatch.setenv("MTPLX_GRAPHBANK_PRESERVE_PAGED_KV", "1")
    first = [_promoted("q8"), _promoted("q8")]
    second = [_promoted("q8")]
    link_paged_window_group(first)
    link_paged_window_group(second)
    rebuilt = [first[0], second[0], first[1]]
    assert graphbank_module.promote_kv_cache_offsets(rebuilt, reserve_tokens=4) == (0, {})
    _assert_one_group(rebuilt)  # old code: first[0]'s group was still [first[0], first[1]]

    reserve_paged_window(rebuilt[0]._window_members(), 4)  # 31 + 4 > 32 for all three
    assert [a.capacity for a in rebuilt] == [48, 48, 48]
