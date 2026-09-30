"""Paged tensor-offset adapters grow on the host when a generation outruns them.

The server reserves at most 16,384 new tokens of pages up front; a thinking-on
decode past that wrote beyond the adapter's fixed capacity and produced
non-finite logits (2026-10-01). ``grow_to`` extends every layout zero-filled
and block-aligned, and writes after the growth must land exactly where a cache
that had the room from the start puts them.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from mtplx.cache_state import (
    TensorOffsetQuantizedPagedKVCache,
    TensorOffsetVllmMetalPagedKVCache,
    VllmMetalPagedKVCache,
)
from mtplx.kv_quant import PagedKVQuantConfig


def _kv(tokens: int, seed: int):
    keys = mx.random.normal((1, 4, tokens, 64), key=mx.random.key(seed)).astype(mx.float16)
    values = mx.random.normal((1, 4, tokens, 64), key=mx.random.key(seed + 1)).astype(mx.float16)
    return keys, values


def _paged(num_blocks: int, bits: int | None) -> VllmMetalPagedKVCache:
    config = PagedKVQuantConfig(mode=f"q{bits}") if bits else None
    return VllmMetalPagedKVCache(block_size=16, num_blocks=num_blocks, kv_quant_config=config)


def _adapter(paged: VllmMetalPagedKVCache, bits: int | None):
    if bits:
        return TensorOffsetQuantizedPagedKVCache.from_paged_cache(paged)
    return TensorOffsetVllmMetalPagedKVCache.from_paged_cache(paged)


@pytest.mark.parametrize(
    "bits,pages",
    [(None, "0"), (8, "0"), (8, "1"), (4, "0")],
)
def test_grow_then_write_matches_a_cache_that_had_the_room(bits, pages, monkeypatch):
    monkeypatch.setenv("MTPLX_KV_QUANT_PAGES_ADAPTER", pages)
    monkeypatch.setenv("MTPLX_GQA_MMA", "1")
    first_k, first_v = _kv(100, 1)
    more_k, more_v = _kv(60, 3)

    small = _paged(8, bits)  # 128 tokens of pages
    small.update_and_fetch(first_k, first_v)
    adapter = _adapter(small, bits)
    assert adapter.capacity == 128

    assert adapter.grow_to(160 + 512)
    assert adapter.capacity >= 672 and adapter.capacity % 16 == 0
    adapter.update_without_fetch(more_k, more_v)
    mx.eval(adapter.cache)

    roomy = _paged(64, bits)
    roomy.update_and_fetch(first_k, first_v)
    roomy.update_and_fetch(more_k, more_v)
    reference = _adapter(roomy, bits)

    assert int(adapter.offset.item()) == 160
    got_k, got_v = (x[..., :160, :] for x in adapter.state)
    want_k, want_v = (x[..., :160, :] for x in reference.state)
    assert mx.array_equal(got_k, want_k).item()
    assert mx.array_equal(got_v, want_v).item()


def test_grow_to_is_a_no_op_within_capacity():
    paged = _paged(8, None)
    paged.update_and_fetch(*_kv(10, 5))
    adapter = _adapter(paged, None)
    before = [leaf for leaf in adapter.cache[:2]]
    assert adapter.grow_to(64)
    assert adapter.capacity == 128
    assert all(a is b for a, b in zip(before, adapter.cache[:2]))
