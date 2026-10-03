"""A meta_state rewind must shorten the kv_quant derived views like trim() does.

``VllmMetalPagedKVCache`` keeps two views of its quantized pages that extend
tail-only from a ``tokens`` count: the q8 dequant mirror and the q4 head-major
quant bank. ``trim()`` shortens that count; the ``meta_state`` setter moved the
offset back without it. The read-time guard (``tokens > offset``) only fires
while the offset is still below the old count, but a forward writes before it
reads: rewind 20 -> 12, write 6 new rows (offset 18), and the next read found
``tokens`` (20) >= offset (18), extended nothing, and served the OLD rows 12..17.
Found by the static review of issue #526; not the #526 route itself.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from mtplx.cache_state import VllmMetalPagedKVCache
from mtplx.kv_quant import PagedKVQuantConfig

HEADS, HEAD_DIM = 2, 64


def _rows(rows: int, seed: int):
    mx.random.seed(seed)
    keys = mx.random.normal((1, HEADS, rows, HEAD_DIM)).astype(mx.bfloat16)
    values = mx.random.normal((1, HEADS, rows, HEAD_DIM)).astype(mx.bfloat16)
    mx.eval(keys, values)
    return keys, values


def _cache(mode: str) -> VllmMetalPagedKVCache:
    cache = VllmMetalPagedKVCache(
        block_size=16, num_blocks=4, kv_quant_config=PagedKVQuantConfig(mode)
    )
    cache.update_without_fetch(*_rows(20, seed=1))
    return cache


def _rewind_then_rewrite(cache: VllmMetalPagedKVCache) -> None:
    assert cache.offset == 20
    cache.meta_state = (str(cache.block_size), str(cache.num_blocks), "12")
    assert cache.offset == 12
    cache.update_without_fetch(*_rows(6, seed=2))  # different rows at 12..17
    assert cache.offset == 18


def test_q8_dequant_mirror_is_truncated_by_a_meta_state_rewind():
    cache = _cache("q8")
    cache._dequant_active_arrays()  # the mirror now covers rows [0, 20)
    assert cache._dequant_memo["tokens"] == 20

    _rewind_then_rewrite(cache)
    keys, values = cache._dequant_active_arrays()
    want_k, want_v = cache._paged_range(0, 18)  # straight from the pages
    # Old code: rows 12..17 of the mirror were still the pre-rewind rows.
    assert np.array_equal(np.array(keys.astype(mx.float32)), np.array(want_k.astype(mx.float32)))
    assert np.array_equal(np.array(values.astype(mx.float32)), np.array(want_v.astype(mx.float32)))


def test_q4_quant_bank_is_truncated_by_a_meta_state_rewind():
    cache = _cache("q4")
    cache._quant_bank_arrays()  # the bank now covers rows [0, 20)
    assert cache._quant_bank["tokens"] == 20

    _rewind_then_rewrite(cache)
    bank = cache._quant_bank_arrays()
    heads = int(cache.key_cache.shape[2])
    for got, pages in zip(
        bank,
        (cache.key_cache, cache.value_cache, cache.key_scale_cache, cache.value_scale_cache),
    ):
        want = pages.reshape(-1, heads, int(pages.shape[3]))[:18].transpose(1, 0, 2)[None]
        # Old code: bank rows 12..17 still held the pre-rewind payloads/scales.
        assert np.array_equal(np.array(got[:, :, :18, :]), np.array(want))


@pytest.mark.parametrize("mode", ["q8", "q4"])
def test_meta_state_rewind_matches_trim_on_the_derived_views(mode):
    via_trim, via_meta = _cache(mode), _cache(mode)
    for cache in (via_trim, via_meta):
        cache._dequant_active_arrays() if mode == "q8" else cache._quant_bank_arrays()
    via_trim.trim(8)
    via_meta.meta_state = (str(via_meta.block_size), str(via_meta.num_blocks), "12")
    for key in ("_dequant_memo", "_quant_bank"):
        a, b = getattr(via_trim, key), getattr(via_meta, key)
        assert (a is None) == (b is None)
        if a is not None:
            assert a["tokens"] == b["tokens"] == 12
