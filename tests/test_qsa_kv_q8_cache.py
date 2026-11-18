"""int8 QSA KV storage: round trip, growth, trim, state contract, size."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

import mtplx.models.qwen4_exp as Q


def _rows(n, seed=0, heads=2, dim=256):
    mx.random.seed(seed)
    k = (mx.random.normal((1, heads, n, dim)) * 2.0).astype(mx.float16)
    v = (mx.random.normal((1, heads, n, dim)) * 0.5).astype(mx.float16)
    mx.eval(k, v)
    return k, v


def _rel(a, b):
    a, b = np.array(a).astype(np.float32), np.array(b).astype(np.float32)
    return float(np.abs(a - b).max() / np.abs(b).max())


def test_the_fetched_views_are_the_written_rows_to_int8_precision():
    cache = Q.QSAKVCacheQ8()
    k, v = _rows(37)
    fk, fv = cache.update_and_fetch(k, v)
    assert fk.shape == k.shape and fk.dtype == mx.float16
    assert _rel(fk, k) < 0.01 and _rel(fv, v) < 0.01
    assert cache.offset == 37


def test_growth_keeps_history_and_matches_the_stock_cache_shape_contract():
    cache = Q.QSAKVCacheQ8()
    stock = Q.QSAKVCache()
    chunks = [_rows(n, seed=n) for n in (5, 300, 1, 700)]
    for k, v in chunks:
        fk, fv = cache.update_and_fetch(k, v)
        sk, sv = stock.update_and_fetch(k, v)
        assert fk.shape == sk.shape
    assert cache.offset == stock.offset == 1006
    assert cache.capacity >= cache.offset
    ref_k = mx.concatenate([c[0] for c in chunks], axis=2)
    assert _rel(fk, ref_k) < 0.01


def test_rows_target_sizes_the_backing_once():
    cache = Q.QSAKVCacheQ8()
    k, v = _rows(10)
    with Q.qsa_rows_target(4096):
        cache.update_and_fetch(k, v)
    assert cache.capacity == 4096


def test_trim_and_overwrite():
    cache = Q.QSAKVCacheQ8()
    k, v = _rows(20)
    cache.update_and_fetch(k, v)
    assert cache.trim(5) == 5 and cache.offset == 15
    k2, v2 = _rows(5, seed=9)
    fk, _ = cache.update_and_fetch(k2, v2)
    assert fk.shape[2] == 20
    assert _rel(fk[:, :, 15:], k2) < 0.01 and _rel(fk[:, :, :15], k[:, :, :15]) < 0.01


def test_state_round_trip_is_float16_four_leaf_compatible():
    cache = Q.QSAKVCacheQ8()
    k, v = _rows(64)
    cache.update_and_fetch(k, v)
    keys, values = cache.state
    assert keys.shape == (1, 2, 64, 256) and keys.dtype == mx.float16
    other = Q.QSAKVCacheQ8()
    other.state = (keys, values)
    assert other.offset == 64
    assert _rel(other.keys, keys) < 0.01


def test_half_the_bytes():
    q8, fp = Q.QSAKVCacheQ8(), Q.QSAKVCache()
    k, v = _rows(512)
    q8.update_and_fetch(k, v)
    fp.update_and_fetch(k, v)
    assert q8.nbytes < 0.55 * fp.nbytes


def test_the_switch(monkeypatch):
    monkeypatch.delenv("MTPLX_QSA_KV_Q8", raising=False)
    assert type(Q.QSACache().kv) is Q.QSAKVCache
    monkeypatch.setenv("MTPLX_QSA_KV_Q8", "1")
    assert type(Q.QSACache().kv) is Q.QSAKVCacheQ8
