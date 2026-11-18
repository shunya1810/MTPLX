"""Verify-row attention over the int8 KV backing against the float16 kernel on the dequantized rows."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

import mtplx.models.qwen4_exp as Q
from mtplx.kernels.qsa_rows_mma import qsa_rows_attention, qsa_rows_attention_q8

pytestmark = pytest.mark.skipif(not (hasattr(mx, "metal") and mx.metal.is_available()), reason="needs Metal")


@pytest.mark.parametrize("rows,tokens,hkv,gqa", [(1, 600, 2, 12), (4, 3000, 2, 12), (8, 9000, 1, 12), (3, 700, 2, 8)])
def test_matches_the_float16_kernel_on_dequantized_rows(rows, tokens, hkv, gqa):
    mx.random.seed(rows * 7 + tokens)
    k = (mx.random.normal((1, hkv, tokens, 256)) * 1.5).astype(mx.float16)
    v = (mx.random.normal((1, hkv, tokens, 256)) * 0.5).astype(mx.float16)
    cache = Q.QSAKVCacheQ8()
    cache.update_and_fetch(k, v)
    q = (mx.random.normal((1, hkv * gqa, rows, 256)) * 0.5).astype(mx.float16)
    rng = np.random.default_rng(tokens)
    klist = min(tokens, 330)
    idx = np.stack([np.sort(rng.choice(tokens, size=klist, replace=False)) for _ in range(rows)]).astype(np.int32)
    ok = np.ones_like(idx, dtype=bool)
    ok[:, -3:] = False
    tok_idx, tok_ok = mx.array(idx), mx.array(ok)
    deq_k = cache.keys.reshape(1, hkv, tokens, 256)
    deq_v = cache.values.reshape(1, hkv, tokens, 256)
    want = qsa_rows_attention(q, deq_k, deq_v, tok_idx, tok_ok, 0.0625)
    got = qsa_rows_attention_q8(q, cache.kq, cache.ks, cache.vq, cache.vs, tok_idx, tok_ok, 0.0625)
    ref = qsa_rows_attention(q, k, v, tok_idx, tok_ok, 0.0625)
    mx.eval(want, got, ref)
    w, g, r = (np.array(a).astype(np.float32) for a in (want, got, ref))
    assert np.abs(w - g).max() < 3e-3 * max(1.0, np.abs(w).max())
    assert np.abs(r - g).max() < 0.03 * max(1.0, np.abs(r).max())
