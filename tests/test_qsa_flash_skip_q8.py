"""Decode flash-skip over the int8 KV backing against the float16 kernel on the dequantized rows."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

import mtplx.models.qwen4_exp as Q
from mtplx.kernels.qsa_flash_skip import qsa_flash_skip, qsa_flash_skip_q8

pytestmark = pytest.mark.skipif(not (hasattr(mx, "metal") and mx.metal.is_available()), reason="needs Metal")


@pytest.mark.parametrize("tokens,hq,hkv", [(64, 24, 2), (1000, 24, 2), (4099, 12, 1)])
def test_matches_the_float16_kernel_on_dequantized_rows(tokens, hq, hkv):
    mx.random.seed(tokens)
    k = (mx.random.normal((1, hkv, tokens, 256)) * 1.5).astype(mx.float16)
    v = (mx.random.normal((1, hkv, tokens, 256)) * 0.5).astype(mx.float16)
    cache = Q.QSAKVCacheQ8()
    cache.update_and_fetch(k, v)
    q = (mx.random.normal((hq, 256)) * 0.5).astype(mx.float16)
    nblocks = tokens // 4
    sel = np.sort(np.random.default_rng(tokens).choice(nblocks, size=min(nblocks, 40), replace=False)).astype(np.int32)
    blocks = mx.array(sel)
    tail = nblocks * 4
    want = qsa_flash_skip(q, cache.keys.reshape(1, hkv, tokens, 256), cache.values.reshape(1, hkv, tokens, 256), blocks, tokens, tail, 0.0625)
    got = qsa_flash_skip_q8(q, cache.kq, cache.ks, cache.vq, cache.vs, blocks, tokens, tail, 0.0625)
    mx.eval(want, got)
    w, g = np.array(want).astype(np.float32), np.array(got).astype(np.float32)
    assert np.abs(w - g).max() < 3e-3 * max(1.0, np.abs(w).max())
    # and close to the unquantized cache
    ref = qsa_flash_skip(q, k, v, blocks, tokens, tail, 0.0625)
    assert np.abs(np.array(ref).astype(np.float32) - g).max() < 0.02 * max(1.0, np.abs(w).max())
