"""The fused simdgroup-matrix prefill score kernel against the float32 MLX expression."""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

import mtplx.kernels.qsa_indexer_prefill as backend

pytestmark = pytest.mark.skipif(
    not (hasattr(mx, "metal") and mx.metal.is_available()), reason="needs a Metal device"
)


def _inputs(rows: int, blocks: int, seed: int = 0):
    mx.random.seed(seed)
    q = (mx.random.normal((1, rows, 4, 128)) * 0.3).astype(mx.float16)
    pooled = (mx.random.normal((1, blocks, 128)) * 0.3).astype(mx.float16)
    mx.eval(q, pooled)
    return q, pooled


def _oracle(q, pooled):
    pooled_t = mx.swapaxes(pooled.astype(mx.float32), 1, 2)[:, None]
    return backend.qsa_indexer_prefill_scores(q, pooled_t, head_dim=128)


@pytest.mark.parametrize("rows,blocks", [(8, 32), (16, 4096), (13, 1000), (200, 8200), (1, 8)])
def test_scores_match_the_float32_expression(rows, blocks):
    q, pooled = _inputs(rows, blocks)
    got = backend.qsa_indexer_prefill_scores_simd(q, pooled)
    want = _oracle(q, pooled)
    mx.eval(got, want)
    assert got.shape == want.shape == (rows, blocks)
    assert float(mx.max(mx.abs(got - want)).item()) < 1e-5


def test_selected_sets_are_the_same():
    q, pooled = _inputs(64, 4096, seed=3)
    got = np.array(backend.qsa_indexer_prefill_scores_simd(q, pooled))
    want = np.array(_oracle(q, pooled))
    for row in range(64):
        a = set(np.argpartition(got[row], -512)[-512:].tolist())
        b = set(np.argpartition(want[row], -512)[-512:].tolist())
        assert a == b


def test_unsupported_signatures_are_refused():
    q, pooled = _inputs(8, 64)
    assert backend.qsa_indexer_prefill_scores_simd_supported(q, pooled)
    assert not backend.qsa_indexer_prefill_scores_simd_supported(q.astype(mx.float32), pooled)
    assert not backend.qsa_indexer_prefill_scores_simd_supported(q, pooled[:, :60])
    assert not backend.qsa_indexer_prefill_scores_simd_supported(q[:, :, :2], pooled)
    with pytest.raises(ValueError):
        backend.qsa_indexer_prefill_scores_simd(q, pooled[:, :60])


def test_the_rollback_switch(monkeypatch):
    q, pooled = _inputs(8, 64)
    monkeypatch.setenv("MTPLX_QSA_PREFILL_SIMD_SCORES", "0")
    assert not backend.qsa_indexer_prefill_scores_simd_supported(q, pooled)


def test_the_two_stage_prefill_is_unchanged_by_the_fused_scores(monkeypatch):
    q, pooled = _inputs(300, 2048, seed=5)
    kwargs = dict(pos_start=7800, total_tokens=8192, block_topk=512, compress_ratio=4)
    on = backend.qsa_indexer_prefill_metal(q, pooled, **kwargs)
    monkeypatch.setenv("MTPLX_QSA_PREFILL_SIMD_SCORES", "0")
    off = backend.qsa_indexer_prefill_metal(q, pooled, **kwargs)
    mx.eval(on, off)
    for a, b in zip(on if isinstance(on, tuple) else (on,), off if isinstance(off, tuple) else (off,)):
        a, b = np.array(a), np.array(b)
        assert a.shape == b.shape
        if a.ndim >= 2:
            same = np.mean([set(a[i].ravel().tolist()) == set(b[i].ravel().tolist()) for i in range(a.shape[0])])
            assert same > 0.99
