"""The fused few-row score kernel against the eager chain it replaces."""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

from mtplx.kernels import qsa_decode_scores as D
from mtplx.kernels import qsa_topk_rows as T

pytestmark = pytest.mark.skipif(
    not (hasattr(mx, "metal") and mx.metal.is_available()), reason="needs a Metal device"
)

RATIO = 4


def _stock(q, pooled, pos, divisor):
    blocks = int(pooled.shape[1])
    rows = int(q.shape[1])
    pt = mx.swapaxes(pooled.astype(mx.float32), 1, 2)[:, None]
    scores = (mx.maximum(mx.matmul(q.astype(mx.float32), pt), 0.0).sum(axis=2) / divisor)[0]
    nb_q = (pos + mx.arange(rows, dtype=mx.int32) + 1) // RATIO
    blk = mx.arange(blocks, dtype=mx.int32)
    kept = mx.where(blk[None, :] < nb_q[:, None], scores, mx.array(-mx.inf, dtype=mx.float32))
    return kept - blk.astype(mx.float32)[None, :] * 1e-12


@pytest.mark.parametrize("rows", [1, 2, 3, 4, 5, 8])
@pytest.mark.parametrize("blocks,position", [(64, 200), (2112, 8431), (8192, 8192 * 4 - 12), (32768, 100000), (2049, 8200), (8195, 32770), (13, 52), (7, 24), (1, 3)])
def test_scores_match_the_eager_chain(rows, blocks, position):
    mx.random.seed(rows * 13 + blocks)
    q = (mx.random.normal((1, rows, 4, 128)) * 0.3).astype(mx.float16)
    pooled = (mx.random.normal((1, blocks, 128)) * 0.3).astype(mx.float16)
    divisor = mx.array([math.sqrt(128)], dtype=mx.float32)
    pos = mx.array(position, dtype=mx.int32)
    want = _stock(q, pooled, pos, divisor)
    got = D.masked_scores(q, pooled, position, divisor, ratio=RATIO)
    mx.eval(want, got)
    a, b = np.array(want), np.array(got)
    assert a.shape == b.shape == (rows, blocks)
    assert np.array_equal(np.isfinite(a), np.isfinite(b))
    fin = np.isfinite(a)
    assert float(np.max(np.abs(a[fin] - b[fin]))) < 1e-5


def test_the_selected_sets_agree():
    rows, blocks = 4, 8192
    mx.random.seed(7)
    q = (mx.random.normal((1, rows, 4, 128)) * 0.3).astype(mx.float16)
    pooled = (mx.random.normal((1, blocks, 128)) * 0.3).astype(mx.float16)
    divisor = mx.array([math.sqrt(128)], dtype=mx.float32)
    want = _stock(q, pooled, mx.array(blocks * 4 - 8, dtype=mx.int32), divisor)
    got = D.masked_scores(q, pooled, blocks * 4 - 8, divisor, ratio=RATIO)
    a, b = np.array(T.topk_rows(want, 512)), np.array(T.topk_rows(got, 512))
    assert all(set(a[i].tolist()) == set(b[i].tolist()) for i in range(rows))


def test_signatures_outside_the_kernel_are_refused(monkeypatch):
    q = mx.zeros((1, 4, 4, 128), dtype=mx.float16)
    pooled = mx.zeros((1, 64, 128), dtype=mx.float16)
    assert D.supported(q, pooled, 4, 128)
    assert not D.supported(q.astype(mx.float32), pooled, 4, 128)
    assert D.supported(q, pooled[:, :60], 4, 128)
    assert not D.supported(q, pooled[:, :0], 4, 128)
    assert not D.supported(mx.zeros((1, 9, 4, 128), dtype=mx.float16), pooled, 4, 128)
    assert not D.supported(q, pooled, 2, 128)
    monkeypatch.setenv(D.ENV, "0")
    assert not D.supported(q, pooled, 4, 128)
