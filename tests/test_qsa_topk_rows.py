"""The radix top-k of a few float32 rows against mx.argpartition."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from mtplx.kernels import qsa_topk_rows as T

pytestmark = pytest.mark.skipif(
    not (hasattr(mx, "metal") and mx.metal.is_available()), reason="needs a Metal device"
)


def _sets(top):
    return [set(row.tolist()) for row in np.array(top)]


@pytest.mark.parametrize("rows", [1, 2, 4, 8])
@pytest.mark.parametrize("blocks", [600, 2112, 8192, 32768, 65536])
def test_the_same_set_as_argpartition(rows, blocks):
    mx.random.seed(rows * 31 + blocks)
    x = mx.random.normal((rows, blocks)).astype(mx.float32) - mx.arange(blocks, dtype=mx.float32)[None] * 1e-12
    x[:, blocks - 7 :] = -mx.inf
    mx.eval(x)
    got = T.topk_rows(x, 512)
    want = mx.argpartition(x, kth=blocks - 512, axis=-1)[:, blocks - 512 :]
    mx.eval(got, want)
    assert got.shape == (rows, 512) and got.dtype == mx.uint32
    assert _sets(got) == _sets(want)


def test_fewer_blocks_than_k_returns_every_block():
    x = mx.random.normal((3, 300)).astype(mx.float32)
    got = T.topk_rows(x, 512)
    assert got.shape == (3, 300)
    assert all(s == set(range(300)) for s in _sets(got))


def test_equal_scores_at_the_cut_take_the_lowest_index():
    x = mx.zeros((2, 4096), dtype=mx.float32)
    x[:, :100] = 5.0
    x[1, 3000:3050] = 5.0
    mx.eval(x)
    got = _sets(T.topk_rows(x, 512))
    assert set(range(100)) <= got[0] and set(range(100, 512)) <= got[0] and len(got[0]) == 512
    assert set(range(100)) | set(range(3000, 3050)) <= got[1]
    assert got[1] - set(range(100)) - set(range(3000, 3050)) == set(range(100, 100 + 512 - 150))


def test_negative_and_infinite_scores_order_correctly():
    x = -mx.abs(mx.random.normal((2, 5000)).astype(mx.float32))
    x[:, 10:20] = -mx.inf
    mx.eval(x)
    got = T.topk_rows(x, 512)
    want = mx.argpartition(x, kth=5000 - 512, axis=-1)[:, 5000 - 512 :]
    assert _sets(got) == _sets(want)


def test_the_switch_is_opt_in(monkeypatch):
    monkeypatch.delenv(T.ENV, raising=False)
    assert not T.enabled()
    monkeypatch.setenv(T.ENV, "1")
    assert T.enabled()
    monkeypatch.setenv(T.ENV, "0")
    assert not T.enabled()


def test_bad_input_is_refused():
    with pytest.raises(ValueError):
        T.topk_rows(mx.zeros((2, 8), dtype=mx.float16), 4)
