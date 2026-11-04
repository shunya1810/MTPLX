import mlx.core as mx
import pytest

from mtplx.kernels.chunk_topk import chunk_topk_supported, topk_candidates

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="Metal kernel")


@pytest.mark.parametrize("vocab", [248320, 103887, 4096])
@pytest.mark.parametrize("k,m", [(20, 80), (20, 20), (1, 4), (64, 64)])
def test_candidates_hold_the_rows_top_k(vocab, k, m):
    if not chunk_topk_supported(k, vocab, m):
        assert vocab // 4096 * k < m
        return
    rows = (mx.random.normal((2, vocab), key=mx.random.key(vocab + k)) * 3).astype(mx.float32)
    idx, vals = topk_candidates(rows, k, m)
    assert idx.shape == (2, m) and vals.shape == (2, m)
    for r in range(2):
        assert mx.array_equal(mx.take(rows[r], idx[r]), vals[r]).item()
        top = mx.sort(rows[r])[-k:]
        assert mx.array_equal(mx.sort(vals[r])[-k:], top).item()


def test_ties_and_masked_tails_still_cover_the_top_k():
    rows = mx.round(mx.random.normal((2, 50000), key=mx.random.key(1)) * 2) / 2
    rows = mx.concatenate([rows, mx.full((2, 1000), -mx.inf)], axis=-1)
    _idx, vals = topk_candidates(rows, 20, 80)
    for r in range(2):
        assert mx.array_equal(mx.sort(vals[r])[-20:], mx.sort(rows[r])[-20:]).item()


def test_support_bounds():
    assert chunk_topk_supported(20, 248320)
    assert not chunk_topk_supported(0, 248320)
    assert not chunk_topk_supported(65, 248320)
    assert not chunk_topk_supported(20, 1000)
    assert not chunk_topk_supported(20, 4096, 80)  # one chunk holds only 20 winners
