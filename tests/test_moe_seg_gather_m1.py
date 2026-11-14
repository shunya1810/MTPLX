"""M1 expert-segmented sorted gather: bit-identical to mx.gather_qmm."""

from __future__ import annotations

import mlx.core as mx
import pytest

from mtplx.kernels import moe_seg_gather_m1 as seg


def _case(rows_per_token: int, skew: bool, dtype=mx.float16, n: int = 256, k: int = 512):
    mx.random.seed(3)
    experts, tokens = 16, 512  # 2,048 routed rows: the seg-gather threshold
    w = (mx.random.normal((experts, n, k)) * 0.02).astype(dtype)
    wq, scales, biases = mx.quantize(w, group_size=64, bits=4)
    x = (mx.random.normal((tokens, k)) * 0.5).astype(dtype)
    logits = mx.random.normal((tokens, experts))
    if skew:
        logits = logits + mx.arange(experts).astype(mx.float32) * 0.5
    inds = mx.argpartition(-logits, kth=rows_per_token - 1, axis=-1)[:, :rows_per_token]
    flat = inds.reshape(-1).astype(mx.uint32)
    order = mx.argsort(flat)
    xs = x[order // rows_per_token][:, None, :]
    return xs, wq, scales, biases, flat[order]


@pytest.mark.parametrize("skew", [False, True])
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_segmented_gather_is_bit_identical(monkeypatch, skew, dtype):
    monkeypatch.setenv("MTPLX_M1_MOE_SEG_GATHER", "1")
    seg._reset_for_tests()
    xs, wq, scales, biases, idx = _case(4, skew, dtype)
    assert seg.applies(xs, wq, scales, biases, idx, group_size=64, bits=4, sorted_indices=True)
    ref = mx.gather_qmm(xs, wq, scales, biases, rhs_indices=idx, transpose=True,
                        group_size=64, bits=4, sorted_indices=True)
    out = seg.gather_qmm(xs, wq, scales, biases, idx, group_size=64, bits=4)
    assert out.shape == ref.shape
    assert bool(mx.array_equal(out, ref).item())
    assert seg.stats()["canary_failures"] == 0


def test_declines_narrow_unsorted_and_off(monkeypatch):
    monkeypatch.setenv("MTPLX_M1_MOE_SEG_GATHER", "1")
    xs, wq, scales, biases, idx = _case(4, False)
    kw = dict(group_size=64, bits=4)
    assert not seg.applies(xs[:512], wq, scales, biases, idx[:512], sorted_indices=True, **kw)
    assert not seg.applies(xs, wq, scales, biases, idx, sorted_indices=False, **kw)
    monkeypatch.setenv("MTPLX_M1_MOE_SEG_GATHER", "0")
    assert not seg.applies(xs, wq, scales, biases, idx, sorted_indices=True, **kw)


def test_narrow_tiles_take_64_deep_k_steps_and_wide_tiles_keep_32():
    from mtplx.kernels import moe_seg_gather_m1 as seg

    assert seg._bm_for(1024) == seg.NARROW_BM and seg._bm_for(8192) == seg.NARROW_BM
    assert seg._bm_for(16384) == seg.BM
    assert seg._bk_for(seg.NARROW_BM, 2048) == 64 and seg._bk_for(seg.NARROW_BM, 512) == 64
    assert seg._bk_for(seg.NARROW_BM, 96) == seg.BK  # K not a multiple of 64
    assert seg._bk_for(seg.BM, 2048) == seg.BK


def test_low_row_class_is_off_by_default():
    from mtplx.kernels import moe_seg_gather_m1 as seg

    assert seg.LOW_MIN_ROWS == seg.MIN_ROWS
