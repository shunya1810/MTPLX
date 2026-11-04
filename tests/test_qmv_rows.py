import mlx.core as mx
import pytest

from mtplx.attention_context import attention_phase, exact_verify
from mtplx.kernels import qmv_rows as Q

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="Metal kernel")


def _weights(n, k, dtype, group_size=64):
    w = (mx.random.normal((n, k), key=mx.random.key(3)) * 0.02).astype(dtype)
    return mx.quantize(w, group_size=group_size, bits=4)


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("group_size", [32, 64])
def test_exact_kernel_is_bit_identical_to_stock(dtype, group_size):
    wq, scales, biases = _weights(2048, 1024, dtype, group_size)
    x = (mx.random.normal((2, 1024), key=mx.random.key(5)) * 0.7).astype(dtype)
    ref = mx.quantized_matmul(x, wq, scales, biases, transpose=True, group_size=group_size, bits=4)
    out = Q.qmv_rows(x, wq, scales, biases, group_size=group_size, exact=True)
    assert mx.array_equal(ref, out).item()


def test_half_x_kernel_stays_within_its_drift():
    wq, scales, biases = _weights(4096, 2048, mx.float16)
    x = (mx.random.normal((2, 2048), key=mx.random.key(9)) * 0.7).astype(mx.float16)
    ref = mx.quantized_matmul(x, wq, scales, biases, transpose=True, group_size=64, bits=4).astype(mx.float32)
    out = Q.qmv_rows(x, wq, scales, biases, group_size=64, exact=False).astype(mx.float32)
    assert (mx.abs(out - ref).max() / mx.abs(ref).max()).item() < 3e-3


def test_route_serves_sampled_verifies_only(monkeypatch):
    monkeypatch.setenv("MTPLX_M1_QMV_ROWS", "1")
    wq, scales, biases = _weights(2048, 1024, mx.float16)
    x = (mx.random.normal((1, 2, 1024), key=mx.random.key(1)) * 0.7).astype(mx.float16)
    kw = dict(group_size=64, bits=4)
    with attention_phase("decode_verify"), exact_verify(False):
        assert Q.sampled_route_active()
        y = Q.maybe_qmv_rows(x, wq, scales, biases, **kw)
        assert y is not None and y.shape == (1, 2, 2048)
    with attention_phase("decode_verify"), exact_verify(True):
        assert not Q.sampled_route_active()
        assert Q.maybe_qmv_rows(x, wq, scales, biases, **kw) is None
    with attention_phase("prefill"), exact_verify(False):
        assert Q.maybe_qmv_rows(x, wq, scales, biases, **kw) is None
    monkeypatch.setenv("MTPLX_M1_QMV_ROWS", "0")
    with attention_phase("decode_verify"), exact_verify(False):
        assert not Q.sampled_route_active()


def test_route_skips_shapes_stock_serves_as_fast(monkeypatch):
    monkeypatch.setenv("MTPLX_M1_QMV_ROWS", "1")
    kw = dict(group_size=64, bits=4)
    with attention_phase("decode_verify"), exact_verify(False):
        small = _weights(256, 1024, mx.float16)
        x2 = mx.zeros((2, 1024), mx.float16)
        assert Q.maybe_qmv_rows(x2, *small, **kw) is None  # N below 2048
        big = _weights(2048, 1024, mx.float16)
        assert Q.maybe_qmv_rows(mx.zeros((1, 1024), mx.float16), *big, **kw) is None  # one row
        assert Q.maybe_qmv_rows(mx.zeros((3, 1024), mx.float16), *big, **kw) is None  # three rows
