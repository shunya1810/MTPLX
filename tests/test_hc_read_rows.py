"""The any-width hyper-connection read for decode/verify rows (M1 default).

Runs the real Metal kernels on a small GatedResidual: against the eager chain
within a few ulp, row-independent (a row reads the same alone or in a verify
batch), and gated off where it does not apply.
"""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import pytest

from mtplx.kernels import hc_read_rows as K
from mtplx.models import qwen4_exp as Q

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="Metal kernels")

ARGS = SimpleNamespace(hidden_size=256, hc_count=4, hc_lowrank=64, rms_norm_eps=1e-6)


def _residual(dtype, seed=0):
    mx.random.seed(seed)
    module = Q.GatedResidual(ARGS)
    module.hc_norm.weight = 1.0 + 0.1 * mx.random.normal(module.hc_norm.weight.shape)
    module.update(
        {k: v for k, v in _cast(module.parameters(), dtype).items()}
    )
    mx.eval(module.parameters())
    return module


def _cast(tree, dtype):
    if isinstance(tree, dict):
        return {k: _cast(v, dtype) for k, v in tree.items()}
    if isinstance(tree, list):
        return [_cast(v, dtype) for v in tree]
    return tree.astype(dtype)


def _eager(module, x, pending, monkeypatch):
    with monkeypatch.context() as m:
        m.setenv("MTPLX_HC_READ_ROWS", "0")
        return module(x, pending=pending)


def _inputs(rows, dtype, seed=3):
    mx.random.seed(seed)
    x = (3.0 * mx.random.normal((1, rows, 4 * ARGS.hidden_size))).astype(dtype)
    bo = mx.random.normal((1, rows, ARGS.hidden_size)).astype(dtype)
    inj = (2.0 * mx.random.uniform(shape=(1, rows, 4))).astype(dtype)
    return x, bo, inj


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("rows", [1, 2, 3, 4])
@pytest.mark.parametrize("with_pending", [False, True], ids=["read", "pending"])
def test_matches_the_eager_chain_within_a_few_ulp(monkeypatch, dtype, rows, with_pending):
    module = _residual(dtype)
    x, bo, inj = _inputs(rows, dtype)
    pending = (bo, inj) if with_pending else None
    assert K.supports(module, x)
    km, kh, ki = K.hc_read_rows(module, x, pending)
    em, eh, ei = _eager(module, x, pending, monkeypatch)
    ulp = 2.0 ** -10 if dtype == mx.float16 else 2.0 ** -7
    scale = float(mx.max(mx.abs(em.astype(mx.float32))).item())
    assert float(mx.max(mx.abs(km.astype(mx.float32) - em.astype(mx.float32))).item()) <= 4 * ulp * max(1.0, scale)
    assert float(mx.max(mx.abs(ki.astype(mx.float32) - ei.astype(mx.float32))).item()) <= 4 * ulp * 2.0
    # The written stream is the same elementwise expression as the eager write.
    assert bool(mx.array_equal(kh, eh).item())


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16], ids=["fp16", "bf16"])
def test_a_row_reads_the_same_alone_and_in_a_verify_batch(dtype):
    module = _residual(dtype)
    x, bo, inj = _inputs(4, dtype)
    batch = K.hc_read_rows(module, x, (bo, inj))
    for row in range(4):
        alone = K.hc_read_rows(module, x[:, row : row + 1], (bo[:, row : row + 1], inj[:, row : row + 1]))
        for got, want in zip(alone, batch):
            assert bool(mx.array_equal(got, want[:, row : row + 1]).item()), row


def test_gates(monkeypatch):
    module = _residual(mx.float16)
    x, _, _ = _inputs(4, mx.float16)
    wide, _, _ = _inputs(K.MAX_ROWS + 1, mx.float16)
    assert K.supports(module, x)
    assert not K.supports(module, wide)
    assert not K.supports(module, x.astype(mx.float32))
    other = _residual(mx.bfloat16)
    assert not K.supports(other, x)  # weights must be of the activation dtype
    monkeypatch.setenv("MTPLX_HC_READ_ROWS", "0")
    assert not K.enabled()
    monkeypatch.setenv("MTPLX_HC_READ_ROWS", "1")
    assert K.enabled()
    assert Q._hc_rows_read_applies(module, x)
