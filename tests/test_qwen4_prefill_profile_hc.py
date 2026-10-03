"""The prefill profiler keeps hc_read and hc_write apart when the write is handed over."""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="MLX GPU arrays")


def test_a_handed_over_write_is_timed_as_a_write_and_read_as_before(monkeypatch):
    from mtplx import qwen4_prefill_profile as profile
    from mtplx.models import qwen4_exp as family

    monkeypatch.setenv("MTPLX_QWEN4_PREFILL_PROFILE", "1")
    monkeypatch.setenv("MTPLX_QWEN4_PREFILL_PROFILE_MIN_ROWS", "4")
    # install() rebinds these globally; the monkeypatch restores them.
    for owner, name in (
        (family.GatedResidual, "__call__"),
        (family, "_hyper_residual_write"),
        (family.GatedDeltaNet, "__call__"),
        (family.Attention, "__call__"),
        (family.SparseMoeBlock, "__call__"),
        (family.PLELayer, "__call__"),
        (family._FusedGateUpSwitchGLU, "__call__"),
        (family._FusedGateUpMLP, "__call__"),
        (family.QSAIndexer, "__call__"),
    ):
        monkeypatch.setattr(owner, name, getattr(owner, name))
    from mtplx.kernels import qsa_prefill_flash

    monkeypatch.setattr(
        qsa_prefill_flash, "qsa_prefill_flash", qsa_prefill_flash.qsa_prefill_flash
    )
    monkeypatch.setattr(profile, "_INSTALLED", False)
    monkeypatch.setattr(profile, "_TOTALS", {})
    monkeypatch.setattr(profile, "_CALLS", {})
    monkeypatch.setattr(profile, "_ROWS", {})
    stock_write = family._hyper_residual_write
    args = SimpleNamespace(hc_count=2, hidden_size=64, hc_lowrank=16, rms_norm_eps=1e-6)
    module = family.GatedResidual(args)
    mx.eval(module.parameters())
    x = mx.random.normal((1, 8, 128), key=mx.random.key(0)).astype(mx.float32)
    block = mx.random.normal((1, 8, 64), key=mx.random.key(1)).astype(mx.float32)
    gates = mx.random.uniform(0.0, 2.0, (1, 8, 2), key=mx.random.key(2)).astype(mx.float32)
    expected = module(stock_write(x, block, gates))

    assert profile.install()
    got = module(x, pending=(block, gates))
    mx.eval(*expected, *got)
    for want, have in zip(expected, got):
        np.testing.assert_array_equal(np.array(want), np.array(have))
    assert profile._CALLS.get("hc_write") == 1
    assert profile._CALLS.get("hc_read") == 1
