"""Fused GDN output kernel parity (GPU: Metal).

The kernel must reproduce SigmoidRMSNormGated + QuantizedLinear on the SAME
4-bit pack, at both shipped forge group sizes, within accumulation-order
tolerance (the kernel casts the gated-normed values through bf16 exactly at
the module boundary the eager chain has)."""

import mlx.core as mx
import mlx.nn as nn
import pytest

from mtplx.kernels.gdn_out_fused import fused_gdn_out
from mtplx.models.qwen4_exp import SigmoidRMSNormGated

NH, HD = 48, 128
K = NH * HD
DMODEL = 2560


@pytest.fixture(params=[32, 64], ids=["g32", "g64"])
def setup(request):
    if mx.default_device().type != mx.DeviceType.gpu:
        pytest.skip("Metal kernel needs the GPU")
    gs = request.param
    mx.random.seed(23 + gs)
    w = (mx.random.normal((DMODEL, K)) * 0.03).astype(mx.bfloat16)
    qw, qs, qb = mx.quantize(w, group_size=gs, bits=4)
    x = (mx.random.normal((K,)) * 0.7).astype(mx.bfloat16)
    z = (mx.random.normal((K,)) * 0.8).astype(mx.bfloat16)
    norm = SigmoidRMSNormGated(HD)
    norm.weight = 1.0 + mx.random.normal((HD,)) * 0.05
    mx.eval(qw, qs, qb, x, z, norm.weight)
    return gs, x, z, norm, (qw, qs, qb)


def _reference(gs, x, z, norm, pack):
    qw, qs, qb = pack
    v = norm(x.reshape(NH, HD), z.reshape(NH, HD)).reshape(K)
    wd = mx.dequantize(qw, qs, qb, group_size=gs, bits=4).astype(mx.float32)
    return wd @ v.astype(mx.float32)


def test_kernel_matches_module_chain(setup):
    gs, x, z, norm, pack = setup
    y_k = fused_gdn_out(x, z, norm.weight.astype(mx.float32), *pack, group_size=gs)
    y_r = _reference(gs, x, z, norm, pack)
    mx.eval(y_k, y_r)
    scale = mx.abs(y_r).max().item() + 1e-6
    err = (mx.abs(y_k.astype(mx.float32) - y_r) / scale).max().item()
    assert err < 2e-2, f"g{gs} rel err {err}"


def test_rejects_bad_group_size(setup):
    gs, x, z, norm, pack = setup
    with pytest.raises(ValueError):
        fused_gdn_out(x, z, norm.weight.astype(mx.float32), *pack, group_size=80)


def test_gdn_module_fused_out_matches_stock(monkeypatch):
    if mx.default_device().type != mx.DeviceType.gpu:
        pytest.skip("Metal kernel needs the GPU")
    from mtplx.models.qwen4_exp import GatedDeltaNet, TextArgs

    mx.random.seed(31)
    gdn = GatedDeltaNet(TextArgs())  # family geometry: 48x128 -> 2560
    lin = gdn.out_proj
    lin.weight = (mx.random.normal(lin.weight.shape) * 0.03).astype(mx.bfloat16)
    gdn.out_proj = nn.QuantizedLinear.from_linear(lin, group_size=64, bits=4)
    gdn.norm.weight = 1.0 + mx.random.normal(gdn.norm.weight.shape) * 0.05
    mx.eval(gdn.parameters())

    x = (mx.random.normal((1, 1, 2560)) * 0.5).astype(mx.bfloat16)
    monkeypatch.setenv("MTPLX_FUSED_GDN_OUT", "0")
    ref = gdn(x)
    monkeypatch.setenv("MTPLX_FUSED_GDN_OUT", "1")
    fused = gdn(x)
    mx.eval(ref, fused)
    scale = mx.abs(ref.astype(mx.float32)).max().item() + 1e-6
    err = (
        mx.abs(fused.astype(mx.float32) - ref.astype(mx.float32)) / scale
    ).max().item()
    assert err < 2e-2, f"module fused-out rel err {err}"

    # multi-row inputs must never take the fused path
    multi = (mx.random.normal((1, 3, 2560)) * 0.5).astype(mx.bfloat16)
    assert not gdn._fused_out_applies(1, 3)
    out_multi = gdn(multi)
    assert out_multi.shape == (1, 3, 2560)


def test_a_wider_trunk_keeps_the_stock_out_proj(monkeypatch):
    """The kernel's output width (2560) is compiled in: a 5120-wide Qwen4 trunk
    with the same 48x128 GDN values must not take it."""

    from mtplx.models.qwen4_exp import GatedDeltaNet, TextArgs

    gdn = GatedDeltaNet(TextArgs(hidden_size=5120))
    gdn.out_proj = nn.QuantizedLinear.from_linear(gdn.out_proj, group_size=64, bits=4)
    monkeypatch.setenv("MTPLX_FUSED_GDN_OUT", "1")
    assert tuple(gdn.out_proj.weight.shape) == (5120, 768)
    assert not gdn._fused_out_applies(1, 1)
