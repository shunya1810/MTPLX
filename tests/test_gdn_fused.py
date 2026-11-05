from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="Metal kernel")


def _fake_gdn(seed):
    from mlx_lm.models.qwen3_next import Qwen3NextRMSNormGated

    k = mx.random.split(mx.random.key(seed), 4)
    conv = nn.Conv1d(8192, 8192, 4, groups=8192, bias=False)
    conv.weight = (mx.random.normal((8192, 4, 1), key=k[0]) * 0.3).astype(mx.float16)
    norm = Qwen3NextRMSNormGated(128, eps=1e-6)
    norm.weight = (1.0 + 0.1 * mx.random.normal((128,), key=k[1])).astype(mx.float16)
    return SimpleNamespace(
        conv1d=conv,
        norm=norm,
        A_log=(mx.random.normal((32,), key=k[2]) * 0.5).astype(mx.float16),
        dt_bias=(mx.random.normal((32,), key=k[3]) * 0.5).astype(mx.float16),
    )


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_fused_gdn_matches_the_stock_chain_bit_for_bit(seed, request):
    import mtplx.gdn_capture as gc
    from mtplx.kernels import gdn_fused as F
    from mtplx.kernels import moe_scalar

    moe_scalar.set_scalar(mx.float16)
    request.addfinalizer(lambda: moe_scalar.set_scalar(mx.bfloat16))
    gdn = _fake_gdn(seed)
    k = mx.random.split(mx.random.key(100 + seed), 3)
    proj = (mx.random.normal((1, 2, 12352), key=k[0]) * 0.8).astype(mx.float16)
    conv_state = (mx.random.normal((1, 3, 8192), key=k[1]) * 0.8).astype(mx.float16)
    state = mx.random.normal((1, 32, 128, 128), key=k[2]) * 0.05

    qkv, z, b, a = mx.split(proj, [8192, 12288, 12320], axis=-1)
    conv_out, conv_states = gc._stock_conv1d_capture(qkv, conv_state, gdn)
    out, states = gc._a3b_compiled_target_gdn_postconv_m2_tgy4(
        conv_out, a, b, state, A_log=gdn.A_log, dt_bias=gdn.dt_bias
    )
    y_ref = gdn.norm(out, z.reshape(1, 2, 32, 128))

    assert F.eligible(gdn, proj, conv_state, state)
    y, st, cs = F.gdn_fused_m2(gdn, proj, conv_state, state)
    assert mx.array_equal(y, y_ref).item()
    assert mx.array_equal(st, states).item()
    assert mx.array_equal(cs, conv_states).item()


def test_fused_gdn_rejects_other_shapes():
    from mtplx.kernels import gdn_fused as F

    gdn = _fake_gdn(0)
    state = mx.zeros((1, 32, 128, 128))
    conv_state = mx.zeros((1, 3, 8192), mx.float16)
    assert not F.eligible(gdn, mx.zeros((1, 3, 12352), mx.float16), conv_state, state)
    assert not F.eligible(gdn, mx.zeros((1, 2, 12352), mx.bfloat16), conv_state, state)


def test_install_wraps_only_after_the_selfcheck_and_the_layer_routes_to_it(request):
    import mtplx.gdn_capture as gc
    from mtplx.kernels import gdn_fused as F
    from mtplx.kernels import moe_scalar

    moe_scalar.set_scalar(mx.float16)
    request.addfinalizer(lambda: moe_scalar.set_scalar(mx.bfloat16))
    gdn = _fake_gdn(5)
    impl = lambda c, a, b, s: gc._a3b_compiled_target_gdn_postconv_m2_tgy4(  # noqa: E731
        c, a, b, s, A_log=gdn.A_log, dt_bias=gdn.dt_bias
    )
    kw = dict(stock_layer=gc._a3b_gdn_forward_with_fixed_postconv, stock_conv=gc._stock_conv1d_capture)

    gdn.in_proj_qkv = SimpleNamespace()  # projections not fused: unchanged
    assert F.install_m2([gdn], (impl,), **kw) == (impl,)
    assert F.status()["state"] == "skipped"

    proj = (mx.random.normal((1, 2, 12352), key=mx.random.key(9)) * 0.8).astype(mx.float16)
    gdn.in_proj_qkv = SimpleNamespace(_hub=SimpleNamespace(fused=lambda x: proj))
    gdn.out_proj = lambda v: v * 2
    (wrapped,) = F.install_m2([gdn], (impl,), **kw)
    assert isinstance(wrapped, F.FusedM2Layer) and F.status()["state"] == "installed"

    conv_state = (mx.random.normal((1, 3, 8192), key=mx.random.key(10)) * 0.8).astype(mx.float16)
    state = mx.random.normal((1, 32, 128, 128), key=mx.random.key(11)) * 0.05
    cache = [conv_state, state]
    out, captures = gc._a3b_gdn_forward_with_fixed_postconv(gdn, mx.zeros((1, 2, 2048), mx.float16), cache, wrapped)
    y, st, cs = F.gdn_fused_m2(gdn, proj, conv_state, state)
    assert mx.array_equal(out, y.reshape(1, 2, -1) * 2).item()
    assert mx.array_equal(captures["states"], st).item() and mx.array_equal(captures["conv_states"], cs).item()
    assert mx.array_equal(cache[0], cs[:, -1]).item() and mx.array_equal(cache[1], st[:, -1]).item()


@pytest.mark.parametrize("seed", [3, 4])
def test_tape_mode_matches_the_generic_tape_capture_bit_for_bit(seed):
    from mlx_lm.models.gated_delta import compute_g

    import mtplx.gdn_capture as gc
    from mtplx.kernels import gdn_fused as F

    gdn = _fake_gdn(seed)
    gdn.conv_dim, gdn.head_k_dim, gdn.head_v_dim, gdn.num_k_heads, gdn.num_v_heads, gdn.key_dim = (
        8192, 128, 128, 16, 32, 2048,
    )
    k = mx.random.split(mx.random.key(200 + seed), 3)
    proj = (mx.random.normal((1, 2, 12352), key=k[0]) * 0.8).astype(mx.float16)
    conv_state = (mx.random.normal((1, 3, 8192), key=k[1]) * 0.8).astype(mx.float16)
    state = mx.random.normal((1, 32, 128, 128), key=k[2]) * 0.05

    qkv, z, b, a = mx.split(proj, [8192, 12288, 12320], axis=-1)
    conv_out, conv_states = gc._stock_conv1d_capture(qkv, conv_state, gdn)
    g = compute_g(gdn.A_log, a, gdn.dt_bias)
    out, final_state, tape = gc._linear_gated_delta_from_conv_tape_capture(conv_out, g, mx.sigmoid(b), state, gdn)
    ref = (gdn.norm(out, z.reshape(1, 2, 32, 128)), final_state, conv_states, conv_out, g.astype(mx.float32), tape)

    got = F.gdn_fused_m2_tape(gdn, proj, conv_state, state)
    for name, x, y in zip(("y", "final_state", "conv_states", "conv_out", "g", "tape"), got, ref):
        assert mx.array_equal(x, y).item(), name
