"""CPU admission/rounding checks and separately allocated Metal exactness.

Use ``-k 'not TestMetal'`` with MLX imports blocked for CPU-only work. Metal
cases establish stage equality on small tensors; full model states/logits and
sampled product A/B remain separate promotion gates.
"""

from __future__ import annotations

import importlib.util
import struct
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

_spec = importlib.util.spec_from_file_location(
    "hc_prefill_contract", Path(__file__).parents[1] / "mtplx/kernels/hc_prefill.py"
)
hc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hc)


@pytest.mark.parametrize(
    "shape,count,hidden,rows",
    [
        ((1, 32, 10240), 4, 2560, 32),
        ((1, 33, 10240), 4, 2560, 33),
        ((1, 4096, 10240), 4, 2560, 4096),
        ((1, 8192, 10240), 4, 2560, 8192),
        ((2, 4096, 10240), 4, 2560, 8192),
        ((1, 8193, 10240), 4, 2560, None),
        ((2, 4097, 10240), 4, 2560, None),
        ((1, 4, 10240), 4, 2560, None),
        ((32, 1, 10240), 4, 2560, None),
        ((1, 31, 10240), 4, 2560, None),
        ((1, 32, 10240), 8, 1280, None),
        ((1, 32, 512), 4, 128, None),
        ((1, 32, 10239), 4, 2560, None),
        ((32, 10240), 4, 2560, None),
    ],
)
def test_cpu_only_prefill_family_geometry(shape, count, hidden, rows):
    assert hc._geometry(shape, count, hidden) == rows


def _bf16(value):
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return struct.unpack("<f", struct.pack("<I", rounded))[0]


def _bf16_mean(products):
    # Four MLX reduction lanes, each initialized to +0, followed by the
    # ascending lane fold and a BF16 reciprocal multiply.
    partials = [_bf16(_bf16(value) + 0.0) for value in products]
    value = partials[0]
    for partial in partials[1:]:
        value = _bf16(partial + value)
    return _bf16(value * _bf16(0.25))


def test_cpu_mean_contract_rejects_float32_accumulation():
    products = [256.0, 1.0, -256.0, 1.0]
    assert _bf16_mean(products) == 0.25
    assert sum(products) / 4 == 0.5


def test_cpu_initial_reduction_zero_is_observable():
    result = _bf16_mean([-0.0] * 4)
    assert struct.pack("<f", result) == struct.pack("<f", 0.0)
    assert struct.pack("<f", result) != struct.pack("<f", -0.0)


class _Owner(SimpleNamespace):
    def __contains__(self, name):
        return hasattr(self, name)


@pytest.mark.parametrize(
    "unsupported",
    ["dtype", "cpu", "backend", "norm", "down", "up", "inject", "quantized", "bias"],
)
def test_cpu_unsupported_contract_declines_before_any_tensor_work(
    monkeypatch, unsupported
):
    mlx, mx, nn = ModuleType("mlx"), ModuleType("mlx.core"), ModuleType("mlx.nn")
    mx.bfloat16, mx.float16, mx.gpu = object(), object(), object()
    mx.metal = SimpleNamespace(is_available=lambda: True)
    mx.default_device = lambda: mx.gpu
    mlx.core, mlx.nn = mx, nn
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", mx)
    monkeypatch.setitem(sys.modules, "mlx.nn", nn)

    def projection(shape):
        return SimpleNamespace(weight=SimpleNamespace(shape=shape, dtype=mx.bfloat16))

    owner = _Owner(
        hc_count=4,
        hidden_size=2560,
        hc_norm=SimpleNamespace(
            weight=SimpleNamespace(shape=(10240,), dtype=mx.bfloat16), group_size=2560
        ),
        input_mix_weight_down=projection((320, 10240)),
        input_mix_weight_up=projection((10240, 320)),
        block_inject_weight=projection((4, 10240)),
    )
    x = SimpleNamespace(shape=(1, 64, 10240), dtype=mx.bfloat16)
    if unsupported == "dtype":
        x.dtype = object()
    elif unsupported == "cpu":
        mx.default_device = lambda: object()
    elif unsupported == "backend":
        mx.metal.is_available = lambda: False
    elif unsupported == "norm":
        owner.hc_norm.weight.dtype = object()
    elif unsupported in ("down", "up", "inject"):
        names = {
            "down": "input_mix_weight_down",
            "up": "input_mix_weight_up",
            "inject": "block_inject_weight",
        }
        getattr(owner, names[unsupported]).weight.dtype = object()
    elif unsupported == "quantized":
        owner.input_mix_weight_down.scales = object()
    else:
        owner.input_mix_weight_up.bias = object()
    assert hc.hc_prefill_read(owner, x) is None


class TestMetal:
    @staticmethod
    def _mlx():
        mx = pytest.importorskip("mlx.core")
        if not mx.metal.is_available():
            pytest.skip("Metal unavailable")
        return mx

    @staticmethod
    def _same(mx, actual, expected):
        different = mx.sum(actual.view(mx.uint16) != expected.view(mx.uint16))
        mx.eval(different)
        assert different.item() == 0, f"{different.item()} BF16 words differ"

    @pytest.mark.parametrize(
        "rows,hidden",
        [
            (32, 2560),
            (33, 2560),
            (64, 2560),
            (257, 2560),
            (1024, 128),
            (4096, 128),
            (8192, 128),
        ],
    )
    def test_norm_and_mix_stages(self, rows, hidden):
        mx = self._mlx()
        with mx.stream(mx.gpu):
            mx.random.seed(rows)
            x = (3 * mx.random.normal((1, rows, 4 * hidden))).astype(mx.bfloat16)
            weight = (1 + 0.1 * mx.random.normal((4 * hidden,))).astype(mx.bfloat16)
            up = (12 * mx.random.normal(x.shape)).astype(mx.bfloat16)
            grouped = x.reshape(1, rows, 4, hidden)
            expected_norm = (
                mx.fast.rms_norm(grouped, None, 1e-6).reshape(x.shape) * weight
            )
            actual_norm = hc._normalize(x, weight, 1e-6, hidden=hidden)
            self._same(mx, actual_norm, expected_norm)
            expected_mix = mx.mean(
                mx.sigmoid(up).reshape(grouped.shape)
                * expected_norm.reshape(grouped.shape),
                axis=-2,
            )
            actual_mix = hc._mix(up, actual_norm, hidden=hidden)
            self._same(mx, actual_mix, expected_mix)

    def test_mix_exhausts_bf16_sigmoid_inputs(self):
        mx = self._mlx()
        with mx.stream(mx.gpu):
            bits = mx.arange(65536, dtype=mx.uint32).astype(mx.uint16)
            up = bits.view(mx.bfloat16).reshape(1, 128, 512)
            normed = mx.full(up.shape, -0.75, dtype=mx.bfloat16)
            expected = mx.mean(
                (mx.sigmoid(up) * normed).reshape(1, 128, 4, 128), axis=-2
            )
            self._same(mx, hc._mix(up, normed, hidden=128), expected)

    @pytest.mark.parametrize("products", [[-0.0] * 4, [256.0, 1.0, -256.0, 1.0]])
    def test_mix_preserves_reduction_rounding_and_signed_zero(self, products):
        mx = self._mlx()
        with mx.stream(mx.gpu):
            up = mx.full((1, 32, 512), float("inf"), dtype=mx.bfloat16)
            normed = mx.broadcast_to(
                mx.array(products, dtype=mx.bfloat16)[None, None, :, None],
                (1, 32, 4, 128),
            ).reshape(up.shape)
            expected = mx.mean(
                (mx.sigmoid(up) * normed).reshape(1, 32, 4, 128), axis=-2
            )
            self._same(mx, hc._mix(up, normed, hidden=128), expected)

    @pytest.mark.parametrize("combine", [False, True])
    def test_complete_native_projection_read(self, monkeypatch, combine):
        mx = self._mlx()
        import mlx.nn as nn
        from mtplx.kernels import hc_prefill as live
        from mtplx.models.qwen4_exp import GatedResidual

        calls = []
        native_helper = live.hc_prefill_read

        def checked_helper(owner, value):
            result = native_helper(owner, value)
            assert result is not None
            calls.append(value.shape)
            return result

        monkeypatch.setattr(live, "hc_prefill_read", checked_helper)
        with mx.stream(mx.gpu):
            mx.random.seed(71)
            owner = GatedResidual(
                SimpleNamespace(
                    hc_count=4, hidden_size=2560, hc_lowrank=320, rms_norm_eps=1e-6
                ),
                use_combine=combine,
            )
            owner.set_dtype(mx.bfloat16)
            x = mx.random.normal((1, 33, 10240)).astype(mx.bfloat16)
            normed = owner.hc_norm(x)
            mix = nn.silu(owner.input_mix_weight_down(normed) / owner.hc_count)
            up = mx.sigmoid(owner.input_mix_weight_up(mix))
            expected_mix = mx.mean(
                up.reshape(1, 33, 4, 2560) * normed.reshape(1, 33, 4, 2560), axis=-2
            )
            actual = owner(x)
            assert calls == [x.shape]
            if combine:
                expected_inject = 2.0 * mx.sigmoid(
                    owner.block_inject_weight(normed) / owner.hc_count
                )
                assert actual[1] is x
                self._same(mx, actual[0], expected_mix)
                self._same(mx, actual[2], expected_inject)
            else:
                self._same(mx, actual, expected_mix)
