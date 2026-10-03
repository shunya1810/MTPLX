"""A quantized projection's bf16 cast must not change eager verify numerics."""

import mlx.core as mx
import pytest

from mtplx.attention_math import attention_gate
from mtplx.attention_context import attention_phase


def test_bf16_eager_gate_matches_the_existing_compiled_projection_gate():
    # The real 27B first differed at this input, after identical Q/K/V and
    # attention outputs. Include both tails and the ordinary operating range.
    raw = mx.array([-30.0, -10.0, -6.85, -6.84, -1.0, 0.0, 1.0, 10.0])
    output = mx.array([0.5, -2.0, 1.0, 4.0, -0.25, 2.0, 0.1, -0.5], dtype=mx.bfloat16)
    mx.eval(raw, output)

    def existing_trace(raw, output):
        return output * mx.sigmoid(raw.astype(mx.bfloat16))

    def candidate_trace(raw, output):
        return attention_gate(output, raw.astype(mx.bfloat16))

    with attention_phase("decode_verify"):
        reference = mx.compile(existing_trace)(raw, output)
        eager = candidate_trace(raw, output)
        compiled = mx.compile(candidate_trace)(raw, output)
    assert mx.array_equal(eager, reference).item()
    assert mx.array_equal(compiled, reference).item()


@pytest.mark.parametrize("dtype", [mx.float16, mx.float32])
def test_other_gate_dtypes_keep_the_stock_expression(dtype):
    gate = mx.array([-10.0, -1.0, 0.0, 1.0, 10.0], dtype=dtype)
    output = mx.array([0.5, -2.0, 1.0, 4.0, -0.25], dtype=dtype)
    assert mx.array_equal(attention_gate(output, gate), output * mx.sigmoid(gate)).item()


@pytest.mark.parametrize("phase", ["prefill", "ar_decode"])
def test_bf16_prefill_and_plain_decode_keep_the_stock_expression(phase):
    gate = mx.array([-6.85, -6.84, 0.0, 1.0], dtype=mx.bfloat16)
    output = mx.ones(gate.shape, dtype=mx.bfloat16)
    with attention_phase(phase):
        assert mx.array_equal(attention_gate(output, gate), output * mx.sigmoid(gate)).item()


def _dense_gate_grid(dtype):
    # Both tails and the operating range, densely: the fused and standalone
    # sigmoid lowerings differ at scattered inputs, not at a few landmarks.
    raw = mx.concatenate(
        [mx.linspace(-30.0, 30.0, 1 << 15), mx.random.normal((1 << 15,), key=mx.random.key(5)) * 4.0]
    ).astype(dtype)
    output = (mx.random.normal(raw.shape, key=mx.random.key(6)) * 2.0).astype(dtype)
    mx.eval(raw, output)
    return raw, output


def test_f32_eager_verify_gate_matches_the_compiled_trace_bit_for_bit():
    # A float32 verify trace fuses the gate's sigmoid with its multiply (one
    # JIT kernel, fast exp); the stock eager expression runs the precompiled
    # sigmoid (precise exp). On an M5 the TF32 o_proj GEMM rounded the
    # difference away; the M1 to M4 float32 kernels did not, and the
    # compiled verifier's hidden state and logits stopped matching eager.
    raw, output = _dense_gate_grid(mx.float32)

    def existing_trace(raw, output):
        return output * mx.sigmoid(raw)

    with attention_phase("decode_verify"):
        reference = mx.compile(existing_trace)(raw, output)
        eager = attention_gate(output, raw)
        compiled = mx.compile(lambda r, o: attention_gate(o, r))(raw, output)
    assert mx.array_equal(eager, reference).item()
    assert mx.array_equal(compiled, reference).item()


def test_f16_gate_needs_no_helper_because_both_lowerings_agree_on_every_value():
    # Every finite float16 value: MLX 0.32.2's fused and standalone float16
    # sigmoid agree bit for bit, so float16 keeps the stock expression on
    # decode_verify. If an MLX upgrade breaks this, give float16 a compiled
    # gate like float32's.
    import numpy as np

    every = np.arange(1 << 16, dtype=np.uint16).view(np.float16)
    raw = mx.array(every[np.isfinite(every)])
    output = mx.ones(raw.shape, dtype=mx.float16)
    mx.eval(raw, output)

    def trace(raw, output):
        return output * mx.sigmoid(raw)

    with attention_phase("decode_verify"):
        reference = mx.compile(trace)(raw, output)
        eager = attention_gate(output, raw)
    assert mx.array_equal(eager, reference).item()


@pytest.mark.parametrize("phase", ["prefill", "ar_decode"])
def test_f32_prefill_and_plain_decode_keep_the_stock_expression(phase):
    gate = mx.array([-6.85, -6.84, 0.0, 1.0], dtype=mx.float32)
    output = mx.ones(gate.shape, dtype=mx.float32)
    with attention_phase(phase):
        assert mx.array_equal(attention_gate(output, gate), output * mx.sigmoid(gate)).item()
