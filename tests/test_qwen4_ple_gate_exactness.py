"""Flash-Next's PLE gate: the compiled verifier's lowering on eager verify forwards.

PLELayer (mtplx/models/qwen4_exp.py) gates each hyper-connection stream by
``sigmoid(gate) * value`` in bfloat16. Inside a compiled verify trace that
sigmoid fuses with its multiply and runs Metal's fast exp; an eager verify
forward ran MLX's standalone sigmoid (precise exp), and in bfloat16 the two
differ at gate -6.84375 (MLX 0.32.2). The layer here is the tiny pack's (the
Flash-Next compiled-route tests use it), with its key and query norms
replaced so that stream 0's gate is exactly -6.84375 in every row: its dot
is -376, over sqrt(64) that is -47, and sqrt(47) rounds to 6.84375 in
bfloat16.
"""

from __future__ import annotations

import dataclasses
import importlib.util
from pathlib import Path

import mlx.core as mx
import mlx.utils
import pytest

from mtplx.attention_context import attention_phase

ROWS = 4


def _tiny_args():
    path = Path(__file__).resolve().parents[1] / "scripts" / "qwen4exp_mtp_tiny_smoke.py"
    spec = importlib.util.spec_from_file_location("qwen4exp_mtp_tiny_smoke", path)
    smoke = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(smoke)
    return dataclasses.replace(smoke._tiny_text_args(), ple_layer_ids=[1])


def _layer(monkeypatch):
    from mtplx.models.qwen4_exp import PLELayer

    args = _tiny_args()
    mx.random.seed(0)
    layer = PLELayer(args, 0)
    layer.update(
        mlx.utils.tree_map(
            lambda p: p.astype(mx.bfloat16) if p.dtype == mx.float32 else p,
            layer.parameters(),
        )
    )
    mx.eval(layer.parameters())
    width = args.hidden_size * args.hc_count
    assert args.hidden_size == 64
    key = mx.zeros((1, ROWS, width), dtype=mx.bfloat16)
    query = mx.zeros((1, ROWS, width), dtype=mx.bfloat16)
    key[..., 0] = -16.0
    query[..., 0] = 23.5
    # A fresh layer's n-gram table is zero until a checkpoint fills it, and
    # a zero value hides any gate; give the streams a value to carry.
    value = (mx.random.normal((1, ROWS, args.hidden_size), key=mx.random.key(2)) + 2.0).astype(
        mx.bfloat16
    )
    mx.eval(key, query, value)
    monkeypatch.setattr(layer, "norm_key", lambda _x: key)
    monkeypatch.setattr(layer, "norm_query", lambda _x: query)
    monkeypatch.setattr(layer, "value_proj", lambda _x: value)
    # The layer's own gate arithmetic on these rows gives -6.84375.
    streams = (1, ROWS, args.hc_count, args.hidden_size)
    gate = (key.reshape(streams) * query.reshape(streams)).sum(axis=-1, keepdims=True) / 8.0
    gate = mx.sqrt(mx.maximum(mx.abs(gate), 1e-6)) * mx.sign(gate)
    assert gate[0, :, 0, 0].tolist() == [-6.84375] * ROWS
    return layer, width


def _inputs(width):
    hidden = (mx.random.normal((1, ROWS, width), key=mx.random.key(1)) * 0.5).astype(
        mx.bfloat16
    )
    ids = mx.array([[5, 6, 7, 8]], dtype=mx.int32)
    mx.eval(hidden, ids)
    return hidden, ids


@pytest.mark.skipif(not mx.metal.is_available(), reason="fused kernels are the GPU's")
def test_an_eager_verify_forward_takes_the_compiled_verifiers_gate(monkeypatch):
    layer, width = _layer(monkeypatch)
    hidden, ids = _inputs(width)

    def forward(hidden, ids):
        return layer(hidden, ids, None)

    with attention_phase("decode_verify"):
        eager = forward(hidden, ids)
        compiled = mx.compile(forward)(hidden, ids)
        mx.eval(eager, compiled)
    assert mx.array_equal(compiled, eager).item()


@pytest.mark.parametrize("phase", ["prefill", "ar_decode", "postcommit", "unknown"])
def test_every_other_phase_keeps_the_stock_gate_bit_for_bit(monkeypatch, phase):
    # Outside decode_verify the gate is the expression the layer always had,
    # sigmoid(gate) * value, with its factors swapped (IEEE multiplication
    # commutes): the same bits, eagerly and compiled.
    import mtplx.models.qwen4_exp as qwen4_exp

    layer, width = _layer(monkeypatch)
    hidden, ids = _inputs(width)

    def forward(hidden, ids):
        return layer(hidden, ids, None)

    with attention_phase(phase):
        new = forward(hidden, ids)
        new_compiled = mx.compile(forward)(hidden, ids)
        mx.eval(new, new_compiled)
    monkeypatch.setattr(
        qwen4_exp, "attention_gate", lambda output, gate: mx.sigmoid(gate) * output
    )
    with attention_phase(phase):
        old = forward(hidden, ids)
        old_compiled = mx.compile(forward)(hidden, ids)
        mx.eval(old, old_compiled)
    assert mx.array_equal(new, old).item()
    assert mx.array_equal(new_compiled, old_compiled).item()
