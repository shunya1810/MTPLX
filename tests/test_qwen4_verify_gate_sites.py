"""Flash-Next's inject and shared-expert gates: eager verify forwards take the
compiled verifier's lowering.

Inside the compiled verifier's trace MLX 0.32.2 fuses two more bfloat16
sigmoids with their neighbours: the hyper-connection inject gate
``2 * sigmoid(logits / hc_count)`` and the MoE block's shared-expert gate
``sigmoid(gate) * shared + routed``. A fused sigmoid runs Metal's fast exp,
the standalone kernel the precise one, and in bfloat16 they differ at
-6.84375 (mtplx/attention_math.py). Each test plants that value through a
projection with a zero weight and a bias, so the sigmoid reads it from memory
as it reads a projection's output, and compares the module on decode_verify
eagerly and compiled. The compiled result is also checked against the stock
expression's compiled result: moving the eager forward onto the compiled
lowering leaves the compiled verifier's numbers as they were.
"""

from __future__ import annotations

import dataclasses
import importlib.util
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.utils
import pytest

from mtplx.attention_context import attention_phase

pytestmark = pytest.mark.skipif(
    not mx.metal.is_available(), reason="fused kernels are the GPU's"
)

GATE = -6.84375
ROWS = 4


def _tiny_args():
    path = Path(__file__).resolve().parents[1] / "scripts" / "qwen4exp_mtp_tiny_smoke.py"
    spec = importlib.util.spec_from_file_location("qwen4exp_mtp_tiny_smoke", path)
    smoke = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(smoke)
    return smoke._tiny_text_args()


def _bfloat16(module):
    module.update(
        mlx.utils.tree_map(
            lambda p: p.astype(mx.bfloat16) if p.dtype == mx.float32 else p,
            module.parameters(),
        )
    )
    mx.eval(module.parameters())
    return module


def _constant_projection(in_dims: int, out_dims: int, value: float) -> nn.Linear:
    """A bfloat16 projection that outputs ``value`` in every row."""

    projection = nn.Linear(in_dims, out_dims, bias=True)
    projection.weight = mx.zeros((out_dims, in_dims), dtype=mx.bfloat16)
    projection.bias = mx.full((out_dims,), value, dtype=mx.bfloat16)
    mx.eval(projection.parameters())
    return projection


def _rows(width: int, seed: int) -> mx.array:
    rows = mx.random.normal((1, ROWS, width), key=mx.random.key(seed)).astype(mx.bfloat16)
    mx.eval(rows)
    return rows


def _eager_and_compiled(forward, *inputs):
    with attention_phase("decode_verify"):
        eager = forward(*inputs)
        compiled = mx.compile(forward)(*inputs)
        mx.eval(eager, compiled)
    return eager, compiled


def test_the_inject_gate_takes_the_compiled_lowering_on_verify():
    from mtplx.models.qwen4_exp import GatedResidual

    args = _tiny_args()
    mx.random.seed(0)
    connection = _bfloat16(GatedResidual(args))
    width = args.hc_count * args.hidden_size
    connection.block_inject_weight = _constant_projection(
        width, args.hc_count, GATE * args.hc_count
    )
    hyper = _rows(width, 1)

    def inject(hyper):
        return connection(hyper)[2]

    def stock_inject(hyper):
        logits = connection.block_inject_weight(connection.hc_norm(hyper))
        return 2.0 * mx.sigmoid(logits / connection.hc_count)

    eager, compiled = _eager_and_compiled(inject, hyper)
    _, stock_compiled = _eager_and_compiled(stock_inject, hyper)
    assert eager.dtype == mx.bfloat16
    assert mx.array_equal(compiled, eager).item()
    assert mx.array_equal(compiled, stock_compiled).item()


def _moe_block():
    from mtplx.models.qwen4_exp import SparseMoeBlock

    args = _tiny_args()
    mx.random.seed(0)
    block = _bfloat16(SparseMoeBlock(args))
    return block, args


def test_the_shared_expert_gate_takes_the_compiled_lowering_on_verify():
    from mlx_lm.models.qwen3_next import Qwen3NextSparseMoeBlock

    block, args = _moe_block()
    block.shared_expert_gate = _constant_projection(args.hidden_size, 1, GATE)
    # Zero routed experts: the block's output is then the gated shared expert
    # alone, and a moved gate is not rounded away in the sum.
    down = block.switch_mlp.down_proj
    down.weight = mx.zeros_like(down.weight)
    mx.eval(down.weight)
    x = _rows(args.hidden_size, 1)

    def stock(x):
        return Qwen3NextSparseMoeBlock.__call__(block, x)

    def forward(x):
        return block(x)

    eager, compiled = _eager_and_compiled(forward, x)
    _, stock_compiled = _eager_and_compiled(stock, x)
    assert eager.dtype == mx.bfloat16
    assert mx.array_equal(compiled, eager).item()
    assert mx.array_equal(compiled, stock_compiled).item()


def test_the_verify_forward_is_mlx_lms_forward_away_from_the_gate_value():
    # The verify forward restates mlx_lm's Qwen3NextSparseMoeBlock forward with
    # only the gate's lowering changed. On gates other than -6.84375 the two
    # lowerings agree, so the whole block must match mlx_lm's forward bit for
    # bit; a change in mlx_lm's routing that the restatement missed fails here.
    from mlx_lm.models.qwen3_next import Qwen3NextSparseMoeBlock

    block, args = _moe_block()
    x = _rows(args.hidden_size, 2) * 4.0
    gates = block.shared_expert_gate(x)
    mx.eval(gates)
    assert not mx.any(gates == GATE).item()
    with attention_phase("decode_verify"):
        verify = block(x)
        stock = Qwen3NextSparseMoeBlock.__call__(block, x)
        mx.eval(verify, stock)
    assert mx.array_equal(verify, stock).item()
