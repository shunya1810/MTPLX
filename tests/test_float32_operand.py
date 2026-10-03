"""The MLX property mtplx/float32_operand.py rests on, checked, not trusted.

MLX 0.32.2 (and main as of 2026-09) writes an input of a fused kernel into the
kernel's source when it is a one-element array without a primitive, and
writes a float32 with 7 significant digits. The helper hands out a fresh
slice of an evaluated two-element buffer: a one-element array WITH a
primitive, which a kernel must read from memory. If a later MLX inlined that
slice too, the compiled division below would round sqrt(128) to 11.31371 and
tie the pair that the eager division keeps apart, and this test would fail.
"""

from __future__ import annotations

import io
import math

import mlx.core as mx
import numpy as np
import pytest

from mtplx.float32_operand import float32_operand

PAIR = (1.5 + 2.0**-23, 1.5 + 2.0**-22)


def _has_primitive(array: mx.array) -> bool:
    dot = io.StringIO()
    mx.export_to_dot(dot, array)
    return "shape=rectangle" in dot.getvalue()


def test_the_operand_is_the_float32_value_and_carries_a_primitive():
    operand = float32_operand(math.sqrt(128))
    assert operand.shape == (1,) and operand.dtype == mx.float32
    assert _has_primitive(operand)
    assert operand.item() == float(np.float32(math.sqrt(128)))
    # A fresh one at every call: an evaluated slice has no primitive left.
    mx.eval(operand)
    assert not _has_primitive(operand)
    assert _has_primitive(float32_operand(math.sqrt(128)))


@pytest.mark.skipif(not mx.metal.is_available(), reason="fused kernels are the GPU's")
def test_a_fused_division_by_the_operand_keeps_the_near_tie_apart():
    pair = mx.array(PAIR, dtype=mx.float32)
    mx.eval(pair)

    def scores(values):
        # The division joined to a fused elementwise op, as in a verify trace.
        return mx.maximum(values / float32_operand(math.sqrt(128)), 0.0)

    with mx.stream(mx.gpu):
        eager = scores(pair)
        compiled = mx.compile(scores)(pair)
        mx.eval(eager, compiled)
    assert mx.array_equal(compiled, eager).item()
    low, high = compiled.tolist()
    assert low < high
