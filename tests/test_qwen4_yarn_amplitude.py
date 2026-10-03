"""Static-YaRN rotary tables: the compiled verifier's values are the eager ones.

Flash-Next multiplies its float32 cos/sin tables by the YaRN amplitude
0.1 * ln(factor) + 1 inside the verify graph. At factor 4 that is 1.1386294,
which a fused kernel's 7-significant-digit constant would move to 1.1386290
(mtplx/float32_operand.py); the tables now read it from memory.
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest


@pytest.mark.parametrize("half", [False, True], ids=["full-table", "half-table"])
def test_the_yarn_rotary_tables_compile_to_the_eager_values(half):
    from mtplx.models.qwen4_exp import _rope_cos_sin, _rope_cos_sin_half

    tables = _rope_cos_sin_half if half else _rope_cos_sin
    scaling = 0.1 * math.log(4.0) + 1.0
    inv_freq = mx.array(1.0 / (10000.0 ** (np.arange(0, 16, 2) / 16.0)), dtype=mx.float32)
    positions = mx.arange(0, 4096, 7, dtype=mx.int32)
    mx.eval(inv_freq, positions)

    def run(positions):
        cos, sin = tables(positions, inv_freq, scaling)
        return cos, sin

    eager = run(positions)
    compiled = mx.compile(run)(positions)
    mx.eval(eager, compiled)
    for got, want in zip(compiled, eager):
        assert mx.array_equal(got, want).item()
