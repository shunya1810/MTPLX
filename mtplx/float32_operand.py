"""A float32 scalar operand that no compiled kernel writes into its source.

``mx.compile`` fuses runs of elementwise ops into one JIT kernel, and it
treats an input of such a kernel as a constant, written into the kernel's
source, when the input is a one-element array that has no primitive and is
not an input of the compiled function (``compile_fuse`` in mlx/compile.cpp,
MLX 0.32.2 and main as of 2026-09). MLX 0.32.2 writes a float32 constant
with ``std::numeric_limits<float>::digits10 + 1`` = 7 significant digits
(mlx/backend/common/compiled.h), and a float32 needs 9 to read back exactly:
sqrt(128) = 11.3137083 comes back as 11.31371 and a YaRN amplitude of
1.1386294 as 1.1386290. A Python float qualifies, and so does an evaluated
one-element array. A slice of an evaluated two-element buffer is a
one-element array WITH a primitive (a Slice, which no fused kernel absorbs),
so a kernel reads it from memory: the float32 that the eager op uses.

Take a fresh operand at every use and never keep one: once evaluated, a
slice drops its primitive and would qualify as a constant again.
tests/test_float32_operand.py checks the property against the installed MLX.
"""

from __future__ import annotations

import mlx.core as mx

_BUFFERS: dict[float, mx.array] = {}


def float32_operand(value: float) -> mx.array:
    """float32(value) as a one-element operand a fused kernel reads from memory."""

    key = float(value)
    buffer = _BUFFERS.get(key)
    if buffer is None:
        # Built from host data: evaluated from the start, with no primitive.
        buffer = mx.array([key, key], dtype=mx.float32)
        _BUFFERS[key] = buffer
    return buffer[:1]
