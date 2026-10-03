"""The QSA rows-gather attention's float32 scores, compiled against eager.

``_qsa_rows_gather_attention`` scales float32 scores by the softmax scale
head_dim ** -0.5 inside the verify graph. As a Python float that is a
constant of the fused kernel, written with 7 significant digits: exact for
the shipping head size 256 (0.0625), not for 32 or 128. The scale is read
from memory now (mtplx/float32_operand.py). float32 queries and keys at head
size 32 keep the moved scale visible through the softmax.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from mtplx.models.qwen4_exp import _qsa_rows_gather_attention, _qsa_stock_rows_gather_kv

ROWS, SELECTED, CONTEXT, HEAD = 4, 12, 64, 32


@pytest.mark.skipif(not mx.metal.is_available(), reason="fused kernels are the GPU's")
@pytest.mark.parametrize("heads", [(4, 1), (2, 2)], ids=["gqa", "mha"])
def test_rows_gather_scores_compile_to_the_eager_values(heads):
    n_heads, n_kv = heads
    scale = HEAD**-0.5
    assert np.float32(float(f"{float(np.float32(scale)):.7g}")) != np.float32(scale)
    key = mx.random.key(3)
    q = mx.random.normal((1, n_heads, ROWS, HEAD), key=key) * 2.0
    k = mx.random.normal((1, n_kv, CONTEXT, HEAD), key=mx.random.key(4)) * 2.0
    v = mx.random.normal((1, n_kv, CONTEXT, HEAD), key=mx.random.key(5))
    token_idx = mx.array(
        np.stack([np.arange(r, r + SELECTED) for r in range(ROWS)]).astype(np.int32)
    )
    token_ok = mx.array(np.ones((ROWS, SELECTED), dtype=bool))
    mx.eval(q, k, v, token_idx, token_ok)

    def attend(q, k, v, token_idx, token_ok):
        return _qsa_rows_gather_attention(
            q, k, v, token_idx, token_ok, scale, _qsa_stock_rows_gather_kv
        )

    eager = attend(q, k, v, token_idx, token_ok)
    compiled = mx.compile(attend)(q, k, v, token_idx, token_ok)
    mx.eval(eager, compiled)
    assert eager.dtype == mx.float32
    assert mx.array_equal(compiled, eager).item()
