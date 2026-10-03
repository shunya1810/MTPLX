"""Every routed row of a wide Flash-Next MoE forward is computed.

Flash-Next routes ten rows per token through ``gather_qmm(sorted_indices=True)``
twice per layer.  On a tensor-unit GPU with MLX 0.32.2 that kernel leaves the
leading rows of a call unwritten once a call carries more than 32,767 rows that
are not a multiple of its row tile (mtplx/moe_sorted_gather.py): a 3,404-row
prefill forward (34,040 routed rows, the last chunk of a 7,500-token prompt on
the 4,096-row plan) lost rows 0-1,279 of both gathers.  These tests run the
model's own expert module at that width and check every row against a float64
reference; the output buffers are pre-filled with NaN so an unwritten row
cannot pass by inheriting a correct value from an earlier computation.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest
from mlx_lm.models.switch_layers import QuantizedSwitchLinear

from mtplx.models import qwen4_exp

EXPERTS = 64
HIDDEN = 256
INTER = 128
TOP_K = 10
GROUP = 32
BITS = 4


def _switch(seed: int = 0):
    mx.random.seed(seed)
    gate_up = (mx.random.normal((EXPERTS, 2 * INTER, HIDDEN)) * 0.05).astype(mx.bfloat16)
    gu_w, gu_s, gu_b = mx.quantize(gate_up, group_size=GROUP, bits=BITS)
    down = QuantizedSwitchLinear(
        INTER, HIDDEN, EXPERTS, bias=False, group_size=GROUP, bits=BITS
    )
    dn = (mx.random.normal((EXPERTS, HIDDEN, INTER)) * 0.05).astype(mx.bfloat16)
    down.weight, down.scales, down.biases = mx.quantize(dn, group_size=GROUP, bits=BITS)
    switch = qwen4_exp._FusedGateUpSwitchGLU(down, gu_w, gu_s, gu_b, GROUP, BITS, "affine")
    mx.eval(switch.parameters())
    return switch


def _dequantized(switch):
    gu = mx.dequantize(
        switch.gu_weight, switch.gu_scales, switch.gu_biases, group_size=GROUP, bits=BITS
    )
    dn = switch.down_proj
    down = mx.dequantize(dn.weight, dn.scales, dn.biases, group_size=GROUP, bits=BITS)
    return (
        np.array(gu.astype(mx.float32)).astype(np.float64),
        np.array(down.astype(mx.float32)).astype(np.float64),
    )


def _tokens(rows: int, seed: int = 1):
    mx.random.seed(seed)
    x = (mx.random.normal((1, rows, HIDDEN)) * 0.5).astype(mx.bfloat16)
    # TOP_K distinct experts per token, as the router picks them.
    inds = mx.argsort(mx.random.uniform(shape=(rows, EXPERTS)), axis=-1)[:, :TOP_K]
    inds = inds.astype(mx.uint32).reshape(1, rows, TOP_K)
    mx.eval(x, inds)
    return x, inds


def _reference(switch, x, inds) -> np.ndarray:
    """[rows, TOP_K, HIDDEN]: silu(gate) * up through down, per routed row, float64."""

    gu, down = _dequantized(switch)
    xs = np.array(x.astype(mx.float32)).astype(np.float64)[0]
    ids = np.array(inds)[0]
    out = np.zeros((ids.shape[0], TOP_K, HIDDEN))
    for e in np.unique(ids):
        r, s = np.nonzero(ids == e)
        h = xs[r] @ gu[e].T
        g, u = h[:, :INTER], h[:, INTER:]
        out[r, s] = (g / (1.0 + np.exp(-g)) * u) @ down[e].T
    return out


def _poison_allocator(nbytes: int, count: int = 6) -> None:
    """Park NaN-filled buffers of the output size in MLX's buffer cache."""

    mx.synchronize()
    mx.clear_cache()
    parked = [mx.full((nbytes // 2,), float("nan"), dtype=mx.bfloat16) for _ in range(count)]
    mx.eval(*parked)
    del parked


def _bad_rows(actual: np.ndarray, reference: np.ndarray, tol: float = 0.05) -> np.ndarray:
    """Routed rows whose worst error exceeds ``tol`` of their largest value
    (bf16 rounding of the two intermediates stays near 1%; an unwritten row is
    NaN, zero or someone else's data)."""

    err = np.abs(actual - reference).max(axis=-1)
    scale = np.abs(reference).max(axis=-1) + 1e-6
    ok = np.isfinite(actual).all(axis=-1) & (err / scale <= tol)
    return ~ok


# 3,404 rows is the 7,500-token prompt's last chunk; 4,095 the widest
# unaligned forward of the 4,096-row plan (40,950 routed rows); 3,277 the
# narrowest one past the bound (32,770 routed rows).
WIDTHS = [3404, 4095, 3277]


@pytest.mark.parametrize("rows", WIDTHS)
def test_prefill_sorted_experts_computes_every_routed_row(rows):
    switch = _switch()
    x, inds = _tokens(rows)
    reference = _reference(switch, x, inds)
    _poison_allocator(rows * TOP_K * HIDDEN * 2)
    y_sorted, inv_order = switch.sorted_experts(x, inds)
    y = y_sorted[inv_order].reshape(rows, TOP_K, HIDDEN)
    mx.eval(y)
    assert tuple(y_sorted.shape) == (rows * TOP_K, HIDDEN)
    bad = _bad_rows(np.array(y.astype(mx.float32)).astype(np.float64), reference)
    assert int(bad.sum()) == 0, (
        f"{int(bad.sum())} of {rows * TOP_K} routed rows wrong at {rows} rows"
    )


@pytest.mark.parametrize("rows", WIDTHS)
def test_block_forward_computes_every_routed_row(rows):
    """The same module through ``__call__`` (the path a block takes when the
    fused combine does not apply, for example an fp16 pack)."""

    switch = _switch(seed=3)
    x, inds = _tokens(rows, seed=4)
    reference = _reference(switch, x, inds)
    _poison_allocator(rows * TOP_K * HIDDEN * 2)
    y = switch(x, inds)
    mx.eval(y)
    assert tuple(y.shape) == (1, rows, TOP_K, HIDDEN)
    bad = _bad_rows(np.array(y.astype(mx.float32)).astype(np.float64)[0], reference)
    assert int(bad.sum()) == 0, (
        f"{int(bad.sum())} of {rows * TOP_K} routed rows wrong at {rows} rows"
    )


def test_aligned_and_narrow_widths_stay_correct():
    """4,096 rows (aligned) and 2,048 rows (under the bound) were never
    affected; they must stay exact through the same module."""

    switch = _switch(seed=5)
    for rows in (4096, 2048):
        x, inds = _tokens(rows, seed=rows)
        reference = _reference(switch, x, inds)
        y_sorted, inv_order = switch.sorted_experts(x, inds)
        y = y_sorted[inv_order].reshape(rows, TOP_K, HIDDEN)
        mx.eval(y)
        bad = _bad_rows(np.array(y.astype(mx.float32)).astype(np.float64), reference)
        assert int(bad.sum()) == 0
