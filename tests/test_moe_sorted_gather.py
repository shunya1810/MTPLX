"""The sorted-rows guard around MLX's expert-sorted quantized gather
(mtplx/moe_sorted_gather.py).

The guard pads an affected call to a multiple of 64 rows, runs it once and
slices the real rows back.  These tests pin: when it engages (and that it
engages only there), that the real rows are bit-identical to an unpadded call
wherever that call is correct, that the rows the unpadded call drops come back
right, and that every entry point (the direct op, a switch-linear module, mlx-lm's
switch layers, the packed and Laguna projections) keeps it.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx_lm.models import switch_layers
from mlx_lm.models.switch_layers import QuantizedSwitchLinear, SwitchGLU

from mtplx import moe_sorted_gather as msg
from mtplx import nax_detect

HIDDEN = 256
OUT = 256


@pytest.fixture
def guard_on(monkeypatch):
    """Pretend the affected kernel is present (pure-logic tests)."""

    monkeypatch.setattr(msg, "guard_active", lambda: True)


def _bits(a: mx.array) -> np.ndarray:
    mx.eval(a)
    view = mx.uint16 if a.dtype in (mx.bfloat16, mx.float16) else mx.uint32
    return np.array(a.view(view))


def _weights(experts=64, n=OUT, k=HIDDEN, group_size=32, bits=4, dtype=mx.bfloat16, seed=0):
    mx.random.seed(seed)
    w = (mx.random.normal((experts, n, k)) * 0.05).astype(dtype)
    wq, s, b = mx.quantize(w, group_size=group_size, bits=bits)
    mx.eval(wq, s, b)
    return wq, s, b


def _sorted_rows(rows, experts=64, k=HIDDEN, dtype=mx.bfloat16, seed=1):
    mx.random.seed(seed)
    x = (mx.random.normal((rows, 1, k)) * 0.5).astype(dtype)
    idx = mx.sort(mx.random.randint(0, experts, (rows,))).astype(mx.uint32)
    mx.eval(x, idx)
    return x, idx


def _reference(x, idx, wq, s, b, group_size=32, bits=4) -> np.ndarray:
    w = np.array(mx.dequantize(wq, s, b, group_size=group_size, bits=bits).astype(mx.float32))
    xs = np.array(x.astype(mx.float32))[:, 0, :].astype(np.float64)
    ids = np.array(idx)
    out = np.zeros((xs.shape[0], w.shape[1]))
    for e in np.unique(ids):
        rows = np.nonzero(ids == e)[0]
        out[rows] = xs[rows] @ w[e].astype(np.float64).T
    return out


def _bad_rows(y: mx.array, reference: np.ndarray, tol: float = 0.05) -> np.ndarray:
    actual = np.array(y.astype(mx.float32)).reshape(reference.shape).astype(np.float64)
    err = np.abs(actual - reference).max(axis=-1)
    scale = np.abs(reference).max(axis=-1) + 1e-6
    return ~(np.isfinite(actual).all(axis=-1) & (err / scale <= tol))


def _poison_allocator(nbytes: int, count: int = 6) -> None:
    mx.synchronize()
    mx.clear_cache()
    parked = [mx.full((nbytes // 2,), float("nan"), dtype=mx.bfloat16) for _ in range(count)]
    mx.eval(*parked)
    del parked


def _stock(x, idx, wq, s, b, group_size=32, bits=4):
    return mx.gather_qmm(
        x, wq, s, b, rhs_indices=idx, transpose=True,
        group_size=group_size, bits=bits, sorted_indices=True,
    )


# --- when the guard engages -------------------------------------------------


@pytest.mark.parametrize(
    "rows,pad",
    [(1, 0), (32767, 0), (32768, 0), (32769, 63), (32770, 62), (34040, 8),
     (40950, 10), (40960, 0), (65535, 1), (65537, 63), (81910, 10)],
)
def test_pad_rule(guard_on, rows, pad):
    assert msg.pad_rows(rows) == pad
    assert (rows + pad) % msg.ROW_ALIGN == 0 or pad == 0


@pytest.mark.parametrize(
    "version,affected",
    [("0.32.2", True), ("0.31.2", True), ("0.32.3", False), ("0.32.4", False),
     ("0.33.0", False), ("1.0.0", False), ("0.32.3.dev20260929", True),
     ("0.32.3rc1", True), ("0.32.3+local", False), ("v0.32.2", True),
     ("unknown", True)],
)
def test_mlx_release_rule(version, affected):
    assert msg.mlx_release_affected(version) is affected


def test_no_guard_without_the_tensor_unit_kernel(monkeypatch):
    monkeypatch.setattr(nax_detect, "nax_hardware_available", lambda: False)
    msg.guard_active.cache_clear()
    try:
        assert msg.guard_active() is False
        assert msg.pad_rows(34040) == 0
    finally:
        msg.guard_active.cache_clear()


def test_no_guard_on_a_fixed_mlx(monkeypatch):
    monkeypatch.setattr(nax_detect, "nax_hardware_available", lambda: True)
    monkeypatch.setattr(msg.mx, "__version__", "0.32.3")
    msg.guard_active.cache_clear()
    try:
        assert msg.guard_active() is False
        assert msg.pad_rows(34040) == 0
    finally:
        msg.guard_active.cache_clear()


def test_the_route_switch_does_not_hide_mlx_kernel(monkeypatch):
    """MTPLX_FORCE_GPU_FAMILY_FALLBACK sends our routes down the M1-M4 path but
    MLX still runs its tensor-unit kernel, so the guard must not follow it."""

    before = msg.guard_active()
    monkeypatch.setenv("MTPLX_FORCE_GPU_FAMILY_FALLBACK", "1")
    msg.guard_active.cache_clear()
    try:
        assert nax_detect.nax_available() is False
        assert msg.guard_active() is before
    finally:
        msg.guard_active.cache_clear()


def test_only_affine_transposed_sorted_rows_are_padded(guard_on):
    x = mx.zeros((34040, 1, 64), dtype=mx.bfloat16)
    idx = mx.zeros((34040,), dtype=mx.uint32)
    call = msg._call_pad
    assert call(x, idx, None, sorted_indices=True, transpose=True, mode="affine") == 8
    assert call(x, idx, None, sorted_indices=False, transpose=True, mode="affine") == 0
    assert call(x, idx, None, sorted_indices=True, transpose=False, mode="affine") == 0
    assert call(x, idx, None, sorted_indices=True, transpose=True, mode="mxfp4") == 0
    assert call(x, idx, None, sorted_indices=True, transpose=True, mode=None) == 0
    assert call(x, idx.reshape(2, -1), None, sorted_indices=True, transpose=True, mode="affine") == 0
    assert call(x[:-1], idx, None, sorted_indices=True, transpose=True, mode="affine") == 0
    lhs = mx.arange(34040, dtype=mx.uint32)
    assert call(x, idx, lhs, sorted_indices=True, transpose=True, mode="affine") == 8
    assert call(x, idx, lhs[:-1], sorted_indices=True, transpose=True, mode="affine") == 0


# --- exactness ----------------------------------------------------------------


@pytest.mark.parametrize("rows", [34040, 40950, 32770])
def test_guarded_rows_equal_the_stock_rows_it_computes_and_fix_the_rest(rows):
    """Bit-identical to the unpadded call on every row that call writes; the
    rows it leaves unwritten (MLX 0.32.2 on a tensor-unit GPU) are right."""

    wq, s, b = _weights()
    x, idx = _sorted_rows(rows)
    reference = _reference(x, idx, wq, s, b)
    _poison_allocator(rows * OUT * 2)
    stock = _stock(x, idx, wq, s, b)
    mx.eval(stock)
    stock_bad = _bad_rows(stock, reference)
    guarded = msg.gather_qmm(
        x, wq, s, b, rhs_indices=idx, transpose=True,
        group_size=32, bits=4, sorted_indices=True,
    )
    mx.eval(guarded)
    assert tuple(guarded.shape) == tuple(stock.shape)
    assert not _bad_rows(guarded, reference).any()
    good = ~stock_bad
    assert np.array_equal(_bits(guarded)[good], _bits(stock)[good])
    if msg.guard_active():
        # The rows the stock kernel drops: every 32-row simdgroup block that
        # starts at or before rows - 32,768.
        dropped = ((rows - 32768) // 32 + 1) * 32
        assert np.nonzero(stock_bad)[0].tolist() == list(range(dropped))


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("group_size,bits", [(32, 4), (64, 4), (64, 8)])
@pytest.mark.parametrize("rows,experts", [(1000, 64), (4090, 64), (32700, 512)])
def test_a_padded_call_is_bit_identical_where_the_stock_call_is_correct(
    monkeypatch, guard_on, dtype, group_size, bits, rows, experts
):
    """Force the pad below the bound, where the stock call is correct.  4,090
    rows over 64 experts pads across MLX's 32-row/64-row tile switch."""

    monkeypatch.setattr(msg, "INT16_ROW_BOUND", 0)
    assert msg.pad_rows(rows) > 0
    wq, s, b = _weights(experts=experts, group_size=group_size, bits=bits, dtype=dtype)
    x, idx = _sorted_rows(rows, experts=experts, dtype=dtype)
    stock = _stock(x, idx, wq, s, b, group_size, bits)
    guarded = msg.gather_qmm(
        x, wq, s, b, rhs_indices=idx, transpose=True,
        group_size=group_size, bits=bits, sorted_indices=True,
    )
    assert tuple(guarded.shape) == tuple(stock.shape)
    assert np.array_equal(_bits(guarded), _bits(stock))


def test_unsorted_calls_pass_through_unchanged(guard_on):
    wq, s, b = _weights()
    mx.random.seed(2)
    x = mx.random.normal((3, 4, 1, 1, HIDDEN)).astype(mx.bfloat16)
    idx = mx.random.randint(0, 64, (3, 4, 10)).astype(mx.uint32)
    stock = mx.gather_qmm(x, wq, s, b, rhs_indices=idx, transpose=True, group_size=32, bits=4)
    ours = msg.gather_qmm(x, wq, s, b, rhs_indices=idx, transpose=True, group_size=32, bits=4)
    assert np.array_equal(_bits(ours), _bits(stock))


def test_padded_calls_are_counted(monkeypatch, guard_on):
    monkeypatch.setattr(msg, "_STATS", {"padded_calls": 0, "padded_rows": 0})
    wq, s, b = _weights()
    x, idx = _sorted_rows(34040)
    mx.eval(msg.gather_qmm(x, wq, s, b, rhs_indices=idx, group_size=32, bits=4, sorted_indices=True))
    assert msg.stats() == {"padded_calls": 1, "padded_rows": 8}


# --- entry points ---------------------------------------------------------------


def _quantized_switch_linear(n_in, n_out, experts=64, seed=0):
    layer = QuantizedSwitchLinear(n_in, n_out, experts, bias=False, group_size=32, bits=4)
    wq, s, b = _weights(experts=experts, n=n_out, k=n_in, seed=seed)
    layer.weight, layer.scales, layer.biases = wq, s, b
    return layer


def test_switch_linear_module_call_is_guarded():
    layer = _quantized_switch_linear(HIDDEN, OUT)
    x, idx = _sorted_rows(34040)
    reference = _reference(x, idx, layer.weight, layer.scales, layer.biases)
    _poison_allocator(34040 * OUT * 2)
    y = msg.switch_linear(layer, x, idx, sorted_indices=True)
    mx.eval(y)
    assert tuple(y.shape) == (34040, 1, OUT)
    assert not _bad_rows(y, reference).any()


def test_mlx_lm_switch_layers_are_guarded_once_installed(monkeypatch):
    if not msg.guard_active():
        assert msg.install_switch_linear_guard() is False
        pytest.skip("MLX's sorted kernel is correct here: nothing to install")
    # Restore mlx-lm's class after the test, whatever the install does.
    monkeypatch.setattr(
        QuantizedSwitchLinear, "__call__", QuantizedSwitchLinear.__call__
    )
    assert msg.install_switch_linear_guard() is True
    installed = QuantizedSwitchLinear.__call__
    assert msg.install_switch_linear_guard() is True
    assert QuantizedSwitchLinear.__call__ is installed  # idempotent, no double wrap

    # A top-8 family: 4,255 tokens route 34,040 rows through three gathers.
    mx.random.seed(7)
    glu = SwitchGLU(HIDDEN, 128, 64)
    nn.quantize(glu, group_size=32, bits=4)
    for proj in (glu.gate_proj, glu.up_proj, glu.down_proj):
        assert isinstance(proj, QuantizedSwitchLinear)
    mx.eval(glu.parameters())
    rows, top_k = 4255, 8
    x = (mx.random.normal((1, rows, HIDDEN)) * 0.5).astype(mx.bfloat16)
    inds = mx.argsort(mx.random.uniform(shape=(rows, 64)), axis=-1)[:, :top_k]
    inds = inds.astype(mx.uint32).reshape(1, rows, top_k)

    def dq(proj):
        w = mx.dequantize(proj.weight, proj.scales, proj.biases, group_size=32, bits=4)
        return np.array(w.astype(mx.float32)).astype(np.float64)

    gate, up, down = dq(glu.gate_proj), dq(glu.up_proj), dq(glu.down_proj)
    xs = np.array(x.astype(mx.float32)).astype(np.float64)[0]
    ids = np.array(inds)[0]
    reference = np.zeros((rows, top_k, HIDDEN))
    for e in np.unique(ids):
        r, c = np.nonzero(ids == e)
        g, u = xs[r] @ gate[e].T, xs[r] @ up[e].T
        reference[r, c] = (g / (1.0 + np.exp(-g)) * u) @ down[e].T
    _poison_allocator(rows * top_k * HIDDEN * 2)
    y = glu(x, inds)
    mx.eval(y)
    assert tuple(y.shape) == (1, rows, top_k, HIDDEN)
    assert not _bad_rows(y[0], reference).any()


def test_packed_projection_gather_is_guarded():
    from mtplx.moe_packed_projections import _PackedQuantizedProjection

    wq, s, b = _weights()
    proj = _PackedQuantizedProjection(wq, s, b, group_size=32, bits=4, mode="affine")
    x, idx = _sorted_rows(34040)
    reference = _reference(x, idx, wq, s, b)
    _poison_allocator(34040 * OUT * 2)
    y = proj.gather(x, idx, True)
    mx.eval(y)
    assert not _bad_rows(y, reference).any()


def test_laguna_patched_switch_call_is_guarded_with_its_explicit_lhs():
    from mtplx.models import laguna_fused

    layer = _quantized_switch_linear(HIDDEN, OUT)
    x, idx = _sorted_rows(34040)
    reference = _reference(x, idx, layer.weight, layer.scales, layer.biases)
    _poison_allocator(34040 * OUT * 2)
    y = laguna_fused._patched_quantized_switch_call(layer, x, idx, sorted_indices=True)
    mx.eval(y)
    assert tuple(y.shape) == (34040, 1, OUT)
    assert not _bad_rows(y, reference).any()


def test_install_is_a_no_op_where_the_kernel_is_correct(monkeypatch):
    monkeypatch.setattr(msg, "guard_active", lambda: False)
    before = switch_layers.QuantizedSwitchLinear.__call__
    assert msg.install_switch_linear_guard() is False
    assert switch_layers.QuantizedSwitchLinear.__call__ is before


def test_no_sorted_gather_in_mtplx_bypasses_the_entry_point():
    """Every call site that serves a prefill (cold chunks, warm suffixes,
    image embeddings, batched forwards) reaches MLX's sorted gather through
    ``mtplx.moe_sorted_gather`` or mlx-lm's switch layers, which the loader
    guards.  A direct ``mx.gather_qmm`` with sorted indices anywhere else in
    the package would skip the row guard; only literal unsorted calls may
    stay direct."""

    import ast
    from pathlib import Path

    import mtplx

    root = Path(mtplx.__file__).resolve().parent
    offenders = []
    for path in sorted(root.rglob("*.py")):
        if path.name == "moe_sorted_gather.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "gather_qmm"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "mx"
            ):
                continue
            flag = next((kw.value for kw in node.keywords if kw.arg == "sorted_indices"), None)
            if flag is None or (isinstance(flag, ast.Constant) and flag.value is False):
                continue
            offenders.append(f"{path.relative_to(root)}:{node.lineno}")
    assert offenders == []
