"""Flash-Next's routed gate/up gather reading token rows in place
(mtplx/kernels/moe_sorted_gather_nax.py).

Class A: every output row must be bit-identical to ``mx.gather_qmm(tokens[row_map],
..., sorted_indices=True)`` (the copy plus stock gather the model ran before),
and with the SwiGLU epilogue to that gather's split and ``nn.silu(gate) * up``,
wherever the stock kernel is correct, and right where it is not (past 32,767
unaligned rows on MLX 0.32.2).  The kernel runs only on tensor-unit GPUs;
everywhere else these tests check that the stock path runs.
"""

from __future__ import annotations

import os
import shutil

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx_lm.models.switch_layers import QuantizedSwitchLinear

from mtplx import moe_sorted_gather as msg
from mtplx import nax_detect
from mtplx.kernels import moe_sorted_gather_nax as nax_gather

needs_tensor_units = pytest.mark.skipif(
    not nax_detect.nax_available() or nax_gather._mlx_headers() is None,
    reason="the kernel runs only on tensor-unit GPUs with MLX's kernel headers installed",
)


def _bits(a: mx.array) -> np.ndarray:
    mx.eval(a)
    return np.array(a.view(mx.uint16))


def _weights(experts, n, k, group_size, bits, dtype, seed=0):
    mx.random.seed(seed)
    w = (mx.random.normal((experts, n, k)) * 0.05).astype(dtype)
    wq, s, b = mx.quantize(w, group_size=group_size, bits=bits)
    mx.eval(wq, s, b)
    return wq, s, b


def _routed(tokens, experts, top_k, k, dtype, seed):
    """Tokens and their expert-sorted routing, the way the model sorts them."""

    mx.random.seed(seed)
    x = (mx.random.normal((tokens, k)) * 0.5).astype(dtype)
    inds = mx.argsort(mx.random.uniform(shape=(tokens, experts)), axis=-1)[:, :top_k]
    tok, row_map, idx, _inv = msg.sort_rows(x, inds.astype(mx.uint32))
    mx.eval(tok, row_map, idx)
    return tok, row_map, idx


def _stock(tokens, row_map, wq, s, b, idx, group_size, bits):
    return mx.gather_qmm(
        tokens[row_map], wq, s, b, rhs_indices=idx, transpose=True,
        group_size=group_size, bits=bits, sorted_indices=True,
    )


def _stock_swiglu(tokens, row_map, wq, s, b, idx, group_size, bits):
    """The chain Flash-Next's expert module ran before the epilogue."""

    gate, up = mx.split(_stock(tokens, row_map, wq, s, b, idx, group_size, bits), 2, axis=-1)
    return nn.silu(gate) * up


@pytest.fixture(autouse=True)
def _fresh_canaries(monkeypatch):
    monkeypatch.setattr(nax_gather, "_CANARY", {})
    monkeypatch.setattr(
        nax_gather,
        "_STATS",
        {"calls": 0, "fallbacks": 0, "canaries": 0, "canary_failures": 0, "header_failures": 0},
    )


@needs_tensor_units
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("group_size,bits", [(32, 4), (64, 4), (64, 8), (128, 4)])
@pytest.mark.parametrize("n,k", [(128, 256), (256, 128)])
def test_bit_identical_to_copy_plus_stock_gather(dtype, group_size, bits, n, k):
    tok, row_map, idx = _routed(500, 64, 10, k, dtype, seed=3)
    wq, s, b = _weights(64, n, k, group_size, bits, dtype)
    ours = nax_gather.gather_rows_qmm(tok, row_map, wq, s, b, idx, group_size=group_size, bits=bits)
    assert ours is not None, nax_gather.stats()
    stock = _stock(tok, row_map, wq, s, b, idx, group_size, bits)
    assert tuple(ours.shape) == tuple(stock.shape) == (5000, 1, n)
    assert np.array_equal(_bits(ours), _bits(stock))
    assert nax_gather.stats()["canary_failures"] == 0


@needs_tensor_units
def test_ragged_runs_empty_experts_and_tile_edges():
    """Runs of 0, 1, 63, 64, 65, 127, 128 and 129 rows: tiles end inside and at
    the edge of an expert's run, and empty experts own no tile."""

    counts = [0, 1, 63, 64, 65, 0, 127, 128, 129, 0, 2000, 1, 0, 3000]
    counts += [0] * (64 - len(counts))
    idx = mx.array(np.repeat(np.arange(64, dtype=np.uint32), counts))
    rows = int(idx.shape[0])
    mx.random.seed(6)
    tok = (mx.random.normal((900, 1, 256)) * 0.5).astype(mx.bfloat16)
    row_map = mx.random.randint(0, 900, (rows,)).astype(mx.uint32)
    wq, s, b = _weights(64, 128, 256, 32, 4, mx.bfloat16, seed=5)
    ours = nax_gather.gather_rows_qmm(tok, row_map, wq, s, b, idx, group_size=32, bits=4)
    assert ours is not None
    assert np.array_equal(_bits(ours), _bits(_stock(tok, row_map, wq, s, b, idx, 32, 4)))


@needs_tensor_units
@pytest.mark.parametrize("tokens", [3404, 4095])
def test_no_row_bound(tokens):
    """Past 32,767 unaligned rows: equal, bit for bit, to the padded stock call
    (correct, and equal to the stock call wherever that is correct)."""

    tok, row_map, idx = _routed(tokens, 64, 10, 256, mx.bfloat16, seed=9)
    rows = int(idx.shape[0])
    wq, s, b = _weights(64, 128, 256, 32, 4, mx.bfloat16, seed=7)
    ours = nax_gather.gather_rows_qmm(tok, row_map, wq, s, b, idx, group_size=32, bits=4)
    assert ours is not None
    pad = 64 - rows % 64
    xp = mx.concatenate([tok[row_map], mx.zeros((pad, 1, 256), dtype=tok.dtype)])
    ip = mx.concatenate([idx, mx.broadcast_to(idx[-1:], (pad,))])
    padded = mx.gather_qmm(
        xp, wq, s, b, rhs_indices=ip, transpose=True, group_size=32, bits=4, sorted_indices=True
    )[:rows]
    assert np.array_equal(_bits(ours), _bits(padded))


@needs_tensor_units
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("group_size,bits", [(32, 4), (64, 4), (64, 8), (128, 4)])
@pytest.mark.parametrize("n,k", [(128, 256), (256, 128)])
def test_swiglu_bit_identical_to_the_stock_chain(dtype, group_size, bits, n, k):
    tok, row_map, idx = _routed(500, 64, 10, k, dtype, seed=4)
    wq, s, b = _weights(64, n, k, group_size, bits, dtype, seed=2)
    ours = nax_gather.gather_rows_qmm(
        tok, row_map, wq, s, b, idx, group_size=group_size, bits=bits, swiglu=True
    )
    assert ours is not None, nax_gather.stats()
    stock = _stock_swiglu(tok, row_map, wq, s, b, idx, group_size, bits)
    assert tuple(ours.shape) == tuple(stock.shape) == (5000, 1, n // 2)
    assert np.array_equal(_bits(ours), _bits(stock))
    assert nax_gather.stats()["canary_failures"] == 0


@needs_tensor_units
def test_swiglu_ragged_runs_empty_experts_and_tile_edges():
    counts = [0, 1, 63, 64, 65, 0, 127, 128, 129, 0, 2000, 1, 0, 3000]
    counts += [0] * (64 - len(counts))
    idx = mx.array(np.repeat(np.arange(64, dtype=np.uint32), counts))
    rows = int(idx.shape[0])
    mx.random.seed(16)
    tok = (mx.random.normal((900, 1, 256)) * 0.5).astype(mx.bfloat16)
    row_map = mx.random.randint(0, 900, (rows,)).astype(mx.uint32)
    wq, s, b = _weights(64, 192, 256, 32, 4, mx.bfloat16, seed=15)
    ours = nax_gather.gather_rows_qmm(tok, row_map, wq, s, b, idx, group_size=32, bits=4, swiglu=True)
    assert ours is not None
    stock = _stock_swiglu(tok, row_map, wq, s, b, idx, 32, 4)
    assert np.array_equal(_bits(ours), _bits(stock))


@needs_tensor_units
@pytest.mark.parametrize("tokens", [3404, 4095])
def test_swiglu_no_row_bound(tokens):
    tok, row_map, idx = _routed(tokens, 64, 10, 256, mx.bfloat16, seed=19)
    rows = int(idx.shape[0])
    wq, s, b = _weights(64, 128, 256, 32, 4, mx.bfloat16, seed=17)
    ours = nax_gather.gather_rows_qmm(tok, row_map, wq, s, b, idx, group_size=32, bits=4, swiglu=True)
    assert ours is not None
    pad = 64 - rows % 64
    xp = mx.concatenate([tok[row_map], mx.zeros((pad, 1, 256), dtype=tok.dtype)])
    ip = mx.concatenate([idx, mx.broadcast_to(idx[-1:], (pad,))])
    gu = mx.gather_qmm(
        xp, wq, s, b, rhs_indices=ip, transpose=True, group_size=32, bits=4, sorted_indices=True
    )[:rows]
    gate, up = mx.split(gu, 2, axis=-1)
    assert np.array_equal(_bits(ours), _bits(nn.silu(gate) * up))


def test_short_chunks_follow_the_installed_mlx():
    """MLX before 0.32.3 tiles rows across experts and ours wins from 4,096
    rows; MLX 0.32.3 tiles per expert and keeps short chunks."""

    assert nax_gather.min_rows("0.32.2") == 4096
    assert nax_gather.min_rows("0.32.3.dev20260920") == 4096
    assert nax_gather.min_rows("0.32.3") == 16384
    assert nax_gather.min_rows("0.32.4") == 16384
    assert nax_gather.min_rows() == nax_gather.min_rows(mx.__version__)


def test_small_widths_and_other_layouts_keep_the_stock_kernel():
    wq, s, b = _weights(64, 128, 256, 32, 4, mx.bfloat16)
    rows = nax_gather.min_rows()
    tok = mx.zeros((rows, 1, 256), dtype=mx.bfloat16)
    idx = mx.zeros((rows,), dtype=mx.uint32)
    kw = dict(group_size=32, bits=4, mode="affine")
    small = mx.zeros((rows - 1,), dtype=mx.uint32)
    assert not nax_gather.applies(tok, small, wq, s, b, small, **kw)
    assert not nax_gather.applies(tok, idx, wq, s, b, idx, group_size=32, bits=4, mode="mxfp4")
    assert not nax_gather.applies(tok.astype(mx.float32), idx, wq, s, b, idx, **kw)
    assert not nax_gather.applies(tok, idx.astype(mx.int32), wq, s, b, idx, **kw)
    assert not nax_gather.applies(tok, idx, wq, s, None, idx, **kw)
    assert not nax_gather.applies(tok, idx, wq, s.astype(mx.float16), b, idx, **kw)
    assert not nax_gather.applies(tok.reshape(rows, 256), idx, wq, s, b, idx, **kw)


def test_rehearsal_switch_and_kill_switch_keep_the_stock_kernel(monkeypatch):
    wq, s, b = _weights(64, 128, 256, 32, 4, mx.bfloat16)
    rows = nax_gather.min_rows()
    tok = mx.zeros((rows, 1, 256), dtype=mx.bfloat16)
    idx = mx.zeros((rows,), dtype=mx.uint32)
    kw = dict(group_size=32, bits=4, mode="affine")
    monkeypatch.setenv("MTPLX_FORCE_GPU_FAMILY_FALLBACK", "1")
    assert not nax_gather.applies(tok, idx, wq, s, b, idx, **kw)
    monkeypatch.delenv("MTPLX_FORCE_GPU_FAMILY_FALLBACK")
    monkeypatch.setenv("MTPLX_MOE_SORTED_GATHER_KERNEL", "0")
    assert not nax_gather.applies(tok, idx, wq, s, b, idx, **kw)


@needs_tensor_units
def test_a_failed_canary_falls_back_to_the_stock_op(monkeypatch):
    real_launch = nax_gather._launch

    def wrong(*args, **kwargs):
        return real_launch(*args, **kwargs) + 1

    monkeypatch.setattr(nax_gather, "_launch", wrong)
    # Enough rows for the entry point to take the kernel under any MLX.
    tok, row_map, idx = _routed(2000, 64, 10, 256, mx.bfloat16, seed=13)
    assert int(idx.shape[0]) >= nax_gather.min_rows()
    wq, s, b = _weights(64, 128, 256, 32, 4, mx.bfloat16)
    assert nax_gather.gather_rows_qmm(tok, row_map, wq, s, b, idx, group_size=32, bits=4) is None
    assert nax_gather.stats()["canary_failures"] == 1
    # The entry point then copies the rows and runs the stock op.
    y = msg.gather_qmm_rows(tok, row_map, wq, s, b, idx, group_size=32, bits=4)
    assert np.array_equal(_bits(y), _bits(_stock(tok, row_map, wq, s, b, idx, 32, 4)))
    assert nax_gather.stats()["fallbacks"] == 2


@needs_tensor_units
def test_a_failed_swiglu_canary_falls_back_to_the_stock_chain(monkeypatch):
    real_launch = nax_gather._launch

    def wrong(*args, **kwargs):
        return real_launch(*args, **kwargs) + 1

    monkeypatch.setattr(nax_gather, "_launch", wrong)
    tok, row_map, idx = _routed(2000, 64, 10, 256, mx.bfloat16, seed=23)
    assert int(idx.shape[0]) >= nax_gather.min_rows()
    wq, s, b = _weights(64, 128, 256, 32, 4, mx.bfloat16)
    y = msg.swiglu_rows(tok, row_map, wq, s, b, idx, group_size=32, bits=4)
    assert np.array_equal(_bits(y), _bits(_stock_swiglu(tok, row_map, wq, s, b, idx, 32, 4)))
    # The fused instantiation failed, then the plain one, then the stock chain ran.
    assert nax_gather.stats()["canary_failures"] == 2
    assert nax_gather.stats()["calls"] == 0


@pytest.mark.parametrize("batch,tokens", [(1, 4096), (2, 1100)])
def test_flash_next_expert_module_equals_the_code_it_replaced(monkeypatch, batch, tokens):
    """Flash-Next's own gate/up + down module at a 4,096-token chunk against
    the code it ran before (mlx-lm's gather-sort copy, the stock gate/up gather,
    split, ``nn.silu(gate) * up``, the down gather, the unsort), bit for bit, in
    both the block-forward and the prefill-combine entries.  On a tensor-unit
    GPU the module runs the fused kernel; elsewhere (and with the kill switch)
    the stock chain."""

    from mlx_lm.models.switch_layers import _gather_sort, _scatter_unsort

    from mtplx.models import qwen4_exp

    experts, hidden, inter, top_k = 64, 256, 128, 10
    mx.random.seed(14)
    gu = (mx.random.normal((experts, 2 * inter, hidden)) * 0.05).astype(mx.bfloat16)
    gu_w, gu_s, gu_b = mx.quantize(gu, group_size=32, bits=4)
    down = QuantizedSwitchLinear(inter, hidden, experts, bias=False, group_size=32, bits=4)
    dn = (mx.random.normal((experts, hidden, inter)) * 0.05).astype(mx.bfloat16)
    down.weight, down.scales, down.biases = mx.quantize(dn, group_size=32, bits=4)
    switch = qwen4_exp._FusedGateUpSwitchGLU(down, gu_w, gu_s, gu_b, 32, 4, "affine")
    x = (mx.random.normal((batch, tokens, hidden)) * 0.5).astype(mx.bfloat16)
    inds = mx.argsort(mx.random.uniform(shape=(batch * tokens, experts)), axis=-1)[:, :top_k]
    inds = inds.astype(mx.uint32).reshape(batch, tokens, top_k)

    xs, idx, inv_ref = _gather_sort(mx.expand_dims(x, (-2, -3)), inds)
    gate, up = mx.split(
        mx.gather_qmm(
            xs, gu_w, gu_s, gu_b, rhs_indices=idx, transpose=True,
            group_size=32, bits=4, sorted_indices=True,
        ),
        2,
        axis=-1,
    )
    y_ref = down(nn.silu(gate) * up, idx, sorted_indices=True)
    block_ref = _scatter_unsort(y_ref, inv_ref, inds.shape).squeeze(-2)

    rows = mx.zeros((batch * tokens * top_k,), dtype=mx.uint32)
    on_tensor_units = nax_gather.applies(
        x.reshape(-1, 1, hidden), rows, gu_w, gu_s, gu_b, rows,
        group_size=32, bits=4, mode="affine",
    )
    for label, env in (("default", None), ("kill switch", "0")):
        if env is not None:
            monkeypatch.setenv("MTPLX_MOE_SORTED_GATHER_KERNEL", env)
        calls = nax_gather.stats()["calls"]
        y, inv = switch.sorted_experts(x, inds)
        block = switch(x, inds)
        mx.eval(y, inv, block)
        kernel_calls = nax_gather.stats()["calls"] - calls
        assert kernel_calls == (2 if on_tensor_units and env is None else 0), label
        assert np.array_equal(np.array(inv), np.array(inv_ref)), label
        assert np.array_equal(_bits(y), _bits(y_ref.reshape(y_ref.shape[0], -1))), label
        assert np.array_equal(_bits(block), _bits(block_ref)), label


@needs_tensor_units
def test_canary_rejects_a_difference_only_in_the_sign_of_zero(monkeypatch):
    """A first call whose output differs from the stock chain only by -0.0
    where the stock wrote +0.0 is value-equal but not bit-identical: the
    canary must refuse it (a float comparison would not)."""

    real_launch = nax_gather._launch

    def negative_zeros(*args, **kwargs):
        y = real_launch(*args, **kwargs)
        return mx.where(y == 0, mx.zeros_like(y) * -1.0, y)

    monkeypatch.setattr(nax_gather, "_launch", negative_zeros)
    tok, row_map, idx = _routed(500, 64, 10, 256, mx.bfloat16, seed=61)
    tok = mx.concatenate([mx.zeros((50, 1, 256), dtype=tok.dtype), tok[50:]])  # rows of exact zeros
    wq, s, b = _weights(64, 128, 256, 32, 4, mx.bfloat16)
    stock = _stock_swiglu(tok, row_map, wq, s, b, idx, 32, 4)
    faked = negative_zeros(tok, row_map, wq, s, b, idx, group_size=32, bits=4, swiglu=True)
    assert bool(mx.array_equal(faked, stock).item())  # equal as floats ...
    assert not np.array_equal(_bits(faked), _bits(stock))  # ... not as bits
    ours = nax_gather.gather_rows_qmm(tok, row_map, wq, s, b, idx, group_size=32, bits=4, swiglu=True)
    assert ours is None
    assert nax_gather.stats()["canary_failures"] == 1


@needs_tensor_units
def test_canary_checks_the_whole_first_call_including_its_tail(monkeypatch):
    """A kernel wrong only for the highest experts' rows, the tail of the
    sorted order (rows 38,000 and up of 40,950), must be caught on its first
    call: a check that runs a strided sample of the call (at most 4,096 rows,
    the last at 36,855) never sees them."""

    real_launch = nax_gather._launch

    def bad_top_experts(tokens, row_map, w, scales, biases, rhs_indices, **kwargs):
        y = real_launch(tokens, row_map, w, scales, biases, rhs_indices, **kwargs)
        return mx.where((rhs_indices >= 60).reshape(-1, 1, 1), y + 1, y)

    monkeypatch.setattr(nax_gather, "_launch", bad_top_experts)
    tok, row_map, idx = _routed(4095, 64, 10, 256, mx.bfloat16, seed=62)
    assert int(idx.shape[0]) == 40950
    first_bad = int(np.argmax(np.array(idx) >= 60))
    assert first_bad > 9 * 4095  # past the last row a strided 4,096-row sample reaches
    wq, s, b = _weights(64, 128, 256, 32, 4, mx.bfloat16)
    assert nax_gather.gather_rows_qmm(tok, row_map, wq, s, b, idx, group_size=32, bits=4) is None
    assert nax_gather.stats()["canary_failures"] == 1


@needs_tensor_units
def test_each_row_regime_gets_its_own_first_call_check():
    """MLX tiles narrow calls (under 64 rows per expert) and wide ones
    differently; the first call of each runs the check."""

    wq, s, b = _weights(64, 128, 256, 32, 4, mx.bfloat16)
    narrow = _routed(300, 64, 10, 256, mx.bfloat16, seed=63)  # 3,000 rows, 47 per expert
    wide = _routed(500, 64, 10, 256, mx.bfloat16, seed=64)  # 5,000 rows, 78 per expert
    for tok, row_map, idx in (narrow, wide, narrow, wide):
        ours = nax_gather.gather_rows_qmm(tok, row_map, wq, s, b, idx, group_size=32, bits=4)
        assert np.array_equal(_bits(ours), _bits(_stock(tok, row_map, wq, s, b, idx, 32, 4)))
    assert nax_gather.stats()["canaries"] == 2
    assert nax_gather.stats()["calls"] == 4


@needs_tensor_units
def test_each_passing_first_call_check_logs_one_engagement_line(capsys):
    """The server log shows that the kernel serves: one line per
    instantiation and row regime when its first call passes the check, and
    none on later calls."""

    wq, s, b = _weights(64, 128, 256, 32, 4, mx.bfloat16)
    narrow = _routed(300, 64, 10, 256, mx.bfloat16, seed=63)
    wide = _routed(500, 64, 10, 256, mx.bfloat16, seed=64)
    for tok, row_map, idx in (narrow, wide, narrow, wide):
        assert nax_gather.gather_rows_qmm(tok, row_map, wq, s, b, idx, group_size=32, bits=4) is not None
    err = capsys.readouterr().err.splitlines()
    engaged = [line for line in err if line.startswith("[moe-sorted-gather] tensor-unit kernel on for ")]
    assert len(engaged) == 2
    assert "'narrow')" in engaged[0] and "'wide')" in engaged[1]


@pytest.mark.skipif(os.geteuid() == 0, reason="permissions do not stop root")
def test_an_untraversable_include_directory_keeps_the_stock_path(tmp_path, monkeypatch):
    """Where MLX's include directory cannot be traversed, the header lookup
    itself raises (``Path.is_file()`` propagates PermissionError); the lookup
    sits inside the same fallback, so the prefill keeps the stock path."""

    import functools

    package = tmp_path / "mlx"
    locked = package / "include"
    (locked / "mlx").mkdir(parents=True)
    locked.chmod(0)
    try:
        with pytest.raises(PermissionError):
            nax_gather._include_root(package)
        monkeypatch.setattr(nax_gather, "_include_root", functools.partial(nax_gather._include_root, package))
        nax_gather._mlx_headers.cache_clear()
        assert nax_gather._mlx_headers() is None
        assert nax_gather.stats()["header_failures"] == 1
        assert not nax_gather.available()
    finally:
        locked.chmod(0o755)
        nax_gather._mlx_headers.cache_clear()


def test_unreadable_headers_keep_the_stock_path(tmp_path, monkeypatch):
    """An install with the four checked headers but without unary_ops.h: no
    exception reaches the prefill, the reason is counted once, the kernel
    reports itself unavailable."""

    root = nax_gather._include_root()
    if root is None:
        pytest.skip("this MLX install ships no kernel headers")
    kernels = "mlx/backend/metal/kernels"
    copy = tmp_path / "include"
    shutil.copytree(root / kernels, copy / kernels, ignore=shutil.ignore_patterns("unary_ops.h"))
    monkeypatch.setattr(nax_gather, "_include_root", lambda: copy)
    nax_gather._mlx_headers.cache_clear()
    try:
        assert nax_gather._mlx_headers() is None
        assert nax_gather._mlx_headers() is None
        assert nax_gather.stats()["header_failures"] == 1
        assert not nax_gather.available()
        rows = nax_gather.min_rows()
        tok = mx.zeros((rows, 1, 256), dtype=mx.bfloat16)
        idx = mx.zeros((rows,), dtype=mx.uint32)
        wq, s, b = _weights(64, 128, 256, 32, 4, mx.bfloat16)
        assert not nax_gather.applies(tok, idx, wq, s, b, idx, group_size=32, bits=4, mode="affine")
    finally:
        nax_gather._mlx_headers.cache_clear()


def test_model_load_installs_the_row_guard_and_the_switch_glu_route(monkeypatch, tmp_path):
    from mtplx import moe_sorted_gather as entry
    from mtplx import runtime

    installed = []
    monkeypatch.setattr(entry, "install_switch_linear_guard", lambda: installed.append("guard") or True)
    monkeypatch.setattr(entry, "install_switch_glu_rows", lambda: installed.append("glu") or True)

    def stop(_metadata):
        raise RuntimeError("stopped after the install step")

    monkeypatch.setattr(runtime, "engine_version_blocker", stop)
    with pytest.raises(RuntimeError, match="stopped after the install step"):
        runtime.load(tmp_path / "pack")
    assert installed == ["guard", "glu"]


@pytest.mark.parametrize("combine", ["1", "0"])
def test_flash_next_moe_block_is_bit_identical_with_and_without_the_kernel(monkeypatch, combine):
    """Flash-Next's whole MoE block (router, routed experts, shared expert,
    combine) at a 2,100-token prefill forward, through the fused prefill
    combine (``combine=1``) and through the parent block's own tail
    (``combine=0``): the same bits with the kernel as with the stock chain."""

    from types import SimpleNamespace

    from mtplx.attention_context import attention_phase
    from mtplx.models import qwen4_exp

    experts, hidden, inter, top_k = 64, 256, 64, 10
    args = SimpleNamespace(
        hidden_size=hidden, moe_intermediate_size=inter, shared_expert_intermediate_size=inter,
        norm_topk_prob=True, num_experts=experts, num_experts_per_tok=top_k,
    )
    block = qwen4_exp.SparseMoeBlock(args)
    mx.random.seed(71)
    gu = (mx.random.normal((experts, 2 * inter, hidden)) * 0.05).astype(mx.bfloat16)
    gu_w, gu_s, gu_b = mx.quantize(gu, group_size=32, bits=4)
    down = QuantizedSwitchLinear(inter, hidden, experts, bias=False, group_size=32, bits=4)
    down.set_dtype(mx.bfloat16)
    block.switch_mlp = qwen4_exp._FusedGateUpSwitchGLU(down, gu_w, gu_s, gu_b, 32, 4, "affine")
    block.set_dtype(mx.bfloat16)
    x = (mx.random.normal((1, 2100, hidden)) * 0.5).astype(mx.bfloat16)
    monkeypatch.setenv("MTPLX_QWEN4_MOE_PREFILL_COMBINE", combine)
    with attention_phase("prefill"):
        with_kernel = block(x)
        mx.eval(with_kernel)
        calls = nax_gather.stats()["calls"]
        monkeypatch.setenv("MTPLX_MOE_SORTED_GATHER_KERNEL", "0")
        stock = block(x)
        mx.eval(stock)
    assert nax_gather.stats()["calls"] == calls  # the second pass never reached the kernel
    if nax_gather.available():
        assert calls == 1
    assert tuple(with_kernel.shape) == tuple(stock.shape) == (1, 2100, hidden)
    assert np.array_equal(_bits(with_kernel), _bits(stock))


# mlx-lm's SwitchGLU (separate gate and up weights; Qwen3.5 and 3.6 MoE).


def _switch_glu(experts, hidden, inter, group_size, bits, dtype, seed):
    from mlx_lm.models.switch_layers import SwitchGLU

    mx.random.seed(seed)
    glu = SwitchGLU(hidden, inter, experts)
    for proj in (glu.gate_proj, glu.up_proj, glu.down_proj):
        proj.weight = (mx.random.normal(proj.weight.shape) * 0.05).astype(dtype)
    nn.quantize(glu, group_size=group_size, bits=bits)
    mx.eval(glu.parameters())
    return glu.eval()  # inference mode, as mlx-lm's loader leaves every model


def _original_switch_glu_call():
    from mlx_lm.models.switch_layers import SwitchGLU

    return getattr(SwitchGLU.__call__, "__wrapped__", SwitchGLU.__call__)


@needs_tensor_units
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("group_size,bits", [(64, 4), (32, 4), (64, 8), (128, 4)])
def test_split_gate_up_bit_identical_to_the_stock_chain(dtype, group_size, bits):
    from mlx_lm.models.switch_layers import SwiGLU

    glu = _switch_glu(64, 256, 128, group_size, bits, dtype, seed=31)
    tok, row_map, idx = _routed(500, 64, 8, 256, dtype, seed=32)
    gate, up = glu.gate_proj, glu.up_proj
    ours = nax_gather.gather_rows_qmm(
        tok, row_map, gate.weight, gate.scales, gate.biases, idx, group_size=group_size, bits=bits,
        swiglu=True, up=(up.weight, up.scales, up.biases), act=SwiGLU(),
    )
    assert ours is not None, nax_gather.stats()
    x = tok[row_map]
    stock = SwiGLU()(up(x, idx, sorted_indices=True), gate(x, idx, sorted_indices=True))
    assert tuple(ours.shape) == tuple(stock.shape) == (4000, 1, 128)
    assert np.array_equal(_bits(ours), _bits(stock))


def test_mlx_lm_switch_glu_equals_its_own_call():
    """A quantized mlx-lm SwitchGLU at a 2,100-token chunk, top-8 of 64
    experts (16,800 routed rows, past the threshold under either MLX): the
    row-map route equals the module's own call bit for bit, through the
    kernel on an M5 and through the original call everywhere else."""

    from mtplx import moe_sorted_gather as entry

    glu = _switch_glu(64, 256, 128, 64, 4, mx.bfloat16, seed=41)
    mx.random.seed(42)
    x = (mx.random.normal((1, 2100, 256)) * 0.5).astype(mx.bfloat16)
    inds = mx.argsort(mx.random.uniform(shape=(2100, 64)), axis=-1)[:, :8]
    inds = inds.astype(mx.uint32).reshape(1, 2100, 8)
    original = _original_switch_glu_call()
    reference = original(glu, x, inds)
    calls = nax_gather.stats()["calls"]
    routed = entry.switch_glu_rows(glu, x, inds, original)
    mx.eval(reference, routed)
    assert tuple(routed.shape) == tuple(reference.shape) == (1, 2100, 8, 256)
    assert np.array_equal(_bits(routed), _bits(reference))
    assert nax_gather.stats()["calls"] - calls == (1 if nax_gather.available() else 0)


def test_switch_glu_other_widths_and_layouts_keep_the_original_call():
    from mlx_lm.models.switch_layers import SwitchGLU

    from mtplx import moe_sorted_gather as entry

    seen = []

    def original(module, x, indices):
        seen.append(int(indices.size))
        return x

    glu = _switch_glu(64, 256, 128, 64, 4, mx.bfloat16, seed=51)
    x = mx.zeros((1, 4, 256), dtype=mx.bfloat16)
    decode = mx.zeros((1, 4, 8), dtype=mx.uint32)
    entry.switch_glu_rows(glu, x, decode, original)  # verify width
    wide = mx.zeros((1, 2100, 8), dtype=mx.uint32)
    wide_x = mx.zeros((1, 2100, 256), dtype=mx.bfloat16)
    other = SwitchGLU(256, 128, 64, activation=nn.GELU()).eval()
    entry.switch_glu_rows(other, wide_x, wide, original)  # not SwiGLU
    dense = SwitchGLU(256, 128, 64).eval()
    entry.switch_glu_rows(dense, wide_x, wide, original)  # not quantized
    entry.switch_glu_rows(glu.train(), wide_x, wide, original)  # training: keeps the gradient path
    assert seen == [32, 16800, 16800, 16800]


def test_install_switch_glu_rows_is_idempotent_and_off_without_the_kernel(monkeypatch):
    from mlx_lm.models import switch_layers

    from mtplx import moe_sorted_gather as entry

    monkeypatch.setattr(switch_layers.SwitchGLU, "__call__", _original_switch_glu_call())
    monkeypatch.setenv("MTPLX_FORCE_GPU_FAMILY_FALLBACK", "1")
    assert entry.install_switch_glu_rows() is False
    assert not hasattr(switch_layers.SwitchGLU.__call__, "__wrapped__")
    monkeypatch.delenv("MTPLX_FORCE_GPU_FAMILY_FALLBACK")
    installed = entry.install_switch_glu_rows()
    assert installed is nax_gather.available()
    if installed:
        wrapped = switch_layers.SwitchGLU.__call__
        assert entry.install_switch_glu_rows() is True
        assert switch_layers.SwitchGLU.__call__ is wrapped


@pytest.mark.parametrize("split", [None, "stable-prefix"])
def test_the_serving_prefill_loop_is_bit_identical_through_the_switch_glu_route(monkeypatch, tmp_path, split):
    """The product's cold streaming prefill loop on a tiny quantized
    Qwen3.5-MoE (the A3B layout, with its draft head), chunked so every
    forward routes past the kernel's threshold, with and without the
    SwitchGLU route: the logits, the last hidden, every trunk cache leaf and
    the draft cache carry the same bits.  ``stable-prefix`` splits the body
    at a stable-prefix edge (as an image or a cache boundary does)."""

    from mlx_lm.models import switch_layers

    from mtplx import generation
    from mtplx import moe_sorted_gather as entry
    from tests.a3b_tiny_synth import assert_bit_equal, prompt, tiny_model_with_draft_head
    from tests.test_mtp_history_cache_only import _LoopRuntime

    model = tiny_model_with_draft_head(tmp_path).eval()
    top_k = 4
    chunk = -(-nax_gather.min_rows() // top_k) + 40  # every full chunk routes past the threshold
    tokens = prompt(2 * chunk + 31, seed=12)
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", "1")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE", str(chunk))
    monkeypatch.setattr(switch_layers.SwitchGLU, "__call__", _original_switch_glu_call())
    installed = entry.install_switch_glu_rows()
    stable = chunk + 17 if split else None

    def cold():
        rt = _LoopRuntime(model, tmp_path)
        out = generation._prefill_committed_mtp_history_streaming(rt, list(tokens), stable_prefix_len=stable)
        cache, logits, hidden, mtp_cache = out[:4]
        mx.eval(logits, hidden)
        leaves = [leaf for entry_ in cache for leaf in entry_.state if leaf is not None]
        return [logits, hidden, *leaves, *mtp_cache[0].state]

    routed = cold()
    calls = nax_gather.stats()["calls"]
    monkeypatch.setenv("MTPLX_MOE_SORTED_GATHER_KERNEL", "0")
    stock = cold()
    assert nax_gather.stats()["calls"] == calls
    assert (calls > 0) == installed
    assert_bit_equal(routed, stock)


def test_the_warm_restored_suffix_loop_is_bit_identical_through_the_switch_glu_route(monkeypatch, tmp_path):
    """A warm agent turn: a restored prefix, then the suffix through the
    product's restored-suffix prefill loop, chunked so every full chunk routes
    past the kernel's threshold, with and without the SwitchGLU route: the
    logits, the last hidden and every trunk cache leaf carry the same bits."""

    from types import SimpleNamespace

    from mlx_lm.models import switch_layers

    from mtplx import generation
    from mtplx import moe_sorted_gather as entry
    from tests.a3b_tiny_synth import assert_bit_equal, prompt, tiny_model_with_draft_head
    from tests.test_mtp_history_cache_only import _LoopRuntime

    model = tiny_model_with_draft_head(tmp_path).eval()
    top_k, cached = 4, 300
    chunk = -(-nax_gather.min_rows() // top_k) + 40
    tokens = prompt(cached + 2 * chunk + 31, seed=13)
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", "1")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE", str(chunk))
    monkeypatch.setenv("MTPLX_SMALL_SUFFIX_FUSED_MAX", "0")
    monkeypatch.setattr(switch_layers.SwitchGLU, "__call__", _original_switch_glu_call())
    installed = entry.install_switch_glu_rows()

    def warm():
        rt = _LoopRuntime(model, tmp_path)
        cache = rt.make_cache()
        out = rt.forward_ar(mx.array([tokens[:cached]]), cache=cache, return_hidden=True, emit_logits=False)
        mx.eval(out[1])
        restored = SimpleNamespace(
            cache=cache,
            mtp_history_cache=rt.make_mtp_cache(),
            hidden=None,
            entry=SimpleNamespace(prefix_len=cached),
        )
        logits, hidden, _forward_s, _history_s = generation._prefill_restored_prompt_suffix(
            rt,
            restored,
            list(tokens[cached:]),
            base_hidden_variant="post_norm",
            mtp_hidden_variant="post_norm",
            mtp_history_policy="committed",
            cached_tokens=cached,
        )
        mx.eval(logits, hidden)
        leaves = [leaf for entry_ in cache for leaf in entry_.state if leaf is not None]
        return [logits, hidden, *leaves]

    routed = warm()
    calls = nax_gather.stats()["calls"]
    monkeypatch.setenv("MTPLX_MOE_SORTED_GATHER_KERNEL", "0")
    stock = warm()
    assert nax_gather.stats()["calls"] == calls
    assert (calls > 0) == installed
    assert_bit_equal(routed, stock)
