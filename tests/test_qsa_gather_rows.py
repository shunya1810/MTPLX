"""QSA rows-gather lane parity (S>1), GPU: Metal.

Adapting community PR #380 by @maceip: past the engage threshold, an S>1
forward (batched-verify widths, AR pipelining) routed through the
rows-gather lane must produce the same attention output as the dense
bool-mask lane — identical per-row visible sets, so outputs differ only by
reduction-order bf16 noise. Below MTPLX_QSA_GATHER_MIN_CONTEXT or above
MTPLX_QSA_GATHER_MAX_ROWS the indexer must keep returning the dense mask.
"""

import mlx.core as mx
import pytest

from mtplx.models.qwen4_exp import (
    Attention,
    QSACache,
    TextArgs,
    _qsa_gather_max_rows,
    _qsa_gather_min_context,
)


@pytest.fixture()
def attn():
    if mx.default_device().type != mx.DeviceType.gpu:
        pytest.skip("Metal kernel needs the GPU")
    mx.random.seed(23)
    layer = Attention(TextArgs())
    layer.eval()
    mx.eval(layer.parameters())
    return layer


def _fresh_cache_run(layer, prefill, steps):
    cache = QSACache(compress_ratio=layer.indexer.ratio)
    out_p = layer(prefill, cache)
    outs = [layer(s, cache) for s in steps]
    mx.eval(out_p, *outs)
    return outs


def _engaged_prefill(layer, extra_blocks=9):
    ratio = layer.indexer.ratio
    t0 = layer.indexer.block_topk * ratio + extra_blocks * ratio + 1
    mx.random.seed(31)
    return (mx.random.normal((1, t0, 2560)) * 0.3).astype(mx.bfloat16)


def test_rows_gather_parity_at_verify_widths(attn, monkeypatch):
    prefill = _engaged_prefill(attn)
    mx.random.seed(41)
    # Verify-shaped steps: 4 rows, then 3 rows (row tails straddle block
    # boundaries because positions advance by the step width each call).
    steps = [
        (mx.random.normal((1, 4, 2560)) * 0.3).astype(mx.bfloat16),
        (mx.random.normal((1, 3, 2560)) * 0.3).astype(mx.bfloat16),
    ]

    monkeypatch.setenv("MTPLX_QSA_GATHER", "0")
    ref = _fresh_cache_run(attn, prefill, steps)

    monkeypatch.setenv("MTPLX_QSA_GATHER", "1")
    monkeypatch.setenv("MTPLX_QSA_GATHER_MIN_CONTEXT", "0")
    sel = attn.indexer(steps[0], prefill.shape[1], _prefilled_cache(attn, prefill))
    assert isinstance(sel, tuple) and sel[0] == "gather_rows", "lane did not engage"
    got = _fresh_cache_run(attn, prefill, steps)

    for i, (r, g) in enumerate(zip(ref, got)):
        scale = mx.abs(r.astype(mx.float32)).max().item() + 1e-6
        err = (
            mx.abs(g.astype(mx.float32) - r.astype(mx.float32)) / scale
        ).max().item()
        assert err < 2e-2, f"step {i} rel err {err}"


def _prefilled_cache(layer, prefill):
    cache = QSACache(compress_ratio=layer.indexer.ratio)
    mx.eval(layer(prefill, cache))
    return cache


def test_rows_gather_selection_shape_and_disjointness(attn, monkeypatch):
    monkeypatch.setenv("MTPLX_QSA_GATHER", "1")
    monkeypatch.setenv("MTPLX_QSA_GATHER_MIN_CONTEXT", "0")
    prefill = _engaged_prefill(attn)
    cache = _prefilled_cache(attn, prefill)
    ratio = attn.indexer.ratio
    step = (mx.random.normal((1, 4, 2560)) * 0.3).astype(mx.bfloat16)
    sel = attn.indexer(step, cache.offset, cache)
    assert isinstance(sel, tuple) and sel[0] == "gather_rows"
    _, token_idx, token_ok = sel
    k_eff = attn.indexer.block_topk
    assert tuple(token_idx.shape) == (4, k_eff * ratio + ratio)
    assert tuple(token_ok.shape) == tuple(token_idx.shape)
    idx = token_idx.tolist()
    ok = token_ok.tolist()
    T = cache.offset + 4
    for row_i, (row, row_ok) in enumerate(zip(idx, ok)):
        visible = [t for t, o in zip(row, row_ok) if o]
        qpos = cache.offset + row_i
        assert visible, "a row selected nothing"
        assert len(set(visible)) == len(visible), "double-counted token"
        assert max(visible) <= qpos < T, "non-causal token gathered"


def test_rows_gather_stays_dense_below_min_context(attn, monkeypatch):
    monkeypatch.setenv("MTPLX_QSA_GATHER", "1")
    monkeypatch.setenv("MTPLX_QSA_GATHER_MIN_CONTEXT", "1000000")
    prefill = _engaged_prefill(attn)
    cache = _prefilled_cache(attn, prefill)
    step = (mx.random.normal((1, 4, 2560)) * 0.3).astype(mx.bfloat16)
    sel = attn.indexer(step, cache.offset, cache)
    assert isinstance(sel, mx.array) and sel.ndim == 4, "must stay dense"


def test_rows_gather_stays_dense_above_max_rows(attn, monkeypatch):
    monkeypatch.setenv("MTPLX_QSA_GATHER", "1")
    monkeypatch.setenv("MTPLX_QSA_GATHER_MIN_CONTEXT", "0")
    monkeypatch.setenv("MTPLX_QSA_GATHER_MAX_ROWS", "8")
    prefill = _engaged_prefill(attn)
    cache = _prefilled_cache(attn, prefill)
    step = (mx.random.normal((1, 25, 2560)) * 0.3).astype(mx.bfloat16)
    sel = attn.indexer(step, cache.offset, cache)
    assert isinstance(sel, mx.array) and sel.ndim == 4, "copy-block widths stay dense"


def test_qsa_gather_fence_defaults(monkeypatch):
    """The shipped fences behind the family default (founder release call
    2026-08-28): rows-gather engages only at KV >= 16384 and serves 2..8
    rows when the stock rows-gather attention serves it. With the in-place
    rows kernel (mtplx.kernels.qsa_rows_mma, default on) rows-gather is the
    faster route at every context, so its floor drops to 0."""
    monkeypatch.delenv("MTPLX_QSA_GATHER_MIN_CONTEXT", raising=False)
    monkeypatch.setenv("MTPLX_QSA_ROWS_MMA", "0")
    assert _qsa_gather_min_context() == 16384
    monkeypatch.delenv("MTPLX_QSA_ROWS_MMA")
    assert _qsa_gather_min_context() == 0
    assert _qsa_gather_max_rows() == 8
