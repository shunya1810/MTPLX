"""QSA verify attention read in place (mtplx.kernels.qsa_rows_mma) against the
stock rows-gather product over the same per-row token lists."""

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

import mtplx.models.qwen4_exp as Q
from mtplx.kernels.qsa_rows_mma import qsa_rows_attention, supported


def _lists(rng, T, S, ksel, ratio=4):
    """Selected complete blocks below each row's tail plus its visible tail."""
    idx = np.zeros((S, ksel), np.int32)
    ok = np.zeros((S, ksel), bool)
    nblk = (ksel - ratio) // ratio
    for s in range(S):
        qpos = T - S + s
        tail0 = (qpos // ratio) * ratio
        nb = tail0 // ratio
        take = min(nblk, nb)
        blks = rng.choice(nb, take, replace=False) if nb else np.array([], int)
        toks = (blks[:, None] * ratio + np.arange(ratio)).reshape(-1)
        idx[s, : toks.size] = toks
        ok[s, : toks.size] = True
        tt = tail0 + np.arange(ratio)
        idx[s, nblk * ratio :] = np.where(tt <= qpos, tt, 0)
        ok[s, nblk * ratio :] = tt <= qpos
    return mx.array(np.where(ok, idx, 0)), mx.array(ok)


def _inputs(rng, S, T, cap, dt, H=24, hk=2, D=256):
    q = mx.array(rng.standard_normal((1, H, S, D)).astype(np.float32) * 0.5).astype(dt)
    keys = mx.array(rng.standard_normal((1, hk, cap, D)).astype(np.float32)).astype(dt)
    vals = mx.array(rng.standard_normal((1, hk, cap, D)).astype(np.float32)).astype(dt)
    return q, keys, vals


@pytest.mark.parametrize(
    "S,T,cap,ksel,dt",
    [
        (4, 9000, 9216, 2304, mx.float16),
        (4, 20000, 20480, 2304, mx.bfloat16),
        (2, 6000, 6144, 2304, mx.float16),
        (8, 3000, 3072, 2304, mx.float16),
    ],
)
def test_matches_rows_gather(S, T, cap, ksel, dt):
    rng = np.random.default_rng(S * 7 + T)
    q, keys, vals = _inputs(rng, S, T, cap, dt)
    ti, to = _lists(rng, T, S, ksel)
    assert supported(q, keys, ti)
    scale = 256 ** -0.5
    got = qsa_rows_attention(q, keys, vals, ti, to, scale).astype(mx.float32)
    ref = Q._qsa_rows_gather_attention(
        q.astype(mx.float32),
        keys[:, :, :T].astype(mx.float32),
        vals[:, :, :T].astype(mx.float32),
        ti,
        to,
        scale,
        Q._qsa_stock_rows_gather_kv,
    )
    tol = 1e-4 if dt == mx.float16 else 6e-4
    assert float(mx.abs(got - ref).max()) < tol


def test_causal_prefix_lists_match_masked_sdpa():
    """Below the selection budget every row sees its causal prefix."""
    rng = np.random.default_rng(3)
    S, T, cap = 4, 1500, 1792
    q, keys, vals = _inputs(rng, S, T, cap, mx.float16)
    tpos = mx.arange(T, dtype=mx.int32)
    qpos = T - S + mx.arange(S, dtype=mx.int32)
    ok = tpos[None, :] <= qpos[:, None]
    got = qsa_rows_attention(q, keys, vals, mx.broadcast_to(tpos[None, :], ok.shape), ok, 0.0625)
    ref = mx.fast.scaled_dot_product_attention(
        q.astype(mx.float32),
        keys[:, :, :T].astype(mx.float32),
        vals[:, :, :T].astype(mx.float32),
        scale=0.0625,
        mask=ok[None, None],
    )
    assert float(mx.abs(got.astype(mx.float32) - ref).max()) < 1e-4


def test_unsupported_shapes_decline():
    q = mx.zeros((1, 24, 4, 256), mx.float16)
    assert not supported(q, mx.zeros((1, 2, 64, 256), mx.float32), mx.zeros((4, 8), mx.int32))
    assert not supported(q, mx.zeros((1, 5, 64, 256), mx.float16), mx.zeros((4, 8), mx.int32))
    assert not supported(q, mx.zeros((1, 2, 64, 256), mx.float16), mx.zeros((3, 8), mx.int32))
