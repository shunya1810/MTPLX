"""ISTA GGUF formats read in place by qwen4_exp: Q2_0 routed experts built as
MLX affine 2-bit g64 at load, and IQ4_NL n-gram rows decoded after the gather.
Both are checked against llama.cpp's reference dequantization written in numpy."""

from __future__ import annotations

import numpy as np
import mlx.core as mx

from mtplx.models.qwen4_exp import (
    NGramTable,
    _decode_iq4_nl_rows,
    _IQ4_NL_VALUES,
    _q2_0_affine_from_gguf,
)


def _q2_0_reference(blocks: np.ndarray) -> np.ndarray:
    """dequantize_row_q2_0: value = d * (q - 1), element l in bits 2 (l % 4) of byte l / 4."""
    d = blocks[..., :2].copy().view(np.float16).astype(np.float32)
    qs = blocks[..., 2:]
    q = np.stack([(qs >> (2 * k)) & 3 for k in range(4)], axis=-1).reshape(*qs.shape[:-1], 64)
    return d * (q.astype(np.float32) - 1.0)


def _iq4_nl_reference(blocks: np.ndarray) -> np.ndarray:
    d = blocks[..., :2].copy().view(np.float16).astype(np.float32)
    qs = blocks[..., 2:]
    q = np.concatenate([qs & 15, qs >> 4], axis=-1)
    return d * np.array(_IQ4_NL_VALUES, dtype=np.float32)[q]


def _random_blocks(rng, shape) -> np.ndarray:
    blocks = rng.integers(0, 256, size=(*shape, 18), dtype=np.uint8)
    d = rng.uniform(-0.05, 0.05, size=shape).astype(np.float16)
    blocks[..., :2] = d[..., None].view(np.uint8)
    return blocks


def test_q2_0_experts_load_as_exact_affine_2bit(tmp_path):
    rng = np.random.default_rng(0)
    n_exp, rows, cols = 3, 8, 256
    blocks = _random_blocks(rng, (n_exp, rows, cols // 64))
    path = tmp_path / "experts.gguf"
    pad = 96  # the tensor sits at an offset inside the file
    path.write_bytes(b"\0" * pad + blocks.tobytes())

    w, s, b = _q2_0_affine_from_gguf(
        {"path": str(path), "offset": pad, "experts": n_exp, "rows": rows, "cols": cols}
    )
    assert w.dtype == mx.uint32 and w.shape == (n_exp, rows, cols // 16)
    assert s.shape == b.shape == (n_exp, rows, cols // 64)
    got = mx.dequantize(w, s, b, group_size=64, bits=2).astype(mx.float32)
    want = _q2_0_reference(blocks).reshape(n_exp, rows, cols)
    np.testing.assert_array_equal(np.array(got), want)


def test_iq4_nl_rows_decode_like_llama_cpp():
    rng = np.random.default_rng(1)
    blocks = _random_blocks(rng, (5, 160 // 32))
    got = _decode_iq4_nl_rows(mx.array(blocks.reshape(5, -1)), mx.float32)
    want = _iq4_nl_reference(blocks).reshape(5, 160)
    np.testing.assert_array_equal(np.array(got), want)


def test_ngram_table_gathers_iq4_nl_rows_from_gguf(tmp_path, monkeypatch):
    monkeypatch.setenv("MTPLX_NGRAM_PREWARM", "off")
    rng = np.random.default_rng(2)
    rows, dim = 64, 160
    blocks = _random_blocks(rng, (rows, dim // 32))
    path = tmp_path / "table.gguf"
    pad = 192
    path.write_bytes(b"\0" * pad + blocks.tobytes())

    table = NGramTable(rows, dim, sidecar=True)
    table.attach_gguf(
        {"path": str(path), "offset": pad, "rows": rows, "dim": dim,
         "dtype": "IQ4_NL", "out_dtype": "float32"}
    )
    ids = mx.array([[3, 0, 63], [17, 17, 5]])
    got = table(ids)
    assert got.shape == (2, 3, dim)
    want = _iq4_nl_reference(blocks).reshape(rows, dim)[np.array(ids)]
    np.testing.assert_array_equal(np.array(got), want)


def test_memory_plan_counts_gguf_experts_and_table(tmp_path, monkeypatch):
    import json

    # the affine form (MTPLX_GGUF_Q2_0_BLOCKS=0; the M1 default keeps blocks,
    # test_qwen4_gguf_mixed_experts.test_q2_0_blocks_weight_estimate)
    monkeypatch.setenv("MTPLX_GGUF_Q2_0_BLOCKS", "0")

    from mtplx.engine_session import model_weights_bytes, ngram_table_bytes

    (tmp_path / "model-00001.safetensors").write_bytes(b"\0" * 1000)
    proj = {"path": "x.gguf", "offset": 0, "experts": 512, "rows": 640, "cols": 2560}
    (tmp_path / "config.json").write_text(json.dumps({"text_config": {
        "routed_experts_gguf": {"format": "Q2_0", "layers": {"0": {"gate_proj": proj}}},
        "ngram_gguf": {"path": "t.gguf", "offset": 0, "rows": 1000, "dim": 160, "dtype": "IQ4_NL"},
    }}))
    weights = 512 * 640 * 2560
    assert model_weights_bytes(tmp_path) == 1000 + weights // 4 + weights // 64 * 4
    assert ngram_table_bytes(tmp_path) == 1000 * 90


def test_q2_moe_glu_matches_float_reference():
    from mtplx.kernels.moe_q2_glu import moe_q2_glu

    mx.random.seed(3)
    E, H, F, K, M = 16, 256, 128, 4, 3
    gu = (mx.random.normal((E, 2 * F, H)) * 0.05).astype(mx.float16)
    dn = (mx.random.normal((E, H, F)) * 0.05).astype(mx.float16)
    gw, gs, gb = mx.quantize(gu, group_size=64, bits=2)
    dw, ds, db = mx.quantize(dn, group_size=64, bits=2)
    rng = np.random.default_rng(3)
    idx = np.stack([rng.choice(E, K, replace=False) for _ in range(M)]).astype(np.uint32)
    w = rng.dirichlet(np.ones(K), size=M).astype(np.float32)
    x = mx.random.normal((M, H)).astype(mx.float16)
    got = np.array(moe_q2_glu(x, gw, gs, gb, dw, ds, db, mx.array(idx.reshape(-1)),
                              mx.array(w.reshape(-1))).astype(mx.float32))
    G = np.array(mx.dequantize(gw, gs, gb, group_size=64, bits=2).astype(mx.float32))
    D = np.array(mx.dequantize(dw, ds, db, group_size=64, bits=2).astype(mx.float32))
    xf = np.array(x.astype(mx.float32))
    want = np.zeros((M, H), np.float32)
    for t in range(M):
        for s in range(K):
            g = G[idx[t, s]] @ xf[t]
            h = g[:F] / (1 + np.exp(-g[:F])) * g[F:]
            want[t] += w[t, s] * (D[idx[t, s]] @ h)
    assert np.abs(got - want).max() <= 2e-3 * np.abs(want).max()
    # a row's output does not depend on the other rows of the call
    one = np.array(moe_q2_glu(x[1:2], gw, gs, gb, dw, ds, db, mx.array(idx[1]),
                              mx.array(w[1])).astype(mx.float32))
    np.testing.assert_array_equal(one[0], got[1])


def test_hc_read_q8_matches_hc_read_rows():
    import types

    from mtplx.kernels.hc_read_q8 import hc_read_q8, supports
    from mtplx.kernels.hc_read_rows import hc_read_rows
    from mtplx.models.qwen4_exp import GatedResidual

    args = types.SimpleNamespace(hc_count=4, hidden_size=512, rms_norm_eps=1e-6, hc_lowrank=128)
    m = GatedResidual(args)
    m.input_mix_weight_down.weight = (mx.random.normal((128, 2048)) * 0.03).astype(mx.float16)
    m.input_mix_weight_up.weight = (mx.random.normal((2048, 128)) * 0.1).astype(mx.float16)
    m.block_inject_weight.weight = (mx.random.normal((4, 2048)) * 0.03).astype(mx.float16)
    m.hc_norm.weight = (1 + mx.random.normal((2048,)) * 0.1).astype(mx.float16)
    for rows in (1, 3):
        x = mx.random.normal((1, rows, 2048)).astype(mx.float16)
        pending = (mx.random.normal((1, rows, 512)).astype(mx.float16),
                   mx.random.uniform(shape=(1, rows, 4)).astype(mx.float16))
        assert supports(m, x)
        for p in (None, pending):
            ref = hc_read_rows(m, x, p)
            got = hc_read_q8(m, x, p)
            for a, b in zip(ref, got):
                a = np.array(a.astype(mx.float32)); b = np.array(b.astype(mx.float32))
                assert np.abs(a - b).max() <= 1e-2 * np.abs(a).max()
    x3 = mx.random.normal((1, 3, 2048)).astype(mx.float16)
    row = hc_read_q8(m, x3[:, 1:2])
    full = hc_read_q8(m, x3)
    np.testing.assert_array_equal(np.array(row[0][0, 0]), np.array(full[0][0, 1]))


def test_route_topk_matches_softmax_topk():
    from mtplx.kernels.moe_q2_glu import moe_route_topk

    rng = np.random.default_rng(5)
    logits = rng.normal(size=(4, 512)).astype(np.float32) * 3
    idx, w = moe_route_topk(mx.array(logits), 10, normalize=True)
    idx = np.array(idx).reshape(4, 10)
    w = np.array(w).reshape(4, 10)
    p = np.exp(logits - logits.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    for t in range(4):
        want = np.argsort(-p[t])[:10]
        assert list(idx[t]) == list(want)
        np.testing.assert_allclose(w[t], p[t][want] / p[t][want].sum(), rtol=1e-5)
