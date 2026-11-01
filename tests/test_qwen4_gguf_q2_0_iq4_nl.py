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


def test_memory_plan_counts_gguf_experts_and_table(tmp_path):
    import json

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
