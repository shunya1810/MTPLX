"""Routed experts kept as GGUF blocks of mixed types (Unsloth UD-Q2_0 Flash-Next):
qwen4_exp._gguf_blocks_soa splits the blocks, mtplx.kernels.moe_gguf_iq reads
them. Dequantization is checked bit for bit against llama.cpp's
dequantize_row_* written in numpy; the decode GLU and the prefill matmuls
against a float reference over those weights."""

from __future__ import annotations

import numpy as np
import pytest
import mlx.core as mx
import mlx.nn as nn

from mtplx.kernels import moe_gguf_iq as K
from mtplx.kernels.gguf_iq_tables import IQ2S_GRID, IQ2XXS_GRID, IQ4NL_VALUES
from mtplx.models.qwen4_exp import GGUFMixedSwitchGLU, _gguf_blocks_soa


def _grid(vals) -> np.ndarray:
    return np.array([[(v >> (8 * j)) & 0xFF for j in range(8)] for v in vals], dtype=np.float32)


def _d(b):
    return b[..., :2].copy().view(np.float16).astype(np.float32)[..., 0]


def _reference(blocks: np.ndarray, typ: str) -> np.ndarray:
    """llama.cpp dequantize_row_* for one row of blocks [nb, bytes] -> [nb * qk]."""
    out = []
    for b in blocks:
        d = float(_d(b[None])[0])
        if typ == "Q2_0":
            qs = b[2:18]
            q = np.stack([(qs >> (2 * k)) & 3 for k in range(4)], axis=-1).reshape(64)
            out.append(d * (q.astype(np.float32) - 1.0))
        elif typ == "IQ4_NL":
            qs = b[2:18]
            out.append(d * np.array(IQ4NL_VALUES, np.float32)[np.concatenate([qs & 15, qs >> 4])])
        elif typ == "IQ2_XXS":
            grid = _grid(IQ2XXS_GRID)
            vals = []
            for ib in range(8):
                w = b[2 + 8 * ib: 10 + 8 * ib]
                aux = int(w[4]) | int(w[5]) << 8 | int(w[6]) << 16 | int(w[7]) << 24
                db = d * (0.5 + (aux >> 28)) * 0.25
                for l in range(4):
                    s7 = (aux >> (7 * l)) & 127
                    signs = s7 | ((bin(s7).count("1") & 1) << 7)
                    g = grid[w[l]]
                    vals.extend(db * g[j] * (-1.0 if (signs >> j) & 1 else 1.0) for j in range(8))
            out.append(np.array(vals, np.float32))
        else:  # IQ2_S
            grid = _grid(IQ2S_GRID)
            qs, sg, qh, sc = b[2:34], b[34:66], b[66:74], b[74:82]
            vals = []
            for ib in range(8):
                db = [d * (0.5 + (sc[ib] & 15)) * 0.25, d * (0.5 + (sc[ib] >> 4)) * 0.25]
                for l in range(4):
                    g = grid[int(qs[4 * ib + l]) | ((int(qh[ib]) << (8 - 2 * l)) & 0x300)]
                    s = sg[4 * ib + l]
                    vals.extend(db[l // 2] * g[j] * (-1.0 if (s >> j) & 1 else 1.0) for j in range(8))
            out.append(np.array(vals, np.float32))
    return np.concatenate(out)


def _blocks(rng, typ, shape):
    qk, bs, _ = K.BLOCKS[typ]
    b = rng.integers(0, 256, size=(*shape, bs), dtype=np.uint8)
    b[..., :2] = rng.uniform(0.001, 0.01, size=shape).astype(np.float16)[..., None].view(np.uint8)
    return b


def _write(tmp_path, name, blocks, typ, n_exp, rows, cols, pad=64):
    path = tmp_path / f"{name}.gguf"
    path.write_bytes(b"\0" * pad + blocks.tobytes())
    return {"path": str(path), "offset": pad, "experts": n_exp, "rows": rows, "cols": cols, "type": typ}


def _module(tmp_path, rng, gate_type, down_type, n_exp=6, k=512, n=64):
    dm = k  # the down projection returns to the hidden width
    specs, refs, arrs, shapes = {}, {}, {}, {}
    for name, typ, rows, cols in (("gate", gate_type, n, k), ("up", gate_type, n, k), ("down", down_type, dm, n)):
        qk = K.BLOCKS[typ][0]
        blocks = _blocks(rng, typ, (n_exp, rows, cols // qk))
        spec = _write(tmp_path, name, blocks, typ, n_exp, rows, cols)
        refs[name] = np.stack([[_reference(blocks[e, r], typ) for r in range(rows)] for e in range(n_exp)])
        d, q = _gguf_blocks_soa(spec)
        arrs[f"{name}_d"], arrs[f"{name}_q"] = d, q
        shapes[f"{name}_d"], shapes[f"{name}_q"] = (d.shape, d.dtype), (q.shape, q.dtype)
    sw = GGUFMixedSwitchGLU(n_exp, k, n, gate_type, down_type, shapes)
    sw.update(arrs)
    return sw, refs


@pytest.mark.parametrize("typ,cols", [("Q2_0", 128), ("IQ4_NL", 64), ("IQ2_XXS", 512), ("IQ2_S", 512)])
def test_dequant_matches_llama_cpp(tmp_path, typ, cols):
    rng = np.random.default_rng(len(typ))
    n_exp, rows = 3, 4
    qk = K.BLOCKS[typ][0]
    blocks = _blocks(rng, typ, (n_exp, rows, cols // qk))
    d, q = _gguf_blocks_soa(_write(tmp_path, typ, blocks, typ, n_exp, rows, cols))
    got = K.dequant(d, q, typ, mx.array([2, 0], mx.uint32), cols=cols, dtype=mx.float32)
    want = np.stack([[_reference(blocks[e, r], typ) for r in range(rows)] for e in (2, 0)])
    np.testing.assert_allclose(np.array(got), want, rtol=1e-6, atol=1e-7)


def _glu_reference(refs, x, idx, w, dtype):
    xs = np.array(x.astype(mx.float32))
    out = np.zeros((xs.shape[0], refs["down"].shape[1]), np.float32)
    for t in range(xs.shape[0]):
        for s in range(idx.shape[1]):
            e = int(idx[t, s])
            g = mx.array(refs["gate"][e] @ xs[t]).astype(dtype)
            u = mx.array(refs["up"][e] @ xs[t]).astype(dtype)
            h = np.array((nn.silu(g) * u).astype(dtype).astype(mx.float32))
            out[t] += float(w[t, s]) * (refs["down"][e] @ h)
    return out


@pytest.mark.parametrize("gate_type,down_type", [("IQ2_XXS", "IQ4_NL"), ("IQ2_S", "Q2_0"), ("Q2_0", "Q2_0")])
def test_decode_glu_and_prefill_matmul_match_reference(tmp_path, gate_type, down_type):
    rng = np.random.default_rng(3)
    sw, refs = _module(tmp_path, rng, gate_type, down_type)
    dtype = mx.float16
    for m in (1, 3, 40, 200):  # 200 rows: several token tiles an expert
        x = mx.array(rng.normal(0, 0.5, size=(m, 512)).astype(np.float32)).astype(dtype)
        idx = np.stack([rng.choice(6, 2, replace=False) for _ in range(m)])
        w = rng.dirichlet([1, 1], size=m).astype(np.float32)
        want = _glu_reference(refs, x, idx, w, dtype)
        flat = mx.array(idx.reshape(-1), mx.uint32)
        if m <= 8:
            got = K.moe_iq_glu(x, sw, flat, mx.array(w.reshape(-1)))
        else:
            part = K.moe_iq_prefill_parts(x, sw, flat)
            got = (part.reshape(m, 2, -1) * mx.array(w)[..., None]).sum(axis=1)
        got = np.array(got.astype(mx.float32))
        scale = np.abs(want).max()
        np.testing.assert_allclose(got, want, atol=4e-3 * scale + 1e-4)


def test_switch_call_keeps_switchglu_contract(tmp_path):
    rng = np.random.default_rng(5)
    sw, refs = _module(tmp_path, rng, "IQ2_XXS", "Q2_0")
    x = mx.array(rng.normal(0, 0.5, size=(1, 5, 512)).astype(np.float32)).astype(mx.float16)
    idx = mx.array(np.stack([rng.choice(6, 2, replace=False) for _ in range(5)])[None], mx.uint32)
    y = sw(x, idx)
    assert y.shape == (1, 5, 2, 512) and y.dtype == mx.float16


@pytest.mark.parametrize("gate_type,down_type", [("IQ2_XXS", "IQ4_NL"), ("IQ2_S", "Q2_0"), ("IQ2_S", "IQ4_NL"), ("Q2_0", "Q2_0")])
@pytest.mark.parametrize("bm", ["16", "32"])
def test_prefill_on_mlx_qmm_tile_matches_reference(tmp_path, monkeypatch, gate_type, down_type, bm):
    """moe_gguf_qmm: GGUF loader inside MLX's quantized-matmul tile."""
    from mtplx.kernels import moe_gguf_qmm as G

    if not G.available(mx.float16):
        pytest.skip("MLX kernel headers unavailable")
    monkeypatch.setenv("MTPLX_GGUF_QMM_BM", bm)
    rng = np.random.default_rng(7)
    sw, refs = _module(tmp_path, rng, gate_type, down_type)
    dtype = mx.float16
    for m in (40, 200):
        x = mx.array(rng.normal(0, 0.5, size=(m, 512)).astype(np.float32)).astype(dtype)
        idx = np.stack([rng.choice(6, 2, replace=False) for _ in range(m)])
        w = rng.dirichlet([1, 1], size=m).astype(np.float32)
        want = _glu_reference(refs, x, idx, w, dtype)
        part = G.prefill_parts(x, sw, mx.array(idx.reshape(-1), mx.uint32))
        got = np.array((part.reshape(m, 2, -1) * mx.array(w)[..., None]).sum(axis=1).astype(mx.float32))
        np.testing.assert_allclose(got, want, atol=4e-3 * np.abs(want).max() + 1e-4)


def test_q2_0_blocks_decode_is_the_affine_kernel_bit_for_bit(tmp_path):
    """All-Q2_0 blocks (ISTA pack, MTPLX_GGUF_Q2_0_BLOCKS) at decode/verify
    widths: moe_q2_blocks_glu equals moe_q2_glu over the affine form the
    default load builds (scale = d, bias = -d, gate+up fused)."""
    from mtplx.kernels.moe_q2_glu import moe_q2_blocks_glu, moe_q2_glu

    rng = np.random.default_rng(11)
    sw, _ = _module(tmp_path, rng, "Q2_0", "Q2_0", n=128)  # the 2-bit kernels take N % 128 == 0
    gu_w = mx.concatenate([sw.gate_q, sw.up_q], axis=1)
    gu_s = mx.concatenate([sw.gate_d, sw.up_d], axis=1)
    for m in (1, 4):
        x = mx.array(rng.normal(0, 0.5, size=(m, 512)).astype(np.float32)).astype(mx.float16)
        flat = mx.array(np.stack([rng.choice(6, 2, replace=False) for _ in range(m)]).reshape(-1), mx.uint32)
        w = mx.array(rng.dirichlet([1, 1], size=m).astype(np.float32).reshape(-1))
        shared = mx.array(rng.normal(0, 0.1, size=(m, 512)).astype(np.float32)).astype(mx.float16)
        gate = mx.array(rng.normal(size=(m,)).astype(np.float32)).astype(mx.float16)
        want = moe_q2_glu(x, gu_w, gu_s, -gu_s, sw.down_q, sw.down_d, -sw.down_d, flat, w, shared=shared, shared_gate=gate)
        got = moe_q2_blocks_glu(x, sw, flat, w, shared=shared, shared_gate=gate)
        assert np.array_equal(np.array(got), np.array(want))


def test_q2_0_blocks_weight_estimate(tmp_path, monkeypatch):
    """The memory plan prices Q2_0 experts at 18 bytes a 64-weight block when
    they stay blocks, at codes + fp16 scale and bias otherwise."""
    import json

    from mtplx.engine_session import gguf_routed_expert_bytes

    t = {"path": "x", "offset": 0, "experts": 4, "rows": 64, "cols": 128}
    (tmp_path / "config.json").write_text(json.dumps(
        {"text_config": {"routed_experts_gguf": {"format": "Q2_0", "layers": {"0": {"gate_proj": t}}}}}))
    n = 4 * 64 * 128
    monkeypatch.delenv("MTPLX_GGUF_Q2_0_BLOCKS", raising=False)
    assert gguf_routed_expert_bytes(tmp_path) == n // 4 + n // 64 * 4
    monkeypatch.setenv("MTPLX_GGUF_Q2_0_BLOCKS", "1")
    assert gguf_routed_expert_bytes(tmp_path) == n // 64 * 18
