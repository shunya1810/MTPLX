"""Numerics of the M1 MMA split-K attention kernels (kernels/sdpa_gqa_mma)."""

from __future__ import annotations

import mlx.core as mx
import pytest

from mtplx.kernels.sdpa_gqa_mma import sdpa_gqa_mma, sdpa_gqa_mma_prefill
from mtplx.kv_quant import dequantize_symmetric, quantize_symmetric

HQ, HK, D = 24, 4, 256


def _require_metal() -> None:
    if not mx.metal.is_available():
        pytest.skip("Metal is unavailable")


def _reference(queries, keys, values, n_kv, scale):
    """fp32 tail-causal attention: query row j sees keys n <= n_kv - q_len + j."""
    _, hq, q_len, d = queries.shape
    hk = keys.shape[1]
    gqa = hq // hk
    q = queries.astype(mx.float32).reshape(1, hk, gqa * q_len, d)
    k = keys[:, :, :n_kv, :].astype(mx.float32)
    v = values[:, :, :n_kv, :].astype(mx.float32)
    scores = (q * scale) @ k.transpose(0, 1, 3, 2)
    limit = n_kv - q_len + (mx.arange(gqa * q_len) % q_len)
    visible = mx.arange(n_kv)[None, :] <= limit[:, None]
    scores = mx.where(visible[None, None], scores, -mx.inf)
    return (mx.softmax(scores, axis=-1) @ v).reshape(1, hq, q_len, d)


def _kv(n: int, seed: int):
    k = (mx.random.normal((1, HK, n, D), key=mx.random.key(seed)) * 0.6).astype(mx.float16)
    v = (mx.random.normal((1, HK, n, D), key=mx.random.key(seed + 1)) * 0.6).astype(mx.float16)
    return k, v


@pytest.mark.parametrize("q_len", [1, 3, 4, 9])
@pytest.mark.parametrize("n_kv", [700, 5003])
def test_dense_layout_matches_reference(q_len: int, n_kv: int) -> None:
    _require_metal()
    cap = n_kv + 64
    k, v = _kv(cap, n_kv)
    q = (mx.random.normal((1, HQ, q_len, D), key=mx.random.key(7)) * 0.6).astype(mx.float16)
    out = sdpa_gqa_mma(
        queries=q, keys=k, values=v, offset=n_kv, scale=D**-0.5, num_kv_heads=HK,
        k_strides=(cap * D, D), v_strides=(cap * D, D), ceiling=cap, max_q_len=16,
    )
    ref = _reference(q, k, v, n_kv, D**-0.5)
    assert out is not None
    assert float(mx.max(mx.abs(out.astype(mx.float32) - ref)).item()) < 2e-4


@pytest.mark.parametrize("bits", [8, 4])
def test_quantized_token_major_pages_match_reference(bits: int) -> None:
    _require_metal()
    n_kv, cap = 3000, 3072
    k, v = _kv(cap, 3)
    kq, ks = quantize_symmetric(k, bits=bits)
    vq, vs = quantize_symmetric(v, bits=bits)
    kd = dequantize_symmetric(kq, ks, bits=bits, head_dim=D).astype(mx.float16)
    vd = dequantize_symmetric(vq, vs, bits=bits, head_dim=D).astype(mx.float16)
    packed = int(kq.shape[3])

    def pages(a, width):
        return mx.contiguous(a[0].transpose(1, 0, 2)).reshape(cap // 16, 16, HK, width)

    q = (mx.random.normal((1, HQ, 4, D), key=mx.random.key(11)) * 0.6).astype(mx.float16)
    out = sdpa_gqa_mma(
        queries=q, keys=pages(kq, packed), values=pages(vq, packed),
        offset=mx.array(n_kv, dtype=mx.int32), scale=D**-0.5, num_kv_heads=HK,
        k_strides=(packed, HK * packed), v_strides=(packed, HK * packed), ceiling=cap,
        kv_bits=bits, k_scales=pages(ks, 1), v_scales=pages(vs, 1), s_strides=(1, HK),
    )
    ref = _reference(q, kd, vd, n_kv, D**-0.5)
    assert out is not None
    assert float(mx.max(mx.abs(out.astype(mx.float32) - ref)).item()) < 2e-4


@pytest.mark.parametrize(("prefix", "chunk"), [(0, 7), (700, 513), (3000, 64)])
def test_prefill_matches_fused_causal_sdpa(prefix: int, chunk: int) -> None:
    _require_metal()
    cap = prefix + chunk + 32
    k, v = _kv(cap, prefix + chunk)
    q = (mx.random.normal((1, HQ, chunk, D), key=mx.random.key(5)) * 0.6).astype(mx.float16)
    n = prefix + chunk
    ref = mx.fast.scaled_dot_product_attention(
        q, k[:, :, :n], v[:, :, :n], scale=D**-0.5, mask="causal"
    ).astype(mx.float32)
    out = sdpa_gqa_mma_prefill(
        queries=q, keys=k, values=v, prefix=prefix, scale=D**-0.5, num_kv_heads=HK
    )
    assert out is not None
    # <= 1 fp16 ulp at |value| < 2 (the first rows attend to a single key).
    assert float(mx.max(mx.abs(out.astype(mx.float32) - ref)).item()) <= 1e-3


def test_contract_bails_return_none() -> None:
    _require_metal()
    k, v = _kv(128, 1)
    q = mx.zeros((1, HQ, 4, D), dtype=mx.float16)
    assert (
        sdpa_gqa_mma(
            queries=q, keys=k.astype(mx.bfloat16), values=v, offset=64, scale=1.0,
            num_kv_heads=HK, k_strides=(128 * D, D), v_strides=(128 * D, D), ceiling=128,
        )
        is None
    )
    assert (
        sdpa_gqa_mma_prefill(
            queries=q, keys=k, values=v, prefix=126, scale=1.0, num_kv_heads=HK
        )
        is None
    )


def test_dense_verify_route_min_capacity_floor(monkeypatch):
    from mtplx.attention_split import _gqa_mma_min_capacity

    monkeypatch.delenv("MTPLX_GQA_MMA_MIN_CAPACITY", raising=False)
    assert _gqa_mma_min_capacity() == 4096
    monkeypatch.setenv("MTPLX_GQA_MMA_MIN_CAPACITY", "0")
    assert _gqa_mma_min_capacity() == 0
    monkeypatch.setenv("MTPLX_GQA_MMA_MIN_CAPACITY", "junk")
    assert _gqa_mma_min_capacity() == 4096


@pytest.mark.parametrize("prefix,q_len", [(0, 37), (300, 21), (1000, 64)])
def test_prefill_over_q8_pages_matches_dequantized_reference(prefix, q_len):
    from mtplx.kernels.sdpa_gqa_mma import sdpa_gqa_mma_prefill_q8_pages
    from mtplx.kv_quant import dequantize_symmetric, quantize_symmetric

    hk, hq, d, block = 4, 24, 256, 16
    total = prefix + q_len
    blocks = (total + block - 1) // block + 2
    rows = blocks * block
    k = mx.random.normal((rows, hk, d), key=mx.random.key(1)).astype(mx.float16)
    v = mx.random.normal((rows, hk, d), key=mx.random.key(2)).astype(mx.float16)
    kq, ks = quantize_symmetric(k, bits=8)
    vq, vs = quantize_symmetric(v, bits=8)
    q = (mx.random.normal((1, hq, q_len, d), key=mx.random.key(3)) * 0.5).astype(mx.float16)
    out = sdpa_gqa_mma_prefill_q8_pages(
        queries=q,
        key_pages=kq.reshape(blocks, block, hk, d),
        value_pages=vq.reshape(blocks, block, hk, d),
        key_scales=ks.reshape(blocks, block, hk, 1),
        value_scales=vs.reshape(blocks, block, hk, 1),
        prefix=prefix,
        scale=d ** -0.5,
    )
    assert out is not None
    kd = dequantize_symmetric(kq, ks, bits=8, head_dim=d)[:total].transpose(1, 0, 2)[None]
    vd = dequantize_symmetric(vq, vs, bits=8, head_dim=d)[:total].transpose(1, 0, 2)[None]
    kd = mx.repeat(kd, hq // hk, axis=1).astype(mx.float32)
    vd = mx.repeat(vd, hq // hk, axis=1).astype(mx.float32)
    scores = (q.astype(mx.float32) @ kd.transpose(0, 1, 3, 2)) * d ** -0.5
    qpos = prefix + mx.arange(q_len)[:, None]
    kpos = mx.arange(total)[None, :]
    scores = mx.where(kpos <= qpos, scores, -1e30)
    ref = mx.softmax(scores, axis=-1) @ vd
    err = mx.max(mx.abs(out.astype(mx.float32) - ref)).item()
    assert err < 2e-2, err


@pytest.mark.parametrize("positions,simdgroups", [(5, 4), (4, 8), (6, 8)])
def test_prefill_row_geometries_match_the_default(positions, simdgroups):
    from mtplx.kernels.sdpa_gqa_mma import sdpa_gqa_mma_prefill

    hk, hq, d, prefix, q_len = 4, 24, 256, 700, 45
    cap = prefix + q_len + 16
    k = mx.random.normal((1, hk, cap, d), key=mx.random.key(4)).astype(mx.float16)
    v = mx.random.normal((1, hk, cap, d), key=mx.random.key(5)).astype(mx.float16)
    q = (mx.random.normal((1, hq, q_len, d), key=mx.random.key(6)) * 0.5).astype(mx.float16)

    def run(ql, nsg):
        return sdpa_gqa_mma_prefill(
            queries=q, keys=k, values=v, prefix=prefix, scale=d ** -0.5,
            num_kv_heads=hk, block_positions=ql, simdgroups=nsg,
        )

    ref = run(4, 4)
    out = run(positions, simdgroups)
    assert out is not None and ref is not None
    err = mx.max(mx.abs(out.astype(mx.float32) - ref.astype(mx.float32))).item()
    assert err < 1e-2, err
