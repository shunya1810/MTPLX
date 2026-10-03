"""fp16 partial numerators in the two quantized attention kernels (#526 review).

Both kv_quant kernels (the q8 paged two-pass kernel and the q8/q4 packed-quant
bank kernel) accumulate each block's softmax numerator in fp32, then stored it
UNNORMALIZED in the query dtype before the shared fp32 reducer divides by the
exp-sums. With fp16 activations (the FP16 packs, fp16 ternary packs) a block
of 16 equal rows of 8192 stores 131072 -> inf, and when another block's max is
~200 higher the reducer's ``exp(m_block - m_global) * partial`` is 0 * inf =
NaN in every output element. The fix stores those numerators in fp32 for fp16
queries; bf16 keeps float32's exponent range and is unchanged.

This is not the #526 route itself (the reporter's pack is BF16 on an M3); it
is the second defect the static review of #526 found in the same kernels.
"""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from mtplx.kv_quant import dequantize_symmetric, quantize_symmetric  # noqa: E402

METAL = mx.metal.is_available()


# --- NumPy replay of the kernels' split arithmetic ---------------------------


def _np_quantize(x: np.ndarray, bits: int) -> tuple[np.ndarray, np.ndarray]:
    """kv_quant.quantize_symmetric on host: integer rows + one fp32 scale per row."""
    qmax = 127.0 if bits == 8 else 7.0
    scale = np.maximum(np.abs(x).max(axis=-1, keepdims=True) / qmax, 1e-6).astype(np.float32)
    ints = np.clip(np.round(x / scale), -qmax, qmax).astype(np.float32)
    return ints, scale


def _replay_split(q, k_int, k_scale, v_int, v_scale, *, blocks, partial_dtype):
    """Pass one per block (strided rows, online softmax, the partial store under
    test) and the shared reducer, in the kernels' order of operations: the row
    scale multiplies the integer dot product once, and folds into the
    exp-weight before the value FMA."""
    rows, width = v_int.shape
    lowest = np.finfo(np.float32).min
    maxs = np.full(blocks, lowest, np.float32)
    sums = np.zeros(blocks, np.float32)
    partials = np.zeros((blocks, width), partial_dtype)
    for b in range(blocks):
        m, s = np.float32(lowest), np.float32(0.0)
        o = np.zeros(width, np.float32)
        for n in range(b, rows, blocks):
            score = np.float32(q @ k_int[n]) * k_scale[n, 0]
            new_m = np.float32(max(m, score))
            factor = np.exp(np.float32(m - new_m))
            weight = np.exp(np.float32(score - new_m))
            s = s * factor + weight
            o = o * factor + (weight * v_scale[n, 0]) * v_int[n]
            m = new_m
        maxs[b], sums[b] = m, s
        partials[b] = o.astype(partial_dtype)  # the store: InT before, PartT now
    top = maxs.max()
    factors = np.exp(maxs - top).astype(np.float32)
    total = np.float32((factors * sums).sum(dtype=np.float32))
    acc = (factors[:, None] * partials.astype(np.float32)).sum(axis=0, dtype=np.float32)
    return (acc / total).astype(np.float16)  # the normalized output in InT


def _overflow_fixture(width: int, rows: int):
    """Every row scores 0 and carries v = 8192, except row 1: score 200, small v."""
    q = np.zeros(width, np.float32)
    q[0] = 1.0
    keys = np.zeros((rows, width), np.float32)
    keys[1, 0] = 200.0
    values = np.full((rows, width), 8192.0, np.float32)
    values[1] = np.linspace(-4.0, 4.0, width, dtype=np.float32)
    return q, keys, values


@pytest.mark.parametrize("bits", [8, 4])
def test_split_reduction_replay_fp16_partials_nan_and_product_dtype_exact(bits):
    q, keys, values = _overflow_fixture(width=8, rows=34)
    k_int, k_scale = _np_quantize(keys, bits)
    v_int, v_scale = _np_quantize(values, bits)
    # Unsplit reference over the same dequantized rows (float64 softmax).
    scores = (k_int * k_scale) @ q
    assert scores[1] == pytest.approx(200.0, rel=1e-5) and scores[0] == 0.0
    weights = np.exp((scores - scores.max()).astype(np.float64))
    reference = ((weights / weights.sum()) @ (v_int * v_scale).astype(np.float64)).astype(np.float16)
    assert np.isfinite(reference).all()

    with np.errstate(over="ignore", invalid="ignore"):
        old = _replay_split(q, k_int, k_scale, v_int, v_scale, blocks=2, partial_dtype=np.float16)
    # Block 0 holds 17 rows of 8192: 139264 stored as fp16 is inf, and the
    # reducer weighs it by exp(0 - 200) = 0.
    assert np.isnan(old).all()

    # The product's storage dtype for fp16 queries. Old code: ImportError
    # (no unnormalized_partials_dtype; the store was always the query dtype).
    from mtplx.kernels.sdpa_2pass_paged import unnormalized_partials_dtype

    product = np.dtype(str(unnormalized_partials_dtype(mx.float16)).removeprefix("mlx.core."))
    new = _replay_split(q, k_int, k_scale, v_int, v_scale, blocks=2, partial_dtype=product)
    assert np.array_equal(new, reference)


def test_partial_dtype_policy_keeps_bf16_and_widens_fp16_only():
    from mtplx.kernels.sdpa_2pass_paged import unnormalized_partials_dtype

    assert unnormalized_partials_dtype(mx.float16) == mx.float32
    assert unnormalized_partials_dtype(mx.bfloat16) == mx.bfloat16
    assert unnormalized_partials_dtype(mx.float32) == mx.float32


# --- Real Metal dispatch ---------------------------------------------------------


def _mx_overflow_rows(head_dim: int, rows: int, bits: int):
    q, keys, values = _overflow_fixture(head_dim, rows)
    k_q, k_s = quantize_symmetric(mx.array(keys), bits=bits)
    v_q, v_s = quantize_symmetric(mx.array(values), bits=bits)
    v_row1 = dequantize_symmetric(v_q[1:2], v_s[1:2], bits=bits, head_dim=head_dim)
    mx.eval(k_q, k_s, v_q, v_s, v_row1)
    return q, (k_q, k_s, v_q, v_s), np.array(v_row1)[0]


@pytest.mark.skipif(not METAL, reason="requires Metal")
def test_q8_paged_two_pass_kernel_fp16_overflow_fixture_is_finite(monkeypatch):
    from mtplx.kernels.sdpa_2pass_paged_q8 import sdpa_2pass_paged_q8_tail

    # 32 blocks over 320 rows: every block but the one holding row 1 sums ten
    # rows of 8192 (81920 > 65504) at score 0, 200 below the global max.
    monkeypatch.setenv("MTPLX_SDPA_2PASS_BLOCKS", "32")
    head_dim, rows, page = 64, 320, 16
    q, (k_q, k_s, v_q, v_s), want = _mx_overflow_rows(head_dim, rows, bits=8)

    def pages(x):
        return x.reshape(rows // page, page, 1, x.shape[-1])

    out = sdpa_2pass_paged_q8_tail(
        queries=mx.array(q).astype(mx.float16).reshape(1, 1, 1, head_dim),
        key_q=pages(k_q),
        key_scales=pages(k_s)[..., 0],
        value_q=pages(v_q),
        value_scales=pages(v_s)[..., 0],
        offset=rows,
        block_size=page,
        scale=1.0,
    )
    assert out is not None and out.dtype == mx.float16
    got = np.array(out.astype(mx.float32)).reshape(-1)
    # Old code: every element NaN.
    assert np.isfinite(got).all()
    assert np.abs(got - want).max() < 1e-2


@pytest.mark.skipif(not METAL, reason="requires Metal")
@pytest.mark.parametrize("bits", [8, 4])
def test_packed_quant_kernel_fp16_overflow_fixture_is_finite(bits, monkeypatch):
    from mtplx.kernels.sdpa_gqa_packed_quant import sdpa_gqa_packed_tail_quant

    monkeypatch.setenv("MTPLX_GQA_PACKED_SDPA_BLOCKS", "32")
    head_dim, rows = 256, 320  # q4 is enveloped to head dim 256
    q, (k_q, k_s, v_q, v_s), want = _mx_overflow_rows(head_dim, rows, bits=bits)

    def bank(x):
        return x[None, None, ...]  # (1, H_kv=1, rows, width)

    out = sdpa_gqa_packed_tail_quant(
        queries=mx.array(q).astype(mx.float16).reshape(1, 1, 1, head_dim),
        k_q=bank(k_q),
        k_scale=bank(k_s),
        v_q=bank(v_q),
        v_scale=bank(v_s),
        offset=rows,
        scale=1.0,
        bits=bits,
    )
    assert out is not None and out.dtype == mx.float16
    got = np.array(out.astype(mx.float32)).reshape(-1)
    # Old code: every element NaN.
    assert np.isfinite(got).all()
    assert np.abs(got - want).max() < 1e-2


def _ref_tail_causal(q, k, v, scale):
    qf = q.astype(mx.float32)
    kf = mx.repeat(k.astype(mx.float32), qf.shape[1] // k.shape[1], axis=1)
    vf = mx.repeat(v.astype(mx.float32), qf.shape[1] // v.shape[1], axis=1)
    n_kv, q_len = kf.shape[2], qf.shape[2]
    scores = (qf * scale) @ kf.transpose(0, 1, 3, 2)
    mask = (mx.arange(n_kv)[None, :] <= mx.arange(n_kv - q_len, n_kv)[:, None])[None, None]
    scores = mx.where(mask, scores, mx.full(scores.shape, -1e30))
    return mx.softmax(scores, axis=-1) @ vf


@pytest.mark.skipif(not METAL, reason="requires Metal")
@pytest.mark.parametrize("bits", [8, 4])
@pytest.mark.parametrize("q_len", [1, 4])
@pytest.mark.parametrize("offset", [515, 2051])
def test_packed_quant_kernel_fp16_matches_fp32_reference(bits, q_len, offset):
    # The fp32-partial path on real dispatch, random fp16 rows, 27B geometry.
    hq, hk, d = 24, 4, 256
    mx.random.seed(offset * 10 + q_len + bits)
    capacity = offset + 37
    q = mx.random.normal((1, hq, q_len, d)).astype(mx.float16)
    k_q, k_s = quantize_symmetric(mx.random.normal((1, hk, capacity, d)).astype(mx.float16), bits=bits)
    v_q, v_s = quantize_symmetric(mx.random.normal((1, hk, capacity, d)).astype(mx.float16), bits=bits)
    k_deq = dequantize_symmetric(k_q, k_s, bits=bits, head_dim=d)[..., :offset, :]
    v_deq = dequantize_symmetric(v_q, v_s, bits=bits, head_dim=d)[..., :offset, :]
    ref = _ref_tail_causal(q, k_deq, v_deq, d**-0.5)
    out = _packed_quant(q, k_q, k_s, v_q, v_s, offset, d, bits)
    assert out.dtype == mx.float16
    assert float(mx.max(mx.abs(out.astype(mx.float32) - ref)).item()) < 2e-2


def _packed_quant(q, k_q, k_s, v_q, v_s, offset, d, bits):
    from mtplx.kernels.sdpa_gqa_packed_quant import sdpa_gqa_packed_tail_quant

    out = sdpa_gqa_packed_tail_quant(
        queries=q, k_q=k_q, k_scale=k_s, v_q=v_q, v_scale=v_s,
        offset=offset, scale=d**-0.5, bits=bits,
    )
    assert out is not None, "packed-quant kernel bailed"
    return out


@pytest.mark.skipif(not METAL, reason="requires Metal")
@pytest.mark.parametrize("q_len", [1, 4])
@pytest.mark.parametrize("offset", [230, 1009])
def test_q8_paged_two_pass_kernel_fp16_matches_fp32_reference(q_len, offset):
    from mtplx.kernels.sdpa_2pass_paged_q8 import sdpa_2pass_paged_q8_tail

    hq, hk, d, page = 16, 4, 128, 16
    pages_n = (offset + page - 1) // page + 1
    mx.random.seed(offset + q_len)
    q = mx.random.normal((1, hq, q_len, d)).astype(mx.float16)
    k_q, k_s = quantize_symmetric(mx.random.normal((pages_n, page, hk, d)).astype(mx.float16), bits=8)
    v_q, v_s = quantize_symmetric(mx.random.normal((pages_n, page, hk, d)).astype(mx.float16), bits=8)

    def head_major(ints, scales):
        rows = dequantize_symmetric(ints, scales, bits=8, head_dim=d).reshape(-1, hk, d)
        return rows[:offset].transpose(1, 0, 2)[None]

    ref = _ref_tail_causal(q, head_major(k_q, k_s), head_major(v_q, v_s), d**-0.5)
    out = sdpa_2pass_paged_q8_tail(
        queries=q, key_q=k_q, key_scales=k_s[..., 0], value_q=v_q,
        value_scales=v_s[..., 0], offset=offset, block_size=page, scale=d**-0.5,
    )
    assert out is not None and out.dtype == mx.float16
    assert float(mx.max(mx.abs(out.astype(mx.float32) - ref)).item()) < 2e-2
