"""fp16 partial numerators in MTPLX's unquantized split attention kernels.

The #526 review found that the two quantized kernels stored each block's
UNNORMALIZED softmax numerator in the query dtype before the float32 reducer
divided by the exp-sums (tests/test_quant_partials_fp16_526.py). The same
store sits in every MTPLX split kernel over unquantized KV: the dense and
paged two-pass kernels, the packed and grouped GQA verify kernels and the
three NAX flash kernels. With fp16 activations a block that sums 8 or more
rows of 8192 stores inf (fp16 tops out at 65504), and when another block's
max is ~200 higher the reducer's exp(m_block - m_global) * partial is
0 * inf = NaN in every output element. The partials are float32 for fp16
queries now (sdpa_2pass.unnormalized_partials_dtype); bf16 has float32's
exponent range and keeps its bf16 partials.

MLX's own fused sdpa_vector two-pass kernel is outside this package and is
not changed here.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

METAL = mx.metal.is_available()


def _nax_ready() -> bool:
    if not METAL:
        return False
    from mtplx.kernels.sdpa_nax_flash import _nax_flash_kernel
    from mtplx.nax_verify import nax_available

    return bool(nax_available()) and _nax_flash_kernel() is not None


NAX = _nax_ready()
requires_metal = pytest.mark.skipif(not METAL, reason="requires Metal")
requires_nax = pytest.mark.skipif(not NAX, reason="requires the TensorOps (NAX) kernels")


def _overflow(head_dim: int, rows: int, *, q_rows: int = 1):
    """One query head, one KV head. Every KV row scores 0 and carries v = 8192,
    except row 1: score 200 and a small v, which is therefore the answer."""

    q = np.zeros((1, 1, q_rows, head_dim), np.float32)
    q[..., 0] = 1.0
    k = np.zeros((1, 1, rows, head_dim), np.float32)
    k[:, :, 1, 0] = 200.0
    v = np.full((1, 1, rows, head_dim), 8192.0, np.float32)
    row1 = np.linspace(-4.0, 4.0, head_dim, dtype=np.float32)
    v[:, :, 1, :] = row1
    arrays = [mx.array(x).astype(mx.float16) for x in (q, k, v)]
    mx.eval(*arrays)
    return (*arrays, row1)


def _pages(head_major: mx.array, page: int) -> mx.array:
    """[1, Hk, rows, D] -> [rows / page, page, Hk, D], the paged caches' layout."""
    _, hk, rows, d = head_major.shape
    return mx.contiguous(head_major[0].transpose(1, 0, 2).reshape(rows // page, page, hk, d))


def _assert_answer(out, row1: np.ndarray, q_rows: int) -> None:
    assert out is not None, "the kernel bailed; the fixture must dispatch it"
    assert out.dtype == mx.float16
    got = np.array(out.astype(mx.float32)).reshape(q_rows, -1)
    # Old code: every element NaN.
    assert np.isfinite(got).all()
    assert np.abs(got - row1[None, :]).max() < 1e-2


@requires_metal
def test_dense_two_pass_kernel_fp16_overflow_fixture_is_finite(monkeypatch):
    from mtplx.kernels.sdpa_2pass import sdpa_2pass_tail

    # 32 strided blocks over 320 rows: ten rows of 8192 per block (81920).
    monkeypatch.setenv("MTPLX_SDPA_2PASS_BLOCKS", "32")
    q, k, v, row1 = _overflow(128, 320)
    out = sdpa_2pass_tail(queries=q, keys=k, values=v, scale=1.0)
    _assert_answer(out, row1, 1)


@requires_metal
@pytest.mark.parametrize("offset_kind", ["int", "array"])
def test_paged_two_pass_kernels_fp16_overflow_fixture_is_finite(offset_kind, monkeypatch):
    from mtplx.kernels.sdpa_2pass_paged import (
        sdpa_2pass_paged_tail,
        sdpa_2pass_paged_tail_dynamic_offset,
    )

    monkeypatch.setenv("MTPLX_SDPA_2PASS_BLOCKS", "32")
    q, k, v, row1 = _overflow(64, 320)
    args = dict(queries=q, key_cache=_pages(k, 16), value_cache=_pages(v, 16), block_size=16, scale=1.0)
    if offset_kind == "int":
        out = sdpa_2pass_paged_tail(offset=320, **args)
    else:
        out = sdpa_2pass_paged_tail_dynamic_offset(offset=mx.array(320, dtype=mx.int32), **args)
    _assert_answer(out, row1, 1)


@requires_metal
def test_packed_gqa_kernel_fp16_overflow_fixture_is_finite(monkeypatch):
    from mtplx.kernels.sdpa_gqa_packed import sdpa_gqa_packed_tail

    monkeypatch.setenv("MTPLX_GQA_PACKED_SDPA_BLOCKS", "32")
    q, k, v, row1 = _overflow(64, 320, q_rows=2)
    out = sdpa_gqa_packed_tail(queries=q, keys=k, values=v, offset=320, scale=1.0)
    _assert_answer(out, row1, 2)


@requires_metal
def test_grouped_packed_gqa_kernel_fp16_overflow_fixture_is_finite():
    from mtplx.kernels.sdpa_gqa_packed import sdpa_gqa_packed_tail_grouped

    # The grouped kernel walks at least 256 blocks: 2560 rows is ten per block.
    q, k, v, row1 = _overflow(64, 2560, q_rows=6)
    out = sdpa_gqa_packed_tail_grouped(queries=q, keys=k, values=v, offset=2560, scale=1.0)
    _assert_answer(out, row1, 6)


@requires_nax
@pytest.mark.parametrize("kernel", ["flash", "flash_dsplit", "tile"])
def test_nax_flash_kernels_fp16_overflow_fixture_is_finite(kernel, monkeypatch):
    from mtplx.kernels.sdpa_nax_flash import sdpa_nax_flash
    from mtplx.kernels.sdpa_nax_flash_dsplit import sdpa_nax_flash_dsplit
    from mtplx.kernels.sdpa_nax_tile import sdpa_nax_tile

    # 32 blocks over 320 rows: contiguous 32-row chunks, 262144 per chunk.
    monkeypatch.setenv("MTPLX_NAX_FLASH_BLOCKS", "32")
    monkeypatch.setenv("MTPLX_NAX_FLASH_KS", "1")
    monkeypatch.setenv("MTPLX_NAX_FLASH_DSPLIT_BLOCKS", "32")
    monkeypatch.setenv("MTPLX_GQA_PACKED_SDPA_BLOCKS", "32")  # the tile kernel's block count
    q, k, v, row1 = _overflow(256, 320, q_rows=2)
    fn = {"flash": sdpa_nax_flash, "flash_dsplit": sdpa_nax_flash_dsplit, "tile": sdpa_nax_tile}[kernel]
    out = fn(queries=q, keys=k, values=v, offset=320, scale=1.0)
    _assert_answer(out, row1, 2)


# --- ordinary fp16 inputs against an fp32 reference ----------------------------


def _ref_tail_causal(q, k, v, offset, scale):
    q_len = q.shape[2]
    gqa = q.shape[1] // k.shape[1]
    kf = mx.repeat(k[:, :, :offset, :].astype(mx.float32), gqa, axis=1)
    vf = mx.repeat(v[:, :, :offset, :].astype(mx.float32), gqa, axis=1)
    scores = (q.astype(mx.float32) * scale) @ kf.transpose(0, 1, 3, 2)
    mask = mx.arange(offset)[None, :] <= mx.arange(offset - q_len, offset)[:, None]
    scores = mx.where(mask[None, None], scores, mx.full(scores.shape, -1e30))
    return mx.softmax(scores, axis=-1) @ vf


def _random(hq, hk, q_len, capacity, d, seed):
    mx.random.seed(seed)
    q = mx.random.normal((1, hq, q_len, d)).astype(mx.float16)
    k = mx.random.normal((1, hk, capacity, d)).astype(mx.float16)
    v = mx.random.normal((1, hk, capacity, d)).astype(mx.float16)
    mx.eval(q, k, v)
    return q, k, v


def _dispatch(kernel: str, q, k, v, offset: int, scale: float):
    from mtplx.kernels import sdpa_2pass, sdpa_2pass_paged, sdpa_gqa_packed

    if kernel == "dense_two_pass":
        return sdpa_2pass.sdpa_2pass_tail(
            queries=q, keys=k[:, :, :offset, :], values=v[:, :, :offset, :], scale=scale
        )
    if kernel in ("paged", "paged_dynamic_offset"):
        pages = (k.shape[2] + 15) // 16 * 16
        pad = pages - k.shape[2]
        kp = mx.pad(k, [(0, 0), (0, 0), (0, pad), (0, 0)])
        vp = mx.pad(v, [(0, 0), (0, 0), (0, pad), (0, 0)])
        args = dict(queries=q, key_cache=_pages(kp, 16), value_cache=_pages(vp, 16), block_size=16, scale=scale)
        if kernel == "paged":
            return sdpa_2pass_paged.sdpa_2pass_paged_tail(offset=offset, **args)
        return sdpa_2pass_paged.sdpa_2pass_paged_tail_dynamic_offset(
            offset=mx.array(offset, dtype=mx.int32), **args
        )
    if kernel == "packed":
        return sdpa_gqa_packed.sdpa_gqa_packed_tail(queries=q, keys=k, values=v, offset=offset, scale=scale)
    return sdpa_gqa_packed.sdpa_gqa_packed_tail_grouped(queries=q, keys=k, values=v, offset=offset, scale=scale)


@requires_metal
@pytest.mark.parametrize(
    "kernel", ["dense_two_pass", "paged", "paged_dynamic_offset", "packed", "grouped"]
)
@pytest.mark.parametrize("offset", [515, 2051])
def test_split_kernels_fp16_match_an_fp32_reference(kernel, offset):
    hq, hk, d = 16, 4, 128
    q_len = 1 if kernel in ("dense_two_pass", "paged", "paged_dynamic_offset") else 4
    if kernel == "grouped":
        q_len = 6
    q, k, v = _random(hq, hk, q_len, offset + 37, d, seed=offset + q_len)
    out = _dispatch(kernel, q, k, v, offset, d**-0.5)
    assert out is not None, f"{kernel} bailed"
    assert out.dtype == mx.float16
    ref = _ref_tail_causal(q, k, v, offset, d**-0.5)
    assert float(mx.max(mx.abs(out.astype(mx.float32) - ref)).item()) < 2e-2


@requires_nax
@pytest.mark.parametrize("kernel", ["flash", "flash_dsplit", "tile"])
@pytest.mark.parametrize("offset", [515, 2051])
def test_nax_flash_kernels_fp16_match_an_fp32_reference(kernel, offset):
    from mtplx.kernels.sdpa_nax_flash import sdpa_nax_flash
    from mtplx.kernels.sdpa_nax_flash_dsplit import sdpa_nax_flash_dsplit
    from mtplx.kernels.sdpa_nax_tile import sdpa_nax_tile

    hq, hk, d, q_len = 24, 4, 256, 4  # the 27B verify shape
    q, k, v = _random(hq, hk, q_len, offset + 37, d, seed=offset + 5)
    fn = {"flash": sdpa_nax_flash, "flash_dsplit": sdpa_nax_flash_dsplit, "tile": sdpa_nax_tile}[kernel]
    out = fn(queries=q, keys=k, values=v, offset=offset, scale=1.0 / math.sqrt(d))
    assert out is not None, f"{kernel} bailed"
    assert out.dtype == mx.float16
    ref = _ref_tail_causal(q, k, v, offset, 1.0 / math.sqrt(d))
    assert float(mx.max(mx.abs(out.astype(mx.float32) - ref)).item()) < 2e-2


def test_policy_lives_in_the_base_module_and_the_paged_module_reexports_it():
    from mtplx.kernels import sdpa_2pass, sdpa_2pass_paged

    assert sdpa_2pass_paged.unnormalized_partials_dtype is sdpa_2pass.unnormalized_partials_dtype
    assert sdpa_2pass.unnormalized_partials_dtype(mx.float16) == mx.float32
    assert sdpa_2pass.unnormalized_partials_dtype(mx.bfloat16) == mx.bfloat16
