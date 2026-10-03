"""Tiny-fixture exactness gates for the fused QSA Metal selector.

No model weights are loaded here.  The oracle is the v2.10 eager MLX op
chain, including its complete-block causal mask and ``block * 1e-12`` tie
adjustment.  Selection outputs are compared with zero index/validity
tolerance; only the fp32 dot-product reduction receives a rounding tolerance.
For direct helper geometries with fewer than K logical blocks, the oracle pads
to the custom helper's fixed-K contract; production QSAIndexer short-circuits
before either selector at that unreachable dense-equals-sparse boundary.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
import textwrap

import mlx.core as mx
import pytest

import mtplx.kernels.qsa_indexer_select as selector_module
from mtplx.kernels.qsa_indexer_select import (
    qsa_indexer_select_blocks_metal,
    qsa_indexer_select_dense_mask_metal,
    qsa_indexer_select_row_tokens_metal,
)

pytestmark = pytest.mark.skipif(
    not mx.metal.is_available(), reason="custom QSA selector requires Metal"
)


@pytest.mark.parametrize(
    "macos_version,architecture,expected",
    [
        ("26.2", "applegpu_g17s", True),
        ("26.5.1", "applegpu_g17s", True),
        ("27.0", "applegpu_g18p", True),
        ("26.2", "applegpu_g17p", False),
        ("26.2", "applegpu_g16s", False),
        ("26.1.9", "applegpu_g19s", False),
        ("", "applegpu_g17s", False),
        ("26.2", "unknown", False),
        (None, "applegpu_g17s", False),
        ("26.2", None, False),
    ],
)
def test_nax_availability_parser(
    macos_version: str | None, architecture: str | None, expected: bool
):
    assert (
        selector_module._nax_available_for_platform(macos_version, architecture)
        is expected
    )


def test_nax_detection_reads_platform_and_device_info(monkeypatch):
    # Hardware detection only. The rehearsal switch closes this route gate on
    # every Mac by design (tests/test_nax_detect.py covers it), so a suite run
    # with MTPLX_FORCE_GPU_FAMILY_FALLBACK=1 must not reach this check.
    monkeypatch.delenv("MTPLX_FORCE_GPU_FAMILY_FALLBACK", raising=False)
    requested_devices = []
    monkeypatch.setattr(
        selector_module.platform,
        "mac_ver",
        lambda: ("26.2.0", ("", "", ""), "arm64"),
    )
    monkeypatch.setattr(
        selector_module.mx,
        "device_info",
        lambda device: (
            requested_devices.append(device) or {"architecture": "applegpu_g17s"}
        ),
    )
    selector_module._mlx_nax_available.cache_clear()
    try:
        assert selector_module._mlx_nax_available()
        assert selector_module.qsa_indexer_select_nax_available()
        assert requested_devices == [mx.gpu]
        monkeypatch.setattr(
            selector_module.mx,
            "device_info",
            lambda device: (
                requested_devices.append(device) or {"architecture": "applegpu_g16s"}
            ),
        )
        selector_module._mlx_nax_available.cache_clear()
        assert not selector_module.qsa_indexer_select_nax_available()
        assert requested_devices == [mx.gpu, mx.gpu]
    finally:
        selector_module._mlx_nax_available.cache_clear()


def _eager_oracle(
    q: mx.array,
    pooled_backing: mx.array,
    *,
    pos_start: int,
    total_tokens: int,
    block_topk: int,
    compress_ratio: int,
    logical_blocks: int | None = None,
):
    """Mirror v2.10 selection plus the helper's fixed-K output contract."""

    _, rows, _, head_dim = map(int, q.shape)
    logical = (
        int(pooled_backing.shape[1]) if logical_blocks is None else int(logical_blocks)
    )
    pooled = pooled_backing[:, :logical, :]
    if logical:
        pooled_t = mx.swapaxes(pooled.astype(mx.float32), 1, 2)[:, None]
        scores = mx.matmul(q.astype(mx.float32), pooled_t)
        scores = mx.maximum(scores, 0.0).sum(axis=2) / math.sqrt(head_dim)
        scores = scores[0]
    else:
        scores = mx.zeros((rows, 0), dtype=mx.float32)

    qpos = mx.arange(pos_start, pos_start + rows, dtype=mx.int32)
    complete = (qpos + 1) // compress_ratio
    block = mx.arange(logical, dtype=mx.int32)
    visible = block[None, :] < complete[:, None]
    adjusted = mx.where(
        visible,
        scores,
        mx.array(-mx.inf, dtype=mx.float32),
    )
    adjusted = adjusted - block.astype(mx.float32)[None, :] * 1e-12

    k_eff = min(block_topk, logical)
    if k_eff:
        # This is exactly v2.10's argpartition direction/slice. Invalid -inf
        # entries may occupy padding slots when visible<K; validity removes
        # them before any production epilogue observes the set.
        top_idx = mx.argpartition(
            adjusted,
            kth=logical - k_eff,
            axis=-1,
        )[:, logical - k_eff :]
        top_valid = mx.take_along_axis(visible, top_idx.astype(mx.int64), axis=-1)
        top_scores = mx.take_along_axis(adjusted, top_idx, axis=-1)
        top_idx = top_idx.astype(mx.int32)
    else:
        top_idx = mx.zeros((rows, 0), dtype=mx.int32)
        top_valid = mx.zeros((rows, 0), dtype=mx.bool_)
        top_scores = mx.zeros((rows, 0), dtype=mx.float32)

    if k_eff < block_topk:
        pad = block_topk - k_eff
        top_idx = mx.concatenate(
            [top_idx, mx.zeros((rows, pad), dtype=mx.int32)], axis=1
        )
        top_valid = mx.concatenate(
            [top_valid, mx.zeros((rows, pad), dtype=mx.bool_)], axis=1
        )
        top_scores = mx.concatenate(
            [
                top_scores,
                mx.full((rows, pad), -mx.inf, dtype=mx.float32),
            ],
            axis=1,
        )

    # Block mode deliberately canonicalizes by block id. Rows mode below keeps
    # v2.10's raw GPU ArgPartition order: MLX currently performs a stable full
    # ascending sort and the model consumes its final K-index slice directly.
    row_block_ids = top_idx
    row_block_valid = top_valid
    sentinel = mx.array(2**31 - 1, dtype=mx.int32)
    id_sort_key = mx.where(top_valid, top_idx, sentinel)
    id_order = mx.argsort(id_sort_key, axis=-1)
    selected_ids = mx.take_along_axis(top_idx, id_order, axis=-1)
    selected_valid = mx.take_along_axis(top_valid, id_order, axis=-1)
    selected_scores = mx.take_along_axis(top_scores, id_order, axis=-1)
    selected_ids = mx.where(selected_valid, selected_ids, mx.array(0, dtype=mx.int32))
    selected_scores = mx.where(
        selected_valid,
        selected_scores,
        mx.array(-mx.inf, dtype=mx.float32),
    )

    token_pos = mx.arange(total_tokens, dtype=mx.int32)
    selected_token = mx.any(
        (selected_ids[:, :, None] == (token_pos // compress_ratio)[None, None, :])
        & selected_valid[:, :, None],
        axis=1,
    )
    tail = token_pos[None, :] >= complete[:, None] * compress_ratio
    causal = token_pos[None, :] <= qpos[:, None]
    dense_mask = ((selected_token | tail) & causal)[None, None]

    block_tokens = (
        row_block_ids[:, :, None] * compress_ratio
        + mx.arange(compress_ratio, dtype=mx.int32)[None, None, :]
    ).reshape(rows, block_topk * compress_ratio)
    block_token_valid = mx.broadcast_to(
        row_block_valid[:, :, None],
        (rows, block_topk, compress_ratio),
    ).reshape(rows, block_topk * compress_ratio)
    tail_tokens = (
        complete[:, None] * compress_ratio
        + mx.arange(compress_ratio, dtype=mx.int32)[None, :]
    )
    tail_valid = tail_tokens <= qpos[:, None]
    row_tokens = mx.concatenate([block_tokens, tail_tokens], axis=1)
    row_valid = mx.concatenate([block_token_valid, tail_valid], axis=1)
    row_tokens = mx.where(row_valid, row_tokens, mx.array(0, dtype=mx.int32))

    return (
        selected_ids,
        selected_valid,
        selected_scores,
        dense_mask,
        row_tokens,
        row_valid,
    )


def _fixture(
    seed: int,
    *,
    rows: int,
    heads: int,
    head_dim: int,
    backing_blocks: int,
    dtype: mx.Dtype = mx.float32,
) -> tuple[mx.array, mx.array]:
    mx.random.seed(seed)
    q = mx.contiguous(mx.random.normal((1, rows, heads, head_dim)).astype(dtype))
    pooled = mx.contiguous(
        mx.random.normal((1, backing_blocks, head_dim)).astype(dtype)
    )
    return q, pooled


def _assert_scores_close(
    actual: mx.array,
    expected: mx.array,
    valid: mx.array,
    *,
    atol: float = 5e-5,
) -> None:
    mx.eval(actual, expected, valid)
    delta = mx.where(valid, mx.abs(actual - expected), 0.0)
    maximum = 0.0 if int(delta.size) == 0 else float(mx.max(delta).item())
    assert maximum <= atol, f"selected adjusted-score max abs {maximum} > {atol}"
    actual_rows = actual.tolist()
    valid_rows = valid.tolist()
    for score_row, valid_row in zip(actual_rows, valid_rows, strict=True):
        for score, ok in zip(score_row, valid_row, strict=True):
            if not ok:
                assert math.isinf(score) and score < 0


def _assert_all_modes(
    q: mx.array,
    pooled: mx.array,
    *,
    pos_start: int,
    total_tokens: int,
    block_topk: int,
    compress_ratio: int,
    logical_blocks: int | None = None,
    score_atol: float = 5e-5,
) -> None:
    kwargs = {
        "pos_start": pos_start,
        "total_tokens": total_tokens,
        "block_topk": block_topk,
        "compress_ratio": compress_ratio,
        "logical_blocks": logical_blocks,
    }
    expected = _eager_oracle(q, pooled, **kwargs)
    ids, valid, selected_scores = qsa_indexer_select_blocks_metal(q, pooled, **kwargs)
    dense = qsa_indexer_select_dense_mask_metal(q, pooled, **kwargs)
    token_ids, token_valid = qsa_indexer_select_row_tokens_metal(q, pooled, **kwargs)
    mx.eval(ids, valid, selected_scores, dense, token_ids, token_valid, *expected)

    exp_ids, exp_valid, exp_scores, exp_dense, exp_tokens, exp_token_valid = expected
    assert ids.dtype == mx.int32
    assert valid.dtype == mx.bool_
    assert selected_scores.dtype == mx.float32
    assert dense.dtype == mx.bool_
    assert token_ids.dtype == mx.int32
    assert token_valid.dtype == mx.bool_
    assert tuple(dense.shape) == (1, 1, int(q.shape[1]), total_tokens)
    assert tuple(token_ids.shape) == (
        int(q.shape[1]),
        block_topk * compress_ratio + compress_ratio,
    )

    assert ids.tolist() == exp_ids.tolist()
    assert valid.tolist() == exp_valid.tolist()
    assert dense.tolist() == exp_dense.tolist()
    assert token_ids.tolist() == exp_tokens.tolist()
    assert token_valid.tolist() == exp_token_valid.tolist()
    _assert_scores_close(selected_scores, exp_scores, exp_valid, atol=score_atol)


@pytest.mark.parametrize(
    "seed,rows,pos_start,total_tokens",
    [
        # N<K, N==K, an incomplete tail, and the first N>K sparse row.
        (1, 1, 10, 11),
        (2, 2, 10, 12),
        (3, 2, 11, 13),
        (4, 4, 12, 16),
        # Multi-row prefill/verify shapes crossing several block boundaries.
        (5, 7, 18, 25),
        (6, 9, 28, 37),
    ],
)
def test_all_modes_match_eager_around_block_and_dense_boundaries(
    seed: int, rows: int, pos_start: int, total_tokens: int
):
    ratio = 4
    q, pooled = _fixture(
        seed,
        rows=rows,
        heads=2,
        head_dim=8,
        backing_blocks=total_tokens // ratio,
    )
    _assert_all_modes(
        q,
        pooled,
        pos_start=pos_start,
        total_tokens=total_tokens,
        block_topk=3,
        compress_ratio=ratio,
    )


@pytest.mark.parametrize("seed", range(10, 22))
def test_many_random_seeds_are_index_identical(seed: int):
    q, pooled = _fixture(
        seed,
        rows=5,
        heads=3,
        head_dim=12,
        backing_blocks=11,
    )
    _assert_all_modes(
        q,
        pooled,
        pos_start=29,
        total_tokens=34,
        block_topk=4,
        compress_ratio=3,
        score_atol=8e-5,
    )


def test_exact_zero_relu_ties_always_choose_lowest_blocks():
    rows, heads, head_dim = 6, 3, 8
    q = mx.zeros((1, rows, heads, head_dim), dtype=mx.float32)
    _, pooled = _fixture(
        90,
        rows=rows,
        heads=heads,
        head_dim=head_dim,
        backing_blocks=9,
    )
    _assert_all_modes(
        q,
        pooled,
        pos_start=31,
        total_tokens=37,
        block_topk=4,
        compress_ratio=4,
        score_atol=0.0,
    )
    ids, valid, scores = qsa_indexer_select_blocks_metal(
        q,
        pooled,
        pos_start=31,
        total_tokens=37,
        block_topk=4,
        compress_ratio=4,
    )
    mx.eval(ids, valid, scores)
    assert ids.tolist() == [[0, 1, 2, 3]] * rows
    assert valid.tolist() == [[True, True, True, True]] * rows

    # v2.10's rows-gather path consumes the stable ascending-score tail of
    # MLX's GPU ArgPartition directly.  The selected set is the same lowest-id
    # tie winner, but its block-token groups run from worst to best score.
    token_ids, token_valid = qsa_indexer_select_row_tokens_metal(
        q,
        pooled,
        pos_start=31,
        total_tokens=37,
        block_topk=4,
        compress_ratio=4,
    )
    mx.eval(token_ids, token_valid)
    assert token_ids[0].tolist() == [
        12,
        13,
        14,
        15,
        8,
        9,
        10,
        11,
        4,
        5,
        6,
        7,
        0,
        1,
        2,
        3,
        0,
        0,
        0,
        0,
    ]
    assert token_valid[0].tolist() == [True] * 16 + [False] * 4


def test_rounded_away_positive_tie_matches_eager_stable_cutoff():
    """When epsilon rounds away, v2.10's stable final-K keeps later ids."""

    heads, head_dim, logical, topk = 4, 128, 1_025, 17
    q = mx.ones((1, 1, heads, head_dim), dtype=mx.float32)
    pooled = mx.ones((1, logical, head_dim), dtype=mx.float32)
    kwargs = {
        "pos_start": logical * 4 - 1,
        "total_tokens": logical * 4,
        "logical_blocks": logical,
        "block_topk": topk,
        "compress_ratio": 4,
    }
    _assert_all_modes(q, pooled, **kwargs, score_atol=0.0)
    ids, valid, scores = qsa_indexer_select_blocks_metal(
        q,
        pooled,
        **kwargs,
    )
    mx.eval(ids, valid, scores)
    assert ids.tolist() == [list(range(logical - topk, logical))]
    assert valid.tolist() == [[True] * topk]
    # At a score of ~45, subtracting <=1.6e-11 is below one float32 ULP.
    # Exact eager parity therefore requires the sort's stable cutoff behavior,
    # not a stronger semantic than the float32 adjustment actually provides.
    assert len(set(scores[0].tolist())) == 1


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16, mx.float32])
def test_common_mlx_input_dtypes(dtype: mx.Dtype):
    q, pooled = _fixture(
        101,
        rows=3,
        heads=2,
        head_dim=8,
        backing_blocks=5,
        dtype=dtype,
    )
    _assert_all_modes(
        q,
        pooled,
        pos_start=17,
        total_tokens=20,
        block_topk=3,
        compress_ratio=4,
        score_atol=8e-5,
    )


@pytest.mark.parametrize(
    "seed,logical,block_topk",
    [
        *[(seed, 700, 512) for seed in range(120, 126)],
        *[(seed, 1_025, 64) for seed in range(130, 136)],
    ],
)
def test_float32_tf32_cutoff_sets_match_eager(seed: int, logical: int, block_topk: int):
    """Adversarial production head geometry around a large top-k cutoff.

    MLX's default float32 GPU matmul truncates operands to a 10-bit mantissa.
    A serial full-fp32 Metal dot mismatches these winner sets on sampled seeds;
    the fused kernel must reproduce MLX's TF32-style operand semantics.
    """

    q, pooled = _fixture(
        seed,
        rows=1,
        heads=4,
        head_dim=128,
        backing_blocks=logical,
        dtype=mx.float32,
    )
    kwargs = {
        "pos_start": logical * 4 - 1,
        "total_tokens": logical * 4,
        "logical_blocks": logical,
        "block_topk": block_topk,
        "compress_ratio": 4,
    }
    exp_ids, exp_valid, exp_scores, *_ = _eager_oracle(q, pooled, **kwargs)
    ids, valid, scores = qsa_indexer_select_blocks_metal(q, pooled, **kwargs)
    mx.eval(ids, valid, scores, exp_ids, exp_valid, exp_scores)
    assert ids.tolist() == exp_ids.tolist()
    assert valid.tolist() == exp_valid.tolist()
    _assert_scores_close(scores, exp_scores, exp_valid, atol=1e-5)


@pytest.mark.parametrize("seed", [9, 18, 29])
def test_float32_single_query_head_uses_full_precision_gemv(seed: int):
    """S*H==1 takes MLX's full-fp32 GEMV route, even on a NAX host."""

    logical = 700
    block_topk = 512
    q, pooled = _fixture(
        seed,
        rows=1,
        heads=1,
        head_dim=128,
        backing_blocks=logical,
        dtype=mx.float32,
    )
    kwargs = {
        "pos_start": logical * 4 - 1,
        "total_tokens": logical * 4,
        "logical_blocks": logical,
        "block_topk": block_topk,
        "compress_ratio": 4,
    }
    exp_ids, exp_valid, exp_scores, *_ = _eager_oracle(q, pooled, **kwargs)
    ids, valid, scores = qsa_indexer_select_blocks_metal(q, pooled, **kwargs)
    mx.eval(ids, valid, scores, exp_ids, exp_valid, exp_scores)
    assert ids.tolist() == exp_ids.tolist()
    assert valid.tolist() == exp_valid.tolist()
    _assert_scores_close(scores, exp_scores, exp_valid, atol=1e-5)


@pytest.mark.parametrize("seed", [3, 16])
def test_float32_strided_single_head_batch_uses_gemv(seed: int):
    """A gapped S stride prevents MLX from collapsing S into matmul M."""

    logical = 700
    mx.random.seed(seed)
    q_full = mx.random.normal((1, 4, 1, 128)).astype(mx.float32)
    q = q_full[:, ::2]
    pooled = mx.random.normal((1, logical, 128)).astype(mx.float32)
    kwargs = {
        "pos_start": logical * 4 - 1,
        "total_tokens": logical * 4 + 1,
        "logical_blocks": logical,
        "block_topk": 512,
        "compress_ratio": 4,
    }
    exp_ids, exp_valid, exp_scores, *_ = _eager_oracle(q, pooled, **kwargs)
    ids, valid, scores = qsa_indexer_select_blocks_metal(q, pooled, **kwargs)
    mx.eval(ids, valid, scores, exp_ids, exp_valid, exp_scores)
    assert ids.tolist() == exp_ids.tolist()
    assert valid.tolist() == exp_valid.tolist()
    _assert_scores_close(scores, exp_scores, exp_valid, atol=1e-5)


def test_float32_copied_broadcast_pooled_view_prevents_batch_collapse():
    """A copied, broadcast B loses the zero batch stride used for collapse."""

    logical = 700
    mx.random.seed(503)
    q = mx.contiguous(mx.random.normal((1, 2, 1, 128)).astype(mx.float32))
    pooled_storage = mx.random.normal((1, logical, 256)).astype(mx.float32)
    pooled = pooled_storage[..., ::2]
    kwargs = {
        "pos_start": logical * 4 - 1,
        "total_tokens": logical * 4 + 1,
        "logical_blocks": logical,
        "block_topk": 512,
        "compress_ratio": 4,
    }
    exp_ids, exp_valid, exp_scores, *_ = _eager_oracle(q, pooled, **kwargs)
    ids, valid, scores = qsa_indexer_select_blocks_metal(q, pooled, **kwargs)
    mx.eval(ids, valid, scores, exp_ids, exp_valid, exp_scores)
    assert ids.tolist() == exp_ids.tolist()
    assert valid.tolist() == exp_valid.tolist()
    _assert_scores_close(scores, exp_scores, exp_valid, atol=1e-5)


def test_float32_single_logical_block_uses_full_precision_gemv():
    """N==1 is the other MLX GEMV boundary after S batches collapse into M."""

    q, pooled = _fixture(
        300,
        rows=2,
        heads=4,
        head_dim=128,
        backing_blocks=1,
        dtype=mx.float32,
    )
    _assert_all_modes(
        q,
        pooled,
        pos_start=3,
        total_tokens=5,
        logical_blocks=1,
        block_topk=1,
        compress_ratio=4,
        score_atol=1e-5,
    )


def test_strided_q_and_pooled_views_match_eager_without_copy_dispatch():
    mx.random.seed(180)
    q_backing = mx.random.normal((1, 3, 2, 16)).astype(mx.float32)
    pooled_backing = mx.random.normal((1, 5, 16)).astype(mx.float32)
    q = q_backing[..., ::2]
    pooled = pooled_backing[..., ::2]
    assert tuple(q.shape) == (1, 3, 2, 8)
    assert tuple(pooled.shape) == (1, 5, 8)
    _assert_all_modes(
        q,
        pooled,
        pos_start=17,
        total_tokens=20,
        block_topk=3,
        compress_ratio=4,
    )


def test_float32_full_precision_specialization_in_fresh_process():
    """MLX caches MLX_ENABLE_TF32, so the off-path needs a fresh process."""

    code = textwrap.dedent(
        """
        import math
        import mlx.core as mx
        from mtplx.kernels.qsa_indexer_select import qsa_indexer_select_blocks_metal

        mx.random.seed(190)
        q = mx.random.normal((1, 1, 4, 128)).astype(mx.float32)
        pooled = mx.random.normal((1, 257, 128)).astype(mx.float32)
        pooled_t = mx.swapaxes(pooled, 1, 2)[:, None]
        score = mx.maximum(mx.matmul(q, pooled_t), 0.0).sum(axis=2)[0] / math.sqrt(128)
        block = mx.arange(257, dtype=mx.int32)
        adjusted = score - block.astype(mx.float32)[None] * 1e-12
        top = mx.argpartition(adjusted, kth=257 - 32, axis=-1)[:, 257 - 32:]
        expected = mx.sort(top.astype(mx.int32), axis=-1)
        ids, valid, selected = qsa_indexer_select_blocks_metal(
            q,
            pooled,
            pos_start=1027,
            total_tokens=1028,
            logical_blocks=257,
            block_topk=32,
            compress_ratio=4,
        )
        mx.eval(ids, valid, selected, expected, adjusted)
        assert ids.tolist() == expected.tolist(), (ids, expected)
        gathered = mx.take_along_axis(adjusted, ids.astype(mx.int64), axis=-1)
        delta = float(mx.max(mx.abs(selected - gathered)).item())
        assert delta <= 2e-5, delta
        assert bool(mx.all(valid).item())
        """
    )
    env = dict(os.environ)
    env["MLX_ENABLE_TF32"] = "0"
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=os.getcwd(),
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_history_longer_than_threadgroup_matches_eager():
    """N is not WIDTH: radix selection streams a 1,301-block history."""

    logical = 1_301
    ratio = 4
    total = logical * ratio + 1
    q, pooled = _fixture(
        202,
        rows=3,
        heads=2,
        head_dim=8,
        backing_blocks=2_048,
    )
    _assert_all_modes(
        q,
        pooled,
        pos_start=total - 3,
        total_tokens=total,
        logical_blocks=logical,
        block_topk=17,
        compress_ratio=ratio,
        score_atol=1e-4,
    )


def test_real_prefill_sized_query_chunk_matches_eager_blocks():
    """A 2,048-row chunk keeps exact winners across block boundaries."""

    rows, logical, ratio, topk = 2_048, 1_024, 4, 32
    total = logical * ratio
    q, pooled = _fixture(
        250,
        rows=rows,
        heads=2,
        head_dim=8,
        backing_blocks=logical,
        dtype=mx.float16,
    )
    kwargs = {
        "pos_start": total - rows,
        "total_tokens": total,
        "logical_blocks": logical,
        "block_topk": topk,
        "compress_ratio": ratio,
    }
    exp_ids, exp_valid, exp_scores, *_ = _eager_oracle(q, pooled, **kwargs)
    ids, valid, scores = qsa_indexer_select_blocks_metal(q, pooled, **kwargs)
    mx.eval(ids, valid, scores, exp_ids, exp_valid, exp_scores)
    assert ids.tolist() == exp_ids.tolist()
    assert valid.tolist() == exp_valid.tolist()
    _assert_scores_close(scores, exp_scores, exp_valid, atol=1e-5)


def test_262k_context_geometry_selects_exact_lowest_zero_ties():
    """65,536 pooled blocks (262,144 tokens at r=4) remain one dispatch."""

    logical = 65_536
    q = mx.zeros((1, 1, 1, 2), dtype=mx.float32)
    pooled = mx.zeros((1, logical, 2), dtype=mx.float32)
    ids, valid, scores = qsa_indexer_select_blocks_metal(
        q,
        pooled,
        pos_start=262_143,
        total_tokens=262_144,
        logical_blocks=logical,
        block_topk=8,
        compress_ratio=4,
    )
    mx.eval(ids, valid, scores)
    assert ids.tolist() == [list(range(8))]
    assert valid.tolist() == [[True] * 8]
    expected_scores = [-(i * 1e-12) for i in range(8)]
    for actual, expected in zip(scores[0].tolist(), expected_scores, strict=True):
        assert abs(actual - expected) <= 1e-18


def test_dynamic_scalar_frontiers_and_dense_capacity_stride():
    rows, ratio, logical, capacity = 3, 4, 9, 48
    total = logical * ratio + 1
    q, pooled = _fixture(
        303,
        rows=rows,
        heads=2,
        head_dim=8,
        backing_blocks=64,
    )
    pos_tensor = mx.array([total - rows], dtype=mx.int32)
    total_tensor = mx.array([total], dtype=mx.int32)
    logical_tensor = mx.array([logical], dtype=mx.int32)
    scalar_kwargs = {
        "pos_start": pos_tensor,
        "total_tokens": total_tensor,
        "logical_blocks": logical_tensor,
        "block_topk": 4,
        "compress_ratio": ratio,
    }

    got_ids, got_valid, got_scores = qsa_indexer_select_blocks_metal(
        q, pooled, **scalar_kwargs
    )
    got_tokens, got_token_valid = qsa_indexer_select_row_tokens_metal(
        q, pooled, **scalar_kwargs
    )
    got_dense = qsa_indexer_select_dense_mask_metal(
        q,
        pooled,
        **scalar_kwargs,
        output_total_tokens=capacity,
    )
    expected = _eager_oracle(
        q,
        pooled,
        pos_start=total - rows,
        total_tokens=total,
        logical_blocks=logical,
        block_topk=4,
        compress_ratio=ratio,
    )
    mx.eval(
        got_ids,
        got_valid,
        got_scores,
        got_tokens,
        got_token_valid,
        got_dense,
        *expected,
    )
    exp_ids, exp_valid, exp_scores, exp_dense, exp_tokens, exp_token_valid = expected
    assert got_ids.tolist() == exp_ids.tolist()
    assert got_valid.tolist() == exp_valid.tolist()
    assert got_tokens.tolist() == exp_tokens.tolist()
    assert got_token_valid.tolist() == exp_token_valid.tolist()
    _assert_scores_close(got_scores, exp_scores, exp_valid)
    assert tuple(got_dense.shape) == (1, 1, rows, capacity)
    assert got_dense[..., :total].tolist() == exp_dense.tolist()
    assert not bool(mx.any(got_dense[..., total:]).item())


def test_compiled_selector_reads_changed_dynamic_frontiers():
    """Scalar tensor values must remain runtime inputs under ``mx.compile``."""

    q, pooled = _fixture(
        350,
        rows=1,
        heads=2,
        head_dim=8,
        backing_blocks=32,
    )

    @mx.compile
    def compiled_select(q_value, pooled_value, pos, total, logical):
        return qsa_indexer_select_blocks_metal(
            q_value,
            pooled_value,
            pos_start=pos,
            total_tokens=total,
            logical_blocks=logical,
            block_topk=4,
            compress_ratio=4,
        )

    for case, logical in enumerate((9, 5, 13)):
        total = logical * 4 + 1
        pos = total - 1
        scalar = (
            (lambda value: mx.array(value, dtype=mx.int32))
            if case == 1
            else (lambda value: mx.array([value], dtype=mx.int32))
        )
        scalar_args = (
            scalar(pos),
            scalar(total),
            scalar(logical),
        )
        ids, valid, scores = compiled_select(q, pooled, *scalar_args)
        exp_ids, exp_valid, exp_scores, *_ = _eager_oracle(
            q,
            pooled,
            pos_start=pos,
            total_tokens=total,
            logical_blocks=logical,
            block_topk=4,
            compress_ratio=4,
        )
        mx.eval(ids, valid, scores, exp_ids, exp_valid, exp_scores)
        assert ids.tolist() == exp_ids.tolist()
        assert valid.tolist() == exp_valid.tolist()
        _assert_scores_close(scores, exp_scores, exp_valid)


def test_query_chunking_is_identical_to_one_call():
    rows, split, ratio, total = 8, 3, 4, 43
    pos_start = 31
    q, pooled = _fixture(
        404,
        rows=rows,
        heads=2,
        head_dim=8,
        backing_blocks=total // ratio,
    )
    common = {
        "total_tokens": total,
        "block_topk": 3,
        "compress_ratio": ratio,
    }

    full_blocks = qsa_indexer_select_blocks_metal(
        q, pooled, pos_start=pos_start, **common
    )
    full_mask = qsa_indexer_select_dense_mask_metal(
        q, pooled, pos_start=pos_start, **common
    )
    full_rows = qsa_indexer_select_row_tokens_metal(
        q, pooled, pos_start=pos_start, **common
    )

    q0 = mx.contiguous(q[:, :split])
    q1 = mx.contiguous(q[:, split:])
    chunks = [(q0, pos_start), (q1, pos_start + split)]
    block_chunks = [
        qsa_indexer_select_blocks_metal(qc, pooled, pos_start=pc, **common)
        for qc, pc in chunks
    ]
    mask_chunks = [
        qsa_indexer_select_dense_mask_metal(qc, pooled, pos_start=pc, **common)
        for qc, pc in chunks
    ]
    row_chunks = [
        qsa_indexer_select_row_tokens_metal(qc, pooled, pos_start=pc, **common)
        for qc, pc in chunks
    ]

    merged_blocks = tuple(
        mx.concatenate([block_chunks[0][i], block_chunks[1][i]], axis=0)
        for i in range(3)
    )
    merged_mask = mx.concatenate(mask_chunks, axis=2)
    merged_rows = tuple(
        mx.concatenate([row_chunks[0][i], row_chunks[1][i]], axis=0) for i in range(2)
    )
    mx.eval(
        *full_blocks, full_mask, *full_rows, *merged_blocks, merged_mask, *merged_rows
    )
    for whole, chunked in zip(full_blocks, merged_blocks, strict=True):
        assert whole.tolist() == chunked.tolist()
    assert full_mask.tolist() == merged_mask.tolist()
    for whole, chunked in zip(full_rows, merged_rows, strict=True):
        assert whole.tolist() == chunked.tolist()


def test_guards_reject_cpu_and_oversized_k_without_fallback():
    q = mx.zeros((1, 1, 1, 4), dtype=mx.float32)
    pooled = mx.zeros((1, 2, 4), dtype=mx.float32)
    with pytest.raises(ValueError, match="block_topk"):
        qsa_indexer_select_blocks_metal(
            q,
            pooled,
            pos_start=7,
            total_tokens=8,
            block_topk=513,
            compress_ratio=4,
        )

    previous = mx.default_device()
    try:
        mx.set_default_device(mx.cpu)
        with pytest.raises(RuntimeError, match="default device"):
            qsa_indexer_select_blocks_metal(
                q,
                pooled,
                pos_start=7,
                total_tokens=8,
                block_topk=1,
                compress_ratio=4,
            )
    finally:
        mx.set_default_device(previous)
