"""Paged-KV variant of MLX vector two-pass SDPA for tiny verify windows.

This is an opt-in verifier-correctness probe.  It keeps the vLLM-style physical
page layout, but matches MLX's vector two-pass reduction topology instead of
using the vLLM-Metal online softmax kernel.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import mlx.core as mx

from .sdpa_2pass import _compute_blocks, unnormalized_partials_dtype


@lru_cache(maxsize=2)
def _paged_partials_kernel(*, has_window: bool = False):
    if not mx.metal.is_available():
        return None

    if has_window:
        loop_header = (
            "for (int rel_n = block_idx; "
            "rel_n < (N - window_start); rel_n += blocks) {"
        )
        loop_prefix = "const int n = window_start + rel_n;"
    else:
        loop_header = "for (int n = block_idx; n < N; n += blocks) {"
        loop_prefix = ""

    source = """
        constexpr int BD = 32;
        constexpr int qk_per_thread = D / BD;
        constexpr int v_per_thread = V / BD;

        typedef float U;

        thread U q[qk_per_thread];
        thread U o[v_per_thread] = {0};

        const int kv_head_idx = threadgroup_position_in_grid.x;
        const int batch_idx = threadgroup_position_in_grid.y;
        const int block_idx = threadgroup_position_in_grid.z;
        const int gqa_factor = threads_per_threadgroup.y;
        const int q_seq_len = threads_per_threadgroup.z;
        const int q_seq_idx = thread_position_in_threadgroup.z;
        const int q_head_idx = gqa_factor * kv_head_idx + thread_position_in_threadgroup.y;
        const int num_kv_heads = threadgroups_per_grid.x;
        const int num_q_heads = num_kv_heads * gqa_factor;
        const int q_batch_head_idx = batch_idx * num_q_heads + q_head_idx;
        const int o_offset = q_batch_head_idx * q_seq_len + q_seq_idx;

        queries += o_offset * D + thread_index_in_simdgroup * qk_per_thread;
        partials += (o_offset * blocks + block_idx) * V
            + thread_index_in_simdgroup * v_per_thread;
        sums += o_offset * blocks + block_idx;
        maxs += o_offset * blocks + block_idx;

        for (int i = 0; i < qk_per_thread; ++i) {
            q[i] = static_cast<U>(scale) * static_cast<U>(queries[i]);
        }

        U max_score = Limits<U>::finite_min;
        U sum_exp_score = 0.0f;

        __LOOP_HEADER__
            __LOOP_PREFIX__
            bool use_key = n <= (N - q_seq_len + q_seq_idx);
            if (use_key) {
                const int page_idx = n / PAGE_SIZE;
                const int page_offset = n - page_idx * PAGE_SIZE;
                const device InT* key_ptr = key_cache
                    + (((page_idx * PAGE_SIZE + page_offset) * Hk + kv_head_idx) * D)
                    + thread_index_in_simdgroup * qk_per_thread;
                const device InT* value_ptr = value_cache
                    + (((page_idx * PAGE_SIZE + page_offset) * Hk + kv_head_idx) * V)
                    + thread_index_in_simdgroup * v_per_thread;

                U score = 0.0f;
                for (int i = 0; i < qk_per_thread; ++i) {
                    score += q[i] * static_cast<U>(key_ptr[i]);
                }
                score = simd_sum(score);

                U new_max = metal::max(max_score, score);
                U factor = fast::exp(max_score - new_max);
                U exp_score = fast::exp(score - new_max);

                max_score = new_max;
                sum_exp_score = sum_exp_score * factor + exp_score;
                for (int i = 0; i < v_per_thread; ++i) {
                    o[i] = o[i] * factor + exp_score * static_cast<U>(value_ptr[i]);
                }
            }
        }

        if (thread_index_in_simdgroup == 0) {
            sums[0] = sum_exp_score;
            maxs[0] = max_score;
        }
        for (int i = 0; i < v_per_thread; ++i) {
            partials[i] = static_cast<PartT>(o[i]);
        }
    """
    source = source.replace("__LOOP_HEADER__", loop_header).replace(
        "__LOOP_PREFIX__",
        loop_prefix,
    )
    input_names = [
        "queries",
        "key_cache",
        "value_cache",
        "N",
    ]
    if has_window:
        input_names.append("window_start")
    input_names.extend(["scale", "blocks"])
    return mx.fast.metal_kernel(
        name=(
            "mtplx_sdpa_2pass_paged_window_partials"
            if has_window
            else "mtplx_sdpa_2pass_paged_partials"
        ),
        input_names=input_names,
        output_names=["partials", "sums", "maxs"],
        source=source,
    )


@lru_cache(maxsize=1)
def _paged_reduce_kernel():
    if not mx.metal.is_available():
        return None

    source = """
        constexpr int BN = 32;
        constexpr int BD = 32;
        constexpr int elem_per_thread = V / BD;

        typedef float U;

        thread U o[elem_per_thread] = {0};
        threadgroup U outputs[BN * BD];

        const int head_idx = threadgroup_position_in_grid.x;
        const int q_seq_idx = threadgroup_position_in_grid.y;
        const int q_offset = head_idx * threadgroups_per_grid.y + q_seq_idx;
        partials += q_offset * blocks * V + simdgroup_index_in_threadgroup * V
            + thread_index_in_simdgroup * elem_per_thread;
        sums += q_offset * blocks;
        maxs += q_offset * blocks;
        out += q_offset * V + simdgroup_index_in_threadgroup * elem_per_thread;

        U sum_exp_score = 0.0f;
        U max_score = Limits<U>::finite_min;

        for (int b = 0; b < blocks / BN; ++b) {
            max_score = metal::max(max_score, maxs[thread_index_in_simdgroup + BN * b]);
        }
        max_score = simd_max(max_score);

        for (int b = 0; b < blocks / BN; ++b) {
            U factor = fast::exp(maxs[thread_index_in_simdgroup + BN * b] - max_score);
            sum_exp_score += factor * sums[thread_index_in_simdgroup + BN * b];
        }
        sum_exp_score = simd_sum(sum_exp_score);

        for (int b = 0; b < blocks / BN; ++b) {
            U factor = fast::exp(maxs[simdgroup_index_in_threadgroup] - max_score);
            for (int i = 0; i < elem_per_thread; ++i) {
                o[i] += factor * static_cast<U>(partials[i]);
            }
            maxs += BN;
            sums += BN;
            partials += BN * V;
        }

        for (int i = 0; i < elem_per_thread; ++i) {
            outputs[thread_index_in_simdgroup * BD + simdgroup_index_in_threadgroup] = o[i];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            o[i] = simd_sum(outputs[simdgroup_index_in_threadgroup * BD + thread_index_in_simdgroup]);
            o[i] = sum_exp_score == 0.0f ? o[i] : (o[i] / sum_exp_score);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (thread_index_in_simdgroup == 0) {
            for (int i = 0; i < elem_per_thread; ++i) {
                out[i] = static_cast<InT>(o[i]);
            }
        }
    """
    return mx.fast.metal_kernel(
        name="mtplx_sdpa_2pass_paged_reduce",
        input_names=["partials", "sums", "maxs", "blocks"],
        output_names=["out"],
        source=source,
    )


@lru_cache(maxsize=1)
def _paged_partials_dynamic_offset_kernel():
    if not mx.metal.is_available():
        return None

    source = """
        constexpr int BD = 32;
        constexpr int qk_per_thread = D / BD;
        constexpr int v_per_thread = V / BD;

        typedef float U;

        thread U q[qk_per_thread];
        thread U o[v_per_thread] = {0};

        // ``offset`` is an int32 mx.array input (device pointer), not a
        // host scalar: static_cast<int>(offset) never compiled — this
        // kernel was dead on arrival behind its python mask gate
        // (found 2026-08-26 by the tailmask-elide equivalence test).
        const int N = offset[0];
        const int kv_head_idx = threadgroup_position_in_grid.x;
        const int batch_idx = threadgroup_position_in_grid.y;
        const int block_idx = threadgroup_position_in_grid.z;
        const int gqa_factor = threads_per_threadgroup.y;
        const int q_seq_len = threads_per_threadgroup.z;
        const int q_seq_idx = thread_position_in_threadgroup.z;
        const int q_head_idx = gqa_factor * kv_head_idx + thread_position_in_threadgroup.y;
        const int num_kv_heads = threadgroups_per_grid.x;
        const int num_q_heads = num_kv_heads * gqa_factor;
        const int q_batch_head_idx = batch_idx * num_q_heads + q_head_idx;
        const int o_offset = q_batch_head_idx * q_seq_len + q_seq_idx;

        queries += o_offset * D + thread_index_in_simdgroup * qk_per_thread;
        partials += (o_offset * blocks + block_idx) * V
            + thread_index_in_simdgroup * v_per_thread;
        sums += o_offset * blocks + block_idx;
        maxs += o_offset * blocks + block_idx;

        for (int i = 0; i < qk_per_thread; ++i) {
            q[i] = static_cast<U>(scale) * static_cast<U>(queries[i]);
        }

        U max_score = Limits<U>::finite_min;
        U sum_exp_score = 0.0f;

        for (int n = block_idx; n < N; n += blocks) {
            bool use_key = n <= (N - q_seq_len + q_seq_idx);
            if (use_key) {
                const int page_idx = n / PAGE_SIZE;
                const int page_offset = n - page_idx * PAGE_SIZE;
                const device InT* key_ptr = key_cache
                    + (((page_idx * PAGE_SIZE + page_offset) * Hk + kv_head_idx) * D)
                    + thread_index_in_simdgroup * qk_per_thread;
                const device InT* value_ptr = value_cache
                    + (((page_idx * PAGE_SIZE + page_offset) * Hk + kv_head_idx) * V)
                    + thread_index_in_simdgroup * v_per_thread;

                U score = 0.0f;
                for (int i = 0; i < qk_per_thread; ++i) {
                    score += q[i] * static_cast<U>(key_ptr[i]);
                }
                score = simd_sum(score);

                U new_max = metal::max(max_score, score);
                U factor = fast::exp(max_score - new_max);
                U exp_score = fast::exp(score - new_max);

                max_score = new_max;
                sum_exp_score = sum_exp_score * factor + exp_score;
                for (int i = 0; i < v_per_thread; ++i) {
                    o[i] = o[i] * factor + exp_score * static_cast<U>(value_ptr[i]);
                }
            }
        }

        if (thread_index_in_simdgroup == 0) {
            sums[0] = sum_exp_score;
            maxs[0] = max_score;
        }
        for (int i = 0; i < v_per_thread; ++i) {
            partials[i] = static_cast<PartT>(o[i]);
        }
    """
    return mx.fast.metal_kernel(
        name="mtplx_sdpa_2pass_paged_dynamic_offset_partials",
        input_names=["queries", "key_cache", "value_cache", "offset", "scale", "blocks"],
        output_names=["partials", "sums", "maxs"],
        source=source,
    )


def sdpa_2pass_paged_tail(
    *,
    queries: mx.array,
    key_cache: mx.array,
    value_cache: mx.array,
    offset: int,
    block_size: int,
    scale: float,
    mask: Any = None,
    max_q_len: int = 16,
    sliding_window: int = -1,
) -> mx.array | None:
    """Return causal SDPA over contiguous logical tokens stored in KV pages."""

    if not mx.metal.is_available():
        return None
    if mask is not None and not (isinstance(mask, str) and mask == "causal"):
        return None
    if queries.ndim != 4 or key_cache.ndim != 4 or value_cache.ndim != 4:
        return None
    bsz, hq, q_len, d = queries.shape
    if int(bsz) != 1:
        return None
    if q_len <= 0 or q_len > int(max_q_len):
        return None
    if int(offset) <= 0 or int(offset) > int(key_cache.shape[0]) * int(block_size):
        return None
    if int(block_size) != int(key_cache.shape[1]) or int(block_size) != int(value_cache.shape[1]):
        return None
    hk = int(key_cache.shape[2])
    kd = int(key_cache.shape[3])
    vdim = int(value_cache.shape[3])
    if kd != int(d) or int(value_cache.shape[2]) != hk:
        return None
    if int(d) not in {64, 96, 128, 256} or vdim not in {64, 96, 128, 256}:
        return None
    if int(d) != vdim:
        return None
    if hk <= 0 or int(hq) % hk:
        return None
    if queries.dtype not in (mx.bfloat16, mx.float16):
        return None
    if key_cache.dtype != queries.dtype or value_cache.dtype != queries.dtype:
        return None

    queries = mx.contiguous(queries)
    gqa_factor = int(hq) // hk
    if 32 * gqa_factor * int(q_len) > 1024:
        return None
    window_start = 0
    if int(sliding_window) > 0:
        window_start = max(0, int(offset) - int(sliding_window))
    effective_offset = int(offset) - int(window_start)
    if effective_offset <= 0:
        return None

    blocks = _compute_blocks(gqa_factor * int(q_len), effective_offset)
    if blocks <= 0 or blocks % 32:
        return None

    has_window = window_start > 0
    partials_kernel = _paged_partials_kernel(has_window=has_window)
    reduce_kernel = _paged_reduce_kernel()
    if partials_kernel is None or reduce_kernel is None:
        return None

    partial_shape = (int(bsz), int(hq), int(q_len), int(blocks), int(vdim))
    stats_shape = (int(bsz), int(hq), int(q_len), int(blocks))
    partial_dtype = unnormalized_partials_dtype(queries.dtype)
    partial_inputs = [
        queries,
        key_cache,
        value_cache,
        int(offset),
    ]
    if has_window:
        partial_inputs.append(int(window_start))
    partial_inputs.extend([float(scale), int(blocks)])
    partials, sums, maxs = partials_kernel(
        inputs=partial_inputs,
        template=[
            ("InT", queries.dtype),
            ("PartT", partial_dtype),
            ("D", int(d)),
            ("V", int(vdim)),
            ("Hk", int(hk)),
            ("PAGE_SIZE", int(block_size)),
        ],
        grid=(hk * 32, int(bsz) * gqa_factor, int(blocks) * int(q_len)),
        threadgroup=(32, gqa_factor, int(q_len)),
        output_shapes=[partial_shape, stats_shape, stats_shape],
        output_dtypes=[partial_dtype, mx.float32, mx.float32],
    )

    (out,) = reduce_kernel(
        inputs=[partials, sums, maxs, int(blocks)],
        template=[
            ("InT", queries.dtype),
            ("V", int(vdim)),
        ],
        grid=(int(bsz) * int(hq) * 1024, int(q_len), 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[queries.shape],
        output_dtypes=[queries.dtype],
    )
    return out


def sdpa_2pass_paged_tail_dynamic_offset(
    *,
    queries: mx.array,
    key_cache: mx.array,
    value_cache: mx.array,
    offset: mx.array,
    block_size: int,
    scale: float,
    mask: Any = None,
    max_q_len: int = 16,
    max_offset: int | None = None,
) -> mx.array | None:
    """Paged causal SDPA with an MLX-array offset for compiled cache state.

    This variant is intentionally conservative: it uses a static block count
    derived from the cache capacity so the compiled graph can treat ``offset``
    as a data input/output instead of a Python integer captured at trace time.
    """

    if not mx.metal.is_available():
        return None
    if mask is not None and not (isinstance(mask, str) and mask == "causal"):
        return None
    if queries.ndim != 4 or key_cache.ndim != 4 or value_cache.ndim != 4:
        return None
    if not isinstance(offset, mx.array) or offset.size != 1:
        return None
    bsz, hq, q_len, d = queries.shape
    if int(bsz) != 1:
        return None
    if q_len <= 0 or q_len > int(max_q_len):
        return None
    if int(block_size) != int(key_cache.shape[1]) or int(block_size) != int(value_cache.shape[1]):
        return None
    hk = int(key_cache.shape[2])
    kd = int(key_cache.shape[3])
    vdim = int(value_cache.shape[3])
    if kd != int(d) or int(value_cache.shape[2]) != hk:
        return None
    if int(d) not in {64, 96, 128, 256} or vdim not in {64, 96, 128, 256}:
        return None
    if int(d) != vdim:
        return None
    if hk <= 0 or int(hq) % hk:
        return None
    if queries.dtype not in (mx.bfloat16, mx.float16):
        return None
    if key_cache.dtype != queries.dtype or value_cache.dtype != queries.dtype:
        return None

    queries = mx.contiguous(queries)
    gqa_factor = int(hq) // hk
    if 32 * gqa_factor * int(q_len) > 1024:
        # Metal threadgroup ceiling: BD(32) x gqa x q_len rows must fit in
        # 1024 threads — at GQA 6 that is q_len <= 5. Launching past it
        # raises; decline instead (2026-08-26, found by the elide test).
        return None
    capacity = int(key_cache.shape[0]) * int(block_size)
    static_offset = int(max_offset) if max_offset is not None else capacity
    static_offset = max(1, min(static_offset, capacity))
    blocks = _compute_blocks(gqa_factor * int(q_len), static_offset)
    if blocks <= 0 or blocks % 32:
        return None

    partials_kernel = _paged_partials_dynamic_offset_kernel()
    reduce_kernel = _paged_reduce_kernel()
    if partials_kernel is None or reduce_kernel is None:
        return None

    # A 0-d size-1 array passes the gate above but binds BY VALUE in
    # mx.fast.metal_kernel, so the source's ``offset[0]`` fails to compile
    # (the TensorOffset adapter hands a 0-d offset; serve hands shape [1]).
    # Reshape to [1] so the offset always binds as a device pointer.
    offset_input = offset.astype(mx.int32).reshape(1)

    partial_shape = (int(bsz), int(hq), int(q_len), int(blocks), int(vdim))
    stats_shape = (int(bsz), int(hq), int(q_len), int(blocks))
    partial_dtype = unnormalized_partials_dtype(queries.dtype)
    partials, sums, maxs = partials_kernel(
        inputs=[
            queries,
            key_cache,
            value_cache,
            offset_input,
            float(scale),
            int(blocks),
        ],
        template=[
            ("InT", queries.dtype),
            ("PartT", partial_dtype),
            ("D", int(d)),
            ("V", int(vdim)),
            ("Hk", int(hk)),
            ("PAGE_SIZE", int(block_size)),
        ],
        grid=(hk * 32, int(bsz) * gqa_factor, int(blocks) * int(q_len)),
        threadgroup=(32, gqa_factor, int(q_len)),
        output_shapes=[partial_shape, stats_shape, stats_shape],
        output_dtypes=[partial_dtype, mx.float32, mx.float32],
    )

    (out,) = reduce_kernel(
        inputs=[partials, sums, maxs, int(blocks)],
        template=[
            ("InT", queries.dtype),
            ("V", int(vdim)),
        ],
        grid=(int(bsz) * int(hq) * 1024, int(q_len), 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[queries.shape],
        output_dtypes=[queries.dtype],
    )
    return out
