"""Two-pass Metal SDPA probe for tiny Qwen3Next verify windows.

This is an exact-attention diagnostic kernel adapted for MTPLX's verify shapes:
``B=1``, GQA full-attention layers, ``q_len`` usually depth+1, and long cached
KV.  It keeps the product path opt-in because reduction order differs from MLX's
stock fused SDPA and must pass acceptance parity before any serving claim.
"""

from __future__ import annotations

from functools import lru_cache
import os
from typing import Any

import mlx.core as mx


def _compute_blocks(gqa_factor: int, n_kv: int) -> int:
    override = (os.environ.get("MTPLX_SDPA_2PASS_BLOCKS") or "").strip()
    if override:
        try:
            blocks = int(override)
        except ValueError:
            blocks = 0
        if blocks > 0:
            return int(blocks)
    arch = str(mx.device_info().get("architecture", ""))
    devc = arch[-1] if arch else ""
    n_simds = int(gqa_factor)
    n = int(n_kv)
    if devc == "d":
        blocks = 128
        if n_simds <= 2 and n > 8192:
            blocks = 256
        elif n_simds >= 6:
            if 16384 <= n < 65536:
                blocks = 512
            elif n >= 65536:
                blocks = 1024
    elif devc == "s":
        blocks = 64
        if n > 1024 and n_simds > 4:
            if n <= 8192:
                blocks = 128
            elif n <= 32768:
                blocks = 256
            elif n <= 65536:
                blocks = 512
            else:
                blocks = 1024
    else:
        blocks = 64 if n_simds >= 4 else 32
    return int(blocks)


def unnormalized_partials_dtype(query_dtype: Any) -> Any:
    """Storage dtype for a two-pass kernel's UNNORMALIZED partial numerators.

    Pass one stores ``sum_j exp(s_j - m_block) * v_j`` per block, before the
    division by the block's exp-sum, so its magnitude is up to (rows walked
    by the block) x max|v|. fp16 overflows that at 65504: 16 equal rows of
    8192 store inf, and when another block's max is ~200 higher the
    reducer's ``exp(m_block - m_global) * partial`` is ``0 * inf`` = NaN in
    every output element (issue #526; tests/test_quant_partials_fp16_526.py
    and tests/test_split_partials_fp16_526.py reproduce it on every kernel
    that stores such partials). bfloat16 shares float32's exponent range, so
    no walk over finite values can overflow it; its partials stay
    bit-for-bit what they were, and so does their buffer. For fp16 queries
    the buffer doubles: 27B verify at 32k is 24 heads x 4 rows x 512 blocks
    x 256 dims, 25 MB in fp16 and 50 MB in float32, written and read back
    once per layer call, against a 64 MB q8 KV walk. Both reducers
    (``_reduce_kernel`` here and ``sdpa_2pass_paged._paged_reduce_kernel``)
    accumulate in float32 and cast only the normalized output, so a float32
    partial needs no change there: MLX types the ``partials`` pointer from
    the input array. Every MTPLX split kernel that stores unnormalized
    partials follows this (the two-pass, paged, packed, packed-quant, q8
    and NAX flash kernels); MLX's own sdpa_vector is outside this package.
    """

    return mx.float32 if query_dtype == mx.float16 else query_dtype


@lru_cache(maxsize=None)
def _partials_kernel(*, has_mask: bool):
    if not mx.metal.is_available():
        return None

    mask_setup = ""
    mask_use_key = ""
    mask_score = ""
    mask_advance = ""
    inputs = [
        "queries",
        "keys",
        "values",
        "gqa_factor",
        "N",
        "k_head_stride",
        "k_seq_stride",
        "v_head_stride",
        "v_seq_stride",
        "scale",
        "blocks",
    ]
    if has_mask:
        inputs.append("mask")
        mask_setup = """
        auto mask_ = mask + (((b_idx * Hq + q_head_idx) * M_FIXED + q_seq_idx) * N + block_idx);
        """
        mask_use_key = """
            auto mask_value = static_cast<float>(mask_[0]);
            use_key = use_key && (mask_value >= Limits<InT>::finite_min);
        """
        mask_score = """
                score += static_cast<float>(mask_[0]);
        """
        mask_advance = """
            mask_ += blocks;
        """

    source = f"""
        constexpr int BD = 32;
        constexpr int qk_per_thread = D / BD;
        constexpr int v_per_thread = V / BD;

        auto q_head_idx = threadgroup_position_in_grid.x;
        auto b_idx = threadgroup_position_in_grid.y;
        auto block_idx = threadgroup_position_in_grid.z;
        auto q_seq_idx = thread_position_in_threadgroup.z;
        auto simd_lid = thread_index_in_simdgroup;

        auto Hq = threadgroups_per_grid.x;
        auto hk_idx = q_head_idx / gqa_factor;
        auto q_batch_head_idx = b_idx * Hq + q_head_idx;
        auto o_offset = q_batch_head_idx * M_FIXED + q_seq_idx;

        auto q_ = queries + (o_offset * D) + simd_lid * qk_per_thread;
        auto k_ = keys + ((b_idx * Hk + hk_idx) * k_head_stride) + block_idx * k_seq_stride + simd_lid * qk_per_thread;
        auto v_ = values + ((b_idx * Hk + hk_idx) * v_head_stride) + block_idx * v_seq_stride + simd_lid * v_per_thread;

        partials += (o_offset * blocks + block_idx) * V + simd_lid * v_per_thread;
        sums += o_offset * blocks + block_idx;
        maxs += o_offset * blocks + block_idx;
        {mask_setup}

        thread float q[qk_per_thread];
        thread float o[v_per_thread];
        threadgroup InT tg_k[BD * qk_per_thread];
        threadgroup InT tg_v[BD * v_per_thread];

        for (int i = 0; i < qk_per_thread; ++i) {{
            q[i] = static_cast<float>(scale) * static_cast<float>(q_[i]);
        }}
        for (int i = 0; i < v_per_thread; ++i) {{
            o[i] = 0.0f;
        }}

        float max_score = Limits<float>::finite_min;
        float sum_exp_score = 0.0f;

        for (int n = block_idx; n < N; n += blocks) {{
            if (q_seq_idx == 0) {{
                for (int i = 0; i < qk_per_thread; ++i) {{
                    tg_k[simd_lid * qk_per_thread + i] = k_[i];
                }}
                for (int i = 0; i < v_per_thread; ++i) {{
                    tg_v[simd_lid * v_per_thread + i] = v_[i];
                }}
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);

            bool use_key = (n <= (N - M_FIXED + q_seq_idx));
            {mask_use_key}

            if (use_key) {{
                float score = 0.0f;
                for (int i = 0; i < qk_per_thread; ++i) {{
                    score += q[i] * static_cast<float>(tg_k[simd_lid * qk_per_thread + i]);
                }}
                score = simd_sum(score);
                {mask_score}

                float new_max = metal::max(max_score, score);
                float factor = fast::exp(max_score - new_max);
                float exp_score = fast::exp(score - new_max);

                max_score = new_max;
                sum_exp_score = sum_exp_score * factor + exp_score;
                for (int i = 0; i < v_per_thread; ++i) {{
                    o[i] = o[i] * factor + exp_score * static_cast<float>(tg_v[simd_lid * v_per_thread + i]);
                }}
            }}

            threadgroup_barrier(mem_flags::mem_threadgroup);
            k_ += blocks * int(k_seq_stride);
            v_ += blocks * int(v_seq_stride);
            {mask_advance}
        }}

        if (simd_lid == 0) {{
            sums[0] = sum_exp_score;
            maxs[0] = max_score;
        }}
        for (int i = 0; i < v_per_thread; ++i) {{
            partials[i] = static_cast<PartT>(o[i]);
        }}
    """
    return mx.fast.metal_kernel(
        name=f"mtplx_sdpa_2pass_partials{'_mask' if has_mask else ''}",
        input_names=inputs,
        output_names=["partials", "sums", "maxs"],
        source=source,
    )


@lru_cache(maxsize=1)
def _reduce_kernel():
    if not mx.metal.is_available():
        return None

    source = """
        constexpr int BN = 32;
        constexpr int BD = 32;
        constexpr int elem_per_thread = V / BD;

        auto head_idx = threadgroup_position_in_grid.x;
        auto q_seq_idx = threadgroup_position_in_grid.y;
        auto simd_gid = simdgroup_index_in_threadgroup;
        auto simd_lid = thread_index_in_simdgroup;

        auto q_offset = head_idx * M_FIXED + q_seq_idx;
        partials += (q_offset * blocks + simd_gid) * V + simd_lid * elem_per_thread;
        sums += q_offset * blocks;
        maxs += q_offset * blocks;
        out += q_offset * V + simd_gid * elem_per_thread;

        thread float o[elem_per_thread];
        threadgroup float outputs[BN * BD];

        for (int i = 0; i < elem_per_thread; ++i) {
            o[i] = 0.0f;
        }

        float sum_exp_score = 0.0f;
        float max_score = Limits<float>::finite_min;

        for (int b = 0; b < blocks / BN; ++b) {
            max_score = metal::max(max_score, maxs[simd_lid + BN * b]);
        }
        max_score = simd_max(max_score);

        for (int b = 0; b < blocks / BN; ++b) {
            float factor = fast::exp(maxs[simd_lid + BN * b] - max_score);
            sum_exp_score += factor * sums[simd_lid + BN * b];
        }
        sum_exp_score = simd_sum(sum_exp_score);

        for (int b = 0; b < blocks / BN; ++b) {
            float factor = fast::exp(maxs[simd_gid] - max_score);
            for (int i = 0; i < elem_per_thread; ++i) {
                o[i] += factor * static_cast<float>(partials[i]);
            }
            maxs += BN;
            partials += BN * V;
        }

        for (int i = 0; i < elem_per_thread; ++i) {
            outputs[simd_lid * BD + simd_gid] = o[i];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            o[i] = simd_sum(outputs[simd_gid * BD + simd_lid]);
            o[i] = sum_exp_score == 0.0f ? o[i] : (o[i] / sum_exp_score);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (simd_lid == 0) {
            for (int i = 0; i < elem_per_thread; ++i) {
                out[i] = static_cast<InT>(o[i]);
            }
        }
    """
    return mx.fast.metal_kernel(
        name="mtplx_sdpa_2pass_reduce",
        input_names=["partials", "sums", "maxs", "blocks"],
        output_names=["out"],
        source=source,
    )


def _causal_mask_for_tail(mask: Any, *, bsz: int, hq: int, q_len: int, n_kv: int, dtype: mx.Dtype):
    if mask is None or (isinstance(mask, str) and mask == "causal"):
        return None
    if mask.dtype == mx.bool_:
        converted = mx.where(
            mask,
            mx.zeros(mask.shape, dtype=dtype),
            mx.full(mask.shape, mx.finfo(dtype).min, dtype=dtype),
        )
    else:
        converted = mask.astype(dtype) if mask.dtype != dtype else mask
    return mx.contiguous(mx.broadcast_to(converted, (bsz, hq, q_len, n_kv)))


def sdpa_2pass_tail(
    *,
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    scale: float,
    mask: Any = None,
    max_q_len: int = 16,
) -> mx.array | None:
    """Return causal SDPA for the newest ``q_len`` rows, or ``None`` if unsupported."""

    if not mx.metal.is_available():
        return None
    if queries.ndim != 4 or keys.ndim != 4 or values.ndim != 4:
        return None
    bsz, hq, q_len, d = queries.shape
    if q_len <= 0 or q_len > int(max_q_len):
        return None
    _, hk, n_kv, kd = keys.shape
    if int(kd) != int(d):
        return None
    vdim = int(values.shape[-1])
    input_type = queries.dtype
    if input_type not in (mx.bfloat16, mx.float16):
        return None
    if int(d) not in {128, 256} or vdim not in {128, 256} or int(d) != vdim:
        return None
    if int(hk) <= 0 or int(hq) % int(hk):
        return None

    queries = mx.contiguous(queries)
    keys = mx.contiguous(keys)
    values = mx.contiguous(values)

    gqa_factor = int(hq) // int(hk)
    blocks = _compute_blocks(gqa_factor, int(n_kv))
    if blocks <= 0 or blocks % 32:
        return None

    mask_tensor = _causal_mask_for_tail(
        mask,
        bsz=int(bsz),
        hq=int(hq),
        q_len=int(q_len),
        n_kv=int(n_kv),
        dtype=input_type,
    )
    kernel = _partials_kernel(has_mask=mask_tensor is not None)
    reduce_kernel = _reduce_kernel()
    if kernel is None or reduce_kernel is None:
        return None

    inputs = [
        queries,
        keys,
        values,
        int(gqa_factor),
        int(n_kv),
        int(keys.shape[2] * keys.shape[3]),
        int(keys.shape[3]),
        int(values.shape[2] * values.shape[3]),
        int(values.shape[3]),
        float(scale),
        int(blocks),
    ]
    if mask_tensor is not None:
        inputs.append(mask_tensor)

    partial_shape = (int(bsz) * int(hq), int(q_len), int(blocks), int(vdim))
    stats_shape = (int(bsz) * int(hq), int(q_len), int(blocks))
    partial_dtype = unnormalized_partials_dtype(input_type)
    partials, sums, maxs = kernel(
        inputs=inputs,
        template=[
            ("InT", input_type),
            ("PartT", partial_dtype),
            ("D", int(d)),
            ("V", int(vdim)),
            ("Hk", int(hk)),
            ("M_FIXED", int(q_len)),
        ],
        grid=(int(hq) * 32, int(bsz), int(blocks) * int(q_len)),
        threadgroup=(32, 1, int(q_len)),
        output_shapes=[partial_shape, stats_shape, stats_shape],
        output_dtypes=[partial_dtype, mx.float32, mx.float32],
    )

    (out,) = reduce_kernel(
        inputs=[partials, sums, maxs, int(blocks)],
        template=[
            ("InT", input_type),
            ("V", int(vdim)),
            ("M_FIXED", int(q_len)),
        ],
        grid=((int(bsz) * int(hq)) * 1024, int(q_len), 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[queries.shape],
        output_dtypes=[input_type],
    )
    return out
