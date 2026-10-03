"""Two-pass paged SDPA reading q8-quantized KV pages directly.

Wired as the q8 kv_quant decode route (cache_state
VllmMetalPagedKVCache._kv_quant_2pass_attention), where the alternative is the
dequant fallback rather than the dense kernel the verdict below measured.

CLOSED LANE (2026-06-12, [CONFIDENT DOES NOT WORK as a kernel-variant
family]) as a replacement for the dense kernel. Three numerically
verified variants (per-token scales, char4-vectorized loads, per-page
scales) all measured ~0.8x vs the dense two-pass kernel at the target
8k-32k contexts on M5 (short-context 1k reached 1.72x with vectorization,
but short context is not where KV bytes matter). Conclusion: int8 loads +
inline conversion cannot beat the GPU's native bf16 vector-load FMA path at
these shapes; the kernel is not purely KV-DRAM-bound the way the premise
assumed. Re-litigation bar: a threadgroup-staged page-dequant redesign or
different silicon must beat dense at 8k+ in the kill-test microbench
(see LOG.md 2026-06-12 15:10-15:14 entries) before any wiring work.

self_test() remains the numeric gate (worst 0.00195 across 7 shape/edge
configs vs the dense kernel on identically-dequantized pages). Causal mask
only, batch 1, no sliding window.
"""

from __future__ import annotations

from functools import lru_cache

import mlx.core as mx

from .sdpa_2pass_paged import (  # reuse the proven pieces
    _compute_blocks,
    _paged_reduce_kernel,
    sdpa_2pass_paged_tail,
    unnormalized_partials_dtype,
)


@lru_cache(maxsize=1)
def _paged_partials_kernel_q8():
    source = """
        typedef float U;
        constexpr int qk_per_thread = D / 32;
        constexpr int v_per_thread = V / 32;

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

        const int N = int(N_tokens);
        for (int n = block_idx; n < N; n += blocks) {
            bool use_key = n <= (N - q_seq_len + q_seq_idx);
            if (use_key) {
                const int page_idx = n / PAGE_SIZE;
                const int page_offset = n - page_idx * PAGE_SIZE;
                const int tok_head = (page_idx * PAGE_SIZE + page_offset) * Hk + kv_head_idx;
                // Vectorized byte loads: qk_per_thread/v_per_thread int8s per
                // thread are read as packed char4 words (one 32-bit load per
                // 4 elements) instead of per-element byte loads.
                const device char4* key_ptr4 = (const device char4*)(key_q
                    + tok_head * D
                    + thread_index_in_simdgroup * qk_per_thread);
                const device char4* value_ptr4 = (const device char4*)(value_q
                    + tok_head * V
                    + thread_index_in_simdgroup * v_per_thread);
                const U k_scale = static_cast<U>(key_scales[tok_head]);
                const U v_scale = static_cast<U>(value_scales[tok_head]);

                U score = 0.0f;
                for (int w = 0; w < qk_per_thread / 4; ++w) {
                    const float4 kw = static_cast<float4>(key_ptr4[w]);
                    score += q[4 * w + 0] * kw.x;
                    score += q[4 * w + 1] * kw.y;
                    score += q[4 * w + 2] * kw.z;
                    score += q[4 * w + 3] * kw.w;
                }
                for (int i = 4 * (qk_per_thread / 4); i < qk_per_thread; ++i) {
                    score += q[i]
                        * static_cast<U>(((const device int8_t*)key_ptr4)[i]);
                }
                score *= k_scale;
                score = simd_sum(score);

                U new_max = metal::max(max_score, score);
                U factor = fast::exp(max_score - new_max);
                U exp_score = fast::exp(score - new_max);

                max_score = new_max;
                sum_exp_score = sum_exp_score * factor + exp_score;
                const U ev_scale = exp_score * v_scale;
                for (int w = 0; w < v_per_thread / 4; ++w) {
                    const float4 vw = static_cast<float4>(value_ptr4[w]);
                    o[4 * w + 0] = o[4 * w + 0] * factor + ev_scale * vw.x;
                    o[4 * w + 1] = o[4 * w + 1] * factor + ev_scale * vw.y;
                    o[4 * w + 2] = o[4 * w + 2] * factor + ev_scale * vw.z;
                    o[4 * w + 3] = o[4 * w + 3] * factor + ev_scale * vw.w;
                }
                for (int i = 4 * (v_per_thread / 4); i < v_per_thread; ++i) {
                    o[i] = o[i] * factor
                        + ev_scale
                        * static_cast<U>(((const device int8_t*)value_ptr4)[i]);
                }
            }
        }

        if (thread_index_in_simdgroup == 0) {
            sums[0] = sum_exp_score;
            maxs[0] = max_score;
        }
        // PartT, not InT: o[] is not yet divided by sum_exp_score
        // (see sdpa_2pass.unnormalized_partials_dtype).
        for (int i = 0; i < v_per_thread; ++i) {
            partials[i] = static_cast<PartT>(o[i]);
        }
    """
    return mx.fast.metal_kernel(
        name="mtplx_sdpa_2pass_paged_q8_partials",
        input_names=[
            "queries",
            "key_q",
            "key_scales",
            "value_q",
            "value_scales",
            "N_tokens",
            "scale",
            "blocks",
        ],
        output_names=["partials", "sums", "maxs"],
        source=source,
    )


def quantize_pages_q8(cache: mx.array) -> tuple[mx.array, mx.array]:
    """Symmetric per-(token, head) q8 of a (pages, block, Hk, D) cache."""
    amax = mx.max(mx.abs(cache.astype(mx.float32)), axis=-1, keepdims=True)
    scales = mx.maximum(amax / 127.0, 1e-8)
    q = mx.round(cache.astype(mx.float32) / scales)
    q = mx.clip(q, -127, 127).astype(mx.int8)
    return q, scales.squeeze(-1).astype(mx.float32)


def sdpa_2pass_paged_q8_tail(
    *,
    queries: mx.array,
    key_q: mx.array,
    key_scales: mx.array,
    value_q: mx.array,
    value_scales: mx.array,
    offset: int,
    block_size: int,
    scale: float,
    max_q_len: int = 16,
) -> mx.array | None:
    """q8-KV variant of sdpa_2pass_paged_tail. Causal, batch 1, no window."""
    if not mx.metal.is_available():
        return None
    if queries.ndim != 4 or key_q.ndim != 4 or value_q.ndim != 4:
        return None
    bsz, hq, q_len, d = queries.shape
    if int(bsz) != 1 or q_len <= 0 or q_len > int(max_q_len):
        return None
    hk = int(key_q.shape[2])
    vdim = int(value_q.shape[3])
    if int(key_q.shape[3]) != int(d) or int(d) != vdim:
        return None
    if int(d) not in {64, 96, 128, 256}:
        return None
    if hk <= 0 or int(hq) % hk:
        return None
    if key_q.dtype != mx.int8 or value_q.dtype != mx.int8:
        return None
    if int(offset) <= 0 or int(offset) > int(key_q.shape[0]) * int(block_size):
        return None

    queries = mx.contiguous(queries)
    gqa_factor = int(hq) // hk
    if 32 * gqa_factor * int(q_len) > 1024:
        return None
    blocks = _compute_blocks(gqa_factor * int(q_len), int(offset))
    if blocks <= 0 or blocks % 32:
        return None

    kernel = _paged_partials_kernel_q8()
    reduce_kernel = _paged_reduce_kernel()
    if kernel is None or reduce_kernel is None:
        return None

    partial_shape = (int(bsz), int(hq), int(q_len), int(blocks), int(vdim))
    stats_shape = (int(bsz), int(hq), int(q_len), int(blocks))
    partial_dtype = unnormalized_partials_dtype(queries.dtype)
    partials, sums, maxs = kernel(
        inputs=[
            queries,
            mx.contiguous(key_q),
            mx.contiguous(key_scales.astype(mx.float32)),
            mx.contiguous(value_q),
            mx.contiguous(value_scales.astype(mx.float32)),
            int(offset),
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
        template=[("InT", queries.dtype), ("V", int(vdim))],
        grid=(int(bsz) * int(hq) * 1024, int(q_len), 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[(int(bsz), int(hq), int(q_len), int(vdim))],
        output_dtypes=[queries.dtype],
    )
    return out


def self_test(verbose: bool = True) -> dict[str, float]:
    """Numeric gate: q8 kernel vs dense kernel run on the dequantized pages.

    Both arms see the SAME quantization error, so they must agree to kernel
    arithmetic precision, isolating kernel correctness from quantization
    quality. Sweeps product-shape GQA configs and page-boundary edges.
    """
    mx.random.seed(20260612)
    configs = [
        # (hq, hk, d, block, pages, offset, q_len, label)
        (8, 2, 128, 16, 16, 230, 4, "smoke"),
        (16, 4, 128, 16, 64, 1009, 4, "deep offset, mid-page"),
        (16, 4, 128, 16, 64, 1024, 4, "offset page-aligned"),
        (16, 4, 128, 16, 4, 15, 1, "tiny prefix, q1"),
        (16, 4, 128, 16, 4, 16, 2, "prefix=one page, q2"),
        (32, 8, 128, 16, 128, 2041, 4, "2k-deep, q4"),
        (8, 8, 64, 16, 32, 333, 3, "MHA d64, q3"),
    ]
    worst = 0.0
    for hq, hk, d, block, pages, offset, q_len, label in configs:
        q = (mx.random.normal((1, hq, q_len, d)) * 0.3).astype(mx.bfloat16)
        kc = (mx.random.normal((pages, block, hk, d)) * 0.5).astype(mx.bfloat16)
        vc = (mx.random.normal((pages, block, hk, d)) * 0.5).astype(mx.bfloat16)
        kq, ks = quantize_pages_q8(kc)
        vq, vs = quantize_pages_q8(vc)
        kd = (kq.astype(mx.float32) * ks[..., None]).astype(mx.bfloat16)
        vd = (vq.astype(mx.float32) * vs[..., None]).astype(mx.bfloat16)
        dense = sdpa_2pass_paged_tail(
            queries=q, key_cache=kd, value_cache=vd,
            offset=offset, block_size=block, scale=d ** -0.5, mask="causal",
        )
        quant = sdpa_2pass_paged_q8_tail(
            queries=q, key_q=kq, key_scales=ks, value_q=vq, value_scales=vs,
            offset=offset, block_size=block, scale=d ** -0.5,
        )
        assert dense is not None and quant is not None, f"kernel refused: {label}"
        mx.eval(dense, quant)
        diff = float(mx.abs(dense.astype(mx.float32) - quant.astype(mx.float32)).max())
        worst = max(worst, diff)
        if verbose:
            print(f"{label:28s} max_abs_diff={diff:.5f}")
    if verbose:
        print(f"worst across {len(configs)} configs: {worst:.5f}")
    return {"max_abs_diff": worst, "configs": float(len(configs))}


if __name__ == "__main__":
    r = self_test()
    raise SystemExit(0 if r["max_abs_diff"] < 0.05 else 1)
