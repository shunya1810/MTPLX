"""Dequant-in-flight packed GQA attention over q8/q4 KV (2026-08-25).

The product's KV-quant lane stores rowwise-symmetric quantized KV
(kv_quant.quantize_symmetric): int8 (q8) or nibble-packed uint8 (q4) with ONE
fp32 scale per row. The existing consumption path dequantizes through generic
ops and runs the walk at ~28GB/s — 2.5x SLOWER than bf16 (kv_walk_bench
receipt). This kernel walks the PACKED buffers directly and dequantizes in
registers, so the memory traffic drops to 0.51x (q8) / 0.26x (q4) of bf16.

Topology is byte-for-byte the proven sdpa_gqa_packed_tail one-bank+two-bank
design (one simdgroup per KV head, float4 query banks, blocks-strided walk,
fp32 LSE partials; the reduce kernel is REUSED from the bf16 module). Only the
K/V load path differs:

- q8: per-thread 8x int8 loads; the row dot runs on raw integer values in
  float and the fp32 row scale multiplies the SCORE once per row (1 mult
  instead of 8).
- q4: per-thread 4x uint8 loads, nibbles unpacked as (b & 0xF) - 8 and
  (b >> 4) - 8 (matching kv_quant's even|odd<<4 packing), same fold.
- V: the row scale folds into the exp-weight before the FMA loop.

Exactness contract: identical math to dequantize_symmetric + the bf16 kernel
in fp32 accumulation; the test suite pins it against that reference.
"""

from __future__ import annotations

from functools import lru_cache

import mlx.core as mx

from ..compile_state import is_compile_trace_error
from .sdpa_2pass_paged import unnormalized_partials_dtype
from .sdpa_gqa_packed import (  # shared, proven pieces
    _bail,
    _blocks_for_capacity,
    _paged_reduce_kernel,
)


def _concrete_offset(offset) -> int | None:
    """The host value of an array offset, or None while ``mx.compile`` traces.

    Eagerly this is one scalar sync. Inside a trace ``.item()`` raises MLX's
    refusal, recognised by ``compile_state.is_compile_trace_error`` exactly
    as ``cache_state._concrete_offset`` does; every other ValueError (a NaN
    offset that ``int()`` cannot convert, a failed evaluation) is a real fault
    and propagates instead of skipping the range check below.
    """

    try:
        return int(offset.item())
    except ValueError as exc:
        if is_compile_trace_error(exc):
            return None
        raise


def _static_blocks(capacity: int, max_offset: int | None) -> int:
    """Block-lane count from the STATIC attention ceiling.

    ``capacity`` is the whole allocated seq axis; ``max_offset`` (a compiled
    verify bucket) tightens the ceiling so small-context buckets keep small
    partial/reduce shapes and a stable compiled graph. Blocks only set the
    walk stride, grid size, and reduce width — the walk itself always reads
    exactly ``offset`` rows, so the math is invariant to this value.
    """
    capacity = int(capacity)
    ceiling = (
        capacity
        if max_offset is None
        else max(1, min(int(max_offset), capacity))
    )
    return _blocks_for_capacity(ceiling)


@lru_cache(maxsize=None)
def _quant_partials_kernel():
    if not mx.metal.is_available():
        return None

    # KBITS in {8, 4}. QL in [1, 8]: rows 0..3 on bank 1, 4..7 on bank 2
    # (banks compile out below their row counts exactly like the bf16 kernel).
    source = """
        constexpr int BD = 32;
        constexpr int qk_per_thread = D / BD;
        constexpr int v_per_thread = V / BD;
        constexpr int QH = (QL > 4) ? (QL - 4) : 0;
        constexpr int k_bytes_per_thread =
            (KBITS == 8) ? qk_per_thread : (qk_per_thread / 2);
        constexpr int KROW = (KBITS == 8) ? D : (D / 2);

        typedef float U;

        const int kv_head_idx = threadgroup_position_in_grid.x;
        const int block_idx = threadgroup_position_in_grid.z;
        const int gqa_idx = thread_position_in_threadgroup.y;
        const int simd_lid = thread_index_in_simdgroup;
        const int n_kv = static_cast<int>(offset[0]);

        const int q_head_idx = kv_head_idx * GQA_F + gqa_idx;

        // Pro-consult #3 register layout (2026-08-25): the query bank is
        // TRANSPOSED — qbank[d] holds element d across query rows 0..3 as a
        // float4, so each K element costs ONE float4 fma covering four rows
        // (the row-scalar layout paid up to 8 scalar FMAs per element).
        // Output accumulators are transposed the same way. No dequantized
        // float arrays are ever materialized: uchar4 components are consumed
        // inline. Rows 4..7 ride second banks that compile out at QL <= 4.
        float4 qbank[qk_per_thread];
        float4 qbank2[qk_per_thread];
        float4 obank[v_per_thread];
        float4 obank2[v_per_thread];
        for (int d = 0; d < qk_per_thread; ++d) {
            const int col = simd_lid * qk_per_thread + d;
            float4 qv = 0.0f;
            float4 qv2 = 0.0f;
            qv.x = static_cast<U>(queries[(q_head_idx * QL + 0) * D + col]);
            if (QL > 1) qv.y = static_cast<U>(queries[(q_head_idx * QL + 1) * D + col]);
            if (QL > 2) qv.z = static_cast<U>(queries[(q_head_idx * QL + 2) * D + col]);
            if (QL > 3) qv.w = static_cast<U>(queries[(q_head_idx * QL + 3) * D + col]);
            if (QH > 0) qv2.x = static_cast<U>(queries[(q_head_idx * QL + 4) * D + col]);
            if (QH > 1) qv2.y = static_cast<U>(queries[(q_head_idx * QL + 5) * D + col]);
            if (QH > 2) qv2.z = static_cast<U>(queries[(q_head_idx * QL + 6) * D + col]);
            if (QH > 3) qv2.w = static_cast<U>(queries[(q_head_idx * QL + 7) * D + col]);
            qbank[d] = static_cast<U>(scale) * qv;
            qbank2[d] = static_cast<U>(scale) * qv2;
            obank[d] = 0.0f;
            obank2[d] = 0.0f;
        }
        float4 max_score = Limits<float>::finite_min;
        float4 sum_exp = 0.0f;
        float4 max_score2 = Limits<float>::finite_min;
        float4 sum_exp2 = 0.0f;

        const device uint8_t* k_ptr = k_q
            + (size_t)kv_head_idx * k_head_seq * KROW
            + (size_t)block_idx * KROW
            + simd_lid * k_bytes_per_thread;
        const device uint8_t* v_ptr = v_q
            + (size_t)kv_head_idx * v_head_seq * KROW
            + (size_t)block_idx * KROW
            + simd_lid * k_bytes_per_thread;
        const device float* ks_ptr = k_scale
            + (size_t)kv_head_idx * k_head_seq + block_idx;
        const device float* vs_ptr = v_scale
            + (size_t)kv_head_idx * v_head_seq + block_idx;

        const int n_full = n_kv - QL;

        #define MTPLX_QDOT(SRC, D1, D2) \
            { \
                const device uchar4* p4 = \
                    reinterpret_cast<const device uchar4*>(SRC); \
                if (KBITS == 8) { \
                    uchar4 a = p4[0]; \
                    char4 c = as_type<char4>(a); \
                    D1 = fma(qbank[0], (U)c.x, D1); \
                    D1 = fma(qbank[1], (U)c.y, D1); \
                    D1 = fma(qbank[2], (U)c.z, D1); \
                    D1 = fma(qbank[3], (U)c.w, D1); \
                    if (QH > 0) { \
                        D2 = fma(qbank2[0], (U)c.x, D2); \
                        D2 = fma(qbank2[1], (U)c.y, D2); \
                        D2 = fma(qbank2[2], (U)c.z, D2); \
                        D2 = fma(qbank2[3], (U)c.w, D2); \
                    } \
                    a = p4[1]; \
                    c = as_type<char4>(a); \
                    D1 = fma(qbank[4], (U)c.x, D1); \
                    D1 = fma(qbank[5], (U)c.y, D1); \
                    D1 = fma(qbank[6], (U)c.z, D1); \
                    D1 = fma(qbank[7], (U)c.w, D1); \
                    if (QH > 0) { \
                        D2 = fma(qbank2[4], (U)c.x, D2); \
                        D2 = fma(qbank2[5], (U)c.y, D2); \
                        D2 = fma(qbank2[6], (U)c.z, D2); \
                        D2 = fma(qbank2[7], (U)c.w, D2); \
                    } \
                } else { \
                    const uchar4 a = p4[0]; \
                    D1 = fma(qbank[0], (U)((int)(a.x & 0x0F) - 8), D1); \
                    D1 = fma(qbank[1], (U)((int)(a.x >> 4) - 8), D1); \
                    D1 = fma(qbank[2], (U)((int)(a.y & 0x0F) - 8), D1); \
                    D1 = fma(qbank[3], (U)((int)(a.y >> 4) - 8), D1); \
                    D1 = fma(qbank[4], (U)((int)(a.z & 0x0F) - 8), D1); \
                    D1 = fma(qbank[5], (U)((int)(a.z >> 4) - 8), D1); \
                    D1 = fma(qbank[6], (U)((int)(a.w & 0x0F) - 8), D1); \
                    D1 = fma(qbank[7], (U)((int)(a.w >> 4) - 8), D1); \
                    if (QH > 0) { \
                        D2 = fma(qbank2[0], (U)((int)(a.x & 0x0F) - 8), D2); \
                        D2 = fma(qbank2[1], (U)((int)(a.x >> 4) - 8), D2); \
                        D2 = fma(qbank2[2], (U)((int)(a.y & 0x0F) - 8), D2); \
                        D2 = fma(qbank2[3], (U)((int)(a.y >> 4) - 8), D2); \
                        D2 = fma(qbank2[4], (U)((int)(a.z & 0x0F) - 8), D2); \
                        D2 = fma(qbank2[5], (U)((int)(a.z >> 4) - 8), D2); \
                        D2 = fma(qbank2[6], (U)((int)(a.w & 0x0F) - 8), D2); \
                        D2 = fma(qbank2[7], (U)((int)(a.w >> 4) - 8), D2); \
                    } \
                } \
            }

        #define MTPLX_VUPD(SRC, ES1, ES2, F1, F2) \
            { \
                for (int d = 0; d < v_per_thread; ++d) { \
                    obank[d] = obank[d] * F1; \
                    if (QH > 0) obank2[d] = obank2[d] * F2; \
                } \
                const device uchar4* p4 = \
                    reinterpret_cast<const device uchar4*>(SRC); \
                if (KBITS == 8) { \
                    uchar4 a = p4[0]; \
                    char4 c = as_type<char4>(a); \
                    obank[0] = fma(ES1, (U)c.x, obank[0]); \
                    obank[1] = fma(ES1, (U)c.y, obank[1]); \
                    obank[2] = fma(ES1, (U)c.z, obank[2]); \
                    obank[3] = fma(ES1, (U)c.w, obank[3]); \
                    if (QH > 0) { \
                        obank2[0] = fma(ES2, (U)c.x, obank2[0]); \
                        obank2[1] = fma(ES2, (U)c.y, obank2[1]); \
                        obank2[2] = fma(ES2, (U)c.z, obank2[2]); \
                        obank2[3] = fma(ES2, (U)c.w, obank2[3]); \
                    } \
                    a = p4[1]; \
                    c = as_type<char4>(a); \
                    obank[4] = fma(ES1, (U)c.x, obank[4]); \
                    obank[5] = fma(ES1, (U)c.y, obank[5]); \
                    obank[6] = fma(ES1, (U)c.z, obank[6]); \
                    obank[7] = fma(ES1, (U)c.w, obank[7]); \
                    if (QH > 0) { \
                        obank2[4] = fma(ES2, (U)c.x, obank2[4]); \
                        obank2[5] = fma(ES2, (U)c.y, obank2[5]); \
                        obank2[6] = fma(ES2, (U)c.z, obank2[6]); \
                        obank2[7] = fma(ES2, (U)c.w, obank2[7]); \
                    } \
                } else { \
                    const uchar4 a = p4[0]; \
                    obank[0] = fma(ES1, (U)((int)(a.x & 0x0F) - 8), obank[0]); \
                    obank[1] = fma(ES1, (U)((int)(a.x >> 4) - 8), obank[1]); \
                    obank[2] = fma(ES1, (U)((int)(a.y & 0x0F) - 8), obank[2]); \
                    obank[3] = fma(ES1, (U)((int)(a.y >> 4) - 8), obank[3]); \
                    obank[4] = fma(ES1, (U)((int)(a.z & 0x0F) - 8), obank[4]); \
                    obank[5] = fma(ES1, (U)((int)(a.z >> 4) - 8), obank[5]); \
                    obank[6] = fma(ES1, (U)((int)(a.w & 0x0F) - 8), obank[6]); \
                    obank[7] = fma(ES1, (U)((int)(a.w >> 4) - 8), obank[7]); \
                    if (QH > 0) { \
                        obank2[0] = fma(ES2, (U)((int)(a.x & 0x0F) - 8), obank2[0]); \
                        obank2[1] = fma(ES2, (U)((int)(a.x >> 4) - 8), obank2[1]); \
                        obank2[2] = fma(ES2, (U)((int)(a.y & 0x0F) - 8), obank2[2]); \
                        obank2[3] = fma(ES2, (U)((int)(a.y >> 4) - 8), obank2[3]); \
                        obank2[4] = fma(ES2, (U)((int)(a.z & 0x0F) - 8), obank2[4]); \
                        obank2[5] = fma(ES2, (U)((int)(a.z >> 4) - 8), obank2[5]); \
                        obank2[6] = fma(ES2, (U)((int)(a.w & 0x0F) - 8), obank2[6]); \
                        obank2[7] = fma(ES2, (U)((int)(a.w >> 4) - 8), obank2[7]); \
                    } \
                } \
            }

        for (int n = block_idx; n <= n_full; n += blocks) {
            const U ks = static_cast<U>(*ks_ptr);
            float4 dot1 = 0.0f;
            float4 dot2 = 0.0f;
            MTPLX_QDOT(k_ptr, dot1, dot2)
            float4 score = dot1 * ks;
            float4 score2 = dot2 * ks;
            for (int off = 16; off > 0; off >>= 1) {
                score += simd_shuffle_xor(score, off);
                if (QH > 0) score2 += simd_shuffle_xor(score2, off);
            }
            float4 new_max = metal::max(max_score, score);
            float4 factor = fast::exp(max_score - new_max);
            float4 exp_score = fast::exp(score - new_max);
            max_score = new_max;
            sum_exp = sum_exp * factor + exp_score;
            float4 factor2 = 1.0f;
            float4 exp_score2 = 0.0f;
            if (QH > 0) {
                float4 new_max2 = metal::max(max_score2, score2);
                factor2 = fast::exp(max_score2 - new_max2);
                exp_score2 = fast::exp(score2 - new_max2);
                max_score2 = new_max2;
                sum_exp2 = sum_exp2 * factor2 + exp_score2;
            }
            const U vs = static_cast<U>(*vs_ptr);
            const float4 es1 = exp_score * vs;
            const float4 es2v = exp_score2 * vs;
            MTPLX_VUPD(v_ptr, es1, es2v, factor, factor2)
            k_ptr += (size_t)blocks * KROW;
            v_ptr += (size_t)blocks * KROW;
            ks_ptr += blocks;
            vs_ptr += blocks;
        }

        {
            const int first = n_full + 1;
            int n0 = block_idx;
            if (n0 < first) {
                const int steps = (first - n0 + blocks - 1) / blocks;
                n0 += steps * blocks;
            }
            const device uint8_t* k_tail = k_q
                + (size_t)kv_head_idx * k_head_seq * KROW
                + (size_t)n0 * KROW + simd_lid * k_bytes_per_thread;
            const device uint8_t* v_tail = v_q
                + (size_t)kv_head_idx * v_head_seq * KROW
                + (size_t)n0 * KROW + simd_lid * k_bytes_per_thread;
            const device float* ks_tail = k_scale
                + (size_t)kv_head_idx * k_head_seq + n0;
            const device float* vs_tail = v_scale
                + (size_t)kv_head_idx * v_head_seq + n0;
            for (int n = n0; n < n_kv; n += blocks) {
                const U ks = static_cast<U>(*ks_tail);
                float4 dot1 = 0.0f;
                float4 dot2 = 0.0f;
                MTPLX_QDOT(k_tail, dot1, dot2)
                float4 score = dot1 * ks;
                float4 score2 = dot2 * ks;
                for (int off = 16; off > 0; off >>= 1) {
                    score += simd_shuffle_xor(score, off);
                    if (QH > 0) score2 += simd_shuffle_xor(score2, off);
                }
                float4 vis;
                vis.x = (n <= n_kv - QL + 0) ? 1.0f : 0.0f;
                vis.y = (QL > 1 && n <= n_kv - QL + 1) ? 1.0f : 0.0f;
                vis.z = (QL > 2 && n <= n_kv - QL + 2) ? 1.0f : 0.0f;
                vis.w = (QL > 3 && n <= n_kv - QL + 3) ? 1.0f : 0.0f;
                score = score * vis + (1.0f - vis) * Limits<float>::finite_min;
                float4 new_max = metal::max(max_score, score);
                float4 factor = fast::exp(max_score - new_max);
                float4 exp_score = fast::exp(score - new_max) * vis;
                max_score = new_max;
                sum_exp = sum_exp * factor + exp_score;
                float4 factor2 = 1.0f;
                float4 exp_score2 = 0.0f;
                if (QH > 0) {
                    float4 vis2;
                    vis2.x = (n <= n_kv - QL + 4) ? 1.0f : 0.0f;
                    vis2.y = (QH > 1 && n <= n_kv - QL + 5) ? 1.0f : 0.0f;
                    vis2.z = (QH > 2 && n <= n_kv - QL + 6) ? 1.0f : 0.0f;
                    vis2.w = (QH > 3 && n <= n_kv - QL + 7) ? 1.0f : 0.0f;
                    score2 = score2 * vis2
                        + (1.0f - vis2) * Limits<float>::finite_min;
                    float4 new_max2 = metal::max(max_score2, score2);
                    factor2 = fast::exp(max_score2 - new_max2);
                    exp_score2 = fast::exp(score2 - new_max2) * vis2;
                    max_score2 = new_max2;
                    sum_exp2 = sum_exp2 * factor2 + exp_score2;
                }
                const U vs = static_cast<U>(*vs_tail);
                const float4 es1 = exp_score * vs;
                const float4 es2v = exp_score2 * vs;
                MTPLX_VUPD(v_tail, es1, es2v, factor, factor2)
                k_tail += (size_t)blocks * KROW;
                v_tail += (size_t)blocks * KROW;
                ks_tail += blocks;
                vs_tail += blocks;
            }
        }

        for (int j = 0; j < QL; ++j) {
            const int o_offset = q_head_idx * QL + j;
            // PartT, not InT: the numerator is not yet divided by sum_exp
            // (see sdpa_2pass.unnormalized_partials_dtype).
            device PartT* p = partials
                + ((size_t)o_offset * blocks + block_idx) * V
                + simd_lid * v_per_thread;
            for (int i = 0; i < v_per_thread; ++i) {
                const float4 ob = (j < 4) ? obank[i] : obank2[i];
                const float val = (j % 4 == 0) ? ob.x
                    : (j % 4 == 1) ? ob.y : (j % 4 == 2) ? ob.z : ob.w;
                p[i] = static_cast<PartT>(val);
            }
            if (simd_lid == 0) {
                const float se = (j < 4) ? sum_exp[j] : sum_exp2[j - 4];
                const float ms = (j < 4) ? max_score[j] : max_score2[j - 4];
                sums[o_offset * blocks + block_idx] = se;
                maxs[o_offset * blocks + block_idx] = ms;
            }
        }
    
    """
    return mx.fast.metal_kernel(
        name="mtplx_sdpa_gqa_packed_quant_partials",
        input_names=[
            "queries",
            "k_q",
            "k_scale",
            "v_q",
            "v_scale",
            "offset",
            "k_head_seq",
            "v_head_seq",
            "scale",
            "blocks",
        ],
        output_names=["partials", "sums", "maxs"],
        source=source,
    )


def sdpa_gqa_packed_tail_quant(
    *,
    queries,
    k_q,
    k_scale,
    v_q,
    v_scale,
    offset,
    scale,
    bits: int,
    max_q_len: int = 8,
    max_offset: int | None = None,
):
    """Packed-quant GQA tail attention. Returns None (bail) on contract miss.

    queries: (1, HQ, QL, D) bf16/fp16. k_q/v_q: (1, HKV, capacity, D) int8 for
    q8 or (1, HKV, capacity, D//2) uint8 for q4 (kv_quant even|odd<<4 packing).
    k_scale/v_scale: (1, HKV, capacity, 1) fp32 rowwise. Same whole-buffer
    contract as the bf16 kernel: pass allocated buffers, never sliced views.

    ``offset`` may be an ``mx.array`` scalar (compiled cache state) and
    ``max_offset`` a static attention ceiling (compiled verify bucket): with
    both, every host-side value feeding shapes/geometry is static per bucket,
    so the call is safe inside an ``mx.compile`` graph.
    """
    bits = int(bits)
    if bits not in (8, 4):
        return _bail("quant_bits")
    bsz, hq, q_len, d = (int(x) for x in queries.shape)
    if bsz != 1:
        return _bail("batch_size")
    if q_len < 1 or q_len > min(8, int(max_q_len)):
        return _bail("q_len")
    hk = int(k_q.shape[1])
    capacity = int(k_q.shape[2])
    if int(v_q.shape[1]) != hk or int(v_q.shape[2]) != capacity:
        return _bail("kv_layout_mismatch")
    packed = d if bits == 8 else d // 2
    if int(k_q.shape[3]) != packed or int(v_q.shape[3]) != packed:
        return _bail("packed_dim_mismatch")
    for s in (k_scale, v_scale):
        if int(s.shape[1]) != hk or int(s.shape[2]) != capacity:
            return _bail("scale_layout_mismatch")
        if s.dtype != mx.float32:
            return _bail("scale_dtype")
    if d not in (64, 128, 256):
        return _bail("head_dim_unsupported")
    if bits == 4 and d != 256:
        # 2026-08-26 forensics: the q4 nibble path deviates from its own
        # dequant math at D=128 (0.006-0.026 abs vs <2e-4 at D=256) —
        # geometry-scoped defect, receipts in MEASUREMENTS.md. Fail closed
        # outside the verified envelope until the D<256 path is fixed.
        return _bail("q4_head_dim_envelope")
    if hk <= 0 or hq % hk:
        return _bail("gqa_heads")
    gqa_factor = hq // hk
    if 32 * gqa_factor > 1024:
        return _bail("threadgroup_width")
    if queries.dtype not in (mx.bfloat16, mx.float16):
        return _bail("query_dtype")
    expect_kv = mx.int8 if bits == 8 else mx.uint8
    if k_q.dtype != expect_kv or v_q.dtype != expect_kv:
        return _bail("kv_dtype")

    if isinstance(offset, mx.array):
        if offset.size != 1:
            return _bail("offset_shape")
        # The integer branch below bails past the buffers; an array offset
        # used to reach the kernel unchecked, and its walk reads offset rows
        # of k_q/v_q and both scale planes whatever their size. An eager
        # offset past the capacity means the cache already outran its
        # buffers, so this refuses loudly instead of reading past them. A
        # traced offset cannot be read here; the compiled verify bank only
        # dispatches a bucket whose offset + window fits the capacity.
        concrete = _concrete_offset(offset)
        if concrete is not None and concrete > capacity:
            raise ValueError(
                f"packed-quant attention offset {concrete} is past the KV "
                f"buffers' {capacity} rows; the cache wrote beyond its capacity"
            )
        offset_arr = offset.astype(mx.int32).reshape(1)
    else:
        offset_int = int(offset)
        if offset_int <= 0 or offset_int > capacity:
            return _bail("offset_range")
        offset_arr = mx.array([offset_int], dtype=mx.int32)

    blocks = _static_blocks(capacity, max_offset)
    if blocks <= 0 or blocks % 32:
        return _bail("blocks_geometry")

    kernel = _quant_partials_kernel()
    reduce_kernel = _paged_reduce_kernel()
    if kernel is None or reduce_kernel is None:
        return _bail("kernel_unavailable")

    # int8 buffers reach Metal as uint8 pointers; the kernel casts per KBITS.
    k_q_view = k_q.view(mx.uint8) if bits == 8 else k_q
    v_q_view = v_q.view(mx.uint8) if bits == 8 else v_q

    partial_shape = (bsz, hq, q_len, blocks, d)
    stats_shape = (bsz, hq, q_len, blocks)
    partial_dtype = unnormalized_partials_dtype(queries.dtype)
    partials, sums, maxs = kernel(
        inputs=[
            queries,
            k_q_view,
            k_scale,
            v_q_view,
            v_scale,
            offset_arr,
            capacity,
            capacity,
            float(scale),
            int(blocks),
        ],
        template=[
            ("InT", queries.dtype),
            ("PartT", partial_dtype),
            ("D", d),
            ("V", d),
            ("GQA_F", gqa_factor),
            ("QL", q_len),
            ("KBITS", bits),
        ],
        grid=(hk * 32, gqa_factor, blocks),
        threadgroup=(32, gqa_factor, 1),
        output_shapes=[partial_shape, stats_shape, stats_shape],
        output_dtypes=[partial_dtype, mx.float32, mx.float32],
    )

    (out,) = reduce_kernel(
        inputs=[partials, sums, maxs, int(blocks)],
        template=[
            ("InT", queries.dtype),
            ("V", d),
        ],
        grid=(bsz * hq * 1024, q_len, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[queries.shape],
        output_dtypes=[queries.dtype],
    )
    return out
