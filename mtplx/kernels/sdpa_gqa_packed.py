"""Packed-row GQA verify attention for tiny query windows over long KV.

Why this kernel exists (2026-07-05 speed war, Lane A):

MLX's fused SDPA routes q_len 2..8 to ``sdpa_vector_2pass`` with a
``(32, gqa, q_len)`` threadgroup — every (gqa x q_len) simdgroup issues its
own device loads for the *same* KV rows.  At the Qwen3.6-27B verify shape
(Hq=24, Hk=4, D=256, q_len=4) that measured ~160 GB/s useful KV bandwidth at
128k context vs ~387 GB/s for the q=1 vector kernel — the single largest
term of the long-context decode wall (49 ms of a 152 ms verify call).

This kernel keeps the q=1 thread topology — threadgroup ``(32, gqa, 1)``,
one KV block-lane streamed exactly once per simdgroup — and carries all
``q_len`` query rows in registers.  The per-row score reductions are packed
into a single ``float4`` simd_shuffle_xor butterfly so the shuffle-chain
latency is paid once per KV row instead of ``q_len`` times.  Measured
2026-07-05 (M5 Max, bf16, per-iteration eval): 207 GB/s useful at 128k
(+27% vs fused q=4), 185 GB/s at 65k (+22%), ties-or-wins at 16k.  Max
|diff| vs an fp32 reference: 0.0011 (stock fused: 0.0006) — same numeric
class; acceptance-decision parity is gated separately before any default.

Contract:
- ``queries``: ``[1, Hq, q_len, D]`` bf16/fp16, ``2 <= q_len <= 8`` (rows
  0..3 in the original float4 bank; rows 4..7 in a second bank added for
  the 2026-07-21 D4 campaign — q_len 5 is depth 4's verify window. The
  second bank compiles out at QL <= 4, keeping the shipping D3 shape
  byte-identical).
- ``keys``/``values``: the FULL capacity-padded contiguous buffers
  (``KVCache.keys`` / ``TensorOffsetKVCache.cache[0]``) — never the
  ``[..., :offset, :]`` views, which would force a whole-buffer copy.
- ``offset``: rows in use (python int or int32 scalar ``mx.array`` for
  compiled graphs).  Rows ``>= offset`` are never read.
- Semantics: tail-causal — query row ``j`` attends to rows
  ``n <= offset - q_len + j`` — identical to ``make_mask``'s tail window
  and to fused SDPA's ``mask="causal"`` with a KV cache.
"""

from __future__ import annotations

from functools import lru_cache
import os

import mlx.core as mx

from .sdpa_2pass_paged import _paged_reduce_kernel, unnormalized_partials_dtype

# F23b (2026-08-16): contract bails, keyed by the first gate that declined.
# Every ``return None`` below silently routes the caller back to fused SDPA
# — "looks like turbo, runs stock" is invisible without this. Import-stable
# surface for /health; increments happen ONLY on bail paths (an engaged
# call never touches this dict).
gqa_packed_bail_counts: dict[str, int] = {}


def _bail(reason: str) -> None:
    gqa_packed_bail_counts[reason] = gqa_packed_bail_counts.get(reason, 0) + 1
    return None


def _env_blocks_override() -> int:
    raw = (os.environ.get("MTPLX_GQA_PACKED_SDPA_BLOCKS") or "").strip()
    if not raw:
        return 0
    try:
        return max(0, int(raw))
    except ValueError:
        return 0


def _blocks_for_capacity(capacity: int) -> int:
    """Block-lane count tuned on M5 Max (2026-07-05 race, proto5)."""

    override = _env_blocks_override()
    if override:
        return override
    if capacity >= 65536:
        return 1024
    if capacity >= 16384:
        return 512
    return 256


@lru_cache(maxsize=None)
def _packed_partials_kernel():
    if not mx.metal.is_available():
        return None

    # QL is a template constant in [2, 8]. Rows 0..3 ride the first float4
    # bank; rows 4..7 ride a second bank that compiles out entirely at
    # QL <= 4 (the 2026-07-05 shipping shape pays zero extra cost). Lanes
    # above QL inside a bank are dead weight (zero query, never written).
    source = """
        constexpr int BD = 32;
        constexpr int qk_per_thread = D / BD;
        constexpr int v_per_thread = V / BD;
        constexpr int QH = (QL > 4) ? (QL - 4) : 0;

        typedef float U;

        const int kv_head_idx = threadgroup_position_in_grid.x;
        const int block_idx = threadgroup_position_in_grid.z;
        const int gqa_idx = thread_position_in_threadgroup.y;
        const int simd_lid = thread_index_in_simdgroup;
        const int n_kv = static_cast<int>(offset[0]);

        const int q_head_idx = kv_head_idx * GQA_F + gqa_idx;

        thread U q[QL][qk_per_thread];
        thread U o[QL][v_per_thread];
        float4 max_score = Limits<U>::finite_min;
        float4 sum_exp = 0.0f;
        float4 max_score2 = Limits<U>::finite_min;
        float4 sum_exp2 = 0.0f;

        for (int j = 0; j < QL; ++j) {
            const device InT* q_ptr = queries
                + (q_head_idx * QL + j) * D + simd_lid * qk_per_thread;
            for (int i = 0; i < qk_per_thread; ++i) {
                q[j][i] = static_cast<U>(scale) * static_cast<U>(q_ptr[i]);
            }
            for (int i = 0; i < v_per_thread; ++i) {
                o[j][i] = 0.0f;
            }
        }

        const device InT* k_ptr = keys
            + (size_t)kv_head_idx * k_head_seq * D
            + (size_t)block_idx * D
            + simd_lid * qk_per_thread;
        const device InT* v_ptr = values
            + (size_t)kv_head_idx * v_head_seq * D
            + (size_t)block_idx * D
            + simd_lid * v_per_thread;

        // Rows visible to every query row run branch-free; the last QL-1
        // rows take the per-row causal predicate in the tail loop below.
        const int n_full = n_kv - QL;

        for (int n = block_idx; n <= n_full; n += blocks) {
            U k_vec[qk_per_thread];
            for (int i = 0; i < qk_per_thread; ++i) {
                k_vec[i] = static_cast<U>(k_ptr[i]);
            }
            float4 score = 0.0f;
            float4 score2 = 0.0f;
            for (int i = 0; i < qk_per_thread; ++i) {
                score.x += q[0][i] * k_vec[i];
                if (QL > 1) score.y += q[1][i] * k_vec[i];
                if (QL > 2) score.z += q[2][i] * k_vec[i];
                if (QL > 3) score.w += q[3][i] * k_vec[i];
                if (QH > 0) score2.x += q[4][i] * k_vec[i];
                if (QH > 1) score2.y += q[5][i] * k_vec[i];
                if (QH > 2) score2.z += q[6][i] * k_vec[i];
                if (QH > 3) score2.w += q[7][i] * k_vec[i];
            }
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
            for (int i = 0; i < v_per_thread; ++i) {
                const U v = static_cast<U>(v_ptr[i]);
                o[0][i] = o[0][i] * factor.x + exp_score.x * v;
                if (QL > 1) o[1][i] = o[1][i] * factor.y + exp_score.y * v;
                if (QL > 2) o[2][i] = o[2][i] * factor.z + exp_score.z * v;
                if (QL > 3) o[3][i] = o[3][i] * factor.w + exp_score.w * v;
                if (QH > 0) o[4][i] = o[4][i] * factor2.x + exp_score2.x * v;
                if (QH > 1) o[5][i] = o[5][i] * factor2.y + exp_score2.y * v;
                if (QH > 2) o[6][i] = o[6][i] * factor2.z + exp_score2.z * v;
                if (QH > 3) o[7][i] = o[7][i] * factor2.w + exp_score2.w * v;
            }
            k_ptr += (size_t)blocks * D;
            v_ptr += (size_t)blocks * D;
        }

        {
            const int first = n_full + 1;
            int n0 = block_idx;
            if (n0 < first) {
                const int steps = (first - n0 + blocks - 1) / blocks;
                n0 += steps * blocks;
            }
            const device InT* k_tail = keys
                + (size_t)kv_head_idx * k_head_seq * D
                + (size_t)n0 * D + simd_lid * qk_per_thread;
            const device InT* v_tail = values
                + (size_t)kv_head_idx * v_head_seq * D
                + (size_t)n0 * D + simd_lid * v_per_thread;
            for (int n = n0; n < n_kv; n += blocks) {
                U k_vec[qk_per_thread];
                for (int i = 0; i < qk_per_thread; ++i) {
                    k_vec[i] = static_cast<U>(k_tail[i]);
                }
                float4 score = 0.0f;
                float4 score2 = 0.0f;
                for (int i = 0; i < qk_per_thread; ++i) {
                    score.x += q[0][i] * k_vec[i];
                    if (QL > 1) score.y += q[1][i] * k_vec[i];
                    if (QL > 2) score.z += q[2][i] * k_vec[i];
                    if (QL > 3) score.w += q[3][i] * k_vec[i];
                    if (QH > 0) score2.x += q[4][i] * k_vec[i];
                    if (QH > 1) score2.y += q[5][i] * k_vec[i];
                    if (QH > 2) score2.z += q[6][i] * k_vec[i];
                    if (QH > 3) score2.w += q[7][i] * k_vec[i];
                }
                for (int off = 16; off > 0; off >>= 1) {
                    score += simd_shuffle_xor(score, off);
                    if (QH > 0) score2 += simd_shuffle_xor(score2, off);
                }
                // Query row j attends to n iff n <= n_kv - QL + j.
                float4 vis;
                vis.x = (n <= n_kv - QL + 0) ? 1.0f : 0.0f;
                vis.y = (QL > 1 && n <= n_kv - QL + 1) ? 1.0f : 0.0f;
                vis.z = (QL > 2 && n <= n_kv - QL + 2) ? 1.0f : 0.0f;
                vis.w = (QL > 3 && n <= n_kv - QL + 3) ? 1.0f : 0.0f;
                score = score * vis + (1.0f - vis) * Limits<U>::finite_min;
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
                    score2 = score2 * vis2 + (1.0f - vis2) * Limits<U>::finite_min;
                    float4 new_max2 = metal::max(max_score2, score2);
                    factor2 = fast::exp(max_score2 - new_max2);
                    exp_score2 = fast::exp(score2 - new_max2) * vis2;
                    max_score2 = new_max2;
                    sum_exp2 = sum_exp2 * factor2 + exp_score2;
                }
                for (int i = 0; i < v_per_thread; ++i) {
                    const U v = static_cast<U>(v_tail[i]);
                    o[0][i] = o[0][i] * factor.x + exp_score.x * v;
                    if (QL > 1) o[1][i] = o[1][i] * factor.y + exp_score.y * v;
                    if (QL > 2) o[2][i] = o[2][i] * factor.z + exp_score.z * v;
                    if (QL > 3) o[3][i] = o[3][i] * factor.w + exp_score.w * v;
                    if (QH > 0) o[4][i] = o[4][i] * factor2.x + exp_score2.x * v;
                    if (QH > 1) o[5][i] = o[5][i] * factor2.y + exp_score2.y * v;
                    if (QH > 2) o[6][i] = o[6][i] * factor2.z + exp_score2.z * v;
                    if (QH > 3) o[7][i] = o[7][i] * factor2.w + exp_score2.w * v;
                }
                k_tail += (size_t)blocks * D;
                v_tail += (size_t)blocks * D;
            }
        }

        for (int j = 0; j < QL; ++j) {
            const int o_offset = q_head_idx * QL + j;
            device PartT* p = partials
                + ((size_t)o_offset * blocks + block_idx) * V
                + simd_lid * v_per_thread;
            for (int i = 0; i < v_per_thread; ++i) {
                p[i] = static_cast<PartT>(o[j][i]);
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
        name="mtplx_sdpa_gqa_packed_partials",
        input_names=[
            "queries",
            "keys",
            "values",
            "offset",
            "k_head_seq",
            "v_head_seq",
            "scale",
            "blocks",
        ],
        output_names=["partials", "sums", "maxs"],
        source=source,
    )


def sdpa_gqa_packed_tail(
    *,
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    offset: int | mx.array,
    scale: float,
    max_q_len: int = 8,
) -> mx.array | None:
    """Tail-causal SDPA over the first ``offset`` rows of full KV buffers.

    Returns ``None`` when the shape/dtype contract is not met so callers can
    fall back to the stock fused path.
    """

    if not mx.metal.is_available():
        return _bail("metal_unavailable")
    if queries.ndim != 4 or keys.ndim != 4 or values.ndim != 4:
        return _bail("ndim")
    bsz, hq, q_len, d = (int(x) for x in queries.shape)
    if bsz != 1:
        return _bail("batch_size")
    if q_len < 2 or q_len > min(8, int(max_q_len)):
        return _bail("q_len")
    hk = int(keys.shape[1])
    capacity = int(keys.shape[2])
    if int(values.shape[1]) != hk or int(values.shape[2]) != capacity:
        return _bail("kv_layout_mismatch")
    kd = int(keys.shape[3])
    vdim = int(values.shape[3])
    if kd != d or vdim != d:
        return _bail("head_dim_mismatch")
    if d not in (64, 96, 128, 256):
        return _bail("head_dim_unsupported")
    if hk <= 0 or hq % hk:
        return _bail("gqa_heads")
    gqa_factor = hq // hk
    if 32 * gqa_factor > 1024:
        return _bail("threadgroup_width")
    if queries.dtype not in (mx.bfloat16, mx.float16):
        return _bail("query_dtype")
    if keys.dtype != queries.dtype or values.dtype != queries.dtype:
        return _bail("kv_dtype_mismatch")
    # NOTE: callers must pass the whole allocated buffers (contiguous by
    # construction), never `[..., :offset, :]` views. MLX python exposes no
    # contiguity flag to assert on; a sliced view would still be CORRECT
    # (metal_kernel's ensure_row_contiguous copies it) but would silently
    # reintroduce the whole-buffer copy this kernel exists to avoid.

    if isinstance(offset, mx.array):
        if offset.size != 1:
            return _bail("offset_shape")
        offset_arr = offset.astype(mx.int32).reshape(1)
    else:
        offset_int = int(offset)
        if offset_int <= 0 or offset_int > capacity:
            return _bail("offset_range")
        offset_arr = mx.array([offset_int], dtype=mx.int32)

    blocks = _blocks_for_capacity(capacity)
    if blocks <= 0 or blocks % 32:
        return _bail("blocks_geometry")

    kernel = _packed_partials_kernel()
    reduce_kernel = _paged_reduce_kernel()
    if kernel is None or reduce_kernel is None:
        return _bail("kernel_unavailable")

    partial_shape = (bsz, hq, q_len, blocks, vdim)
    stats_shape = (bsz, hq, q_len, blocks)
    partial_dtype = unnormalized_partials_dtype(queries.dtype)
    partials, sums, maxs = kernel(
        inputs=[
            queries,
            keys,
            values,
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
            ("V", vdim),
            ("GQA_F", gqa_factor),
            ("QL", q_len),
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
            ("V", vdim),
        ],
        grid=(bsz * hq * 1024, q_len, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[queries.shape],
        output_dtypes=[queries.dtype],
    )
    return out


@lru_cache(maxsize=None)
def _grouped_partials_kernel():
    """Query-group variant: each workgroup owns <=4 rows of a wide QL window.

    The 2026-08-25 QL sweep measured the second float4 bank as the depth
    cliff (QL4->5 = 2.3x, QL8 = 4-8x vs QL4 at 40-71k): activating bank 2
    doubles per-thread register state and collapses occupancy. This variant
    keeps the PROVEN one-bank topology and widens by threadgroup grid
    instead: grid.y = GQA_F * QGROUPS, each workgroup handling query rows
    [qgroup*4, qgroup*4 + nq). KV rows are re-streamed per group (K1/K3:
    redundant GQA loads are cache-served; register fan-out is what kills).
    """
    if not mx.metal.is_available():
        return None

    source = """
        constexpr int BD = 32;
        constexpr int qk_per_thread = D / BD;
        constexpr int v_per_thread = V / BD;

        typedef float U;

        const int kv_head_idx = threadgroup_position_in_grid.x;
        const int qgroup = threadgroup_position_in_grid.y;
        const int block_idx = threadgroup_position_in_grid.z;
        const int gqa_idx = thread_position_in_threadgroup.y;
        const int simd_lid = thread_index_in_simdgroup;
        const int n_kv = static_cast<int>(offset[0]);

        const int q_head_idx = kv_head_idx * GQA_F + gqa_idx;
        const int q0 = qgroup * 4;
        // NQ_LAST/QGROUPS are template constants: full groups fold nq to a
        // literal 4 and the tail group folds to NQ_LAST, so the dead-lane
        // predicates compile out exactly like the single-bank kernel's QL.
        // NQ_LAST may exceed 4 (mixed grouping, 2026-08-25 v3): the tail
        // group then carries QH2 = NQ_LAST - 4 rows in a small second bank.
        // QH2 <= 2 keeps register state far from the QL8 spill cliff.
        const bool is_tail = qgroup == QGROUPS - 1;
        const int nq = is_tail ? metal::min(NQ_LAST, 4) : 4;
        constexpr int QH2 = (NQ_LAST > 4) ? (NQ_LAST - 4) : 0;
        const int nq2 = is_tail ? QH2 : 0;

        thread U q[4][qk_per_thread];
        thread U o[4][v_per_thread];
        float4 max_score = Limits<U>::finite_min;
        float4 sum_exp = 0.0f;
        thread U q2[QH2 > 0 ? QH2 : 1][qk_per_thread];
        thread U o2[QH2 > 0 ? QH2 : 1][v_per_thread];
        float2 max_score2 = Limits<U>::finite_min;
        float2 sum_exp2 = 0.0f;

        for (int j = 0; j < 4; ++j) {
            const bool live = j < nq;
            const device InT* q_ptr = queries
                + ((size_t)q_head_idx * QL + q0 + (live ? j : 0)) * D
                + simd_lid * qk_per_thread;
            for (int i = 0; i < qk_per_thread; ++i) {
                q[j][i] = live ? static_cast<U>(scale) * static_cast<U>(q_ptr[i]) : U(0);
            }
            for (int i = 0; i < v_per_thread; ++i) {
                o[j][i] = 0.0f;
            }
        }
        if (QH2 > 0) {
            for (int j = 0; j < QH2; ++j) {
                const bool live = j < nq2;
                const device InT* q_ptr = queries
                    + ((size_t)q_head_idx * QL + q0 + 4 + (live ? j : 0)) * D
                    + simd_lid * qk_per_thread;
                for (int i = 0; i < qk_per_thread; ++i) {
                    q2[j][i] = live ? static_cast<U>(scale) * static_cast<U>(q_ptr[i]) : U(0);
                }
                for (int i = 0; i < v_per_thread; ++i) {
                    o2[j][i] = 0.0f;
                }
            }
        }

        const device InT* k_ptr = keys
            + (size_t)kv_head_idx * k_head_seq * D
            + (size_t)block_idx * D
            + simd_lid * qk_per_thread;
        const device InT* v_ptr = values
            + (size_t)kv_head_idx * v_head_seq * D
            + (size_t)block_idx * D
            + simd_lid * v_per_thread;

        // Rows visible to every live row of this group run branch-free;
        // the per-row causal predicate only matters in the last QL rows.
        const int n_full = n_kv - QL + q0;

        for (int n = block_idx; n <= n_full; n += blocks) {
            U k_vec[qk_per_thread];
            for (int i = 0; i < qk_per_thread; ++i) {
                k_vec[i] = static_cast<U>(k_ptr[i]);
            }
            float4 score = 0.0f;
            float2 score2 = 0.0f;
            for (int i = 0; i < qk_per_thread; ++i) {
                score.x += q[0][i] * k_vec[i];
                score.y += q[1][i] * k_vec[i];
                score.z += q[2][i] * k_vec[i];
                score.w += q[3][i] * k_vec[i];
                if (QH2 > 0) score2.x += q2[0][i] * k_vec[i];
                if (QH2 > 1) score2.y += q2[QH2 > 1 ? 1 : 0][i] * k_vec[i];
            }
            for (int off = 16; off > 0; off >>= 1) {
                score += simd_shuffle_xor(score, off);
                if (QH2 > 0) score2 += simd_shuffle_xor(score2, off);
            }
            float4 vis;
            vis.x = (0 < nq) ? 1.0f : 0.0f;
            vis.y = (1 < nq) ? 1.0f : 0.0f;
            vis.z = (2 < nq) ? 1.0f : 0.0f;
            vis.w = (3 < nq) ? 1.0f : 0.0f;
            score = score * vis + (1.0f - vis) * Limits<U>::finite_min;
            float4 new_max = metal::max(max_score, score);
            float4 factor = fast::exp(max_score - new_max);
            float4 exp_score = fast::exp(score - new_max) * vis;
            max_score = new_max;
            sum_exp = sum_exp * factor + exp_score;
            float2 factor2 = 1.0f;
            float2 exp_score2 = 0.0f;
            if (QH2 > 0) {
                float2 vis2;
                vis2.x = (0 < nq2) ? 1.0f : 0.0f;
                vis2.y = (1 < nq2) ? 1.0f : 0.0f;
                score2 = score2 * vis2 + (1.0f - vis2) * Limits<U>::finite_min;
                float2 new_max2 = metal::max(max_score2, score2);
                factor2 = fast::exp(max_score2 - new_max2);
                exp_score2 = fast::exp(score2 - new_max2) * vis2;
                max_score2 = new_max2;
                sum_exp2 = sum_exp2 * factor2 + exp_score2;
            }
            for (int i = 0; i < v_per_thread; ++i) {
                const U v = static_cast<U>(v_ptr[i]);
                o[0][i] = o[0][i] * factor.x + exp_score.x * v;
                o[1][i] = o[1][i] * factor.y + exp_score.y * v;
                o[2][i] = o[2][i] * factor.z + exp_score.z * v;
                o[3][i] = o[3][i] * factor.w + exp_score.w * v;
                if (QH2 > 0) o2[0][i] = o2[0][i] * factor2.x + exp_score2.x * v;
                if (QH2 > 1) o2[QH2 > 1 ? 1 : 0][i]
                    = o2[QH2 > 1 ? 1 : 0][i] * factor2.y + exp_score2.y * v;
            }
            k_ptr += (size_t)blocks * D;
            v_ptr += (size_t)blocks * D;
        }

        {
            const int first = n_full + 1;
            int n0 = block_idx;
            if (n0 < first) {
                const int steps = (first - n0 + blocks - 1) / blocks;
                n0 += steps * blocks;
            }
            const device InT* k_tail = keys
                + (size_t)kv_head_idx * k_head_seq * D
                + (size_t)n0 * D + simd_lid * qk_per_thread;
            const device InT* v_tail = values
                + (size_t)kv_head_idx * v_head_seq * D
                + (size_t)n0 * D + simd_lid * v_per_thread;
            for (int n = n0; n < n_kv; n += blocks) {
                U k_vec[qk_per_thread];
                for (int i = 0; i < qk_per_thread; ++i) {
                    k_vec[i] = static_cast<U>(k_tail[i]);
                }
                float4 score = 0.0f;
                float2 score2 = 0.0f;
                for (int i = 0; i < qk_per_thread; ++i) {
                    score.x += q[0][i] * k_vec[i];
                    score.y += q[1][i] * k_vec[i];
                    score.z += q[2][i] * k_vec[i];
                    score.w += q[3][i] * k_vec[i];
                    if (QH2 > 0) score2.x += q2[0][i] * k_vec[i];
                    if (QH2 > 1) score2.y += q2[QH2 > 1 ? 1 : 0][i] * k_vec[i];
                }
                for (int off = 16; off > 0; off >>= 1) {
                    score += simd_shuffle_xor(score, off);
                    if (QH2 > 0) score2 += simd_shuffle_xor(score2, off);
                }
                // Row j of this group is global row q0+j: visible iff
                // n <= n_kv - QL + q0 + j (and the row exists).
                float4 vis;
                vis.x = (0 < nq && n <= n_full + 0) ? 1.0f : 0.0f;
                vis.y = (1 < nq && n <= n_full + 1) ? 1.0f : 0.0f;
                vis.z = (2 < nq && n <= n_full + 2) ? 1.0f : 0.0f;
                vis.w = (3 < nq && n <= n_full + 3) ? 1.0f : 0.0f;
                score = score * vis + (1.0f - vis) * Limits<U>::finite_min;
                float4 new_max = metal::max(max_score, score);
                float4 factor = fast::exp(max_score - new_max);
                float4 exp_score = fast::exp(score - new_max) * vis;
                max_score = new_max;
                sum_exp = sum_exp * factor + exp_score;
                float2 factor2 = 1.0f;
                float2 exp_score2 = 0.0f;
                if (QH2 > 0) {
                    // Bank-2 row j is global row q0+4+j.
                    float2 vis2;
                    vis2.x = (0 < nq2 && n <= n_full + 4) ? 1.0f : 0.0f;
                    vis2.y = (1 < nq2 && n <= n_full + 5) ? 1.0f : 0.0f;
                    score2 = score2 * vis2 + (1.0f - vis2) * Limits<U>::finite_min;
                    float2 new_max2 = metal::max(max_score2, score2);
                    factor2 = fast::exp(max_score2 - new_max2);
                    exp_score2 = fast::exp(score2 - new_max2) * vis2;
                    max_score2 = new_max2;
                    sum_exp2 = sum_exp2 * factor2 + exp_score2;
                }
                for (int i = 0; i < v_per_thread; ++i) {
                    const U v = static_cast<U>(v_tail[i]);
                    o[0][i] = o[0][i] * factor.x + exp_score.x * v;
                    o[1][i] = o[1][i] * factor.y + exp_score.y * v;
                    o[2][i] = o[2][i] * factor.z + exp_score.z * v;
                    o[3][i] = o[3][i] * factor.w + exp_score.w * v;
                    if (QH2 > 0) o2[0][i] = o2[0][i] * factor2.x + exp_score2.x * v;
                    if (QH2 > 1) o2[QH2 > 1 ? 1 : 0][i]
                        = o2[QH2 > 1 ? 1 : 0][i] * factor2.y + exp_score2.y * v;
                }
                k_tail += (size_t)blocks * D;
                v_tail += (size_t)blocks * D;
            }
        }

        for (int j = 0; j < nq; ++j) {
            const int o_offset = q_head_idx * QL + q0 + j;
            device PartT* p = partials
                + ((size_t)o_offset * blocks + block_idx) * V
                + simd_lid * v_per_thread;
            for (int i = 0; i < v_per_thread; ++i) {
                p[i] = static_cast<PartT>(o[j][i]);
            }
            if (simd_lid == 0) {
                sums[o_offset * blocks + block_idx] = sum_exp[j];
                maxs[o_offset * blocks + block_idx] = max_score[j];
            }
        }
        for (int j = 0; j < nq2; ++j) {
            const int o_offset = q_head_idx * QL + q0 + 4 + j;
            device PartT* p = partials
                + ((size_t)o_offset * blocks + block_idx) * V
                + simd_lid * v_per_thread;
            for (int i = 0; i < v_per_thread; ++i) {
                p[i] = static_cast<PartT>(o2[j][i]);
            }
            if (simd_lid == 0) {
                sums[o_offset * blocks + block_idx] = sum_exp2[j];
                maxs[o_offset * blocks + block_idx] = max_score2[j];
            }
        }
    """
    return mx.fast.metal_kernel(
        name="mtplx_sdpa_gqa_packed_grouped_partials",
        input_names=[
            "queries",
            "keys",
            "values",
            "offset",
            "k_head_seq",
            "v_head_seq",
            "scale",
            "blocks",
        ],
        output_names=["partials", "sums", "maxs"],
        source=source,
    )


def sdpa_gqa_packed_tail_grouped(
    *,
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    offset: int | mx.array,
    scale: float,
    max_q_len: int = 16,
) -> mx.array | None:
    """Wide-QL tail-causal SDPA: query groups of <=4 rows per workgroup.

    Same contract as :func:`sdpa_gqa_packed_tail` but for ``2 <= q_len <= 16``.
    Returns ``None`` on any contract miss (callers fall back to stock).
    """

    if not mx.metal.is_available():
        return _bail("metal_unavailable")
    if queries.ndim != 4 or keys.ndim != 4 or values.ndim != 4:
        return _bail("ndim")
    bsz, hq, q_len, d = (int(x) for x in queries.shape)
    if bsz != 1:
        return _bail("batch_size")
    if q_len < 2 or q_len > min(16, int(max_q_len)):
        return _bail("q_len")
    hk = int(keys.shape[1])
    capacity = int(keys.shape[2])
    vdim = int(values.shape[3])
    if int(values.shape[1]) != hk or int(values.shape[2]) != capacity:
        return _bail("kv_shape_mismatch")
    if int(keys.shape[3]) != d:
        return _bail("head_dim_mismatch")
    if d not in (64, 96, 128, 256):
        return _bail("head_dim_unsupported")
    if hk <= 0 or hq % hk:
        return _bail("gqa_heads")
    gqa_factor = hq // hk
    if 32 * gqa_factor > 1024:
        return _bail("threadgroup_width")
    if queries.dtype not in (mx.bfloat16, mx.float16):
        return _bail("query_dtype")
    if keys.dtype != queries.dtype or values.dtype != queries.dtype:
        return _bail("kv_dtype_mismatch")

    if isinstance(offset, mx.array):
        if offset.size != 1:
            return _bail("offset_shape")
        offset_arr = offset.astype(mx.int32).reshape(1)
    else:
        offset_int = int(offset)
        if offset_int <= 0 or offset_int > capacity:
            return _bail("offset_range")
        offset_arr = mx.array([offset_int], dtype=mx.int32)

    # Mixed grouping (v3, 2026-08-25): at QL9/10 a 5-6 row tail group saves
    # a whole extra KV walk (4+5 / 4+6 vs 4+4+1 / 4+4+2). The sweep priced
    # per-walk costs 4-row 1.65ms/layer-16th, 5-row 2.62, 6-row 3.03 at 71k
    # -- two mixed walks beat three narrow ones only at QL9-10.
    if q_len in (9, 10):
        qgroups = 2
    else:
        qgroups = (q_len + 3) // 4
    base_blocks = _blocks_for_capacity(capacity)
    blocks = max(256, base_blocks // qgroups)
    blocks -= blocks % 32
    if blocks <= 0 or blocks % 32:
        return _bail("blocks_geometry")

    kernel = _grouped_partials_kernel()
    reduce_kernel = _paged_reduce_kernel()
    if kernel is None or reduce_kernel is None:
        return _bail("kernel_unavailable")

    partial_shape = (bsz, hq, q_len, blocks, vdim)
    stats_shape = (bsz, hq, q_len, blocks)
    partial_dtype = unnormalized_partials_dtype(queries.dtype)
    partials, sums, maxs = kernel(
        inputs=[
            queries,
            keys,
            values,
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
            ("V", vdim),
            ("GQA_F", gqa_factor),
            ("QL", q_len),
            ("QGROUPS", qgroups),
            ("NQ_LAST", q_len - 4 * (qgroups - 1)),
        ],
        grid=(hk * 32, gqa_factor * qgroups, blocks),
        threadgroup=(32, gqa_factor, 1),
        output_shapes=[partial_shape, stats_shape, stats_shape],
        output_dtypes=[partial_dtype, mx.float32, mx.float32],
    )

    (out,) = reduce_kernel(
        inputs=[partials, sums, maxs, int(blocks)],
        template=[
            ("InT", queries.dtype),
            ("V", vdim),
        ],
        grid=(bsz * hq * 1024, q_len, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[queries.shape],
        output_dtypes=[queries.dtype],
    )
    return out
