"""Prefill relative attention on MLX's steel attention kernel.

MLX's own causal attention reaches ~7 TFLOPS at head dim 128 on M1 Max; the
relative-bias kernel from the upstream port ~4.2 (scalar transposing tile
loads, half fragments). This is MLX's steel_attention body (vectorised
BlockLoaderT tiles, float fragments, BQ 32 / BK 16, 4 simdgroups) with its
mask step replaced by Inkling's: causal by key position, the sliding window,
the first visible key, and the relative bias added for distances below R;
the online softmax of each row starts from that row's sink logit. Output
layout (B, L, Hq, D), as ``relative_attention``.
"""

from __future__ import annotations

from functools import lru_cache

import mlx.core as mx

from .inkling2_seg_gemm import _inline

_BQ, _BK, _WM, _WN = 32, 16, 4, 1


@lru_cache(maxsize=None)
def _header() -> str:
    seen = {"mlx/backend/metal/kernels/utils.h", "mlx/backend/metal/kernels/bf16.h",
            "mlx/backend/metal/kernels/bf16_math.h", "mlx/backend/metal/kernels/complex.h"}
    return _inline("mlx/backend/metal/kernels/steel/attn/attn.h", seen) + r"""
using namespace mlx::steel;
struct InkMaxOp { template <typename T> METAL_FUNC static constexpr T apply(T x, T y) { return metal::max(x, y); } };
struct InkSumOp { template <typename T> METAL_FUNC static constexpr T apply(T x, T y) { return x + y; } };
struct InkMulOp { template <typename T> METAL_FUNC static constexpr T apply(T x, T y) { return x * y; } };
struct InkExpSubOp { template <typename T> METAL_FUNC static constexpr T apply(T x, T y) { return fast::exp2(x - y); } };
struct InkDivOp { template <typename T> METAL_FUNC static constexpr T apply(T x, T y) { return x / y; } };
"""


_SRC = r"""
    typedef float AccumType;
    const uint simd_lane_id = thread_index_in_simdgroup;
    const uint simd_group_id = simdgroup_index_in_threadgroup;
    const uint3 tid = threadgroup_position_in_grid;
    const int L = sizes[0];
    const int S = sizes[1];
    const int qL_off = S - L;                  // query row r is key position qL_off + r
    const int start = key_start[tid.z];
    const int h = tid.y;
    const ulong kv_head_idx = h / (HQ / HKV);
    const int q0 = tid.x * BQ;

    const device T* Qp = q + tid.z * q_strides[0] + h * q_strides[1] + (ulong)q0 * q_strides[2];
    const device T* Kp = k + tid.z * k_strides[0] + kv_head_idx * k_strides[1];
    const device T* Vp = v + tid.z * v_strides[0] + kv_head_idx * v_strides[1];
    const ulong O_ld = (ulong)HQ * BD;
    device T* Op = out + ((ulong)tid.z * L + q0) * O_ld + (ulong)h * BD;

    constexpr short padQ = 16 / sizeof(T);
    constexpr short padK = 16 / sizeof(T);
    constexpr short padV = 16 / sizeof(T);
    constexpr short LDQ_tgp = BD + padQ;
    constexpr short LDK_tgp = BK + padK;
    constexpr short LDV_tgp = BD + padV;
    constexpr short tgp_mem_0 = (BK + padK) * (BD);
    constexpr short tgp_mem_1 = BK * (BD + padV);
    constexpr short tgp_mem_s = tgp_mem_0 > tgp_mem_1 ? tgp_mem_0 : tgp_mem_1;
    threadgroup T Q_smem[BQ * (BD + padQ)];
    threadgroup T KV_smem[tgp_mem_s];
    threadgroup T* Qs = Q_smem;
    threadgroup T* Ks = KV_smem;
    threadgroup T* Vs = KV_smem;

    using QBlockLoader = BlockLoaderT<T, BQ, BD, LDQ_tgp, 1, 1, WM * WN * 32>;
    using KBlockLoader = BlockLoaderT<T, BK, BD, 1, LDK_tgp, 0, WM * WN * 32>;
    using VBlockLoader = BlockLoaderT<T, BK, BD, LDV_tgp, 1, 0, WM * WN * 32>;

    // the first key block the rows of this query block can see
    const int q_last = min(L, q0 + BQ) - 1;
    int first_key = WINDOW > 0 ? max(0, qL_off + q0 - WINDOW + 1) : 0;
    first_key = max(first_key, start);
    const int kb_start = first_key / BK;
    const int kb_lim = min((S + BK - 1) / BK, (qL_off + q_last) / BK + 1);

    QBlockLoader loader_q(Qp, (int)q_strides[2], Qs, simd_group_id, simd_lane_id);
    KBlockLoader loader_k(Kp + (ulong)kb_start * BK * k_strides[2], (int)k_strides[2], Ks, simd_group_id, simd_lane_id);
    VBlockLoader loader_v(Vp + (ulong)kb_start * BK * v_strides[2], (int)v_strides[2], Vs, simd_group_id, simd_lane_id);

    const AccumType scale = sc[0] * M_LOG2E_F;
    constexpr short kFragSize = 8;
    using MMAFrag_acc_t = BaseMMAFrag<AccumType, kFragSize, kFragSize>;
    constexpr int kNWarps = WM * WN;
    constexpr int TQ = BQ / (kNWarps * kFragSize);
    constexpr int TK = BK / kFragSize;
    constexpr int TD = BD / kFragSize;
    static_assert(TQ == 1, "Check TQ");

    MMATile<AccumType, TQ, 1, MMAFrag_acc_t> Qtile;
    MMATile<AccumType, 1, TK, MMAFrag_acc_t> Ktile;
    MMATile<AccumType, TQ, TK, MMAFrag_acc_t> Stile;
    MMATile<AccumType, 1, 1, MMAFrag_acc_t> Vtile;
    MMATile<AccumType, TQ, TD, MMAFrag_acc_t> Otile;
    Otile.clear();

    const short2 simd_coord = MMAFrag_acc_t::get_coord(simd_lane_id);
    const short sm = simd_coord.y;
    const short sn = simd_coord.x;
    const short tm = kFragSize * TQ * simd_group_id;
    const short Qs_offset = (tm + sm) * LDQ_tgp + sn;
    const short Ks_offset = sm * LDK_tgp + sn;
    const short Vs_offset = sm * LDV_tgp + sn;
    constexpr short Qs_tile_stride = kFragSize;
    constexpr short Ks_tile_stride = kFragSize * LDK_tgp;

    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (q0 + BQ > L) {
        loader_q.load_safe(short2(BD, L - q0));
    } else {
        loader_q.load_unsafe();
    }

    constexpr short kRowsPT = decltype(Stile)::kRowsPerThread;
    AccumType max_score[kRowsPT];
    AccumType sum_score[kRowsPT];
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
        const int row = q0 + tm + sm + i * decltype(Stile)::kFragRows;
        if (row < L) {
            max_score[i] = M_LOG2E_F * static_cast<AccumType>(sinks[((ulong)tid.z * L + row) * HQ + h]);
            sum_score[i] = 1;
        } else {
            max_score[i] = Limits<AccumType>::finite_min;
            sum_score[i] = 0;
        }
    }

    for (int kb = kb_start; kb < kb_lim; kb++) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (kb * BK + BK > S) {
            loader_k.load_safe(short2(BD, S - kb * BK));
        } else {
            loader_k.load_unsafe();
        }
        Stile.clear();
        threadgroup_barrier(mem_flags::mem_threadgroup);
        STEEL_PRAGMA_UNROLL
        for (short dd = 0; dd < TD; dd++) {
            simdgroup_barrier(mem_flags::mem_none);
            Qtile.template load<T, 1, 1, LDQ_tgp, 1>(&Qs[Qs_offset + dd * Qs_tile_stride]);
            Ktile.template load<T, 1, 1, LDK_tgp, 1>(&Ks[Ks_offset + dd * Ks_tile_stride]);
            simdgroup_barrier(mem_flags::mem_none);
            tile_matmad(Stile, Qtile, Ktile, Stile);
        }

        // scale, mask (causal / window / first key / length) and relative bias
        {
            using stile_t = decltype(Stile);
            using selem_t = typename stile_t::elem_type;
            constexpr auto neg_inf = Limits<selem_t>::finite_min;
            STEEL_PRAGMA_UNROLL
            for (short i = 0; i < stile_t::kTileRows; i++) {
                const int row = q0 + tm + sm + (i * stile_t::kFragRows);
                const device T* bias = rel + (((ulong)tid.z * L + min(row, L - 1)) * HQ + h) * R;
                STEEL_PRAGMA_UNROLL
                for (short j = 0; j < stile_t::kTileCols; j++) {
                    const int col0 = kb * BK + sn + (j * stile_t::kFragCols);
                    STEEL_PRAGMA_UNROLL
                    for (short jj = 0; jj < stile_t::MMAFrag_t::kElemCols; jj++) {
                        const int col = col0 + jj;
                        const int d = qL_off + row - col;
                        const bool vis = row < L && col < S && d >= 0 && col >= start && (WINDOW == 0 || d < WINDOW);
                        selem_t s = Stile.frag_at(i, j)[jj] * scale;
                        if (!vis) {
                            s = neg_inf;
                        } else if (d < R) {
                            s += M_LOG2E_F * selem_t(bias[d]);
                        }
                        Stile.frag_at(i, j)[jj] = s;
                    }
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (kb * BK + BK > S) {
            loader_v.load_safe(short2(BD, S - kb * BK));
        } else {
            loader_v.load_unsafe();
        }

        AccumType new_max[kRowsPT];
        AccumType factor[kRowsPT];
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < kRowsPT; ++i) new_max[i] = max_score[i];
        Stile.template row_reduce<InkMaxOp>(new_max);
        Stile.template row_bin_op<InkExpSubOp>(new_max);
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < kRowsPT; ++i) factor[i] = fast::exp2(max_score[i] - new_max[i]);
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < kRowsPT; ++i) max_score[i] = new_max[i];
        AccumType sum_score_tmp[kRowsPT] = {0};
        Stile.template row_reduce<InkSumOp>(sum_score_tmp);
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < kRowsPT; ++i) sum_score[i] = sum_score[i] * factor[i] + sum_score_tmp[i];
        Otile.template row_bin_op<InkMulOp>(factor);

        threadgroup_barrier(mem_flags::mem_threadgroup);
        STEEL_PRAGMA_UNROLL
        for (short iq = 0; iq < TQ; iq++) {
            STEEL_PRAGMA_UNROLL
            for (short id = 0; id < TD; id++) {
                STEEL_PRAGMA_UNROLL
                for (short ik = 0; ik < TK; ik++) {
                    if constexpr (BD >= 128) simdgroup_barrier(mem_flags::mem_none);
                    const short kk = ik * kFragSize;
                    const short dd = id * kFragSize;
                    Vtile.template load<T, 1, 1, LDV_tgp, 1>(&Vs[Vs_offset + kk * LDV_tgp + dd]);
                    if constexpr (BD >= 128) simdgroup_barrier(mem_flags::mem_none);
                    MMAFrag_acc_t::mma(Otile.frag_at(iq, id), Stile.frag_at(iq, ik), Vtile.frag_at(0, 0), Otile.frag_at(iq, id));
                }
            }
        }
        loader_k.next();
        loader_v.next();
    }

    Otile.template row_bin_op<InkDivOp>(sum_score);
    threadgroup_barrier(mem_flags::mem_none);
    Op += (tm + sm) * O_ld + sn;
    const int rows_left = L - (q0 + tm + sm);
    if (rows_left <= 0) return;
    if (q0 + BQ > L) {
        Otile.template store_safe<T, 1, 1>(Op, (int)O_ld, short2(BD - sn, rows_left));
    } else {
        Otile.template store<T, 1, 1>(Op, (int)O_ld);
    }
"""


@lru_cache(maxsize=None)
def _kernel():
    return mx.fast.metal_kernel(
        name="mtplx_inkling2_steel_rel_attn",
        input_names=["q", "k", "v", "rel", "sinks", "key_start", "sizes", "sc"],
        output_names=["out"],
        header=_header(),
        source=_SRC,
        ensure_row_contiguous=False,
    )


def steel_relative_attention(q, k, v, rel, sinks, *, scale: float, window: int, key_start: mx.array) -> mx.array:
    """Same arguments and result as ``relative_attention`` (head dim 128)."""
    B, Hq, L, D = q.shape
    Hkv, S = k.shape[1], k.shape[2]
    if D != 128 or Hq % Hkv:
        raise ValueError("steel_relative_attention: head dim 128 and grouped heads only")
    (out,) = _kernel()(
        inputs=[q, k, v, mx.contiguous(rel), mx.contiguous(sinks.astype(mx.float32)),
                mx.contiguous(key_start.astype(mx.int32)), mx.array([L, S], dtype=mx.int32),
                mx.array([scale], dtype=mx.float32)],
        template=[("T", q.dtype), ("BQ", _BQ), ("BK", _BK), ("BD", D), ("WM", _WM), ("WN", _WN),
                  ("HQ", Hq), ("HKV", Hkv), ("R", int(rel.shape[-1])), ("WINDOW", int(window))],
        grid=(-(-L // _BQ) * 32 * _WM * _WN, Hq, B),
        threadgroup=(32 * _WM * _WN, 1, 1),
        output_shapes=[(B, L, Hq, D)],
        output_dtypes=[q.dtype],
    )
    return out
