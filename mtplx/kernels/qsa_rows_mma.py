"""QSA verify attention over per-row token lists, read in place (M1 simdgroup MMA).

Why (2026-11-16, Flash-Next on M1 Max): at verify widths the QSA layers either
ran fused SDPA over the FULL context through a dense [S, T] mask (below
MTPLX_QSA_GATHER_MIN_CONTEXT, 16K) or, past it, gathered every row's selected
keys/values into fresh [S, K, D] copies and ran broadcast matmuls + an fp32
softmax (_qsa_rows_gather_attention). Both cost about 1 ms a layer for 4 rows
(12 layers: ~12 ms of a ~47 ms verify) although a row only sees its 2,304
selected tokens.

This kernel reads each row's selected keys/values straight from the cache's
backing arrays through the row's token list. One threadgroup = one KV head x
one query row x one split of the row's list; its GQA query heads (12) are the
rows of the MMA tile, exactly as sdpa_gqa_mma lays out a KV head. Invalid slots
(token_ok False) score -inf. Split partials (fp32) are merged by a reduce
kernel. Same visible set and math as the gathered product; summation order
differs (fp32 MMA accumulation, online softmax), so it is not bit-identical.

Layouts: q [1, H, S, D]; keys/values [1, H_kv, cap, D] (contiguous backing,
any cap >= every listed index); token_idx int32 [S, K]; token_ok bool [S, K].
Output [1, H, S, D] in q's dtype.
"""

from __future__ import annotations

from functools import lru_cache
import os

import mlx.core as mx

_SOURCE = r"""
    constexpr int R = GQA;
    constexpr int RT = ((R + 7) / 8) * 8;
    constexpr int NRT = RT / 8;
    constexpr int NSG = 4;
    constexpr int BK = 8 * KT * NSG;
    constexpr int LDS = BK + 4;
    constexpr int DCOLS = D / NSG;
    constexpr int NDT = DCOLS / 8;
    constexpr int D2 = D / 2;
    constexpr float NEG = -1.0e30f;

    threadgroup half Qt[D * RT];
    threadgroup float Ss[RT * LDS];
    threadgroup float Fs[RT];
    threadgroup int Sidx[BK];
    threadgroup uchar Sok[BK];

    const int h = threadgroup_position_in_grid.x;
    const int s = threadgroup_position_in_grid.y;
    const int split = threadgroup_position_in_grid.z;
    const int sg = simdgroup_index_in_threadgroup;
    const int lane = thread_index_in_simdgroup;
    const int tid = sg * 32 + lane;
    const int NS = token_idx_shape[0];
    const int KSEL = token_idx_shape[1];
    const int CAP = keys_shape[2];
    const int S_ = SPLITS;

    int per = (KSEL + S_ - 1) / S_;
    per = ((per + BK - 1) / BK) * BK;
    const int k_begin = split * per;
    const int k_end = min(KSEL, k_begin + per);

    for (int i = tid; i < RT * D2; i += NSG * 32) {
        const int r = i / D2;
        const int d = (i - r * D2) * 2;
        vec<InT, 2> v2 = vec<InT, 2>(0);
        if (r < R) {
            v2 = *((const device vec<InT, 2>*)(queries + (size_t)((h * GQA + r) * NS + s) * D + d));
        }
        Qt[d * RT + r] = static_cast<half>(v2.x);
        Qt[(d + 1) * RT + r] = static_cast<half>(v2.y);
    }

    const float qscale = scale[0] * 1.44269504088896f;
    const short qid = lane / 4;
    const short fm = (qid & 4) + ((lane / 2) % 4);
    const short fn = (qid & 2) * 2 + (lane % 2) * 2;
    const size_t k_hs = (size_t)CAP * D;
    const device int* tix = token_idx + (size_t)s * KSEL;
    const device bool* tok = token_ok + (size_t)s * KSEL;

    simdgroup_matrix<float, 8, 8> Oacc[NRT * NDT];
    _Pragma("clang loop unroll(full)")
    for (int i = 0; i < NRT * NDT; ++i) {
        Oacc[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
    const int srow = tid / 4;
    const int squad = tid % 4;
    float m_i = NEG;
    float l_i = 0.0f;

    for (int n0 = k_begin; n0 < k_end; n0 += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < BK) {
            const int j = n0 + tid;
            const bool okv = j < k_end && tok[j];
            Sidx[tid] = okv ? tix[j] : 0;
            Sok[tid] = okv ? 1 : 0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_matrix<float, 8, 8> Sacc[KT * NRT * QKS];
        _Pragma("clang loop unroll(full)")
        for (int i = 0; i < KT * NRT * QKS; ++i) {
            Sacc[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
        size_t kbase[KT];
        _Pragma("clang loop unroll(full)")
        for (int c = 0; c < KT; ++c) {
            kbase[c] = (size_t)h * k_hs + (size_t)Sidx[(sg * KT + c) * 8 + fm] * D;
        }
        _Pragma("clang loop unroll(full)")
        for (int kb = 0; kb < D / 8; ++kb) {
            simdgroup_matrix<float, 8, 8> B[NRT];
            _Pragma("clang loop unroll(full)")
            for (int rt = 0; rt < NRT; ++rt) {
                const half2 q2 = *((const threadgroup half2*)(&Qt[(kb * 8 + fm) * RT + rt * 8 + fn]));
                B[rt].thread_elements()[0] = static_cast<float>(q2.x);
                B[rt].thread_elements()[1] = static_cast<float>(q2.y);
            }
            _Pragma("clang loop unroll(full)")
            for (int c = 0; c < KT; ++c) {
                simdgroup_matrix<float, 8, 8> A;
                const int d = kb * 8 + fn;
                const vec<InT, 2> k2 = *((const device vec<InT, 2>*)(keys + kbase[c] + d));
                A.thread_elements()[0] = static_cast<float>(k2.x);
                A.thread_elements()[1] = static_cast<float>(k2.y);
                _Pragma("clang loop unroll(full)")
                for (int rt = 0; rt < NRT; ++rt) {
                    const int si = ((kb % QKS) * KT + c) * NRT + rt;
                    simdgroup_multiply_accumulate(Sacc[si], A, B[rt], Sacc[si]);
                }
            }
        }
        _Pragma("clang loop unroll(full)")
        for (int part = 1; part < QKS; ++part) {
            _Pragma("clang loop unroll(full)")
            for (int i = 0; i < KT * NRT; ++i) {
                Sacc[i].thread_elements()[0] += Sacc[part * KT * NRT + i].thread_elements()[0];
                Sacc[i].thread_elements()[1] += Sacc[part * KT * NRT + i].thread_elements()[1];
            }
        }
        _Pragma("clang loop unroll(full)")
        for (int c = 0; c < KT; ++c) {
            const int key = (sg * KT + c) * 8 + fm;
            _Pragma("clang loop unroll(full)")
            for (int rt = 0; rt < NRT; ++rt) {
                Ss[(rt * 8 + fn) * LDS + key] = Sacc[c * NRT + rt].thread_elements()[0] * qscale;
                Ss[(rt * 8 + fn + 1) * LDS + key] = Sacc[c * NRT + rt].thread_elements()[1] * qscale;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (srow < RT) {
            const bool rv = srow < R;
            float mx_ = NEG;
            for (int j = squad; j < BK; j += 4) {
                const bool ok = rv && Sok[j];
                mx_ = max(mx_, ok ? Ss[srow * LDS + j] : NEG);
            }
            mx_ = max(mx_, simd_shuffle_xor(mx_, 1));
            mx_ = max(mx_, simd_shuffle_xor(mx_, 2));
            const float new_m = max(m_i, mx_);
            const float factor = fast::exp2(m_i - new_m);
            float sm = 0.0f;
            for (int j = squad; j < BK; j += 4) {
                const bool ok = rv && Sok[j];
                const float p = ok ? fast::exp2(Ss[srow * LDS + j] - new_m) : 0.0f;
                Ss[srow * LDS + j] = p;
                sm += p;
            }
            sm += simd_shuffle_xor(sm, 1);
            sm += simd_shuffle_xor(sm, 2);
            l_i = l_i * factor + sm;
            m_i = new_m;
            if (squad == 0) {
                Fs[srow] = factor;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        _Pragma("clang loop unroll(full)")
        for (int rt = 0; rt < NRT; ++rt) {
            const float f = Fs[rt * 8 + fm];
            _Pragma("clang loop unroll(full)")
            for (int t = 0; t < NDT; ++t) {
                Oacc[rt * NDT + t].thread_elements()[0] *= f;
                Oacc[rt * NDT + t].thread_elements()[1] *= f;
            }
        }
        _Pragma("clang loop unroll(full)")
        for (int c = 0; c < BK / 8; ++c) {
            simdgroup_matrix<float, 8, 8> P[NRT];
            _Pragma("clang loop unroll(full)")
            for (int rt = 0; rt < NRT; ++rt) {
                P[rt].thread_elements()[0] = Ss[(rt * 8 + fm) * LDS + c * 8 + fn];
                P[rt].thread_elements()[1] = Ss[(rt * 8 + fm) * LDS + c * 8 + fn + 1];
            }
            const size_t vbase = (size_t)h * k_hs + (size_t)Sidx[c * 8 + fm] * D;
            _Pragma("clang loop unroll(full)")
            for (int t = 0; t < NDT; ++t) {
                const int d = sg * DCOLS + t * 8 + fn;
                simdgroup_matrix<float, 8, 8> W;
                const vec<InT, 2> w2 = *((const device vec<InT, 2>*)(values + vbase + d));
                W.thread_elements()[0] = static_cast<float>(w2.x);
                W.thread_elements()[1] = static_cast<float>(w2.y);
                _Pragma("clang loop unroll(full)")
                for (int rt = 0; rt < NRT; ++rt) {
                    simdgroup_multiply_accumulate(Oacc[rt * NDT + t], P[rt], W, Oacc[rt * NDT + t]);
                }
            }
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (srow < RT && squad == 0) {
        Ss[srow * LDS + 0] = m_i;
        Ss[srow * LDS + 1] = l_i;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const size_t rowbase = (size_t)(h * NS + s) * GQA;
    _Pragma("clang loop unroll(full)")
    for (int rt = 0; rt < NRT; ++rt) {
        const int r = rt * 8 + fm;
        if (r < R) {
            const size_t prow = (rowbase + r) * S_ + split;
            device float* p = partials + prow * D + sg * DCOLS;
            _Pragma("clang loop unroll(full)")
            for (int t = 0; t < NDT; ++t) {
                p[t * 8 + fn] = Oacc[rt * NDT + t].thread_elements()[0];
                p[t * 8 + fn + 1] = Oacc[rt * NDT + t].thread_elements()[1];
            }
        }
    }
    if (tid < R) {
        const size_t prow = (rowbase + tid) * S_ + split;
        maxs[prow] = Ss[tid * LDS + 0];
        sums[prow] = Ss[tid * LDS + 1];
    }
"""

# One threadgroup per (kv head, row, query head); D threads. Partial rows are
# ordered (h, s, r); the output is [1, H, S, D] with H = h * GQA + r.
_REDUCE = r"""
    const int row = threadgroup_position_in_grid.x;
    const int d = thread_position_in_threadgroup.x;
    const int S_ = SPLITS;
    const int r = row % GQA;
    const int hs = row / GQA;
    const int s = hs % NROWS;
    const int h = hs / NROWS;
    const device float* mrow = maxs + (size_t)row * S_;
    const device float* lrow = sums + (size_t)row * S_;
    float M = -1.0e30f;
    for (int i = 0; i < S_; ++i) {
        M = max(M, mrow[i]);
    }
    float L = 0.0f;
    float acc = 0.0f;
    const device float* prow = partials + (size_t)row * S_ * D + d;
    for (int i = 0; i < S_; ++i) {
        const float f = fast::exp2(mrow[i] - M);
        L += f * lrow[i];
        acc += f * prow[(size_t)i * D];
    }
    out[((size_t)(h * GQA + r) * NROWS + s) * D + d] = static_cast<OutT>(L > 0.0f ? acc / L : 0.0f);
"""


@lru_cache(maxsize=None)
def _kernels():
    attn = mx.fast.metal_kernel(
        name="mtplx_qsa_rows_mma",
        input_names=["queries", "keys", "values", "token_idx", "token_ok", "scale"],
        output_names=["partials", "maxs", "sums"],
        source=_SOURCE,
    )
    red = mx.fast.metal_kernel(
        name="mtplx_qsa_rows_mma_reduce",
        input_names=["partials", "maxs", "sums"],
        output_names=["out"],
        source=_REDUCE,
    )
    return attn, red


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name) or default)
    except ValueError:
        return default


@lru_cache(maxsize=None)
def _scale_array(scale: float) -> mx.array:
    return mx.array([scale], dtype=mx.float32)


def supported(q: mx.array, keys: mx.array, token_idx: mx.array) -> bool:
    if q.ndim != 4 or keys.ndim != 4 or int(q.shape[0]) != 1 or int(keys.shape[0]) != 1:
        return False
    H, D = int(q.shape[1]), int(q.shape[3])
    hk = int(keys.shape[1])
    if hk <= 0 or H % hk or D % 32 or int(keys.shape[3]) != D:
        return False
    if q.dtype not in (mx.float16, mx.bfloat16) or keys.dtype != q.dtype:
        return False
    return H // hk <= 16 and token_idx.ndim == 2 and int(token_idx.shape[0]) == int(q.shape[2])


def qsa_rows_attention(q, keys, values, token_idx, token_ok, scale: float):
    """Attention of each row of q over its own token list, in place.

    q [1, H, S, D]; keys/values [1, H_kv, cap, D] contiguous backing; token_idx
    int32 [S, K] (every index < cap); token_ok bool [S, K]. Returns [1, H, S, D]."""
    _, H, S, D = (int(v) for v in q.shape)
    hk, cap = int(keys.shape[1]), int(keys.shape[2])
    gqa = H // hk
    ksel = int(token_idx.shape[1])
    kt = _env_int("MTPLX_QSA_ROWS_MMA_KT", 1)
    qks = _env_int("MTPLX_QSA_ROWS_MMA_QKS", 2)
    bk = 32 * kt
    splits = _env_int("MTPLX_QSA_ROWS_MMA_SPLITS", 0)
    if splits <= 0:
        # ~64-128 threadgroups across the GPU; never a split shorter than a tile
        splits = max(1, min(16, (128 + hk * S - 1) // (hk * S), (ksel + bk - 1) // bk))
    attn, red = _kernels()
    rows = hk * S * gqa
    tmpl = [("InT", q.dtype), ("D", D), ("GQA", gqa), ("KT", kt), ("QKS", qks), ("SPLITS", splits)]
    if token_idx.dtype != mx.int32:
        token_idx = token_idx.astype(mx.int32)
    if token_ok.dtype != mx.bool_:
        token_ok = token_ok.astype(mx.bool_)
    partials, maxs, sums = attn(
        inputs=[q, keys, values, token_idx, token_ok, _scale_array(float(scale))],
        template=tmpl,
        grid=(hk * 128, S, splits),
        threadgroup=(128, 1, 1),
        output_shapes=[(rows * splits, D), (rows * splits,), (rows * splits,)],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )
    (out,) = red(
        inputs=[partials, maxs, sums],
        template=[("OutT", q.dtype), ("D", D), ("GQA", gqa), ("NROWS", S), ("SPLITS", splits)],
        grid=(rows * D, 1, 1),
        threadgroup=(D, 1, 1),
        output_shapes=[(1, H, S, D)],
        output_dtypes=[q.dtype],
    )
    return out
