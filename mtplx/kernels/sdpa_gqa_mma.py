"""Simdgroup-matrix (MMA) split-K verify/draft attention for Apple7 (M1) GPUs.

Why (2026-09-24, M1 Max long-context work): the packed-GQA kernels
(sdpa_gqa_packed / sdpa_gqa_packed_quant) stream one KV row per simdgroup and
reduce every score with a 5-step shuffle butterfly. On M1 Max that is
instruction-bound, ~1.6 TFLOP/s effective: 16.3 ms/layer (fp16) and
14.6 ms/layer (q8 bank) at 256K for the Qwen3.8 verify shape (Hq=24, Hk=4,
D=256, q_len=4), i.e. 60-70 % of a 256K verify. The attention for one KV head
is a skinny GEMM: all GQA_F*QL (=24) query rows against the same keys.

Structure (one threadgroup = one KV head x one key split, 4 simdgroups,
blocks of 32 keys):
  QK  S^T = K Q^T: each simdgroup owns 8 keys; A = K fragment read straight
      from device (dequantized in registers for q8/q4, the per-key scale
      folded into the score), B = Q^T fragment from threadgroup memory (Q is
      fp16/bf16 input, stored unscaled).
  softmax  S [rows x 32] in threadgroup memory, 4 threads per row, fp32
      online max/sum in the log2 domain.
  PV  each simdgroup owns D/4 output columns for every row tile; V fragments
      straight from device, P fragments from threadgroup memory.
Split-K partials (fp32) are merged by a small reduce kernel.

Measured per layer on M1 Max (qwen38-mlx-research/mma_attn_v3.py sweep,
2026-09-24), q_len 4 / q_len 1, vs packed-GQA / fused SDPA:
  256K fp16 5.1 vs 15.8 ms, q8 6.6 vs 14.5 ms; q_len 1 fp16 3.3 vs 16.7 ms.
  128K fp16 2.7 vs 7.7 ms, q8 3.6 vs 7.2 ms.  32K fp16 1.0 vs 1.9 ms.

One kernel serves every cache layout the long-context lanes use, through
(head, token) element strides:

- dense ``KVCache`` / ``TensorOffsetKVCache`` buffers ``[1, H, cap, D]``
- token-major paged pages ``[blocks, block_size, H, D]`` (fp16, q8, q4)
- head-major quantized banks ``[1, H, cap, packed]`` + fp32 scales

Semantics are the tail-causal contract of the packed kernels: query row j of a
q_len window attends to key n iff n <= offset - q_len + j. ``offset`` may be a
traced int32 array (compiled verify); the split count comes from the STATIC
ceiling (capacity or compiled bucket) so graph shapes stay stable, and each
split derives its key range from the dynamic offset.

Numerics: fragments are fp32 (K/V/Q converted from their storage type,
q8/q4 dequantized exactly as kv_quant.dequantize_symmetric), MMAs accumulate
in fp32, softmax and partials are fp32. Max |diff| vs an fp32 reference
<= 4e-5 from 700 to 256K keys (packed kernels ~2.5e-5 at 2K).
"""

from __future__ import annotations

from functools import lru_cache
import math
import os

import mlx.core as mx

# Contract bails keyed by the first gate that declined (same convention as
# sdpa_gqa_packed.gqa_packed_bail_counts); engaged calls never touch it.
gqa_mma_bail_counts: dict[str, int] = {}


def _bail(reason: str) -> None:
    gqa_mma_bail_counts[reason] = gqa_mma_bail_counts.get(reason, 0) + 1
    return None


_SOURCE = r"""
    constexpr int R = GQA_F * QL;
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

    const int h = threadgroup_position_in_grid.x;
    const int split = threadgroup_position_in_grid.z;
    const int sg = simdgroup_index_in_threadgroup;
    const int lane = thread_index_in_simdgroup;
    const int tid = sg * 32 + lane;
    const int n_kv = static_cast<int>(offset[0]);
    const int S_ = splits;

    int per = (n_kv + S_ - 1) / S_;
    per = ((per + BK - 1) / BK) * BK;
    const int k_begin = split * per;
    const int k_end = min(n_kv, k_begin + per);

    // Q^T into threadgroup memory: Qt[d * RT + r]
    for (int i = tid; i < RT * D2; i += NSG * 32) {
        const int r = i / D2;
        const int d = (i - r * D2) * 2;
        vec<InT, 2> v2 = vec<InT, 2>(0);
        if (r < R) {
            v2 = *((const device vec<InT, 2>*)(queries + (size_t)(h * R + r) * D + d));
        }
        Qt[d * RT + r] = static_cast<half>(v2.x);
        Qt[(d + 1) * RT + r] = static_cast<half>(v2.y);
    }

    const float qscale = scale * 1.44269504088896f;
    const short qid = lane / 4;
    const short fm = (qid & 4) + ((lane / 2) % 4);
    const short fn = (qid & 2) * 2 + (lane % 2) * 2;

    simdgroup_matrix<float, 8, 8> Oacc[NRT * NDT];
    _Pragma("clang loop unroll(full)")
    for (int i = 0; i < NRT * NDT; ++i) {
        Oacc[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
    const int srow = tid / 4;
    const int squad = tid % 4;
    float m_i = NEG;
    float l_i = 0.0f;
    const int srow_lim = n_kv - QL + (srow % QL);

    for (int n0 = k_begin; n0 < k_end; n0 += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_matrix<float, 8, 8> Sacc[KT * NRT];
        _Pragma("clang loop unroll(full)")
        for (int i = 0; i < KT * NRT; ++i) {
            Sacc[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
        size_t kbase[KT];
        float kscl[KT];
        _Pragma("clang loop unroll(full)")
        for (int c = 0; c < KT; ++c) {
            const int n = min(n0 + (sg * KT + c) * 8 + fm, n_kv - 1);
            kbase[c] = (size_t)h * k_hs + (size_t)n * k_ts;
            kscl[c] = (KQ == 0) ? 1.0f : k_scales[(size_t)h * s_hs + (size_t)n * s_ts];
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
                if (KQ == 0) {
                    const vec<InT, 2> k2 = *((const device vec<InT, 2>*)(keys + kbase[c] + d));
                    A.thread_elements()[0] = static_cast<float>(k2.x);
                    A.thread_elements()[1] = static_cast<float>(k2.y);
                } else if (KQ == 8) {
                    const char2 k2 = *((const device char2*)((const device char*)keys + kbase[c] + d));
                    A.thread_elements()[0] = static_cast<float>(k2.x);
                    A.thread_elements()[1] = static_cast<float>(k2.y);
                } else {
                    const uchar b = ((const device uchar*)keys)[kbase[c] + d / 2];
                    A.thread_elements()[0] = float(b & 15) - 8.0f;
                    A.thread_elements()[1] = float(b >> 4) - 8.0f;
                }
                _Pragma("clang loop unroll(full)")
                for (int rt = 0; rt < NRT; ++rt) {
                    simdgroup_multiply_accumulate(Sacc[c * NRT + rt], A, B[rt], Sacc[c * NRT + rt]);
                }
            }
        }
        // S^T tile (key fm, rows fn, fn+1) -> Ss[row][key]
        _Pragma("clang loop unroll(full)")
        for (int c = 0; c < KT; ++c) {
            const int key = (sg * KT + c) * 8 + fm;
            // q8/q4: the per-key row scale folds into the score (S^T row = key).
            const float ksc = qscale * kscl[c];
            _Pragma("clang loop unroll(full)")
            for (int rt = 0; rt < NRT; ++rt) {
                Ss[(rt * 8 + fn) * LDS + key] = Sacc[c * NRT + rt].thread_elements()[0] * ksc;
                Ss[(rt * 8 + fn + 1) * LDS + key] = Sacc[c * NRT + rt].thread_elements()[1] * ksc;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (srow < RT) {
            const bool rv = srow < R;
            const int lim = min(k_end - 1, srow_lim);
            float mx_ = NEG;
            for (int j = squad; j < BK; j += 4) {
                const bool ok = rv && (n0 + j) <= lim;
                mx_ = max(mx_, ok ? Ss[srow * LDS + j] : NEG);
            }
            mx_ = max(mx_, simd_shuffle_xor(mx_, 1));
            mx_ = max(mx_, simd_shuffle_xor(mx_, 2));
            const float new_m = max(m_i, mx_);
            const float factor = fast::exp2(m_i - new_m);
            float sm = 0.0f;
            for (int j = squad; j < BK; j += 4) {
                const bool ok = rv && (n0 + j) <= lim;
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
        for (int c = 0; c < BK / 8; ++c) {
            if (n0 + c * 8 >= k_end) {
                break;
            }
            simdgroup_matrix<float, 8, 8> P[NRT];
            _Pragma("clang loop unroll(full)")
            for (int rt = 0; rt < NRT; ++rt) {
                P[rt].thread_elements()[0] = Ss[(rt * 8 + fm) * LDS + c * 8 + fn];
                P[rt].thread_elements()[1] = Ss[(rt * 8 + fm) * LDS + c * 8 + fn + 1];
            }
            const int n = min(n0 + c * 8 + fm, n_kv - 1);
            const size_t vbase = (size_t)h * v_hs + (size_t)n * v_ts;
            const float vs = (KQ == 0) ? 1.0f : v_scales[(size_t)h * s_hs + (size_t)n * s_ts];
            _Pragma("clang loop unroll(full)")
            for (int t = 0; t < NDT; ++t) {
                const int d = sg * DCOLS + t * 8 + fn;
                simdgroup_matrix<float, 8, 8> W;
                if (KQ == 0) {
                    const vec<InT, 2> w2 = *((const device vec<InT, 2>*)(values + vbase + d));
                    W.thread_elements()[0] = static_cast<float>(w2.x);
                    W.thread_elements()[1] = static_cast<float>(w2.y);
                } else if (KQ == 8) {
                    const char2 w2 = *((const device char2*)((const device char*)values + vbase + d));
                    W.thread_elements()[0] = static_cast<float>(w2.x) * vs;
                    W.thread_elements()[1] = static_cast<float>(w2.y) * vs;
                } else {
                    const uchar b = ((const device uchar*)values)[vbase + d / 2];
                    W.thread_elements()[0] = (float(b & 15) - 8.0f) * vs;
                    W.thread_elements()[1] = (float(b >> 4) - 8.0f) * vs;
                }
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
    _Pragma("clang loop unroll(full)")
    for (int rt = 0; rt < NRT; ++rt) {
        const int r = rt * 8 + fm;
        if (r < R) {
            const size_t prow = (size_t)(h * R + r) * S_ + split;
            device float* p = partials + prow * D + sg * DCOLS;
            _Pragma("clang loop unroll(full)")
            for (int t = 0; t < NDT; ++t) {
                p[t * 8 + fn] = Oacc[rt * NDT + t].thread_elements()[0];
                p[t * 8 + fn + 1] = Oacc[rt * NDT + t].thread_elements()[1];
            }
        }
    }
    if (tid < R) {
        const size_t prow = (size_t)(h * R + tid) * S_ + split;
        maxs[prow] = Ss[tid * LDS + 0];
        sums[prow] = Ss[tid * LDS + 1];
    }
"""

_REDUCE = r"""
    const int row = threadgroup_position_in_grid.x;
    const int d = thread_position_in_threadgroup.x;
    const int S_ = splits;
    const device float* mrow = maxs + (size_t)row * S_;
    const device float* lrow = sums + (size_t)row * S_;
    float M = -1.0e30f;
    for (int s = 0; s < S_; ++s) {
        M = max(M, mrow[s]);
    }
    float L = 0.0f;
    float acc = 0.0f;
    const device float* prow = partials + (size_t)row * S_ * D + d;
    for (int s = 0; s < S_; ++s) {
        const float f = fast::exp2(mrow[s] - M);
        L += f * lrow[s];
        acc += f * prow[(size_t)s * D];
    }
    out[(size_t)row * D + d] = static_cast<OutT>(L > 0.0f ? acc / L : 0.0f);
"""


@lru_cache(maxsize=None)
def _kernels():
    if not mx.metal.is_available():
        return None, None
    partials = mx.fast.metal_kernel(
        name="mtplx_sdpa_gqa_mma_partials",
        input_names=[
            "queries", "keys", "values", "k_scales", "v_scales", "offset",
            "k_hs", "k_ts", "v_hs", "v_ts", "s_hs", "s_ts", "scale", "splits",
        ],
        output_names=["partials", "sums", "maxs"],
        source=_SOURCE,
    )
    reduce = mx.fast.metal_kernel(
        name="mtplx_sdpa_gqa_mma_reduce",
        input_names=["partials", "sums", "maxs", "splits"],
        output_names=["out"],
        source=_REDUCE,
    )
    return partials, reduce


def _env_int(name: str, default: int) -> int:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return int(default)
    try:
        return int(raw)
    except ValueError:
        return int(default)


def splits_for_ceiling(ceiling: int) -> int:
    """Static split-K count from the static key ceiling (never the dynamic offset)."""
    override = _env_int("MTPLX_GQA_MMA_SPLITS", 0)
    if override > 0:
        return override
    # M1 Max sweep: 64 splits best up to 32K, 128-256 at 128K-256K.
    target = max(64, _env_int("MTPLX_GQA_MMA_KEYS_PER_SPLIT", 1024))
    return int(max(64, min(256, math.ceil(max(1, int(ceiling)) / target))))


_DUMMY_SCALES: mx.array | None = None


def _dummy_scales() -> mx.array:
    global _DUMMY_SCALES
    if _DUMMY_SCALES is None:
        _DUMMY_SCALES = mx.zeros((1,), dtype=mx.float32)
    return _DUMMY_SCALES


def sdpa_gqa_mma(
    *,
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    offset: int | mx.array,
    scale: float,
    num_kv_heads: int,
    k_strides: tuple[int, int],
    v_strides: tuple[int, int],
    ceiling: int,
    kv_bits: int = 0,
    k_scales: mx.array | None = None,
    v_scales: mx.array | None = None,
    s_strides: tuple[int, int] = (0, 0),
    max_q_len: int = 4,
) -> mx.array | None:
    """Tail-causal GQA attention of ``queries`` [1, Hq, QL, D] over ``offset`` keys.

    ``keys``/``values`` are the WHOLE allocated buffers (never offset views);
    element (head h, token n, dim d) lives at ``h * strides[0] + n *
    strides[1] + d`` in payload units (fp16 elements, int8 bytes, or q4
    bytes = d / 2). ``ceiling`` is the static maximum offset (capacity or the
    compiled bucket). Returns None when the contract is not met.
    """
    if not mx.metal.is_available():
        return _bail("metal_unavailable")
    if queries.ndim != 4:
        return _bail("ndim")
    bsz, hq, q_len, d = (int(x) for x in queries.shape)
    if bsz != 1:
        return _bail("batch_size")
    if q_len < 1 or q_len > int(max_q_len):
        return _bail("q_len")
    chunk_rows = 4
    if q_len > chunk_rows:
        # Wide windows (context-copy blocks, q_len up to 16) run as <=4-row
        # query chunks. Chunk [a, b) with offset' = offset - (q_len - b) and
        # q_len' = b - a keeps the exact tail-causal visibility of every row:
        # n <= offset' - q_len' + (j - a) = offset - q_len + j.
        pieces = []
        for a in range(0, q_len, chunk_rows):
            b = min(q_len, a + chunk_rows)
            sub_offset = offset - (q_len - b)
            piece = sdpa_gqa_mma(
                queries=queries[:, :, a:b, :],
                keys=keys,
                values=values,
                offset=sub_offset,
                scale=scale,
                num_kv_heads=num_kv_heads,
                k_strides=k_strides,
                v_strides=v_strides,
                ceiling=ceiling,
                kv_bits=kv_bits,
                k_scales=k_scales,
                v_scales=v_scales,
                s_strides=s_strides,
                max_q_len=chunk_rows,
            )
            if piece is None:
                return None
            pieces.append(piece)
        return mx.concatenate(pieces, axis=2)
    if d % 32 or d > 256:
        return _bail("head_dim")
    hk = int(num_kv_heads)
    if hk <= 0 or hq % hk:
        return _bail("gqa_heads")
    gqa = hq // hk
    rows = gqa * q_len
    # Q^T (D x RT halves) + scores (RT x 36 floats) in threadgroup memory,
    # and the softmax runs 4 threads per row over 128 threads: RT <= 32.
    if rows > 32:
        return _bail("rows")
    if queries.dtype not in (mx.float16, mx.bfloat16):
        return _bail("query_dtype")
    kv_bits = int(kv_bits)
    if kv_bits == 0:
        # fp16/bf16 K/V are read through the query element type.
        if keys.dtype != queries.dtype or values.dtype != queries.dtype:
            return _bail("kv_dtype")
    elif kv_bits == 8:
        if keys.dtype != mx.int8 or values.dtype != mx.int8:
            return _bail("kv_dtype")
    elif kv_bits == 4:
        if keys.dtype != mx.uint8 or values.dtype != mx.uint8:
            return _bail("kv_dtype")
    else:
        return _bail("kv_bits")
    if kv_bits and (k_scales is None or v_scales is None):
        return _bail("scales_missing")
    if kv_bits and (k_scales.dtype != mx.float32 or v_scales.dtype != mx.float32):
        return _bail("scale_dtype")
    partials_kernel, reduce_kernel = _kernels()
    if partials_kernel is None:
        return _bail("kernel_unavailable")

    if isinstance(offset, mx.array):
        if offset.size != 1:
            return _bail("offset_shape")
        offset_arr = offset.astype(mx.int32).reshape(1)
    else:
        offset_int = int(offset)
        if offset_int <= 0:
            return _bail("offset_range")
        offset_arr = mx.array([offset_int], dtype=mx.int32)

    splits = splits_for_ceiling(ceiling)
    dummy = _dummy_scales()
    partials, sums, maxs = partials_kernel(
        inputs=[
            queries,
            keys,
            values,
            k_scales if kv_bits else dummy,
            v_scales if kv_bits else dummy,
            offset_arr,
            int(k_strides[0]),
            int(k_strides[1]),
            int(v_strides[0]),
            int(v_strides[1]),
            int(s_strides[0]),
            int(s_strides[1]),
            float(scale),
            int(splits),
        ],
        template=[
            ("InT", queries.dtype),
            ("D", d),
            ("GQA_F", gqa),
            ("QL", q_len),
            ("KQ", kv_bits),
            ("KT", 1),
        ],
        grid=(hk * 32, 4, splits),
        threadgroup=(32, 4, 1),
        output_shapes=[(hq * q_len * splits, d), (hq * q_len * splits,), (hq * q_len * splits,)],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )
    (out,) = reduce_kernel(
        inputs=[partials, sums, maxs, int(splits)],
        template=[("D", d), ("OutT", queries.dtype)],
        grid=(hq * q_len * d, 1, 1),
        threadgroup=(d, 1, 1),
        output_shapes=[queries.shape],
        output_dtypes=[queries.dtype],
    )
    return out
