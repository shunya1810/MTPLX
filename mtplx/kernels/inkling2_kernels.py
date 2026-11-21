# Copyright © 2026 Apple Inc. (MIT), from ml-explore/mlx-lm PR #2086
# (mlx_lm/models/inkling2/kernels.py and mlx_lm/kernels/{common,expert_routing}.py),
# which the pinned mlx-lm does not ship yet. Changes for MTPLX: the helpers
# and the routing kernel live here; relative_attention also takes float16.
"""Metal kernels of the Inkling 2 trunk: depth attention over the block
residuals, causal attention with a relative position bias and sinks, and the
sigmoid top-k routing."""

import os
from functools import lru_cache

import mlx.core as mx


@lru_cache
def scalar(value: float, dtype: mx.Dtype = mx.float32) -> mx.array:
    """A cached 0-d array, which a kernel gets as a constant."""
    return mx.array(value, dtype=dtype)


def row_threads(dim: int) -> int:
    """Threads for a threadgroup that handles one row of dim values."""
    return min(1024, max(32, dim // 8 // 32 * 32))


__all__ = [
    "depth_attention",
    "expert_routing",
    "relative_attention",
    "row_threads",
    "scalar",
]


# Attention of learned queries over the depth sources of each token, in one
# kernel.
_DEPTH_ATTENTION_SOURCE = """
    constexpr int N = K + 1;
    uint row = threadgroup_position_in_grid.x;
    uint rows = threadgroups_per_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;

    float acc[N * Q];
    for (int i = 0; i < N * Q; i++) {
        acc[i] = 0.0f;
    }
    for (int d = tid; d < D; d += TG) {
        float x[N];
        for (int i = 0; i < K; i++) {
            x[i] = blocks[(size_t(i) * rows + row) * D + d];
        }
        x[K] = partial[size_t(row) * D + d];
        for (int j = 0; j < Q; j++) {
            float qv = queries[j * D + d];
            for (int i = 0; i < N; i++) {
                acc[i * Q + j] += x[i] * qv;
            }
        }
    }

    threadgroup float buf[N * Q][TG / 32];
    for (int i = 0; i < N * Q; i++) {
        float v = simd_sum(acc[i]);
        if (lane == 0) {
            buf[i][sg] = v;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float w[N * Q];
    for (int i = 0; i < N * Q; i++) {
        float t = 0.0f;
        for (int s = 0; s < TG / 32; s++) {
            t += buf[i][s];
        }
        w[i] = t * scale;
    }
    for (int j = 0; j < Q; j++) {
        float m = -INFINITY;
        for (int i = 0; i < N; i++) {
            m = max(m, w[i * Q + j]);
        }
        float z = 0.0f;
        for (int i = 0; i < N; i++) {
            w[i * Q + j] = metal::exp(w[i * Q + j] - m);
            z += w[i * Q + j];
        }
        for (int i = 0; i < N; i++) {
            w[i * Q + j] /= z;
        }
    }

    for (int d = tid; d < D; d += TG) {
        float x[N];
        for (int i = 0; i < K; i++) {
            x[i] = blocks[(size_t(i) * rows + row) * D + d];
        }
        x[K] = partial[size_t(row) * D + d];
        for (int j = 0; j < Q; j++) {
            float o = 0.0f;
            for (int i = 0; i < N; i++) {
                o += w[i * Q + j] * x[i];
            }
            out[(size_t(row) * Q + j) * D + d] = o;
        }
    }
"""

_depth_attention_kernel = mx.fast.metal_kernel(
    name="depth_attention",
    input_names=["blocks", "partial", "queries", "scale"],
    output_names=["out"],
    source=_DEPTH_ATTENTION_SOURCE,
)


def depth_attention(
    blocks: mx.array, partial: mx.array, queries: mx.array, *, scale: float
) -> mx.array:
    """blocks: (K, ..., D) and partial: (..., D), float32, also the values.
    queries: float32 (Q, D). Returns (..., Q, D)."""
    D = partial.shape[-1]
    Q = queries.shape[0]
    TG = row_threads(D)
    return _depth_attention_kernel(
        inputs=[blocks, partial, queries, scalar(scale)],
        template=[("K", blocks.shape[0]), ("D", D), ("Q", Q), ("TG", TG)],
        grid=(partial.size // D * TG, 1, 1),
        threadgroup=(TG, 1, 1),
        output_shapes=[partial.shape[:-1] + (Q, D)],
        output_dtypes=[mx.float32],
    )[0]


# Causal flash attention with a relative position bias and a sink logit per
# query.
# Queries per threadgroup, 8 for each of its 4 simdgroups.
_BQ = 32

_TG = 128

# The online softmax of each row starts from its sink logit, so the zero-value
# sink needs no key.
_RELATIVE_ATTENTION_SOURCE = """
    constexpr float LOG2E = 1.4426950408889634f;
    constexpr int BK = 16;
    constexpr int LDK = BK + 8;
    constexpr int LDV = D + 8;
    const int L = sizes[0];
    const int S = sizes[1];
    const uint tid = thread_position_in_threadgroup.x;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const int batch = threadgroup_position_in_grid.z;
    const int head = threadgroup_position_in_grid.y;
    const int kv_head = head / (HQ / HKV);
    const int q0 = threadgroup_position_in_grid.x * BQ;
    const int start = key_start[batch];
    const float qscale = scale * LOG2E;

    // The lane holds row fm and columns fn, fn + 1 of each 8x8 fragment.
    const short fm = ((lane / 4) & 4) + (lane / 2) % 4;
    const short fn = ((lane / 4) & 2) * 2 + (lane % 2) * 2;
    const int row = q0 + int(sg) * 8 + fm;
    const bool row_ok = row < L;
    const int diag = S - L + row;

    // q and k are clipped and v has unit RMS, so half holds them.
    threadgroup half Ks[D * LDK];
    threadgroup half Vs[BK * LDV];

    simdgroup_half8x8 q_frag[D / 8];
    for (int db = 0; db < D / 8; ++db) {
        half2 qv = half2(0.0h);
        if (row_ok) {
            const device T* qp = q + batch * q_strides[0] + head * q_strides[1]
                + row * q_strides[2] + db * 8 + fn;
            qv = half2(float(qp[0]), float(qp[1]));
        }
        q_frag[db].thread_elements()[0] = qv.x;
        q_frag[db].thread_elements()[1] = qv.y;
    }
    simdgroup_float8x8 acc[D / 8];
    for (int db = 0; db < D / 8; ++db) {
        acc[db] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
    float row_max = row_ok ? float(sinks[(batch * L + row) * HQ + head]) * LOG2E : 0.0f;
    float row_sum = 1.0f;
    const int bias_row = batch * L + (row_ok ? row : 0);
    const device T* bias = rel + (size_t(bias_row) * HQ + head) * R;

    const int first_diag = S - L + q0;
    const int last_diag = S - L + min(q0 + BQ, L) - 1;
    int first_key = WINDOW > 0 ? max(0, first_diag - WINDOW + 1) : 0;
    first_key = max(first_key, start);
    const device T* k_head = k + batch * k_strides[0] + kv_head * k_strides[1];
    const device T* v_head = v + batch * v_strides[0] + kv_head * v_strides[1];

    for (int k0 = first_key / BK * BK; k0 <= last_diag; k0 += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int c = tid; c < BK * D / 8; c += TG) {
            const int r = c / (D / 8);
            const int col = (c % (D / 8)) * 8;
            const int key = k0 + r;
            vec<T, 4> kv[2] = {vec<T, 4>(0), vec<T, 4>(0)};
            vec<T, 4> vv[2] = {vec<T, 4>(0), vec<T, 4>(0)};
            if (key < S) {
                auto kp = (const device vec<T, 4>*)(k_head + key * k_strides[2] + col);
                auto vp = (const device vec<T, 4>*)(v_head + key * v_strides[2] + col);
                kv[0] = kp[0];
                kv[1] = kp[1];
                vv[0] = vp[0];
                vv[1] = vp[1];
            }
            for (int j = 0; j < 8; ++j) {
                Ks[(col + j) * LDK + r] = half(float(kv[j / 4][j % 4]));
                Vs[r * LDV + col + j] = half(float(vv[j / 4][j % 4]));
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (q0 + int(sg) * 8 >= L) {
            continue;
        }

        simdgroup_float8x8 scores[BK / 8];
        for (int kb = 0; kb < BK / 8; ++kb) {
            scores[kb] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            for (int db = 0; db < D / 8; ++db) {
                simdgroup_half8x8 k_frag;
                simdgroup_load(k_frag, Ks + db * 8 * LDK + kb * 8, LDK);
                simdgroup_multiply_accumulate(scores[kb], q_frag[db], k_frag, scores[kb]);
            }
        }

        float tile_max = -INFINITY;
        for (int kb = 0; kb < BK / 8; ++kb) {
            for (int e = 0; e < 2; ++e) {
                const int key = k0 + kb * 8 + fn + e;
                const int d = diag - key;
                const bool visible = row_ok && d >= 0 && key >= start
                    && (WINDOW == 0 || d < WINDOW);
                float s = -INFINITY;
                if (visible) {
                    s = scores[kb].thread_elements()[e] * qscale;
                    if (d < R) {
                        s += float(bias[d]) * LOG2E;
                    }
                }
                scores[kb].thread_elements()[e] = s;
                tile_max = max(tile_max, s);
            }
        }
        // The 4 lanes of a row are lane ^ 1 and lane ^ 8 apart.
        tile_max = max(tile_max, simd_shuffle_xor(tile_max, 1));
        tile_max = max(tile_max, simd_shuffle_xor(tile_max, 8));
        const float new_max = max(row_max, tile_max);
        const float factor = fast::exp2(row_max - new_max);
        float tile_sum = 0.0f;
        simdgroup_half8x8 probs[BK / 8];
        for (int kb = 0; kb < BK / 8; ++kb) {
            for (int e = 0; e < 2; ++e) {
                const float p = fast::exp2(scores[kb].thread_elements()[e] - new_max);
                probs[kb].thread_elements()[e] = half(p);
                tile_sum += p;
            }
        }
        tile_sum += simd_shuffle_xor(tile_sum, 1);
        tile_sum += simd_shuffle_xor(tile_sum, 8);
        row_sum = row_sum * factor + tile_sum;
        row_max = new_max;
        for (int db = 0; db < D / 8; ++db) {
            acc[db].thread_elements()[0] *= factor;
            acc[db].thread_elements()[1] *= factor;
            for (int kb = 0; kb < BK / 8; ++kb) {
                simdgroup_half8x8 v_frag;
                simdgroup_load(v_frag, Vs + kb * 8 * LDV + db * 8, LDV);
                simdgroup_multiply_accumulate(acc[db], probs[kb], v_frag, acc[db]);
            }
        }
    }

    if (row_ok) {
        device T* op = out + (size_t(batch * L + row) * HQ + head) * D;
        for (int db = 0; db < D / 8; ++db) {
            op[db * 8 + fn] = T(acc[db].thread_elements()[0] / row_sum);
            op[db * 8 + fn + 1] = T(acc[db].thread_elements()[1] / row_sum);
        }
    }
"""

_relative_attention_kernel = mx.fast.metal_kernel(
    name="relative_attention",
    input_names=["q", "k", "v", "rel", "sinks", "key_start", "sizes", "scale"],
    output_names=["out"],
    source=_RELATIVE_ATTENTION_SOURCE,
    ensure_row_contiguous=False,
)


def relative_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    rel: mx.array,
    sinks: mx.array,
    *,
    scale: float,
    window: int,
    key_start: mx.array,
) -> mx.array:
    """q: (B, Hq, L, D), k and v: (B, Hkv, S, D), all bfloat16 or all float16, with the queries
    at the last L keys. rel: (B, L, Hq, R), the bias at distances below R.
    sinks: (B, L, Hq). key_start: (B,), the first key of each row. Returns
    (B, L, Hq, D)."""
    B, Hq, L, D = q.shape
    Hkv, S = k.shape[1], k.shape[2]
    return _relative_attention_kernel(
        inputs=[
            q,
            k,
            v,
            mx.contiguous(rel),
            mx.contiguous(sinks.astype(mx.float32)),
            mx.contiguous(key_start.astype(mx.int32)),
            mx.array([L, S], dtype=mx.int32),
            scalar(scale),
        ],
        template=[
            ("T", q.dtype),
            ("D", D),
            ("HQ", Hq),
            ("HKV", Hkv),
            ("R", rel.shape[-1]),
            ("WINDOW", window),
            ("BQ", _BQ),
            ("TG", _TG),
        ],
        grid=((L + _BQ - 1) // _BQ * _TG, Hq, B),
        threadgroup=(_TG, 1, 1),
        output_shapes=[(B, L, Hq, D)],
        output_dtypes=[q.dtype],
    )[0]


# The routing after the router matmul, in one kernel.
_ROUTING_SOURCE = """
    constexpr int R = (E + 31) / 32;
    uint token = thread_position_in_grid.x / 32;
    uint lane = thread_index_in_simdgroup;
    auto row = logits + size_t(token) * E;

    float score[R];
    float biased[R];
    for (int i = 0; i < R; i++) {
        int e = lane + 32 * i;
        score[i] = 0.0f;
        biased[i] = -INFINITY;
        if (e < E) {
            float x = row[e] * multiplier;
            float y = 1.0f / (1.0f + metal::precise::exp(metal::abs(x)));
            score[i] = x < 0 ? y : 1.0f - y;
            biased[i] = score[i] + bias[e];
        }
    }

    float w[K];
    for (int k = 0; k < K; k++) {
        float best = -INFINITY;
        int best_e = E;
        for (int i = 0; i < R; i++) {
            int e = lane + 32 * i;
            if (biased[i] > best || (biased[i] == best && e < best_e)) {
                best = biased[i];
                best_e = e;
            }
        }
        float top = simd_max(best);
        int winner = simd_min(best == top ? best_e : E);
        float s = 0.0f;
        for (int i = 0; i < R; i++) {
            if (lane + 32 * i == winner) {
                s = score[i];
                biased[i] = -INFINITY;
            }
        }
        w[k] = simd_sum(s);
        if (lane == 0) {
            inds[size_t(token) * K + k] = winner;
        }
    }

    float total = 0.0f;
    for (int k = 0; k < K; k++) {
        total = w[k] + total;
    }
    if (lane == 0) {
        for (int k = 0; k < K; k++) {
            weights[size_t(token) * K + k] = w[k] / (total + eps) * scale;
        }
    }
"""

_routing_kernel = mx.fast.metal_kernel(
    name="expert_routing",
    input_names=["logits", "bias", "multiplier", "eps", "scale"],
    output_names=["inds", "weights"],
    source=_ROUTING_SOURCE,
)


def expert_routing(
    logits: mx.array,
    expert_bias: mx.array,
    *,
    top_k: int,
    multiplier: float,
    score_eps: float,
    route_scale: float,
) -> list[mx.array]:
    """The experts and their weights from the router logits, in one kernel."""
    E = logits.shape[-1]
    return _routing_kernel(
        inputs=[
            logits,
            expert_bias,
            scalar(multiplier),
            scalar(score_eps),
            scalar(route_scale),
        ],
        template=[("E", E), ("K", top_k)],
        grid=(logits.size // E * 32, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[logits.shape[:-1] + (top_k,)] * 2,
        output_dtypes=[mx.uint32, mx.float32],
    )


# Few-query relative attention (speculative verify, L <= 8 rows): the key
# range is split over threadgroups and merged in a second pass, and one
# threadgroup serves the G query heads of a KV head (one simdgroup a head,
# 8 query rows each, sharing the staged K/V tile). Same fragment layout and
# masking as the prefill kernel above; partials are in the log2 domain.
_SMALL_ATTENTION_SOURCE = """
    constexpr float LOG2E = 1.4426950408889634f;
    constexpr int G = HQ / HKV;
    constexpr int LDK = BK + 8;
    constexpr int LDV = D + 8;
    const int L = sizes[0];
    const int S = sizes[1];
    const int NS = sizes[2];
    const int chunk = sizes[3];
    const uint tid = thread_position_in_threadgroup.x;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const int batch = threadgroup_position_in_grid.z;
    const int kv_head = threadgroup_position_in_grid.y;
    const int split = threadgroup_position_in_grid.x;
    const int head = kv_head * G + int(sg);
    const int start = key_start[batch];
    const float qscale = scale * LOG2E;

    const short fm = ((lane / 4) & 4) + (lane / 2) % 4;
    const short fn = ((lane / 4) & 2) * 2 + (lane % 2) * 2;
    const int row = fm;
    const bool row_ok = row < L;
    const int diag = S - L + row;

    threadgroup half Ks[D * LDK];
    threadgroup half Vs[BK * LDV];

    simdgroup_half8x8 q_frag[D / 8];
    for (int db = 0; db < D / 8; ++db) {
        half2 qv = half2(0.0h);
        if (row_ok) {
            const device T* qp = q + batch * q_strides[0] + head * q_strides[1]
                + row * q_strides[2] + db * 8 + fn;
            qv = half2(float(qp[0]), float(qp[1]));
        }
        q_frag[db].thread_elements()[0] = qv.x;
        q_frag[db].thread_elements()[1] = qv.y;
    }
    simdgroup_float8x8 acc[D / 8];
    for (int db = 0; db < D / 8; ++db) {
        acc[db] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
    float row_max = -INFINITY;
    float row_sum = 0.0f;
    const device T* bias = rel + (size_t(batch * L + (row_ok ? row : 0)) * HQ + head) * R;

    int first_key = WINDOW > 0 ? max(0, (S - L) - WINDOW + 1) : 0;
    first_key = max(first_key, start);
    const int lo = first_key + split * chunk;
    const int hi = min(lo + chunk, S);
    const device T* k_head = k + batch * k_strides[0] + kv_head * k_strides[1];
    const device T* v_head = v + batch * v_strides[0] + kv_head * v_strides[1];

    for (int k0 = lo; k0 < hi; k0 += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int c = tid; c < BK * D / 8; c += TG) {
            const int r = c / (D / 8);
            const int col = (c % (D / 8)) * 8;
            const int key = k0 + r;
            vec<T, 4> kv[2] = {vec<T, 4>(0), vec<T, 4>(0)};
            vec<T, 4> vv[2] = {vec<T, 4>(0), vec<T, 4>(0)};
            if (key < hi) {
                auto kp = (const device vec<T, 4>*)(k_head + key * k_strides[2] + col);
                auto vp = (const device vec<T, 4>*)(v_head + key * v_strides[2] + col);
                kv[0] = kp[0];
                kv[1] = kp[1];
                vv[0] = vp[0];
                vv[1] = vp[1];
            }
            for (int j = 0; j < 8; ++j) {
                Ks[(col + j) * LDK + r] = half(float(kv[j / 4][j % 4]));
                Vs[r * LDV + col + j] = half(float(vv[j / 4][j % 4]));
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 scores[BK / 8];
        for (int kb = 0; kb < BK / 8; ++kb) {
            scores[kb] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            for (int db = 0; db < D / 8; ++db) {
                simdgroup_half8x8 k_frag;
                simdgroup_load(k_frag, Ks + db * 8 * LDK + kb * 8, LDK);
                simdgroup_multiply_accumulate(scores[kb], q_frag[db], k_frag, scores[kb]);
            }
        }
        float tile_max = -INFINITY;
        for (int kb = 0; kb < BK / 8; ++kb) {
            for (int e = 0; e < 2; ++e) {
                const int key = k0 + kb * 8 + fn + e;
                const int d = diag - key;
                const bool visible = row_ok && key < hi && d >= 0 && key >= start
                    && (WINDOW == 0 || d < WINDOW);
                float s = -INFINITY;
                if (visible) {
                    s = scores[kb].thread_elements()[e] * qscale;
                    if (d < R) {
                        s += float(bias[d]) * LOG2E;
                    }
                }
                scores[kb].thread_elements()[e] = s;
                tile_max = max(tile_max, s);
            }
        }
        tile_max = max(tile_max, simd_shuffle_xor(tile_max, 1));
        tile_max = max(tile_max, simd_shuffle_xor(tile_max, 8));
        const float new_max = max(row_max, tile_max);
        const float factor = (new_max == -INFINITY) ? 1.0f : fast::exp2(row_max - new_max);
        float tile_sum = 0.0f;
        simdgroup_half8x8 probs[BK / 8];
        for (int kb = 0; kb < BK / 8; ++kb) {
            for (int e = 0; e < 2; ++e) {
                const float sv = scores[kb].thread_elements()[e];
                const float p = (sv == -INFINITY) ? 0.0f : fast::exp2(sv - new_max);
                probs[kb].thread_elements()[e] = half(p);
                tile_sum += p;
            }
        }
        tile_sum += simd_shuffle_xor(tile_sum, 1);
        tile_sum += simd_shuffle_xor(tile_sum, 8);
        row_sum = row_sum * factor + tile_sum;
        row_max = new_max;
        for (int db = 0; db < D / 8; ++db) {
            acc[db].thread_elements()[0] *= factor;
            acc[db].thread_elements()[1] *= factor;
            for (int kb = 0; kb < BK / 8; ++kb) {
                simdgroup_half8x8 v_frag;
                simdgroup_load(v_frag, Vs + kb * 8 * LDV + db * 8, LDV);
                simdgroup_multiply_accumulate(acc[db], probs[kb], v_frag, acc[db]);
            }
        }
    }

    if (row_ok) {
        const size_t pi = (size_t(batch * HQ + head) * NS + split) * 8 + row;
        if (fn == 0 && (lane % 2) == 0 && ((lane / 4) & 2) == 0) {
            pm[pi] = row_max;
            pl[pi] = row_sum;
        }
        device float* op = po + pi * D;
        for (int db = 0; db < D / 8; ++db) {
            op[db * 8 + fn] = acc[db].thread_elements()[0];
            op[db * 8 + fn + 1] = acc[db].thread_elements()[1];
        }
    }
"""

_SMALL_MERGE_SOURCE = """
    constexpr float LOG2E = 1.4426950408889634f;
    const int L = sizes[0];
    const int NS = sizes[2];
    const uint d = thread_position_in_threadgroup.x;
    const int idx = threadgroup_position_in_grid.x;     // over B * HQ * L
    const int row = idx % L;
    const int head = (idx / L) % HQ;
    const int batch = idx / (L * HQ);
    const float sink = float(sinks[(batch * L + row) * HQ + head]) * LOG2E;
    const size_t base = size_t(batch * HQ + head) * NS;
    float M = sink;
    for (int s = 0; s < NS; ++s) M = max(M, pm[(base + s) * 8 + row]);
    float Z = fast::exp2(sink - M);
    float O = 0.0f;
    for (int s = 0; s < NS; ++s) {
        const size_t pi = (base + s) * 8 + row;
        const float m = pm[pi];
        if (m == -INFINITY) continue;
        const float w = fast::exp2(m - M);
        Z += w * pl[pi];
        O += w * po[pi * D + d];
    }
    out[(size_t(batch * L + row) * HQ + head) * D + d] = T(O / Z);
"""

_small_attention_kernel = mx.fast.metal_kernel(
    name="relative_attention_small",
    input_names=["q", "k", "v", "rel", "key_start", "sizes", "scale"],
    output_names=["pm", "pl", "po"],
    source=_SMALL_ATTENTION_SOURCE,
    ensure_row_contiguous=False,
)

_small_merge_kernel = mx.fast.metal_kernel(
    name="relative_attention_small_merge",
    input_names=["pm", "pl", "po", "sinks", "sizes"],
    output_names=["out"],
    source=_SMALL_MERGE_SOURCE,
)

_SMALL_BK = 32
_SMALL_KEYS_PER_SPLIT = int(os.environ.get("MTPLX_INKLING2_SMALL_ATTN_KEYS", "128"))
_SMALL_MAX_SPLITS = int(os.environ.get("MTPLX_INKLING2_SMALL_ATTN_SPLITS", "64"))


def relative_attention_small(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    rel: mx.array,
    sinks: mx.array,
    *,
    scale: float,
    window: int,
    key_start: mx.array,
) -> mx.array:
    """relative_attention for at most 8 query rows (same arguments and
    result layout, (B, L, Hq, D))."""
    B, Hq, L, D = q.shape
    Hkv, S = k.shape[1], k.shape[2]
    G = Hq // Hkv
    if L > 8 or Hq % Hkv or G > 8:
        raise ValueError("relative_attention_small: L <= 8 and G <= 8 only")
    first = max(0, (S - L) - window + 1) if window > 0 else 0
    span = max(1, S - first)
    splits = max(1, min(_SMALL_MAX_SPLITS, -(-span // _SMALL_KEYS_PER_SPLIT)))
    chunk = -(-span // splits)
    chunk = -(-chunk // _SMALL_BK) * _SMALL_BK
    splits = -(-span // chunk)
    sizes = mx.array([L, S, splits, chunk], dtype=mx.int32)
    tg = 32 * G
    pm, pl, po = _small_attention_kernel(
        inputs=[q, k, v, mx.contiguous(rel), mx.contiguous(key_start.astype(mx.int32)), sizes, scalar(scale)],
        template=[("T", q.dtype), ("D", D), ("HQ", Hq), ("HKV", Hkv), ("R", rel.shape[-1]),
                  ("WINDOW", window), ("BK", _SMALL_BK), ("TG", tg)],
        grid=(splits * tg, Hkv, B),
        threadgroup=(tg, 1, 1),
        output_shapes=[(B * Hq * splits * 8,), (B * Hq * splits * 8,), (B * Hq * splits * 8 * D,)],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )
    return _small_merge_kernel(
        inputs=[pm, pl, po, mx.contiguous(sinks.astype(mx.float32)), sizes],
        template=[("T", q.dtype), ("D", D), ("HQ", Hq)],
        grid=(B * Hq * L * D, 1, 1),
        threadgroup=(D, 1, 1),
        output_shapes=[(B, L, Hq, D)],
        output_dtypes=[q.dtype],
    )[0]
