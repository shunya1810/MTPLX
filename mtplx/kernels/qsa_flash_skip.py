"""Block-sparse flash attention for QSA decode rows (MTPLX_QSA_FLASH).

The indexer's selected blocks + visible tail define the visible set; this
kernel computes softmax attention over EXACTLY that set by iterating the
selected blocks inside the kernel — skipped blocks are never staged, and
nothing is gathered into temporaries (the falsified MTPLX_QSA_GATHER lane
paid two materialized copies per layer per token; the dense bool-mask lane
stages the full context). Standard block-sparse flash-attention structure
(online softmax with a two-level simdgroup merge), fitted to the family
geometry: head_dim 256, GQA, indexer block size 4.

One dispatch per (token, layer): grid = Hq threadgroups x 256 threads
(8 simdgroups). Each simdgroup strides the selected-block list and then
the tail keys; per key the 32 lanes hold 8 dims each of q/k/v, scores are
simd_sum reductions, and the output accumulates per-lane in fp32. The KV
backing arrays are read IN PLACE at their allocation stride (cap), so the
non-contiguous :T slice never forces a copy.
"""

from functools import lru_cache

import mlx.core as mx

_HEADER = """
#include <metal_stdlib>
using namespace metal;
"""

_SRC = """
    constexpr int HD = 256;
    constexpr int BLK = 4;                    // indexer compress ratio

    const uint hq = threadgroup_position_in_grid.x;
    const uint tid = thread_position_in_threadgroup.x;
    const uint sg = tid / 32;
    const uint lane = tid % 32;

    const int t_total = params[0];            // valid kv length T
    const int tail_start = params[1];         // first tail token
    const int nsel = params[2];               // selected block count
    const int cap = params[3];                // kv allocation stride (dim2)

    threadgroup float q_s[HD];
    if (tid < (uint)HD) q_s[tid] = (float)q[hq * HD + tid] * scale[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint hkv = hq / GQA;
    device const T* kh = k + (size_t)hkv * (size_t)cap * HD;
    device const T* vh = v + (size_t)hkv * (size_t)cap * HD;

    float m = -3.0e38f;
    float l = 0.0f;
    float o[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    const uint dbase = lane * 8;

    // selected complete blocks, striped across the 8 simdgroups
    for (int bi = (int)sg; bi < nsel; bi += 8) {
        const int t0 = blocks[bi] * BLK;
        for (int j = 0; j < BLK; ++j) {
            const size_t t = (size_t)(t0 + j);
            float part = 0.0f;
            device const T* kr = kh + t * HD + dbase;
            for (int d = 0; d < 8; ++d) part += q_s[dbase + d] * (float)kr[d];
            const float s = simd_sum(part);
            const float mn = metal::max(m, s);
            const float c = metal::exp(m - mn);
            const float p = metal::exp(s - mn);
            l = l * c + p;
            device const T* vr = vh + t * HD + dbase;
            for (int d = 0; d < 8; ++d) o[d] = o[d] * c + p * (float)vr[d];
            m = mn;
        }
    }
    // visible tail keys, same striping
    for (int t = tail_start + (int)sg; t < t_total; t += 8) {
        float part = 0.0f;
        device const T* kr = kh + (size_t)t * HD + dbase;
        for (int d = 0; d < 8; ++d) part += q_s[dbase + d] * (float)kr[d];
        const float s = simd_sum(part);
        const float mn = metal::max(m, s);
        const float c = metal::exp(m - mn);
        const float p = metal::exp(s - mn);
        l = l * c + p;
        device const T* vr = vh + (size_t)t * HD + dbase;
        for (int d = 0; d < 8; ++d) o[d] = o[d] * c + p * (float)vr[d];
        m = mn;
    }

    // two-level merge across the 8 simdgroups
    threadgroup float m_tg[8];
    threadgroup float l_tg[8];
    threadgroup float o_tg[8 * HD];
    if (lane == 0) { m_tg[sg] = m; l_tg[sg] = l; }
    for (int d = 0; d < 8; ++d) o_tg[sg * HD + dbase + d] = o[d];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // each of the 256 threads owns one output dim
    float M = m_tg[0];
    for (int s2 = 1; s2 < 8; ++s2) M = metal::max(M, m_tg[s2]);
    float L = 0.0f;
    float acc = 0.0f;
    for (int s2 = 0; s2 < 8; ++s2) {
        const float w = metal::exp(m_tg[s2] - M);
        L += l_tg[s2] * w;
        acc += o_tg[s2 * HD + tid] * w;
    }
    out[hq * HD + tid] = (T)(acc / L);
"""


@lru_cache(maxsize=4)
def _kernel():
    return mx.fast.metal_kernel(
        name="mtplx_qsa_flash_skip",
        input_names=["q", "k", "v", "blocks", "params", "scale"],
        output_names=["out"],
        header=_HEADER,
        source=_SRC,
    )


def qsa_flash_skip(q_row, k_backing, v_backing, blocks, t_total, tail_start, scale):
    """q_row [Hq, 256] (post-norm, post-rope); k/v_backing the FULL cache
    arrays [1, Hkv, cap, 256] (read in place — no slicing copies); blocks
    [nsel] int32 sorted selected block ids; t_total the valid kv length;
    tail_start the first tail token (host int). Returns out [Hq, 256]."""
    hq = int(q_row.shape[0])
    hkv = int(k_backing.shape[1])
    cap = int(k_backing.shape[2])
    params = mx.array([int(t_total), int(tail_start), int(blocks.shape[0]), cap], dtype=mx.int32)
    k = _kernel()
    (out,) = k(
        inputs=[
            q_row.reshape(-1),
            k_backing.reshape(-1),
            v_backing.reshape(-1),
            blocks,
            params,
            mx.array([float(scale)], dtype=mx.float32),
        ],
        template=[("T", q_row.dtype), ("GQA", hq // hkv)],
        grid=(hq * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(hq, 256)],
        output_dtypes=[q_row.dtype],
    )
    return out


def _q8_source() -> str:
    """The same kernel over int8 K/V with a float16 scale per (head, token).

    q.k = scale_k * sum(q * k_int8); the V rows enter the online softmax as
    p * scale_v * v_int8, so the int8 values never need a float16 copy."""

    src = _SRC
    src = src.replace("device const T* kh = k + (size_t)hkv * (size_t)cap * HD;",
                      "device const int8_t* kh = k + (size_t)hkv * (size_t)cap * HD;\n"
                      "    device const half* ksh = ksc + (size_t)hkv * (size_t)cap;\n"
                      "    device const half* vsh = vsc + (size_t)hkv * (size_t)cap;")
    src = src.replace("device const T* vh = v + (size_t)hkv * (size_t)cap * HD;",
                      "device const int8_t* vh = v + (size_t)hkv * (size_t)cap * HD;")
    src = src.replace("device const T* kr = kh + t * HD + dbase;", "device const int8_t* kr = kh + t * HD + dbase;")
    src = src.replace("device const T* kr = kh + (size_t)t * HD + dbase;", "device const int8_t* kr = kh + (size_t)t * HD + dbase;")
    src = src.replace("device const T* vr = vh + t * HD + dbase;", "device const int8_t* vr = vh + t * HD + dbase;")
    src = src.replace("device const T* vr = vh + (size_t)t * HD + dbase;", "device const int8_t* vr = vh + (size_t)t * HD + dbase;")
    # score scale (selected blocks: index t; tail: index t) and V scale
    src = src.replace("const float s = simd_sum(part);\n            const float mn", "const float s = simd_sum(part) * (float)ksh[t];\n            const float mn")
    src = src.replace("const float s = simd_sum(part);\n        const float mn", "const float s = simd_sum(part) * (float)ksh[t];\n        const float mn")
    src = src.replace("o[d] = o[d] * c + p * (float)vr[d];", "o[d] = o[d] * c + pv * (float)vr[d];")
    src = src.replace("l = l * c + p;\n            device const int8_t* vr", "l = l * c + p;\n            const float pv = p * (float)vsh[t];\n            device const int8_t* vr")
    src = src.replace("l = l * c + p;\n        device const int8_t* vr", "l = l * c + p;\n        const float pv = p * (float)vsh[t];\n        device const int8_t* vr")
    assert src.count("ksh[t]") == 2 and src.count("vsh[t]") == 2 and src.count("pv * (float)vr[d]") == 2, (src.count("ksh[t]"), src.count("vsh[t]"))
    return src


@lru_cache(maxsize=4)
def _kernel_q8():
    return mx.fast.metal_kernel(
        name="mtplx_qsa_flash_skip_q8",
        input_names=["q", "k", "v", "ksc", "vsc", "blocks", "params", "scale"],
        output_names=["out"],
        header=_HEADER,
        source=_q8_source(),
    )


def qsa_flash_skip_q8(q_row, kq, ks, vq, vs, blocks, t_total, tail_start, scale):
    """``qsa_flash_skip`` over the int8 backing: kq/vq [1, Hkv, cap, 256] int8,
    ks/vs [1, Hkv, cap, 1] float16 scales (read in place)."""
    hq = int(q_row.shape[0])
    hkv = int(kq.shape[1])
    cap = int(kq.shape[2])
    params = mx.array([int(t_total), int(tail_start), int(blocks.shape[0]), cap], dtype=mx.int32)
    (out,) = _kernel_q8()(
        inputs=[
            q_row.reshape(-1),
            kq.reshape(-1),
            vq.reshape(-1),
            ks.reshape(-1),
            vs.reshape(-1),
            blocks,
            params,
            mx.array([float(scale)], dtype=mx.float32),
        ],
        template=[("T", q_row.dtype), ("GQA", hq // hkv)],
        grid=(hq * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(hq, 256)],
        output_dtypes=[q_row.dtype],
    )
    return out
