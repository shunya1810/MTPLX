"""Hyper-connection (GatedResidual) read for decode and verify rows, at any width.

The eager read is ~10 small kernels per call (grouped RMS norm, down-proj, /hc, silu,
up-proj, sigmoid, multiply, mean, inject projection, inject gate), two calls per layer.
mtplx/kernels/hyper_connection.py fuses it into one threadgroup per row, which hardcodes
Flash-Next's geometry (hidden 2560, low rank 320) and stages the normed row in
threadgroup memory (20 KB; a 5120-wide model would need 40 KB, over the 32 KB of an
M1-class GPU). This version spreads the read over the whole GPU in four dispatches and
works for any hidden size, low rank and stream count:

  1. norm      one threadgroup per (row, stream): optional pending residual write
               (``hyper + block_out * inject``), grouped RMS norm, x gamma.
  2. down      the down projection and the inject projection, split over the input
               width: each threadgroup stages one chunk of every row's normed input
               and its simdgroups take two output rows each, reading the weights 16
               bytes per lane. Writes fp32 partial sums per chunk.
  3. finalize  one thread per (row, output): the chunk partials in a fixed order, then
               mix = silu(down / hc) and inject = 2 * sigmoid(inject_logits / hc).
  4. up        one simdgroup per hidden position: each stream's up-projection row
               (16 bytes per lane), sigmoid, x normed, then the mean over streams.

The mix projections are streamed once per read whatever the row count (2 x 13 MB at
hidden 5120), so a 4-row verify read costs about what a 1-row read does.

Each row is computed the same way whatever the number of rows, so a verify row and a
decode row of the same input give the same result. Op boundaries round to the input
dtype like the eager chain (norm output, x gamma, each projection output, /hc, silu,
sigmoid, the product, the mean). Not bit-identical to the eager chain (dot-product
order differs from MLX's matmuls), within a few ulp.

Weights must be unquantized and of the activation dtype (the shipped hc mixes are).
"""

from __future__ import annotations

import os
from functools import lru_cache

import mlx.core as mx

# M1 Max, hidden 5120, fp16, 32 distinct modules read as a dependent chain (each read
# folds the previous write), per read: eager 0.181 / 0.288 / 0.319 / 0.324 / 0.329 ms at
# 1 / 2 / 4 / 6 / 8 rows, these kernels 0.180 / 0.183 / 0.252 / 0.373 / 0.453 ms. Past 4
# rows the per-row work (register-held partial sums, the staged mix) outgrows the eager
# chain's matmuls, so wider verify forwards keep the eager read. A two-dispatch variant
# (norm folded into the down pass, finalize and the write into the up pass) re-reads
# each row's input per simdgroup and measured slower: 0.186 / 0.352 ms at 1 / 4 rows at
# best. In the engine (64-layer synthetic Qwen4 27B, fp16): verify 136.2 -> 128.4 ms at
# 4 rows, 126.7 -> 115.4 ms at 2 rows; 1-row decode unchanged.
MAX_ROWS = 4
_ROWS_PER_SIMDGROUP = 2  # down-projection output rows per simdgroup


def _chunk(hcd: int) -> int:
    """Input-width chunk of the down kernel: 8 rows x 1024 fp16 = 16 KB staged."""

    for ck in (1024, 512, 256, 128):
        if hcd % ck == 0:
            return ck
    raise ValueError(f"hc input width {hcd} is not a multiple of 128")

_NORM_SRC = r"""
    const uint tg = threadgroup_position_in_grid.x;   // row * HC + stream
    const uint row = tg / HC;
    const uint g = tg % HC;
    const uint t = thread_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    threadgroup float partial[32];
    threadgroup float scale_s;
    const size_t base = (size_t)row * (HC * D) + (size_t)g * D;
    float ss = 0.0f;
    for (uint i = t; i < D; i += TPG) {
        T v = x[base + i];
        if (WRITE) {
            const T prod = (T)((float)bo[(size_t)row * D + i] * (float)inj[row * HC + g]);
            v = (T)((float)v + (float)prod);
            hyper_out[base + i] = v;
        }
        const float f = (float)v;
        ss += f * f;
    }
    ss = simd_sum(ss);
    if (lane == 0) partial[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float s = lane < (TPG / 32) ? partial[lane] : 0.0f;
        s = simd_sum(s);
        if (lane == 0) scale_s = metal::rsqrt(s / (float)D + eps_arr[0]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float scale = scale_s;
    for (uint i = t; i < D; i += TPG) {
        const T v = WRITE ? hyper_out[base + i] : x[base + i];
        const T nv = (T)((float)v * scale);
        normed[base + i] = (T)((float)nv * (float)gamma[(size_t)g * D + i]);
    }
"""

_DOWN_SRC = r"""
    // threadgroup = (input chunk c, block of NSG * NR output rows); output rows are
    // [0, R) of wd, then [R, R + HC) of wi.
    const uint tg = threadgroup_position_in_grid.x;
    const uint c = tg % KS;
    const uint blk = tg / KS;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint t = thread_index_in_threadgroup;
    constexpr uint NOUT = R + HC;
    constexpr uint CV = CK / 8;  // 16-byte vectors per chunk row
    threadgroup uint4 tile[S * CV];
    for (uint i = t; i < S * CV; i += NSG * 32) {
        const uint s = i / CV;
        tile[i] = ((device const uint4*)(normed + (size_t)s * HCD + (size_t)c * CK))[i % CV];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint row0 = (blk * NSG + sg) * NR;
    if (row0 >= NOUT) return;
    float acc[NR][S];
    for (int r = 0; r < NR; ++r)
        for (int s = 0; s < S; ++s) acc[r][s] = 0.0f;
    for (uint v = lane; v < CV; v += 32) {
        uint4 xv[S];
        for (int s = 0; s < S; ++s) xv[s] = tile[s * CV + v];
        for (int r = 0; r < NR; ++r) {
            const uint n = row0 + r;
            if (n >= NOUT) break;
            device const T* w = n < (uint)R ? wd + (size_t)n * HCD : wi + (size_t)(n - R) * HCD;
            const uint4 wv = ((device const uint4*)(w + (size_t)c * CK))[v];
            const thread T* wh = (const thread T*)&wv;
            for (int s = 0; s < S; ++s) {
                const thread T* xh = (const thread T*)&xv[s];
                float a = acc[r][s];
                for (int e = 0; e < 8; ++e) a += (float)wh[e] * (float)xh[e];
                acc[r][s] = a;
            }
        }
    }
    for (int r = 0; r < NR; ++r) {
        const uint n = row0 + r;
        if (n >= NOUT) break;
        for (int s = 0; s < S; ++s) {
            const float a = simd_sum(acc[r][s]);
            if (lane == 0) partial[((size_t)c * S + s) * NOUT + n] = a;
        }
    }
"""

_FINALIZE_SRC = r"""
    // one thread per (row s, output n): chunk partials in ascending order
    constexpr uint NOUT = R + HC;
    const uint i = thread_position_in_grid.x;
    if (i >= (uint)S * NOUT) return;
    const uint s = i / NOUT;
    const uint n = i % NOUT;
    float a = 0.0f;
    for (uint c = 0; c < (uint)KS; ++c) a += partial[((size_t)c * S + s) * NOUT + n];
    const float lin = (float)((T)a);                    // Linear out
    const float div = (float)((T)(lin / (float)HC));     // / hc_count
    const float sig = 1.0f / (1.0f + metal::exp(-div));
    if (n < (uint)R) {
        mix[(size_t)s * R + n] = (T)(div * sig);          // silu
    } else {
        inject[(size_t)s * HC + (n - R)] = (T)(2.0f * (float)((T)sig));
    }
"""

_UP_SRC = r"""
    // one simdgroup per hidden position d: HC up-projection rows (g * D + d), all rows
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint t = thread_index_in_threadgroup;
    const uint d = threadgroup_position_in_grid.x * NSG + sg;
    constexpr uint RV = R / 8;
    threadgroup float mixs[S * R];
    for (uint i = t; i < (uint)(S * R); i += NSG * 32) mixs[i] = (float)mix[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (d >= (uint)D) return;
    float out[S];
    for (int s = 0; s < S; ++s) out[s] = 0.0f;
    for (int g = 0; g < HC; ++g) {
        device const uint4* urow = (device const uint4*)(wu + (size_t)(g * D + d) * R);
        float acc[S];
        for (int s = 0; s < S; ++s) acc[s] = 0.0f;
        for (uint jv = lane; jv < RV; jv += 32) {
            const uint4 uv = urow[jv];
            const thread T* uh = (const thread T*)&uv;
            for (int s = 0; s < S; ++s) {
                threadgroup const float* m = mixs + s * R + jv * 8;
                float a = acc[s];
                for (int e = 0; e < 8; ++e) a += m[e] * (float)uh[e];
                acc[s] = a;
            }
        }
        for (int s = 0; s < S; ++s) {
            const float a = (float)((T)simd_sum(acc[s]));           // Linear out
            const float w = (float)((T)(1.0f / (1.0f + metal::exp(-a))));  // sigmoid
            const T prod = (T)(w * (float)normed[(size_t)s * HCD + g * D + d]);
            out[s] += (float)prod;
        }
    }
    if (lane == 0) {
        for (int s = 0; s < S; ++s) mixed[(size_t)s * D + d] = (T)(out[s] / (float)HC);
    }
"""


@lru_cache(maxsize=8)
def _eps(eps: float) -> mx.array:
    return mx.array([eps], dtype=mx.float32)


@lru_cache(maxsize=None)
def _norm_kernel(write: bool):
    return mx.fast.metal_kernel(
        name=f"mtplx_hc_rows_norm_{'w' if write else 'r'}",
        input_names=["x", "gamma", "bo", "inj", "eps_arr"],
        output_names=["normed", "hyper_out"],
        source=_NORM_SRC,
    )


@lru_cache(maxsize=None)
def _down_kernel():
    return mx.fast.metal_kernel(
        name="mtplx_hc_rows_down",
        input_names=["normed", "wd", "wi"],
        output_names=["partial"],
        source=_DOWN_SRC,
    )


@lru_cache(maxsize=None)
def _finalize_kernel():
    return mx.fast.metal_kernel(
        name="mtplx_hc_rows_finalize",
        input_names=["partial"],
        output_names=["mix", "inject"],
        source=_FINALIZE_SRC,
    )


@lru_cache(maxsize=None)
def _up_kernel():
    return mx.fast.metal_kernel(
        name="mtplx_hc_rows_up",
        input_names=["mix", "wu", "normed"],
        output_names=["mixed"],
        source=_UP_SRC,
    )


def enabled() -> bool:
    """``MTPLX_HC_READ_ROWS``: 1/0 forces; unset follows the M1-family gate."""
    raw = (os.environ.get("MTPLX_HC_READ_ROWS") or "").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    try:
        from mtplx.cache_state import m1_long_context_defaults

        return bool(m1_long_context_defaults())
    except Exception:
        return False


def supports(module, hyper_input: mx.array) -> bool:
    """Shape/dtype contract of the three kernels for this GatedResidual and input."""
    if "block_inject_weight" not in module:
        return False
    rows = 1
    for s in hyper_input.shape[:-1]:
        rows *= int(s)
    if not 1 <= rows <= MAX_ROWS:
        return False
    dt = hyper_input.dtype
    if dt not in (mx.float16, mx.bfloat16):
        return False
    hc, d = int(module.hc_count), int(module.hidden_size)
    wd = module.input_mix_weight_down
    wu = module.input_mix_weight_up
    wi = module.block_inject_weight
    for lin in (wd, wu, wi):
        if hasattr(lin, "scales") or "bias" in lin or lin.weight.dtype != dt:
            return False
    r = int(wd.weight.shape[0])
    # The up kernel stages every row's mix (rows x r floats) in threadgroup
    # memory: r <= 768 keeps 8 rows within 24 KB of the 32 KB an M1 allows.
    # 16-byte weight loads need r and the chunked input width in whole vectors.
    if hyper_input.shape[-1] != hc * d or d % 32 or r % 32 or r > 768:
        return False
    if module.hc_norm.weight.dtype != dt:
        return False
    return True


def hc_read_rows(module, hyper_input: mx.array, pending=None):
    """(mixed, written hyper, inject) for 1..MAX_ROWS rows; ``pending`` = (block_out, inject)."""
    lead = hyper_input.shape[:-1]
    rows = 1
    for s in lead:
        rows *= int(s)
    hc, d = int(module.hc_count), int(module.hidden_size)
    hcd = hc * d
    dt = hyper_input.dtype
    wd = module.input_mix_weight_down.weight
    wu = module.input_mix_weight_up.weight
    wi = module.block_inject_weight.weight
    r = int(wd.shape[0])
    x2 = hyper_input.reshape(rows, hcd)
    write = pending is not None
    if write:
        bo, inj = pending
        bo2 = bo.reshape(rows, d).astype(dt)
        inj2 = inj.reshape(rows, hc).astype(dt)
    else:
        bo2 = inj2 = x2  # unused
    tpg = 1024 if d >= 4096 else 256
    normed, hyper_out = _norm_kernel(write)(
        inputs=[x2, module.hc_norm.weight, bo2, inj2, _eps(float(module.hc_norm.eps))],
        template=[("T", dt), ("HC", hc), ("D", d), ("TPG", tpg), ("WRITE", int(write))],
        grid=(tpg * rows * hc, 1, 1),
        threadgroup=(tpg, 1, 1),
        output_shapes=[(rows, hcd), (rows, hcd) if write else (1,)],
        output_dtypes=[dt, dt],
    )
    nsg = 8
    n_out = r + hc
    ck = _chunk(hcd)
    ks = hcd // ck
    rows_per_tg = nsg * _ROWS_PER_SIMDGROUP
    (partial,) = _down_kernel()(
        inputs=[normed, wd, wi],
        template=[("T", dt), ("HC", hc), ("R", r), ("HCD", hcd), ("S", rows), ("NSG", nsg),
                  ("NR", _ROWS_PER_SIMDGROUP), ("CK", ck), ("KS", ks)],
        grid=(32 * nsg * ks * ((n_out + rows_per_tg - 1) // rows_per_tg), 1, 1),
        threadgroup=(32 * nsg, 1, 1),
        output_shapes=[(ks * rows * n_out,)],
        output_dtypes=[mx.float32],
    )
    mix, inject = _finalize_kernel()(
        inputs=[partial],
        template=[("T", dt), ("HC", hc), ("R", r), ("S", rows), ("KS", ks)],
        grid=(rows * n_out, 1, 1),
        threadgroup=(min(256, rows * n_out), 1, 1),
        output_shapes=[(rows, r), (rows, hc)],
        output_dtypes=[dt, dt],
    )
    (mixed,) = _up_kernel()(
        inputs=[mix, wu, normed],
        template=[("T", dt), ("HC", hc), ("R", r), ("D", d), ("HCD", hcd), ("S", rows),
                  ("NSG", nsg)],
        grid=(32 * nsg * ((d + nsg - 1) // nsg), 1, 1),
        threadgroup=(32 * nsg, 1, 1),
        output_shapes=[(rows, d)],
        output_dtypes=[dt],
    )
    written = hyper_out.reshape(hyper_input.shape) if write else hyper_input
    return mixed.reshape(*lead, d), written, inject.reshape(*lead, hc)
