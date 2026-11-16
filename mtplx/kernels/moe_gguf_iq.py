"""Routed experts kept as GGUF blocks of mixed types (Unsloth UD-Q2_0 Flash-Next).

Unsloth's UD-Q2_0 Qwen3.8-Flash-Next GGUF stores each layer's routed experts
as IQ2_XXS or IQ2_S gate/up rows and Q2_0 or IQ4_NL down rows (the type
varies per layer). None of those codebooks is an MLX affine grid, so the
blocks are kept as they are, split at load into a structure of arrays per
projection:

  d  float16 [E, R, nb]       the block scale
  q  uint32  [E, R, nb * W]   the block payload, W words a block

  type     values/block  bytes/block  W   payload words
  Q2_0     64            18           4   16 bytes, 2-bit codes, w = d (q - 1)
  IQ4_NL   32            18           4   16 bytes, nibbles into a 16-entry table
  IQ2_XXS  256           66           16  8 x (4 grid bytes, 4 bytes of signs + scale)
  IQ2_S    256           82           20  qs[32] signs[32] qh[8] scales[8]

Formulas follow ggml's dequantize_row_* (llama.cpp, MIT).

Kernels:
  dequant    rows of chosen experts -> activation dtype (tests, fallbacks).
  mid        decode / verify widths: h = silu(gate . x) * (up . x) per
             (token, slot) pair; IQ2 grid staged in threadgroup memory.
  down       y = sum_slot w * (down . h) (+ shared * sigmoid(shared_gate)).
  mm_mid /   prefill widths: pairs grouped by expert; each expert's rows are
  mm_down    dequantized one 64-wide K step at a time into threadgroup memory
             and multiplied with simdgroup 8x8 matrices against tiles of the
             routed tokens.
"""

from __future__ import annotations

import os
from functools import lru_cache

import mlx.core as mx

from mtplx.kernels.gguf_iq_tables import IQ2S_GRID, IQ2XXS_GRID, IQ4NL_VALUES

TYPE_CODES = {"Q2_0": 0, "IQ4_NL": 1, "IQ2_XXS": 2, "IQ2_S": 3}
#: type -> (values a block, bytes a block, payload words a block)
BLOCKS = {"Q2_0": (64, 18, 4), "IQ4_NL": (32, 18, 4), "IQ2_XXS": (256, 66, 16), "IQ2_S": (256, 82, 20)}


def _uint2_rows(vals) -> str:
    return ",".join(f"uint2({v & 0xFFFFFFFF}u,{v >> 32}u)" for v in vals)


_HEADER = (
    "#include <metal_stdlib>\nusing namespace metal;\n"
    "constant uint2 iq2xxs_grid_c[256] = {" + _uint2_rows(IQ2XXS_GRID) + "};\n"
    "constant uint2 iq2s_grid_c[1024] = {" + _uint2_rows(IQ2S_GRID) + "};\n"
    "constant float iq4nl_values_c[16] = {" + ",".join(f"{v}.0f" for v in IQ4NL_VALUES) + "};\n"
    + r"""
// Type codes: 0 Q2_0, 1 IQ4_NL, 2 IQ2_XXS, 3 IQ2_S.
constexpr int gq_qk(int ty) { return ty == 0 ? 64 : ty == 1 ? 32 : 256; }   // values a block
constexpr int gq_w(int ty) { return ty == 2 ? 16 : ty == 3 ? 20 : 4; }      // payload words a block

inline uint gq_byte(uint w, uint i) { return (w >> (8u * i)) & 0xFFu; }

// Sign bit masks (bit 31) for 4 weights from sign bits b0..b3 of s.
inline uint4 gq_sign_lo(uint s) { return (uint4(s) << uint4(31u, 30u, 29u, 28u)) & 0x80000000u; }
inline uint4 gq_sign_hi(uint s) { return (uint4(s) << uint4(27u, 26u, 25u, 24u)) & 0x80000000u; }

// The 8 weights of 8-value group g of one block (q = the block's payload),
// unscaled by nothing: final values. grid = the IQ2 grid for TY 2 / 3.
template <int TY, typename G>
inline void gq_deq8(float d, const device uint* q, uint g, G grid, thread float* v) {
    if (TY == 0) {
        const uint word = q[g >> 1];
        const uint sh = (g & 1u) * 16u;
        for (uint j = 0; j < 8; ++j) v[j] = d * ((float)((word >> (sh + 2u * j)) & 3u) - 1.0f);
    } else if (TY == 1) {
        const uint w0 = q[(g & 1u) * 2u], w1 = q[(g & 1u) * 2u + 1u];
        const uint sh = (g >> 1) * 4u;
        for (uint j = 0; j < 4; ++j) {
            v[j] = d * iq4nl_values_c[(gq_byte(w0, j) >> sh) & 15u];
            v[4 + j] = d * iq4nl_values_c[(gq_byte(w1, j) >> sh) & 15u];
        }
    } else if (TY == 2) {
        const uint s = g >> 2, l = g & 3u;
        const uint a0 = q[2u * s], a1 = q[2u * s + 1u];
        const float db = d * (0.5f + (float)(a1 >> 28)) * 0.25f;
        const uint s7 = (a1 >> (7u * l)) & 127u;
        const uint sg = s7 | ((popcount(s7) & 1u) << 7);
        const uint2 gw = grid[gq_byte(a0, l)];
        for (uint j = 0; j < 4; ++j) {
            v[j] = db * (float)gq_byte(gw.x, j) * (((sg >> j) & 1u) ? -1.0f : 1.0f);
            v[4 + j] = db * (float)gq_byte(gw.y, j) * (((sg >> (4u + j)) & 1u) ? -1.0f : 1.0f);
        }
    } else {
        const uint s = g >> 2, l = g & 3u;
        const uint qs = gq_byte(q[s], l);
        const uint sg = gq_byte(q[8u + s], l);
        const uint qh = gq_byte(q[16u + (s >> 2)], s & 3u);
        const uint sc = gq_byte(q[18u + (s >> 2)], s & 3u);
        const float db = d * (0.5f + (float)(l < 2u ? (sc & 15u) : (sc >> 4))) * 0.25f;
        const uint2 gw = grid[qs | ((qh << (8u - 2u * l)) & 0x300u)];
        for (uint j = 0; j < 4; ++j) {
            v[j] = db * (float)gq_byte(gw.x, j) * (((sg >> j) & 1u) ? -1.0f : 1.0f);
            v[4 + j] = db * (float)gq_byte(gw.y, j) * (((sg >> (4u + j)) & 1u) ? -1.0f : 1.0f);
        }
    }
}

// The same 8 weights written to threadgroup memory as half, with vector ops
// (same float products as gq_deq8, one rounding to half).
template <int TY, typename G>
inline void gq_deq8h(float d, const device uint* q, uint g, G grid, threadgroup half* dst) {
    float4 lo, hi;
    if (TY == 0) {
        const uint word = q[g >> 1];
        const uint sh = (g & 1u) * 16u;
        lo = (float4((uint4((word >> sh) & 0xFFu) >> uint4(0u, 2u, 4u, 6u)) & 3u) - 1.0f) * d;
        hi = (float4((uint4((word >> (sh + 8u)) & 0xFFu) >> uint4(0u, 2u, 4u, 6u)) & 3u) - 1.0f) * d;
    } else if (TY == 1) {
        const uint w0 = q[(g & 1u) * 2u], w1 = q[(g & 1u) * 2u + 1u];
        const uint4 sh = uint4(0u, 8u, 16u, 24u) + (g >> 1) * 4u;
        const uint4 n0 = (uint4(w0) >> sh) & 15u, n1 = (uint4(w1) >> sh) & 15u;
        lo = float4(iq4nl_values_c[n0.x], iq4nl_values_c[n0.y], iq4nl_values_c[n0.z], iq4nl_values_c[n0.w]) * d;
        hi = float4(iq4nl_values_c[n1.x], iq4nl_values_c[n1.y], iq4nl_values_c[n1.z], iq4nl_values_c[n1.w]) * d;
    } else {
        const uint s = g >> 2, l = g & 3u;
        float db;
        uint sg;
        uint2 gw;
        if (TY == 2) {
            const uint a0 = q[2u * s], a1 = q[2u * s + 1u];
            db = d * (0.5f + (float)(a1 >> 28)) * 0.25f;
            const uint s7 = (a1 >> (7u * l)) & 127u;
            sg = s7 | ((popcount(s7) & 1u) << 7);
            gw = grid[gq_byte(a0, l)];
        } else {
            const uint qs = gq_byte(q[s], l);
            sg = gq_byte(q[8u + s], l);
            const uint qh = gq_byte(q[16u + (s >> 2)], s & 3u);
            const uint sc = gq_byte(q[18u + (s >> 2)], s & 3u);
            db = d * (0.5f + (float)(l < 2u ? (sc & 15u) : (sc >> 4))) * 0.25f;
            gw = grid[qs | ((qh << (8u - 2u * l)) & 0x300u)];
        }
        lo = as_type<float4>(as_type<uint4>(float4(as_type<uchar4>(gw.x))) ^ gq_sign_lo(sg)) * db;
        hi = as_type<float4>(as_type<uint4>(float4(as_type<uchar4>(gw.y))) ^ gq_sign_hi(sg)) * db;
    }
    *(threadgroup half4*)dst = half4(lo);
    *(threadgroup half4*)(dst + 4) = half4(hi);
}

// Raw words of the 8-weight groups gk, gk + 1 (gk even) of one row: what
// gq_deq16_raw needs, loaded a K step ahead into registers.
template <int TY>
inline void gq_load_raw(const device half* drow, const device uint* qrow, uint gk, thread float& d, thread uint4& w) {
    constexpr int QK = gq_qk(TY), W = gq_w(TY);
    const uint blk = gk / (QK / 8), g = gk % (QK / 8);
    d = (float)drow[blk];
    const device uint* q = qrow + blk * W;
    if (TY == 0) {
        w = uint4(q[g >> 1], 0u, 0u, 0u);
    } else if (TY == 1) {
        w = uint4(q[0], q[1], q[2], q[3]);
    } else if (TY == 2) {
        const uint s = g >> 2;
        w = uint4(q[2u * s], q[2u * s + 1u], 0u, 0u);
    } else {
        const uint s = g >> 2;
        w = uint4(q[s], q[8u + s], q[16u + (s >> 2)], q[18u + (s >> 2)]);
    }
}

// Half sign masks for 4 halves (uint2) from sign bits b0..b3 of s.
inline uint2 gq_hsign(uint s) {
    return uint2(((s & 1u) << 15) | ((s & 2u) << 30), ((s & 4u) << 13) | ((s & 8u) << 28));
}

// Groups g0 .. g0 + GPT - 1 of a block from the raw words, as halves at dst.
// Products are formed in half and are bit-identical to rounding gq_deq8's
// float values once: the integer factors (grid magnitude x (2 * scale + 1)
// <= 1333, q - 1, the IQ4_NL table) are exact in half, a single multiply by
// d rounds, and the remaining factor (1/8) is a power of two.
template <int TY, int GPT, typename G>
inline void gq_deq16_raw(float d, uint4 w, uint g0, G grid, threadgroup half* dst) {
    const half hd = (half)d;
    for (uint hh = 0; hh < (uint)GPT; ++hh) {
        const uint g = g0 + hh;
        half4 lo, hi;
        if (TY == 0) {
            const uint sh = (g & 1u) * 16u;
            lo = (half4((uint4((w.x >> sh) & 0xFFu) >> uint4(0u, 2u, 4u, 6u)) & 3u) - 1.0h) * hd;
            hi = (half4((uint4((w.x >> (sh + 8u)) & 0xFFu) >> uint4(0u, 2u, 4u, 6u)) & 3u) - 1.0h) * hd;
        } else if (TY == 1) {
            const uint w0 = (g & 1u) ? w.z : w.x, w1 = (g & 1u) ? w.w : w.y;
            const uint4 sh = uint4(0u, 8u, 16u, 24u) + (g >> 1) * 4u;
            const uint4 n0 = (uint4(w0) >> sh) & 15u, n1 = (uint4(w1) >> sh) & 15u;
            lo = half4(iq4nl_values_c[n0.x], iq4nl_values_c[n0.y], iq4nl_values_c[n0.z], iq4nl_values_c[n0.w]) * hd;
            hi = half4(iq4nl_values_c[n1.x], iq4nl_values_c[n1.y], iq4nl_values_c[n1.z], iq4nl_values_c[n1.w]) * hd;
        } else {
            const uint s = g >> 2, l = g & 3u;
            uint m2;     // 2 * scale + 1
            uint sg;
            uint2 gw;
            if (TY == 2) {
                m2 = 2u * (w.y >> 28) + 1u;
                const uint s7 = (w.y >> (7u * l)) & 127u;
                sg = s7 | ((popcount(s7) & 1u) << 7);
                gw = grid[gq_byte(w.x, l)];
            } else {
                const uint qs = gq_byte(w.x, l);
                sg = gq_byte(w.y, l);
                const uint qh = gq_byte(w.z, s & 3u);
                const uint sc = gq_byte(w.w, s & 3u);
                m2 = 2u * (l < 2u ? (sc & 15u) : (sc >> 4)) + 1u;
                gw = grid[qs | ((qh << (8u - 2u * l)) & 0x300u)];
            }
            const half hm = (half)m2;
            lo = ((half4(as_type<uchar4>(gw.x)) * hm) * hd) * 0.125h;
            hi = ((half4(as_type<uchar4>(gw.y)) * hm) * hd) * 0.125h;
            lo = as_type<half4>(as_type<uint2>(lo) ^ gq_hsign(sg));
            hi = as_type<half4>(as_type<uint2>(hi) ^ gq_hsign(sg >> 4));
        }
        *(threadgroup half4*)(dst + hh * 8) = lo;
        *(threadgroup half4*)(dst + hh * 8 + 4) = hi;
    }
}

inline float gq_silu(float v) { return v / (1.0f + metal::exp(-v)); }
"""
)

def _half_bits(v: float) -> int:
    import struct

    return struct.unpack("<H", struct.pack("<e", float(v)))[0]


@lru_cache(maxsize=None)
def half8_tables(typ: str):
    """(grid, signs) as uint32 [n, 4] arrays: each grid entry's 8 magnitudes as
    half8, each sign byte as a half8 mask on the sign bits (IQ2_XXS: the 128
    stored 7-bit patterns with their parity bit; IQ2_S: all 256 bytes). One
    XOR of the two gives the 8 signed grid values."""
    import numpy as np

    vals = IQ2S_GRID if typ == "IQ2_S" else IQ2XXS_GRID
    grid = np.array(
        [[_half_bits((v >> (8 * j)) & 0xFF) for j in range(8)] for v in vals], dtype=np.uint16
    ).view(np.uint32)
    if typ == "IQ2_S":
        signs = list(range(256))
    else:
        signs = [i | ((bin(i).count("1") & 1) << 7) for i in range(128)]
    sign = np.array(
        [[0x8000 if (sv >> j) & 1 else 0 for j in range(8)] for sv in signs], dtype=np.uint16
    ).view(np.uint32)
    return mx.array(grid), mx.array(sign)


@lru_cache(maxsize=None)
def grid_u2(typ: str):
    """The IQ2 grid as uint32 [n, 2]: 8 magnitude bytes an entry."""
    import numpy as np

    vals = IQ2S_GRID if typ == "IQ2_S" else IQ2XXS_GRID
    return mx.array(np.array([[v & 0xFFFFFFFF, v >> 32] for v in vals], dtype=np.uint32))


# ------------------------------------------------------------------ dequant

_DEQ_SRC = r"""
    constexpr int QK = gq_qk(TY), W = gq_w(TY);
    constexpr int NB = C / QK, GPR = C / 8;   // blocks / 8-groups a row
    const uint gid = thread_position_in_grid.x;
    const uint i = gid / (R * GPR);           // which selected expert
    if (i >= (uint)NSEL) return;
    const uint rem = gid % (R * GPR);
    const uint r = rem / GPR, gc = rem % GPR;
    const uint blk = gc / (QK / 8), g = gc % (QK / 8);
    const ulong row = (ulong)experts[i] * R + r;
    const float dv = (float)dd[row * NB + blk];
    float v[8];
    if (TY == 3) gq_deq8<TY>(dv, qq + (row * NB + blk) * W, g, iq2s_grid_c, v);
    else gq_deq8<TY>(dv, qq + (row * NB + blk) * W, g, iq2xxs_grid_c, v);
    device T* o = out + ((ulong)i * R + r) * C + gc * 8;
    for (uint j = 0; j < 8; ++j) o[j] = (T)v[j];
"""


@lru_cache(maxsize=None)
def _deq_kernel():
    return mx.fast.metal_kernel(
        name="mtplx_gguf_iq_dequant",
        input_names=["dd", "qq", "experts"],
        output_names=["out"],
        header=_HEADER,
        source=_DEQ_SRC,
    )


def dequant(d: mx.array, q: mx.array, typ: str, experts: mx.array, *, cols: int, dtype=mx.float16) -> mx.array:
    """Rows of the chosen experts as ``dtype``: d [E, R, nb], q [E, R, nb * W],
    experts [n] -> [n, R, cols]."""
    rows = int(d.shape[1])
    n = int(experts.shape[0])
    total = n * rows * (cols // 8)
    (out,) = _deq_kernel()(
        inputs=[d, q, experts.astype(mx.uint32)],
        template=[("T", dtype), ("TY", TYPE_CODES[typ]), ("C", cols), ("R", rows), ("NSEL", n)],
        grid=(total, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(n, rows, cols)],
        output_dtypes=[dtype],
    )
    return out


# ------------------------------------------------------------------ decode / verify

# mid: threadgroup = one (token, slot) pair x ROWS expert rows. 8 simdgroups,
# 2 rows each (16 lanes a row); a lane takes every 16th 32-weight sub-block.
# x is staged once as float4; the IQ2 grid is staged as uint2 (8 magnitude
# bytes) and signs are applied to x by XOR on the float sign bit.
_MID_SRC = r"""
    constexpr int NGRID = TY == 3 ? 1024 : 256;
    constexpr int NSIGN = TY == 3 ? 256 : 128;
    constexpr int NB = K / 256;          // blocks a row
    constexpr int W = gq_w(TY);
    threadgroup float4 xs[XTG ? K / 4 : 1];
    threadgroup uint4 tgrid[NGRID];
    threadgroup uint4 tsign[NSIGN];
    const uint tid = thread_position_in_threadgroup.x;
    const uint pair = threadgroup_position_in_grid.y;
    const uint t = pair / TOPK;
    const uint e = experts[pair];
    if (TY != 0) {
        for (uint i = tid; i < (uint)NGRID; i += 256) tgrid[i] = ((const device uint4*)gridh)[i];
        for (uint i = tid; i < (uint)NSIGN; i += 256) tsign[i] = ((const device uint4*)signh)[i];
    }
    if (XTG) {
        for (uint i = tid; i < (uint)(K / 4); i += 256) {
            const device T* src = x + (ulong)t * K + 4 * i;
            xs[i] = float4(src[0], src[1], src[2], src[3]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint sg = tid / 32, lane = tid % 32;
    const uint l16 = lane % 16;
    const device vec<T, 4>* xd = (const device vec<T, 4>*)(x + (ulong)t * K);
    for (uint r0 = 0; r0 < (uint)ROWS; r0 += 16 * NR) {
        const uint j0 = threadgroup_position_in_grid.x * ROWS + r0 + (sg * 2 + lane / 16) * NR;
        if (j0 >= (uint)N) break;
        float acc[2 * NR];
        for (uint i = 0; i < 2 * NR; ++i) acc[i] = 0.0f;
        for (uint sb = l16; sb < (uint)(K / 32); sb += 16) {
            const uint blk = sb / 8, s = sb % 8;
            float4 xv[8];
            for (uint i = 0; i < 8; ++i) xv[i] = XTG ? xs[sb * 8 + i] : float4(xd[sb * 8 + i]);
            float xsum = 0.0f;
            if (TY == 0)
                for (uint i = 0; i < 8; ++i) xsum += xv[i].x + xv[i].y + xv[i].z + xv[i].w;
            for (uint rp = 0; rp < 2 * NR; ++rp) {
                const uint r = rp / 2, hh = rp % 2;
                const ulong row = (ulong)e * N + j0 + r;
                if (TY == 0) {
                    // Q2_0 gate/up (the ISTA pack): 64-value blocks, so this
                    // 32-weight sub-block is half a block, two payload words;
                    // w = d (q - 1), element j in bits 2 (j % 16) of word j / 16.
                    const uint qblk = sb / 2;
                    const float d = (float)(hh == 0 ? gd : ud)[row * (K / 64) + qblk];
                    const device uint* qw = (hh == 0 ? gq_ : uq) + (row * (K / 64) + qblk) * 4 + (sb % 2) * 2;
                    float sq = 0.0f;
                    for (uint wi = 0; wi < 2; ++wi) {
                        const uint wv = qw[wi];
                        for (uint b4 = 0; b4 < 4; ++b4) {
                            const uint by = (wv >> (8 * b4)) & 0xFFu;
                            const float4 qv = float4(by & 3u, (by >> 2) & 3u, (by >> 4) & 3u, by >> 6);
                            sq += dot(qv, xv[wi * 4 + b4]);
                        }
                    }
                    acc[rp] += d * (sq - xsum);
                    continue;
                }
                const float d = (float)(hh == 0 ? gd : ud)[row * NB + blk];
                const device uint* qb = (hh == 0 ? gq_ : uq) + (row * NB + blk) * W;
                if (TY == 2) {
                    const uint2 a = ((const device uint2*)qb)[s];
                    float sub = 0.0f;
                    for (uint l = 0; l < 4; ++l) {
                        const uint4 g = tgrid[gq_byte(a.x, l)] ^ tsign[(a.y >> (7 * l)) & 127u];
                        sub += dot(float4(as_type<half4>(g.xy)), xv[2 * l]) + dot(float4(as_type<half4>(g.zw)), xv[2 * l + 1]);
                    }
                    acc[rp] += d * (0.5f + (float)(a.y >> 28)) * 0.25f * sub;
                } else {
                    const uint qs = qb[s], sw = qb[8 + s];
                    const uint qh = gq_byte(qb[16 + (s >> 2)], s & 3u);
                    const uint sc = gq_byte(qb[18 + (s >> 2)], s & 3u);
                    float sub0 = 0.0f, sub1 = 0.0f;
                    for (uint l = 0; l < 4; ++l) {
                        const uint4 g = tgrid[gq_byte(qs, l) | ((qh << (8 - 2 * l)) & 0x300u)] ^ tsign[gq_byte(sw, l)];
                        const float p = dot(float4(as_type<half4>(g.xy)), xv[2 * l]) + dot(float4(as_type<half4>(g.zw)), xv[2 * l + 1]);
                        if (l < 2) sub0 += p; else sub1 += p;
                    }
                    acc[rp] += d * 0.25f * ((0.5f + (float)(sc & 15u)) * sub0 + (0.5f + (float)(sc >> 4)) * sub1);
                }
            }
        }
        for (uint r = 0; r < (uint)NR; ++r) {
            float g = acc[2 * r], u = acc[2 * r + 1];
            for (uint o = 1; o < 16; o *= 2) { g += simd_shuffle_xor(g, o); u += simd_shuffle_xor(u, o); }
            if (l16 == 0 && j0 + r < (uint)N) {
                const float gv = (float)((T)g);
                const float uv = (float)((T)u);
                h[(ulong)pair * N + j0 + r] = (T)((float)((T)gq_silu(gv)) * uv);
            }
        }
    }
"""

# down: threadgroup = one token x 32 output dims: 8 simdgroups, 4 dims each,
# 8 lanes a dim; the token's TOPK h rows are staged in threadgroup memory.
# Q2_0 uses the pre-scaled float4 staging of moe_q2_glu (codes masked in
# place, no shifts) with w = d * (q - 1) = d * q - d.
_DOWN_SRC = r"""
    constexpr int QK = gq_qk(TY), W = gq_w(TY);
    constexpr int NVALS = TOPK * N;
    constexpr int NB = N / QK;
    constexpr int WPR = NB * W;             // payload words a row
    threadgroup float4 xs[NVALS / 4];
    threadgroup float wsum[TY == 0 ? NVALS / 16 : 1];
    threadgroup half2 kv2[TY == 1 ? 256 : 1];   // IQ4_NL byte -> (low, high) nibble values
    const uint tid = thread_position_in_threadgroup.x;
    const uint t = threadgroup_position_in_grid.y;
    const device T* src = h + (ulong)t * NVALS;
    if (TY == 0) {
        for (uint w = tid; w < (uint)(NVALS / 16); w += 256) {
            float sum = 0.0f;
            for (uint qd = 0; qd < 4; ++qd) {
                const uint i = 4 * w + qd;
                const float4 v = float4(src[4 * i], src[4 * i + 1], src[4 * i + 2], src[4 * i + 3]);
                xs[i] = v * float4(1.0f, 0.25f, 0.0625f, 0.015625f);
                sum += v.x + v.y + v.z + v.w;
            }
            wsum[w] = sum;
        }
    } else {
        for (uint i = tid; i < (uint)(NVALS / 4); i += 256)
            xs[i] = float4(src[4 * i], src[4 * i + 1], src[4 * i + 2], src[4 * i + 3]);
        if (TY == 1) kv2[tid] = half2(iq4nl_values_c[tid & 15u], iq4nl_values_c[tid >> 4]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint sg = tid / 32, lane = tid % 32, l8 = lane % LPD;
    const uint dim = threadgroup_position_in_grid.x * (256 / LPD) + sg * (32 / LPD) + lane / LPD;
    if (dim >= (uint)DM) return;
    float acc = 0.0f;
    for (uint slot = 0; slot < (uint)TOPK; ++slot) {
        const ulong row = (ulong)experts[t * TOPK + slot] * DM + dim;
        const device uint* qrow = dq + row * WPR;
        const device half* drow = ddv + row * NB;
        float dot_ = 0.0f;
        for (uint w = l8; w < (uint)WPR; w += LPD) {
            const uint word = qrow[w];
            const uint blk = w / W, j = w % W;
            const float d = (float)drow[blk];
            if (TY == 0) {
                const uint c = slot * (N / 16) + w;              // 16-value word index in the slot
                const threadgroup float4* xv = xs + 4 * c;
                const uint4 mask = uint4(3u, 12u, 48u, 192u);
                const uchar4 b = as_type<uchar4>(word);
                float s = dot(float4(uint4(b.x) & mask), xv[0]);
                s += dot(float4(uint4(b.y) & mask), xv[1]);
                s += dot(float4(uint4(b.z) & mask), xv[2]);
                s += dot(float4(uint4(b.w) & mask), xv[3]);
                dot_ += d * (s - wsum[c]);
            } else {
                // bytes 4j..4j+3: low nibbles = values 4j.., high = 16+4j..
                const uint base = slot * N + blk * 32 + 4 * j;
                const float4 xl = xs[base / 4], xh = xs[(base + 16) / 4];
                const half2 p0 = kv2[word & 0xFFu], p1 = kv2[(word >> 8) & 0xFFu];
                const half2 p2 = kv2[(word >> 16) & 0xFFu], p3 = kv2[word >> 24];
                const float4 vl = float4(p0.x, p1.x, p2.x, p3.x);
                const float4 vh = float4(p0.y, p1.y, p2.y, p3.y);
                dot_ += d * (dot(vl, xl) + dot(vh, xh));
            }
        }
        acc = metal::fma(rw[t * TOPK + slot], dot_, acc);
    }
    for (uint o = 1; o < (uint)LPD; o *= 2) acc += simd_shuffle_xor(acc, o);
    if (l8 == 0) {
        if (SHARED) {
            const float sg_ = (float)sgate[t];
            const float gate = (float)((T)(1.0f / (1.0f + metal::exp(-sg_))));
            const float sh = (float)((T)((float)shared[(ulong)t * DM + dim] * gate));
            y[(ulong)t * DM + dim] = (T)((float)((T)acc) + sh);
        } else {
            y[(ulong)t * DM + dim] = (T)acc;
        }
    }
"""


@lru_cache(maxsize=None)
def _mid_kernel():
    return mx.fast.metal_kernel(
        name="mtplx_gguf_iq_mid",
        input_names=["x", "gd", "gq_", "ud", "uq", "experts", "gridh", "signh"],
        output_names=["h"],
        header=_HEADER,
        source=_MID_SRC,
    )


@lru_cache(maxsize=None)
def _down_kernel():
    return mx.fast.metal_kernel(
        name="mtplx_gguf_iq_down",
        input_names=["h", "ddv", "dq", "experts", "rw", "shared", "sgate"],
        output_names=["y"],
        header=_HEADER,
        source=_DOWN_SRC,
    )


def _mid_nr() -> int:
    v = int(os.environ.get("MTPLX_GGUF_IQ_MID_NR", "2"))
    return v if v in (1, 2, 4) else 2


def _mid_rows() -> int:
    step = 16 * _mid_nr()
    r = int(os.environ.get("MTPLX_GGUF_IQ_MID_ROWS", "64"))
    return r if r > 0 and r % step == 0 else max(step, 64 // step * step)




def _mid_xtg(typ: str) -> int:
    raw = (os.environ.get("MTPLX_GGUF_IQ_MID_XTG") or "").strip()
    if raw in ("0", "1"):
        return int(raw)
    # x read through the cache beats staging it (M1 Max, M=1: 83-88 vs 105-118 us a layer)
    return 0


def run_mid(x2, sw, experts):
    """h [M * topk, N] = silu(gate . x) * (up . x) for each (token, slot) pair."""
    m_rows, k = int(x2.shape[0]), int(x2.shape[1])
    n_inter = sw.n_inter
    pairs = int(experts.shape[0])
    rows = _mid_rows()
    gridh, signh = half8_tables(sw.gate_type)
    (h,) = _mid_kernel()(
        inputs=[x2, sw.gate_d, sw.gate_q, sw.up_d, sw.up_q, experts, gridh, signh],
        template=[("T", x2.dtype), ("TY", TYPE_CODES[sw.gate_type]), ("K", k), ("N", n_inter),
                  ("TOPK", pairs // m_rows), ("ROWS", rows), ("NR", _mid_nr()), ("XTG", _mid_xtg(sw.gate_type))],
        grid=(((n_inter + rows - 1) // rows) * 256, pairs, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(pairs, n_inter)],
        output_dtypes=[x2.dtype],
    )
    return h


def run_down(h, sw, experts, route_w, m_rows, shared=None, shared_gate=None):
    """y [M, D] = sum_slot w * (down . h) (+ shared * sigmoid(shared_gate))."""
    dmodel = sw.hidden
    topk = int(experts.shape[0]) // m_rows
    with_shared = shared is not None
    if not with_shared:
        shared = shared_gate = h  # unused
    # Lanes a output dim: 4 for Q2_0 at verify widths (one staging of h then
    # serves 64 dims; M1 Max, M=4: 147 vs 168 us a layer), else 8.
    lpd = 4 if sw.down_type == "Q2_0" and m_rows >= 2 else 8
    dpt = 256 // lpd
    (y,) = _down_kernel()(
        inputs=[h, sw.down_d, sw.down_q, experts, route_w, shared, shared_gate],
        template=[("T", h.dtype), ("TY", TYPE_CODES[sw.down_type]), ("N", sw.n_inter), ("TOPK", topk),
                  ("DM", dmodel), ("SHARED", int(with_shared)), ("LPD", lpd)],
        grid=(((dmodel + dpt - 1) // dpt) * 256, m_rows, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(m_rows, dmodel)],
        output_dtypes=[h.dtype],
    )
    return y


def moe_iq_glu(x2, sw, experts, route_w, shared=None, shared_gate=None):
    """Routed experts at decode / verify widths for a GGUFMixedSwitchGLU ``sw``.
    x2 [M, K]; experts uint32 [M * topk] (token-major); route_w float32
    [M * topk]. With ``shared`` [M, D] and ``shared_gate`` [M] the down pass
    adds shared * sigmoid(shared_gate). Returns y [M, D] in x2's dtype."""
    h = run_mid(x2, sw, experts)
    return run_down(h, sw, experts, route_w, int(x2.shape[0]), shared, shared_gate)


# ------------------------------------------------------------------ prefill (expert-grouped MMA)

# grid.x = 32-row blocks of the output, grid.y = experts; 128 threads, 4
# simdgroups of 8 rows each. The pairs routed to expert e are
# order[start[e] : start[e] + count[e]] (pair = token * TOPK + slot); they are
# taken in tiles of TT = 8 * NT. Each K step of 64 dequantizes the 32 x 64
# weight tile into threadgroup memory (thread = row tid / 4, quarter tid % 4
# = two 8-weight groups) and the TT routed activations next to it.
_MM_MID_SRC = r"""
    constexpr uint ROWS = 8 * NSG, KS = KSC, TT = 8 * NT, NTH = 32 * NSG, GPT = KS / 32;
    // padded tile strides (halves) against threadgroup bank conflicts
    constexpr uint AST = KS + APAD, BST = TT + BPAD;
    constexpr int QK = gq_qk(TY), W = gq_w(TY);
    constexpr int NB = K / QK;
    constexpr int NGRID = TY == 3 ? 1024 : 256;
    threadgroup half4 Ag4[ROWS * AST / 4];
    threadgroup half4 Au4[ROWS * AST / 4];
    threadgroup half4 Bs4[KS * BST / 4];
    threadgroup half* Ag = (threadgroup half*)Ag4;
    threadgroup half* Au = (threadgroup half*)Au4;
    threadgroup half* Bs = (threadgroup half*)Bs4;
    // C staging after the K loop reuses the A tiles (NSG x 2 x 64 floats).
    threadgroup float (*Cs)[2][64] = (threadgroup float (*)[2][64])Ag4;
    threadgroup uint2 tgrid[NGRID];
    const uint tid = thread_position_in_threadgroup.x;
    const uint sgi = tid / 32;
    const uint rb = threadgroup_position_in_grid.x, tile = threadgroup_position_in_grid.y;
    // One token tile of one expert per threadgroup: the expert is the first
    // whose running tile count passes this tile (tile_ends = cumsum of tiles).
    if (tile >= (uint)tile_ends[NE - 1]) return;
    uint lo_ = 0, hi_ = NE - 1;
    while (lo_ < hi_) {
        const uint m_ = (lo_ + hi_) / 2;
        if ((uint)tile_ends[m_] > tile) hi_ = m_; else lo_ = m_ + 1;
    }
    const uint e = lo_;
    const uint count = (uint)counts[e];
    const uint t0_first = (tile - ((uint)tile_ends[e] - (count + TT - 1) / TT)) * TT;
    for (uint i = tid; i < (uint)NGRID; i += NTH) tgrid[i] = ((const device uint2*)gridb)[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint start = (uint)starts[e];
    const uint row0 = rb * ROWS;
    constexpr uint NK = K / KS;
    for (uint t0 = t0_first; t0 < t0_first + 1; t0 += TT) {   // this threadgroup's tile
        const uint n_tile = min(TT, count - t0);
        simdgroup_float8x8 Cg[NT], Cu[NT];
        for (uint nt = 0; nt < NT; nt++) {
            Cg[nt] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            Cu[nt] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
        const uint tok = tid % TT, kq = tid / TT;
        constexpr uint KPT = KS * TT / NTH;
        const int my_pair = tok < n_tile ? order[start + t0 + tok] : -1;
        const device T* xrow = x + (ulong)(my_pair >= 0 ? (uint)my_pair / TOPK : 0) * K + kq * KPT;
        const uint r = tid / 4, qd = tid % 4;
        const ulong row = (ulong)e * N + row0 + r;
        const device half* gdrow = gd + row * NB;
        const device half* udrow = ud + row * NB;
        const device uint* gqrow = gq_ + row * NB * W;
        const device uint* uqrow = uq + row * NB * W;
        threadgroup half* gdst = Ag + r * AST + qd * 8 * GPT;
        threadgroup half* udst = Au + r * AST + qd * 8 * GPT;
        // registers for the next K step: raw gate/up words and activations
        float dg, du;
        uint4 wg, wu;
        T bx[KPT];
        gq_load_raw<TY>(gdrow, gqrow, qd * GPT, dg, wg);
        gq_load_raw<TY>(udrow, uqrow, qd * GPT, du, wu);
        if (my_pair >= 0)
            for (uint j = 0; j < KPT; j++) bx[j] = xrow[j];
        for (uint kb = 0; kb < NK; kb++) {
            const uint g0 = (kb * (KS / 8) + qd * GPT) % (QK / 8);
            gq_deq16_raw<TY, GPT>(dg, wg, g0, tgrid, gdst);
            gq_deq16_raw<TY, GPT>(du, wu, g0, tgrid, udst);
            if (my_pair >= 0)
                for (uint j = 0; j < KPT; j++) Bs[(kq * KPT + j) * BST + tok] = (half)bx[j];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (kb + 1 < NK) {
                const uint gk = (kb + 1) * (KS / 8) + qd * GPT;
                gq_load_raw<TY>(gdrow, gqrow, gk, dg, wg);
                gq_load_raw<TY>(udrow, uqrow, gk, du, wu);
                if (my_pair >= 0)
                    for (uint j = 0; j < KPT; j++) bx[j] = xrow[(kb + 1) * KS + j];
            }
            for (uint sub = 0; sub < KS / 8; sub++) {
                simdgroup_half8x8 ag, au, b;
                simdgroup_load(ag, Ag + (sgi * 8) * AST + sub * 8, AST, 0, false);
                simdgroup_load(au, Au + (sgi * 8) * AST + sub * 8, AST, 0, false);
                // Every sub-tile is multiplied (a guard on n_tile here costs ~35%:
                // M1 Max, 2048 rows); columns past n_tile hold stale tokens and are
                // never stored.
                for (uint nt = 0; nt < NT; nt++) {
                    simdgroup_load(b, Bs + sub * 8 * BST + nt * 8, BST, 0, false);
                    simdgroup_multiply_accumulate(Cg[nt], ag, b, Cg[nt]);
                    simdgroup_multiply_accumulate(Cu[nt], au, b, Cu[nt]);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
#pragma unroll
        for (uint nt = 0; nt < NT; nt++) {
            if (nt * 8 >= n_tile) continue;
            simdgroup_store(Cg[nt], Cs[sgi][0], 8, 0, false);
            simdgroup_store(Cu[nt], Cs[sgi][1], 8, 0, false);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint idx = tid; idx < NSG * 64; idx += NTH) {
                const uint sgx = idx / 64, el = idx % 64, r = el / 8, tok = nt * 8 + el % 8;
                if (tok >= n_tile) continue;
                const uint pair = (uint)order[start + t0 + tok];
                const float gv = (float)((T)Cs[sgx][0][el]);
                const float uv = (float)((T)Cs[sgx][1][el]);
                mid[(ulong)pair * N + row0 + sgx * 8 + r] = (T)((float)((T)gq_silu(gv)) * uv);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
"""

_MM_DOWN_SRC = r"""
    constexpr uint ROWS = 8 * NSG, KS = KSC, TT = 8 * NT, NTH = 32 * NSG, GPT = KS / 32;
    // padded tile strides (halves) against threadgroup bank conflicts
    constexpr uint AST = KS + APAD, BST = TT + BPAD;
    constexpr int QK = gq_qk(TY), W = gq_w(TY);
    constexpr int NB = N / QK;
    threadgroup half4 As4[ROWS * AST / 4];
    threadgroup half4 Bs4[KS * BST / 4];
    threadgroup half* As = (threadgroup half*)As4;
    threadgroup half* Bs = (threadgroup half*)Bs4;
    threadgroup float (*Cs)[64] = (threadgroup float (*)[64])As4;   // after the K loop
    const uint tid = thread_position_in_threadgroup.x;
    const uint sgi = tid / 32;
    const uint rb = threadgroup_position_in_grid.x, tile = threadgroup_position_in_grid.y;
    // One token tile of one expert per threadgroup: the expert is the first
    // whose running tile count passes this tile (tile_ends = cumsum of tiles).
    if (tile >= (uint)tile_ends[NE - 1]) return;
    uint lo_ = 0, hi_ = NE - 1;
    while (lo_ < hi_) {
        const uint m_ = (lo_ + hi_) / 2;
        if ((uint)tile_ends[m_] > tile) hi_ = m_; else lo_ = m_ + 1;
    }
    const uint e = lo_;
    const uint count = (uint)counts[e];
    const uint t0_first = (tile - ((uint)tile_ends[e] - (count + TT - 1) / TT)) * TT;
    const uint start = (uint)starts[e];
    const uint row0 = rb * ROWS;
    constexpr uint NK = N / KS;
    for (uint t0 = t0_first; t0 < t0_first + 1; t0 += TT) {   // this threadgroup's tile
        const uint n_tile = min(TT, count - t0);
        simdgroup_float8x8 C[NT];
        for (uint nt = 0; nt < NT; nt++) C[nt] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        const uint tok = tid % TT, kq = tid / TT;
        constexpr uint KPT = KS * TT / NTH;
        const int my_pair = tok < n_tile ? order[start + t0 + tok] : -1;
        const device T* mrow = mid + (ulong)(my_pair >= 0 ? my_pair : 0) * N + kq * KPT;
        const uint r = tid / 4, qd = tid % 4;
        const ulong row = (ulong)e * DM + row0 + r;
        const device half* drow = ddv + row * NB;
        const device uint* qrow = dq + row * NB * W;
        threadgroup half* dst = As + r * AST + qd * 8 * GPT;
        float dd_;
        uint4 ww;
        T bx[KPT];
        gq_load_raw<TY>(drow, qrow, qd * GPT, dd_, ww);
        if (my_pair >= 0)
            for (uint j = 0; j < KPT; j++) bx[j] = mrow[j];
        for (uint kb = 0; kb < NK; kb++) {
            gq_deq16_raw<TY, GPT>(dd_, ww, (kb * (KS / 8) + qd * GPT) % (QK / 8), iq2xxs_grid_c, dst);
            if (my_pair >= 0)
                for (uint j = 0; j < KPT; j++) Bs[(kq * KPT + j) * BST + tok] = (half)bx[j];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (kb + 1 < NK) {
                gq_load_raw<TY>(drow, qrow, (kb + 1) * (KS / 8) + qd * GPT, dd_, ww);
                if (my_pair >= 0)
                    for (uint j = 0; j < KPT; j++) bx[j] = mrow[(kb + 1) * KS + j];
            }
            for (uint sub = 0; sub < KS / 8; sub++) {
                simdgroup_half8x8 a, b;
                simdgroup_load(a, As + (sgi * 8) * AST + sub * 8, AST, 0, false);
                for (uint nt = 0; nt < NT; nt++) {
                    simdgroup_load(b, Bs + sub * 8 * BST + nt * 8, BST, 0, false);
                    simdgroup_multiply_accumulate(C[nt], a, b, C[nt]);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
#pragma unroll
        for (uint nt = 0; nt < NT; nt++) {
            if (nt * 8 >= n_tile) continue;
            simdgroup_store(C[nt], Cs[sgi], 8, 0, false);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint idx = tid; idx < NSG * 64; idx += NTH) {
                const uint sgx = idx / 64, el = idx % 64, r = el / 8, tok = nt * 8 + el % 8;
                if (tok >= n_tile) continue;
                const uint pair = (uint)order[start + t0 + tok];
                part[(ulong)pair * DM + row0 + sgx * 8 + r] = Cs[sgx][el];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
"""


@lru_cache(maxsize=None)
def _mm_mid_kernel():
    return mx.fast.metal_kernel(
        name="mtplx_gguf_iq_mm_mid",
        input_names=["x", "gd", "gq_", "ud", "uq", "order", "starts", "counts", "gridb", "tile_ends"],
        output_names=["mid"],
        header=_HEADER,
        source=_MM_MID_SRC,
    )


@lru_cache(maxsize=None)
def _mm_down_kernel():
    return mx.fast.metal_kernel(
        name="mtplx_gguf_iq_mm_down",
        input_names=["mid", "ddv", "dq", "order", "starts", "counts", "tile_ends"],
        output_names=["part"],
        header=_HEADER,
        source=_MM_DOWN_SRC,
    )


def _mm_nt(pairs: int = 0, n_experts: int = 1) -> int:
    """Routed tokens a weight tile is multiplied with (8 * NT). Unset: 32 when
    experts average 32 or more pairs, else 16 (M1 Max: 2,048 rows 66 vs 72 ms a
    layer, 512 rows 29 vs 41 ms)."""
    raw = (os.environ.get("MTPLX_GGUF_IQ_MM_NT") or "").strip()
    if raw in ("1", "2", "4", "8"):
        return int(raw)
    return 4 if pairs >= 32 * n_experts else 2


def _mm_nsg() -> int:
    raw = (os.environ.get("MTPLX_GGUF_IQ_MM_NSG") or "").strip()
    return int(raw) if raw in ("4", "8") else 8


def _mm_ks() -> int:
    raw = (os.environ.get("MTPLX_GGUF_IQ_MM_KS") or "").strip()
    return int(raw) if raw in ("32", "64") else 64


def _mm_pad(name: str, default: int) -> int:
    raw = (os.environ.get(f"MTPLX_GGUF_IQ_MM_{name}") or "").strip()
    return int(raw) if raw in ("0", "4", "8", "16") else default


def expert_lists(experts_flat: mx.array, n_experts: int):
    """Pairs grouped by expert: (order int32 [P], starts int32 [E], counts int32 [E])."""
    flat = experts_flat.astype(mx.int32)
    order = mx.argsort(flat).astype(mx.int32)
    counts = mx.zeros((n_experts,), dtype=mx.int32).at[flat].add(1)
    starts = (mx.cumsum(counts) - counts).astype(mx.int32)
    return order, starts, counts


def _tiles(counts, pairs: int, n_experts: int, nt: int):
    """(tile_ends int32 [E], an upper bound on the tile count) for token tiles of 8 * nt."""
    tt = 8 * nt
    tile_ends = mx.cumsum((counts + (tt - 1)) // tt).astype(mx.int32)
    return tile_ends, pairs // tt + n_experts


def run_mm_mid(x2, sw, order, starts, counts, topk):
    """Expert-grouped GLU at prefill widths: h [M * topk, N] in x2's dtype."""
    m_rows, k = int(x2.shape[0]), int(x2.shape[1])
    n_inter = sw.n_inter
    nsg = _mm_nsg()
    nt = _mm_nt(m_rows * topk, sw.num_experts)
    tile_ends, n_tiles = _tiles(counts, m_rows * topk, sw.num_experts, nt)
    (mid,) = _mm_mid_kernel()(
        inputs=[x2, sw.gate_d, sw.gate_q, sw.up_d, sw.up_q, order, starts, counts, grid_u2(sw.gate_type), tile_ends],
        template=[("T", x2.dtype), ("TY", TYPE_CODES[sw.gate_type]), ("K", k), ("N", n_inter),
                  ("TOPK", topk), ("NT", nt), ("NSG", nsg), ("KSC", _mm_ks()), ("NE", sw.num_experts), ("APAD", _mm_pad("APAD", 0)), ("BPAD", _mm_pad("BPAD", 8))],
        grid=((n_inter // (8 * nsg)) * 32 * nsg, n_tiles, 1),
        threadgroup=(32 * nsg, 1, 1),
        output_shapes=[(m_rows * topk, n_inter)],
        output_dtypes=[x2.dtype],
    )
    return mid


def run_mm_down(mid, sw, order, starts, counts):
    """Expert-grouped down at prefill widths: float32 [pairs, D] in pair order."""
    pairs = int(mid.shape[0])
    nsg = _mm_nsg()
    raw = (os.environ.get("MTPLX_GGUF_IQ_MM_NT_DOWN") or "").strip()
    nt = int(raw) if raw in ("1", "2", "4", "8") else _mm_nt(pairs, sw.num_experts)
    tile_ends, n_tiles = _tiles(counts, pairs, sw.num_experts, nt)
    (part,) = _mm_down_kernel()(
        inputs=[mid, sw.down_d, sw.down_q, order, starts, counts, tile_ends],
        template=[("T", mid.dtype), ("TY", TYPE_CODES[sw.down_type]), ("N", sw.n_inter), ("DM", sw.hidden),
                  ("NT", nt), ("NSG", nsg), ("KSC", _mm_ks()), ("NE", sw.num_experts), ("APAD", _mm_pad("APAD", 0)), ("BPAD", _mm_pad("BPAD", 8))],
        grid=((sw.hidden // (8 * nsg)) * 32 * nsg, n_tiles, 1),
        threadgroup=(32 * nsg, 1, 1),
        output_shapes=[(pairs, sw.hidden)],
        output_dtypes=[mx.float32],
    )
    return part


def moe_iq_prefill_parts(x2, sw, experts_flat):
    """Per-pair expert outputs at prefill widths: x2 [M, K], experts_flat
    [M * topk] -> float32 [M * topk, D] (pair = token * topk + slot)."""
    m_rows = int(x2.shape[0])
    topk = int(experts_flat.shape[0]) // m_rows
    order, starts, counts = expert_lists(experts_flat, sw.num_experts)
    mid = run_mm_mid(x2, sw, order, starts, counts, topk)
    return run_mm_down(mid, sw, order, starts, counts)
