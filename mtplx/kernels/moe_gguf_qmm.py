"""Prefill-width GGUF routed experts on MLX's quantized-matmul tile body (M1).

The expert-grouped MMA of ``moe_gguf_iq`` (its own 64-row x 16/32-token tiles)
ran the Unsloth UD-Q2_0 prefill at about 55 ms a layer for a 2,048-token chunk
on an M1 Max, where the ISTA Q2_0 experts on the M1 segmented gather (MLX's
``qmm_t_impl`` tile, ``moe_seg_gather_m1``) take 40 ms. The difference is the
MMA structure, not the dequantization: per expert tile both decode every weight
once and reuse it for every token of the tile.

Here each threadgroup runs MLX's own tile (steel ``BlockMMA``, the activation
``BlockLoader``, 32 tokens x 64 output rows x 32-deep K steps, 128 threads)
over up to 32 expert-sorted rows of ONE expert, as the segmented gather does;
only the weight loader differs: it decodes GGUF blocks (IQ2_XXS, IQ2_S, Q2_0,
IQ4_NL; ``moe_gguf_iq``'s ``gq_deq16_raw``) into the same threadgroup tile
MLX's ``QuantizedBlockLoader`` would fill. Gate and up are one [2N]-wide
projection whose 64-row output tiles each read one of the two matrices.
"""

from __future__ import annotations

import os
from functools import lru_cache

import mlx.core as mx

from mtplx.kernels import moe_gguf_iq as K
from mtplx.kernels import moe_seg_gather_m1 as SG

BM = 32
BN = 64
BK = 32
THREADS = 128


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name) or default)
    except ValueError:
        return default


#: Routed rows up to this take 16-row tiles (fewer padded rows per expert):
#: M1 Max, IQ2_S/Q2_0 layer, 1,024 tokens 31.5 (16) vs 33.1 ms (32), 512:
#: 22.6 vs 28.9, 128: 15.8 vs 22.6; 2,048 tokens 32-row tiles 46.8 vs 56.9.
NARROW_MAX_ROWS = 16384


def _shape(pairs: int):
    """(BM, BK, BMS) of the tiles: MTPLX_GGUF_QMM_BM (16/32/64), _BK (32/64).

    BMS is the size of the small tile of the hybrid layout (0 = single size):
    full 32-row tiles, and each expert's last tile as a 16-row tile when 16 or
    fewer of its rows are left (hybrid_enabled; an explicit MTPLX_GGUF_QMM_BM
    keeps one size, which is what the shape tests and sweeps ask for)."""
    bk = _env_int("MTPLX_GGUF_QMM_BK", BK)
    bk = bk if bk in (32, 64) else BK
    if hybrid_enabled() and not (os.environ.get("MTPLX_GGUF_QMM_BM") or "").strip():
        return BM, bk, HYBRID_SMALL
    bm = _env_int("MTPLX_GGUF_QMM_BM", 16 if pairs <= NARROW_MAX_ROWS else BM)
    return (bm if bm in (16, 32, 64) else BM), bk, 0


HYBRID_SMALL = 16


def hybrid_enabled() -> bool:
    """MTPLX_GGUF_QMM_HYBRID (default on): 32-row expert tiles with a 16-row
    tile for each expert's remainder of 16 rows or fewer, in one dispatch."""
    raw = (os.environ.get("MTPLX_GGUF_QMM_HYBRID") or "").strip().lower()
    return raw not in {"0", "off", "false", "no"}


_LOADER = r"""
// [BROWS output rows, BCOLS K] weight tile decoded from GGUF blocks into the
// layout MLX's QuantizedBlockLoader writes (row = output row, K contiguous).
// Each thread decodes GPT 8-weight groups of one row.
template <typename T, short BROWS, short BCOLS, short dst_ld, short tgp_size, int TY, int HALF8>
struct MtplxGGUFLoader {
  STEEL_CONST short n_vals = BROWS * BCOLS / tgp_size;
  STEEL_CONST short GPT = n_vals / 8;
  STEEL_CONST short TPR = BCOLS / n_vals;
  STEEL_CONST int QK = gq_qk(TY);
  STEEL_CONST int W = gq_w(TY);
  const short bi;
  const short bj;
  threadgroup T* dst;
  const device half* drow;
  const device uint* qrow;
  threadgroup const uint2* grid;
  threadgroup const uint4* grid8;
  int k_off;
  METAL_FUNC MtplxGGUFLoader(const device half* d_, const device uint* q_, int nb,
                             threadgroup const uint2* grid_, threadgroup const uint4* grid8_, threadgroup T* dst_,
                             ushort simd_group_id, ushort simd_lane_id) thread
      : bi((simd_group_id * 32 + simd_lane_id) / TPR),
        bj(((simd_group_id * 32 + simd_lane_id) % TPR) * n_vals),
        dst(dst_ + bi * dst_ld + bj),
        drow(d_ + size_t(bi) * nb),
        qrow(q_ + size_t(bi) * nb * W),
        grid(grid_),
        grid8(grid8_),
        k_off(0) {}
  // Q2_0 raw words hold two 8-weight groups; the other types load one
  // block / sub-block that covers all GPT groups (GPT <= 4, aligned).
  STEEL_CONST short STEP = TY == 0 ? 2 : GPT;
  METAL_FUNC void load_unsafe() const thread {
    const uint gk0 = uint(k_off + bj) / 8u;
    if (HALF8 && TY >= 2) {
      // grid entries as half8 magnitudes (threadgroup uint4 table): no byte
      // conversion; same products as gq_deq16_raw, bit for bit.
      float d;
      uint4 w;
      gq_load_raw<TY>(drow, qrow, gk0, d, w);
      const half hd = (half)d;
      threadgroup half* out = (threadgroup half*)dst;
      for (short hh = 0; hh < GPT; ++hh) {
        const uint g = gk0 % uint(QK / 8) + hh;
        const uint s = g >> 2, l = g & 3u;
        uint m2, sg;
        uint4 gv;
        if (TY == 2) {
          m2 = 2u * (w.y >> 28) + 1u;
          const uint s7 = (w.y >> (7u * l)) & 127u;
          sg = s7 | ((popcount(s7) & 1u) << 7);
          gv = grid8[gq_byte(w.x, l)];
        } else {
          sg = gq_byte(w.y, l);
          const uint qh = gq_byte(w.z, s & 3u);
          const uint sc = gq_byte(w.w, s & 3u);
          m2 = 2u * (l < 2u ? (sc & 15u) : (sc >> 4)) + 1u;
          gv = grid8[gq_byte(w.x, l) | ((qh << (8u - 2u * l)) & 0x300u)];
        }
        const half hm = (half)m2;
        half4 lo = ((as_type<half4>(gv.xy) * hm) * hd) * 0.125h;
        half4 hi = ((as_type<half4>(gv.zw) * hm) * hd) * 0.125h;
        lo = as_type<half4>(as_type<uint2>(lo) ^ gq_hsign(sg));
        hi = as_type<half4>(as_type<uint2>(hi) ^ gq_hsign(sg >> 4));
        *(threadgroup half4*)(out + hh * 8) = lo;
        *(threadgroup half4*)(out + hh * 8 + 4) = hi;
      }
      return;
    }
    for (short hh = 0; hh < GPT; hh += STEP) {
      const uint gk = gk0 + hh;
      float d;
      uint4 w;
      gq_load_raw<TY>(drow, qrow, gk, d, w);
      gq_deq16_raw<TY, STEP>(d, w, gk % uint(QK / 8), grid, (threadgroup half*)dst + hh * 8);
    }
  }
  METAL_FUNC void next() thread { k_off += BCOLS; }
};

template <typename T, int TY, int BM_, int BK_, int BN_, int HALF8>
METAL_FUNC void mtplx_qmm_t_gguf(
    const device half* wd, const device uint* wq, int nb, threadgroup const uint2* grid,
    threadgroup const uint4* grid8,
    const device T* x, device T* y, threadgroup T* Xs, threadgroup T* Ws,
    const int Kd, const int ldy, const int rows,
    uint simd_gid, uint simd_lid) {
  constexpr int WM = 2;
  constexpr int WN = 2;
  constexpr int BK_padded = (BK_ + 16 / sizeof(T));
  using mma_t = mlx::steel::BlockMMA<T, T, BM_, BN_, BK_, WM, WN, false, true, BK_padded, BK_padded>;
  using loader_x_t = mlx::steel::BlockLoader<T, BM_, BK_, BK_padded, 1, WM * WN * SIMD_SIZE>;
  using loader_w_t = MtplxGGUFLoader<T, BN_, BK_, BK_padded, WM * WN * SIMD_SIZE, TY, HALF8>;
  const short num_els = min(BM_, rows);
  loader_x_t loader_x(x, Kd, Xs, simd_gid, simd_lid);
  loader_w_t loader_w(wd, wq, nb, grid, grid8, Ws, simd_gid, simd_lid);
  mma_t mma_op(simd_gid, simd_lid);
  if (num_els < BM_) {
    for (int k = 0; k < Kd; k += BK_) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      loader_x.load_safe(short2(BK_, num_els));
      loader_w.load_unsafe();
      threadgroup_barrier(mem_flags::mem_threadgroup);
      mma_op.mma(Xs, Ws);
      loader_x.next();
      loader_w.next();
    }
  } else {
    for (int k = 0; k < Kd; k += BK_) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      loader_x.load_unsafe();
      loader_w.load_unsafe();
      threadgroup_barrier(mem_flags::mem_threadgroup);
      mma_op.mma(Xs, Ws);
      loader_x.next();
      loader_w.next();
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (num_els < BM_) {
    mma_op.store_result_safe(y, ldy, short2(BN_, num_els));
  } else {
    mma_op.store_result(y, ldy);
  }
}
"""

# One threadgroup = one column tile x one row tile of one expert (binary search
# over the tiles-per-expert prefix, as moe_seg_gather_m1). Gate/up: column
# tiles below N read the gate rows, the rest the up rows; y is [P, 2N].
_SRC_GU = r"""
    threadgroup T Xs[BM * (BK + 16 / sizeof(T))];
    threadgroup T Ws[BN * (BK + 16 / sizeof(T))];
    threadgroup uint2 tgrid[(TY >= 2 && !HALF8) ? (TY == 3 ? 1024 : 256) : 1];
    threadgroup uint4 tgrid8[(TY >= 2 && HALF8) ? (TY == 3 ? 1024 : 256) : 1];
    const int tile = int(threadgroup_position_in_grid.y);
    if (tile >= tile_start[ED]) return;
    if (TY >= 2) {
        const uint n = TY == 3 ? 1024u : 256u;
        if (HALF8) {
            for (uint i = thread_index_in_threadgroup; i < n; i += 128u) tgrid8[i] = ((const device uint4*)gridb)[i];
        } else {
            for (uint i = thread_index_in_threadgroup; i < n; i += 128u) tgrid[i] = ((const device uint2*)gridb)[i];
        }
    }
    int lo = 0, hi = ED - 1;
    while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (tile_start[mid] <= tile) lo = mid; else hi = mid - 1;
    }
    const int expert = lo;
    const int local = tile - tile_start[expert];
    const int rem_rows = row_start[expert + 1] - row_start[expert] - local * BM;   // rows from this tile on
    const int row0 = row_start[expert] + local * BM;
    const int tile_rows = min(BM, rem_rows);
    const int col = int(threadgroup_position_in_grid.x) * BN;
    constexpr int NB = KD / gq_qk(TY);
    const bool is_up = col >= ND;
    const int r0 = is_up ? col - ND : col;
    const size_t wrow = size_t(expert) * ND + r0;
    const device half* wd = (is_up ? ud : gd) + wrow * NB;
    const device uint* wq = (is_up ? uq : gq_) + wrow * NB * gq_w(TY);
    if (BMS > 0 && rem_rows <= BMS) {
        // an expert's last tile with BMS rows or fewer left: the small tile (same dot products, same order)
        mtplx_qmm_t_gguf<T, TY, (BMS > 0 ? BMS : BM), BK, BN, HALF8>(
            wd, wq, NB, tgrid, tgrid8, x + size_t(row0) * KD, y + size_t(row0) * (2 * ND) + col, Xs, Ws,
            KD, 2 * ND, tile_rows, simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
    } else {
        mtplx_qmm_t_gguf<T, TY, BM, BK, BN, HALF8>(
            wd, wq, NB, tgrid, tgrid8, x + size_t(row0) * KD, y + size_t(row0) * (2 * ND) + col, Xs, Ws,
            KD, 2 * ND, tile_rows, simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
    }
"""

_SRC_DOWN = r"""
    threadgroup T Xs[BM * (BK + 16 / sizeof(T))];
    threadgroup T Ws[BN * (BK + 16 / sizeof(T))];
    threadgroup uint2 tgrid[1];
    threadgroup uint4 tgrid8[1];
    const int tile = int(threadgroup_position_in_grid.y);
    if (tile >= tile_start[ED]) return;
    int lo = 0, hi = ED - 1;
    while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (tile_start[mid] <= tile) lo = mid; else hi = mid - 1;
    }
    const int expert = lo;
    const int local = tile - tile_start[expert];
    const int rem_rows = row_start[expert + 1] - row_start[expert] - local * BM;
    const int row0 = row_start[expert] + local * BM;
    const int tile_rows = min(BM, rem_rows);
    const int col = int(threadgroup_position_in_grid.x) * BN;
    constexpr int NB = KD / gq_qk(TY);
    const size_t wrow = size_t(expert) * ND + col;
    if (BMS > 0 && rem_rows <= BMS) {
        mtplx_qmm_t_gguf<T, TY, (BMS > 0 ? BMS : BM), BK, BN, 0>(
            dd + wrow * NB, dq + wrow * NB * gq_w(TY), NB, tgrid, tgrid8, x + size_t(row0) * KD,
            y + size_t(row0) * ND + col, Xs, Ws, KD, ND, tile_rows,
            simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
    } else {
        mtplx_qmm_t_gguf<T, TY, BM, BK, BN, 0>(
            dd + wrow * NB, dq + wrow * NB * gq_w(TY), NB, tgrid, tgrid8, x + size_t(row0) * KD,
            y + size_t(row0) * ND + col, Xs, Ws, KD, ND, tile_rows,
            simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
    }
"""


@lru_cache(maxsize=1)
def _header():
    mlx = SG._mlx_headers()
    if mlx is None:
        return None
    return K._HEADER + "\n" + mlx + "\n" + _LOADER


@lru_cache(maxsize=None)
def _kernels():
    head = _header()
    if head is None:
        return None
    gu = mx.fast.metal_kernel(
        name="mtplx_gguf_qmm_gate_up",
        input_names=["x", "gd", "gq_", "ud", "uq", "gridb", "tile_start", "row_start"],
        output_names=["y"], header=head, source=_SRC_GU, ensure_row_contiguous=True)
    dn = mx.fast.metal_kernel(
        name="mtplx_gguf_qmm_down",
        input_names=["x", "dd", "dq", "tile_start", "row_start"],
        output_names=["y"], header=head, source=_SRC_DOWN, ensure_row_contiguous=True)
    return gu, dn


def available(dtype=None) -> bool:
    """MTPLX_GGUF_QMM (default on): fp16 activations only (the decode writes
    half into the tile)."""
    raw = (os.environ.get("MTPLX_GGUF_QMM") or "1").strip().lower()
    if raw in {"0", "off", "false", "no"}:
        return False
    if dtype is not None and dtype != mx.float16:
        return False
    return _kernels() is not None


def prefill_parts(x2, sw, idx):
    """Token-major [M * topk, D] parts, as moe_gguf_iq.moe_iq_prefill_parts."""
    y, order = prefill_sorted(x2, sw, idx)
    return mx.take(y, mx.argsort(order), axis=0)


def _flag(name: str, default: bool = True) -> bool:
    raw = (os.environ.get(name) or "").strip().lower()
    if raw in {"0", "off", "false", "no"}:
        return False
    if raw in {"1", "on", "true", "yes"}:
        return True
    return default


def fused_epilogue_enabled() -> bool:
    """MTPLX_GGUF_QMM_FUSED_EPILOGUE (default on): the un-sort and the routing
    weighted sum after the expert kernels are one pass."""
    return _flag("MTPLX_GGUF_QMM_FUSED_EPILOGUE")


def fused_mid_enabled() -> bool:
    """MTPLX_GGUF_QMM_FUSED_MID (default on): silu(gate) * up between the two
    expert kernels is one pass over the [P, 2N] gate/up output."""
    return _flag("MTPLX_GGUF_QMM_FUSED_MID")


# out[t, d] = sum_s w[t, s] * y_sorted[inv[t * TOPK + s], d]: the token-major
# gather of the sorted expert outputs and the weighted sum over a token's
# TOPK picks, accumulated in float32 in slot order (mx.take, a float32 [P, D]
# product and a sum over the slot axis, as _gguf_mixed_call wrote it before).
_UNSORT_SRC = r"""
    const uint d = thread_position_in_grid.x;
    const uint t = thread_position_in_grid.y;
    if (d >= (uint)D) return;
    float acc = 0.0f;
    for (uint s = 0; s < (uint)TOPK; ++s) {
        const uint p = t * (uint)TOPK + s;
        acc += w[p] * (float)y[(ulong)inv[p] * (ulong)D + d];
    }
    out[(ulong)t * (ulong)D + d] = (T)acc;
"""

# h[p, j] = silu(gate) * up from the fused gate/up output [P, 2N], with the
# roundings of the expression it replaces, (g * sigmoid(float32(g)).astype(T)) * u:
# MLX's sigmoid form, the sigmoid rounded to T, then each product rounded to T.
_MID_SRC = r"""
    const uint j = thread_position_in_grid.x;
    const uint p = thread_position_in_grid.y;
    if (j >= (uint)NI) return;
    const ulong base = (ulong)p * (ulong)(2 * NI);
    const T g = gu[base + j];
    const float gf = (float)g;
    const float e = 1.0f / (1.0f + metal::exp(metal::abs(gf)));
    const float sig = (gf < 0.0f) ? e : 1.0f - e;
    const T gs = (T)((float)g * (float)((T)sig));
    h[(ulong)p * (ulong)NI + j] = (T)((float)gs * (float)gu[base + (ulong)NI + j]);
"""


@lru_cache(maxsize=1)
def _epilogue_kernels():
    unsort = mx.fast.metal_kernel(
        name="mtplx_gguf_qmm_unsort_wsum", input_names=["y", "inv", "w"], output_names=["out"],
        source=_UNSORT_SRC, ensure_row_contiguous=True)
    mid = mx.fast.metal_kernel(
        name="mtplx_gguf_qmm_silu_up", input_names=["gu"], output_names=["h"],
        source=_MID_SRC, ensure_row_contiguous=True)
    return unsort, mid


def silu_up(gu, ni: int):
    """silu(gu[:, :ni]) * gu[:, ni:] over [P, 2 * ni] in one pass (fp16/bf16/fp32)."""
    _unsort, mid = _epilogue_kernels()
    pairs = int(gu.shape[0])
    (h,) = mid(
        inputs=[gu], template=[("T", gu.dtype), ("NI", int(ni))],
        grid=(((int(ni) + 127) // 128) * 128, pairs, 1), threadgroup=(128, 1, 1),
        output_shapes=[(pairs, int(ni))], output_dtypes=[gu.dtype])
    return h


def prefill_weighted(x2, sw, idx, w):
    """[M, D] routed output: the expert kernels, then the un-sort and the
    routing-weighted sum over each token's picks in one pass. ``w`` is the
    float32 routing weight per pair (token-major, [M * topk])."""
    y, order = prefill_sorted(x2, sw, idx)
    rows = int(x2.shape[0])
    topk = int(idx.size) // rows
    dm = int(sw.hidden)
    inv = mx.argsort(order).astype(mx.uint32)
    unsort, _mid = _epilogue_kernels()
    (out,) = unsort(
        inputs=[y, inv, w.reshape(-1).astype(mx.float32)],
        template=[("T", y.dtype), ("TOPK", topk), ("D", dm)],
        grid=(((dm + 255) // 256) * 256, rows, 1), threadgroup=(256, 1, 1),
        output_shapes=[(rows, dm)], output_dtypes=[y.dtype])
    return out


def _tables(experts_sorted_counts, n_experts: int, bm: int = BM):
    zero = mx.zeros((1,), dtype=mx.int32)
    row_start = mx.concatenate([zero, mx.cumsum(experts_sorted_counts)])
    tile_start = mx.concatenate([zero, mx.cumsum((experts_sorted_counts + bm - 1) // bm)])
    return tile_start, row_start


def prefill_sorted(x2, sw, idx):
    """Routed experts for prefill rows in expert-sorted order.

    x2 [M, K] tokens; idx uint32 [M * topk] (token-major). Returns
    (y_sorted [P, D], order [P]) where y_sorted[r] is the down projection of
    pair order[r] (pair = token * topk + slot)."""
    gu_k, dn_k = _kernels()
    m_rows, kd = int(x2.shape[0]), int(x2.shape[1])
    pairs = int(idx.size)
    topk = pairs // m_rows
    n_exp = int(sw.num_experts) if hasattr(sw, "num_experts") else int(sw.gate_d.shape[0])
    ni = int(sw.n_inter)
    dm = int(sw.hidden)
    flat = idx.reshape(-1).astype(mx.int32)
    order = mx.argsort(flat).astype(mx.uint32)
    counts = mx.zeros((n_exp,), dtype=mx.int32).at[flat].add(1)
    bm, bk, bms = _shape(pairs)
    tile_start, row_start = _tables(counts, n_exp, bm)
    xs = mx.take(x2, order // topk, axis=0)
    ty_g = K.TYPE_CODES[sw.gate_type]
    half8 = _env_int("MTPLX_GGUF_QMM_HALF8", 0)
    if ty_g >= 2:
        gridb = K.half8_tables(sw.gate_type)[0] if half8 else K.grid_u2(sw.gate_type)
    else:
        gridb = mx.zeros((2, 4), mx.uint32)
    n_tiles = pairs // bm + n_exp
    (gu,) = gu_k(
        inputs=[xs, sw.gate_d, sw.gate_q, sw.up_d, sw.up_q, gridb, tile_start, row_start],
        template=[("T", x2.dtype), ("TY", ty_g), ("HALF8", int(bool(half8))), ("KD", kd), ("ND", ni), ("ED", n_exp),
                  ("BM", bm), ("BMS", bms), ("BN", BN), ("BK", bk)],
        grid=((2 * ni // BN) * THREADS, n_tiles, 1),
        threadgroup=(THREADS, 1, 1),
        output_shapes=[(pairs, 2 * ni)],
        output_dtypes=[x2.dtype],
    )
    if fused_mid_enabled():
        h = silu_up(gu, ni)
    else:
        g, u = gu[:, :ni], gu[:, ni:]
        h = (g * mx.sigmoid(g.astype(mx.float32)).astype(g.dtype)) * u
    ty_d = K.TYPE_CODES[sw.down_type]
    (y,) = dn_k(
        inputs=[h, sw.down_d, sw.down_q, tile_start, row_start],
        template=[("T", x2.dtype), ("TY", ty_d), ("KD", ni), ("ND", dm), ("ED", n_exp),
                  ("BM", bm), ("BMS", bms), ("BN", BN), ("BK", bk)],
        grid=((dm // BN) * THREADS, n_tiles, 1),
        threadgroup=(THREADS, 1, 1),
        output_shapes=[(pairs, dm)],
        output_dtypes=[x2.dtype],
    )
    return y, order
