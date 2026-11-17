"""Large-query prefill backend for the Qwen sparse-attention indexer.

The decode-oriented selector in :mod:`qsa_indexer_select` assigns one Metal
threadgroup to one query row and performs the score dot products inside that
threadgroup.  That is an appropriate shape for ``S == 1``, but a large prefill
would make every row load the same pooled-key history independently.

This module follows vLLM's prefill split instead:

1. The production ``B=1,H=4,D=128`` lane uses a Metal 4 TensorOps kernel to
   compute a 16-query by 32-key tile, fuse the per-head ReLU and head reduction,
   and emit only the byte-budgeted float32 ``[C,N]`` score plane.  The kernel
   consumes MLX's runtime-injected strides, so ordinary views do not require a
   hidden contiguous copy. Unsupported shapes and toolchains retain the
   vectorized MLX expression as the oracle.
2. One dedicated Metal threadgroup per score row applies the complete-block
   causal frontier, the load-bearing ``score - block_id * 1e-12`` adjustment,
   exact top-k selection through an adaptive insertion/full-radix hybrid, and
   the requested output epilogue.

There is deliberately no fallback for the exact selector itself.  Only the
score producer has a static production specialization plus the eager-equivalent
MLX fallback.  The Python chunk loop is shape-static and is suitable for
enclosure by the existing ``mx.compile`` graph bank.
"""

from __future__ import annotations

import math
import os
from functools import lru_cache
from typing import Literal

import mlx.core as mx

from mtplx.kernels.qsa_indexer_select import (
    _HEADER as _SELECT_HEADER,
)
from mtplx.kernels.qsa_indexer_select import (
    _as_i32_scalar,
    _epilogue,
    _next_power_of_two,
    _require_metal,
    _static_int,
    qsa_indexer_select_nax_available,
)

QSAPrefillMode = Literal["blocks", "dense_mask", "row_tokens"]
QSAPrefillScoreProducer = Literal["mpp", "mlx", "simd"]

_SUPPORTED_DTYPES = (mx.float16, mx.bfloat16, mx.float32)
_MAX_TOPK = 512
_MAX_WIDTH = 1024
_MIN_SELECTOR_WIDTH = 256
_RADIX_BINS = 2048
_FINAL_CANDIDATES = 2048
_INSERTION_CANDIDATES = 64

# This bounds score-stage planes, not every live allocation in an enclosing
# graph. At N=65,536 the fused MPP producer fits 512 rows; the general H=4 MLX
# producer fits approximately 102 rows (rounded down to 96 for GEMM tiling).
DEFAULT_PREFILL_SCORE_WORKSPACE_BYTES = 128 * 1024 * 1024
PREFILL_ROW_ALIGNMENT = 32

__all__ = [
    "DEFAULT_PREFILL_SCORE_WORKSPACE_BYTES",
    "PREFILL_ROW_ALIGNMENT",
    "qsa_indexer_prefill_blocks_metal",
    "qsa_indexer_prefill_chunk_rows",
    "qsa_indexer_prefill_dense_mask_metal",
    "qsa_indexer_prefill_metal",
    "qsa_indexer_prefill_prepared_scores_mpp_supported",
    "qsa_indexer_prefill_row_tokens_metal",
    "qsa_indexer_prefill_score_chunk_rows",
    "qsa_indexer_prefill_scores",
    "qsa_indexer_prefill_scores_mpp",
    "qsa_indexer_prefill_scores_mpp_supported",
    "qsa_indexer_prefill_topk_metal",
]


def qsa_indexer_prefill_chunk_rows(
    rows: int,
    heads: int,
    backing_blocks: int,
    workspace_bytes: int = DEFAULT_PREFILL_SCORE_WORKSPACE_BYTES,
    *,
    row_alignment: int = PREFILL_ROW_ALIGNMENT,
) -> int:
    """Return the vectorized-MLX chunk width under the score-plane budget.

    This compatibility wrapper describes the general fallback, whose live
    score tensors are the ``H`` per-head planes plus one reduced plane.  New
    dispatch code should call :func:`qsa_indexer_prefill_score_chunk_rows`
    with the statically selected producer so the fused MPP lane is not charged
    for intermediates it never materializes.
    """

    return qsa_indexer_prefill_score_chunk_rows(
        rows,
        heads,
        backing_blocks,
        workspace_bytes,
        producer="mlx",
        row_alignment=row_alignment,
    )


def qsa_indexer_prefill_score_chunk_rows(
    rows: int,
    heads: int,
    backing_blocks: int,
    workspace_bytes: int = DEFAULT_PREFILL_SCORE_WORKSPACE_BYTES,
    *,
    producer: QSAPrefillScoreProducer,
    row_alignment: int = PREFILL_ROW_ALIGNMENT,
) -> int:
    """Return a producer-aware static score-chunk width.

    The vectorized expression has a float32 ``[1,C,H,N]`` per-head logits plane
    and a float32 ``[C,N]`` reduced plane.  Counting ``H + 1`` planes is a
    conservative bound for those dominant intermediates.  The production MPP
    kernel emits only the reduced float32 ``[C,N]`` plane, so it is charged one
    plane.  A selected MPP failure is intentionally not allowed to fall back;
    this byte contract therefore cannot under-budget a hidden fallback graph.

    The returned width is rounded *down* for large chunks so alignment can
    never violate the requested byte ceiling.  A single irreducible row is
    allowed to exceed a smaller caller-provided budget.
    """

    query_rows = int(rows)
    query_heads = int(heads)
    blocks = int(backing_blocks)
    budget = int(workspace_bytes)
    alignment = int(row_alignment)
    if producer not in ("mpp", "mlx", "simd"):
        raise ValueError(f"unknown score producer {producer!r}")
    if query_rows <= 0:
        raise ValueError(f"rows must be positive, got {query_rows}")
    if query_heads <= 0:
        raise ValueError(f"heads must be positive, got {query_heads}")
    if blocks <= 0:
        raise ValueError(f"backing_blocks must be positive, got {blocks}")
    if budget <= 0:
        raise ValueError(f"workspace_bytes must be positive, got {budget}")
    if alignment <= 0:
        raise ValueError(f"row_alignment must be positive, got {alignment}")

    score_planes = 1 if producer in ("mpp", "simd") else query_heads + 1
    bytes_per_row = blocks * 4 * score_planes
    chunk = min(query_rows, max(1, budget // bytes_per_row))
    if chunk >= alignment and chunk < query_rows:
        chunk = max(alignment, (chunk // alignment) * alignment)
    return chunk


def qsa_indexer_prefill_scores(
    q_chunk: mx.array,
    pooled_t_float32: mx.array,
    *,
    head_dim: int,
) -> mx.array:
    """Return oracle/fallback float32 indexer scores ``[C,N]`` for one chunk.

    Keep this expression structurally identical to ``QSAIndexer._select_eager``.
    In particular, both operands are float32 before matmul and ReLU is applied
    independently to every head before the head reduction.  The batched matmul
    is what gives a large-S prefill MLX's tiled/NAX cross-query key reuse.
    """

    if q_chunk.ndim != 4 or int(q_chunk.shape[0]) != 1:
        raise ValueError(f"q_chunk must have shape [1,C,H,D], got {q_chunk.shape}")
    dim = int(head_dim)
    if dim <= 0 or int(q_chunk.shape[3]) != dim:
        raise ValueError(
            f"head_dim must match q_chunk.shape[3], got {dim} and {q_chunk.shape[3]}"
        )
    if (
        pooled_t_float32.ndim != 4
        or int(pooled_t_float32.shape[0]) != 1
        or int(pooled_t_float32.shape[1]) != 1
        or int(pooled_t_float32.shape[2]) != dim
        or pooled_t_float32.dtype != mx.float32
    ):
        raise ValueError(
            "pooled_t_float32 must have shape [1,1,D,N] and dtype float32; "
            f"got {pooled_t_float32.shape} and {pooled_t_float32.dtype}"
        )

    per_head = mx.matmul(q_chunk.astype(mx.float32), pooled_t_float32)
    scores = mx.maximum(per_head, 0.0).sum(axis=2) / math.sqrt(dim)
    return scores[0]


# ---------------------------------------------------------------------------
# Fused simdgroup-matrix score kernel (Apple7+, no tensor ops needed).
#
# ``out[c, n] = sum_h relu(q[c, h, :] . pooled[n, :]) / sqrt(D)`` in one pass:
# the per-head float32 planes of the MLX expression never reach memory.  q and
# pooled are float16, so every product is exact in float32 and the accumulation
# differs from the float32 matmul only in summation order (max |diff| ~1e-7 on
# scores of order 1; the selected top-k sets are identical in the receipts).
# Eight query rows by NF*8 keys per simdgroup; heads are four fragments with
# the same element mapping, so relu + head sum happen in registers.
# ---------------------------------------------------------------------------

_SIMD_SCORE_NF = 4
_SIMD_SCORE_SGS = 4
_SIMD_SCORE_SOURCE = """
    uint sg = simdgroup_index_in_threadgroup;
    uint col0 = (threadgroup_position_in_grid.x * SGS + sg) * (NF * 8);
    uint row0 = threadgroup_position_in_grid.y * 8;
    uint n = (uint)p_shape[0];
    if (col0 >= n) return;
    simdgroup_float8x8 acc[H_][NF];
    #pragma clang loop unroll(full)
    for (int h = 0; h < H_; ++h) {
        #pragma clang loop unroll(full)
        for (int j = 0; j < NF; ++j) acc[h][j] = simdgroup_float8x8(0.0f);
    }
    for (int k = 0; k < D_; k += 8) {
        simdgroup_half8x8 a[H_];
        #pragma clang loop unroll(full)
        for (int h = 0; h < H_; ++h)
            simdgroup_load(a[h], q + ((ulong)row0 * H_ + h) * D_ + k, (ulong)H_ * D_);
        #pragma clang loop unroll(full)
        for (int j = 0; j < NF; ++j) {
            if (col0 + (uint)j * 8 < n) {
                simdgroup_half8x8 b;
                simdgroup_load(b, p + (ulong)(col0 + j * 8) * D_ + k, (ulong)D_, ulong2(0, 0), true);
                #pragma clang loop unroll(full)
                for (int h = 0; h < H_; ++h) simdgroup_multiply_accumulate(acc[h][j], a[h], b, acc[h][j]);
            }
        }
    }
    #pragma clang loop unroll(full)
    for (int j = 0; j < NF; ++j) {
        if (col0 + (uint)j * 8 < n) {
            float v0 = 0.0f, v1 = 0.0f;
            #pragma clang loop unroll(full)
            for (int h = 0; h < H_; ++h) {
                auto e = acc[h][j].thread_elements();
                v0 += metal::max(e[0], 0.0f);
                v1 += metal::max(e[1], 0.0f);
            }
            simdgroup_float8x8 o = simdgroup_float8x8(0.0f);
            o.thread_elements()[0] = v0 / 11.313708498984761f;
            o.thread_elements()[1] = v1 / 11.313708498984761f;
            simdgroup_store(o, out + (ulong)row0 * n + col0 + j * 8, (ulong)n);
        }
    }
"""


def _simd_scores_enabled() -> bool:
    raw = (os.environ.get("MTPLX_QSA_PREFILL_SIMD_SCORES") or "1").strip().lower()
    return raw not in {"0", "false", "off", "no"}


def qsa_indexer_prefill_scores_simd_supported(q_chunk: mx.array, pooled: mx.array) -> bool:
    """float16 [1,C,4,128] x [1,N,128] with N a multiple of 8 on a Metal device."""

    if not _simd_scores_enabled() or not hasattr(mx, "metal") or not mx.metal.is_available():
        return False
    if q_chunk.dtype != mx.float16 or pooled.dtype != mx.float16:
        return False
    if q_chunk.ndim != 4 or int(q_chunk.shape[0]) != 1 or int(q_chunk.shape[2]) != 4:
        return False
    if int(q_chunk.shape[3]) != 128 or pooled.ndim != 3 or int(pooled.shape[0]) != 1:
        return False
    blocks = int(pooled.shape[1])
    return int(pooled.shape[2]) == 128 and blocks > 0 and blocks % 8 == 0


@lru_cache(maxsize=1)
def _prefill_simd_score_kernel():
    return mx.fast.metal_kernel(
        name="mtplx_qsa_prefill_simd_scores_h4d128",
        input_names=["q", "p"],
        output_names=["out"],
        source=_SIMD_SCORE_SOURCE,
    )


def qsa_indexer_prefill_scores_simd(q_chunk: mx.array, pooled: mx.array) -> mx.array:
    """Fused float32 scores ``[C,N]`` for a supported ``q_chunk``/``pooled`` pair."""

    if not qsa_indexer_prefill_scores_simd_supported(q_chunk, pooled):
        raise ValueError("QSA simd prefill scores called for an unsupported signature")
    rows = int(q_chunk.shape[1])
    blocks = int(pooled.shape[1])
    padded = (rows + 7) // 8 * 8
    q = q_chunk[0]
    if padded != rows:
        q = mx.concatenate([q, mx.zeros((padded - rows, 4, 128), dtype=q.dtype)], axis=0)
    span = _SIMD_SCORE_NF * 8 * _SIMD_SCORE_SGS
    (scores,) = _prefill_simd_score_kernel()(
        inputs=[q, pooled[0]],
        template=[("H_", 4), ("D_", 128), ("NF", _SIMD_SCORE_NF), ("SGS", _SIMD_SCORE_SGS)],
        grid=(((blocks + span - 1) // span) * 32 * _SIMD_SCORE_SGS, padded // 8, 1),
        threadgroup=(32 * _SIMD_SCORE_SGS, 1, 1),
        output_shapes=[(padded, blocks)],
        output_dtypes=[mx.float32],
    )
    return scores if padded == rows else scores[:rows]



def _mpp_score_geometry_supported(
    q_shape: tuple[int, ...],
    pooled_shape: tuple[int, ...],
) -> bool:
    """Pure static geometry gate for the production score tile.

    The MPP tile treats four QSA heads for four query rows as its sixteen M
    rows.  MLX 0.32 does not expose an array ``strides`` or contiguity property
    to Python.  Custom Metal kernels do receive each input's real C++ strides,
    however, and this kernel gathers source fragments through those injected
    strides.  Layout is therefore a runtime kernel concern rather than a host
    eligibility guess.
    """

    if len(q_shape) != 4 or len(pooled_shape) != 3:
        return False
    if q_shape[0] != 1 or q_shape[1] <= 0 or q_shape[2:] != (4, 128):
        return False
    return pooled_shape[0] == 1 and pooled_shape[1] > 0 and pooled_shape[2] == 128


def qsa_indexer_prefill_scores_mpp_supported(
    q_chunk: mx.array,
    pooled: mx.array,
) -> bool:
    """Whether the exact production MPP score signature is statically safe."""

    return _mpp_score_signature_supported(
        tuple(int(dim) for dim in q_chunk.shape),
        tuple(int(dim) for dim in pooled.shape),
        q_chunk.dtype,
        pooled.dtype,
    )


def _mpp_score_signature_supported(
    q_shape: tuple[int, ...],
    pooled_shape: tuple[int, ...],
    q_dtype: mx.Dtype,
    pooled_dtype: mx.Dtype,
) -> bool:
    """Shared host/device gate for actual and compile-predicted Q outputs.

    Deliberately do not probe Python-side layout metadata here: it is absent
    from the public MLX array API.  The selected kernel is stride-general and
    fail-closed at dispatch, so this receipt remains truthful for both dense
    arrays and views without inserting a copy.
    """

    return (
        mx.metal.is_available()
        and mx.default_device() == mx.gpu
        and qsa_indexer_select_nax_available()
        and q_dtype in (mx.float16, mx.bfloat16)
        and pooled_dtype == q_dtype
        and _mpp_score_geometry_supported(
            q_shape,
            pooled_shape,
        )
    )


def qsa_indexer_prefill_prepared_scores_mpp_supported(
    rows: int,
    heads: int,
    head_dim: int,
    q_dtype: mx.Dtype,
    pooled: mx.array,
) -> bool:
    """Predict the MPP lane for the contiguous prepared-Q kernel output.

    ``qsa_indexer_prepare_queries_metal`` creates a dense output with the same
    ``[1,S,H,D]`` shape and dtype as its input.  The compiled graph bank uses
    this helper before tracing so its producer receipt and chunk width are the
    same static decision the prefill backend will make for that output.
    """

    query_rows = int(rows)
    query_heads = int(heads)
    query_dim = int(head_dim)
    q_shape = (1, query_rows, query_heads, query_dim)
    return _mpp_score_signature_supported(
        q_shape,
        tuple(int(dim) for dim in pooled.shape),
        q_dtype,
        pooled.dtype,
    )


_MPP_SCORE_HEADER = r"""
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;

constant constexpr uint QSA_SCORE_HEADS = 4;
constant constexpr uint QSA_SCORE_HEAD_DIM = 128;
constant constexpr uint QSA_SCORE_QUERY_TILE = 16;
constant constexpr uint QSA_SCORE_KEY_TILE = 32;
constant constexpr uint QSA_SCORE_QUERIES_PER_SIMDGROUP = 4;
constant constexpr uint QSA_SCORE_SIMDGROUPS = 4;
constant constexpr uint QSA_SCORE_THREADS = 128;
constant constexpr uint QSA_SCORE_K_FRAGMENTS = 8;
constant constexpr float QSA_SCORE_SQRT_HEAD_DIM = 11.313708498984761f;

// Metal 4's 16x32x16 cooperative fragment maps each lane to two rows eight
// apart and four adjacent columns. This is the same proven map used by MLX's
// NAX tiles and mtplx/kernels/sdpa_nax_tile.py.
inline short2 qsa_score_nax_coord(ushort lane) {
    const short qid = short(lane >> 2);
    const short fragment_row =
        ((qid & 4) | ((short(lane) >> 1) & 3));
    const short fragment_col =
        ((qid & 2) | (short(lane) & 1)) * 4;
    return short2{fragment_col, fragment_row};
}
"""


_MPP_SCORE_SOURCE = r"""
    const uint tid = thread_position_in_threadgroup.x;
    const uint simdgroup = tid >> 5;
    const ushort lane = ushort(tid & 31u);
    const uint rows = q_shape[1];
    const uint blocks = pooled_shape[1];
    const uint key_tiles = (blocks + QSA_SCORE_KEY_TILE - 1u) /
        QSA_SCORE_KEY_TILE;
    const uint tile = threadgroup_position_in_grid.x;
    const uint query_tile = tile / key_tiles;
    const uint key_tile = tile - query_tile * key_tiles;
    const uint query0 =
        query_tile * QSA_SCORE_QUERY_TILE +
        simdgroup * QSA_SCORE_QUERIES_PER_SIMDGROUP;
    const uint block0 = key_tile * QSA_SCORE_KEY_TILE;

    // Four simdgroups share one [32,128] pooled-key tile (8 KiB for fp16 or
    // bf16). Each key is therefore fetched once per sixteen output query rows,
    // rather than once per head or per query row.
    threadgroup InT pooled_tile[QSA_SCORE_KEY_TILE * QSA_SCORE_HEAD_DIM];
    constexpr uint VECTORS_PER_KEY = QSA_SCORE_HEAD_DIM / 4u;
    constexpr uint TILE_VECTORS = QSA_SCORE_KEY_TILE * VECTORS_PER_KEY;
    for (uint item = tid; item < TILE_VECTORS; item += QSA_SCORE_THREADS) {
        const uint key_local = item / VECTORS_PER_KEY;
        const uint dim4 = item - key_local * VECTORS_PER_KEY;
        const uint block = block0 + key_local;
        vec<InT, 4> values = vec<InT, 4>(InT(0));
        if (block < blocks) {
            const int64_t source =
                int64_t(block) * pooled_strides[1] +
                int64_t(dim4 * 4u) * pooled_strides[2];
            // Source layout is intentionally not guessed on the host.  MLX
            // injects the real strides for this invocation; scalar gathers
            // also avoid imposing an unobservable vec4 base-alignment
            // contract on sliced/as_strided views.  Contiguous lanes remain
            // adjacent and can still be coalesced by the compiler/hardware.
            for (uint elem = 0u; elem < 4u; ++elem) {
                values[elem] = pooled[
                    source + int64_t(elem) * pooled_strides[2]];
            }
        }
        const uint destination =
            key_local * QSA_SCORE_HEAD_DIM + dim4 * 4u;
        for (uint elem = 0u; elem < 4u; ++elem) {
            pooled_tile[destination + elem] = values[elem];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    constexpr auto descriptor = mpp::tensor_ops::matmul2d_descriptor(
        16, 32, 16, false, true, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<descriptor, metal::execution_simdgroup> matmul;
    auto left = matmul.get_left_input_cooperative_tensor<InT, InT, float>();
    auto right = matmul.get_right_input_cooperative_tensor<InT, InT, float>();
    // macOS 27 MPP SDK rejects address-space-qualified template operands
    // (issue #404); strip them like mlx's steel/gemm/nax.h does.
    auto accumulator = matmul.get_destination_cooperative_tensor<
        metal::remove_addrspace_t<decltype(left)>,
        metal::remove_addrspace_t<decltype(right)>, float>();

    constexpr short ELEMENTS_PER_FRAGMENT = 8;
    constexpr short ELEMENT_COLUMNS = 4;
    constexpr short ELEMENT_ROW_JUMP = 8;
    const short2 coordinate = qsa_score_nax_coord(lane);
    for (short item = 0; item < 2 * ELEMENTS_PER_FRAGMENT; ++item) {
        accumulator[item] = 0.0f;
    }

    // A's sixteen rows are head-major [h0:q0..q3, h1:q0..q3, ...].
    // This layout makes lane^16 pair the h0/h2 carrier with h1/h3 for the
    // same query and output columns after the TensorOp completes.
    for (uint k_frag = 0u;
         k_frag < QSA_SCORE_K_FRAGMENTS;
         ++k_frag) {
        for (short row_part = 0; row_part < 2; ++row_part) {
            const uint matrix_row =
                uint(coordinate.y + row_part * ELEMENT_ROW_JUMP);
            const uint head = matrix_row / QSA_SCORE_QUERIES_PER_SIMDGROUP;
            const uint query_local =
                matrix_row - head * QSA_SCORE_QUERIES_PER_SIMDGROUP;
            const uint query = query0 + query_local;
            vec<InT, 4> values = vec<InT, 4>(InT(0));
            if (query < rows) {
                const int64_t source =
                    int64_t(query) * q_strides[1] +
                    int64_t(head) * q_strides[2] +
                    int64_t(k_frag * 16u + uint(coordinate.x)) *
                        q_strides[3];
                for (uint elem = 0u; elem < 4u; ++elem) {
                    values[elem] = q[
                        source + int64_t(elem) * q_strides[3]];
                }
            }
            for (short elem = 0; elem < ELEMENT_COLUMNS; ++elem) {
                left[row_part * ELEMENT_COLUMNS + elem] = values[elem];
            }
        }

        for (short key_half = 0; key_half < 2; ++key_half) {
            for (short row_part = 0; row_part < 2; ++row_part) {
                const uint key_local = uint(
                    key_half * 16 + coordinate.y +
                    row_part * ELEMENT_ROW_JUMP);
                const uint source =
                    key_local * QSA_SCORE_HEAD_DIM +
                    k_frag * 16u + uint(coordinate.x);
                const threadgroup vec<InT, 4>* source4 =
                    reinterpret_cast<const threadgroup vec<InT, 4>*>(
                        pooled_tile + source);
                const vec<InT, 4> values = source4[0];
                for (short elem = 0; elem < ELEMENT_COLUMNS; ++elem) {
                    right[
                        key_half * ELEMENTS_PER_FRAGMENT +
                        row_part * ELEMENT_COLUMNS + elem] = values[elem];
                }
            }
        }
        matmul.run(left, right, accumulator);
    }

    // For a lane with bit 4 clear, row_part 0 carries h0 and row_part 1
    // carries h2. lane^16 carries h1 and h3 for exactly the same query and
    // four output columns. Apply ReLU before the strictly h0,h1,h2,h3-ordered
    // float32 sum, matching the eager expression's reduction contract.
    for (short key_half = 0; key_half < 2; ++key_half) {
        for (short elem = 0; elem < ELEMENT_COLUMNS; ++elem) {
            const float head0_or_1 = metal::max(
                accumulator[key_half * ELEMENTS_PER_FRAGMENT + elem], 0.0f);
            const float head2_or_3 = metal::max(
                accumulator[
                    key_half * ELEMENTS_PER_FRAGMENT +
                    ELEMENT_COLUMNS + elem],
                0.0f);
            const float paired_head1_or_0 =
                simd_shuffle_xor(head0_or_1, ushort(16));
            const float paired_head3_or_2 =
                simd_shuffle_xor(head2_or_3, ushort(16));
            if ((lane & 16u) == 0u) {
                const uint query = query0 + uint(coordinate.y);
                const uint block =
                    block0 + uint(key_half * 16 + coordinate.x + elem);
                if (query < rows && block < blocks) {
                    const float head_sum =
                        ((head0_or_1 + paired_head1_or_0) + head2_or_3) +
                        paired_head3_or_2;
                    scores[size_t(query) * blocks + block] =
                        head_sum / QSA_SCORE_SQRT_HEAD_DIM;
                }
            }
        }
    }
"""


@lru_cache(maxsize=1)
def _prefill_mpp_score_kernel():
    return mx.fast.metal_kernel(
        name="mtplx_qsa_prefill_mpp_scores_h4d128",
        input_names=["q", "pooled"],
        output_names=["scores"],
        header=_MPP_SCORE_HEADER,
        source=_MPP_SCORE_SOURCE,
        # Both input stride arrays are consumed explicitly.  A hidden
        # contiguous-copy dispatch would defeat graph accounting and key reuse.
        ensure_row_contiguous=False,
    )


def qsa_indexer_prefill_scores_mpp(
    q_chunk: mx.array,
    pooled: mx.array,
) -> mx.array:
    """Emit fused production scores for the exact supported signature.

    Unsupported signatures are a caller routing decision.  Once this helper
    is selected, construction and dispatch errors propagate; silently taking
    the oracle would invalidate path and performance receipts.
    """

    if not qsa_indexer_prefill_scores_mpp_supported(q_chunk, pooled):
        raise ValueError("QSA MPP prefill scores called for an unsupported signature")
    kernel = _prefill_mpp_score_kernel()
    rows = int(q_chunk.shape[1])
    blocks = int(pooled.shape[1])
    query_tiles = (rows + 15) // 16
    key_tiles = (blocks + 31) // 32
    (scores,) = kernel(
        inputs=[q_chunk, pooled],
        template=[("InT", q_chunk.dtype)],
        grid=(query_tiles * key_tiles * 128, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(rows, blocks)],
        output_dtypes=[mx.float32],
    )
    return scores


def _validate_topk(
    scores: mx.array,
    *,
    pos_start: int | mx.array,
    total_tokens: int | mx.array,
    logical_blocks: int | mx.array | None,
    block_topk: int,
    compress_ratio: int,
) -> tuple[int, int, int, int]:
    _require_metal()
    if scores.ndim != 2 or scores.dtype != mx.float32:
        raise TypeError(
            f"scores must be a float32 [rows,blocks] array, got {scores.shape} "
            f"and {scores.dtype}"
        )
    rows, blocks = map(int, scores.shape)
    if rows <= 0 or blocks <= 0:
        raise ValueError(f"score dimensions must be positive, got {scores.shape}")

    topk = int(block_topk)
    ratio = int(compress_ratio)
    if not 1 <= topk <= _MAX_TOPK:
        raise ValueError(f"block_topk must be in [1,{_MAX_TOPK}], got {topk}")
    if ratio <= 0:
        raise ValueError(f"compress_ratio must be positive, got {ratio}")

    start = _static_int(pos_start)
    total = _static_int(total_tokens)
    logical = blocks if logical_blocks is None else _static_int(logical_blocks)
    if start is not None and start < 0:
        raise ValueError(f"pos_start must be non-negative, got {start}")
    if total is not None and total < 0:
        raise ValueError(f"total_tokens must be non-negative, got {total}")
    if start is not None and total is not None and total < start + rows:
        raise ValueError(
            "total_tokens must include every query row: "
            f"got total_tokens={total}, pos_start={start}, rows={rows}"
        )
    if logical is not None:
        if not 0 <= logical <= blocks:
            raise ValueError(f"logical_blocks must be in [0,{blocks}], got {logical}")
        if total is not None and logical != total // ratio:
            raise ValueError(
                "logical_blocks must equal total_tokens // compress_ratio: "
                f"got logical_blocks={logical}, total={total}, ratio={ratio}"
            )

    width = _next_power_of_two(max(_MIN_SELECTOR_WIDTH, topk))
    if width > _MAX_WIDTH:
        raise ValueError(f"selector width {width} exceeds Metal guard {_MAX_WIDTH}")
    return rows, blocks, topk, width


#: The score plane's width (pooled blocks, context // ratio) is different for
#: every prefill chunk of every prompt.  Baked into the kernel source it made
#: every chunk compile a fresh Metal pipeline: 40 ms for the simd selector and
#: 56 ms for the network one on an M5 Max (2026-09-20, tiny score planes, so
#: all of it is the compiler), with the GPU idle behind the encoder.  A 128K
#: cold prompt at 2,048-row chunks paid it 56 times and every warm agent turn
#: past the sparse crossover paid it on its first forward.  The kernels use
#: the width in two places only, the row stride and a clamp, so it is read at
#: run time from the score plane's own shape: same integer arithmetic, same
#: outputs.
_RUNTIME_BLOCKS_PREAMBLE = """
        const uint BACKING_BLOCKS = uint(scores_shape[1]);
"""


def _topk_static_blocks() -> bool:
    """``MTPLX_QSA_PREFILL_TOPK_STATIC_BLOCKS=1``: bake the block count into
    the selector source again (one compile per distinct count; rollback)."""

    raw = (os.environ.get("MTPLX_QSA_PREFILL_TOPK_STATIC_BLOCKS") or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _blocks_constant(static_blocks: int) -> str:
    if static_blocks > 0:
        return f"constant constexpr uint BACKING_BLOCKS = {int(static_blocks)};"
    return "// BACKING_BLOCKS is read from scores_shape at run time."


@lru_cache(maxsize=128)
def _prefill_topk_kernel(
    mode: QSAPrefillMode,
    topk: int,
    ratio: int,
    width: int,
    output_tokens: int,
    static_blocks: int = 0,
):
    output_names, epilogue = _epilogue(mode)
    header = (
        _SELECT_HEADER
        + f"""
{_blocks_constant(static_blocks)}
constant constexpr uint TOP_K = {topk};
constant constexpr uint RATIO = {ratio};
constant constexpr uint WIDTH = {width};
constant constexpr uint RADIX_BINS = {_RADIX_BINS};
constant constexpr uint FINAL_CANDIDATES = {_FINAL_CANDIDATES};
constant constexpr uint INSERTION_CANDIDATES = {_INSERTION_CANDIDATES};
constant constexpr uint RADIX_PASSES = 6;
constant constexpr uint OUTPUT_TOKENS = {output_tokens};
constant constexpr bool ROW_TOKEN_MODE = {str(mode == "row_tokens").lower()};

inline uint qsa_prefill_radix_shift(uint pass) {{
    return pass == 0u ? 53u
        : pass == 1u ? 42u
        : pass == 2u ? 31u
        : pass == 3u ? 20u
        : pass == 4u ? 9u
        : 0u;
}}

inline uint qsa_prefill_radix_bits(uint pass) {{
    return pass < 5u ? 11u : 9u;
}}
"""
    )

    source = (
        ("" if static_blocks > 0 else _RUNTIME_BLOCKS_PREAMBLE)
        + r"""
        const uint row = threadgroup_position_in_grid.x;
        const uint lane = thread_position_in_threadgroup.x;
        const int qpos = pos_start[0] + int(row);
        const int total = total_tokens[0];
        const int logical_value = logical_blocks[0];
        const uint logical = logical_value > 0
            ? metal::min(uint(logical_value), BACKING_BLOCKS)
            : 0u;
        const int complete_value = (qpos + 1) / int(RATIO);
        const uint complete = complete_value > 0 ? uint(complete_value) : 0u;
        const uint valid_count = metal::min(logical, complete);
        const uint k_eff = metal::min(TOP_K, valid_count);
        const size_t score_base = (size_t)row * BACKING_BLOCKS;

        threadgroup float exchange_scores[WIDTH];
        threadgroup uint exchange_indices[WIDTH];
        threadgroup uchar exchange_valid[WIDTH];
        threadgroup atomic_uint radix_histogram[RADIX_BINS];
        threadgroup ulong final_candidate_keys[FINAL_CANDIDATES];
        threadgroup atomic_uint selected_count;
        threadgroup atomic_uint final_candidate_count;
        threadgroup uint final_candidate_total;
        threadgroup ulong radix_prefix;
        threadgroup uint radix_rank;
        threadgroup uint radix_done;
        threadgroup uint radix_final_shift;
        threadgroup uint radix_next_pass;
        threadgroup ulong threshold_key;

        // Find a <=2048-item bucket containing the Kth-largest strict
        // (adjusted-float, block-id) key. Six high-to-low radix digits cover
        // all 64 bits as 11/11/11/11/11/9. Usually the first 11-bit pass is
        // already selective enough; unlike the decode selector, prefill never
        // performs eight unconditional full-row byte scans.
        if (lane == 0) {
            radix_prefix = 0ul;
            radix_rank = k_eff > 0 ? k_eff - 1 : 0;
            radix_done = k_eff > 0 ? 0u : 1u;
            radix_final_shift = 0u;
            radix_next_pass = 0u;
            threshold_key = 0xfffffffffffffffful;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (k_eff > 0) {
            for (uint pass = 0; pass < RADIX_PASSES; ++pass) {
                if (radix_done == 0u) {
                    for (uint bin = lane; bin < RADIX_BINS; bin += WIDTH) {
                        atomic_store_explicit(
                            &radix_histogram[bin], 0u, memory_order_relaxed);
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);

                    const uint shift = qsa_prefill_radix_shift(pass);
                    const uint bits = qsa_prefill_radix_bits(pass);
                    const uint digit_mask = (1u << bits) - 1u;
                    const ulong prefix = radix_prefix;
                    for (uint block = lane; block < valid_count; block += WIDTH) {
                        const float adjusted =
                            scores[score_base + block] - float(block) * 1.0e-12f;
                        const ulong key = qsa_composite_key(adjusted, block);
                        const bool prefix_matches = pass == 0
                            ? true
                            : (key >> (shift + bits)) == prefix;
                        if (prefix_matches) {
                            const uint digit = uint((key >> shift) & digit_mask);
                            atomic_fetch_add_explicit(
                                &radix_histogram[digit], 1u,
                                memory_order_relaxed);
                        }
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);

                    if (lane == 0) {
                        uint rank = radix_rank;
                        uint chosen = 0;
                        uint chosen_count = 0;
                        const int max_digit = int(digit_mask);
                        for (int digit = max_digit; digit >= 0; --digit) {
                            const uint count = atomic_load_explicit(
                                &radix_histogram[uint(digit)],
                                memory_order_relaxed);
                            if (rank < count) {
                                chosen = uint(digit);
                                chosen_count = count;
                                break;
                            }
                            rank -= count;
                        }
                        radix_prefix = (radix_prefix << bits) | ulong(chosen);
                        radix_rank = rank;
                        radix_final_shift = shift;
                        radix_next_pass = pass + 1u;
                        if (chosen_count <= FINAL_CANDIDATES) {
                            radix_done = 1u;
                        }
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
                // All lanes observe radix_done uniformly before the next pass.
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            // Collect only the final threshold bucket. The adaptive descent
            // guarantees that its cardinality is at most FINAL_CANDIDATES.
            if (lane == 0) {
                atomic_store_explicit(
                    &final_candidate_count, 0u, memory_order_relaxed);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            const ulong final_prefix = radix_prefix;
            const uint final_shift = radix_final_shift;
            for (uint block = lane; block < valid_count; block += WIDTH) {
                const float adjusted =
                    scores[score_base + block] - float(block) * 1.0e-12f;
                const ulong key = qsa_composite_key(adjusted, block);
                if ((key >> final_shift) == final_prefix) {
                    const uint slot = atomic_fetch_add_explicit(
                        &final_candidate_count, 1u, memory_order_relaxed);
                    if (slot < FINAL_CANDIDATES) {
                        final_candidate_keys[slot] = key;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (lane == 0) {
                final_candidate_total = atomic_load_explicit(
                    &final_candidate_count, memory_order_relaxed);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // vLLM's prefill selector uses insertion for the common small-row
            // regime and radix work for the long-row regime. Make the same
            // choice from the actual threshold-bucket cardinality: insertion
            // is capped at 64 candidates (at most 4096 key comparisons), while
            // a larger bucket finishes the remaining 64-bit radix digits over
            // at most 2048 resident keys. This avoids both an O(2048^2) tail
            // and a padded 2048-key bitonic network's 66 barriers.
            const uint candidate_count = final_candidate_total;
            if (candidate_count <= INSERTION_CANDIDATES) {
                // Composite keys are strict because block_id occupies their
                // low 32 bits, so exactly one candidate has this rank.
                for (uint item = lane; item < candidate_count; item += WIDTH) {
                    const ulong key = final_candidate_keys[item];
                    uint greater_rank = 0u;
                    for (uint other = 0u; other < candidate_count; ++other) {
                        greater_rank +=
                            final_candidate_keys[other] > key ? 1u : 0u;
                    }
                    if (greater_rank == radix_rank) {
                        threshold_key = key;
                    }
                }
            } else {
                // The first phase has already consumed [0, radix_next_pass)
                // and all resident candidates share radix_prefix. Complete
                // every remaining digit using the same 2048-bin histogram,
                // but scan only the bounded candidate array, never the full
                // score row. After pass 5, radix_prefix is the exact unique
                // 64-bit threshold key.
                const uint first_refine_pass = radix_next_pass;
                for (uint pass = 0u; pass < RADIX_PASSES; ++pass) {
                    if (pass >= first_refine_pass) {
                        for (uint bin = lane; bin < RADIX_BINS; bin += WIDTH) {
                            atomic_store_explicit(
                                &radix_histogram[bin], 0u,
                                memory_order_relaxed);
                        }
                        threadgroup_barrier(mem_flags::mem_threadgroup);

                        const uint shift = qsa_prefill_radix_shift(pass);
                        const uint bits = qsa_prefill_radix_bits(pass);
                        const uint digit_mask = (1u << bits) - 1u;
                        const ulong prefix = radix_prefix;
                        for (uint item = lane;
                             item < candidate_count;
                             item += WIDTH) {
                            const ulong key = final_candidate_keys[item];
                            if ((key >> (shift + bits)) == prefix) {
                                const uint digit =
                                    uint((key >> shift) & digit_mask);
                                atomic_fetch_add_explicit(
                                    &radix_histogram[digit], 1u,
                                    memory_order_relaxed);
                            }
                        }
                        threadgroup_barrier(mem_flags::mem_threadgroup);

                        if (lane == 0) {
                            uint rank = radix_rank;
                            uint chosen = 0u;
                            const int max_digit = int(digit_mask);
                            for (int digit = max_digit; digit >= 0; --digit) {
                                const uint count = atomic_load_explicit(
                                    &radix_histogram[uint(digit)],
                                    memory_order_relaxed);
                                if (rank < count) {
                                    chosen = uint(digit);
                                    break;
                                }
                                rank -= count;
                            }
                            radix_prefix =
                                (radix_prefix << bits) | ulong(chosen);
                            radix_rank = rank;
                        }
                        threadgroup_barrier(mem_flags::mem_threadgroup);
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
                if (lane == 0) {
                    threshold_key = radix_prefix;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compact the exact winners. Atomic arrival order is canonicalized by
        // the same bitonic output network as the decode-oriented selector.
        exchange_scores[lane] = -INFINITY;
        exchange_indices[lane] = 0xffffffffu;
        exchange_valid[lane] = 0;
        if (lane == 0) {
            atomic_store_explicit(&selected_count, 0u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (k_eff > 0) {
            const ulong threshold = threshold_key;
            for (uint block = lane; block < valid_count; block += WIDTH) {
                const float adjusted =
                    scores[score_base + block] - float(block) * 1.0e-12f;
                if (qsa_composite_key(adjusted, block) >= threshold) {
                    const uint slot = atomic_fetch_add_explicit(
                        &selected_count, 1u, memory_order_relaxed);
                    if (slot < TOP_K) {
                        exchange_scores[slot] = adjusted;
                        exchange_indices[slot] = block;
                        exchange_valid[slot] = 1;
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint my_index = exchange_indices[lane];
        bool my_valid = exchange_valid[lane] != 0;
        float my_score = exchange_scores[lane];
        for (uint sequence = 2; sequence <= WIDTH; sequence <<= 1) {
            for (uint stride = sequence >> 1; stride > 0; stride >>= 1) {
                exchange_scores[lane] = my_score;
                exchange_indices[lane] = my_index;
                exchange_valid[lane] = my_valid ? 1 : 0;
                threadgroup_barrier(mem_flags::mem_threadgroup);

                const uint partner = lane ^ stride;
                const float other_score = exchange_scores[partner];
                const uint other_index = exchange_indices[partner];
                const bool other_valid = exchange_valid[partner] != 0;
                threadgroup_barrier(mem_flags::mem_threadgroup);

                const bool is_lower = (lane & stride) == 0;
                const float a_score = is_lower ? my_score : other_score;
                const uint a_index = is_lower ? my_index : other_index;
                const bool a_valid = is_lower ? my_valid : other_valid;
                const float b_score = is_lower ? other_score : my_score;
                const uint b_index = is_lower ? other_index : my_index;
                const bool b_valid = is_lower ? other_valid : my_valid;

                const bool lower_wants_before = (lane & sequence) == 0;
                const bool b_before_a = ROW_TOKEN_MODE
                    ? qsa_row_score_before(
                        b_score, b_index, b_valid,
                        a_score, a_index, a_valid)
                    : qsa_index_before(
                        b_index, b_valid, a_index, a_valid);
                const bool a_before_b = ROW_TOKEN_MODE
                    ? qsa_row_score_before(
                        a_score, a_index, a_valid,
                        b_score, b_index, b_valid)
                    : qsa_index_before(
                        a_index, a_valid, b_index, b_valid);
                const bool swap = lower_wants_before ? b_before_a : a_before_b;
                if (swap) {
                    my_score = is_lower ? b_score : a_score;
                    my_index = is_lower ? b_index : a_index;
                    my_valid = is_lower ? b_valid : a_valid;
                }
            }
        }
        """
        + epilogue
    )

    return mx.fast.metal_kernel(
        name=(
            f"mtplx_qsa_prefill_topk_{mode}_"
            + (f"n{static_blocks}" if static_blocks > 0 else "nrt")
            + f"_k{topk}_r{ratio}_w{width}_t{output_tokens}"
        ),
        input_names=["scores", "pos_start", "total_tokens", "logical_blocks"],
        output_names=output_names,
        header=header,
        source=source,
        ensure_row_contiguous=True,
    )


_SIMD_TOPK_LANES = 32
#: Pooled blocks (context / ratio) up to which the simd selector serves
#: ``blocks`` mode. Its cost is ten reads of the score row, so it grows with
#: the row: on an M5 Max (kernel alone, 2,048 rows, 2026-09-18) 0.40
#: microseconds per row against 2.43 at 4,608 blocks, 1.68 against 2.76 at
#: 16,384, and 3.51 against 3.05 at 32,768. The network selector reads the row
#: four times behind a fixed 2.3 microseconds of barriers, so it keeps the
#: long rows.
_SIMD_TOPK_MAX_BLOCKS = 24_576
_TOPK_KERNEL_VARIANTS = ("auto", "network", "simd")

_SIMD_TOPK_SOURCE = r"""
        const uint lane = uint(thread_index_in_simdgroup);
        const uint row = threadgroup_position_in_grid.x;
        const int qpos = pos_start[0] + int(row);
        const int logical_value = logical_blocks[0];
        const uint logical = logical_value > 0
            ? metal::min(uint(logical_value), BACKING_BLOCKS)
            : 0u;
        const int complete_value = (qpos + 1) / int(RATIO);
        const uint complete = complete_value > 0 ? uint(complete_value) : 0u;
        const uint valid_count = metal::min(logical, complete);
        const uint k_eff = metal::min(TOP_K, valid_count);
        const size_t score_base = (size_t)row * BACKING_BLOCKS;
        const size_t out_base = (size_t)row * TOP_K;

        const uint seg_len = (valid_count + LANES - 1u) / LANES;
        const uint seg_start = metal::min(lane * seg_len, valid_count);
        const uint seg_end = metal::min(seg_start + seg_len, valid_count);

        uint threshold = 0u;
        uint first_winning_tie = 0u;
        uint ties_before = 0u;
        uint slot = seg_start;
        const bool selective = k_eff > 0u && k_eff < valid_count;
        if (selective) {
            uint rank = k_eff - 1u;       // rank 0 = the largest key
            uint resolved = 0u;           // high bits fixed so far
            for (uint pass = 0u; pass < DIGIT_PASSES; ++pass) {
                const uint bits_left = 32u - resolved;
                const uint bits = metal::min(DIGIT_BITS, bits_left);
                const uint shift = bits_left - bits;
                const uint digit_mask = (1u << bits) - 1u;
                uint4 counts[DIGIT_VECS];
                for (uint v = 0u; v < DIGIT_VECS; ++v) {
                    counts[v] = uint4(0u);
                }
                for (uint block = seg_start; block < seg_end; ++block) {
                    const float adjusted =
                        scores[score_base + block] - float(block) * 1.0e-12f;
                    const uint key = qsa_float_order_key(adjusted);
                    const bool match = resolved == 0u
                        || (key >> bits_left) == (threshold >> bits_left);
                    const uint4 digit = uint4(match
                        ? ((key >> shift) & digit_mask) : 0xffffu);
                    counts[0] += uint4(digit == uint4(0u, 1u, 2u, 3u));
                    counts[1] += uint4(digit == uint4(4u, 5u, 6u, 7u));
                    counts[2] += uint4(digit == uint4(8u, 9u, 10u, 11u));
                    counts[3] += uint4(digit == uint4(12u, 13u, 14u, 15u));
                }
                uint chosen = 0u;
                bool found = false;
                for (int v = int(DIGIT_VECS) - 1; v >= 0; --v) {
                    const uint4 total = simd_sum(counts[v]);
                    for (int e = 3; e >= 0; --e) {
                        const uint count = total[e];
                        if (!found) {
                            if (rank < count) {
                                chosen = uint(v) * 4u + uint(e);
                                found = true;
                            } else {
                                rank -= count;
                            }
                        }
                    }
                }
                threshold |= chosen << shift;
                resolved += bits;
            }
            // threshold is the K-th largest 32-bit key; rank is its position
            // among the elements that tie with it (0 = the largest block id).
            uint local_greater = 0u;
            uint local_ties = 0u;
            for (uint block = seg_start; block < seg_end; ++block) {
                const float adjusted =
                    scores[score_base + block] - float(block) * 1.0e-12f;
                const uint key = qsa_float_order_key(adjusted);
                local_greater += key > threshold ? 1u : 0u;
                local_ties += key == threshold ? 1u : 0u;
            }
            const uint greater = simd_sum(local_greater);
            const uint ties = simd_sum(local_ties);
            const uint need = k_eff - greater;
            first_winning_tie = ties - need;
            ties_before = simd_prefix_exclusive_sum(local_ties);
            const uint my_ties_end = ties_before + local_ties;
            const uint my_first = metal::max(first_winning_tie, ties_before);
            const uint my_tie_winners = my_ties_end > my_first ? my_ties_end - my_first : 0u;
            slot = simd_prefix_exclusive_sum(local_greater + my_tie_winners);
        }
        if (k_eff > 0u) {
            uint tie_index = ties_before;
            for (uint block = seg_start; block < seg_end; ++block) {
                const float adjusted =
                    scores[score_base + block] - float(block) * 1.0e-12f;
                bool wins = true;
                if (selective) {
                    const uint key = qsa_float_order_key(adjusted);
                    wins = key > threshold;
                    if (key == threshold) {
                        wins = tie_index >= first_winning_tie;
                        tie_index += 1u;
                    }
                }
                if (wins && slot < TOP_K) {
                    block_ids[out_base + slot] = int(block);
                    block_valid[out_base + slot] = true;
                    adjusted_scores[out_base + slot] = adjusted;
                    slot += 1u;
                }
            }
        }
        for (uint empty = k_eff + lane; empty < TOP_K; empty += LANES) {
            block_ids[out_base + empty] = 0;
            block_valid[out_base + empty] = false;
            adjusted_scores[out_base + empty] = -INFINITY;
        }
"""


def _prefill_topk_variant() -> str:
    raw = (os.environ.get("MTPLX_QSA_PREFILL_TOPK_KERNEL") or "auto").strip().lower()
    return raw if raw in _TOPK_KERNEL_VARIANTS else "auto"


def _simd_topk_applies(mode: QSAPrefillMode, blocks: int) -> bool:
    """Whether the simd selector serves this call.

    ``blocks`` mode only (the other two outputs keep the network selector:
    ``row_tokens`` is ordered by score, and the dense mask is off the fast
    lane). Tensor-unit GPUs only: that is the one GPU class the 32-lane
    simdgroup layout was run on, and the same gate the flash consumer these
    selections feed already needs.
    """

    variant = _prefill_topk_variant()
    if mode != "blocks" or variant == "network":
        return False
    if not qsa_indexer_select_nax_available():
        return False
    return variant == "simd" or int(blocks) <= _SIMD_TOPK_MAX_BLOCKS


@lru_cache(maxsize=128)
def _prefill_topk_simd_kernel(topk: int, ratio: int, static_blocks: int = 0):
    """Exact top-k of one score row on one 32-lane simdgroup.

    The network selector gives a row 512 threads that share one histogram
    through atomics, wait on a threadgroup barrier after every step, and then
    order the winners with a 45-stage bitonic network (90 more barriers). Here
    a row is one simdgroup and the lanes talk through simd reductions only: no
    threadgroup memory, no barrier, no atomic.

    * Every lane owns one contiguous 1/32 of the row.
    * The K-th largest 32-bit score key is resolved four bits a scan: sixteen
      register counters per lane (one-hot vector adds, no data-dependent
      index), one ``simd_sum`` per uint4 of counters, eight scans.
    * Ties at that key are broken by block id exactly as the composite key
      does (the larger id ranks higher): the winners among the ties are the
      LAST ``need`` of them in ascending block order.
    * ``simd_prefix_exclusive_sum`` gives each lane its output offset and its
      first tie index, and the lane writes its winners straight into their
      final slots, so the row is born in ascending block order.

    Same adjusted score, same order key, same winners, same slots: every
    output element equals the network selector's (normal, ReLU, heavy-tie,
    constant and inf / -inf / -0.0 score planes; rows that see fewer than K
    blocks; 2,049 to 131,072 tokens).
    """

    header = (
        _SELECT_HEADER
        + f"""
{_blocks_constant(static_blocks)}
constant constexpr uint TOP_K = {topk};
constant constexpr uint RATIO = {ratio};
constant constexpr uint LANES = {_SIMD_TOPK_LANES};
constant constexpr uint DIGIT_BITS = 4;
constant constexpr uint DIGIT_PASSES = 8;
constant constexpr uint DIGIT_VECS = 4;
"""
    )
    return mx.fast.metal_kernel(
        name=(
            "mtplx_qsa_prefill_topk_blocks_simd_"
            + (f"n{static_blocks}" if static_blocks > 0 else "nrt")
            + f"_k{topk}_r{ratio}"
        ),
        input_names=["scores", "pos_start", "total_tokens", "logical_blocks"],
        output_names=["block_ids", "block_valid", "adjusted_scores"],
        header=header,
        source=(
            ("" if static_blocks > 0 else _RUNTIME_BLOCKS_PREAMBLE) + _SIMD_TOPK_SOURCE
        ),
        ensure_row_contiguous=True,
    )


def qsa_indexer_prefill_topk_metal(
    scores: mx.array,
    *,
    pos_start: int | mx.array,
    total_tokens: int | mx.array,
    block_topk: int,
    compress_ratio: int,
    logical_blocks: int | mx.array | None = None,
    output_total_tokens: int | None = None,
    mode: QSAPrefillMode = "blocks",
):
    """Select exact QSA blocks from an unmasked float32 prefill score plane."""

    if mode not in ("blocks", "dense_mask", "row_tokens"):
        raise ValueError(f"unknown QSA prefill mode {mode!r}")
    rows, blocks, topk, width = _validate_topk(
        scores,
        pos_start=pos_start,
        total_tokens=total_tokens,
        logical_blocks=logical_blocks,
        block_topk=block_topk,
        compress_ratio=compress_ratio,
    )
    ratio = int(compress_ratio)
    kernel_output_tokens = 0
    if mode == "blocks":
        output_shapes = [(rows, topk), (rows, topk), (rows, topk)]
        output_dtypes = [mx.int32, mx.bool_, mx.float32]
    elif mode == "dense_mask":
        static_total = _static_int(total_tokens)
        dense_width = (
            static_total if output_total_tokens is None else int(output_total_tokens)
        )
        if dense_width is None:
            raise ValueError(
                "output_total_tokens is required when total_tokens is a tensor"
            )
        if dense_width < 0:
            raise ValueError(
                f"output_total_tokens must be non-negative, got {dense_width}"
            )
        if static_total is not None and dense_width < static_total:
            raise ValueError(
                f"output_total_tokens={dense_width} is smaller than total={static_total}"
            )
        kernel_output_tokens = dense_width
        output_shapes = [(1, 1, rows, dense_width)]
        output_dtypes = [mx.bool_]
    else:
        token_width = topk * ratio + ratio
        output_shapes = [(rows, token_width), (rows, token_width)]
        output_dtypes = [mx.int32, mx.bool_]

    pos_scalar = _as_i32_scalar(pos_start, "pos_start")
    total_scalar = _as_i32_scalar(total_tokens, "total_tokens")
    logical_scalar = _as_i32_scalar(
        blocks if logical_blocks is None else logical_blocks,
        "logical_blocks",
    )
    static_blocks = blocks if _topk_static_blocks() else 0
    if _simd_topk_applies(mode, blocks):
        return tuple(
            _prefill_topk_simd_kernel(topk, ratio, static_blocks)(
                inputs=[scores, pos_scalar, total_scalar, logical_scalar],
                grid=(rows * _SIMD_TOPK_LANES, 1, 1),
                threadgroup=(_SIMD_TOPK_LANES, 1, 1),
                output_shapes=output_shapes,
                output_dtypes=output_dtypes,
            )
        )
    kernel = _prefill_topk_kernel(
        mode,
        topk,
        ratio,
        width,
        kernel_output_tokens,
        static_blocks,
    )
    outputs = kernel(
        inputs=[scores, pos_scalar, total_scalar, logical_scalar],
        grid=(rows * width, 1, 1),
        threadgroup=(width, 1, 1),
        output_shapes=output_shapes,
        output_dtypes=output_dtypes,
    )
    return outputs[0] if mode == "dense_mask" else tuple(outputs)


def qsa_indexer_prefill_metal(
    q: mx.array,
    pooled: mx.array,
    *,
    pos_start: int | mx.array,
    total_tokens: int | mx.array,
    block_topk: int,
    compress_ratio: int,
    logical_blocks: int | mx.array | None = None,
    output_total_tokens: int | None = None,
    mode: QSAPrefillMode = "blocks",
    score_workspace_bytes: int = DEFAULT_PREFILL_SCORE_WORKSPACE_BYTES,
):
    """Run the producer-selected two-stage backend for a large ``S > 1`` prefill."""

    _require_metal()
    if q.ndim != 4 or int(q.shape[0]) != 1:
        raise ValueError(f"q must have shape [1,S,H,D], got {q.shape}")
    if pooled.ndim != 3 or int(pooled.shape[0]) != 1:
        raise ValueError(f"pooled must have shape [1,N,D], got {pooled.shape}")
    if q.dtype not in _SUPPORTED_DTYPES or pooled.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(
            "q and pooled must be float16, bfloat16, or float32; "
            f"got {q.dtype} and {pooled.dtype}"
        )
    rows, heads, head_dim = map(int, q.shape[1:])
    blocks, pooled_dim = map(int, pooled.shape[1:])
    if rows <= 1:
        raise ValueError(f"prefill backend requires S > 1, got S={rows}")
    if pooled_dim != head_dim or blocks <= 0:
        raise ValueError(
            f"pooled shape {pooled.shape} is incompatible with head_dim={head_dim}"
        )

    use_mpp_scores = qsa_indexer_prefill_scores_mpp_supported(q, pooled)
    use_simd_scores = (
        not use_mpp_scores and qsa_indexer_prefill_scores_simd_supported(q[:, :1], pooled)
    )
    score_producer: QSAPrefillScoreProducer = (
        "mpp" if use_mpp_scores else "simd" if use_simd_scores else "mlx"
    )
    chunk_rows = qsa_indexer_prefill_score_chunk_rows(
        rows,
        heads,
        blocks,
        score_workspace_bytes,
        producer=score_producer,
    )
    pooled_t = None
    chunks = []
    for row_start in range(0, rows, chunk_rows):
        row_stop = min(rows, row_start + chunk_rows)
        q_chunk = q[:, row_start:row_stop]
        if use_mpp_scores:
            scores = qsa_indexer_prefill_scores_mpp(q_chunk, pooled)
        elif use_simd_scores:
            scores = qsa_indexer_prefill_scores_simd(q_chunk, pooled)
        else:
            if pooled_t is None:
                pooled_t = mx.swapaxes(pooled.astype(mx.float32), 1, 2)[:, None]
            scores = qsa_indexer_prefill_scores(
                q_chunk,
                pooled_t,
                head_dim=head_dim,
            )
        chunks.append(
            qsa_indexer_prefill_topk_metal(
                scores,
                pos_start=pos_start + row_start,
                total_tokens=total_tokens,
                logical_blocks=logical_blocks,
                block_topk=block_topk,
                compress_ratio=compress_ratio,
                output_total_tokens=output_total_tokens,
                mode=mode,
            )
        )

    if mode == "dense_mask":
        return chunks[0] if len(chunks) == 1 else mx.concatenate(chunks, axis=2)
    if len(chunks) == 1:
        return chunks[0]
    return tuple(
        mx.concatenate([chunk[leaf] for chunk in chunks], axis=0)
        for leaf in range(len(chunks[0]))
    )


def qsa_indexer_prefill_blocks_metal(
    q: mx.array,
    pooled: mx.array,
    **kwargs,
) -> tuple[mx.array, mx.array, mx.array]:
    """Return canonical block ids, validity, and adjusted scores ``[S,K]``."""

    return qsa_indexer_prefill_metal(q, pooled, mode="blocks", **kwargs)


def qsa_indexer_prefill_dense_mask_metal(
    q: mx.array,
    pooled: mx.array,
    **kwargs,
) -> mx.array:
    """Return the dense QSA selection mask ``[1,1,S,capacity]``."""

    return qsa_indexer_prefill_metal(q, pooled, mode="dense_mask", **kwargs)


def qsa_indexer_prefill_row_tokens_metal(
    q: mx.array,
    pooled: mx.array,
    **kwargs,
) -> tuple[mx.array, mx.array]:
    """Return fixed-width gathered token ids and validity ``[S,K*r+r]``."""

    return qsa_indexer_prefill_metal(q, pooled, mode="row_tokens", **kwargs)
