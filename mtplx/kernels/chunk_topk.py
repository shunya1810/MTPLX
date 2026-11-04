"""Top-k candidates of a few full-vocabulary rows without a full-row sort.

``mx.argpartition`` over a 248K-wide row sorts it (multi-block merge sort):
~0.5 ms of GPU time for the two rows of an A3B sampled verify, every round.
Sampling only needs a superset of each row's top ``k`` (top-k / top-p then
work on those few). Here one threadgroup per 4096-value chunk extracts the
chunk's ``k`` largest values by repeated threadgroup argmax; the union of the
chunk winners holds the row's top ``k`` exactly, and a small ``argpartition``
over it (61 x k values per row) picks the final candidates.
"""

from __future__ import annotations

from functools import lru_cache

import mlx.core as mx

_CHUNK = 4096
_THREADS = 256
_PER_THREAD = _CHUNK // _THREADS
_MAX_K = 64


@lru_cache(maxsize=None)
def _kernel(k: int):
    source = f"""
        constexpr uint K = {k};
        constexpr uint CHUNK = {_CHUNK};
        constexpr uint THREADS = {_THREADS};
        constexpr uint PER = {_PER_THREAD};
        constexpr uint SIMDS = THREADS / 32;

        const uint V = uint(V_size);
        const uint nchunk = (V + CHUNK - 1) / CHUNK;
        const uint chunk = threadgroup_position_in_grid.x;
        const uint row = threadgroup_position_in_grid.y;
        const uint tid = thread_position_in_threadgroup.x;
        const uint lane = thread_index_in_simdgroup;
        const uint sg = simdgroup_index_in_threadgroup;
        const uint base = chunk * CHUNK;

        float v[PER];
        for (uint j = 0; j < PER; ++j) {{
            uint col = base + tid + j * THREADS;
            v[j] = col < V ? float(rows[row * V + col]) : -INFINITY;
        }}
        threadgroup float sg_val[SIMDS];
        threadgroup uint sg_idx[SIMDS];
        threadgroup uint winner;
        for (uint it = 0; it < K; ++it) {{
            float best = -INFINITY;
            uint bj = 0;
            for (uint j = 0; j < PER; ++j) {{
                if (v[j] > best) {{
                    best = v[j];
                    bj = j;
                }}
            }}
            uint my_col = base + tid + bj * THREADS;
            float m = simd_max(best);
            uint c = simd_min(best == m ? my_col : 0xffffffffu);
            if (lane == 0) {{
                sg_val[sg] = m;
                sg_idx[sg] = c;
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sg == 0) {{
                float mv = lane < SIMDS ? sg_val[lane] : -INFINITY;
                uint mi = lane < SIMDS ? sg_idx[lane] : 0xffffffffu;
                float tm = simd_max(mv);
                uint ti = simd_min(mv == tm ? mi : 0xffffffffu);
                if (lane == 0) {{
                    winner = ti;
                    uint out = (row * nchunk + chunk) * K + it;
                    out_idx[out] = int(ti < V ? ti : 0u);
                    out_val[out] = ti < V ? tm : -INFINITY;
                }}
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (my_col == winner) {{
                v[bj] = -INFINITY;
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
    """
    return mx.fast.metal_kernel(
        name=f"mtplx_chunk_topk_k{k}",
        input_names=["rows", "V_size"],
        output_names=["out_idx", "out_val"],
        source=source,
    )


def chunk_topk_supported(k: int, vocab: int, m: int | None = None) -> bool:
    """k in 1..64, a row of at least one chunk, and enough chunk winners for m."""

    nchunk = -(-int(vocab) // _CHUNK)
    return (
        1 <= int(k) <= _MAX_K
        and int(vocab) >= _CHUNK
        and nchunk * int(k) >= int(m if m is not None else k)
    )


def topk_candidates(rows: mx.array, k: int, m: int) -> tuple[mx.array, mx.array]:
    """Indices ``[R, m]`` and values of ``m`` candidates per row that contain
    the row's ``k`` largest (``k <= m``); ``rows`` is ``[R, V]``."""

    r, v = int(rows.shape[0]), int(rows.shape[1])
    nchunk = -(-v // _CHUNK)
    idx, val = _kernel(int(k))(
        inputs=[rows, v],
        grid=(nchunk * _THREADS, r, 1),
        threadgroup=(_THREADS, 1, 1),
        output_shapes=[(r, nchunk * int(k)), (r, nchunk * int(k))],
        output_dtypes=[mx.int32, mx.float32],
    )
    sel = mx.argpartition(-val, kth=m - 1, axis=-1)[:, :m]
    return mx.take_along_axis(idx, sel, axis=-1), mx.take_along_axis(val, sel, axis=-1)
