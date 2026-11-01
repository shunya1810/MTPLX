"""One-dispatch K/V row gather for the Qwen4 fixed-M4 QSA verifier."""

from __future__ import annotations

from functools import lru_cache
from typing import Callable

import mlx.core as mx


_H_KV = 2
_ROWS = 4
_SELECTED = 2052
_HEAD_DIM = 256
_VECTOR_WIDTH = 4
_VECTORS_PER_HEAD = _HEAD_DIM // _VECTOR_WIDTH
_ELEMENTS = _H_KV * _ROWS * _SELECTED * _VECTORS_PER_HEAD
_OUTPUT_SHAPE = (1, _H_KV, _ROWS, _SELECTED, _HEAD_DIM)


@lru_cache(maxsize=None)
def _fused_kv_gather_kernel(capacity: int):
    source = f"""
        constexpr uint H_KV = {_H_KV};
        constexpr uint ROWS = {_ROWS};
        constexpr uint SELECTED = {_SELECTED};
        constexpr uint HEAD_DIM = {_HEAD_DIM};
        constexpr uint VECTORS_PER_HEAD = {_VECTORS_PER_HEAD};
        constexpr uint CAPACITY = {capacity};

        uint output = thread_position_in_grid.x;
        if (output >= H_KV * ROWS * SELECTED * VECTORS_PER_HEAD) return;
        uint dimension = output % VECTORS_PER_HEAD;
        uint selected_offset = (output / VECTORS_PER_HEAD) % (ROWS * SELECTED);
        uint head = output / (VECTORS_PER_HEAD * ROWS * SELECTED);
        uint token = uint(token_idx[selected_offset]);
        uint input = (head * CAPACITY + token) * VECTORS_PER_HEAD + dimension;
        const device vec<T, 4>* keys4 =
            reinterpret_cast<const device vec<T, 4>*>(keys);
        const device vec<T, 4>* values4 =
            reinterpret_cast<const device vec<T, 4>*>(values);
        device vec<T, 4>* selected_keys4 =
            reinterpret_cast<device vec<T, 4>*>(selected_keys);
        device vec<T, 4>* selected_values4 =
            reinterpret_cast<device vec<T, 4>*>(selected_values);
        selected_keys4[output] = keys4[input];
        selected_values4[output] = values4[input];
    """
    return mx.fast.metal_kernel(
        name=f"mtplx_qwen4_qsa_m4_fused_kv_gather_c{capacity}",
        input_names=["keys", "values", "token_idx"],
        output_names=["selected_keys", "selected_values"],
        source=source,
        ensure_row_contiguous=True,
    )


def bind_qwen4_qsa_m4_fused_kv_gather(
    *, capacity: int
) -> Callable[[mx.array, mx.array, mx.array], tuple[mx.array, mx.array]]:
    """Bind the exact 1K/16K fixed-M4 gather geometry once at cache install."""

    capacity = int(capacity)
    if capacity < 16_384 or capacity % 4:
        raise ValueError(
            "Qwen4 fixed-M4 fused K/V gather requires a 16K context cache "
            "whose capacity is divisible by the QSA ratio"
        )
    kernel = _fused_kv_gather_kernel(capacity)

    def gather(
        keys: mx.array, values: mx.array, token_idx: mx.array
    ) -> tuple[mx.array, mx.array]:
        return kernel(
            inputs=[keys, values, token_idx],
            # A pure row copy: any 16-bit cache dtype (bf16 packs, fp16 M1 packs)
            template=[("T", keys.dtype)],
            grid=(_ELEMENTS, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[_OUTPUT_SHAPE, _OUTPUT_SHAPE],
            output_dtypes=[keys.dtype, values.dtype],
        )

    return gather
