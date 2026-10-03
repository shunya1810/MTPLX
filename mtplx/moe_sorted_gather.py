"""The one entry point for MLX's expert-sorted quantized gather, with its row guard.

Every MoE in MTPLX that sorts its routed rows by expert multiplies them with
``mx.gather_qmm(..., sorted_indices=True)`` over a ``[routed_rows, 1, K]``
activation.  Those calls go through :func:`gather_qmm` (a direct call) or
:func:`switch_linear` (a switch-linear module call), and mlx-lm's own switch
layers get the same guard from :func:`install_switch_linear_guard`.

Why the guard exists
--------------------
On a tensor-unit (NAX) GPU, MLX 0.32.2 runs that call on its sorted-rows
tensor-unit kernel for affine weights.  The kernel turns "rows left from this
simdgroup's first row" into a 16-bit integer before clamping it to the tile.
When a call routes more than 32,767 rows and the row count is not a multiple of
the kernel's row tile (32 or 64 rows, picked from rows per expert), the leading
simdgroups read a negative or wrapped count, skip their matmul and never store:
their output rows keep whatever the allocator handed back.  An aligned count
takes a branch that never computes that number, which is why 32,768 and 40,960
rows are fine.  Measured on an M5 Max (applegpu_g17s, MLX 0.32.2): 32,770 rows
leave rows 0-31 unwritten, 34,040 rows 0-1,279, 40,950 rows 0-8,191.  MLX 0.32.3
schedules the kernel per expert segment and is correct.

Flash-Next routes ten rows per token, so any prefill forward of 3,277 to 4,095
tokens whose width is not a multiple of 32 crosses the bound twice per layer
(the gate/up gather and the down gather).

The guard
---------
Such a call is padded to the next multiple of 64 rows (zero activations, the
last expert index repeated so the order stays sorted), run once, and sliced
back to its real rows.  Every tile is then full, so the kernel takes its
aligned branch.  Each output row depends only on its own input row and its
expert's weights, so the real rows are bit-identical to an unpadded call
wherever that call is correct (``tests/test_moe_sorted_gather.py`` pins it).

It engages only where the defect can occur: MLX's own tensor-unit test
(``nax_detect.nax_hardware_available``: the M1 to M4 rehearsal architecture
turns it off exactly as it turns off MLX's kernel, and the route switch
``MTPLX_FORCE_GPU_FAMILY_FALLBACK`` does not, because MLX does not read it) and
an MLX release before 0.32.3.  Every other call passes through unchanged.

Token rows in place
-------------------
:func:`sort_rows` orders the routed rows exactly as mlx-lm's gather-sort does
but returns a row map instead of copying each token row once per expert it is
routed to.  :func:`gather_qmm_rows` (a sorted gather) and :func:`swiglu_rows`
(Flash-Next's fused gate/up gather followed by ``silu(gate) * up``) take that
map: on a tensor-unit GPU our kernel (``mtplx.kernels.moe_sorted_gather_nax``)
reads the rows through it, bit-identical to the copy plus stock chain and
with no row bound; everywhere else, and whenever the kernel declines, the rows
are copied and the stock op runs behind the guard.
"""

from __future__ import annotations

import functools
import re
import sys
from functools import lru_cache
from typing import Any

import mlx.core as mx

from mtplx import nax_detect

#: Row counts up to this bound cannot wrap the kernel's 16-bit row count.
INT16_ROW_BOUND = 32767
#: Padding to a multiple of 64 fills every row tile MLX may pick (32 or 64).
ROW_ALIGN = 64
#: The first MLX release whose sorted tensor-unit gather counts rows in 32 bits.
FIXED_MLX_RELEASE = (0, 32, 3)

_PRE_RELEASE = re.compile(r"^[.\-_]?(dev|a|b|c|rc|alpha|beta|pre|preview)\d*", re.I)
_GUARD_MARK = "_mtplx_sorted_rows_guard"

_STATS: dict[str, int] = {"padded_calls": 0, "padded_rows": 0}
_ANNOUNCED = False


def mlx_release_affected(version: str) -> bool:
    """Whether an MLX version string predates the sorted-gather row fix.

    A pre-release of the fixing release counts as affected (it may predate the
    fix); an unreadable version counts as affected (the guard is exact either
    way, so the safe reading costs nothing but a pad).
    """

    match = re.match(r"^\s*v?(\d+)\.(\d+)\.(\d+)(.*)$", str(version))
    if match is None:
        return True
    release = tuple(int(part) for part in match.groups()[:3])
    if release < FIXED_MLX_RELEASE:
        return True
    if release == FIXED_MLX_RELEASE and _PRE_RELEASE.match(match.group(4) or ""):
        return True
    return False


@lru_cache(maxsize=1)
def guard_active() -> bool:
    """MLX will run its sorted tensor-unit kernel here, and that kernel has the
    16-bit row count.  Both halves are fixed for the life of the process."""

    return mlx_release_affected(mx.__version__) and nax_detect.nax_hardware_available()


def pad_rows(rows: int) -> int:
    """Rows to append to a sorted call of ``rows`` routed rows (0 = none)."""

    rows = int(rows)
    if rows <= INT16_ROW_BOUND or rows % ROW_ALIGN == 0 or not guard_active():
        return 0
    return ROW_ALIGN - rows % ROW_ALIGN


def stats() -> dict[str, int]:
    """How many sorted gathers this process padded, and by how many rows."""

    return dict(_STATS)


def _call_pad(
    x: mx.array,
    rhs_indices: Any,
    lhs_indices: Any,
    *,
    sorted_indices: bool,
    transpose: bool,
    mode: Any,
) -> int:
    """The pad for one call, 0 unless it is an affine, transposed, expert-sorted
    call in the ``[rows, 1, K]`` layout MLX sends to the affected kernel."""

    if not sorted_indices or not transpose or mode != "affine" or rhs_indices is None:
        return 0
    if rhs_indices.ndim != 1 or x.ndim < 2 or x.shape[-2] != 1:
        return 0
    rows = int(rhs_indices.shape[0])
    if int(x.shape[0]) != rows:
        return 0
    if lhs_indices is not None and (
        lhs_indices.ndim != 1 or int(lhs_indices.shape[0]) != rows
    ):
        return 0
    return pad_rows(rows)


def _pad(x: mx.array, indices: mx.array, pad: int) -> tuple[mx.array, mx.array]:
    """``x`` with ``pad`` zero rows and ``indices`` with its last expert repeated."""

    global _ANNOUNCED
    rows = int(indices.shape[0])
    _STATS["padded_calls"] += 1
    _STATS["padded_rows"] += pad
    if not _ANNOUNCED:
        _ANNOUNCED = True
        print(
            f"[moe-sorted-gather] padding a {rows:,}-row sorted expert gather to "
            f"{rows + pad:,} rows: MLX {mx.__version__} leaves rows unwritten past "
            f"{INT16_ROW_BOUND:,} unaligned rows on tensor-unit GPUs (fixed in "
            "0.32.3); the real rows are unchanged",
            file=sys.stderr,
            flush=True,
        )
    x = mx.concatenate([x, mx.zeros((pad, *x.shape[1:]), dtype=x.dtype)], axis=0)
    indices = mx.concatenate([indices, mx.broadcast_to(indices[-1:], (pad,))], axis=0)
    return x, indices


def gather_qmm(
    x: mx.array,
    w: mx.array,
    scales: mx.array,
    biases: mx.array | None = None,
    lhs_indices: mx.array | None = None,
    rhs_indices: mx.array | None = None,
    transpose: bool = True,
    group_size: int | None = None,
    bits: int | None = None,
    mode: str = "affine",
    *,
    sorted_indices: bool = False,
) -> mx.array:
    """``mx.gather_qmm`` with the sorted-rows guard: same arguments, same result."""

    pad = (
        _call_pad(
            x,
            rhs_indices,
            lhs_indices,
            sorted_indices=True,
            transpose=transpose,
            mode=mode,
        )
        if sorted_indices
        else 0
    )
    if pad:
        rows = int(x.shape[0])
        x, rhs_indices = _pad(x, rhs_indices, pad)
        if lhs_indices is not None:
            lhs_indices = mx.concatenate(
                [lhs_indices, mx.arange(rows, rows + pad, dtype=lhs_indices.dtype)]
            )
    y = mx.gather_qmm(
        x,
        w,
        scales,
        biases,
        lhs_indices=lhs_indices,
        rhs_indices=rhs_indices,
        transpose=transpose,
        group_size=group_size,
        bits=bits,
        mode=mode,
        sorted_indices=sorted_indices,
    )
    return y[: x.shape[0] - pad] if pad else y


def sort_rows(x: mx.array, indices: mx.array):
    """Expert-sort the routed rows of ``x`` without copying them.

    ``x`` is ``[..., K]`` tokens and ``indices`` ``[..., top_k]`` experts.
    Returns ``(tokens, row_map, sorted_indices, inv_order)``: ``tokens`` is
    ``[n_tokens, 1, K]``, sorted row ``i`` is token ``row_map[i]`` routed to
    expert ``sorted_indices[i]``, and ``inv_order`` unsorts, exactly as
    mlx-lm's gather-sort orders them (whose ``x[order // top_k]`` copy is what
    :func:`gather_qmm_rows` avoids on the tensor-unit kernel).
    """

    top_k = int(indices.shape[-1])
    flat = indices.flatten()
    order = mx.argsort(flat)
    inv_order = mx.argsort(order)
    row_map = (order // top_k).astype(mx.uint32)
    tokens = x.reshape(-1, 1, x.shape[-1])
    return tokens, row_map, flat[order], inv_order


def gather_qmm_rows(
    tokens: mx.array,
    row_map: mx.array,
    w: mx.array,
    scales: mx.array,
    biases: mx.array | None,
    rhs_indices: mx.array,
    *,
    group_size: int,
    bits: int,
    mode: str = "affine",
) -> mx.array:
    """``gather_qmm(tokens[row_map], ..., sorted_indices=True)``.

    On a tensor-unit GPU the kernel in ``mtplx.kernels.moe_sorted_gather_nax``
    reads each token row in place through the map instead of copying it
    (bit-identical; the ten routed copies of a token row then come from cache).
    Elsewhere, and whenever that kernel declines, the rows are copied and the
    stock op runs behind the row guard.
    """

    from mtplx.kernels import moe_sorted_gather_nax as kernel

    if kernel.applies(
        tokens, row_map, w, scales, biases, rhs_indices,
        group_size=group_size, bits=bits, mode=mode,
    ):
        y = kernel.gather_rows_qmm(
            tokens, row_map, w, scales, biases, rhs_indices,
            group_size=int(group_size), bits=int(bits),
        )
        if y is not None:
            return y
    return gather_qmm(
        tokens[row_map],
        w,
        scales,
        biases,
        rhs_indices=rhs_indices,
        transpose=True,
        group_size=group_size,
        bits=bits,
        mode=mode,
        sorted_indices=True,
    )


def swiglu_rows(
    tokens: mx.array,
    row_map: mx.array,
    w: mx.array,
    scales: mx.array,
    biases: mx.array | None,
    rhs_indices: mx.array,
    *,
    group_size: int,
    bits: int,
    mode: str = "affine",
) -> mx.array:
    """``nn.silu(gate) * up`` of the fused ``[gate | up]`` sorted gather
    ``gather_qmm(tokens[row_map], w, ...)``, as ``[rows, 1, N / 2]``.

    On a tensor-unit GPU one kernel computes it without the row copy or the
    full-width gate/up output (bit-identical to the chain below).  Elsewhere,
    and whenever that kernel declines, the chain runs: :func:`gather_qmm_rows`,
    the split and ``nn.silu(gate) * up``.
    """

    from mtplx.kernels import moe_sorted_gather_nax as kernel

    if kernel.applies(
        tokens, row_map, w, scales, biases, rhs_indices,
        group_size=group_size, bits=bits, mode=mode,
    ):
        y = kernel.gather_rows_qmm(
            tokens, row_map, w, scales, biases, rhs_indices,
            group_size=int(group_size), bits=int(bits), swiglu=True,
        )
        if y is not None:
            return y
    import mlx.nn as nn

    gu = gather_qmm_rows(
        tokens, row_map, w, scales, biases, rhs_indices,
        group_size=group_size, bits=bits, mode=mode,
    )
    gate, up = mx.split(gu, 2, axis=-1)
    return nn.silu(gate) * up


def _module_mode(module: Any) -> Any:
    """A switch linear's quantization mode, None when it holds dense weights."""

    if "scales" not in module:
        return None
    return getattr(module, "mode", "affine")


def switch_linear(
    module: Any, x: mx.array, indices: mx.array, *, sorted_indices: bool
) -> mx.array:
    """``module(x, indices, sorted_indices=...)`` for a switch linear, guarded."""

    pad = _call_pad(
        x,
        indices,
        None,
        sorted_indices=sorted_indices,
        transpose=True,
        mode=_module_mode(module),
    )
    if not pad:
        return module(x, indices, sorted_indices=sorted_indices)
    rows = int(indices.shape[0])
    x, indices = _pad(x, indices, pad)
    return module(x, indices, sorted_indices=True)[:rows]


def install_switch_linear_guard() -> bool:
    """Guard mlx-lm's quantized switch linear, which every mlx-lm MoE family
    (and any MTPLX block that keeps a stock projection) calls with sorted rows.

    Wraps the class's current ``__call__`` rather than replacing its body, so
    whatever it does (bias, a later patch) runs unchanged on the padded rows.
    Idempotent; a no-op wherever :func:`guard_active` is false.  Returns
    whether the guard is installed.
    """

    if not guard_active():
        return False
    from mlx_lm.models import switch_layers

    cls = switch_layers.QuantizedSwitchLinear
    current = cls.__call__
    if getattr(current, _GUARD_MARK, False):
        return True

    @functools.wraps(current)
    def guarded_call(self, x, indices, sorted_indices=False):
        if sorted_indices:
            pad = _call_pad(
                x,
                indices,
                None,
                sorted_indices=True,
                transpose=True,
                mode=getattr(self, "mode", "affine"),
            )
            if pad:
                rows = int(indices.shape[0])
                x, indices = _pad(x, indices, pad)
                return current(self, x, indices, sorted_indices=True)[:rows]
        return current(self, x, indices, sorted_indices=sorted_indices)

    setattr(guarded_call, _GUARD_MARK, True)
    cls.__call__ = guarded_call
    return True


_GLU_MARK = "_mtplx_switch_glu_rows"


def _glu_projections(module: Any):
    """The gate and up projections of a quantized mlx-lm ``SwitchGLU`` with the
    standard SwiGLU activation and no bias, or None for any other layout."""

    from mlx_lm.models.switch_layers import QuantizedSwitchLinear, SwiGLU

    if type(getattr(module, "activation", None)) is not SwiGLU:
        return None
    gate, up = getattr(module, "gate_proj", None), getattr(module, "up_proj", None)
    for proj in (gate, up):
        if type(proj) is not QuantizedSwitchLinear or "bias" in proj or _module_mode(proj) != "affine":
            return None
    if gate.group_size != up.group_size or gate.bits != up.bits:
        return None
    return gate, up


def switch_glu_rows(module: Any, x: mx.array, indices: mx.array, original) -> mx.array:
    """mlx-lm's ``SwitchGLU.__call__`` with its sorted regime on the row-map kernel.

    The rows are sorted as mlx-lm sorts them, the gate and up gathers, the row
    copy and the SwiGLU run as one tensor-unit kernel reading token rows in
    place (bit-identical to ``activation(up_proj(x), gate_proj(x))``; checked
    on first use per shape), and the down projection and the unsort are
    mlx-lm's own calls.  Every other case (training, decode and verify widths,
    another activation or layout, M1 to M4, a failed check) runs ``original``.
    """

    from mtplx.kernels import moe_sorted_gather_nax as kernel

    if getattr(module, "training", False) or int(indices.size) < kernel.min_rows():
        return original(module, x, indices)
    projections = _glu_projections(module)
    if projections is None or not kernel.available():
        return original(module, x, indices)
    gate, up = projections
    up_weights = (up.weight, up.scales, up.biases)
    tokens, row_map, idx, inv_order = sort_rows(x, indices)
    kw = dict(group_size=int(gate.group_size), bits=int(gate.bits))
    if not kernel.applies(
        tokens, row_map, gate.weight, gate.scales, gate.biases, idx, mode="affine", up=up_weights, **kw
    ):
        return original(module, x, indices)
    h = kernel.gather_rows_qmm(
        tokens, row_map, gate.weight, gate.scales, gate.biases, idx,
        swiglu=True, up=up_weights, act=module.activation, **kw,
    )
    if h is None:
        return original(module, x, indices)
    from mlx_lm.models.switch_layers import _scatter_unsort

    y = module.down_proj(h, idx, sorted_indices=True)
    return _scatter_unsort(y, inv_order, indices.shape).squeeze(-2)


def install_switch_glu_rows() -> bool:
    """Route mlx-lm's quantized ``SwitchGLU`` (Qwen3.5 and 3.6 MoE and every
    other mlx-lm family built on it) through :func:`switch_glu_rows` on
    tensor-unit GPUs.  Wraps the class's current ``__call__``; idempotent; a
    no-op where the kernel cannot run.  Returns whether it is installed."""

    from mtplx.kernels import moe_sorted_gather_nax as kernel

    if not kernel.available():
        return False
    from mlx_lm.models import switch_layers

    cls = switch_layers.SwitchGLU
    current = cls.__call__
    if getattr(current, _GLU_MARK, False):
        return True

    @functools.wraps(current)
    def rows_call(self, x, indices):
        return switch_glu_rows(self, x, indices, current)

    setattr(rows_call, _GLU_MARK, True)
    cls.__call__ = rows_call
    return True
