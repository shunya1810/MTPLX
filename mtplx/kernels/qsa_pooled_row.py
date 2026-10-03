"""The fixed QSA bank's pooled-key row at verify widths, in one kernel.

Each verify, every QSA layer's indexer brings its pooled-key bank up to date
(``QSAIndexer._extend_pooled_fixed``): the block that may have just completed
is averaged over its ``ratio`` raw keys, RMS-normed with ``k_layernorm``,
rotated at the block's first position, and written into the bank if it really
completed. Inside a compiled verify body the stock spelling is about sixteen
graph nodes around the one bank write (two dynamic slices, the float32 cast,
the column sum, the mean, the norm, the position arithmetic, the rotary table's
multiply, cos and sin, the two rotation halves, the concatenation and the
select), in each of Flash-Next's 12 QSA layers. ``pooled_row`` computes the row
the bank write stores in one dispatch; the write itself stays the stock
``mx.slice_update``.

EXACTNESS. The kernel repeats the stock arithmetic in its order:

* the block's raw keys summed per value in float32, as the library's small
  column reduce sums them (from +0, rows in order; a split-row spelling is kept
  for the probe to choose), times ``1 / ratio`` (a power of two, exact), cast
  to the element type;
* the RMS norm as the library's single-row kernel does it for a 128-value row:
  32 lanes of four consecutive values, squares summed in order, a ``simd_sum``,
  a second ``simd_sum`` over the one partial, IEEE division by the width,
  precise ``rsqrt``, the cast to the element type before the weight multiply;
* the rotation at ``float(start + delta) * inv_freq``, precise ``cos`` and
  ``sin`` (the library's standalone kernels), the amplitude multiply when it is
  not 1, and each half's two products and sum spelled as the compiled graph's
  fused kernel evaluates them (``ROTATIONS``: separate products, or a fused
  multiply-add on either product); install picks the spelling that matches;
* the select: the new row where the block completed, else the bank's own row.

``install`` compares the kernel with the stock update compiled on this GPU,
bank for bank, as stored bits, on several positions (inside a block, on block
boundaries, both select outcomes, with and without a rotary delta), before the
first verify trace; a difference or a build failure leaves the stock update
serving with a printed reason.

Rollback: ``MTPLX_QWEN4_QSA_POOLED_ROW=0`` keeps the stock update.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Iterable, Optional

import mlx.core as mx
import numpy as np

from mtplx.float32_operand import float32_operand

ENV = "MTPLX_QWEN4_QSA_POOLED_ROW"

#: The row width the single-row norm spelling covers: 32 lanes of 4 values.
HEAD_DIM = 128
_LANES = 32

#: How the compiled graph's fused rotation kernel may evaluate
#: ``a * c + b * s`` (a, b the rotated values in float32, c, s the table):
ROTATIONS = {
    0: "two products, then the sum",
    1: "fused multiply-add on the cos product",
    2: "fused multiply-add on the sin product",
}
#: How the library's column reduce may sum the block's rows.
SUMS = {0: "rows in order from +0", 1: "even and odd rows apart, then combined"}

_STATE: dict[str, Any] = {
    "installed": False,
    "disabled_reason": None,
    "geometry": None,
    "spelling": None,
    "probing": False,
}
_COUNTS: dict[str, int] = {"installs": 0, "probe_cases": 0, "probe_failures": 0, "traces": 0}
_ENGAGED: set[str] = set()


_SOURCE = r"""
    const uint lane = thread_position_in_threadgroup.x;
    threadgroup T normed_row[D];
    const int block_index = block[0];
    // The bank's capacity is read from its shape, so one build serves every
    // bank size.
    const int blk = min(block_index, int(pooled_shape[1]) - 1);
    const int start = blk * RATIO;

    // The block's mean, per value, then its sum of squares in lane order.
    T cand[4];
    float squares = 0.0f;
    for (uint i = 0; i < 4u; ++i) {
        const uint d = lane * 4u + i;
#if POOL_SUM == 1
        float even = 0.0f;
        float odd = 0.0f;
        for (int r = 0; r < RATIO; r += 2) {
            even = float(raw[(size_t)(start + r) * D + d]) + even;
            odd = float(raw[(size_t)(start + r + 1) * D + d]) + odd;
        }
        const float sum = odd + even;
#else
        float sum = 0.0f;
        for (int r = 0; r < RATIO; ++r) {
            sum = float(raw[(size_t)(start + r) * D + d]) + sum;
        }
#endif
        cand[i] = static_cast<T>(sum * MEAN_SCALE);
        const float f = float(cand[i]);
        squares += f * f;
    }
    // One simdgroup covers the row: its sum, then the sum of that one partial
    // with the idle partial slots (+0).
    const float first = simd_sum(squares);
    const float total = simd_sum(lane == 0 ? first : 0.0f);
    const float inv = metal::precise::rsqrt(metal::precise::divide(total, float(D)) + NORM_EPS);
    for (uint i = 0; i < 4u; ++i) {
        const uint d = lane * 4u + i;
        const T unit = static_cast<T>(float(cand[i]) * inv);
        normed_row[d] = gamma[d] * unit;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float position = float(start + delta[0]);
    const bool completed = nb_total[0] > block_index;
    for (uint i = 0; i < 4u; ++i) {
        const uint d = lane * 4u + i;
        T value = normed_row[d];
        if (d < 2u * HALF) {
            const uint j = d < HALF ? d : d - HALF;
            const float angle = position * inv_freq[j];
#if POOL_AMP
            const float c = metal::precise::cos(angle) * amp[0];
            const float s = metal::precise::sin(angle) * amp[0];
#else
            const float c = metal::precise::cos(angle);
            const float s = metal::precise::sin(angle);
#endif
            // First half: x1 * cos + (-x2) * sin; second: x2 * cos + x1 * sin.
            const float a = float(normed_row[d]);
            const float b = d < HALF ? float(-normed_row[d + HALF]) : float(normed_row[j]);
#if POOL_ROTATION == 1
            const float rotated = metal::fma(a, c, b * s);
#elif POOL_ROTATION == 2
            const float rotated = metal::fma(b, s, a * c);
#else
            const float cos_part = a * c;
            const float sin_part = b * s;
            const float rotated = cos_part + sin_part;
#endif
            value = static_cast<T>(rotated);
        }
        row[d] = completed ? value : pooled[(size_t)blk * D + d];
    }
"""


@lru_cache(maxsize=None)
def _kernel(dtype_name: str, head_dim: int, half: int, ratio: int, eps: float, amp: bool, sum_spelling: int, rotation: int):
    header = (
        f"#define D {int(head_dim)}u\n"
        f"#define HALF {int(half)}u\n"
        f"#define RATIO {int(ratio)}\n"
        f"#define MEAN_SCALE {1.0 / int(ratio)!r}f\n"
        f"#define NORM_EPS {float(np.float32(eps))!r}f\n"
        f"#define POOL_AMP {1 if amp else 0}\n"
        f"#define POOL_SUM {int(sum_spelling)}\n"
        f"#define POOL_ROTATION {int(rotation)}\n"
    )
    inputs = ["raw", "pooled", "block", "nb_total", "delta", "gamma", "inv_freq"]
    if amp:
        inputs.append("amp")
    return mx.fast.metal_kernel(
        name=(
            f"mtplx_qsa_pooled_row_{dtype_name}_d{head_dim}_h{half}_r{ratio}"
            f"_s{sum_spelling}_o{rotation}{'_amp' if amp else ''}"
        ),
        input_names=inputs,
        output_names=["row"],
        header=header,
        source=_SOURCE,
        ensure_row_contiguous=True,
    )


def _scalar_int32(value: Any) -> mx.array:
    array = value if isinstance(value, mx.array) else mx.array(int(value), dtype=mx.int32)
    if array.dtype != mx.int32:
        array = array.astype(mx.int32)
    return array.reshape((1,))


def pooled_row(
    raw_keys: mx.array,
    pooled: mx.array,
    block: Any,
    nb_total: Any,
    delta: Any,
    gamma: mx.array,
    inv_freq: mx.array,
    *,
    ratio: int,
    eps: float,
    amplitude: float = 1.0,
    spelling: Optional[tuple[int, int]] = None,
) -> mx.array:
    """The ``[1, 1, head_dim]`` row the stock update writes at ``min(block,
    capacity - 1)``: the pooled, normed, rotated block when ``nb_total >
    block``, else the bank's own row there."""

    if spelling is None:
        spelling = _STATE["spelling"] or (0, 0)
    head_dim = int(pooled.shape[-1])
    half = int(inv_freq.shape[-1])
    amp = float(amplitude) != 1.0
    kernel = _kernel(
        str(pooled.dtype).split(".")[-1], head_dim, half, int(ratio),
        float(eps), amp, int(spelling[0]), int(spelling[1]),
    )
    inputs = [
        raw_keys,
        pooled,
        _scalar_int32(block),
        _scalar_int32(nb_total),
        _scalar_int32(0 if delta is None else delta),
        gamma,
        inv_freq.astype(mx.float32) if inv_freq.dtype != mx.float32 else inv_freq,
    ]
    if amp:
        inputs.append(float32_operand(float(amplitude)))
    (row,) = kernel(
        inputs=inputs,
        template=[("T", pooled.dtype)],
        grid=(_LANES, 1, 1),
        threadgroup=(_LANES, 1, 1),
        output_shapes=[(1, 1, head_dim)],
        output_dtypes=[pooled.dtype],
    )
    return row


def switched_on() -> bool:
    raw = (os.environ.get(ENV) or "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _geometry(indexer: Any, dtype) -> Optional[tuple]:
    try:
        norm = indexer.k_layernorm
        return (
            int(indexer.head_dim),
            int(indexer.ratio),
            int(indexer._inv_freq.shape[-1]),
            float(np.float32(norm.eps)),
            float(indexer._rope_attention_scaling),
            str(norm.weight.dtype),
            str(dtype),
        )
    except (AttributeError, TypeError, ValueError, IndexError):
        return None


def serves(indexer: Any, pooled: mx.array, raw_keys: mx.array) -> bool:
    """True when ``install`` proved the row for this indexer and bank dtype."""

    if not _STATE["installed"]:
        return False
    if pooled.dtype != raw_keys.dtype or int(raw_keys.shape[1]) < int(pooled.shape[1]) * int(indexer.ratio):
        return False
    return _geometry(indexer, pooled.dtype) == _STATE["geometry"]


def note_engaged(lane: str = "fixed bank") -> None:
    """Counts traced rows; the first leaves one log line (never the probe's)."""

    if _STATE["probing"]:
        return
    _COUNTS["traces"] += 1
    if lane not in _ENGAGED:
        _ENGAGED.add(lane)
        print(
            "[mtplx] QSA pooled-key row engaged in a compiled verify trace "
            "(one dispatch before the bank write)",
            flush=True,
        )


# ---------------------------------------------------------------------------
# Install: the first-use canary
# ---------------------------------------------------------------------------


class _ProbeBank:
    fixed_capacity = True

    def __init__(self, raw_keys, pooled, offset, rows, delta):
        self.raw_keys = raw_keys
        self.pooled = pooled
        self.offset = offset
        self._last_write_rows = rows
        self.rope_delta = delta


def _probe_update(indexer: Any, raw_keys, pooled, offset, total, rows, delta) -> mx.array:
    from mtplx.compile_state import compiled_step_body

    def update(raw_keys, pooled, offset, total):
        bank = _ProbeBank(raw_keys, pooled, offset, rows, delta)
        return indexer._extend_pooled_fixed(bank, total)

    with compiled_step_body():
        return mx.compile(update)(raw_keys, pooled, offset, total)


def _bits(a: mx.array) -> np.ndarray:
    if a.dtype in (mx.bfloat16, mx.float16):
        return np.array(mx.view(a, mx.uint16))
    return np.array(a).view(np.uint8)


def _probe_cases(indexer: Any, dtype, seed: int):
    """Banks and positions: inside a block, the last row of a block, the first
    row of the next, a window crossing a boundary, with and without a rotary
    delta; values with a wide exponent range."""

    head_dim, ratio = int(indexer.head_dim), int(indexer.ratio)
    blocks = 96
    keys = mx.random.split(mx.random.key(seed), 3)
    raw = (
        mx.random.normal((1, blocks * ratio, head_dim), key=keys[0])
        * mx.exp(1.5 * mx.random.normal((1, blocks * ratio, head_dim), key=keys[1]))
    ).astype(dtype)
    pooled = mx.random.normal((1, blocks, head_dim), key=keys[2]).astype(dtype)
    mx.eval(raw, pooled)
    cases = []
    for offset, rows in ((37 * ratio + 1, 1), (40 * ratio - 1, 1), (40 * ratio, 4), (51 * ratio - 2, 4), (63 * ratio - 4, 4), (blocks * ratio - 4, 4)):
        for delta in (None, 7):
            cases.append((raw, pooled, offset, rows, delta))
    return cases


def install(model: Any, *, dtype=None) -> dict:
    """Prove the row on this GPU against the stock update, then enable it.

    ``dtype`` is the banks' dtype (default: the indexer's key-norm weight
    dtype, which is the model's activation dtype); banks of another dtype keep
    the stock update.
    """

    reset_state()
    _COUNTS["installs"] += 1
    report: dict[str, Any] = {}

    def _off(reason: str, *, failure: bool = False) -> dict:
        _STATE["disabled_reason"] = reason
        if failure:
            _COUNTS["probe_failures"] += 1
        print(f"[mtplx] QSA pooled-key row off: {reason}; the stock update serves", flush=True)
        report.update(engagement())
        return report

    if not switched_on():
        return _off(f"{ENV}=0")
    if not mx.metal.is_available():
        return _off("no Metal GPU")
    indexer = None
    for layer in list(getattr(model, "layers", ()) or ()):
        candidate = getattr(getattr(layer, "self_attn", None), "indexer", None)
        if candidate is not None:
            indexer = candidate
            break
    if indexer is None:
        return _off("the model has no QSA indexer")
    if dtype is None:
        dtype = getattr(getattr(getattr(indexer, "k_layernorm", None), "weight", None), "dtype", None)
    if dtype not in (mx.bfloat16, mx.float16):
        return _off(f"bank dtype {dtype} is not bfloat16 or float16")
    geometry = _geometry(indexer, dtype)
    if geometry is None:
        return _off("unreadable QSA indexer geometry")
    head_dim, ratio, half = geometry[0], geometry[1], geometry[2]
    if geometry[5] != str(dtype):
        return _off(f"key-norm weight dtype {geometry[5]} differs from the bank dtype {dtype}")
    if head_dim != HEAD_DIM or ratio not in (2, 4, 8) or not 0 < 2 * half <= head_dim:
        return _off(f"geometry head_dim={head_dim}, ratio={ratio}, rotary half={half} is not the one the row covers")

    # Stage 1, a float32 twin of the indexer: at float32 the spellings part
    # on a large share of values (at bfloat16 the final cast hides most of
    # them), so this stage decides which ones the compiled update uses.
    # Stage 2 proves the chosen spelling at the bank dtype.
    twin = _float32_twin(indexer)
    stages = (
        ("float32", twin, mx.float32, _geometry(twin, mx.float32)),
        (str(dtype).split(".")[-1], indexer, dtype, geometry),
    )
    candidates = [(s, r) for s in SUMS for r in ROTATIONS]
    differing: dict[tuple[int, int], int] = {}
    for stage, owner, stage_dtype, stage_geometry in stages:
        cases = _probe_cases(owner, stage_dtype, 20260929)
        wants = []
        try:
            _STATE.update(installed=False, geometry=stage_geometry)
            for raw, pooled, offset, rows, delta in cases:
                off = mx.array(offset, dtype=mx.int32)
                tot = mx.array(offset + rows, dtype=mx.int32)
                want = _probe_update(owner, raw, pooled, off, tot, rows, delta)
                mx.eval(want)
                wants.append(_bits(want))
        except Exception as exc:  # the stock update itself failed: leave it alone
            _STATE.update(geometry=None)
            return _off(f"stock update probe ({stage}): {type(exc).__name__}: {exc}", failure=True)
        matching = []
        for spelling in candidates:
            _STATE.update(spelling=spelling, installed=True, probing=True)
            wrong = 0
            try:
                for (raw, pooled, offset, rows, delta), want in zip(cases, wants):
                    _COUNTS["probe_cases"] += 1
                    off = mx.array(offset, dtype=mx.int32)
                    tot = mx.array(offset + rows, dtype=mx.int32)
                    got = _probe_update(owner, raw, pooled, off, tot, rows, delta)
                    mx.eval(got)
                    wrong += int(np.count_nonzero(_bits(got) != want))
            except Exception as exc:  # a build failure or a refused dispatch
                _STATE.update(installed=False, probing=False, spelling=None, geometry=None)
                return _off(f"{type(exc).__name__}: {exc}", failure=True)
            finally:
                _STATE.update(probing=False, installed=False)
            if stage == "float32":
                differing[spelling] = wrong
            if wrong == 0:
                matching.append(spelling)
        if not matching:
            _STATE.update(spelling=None, geometry=None)
            return _off(
                f"no sum and rotation spelling reproduces the stock pooled-key update at {stage}",
                failure=True,
            )
        candidates = matching

    chosen = candidates[0]
    _STATE.update(installed=True, spelling=chosen, geometry=geometry, disabled_reason=None)
    report["float32_differing_values"] = {f"{s}/{r}": n for (s, r), n in differing.items()}
    others = [n for spelling, n in differing.items() if spelling != chosen]
    print(
        f"[mtplx] QSA pooled-key row on: 12 float32 and 12 {stages[1][0]} probe cases bit-equal to the "
        f"compiled stock update (sum {SUMS[chosen[0]]}, rotation {ROTATIONS[chosen[1]]}; the other spellings "
        f"differ on {min(others) if others else 0} to {max(others) if others else 0} float32 values)",
        flush=True,
    )
    report.update(engagement())
    return report


def _float32_twin(indexer: Any):
    """The indexer's pooled-key update with a float32 key norm, for the probe."""

    import mlx.nn as nn

    cls = type(indexer)
    twin_cls = type(
        "_Float32Twin",
        (),
        {
            "_extend_pooled_fixed": cls._extend_pooled_fixed,
            "_pooled_row_applies": cls._pooled_row_applies,
        },
    )
    twin = twin_cls()
    twin.ratio = int(indexer.ratio)
    twin.head_dim = int(indexer.head_dim)
    norm = nn.RMSNorm(int(indexer.head_dim), eps=indexer.k_layernorm.eps)
    norm.weight = indexer.k_layernorm.weight.astype(mx.float32)
    mx.eval(norm.weight)
    twin.k_layernorm = norm
    twin._inv_freq = indexer._inv_freq
    twin._rope_attention_scaling = indexer._rope_attention_scaling
    return twin


def engagement() -> dict:
    return {
        "installed": bool(_STATE["installed"]),
        "disabled_reason": _STATE["disabled_reason"],
        "geometry": _STATE["geometry"],
        "spelling": _STATE["spelling"],
        "engaged": tuple(sorted(_ENGAGED)),
        **dict(_COUNTS),
    }


def reset_state() -> None:
    _STATE.update(installed=False, disabled_reason=None, geometry=None, spelling=None, probing=False)
    _ENGAGED.clear()


def reset_for_tests() -> None:
    reset_state()
    for key in _COUNTS:
        _COUNTS[key] = 0


__all__ = [
    "ENV",
    "HEAD_DIM",
    "ROTATIONS",
    "SUMS",
    "engagement",
    "install",
    "note_engaged",
    "pooled_row",
    "reset_for_tests",
    "reset_state",
    "serves",
    "switched_on",
]
