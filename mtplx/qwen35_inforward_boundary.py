"""In-forward GDN boundary captures for the mlx-lm Qwen3.5/3.6 MoE models.

``MTPLX_QWEN35_INFORWARD_BOUNDARY=1`` (the M1 MoE launch lane sets it).

Why: a cold prefill records recurrent-state boundaries near the prompt end
(``generation._prefill_spans_with_tail_grid``) so a follow-up that diverges in
the tail resumes from one of them. Without the two model hooks
(``boundary_capture_scope`` / ``take_boundary_captures``) the prefill loop CUTS
its last chunk at those positions, and narrow forwards are slow on this model:
all 256 experts of every layer are active from ~256 rows on, so a forward pays
their dequantization whatever its width (35B-A3B on M1 Max: MoE 1.3 ms/token
at 256 rows, 0.57 at 1,024, 0.40 at 4,096). A 2,093-token prompt ran as
1,024 + 1,024 + 45 rows: 2.36 s against 2.02 s for 2,012 tokens (a3b-s7).

How: the gated-delta recurrence is sequential in the token, so running it as
[0, p) then [p, S) from the returned state is what a forward that ends at p
computes; the conv tail at p is a slice of the pre-conv stream the layer
already holds. This is the scheme ``models/qwen4_exp`` uses for its own
recurrent layers, applied to mlx-lm's ``qwen3_5.GatedDeltaNet`` and to the
Neural Engine pipeline's two halves (``ane_prefill_moe``).

Scope: MoE models only. The dense build's ANE pipeline (``ane_prefill``) runs
a chunk as two halves through ``GatedDeltaNet.__call__`` and would read the
chunk's offsets as rows of each half.
"""

from __future__ import annotations

import contextlib
import contextvars
import os
from typing import Any, Optional

import mlx.core as mx

_CAPTURE: contextvars.ContextVar[Optional[tuple]] = contextvars.ContextVar(
    "qwen35_boundary_capture", default=None
)
_ATTR = "_mtplx_boundary_captures"
_STATE: dict[str, Any] = {"installed": False, "original": None}


def enabled() -> bool:
    return (os.environ.get("MTPLX_QWEN35_INFORWARD_BOUNDARY") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


@contextlib.contextmanager
def boundary_capture_scope(offsets):
    """Ask the GDN layers of the next forward to record their state at each
    row offset (0 < offset < rows) on their cache entry."""

    cleaned = tuple(sorted({int(offset) for offset in offsets if int(offset) > 0}))
    token = _CAPTURE.set(cleaned or None)
    try:
        yield
    finally:
        _CAPTURE.reset(token)


def capture_offsets() -> tuple:
    """The armed row offsets of the current forward (rows of the whole chunk)."""

    return _CAPTURE.get() or ()


def _record(cache: Any, offset: int, arrays: tuple) -> None:
    store = getattr(cache, _ATTR, None)
    if store is None:
        store = {}
        setattr(cache, _ATTR, store)
    store[int(offset)] = arrays


def record_entry_state(cache: Any, offset: int, *, conv_tail: mx.array) -> None:
    """Record the state a call STARTS from as the boundary at ``offset`` (the
    seam between the ANE pipeline's two halves). ``conv_tail`` is the entry's
    conv state as the call read it (the entry already holds the next one)."""

    if cache[1] is None:
        return
    _record(cache, offset, (conv_tail, cache[1]))


def take_boundary_captures(cache: list) -> dict:
    """Pop what the last forward captured: ``{offset: [state list or None per
    cache entry]}``, each state list laid out as that entry's ``state`` (conv
    tail, delta state). An offset is returned only when every recurrent entry
    captured it, so a partial capture can never become a boundary."""

    from mlx_lm.models.cache import ArraysCache

    per_entry = []
    offsets: Optional[set] = None
    for entry in cache:
        store = getattr(entry, _ATTR, None)
        if store is not None:
            setattr(entry, _ATTR, None)
        recurrent = isinstance(entry, ArraysCache)
        per_entry.append((store or {}, recurrent))
        if recurrent:
            have = set((store or {}).keys())
            offsets = have if offsets is None else offsets & have
    result: dict = {}
    for offset in sorted(offsets or ()):
        result[offset] = [list(store[offset]) if recurrent else None for store, recurrent in per_entry]
    return result


def segmented_update(
    update: Any,
    q: mx.array,
    k: mx.array,
    v: mx.array,
    a: mx.array,
    b: mx.array,
    A_log: Any,
    dt_bias: Any,
    state: Any,
    *,
    conv_stream: mx.array,
    n_keep: int,
    cache: Any,
    capture_at: tuple,
    row0: int = 0,
):
    """``update`` over [0, S) in segments ending at each of ``capture_at``
    (rows of this call), recording (conv tail, state) at ``row0 + edge``."""

    S = int(q.shape[1])
    pieces = []
    seg_start = 0
    for edge in (*capture_at, S):
        piece, state = update(
            q[:, seg_start:edge],
            k[:, seg_start:edge],
            v[:, seg_start:edge],
            a[:, seg_start:edge],
            b[:, seg_start:edge],
            A_log,
            dt_bias,
            state,
            None,
            use_kernel=True,
        )
        pieces.append(piece)
        if edge < S:
            _record(cache, row0 + edge, (mx.contiguous(conv_stream[:, edge : edge + n_keep, :]), state))
        seg_start = edge
    return (pieces[0] if len(pieces) == 1 else mx.concatenate(pieces, axis=1)), state


def _call_with_captures(gdn: Any, inputs: mx.array, cache: Any, capture_at: tuple) -> mx.array:
    """``qwen3_5.GatedDeltaNet.__call__`` (no mask, cache present, eval) with
    the recurrence cut at ``capture_at``."""

    import mlx.nn as nn
    from mlx_lm.models import qwen3_5 as q5

    B, S, _ = inputs.shape
    qkv = gdn.in_proj_qkv(inputs)
    z = gdn.in_proj_z(inputs).reshape(B, S, gdn.num_v_heads, gdn.head_v_dim)
    b = gdn.in_proj_b(inputs)
    a = gdn.in_proj_a(inputs)
    if cache[0] is not None:
        conv_state = cache[0]
    else:
        conv_state = mx.zeros((B, gdn.conv_kernel_size - 1, gdn.conv_dim), dtype=inputs.dtype)
    conv_input = mx.concatenate([conv_state, qkv], axis=1)
    n_keep = gdn.conv_kernel_size - 1
    cache[0] = mx.contiguous(conv_input[:, -n_keep:, :])
    conv_out = nn.silu(gdn.conv1d(conv_input))
    q, k, v = [
        t.reshape(B, S, h, d)
        for t, h, d in zip(
            mx.split(conv_out, [gdn.key_dim, 2 * gdn.key_dim], -1),
            [gdn.num_k_heads, gdn.num_k_heads, gdn.num_v_heads],
            [gdn.head_k_dim, gdn.head_k_dim, gdn.head_v_dim],
        )
    ]
    inv_scale = k.shape[-1] ** -0.5
    q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
    k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)
    out, state = segmented_update(
        q5.gated_delta_update, q, k, v, a, b, gdn.A_log, gdn.dt_bias, cache[1],
        conv_stream=conv_input, n_keep=n_keep, cache=cache, capture_at=capture_at,
    )
    cache[1] = state
    cache.advance(S)
    out = gdn.norm(out, z)
    return gdn.out_proj(out.reshape(B, S, -1))


def install(model: Any) -> dict:
    """Patch ``qwen3_5.GatedDeltaNet.__call__`` and hang the two hooks on the model. Idempotent."""

    if not enabled():
        return {"installed": False, "reason": "disabled"}
    try:
        from mlx_lm.models import qwen3_5 as q5
    except Exception as exc:  # pragma: no cover - environment without mlx_lm
        return {"installed": False, "reason": f"mlx_lm import failed: {exc}"}
    inner = getattr(model, "language_model", model)
    body = getattr(inner, "model", inner)
    layers = getattr(body, "layers", None) or []
    gdn_layers = [layer.linear_attn for layer in layers if getattr(layer, "is_linear", False)]
    if not gdn_layers or not all(isinstance(g, q5.GatedDeltaNet) for g in gdn_layers):
        return {"installed": False, "reason": "not a qwen3_5 GatedDeltaNet stack"}
    from .ane_prefill_moe import _is_moe_layer

    if not any(_is_moe_layer(layer) for layer in layers):
        return {"installed": False, "reason": "dense model (its ANE pipeline halves are not covered)"}
    if not _STATE["installed"]:
        original = q5.GatedDeltaNet.__call__

        def patched(self, inputs, mask=None, cache=None):
            offsets = _CAPTURE.get()
            if (
                not offsets
                or cache is None
                or mask is not None
                or int(inputs.shape[0]) != 1
                or getattr(cache, "lengths", None) is not None
                or getattr(self, "sharding_group", None) is not None
                or self.training
            ):
                return original(self, inputs, mask, cache)
            rows = int(inputs.shape[1])
            capture_at = tuple(offset for offset in offsets if 0 < offset < rows)
            if not capture_at:
                return original(self, inputs, mask, cache)
            return _call_with_captures(self, inputs, cache, capture_at)

        q5.GatedDeltaNet.__call__ = patched
        _STATE["installed"] = True
        _STATE["original"] = original
    model.boundary_capture_scope = boundary_capture_scope
    model.take_boundary_captures = take_boundary_captures
    return {"installed": True, "gdn_layers": len(gdn_layers)}
