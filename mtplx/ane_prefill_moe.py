"""Prefill of mixture-of-experts Qwen3.5/3.6 models with the GDN input projections on the Neural Engine.

``MTPLX_ANE_PREFILL_MOE=1`` (M1 family, experimental).

Why: the routed experts change per token and cannot run as a fixed ANE program, so the dense build's FFN split
(``ane_prefill``) does not apply to a MoE. The GDN input projections (qkv and z, 2048 -> 12288 on the 35B-A3B)
are fixed matrices and ~27% of a prefill chunk's GPU time; the ANE computes them while the GPU runs the other
half chunk's experts.

How: a chunk of S rows runs as two halves A and B through every layer. In each GDN layer the ANE projects A
while the GPU finishes B's previous layer (its MoE), then projects B while the GPU runs A's rest of the layer
(conv, recurrence, norm, out_proj, MoE). The recurrent state goes A then B, as in a sequential prefill of the
two halves. Attention layers run A then B on the GPU. One ANE program holds every GDN layer's weights in its own
slot (``AneLinear(..., slots=n)``), staged once.

Numerics: not exact. The ANE part takes int8 activations rotated by a 512-block Hadamard (as ``ane_prefill``);
the small a/b gate projections stay on the GPU.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import mlx.core as mx

logger = logging.getLogger(__name__)

_STATE: dict[str, Any] = {"lin": None, "slots": {}, "staged": set(), "chunks": 0, "disabled": False, "error": None,
                          "t_run": 0.0, "t_wait": 0.0, "t_total": 0.0}


# Below ~2,560 rows a chunk is faster on the GPU alone: the two halves are narrow forwards, and every
# forward pays the dequantization of all experts whatever its width (35B-A3B on M1 Max, one chunk at
# offset 0, GPU-only vs pipeline: 2,048 rows 2.07 vs 2.19 s, 2,560 2.55 vs 2.53, 3,072 3.07 vs 2.87,
# 4,096 4.08 vs 3.72; a3b-s7).
_DEFAULT_MIN_ROWS = 2560
_DEFAULT_MIN_OFFSET = 0


def _env_rows(name: str, default: int) -> int:
    try:
        return int((os.environ.get(name) or "").strip() or default)
    except ValueError:
        return default


def enabled() -> bool:
    if (os.environ.get("MTPLX_ANE_PREFILL_MOE") or "").strip().lower() not in {"1", "true", "yes", "on"}:
        return False
    if _STATE["disabled"]:
        return False
    try:
        from .cache_state import m1_long_context_defaults

        return bool(m1_long_context_defaults())
    except Exception:
        return False


def _attn_enabled() -> bool:
    """``MTPLX_ANE_PREFILL_MOE_ATTN=1`` (default off): the attention layers' q/k/v on the ANE too.

    Off because it corrupted prefills: on the 35B-A3B a cold prefill of the first
    130,000 chars of mtplx/server/openai.py (34,179 tokens) ended in all-NaN logits
    (HTTP 500) on every one of 5 launches (m1.5 and later builds, with or without
    the 32K context cap; 129,000 chars too, while 131,000, 125,000 and <=120,000
    passed); with this part off, 0 of 5 at 125,000-131,000. Checking every
    projection's output for finiteness (which syncs each one) made the failure
    disappear, so it looks like an ordering race on the program's output buffer,
    not overflow. The part saved ~3% of a 31K prefill (30.5 vs 31.4 s).
    """

    return (os.environ.get("MTPLX_ANE_PREFILL_MOE_ATTN") or "0").strip() == "1"


def stats() -> dict[str, Any]:
    return {k: v for k, v in _STATE.items() if k not in {"lin", "slots", "staged", "attn_lin", "attn_slots"}} | {
        "slots": len(_STATE["slots"]),
        "staged": len(_STATE["staged"]),
    }


def _disable(exc: BaseException) -> None:
    _STATE["disabled"] = True
    _STATE["error"] = f"{type(exc).__name__}: {exc}"
    logger.warning("[ane-prefill-moe] disabled after a failure: %s", _STATE["error"])


def _qparams(lin: Any) -> tuple[Any, Any, Any, int, int]:
    return lin.weight, lin.scales, getattr(lin, "biases", None), int(lin.group_size), int(lin.bits)


def _is_moe_layer(layer: Any) -> bool:
    return "SparseMoe" in type(getattr(layer, "mlp", None)).__name__


def _program(inner: Any):
    """The one ANE projection program with a weight slot per GDN layer, or None (stock path)."""

    lin = _STATE["lin"]
    if lin is not None:
        return lin
    from .ane_prefill import _ROWS, _ext

    ext = _ext()
    if ext.unavailable():
        raise RuntimeError(f"ANE unavailable: {ext.unavailable()}")
    slots: dict[int, int] = {}
    shape = None
    for index, layer in enumerate(inner.layers):
        if not getattr(layer, "is_linear", False):
            continue
        gdn = layer.linear_attn
        qkv, z = gdn.in_proj_qkv, gdn.in_proj_z
        pq, pz = _qparams(qkv), _qparams(z)
        if pq[2] is None or pz[2] is None or pq[3:] != pz[3:]:
            raise RuntimeError("GDN projections are not one affine format")
        here = (int(pq[0].shape[0]), int(pz[0].shape[0]), pq[3], pq[4])
        if shape is None:
            shape = here
        elif shape != here:
            raise RuntimeError("GDN projection shapes differ between layers")
        slots[index] = len(slots)
    if shape is None:
        raise RuntimeError("no GDN layer")
    hidden = int(inner.layers[0].input_layernorm.weight.shape[0])
    lin = ext.AneLinear(hidden, shape[0] + shape[1], _ROWS, len(slots))
    logger.info("[ane-prefill-moe] program ready: %d slots, %.1f s", len(slots), lin.compile_seconds)
    _STATE["lin"] = lin
    _STATE["slots"] = slots
    _STATE["shape"] = shape
    return lin


class _StashedProjection:
    """Wraps an attention projection: a call with the input the ANE already
    projected returns the ANE's rows; any other call runs the wrapped module."""

    def __init__(self, module: Any):
        self._module = module
        self._key: Any = None
        self._value: Any = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._module, name)

    def give(self, x: mx.array, value: mx.array) -> None:
        self._key, self._value = x, value

    def __call__(self, x: mx.array) -> mx.array:
        if self._key is x:
            value, self._key, self._value = self._value, None, None
            return value
        return self._module(x)


def _attn_program(inner: Any):
    """The ANE projection program for the attention layers' q/k/v (one slot each), or None."""

    lin = _STATE.get("attn_lin")
    if lin is not None:
        return lin
    from .ane_prefill import _ROWS, _ext

    slots: dict[int, int] = {}
    shape = None
    for index, layer in enumerate(inner.layers):
        if getattr(layer, "is_linear", True):
            continue
        attn = layer.self_attn
        parts = [_qparams(getattr(attn, n)) for n in ("q_proj", "k_proj", "v_proj")]
        if any(pp[2] is None for pp in parts) or len({pp[3:] for pp in parts}) != 1:
            raise RuntimeError("attention projections are not one affine format")
        here = tuple(int(pp[0].shape[0]) for pp in parts) + parts[0][3:]
        if shape is None:
            shape = here
        elif shape != here:
            raise RuntimeError("attention projection shapes differ between layers")
        slots[index] = len(slots)
        for n in ("q_proj", "k_proj", "v_proj"):
            mod = getattr(attn, n)
            if not isinstance(mod, _StashedProjection):
                setattr(attn, n, _StashedProjection(mod))
    if shape is None:
        return None
    hidden = int(inner.layers[0].input_layernorm.weight.shape[0])
    lin = _ext().AneLinear(hidden, shape[0] + shape[1] + shape[2], _ROWS, len(slots))
    logger.info("[ane-prefill-moe] attention program ready: %d slots, %.1f s", len(slots), lin.compile_seconds)
    _STATE["attn_lin"] = lin
    _STATE["attn_slots"] = slots
    _STATE["attn_shape"] = shape
    return lin


def _attn_project(lin: Any, layer: Any, x: mx.array, slot: int) -> mx.array:
    from .ane_prefill import _prep_x

    attn = layer.self_attn
    rows = x.shape[0] * x.shape[1]
    xq, tx = _prep_x(x.reshape(rows, x.shape[-1]))
    key = ("attn", slot)
    if key in _STATE["staged"]:
        return lin.stage(xq, tx, [], [], 4, 64, slot)
    parts = [_qparams(getattr(attn, n)._module if isinstance(getattr(attn, n), _StashedProjection) else getattr(attn, n))
             for n in ("q_proj", "k_proj", "v_proj")]
    ws, offs, off = [], [], 0
    for pp in parts:
        ws += [pp[0], pp[1], pp[2]]
        offs.append(off)
        off += int(pp[0].shape[0])
    _STATE["staged"].add(key)
    return lin.stage(xq, tx, ws, offs, parts[0][4], parts[0][3], slot)


def _attn_give(layer: Any, x: mx.array, out: mx.array) -> None:
    nq, nk, _nv = _STATE["attn_shape"][:3]
    attn = layer.self_attn
    o = out[None]
    attn.q_proj.give(x, o[..., :nq])
    attn.k_proj.give(x, o[..., nq : nq + nk])
    attn.v_proj.give(x, o[..., nq + nk :])


def _gdn_rest(gdn: Any, inputs: mx.array, qkv: mx.array, z: mx.array, cache: Any, row0: int = 0) -> mx.array:
    """qwen3_5 ``GatedDeltaNet.__call__`` after the qkv / z projections (no mask, cache present).

    ``row0`` is this call's first row in the chunk: the in-forward boundary offsets
    (``qwen35_inforward_boundary``) are rows of the chunk, the pipeline calls this per half."""

    import mlx.nn as nn
    from mlx_lm.models import qwen3_5 as q5

    B, S, _ = inputs.shape
    z = z.reshape(B, S, gdn.num_v_heads, gdn.head_v_dim)
    b = gdn.in_proj_b(inputs)
    a = gdn.in_proj_a(inputs)
    if cache[0] is not None:
        conv_state = cache[0]
    else:
        conv_state = mx.zeros((B, gdn.conv_kernel_size - 1, gdn.conv_dim), dtype=inputs.dtype)
    conv_input = mx.concatenate([conv_state, qkv], axis=1)
    n_keep = gdn.conv_kernel_size - 1
    if cache.lengths is not None:
        ends = mx.clip(cache.lengths, 0, S)
        positions = (ends[:, None] + mx.arange(n_keep))[..., None]
        cache[0] = mx.take_along_axis(conv_input, positions, axis=1)
    else:
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
    from . import qwen35_inforward_boundary as _ifb

    offsets = _ifb.capture_offsets() if cache.lengths is None else ()
    if row0 in offsets:  # the seam between the two halves: the state the first half left
        _ifb.record_entry_state(cache, row0, conv_tail=conv_state)
    capture_at = tuple(p - row0 for p in offsets if 0 < p - row0 < S)
    if capture_at:
        out, state = _ifb.segmented_update(
            q5.gated_delta_update, q, k, v, a, b, gdn.A_log, gdn.dt_bias, cache[1],
            conv_stream=conv_input, n_keep=n_keep, cache=cache, capture_at=capture_at, row0=row0,
        )
    else:
        out, state = q5.gated_delta_update(q, k, v, a, b, gdn.A_log, gdn.dt_bias, cache[1], None, use_kernel=True)
    cache[1] = state
    cache.advance(S)
    out = gdn.norm(out, z)
    return gdn.out_proj(out.reshape(B, S, -1))


def _project(lin: Any, layer: Any, x: mx.array, slot: int) -> mx.array:
    """Stage ``x`` (and, the first time, the slot's weights) and evaluate the staging."""

    from .ane_prefill import _prep_x

    gdn = layer.linear_attn
    rows = x.shape[0] * x.shape[1]
    xq, tx = _prep_x(x.reshape(rows, x.shape[-1]))
    if slot in _STATE["staged"]:
        st = lin.stage(xq, tx, [], [], 4, 64, slot)
    else:
        q, z = _qparams(gdn.in_proj_qkv), _qparams(gdn.in_proj_z)
        st = lin.stage(xq, tx, [q[0], q[1], q[2], z[0], z[1], z[2]], [0, int(q[0].shape[0])], q[4], q[3], slot)
        _STATE["staged"].add(slot)
    return st


def _copy_output(lin: Any, rows: int, dtype: Any) -> mx.array:
    """The run's rows copied out of the program's one output buffer (evaluate before the next run)."""

    return mx.contiguous(lin.output()[:, :rows].T.astype(dtype))


def _split(out: mx.array) -> tuple[mx.array, mx.array]:
    n_qkv = _STATE["shape"][0]
    return out[None, :, :n_qkv], out[None, :, n_qkv:]


def pipelined_layers(inner: Any, h: mx.array, cache: list[Any], fa_mask: Any, ssm_mask: Any) -> mx.array | None:
    """The decoder stack over one prefill chunk with the GDN projections on the ANE; None when it does not apply."""

    if not enabled() or cache is None or h.shape[0] != 1:
        return None
    from .ane_prefill import _ROWS, MIN_ROWS

    S = int(h.shape[1])
    half = (S // 2 // 32) * 32
    if half < MIN_ROWS or S - half < MIN_ROWS or max(half, S - half) > max(_ROWS):
        return None
    if ssm_mask is not None or not (isinstance(fa_mask, str) or fa_mask is None):
        return None
    if S < _env_rows("MTPLX_ANE_PREFILL_MOE_MIN_ROWS", _DEFAULT_MIN_ROWS):
        return None
    layers = inner.layers
    if not any(_is_moe_layer(layer) for layer in layers):
        return None
    min_offset = _env_rows("MTPLX_ANE_PREFILL_MOE_MIN_OFFSET", _DEFAULT_MIN_OFFSET)
    if min_offset > 0:
        for layer, c in zip(layers, cache):
            if not layer.is_linear and c is not None:
                if int(getattr(c, "offset", 0) or 0) < min_offset:
                    return None
                break
    # Past MTPLX_ANE_PREFILL_MOE_MAX_CONTEXT keys the ANE path gains little (77K
    # turn re-prefill: 106 vs 111 s) and its extra working memory trips the
    # system-memory abort beside a desktop's apps (7 of 17 prefills failed vs 0
    # of 11 GPU-only, 2026-10-15); attention dominates there anyway.
    limit = int(os.environ.get("MTPLX_ANE_PREFILL_MOE_MAX_CONTEXT", "32768"))
    if limit > 0:
        offset = 0
        for layer, c in zip(layers, cache):
            if not layer.is_linear and c is not None:
                try:
                    offset = int(getattr(c, "offset", 0) or 0)
                except Exception:
                    offset = limit + 1
                break
        if offset + S > limit:
            if not _STATE.get("logged_context_cap"):
                _STATE["logged_context_cap"] = True
                print(f"[ane-prefill-moe] GPU-only past {limit} keys (offset {offset} + {S} rows)", flush=True)
            return None
    if not _STATE.get("logged_engaged"):
        _STATE["logged_engaged"] = True
        print(f"[ane-prefill-moe] engaged (offset {offset if limit > 0 else '?'}, {S} rows)", flush=True)
    try:
        lin = _program(inner)
        if _attn_enabled():
            _attn_program(inner)
    except Exception as exc:
        _disable(exc)
        return None
    slots = _STATE["slots"]
    dtype = h.dtype
    ha, hb = h[:, :half], h[:, half:]
    # Both halves run the same program function (the larger half's): alternating two functions of the
    # program between the halves costs more than the padding (3,583 rows: 4.05 s vs 3.3 s, a3b-s7).
    run_rows = max(half, S - half) if os.environ.get("MTPLX_ANE_PREFILL_MOE_SAME_FN", "1") != "0" else 0
    copied = None  # the last run's output copy, evaluated before the next run reuses the buffer
    try:
        for index, (layer, c) in enumerate(zip(layers, cache)):
            if not layer.is_linear:
                alin = _STATE.get("attn_lin") if _attn_enabled() else None
                if alin is None:
                    ha = layer(ha, mask=fa_mask, cache=c)
                    hb = layer(hb, mask=fa_mask, cache=c)
                    continue
                aslot = _STATE["attn_slots"][index]
                xa = layer.input_layernorm(ha)
                mx.eval(_attn_project(alin, layer, xa, aslot), *(() if copied is None else (copied,)))
                mx.async_eval(hb)
                alin.run(run_rows or xa.shape[1], aslot)
                out_a = _copy_output(alin, xa.shape[1], dtype)
                _attn_give(layer, xa, out_a)
                ha = ha + layer.self_attn(xa, mask=fa_mask, cache=c)
                ha = ha + layer.mlp(layer.post_attention_layernorm(ha))
                xb = layer.input_layernorm(hb)
                mx.eval(_attn_project(alin, layer, xb, aslot), out_a)
                mx.async_eval(ha)
                alin.run(run_rows or xb.shape[1], aslot)
                copied = _copy_output(alin, xb.shape[1], dtype)
                _attn_give(layer, xb, copied)
                hb = hb + layer.self_attn(xb, mask=fa_mask, cache=c)
                hb = hb + layer.mlp(layer.post_attention_layernorm(hb))
                continue
            slot = slots[index]
            # A's projection on the ANE while the GPU runs B's previous layer.
            xa = layer.input_layernorm(ha)
            _t0 = time.perf_counter()
            mx.eval(_project(lin, layer, xa, slot), *(() if copied is None else (copied,)))
            _t1 = time.perf_counter()
            mx.async_eval(hb)
            lin.run(run_rows or xa.shape[1], slot)
            _t2 = time.perf_counter()
            _STATE["t_wait"] += _t1 - _t0
            _STATE["t_run"] += _t2 - _t1
            out_a = _copy_output(lin, xa.shape[1], dtype)
            qkv_a, z_a = _split(out_a)
            ra = _gdn_rest(layer.linear_attn, xa, qkv_a, z_a, c)
            ha = ha + ra
            ha = ha + layer.mlp(layer.post_attention_layernorm(ha))
            # B's projection on the ANE while the GPU runs A's rest of this layer.
            xb = layer.input_layernorm(hb)
            _t0 = time.perf_counter()
            mx.eval(_project(lin, layer, xb, slot), out_a)
            _t1 = time.perf_counter()
            mx.async_eval(ha)
            lin.run(run_rows or xb.shape[1], slot)
            _t2 = time.perf_counter()
            _STATE["t_wait"] += _t1 - _t0
            _STATE["t_run"] += _t2 - _t1
            copied = _copy_output(lin, xb.shape[1], dtype)
            qkv_b, z_b = _split(copied)
            rb = _gdn_rest(layer.linear_attn, xb, qkv_b, z_b, c, row0=half)
            hb = hb + rb
            hb = hb + layer.mlp(layer.post_attention_layernorm(hb))
    except Exception as exc:
        _disable(exc)
        raise
    _STATE["chunks"] += 1
    if os.environ.get("MTPLX_ANE_PREFILL_MOE_TRACE") == "1":
        print(f"[ane-prefill-moe] S={S} half={half} run={_STATE['t_run']:.3f}s wait={_STATE['t_wait']:.3f}s (cumulative)", flush=True)
    return mx.concatenate([ha, hb], axis=1)
