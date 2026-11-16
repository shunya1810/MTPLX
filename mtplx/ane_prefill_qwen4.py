"""Flash-Next (qwen4_exp) prefill: the GDN input projection split between the GPU and the Neural Engine.

``MTPLX_ANE_PREFILL_QWEN4_GDN`` (M1 family; "1"/"0" force).

Why: Flash-Next's GDN layers project each row 2560 -> 16480 (qkv | z | b | a, one fused 8-bit matrix), the
largest dense matmul of a prefill chunk (M1 Max, 2,048 rows: 22.7 ms a layer, 36 layers, ~14% of the
chunk) and it runs at the GPU's MMA peak. The ANE runs int8-activation matmuls about as fast and the two
barely slow each other (``ane_prefill``), so the GPU computes the leading qkv rows plus b / a while the ANE
computes the trailing ``MTPLX_ANE_QWEN4_GDN_ROWS`` qkv|z rows of the same input, as ``ane_prefill.
install_gdn`` does for the separate qkv / z modules of the Qwen3.5 family. The chunk is not split, so the
routed experts keep one wide forward (the A3B's half-chunk pipeline would run them twice, and on
Flash-Next two 1,024-row expert passes cost 63 ms against 47 for one 2,048-row pass).

Weights are staged per call (one layer's rows resident on the ANE, no per-layer slots): no standing memory.

Numerics: not exact. The ANE share takes int8 activations rotated by a 512-block Hadamard (W16A8,
``ane_prefill``'s numerics); the GPU share is the stock quantized matmul.
"""

from __future__ import annotations

import logging
import math
import os
import time
from typing import Any

import mlx.core as mx

logger = logging.getLogger(__name__)

_STATE: dict[str, Any] = {"lins": {}, "calls": 0, "disabled": False, "error": None,
                          "t": {}}


def enabled() -> bool:
    raw = (os.environ.get("MTPLX_ANE_PREFILL_QWEN4_GDN") or "").strip().lower()
    if raw in {"0", "false", "no", "off"} or _STATE["disabled"]:
        return False
    if raw in {"1", "true", "yes", "on"}:
        return True
    # Default off until a whole-model receipt (speed and perplexity) says otherwise.
    return False


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name) or default)
    except ValueError:
        return default


def attn_enabled() -> bool:
    """``MTPLX_ANE_PREFILL_QWEN4_ATTN=1`` (default off): the attention q projection's tail on the ANE too."""
    return enabled() and (os.environ.get("MTPLX_ANE_PREFILL_QWEN4_ATTN") or "0").strip() == "1"


def stats() -> dict[str, Any]:
    return {k: v for k, v in _STATE.items() if k != "lins"}


def _disable(exc: BaseException) -> None:
    _STATE["disabled"] = True
    _STATE["error"] = f"{type(exc).__name__}: {exc}"
    logger.warning("[ane-prefill-qwen4] disabled after a failure: %s", _STATE["error"])


def split_projection(fused: Any, x: mx.array, *, ane_end: int, ane_rows: int, tag: str):
    """The outputs of ``fused`` (a row-concatenated affine matrix with ``_splits``) for prefill rows ``x``:
    rows [ane_end - ane_rows, ane_end) on the ANE, the rest on the GPU at the same time; None (stock path)
    when it does not apply."""

    from . import ane_prefill as ap

    rows = math.prod(x.shape[:-1])
    if not enabled() or rows < ap.MIN_ROWS or x.dtype not in (mx.float16, mx.bfloat16):
        return None
    splits = list(getattr(fused, "_splits", ()))
    if getattr(fused, "mode", "affine") != "affine" or getattr(fused, "biases", None) is None:
        return None
    total = int(fused.weight.shape[0])
    cut = ane_end - ane_rows
    if not (0 < ane_rows and 0 <= cut and ane_end <= total) or ane_rows % 256:
        return None
    H = int(x.shape[-1])
    try:
        # MTPLX_ANE_QWEN4_SLOTS=N: keep the weights of up to N projections resident on the ANE (one slot
        # each, staged once) instead of staging them per call.
        n_slots = _env_int("MTPLX_ANE_QWEN4_SLOTS", 0) if tag == "gdn" else 0
        key = (H, ane_rows, n_slots)
        lin = _STATE["lins"].get(key)
        if lin is None:
            ext = ap._ext()
            why = ext.unavailable()
            if why:
                raise RuntimeError(f"ANE unavailable: {why}")
            lin = ext.AneLinear(H, ane_rows, ap._ROWS, max(1, n_slots))
            _STATE["lins"][key] = lin
            _STATE.setdefault("slot_of", {})[key] = {}
            logger.info("[ane-prefill-qwen4] %s program ready (%d rows, %.1f s)", tag, ane_rows, lin.compile_seconds)
        slot, staged = 0, False
        if n_slots:
            table = _STATE["slot_of"][key]
            entry = table.get(id(fused))
            if entry is None and len(table) < n_slots:
                entry = table[id(fused)] = [len(table), False]
            if entry is not None:
                slot, staged = entry[0], entry[1]
                entry[1] = True
        trace = os.environ.get("MTPLX_ANE_PREFILL_TRACE") == "1"
        if trace:
            t0 = time.perf_counter()
            mx.eval(x)
            t1 = time.perf_counter()
        x2 = x.reshape(rows, H)
        w, s, b = fused.weight, fused.scales, fused.biases
        kw = dict(transpose=True, group_size=int(fused.group_size), bits=int(fused.bits))
        gpu = []
        if cut > 0:
            gpu.append(mx.quantized_matmul(x2, w[:cut], s[:cut], b[:cut], **kw))
        tail = mx.quantized_matmul(x2, w[ane_end:], s[ane_end:], b[ane_end:], **kw) if ane_end < total else None
        gpu_all = gpu + ([] if tail is None else [tail])
        # A chunk wider than the program (the server's 4,096-row Flash-Next chunks) runs as blocks of at most
        # max_rows. The program has one output buffer: each block's copy is evaluated before the next run.
        step = int(lin.max_rows)
        mids: list[mx.array] = []
        t2 = t3 = None
        for i, a in enumerate(range(0, rows, step)):
            rb = min(step, rows - a)
            xq, tx = ap._prep_x(x2[a : a + rb])
            ws = [] if (staged or i > 0) else [w[cut:ane_end], ap._f16(s[cut:ane_end]), ap._f16(b[cut:ane_end])]
            st = lin.stage(xq, tx, ws, [0] if ws else [], int(fused.bits), int(fused.group_size), slot)
            mx.eval(st, *mids[-1:])
            if i == 0:
                if trace:
                    t2 = time.perf_counter()
                mx.async_eval(*gpu_all)
            lin.run(max(rb, ap._ROWS[0]) if rb < ap._ROWS[0] else rb, slot)
            mids.append(mx.contiguous(lin.output()[:, :rb].T.astype(x.dtype)))
        if trace:
            t3 = time.perf_counter()
            mx.eval(*gpu_all, *mids)
            t4 = time.perf_counter()
            acc = _STATE["t"].setdefault(tag, [0.0, 0.0, 0.0, 0.0, 0])
            acc[0] += t1 - t0; acc[1] += t2 - t1; acc[2] += t3 - t2; acc[3] += t4 - t3; acc[4] += 1
            if acc[4] % 12 == 0:
                n = acc[4]
                print(f"[ane-prefill-qwen4] {tag} per call: before {acc[0]/n*1e3:.1f} ms  prep+stage "
                      f"{acc[1]/n*1e3:.1f} ms  ane {acc[2]/n*1e3:.1f} ms  gpu-after {acc[3]/n*1e3:.1f} ms "
                      f"(rows {rows}, ane rows {ane_rows})", flush=True)
                _STATE["t"][tag] = [0.0, 0.0, 0.0, 0.0, 0]
        mid = mids[0] if len(mids) == 1 else mx.concatenate(mids, axis=0)
        parts = gpu + [mid] + ([] if tail is None else [tail])
        y = mx.concatenate(parts, axis=-1).reshape(*x.shape[:-1], -1)
        _STATE["calls"] += 1
        return mx.split(y, splits, axis=-1) if splits else [y]
    except Exception as exc:
        _disable(exc)
        return None


def gdn_in_proj(fused: Any, x: mx.array):
    """(qkv, z, b, a) for a GDN layer's prefill rows: the trailing qkv|z rows on the ANE."""
    splits = list(getattr(fused, "_splits", ()))
    if len(splits) != 3:
        return None
    return split_projection(fused, x, ane_end=int(splits[1]),
                            ane_rows=_env_int("MTPLX_ANE_QWEN4_GDN_ROWS", 8192), tag="gdn")


def attn_qkv(fused: Any, x: mx.array):
    """(q, k, v[, index qk]) for an attention layer's prefill rows: the q projection's tail on the ANE."""
    if not attn_enabled():
        return None
    splits = list(getattr(fused, "_splits", ()))
    if len(splits) < 2:
        return None
    return split_projection(fused, x, ane_end=int(splits[0]),
                            ane_rows=_env_int("MTPLX_ANE_QWEN4_ATTN_ROWS", 6144), tag="attn")
