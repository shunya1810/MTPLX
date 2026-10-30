"""Prefill FFN split between the GPU and the Neural Engine (M1 family), ``MTPLX_ANE_PREFILL=1``.

Why (2026-10-12, attn-s4): on an M1 Max the ANE runs int8-activation matmuls at ~9.7 TFLOPS, as fast
as the GPU's MMA peak, and the two barely slow each other (GPU 9.30 -> 9.15, ANE 9.84 -> 9.69 TFLOPS
side by side). Splash 1.3 splits each dense layer's FFN by intermediate channel between them and its
prefill runs 1.35-1.44x faster with the split (measured here, 7.7K-60K prompts).

What: for prefill chunks of at least ``MIN_ROWS`` rows, the trailing ``MTPLX_ANE_PREFILL_CHANNELS``
(default 7680) intermediate channels of every Qwen3Next MLP run on the ANE as one W16A8 program
(native_extensions/ane_ffn): the input rotated by a 512-block Hadamard and quantized to int8 per token,
the weights dequantized to fp16 and rotated alike, the intermediate rotated and quantized per token
inside the program (Splash's numerics). The GPU runs the leading channels meanwhile; their down
projections add up. Decode, verify and short chunks stay on the GPU.

Numerics: not exact. ANE part 1.57% relative RMS against fp32 on synthetic layers; on Youssofal
Speed-FP16 the simulated split changed 1/6 greedy 160-token answers (the KV 8-bit -> fp16 switch
changes 2/6).
"""

from __future__ import annotations

import logging
import math
import os
import sys
import threading
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)

MIN_ROWS = 512
# The program's functions: a chunk pads up to the next one (an evaluation costs its function's rows), so a
# 256-row grid keeps the padding of odd follow-up suffixes small.
_ROWS = list(range(512, 2048 + 1, 256))
_STATE: dict[str, Any] = {"installed": False, "ffn": None, "calls": 0, "fallbacks": 0, "error": None}
_LOCK = threading.Lock()


def enabled() -> bool:
    """``MTPLX_ANE_PREFILL``: "1"/"0" force; unset follows the M1-family default (on)."""

    raw = (os.environ.get("MTPLX_ANE_PREFILL") or "").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    from .cache_state import m1_long_context_defaults

    return m1_long_context_defaults()


def _ext():
    path = Path(__file__).resolve().parents[1] / "native_extensions" / "ane_ffn"
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    import mtplx_ane_ffn

    return mtplx_ane_ffn


def _had(a: mx.array) -> mx.array:
    s = a.shape
    return mx.hadamard_transform(a.reshape(*s[:-1], s[-1] // 512, 512), scale=1 / math.sqrt(512)).reshape(s)


def _qparams(lin: Any) -> tuple[mx.array, mx.array, mx.array | None, int, int]:
    return lin.weight, lin.scales, getattr(lin, "biases", None), int(lin.group_size), int(lin.bits)


def _f16(a: mx.array | None) -> mx.array | None:
    """The ANE staging kernels read quantization scales and biases as fp16: a bf16 model's are
    cast per call (a no-op for fp16 packs). Read as fp16 bits, bf16 scales turned every logit NaN."""

    return None if a is None else a.astype(mx.float16)


def _ane_q(p: tuple) -> list[mx.array | None]:
    """(weight, scales, biases) of one quantized matrix as the ANE staging kernels read them."""

    return [p[0], _f16(p[1]), _f16(p[2])]


def _layer(self: Any, cut: int) -> dict[str, Any]:
    cached = getattr(self, "_mtplx_ane_split", None)
    if cached is not None and cached["cut"] == cut:
        return cached
    g, u, d = _qparams(self.gate_proj), _qparams(self.up_proj), _qparams(self.down_proj)
    pd = 32 // d[4]
    entry = {
        "cut": cut,
        "g": (g[0][:cut], g[1][:cut], None if g[2] is None else g[2][:cut], g[3], g[4]),
        "u": (u[0][:cut], u[1][:cut], None if u[2] is None else u[2][:cut], u[3], u[4]),
        "d": (d[0][:, : cut // pd], d[1][:, : cut // d[3]], None if d[2] is None else d[2][:, : cut // d[3]], d[3], d[4]),
        "ga": (g[0][cut:], g[1][cut:], None if g[2] is None else g[2][cut:], g[3], g[4]),
        "ua": (u[0][cut:], u[1][cut:], None if u[2] is None else u[2][cut:], u[3], u[4]),
        "da": (d[0][:, cut // pd :], d[1][:, cut // d[3] :], None if d[2] is None else d[2][:, cut // d[3] :], d[3], d[4]),
    }
    self._mtplx_ane_split = entry
    return entry


def _qmm(x: mx.array, p: tuple) -> mx.array:
    w, s, b, gs, bits = p
    return mx.quantized_matmul(x, w, s, b, transpose=True, group_size=gs, bits=bits)


def _deq(p: tuple) -> mx.array:
    w, s, b, gs, bits = p
    return mx.dequantize(w, s, b, group_size=gs, bits=bits).astype(mx.float16)


def _prep_x(x2: mx.array) -> tuple[mx.array, mx.array]:
    xr = _had(x2.astype(mx.float32))
    s = mx.maximum(mx.max(mx.abs(xr), axis=-1), 1e-6) / 127.0
    xq = mx.clip(mx.round(xr / s[:, None]), -127, 127).astype(mx.int8)
    return xq, (s * 128.0).astype(mx.float16)


def _w8() -> bool:
    """``MTPLX_ANE_PREFILL_W8=1``: int8 weights on the ANE (per-row scales of the rotated rows, computed
    once per layer), W8A8 instead of W16A8."""

    return os.environ.get("MTPLX_ANE_PREFILL_W8", "0") == "1"


def _row_scales(self: Any, L: dict[str, Any]) -> list[mx.array]:
    cached = L.get("row_scales")
    if cached is None:
        def rs(p: tuple) -> mx.array:
            w = _had(mx.dequantize(*p[:3], group_size=p[3], bits=p[4]).astype(mx.float32))
            return mx.maximum(mx.max(mx.abs(w), axis=-1), 1e-8) / 127.0

        cached = [rs(L["ga"]), rs(L["ua"]), rs(L["da"])]
        mx.eval(*cached)
        L["row_scales"] = cached
    return cached


def _stage_x(ffn: Any, x2: mx.array) -> mx.array:
    """The chunk's input rotated, quantized and staged: one kernel for H = 5120, MLX ops otherwise."""

    if x2.shape[-1] == 5120 and hasattr(ffn, "stage_x"):
        return ffn.stage_x(x2)
    return ffn.stage_input(*_prep_x(x2))


def _disable(exc: BaseException) -> None:
    """Retire the ANE for the life of the process after a failure (the GPU computes its share)."""

    _STATE["disabled"] = True
    _STATE["error"] = f"{type(exc).__name__}: {exc}"
    logger.warning("[ane-prefill] ANE disabled after a failure, the GPU runs the whole FFN: %s", _STATE["error"])


def _gpu_tail(L: dict[str, Any], x2: mx.array) -> mx.array:
    """The ANE's channels of the FFN, computed on the GPU (exact)."""

    h = nn.silu(_qmm(x2, L["ga"])) * _qmm(x2, L["ua"])
    return _qmm(h, L["da"])


def install_gdn(model: Any) -> dict[str, Any]:
    """GDN in-projections split: the GPU computes in_proj_qkv's leading rows while the ANE computes its
    trailing ``MTPLX_ANE_PREFILL_GDN_QKV_TAIL`` rows (default 2048) and all of in_proj_z, in one program
    (``MTPLX_ANE_PREFILL_GDN=1``). in_proj_z's call then returns the ANE's rows for the same input."""

    ext = _ext()
    if ext.unavailable():
        return {"gdn": "unavailable"}
    tail = int(os.environ.get("MTPLX_ANE_PREFILL_GDN_QKV_TAIL", "2048"))
    layers = getattr(getattr(model, "language_model", model), "model", None)
    layers = getattr(layers, "layers", None) or getattr(getattr(model, "model", None), "layers", [])
    tagged = 0
    for layer in layers:
        gdn = getattr(layer, "linear_attn", None)
        qkv, z = getattr(gdn, "in_proj_qkv", None), getattr(gdn, "in_proj_z", None)
        if qkv is None or z is None or not hasattr(qkv, "scales") or not hasattr(z, "scales"):
            continue
        if (qkv.bits, qkv.group_size) != (z.bits, z.group_size) or getattr(qkv, "biases", None) is None:
            continue
        n_qkv = qkv.weight.shape[0]
        qkv._mtplx_ane_gdn = {"z": z, "cut": n_qkv - tail}
        z._mtplx_ane_gdn_z = True
        tagged += 1
    if not tagged:
        return {"gdn": "no layers"}
    cls = type(next(l.linear_attn.in_proj_qkv for l in layers if hasattr(getattr(l, "linear_attn", None), "in_proj_qkv")))
    orig = cls.__call__
    stash: dict[int, mx.array] = {}

    def call(self: Any, x: mx.array) -> mx.array:
        if getattr(self, "_mtplx_ane_gdn_z", False):
            hit = stash.pop(id(x), None)
            if hit is not None:
                return hit
            return orig(self, x)
        role = getattr(self, "_mtplx_ane_gdn", None)
        rows = math.prod(x.shape[:-1])
        if role is None or rows < MIN_ROWS:
            return orig(self, x)
        z = role["z"]
        cut = role["cut"]
        H = x.shape[-1]
        n_qkv, n_z = self.weight.shape[0], z.weight.shape[0]
        n_ane = (n_qkv - cut) + n_z
        lin = _STATE.get("gdn_lin")
        if lin is None:
            lin = ext.AneLinear(H, n_ane, _ROWS)
            _STATE["gdn_lin"] = lin
            logger.info("[ane-prefill] GDN program ready (%.1f s)", lin.compile_seconds)
        if rows > lin.max_rows:
            return orig(self, x)
        trace = os.environ.get("MTPLX_ANE_PREFILL_TRACE") == "1"
        if trace:
            import time as _time

            t0 = _time.perf_counter()
            mx.eval(x)
            t1 = _time.perf_counter()
        x2 = x.reshape(rows, H)
        xq, tx = _prep_x(x2)
        ws = [self.weight[cut:], _f16(self.scales[cut:]), _f16(self.biases[cut:]), z.weight, _f16(z.scales), _f16(z.biases)]
        st = lin.stage(xq, tx, ws, [0, n_qkv - cut], self.bits, self.group_size)
        mx.eval(st)
        gpu = mx.quantized_matmul(x2, self.weight[:cut], self.scales[:cut], self.biases[:cut], transpose=True,
                                  group_size=self.group_size, bits=self.bits)
        if trace:
            t2 = _time.perf_counter()
        mx.async_eval(gpu)
        lin.run(rows)
        if trace:
            t3 = _time.perf_counter()
            mx.eval(gpu)
            t4 = _time.perf_counter()
            acc = _STATE.setdefault("gtrace", [0.0, 0.0, 0.0, 0.0, 0])
            acc[0] += t1 - t0; acc[1] += t2 - t1; acc[2] += t3 - t2; acc[3] += t4 - t3; acc[4] += 1
            if acc[4] % 48 == 0:
                n = acc[4]
                print(f"[ane-prefill] GDN in_proj per layer: before {acc[0]/n*1e3:.1f} ms  prep+stage {acc[1]/n*1e3:.1f} ms  ane {acc[2]/n*1e3:.1f} ms  gpu-after {acc[3]/n*1e3:.1f} ms (rows {rows})", flush=True)
                _STATE["gtrace"] = [0.0, 0.0, 0.0, 0.0, 0]
        o = lin.output()[:, :rows].T.astype(x.dtype)
        _STATE["gdn_calls"] = _STATE.get("gdn_calls", 0) + 1
        stash[id(x)] = o[:, n_qkv - cut :].reshape(*x.shape[:-1], n_z)
        return mx.concatenate([gpu, o[:, : n_qkv - cut]], axis=-1).reshape(*x.shape[:-1], n_qkv)

    cls.__call__ = call
    return {"gdn": tagged, "qkv_tail": tail}


def _is_moe(model: Any) -> bool:
    """A mixture-of-experts model: its shared expert is a Qwen3NextMLP too, but
    the ANE program is built for the dense 27B's FFN shapes (it failed with
    "unsupported shape" on the 35B-A3B and disabled itself on the first call)."""

    if model is None:
        return False
    try:
        for _name, module in model.named_modules():
            if "SparseMoe" in type(module).__name__:
                return True
    except Exception:
        return False
    return False


def install(model: Any = None) -> dict[str, Any]:
    if _STATE["installed"]:
        return dict(_STATE)
    if _is_moe(model):
        _STATE["error"] = "mixture-of-experts model"
        return dict(_STATE)
    ext = _ext()
    why = ext.unavailable()
    if why:
        _STATE["error"] = why
        return dict(_STATE)
    from mlx_lm.models import qwen3_next as qn

    orig = qn.Qwen3NextMLP.__call__
    channels = int(os.environ.get("MTPLX_ANE_PREFILL_CHANNELS", "7680"))

    def call(self: Any, x: mx.array) -> mx.array:
        rows = math.prod(x.shape[:-1])
        if rows < MIN_ROWS or _STATE.get("disabled") or not hasattr(self.gate_proj, "scales"):
            return orig(self, x)
        H = x.shape[-1]
        inter = self.gate_proj.weight.shape[0]
        ffn = _STATE["ffn"]
        if ffn is None:
            with _LOCK:
                if _STATE["ffn"] is None:
                    try:
                        _STATE["ffn"] = ext.AneFfn(H, channels, _ROWS, w8=_w8())
                    except Exception as exc:
                        _disable(exc)
                        return orig(self, x)
                    logger.info("[ane-prefill] program ready (%.1f s)", _STATE["ffn"].compile_seconds)
                ffn = _STATE["ffn"]
        if rows > ffn.max_rows:
            # The FFN is row-wise: a wide chunk (4096 rows with the pipeline's chunk size, past its
            # context limit) runs as blocks the program holds.
            # Each block is evaluated before the next one runs: a block's ANE
            # share is a lazy read of the program's single output buffer, and
            # the next block's run overwrites it. Unevaluated, the first block
            # read the last block's rows (2026-10-20: past the pipeline's 32K
            # limit the 27B prefilled 4,096-row chunks this way; the first
            # 2,048 rows of every chunk got the wrong FFN share, the target's
            # top-1 on those rows fell from ~0.9 to ~0.3 and the draft's
            # acceptance collapsed on cold 250K prompts).
            flat = x.reshape(rows, H)
            step = ffn.max_rows
            parts = []
            for a in range(0, rows, step):
                part = call(self, flat[a : a + step][None])
                mx.eval(part)
                parts.append(part)
            return mx.concatenate([p.reshape(-1, H) for p in parts], axis=0).reshape(x.shape)
        cut = inter - channels
        L = _layer(self, cut)
        trace = os.environ.get("MTPLX_ANE_PREFILL_TRACE") == "1"
        if trace:
            import time as _time

            t0 = _time.perf_counter()
            mx.eval(x)
            t1 = _time.perf_counter()
        x2 = x.reshape(rows, H)
        ga, ua, da = L["ga"], L["ua"], L["da"]
        if ga[2] is None or ga[4] not in (4, 8) or not (ga[4] == ua[4] == da[4] and ga[3] == ua[3] == da[3]):
            xq, tx = _prep_x(x2)
            stage = [ffn.stage(xq, tx, _had(_deq(ga)), _had(_deq(ua)), _had(_deq(da)))]
        else:
            scales = _row_scales(self, L) if _w8() else []
            stage = [_stage_x(ffn, x2), ffn.stage_weights_q(_ane_q(ga), _ane_q(ua), _ane_q(da), ga[4], ga[3], scales)]
        mx.eval(*stage)
        if trace:
            t2 = _time.perf_counter()
        h = nn.silu(_qmm(x2, L["g"])) * _qmm(x2, L["u"])
        y_gpu = _qmm(h, L["d"])
        mx.async_eval(y_gpu)
        try:
            ffn.run(rows)
        except Exception as exc:  # the ANE refused or stalled: this call and every later one on the GPU
            _disable(exc)
            return (y_gpu + _gpu_tail(L, x2)).reshape(x.shape)
        if trace:
            t3 = _time.perf_counter()
            mx.eval(y_gpu)
            t4 = _time.perf_counter()
            acc = _STATE.setdefault("trace", [0.0, 0.0, 0.0, 0.0, 0])
            acc[0] += t1 - t0; acc[1] += t2 - t1; acc[2] += t3 - t2; acc[3] += t4 - t3; acc[4] += 1
            if acc[4] % 64 == 0:
                n = acc[4]
                print(f"[ane-prefill] per layer: before-mlp {acc[0]/n*1e3:.1f} ms  prep+stage {acc[1]/n*1e3:.1f} ms  ane {acc[2]/n*1e3:.1f} ms  gpu-part-after-ane {acc[3]/n*1e3:.1f} ms  (rows {rows})", flush=True)
                _STATE["trace"] = [0.0, 0.0, 0.0, 0.0, 0]
        o = ffn.output()
        y_ane = (o[:H, :rows] * o[H : H + 1, :rows]).T
        _STATE["calls"] += 1
        return (y_gpu + y_ane.astype(y_gpu.dtype)).reshape(x.shape)

    qn.Qwen3NextMLP.__call__ = call
    _STATE["installed"] = True
    _STATE["channels"] = channels
    if model is not None:
        # Build (and on a Mac's first run compile, 10-30 s) the programs at load, not inside the first
        # long prompt that needs them.
        try:
            text = getattr(model, "language_model", model)
            hidden = int(getattr(getattr(text, "args", None), "hidden_size", 0) or 0)
            if hidden and hidden % 2560 == 0:
                _STATE["ffn"] = ext.AneFfn(hidden, channels, _ROWS, w8=_w8())
                if pipeline_enabled():
                    pc = int(os.environ.get("MTPLX_ANE_PREFILL_PIPELINE_CHANNELS", "10240"))
                    _ffn_pair(hidden, pc)
                _STATE["warmed"] = True
        except Exception as exc:
            _disable(exc)
    if model is not None and os.environ.get("MTPLX_ANE_PREFILL_GDN", "0") == "1":
        _STATE["gdn"] = install_gdn(model)
    return {k: v for k, v in _STATE.items() if k not in ("ffn", "gdn_lin")}


def stats() -> dict[str, Any]:
    return {k: v for k, v in _STATE.items() if k != "ffn"}


# ---- two halves of a chunk pipelined across the GPU and the ANE (MTPLX_ANE_PREFILL_PIPELINE=1) ----
#
# Why (2026-10-12, attn-s4 N-9): with the FFN split alone the ANE idles while the GPU runs each layer's
# attention / GDN mixer. Splitting a prefill chunk into halves A and B and interleaving them per layer
# (B's mixer on the GPU while A's FFN share is on the ANE, then the reverse) keeps both busy, and lets
# the ANE take more channels. Outside the server, 8192 tokens: GPU only 155 tok/s, sequential split 190,
# pipelined 208 (7680 ch) .. 240 (11264 ch), final logits identical to the sequential split.
# Caches see A then B in every layer, as a single forward of A followed by B would.


def pipeline_enabled() -> bool:
    return enabled() and os.environ.get("MTPLX_ANE_PREFILL_PIPELINE", "1") != "0"


def wants_wide_chunks() -> bool:
    """The pipeline splits a chunk in halves: an automatic chunk width becomes 4,096 rows."""

    return bool(_STATE.get("installed")) and not _STATE.get("disabled") and pipeline_enabled()


def _ffn_pair(H: int, channels: int):
    pair = _STATE.get("ffn_pair")
    if pair is None:
        ext = _ext()
        pair = (ext.AneFfn(H, channels, _ROWS), ext.AneFfn(H, channels, _ROWS))
        _STATE["ffn_pair"] = pair
        logger.info("[ane-prefill] pipeline programs ready")
    return pair


def _mixer(layer: Any, h: mx.array, mask: Any, cache: Any) -> mx.array:
    normed = layer.input_layernorm(h)
    r = layer.linear_attn(normed, mask, cache) if layer.is_linear else layer.self_attn(normed, mask, cache)
    return h + r


def _ffn_begin(ffn: Any, layer: Any, h2: mx.array, channels: int):
    import threading as _th

    mlp = layer.mlp
    B, S, H = h2.shape
    rows = B * S
    x2 = layer.post_attention_layernorm(h2).reshape(rows, H)
    inter = mlp.gate_proj.weight.shape[0]
    L = _layer(mlp, inter - channels)
    ga, ua, da = L["ga"], L["ua"], L["da"]
    st = [_stage_x(ffn, x2), ffn.stage_weights_q(_ane_q(ga), _ane_q(ua), _ane_q(da), ga[4], ga[3])]
    mx.eval(*st)
    h = nn.silu(_qmm(x2, L["g"])) * _qmm(x2, L["u"])
    y_gpu = _qmm(h, L["d"])
    mx.async_eval(y_gpu)
    err: list[BaseException] = []

    def run() -> None:
        try:
            ffn.run(rows)
        except BaseException as exc:  # surfaced on join
            err.append(exc)

    th = _th.Thread(target=run, name="mtplx-ane-ffn", daemon=True)
    th.start()
    return th, err, ffn, y_gpu, rows, h2, L, x2


def _ffn_end(handle: tuple) -> mx.array:
    th, err, ffn, y_gpu, rows, h2, L, x2 = handle
    th.join()
    if err:
        _disable(err[0])
        return h2 + (y_gpu + _gpu_tail(L, x2)).reshape(h2.shape)
    H = h2.shape[-1]
    o = ffn.output()
    y = y_gpu + (o[:H, :rows] * o[H : H + 1, :rows]).T.astype(y_gpu.dtype)
    _STATE["pipeline_ffn_calls"] = _STATE.get("pipeline_ffn_calls", 0) + 1
    return h2 + y.reshape(h2.shape)


def pipelined_layers(inner: Any, h: mx.array, cache: list[Any], fa_mask: Any, ssm_mask: Any) -> mx.array | None:
    """The decoder stack over one prefill chunk as two pipelined halves; None when it does not apply."""

    if not pipeline_enabled() or _STATE.get("disabled") or cache is None or h.shape[0] != 1:
        return None
    S = h.shape[1]
    half = (S // 2 // 32) * 32
    if half < MIN_ROWS or S - half < MIN_ROWS or S - half > 2048:
        return None
    if not (isinstance(fa_mask, str) or fa_mask is None) or ssm_mask is not None:
        return None
    layers = inner.layers
    if not all(hasattr(l.mlp, "gate_proj") and hasattr(l.mlp.gate_proj, "scales") for l in layers):
        return None
    # Past MAX_CONTEXT keys one half's attention is a ~1 s GPU command at 128K, and with the ANE reading
    # memory beside it macOS can abort it for stalling the display (kIOGPUCommandBufferCallbackError
    # ImpactingInteractivity, seen once at 140K, 2026-10-13). There the gain is small anyway (attention
    # dominates), so long prefixes take the sequential split, whose attention runs alone. The quarter
    # split past the limit (~0.5 s commands) tripped it too at ~130K on a low battery, so the limit is
    # 32K, where a half's attention stays near 0.25 s.
    limit = int(os.environ.get("MTPLX_ANE_PREFILL_PIPELINE_MAX_CONTEXT", "32768"))
    try:
        offset = int(getattr(cache[inner.fa_idx], "offset", 0) or 0)
    except Exception:
        offset = limit + 1
    if offset + S > limit:
        # Experimental (MTPLX_ANE_PREFILL_PIPELINE_LONG=1): past the limit, pipeline quarter-size pieces in
        # two pairs, so each piece's attention is half as long a GPU command.
        if os.environ.get("MTPLX_ANE_PREFILL_PIPELINE_LONG", "0") == "1" and half >= 2 * MIN_ROWS and _STATE.get("_long_depth", 0) == 0:
            _STATE["_long_depth"] = 1
            try:
                first = pipelined_layers_forced(inner, h[:, :half], cache, fa_mask, ssm_mask)
                second = pipelined_layers_forced(inner, h[:, half:], cache, fa_mask, ssm_mask)
            finally:
                _STATE["_long_depth"] = 0
            _STATE["pipeline_long_pairs"] = _STATE.get("pipeline_long_pairs", 0) + 2
            return mx.concatenate([first, second], axis=1)
        _STATE["pipeline_long_context_skips"] = _STATE.get("pipeline_long_context_skips", 0) + 1
        return None
    return _pipeline_core(inner, h, cache, fa_mask, ssm_mask, half)


def pipelined_layers_forced(inner: Any, h: mx.array, cache: list[Any], fa_mask: Any, ssm_mask: Any) -> mx.array:
    """The pipeline over a piece regardless of the context limit (the long-context quarter split)."""

    S = h.shape[1]
    half = (S // 2 // 32) * 32
    out = _pipeline_core(inner, h, cache, fa_mask, ssm_mask, half)
    if out is None:
        raise RuntimeError("ANE pipeline unavailable for a forced piece")
    return out


def _pipeline_core(inner: Any, h: mx.array, cache: list[Any], fa_mask: Any, ssm_mask: Any, half: int) -> mx.array | None:
    layers = inner.layers
    channels = int(os.environ.get("MTPLX_ANE_PREFILL_PIPELINE_CHANNELS", "10240"))
    try:
        ffa, ffb = _ffn_pair(h.shape[-1], channels)
    except Exception as exc:
        _disable(exc)
        return None
    ha, hb = h[:, :half], h[:, half:]
    pend_b = None
    for layer, c in zip(layers, cache):
        mask = ssm_mask if layer.is_linear else fa_mask
        a = _ffn_begin(ffa, layer, _mixer(layer, ha, mask, c), channels)
        if pend_b is not None:
            hb = _ffn_end(pend_b)
        pend_b = _ffn_begin(ffb, layer, _mixer(layer, hb, mask, c), channels)
        ha = _ffn_end(a)
    hb = _ffn_end(pend_b)
    _STATE["pipeline_chunks"] = _STATE.get("pipeline_chunks", 0) + 1
    return mx.concatenate([ha, hb], axis=1)
