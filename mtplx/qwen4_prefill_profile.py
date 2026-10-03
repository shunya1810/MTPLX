"""``MTPLX_QWEN4_PREFILL_PROFILE=1`` -- where a Flash-Next prefill chunk spends its time.

A measuring instrument, never a product lane.  When the env is set at model
load, the Flash-Next decoder components are wrapped so that every call at
prefill width (rows >= ``MTPLX_QWEN4_PREFILL_PROFILE_MIN_ROWS``, default 256)
evaluates its inputs, runs, evaluates its outputs, and adds the wall time to a
per-component total.  The barriers serialize the graph, so absolute chunk time
is higher than a normal run; the shares are what the instrument is for.  The
totals are printed to stderr at process exit and after every
``MTPLX_QWEN4_PREFILL_PROFILE_EVERY`` profiled MoE-block calls (default 48,
one decoder stack, so one line per prefill chunk).

Decode and verify widths are left untouched (the row gate), so a profiled
daemon still decodes on its normal lanes.
"""

from __future__ import annotations

import atexit
import json
import os
import sys
import time
from typing import Any

_TOTALS: dict[str, float] = {}
_CALLS: dict[str, int] = {}
_ROWS: dict[str, int] = {}
_WIDTHS: dict[int, int] = {}
_INSTALLED = False


def enabled() -> bool:
    return os.environ.get("MTPLX_QWEN4_PREFILL_PROFILE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _min_rows() -> int:
    try:
        return max(2, int(os.environ.get("MTPLX_QWEN4_PREFILL_PROFILE_MIN_ROWS", "256")))
    except ValueError:
        return 256


def _report_every() -> int:
    try:
        return max(1, int(os.environ.get("MTPLX_QWEN4_PREFILL_PROFILE_EVERY", "48")))
    except ValueError:
        return 48


def snapshot() -> dict[str, Any]:
    total = sum(v for k, v in _TOTALS.items() if not k.startswith("moe.")) or 1.0
    rows = {
        name: {
            "s": round(_TOTALS[name], 4),
            "calls": _CALLS.get(name, 0),
            "share": round(_TOTALS[name] / total, 4) if not name.startswith("moe.") else None,
            "us_per_row": round(1e6 * _TOTALS[name] / max(1, _ROWS.get(name, 0)), 3),
        }
        for name in sorted(_TOTALS, key=lambda key: -_TOTALS[key])
    }
    return {
        "profiled_total_s": round(total, 4),
        "components": rows,
        "moe_calls_by_rows": {str(k): v for k, v in sorted(_WIDTHS.items())},
    }


def _report() -> None:
    if _TOTALS:
        print("qwen4_prefill_profile=" + json.dumps(snapshot()), file=sys.stderr, flush=True)


def _arrays(tree: Any) -> list[Any]:
    import mlx.core as mx

    out: list[Any] = []
    stack = [tree]
    while stack:
        item = stack.pop()
        if isinstance(item, mx.array):
            out.append(item)
        elif isinstance(item, (list, tuple)):
            stack.extend(item)
        elif isinstance(item, dict):
            stack.extend(item.values())
    return out


def _rows_of(x: Any) -> int:
    shape = getattr(x, "shape", None)
    if not shape or len(shape) < 2:
        return 0
    rows = 1
    for dim in shape[:-1]:
        rows *= int(dim)
    return rows


def _timed(name: str, fn, row_arg: int = 0, rows_fn=None):
    import mlx.core as mx

    floor = _min_rows()

    def wrapper(*args, **kwargs):
        probe = args[row_arg] if len(args) > row_arg else None
        rows = _rows_of(probe) if rows_fn is None else rows_fn(probe)
        if rows < floor:
            return fn(*args, **kwargs)
        mx.eval(_arrays(args) + _arrays(kwargs))
        started = time.perf_counter()
        out = fn(*args, **kwargs)
        mx.eval(_arrays(out))
        elapsed = time.perf_counter() - started
        _TOTALS[name] = _TOTALS.get(name, 0.0) + elapsed
        _CALLS[name] = _CALLS.get(name, 0) + 1
        _ROWS[name] = _ROWS.get(name, 0) + rows
        if name == "moe":
            _WIDTHS[rows] = _WIDTHS.get(rows, 0) + 1
        if name == "moe" and _CALLS[name] % _report_every() == 0:
            # The MoE block closes a decoder layer; a multiple of the layer
            # count is the end of one prefill chunk.  The daemon is stopped
            # by a signal, so exit handlers cannot be the only reporter.
            _report()
        return out

    return wrapper


_OVERLAP: dict[int, list[int]] = {}


def overlap_enabled() -> bool:
    return os.environ.get("MTPLX_QWEN4_EXPERT_OVERLAP_PROBE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _overlap_report() -> None:
    if _OVERLAP:
        rows = {
            str(width): {
                "calls": calls,
                "slots_per_call": width_slots / max(1, calls),
                "unique_per_call": round(unique / max(1, calls), 3),
                "unique_share": round(unique / max(1, width_slots), 4),
            }
            for width, (calls, width_slots, unique) in sorted(_OVERLAP.items())
        }
        print("qwen4_expert_overlap=" + json.dumps(rows), file=sys.stderr, flush=True)


def _install_overlap_probe(family) -> None:
    """MTPLX_QWEN4_EXPERT_OVERLAP_PROBE=1 (eager verify only: it syncs per layer).

    For forwards of 2 to 8 rows, count how many DISTINCT routed experts the
    rows select in each MoE block.  ``unique_share`` under 1.0 is expert
    weight traffic a row-grouped verify kernel would not read twice.
    """

    import mlx.core as mx
    import numpy as np

    original = family.SparseMoeBlock.__call__

    def wrapper(self, x):
        rows = _rows_of(x)
        if 2 <= rows <= 8:
            gates = mx.softmax(self.gate(x), axis=-1, precise=True)
            picks = mx.argpartition(gates, kth=-self.top_k, axis=-1)[..., -self.top_k :]
            try:
                chosen = np.asarray(picks).reshape(-1)
            except Exception:  # inside an mx.compile trace: nothing to read
                return original(self, x)
            entry = _OVERLAP.setdefault(rows, [0, 0, 0])
            entry[0] += 1
            entry[1] += int(chosen.size)
            entry[2] += int(np.unique(chosen).size)
            if entry[0] % 1200 == 0:
                _overlap_report()
        return original(self, x)

    family.SparseMoeBlock.__call__ = wrapper
    atexit.register(_overlap_report)
    print("[qwen4-expert-overlap-probe] installed", file=sys.stderr, flush=True)


def install() -> bool:
    """Wrap the Flash-Next components once per process."""

    global _INSTALLED
    if _INSTALLED or not (enabled() or overlap_enabled()):
        return _INSTALLED
    from .models import qwen4_exp as family

    if overlap_enabled():
        _install_overlap_probe(family)
        if not enabled():
            _INSTALLED = True
            return True

    timed_read = _timed("hc_read", family.GatedResidual.__call__, 1)

    def read_after_write(self, hyper_input, pending=None):
        # At profiled widths a residual write handed to the read is timed as
        # the write it is, so hc_read and hc_write keep their meaning; below
        # the row gate it rides into the read untouched.
        if pending is not None and _rows_of(hyper_input) >= _min_rows():
            hyper_input = family._hyper_residual_write(hyper_input, *pending)
            pending = None
        return timed_read(self, hyper_input, pending=pending)

    family.GatedResidual.__call__ = read_after_write
    family._hyper_residual_write = _timed("hc_write", family._hyper_residual_write, 0)
    family.GatedDeltaNet.__call__ = _timed("gdn", family.GatedDeltaNet.__call__, 1)
    family.Attention.__call__ = _timed("qsa_attention", family.Attention.__call__, 1)
    family.SparseMoeBlock.__call__ = _timed("moe", family.SparseMoeBlock.__call__, 1)
    family.PLELayer.__call__ = _timed("ple", family.PLELayer.__call__, 1)
    # Inside the MoE block (reported beside it, excluded from the share total).
    switch = getattr(family, "_FusedGateUpSwitchGLU", None)
    if switch is not None:
        switch.__call__ = _timed("moe.routed_experts", switch.__call__, 1)
    shared = getattr(family, "_FusedGateUpMLP", None)
    if shared is not None:
        shared.__call__ = _timed("moe.shared_expert", shared.__call__, 1)
    # Inside the QSA attention layer (reported beside it, excluded from the
    # share total): the block indexer (project, score, select) and the
    # block-sparse consumer kernel of the large-prefill lane.
    indexer = getattr(family, "QSAIndexer", None)
    if indexer is not None:
        indexer.__call__ = _timed("moe.zz_qsa_indexer", indexer.__call__, 1)
    try:
        from .kernels import qsa_prefill_flash as flash

        flash.qsa_prefill_flash = _timed(
            "moe.zz_qsa_flash_kernel",
            flash.qsa_prefill_flash,
            0,
            rows_fn=lambda q: int(q.shape[-2]) if getattr(q, "ndim", 0) == 4 else 0,
        )
    except Exception:
        pass
    atexit.register(_report)
    _INSTALLED = True
    print("[qwen4-prefill-profile] installed (rows >= %d)" % _min_rows(), file=sys.stderr, flush=True)
    return True
