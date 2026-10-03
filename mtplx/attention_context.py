"""Lightweight attention-phase telemetry context."""

from __future__ import annotations

import os
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Hashable, Iterator

from .compile_state import compiled_dispatch_scope, current_dispatch_identity

VALID_ATTENTION_PHASES = {
    "prefill",
    "decode_verify",
    "ar_decode",
    "postcommit",
    "unknown",
}
VALID_MODEL_FORWARD_KINDS = {
    "target_verify",
    "repair",
    "other",
}

_ATTENTION_PHASE: ContextVar[str] = ContextVar(
    "mtplx_attention_phase",
    default="unknown",
)
_MODEL_FORWARD_KIND: ContextVar[str] = ContextVar(
    "mtplx_model_forward_kind",
    default="other",
)


def normalize_attention_phase(phase: str | None) -> str:
    value = (phase or "unknown").strip().lower()
    return value if value in VALID_ATTENTION_PHASES else "unknown"


def current_attention_phase() -> str:
    return normalize_attention_phase(_ATTENTION_PHASE.get())


def normalize_model_forward_kind(kind: str | None) -> str:
    value = (kind or "other").strip().lower()
    return value if value in VALID_MODEL_FORWARD_KINDS else "other"


def current_model_forward_kind() -> str:
    return normalize_model_forward_kind(_MODEL_FORWARD_KIND.get())


_EXACT_VERIFY_REQUIRED: ContextVar[bool] = ContextVar(
    "mtplx_exact_verify_required",
    default=False,
)

# The t<=0 stock-matmul guard is OPT-IN as of 2026-08-31 (founder order:
# restore turbo at greedy). Receipts (SPEEDWAR-20260831/turbo-guard-27b):
# on the 27B turbo serve path the guard cost 5-21% greedy decode across a
# position-matched ABBA quad (guard-off 54.3/52.4 vs guard-on 51.0/49.7
# tok/s in the clean pair), while the identity it promised held on NEITHER
# route — 5/6 default-suite prompts diverge from greedy AR at identical
# token indexes with the guard on or off, because the divergence lives in
# the cross-M numeric frame (M=1 AR vs M=4..16 verify shapes), which stock
# kernels share. MTPLX_EXACT_T0_GUARD=1 re-arms the stock frame for
# operators who want it. Read once at import (hot-path flag pattern).
_EXACT_T0_GUARD_ARMED = (
    (os.environ.get("MTPLX_EXACT_T0_GUARD") or "").strip().lower()
    in {"1", "true", "yes", "on"}
)

# Multi-axis rope state for vision requests: (positions [3, prompt_len] mx
# array or None, rope_delta int). Families that implement M-RoPE (qwen4_exp)
# read it inside their attention layers and self-slice by cache offset; every
# other family ignores it. Set per request around generation entry points —
# never stored in cache state, so bank restores stay format-stable (the
# request re-derives it from its own content).
_VISION_ROPE: ContextVar["tuple[object, int] | None"] = ContextVar(
    "mtplx_vision_rope",
    default=None,
)


def vision_rope_state() -> "tuple[object, int] | None":
    return _VISION_ROPE.get()


@contextmanager
def vision_rope(positions: object, delta: int) -> Iterator[None]:
    token = _VISION_ROPE.set((positions, int(delta)))
    try:
        yield
    finally:
        _VISION_ROPE.reset(token)


def exact_verify_required() -> bool:
    """True while the current forward must use stock (bit-exact) matmuls.

    Only meaningful when the operator arms MTPLX_EXACT_T0_GUARD=1: the
    vk/nax verify kernels are argmax- and distribution-validated but not
    bit-exact vs stock (~6e-3 dmax, lane-strided fp32 accumulation), and an
    armed guard makes t<=0 verify forwards fall through to stock so both
    paths share one numeric frame. The shipping default leaves the guard
    dark — turbo kernels run at every temperature — because the guard never
    delivered MTP==AR greedy identity (cross-M frame flips survive stock
    kernels) and cost 5-21% greedy decode on the 27B turbo profile.
    """
    if not _EXACT_T0_GUARD_ARMED:
        return False
    return bool(_EXACT_VERIFY_REQUIRED.get())


@contextmanager
def exact_verify(required: bool) -> Iterator[None]:
    token = _EXACT_VERIFY_REQUIRED.set(bool(required))
    try:
        yield
    finally:
        _EXACT_VERIFY_REQUIRED.reset(token)


@contextmanager
def attention_phase(phase: str | None) -> Iterator[None]:
    token = _ATTENTION_PHASE.set(normalize_attention_phase(phase))
    try:
        yield
    finally:
        _ATTENTION_PHASE.reset(token)


@contextmanager
def model_forward_kind(kind: str | None) -> Iterator[None]:
    """Identify whether one decode-verify-phase target call verifies or repairs."""

    token = _MODEL_FORWARD_KIND.set(normalize_model_forward_kind(kind))
    try:
        yield
    finally:
        _MODEL_FORWARD_KIND.reset(token)


# KV attention record (issue #526): host facts about each full-attention
# layer's latest call, written by the split-attention hook and read when the
# request fails with non-finite logits. It belongs to one request, not to the
# process: the server gives every model-work item its own record
# (kv_attention_request_scope) and NonFiniteLogitsError captures the failure
# line where it is raised, on the model thread, before any other request can
# run attention. A thread with no scope keeps one record for itself.
#
# The Python forward runs for eager calls and for the TRACE of a compiled
# step, never for its replays. Trace-time records are therefore kept apart,
# filed under the specialization being traced (the compiled callable and the
# input shapes that select its graph), and every compiled dispatch names its
# specialization (compiled_dispatch). A failure after a replay reads
# "compiled replay of" the trace of that same graph when this request traced
# it, and says the metadata is unavailable otherwise: never another width's
# trace, never a stale eager record claiming to be the failing dispatch.


@dataclass
class KvAttentionRecords:
    eager: dict[int, tuple[Any, ...]] = field(default_factory=dict)
    traced: dict[Hashable, dict[int, tuple[Any, ...]]] = field(default_factory=dict)
    last_dispatch: str = "eager"
    last_identity: Hashable | None = None


_KV_ATTENTION_RECORDS: ContextVar[KvAttentionRecords | None] = ContextVar(
    "mtplx_kv_attention_records", default=None
)


def _kv_attention_records() -> KvAttentionRecords:
    records = _KV_ATTENTION_RECORDS.get()
    if records is None:
        records = KvAttentionRecords()
        _KV_ATTENTION_RECORDS.set(records)
    return records


@contextmanager
def kv_attention_request_scope() -> Iterator[None]:
    """Give one request (one model-work item) its own KV attention record."""
    token = _KV_ATTENTION_RECORDS.set(KvAttentionRecords())
    try:
        yield
    finally:
        _KV_ATTENTION_RECORDS.reset(token)


def note_kv_attention_record(layer: int, record: tuple[Any, ...], *, traced: bool) -> None:
    records = _kv_attention_records()
    if traced:
        identity = current_dispatch_identity()
        records.traced.setdefault(identity, {})[int(layer)] = record
    else:
        records.eager[int(layer)] = record
        records.last_dispatch = "eager"


@contextmanager
def compiled_dispatch(identity: Hashable) -> Iterator[None]:
    """Wrap a call into a compiled step whose graph ``identity`` names.

    Marks the dispatch (no Python runs for a replay) and files any trace the
    call makes under ``identity``, so a failure can be matched to the trace of
    the graph that ran. Pass the compiled callable's id and the shapes of the
    inputs that select its specialization.
    """

    records = _kv_attention_records()
    records.last_dispatch = "compiled_replay"
    records.last_identity = identity
    with compiled_dispatch_scope(identity):
        yield


def format_kv_attention_record(
    record: tuple[Any, ...], *, dispatch: str | None = None
) -> str:
    (layer, phase, cache_name, bits, route, q_dtype, offset, capacity, q_len,
     mask_kind, fallback, finite) = record
    head = "mtplx_kv_attention " + (f"dispatch={dispatch} " if dispatch else "")
    return head + (
        f"layer={layer} phase={phase} cache={cache_name} bits={bits} route={route} "
        f"q_dtype={str(q_dtype).removeprefix('mlx.core.')} offset={offset} "
        f"capacity={'-' if capacity is None else capacity} q_len={q_len} "
        f"mask={mask_kind} fallback={fallback or '-'} finite={finite}"
    )


def kv_attention_failure_line() -> str | None:
    """The current request's first full-attention layer, for a failure report.

    Host data only (safe after the failing generation unwound). Offsets held
    in an array print as ``array``; rerun with MTPLX_KV_ATTENTION_TRACE=nonfinite
    for per-layer values and verdicts.
    """

    records = _KV_ATTENTION_RECORDS.get()
    if records is None:
        return None
    if records.last_dispatch == "compiled_replay":
        traced = (
            records.traced.get(records.last_identity)
            if records.last_identity is not None
            else None
        )
        if traced:
            record = traced[min(traced)]
            return format_kv_attention_record(record, dispatch="compiled_replay_of_trace")
        return (
            "mtplx_kv_attention dispatch=compiled_replay (this request did not "
            "trace the graph that ran, so its call metadata is unavailable; "
            "MTPLX_KV_ATTENTION_TRACE=1 prints every trace)"
        )
    if records.eager:
        return format_kv_attention_record(records.eager[min(records.eager)], dispatch="eager")
    return None
