"""Device-resident greedy depth-1 decode rounds (``MTPLX_M1_RESIDENT_DECODE=1``).

The stock round reads the verify's argmax on the host, then commits, drafts and
builds the next verify. On M1 Max that host chain leaves the GPU idle for about
3.7 ms of a 19.5 ms A3B round at 2K. Here everything a round decides stays on
the device:

* ``accepted = argmax(row 0) == draft``
* the next primary is ``where(accepted, argmax(row 1), argmax(row 0))``
* the target cache commit picks the captured GDN row ``accepted`` and moves the
  full-attention offsets back by ``1 - accepted``
* the draft history always appends ``(hidden row 0, draft)`` and then moves its
  offset back by ``1 - accepted`` (a rejected row is overwritten by the next
  draft), exactly what the stock rollback + append leaves

so round k+1 is queued before the host reads round k. The host reads each
round one round late, emits its tokens and stops the pipeline when a round
holds a stop token, would pass ``max_tokens`` or the caller asks to stop. On
exit every round the host has not accepted is discarded and the caches are
rewound to the start of the oldest such round (offsets and GDN leaves are kept
per round; KV rows past an offset are dead), so the stock loop resumes exactly
there and handles the stop with its own rules.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import mlx.core as mx

from .kernels.chunk_topk import chunk_topk_supported, topk_candidates
from .progress_heartbeat import tick as _owner_progress_tick


class ResidentUnsupported(RuntimeError):
    pass


def enabled() -> bool:
    return (os.environ.get("MTPLX_M1_RESIDENT_DECODE") or "").strip() == "1"


# Test hook: treat every N-th sampled round as a host/device disagreement so
# the rewind path runs (``MTPLX_RESIDENT_FORCE_MISMATCH_EVERY``).
_FORCE_MISMATCH_EVERY = int(os.environ.get("MTPLX_RESIDENT_FORCE_MISMATCH_EVERY") or 0)


def sampled_enabled() -> bool:
    """Sampled requests too (``MTPLX_M1_RESIDENT_SAMPLED``)."""
    return (os.environ.get("MTPLX_M1_RESIDENT_SAMPLED") or "").strip() == "1"


def copy_hybrid_enabled() -> bool:
    """Resident rounds beside context-copy (``MTPLX_M1_RESIDENT_COPY_HYBRID``,
    default on): with copy active the rounds stay resident until the copy
    proposer finds a prompt match, hand the stream to the stock loop's copy
    block rounds, and come back once the match is spent. Off keeps the old
    rule (copy active means no resident rounds)."""
    return (os.environ.get("MTPLX_M1_RESIDENT_COPY_HYBRID") or "1").strip() != "0"


def max_context() -> int:
    """Context (prompt + completion) up to which rounds stay resident.

    A queued verify cannot donate the KV buffers the previous, still running
    round holds, so its slice_update copies them; the copy grows with the
    context. M1 Max A3B, greedy: +11% at 2K, +13% at 8K, +4.5% at 16K, +1% at
    30K (moe-s3/resident, 2026-10-14). With in-place KV rows
    (``MTPLX_M1_RESIDENT_INPLACE_KV``) nothing is copied and there is no limit:
    +17.5% at 2K, +20% at 8K, +8.8% at 30K, +4.1% at 115K."""
    default = 1 << 30 if inplace_kv_enabled() else 20480
    try:
        return int(os.environ.get("MTPLX_M1_RESIDENT_MAX_CONTEXT") or default)
    except ValueError:
        return default


def inplace_kv_enabled() -> bool:
    """Queued verifies write their KV rows in place (``MTPLX_M1_RESIDENT_INPLACE_KV``)."""
    return (os.environ.get("MTPLX_M1_RESIDENT_INPLACE_KV") or "").strip() == "1"


def own_kv_pages(cache: list[Any]) -> None:
    """Give this cache its own quantized page buffers before in-place writes.

    A no-op elementwise op donates a buffer nothing else holds and copies one
    that a session-bank entry or snapshot still shares, so in-place rows can
    never reach another holder.
    """

    from .cache_state import TensorOffsetQuantizedPagedKVCache

    owned = []
    for entry in cache:
        if isinstance(entry, TensorOffsetQuantizedPagedKVCache) and entry.layout == "pages":
            for slot in (0, 1, 3, 4):
                leaf = entry.cache[slot]
                entry.cache[slot] = None
                leaf = leaf + mx.array(0, dtype=leaf.dtype)
                entry.cache[slot] = leaf
                owned.append(leaf)
    if owned:
        mx.eval(*owned)


def _select_row(value: mx.array, index: mx.array) -> mx.array:
    """``value[:, index]`` for a device scalar ``index`` (no host read)."""

    shape = (int(value.shape[0]), 1) + tuple(int(s) for s in value.shape[2:])
    sel = mx.broadcast_to(index.reshape((1, 1) + (1,) * (value.ndim - 2)), shape)
    return mx.take_along_axis(value, sel, axis=1)[:, 0]


def _tensor_offset(entry: Any) -> bool:
    leaves = getattr(entry, "cache", None)
    return (
        isinstance(leaves, list)
        and len(leaves) >= 3
        and isinstance(leaves[2], mx.array)
        and leaves[2].ndim == 0
    )


def validate_cache(cache: list[Any]) -> None:
    """Raise :class:`ResidentUnsupported` BEFORE any state moves unless every
    full-attention entry is a tensor-offset adapter (the compiled bank has
    promoted the cache). A restored session's bank serves its first rounds
    eager on stock containers; the caller retries a round later."""

    for entry in cache:
        if entry is None or hasattr(entry, "conv_state") or not getattr(entry, "is_trimmable", lambda: False)():
            continue
        if not _tensor_offset(entry):
            raise ResidentUnsupported(f"attention_entry:{type(entry).__name__}")


def device_commit(cache: list[Any], captures: dict, accepted: mx.array) -> None:
    """Commit a 2-row verify window keeping ``1 + accepted`` rows, on device."""

    from .cache_state import replace_recurrent_cache_state
    from .gdn_capture import (
        _maybe_contiguous_authoritative_gdn_leaf,
        linear_gated_delta_from_conv_tape_replay_steps,
    )

    if captures.get("__final_only__"):
        raise RuntimeError("resident commit: final-only captures after a queued verify")
    for layer_idx, entry in enumerate(cache):
        capture = captures.get(layer_idx)
        if capture is not None and hasattr(entry, "state"):
            if "conv_states" not in capture or int(capture.get("capture_start", 0)) != 0:
                # Nothing has moved yet in this commit, but the verify has run:
                # fail the request rather than continue on a half-applied round.
                raise RuntimeError("resident commit: unsupported capture layout")
    keep = accepted.astype(mx.int32)
    trim = (1 - keep).astype(mx.int32)
    host_trim = None
    for layer_idx, entry in enumerate(cache):
        capture = captures.get(layer_idx)
        if capture is not None and hasattr(entry, "state"):
            conv_state = mx.contiguous(_select_row(capture["conv_states"], keep))
            if "tape" in capture:
                replayed = linear_gated_delta_from_conv_tape_replay_steps(
                    capture["tape"],
                    capture["conv_out"],
                    capture["g"],
                    capture["state_in"],
                    capture.get("gdn_meta", capture.get("gdn")),
                    keep + 1,
                )
                if replayed is None:
                    raise RuntimeError("resident commit: tape replay unavailable")
                gdn_state = _maybe_contiguous_authoritative_gdn_leaf(replayed)
            else:
                gdn_state = mx.contiguous(_select_row(capture["states"], keep))
            replace_recurrent_cache_state(entry, [conv_state, gdn_state])
        elif capture is None and entry is not None and getattr(entry, "is_trimmable", lambda: False)():
            if not _tensor_offset(entry):
                # The bank fell back to an eager verify on stock containers
                # (python-int offsets): trim on the host for this round.
                if host_trim is None:
                    host_trim = int(trim.item())
                if host_trim:
                    entry.trim(host_trim)
                continue
            leaves = entry.cache
            rollback = getattr(entry, "rollback_state", None)
            if rollback is not None:
                for slot in range(len(rollback)):
                    rollback[slot] = None
            leaves[2] = mx.maximum(leaves[2] - trim, mx.array(0, dtype=leaves[2].dtype))


def _snapshot(cache: list[Any], mtp_cache: list[Any]) -> list[tuple[Any, int, Any]]:
    """References that rewind both caches to this point (offsets, GDN leaves)."""

    snap: list[tuple[Any, int, Any]] = []
    for entry in list(cache) + list(mtp_cache):
        if entry is None:
            continue
        if hasattr(entry, "state") and not isinstance(getattr(entry, "cache", None), list):
            snap.append((entry, -1, list(entry.state)))
        elif hasattr(entry, "state") and isinstance(getattr(entry, "cache", None), list) and not (
            len(entry.cache) >= 3 and isinstance(entry.cache[2], mx.array) and entry.cache[2].ndim == 0
        ):
            snap.append((entry, -1, list(entry.cache)))
        elif isinstance(getattr(entry, "cache", None), list) and len(entry.cache) >= 3:
            snap.append((entry, 2, entry.cache[2]))
        elif isinstance(getattr(entry, "offset", None), int):
            snap.append((entry, "int", int(entry.offset)))
    return snap


def _restore(snap: list[tuple[Any, int, Any]]) -> None:
    from .cache_state import replace_recurrent_cache_state

    for entry, slot, value in snap:
        if slot == "int":
            entry.offset = value
        elif slot == 2:
            entry.cache[2] = value
            rollback = getattr(entry, "rollback_state", None)
            if rollback is not None:
                for i in range(len(rollback)):
                    rollback[i] = None
        else:
            replace_recurrent_cache_state(entry, list(value))


@dataclass
class _Round:
    out: mx.array  # int32 [4]: draft, t0, t1, accepted
    snap: list  # rewind point = this round's start
    primary: mx.array
    hidden: mx.array
    logits: mx.array


@dataclass
class ResidentResult:
    tokens: list[int] = field(default_factory=list)  # newly committed, in order
    rounds: int = 0
    accepted: int = 0
    primary: int | None = None  # pending primary at the rewind point
    hidden: Any = None  # draft hidden at the rewind point
    logits: Any = None
    exit_reason: str = ""
    verify_time_s: float = 0.0
    draft_time_s: float = 0.0
    wait_s: float = 0.0
    wall_s: float = 0.0
    mismatches: int = 0


class ResidentMTP:
    """Compiled draft / history steps over a tensor-offset MTP cache."""

    def __init__(self, rt: Any, mtp_cache: list[Any], *, mtp_hidden_variant: str, reserve: int, hidden: mx.array, primary: mx.array):
        from .attention_context import attention_phase
        from .generation import _device_core_state_tree
        from .graphbank import TensorOffsetKVCache, promote_kv_cache_offsets

        promote_kv_cache_offsets(mtp_cache, reserve_tokens=reserve, initial_reserve_tokens=reserve)
        if not mtp_cache or not all(isinstance(e, TensorOffsetKVCache) for e in mtp_cache):
            raise ResidentUnsupported("mtp_cache_not_tensor_offset")
        self.mtp_cache = mtp_cache
        variant = mtp_hidden_variant

        def draft(h, p):
            logits, _h = rt.draft_mtp(
                h, p, mtp_cache=mtp_cache, return_hidden=True,
                mtp_hidden_variant=variant, mtp_depth=1,
            )
            return mx.argmax(logits[:, -1, :], axis=-1).reshape(1, 1).astype(mx.int32)

        def commit_draft(h0, d, accepted, h, p):
            with attention_phase("ar_decode"):
                rt.update_mtp_cache(h0, d, mtp_cache=mtp_cache, mtp_hidden_variant=variant)
            trim = (1 - accepted).astype(mx.int32)
            offsets = []
            for entry in mtp_cache:
                entry.cache[2] = entry.cache[2] - trim
                for i in range(len(entry.rollback_state)):
                    entry.rollback_state[i] = None
                offsets.append(entry.cache[2])
            # The committed history offsets (before the next draft's row) are
            # the rewind point of the next round.
            return (draft(h, p), *offsets)

        self.inplace = inplace_kv_enabled()
        if self.inplace:
            # Own the history buffers (a no-op op copies a shared one) so the
            # in-place rows of the traced steps reach no other holder.
            owned = []
            for entry in mtp_cache:
                for slot in (0, 1):
                    leaf = entry.cache[slot]
                    entry.cache[slot] = None
                    entry.cache[slot] = leaf + mx.array(0, dtype=leaf.dtype)
                    owned.append(entry.cache[slot])
            mx.eval(*owned)
        tree = _device_core_state_tree(mtp_cache)
        self._draft = mx.compile(draft, inputs=tree, outputs=tree)
        self._commit_draft = mx.compile(commit_draft, inputs=tree, outputs=tree)

    def _call(self, fn, *args):
        from .cache_state import traced_inplace_kv

        with traced_inplace_kv(self.inplace):
            return fn(*args)

    def draft(self, h, p):
        return self._call(self._draft, h, p)

    def commit_draft(self, h0, d, accepted, h, p):
        """Returns the next draft and the committed history offsets."""
        out = self._call(self._commit_draft, h0, d, accepted, h, p)
        return out[0], list(out[1:])


def run_resident_rounds(
    *,
    cache: list[Any],
    mtp: ResidentMTP,
    verify: Callable[[mx.array, int], tuple[Any, Any, Any]],
    reserve_round: Callable[[int], bool],
    hidden: mx.array,
    primary: int,
    committed: int,
    max_tokens: int,
    stop_token_ids: set[int] | None,
    should_stop: Callable[[], bool] | None = None,
    on_tokens: Callable[[list[int]], None] | None = None,
    max_rounds: int | None = None,
    context_limit: int | None = None,
) -> ResidentResult:
    """Run pipelined rounds from ``primary`` (committed, not yet forwarded).

    ``committed`` is the host count of committed completion tokens (including
    ``primary``). ``verify(ids, committed_upper)`` runs the target verify on the
    two-token window; ``reserve_round(committed_upper)`` grows the paged KV for
    one round and returns False when it cannot.
    """

    validate_cache(cache)
    stops = set(int(t) for t in (stop_token_ids or ()))
    res = ResidentResult()
    started = time.perf_counter()
    p_dev = mx.array([[int(primary)]], dtype=mx.int32)
    h_dev = hidden
    logits_dev = None
    snap = _snapshot(cache, mtp.mtp_cache)
    t0 = time.perf_counter()
    d_dev = mtp.draft(h_dev, p_dev)
    res.draft_time_s += time.perf_counter() - t0
    inflight: list[_Round] = []
    host_committed = int(committed)
    queued_upper = host_committed
    exit_reason = ""
    while True:
        # Queue one more round when it cannot pass max_tokens even if every
        # in-flight round accepts.
        can_queue = (
            not exit_reason
            and queued_upper + 2 < max_tokens
            and (max_rounds is None or res.rounds + len(inflight) < max_rounds)
        )
        if can_queue and context_limit is not None and queued_upper + 2 > context_limit:
            # Past this context a queued verify cannot donate the KV it
            # extends (the previous round still holds it), and the copy eats
            # the gain; the stock loop takes over.
            can_queue = False
            exit_reason = "context"
        if can_queue and not reserve_round(queued_upper + 2):
            can_queue = False
            exit_reason = "reservation"
        if can_queue:
            tv = time.perf_counter()
            ids = mx.concatenate([p_dev, d_dev], axis=1)
            logits, vhidden, captures = verify(ids, queued_upper)
            targets = mx.argmax(logits[0, :2, :], axis=-1).astype(mx.int32)
            accepted = (targets[0] == d_dev.reshape(())).astype(mx.int32)
            device_commit(cache, captures, accepted)
            keep = accepted.astype(mx.bool_)
            p_next = mx.where(keep, targets[1], targets[0]).reshape(1, 1)
            h_next = mx.where(keep, vhidden[:, 1:2, :], vhidden[:, 0:1, :])
            l_next = mx.where(keep, logits[:, 1, :], logits[:, 0, :])
            out = mx.concatenate([d_dev.reshape(1), targets, accepted.reshape(1)])
            res.verify_time_s += time.perf_counter() - tv
            td = time.perf_counter()
            next_snap = _snapshot(cache, [])
            d_next, mtp_offsets = mtp.commit_draft(vhidden[:, 0:1, :], d_dev, accepted, h_next, p_next)
            next_snap += [(e, 2, o) for e, o in zip(mtp.mtp_cache, mtp_offsets)]
            res.draft_time_s += time.perf_counter() - td
            mx.async_eval(out, p_next, h_next, d_next)
            inflight.append(_Round(out=out, snap=snap, primary=p_dev, hidden=h_dev, logits=logits_dev))
            snap, p_dev, h_dev, d_dev, logits_dev = next_snap, p_next, h_next, d_next, l_next
            queued_upper += 2
            if len(inflight) < 2:
                continue
        if not inflight:
            break
        # Read the oldest round (the GPU finished it or is on its last kernels).
        rnd = inflight[0]
        tw = time.perf_counter()
        d, a0, a1, acc = (int(v) for v in rnd.out.tolist())
        _owner_progress_tick()  # a settled round: the stream stall watchdog's heartbeat
        res.wait_s += time.perf_counter() - tw
        emitted = [d, a1] if acc else [a0]
        if any(t in stops for t in emitted) or host_committed + len(emitted) >= max_tokens:
            exit_reason = exit_reason or (
                "stop" if any(t in stops for t in emitted) else "max_tokens"
            )
            break
        inflight.pop(0)
        res.tokens.extend(emitted)
        if on_tokens is not None:
            on_tokens(emitted)
        res.rounds += 1
        res.accepted += acc
        host_committed += len(emitted)
        queued_upper = host_committed + 2 * len(inflight)
        if should_stop is not None and should_stop():
            exit_reason = exit_reason or "caller"
        if exit_reason and not can_queue:
            if not inflight:
                break
    # Let every queued round (and its in-place rows) land before the stock
    # loop writes the same rows again.
    mx.synchronize()
    # Rewind to the start of the oldest round the host did not accept.
    if inflight:
        first = inflight[0]
        _restore(first.snap)
        res.primary = int(first.primary.reshape(-1)[0].item())
        res.hidden = first.hidden
        res.logits = first.logits
    else:
        _restore(snap)
        res.primary = int(p_dev.reshape(-1)[0].item())
        res.hidden = h_dev
        res.logits = logits_dev
    # Hand the stock loop (and the session bank) python-int MTP containers.
    for i, entry in enumerate(mtp.mtp_cache):
        if hasattr(entry, "demote"):
            mtp.mtp_cache[i] = entry.demote()
    res.exit_reason = exit_reason or "drained"
    res.wall_s = time.perf_counter() - started
    if os.environ.get("MTPLX_RESIDENT_DEBUG"):
        n = max(1, res.rounds)
        print(
            f"[resident] rounds={res.rounds} acc={res.accepted} exit={res.exit_reason} "
            f"ms/round wall={1e3 * res.wall_s / n:.2f} verify_host={1e3 * res.verify_time_s / n:.2f} "
            f"draft_host={1e3 * res.draft_time_s / n:.2f} wait={1e3 * res.wait_s / n:.2f}",
            flush=True,
        )
    return res


# --------------------------------------------------------------------------
# Sampled rounds (temperature > 0): the device predicts, the host confirms.
#
# Per round the stock loop draws three uniforms from the request generator, in
# this order: the draft pick, the accept test, then the bonus (accepted) or
# the residual correction (rejected). Here they are drawn ahead when a round
# is queued and handed to the device, which runs the same decisions in
# float32. The host replays the stock float64 arithmetic one round late on the
# same arrays and uniforms; a round where any decision differs (a uniform
# within rounding of a cdf edge) is rewound -- caches AND generator -- and the
# stock loop runs it, so every emitted token is the stock loop's token.
# --------------------------------------------------------------------------


def chunk_topk_enabled() -> bool:
    """MTPLX_M1_CHUNK_TOPK: sampled rounds take their top-k candidates from
    per-chunk winners (mtplx/kernels/chunk_topk.py) instead of an
    argpartition that sorts the whole 248K-wide row (~0.4 ms of GPU per
    round on M1 Max for the two verify rows)."""

    return (os.environ.get("MTPLX_M1_CHUNK_TOPK") or "").strip().lower() in {"1", "true", "on", "yes"}


def target_candidates(rows: mx.array, config: Any) -> tuple:
    """Lazy ``_device_serial_candidates`` (same expressions, no eval).

    Returns (scaled, cand_idx [R, m], cand_vals [R, m], cand_probs or None).
    """

    rows = rows.reshape(-1, int(rows.shape[-1])).astype(mx.float32)
    vocab = int(rows.shape[-1])
    k = min(int(config.top_k), vocab)
    m = min(max(4 * k, k), vocab)
    scaled = rows * (1.0 / float(config.temperature))
    if chunk_topk_enabled() and chunk_topk_supported(k, vocab, m):
        # Same top-k (the m candidates only pad it); no full-row sort.
        cand_idx, cand_vals = topk_candidates(scaled, k, m)
    else:
        cand_idx = mx.argpartition(-scaled, kth=m - 1, axis=-1)[:, :m]
        cand_vals = mx.take_along_axis(scaled, cand_idx, axis=-1)
    cand_probs = None
    if 0.0 < float(config.top_p) < 1.0:
        log_total = mx.logsumexp(scaled, axis=-1, keepdims=True)
        cand_probs = mx.exp(cand_vals - log_total)
    return scaled, cand_idx, cand_vals, cand_probs


def device_target_support(candidates: tuple, config: Any) -> tuple[mx.array, mx.array]:
    """Float32 prediction of the batched serial target support, per row.

    Returns (ids [R, k] ascending, probs [R, k] normalized, top-p dropped = 0),
    the arrays ``BatchedSparseDistributions`` holds after
    ``batched_sparse_distributions_from_mlx_logits``.
    """

    scaled, idx, vals, cand_probs = candidates
    k = min(int(config.top_k), int(scaled.shape[-1]))
    order = mx.argsort(-vals, axis=-1)[:, :k]
    idx = mx.take_along_axis(idx, order, axis=-1)
    vals_k = mx.take_along_axis(vals, order, axis=-1)
    if cand_probs is not None:
        probs = mx.take_along_axis(cand_probs, order, axis=-1)
        before = mx.cumsum(probs, axis=-1) - probs
        probs = mx.where(before < float(config.top_p), probs, 0.0)
    else:
        probs = mx.softmax(vals_k, axis=-1)
    by_id = mx.argsort(idx, axis=-1)
    ids = mx.take_along_axis(idx, by_id, axis=-1).astype(mx.int32)
    probs = mx.take_along_axis(probs, by_id, axis=-1)
    probs = probs / mx.sum(probs, axis=-1, keepdims=True)
    return ids, probs


def device_pick(ids: mx.array, probs: mx.array, uniform: mx.array) -> mx.array:
    """``Generator.choice(ids, p=probs)`` for a given uniform (ids ascending)."""

    cdf = mx.cumsum(probs)
    cdf = cdf / cdf[-1]
    index = mx.minimum(mx.sum(cdf <= uniform), int(ids.shape[0]) - 1)
    return mx.take(ids, index)


def device_draft_distribution(plan: Any, cand_local, cand_vals, cand_probs):
    """The draft proposal (ids ascending, probs normalized) on the device."""

    order = mx.argsort(-cand_vals)
    probs = mx.take(cand_probs, order)
    local = mx.take(cand_local, order)
    if 0.0 < float(plan.top_p) < 1.0:
        before = mx.cumsum(probs) - probs
        probs = mx.where(before < float(plan.top_p), probs, 0.0)
    else:
        probs = mx.softmax(mx.take(cand_vals, order))
    ids = mx.take(plan.ids_mx, local).astype(mx.int32)
    by_id = mx.argsort(ids)
    ids = mx.take(ids, by_id)
    probs = mx.take(probs, by_id)
    return ids, probs / mx.sum(probs)


def device_round_decision(t_ids, t_probs, q_ids, q_probs, draft, u_accept, u_post):
    """accepted, bonus, correction of one depth-1 sampled round (float32)."""

    d = draft.reshape(())
    p_d = mx.sum(mx.where(t_ids[0] == d, t_probs[0], 0.0))
    q_d = mx.sum(mx.where(q_ids == d, q_probs, 0.0))
    ratio = mx.where(q_d > 0, mx.minimum(p_d / mx.maximum(q_d, 1e-30), 1.0), mx.where(p_d > 0, 1.0, 0.0))
    accepted = (u_accept <= ratio).astype(mx.int32)
    bonus = device_pick(t_ids[1], t_probs[1], u_post)
    union = mx.sort(mx.concatenate([t_ids[0], q_ids]))
    unique = mx.concatenate([mx.array([True]), union[1:] != union[:-1]])
    p_at = mx.sum(mx.where(union[:, None] == t_ids[0][None, :], t_probs[0][None, :], 0.0), axis=1)
    q_at = mx.sum(mx.where(union[:, None] == q_ids[None, :], q_probs[None, :], 0.0), axis=1)
    residual = mx.where(unique, mx.maximum(p_at - q_at, 0.0), 0.0)
    total = mx.sum(residual)
    cdf = mx.cumsum(residual) / mx.maximum(total, 1e-30)
    res_index = mx.minimum(mx.sum(cdf <= u_post), int(union.shape[0]) - 1)
    correction = mx.where(
        total > 0, mx.take(union, res_index), device_pick(t_ids[0], t_probs[0], u_post)
    )
    return accepted, bonus.astype(mx.int32), correction.astype(mx.int32)


class ResidentSampledMTP:
    """Compiled sampled draft / history steps over a tensor-offset MTP cache."""

    def __init__(self, rt: Any, mtp_cache: list[Any], *, plan: Any, mtp_hidden_variant: str, reserve: int):
        from .attention_context import attention_phase
        from .generation import _device_core_state_tree
        from .graphbank import TensorOffsetKVCache, promote_kv_cache_offsets
        from . import qwen4_draft_device_chain as chain

        promote_kv_cache_offsets(mtp_cache, reserve_tokens=reserve, initial_reserve_tokens=reserve)
        if not mtp_cache or not all(isinstance(e, TensorOffsetKVCache) for e in mtp_cache):
            raise ResidentUnsupported("mtp_cache_not_tensor_offset")
        self.mtp_cache = mtp_cache
        self.plan = plan
        variant = mtp_hidden_variant

        def draft(h, p, u):
            with chain.inplace_mtp_kv():
                logits, _h = rt.draft_mtp(
                    h, p, mtp_cache=mtp_cache, return_hidden=True,
                    mtp_hidden_variant=variant, mtp_depth=1,
                )
            local, vals, probs = chain.device_support(plan, logits)
            token = chain.device_predict(plan, local, vals, probs, u)
            return token.reshape(1, 1).astype(mx.int32), local, vals, probs

        def commit_draft(h0, d, accepted, h, p, u):
            with attention_phase("ar_decode"):
                rt.update_mtp_cache(h0, d, mtp_cache=mtp_cache, mtp_hidden_variant=variant)
            trim = (1 - accepted).astype(mx.int32)
            offsets = []
            for entry in mtp_cache:
                entry.cache[2] = entry.cache[2] - trim
                for i in range(len(entry.rollback_state)):
                    entry.rollback_state[i] = None
                offsets.append(entry.cache[2])
            return (*draft(h, p, u), *offsets)

        self.inplace = inplace_kv_enabled()
        if self.inplace:
            owned = []
            for entry in mtp_cache:
                for slot in (0, 1):
                    leaf = entry.cache[slot]
                    entry.cache[slot] = None
                    entry.cache[slot] = leaf + mx.array(0, dtype=leaf.dtype)
                    owned.append(entry.cache[slot])
            mx.eval(*owned)
        tree = _device_core_state_tree(mtp_cache)
        self._draft = mx.compile(draft, inputs=tree, outputs=tree)
        self._commit_draft = mx.compile(commit_draft, inputs=tree, outputs=tree)

    def _call(self, fn, *args):
        from .cache_state import traced_inplace_kv

        with traced_inplace_kv(self.inplace):
            return fn(*args)

    def draft(self, h, p, u):
        out = self._call(self._draft, h, p, u)
        return out[0], tuple(out[1:4])

    def commit_draft(self, h0, d, accepted, h, p, u):
        out = self._call(self._commit_draft, h0, d, accepted, h, p, u)
        return out[0], tuple(out[1:4]), list(out[4:])


@dataclass
class _SampledRound:
    out: mx.array  # int32 [4]: draft, accepted, bonus, correction
    snap: list
    rng_state: Any  # generator state at this round's start
    uniforms: tuple[float, float, float]  # draft, accept, post
    support: tuple  # draft (local, vals, probs)
    logits: mx.array  # verify logits [1, 2, V]
    primary: mx.array
    hidden: mx.array
    prev_logits: Any
    candidates: tuple = ()


def _host_batch(sampler: Any, candidates: tuple):
    """``batched_sparse_distributions_from_mlx_logits`` on candidates read with
    the round (no new device work unless a cutoff tie spills)."""

    import numpy as np

    from .fast_sampling import (
        BatchedSparseDistributions,
        _SerialCandidates,
        _serial_support_from_candidates,
    )

    scaled, idx, vals, probs = candidates
    vocab = int(scaled.shape[-1])
    k = min(int(sampler.top_k), vocab)
    host = _SerialCandidates(
        scaled=scaled,
        ids=np.asarray(idx, dtype=np.int64),
        vals=np.asarray(vals, dtype=np.float32),
        probs=None if probs is None else np.asarray(probs, dtype=np.float64),
        vocab_size=vocab,
        k=k,
        m=int(idx.shape[-1]),
    )
    token_rows, prob_rows = _serial_support_from_candidates(host, sampler)
    try:
        return BatchedSparseDistributions._from_execution_arrays(
            token_rows, prob_rows, vocab_size=vocab
        )
    except FloatingPointError:
        return None


def _host_round(plan, sampler, rnd: "_SampledRound"):
    """The stock loop's float64 decisions for one round: (emitted, accepted) or None."""

    import numpy as np

    from . import qwen4_draft_device_chain as chain
    from .sampling import residual_distribution

    u_draft, u_accept, u_post = rnd.uniforms
    local, vals, probs = (np.asarray(a) for a in rnd.support)
    q = chain.host_distribution(plan, local, vals, probs)
    if q is None:
        return None
    draft = chain.host_pick(q, u_draft)
    batch = _host_batch(sampler, rnd.candidates)
    if batch is None:
        return None
    p = batch.probability(0, draft)
    qd = q.probability(draft)
    ratio = (1.0 if p > 0 else 0.0) if qd <= 0 else min(1.0, p / qd)
    accepted = u_accept <= ratio
    if accepted:
        keep = batch.probs[1] > 0
        bonus = chain.host_pick(
            _sparse(batch.token_ids[1, keep], batch.probs[1, keep], batch.vocab_size), u_post
        )
        return draft, [draft, bonus], True
    residual = residual_distribution(batch.to_distribution(0), q)
    return draft, [chain.host_pick(residual, u_post)], False


def _sparse(ids, probs, vocab):
    from .sampling import SparseDistribution

    return SparseDistribution(ids, probs, vocab)


def run_resident_rounds_sampled(
    *,
    cache: list[Any],
    mtp: ResidentSampledMTP,
    sampler: Any,
    rng: Any,
    verify: Callable[[mx.array, int], tuple[Any, Any, Any]],
    reserve_round: Callable[[int], bool],
    hidden: mx.array,
    primary: int,
    committed: int,
    max_tokens: int,
    stop_token_ids: set[int] | None,
    should_stop: Callable[[], bool] | None = None,
    on_tokens: Callable[[list[int]], None] | None = None,
    context_limit: int | None = None,
) -> ResidentResult:
    """Sampled counterpart of :func:`run_resident_rounds` (depth 1)."""

    import copy

    validate_cache(cache)
    stops = set(int(t) for t in (stop_token_ids or ()))
    res = ResidentResult()
    started = time.perf_counter()
    plan = mtp.plan
    p_dev = mx.array([[int(primary)]], dtype=mx.int32)
    h_dev = hidden
    logits_dev = None
    snap = _snapshot(cache, mtp.mtp_cache)
    rng_state = copy.deepcopy(rng.bit_generator.state)
    u_draft = float(rng.random())
    t0 = time.perf_counter()
    d_dev, support = mtp.draft(h_dev, p_dev, mx.array(u_draft, dtype=mx.float32))
    res.draft_time_s += time.perf_counter() - t0
    inflight: list[_SampledRound] = []
    host_committed = int(committed)
    queued_upper = host_committed
    exit_reason = ""
    mismatches = 0
    while True:
        can_queue = not exit_reason and queued_upper + 2 < max_tokens
        if can_queue and context_limit is not None and queued_upper + 2 > context_limit:
            can_queue = False
            exit_reason = "context"
        if can_queue and not reserve_round(queued_upper + 2):
            can_queue = False
            exit_reason = "reservation"
        if can_queue:
            tv = time.perf_counter()
            u_accept = float(rng.random())
            u_post = float(rng.random())
            ids = mx.concatenate([p_dev, d_dev], axis=1)
            logits, vhidden, captures = verify(ids, queued_upper)
            cands = target_candidates(logits[0, :2, :], sampler)
            t_ids, t_probs = device_target_support(cands, sampler)
            q_ids, q_probs = device_draft_distribution(plan, *support)
            accepted, bonus, correction = device_round_decision(
                t_ids, t_probs, q_ids, q_probs, d_dev,
                mx.array(u_accept, dtype=mx.float32), mx.array(u_post, dtype=mx.float32),
            )
            device_commit(cache, captures, accepted)
            keep = accepted.astype(mx.bool_)
            p_next = mx.where(keep, bonus, correction).reshape(1, 1)
            h_next = mx.where(keep, vhidden[:, 1:2, :], vhidden[:, 0:1, :])
            l_next = mx.where(keep, logits[:, 1, :], logits[:, 0, :])
            out = mx.stack([d_dev.reshape(()), accepted, bonus, correction])
            res.verify_time_s += time.perf_counter() - tv
            td = time.perf_counter()
            next_snap = _snapshot(cache, [])
            next_rng_state = copy.deepcopy(rng.bit_generator.state)
            u_draft_next = float(rng.random())
            d_next, support_next, mtp_offsets = mtp.commit_draft(
                vhidden[:, 0:1, :], d_dev, accepted, h_next, p_next,
                mx.array(u_draft_next, dtype=mx.float32),
            )
            next_snap += [(e, 2, o) for e, o in zip(mtp.mtp_cache, mtp_offsets)]
            res.draft_time_s += time.perf_counter() - td
            mx.async_eval(
                out, p_next, h_next, d_next, *support_next,
                *[a for a in cands[1:] if a is not None],
            )
            inflight.append(
                _SampledRound(
                    out=out, snap=snap, rng_state=rng_state,
                    uniforms=(u_draft, u_accept, u_post), support=support,
                    logits=logits, primary=p_dev, hidden=h_dev, prev_logits=logits_dev,
                    candidates=cands,
                )
            )
            snap, rng_state, u_draft = next_snap, next_rng_state, u_draft_next
            p_dev, h_dev, d_dev, support, logits_dev = p_next, h_next, d_next, support_next, l_next
            queued_upper += 2
            if len(inflight) < 2:
                continue
        if not inflight:
            break
        rnd = inflight[0]
        tw = time.perf_counter()
        d, acc, bonus, corr = (int(v) for v in rnd.out.tolist())
        _owner_progress_tick()
        res.wait_s += time.perf_counter() - tw
        host = _host_round(plan, sampler, rnd)
        device_emitted = [d, bonus] if acc else [corr]
        forced = _FORCE_MISMATCH_EVERY > 0 and (res.rounds + 1) % _FORCE_MISMATCH_EVERY == 0
        if forced or host is None or host[0] != d or host[1] != device_emitted:
            mismatches += 1
            exit_reason = "mismatch"
            break
        emitted = host[1]
        if any(t in stops for t in emitted) or host_committed + len(emitted) >= max_tokens:
            exit_reason = exit_reason or ("stop" if any(t in stops for t in emitted) else "max_tokens")
            break
        inflight.pop(0)
        res.tokens.extend(emitted)
        if on_tokens is not None:
            on_tokens(emitted)
        res.rounds += 1
        res.accepted += int(acc)
        host_committed += len(emitted)
        queued_upper = host_committed + 2 * len(inflight)
        if should_stop is not None and should_stop():
            exit_reason = exit_reason or "caller"
        if exit_reason and not can_queue and not inflight:
            break
    mx.synchronize()
    if inflight:
        first = inflight[0]
        _restore(first.snap)
        rng.bit_generator.state = first.rng_state
        res.primary = int(first.primary.reshape(-1)[0].item())
        res.hidden = first.hidden
        res.logits = first.prev_logits
    else:
        _restore(snap)
        rng.bit_generator.state = rng_state
        res.primary = int(p_dev.reshape(-1)[0].item())
        res.hidden = h_dev
        res.logits = logits_dev
    if exit_reason != "mismatch":
        # A mismatch hands one round to the stock loop and comes back to the
        # same compiled steps, which keep the tensor-offset history.
        for i, entry in enumerate(mtp.mtp_cache):
            if hasattr(entry, "demote"):
                mtp.mtp_cache[i] = entry.demote()
    res.exit_reason = exit_reason or "drained"
    res.mismatches = mismatches
    res.wall_s = time.perf_counter() - started
    if os.environ.get("MTPLX_RESIDENT_DEBUG"):
        n = max(1, res.rounds)
        print(
            f"[resident-sampled] rounds={res.rounds} acc={res.accepted} exit={res.exit_reason} "
            f"ms/round wall={1e3 * res.wall_s / n:.2f} verify_host={1e3 * res.verify_time_s / n:.2f} "
            f"draft_host={1e3 * res.draft_time_s / n:.2f} wait={1e3 * res.wait_s / n:.2f}",
            flush=True,
        )
    return res
