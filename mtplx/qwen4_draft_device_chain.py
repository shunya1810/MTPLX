"""``MTPLX_QWEN4_SAMPLED_DRAFT_CHAIN`` -- the sampled draft chain, one eval per round.

The problem
-----------
At ``temperature > 0`` the stock draft loop is strictly serial.  Every draft
depth runs one MTP forward, blocks on one ``mx.eval`` of its K20 support, builds
the proposal distribution on the host, draws the token with the request's NumPy
generator, and only then can the next depth's forward be built, because that
forward consumes the drawn token.  Depth 3 therefore costs three blocking
syncs per round before the verify sync, and the GPU idles while the host builds
and encodes each next depth.  The one-eval chain in ``generation.py`` exists for
greedy requests only, where the next token is an ``argmax`` the device can take
by itself.

What this module does
---------------------
It lets the device *predict* the host's draw so the next depth can be chained
without a sync, and lets the host *prove* every prediction afterwards:

1. Before the round the host draws the ``D`` uniforms the serial lane would
   have drawn, in the same order from the same generator (``Generator.choice``
   with ``p=`` draws exactly one ``random()`` double and does a right-sided
   ``searchsorted`` on the normalized cdf; :func:`host_pick` is that, and
   ``tests/test_qwen4_draft_device_chain.py`` pins the equivalence).
2. On the device, per depth: the K20 support of the FR-Spec head's compact
   65,536-row output (never the 248,320-lane scatter), the nucleus filter, the
   id-ascending cdf, and the inverse-cdf pick for that depth's uniform, all in
   float32.  The picked id feeds the next depth's forward as a lazy array.  One
   ``mx.eval`` materializes all depths.
3. On the host, per depth, the SAME arithmetic the serial reader runs (float64
   cumulative sums, ``cumulative_before < top_p``, id-ascending renormalized
   distribution) on the same three device arrays, and the same inverse-cdf
   pick.  The host result is the authority: it is the token that is drafted and
   the distribution that enters the acceptance ratio.
4. If the device's float32 prediction ever disagrees with the host's float64
   pick (a uniform within rounding distance of a cdf edge), the chain is cut at
   that depth: the MTP cache is rolled back to the offset recorded before the
   first forward that consumed the wrong token, and the remaining depths run
   serially with the already drawn uniforms.  Nothing downstream ever sees a
   device-only decision.

Exactness
---------
The drafted tokens, the proposal distributions and the generator stream are,
by construction, exactly those of the serial compact-row reader
(:func:`serial_read`), for every seed: same candidates, same float64 host
arithmetic, same uniforms in the same order.  The device prediction is never
trusted, only confirmed.  Against the older dense-row reader the proposal
``q`` can differ by a few ULP of ``logsumexp`` (a 65,536-lane reduction against
a 248,320-lane one whose extra lanes contribute exactly ``+0.0``; see
``qwen4_draft_k20_prescatter``), which moves ``q`` and never the output law:
the acceptance ratio is taken against the ``q`` the token was really drawn
from.

Cutoff ties follow ``MTPLX_QWEN4_RELAXED_DRAFT_TIES`` semantics (the Flash-Next
lane default): the support is ``argpartition``'s top ``k`` with no tie proof
superset, so which of several *exactly equal* logits at the cutoff enters the
support is the partition's choice.  Whatever that choice is, it is the support
the token is drawn from and the support the ratio sees.

NO device work happens at import.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np

from .sampling import SamplerConfig, SparseDistribution


_ENV_VAR = "MTPLX_QWEN4_SAMPLED_DRAFT_CHAIN"


def is_enabled(env: Any | None = None) -> bool:
    """On when the env says so; unset, on for the M1 family with the in-place append.

    Without the in-place MTP KV append the chain copies the whole MTP cache per
    depth and loses at long context (143K: draft 19.8 -> 32 ms); with it the
    chain wins at every measured context on M1 Max (32K -2.3%, 110K -1.7% per
    round, byte-identical output).
    """
    source = os.environ if env is None else env
    raw = str(source.get(_ENV_VAR, "")).strip().lower()
    if raw:
        return raw in {"1", "true", "yes", "on"}
    if env is not None:
        return False
    try:
        from .cache_state import m1_long_context_defaults

        if not m1_long_context_defaults():
            return False
    except Exception:
        return False
    return _inplace_writer() is not None


_PIPELINE_ENV_VAR = "MTPLX_QWEN4_DRAFT_CHAIN_PIPELINE"


def pipeline_enabled(env: Any | None = None) -> bool:
    """Overlap the chain's host work with its GPU work (default on).

    The chain builds every depth lazily and evaluates once, so the GPU waits
    for the host to build all the depths, and the host then waits for the GPU
    to run them.  Metal timeline, 2026-09-20, Flash-Next at 4K: 2.3 ms of GPU
    idle between the verify and the draft, 5.5 ms between the draft and the
    next verify, 20% of a 38.5 ms round.  With this on, each depth is handed
    to the GPU as soon as it is built, and the host reads depth d (and warms
    the verify window's n-gram rows for it) while the GPU runs depth d+1.
    The arrays, the uniforms and the host arithmetic are the same, so tokens,
    proposals and the generator stream do not change.
    ``MTPLX_QWEN4_DRAFT_CHAIN_PIPELINE=0`` restores the single evaluation.
    """

    source = os.environ if env is None else env
    raw = str(source.get(_PIPELINE_ENV_VAR, "1")).strip().lower()
    return raw not in {"0", "false", "no", "off"}


@dataclass(frozen=True)
class SampledChainPlan:
    """Request-bound view of the FR-Spec head's compact draft row."""

    head: Any
    ids_np: np.ndarray
    ids_mx: Any
    rows: int
    vocab_rows: int
    top_k: int
    temperature: float
    top_p: float
    #: True when the draft logits ARE the compact row (the FR-Spec legacy swap
    #: of the dense 27B family): no head hook, ``head`` is None.
    compact_logits: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "installed": True,
            "rows": int(self.rows),
            "vocab_rows": int(self.vocab_rows),
            "top_k": int(self.top_k),
            "temperature": float(self.temperature),
            "top_p": float(self.top_p),
        }


def claim(rt: Any, draft_sampler: SamplerConfig) -> tuple[SampledChainPlan | None, str | None]:
    """Bind the chain to the live FR-Spec draft head; ``(None, reason)`` if it cannot.

    Every refusal is a routing answer (the stock serial reader runs), never an
    error: this lane removes syncs, it does not own correctness.
    """

    text = getattr(getattr(rt, "model", None), "language_model", None)
    if text is None:
        text = getattr(rt, "model", None)
    head = getattr(text, "_mtplx_frspec_draft_head", None)
    if head is None:
        return None, "no_frspec_head"
    native = getattr(text, "_mtp_draft_head_logits", None)
    if native is None or getattr(native, "__self__", None) is not head:
        return None, "frspec_head_not_live"
    if not hasattr(head, "arm_prescatter_capture"):
        return None, "no_capture_surface"
    ids = getattr(head, "_ids", None)
    if ids is None:
        return None, "no_ranked_table"
    ids_np = np.asarray(ids, dtype=np.int64).reshape(-1)
    rows = int(ids_np.shape[0])
    if rows < 2 or not bool(np.all(ids_np[1:] > ids_np[:-1])):
        return None, "ranked_table_not_ascending"
    vocab_rows = int(getattr(head, "_vocab_rows", 0))
    if vocab_rows <= rows or int(ids_np[-1]) >= vocab_rows:
        return None, "vocab_does_not_admit_table"
    top_k = int(draft_sampler.top_k)
    if float(draft_sampler.temperature) <= 0.0 or top_k <= 0 or top_k > rows:
        return None, "not_a_sampled_top_k_draft"
    if (
        float(draft_sampler.presence_penalty) != 0.0
        or float(draft_sampler.frequency_penalty) != 0.0
    ):
        return None, "draft_penalties"
    head.arm_prescatter_capture(True)
    return (
        SampledChainPlan(
            head=head,
            ids_np=ids_np,
            ids_mx=ids,
            rows=rows,
            vocab_rows=vocab_rows,
            top_k=top_k,
            temperature=float(draft_sampler.temperature),
            top_p=float(draft_sampler.top_p),
        ),
        None,
    )


def claim_compact_logits(
    ids: Any, vocab_rows: int, draft_sampler: SamplerConfig
) -> tuple[SampledChainPlan | None, str | None]:
    """Bind the chain to draft logits that already are the compact row.

    The FR-Spec legacy swap (``MTPLX_FRSPEC_LEGACY``, the M1 default of the
    dense 27B family) puts the pruned head in the draft graph itself: each draft
    step returns ``len(ids)`` logits whose row ``i`` is token ``ids[i]``. That
    is the row the chain reads from a Flash-Next head's pre-scatter capture, so
    the same support, prediction and host arithmetic apply; ``ids`` need not
    be ascending (the prediction orders by real token id, as the host does).
    """

    import mlx.core as mx

    ids_np = np.asarray(ids, dtype=np.int64).reshape(-1)
    rows = int(ids_np.shape[0])
    if rows < 2:
        return None, "no_ranked_table"
    if int(vocab_rows) <= rows or int(ids_np.max()) >= int(vocab_rows):
        return None, "vocab_does_not_admit_table"
    top_k = int(draft_sampler.top_k)
    if float(draft_sampler.temperature) <= 0.0 or top_k <= 0 or top_k > rows:
        return None, "not_a_sampled_top_k_draft"
    if (
        float(draft_sampler.presence_penalty) != 0.0
        or float(draft_sampler.frequency_penalty) != 0.0
    ):
        return None, "draft_penalties"
    return (
        SampledChainPlan(
            head=None,
            ids_np=ids_np,
            ids_mx=mx.array(ids_np.astype(np.int32)),
            rows=rows,
            vocab_rows=int(vocab_rows),
            top_k=top_k,
            temperature=float(draft_sampler.temperature),
            top_p=float(draft_sampler.top_p),
            compact_logits=True,
        ),
        None,
    )


_INPLACE_ENV_VAR = "MTPLX_MTP_INPLACE_KV"
_INPLACE = {"active": False, "write": None, "patched": False}


def _inplace_writer():
    if _INPLACE["write"] is None:
        _INPLACE["write"] = False
        if str(os.environ.get(_INPLACE_ENV_VAR, "1")).strip().lower() in {"0", "false", "no", "off"}:
            return None
        try:
            from .kernels.mfa_prefill import _ext

            ext = _ext()
            fn = getattr(ext, "inplace_row_write", None) if ext is not None else None
            _INPLACE["write"] = fn if fn is not None else False
        except Exception:
            _INPLACE["write"] = False
    return _INPLACE["write"] or None


def _patch_kv_cache() -> bool:
    if _INPLACE["patched"]:
        return True
    try:
        from mlx_lm.models.cache import KVCache
    except Exception:
        return False
    stock = KVCache.update_and_fetch

    def update_and_fetch(self, keys, values):
        write = _INPLACE["write"] if _INPLACE["active"] else None
        if (
            not write
            or self.keys is None
            or self.offset + keys.shape[2] > self.keys.shape[2]
            or keys.dtype != self.keys.dtype
            or values.dtype != self.values.dtype
        ):
            return stock(self, keys, values)
        prev = self.offset
        self.offset += keys.shape[2]
        self.keys = write(self.keys, keys, prev)
        self.values = write(self.values, values, prev)
        return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]

    KVCache.update_and_fetch = update_and_fetch
    _INPLACE["patched"] = True
    return True


class inplace_mtp_kv:
    """Chain depths append to the MTP KV cache in place (``MTPLX_MTP_INPLACE_KV``, default on).

    Between chained depths nothing syncs, so the previous depth's command
    buffer still references the cache buffer and MLX's ``slice_update`` cannot
    donate it: every depth copied the whole fp16 MTP K/V (about 110 MB at 32K,
    0.6 GB at 143K). Inside this block an append within capacity writes its
    rows straight into the existing buffer instead. Every reader of the buffer
    reads only rows below its own offset, and an append writes at or past the
    current offset, so no reader sees a changed row; growth past capacity
    takes the stock path.
    """

    def __enter__(self):
        if _inplace_writer() is not None and _patch_kv_cache():
            _INPLACE["active"] = True
        return self

    def __exit__(self, *exc):
        _INPLACE["active"] = False
        return False


def release(plan: SampledChainPlan | None) -> None:
    if plan is not None and plan.head is not None:
        plan.head.arm_prescatter_capture(False)


def _compact_row(plan: SampledChainPlan, draft_logits: Any) -> Any:
    if plan.compact_logits:
        import mlx.core as mx

        row = draft_logits.reshape(-1, int(draft_logits.shape[-1]))[-1]
        width = int(row.shape[0])
        if width == plan.rows:
            return row
        if width == plan.vocab_rows:
            # A graph without the swapped head (e.g. warmup before the swap):
            # the same rows, gathered by id, are the compact row's logits.
            return mx.take(row, plan.ids_mx)
        raise RuntimeError(
            f"draft logits are {width} wide, expected {plan.rows} or {plan.vocab_rows}"
        )
    stashed = plan.head.take_prescatter_row(draft_logits)
    if stashed is None:
        raise RuntimeError(
            "the FR-Spec head did not capture a pre-scatter row for this draft step"
        )
    row = stashed.reshape(-1)
    if int(row.shape[0]) != plan.rows:
        raise RuntimeError(
            f"pre-scatter row is {int(row.shape[0])} wide, expected {plan.rows}"
        )
    return row


def device_support(plan: SampledChainPlan, draft_logits: Any) -> tuple[Any, Any, Any]:
    """Lazy ``(cand_local [k], cand_vals [k], cand_probs [k])`` of one draft step.

    The same three arrays, from the same expressions, that the serial
    relaxed-tie reader evaluates -- on the compact row.
    """

    import mlx.core as mx

    row = _compact_row(plan, draft_logits)
    scaled = row.astype(mx.float32) * (1.0 / float(plan.temperature))
    k = int(plan.top_k)
    from .kernels.chunk_topk import chunk_topk_supported, topk_candidates
    from .resident_decode import chunk_topk_enabled

    if scaled.ndim == 1 and chunk_topk_enabled() and chunk_topk_supported(k, int(scaled.shape[0])):
        cand_local, cand_vals = topk_candidates(scaled.reshape(1, -1), k, k)
        cand_local, cand_vals = cand_local.reshape(-1), cand_vals.reshape(-1)
    else:
        cand_local = mx.argpartition(-scaled, kth=k - 1, axis=-1)[:k]
        cand_vals = mx.take(scaled, cand_local)
    log_total = mx.logsumexp(scaled, axis=-1, keepdims=True)
    cand_probs = mx.exp(cand_vals - log_total)
    return cand_local, cand_vals, cand_probs


def device_predict(
    plan: SampledChainPlan,
    cand_local: Any,
    cand_vals: Any,
    cand_probs: Any,
    uniform: Any,
) -> Any:
    """Lazy float32 prediction of :func:`host_pick` for this step: a real token id.

    Value-descending nucleus filter, id-ascending cdf, right-sided search.  A
    prediction only: the host confirms it against its own float64 pick.
    """

    import mlx.core as mx

    k = int(plan.top_k)
    order = mx.argsort(-cand_vals)
    probs_sorted = mx.take(cand_probs, order)
    local_sorted = mx.take(cand_local, order)
    if 0.0 < float(plan.top_p) < 1.0:
        cumulative_before = mx.cumsum(probs_sorted) - probs_sorted
        probs_sorted = mx.where(
            cumulative_before < float(plan.top_p), probs_sorted, 0.0
        )
    # Order by REAL token id, as host_distribution does (identical to ordering
    # by row when the table is ascending, as a Flash-Next head's is).
    by_id = mx.argsort(mx.take(plan.ids_mx, local_sorted))
    local_by_id = mx.take(local_sorted, by_id)
    probs_by_id = mx.take(probs_sorted, by_id)
    cdf = mx.cumsum(probs_by_id)
    cdf = cdf / cdf[-1]
    pick = mx.minimum(mx.sum(cdf <= uniform), k - 1)
    return mx.take(plan.ids_mx, mx.take(local_by_id, pick))


def host_distribution(
    plan: SampledChainPlan,
    cand_local: np.ndarray,
    cand_vals: np.ndarray,
    cand_probs: np.ndarray,
) -> SparseDistribution | None:
    """The serial relaxed-tie reader's host arithmetic, on compact-row candidates.

    Local rows are mapped to real token ids BEFORE the deterministic
    ``(value desc, id asc)`` sort, as ``qwen4_draft_k20_prescatter`` does, so
    the order imposed is the order the full-vocabulary builder would impose.
    ``None`` on a row with no finite positive mass (the caller falls back to
    the stock reader, which raises the shipped non-finite-logits error).
    """

    token_ids = plan.ids_np[np.asarray(cand_local, dtype=np.int64).reshape(-1)]
    values = np.asarray(cand_vals, dtype=np.float32).reshape(-1)
    probs = np.asarray(cand_probs, dtype=np.float64).reshape(-1)
    order = np.lexsort((token_ids, -values))
    token_ids = token_ids[order]
    probs = probs[order]
    if 0.0 < float(plan.top_p) < 1.0:
        cumulative_before = np.concatenate(([0.0], np.cumsum(probs[:-1])))
        probs = np.where(cumulative_before < float(plan.top_p), probs, 0.0)
    else:
        vals64 = values[order].astype(np.float64)
        vals64 -= np.max(vals64)
        probs = np.exp(vals64)
        probs /= np.sum(probs)
    keep = probs > 0
    kept_ids = token_ids[keep]
    kept_probs = probs[keep]
    total = kept_probs.sum()
    if not np.isfinite(total) or total <= 0:
        return None
    by_id = np.argsort(kept_ids)
    return SparseDistribution(
        kept_ids[by_id], kept_probs[by_id] / total, int(plan.vocab_rows)
    )


def host_pick(distribution: SparseDistribution, uniform: float) -> int:
    """``Generator.choice(ids, p=probs)`` given the one uniform it would draw.

    NumPy's weighted choice is ``cdf = p.cumsum(); cdf /= cdf[-1];
    idx = cdf.searchsorted(rng.random(), side="right")``.
    """

    cdf = np.cumsum(distribution.probs)
    cdf /= cdf[-1]
    index = int(np.searchsorted(cdf, float(uniform), side="right"))
    index = min(index, int(distribution.token_ids.shape[0]) - 1)
    return int(distribution.token_ids[index])


def serial_read(
    plan: SampledChainPlan,
    draft_logits: Any,
    uniform: float,
) -> tuple[int, SparseDistribution | None]:
    """One serial draft read on the compact row (the chain's parent and fallback)."""

    import mlx.core as mx

    cand_local, cand_vals, cand_probs = device_support(plan, draft_logits)
    mx.eval(cand_local, cand_vals, cand_probs)
    distribution = host_distribution(
        plan,
        np.asarray(cand_local),
        np.asarray(cand_vals),
        np.asarray(cand_probs),
    )
    if distribution is None:
        return -1, None
    return host_pick(distribution, uniform), distribution
