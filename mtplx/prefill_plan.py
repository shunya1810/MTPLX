"""Why a request reads part of its prompt again, said before the wait.

On 2026-09-29 every screenshot turn of a Pi session re-read 123,000 to 138,000
tokens cold, 126 to 138 s each, while the app showed only "tokens done out of
total", and afterwards raw codes such as "miss · ssd_prefix_miss". The prefill
event now carries, right before the replay starts:

- how much of the prompt matches this conversation's cached history;
- the token the running state resumes at, and where it came from (RAM, SSD,
  or nothing: a cold read);
- how many tokens are computed again;
- the cause as a code (plus an English sentence for API clients and logs),
  which the app turns into a plain sentence in the user's language;
- an estimate from the measured prefill rate (added by the server).

``reread_facts`` gathers the numbers in generation from the session bank's
own accessors (no tensor work; one prefix compare only when this
conversation's cached history reaches past the resume point).
``explain_reread`` is the pure classifier the server applies;
``publishable_reread`` adds the estimate and is what the prefill event, the
in-flight snapshot and the request receipt carry.
"""

from __future__ import annotations

from typing import Any, Iterable

# Why the cache did not cover the prompt (None: the prompt only extends it).
CAUSE_HISTORY_CHANGED = "history_changed"
# The prompt shares only its opening (under _SHORT_SHARED_PREFIX_MAX tokens,
# and under a tenth of the saved history) with the conversation's saved state.
# Pi sends its compaction's two summaries and the turn after them under one
# session id; they share 110 and 41 tokens with what came before (2026-10-01),
# and "History changed at token 41" read as a bug. An early edit of a long
# chat looks the same, so the text states the overlap, not why.
CAUSE_SHORT_SHARED_PREFIX = "short_shared_prefix"
_SHORT_SHARED_PREFIX_MAX = 2048
CAUSE_SCREENSHOT_CHANGED = "screenshot_changed"
CAUSE_NEW_CONVERSATION = "new_conversation"
CAUSE_NOT_CACHED = "not_cached"
CAUSE_SWITCHED_CONVERSATION = "switched_conversation"
CAUSE_FREED_FOR_MEMORY = "freed_for_memory"
CAUSE_TOO_LARGE = "too_large"
CAUSE_EVICTED = "evicted"
CAUSE_SETTINGS_CHANGED = "settings_changed"
CAUSE_CACHE_OFF = "cache_off"

# Why the running state resumes before the end of the matched history.
LIMIT_SCREENSHOT = "screenshot"
LIMIT_SAVED_STATE = "saved_state"

SOURCE_RAM = "ram"
SOURCE_SSD = "ssd"
SOURCE_NONE = "none"

# Bank eviction reasons (session_bank eviction_log; the reasons the memory
# guard, the prefill admission and the idle release pass to the bank).
_MEMORY_EVICTION_PREFIXES = ("memory_pressure", "prefill_admission", "prefill_shed")
_MEMORY_EVICTIONS = frozenset(
    {
        "dynamic_ceiling",
        "allocation_failure",
        "idle_session_release",
        "queued_persistence_release",
        "owner_idle",
    }
)
_TOO_LARGE_EVICTIONS = frozenset(
    {"skipped_oversized_snapshot", "skipped_dense_materializing_snapshot"}
)
_BUDGET_EVICTIONS = frozenset({"evicted", "session_entry_retention"})
# Replacements by the same conversation's newer entry: not a loss.
_SUPERSEDE_EVICTIONS = frozenset(
    {"superseded_by_longer_prefix", "superseded_session_lease"}
)
_SETTINGS_MISSES = frozenset({"model_mismatch", "template_mismatch", "policy_mismatch"})

_CAUSE_TEXT = {
    CAUSE_NEW_CONVERSATION: "New conversation or first turn since MTPLX started",
    CAUSE_NOT_CACHED: "No saved state for this conversation",
    CAUSE_SWITCHED_CONVERSATION: "Switched conversations: this one's saved state was replaced",
    CAUSE_FREED_FOR_MEMORY: "Saved state was freed to relieve memory pressure",
    CAUSE_TOO_LARGE: "The conversation was too large to keep in memory",
    CAUSE_EVICTED: "Saved state was removed from memory",
    CAUSE_SETTINGS_CHANGED: "Settings changed since the last turn",
    CAUSE_CACHE_OFF: "The prompt cache is off for this request",
}


def _session_eviction_reason(eviction_log: Iterable[Any], session_id: str) -> str | None:
    reason: str | None = None
    for record in eviction_log:
        if isinstance(record, dict) and record.get("session_id") == session_id:
            reason = str(record.get("reason") or "") or None
    return reason


def reread_facts(
    *,
    session_bank: Any | None,
    session_id: str | None,
    bank_ids: list[int] | tuple[int, ...],
    restore_point: int,
    source: str,
    image_spans: Iterable[tuple[int, int]] | None = None,
) -> dict[str, Any]:
    """The numbers the prefill event carries right before the replay starts.

    ``bank_ids`` is the prompt as the bank sees it (content-keyed for images),
    ``restore_point`` the token the running state resumes at (0 when cold),
    ``source`` where that state came from, ``image_spans`` the [start, end)
    positions of the prompt's images.
    """

    prompt_tokens = len(bank_ids)
    restore_point = max(0, min(int(restore_point), prompt_tokens))
    facts: dict[str, Any] = {
        "prompt_tokens": prompt_tokens,
        "restore_point_tokens": restore_point,
        "recompute_tokens": prompt_tokens - restore_point,
        "history_matched_tokens": restore_point,
        "session_cached_tokens": 0,
        "source": source if restore_point > 0 else SOURCE_NONE,
        "cache_off": session_bank is None,
    }
    if session_bank is not None:
        _add_bank_facts(facts, session_bank, session_id, bank_ids, restore_point)
    matched = int(facts["history_matched_tokens"])
    spans = [(int(start), int(end)) for start, end in image_spans or ()]
    # The history parts from the prompt inside an image: that image changed.
    facts["image_changed_at"] = (
        next((start for start, end in spans if start <= matched < end), None)
        if matched < prompt_tokens
        else None
    )
    # An image between the resume point and the end of the match: restores
    # never resume inside or past an image the lane cannot prove unchanged.
    facts["image_limit_at"] = next(
        (start for start, _end in spans if restore_point <= start < matched), None
    )
    return facts


def _add_bank_facts(
    facts: dict[str, Any],
    session_bank: Any,
    session_id: str | None,
    bank_ids: list[int] | tuple[int, ...],
    restore_point: int,
) -> None:
    cold_tier = getattr(session_bank, "cold_tier", None)
    facts["ssd_checked"] = bool(
        cold_tier is not None and getattr(cold_tier, "enabled", False)
    )
    miss = getattr(session_bank, "last_miss_reason", None)
    if miss:
        facts["miss_reason"] = str(miss)
    rows: list[dict[str, Any]] = []
    held = getattr(session_bank, "held_by_session", None)
    if callable(held):
        rows = list(held())
    own = [row for row in rows if session_id and row.get("session_id") == session_id]
    own_longest = max((int(row.get("longest_prefix_tokens") or 0) for row in own), default=0)
    facts["other_sessions_cached"] = any(
        row.get("session_id") != session_id for row in rows
    )
    matched = restore_point
    if session_id and own_longest > restore_point:
        # This conversation's cached history reaches past the resume point:
        # find where the prompt leaves it (the only prefix compare here).
        shared = getattr(session_bank, "longest_shared_prefix_tokens", None)
        if callable(shared):
            matched = max(matched, int(shared(bank_ids, session_id=session_id)))
    diagnostic = getattr(session_bank, "last_prefix_diagnostic", None)
    refused = (
        int(diagnostic.get("ram_refused_prefix_len") or 0)
        if isinstance(diagnostic, dict)
        else 0
    )
    if (
        session_id
        and restore_point < refused <= len(bank_ids)
        and diagnostic.get("ram_refused_session_id") == session_id
    ):
        # The restore refused this conversation's longest saved state, a
        # prefix of this prompt (SessionBank.restore), and resumed earlier;
        # a cold restore may have released it since. The history matched
        # that far, and the refusal says why the state resumes before it.
        facts["ram_miss_reason"] = str(diagnostic.get("ram_miss_reason") or "")
        own_longest = max(own_longest, refused)
        matched = max(matched, refused)
    facts["session_cached_tokens"] = own_longest
    facts["history_matched_tokens"] = min(matched, len(bank_ids))
    if session_id and not own:
        eviction_log = getattr(session_bank, "eviction_log", None) or ()
        reason = _session_eviction_reason(list(eviction_log), session_id)
        if reason is not None:
            facts["session_eviction_reason"] = reason


def explain_reread(
    facts: dict[str, Any], *, session_served_before: bool = False
) -> dict[str, Any]:
    """Classify the facts into a cause and a resume limit, with English text.

    ``session_served_before``: this server committed earlier turns of the
    conversation (the server's own session record says so).
    """

    prompt = int(facts.get("prompt_tokens") or 0)
    restore = int(facts.get("restore_point_tokens") or 0)
    matched = max(restore, int(facts.get("history_matched_tokens") or 0))
    cached = int(facts.get("session_cached_tokens") or 0)
    source = str(facts.get("source") or SOURCE_NONE)
    cause: str | None = None
    cause_at: int | None = None
    if facts.get("cache_off"):
        cause = CAUSE_CACHE_OFF
    elif str(facts.get("miss_reason") or "") in _SETTINGS_MISSES and restore == 0:
        cause = CAUSE_SETTINGS_CHANGED
    elif cached > 0 and matched < min(cached, prompt):
        # The conversation's saved history and the prompt part ways here.
        image = facts.get("image_changed_at")
        if image is not None:
            cause = CAUSE_SCREENSHOT_CHANGED
            cause_at = int(image)
        elif matched < _SHORT_SHARED_PREFIX_MAX and matched * 10 < cached:
            cause = CAUSE_SHORT_SHARED_PREFIX
            cause_at = matched
        else:
            cause = CAUSE_HISTORY_CHANGED
            cause_at = matched
    elif cached == 0 and source != SOURCE_SSD:
        reason = facts.get("session_eviction_reason")
        if reason in _MEMORY_EVICTIONS or str(reason or "").startswith(
            _MEMORY_EVICTION_PREFIXES
        ):
            cause = CAUSE_FREED_FOR_MEMORY
        elif reason in _TOO_LARGE_EVICTIONS:
            cause = CAUSE_TOO_LARGE
        elif reason in _BUDGET_EVICTIONS:
            cause = (
                CAUSE_SWITCHED_CONVERSATION
                if facts.get("other_sessions_cached")
                else CAUSE_EVICTED
            )
        elif reason is not None and reason not in _SUPERSEDE_EVICTIONS:
            cause = CAUSE_EVICTED
        elif restore == 0:
            cause = CAUSE_NOT_CACHED if session_served_before else CAUSE_NEW_CONVERSATION
    limit: str | None = None
    limit_at: int | None = None
    if restore < matched:
        image_limit = facts.get("image_limit_at")
        if image_limit is not None:
            limit = LIMIT_SCREENSHOT
            limit_at = int(image_limit)
        else:
            limit = LIMIT_SAVED_STATE
            limit_at = restore
    explanation: dict[str, Any] = {
        "prompt_tokens": prompt,
        "history_matched_tokens": matched,
        "restore_point_tokens": restore,
        "recompute_tokens": max(0, prompt - restore),
        "source": source,
        "cause": cause,
        "cause_at_token": cause_at,
        "resume_limit": limit,
        "resume_limit_at_token": limit_at,
        "ssd_checked": bool(facts.get("ssd_checked")),
    }
    # The raw codes behind the cause, for receipts and logs (the app shows
    # its own sentence for the cause, never these).
    for key in ("miss_reason", "ram_miss_reason", "session_eviction_reason"):
        if facts.get(key):
            explanation[key] = str(facts[key])
    explanation["text"] = reread_text(explanation)
    return explanation


def publishable_reread(
    facts: dict[str, Any],
    *,
    session_served_before: bool,
    rates: dict[str, Any] | None,
) -> dict[str, Any]:
    """The explanation with its time estimate, as the server publishes it.

    ``rates``: the dashboard's recent prefill chunk work
    (``PrefillHistory.rates``). ``eta_s`` is None until a prefill was
    measured on this server.
    """

    explanation = explain_reread(facts, session_served_before=session_served_before)
    eta = estimate_seconds(int(explanation["recompute_tokens"]), rates)
    explanation["eta_s"] = round(eta, 1) if eta is not None else None
    explanation["text"] = reread_text(explanation, eta)
    return explanation


def reread_text(explanation: dict[str, Any], eta_s: float | None = None) -> str:
    """One English sentence per fact, for API clients, logs and receipts."""

    parts: list[str] = []
    cause = explanation.get("cause")
    at = int(explanation.get("cause_at_token") or 0)
    if cause == CAUSE_HISTORY_CHANGED:
        parts.append(
            "History changed from the start" if at <= 0 else f"History changed at token {at:,}"
        )
    elif cause == CAUSE_SHORT_SHARED_PREFIX:
        parts.append(f"Only the first {at:,} tokens match this conversation's saved state")
    elif cause == CAUSE_SCREENSHOT_CHANGED:
        parts.append(f"The screenshot at token {at:,} changed")
    elif cause in _CAUSE_TEXT:
        parts.append(_CAUSE_TEXT[cause])
    limit = explanation.get("resume_limit")
    limit_at = int(explanation.get("resume_limit_at_token") or 0)
    if limit == LIMIT_SCREENSHOT:
        parts.append(f"Resuming before the screenshot at token {limit_at:,}")
    elif limit == LIMIT_SAVED_STATE and limit_at > 0:
        parts.append(f"Resuming from the saved state at token {limit_at:,}")
    recompute = int(explanation.get("recompute_tokens") or 0)
    restore = int(explanation.get("restore_point_tokens") or 0)
    if explanation.get("source") == SOURCE_SSD and restore > 0:
        parts.append(f"Restored {restore:,} tokens from the SSD cache")
    reading = (
        f"Reading {recompute:,} new tokens"
        if cause is None and limit is None
        else f"Re-reading {recompute:,} tokens"
    )
    if eta_s is not None:
        reading += f", about {_duration(eta_s)}"
    parts.append(reading)
    return ". ".join(parts) + "."


def _duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{max(1, round(seconds))} s"
    minutes, rest = divmod(int(round(seconds)), 60)
    return f"{minutes} min {rest} s" if rest else f"{minutes} min"


def estimate_seconds(recompute_tokens: int, rates: dict[str, Any] | None) -> float | None:
    """Recompute time from the measured prefill rate (recent chunk work).

    None without a measurement: an estimate is never invented. The rate is
    the recent average, not a function of depth, so it is an estimate.
    """

    if not rates or recompute_tokens <= 0:
        return None
    tokens = float(rates.get("tokens") or 0)
    seconds = float(rates.get("compute_time_s") or 0)
    if tokens <= 0 or seconds <= 0:
        return None
    return float(recompute_tokens) / (tokens / seconds)
