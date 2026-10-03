"""Live decode rate of the phase a streamed answer is in.

The live gauge used to divide every completion token by the time since the
first one. A slow reasoning phase then dragged the answer's number down for
the whole answer: on 2026-09-29 the gauge read about 40 tok/s through answers
that decoded at 57 to 79 tok/s, because reasoning had run at 36 to 49.

``PhaseRateMeter`` keeps that cumulative figure for receipts and adds the rate
of the current phase (reasoning, answer or tool call), measured over a short
window inside the phase from the producer's own timestamps.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Iterable

DECODE_PHASE_REASONING = "reasoning"
DECODE_PHASE_ANSWER = "answer"
DECODE_PHASE_TOOL_CALL = "tool_call"

# Stream-splitter field that carries the thinking text; every other visible
# field is the answer.
_REASONING_FIELD = "reasoning_content"


def decode_phase_for_fields(
    fields: Iterable[str], *, tool_call_open: bool
) -> str | None:
    """The phase the last drained piece of a token batch belongs to.

    ``fields`` are the splitter fields of the pieces the batch produced, in
    order; ``tool_call_open`` says whether the tool-call translator is inside
    a call after those pieces. A batch whose text is still held back
    (no pieces) returns None: the phase has not changed.
    """

    last: str | None = None
    for field in fields:
        last = field
    if last is None:
        return None
    if tool_call_open:
        return DECODE_PHASE_TOOL_CALL
    if last == _REASONING_FIELD:
        return DECODE_PHASE_REASONING
    return DECODE_PHASE_ANSWER


class PhaseRateMeter:
    """Decode rate inside the current phase, from producer timestamps.

    ``observe(tokens, t)`` records a token batch the generator produced at
    ``t`` (its own ``perf_counter`` stamp). ``note_phase(phase, t)`` records
    that the text drained from the batch at ``t`` belongs to ``phase``; a
    change starts the new phase at ``t``, so the batch that revealed it still
    counts toward the phase before (a batch is at most one verify round).

    ``rate(now)`` divides the tokens produced after a reference point by the
    time since it. The reference is the newest point at least ``window_s``
    back when the phase is that old, else the phase start. When the phase is
    younger than ``min_span_s`` the reference is the newest point at least
    ``min_span_s`` back, which may lie in the previous phase: the reading
    slides from the old phase's rate to the new one instead of dividing one
    batch by a few milliseconds. Before ``min_span_s`` of decode exists the
    rate is None. The span is therefore never shorter than ``min_span_s``, so
    there is no divide-by-zero and no first-batch spike, and a stall inside
    the window lowers the reading for as long as it is in the window.

    The first batch's tokens are the reference itself (their production time
    is unknown), so they never inflate a rate. Memory is bounded: points
    older than the reference are dropped, capped at ``max_points``.
    """

    def __init__(
        self,
        *,
        window_s: float = 5.0,
        min_span_s: float = 1.0,
        max_points: int = 1024,
    ) -> None:
        if min_span_s <= 0 or window_s < min_span_s:
            raise ValueError("need 0 < min_span_s <= window_s")
        self.window_s = float(window_s)
        self.min_span_s = float(min_span_s)
        self._points: deque[tuple[float, int]] = deque(maxlen=int(max_points))
        self._total = 0
        self._last_t: float | None = None
        self._phase: str | None = None
        # (time, cumulative tokens) where the current phase begins.
        self._phase_start: tuple[float, int] | None = None

    @property
    def phase(self) -> str | None:
        return self._phase

    def observe(self, tokens: int, t: float) -> None:
        count = int(tokens)
        if count <= 0:
            return
        t = float(t)
        if self._last_t is not None and t < self._last_t:
            # One producer stamps every batch in order; never let a clock
            # hiccup make a span negative.
            t = self._last_t
        self._total += count
        self._last_t = t
        self._points.append((t, self._total))
        if self._phase_start is None:
            self._phase_start = (t, self._total)

    def note_phase(self, phase: str | None, t: float) -> None:
        if phase is None or phase == self._phase:
            return
        if self._phase is None:
            # The first phase starts where decode started.
            self._phase = phase
            return
        self._phase = phase
        t = float(t)
        if self._last_t is not None and t < self._last_t:
            t = self._last_t
        self._phase_start = (t, self._total)

    def rate(self, now: float) -> float | None:
        if self._phase_start is None or self._last_t is None:
            return None
        now = max(float(now), self._last_t)
        phase_t = self._phase_start[0]
        reference: tuple[float, int] | None = None
        if now - phase_t >= self.window_s:
            # Newest point that still leaves a full window inside the phase.
            cutoff = now - self.window_s
            for point in self._points:
                if point[0] > cutoff:
                    break
                if point[0] >= phase_t:
                    reference = point
            if reference is None:
                reference = self._phase_start
        elif now - phase_t >= self.min_span_s:
            reference = self._phase_start
        else:
            cutoff = now - self.min_span_s
            for point in self._points:
                if point[0] > cutoff:
                    break
                reference = point
            if reference is None:
                return None
        self._prune(reference[0])
        span = now - reference[0]
        if span <= 0:
            return None
        return max(0, self._total - reference[1]) / span

    def _prune(self, keep_from: float) -> None:
        # Points older than the reference can only matter again through a
        # later reference, which is always newer; drop them.
        while len(self._points) > 1 and self._points[1][0] <= keep_from:
            self._points.popleft()

    def snapshot(self, now: float) -> dict[str, Any]:
        """The live fields a progress frame carries."""

        payload: dict[str, Any] = {
            "decode_phase": self._phase,
            "phase_tok_s": self.rate(now),
            "phase_tokens": 0,
            "phase_elapsed_s": 0.0,
        }
        if self._phase_start is not None:
            phase_t, phase_tokens = self._phase_start
            payload["phase_tokens"] = max(0, self._total - phase_tokens)
            payload["phase_elapsed_s"] = max(0.0, float(now) - phase_t)
        return payload
