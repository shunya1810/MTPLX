"""#393: honest 507 refusal + sustained-pressure prefill abort.

The shipped failure: an explicitly overcommitted --context-window (by
design, warn-loudly) admitted a 262k-token prompt on a 128 GB machine; the
prefill grew a ~119 GB footprint, macOS compressed and swapped for minutes,
and no 507 ever fired because the only trigger was a Metal allocation
error that swap prevents. These tests pin the two new arms:

- request-time: prompts past the memory plan's fit are refused with a
  structured 507 (insufficient_memory) naming the honest ceiling, while
  prompts under the fit — and every non-overcommitted serve — pass exactly
  as before;
- runtime backstop: the guard loop arms a pressure-abort event only after
  sustained CRITICAL ticks with the engine busy, and the prefill abort
  plumbing consults it.
"""

from __future__ import annotations

import inspect
import threading

import pytest
from fastapi import HTTPException

import mtplx.server.openai as srv


class _Plan:
    def __init__(self, fit: int, resolved: int, overcommitted: bool) -> None:
        self.context_window_fit = fit
        self.context_window_resolved = resolved
        self.context_overcommitted = overcommitted


class _State:
    def __init__(self, window: int, plan: _Plan | None = None) -> None:
        self.context_window = window
        self.memory_plan = plan


class TestRejectPromptOverContext:
    def test_window_overflow_still_400(self):
        with pytest.raises(HTTPException) as e:
            srv._reject_prompt_over_context(_State(1000), 1000)
        assert e.value.status_code == 400
        assert e.value.detail["code"] == "context_length_exceeded"

    def test_overcommitted_prompt_past_fit_is_507(self):
        state = _State(262144, _Plan(112640, 262144, True))
        with pytest.raises(HTTPException) as e:
            srv._reject_prompt_over_context(state, 200000)
        assert e.value.status_code == 507
        assert e.value.detail["code"] == "insufficient_memory"
        assert "112640" in e.value.detail["message"]
        assert "200000" in e.value.detail["message"]

    def test_prompt_exactly_at_fit_refused(self):
        # At the fit there is zero headroom for even one generated token's
        # KV growth — same >= convention as the 400 window check.
        state = _State(262144, _Plan(112640, 262144, True))
        with pytest.raises(HTTPException) as e:
            srv._reject_prompt_over_context(state, 112640)
        assert e.value.status_code == 507

    def test_overcommitted_prompt_under_fit_serves(self):
        state = _State(262144, _Plan(112640, 262144, True))
        assert srv._reject_prompt_over_context(state, 100000) is None

    def test_honest_window_never_507s(self):
        # Not overcommitted: the plan says the resolved window fits, so the
        # fit branch must be inert even for huge prompts under the window.
        state = _State(262144, _Plan(112640, 262144, False))
        assert srv._reject_prompt_over_context(state, 200000) is None

    def test_no_plan_passes(self):
        assert srv._reject_prompt_over_context(_State(262144), 200000) is None

    def test_zero_fit_plan_is_inert(self):
        # A degenerate plan (fit unknown) must never refuse.
        state = _State(262144, _Plan(0, 262144, True))
        assert srv._reject_prompt_over_context(state, 200000) is None


class TestPressureAbortArming:
    def test_arms_only_after_sustained_busy_critical(self):
        state = _State(4096)
        state.pressure_abort_event = threading.Event()
        streak = 0
        for _ in range(srv._PRESSURE_ABORT_TICKS - 1):
            streak = srv._note_critical_pressure_tick(state, 4, True, streak)
            assert not srv._pressure_abort_requested(state)
        streak = srv._note_critical_pressure_tick(state, 4, True, streak)
        assert srv._pressure_abort_requested(state)

    def test_idle_engine_never_arms(self):
        state = _State(4096)
        state.pressure_abort_event = threading.Event()
        streak = 0
        for _ in range(srv._PRESSURE_ABORT_TICKS * 3):
            streak = srv._note_critical_pressure_tick(state, 4, False, streak)
        assert not srv._pressure_abort_requested(state)
        assert streak == 0

    def test_subcritical_tick_disarms_and_resets(self):
        state = _State(4096)
        state.pressure_abort_event = threading.Event()
        streak = 0
        for _ in range(srv._PRESSURE_ABORT_TICKS):
            streak = srv._note_critical_pressure_tick(state, 4, True, streak)
        assert srv._pressure_abort_requested(state)
        streak = srv._note_critical_pressure_tick(state, 2, True, streak)
        assert not srv._pressure_abort_requested(state)
        assert streak == 0
        # Re-arming requires the full sustained run again.
        streak = srv._note_critical_pressure_tick(state, 4, True, streak)
        assert not srv._pressure_abort_requested(state)

    def test_event_created_on_demand(self):
        state = _State(4096)  # no pressure_abort_event attribute
        streak = 0
        for _ in range(srv._PRESSURE_ABORT_TICKS):
            streak = srv._note_critical_pressure_tick(state, 4, True, streak)
        assert srv._pressure_abort_requested(state)

    def test_busy_flicker_resets_streak(self):
        state = _State(4096)
        state.pressure_abort_event = threading.Event()
        streak = srv._note_critical_pressure_tick(state, 4, True, 0)
        streak = srv._note_critical_pressure_tick(state, 4, False, streak)
        assert streak == 0


class TestPrefillAbortWiring:
    def test_request_lanes_and_postcommit_consult_pressure(self):
        src = inspect.getsource(srv)
        # The postcommit abort predicate.
        assert "or _pressure_abort_requested(state)" in src
        # The two request lanes (AR + MTP) share one prefill abort check. It
        # used to be two lambdas, each "... or _pressure_abort_requested
        # (state)"; since the per-chunk supply check joined them it is one
        # function that consults the pressure abort and that check.
        assert src.count("abort_check=_prefill_abort_check") >= 2
        start = src.index("def _prefill_abort_check() -> bool:")
        body = src[start : start + 400]
        assert "_pressure_abort_requested(state)" in body
        assert "prefill_system_guard()" in body

    def test_pressure_abort_maps_to_507_not_stream_cancel(self):
        src = inspect.getsource(srv)
        assert "sustained critical memory pressure during prefill" in src
        cut = src.index("sustained critical memory pressure during prefill")
        assert "_allocation_failure_http_exception" in src[cut - 2000 : cut]
