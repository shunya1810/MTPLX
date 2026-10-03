"""The live gauge's rate: the current decode phase, not the whole answer.

Receipts keep the cumulative rate since the first token. The live frames also
carry the phase (reasoning, answer, tool call) and that phase's own recent
rate, measured from the producer's timestamps, so a slow reasoning phase no
longer drags the answer's number down (2026-09-29: "40 even in the answer").
All timestamps here are synthetic, so every expected figure is exact.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

from mtplx.server.stream_rate import (
    DECODE_PHASE_ANSWER,
    DECODE_PHASE_REASONING,
    DECODE_PHASE_TOOL_CALL,
    PhaseRateMeter,
    decode_phase_for_fields,
)


def _feed(meter, *, phase, start, seconds, interval, tokens_per_batch, frames):
    """Stream batches the way the SSE loop does: observe, frame, then note."""

    t = start
    steps = int(round(seconds / interval))
    for _ in range(steps):
        t = round(t + interval, 9)
        meter.observe(tokens_per_batch, t)
        frames.append((t, meter.snapshot(t)))
        meter.note_phase(phase, t)
    return t


def test_slow_reasoning_then_fast_answer_reports_the_answer_rate():
    from mtplx.server import openai

    meter = PhaseRateMeter()
    frames: list[tuple[float, dict]] = []
    # 20 s of reasoning at 10 tok/s, then 10 s of answer at 50 tok/s.
    meter.observe(1, 0.0)
    meter.note_phase(DECODE_PHASE_REASONING, 0.0)
    t = _feed(meter, phase=DECODE_PHASE_REASONING, start=0.0, seconds=20.0,
              interval=0.1, tokens_per_batch=1, frames=frames)
    reasoning_frame = frames[-1][1]
    assert reasoning_frame["decode_phase"] == DECODE_PHASE_REASONING
    assert reasoning_frame["phase_tok_s"] == pytest.approx(10.0)

    t = _feed(meter, phase=DECODE_PHASE_ANSWER, start=t, seconds=10.0,
              interval=0.1, tokens_per_batch=5, frames=frames)
    payload = openai._stream_progress_payload(
        completion_tokens=1 + 200 + 500,
        decode_started_s=0.0,
        now_s=t,
        phase_meter=meter,
    )
    # Cumulative stays the receipt figure; the gauge gets the answer's rate.
    assert payload["decode_tok_s"] == pytest.approx(701 / 30.0)
    assert payload["decode_phase"] == DECODE_PHASE_ANSWER
    assert payload["phase_tok_s"] == pytest.approx(50.0)
    # The answer's rate shows within a second, not after the whole answer.
    one_second_in = [snap for when, snap in frames if when == pytest.approx(21.1)]
    assert one_second_in[0]["decode_phase"] == DECODE_PHASE_ANSWER
    assert one_second_in[0]["phase_tok_s"] == pytest.approx(50.0)


def test_phase_start_never_spikes_or_divides_by_zero():
    meter = PhaseRateMeter()
    frames: list[tuple[float, dict]] = []
    meter.observe(1, 0.0)
    meter.note_phase(DECODE_PHASE_REASONING, 0.0)
    t = _feed(meter, phase=DECODE_PHASE_REASONING, start=0.0, seconds=10.0,
              interval=0.1, tokens_per_batch=1, frames=frames)
    # The first answer batches, including two stamped at the same instant.
    for dt in (0.1, 0.1, 0.0, 0.1):
        t = round(t + dt, 9)
        meter.observe(5, t)
        frames.append((t, meter.snapshot(t)))
        meter.note_phase(DECODE_PHASE_ANSWER, t)
    rates = [snap["phase_tok_s"] for _when, snap in frames if snap["phase_tok_s"] is not None]
    assert all(math.isfinite(rate) for rate in rates)
    # Never above the fastest real production rate (5 tokens per 0.1 s).
    assert max(rates) <= 50.0 + 1e-9
    # The reading slides from the reasoning rate towards the answer rate.
    assert frames[-1][1]["phase_tok_s"] > 10.0


def test_no_rate_before_a_minimum_span_and_first_batch_is_not_counted():
    meter = PhaseRateMeter(window_s=5.0, min_span_s=1.0)
    # The first batch arrives after a long prefill: its 4 tokens were made
    # before any timestamp we have, so they must never inflate a rate.
    meter.observe(4, 100.0)
    meter.note_phase(DECODE_PHASE_ANSWER, 100.0)
    assert meter.rate(100.0) is None
    meter.observe(2, 100.5)
    assert meter.rate(100.5) is None
    meter.observe(2, 101.0)
    assert meter.rate(101.0) == pytest.approx(4.0)


def test_a_stall_lowers_the_rate_and_it_recovers_within_the_window():
    meter = PhaseRateMeter(window_s=5.0, min_span_s=1.0)
    frames: list[tuple[float, dict]] = []
    meter.observe(1, 0.0)
    meter.note_phase(DECODE_PHASE_ANSWER, 0.0)
    t = _feed(meter, phase=DECODE_PHASE_ANSWER, start=0.0, seconds=10.0,
              interval=0.1, tokens_per_batch=5, frames=frames)
    assert frames[-1][1]["phase_tok_s"] == pytest.approx(50.0)
    # Four silent seconds, then one batch: the stall is inside the window.
    t += 4.0
    meter.observe(5, t)
    after_stall = meter.snapshot(t)["phase_tok_s"]
    assert after_stall < 50.0 * 0.5
    frames.clear()
    _feed(meter, phase=DECODE_PHASE_ANSWER, start=t, seconds=6.0,
          interval=0.1, tokens_per_batch=5, frames=frames)
    assert frames[-1][1]["phase_tok_s"] == pytest.approx(50.0)


def test_tool_call_output_is_its_own_phase():
    meter = PhaseRateMeter()
    frames: list[tuple[float, dict]] = []
    meter.observe(1, 0.0)
    meter.note_phase(DECODE_PHASE_ANSWER, 0.0)
    t = _feed(meter, phase=DECODE_PHASE_ANSWER, start=0.0, seconds=5.0,
              interval=0.1, tokens_per_batch=3, frames=frames)
    t = _feed(meter, phase=DECODE_PHASE_TOOL_CALL, start=t, seconds=3.0,
              interval=0.1, tokens_per_batch=8, frames=frames)
    last = frames[-1][1]
    assert last["decode_phase"] == DECODE_PHASE_TOOL_CALL
    assert last["phase_tok_s"] == pytest.approx(80.0)
    assert last["phase_tokens"] == 8 * 29  # the switching batch counts before
    assert last["phase_elapsed_s"] == pytest.approx(2.9)


def test_phase_of_a_drained_batch():
    assert decode_phase_for_fields([], tool_call_open=False) is None
    assert decode_phase_for_fields(["reasoning_content"], tool_call_open=False) == (
        DECODE_PHASE_REASONING
    )
    assert decode_phase_for_fields(
        ["reasoning_content", "content"], tool_call_open=False
    ) == DECODE_PHASE_ANSWER
    assert decode_phase_for_fields(["content"], tool_call_open=True) == (
        DECODE_PHASE_TOOL_CALL
    )


def test_meter_memory_stays_bounded_over_a_long_answer():
    meter = PhaseRateMeter(window_s=5.0, min_span_s=1.0)
    t = 0.0
    meter.observe(1, t)
    meter.note_phase(DECODE_PHASE_ANSWER, t)
    for _ in range(100_000):
        t += 0.05
        meter.observe(3, t)
        meter.rate(t)
    assert len(meter._points) <= 5.0 / 0.05 + 2
    assert meter.rate(t) == pytest.approx(60.0)


def test_live_history_records_the_phase_rate_not_the_whole_answer_average():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_server_openai import _fake_state
    from mtplx.server import openai

    state = _fake_state()
    openai._dashboard_publish_progress(
        state,
        request_id="chatcmpl-phase",
        payload={
            "decode_tok_s": 40.0,
            "decode_phase": DECODE_PHASE_ANSWER,
            "phase_tok_s": 70.0,
            "completion_tokens": 900,
            "session_id": "sess-phase",
        },
    )
    rolling = state.dashboard.rolling.snapshot()
    assert rolling["live_history"][-1]["tok_s"] == 70.0
    # A frame without a phase rate (older producer) keeps the old behaviour.
    state = _fake_state()
    openai._dashboard_publish_progress(
        state,
        request_id="chatcmpl-old",
        payload={"decode_tok_s": 40.0, "completion_tokens": 12},
    )
    assert state.dashboard.rolling.snapshot()["live_history"][-1]["tok_s"] == 40.0
