"""A memory guard step that raises admits the request, arms the backstops,
and says so on /health and the dashboard stream (item a of the review of
9c96dd9c)."""

from __future__ import annotations

import pytest

import mtplx.server.openai as srv
from mtplx.server.dashboard_state import DashboardState
from tests.test_memguard_admission import (
    GIB,
    _flash_next_state,
    _install,
    _Machine,
    _manager,
)


@pytest.fixture(autouse=True)
def _served_profile(monkeypatch):
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", "1")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE", "auto")
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL_LAYOUT", "auto")
    monkeypatch.setenv("MTPLX_SUSTAINED_DENSE_DECODE_MAX_CONTEXT", "131072")


def _state():
    state = _flash_next_state(_manager())
    state.dashboard = DashboardState()
    return state


def _boom(*args, **kwargs):
    raise KeyError("broken bank plan")


def test_a_failing_admission_admits_and_reports_guard_degraded(monkeypatch):
    state = _state()
    monkeypatch.setattr(srv, "_run_prefill_admission", _boom)
    receipt = srv._prefill_admission_shed(
        state, prompt_ids=list(range(4_000)), session_bank=None, session_id=None
    )
    assert receipt["admitted_unchecked"] is True
    assert receipt["guard_degraded"] is True
    health = srv._memory_guard_health(state)
    assert health["guard_degraded"] is True
    assert health["errors"] == 1
    [record] = health["degraded"]
    assert record["where"] == "prefill_admission"
    assert "broken bank plan" in record["error"]
    # The guard event the app's dashboard stream carries.
    events = list(state.dashboard.memory_guard_events)
    assert events[-1]["action"] == "prefill_admission_shed_error"


def test_a_clean_admission_clears_it_and_keeps_the_last_error(monkeypatch):
    state = _state()
    monkeypatch.setattr(srv, "_run_prefill_admission", _boom)
    srv._prefill_admission_shed(
        state, prompt_ids=list(range(4_000)), session_bank=None, session_id=None
    )
    monkeypatch.setattr(srv, "_run_prefill_admission", lambda *a, **k: None)
    assert (
        srv._prefill_admission_shed(
            state, prompt_ids=list(range(4_000)), session_bank=None, session_id=None
        )
        is None
    )
    health = srv._memory_guard_health(state)
    assert health["guard_degraded"] is False
    assert health["errors"] == 1
    assert "broken bank plan" in health["last_error"]["error"]


def test_the_backstops_are_armed_without_an_admission_bill(monkeypatch):
    """No bill from the admission: the per-chunk check reserves the widest
    forward the prompt allows and holds the engine's limit."""

    state = _state()
    reserve = srv._prefill_chunk_reserve_bytes(
        state, prompt_tokens=30_000, chunk_tokens=4096, priced=None
    )
    assert reserve > 4096 * 32_448
    guard = srv._PrefillSystemGuard(state, chunk_reserve_bytes=reserve)
    assert guard.limit == 96 * GIB
    assert guard.chunk_reserve_bytes == reserve
    # And it trips on the engine line when the engine is at its limit.
    monkeypatch.setattr(srv, "_PREFILL_SYSTEM_CHECK_INTERVAL_S", 0.0)
    _install(
        monkeypatch,
        _Machine(state.sessions.bank, base_gib=95.0, cache_gib=0.0, host_gib=1.0),
    )
    monkeypatch.setattr(srv, "_read_system_memory", lambda: None)
    assert guard() is True
    assert guard.tripped["reason"] == "engine_limit"


def test_shared_guard_keeps_the_reservation_and_health_fallback(monkeypatch):
    from mtplx.memory_plan import RUNTIME_TRANSIENTS_BYTES

    state = _state()
    monkeypatch.setattr(srv, "_prefill_chunk_reserve_bytes", _boom)
    guard = srv.make_prefill_system_guard(
        state, prompt_tokens=4096, chunk_tokens=256, priced=None
    )

    assert guard.chunk_reserve_bytes == RUNTIME_TRANSIENTS_BYTES
    assert guard.after_prefill_reserve_bytes == RUNTIME_TRANSIENTS_BYTES
    assert guard.limit == 96 * GIB
    health = srv._memory_guard_health(state)
    assert health["guard_degraded"] is True
    assert health["degraded"][0]["where"] == "prefill_chunk_reserve"
    assert "broken bank plan" in health["last_error"]["error"]
    assert state.dashboard.memory_guard_events[-1]["action"] == "prefill_chunk_reserve_error"


def test_health_and_the_dashboard_stream_carry_it():
    import inspect

    health_src = inspect.getsource(srv.create_app)
    assert '"memory_guard": _memory_guard_health(state),' in health_src
    dashboard_src = inspect.getsource(srv._mtplx_dashboard_snapshot)
    assert '"memory_guard": _memory_guard_health(state),' in dashboard_src


def test_a_reclamation_step_that_raises_is_degraded_too(monkeypatch):
    """A bank step that raised was a `bank_error` in the receipt and nothing
    else: the admission's bank doubles once refused a keyword and the whole
    bank step did nothing, silently. It is now guard_degraded until a later
    admission gets through reclamation cleanly."""

    import mtplx.system_memory as sm
    from tests.test_memguard_admission import _put

    state = _state()
    bank = state.sessions.bank
    _put(bank, range(0, 10_000), session_id="old", row_bytes=GIB // 1_000)
    bank._session_last_active["old"] = 0.0

    def broken(*args, **kwargs):
        raise RuntimeError("shrink exploded")

    monkeypatch.setattr(bank, "shrink_to_bytes", broken)
    monkeypatch.setattr(sm, "_reader", lambda: None)
    monkeypatch.setattr(srv, "_record_guard_event", lambda state, payload: None)
    _install(
        monkeypatch,
        _Machine(bank, base_gib=88.0, cache_gib=0.0, host_gib=1.0),
    )
    receipt = srv._prefill_admission_shed(
        state,
        prompt_ids=list(range(1_000_000, 1_030_000)),
        session_bank=bank,
        session_id="fresh",
        prefill_chunk_tokens=4096,
    )
    assert "shrink exploded" in receipt["bank_error"]
    assert receipt["guard_degraded"] is True
    health = srv._memory_guard_health(state)
    assert health["guard_degraded"] is True
    assert health["degraded"][0]["where"] == "prefill_admission_reclamation"


class TestAFailedAllocatorReading:
    """The review of 23a94abf (finding 4): a failed MLX reading was read as
    zero active memory, so the admission returned None and the per-chunk
    check skipped the engine line, with guard_degraded false. A reading that
    fails is not a reading of zero: the step is degraded, the request is
    admitted unchecked, and the backstops run on what can still be read."""

    FAILED = (
        {"ok": False, "error": "mlx unavailable: ImportError()"},
        {"ok": True, "active_memory_bytes": None, "cache_memory_bytes": None},
    )

    @pytest.mark.parametrize("reading", FAILED)
    def test_the_admission_reports_it_and_admits_unchecked(self, monkeypatch, reading):
        state = _state()
        monkeypatch.setattr(srv, "_mlx_memory_stats_live", lambda: dict(reading))
        monkeypatch.setattr(srv, "_record_guard_event", lambda state, payload: None)
        receipt = srv._prefill_admission_shed(
            state,
            prompt_ids=list(range(30_000)),
            session_bank=state.sessions.bank,
            session_id="fresh",
            prefill_chunk_tokens=4096,
        )
        assert receipt is not None
        assert receipt["admitted_unchecked"] is True
        assert receipt["guard_degraded"] is True
        health = srv._memory_guard_health(state)
        assert health["guard_degraded"] is True
        assert health["degraded"][0]["where"] == "prefill_admission"
        assert "_AllocatorReadingError" in health["degraded"][0]["error"]

    @pytest.mark.parametrize("reading", FAILED)
    def test_the_per_chunk_check_reports_it_and_keeps_the_macs_lines(
        self, monkeypatch, reading
    ):
        import mtplx.system_memory as sm

        state = _state()
        monkeypatch.setattr(srv, "_PREFILL_SYSTEM_CHECK_INTERVAL_S", 0.0)
        monkeypatch.setattr(srv, "_mlx_memory_stats_live", lambda: dict(reading))
        supply = [40 * GIB]
        monkeypatch.setattr(
            sm,
            "_reader",
            lambda: sm.SystemMemory(
                available_bytes=supply[0],
                total_bytes=128 * GIB,
                level_percent=30,
                free_bytes=supply[0] // 2,
                file_backed_bytes=supply[0] // 2,
                wired_bytes=80 * GIB,
                compressor_bytes=GIB,
                swap_used_bytes=0,
            ),
        )
        guard = srv._PrefillSystemGuard(state, chunk_reserve_bytes=2 * GIB)
        # The Mac has room: no trip, but the check that could not read the
        # engine's account is reported.
        assert guard() is False
        health = srv._memory_guard_health(state)
        assert health["guard_degraded"] is True
        assert health["degraded"][0]["where"] == "prefill_system_check"
        # The Mac's own line still stops the prefill.
        supply[0] = 6 * GIB
        assert guard() is True
        assert guard.tripped["reason"] == "under_abort_floor"

    def test_a_clean_reading_clears_it(self, monkeypatch):
        import mtplx.system_memory as sm

        state = _state()
        monkeypatch.setattr(srv, "_PREFILL_SYSTEM_CHECK_INTERVAL_S", 0.0)
        monkeypatch.setattr(sm, "_reader", lambda: None)
        readings = [dict(self.FAILED[0])]
        monkeypatch.setattr(srv, "_mlx_memory_stats_live", lambda: readings[0])
        guard = srv._PrefillSystemGuard(state, chunk_reserve_bytes=2 * GIB)
        assert guard() is False
        assert srv._memory_guard_health(state)["guard_degraded"] is True
        readings[0] = {"ok": True, "active_memory_bytes": 40 * GIB, "cache_memory_bytes": 0}
        monkeypatch.setattr(srv, "phys_footprint_bytes", lambda *a, **k: 0)
        assert guard() is False
        assert srv._memory_guard_health(state)["guard_degraded"] is False
