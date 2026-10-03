"""A pressure trim never takes a conversation that is generating.

The review of 23a94abf (finding 1): the WARNING trim halves what the bank
holds (9c96dd9c, #525 C) and named no session to spare. After 60 s of WARNING
during a generation (the busy deferral's limit), an 8 GiB conversation that
was generating, alone in a 16 GiB bank, was evicted with its live cache
reference; the old half-the-budget target (8 GiB) took nothing there. Every
other reclamation step (the admission's LRU pass, the chain walk, the idle
release) already spares in-flight sessions; the pressure trim and the
allocation-failure shed now do too. An idle conversation is still taken.
"""

from __future__ import annotations

import functools
from types import SimpleNamespace

import pytest

import mtplx.server.openai as srv
import mtplx.system_memory as sm
from tests.test_memguard_admission import GIB, _manager, _put, _run_loop

MIB = 1024 * 1024
CONVERSATION = range(8_192)  # 8 GiB at 1 MiB a token
IDLE = range(500_000, 504_096)  # 4 GiB


def _reading(available_gib: float) -> sm.SystemMemory:
    # Nothing wired on a 128 GB Mac: abort floor 3.2 GiB, shed floor 6.4 GiB.
    return sm.SystemMemory(
        available_bytes=int(available_gib * GIB),
        total_bytes=128 * GIB,
        level_percent=20,
        free_bytes=int(min(available_gib, 2.0) * GIB),
        file_backed_bytes=int(max(0.0, available_gib - 2.0) * GIB),
        wired_bytes=0,
        compressor_bytes=4 * GIB,
        swap_used_bytes=0,
        monotonic_s=0.0,
    )


@pytest.fixture(autouse=True)
def _a_busy_engine_past_its_deferral(monkeypatch):
    # A request is running, and the WARNING has outlasted the 60 s the loop
    # waits for an idle engine (a zero deferral stands in for the minute).
    monkeypatch.setattr(srv, "_engine_busy_signal", lambda state: True)
    monkeypatch.setattr(
        srv,
        "_MemoryPressureGuard",
        functools.partial(srv._MemoryPressureGuard, warning_defer_max_s=0.0),
    )


def _state(manager):
    return SimpleNamespace(
        sessions=manager,
        dashboard=SimpleNamespace(last_memory_pressure_level=0),
    )


def _generating(manager, session_id: str):
    session = manager.get_or_create(session_id)
    assert session.try_begin_generation()
    return session


def _trims(state) -> list[dict]:
    events = getattr(state.dashboard, "memory_guard_events", ()) or ()
    return [event for event in events if event.get("action") == "pressure_trim"]


class TestTheTrimSparesTheConversationGenerating:
    def test_a_warning_after_the_deferral_leaves_it_whole(self, monkeypatch):
        """The review's case: the conversation generating is all the 16 GiB
        bank holds (8 GiB). Half of what the bank holds is 4 GiB, so the
        trim used to evict it and drop its live cache reference."""

        manager = _manager(max_bytes=16 * GIB, per_session_max_bytes=16 * GIB)
        entry = _put(
            manager.bank, CONVERSATION, session_id="busy", row_bytes=MIB, live_cache=True
        )
        live = entry.cache_ref
        session = _generating(manager, "busy")
        monkeypatch.setattr(sm, "_reader", lambda: _reading(5.0))
        state = _state(manager)
        try:
            _run_loop(state, monkeypatch, seconds=0.05)
        finally:
            session.end_generation()
        trims = _trims(state)
        assert trims and trims[0]["level"] == 2
        assert trims[0]["bank_entries_evicted"] == 0
        assert manager.bank._entries.get(entry.token_ids) is entry
        assert entry.cache_ref is live
        assert manager.bank.total_nbytes == 8 * GIB

    def test_a_warning_takes_the_idle_conversation_first_and_only(self, monkeypatch):
        manager = _manager(max_bytes=16 * GIB, per_session_max_bytes=16 * GIB)
        busy = _put(manager.bank, CONVERSATION, session_id="busy", row_bytes=MIB)
        idle = _put(manager.bank, IDLE, session_id="idle", row_bytes=MIB)
        busy.last_access_s -= 3_600.0  # older than the idle one: LRU alone would take it
        session = _generating(manager, "busy")
        monkeypatch.setattr(sm, "_reader", lambda: _reading(5.0))
        state = _state(manager)
        try:
            _run_loop(state, monkeypatch, seconds=0.05)
        finally:
            session.end_generation()
        # 12 GiB resident, a 6 GiB target: the idle 4 GiB goes, and the
        # trim stops at the conversation generating.
        assert idle.token_ids not in manager.bank._entries
        assert manager.bank._entries.get(busy.token_ids) is busy
        assert manager.bank.total_nbytes == 8 * GIB

    def test_a_critical_trim_empties_everything_but_it(self, monkeypatch):
        manager = _manager(max_bytes=16 * GIB, per_session_max_bytes=16 * GIB)
        busy = _put(manager.bank, CONVERSATION, session_id="busy", row_bytes=MIB)
        idle = _put(manager.bank, IDLE, session_id="idle", row_bytes=MIB)
        session = _generating(manager, "busy")
        monkeypatch.setattr(sm, "_reader", lambda: _reading(2.0))
        state = _state(manager)
        try:
            _run_loop(state, monkeypatch, seconds=0.05)
        finally:
            session.end_generation()
        trims = _trims(state)
        assert trims and trims[0]["level"] == 4
        assert idle.token_ids not in manager.bank._entries
        assert manager.bank._entries.get(busy.token_ids) is busy

    def test_an_idle_conversation_is_still_taken(self, monkeypatch):
        """Take-anything semantics for what is not generating: the same
        8 GiB conversation, idle, goes to the 4 GiB target."""

        manager = _manager(max_bytes=16 * GIB, per_session_max_bytes=16 * GIB)
        entry = _put(manager.bank, CONVERSATION, session_id="conv", row_bytes=MIB)
        monkeypatch.setattr(sm, "_reader", lambda: _reading(5.0))
        state = _state(manager)
        _run_loop(state, monkeypatch, seconds=0.05)
        assert entry.token_ids not in manager.bank._entries
        assert manager.bank.total_nbytes == 0


class TestTheAllocationFailureShed:
    def test_it_spares_the_failing_requests_conversation(self, monkeypatch):
        """After a Metal allocation failure the request's conversation is
        still in flight; its entries are what the retry restores from."""

        import mlx.core as mx

        monkeypatch.setattr(mx, "clear_cache", lambda: None)
        monkeypatch.setattr(srv, "_record_guard_event", lambda state, payload: None)
        manager = _manager(max_bytes=16 * GIB, per_session_max_bytes=16 * GIB)
        busy = _put(manager.bank, CONVERSATION, session_id="busy", row_bytes=MIB)
        idle = _put(manager.bank, IDLE, session_id="idle", row_bytes=MIB)
        busy.last_access_s -= 3_600.0
        session = _generating(manager, "busy")
        try:
            receipt = srv._shed_after_allocation_failure(_state(manager))
        finally:
            session.end_generation()
        assert "bank_error" not in receipt
        assert idle.token_ids not in manager.bank._entries
        assert manager.bank._entries.get(busy.token_ids) is busy
