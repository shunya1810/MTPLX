"""A memory release owns a session the way a request does, with no gaps.

The review of 9c96dd9c (finding 7):

* ``begin_release_hold`` took the slot and then published an odd release
  sequence; ``end_release_hold`` published the even one and then gave the
  slot back. A request that tried the slot in either gap read an even
  sequence with the slot taken and reported the session busy instead of
  waiting: a 409 for a named session, a forked one for an implicit session;
* a release hold left the session's ``in_flight`` flag down, so stale
  eviction (reached through ``list_sessions``) could drop the record while
  the release still held its lock; the next request created a second record
  under the same id, took its fresh lock, and restored while the first one's
  entries were being evicted.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace

from mtplx.engine_session import EngineSessionManager
from mtplx.session_bank import SessionBank

RUNTIME = SimpleNamespace(model_path=Path("models/example"), mtp_enabled=True)


def _manager() -> EngineSessionManager:
    bank = SessionBank(max_entries=32, max_bytes=10_000, per_session_max_bytes=10_000)
    return EngineSessionManager(bank=bank, idle_ttl_s=3600)


class _GappedLock:
    """A lock that pauses its next holder right after it takes the lock:
    the moment between taking the slot and publishing the sequence."""

    def __init__(self) -> None:
        self._inner = threading.Lock()
        self.pause_next = False
        self.in_gap = threading.Event()
        self.resume = threading.Event()

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        if blocking:
            got = self._inner.acquire(True, timeout)
        else:
            got = self._inner.acquire(False)
        if got and self.pause_next:
            self.pause_next = False
            self.in_gap.set()
            self.resume.wait(5)
        return got

    def release(self) -> None:
        self._inner.release()

    def locked(self) -> bool:
        return self._inner.locked()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *exc) -> None:
        self.release()


class TestTheSequence:
    def test_a_request_between_the_slot_and_the_sequence_waits(self):
        manager = _manager()
        session = manager.get_or_create("named")
        lock = _GappedLock()
        session._lock = lock
        lock.pause_next = True
        held: list[bool] = []
        release = threading.Thread(target=lambda: held.append(session.begin_release_hold()))
        release.start()
        assert lock.in_gap.wait(2)
        results: list[bool] = []
        request = threading.Thread(
            target=lambda: results.append(session.try_begin_generation())
        )
        request.start()
        time.sleep(0.1)
        # Still inside the release's gap: the request must not have been
        # told the session is busy.
        assert results == []
        lock.resume.set()
        release.join(2)
        assert held == [True]
        time.sleep(0.05)
        assert results == []
        session.end_release_hold()
        request.join(3)
        assert results == [True]
        session.end_generation()

    def test_a_generation_still_reads_busy_at_once(self):
        manager = _manager()
        session = manager.get_or_create("named")
        assert session.try_begin_generation()
        try:
            started = time.monotonic()
            assert not session.try_begin_generation()
            assert time.monotonic() - started < 0.5
        finally:
            session.end_generation()


class TestTheRecord:
    def test_stale_eviction_keeps_a_held_record(self):
        manager = _manager()
        session = manager.get_or_create("conv")
        # Idle past its TTL: a stale record the registry would drop.
        session.last_access_s = time.time() - 7_200.0
        assert session.begin_release_hold()
        try:
            assert session.in_flight is True
            assert manager.evict_stale() == 0
            # The next request finds the same record, so no second lock can
            # exist for the id while the release holds the first.
            assert manager.get_or_create("conv") is session
            assert "conv" in manager.in_flight_session_ids()
        finally:
            session.end_release_hold()
        assert session.in_flight is False
        assert session.release_held is False

    def test_a_record_that_is_not_held_still_goes_stale(self):
        manager = _manager()
        session = manager.get_or_create("conv")
        session.last_access_s = time.time() - 7_200.0
        assert manager.evict_stale() == 1
        assert manager.get_or_create("conv") is not session


class TestTheSlotAndItsFlag:
    """The review of 23a94abf (finding 5): the slot was taken before the
    flag went up, and stale eviction read only the flag, without the guard
    the slot is taken under. Paused in that interval, the record was
    dropped and a replacement created and taken: two held slots for one
    session id."""

    def test_a_request_between_its_slot_and_its_flag_keeps_the_record(self):
        manager = _manager()
        session = manager.get_or_create("conv")
        lock = _GappedLock()
        session._lock = lock
        lock.pause_next = True
        results: list[bool] = []
        # A waiting acquire (a handoff poll): it takes the slot outside the
        # guard, and its flag follows.
        request = threading.Thread(
            target=lambda: results.append(session.try_begin_generation(timeout_s=1.0))
        )
        request.start()
        assert lock.in_gap.wait(2)
        try:
            # The slot is taken and the flag is not up; the record has aged
            # past its TTL.
            session.last_access_s = time.time() - 7_200.0
            assert manager.evict_stale() == 0
            assert manager.get_or_create("conv") is session
        finally:
            lock.resume.set()
            request.join(3)
        assert results == [True]
        # One record, one slot: a second request finds it taken.
        assert not manager.get_or_create("conv").try_begin_generation()
        session.end_generation()

    def test_a_release_taking_the_slot_keeps_the_record(self):
        manager = _manager()
        session = manager.get_or_create("conv")
        lock = _GappedLock()
        session._lock = lock
        lock.pause_next = True
        held: list[bool] = []
        release = threading.Thread(target=lambda: held.append(session.begin_release_hold()))
        release.start()
        assert lock.in_gap.wait(2)
        session.last_access_s = time.time() - 7_200.0
        evicted: list[int] = []
        sweeper = threading.Thread(target=lambda: evicted.append(manager.evict_stale()))
        sweeper.start()
        # The sweep waits for the release to finish taking the slot.
        sweeper.join(0.3)
        lock.resume.set()
        release.join(3)
        sweeper.join(3)
        try:
            assert held == [True]
            assert evicted == [0]
            assert manager.get_or_create("conv") is session
            assert session.release_held is True
        finally:
            session.end_release_hold()
        assert session.slot_in_use() is False
