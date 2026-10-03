"""A queued idle-lane job holds its entry's arrays: every eviction accounts for it.

The review of 9c96dd9c (finding 5): the chain walk evicted snapshots without
cancelling their queued SSD encodes, and the whole-session release looked
only at the entries still in the bank, so it could not find them: after a
chain eviction and ``release_sessions(None)`` the bank was empty, the encode
was still queued, its snapshot still resident, and nothing had been
cancelled. The idle lane does not run while the engine is busy, so under a
stream of requests that memory never came back ("nothing left to release,
still 507").

Now every memory-pressure eviction cancels the evicted entry's own queued
jobs (settle and encode), budget and supersede evictions keep the encode
and the bank accounts for what it holds (``queued_persistence``), the
admission cancels those before it touches anyone's RAM state, and the
release finds holders by the queue.
"""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import mtplx.server.openai as srv
import mtplx.system_memory as sm
from mtplx.engine_session import EngineSessionManager
from mtplx.session_bank import SessionBank
from tests.test_memguard_admission import (
    GIB,
    V_SCRATCH_NARROW,
    V_SCRATCH_WIDE,
    _flash_next_state,
    _install,
    _Machine,
    _put,
)

RUNTIME = SimpleNamespace(model_path=Path("models/example"), mtp_enabled=True)


class _Tier:
    """Cold tier double: records encodes, publishes nothing."""

    def __init__(self) -> None:
        self.encoded: list[tuple[int, ...]] = []

    def put_entry(self, entry, capabilities=(), raise_on_yield=False):
        self.encoded.append(entry.token_ids)
        return True

    def is_published(self, entry) -> bool:
        return False


class _Lane:
    """The scheduler's idle persistence lane: newest job per key, cancel by
    key; a queued job keeps what its closure captured."""

    def __init__(self) -> None:
        self.pending: dict[str, object] = {}

    def dispatch(self, job) -> None:
        self.pending[job.coalesce_key] = job

    def cancel(self, key: str) -> int:
        return 1 if self.pending.pop(key, None) is not None else 0


def _bank(lane: _Lane, **kwargs) -> SessionBank:
    defaults = dict(max_entries=64, max_bytes=60 * GIB, per_session_max_bytes=30 * GIB)
    defaults.update(kwargs)
    bank = SessionBank(cold_tier=_Tier(), **defaults)
    bank.cold_enqueue_dispatch = lane.dispatch
    bank.cold_enqueue_cancel = lane.cancel
    return bank


def _age(bank: SessionBank, session_id: str) -> None:
    """Take a session out of the bank's activity pin (put() stamps it)."""

    bank._session_last_active[session_id] = time.monotonic() - 7_200.0


@pytest.fixture(autouse=True)
def _served_profile(monkeypatch):
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", "1")
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL_LAYOUT", "auto")
    monkeypatch.setenv("MTPLX_SUSTAINED_DENSE_DECODE_MAX_CONTEXT", "131072")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE", "auto")
    monkeypatch.delenv("MTPLX_PAGED_KV_QUANT", raising=False)
    monkeypatch.delenv("MTPLX_VLLM_METAL_PAGED_KV_QUANT", raising=False)
    monkeypatch.delenv("MTPLX_HOST_MEMORY_ALLOWANCE_BYTES", raising=False)
    import mtplx.models.qwen4_exp as qwen4

    monkeypatch.setattr(qwen4, "_qsa_prefill_enabled", lambda: True)
    monkeypatch.setattr(srv, "_record_guard_event", lambda state, payload: None)
    monkeypatch.setattr(
        srv,
        "_admission_scratch_bytes",
        lambda state, *, rows, prompt_tokens, geometry: (
            V_SCRATCH_WIDE if rows > 2048 else V_SCRATCH_NARROW,
            "qsa_itemized",
        ),
    )
    # A Mac with room: only the engine's own line binds here.
    monkeypatch.setattr(
        sm,
        "_reader",
        lambda: sm.SystemMemory(
            available_bytes=60 * GIB,
            total_bytes=128 * GIB,
            level_percent=50,
            free_bytes=50 * GIB,
            file_backed_bytes=10 * GIB,
            wired_bytes=40 * GIB,
            compressor_bytes=GIB,
            swap_used_bytes=0,
        ),
    )


class TestTheBank:
    def test_the_chain_walk_cancels_the_encode_of_what_it_evicts(self):
        """The review's reproduction: the chain walk's second phase takes an
        idle session's terminal entry while its encode is queued."""

        lane = _Lane()
        bank = _bank(lane)
        old = _put(bank, range(0, 1_000), session_id="old", row_bytes=1_000)
        _age(bank, "old")
        assert "ssd_cold:old" in lane.pending
        non_terminal, terminal = bank.shrink_for_admission(0, protect_tokens=(7, 7, 7))
        assert (non_terminal, terminal) == (0, 1)
        assert old.token_ids not in bank._entries
        assert "ssd_cold:old" not in lane.pending
        assert bank.queued_persistence_bytes == 0
        assert bank.eviction_log[-1]["persistence_cancelled"] == 1

    def test_a_pressure_shrink_cancels_it_too(self):
        lane = _Lane()
        bank = _bank(lane)
        _put(bank, range(0, 1_000), session_id="old", row_bytes=1_000)
        _age(bank, "old")
        assert bank.shrink_to_bytes(0, reason="memory_pressure_critical") == 1
        assert lane.pending == {}

    def test_a_budget_eviction_keeps_the_encode_and_accounts_for_it(self):
        """The SSD tier still gets an entry the budget pushed out of RAM; the
        bank says what its queued encode holds until the job runs."""

        lane = _Lane()
        bank = _bank(lane, max_bytes=5_000_000)
        first = _put(bank, range(0, 1_000), session_id="old", row_bytes=4_000)
        _age(bank, "old")
        _put(bank, range(5_000, 6_000), session_id="new", row_bytes=4_000)
        assert first.token_ids not in bank._entries
        assert "ssd_cold:old" in lane.pending
        [row] = bank.queued_persistence()
        assert row["session_id"] == "old"
        assert row["nbytes"] == 4_000_000
        assert row["keys"] == ["ssd_cold:old"]
        assert bank.queued_persistence_bytes == 4_000_000

    def test_the_release_finds_holders_by_the_queue(self):
        """The bank no longer lists the pushed-out entry; the release still
        finds its queued encode and cancels it."""

        lane = _Lane()
        bank = _bank(lane, max_bytes=5_000_000)
        _put(bank, range(0, 1_000), session_id="old", row_bytes=4_000)
        _age(bank, "old")
        _put(bank, range(5_000, 6_000), session_id="new", row_bytes=4_000)
        receipt = bank.release_sessions(None, keep_session_ids={"new"})
        assert "ssd_cold:old" not in lane.pending
        assert receipt["queued_persistence_entries"] == 1
        assert receipt["queued_persistence_bytes"] == 4_000_000
        assert receipt["persistence_cancelled"] == 1
        assert bank.queued_persistence_bytes == 0
        # The kept session's own encode stays.
        assert "ssd_cold:new" in lane.pending

    def test_a_queued_settle_is_tracked_and_cancelled_with_its_entry(self, monkeypatch):
        monkeypatch.setenv("MTPLX_SESSION_SNAPSHOT_SETTLE", "1")
        lane = _Lane()
        bank = _bank(lane)
        entry = _put(bank, range(0, 1_000), session_id="old", row_bytes=1_000)
        bank._schedule_snapshot_settle(entry)
        assert set(lane.pending) == {"ssd_cold:old", "snapshot_settle:old"}
        _age(bank, "old")
        bank.shrink_to_bytes(0)
        assert lane.pending == {}


def _manager(bank: SessionBank) -> EngineSessionManager:
    return EngineSessionManager(bank=bank, idle_ttl_s=3600)


def _compaction(state, manager):
    return srv._prefill_admission_shed(
        state,
        prompt_ids=list(range(1_000_000, 1_030_000)),
        session_bank=manager.bank,
        session_id="anon-compaction",
        prefill_chunk_tokens=4096,
        restore_mode="clone",
    )


class TestTheAdmission:
    def test_evicting_an_idle_entry_frees_its_memory(self, monkeypatch):
        """A 9 GiB idle conversation with its encode queued, the engine at
        93 GiB of a 96 GiB limit, a 30,000-token compaction to admit. The
        LRU step evicts the conversation; at 5478586a its encode stayed
        queued, the 9 GiB stayed resident, nothing else could be released,
        and the compaction was refused."""

        lane = _Lane()
        manager = _manager(_bank(lane))
        conversation = _put(
            manager.bank, range(0, 10_000), session_id="old", row_bytes=int(0.9 * GIB) // 1_000
        )
        _age(manager.bank, "old")
        assert "ssd_cold:old" in lane.pending
        machine = _Machine(manager.bank, base_gib=84.0, cache_gib=0.0, host_gib=6.0, lane=lane)
        _install(monkeypatch, machine)
        receipt = _compaction(_flash_next_state(manager), manager)
        assert receipt.get("refused") is not True
        assert conversation.token_ids not in manager.bank._entries
        assert lane.pending == {}
        assert machine.queued() == 0
        assert "lru_idle_entries" in receipt["reclamation_steps"]

    def test_queued_writes_go_before_anyones_ram_state(self, monkeypatch):
        """The budget pushed a 9 GiB conversation out of RAM with its encode
        queued; another session's 4 GiB entry is resident. Cancelling the
        queued encode is enough, and the resident entry stays. At 5478586a
        the admission could not see the queue and took the resident entry."""

        lane = _Lane()
        bank = _bank(lane, max_bytes=10 * GIB)
        manager = _manager(bank)
        pushed_out = _put(
            bank, range(0, 10_000), session_id="old", row_bytes=int(0.9 * GIB) // 1_000
        )
        _age(bank, "old")
        resident = _put(
            bank, range(50_000, 54_000), session_id="other", row_bytes=GIB // 1_000
        )
        assert pushed_out.token_ids not in bank._entries
        assert "ssd_cold:old" in lane.pending
        machine = _Machine(bank, base_gib=80.0, cache_gib=0.0, host_gib=6.0, lane=lane)
        _install(monkeypatch, machine)
        # The pushed-out conversation still counts: its queued encode holds it.
        assert machine.active() == 80 * GIB + resident.nbytes + pushed_out.nbytes
        receipt = _compaction(_flash_next_state(manager), manager)
        assert resident.token_ids in bank._entries
        assert "ssd_cold:old" not in lane.pending
        assert receipt["reclamation_steps"] == ["queued_persistence"]
        assert receipt["queued_persistence_release"]["entries"] == 1
        assert receipt.get("refused") is not True


class TestRetryAdvice:
    """Item (e) of the review: the advice must not claim that only a restart
    frees memory, or that nothing can be released, while queued writes or
    the prompt's own restore sources hold it."""

    def test_queued_writes_of_requests_in_flight_can_make_room(self):
        receipt = {
            "refusal_reason": "projected_over_limit_after_reclamation",
            "projected_bytes_after": 98 * GIB,
        }
        holders = {"in_flight_bytes": GIB, "queued_persistence_bytes": 2 * GIB}
        verdict = srv._admission_retry_verdict(
            receipt, holders, limit=96 * GIB, growth=4 * GIB, weights=77 * GIB
        )
        assert verdict == (True, "after_background_work_finishes")

    def test_host_memory_is_not_said_to_need_a_restart_only(self):
        receipt = {
            "refusal_reason": "projected_over_limit_after_reclamation",
            "projected_bytes_after": 98 * GIB,
        }
        holders = {"in_flight_bytes": 0, "host_overhang_charged_bytes": 3 * GIB}
        verdict = srv._admission_retry_verdict(
            receipt, holders, limit=96 * GIB, growth=4 * GIB, weights=77 * GIB
        )
        assert verdict == (False, "after_host_memory_returns")

    def test_no_sentence_overstates(self):
        text = " ".join(srv._RETRY_SENTENCES.values())
        assert "only a restart" not in text
        assert "nothing else the engine holds can be released" not in text
        assert "queued SSD writes" in srv._RETRY_SENTENCES["after_host_memory_returns"]


class _RefusingLane(_Lane):
    """An idle lane whose cancel raises: the job stays queued, holding its
    entry's arrays."""

    def cancel(self, key: str) -> int:
        raise RuntimeError("idle lane refused the cancel")


def _pushed_out_conversation(lane: _Lane):
    """A 9 GiB conversation the budget pushed out of RAM with its encode
    queued, and another session's 4 GiB entry resident."""

    bank = _bank(lane, max_bytes=10 * GIB)
    manager = _manager(bank)
    pushed_out = _put(
        bank, range(0, 10_000), session_id="old", row_bytes=int(0.9 * GIB) // 1_000
    )
    _age(bank, "old")
    resident = _put(bank, range(50_000, 54_000), session_id="other", row_bytes=GIB // 1_000)
    assert pushed_out.token_ids not in bank._entries
    assert "ssd_cold:old" in lane.pending
    return manager, bank, pushed_out, resident


class TestACancelThatFails:
    """The review of 23a94abf (finding 6): the bank dropped a job's tracking
    before cancelling it and swallowed the cancel's error. The closure kept
    its 1 GiB while the queued bytes read zero, and the error never reached
    guard_degraded. A job that could not be cancelled is still queued and
    still holds its entry: it stays counted, the failure is raised, and the
    guard reports it."""

    def test_the_job_stays_counted_and_the_failure_is_raised(self):
        lane = _RefusingLane()
        _manager_, bank, pushed_out, _resident = _pushed_out_conversation(lane)
        assert bank.queued_persistence_bytes == pushed_out.nbytes
        with pytest.raises(RuntimeError) as raised:
            bank.cancel_queued_persistence(None)
        # Nothing was let go, and the receipt says so.
        assert raised.value.receipt["entries"] == 0
        assert raised.value.receipt["held_bytes"] == 0
        assert "ssd_cold:old" in lane.pending
        assert bank.queued_persistence_bytes == pushed_out.nbytes
        assert bank.eviction_log[-1]["persistence_cancel_failures"] == 1

    def test_an_eviction_that_cannot_cancel_its_job_still_evicts_and_raises(self):
        """A pressure trim takes the entry out of RAM, keeps going, and
        reports the job it could not cancel; the bank keeps counting it."""

        lane = _RefusingLane()
        bank = _bank(lane)
        entry = _put(bank, range(0, 4_000), session_id="old", row_bytes=GIB // 1_000)
        _age(bank, "old")
        other = _put(bank, range(50_000, 52_000), session_id="other", row_bytes=GIB // 1_000)
        _age(bank, "other")
        with pytest.raises(RuntimeError) as raised:
            bank.shrink_to_bytes(0, reason="memory_pressure_critical")
        assert raised.value.receipt == 2
        assert entry.token_ids not in bank._entries
        assert other.token_ids not in bank._entries
        assert bank.queued_persistence_bytes == entry.nbytes + other.nbytes

    def test_the_admission_reports_it_and_does_not_count_it_as_freed(self, monkeypatch):
        lane = _RefusingLane()
        manager, bank, pushed_out, _resident = _pushed_out_conversation(lane)
        machine = _Machine(bank, base_gib=80.0, cache_gib=0.0, host_gib=6.0, lane=lane)
        _install(monkeypatch, machine)
        state = _flash_next_state(manager)
        receipt = _compaction(state, manager)
        assert receipt.get("queued_persistence_error")
        assert receipt.get("guard_degraded") is True
        health = srv._memory_guard_health(state)
        assert health["guard_degraded"] is True
        assert [row["where"] for row in health["degraded"]] == [
            "prefill_admission_reclamation"
        ]
        # The queued closure still holds the conversation (and whatever a
        # later step evicted with its job still queued), and the bank's
        # account agrees with what the lane retains.
        assert machine.queued() >= pushed_out.nbytes
        assert bank.queued_persistence_bytes == machine.queued()
        assert receipt["queued_persistence_release"]["held_bytes"] == 0

    def test_a_pressure_trim_that_cannot_cancel_reports_degraded(self, monkeypatch):
        from tests.test_memguard_admission import _run_loop

        lane = _RefusingLane()
        bank = _bank(lane, max_bytes=16 * GIB)
        manager = _manager(bank)
        _put(bank, range(0, 4_000), session_id="old", row_bytes=GIB // 1_000)
        _age(bank, "old")
        monkeypatch.setattr(
            sm,
            "_reader",
            lambda: sm.SystemMemory(
                available_bytes=2 * GIB,
                total_bytes=128 * GIB,
                level_percent=10,
                free_bytes=GIB,
                file_backed_bytes=GIB,
                wired_bytes=0,
                compressor_bytes=4 * GIB,
                swap_used_bytes=0,
            ),
        )
        state = SimpleNamespace(
            sessions=manager,
            dashboard=SimpleNamespace(last_memory_pressure_level=0),
        )
        _run_loop(state, monkeypatch, seconds=0.05)
        assert bank.total_nbytes == 0
        health = srv._memory_guard_health(state)
        assert health["guard_degraded"] is True
        assert health["degraded"][0]["where"] == "pressure_trim"
        assert "idle lane refused the cancel" in health["degraded"][0]["error"]
