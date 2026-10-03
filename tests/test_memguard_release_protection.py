"""No release step takes what the prompt restores from or what a busy session holds.

The review of 9c96dd9c (finding 4): the admission's LRU step pinned only the
incoming session and called ``shrink_to_bytes`` without the restore plan's
keys, so another idle session's entry that was this prompt's restore source
(a cross-session donor) could go; the chain walk got neither the in-flight
session ids nor a hold; and after any eviction the admission kept its first
reuse, copy decision and bill, pricing a warm extension whose source was
gone. The release steps now spare the plan's keys and every in-flight
session, and the restore is planned again after any step that evicted.
"""

from __future__ import annotations

import time

import pytest

import mtplx.server.openai as srv
import mtplx.system_memory as sm
from tests.test_memguard_admission import (
    FN_ROW,
    V_CACHED,
    V_FILE_BACKED,
    V_FREE,
    V_PROMPT,
    V_ROWS,
    V_SCRATCH_NARROW,
    V_SCRATCH_WIDE,
    _flash_next_state,
    _install,
    _Machine,
    _manager,
    _put,
    _v_reading,
)


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


def _age(bank, session_id: str, seconds: float = 3_600.0) -> None:
    """Take a session out of the bank's active pin (put() stamps it)."""

    bank._session_last_active[session_id] = time.monotonic() - seconds


class TestTheLruStep:
    def test_it_spares_another_sessions_entry_the_prompt_restores_from(
        self, monkeypatch
    ):
        """The validation turn's conversation, banked under another idle
        session (a donor), is this prompt's exact prefix. The LRU pass used
        to take it first (the oldest idle entry) and leave an unrelated idle
        entry standing."""

        manager = _manager()
        prompt = list(range(V_PROMPT))
        donor = _put(manager.bank, prompt[:V_CACHED], session_id="donor", row_bytes=FN_ROW)
        other = _put(
            manager.bank, range(900_000, 902_000), session_id="other", row_bytes=500_000
        )
        donor.last_access_s -= 100.0
        _age(manager.bank, "donor", 7_200.0)
        _age(manager.bank, "other")
        monkeypatch.setattr(
            sm, "_reader", lambda: _v_reading(free=V_FREE, file_backed=V_FILE_BACKED)
        )
        _install(monkeypatch, _Machine(manager.bank, base_gib=84.0, cache_gib=1.0, host_gib=6.0))
        receipt = srv._prefill_admission_shed(
            _flash_next_state(manager),
            prompt_ids=prompt,
            session_bank=manager.bank,
            session_id="julian",
            prefill_chunk_tokens=4096,
            restore_mode="clone",
        )
        assert receipt["reclamation_steps"][0] == "allocator_pool"
        assert donor.token_ids in manager.bank._entries
        assert other.token_ids not in manager.bank._entries
        assert receipt["growth"]["reused_tokens"] == V_CACHED
        assert receipt.get("refused") is not True


class TestTheChainWalk:
    def test_it_spares_a_session_that_is_generating(self, monkeypatch):
        """A session mid-generation keeps its sibling snapshot: the walk's
        first phase took any non-terminal entry of any session."""

        manager = _manager()
        conversation = list(range(9_000))
        # A forked generation: the shared first 4,000 tokens, then its own
        # tail, so no put collapses it into the longer entry.
        sibling = _put(
            manager.bank,
            conversation[:4_000] + list(range(700_000, 701_000)),
            session_id="busy",
            row_bytes=400_000,
        )
        _put(manager.bank, conversation, session_id="busy", row_bytes=400_000)
        busy = manager.get_or_create("busy")
        assert busy.try_begin_generation()
        monkeypatch.setattr(
            sm,
            "_reader",
            lambda: _v_reading(free=40_000_000_000, file_backed=V_FILE_BACKED),
        )
        _install(monkeypatch, _Machine(manager.bank, base_gib=84.0, cache_gib=0.0, host_gib=6.0))
        try:
            receipt = srv._prefill_admission_shed(
                _flash_next_state(manager),
                prompt_ids=list(range(500_000, 530_000)),
                session_bank=manager.bank,
                session_id="fresh",
                prefill_chunk_tokens=4096,
                restore_mode="clone",
            )
        finally:
            busy.end_generation()
        assert "chain_walk" in receipt["reclamation_steps"]
        assert sibling.token_ids in manager.bank._entries
        assert receipt.get("chain_entries_evicted", 0) == 0


class TestTheBillFollowsTheSource:
    def test_a_source_lost_to_a_release_step_is_priced_again(self, monkeypatch):
        """A lease of the conversation's own cache: its 12,091 reused rows
        cost nothing. A release step that takes the source anyway (here the
        LRU pass, forced) leaves a cold prefill of all 18,113 rows, and the
        admission prices that instead of the lease it first found."""

        manager = _manager()
        prompt = list(range(V_PROMPT))
        source = _put(manager.bank, prompt[:V_CACHED], session_id="julian", row_bytes=FN_ROW)
        source.cache_ref = []
        source.lazy_kv = False

        def take_the_source(target, **kwargs):
            manager.bank._evict_entry(source, reason="forced_in_test")
            return 1

        monkeypatch.setattr(manager.bank, "shrink_to_bytes", take_the_source)
        monkeypatch.setattr(
            sm, "_reader", lambda: _v_reading(free=3_850_000_000, file_backed=V_FILE_BACKED)
        )
        _install(monkeypatch, _Machine(manager.bank, base_gib=84.0, cache_gib=1.0, host_gib=6.0))
        session = manager.get_or_create("julian")
        assert session.try_begin_generation()
        try:
            receipt = srv._prefill_admission_shed(
                _flash_next_state(manager),
                prompt_ids=prompt,
                session_bank=manager.bank,
                session_id="julian",
                prefill_chunk_tokens=4096,
                restore_mode="reference",
            )
        finally:
            session.end_generation()
        assert receipt["reusable_prefix_tokens"] == V_CACHED
        assert receipt["restore_copies_prefix"] is False
        assert "lru_idle_entries" in receipt["reclamation_steps"]
        assert receipt["growth"]["reused_tokens"] == 0
        assert receipt["growth"]["growth_bytes"] == V_ROWS + V_SCRATCH_NARROW
        replanned = receipt["restore_replanned"]
        assert replanned[0]["after"] == "lru_idle_entries"
        assert replanned[0]["reusable_prefix_tokens"] == 0
