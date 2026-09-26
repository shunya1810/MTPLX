"""Protected-terminal eviction order (MTPLX_SESSION_BANK_PROTECTED_TERMINAL).

Port of oMLX PR #3330's ``exact_resident.py`` rule: when the shorter
input-prompt fallback and the longer matching terminal compete under one byte
ceiling, publishing the fallback must not be what evicts the terminal that
extends it.

Pure host tests -- synthetic entries, ``cache=[]`` plus ``nbytes_override``, no
MLX, no model, no Metal. Every test builds the bank AFTER setting the env,
because the gate is resolved at construction.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from mtplx.session_bank import SESSION_BANK_PROTECTED_TERMINAL_ENV, SessionBank


RUNTIME = SimpleNamespace(model_path=Path("models/example"), mtp_enabled=True)

FALLBACK = (1, 2, 3)
TERMINAL = (1, 2, 3, 4, 5)
UNRELATED = (9, 9, 9)


@pytest.fixture(autouse=True)
def _no_inherited_gate(monkeypatch):
    monkeypatch.delenv(SESSION_BANK_PROTECTED_TERMINAL_ENV, raising=False)
    # Three entries per session, the default off the M1 GPU family (which
    # keeps two, see test_m1_memory_defaults.py).
    monkeypatch.setenv("MTPLX_SESSION_BANK_PER_SESSION_MAX_ENTRIES", "3")


def _bank(**kwargs) -> SessionBank:
    defaults = dict(max_entries=16, max_bytes=300, per_session_max_bytes=300)
    defaults.update(kwargs)
    return SessionBank(**defaults)


def _put(bank: SessionBank, tokens, *, session_id: str, nbytes: int):
    entry = bank.put(
        runtime=RUNTIME,
        token_ids=list(tokens),
        cache=[],
        logits=None,
        hidden=None,
        session_id=session_id,
        nbytes_override=nbytes,
    )
    assert entry is not None, f"put refused {tokens}"
    return entry


# --------------------------------------------------------------------------
# Gate resolution
# --------------------------------------------------------------------------


def test_gate_defaults_off():
    assert _bank().protect_newest_extending is False


@pytest.mark.parametrize("value", ["1", "true", "YES", "On"])
def test_gate_accepts_truthy(monkeypatch, value):
    monkeypatch.setenv(SESSION_BANK_PROTECTED_TERMINAL_ENV, value)
    assert _bank().protect_newest_extending is True


@pytest.mark.parametrize("value", ["0", "", "off", "nope"])
def test_gate_rejects_everything_else(monkeypatch, value):
    monkeypatch.setenv(SESSION_BANK_PROTECTED_TERMINAL_ENV, value)
    assert _bank().protect_newest_extending is False


def test_gate_is_construction_time(monkeypatch):
    monkeypatch.setenv(SESSION_BANK_PROTECTED_TERMINAL_ENV, "1")
    bank = _bank()
    monkeypatch.delenv(SESSION_BANK_PROTECTED_TERMINAL_ENV)
    assert bank.protect_newest_extending is True
    assert _bank().protect_newest_extending is False


# --------------------------------------------------------------------------
# _newest_extending_entry
# --------------------------------------------------------------------------


def test_newest_extending_entry_is_inert_when_gate_off():
    bank = _bank()
    _put(bank, TERMINAL, session_id="s1", nbytes=10)
    assert bank._newest_extending_entry(FALLBACK) is None


def test_newest_extending_entry_prefers_recency_over_length(monkeypatch):
    monkeypatch.setenv(SESSION_BANK_PROTECTED_TERMINAL_ENV, "1")
    bank = _bank(max_bytes=10_000, per_session_max_bytes=10_000)
    longer_older = _put(bank, (1, 2, 3, 7, 7, 7, 7), session_id="s1", nbytes=10)
    shorter_newer = _put(bank, (1, 2, 3, 8, 8), session_id="s1", nbytes=10)
    assert bank._newest_extending_entry(FALLBACK) is shorter_newer
    assert bank._newest_extending_entry(FALLBACK) is not longer_older


def test_newest_extending_entry_ignores_non_extensions(monkeypatch):
    monkeypatch.setenv(SESSION_BANK_PROTECTED_TERMINAL_ENV, "1")
    bank = _bank(max_bytes=10_000, per_session_max_bytes=10_000)
    _put(bank, UNRELATED, session_id="s2", nbytes=10)
    _put(bank, FALLBACK, session_id="s1", nbytes=10)
    # An entry EQUAL to the prompt is not a terminal extending it.
    assert bank._newest_extending_entry(FALLBACK) is None


# --------------------------------------------------------------------------
# The rule itself
# --------------------------------------------------------------------------


def _setup_contended_bank(bank: SessionBank):
    """Terminal is the LRU victim; one unrelated entry is available instead."""

    terminal = _put(bank, TERMINAL, session_id="s1", nbytes=100)
    other = _put(bank, UNRELATED, session_id="s2", nbytes=100)
    terminal.last_access_s = 0.0
    other.last_access_s = 50.0
    return terminal, other


def test_control_fallback_evicts_the_terminal():
    """Baseline: today's LRU can and does evict the matching terminal."""

    bank = _bank()
    _setup_contended_bank(bank)
    _put(bank, FALLBACK, session_id="s1", nbytes=150)

    assert bank._entries.keys() == {FALLBACK, UNRELATED}
    assert bank.protected_rejections == 0


def test_protected_terminal_survives_the_fallback_publish(monkeypatch):
    monkeypatch.setenv(SESSION_BANK_PROTECTED_TERMINAL_ENV, "1")
    bank = _bank()
    _setup_contended_bank(bank)
    _put(bank, FALLBACK, session_id="s1", nbytes=150)

    assert bank._entries.keys() == {FALLBACK, TERMINAL}
    assert bank.protected_rejections == 1


def test_protection_keeps_recency_not_length(monkeypatch):
    """The newest terminal stays; an older, longer branch is the victim."""

    monkeypatch.setenv(SESSION_BANK_PROTECTED_TERMINAL_ENV, "1")
    bank = _bank(max_bytes=260)
    longer_older = _put(bank, (1, 2, 3, 7, 7, 7, 7), session_id="s1", nbytes=50)
    shorter_newer = _put(bank, (1, 2, 3, 8, 8), session_id="s1", nbytes=50)
    longer_older.last_access_s = 10.0
    shorter_newer.last_access_s = 0.0

    _put(bank, FALLBACK, session_id="s1", nbytes=200)

    assert bank._entries.keys() == {FALLBACK, (1, 2, 3, 8, 8)}
    assert bank.protected_rejections == 1


def test_protection_is_order_only_and_yields_to_the_budget(monkeypatch):
    """A protected terminal that is the LAST candidate is still evicted.

    Refusing here would spin _evict_if_needed forever with the bank over its
    ceiling. The rule reorders victims; it never suspends the budget.
    """

    monkeypatch.setenv(SESSION_BANK_PROTECTED_TERMINAL_ENV, "1")
    bank = _bank(max_bytes=200, per_session_max_bytes=200)
    _put(bank, TERMINAL, session_id="s1", nbytes=100)
    _put(bank, FALLBACK, session_id="s1", nbytes=150)

    assert bank._entries.keys() == {FALLBACK}
    assert bank.protected_rejections == 0


def test_protection_does_not_shield_unrelated_entries(monkeypatch):
    """Only the extending terminal is protected; ordinary LRU is untouched."""

    monkeypatch.setenv(SESSION_BANK_PROTECTED_TERMINAL_ENV, "1")
    bank = _bank()
    stale = _put(bank, UNRELATED, session_id="s2", nbytes=100)
    fresh = _put(bank, (8, 8, 8, 8), session_id="s3", nbytes=100)
    stale.last_access_s = 0.0
    fresh.last_access_s = 50.0

    _put(bank, FALLBACK, session_id="s1", nbytes=150)

    assert bank._entries.keys() == {FALLBACK, (8, 8, 8, 8)}
    assert bank.protected_rejections == 0


def test_eviction_terminates_when_every_candidate_is_the_terminal(monkeypatch):
    """Repeated pressure must not loop: each pass evicts exactly one entry."""

    monkeypatch.setenv(SESSION_BANK_PROTECTED_TERMINAL_ENV, "1")
    bank = _bank(max_bytes=120, per_session_max_bytes=120)
    _put(bank, TERMINAL, session_id="s1", nbytes=60)
    _put(bank, (1, 2, 3, 6), session_id="s1", nbytes=60)
    _put(bank, FALLBACK, session_id="s1", nbytes=60)

    assert bank.total_nbytes <= bank.max_bytes
    assert FALLBACK in bank._entries


# --------------------------------------------------------------------------
# Receipt
# --------------------------------------------------------------------------


def test_to_dict_publishes_the_engagement_receipt(monkeypatch):
    monkeypatch.setenv(SESSION_BANK_PROTECTED_TERMINAL_ENV, "1")
    bank = _bank()
    _setup_contended_bank(bank)
    _put(bank, FALLBACK, session_id="s1", nbytes=150)

    snapshot = bank.to_dict()
    assert snapshot["protect_newest_extending"] is True
    assert snapshot["protected_rejections"] == 1


def test_to_dict_receipt_defaults_off():
    snapshot = _bank().to_dict()
    assert snapshot["protect_newest_extending"] is False
    assert snapshot["protected_rejections"] == 0
