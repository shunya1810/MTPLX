"""The postcommit keeps one copy of the conversation (mtplx/one_copy.py).

After an answer, the idle postcommit re-reads the conversation in the
client's rendering and banks it. With the copying store it restored a clone,
which copied the whole conversation beside the lease while it ran (at 138K
tokens on Flash-Next that second copy alone is 4.2 GiB). In the one-copy
store it takes its own conversation's lease and extends it in place. Pi's
next request usually preempts it, so two things keep the conversation safe:
a lease it took and could not finish goes back to the bank
(tests/test_lease_survives_aborted_prefill.py), and a lease it did finish is
banked before the yield is honoured.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from mtplx.server import openai


class _Tokenizer:
    def apply_chat_template(self, messages, **_kwargs):
        return [10, 11, 12, 13, 14]


def _state(bank, *, one_copy: bool):
    runtime = SimpleNamespace(
        tokenizer=_Tokenizer(),
        mtp_enabled=True,
        qwen4_fixed_m4_compiled_verify=one_copy,
    )
    return SimpleNamespace(
        runtime=runtime,
        sessions=SimpleNamespace(bank=bank),
        template_hash="tmpl",
        draft_head_identity="draft",
        lock=threading.Lock(),
        begin_foreground=lambda: None,
        end_foreground=lambda: None,
        args=SimpleNamespace(strip_assistant_reasoning_history=False),
    )


class _Bank:
    def __init__(self):
        self.puts: list[dict] = []

    def longest_prefix(self, _ids):
        return None

    def put(self, **kwargs):
        self.puts.append(kwargs)
        return SimpleNamespace(prefix_len=5, nbytes=0, token_hash="hash")


def _postcommit(monkeypatch, *, one_copy, restore_mode, yield_after_prefill=False):
    monkeypatch.setenv("MTPLX_ONE_COPY", "1")
    committed_mtp = object()
    captured: dict = {}
    prefill_done = {"flag": False}

    def restore(*_args, **kwargs):
        captured.update(kwargs)
        prefill_done["flag"] = True
        return SimpleNamespace(
            trunk_cache=["cache"],
            logits="logits",
            hidden="hidden",
            committed_mtp_cache=committed_mtp,
            restore_mode=restore_mode,
            cache_hit=True,
            cached_tokens=4,
            suffix_tokens=1,
            cache_miss_reason=None,
            gdn_boundaries=[],
        )

    monkeypatch.setattr(openai, "restore_or_prefill_prompt_state", restore)
    monkeypatch.setattr(openai, "snapshot_cache", lambda cache: ("snapshot", cache))
    monkeypatch.setattr(
        openai, "_history_ids_for_postcommit", lambda *a, **k: ([10, 11, 12, 13, 14], None)
    )
    bank = _Bank()
    result = openai._store_retokenized_history_snapshot(
        _state(bank, one_copy=one_copy),
        session_id="pi",
        messages=[],
        assistant_content="ok",
        thinking_enabled=False,
        policy_fingerprint="fp",
        abort_check=(lambda: prefill_done["flag"]) if yield_after_prefill else None,
    )
    return result, captured, bank, committed_mtp


def test_it_extends_its_own_lease_in_place(monkeypatch):
    _result, captured, bank, committed_mtp = _postcommit(
        monkeypatch, one_copy=True, restore_mode="reference_lease"
    )
    assert captured["restore_mode"] == "reference"
    assert captured["session_id"] == "pi"
    [put] = bank.puts
    # The draft head's committed history rides as a reference, not a copy.
    assert put["mtp_history_cache_ref"] is committed_mtp
    assert put["mtp_history_snapshot"] is None


def test_the_copying_store_still_restores_a_clone(monkeypatch):
    _result, captured, bank, committed_mtp = _postcommit(
        monkeypatch, one_copy=False, restore_mode="clone"
    )
    assert captured["restore_mode"] == "clone"
    assert captured["session_id"] is None
    [put] = bank.puts
    assert put["mtp_history_snapshot"] == ("snapshot", committed_mtp)
    assert put["mtp_history_cache_ref"] is None


@pytest.mark.parametrize(
    "restore_mode", ["reference_lease", "near_prefix_boundary_reference_lease"]
)
def test_a_finished_lease_is_banked_before_the_yield(monkeypatch, restore_mode):
    """The next request preempts the postcommit after its prefill: the
    conversation, which now lives only in this prompt state, is banked
    before the postcommit yields."""

    result, _captured, bank, _mtp = _postcommit(
        monkeypatch, one_copy=True, restore_mode=restore_mode, yield_after_prefill=True
    )
    assert len(bank.puts) == 1
    assert result["stored"] is True


def test_a_clone_is_dropped_on_the_yield_as_before(monkeypatch):
    """A clone is a copy: the entry it came from still holds the
    conversation, so the yield drops it without banking."""

    result, _captured, bank, _mtp = _postcommit(
        monkeypatch, one_copy=False, restore_mode="clone", yield_after_prefill=True
    )
    assert bank.puts == []
    assert result["mode"] == "aborted"
