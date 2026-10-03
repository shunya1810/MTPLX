"""A hybrid SSD entry with no recurrent boundary at or below a partial match
is refused before any tensor is read, and the next ranked row is tried.

The restore serves a hybrid entry's prefix only at a stored recurrent
boundary (SessionBank.restore_entry_prefix_cache, tiny gaps included since
2026-09-08). On 2026-10-01 a Pi turn shared its tool definitions with a
131,735-token conversation on the SSD tier that had no boundaries: every
attempt decoded that entry's 4.17 GB for a candidate the restore refused,
and the decoded copy stayed resident through the cold prefill that followed.
"""

from __future__ import annotations

import time

import mlx.core as mx

import mtplx.cache_bank.cold_tier as cold_tier
from mtplx.cache_bank.cold_tier import SessionBankColdTier
from mtplx.cache_state import CacheSnapshot


def _entry(tokens, *, recurrent: bool):
    class Entry:
        token_ids = tuple(tokens)
        nbytes = 2048
        cache_snapshot = CacheSnapshot(
            states=[mx.zeros((1, 2, 8, 4), dtype=mx.float16)],
            meta_states=[{"offset": len(tokens)}],
        )
        logits = mx.zeros((1, 8), dtype=mx.float16)
        hidden = mx.zeros((1, 8), dtype=mx.float16)
        mtp_history_snapshot = None
        gdn_boundaries = ()
        has_recurrent = recurrent
        session_id = "s1"
        token_hash = f"hash-{len(tokens):04d}" * 2
        prefix_len = len(tokens)
        model_path = "model"
        mtp_enabled = False
        hidden_variant = None
        template_hash = None
        mtp_history_policy = None
        policy_fingerprint = None

    return Entry()


def _tier(tmp_path):
    return SessionBankColdTier(base_dir=tmp_path / "bank", mode="on", min_prefix_tokens=1)


def _store(tier, *entries):
    for entry in entries:
        assert tier.put_entry(entry, capabilities=["ar_insert"]) is True
    deadline = time.time() + 5.0
    while time.time() < deadline:
        if tier.stats()["writes_completed"] >= len(entries):
            return
        time.sleep(0.05)
    raise AssertionError("writer did not complete")


def _lookup(tier, tokens):
    return tier.lookup_prefix_boundary(
        tokens,
        model_path="model",
        mtp_enabled=False,
        max_token_gap=8,
        min_matched_tokens=8,
        block_size=16,
        block_min_matched_tokens=16,
    )


def _no_decode(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("a refused row must not be decoded")

    monkeypatch.setattr(cold_tier, "decode_payload", refuse)
    monkeypatch.setattr(cold_tier, "decode_payload_prefix", refuse)


def test_a_hybrid_entry_without_a_boundary_is_refused_before_decoding(tmp_path, monkeypatch):
    tier = _tier(tmp_path)
    stored = list(range(2000))
    _store(tier, _entry(stored, recurrent=True))
    _no_decode(monkeypatch)

    query = tuple(stored[:1500] + [9999] * 64)
    assert _lookup(tier, query) is None
    stats = tier.stats()
    assert stats["last_miss_reason"] == "ssd_prefix_no_recurrent_boundary"
    assert stats["prefix_restores_without_boundary"] == 1
    assert stats.get("restore_hits", 0) == 0


def test_the_next_ranked_row_serves_when_the_longest_cannot(tmp_path):
    tier = _tier(tmp_path)
    hybrid = list(range(2000))
    attention = list(range(1200)) + [5555] * 100
    _store(tier, _entry(hybrid, recurrent=True), _entry(attention, recurrent=False))

    query = tuple(hybrid[:1500] + [9999] * 64)
    hit = _lookup(tier, query)
    assert hit is not None
    assert hit.record.token_ids == tuple(attention)
    assert hit.matched_tokens == 1200
    assert tier.stats()["prefix_restores_without_boundary"] == 1


def test_with_boundary_true_restores_off_the_full_entry_still_decodes(tmp_path, monkeypatch):
    monkeypatch.setenv("MTPLX_SESSION_BOUNDARY_TRUE_RESTORE", "0")
    tier = _tier(tmp_path)
    stored = list(range(2000))
    _store(tier, _entry(stored, recurrent=True))

    query = tuple(stored[:1500] + [9999] * 64)
    hit = _lookup(tier, query)
    assert hit is not None
    assert hit.record.token_ids == tuple(stored)
