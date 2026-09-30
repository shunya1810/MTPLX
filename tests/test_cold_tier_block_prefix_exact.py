"""A block_prefix cold restore hands _restore_row the exact common prefix (2026-09-29).

The block-aligned length only ranks candidates. The request's tokens match the
stored row exactly up to the common prefix, so the restore may use any persisted
recurrent boundary inside it; restoring at the block edge instead fell back to an
earlier GDN boundary (omp: system prompt boundary 19,337, block edge 19,200,
restored 18,432 and re-prefilled ~1K tokens).
"""

from __future__ import annotations

from mtplx.cache_bank.cold_tier import SessionBankColdTier
from test_cold_tier_min_useful_matched import _lookup, _store, _tier


def _spy_restore(tier, monkeypatch):
    seen: list[int] = []
    original = SessionBankColdTier._restore_row

    def spy(self, row, tokens, **kwargs):
        seen.append(int(kwargs["prefix_restore_tokens"]))
        return original(self, row, tokens, **kwargs)

    monkeypatch.setattr(SessionBankColdTier, "_restore_row", spy)
    return seen


def test_block_prefix_restores_at_the_common_prefix(tmp_path, monkeypatch):
    tier = _tier(tmp_path)
    stored = list(range(600))
    _store(tier, stored)
    seen = _spy_restore(tier, monkeypatch)
    # 590 shared tokens, then a gap of 10 new ones: past max_token_gap (8), so
    # the row matches as block_prefix (block 16 -> aligned 576).
    query = tuple(stored[:590] + list(range(10_000, 10_020)))
    hit = _lookup(tier, query)
    assert hit is not None
    assert hit.restore_kind == "block_prefix"
    assert seen == [590]
    assert hit.matched_tokens == 590


def test_block_prefix_never_restores_past_the_stored_row(tmp_path, monkeypatch):
    tier = _tier(tmp_path)
    stored = list(range(600))
    _store(tier, stored)
    seen = _spy_restore(tier, monkeypatch)
    # The request extends the whole stored row by 40 tokens: gap 0, near_prefix,
    # so the restore length is the plain match (the row's full length).
    query = tuple(stored + list(range(10_000, 10_040)))
    hit = _lookup(tier, query)
    assert hit is not None
    assert hit.restore_kind == "near_prefix"
    assert seen == [600]
