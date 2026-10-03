"""The paged KV (27B) reserves each decode round between rounds, so a refused growth ends the answer.

Before: the growth ran inside the forward that outran the pages, and a refusal
(PagedKVGrowthRefused) failed the request with the streamed answer lost (omp,
2026-10-03: 16.5K tokens into a write). generate_mtpk now reserves the round's
widest window first; a refusal becomes a memory_stop like the fixed-M4 bank's.
"""

from __future__ import annotations

import pytest

import mtplx.cache_state as cache_state
from mtplx.cache_state import PagedKVGrowthRefused
from mtplx.generation import _paged_round_reservation
from tests.test_promoted_paged_capacity_526 import _promoted, _rows


@pytest.fixture(autouse=True)
def _growth_env(monkeypatch):
    monkeypatch.setenv("MTPLX_DYNAMIC_PAGED_KV", "1")


@pytest.mark.parametrize("mode", ["q8", "plain"])
def test_a_round_that_fits_grows_before_it_runs(mode):
    adapter = _promoted(mode)  # 31 of 32 rows
    assert _paged_round_reservation([adapter], depth=3) is None
    assert adapter.capacity >= 31 + 4


@pytest.mark.parametrize("mode", ["q8", "plain"])
def test_a_refused_round_returns_a_receipt_and_moves_nothing(mode, monkeypatch):
    adapter = _promoted(mode)
    before = _rows(adapter, 0, 32)

    def refuse(transient_bytes, *, detail):
        raise PagedKVGrowthRefused(f"insufficient memory to grow the paged KV cache: {detail}")

    monkeypatch.setattr(cache_state, "_admit_paged_growth", refuse)
    receipt = _paged_round_reservation([adapter], depth=3)
    assert receipt is not None
    assert receipt["reason"] == "paged_kv_growth_refused"
    assert receipt["window_tokens"] == 4
    assert adapter.capacity == 32 and adapter.size() == 31
    for got, want in zip(_rows(adapter, 0, 32), before):
        assert (got == want).all()


def test_nothing_paged_is_a_no_op():
    assert _paged_round_reservation([None, object()], depth=3) is None
