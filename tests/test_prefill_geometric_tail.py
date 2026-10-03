"""Geometric tail grid for cold prefill (2026-09-18).

Through 2.11.3 the final prefill chunk was cut into ``tail_interval`` pieces
end to end, so every cold prompt closed with up to eight 256-row forwards (and
a prompt shorter than the chunk ran ENTIRELY at 256 rows).  Measured on
Flash-Next, a 256-row forward runs at about 750 tok/s against 1,720 at 2,048
rows.  The boundary list keeps only a few anchors, so the dense grid captured
records that retention discarded.  The geometric layout captures one boundary
per power-of-two distance from the prompt end, and no rung behind the nearest
boundary is narrower than MTPLX_GDN_BOUNDARY_TAIL_MIN_RUNG (1,024
rows by default: a 512-row forward runs at 1,199 tok/s and a 256-row one at
842, against 1,763 at 2,048 rows).
"""

from __future__ import annotations

import pytest

from mtplx.checkpoint_anchors import AnchorPlan, retain_checkpoints
from mtplx.generation import (
    _geometric_tail_edges,
    _prefill_spans_with_tail_grid,
)

INTERVAL = 256
CASES = [
    (tokens, chunk)
    for chunk in (1000, 2048, 4096, 8192)
    for tokens in (1, 255, 256, 257, 300, 1023, 2047, 2049, 4060, 16349, 65501, 131038)
]


def _plan(tokens, chunk, **kwargs):
    return _prefill_spans_with_tail_grid(
        tokens, tail_interval=INTERVAL, chunk_size=chunk, **kwargs
    )


@pytest.fixture(autouse=True)
def _default_layout(monkeypatch):
    monkeypatch.delenv("MTPLX_GDN_BOUNDARY_TAIL_LAYOUT", raising=False)
    monkeypatch.delenv("MTPLX_GDN_BOUNDARY_TAIL_MIN_RUNG", raising=False)
    monkeypatch.delenv("MTPLX_GDN_BOUNDARY_TAIL_BACKOFF", raising=False)


@pytest.mark.parametrize("tokens,chunk", CASES)
def test_spans_cover_the_prompt_once_in_order(tokens, chunk):
    spans = _plan(tokens, chunk)
    assert spans[0][0] == 0 and spans[-1][1] == tokens
    for (a, b), (c, _d) in zip(spans, spans[1:]):
        assert a < b == c


@pytest.mark.parametrize("tokens,chunk", CASES)
def test_only_the_last_chunk_is_refined(tokens, chunk):
    spans = _plan(tokens, chunk)
    last_start = ((tokens - 1) // chunk) * chunk
    whole = [span for span in spans if span[1] <= last_start]
    assert whole == [(s, s + chunk) for s in range(0, last_start, chunk)]


@pytest.mark.parametrize("tokens,chunk", CASES)
def test_at_most_two_forwards_run_at_or_under_the_interval(tokens, chunk):
    """The dense layout ran chunk/interval of them; this is the whole saving."""

    narrow = [1 for start, end in _plan(tokens, chunk) if end - start <= INTERVAL]
    assert len(narrow) <= 2


@pytest.mark.parametrize("tokens,chunk", CASES)
def test_the_wide_part_of_the_last_chunk_stays_wide(tokens, chunk):
    spans = _plan(tokens, chunk)
    last_start = ((tokens - 1) // chunk) * chunk
    tail = [span for span in spans if span[0] >= last_start]
    length = tokens - last_start
    if length >= 4 * INTERVAL:
        assert tail[0][1] - tail[0][0] >= length // 4
    # Widths never grow toward the prompt end: big rows first, fine grid last.
    widths = [end - start for start, end in tail]
    assert widths[:-1] == sorted(widths[:-1], reverse=True)


@pytest.mark.parametrize("tokens,chunk", CASES)
def test_nearest_boundary_is_never_further_from_the_prompt_end_than_the_dense_grid(
    tokens, chunk, monkeypatch
):
    geometric = _plan(tokens, chunk)
    monkeypatch.setenv("MTPLX_GDN_BOUNDARY_TAIL_BACKOFF", "0")
    on_grid = _plan(tokens, chunk)
    monkeypatch.setenv("MTPLX_GDN_BOUNDARY_TAIL_LAYOUT", "dense")
    dense = _plan(tokens, chunk)
    # The last span's start is the boundary an agent turn restores from (the
    # next request diverges a few tokens before this prompt's end).
    assert on_grid[-1][0] == dense[-1][0]
    assert geometric[-1][0] >= dense[-1][0]
    if tokens - dense[-1][0] > 64 and len(dense) > 1 and dense[-1][0] > 0:
        assert tokens - geometric[-1][0] == 64


def test_edges_fall_one_into_each_retention_bucket():
    start, end = 8192, 16349
    edges = _geometric_tail_edges(start, end, INTERVAL)
    distances = sorted(end - edge for edge in edges)
    assert distances[0] <= INTERVAL
    buckets = [(d - 1).bit_length() for d in distances]
    assert len(set(buckets)) == len(buckets), (distances, buckets)
    assert distances[0] == 64
    on_grid = _geometric_tail_edges(start, end, INTERVAL, backoff=0)
    assert all((edge - start) % INTERVAL == 0 for edge in on_grid)
    assert all(edge - start >= INTERVAL for edge in edges)


@pytest.mark.parametrize("chunk", (2048, 8192))
@pytest.mark.parametrize("min_rung", (256, 1024))
def test_retention_keeps_every_captured_tail_boundary_of_a_short_prompt(
    chunk, min_rung, monkeypatch
):
    """Capture positions = span ends.  On a prompt short enough that the
    shipped cap of 8 starts its distance scale at one interval, every tail
    boundary survives thinning, and a request that diverges ``d`` tokens
    before the prompt end re-prefills at most ``2 * d`` plus two rungs (with
    the full 256-row ladder: ``3 * d`` plus one interval, the bound the
    retention policy documents)."""

    monkeypatch.setenv("MTPLX_GDN_BOUNDARY_TAIL_MIN_RUNG", str(min_rung))
    tokens = 4060
    spans = _plan(tokens, chunk)
    records = [(end, object(), None) for _start, end in spans]
    kept_positions = sorted(
        int(record[0]) for record in retain_checkpoints(records, AnchorPlan(record_count=8))
    )
    assert kept_positions == sorted(end for _start, end in spans)
    for distance in (1, 5, 64, 200, 256, 300, 600, 1000, 1500, 2000):
        matched = tokens - distance
        # Position 0 is the cold start: no boundary below means a full prefill
        # of the matched prefix.
        below = [0] + [pos for pos in kept_positions if pos <= matched]
        bound = (
            3 * distance + INTERVAL
            if min_rung == INTERVAL
            else 2 * distance + 2 * min_rung
        )
        assert matched - below[-1] <= bound, (distance, kept_positions)


@pytest.mark.parametrize("chunk", (2048, 8192))
def test_retention_keeps_the_nearest_tail_boundary_of_a_long_prompt(chunk):
    """At 64K the cap's distance scale starts at 2,048, so thinning keeps one
    near-tail record: the closest.  That is the record an agent turn restores
    from, and the geometric layout places it where the dense grid did."""

    tokens = 65501
    spans = _plan(tokens, chunk)
    records = [(end, object(), None) for _start, end in spans]
    kept_positions = sorted(
        int(record[0]) for record in retain_checkpoints(records, AnchorPlan(record_count=8))
    )
    assert kept_positions[-1] == tokens
    assert spans[-1][0] in kept_positions
    assert tokens - spans[-1][0] <= INTERVAL


def test_dense_layout_is_the_shipped_2_11_3_plan(monkeypatch):
    monkeypatch.setenv("MTPLX_GDN_BOUNDARY_TAIL_LAYOUT", "dense")
    spans = _plan(4060, 2048)
    assert [end - start for start, end in spans] == [2048] + [256] * 7 + [220]


def test_geometric_layout_for_the_measured_cells():
    assert [e - s for s, e in _plan(4060, 2048)] == [2048, 1948, 64]
    assert [e - s for s, e in _plan(4060, 8192)] == [2972, 1024, 64]
    assert [e - s for s, e in _plan(16349, 8192)] == [8192, 5021, 2048, 1024, 64]


def test_backoff_zero_keeps_the_nearest_boundary_on_the_interval_grid(monkeypatch):
    monkeypatch.setenv("MTPLX_GDN_BOUNDARY_TAIL_BACKOFF", "0")
    assert [e - s for s, e in _plan(4060, 2048)] == [2048, 1792, 220]
    assert [e - s for s, e in _plan(16349, 8192)] == [8192, 4864, 2048, 1024, 221]


def test_full_ladder_when_the_minimum_rung_is_one_interval(monkeypatch):
    monkeypatch.setenv("MTPLX_GDN_BOUNDARY_TAIL_MIN_RUNG", "256")
    assert [e - s for s, e in _plan(4060, 2048)] == [2048, 1180, 512, 256, 64]
    assert [e - s for s, e in _plan(16349, 8192)] == [
        8192,
        4253,
        2048,
        1024,
        512,
        256,
        64,
    ]


@pytest.mark.parametrize("tokens,chunk", CASES)
def test_no_rung_behind_the_nearest_boundary_is_narrower_than_the_minimum(
    tokens, chunk
):
    spans = _plan(tokens, chunk)
    last_start = ((tokens - 1) // chunk) * chunk
    tail = [span for span in spans if span[0] >= last_start]
    # The final span is the remainder past the nearest boundary; a chunk too
    # short for one rung keeps its single interval-grid edge.
    for start, end in tail[1:-1]:
        assert end - start >= 1024, (tokens, chunk, tail)


def test_mandatory_edge_still_lands_on_a_span_end():
    spans = _plan(16349, 8192, mandatory_edges=(15000,))
    assert 15000 in [end for _start, end in spans]
    assert spans[0][0] == 0 and spans[-1][1] == 16349
