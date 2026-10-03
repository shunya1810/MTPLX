"""Which recurrent checkpoints a conversation keeps (mtplx.checkpoint_anchors).

The anchors replaced the repeated geometric thinning on 2026-09-30; the
16,371 case and the churn it fixed are in tests/test_gdn_boundary_retention.py.
These tests pin the policy's own promises:

* the anchors come first (prompt end, restore point, stable edge, newest),
  then the absolute grid, coarse cells before fine ones, then the newest;
* the byte budget holds after every capture, every inheritance and every bank
  put, and a set already within it passes through the bank whole;
* within one prefill, trimming after every capture keeps exactly what
  trimming once at the end would;
* over a whole session, coverage is never coarser than the grid the budget
  can hold for the longest prompt, plus one chunk;
* a restore never starts past the matched prefix;
* a ceiling (an image prompt's first image) keeps what is at or before it and
  takes no snapshot past it;
* the receipt reports what is kept.
"""

from __future__ import annotations

import random
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import pytest

import mtplx.generation as generation
from mtplx.cache_state import CacheSnapshot
from mtplx.checkpoint_anchors import (
    GRID_TOKENS,
    AnchorPlan,
    CheckpointSink,
    checkpoint_coverage,
    record_nbytes,
    retain_checkpoints,
)
from mtplx.generation import (
    _append_gdn_boundary_record,
    _capture_gdn_boundary,
    _checkpoint_sink,
    _inherited_gdn_boundaries,
    _prefill_spans_with_tail_grid,
)
from mtplx.session_bank import SessionBank, _recurrent_restore_point


def _plain(position: int):
    return (position, None, None)


def _positions(records) -> list[int]:
    return [int(record[0]) for record in records]


def _at_or_below(positions, matched: int) -> int:
    usable = [p for p in positions if p <= matched]
    return max(usable) if usable else 0


ROWS = 32  # one checkpoint: 32 x 64 float32 state rows + a 16-float hidden row


def _record(position: int, *, rows: int = ROWS):
    snapshot = CacheSnapshot(
        states=(None, [mx.full((rows, 64), float(position), dtype=mx.float32)]),
        meta_states=(None, ("recurrent",)),
    )
    return (position, snapshot, mx.zeros((1, 1, 16), dtype=mx.float32))


ONE = record_nbytes(_record(1))


def test_one_checkpoint_counts_its_state_and_hidden_bytes():
    assert ONE == ROWS * 64 * 4 + 16 * 4


def test_the_anchors_come_first_then_the_grid_then_the_newest():
    positions = [1024, 3072, 9000, 12000, 17000, 20000, 24000, 30000, 33000,
                 40000, 40936, 41000]
    plan = AnchorPlan(
        record_count=6,
        prompt_end=41000,
        restore_point=20000,
        stable_prefix=33000,
    )
    kept = _positions(retain_checkpoints([_plain(p) for p in positions], plan))
    # Protected: the prompt-end anchor (newest at or below 41,000 - 64), the
    # restore point, the stable edge, the newest. Then the grid: 1,024 is the
    # first checkpoint (every cell), 17,000 the first of the 16K cell
    # [16384, 32768); 9,000 and 30,000 only open 8K cells, and the budget
    # has no room left for them.
    assert kept == [1024, 17000, 20000, 33000, 40936, 41000]
    # With two more slots the 8K cells come next, never a record that shares
    # its 8K cell with an earlier one.
    wider = AnchorPlan(
        record_count=8, prompt_end=41000, restore_point=20000, stable_prefix=33000
    )
    kept = _positions(retain_checkpoints([_plain(p) for p in positions], wider))
    assert kept == [1024, 9000, 17000, 20000, 30000, 33000, 40936, 41000]


def test_a_later_capture_of_a_position_replaces_the_earlier_one():
    first, second = _record(4096), _record(4096)
    kept = retain_checkpoints([first, _record(2048), second], AnchorPlan())
    assert _positions(kept) == [2048, 4096]
    assert kept[1] is second


def test_the_default_budget_is_eight_checkpoints_of_the_model_at_hand(monkeypatch):
    records = [_record(p) for p in range(1024, 13 * 1024, 1024)]
    monkeypatch.delenv("MTPLX_GDN_BOUNDARY_MAX", raising=False)
    assert len(retain_checkpoints(records, AnchorPlan())) == 8
    monkeypatch.setenv("MTPLX_GDN_BOUNDARY_MAX", "4")
    assert len(retain_checkpoints(records, AnchorPlan())) == 4
    monkeypatch.setenv("MTPLX_GDN_BOUNDARY_MAX", "1")
    assert len(retain_checkpoints(records, AnchorPlan())) == 2
    monkeypatch.setenv("MTPLX_GDN_BOUNDARY_MAX", "eight")
    assert len(retain_checkpoints(records, AnchorPlan())) == 8


def test_the_byte_budget_holds_after_every_capture():
    budget = 3 * ONE + ONE // 2
    sink = CheckpointSink(
        plan=AnchorPlan(budget_bytes=budget, prompt_end=40000, restore_point=None)
    )
    # The chunk ends, then the tail ladder's top rung and the prompt end.
    for position in [*range(2048, 38913, 2048), 39936, 40000]:
        _append_gdn_boundary_record(sink, position, *_record(position)[1:])
        assert sum(record_nbytes(record) for record in sink) <= budget
        assert len(sink) <= 3
    # The prompt-end anchor and the newest, then the first checkpoint.
    assert _positions(sink) == [2048, 39936, 40000]


def test_an_inherited_set_keeps_its_budget_and_its_restore_point():
    entry = SimpleNamespace(
        gdn_boundaries=[_record(p) for p in range(2048, 30721, 2048)]
    )
    kept = _inherited_gdn_boundaries(entry, 20480, budget_bytes=2 * ONE)
    assert _positions(kept) == [2048, 20480]
    kept = _inherited_gdn_boundaries(entry, 20480, budget_bytes=4 * ONE)
    assert _positions(kept) == [2048, 8192, 16384, 20480]


class _KV:
    """Trimmable attention KV whose offset is its keys' length."""

    def __init__(self, offset: int = 0) -> None:
        self.keys = mx.zeros((1, 1, offset, 1), dtype=mx.float32)
        self.offset = int(offset)

    @property
    def state(self):
        return (self.keys,)

    @state.setter
    def state(self, value) -> None:
        (self.keys,) = value
        self.offset = int(self.keys.shape[2])

    @property
    def meta_state(self):
        return (str(self.offset),)

    @meta_state.setter
    def meta_state(self, value) -> None:
        self.offset = int(value[0])

    def is_trimmable(self) -> bool:
        return True

    def trim(self, n: int) -> int:
        n = min(self.offset, int(n))
        self.offset -= n
        return n


class _Recurrent:
    """Non-trimmable recurrent state stamped with the tokens it consumed."""

    def __init__(self, consumed: int = 0) -> None:
        self.cache = [mx.full((ROWS, 64), float(consumed), dtype=mx.float32)]

    @property
    def state(self):
        return self.cache

    @state.setter
    def state(self, value) -> None:
        self.cache = list(value)

    @property
    def meta_state(self):
        return ("recurrent",)

    @meta_state.setter
    def meta_state(self, value) -> None:
        pass

    def is_trimmable(self) -> bool:
        return False


def _runtime():
    return SimpleNamespace(
        model_path=Path("anchors-tiny"),
        mtp_enabled=False,
        make_cache=lambda: [_KV(), _Recurrent()],
        make_mtp_cache=lambda: [],
    )


def _put(bank: SessionBank, prefix_len: int, boundaries):
    return bank.put(
        runtime=_runtime(),
        token_ids=list(range(prefix_len)),
        cache=[_KV(prefix_len), _Recurrent(prefix_len)],
        logits=mx.zeros((1, 4)),
        hidden=None,
        mtp_history_policy="cycle",
        gdn_boundaries=boundaries,
    )


def _boundary(position: int):
    from mtplx.cache_state import snapshot_untrimmable_cache

    return (
        position,
        snapshot_untrimmable_cache([_KV(position), _Recurrent(position)]),
        None,
    )


def test_the_bank_keeps_every_stored_set_within_its_budget():
    one = record_nbytes(_boundary(1))
    bank = SessionBank(checkpoint_budget_bytes=3 * one)
    entry = _put(bank, 30001, [_boundary(p) for p in range(2048, 30001, 2048)])
    assert entry is not None
    kept = _positions(entry.gdn_boundaries)
    assert sum(record_nbytes(record) for record in entry.gdn_boundaries) <= 3 * one
    # The prompt-end anchor (the newest at or below 30,000 - 64) is also the
    # newest; then the first checkpoint and the first of the 16K cell.
    assert kept == [2048, 16384, 28672]
    assert bank.to_dict()["checkpoint_budget_bytes"] == 3 * one
    with pytest.raises(ValueError):
        SessionBank(checkpoint_budget_bytes=-1)


def test_a_set_within_the_budget_passes_through_the_bank_whole():
    """A prefill's sink already holds its anchors within the bank's budget;
    the put must not re-rank it with less knowledge (it does not know the
    restore point or the stable edge) and drop one."""

    bank = SessionBank()
    inherited = [_boundary(p) for p in (2048, 12288, 20000)]
    sink = _checkpoint_sink(
        inherited,
        session_bank=bank,
        prompt_len=36000,
        restore_point=20000,
        stable_prefix_len=33000,
    )
    for position in sorted([*range(22048, 36000, 2048), 33000, 35935, 35999]):
        _append_gdn_boundary_record(sink, position, *_boundary(position)[1:])
    assert 20000 in _positions(sink) and 33000 in _positions(sink)
    entry = _put(bank, 36000, list(sink))
    assert _positions(entry.gdn_boundaries) == _positions(sink)


def test_no_restore_starts_past_the_matched_prefix():
    bank = SessionBank()
    entry = _put(
        bank,
        46938,
        [_boundary(p) for p in (4096, 8192, 16384, 24576, 32768, 40960, 46873, 46937)],
    )
    assert entry.recurrent_boundary_at_or_below(16371)[0] == 8192
    assert _recurrent_restore_point(entry, 16371) == 8192
    result = bank.restore_entry_prefix_cache(_runtime(), entry, 16371, mode="clone")
    assert result is not None
    cache, _history, _mode, restore_point, _hidden = result
    assert restore_point == 8192
    assert cache[0].offset == 8192
    assert float(cache[1].state[0][0, 0]) == 8192.0

    rng = random.Random(20260930)
    for _ in range(300):
        positions = sorted(rng.sample(range(1, 60000), rng.randrange(1, 12)))
        entry.gdn_boundaries = [_plain(p) for p in positions]
        matched = rng.randrange(1, 60000)
        chosen = entry.recurrent_boundary_at_or_below(matched)
        expected = _at_or_below(positions, matched)
        assert (chosen[0] if chosen else 0) == expected <= matched
        assert _recurrent_restore_point(entry, matched) == expected


def test_a_ceiling_keeps_what_is_before_it_and_snapshots_nothing_past_it(monkeypatch):
    snapshots: list[int] = []
    real = generation.snapshot_untrimmable_cache

    def counting(cache):
        snapshots.append(int(cache[0].offset))
        return real(cache)

    monkeypatch.setattr(generation, "snapshot_untrimmable_cache", counting)
    sink = CheckpointSink(
        plan=AnchorPlan(record_count=8, prompt_end=95, ceiling=40),
        cuts_forwards=False,
    )
    assert not generation._sink_cuts_forwards(sink)
    for position in range(8, 96, 8):
        _capture_gdn_boundary(sink, position, [_KV(position), _Recurrent(position)])
    assert _positions(sink) == [8, 16, 24, 32, 40]
    assert snapshots == [8, 16, 24, 32, 40]
    # A plain list takes every position and may reshape the prefill.
    assert generation._sink_admits([], 10**9)
    assert generation._sink_cuts_forwards([])


def test_image_spans_keep_checkpoints_out_of_images_and_snapshot_none_there(monkeypatch):
    """An image prompt keeps checkpoints before, between and after its
    images, never inside one: a restore there would resume with part of an
    image's rows. The pre-image anchor stays protected when the budget
    thins the rest."""

    snapshots: list[int] = []
    real = generation.snapshot_untrimmable_cache

    def counting(cache):
        snapshots.append(int(cache[0].offset))
        return real(cache)

    monkeypatch.setattr(generation, "snapshot_untrimmable_cache", counting)
    plan = AnchorPlan(record_count=8, prompt_end=95, image_spans=((43, 49), (61, 67)))
    assert plan.admits(43) and plan.admits(49) and plan.admits(61) and plan.admits(67)
    assert not any(plan.admits(p) for p in (44, 48, 62, 66))
    sink = CheckpointSink(plan=plan, cuts_forwards=False)
    for position in range(8, 96, 8):
        _capture_gdn_boundary(sink, position, [_KV(position), _Recurrent(position)])
    # 48 and 64 are inside the images; nothing was snapshotted there.
    assert snapshots == [8, 16, 24, 32, 40, 56, 72, 80, 88]
    # Nine captures for a budget of eight: the prompt-end anchor (24), the
    # pre-image anchor (40) and the newest (88) are protected; 16 goes.
    assert _positions(sink) == [8, 24, 32, 40, 56, 72, 80, 88]


def test_the_receipt_reports_what_is_kept():
    records = [_record(p) for p in (2048, 8192, 16384)]
    report = checkpoint_coverage(records, budget_bytes=5 * ONE, first_image_start=20000)
    assert report == {
        "anchors": [2048, 8192, 16384],
        "count": 3,
        "bytes": 3 * ONE,
        "budget_bytes": 5 * ONE,
        "pre_image": {
            "first_image_start": 20000,
            "anchor": 16384,
            "tokens_to_reread": 3616,
        },
    }
    # Without a budget it reports the default, eight of the largest record.
    assert checkpoint_coverage(records)["budget_bytes"] == 8 * ONE
    # An image with no checkpoint before it re-reads everything before it.
    assert checkpoint_coverage([], first_image_start=500)["pre_image"] == {
        "first_image_start": 500,
        "anchor": None,
        "tokens_to_reread": 500,
    }


def _random_prefill(rng: random.Random):
    """An inherited set, then one prefill's captures on the real plan."""

    start = rng.choice([0, 0, rng.randrange(1, 60000)])
    prompt_len = start + rng.randrange(100, 200000)
    inherited = sorted(
        {rng.randrange(1, start + 1) for _ in range(rng.randrange(0, 12))} | {start}
    ) if start else []
    chunk = rng.choice([512, 1024, 2048, 4096])
    interval = rng.choice([128, 256, 512])
    plan = AnchorPlan(
        record_count=rng.choice([3, 5, 8, 12]),
        prompt_end=prompt_len - 1,
        restore_point=start or None,
        stable_prefix=rng.choice([None, rng.randrange(1, prompt_len)]),
    )
    captures = [
        start + end
        for _span_start, end in _prefill_spans_with_tail_grid(
            prompt_len - start - 1, tail_interval=interval, chunk_size=chunk
        )
    ]
    return plan, inherited, captures


def test_trimming_after_every_capture_keeps_what_trimming_once_would():
    rng = random.Random(20260930)
    for _ in range(300):
        plan, inherited, captures = _random_prefill(rng)
        sink = CheckpointSink([_plain(p) for p in inherited], plan=plan)
        sink.retain()
        carried = list(sink)
        for position in captures:
            sink.append(_plain(position))
            sink.retain()
        once = retain_checkpoints(carried + [_plain(p) for p in captures], plan)
        assert _positions(sink) == _positions(once), (plan, inherited[:4])


def _grid_width(longest: int, slots: int) -> int:
    width = GRID_TOKENS
    while -(-longest // width) > slots:
        width *= 2
    return width


@pytest.mark.parametrize("count", [5, 8, 12])
def test_a_session_never_covers_worse_than_its_budget_affords(count):
    """Random agent sessions: turns that diverge near the end, in the middle
    of the last turn, or anywhere, each restoring at the newest checkpoint at
    or below its match. At the end, every match re-prefills at most one cell
    of the finest grid the budget can hold for the longest prompt (the
    prompt-end anchor, the restore point and the newest take three slots),
    plus the chunk that cell's first checkpoint may sit in. Dropped anchors
    stay dropped, so a session that grew long and then rewound keeps the
    coarser grid; it never loses more."""

    rng = random.Random(1000 + count)
    for _ in range(60):
        chunk = rng.choice([1024, 2048, 4096])
        interval = rng.choice([128, 256])
        prompt_len = rng.randrange(2000, 60000)
        held: list = []
        restore = 0
        longest = 0
        for _turn in range(rng.randrange(2, 30)):
            plan = AnchorPlan(
                record_count=count,
                prompt_end=prompt_len - 1,
                restore_point=restore or None,
            )
            # A plain record counts one byte, so the budget is the count.
            inherited = _inherited_gdn_boundaries(
                SimpleNamespace(gdn_boundaries=held), restore, budget_bytes=count
            ) if restore else []
            sink = CheckpointSink(inherited, plan=plan)
            sink.retain()
            for _s, end in _prefill_spans_with_tail_grid(
                prompt_len - restore - 1, tail_interval=interval, chunk_size=chunk
            ):
                sink.append(_plain(restore + end))
                sink.retain()
            held = list(sink)
            longest = max(longest, prompt_len)
            final_len = prompt_len
            kind = rng.random()
            if kind < 0.6:
                matched = prompt_len - rng.randrange(1, 40)
            elif kind < 0.85:
                matched = prompt_len - rng.randrange(40, 3000)
            else:
                matched = rng.randrange(1, prompt_len)
            restore = _at_or_below(_positions(held), matched)
            prompt_len = matched + rng.randrange(200, 8000)
        kept = _positions(held)
        assert len(kept) <= count
        bound = _grid_width(longest, count - 3) + chunk
        worst = max(
            (m - _at_or_below(kept, m), m) for m in range(1, final_len, 37)
        )
        assert worst[0] <= bound, (worst, bound, longest, kept)
