"""A partial restore of an image conversation never cuts an image.

The session bank holds an image conversation in its content-keyed view
(``vision.splice.vision_bank_key_ids``): every image row is replaced by a
stand-in derived from the image's bytes, its row and how it was positioned, so
a keyed prefix match is content-true. Partial restores of such prompts used to
be impossible past the first image (the near-prefix lane matched raw ids and
was capped there). They are served now, under one rule the bank, the restore
and the admission share: a restore point never lies inside an image, and a
restore that re-forwards its last token never ends on an image row.

These tests pin the rule where the bank applies it: the recurrent boundary a
partial restore lands on, the restore point the admission prices, and the
near-prefix candidate an entry offers. Text entries carry no image rows and
are served exactly as before.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from mtplx.cache_state import CacheSnapshot
from mtplx.session_bank import (
    SessionBankEntry,
    _near_prefix_candidate_len,
    _recurrent_restore_point,
)
from mtplx.vision.splice import VisionSplice, vision_bank_key_ids

PAD = 9_000
# 40 text tokens, a 6-row image, 30 text tokens, a 4-row image, 20 text tokens.
HEAD = [(i * 7 + 3) % 500 for i in range(40)]
MIDDLE = [(i * 11 + 5) % 500 for i in range(30)]
TAIL = [(i * 13 + 1) % 500 for i in range(20)]
RAW = [*HEAD, *([PAD] * 6), *MIDDLE, *([PAD] * 4), *TAIL]
FIRST = (40, 46)
SECOND = (76, 80)


def _splice(digests=(0xA1, 0xB2), pad_counts=(6, 4)) -> VisionSplice:
    return VisionSplice(
        image_pad_token_id=PAD,
        embeddings=mx.zeros((sum(pad_counts), 4)),
        image_digests=tuple(digests),
        pad_counts=tuple(pad_counts),
    )


KEYED = tuple(vision_bank_key_ids(RAW, _splice()))
KNOBS = dict(gap_limit=8, min_match=16, block=16, block_min_match=16)


def _entry(token_ids, *, boundaries=(), recurrent=True) -> SessionBankEntry:
    snapshot = CacheSnapshot(states=(None,), meta_states=(None,))
    return SessionBankEntry(
        token_ids=tuple(token_ids),
        token_hash="h",
        model_path="m",
        mtp_enabled=True,
        hidden_variant=None,
        cache_snapshot=snapshot,
        logits=None,
        hidden=None,
        has_recurrent=recurrent,
        gdn_boundaries=[(int(b), snapshot, None) for b in boundaries],
    )


def test_the_keyed_view_marks_exactly_the_image_rows():
    from mtplx.vision.splice import image_key_runs, is_image_key, unkeyed_ids

    assert [i for i, t in enumerate(KEYED) if is_image_key(t)] == [
        *range(*FIRST),
        *range(*SECOND),
    ]
    assert image_key_runs(KEYED) == [FIRST, SECOND]
    assert image_key_runs(RAW) == []  # raw pads are model ids, not keys
    assert unkeyed_ids(KEYED, PAD) == RAW
    assert image_key_runs([*KEYED[:41]]) == [(40, 41)]


def test_image_safe_restore_len_keeps_every_image_whole():
    from mtplx.vision.splice import image_safe_restore_len, inside_image

    for point in range(len(KEYED) + 1):
        inside = FIRST[0] < point < FIRST[1] or SECOND[0] < point < SECOND[1]
        assert inside_image(KEYED, point) is inside
        safe = image_safe_restore_len(KEYED, point, reforwards_last_token=False)
        if FIRST[0] < point < FIRST[1]:
            assert safe == FIRST[0]
        elif SECOND[0] < point < SECOND[1]:
            assert safe == SECOND[0]
        else:
            assert safe == point
    # A restore that re-forwards its last token never ends on an image row,
    # so the image's end is cut back to its start as well.
    assert image_safe_restore_len(KEYED, FIRST[1], reforwards_last_token=True) == FIRST[0]
    assert image_safe_restore_len(KEYED, FIRST[1] + 1, reforwards_last_token=True) == FIRST[1] + 1
    assert image_safe_restore_len(KEYED, FIRST[1], reforwards_last_token=False) == FIRST[1]
    # Text sequences come back unchanged at every point.
    text = [*HEAD, *MIDDLE, *TAIL]
    for point in range(len(text) + 1):
        for seed in (False, True):
            assert image_safe_restore_len(text, point, reforwards_last_token=seed) == point


def test_a_boundary_inside_an_image_is_never_restored_at():
    # Boundaries before the first image, inside it, right after it, inside
    # the second image and in the text after it.
    entry = _entry(KEYED, boundaries=(32, 43, 46, 78, 84))
    at = lambda matched: entry.recurrent_boundary_at_or_below(matched)[0]  # noqa: E731
    assert at(90) == 84
    assert at(83) == 46  # 78 lies inside the second image
    assert at(79) == 46
    assert at(46) == 46  # the end of an image is text on one side: restorable
    assert at(45) == 32  # 43 lies inside the first image
    assert at(43) == 32
    assert _recurrent_restore_point(entry, 83) == 46
    assert _recurrent_restore_point(entry, 45) == 32
    assert _recurrent_restore_point(entry, 90) == 84


def test_text_entries_keep_every_boundary():
    text = [*HEAD, *MIDDLE, *TAIL]
    entry = _entry(text, boundaries=(32, 43, 46, 78, 84))
    for matched, expected in ((90, 84), (83, 78), (45, 43), (43, 43), (40, 32)):
        assert entry.recurrent_boundary_at_or_below(matched)[0] == expected
        assert _recurrent_restore_point(entry, matched) == expected


@pytest.mark.parametrize("recurrent", [False, True])
def test_a_candidate_never_cuts_an_image(recurrent):
    # The prompt shares the first 43 keyed tokens with the entry: the text
    # and three rows of the first image, then rows keyed for other pixels.
    # Real keys of one image agree on every row, so a match ending inside an
    # image is contrived here; the rule must hold for it all the same.
    other = list(KEYED)
    other[43:46] = vision_bank_key_ids(RAW, _splice(digests=(0xC3, 0xB2)))[43:46]
    entry = _entry(other, boundaries=(32,) if recurrent else (), recurrent=recurrent)
    candidate = _near_prefix_candidate_len(
        entry, KEYED, allow_block_prefix=True, **KNOBS
    )
    assert candidate == FIRST[0]


def test_a_seed_forward_candidate_never_ends_on_an_image_row():
    # The entry diverges right after the first image: an entry without
    # recurrent state would re-forward the image's last row as a pad id.
    other = [*KEYED[: FIRST[1]], 7, 7, 7, *KEYED[FIRST[1] + 3 :]]
    attention_only = _entry(other, recurrent=False)
    assert (
        _near_prefix_candidate_len(attention_only, KEYED, allow_block_prefix=True, **KNOBS)
        == FIRST[0]
    )
    # A recurrent entry lands on a boundary instead: the image end is fine.
    recurrent = _entry(other, boundaries=(FIRST[1],), recurrent=True)
    assert (
        _near_prefix_candidate_len(recurrent, KEYED, allow_block_prefix=True, **KNOBS)
        == FIRST[1]
    )


def test_text_candidates_are_unchanged():
    text = [*HEAD, *MIDDLE, *TAIL]
    other = [*text[:61], 7, 7, *text[63:]]
    for recurrent in (False, True):
        entry = _entry(other, boundaries=(32,), recurrent=recurrent)
        assert (
            _near_prefix_candidate_len(entry, tuple(text), allow_block_prefix=True, **KNOBS)
            == 61
        )
