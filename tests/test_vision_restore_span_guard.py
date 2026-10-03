"""Vision restore-span guard (2026-08-07 pillar alias-leg regression).

On raw token ids every image pad equals every image pad, so a near-prefix
match could run into an image span whose embeddings came from different
pixels, and a boundary restore there resurrected the wrong image's KV. The
lane was capped at the first pad for that. It now matches an image prompt in
its content-keyed view (where rows of different pixels differ), and a restore
never cuts an image (tests/test_vision_keyed_restore_rules.py); these tests
pin the span helpers and that the lane never asks the bank with raw ids.
"""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx

from mtplx.vision.splice import (
    VisionSplice,
    clamp_matched_outside_image_spans,
    vision_bank_key_ids,
    vision_image_spans,
)

PAD = 151655
# What the lane reads of a runtime before it restores anything.
RUNTIME = SimpleNamespace(model_path="m", mtp_enabled=True)


def _splice(pad_counts, digests):
    return VisionSplice(
        image_pad_token_id=PAD,
        embeddings=mx.zeros((sum(pad_counts), 8)),
        image_digests=tuple(digests),
        pad_counts=tuple(pad_counts),
    )


def test_spans_on_raw_and_keyed_ids():
    ids = [1, 2, 3] + [PAD] * 4 + [7, 8] + [PAD] * 2 + [9]
    sp = _splice([4, 2], [0xAA, 0xBB])
    assert vision_image_spans(ids, sp) == [(3, 7), (9, 11)]
    keyed = vision_bank_key_ids(ids, sp)
    assert keyed is not None
    assert vision_image_spans(keyed, sp) == [(3, 7), (9, 11)]


def test_clamp_semantics():
    spans = [(3, 7), (9, 11)]
    assert clamp_matched_outside_image_spans(2, spans) == 2
    assert clamp_matched_outside_image_spans(3, spans) == 3  # at start: safe
    assert clamp_matched_outside_image_spans(5, spans) == 3  # inside: snap
    assert clamp_matched_outside_image_spans(7, spans) == 7  # full span: keep
    assert clamp_matched_outside_image_spans(10, spans) == 9
    assert clamp_matched_outside_image_spans(11, spans) == 11
    assert clamp_matched_outside_image_spans(5, None) == 5


def test_keyed_ids_differ_from_first_pad_for_different_pixels():
    ids = [1, 2, 3] + [PAD] * 4 + [7]
    a = vision_bank_key_ids(ids, _splice([4], [0x1111]))
    b = vision_bank_key_ids(ids, _splice([4], [0x2222]))
    assert a is not None and b is not None
    assert a[:3] == b[:3]
    assert all(x != y for x, y in zip(a[3:7], b[3:7]))


def test_near_prefix_lane_asks_the_bank_with_the_keyed_view():
    from mtplx import generation as g

    raw = [1, 2, 3] + [PAD] * 4 + [7] * 60
    keyed = vision_bank_key_ids(raw, _splice([4], [0xAA]))
    seen = {}

    class Bank:
        def near_prefix_candidates(self, prompt_ids, **kw):
            seen["ids"] = list(prompt_ids)
            return []

    out = g._restore_near_prefix_prompt_state(
        RUNTIME,
        raw,
        base_hidden_variant="b",
        mtp_hidden_variant="m",
        mtp_history_policy="cycle",
        session_bank=Bank(),
        template_hash=None,
        draft_head_identity=None,
        policy_fingerprint=None,
        match_ids=keyed,
    )
    assert out is None
    # The raw ids would match any image with the same pad count.
    assert seen["ids"] == keyed != raw


def test_near_prefix_lane_accepts_and_threads_vision_splice():
    """#296 wiring: the near lane takes the splice (it was vision-blind —
    image pads in the suffix were forwarded as plain ids and the rows never
    reached the KV). With nothing to restore the splice must pass through
    untouched; consumption is guarded downstream by the unconsumed-rows
    assert in _suffix_chunk_embeddings."""
    from mtplx import generation as g

    class Splice:
        image_pad_token_id = 7
        cursor = None

        def remaining(self):
            return 1

    class Bank:
        def near_prefix_candidates(self, prompt_ids, **kw):
            return []

    splice = Splice()
    out = g._restore_near_prefix_prompt_state(
        RUNTIME,
        [1] * 64,
        base_hidden_variant="b",
        mtp_hidden_variant="m",
        mtp_history_policy="cycle",
        session_bank=Bank(),
        template_hash=None,
        draft_head_identity=None,
        policy_fingerprint=None,
        vision_splice=splice,
    )
    assert out is None
    assert splice.cursor is None  # no restore happened; splice untouched
