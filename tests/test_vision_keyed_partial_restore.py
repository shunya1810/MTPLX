"""A screenshot turn restores its conversation up to the change, past its images.

On 2026-09-29 every Pi turn after a screenshot diverged from the banked
conversation inside the newest assistant turn and restored nothing past the
first image: the near-prefix lane matched raw ids and was capped at the first
image pad. 123K to 138K tokens were read again, cold, each time.

The lanes now ask the bank the content-keyed question for an image prompt,
partial matches included, and a restore never cuts an image. These tests run
the real restore_or_prefill_prompt_state on a real SessionBank with the toy
hybrid model of test_checkpoint_pre_image (an attention layer and an
untrimmable running sum; logits are exact integer sums over every input row,
image rows included, so a restore that held other rows cannot match a cold
prefill), for both prefill loops. A turn is banked the way the one-copy store
banks a Flash-Next turn: the prompt keeps its checkpoints up to its first image
and the anchor at its end (one_copy.prompt_lease_fields), and the entry of the
prompt plus its answer inherits them (SessionBank.put).

* a reply that changed after the screenshot resumes at the prompt's end, past
  the screenshot, and equals a cold prefill (fails on the old code: cold);
* a new screenshot after the change is fed from the splice (fails on the old
  code: cold);
* other pixels in the old screenshot, the same bytes at another grid, and an
  edit before the screenshot resume under the change, never past it.

CPU-sized: 16-id vocabulary, no model pack, no tower.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import pytest
from test_checkpoint_pre_image import CHUNK, PAD, VOCAB, HybridToy

from mtplx.cache_state import snapshot_cache, snapshot_untrimmable_cache
from mtplx.generation import (
    _resolve_runtime_base_hidden_variant,
    restore_or_prefill_prompt_state,
)
from mtplx.mtp_patch import MTPContract
from mtplx.runtime import MTPLXRuntime
from mtplx.session_bank import SessionBank
from mtplx.vision.splice import VisionSplice, vision_bank_key_ids

SESSION = "pi"
IMAGE_A = 0x5C4EE45A0F7B1D23
IMAGE_A_OTHER_PIXELS = 0x1D8A6F20C0DE4411
IMAGE_B = 0x2B7E151628AED2A6
PADS = 6
BEFORE = [(7 * i + 3) % 14 for i in range(43)]  # the first image at 43
AFTER = [(5 * i + 1) % 14 for i in range(24)]
REPLY = [(3 * i + 8) % 14 for i in range(20)]
# The client re-renders the answer: it parts from the model's own tokens 8
# tokens in (an edit's tool call rendered with other key order).
RESENT = [*REPLY[:8], (REPLY[8] + 1) % 14, *REPLY[9:]]
NEW_TURN = [(11 * i + 2) % 14 for i in range(18)]


def _runtime() -> MTPLXRuntime:
    return MTPLXRuntime(
        model=HybridToy(),
        tokenizer=SimpleNamespace(),
        model_path=Path("models/keyed-partial-restore"),
        mtp_enabled=True,
        contract=MTPContract(),
    )


def _splice(*images: tuple[int, int]) -> VisionSplice:
    """``(digest, pad count)`` per image, in prompt order; the rows are a pure
    function of the digest, as a tower's are of the pixels."""

    rows = [
        [float((digest % 9973 + 31 * row + column) % 5) for column in range(VOCAB)]
        for digest, pads in images
        for row in range(pads)
    ]
    return VisionSplice(
        image_pad_token_id=PAD,
        embeddings=mx.array(rows, dtype=mx.float32),
        image_digests=tuple(digest for digest, _pads in images),
        pad_counts=tuple(pads for _digest, pads in images),
    )


def _image(pads: int = PADS) -> list[int]:
    return [PAD] * pads


@pytest.fixture(params=["cycle", "committed"])
def policy(request, monkeypatch):
    """Both prefill loops: ``cycle`` runs ``_prefill``, ``committed`` the
    streaming loop that also builds the draft head's history."""

    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", "1")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE", str(CHUNK))
    monkeypatch.setenv("MTPLX_SESSION_STORE_ON_PREFILL_MIN_SUFFIX", "1")
    # The toy prompts are far below the production block and match floors.
    monkeypatch.setenv("MTPLX_SESSION_PREFIX_BLOCK_SIZE", "8")
    monkeypatch.setenv("MTPLX_SESSION_BLOCK_PREFIX_MIN_MATCH_TOKENS", "8")
    # Room for every checkpoint of these prompts: the tests are about where a
    # restore may resume, not about which checkpoints the budget keeps.
    monkeypatch.setenv("MTPLX_GDN_BOUNDARY_MAX", "16")
    for name in (
        "MTPLX_GDN_BOUNDARY_CAPTURE",
        "MTPLX_MTP_HISTORY_POLICY",
        "MTPLX_SESSION_NEAR_PREFIX_RESTORE",
        "MTPLX_ONE_COPY",
    ):
        monkeypatch.delenv(name, raising=False)
    return request.param


def _prefill(prompt, images, *, policy, bank=None):
    return restore_or_prefill_prompt_state(
        _runtime(),
        list(prompt),
        mtp_history_policy=policy,
        session_bank=bank,
        session_id=SESSION if bank is not None else None,
        vision_splice=_splice(*images),
        store_prefix_snapshot=False,
    )


def _put(bank, rt, ids, images, state, *, boundaries=None):
    committed = state.committed_mtp_cache
    kwargs = {} if boundaries is None else {"gdn_boundaries": boundaries}
    return bank.put(
        runtime=rt,
        token_ids=vision_bank_key_ids(list(ids), _splice(*images)),
        cache=state.trunk_cache,
        logits=state.logits,
        hidden=state.hidden,
        hidden_variant=_resolve_runtime_base_hidden_variant(rt, None),
        session_id=SESSION,
        mtp_history_policy=state.mtp_history_policy,
        snapshot_epoch=len(ids),
        mtp_snapshot_epoch=len(ids) if committed is not None else None,
        mtp_history_snapshot=snapshot_cache(committed) if committed is not None else None,
        **kwargs,
    )


def _bank_turn(bank, prompt, reply, images, *, policy, expect=None):
    """One finished image turn: the prompt with its checkpoints (outside its
    images) plus the anchor at its end, then the prompt plus its answer
    (which inherits them, as the generation-final commit does)."""

    rt = _runtime()
    # A bank in the prefill makes it keep its pre-image checkpoints.
    state = _prefill(prompt, images, policy=policy, bank=SessionBank())
    anchor = (len(prompt), snapshot_untrimmable_cache(state.trunk_cache), state.hidden)
    _put(bank, rt, prompt, images, state, boundaries=[*state.gdn_boundaries, anchor])
    full = [*prompt, *reply]
    entry = _put(bank, rt, full, images, _prefill(full, images, policy=policy))
    kept = [int(record[0]) for record in entry.gdn_boundaries]
    # Every chunk end but 48, inside the screenshot's rows [43, 49).
    assert kept == (expect or [8, 16, 24, 32, 40, 56, 64, 72, len(prompt)])
    return entry


def _check(bank, prompt, images, *, policy, cached):
    warm = _prefill(prompt, images, policy=policy, bank=bank)
    assert warm.cached_tokens == cached, (warm.restore_mode, warm.cache_miss_reason)
    assert warm.suffix_tokens == len(prompt) - cached
    cold = _prefill(prompt, images, policy=policy)
    assert warm.logits.tolist() == cold.logits.tolist()
    return warm


FIRST_PROMPT = [*BEFORE, *_image(), *AFTER]


def test_a_reply_changed_after_the_screenshot_resumes_at_the_prompt_end(policy):
    bank = SessionBank()
    _bank_turn(bank, FIRST_PROMPT, REPLY, [(IMAGE_A, PADS)], policy=policy)
    second = [*FIRST_PROMPT, *RESENT, *NEW_TURN]
    warm = _check(bank, second, [(IMAGE_A, PADS)], policy=policy, cached=len(FIRST_PROMPT))
    assert warm.cached_tokens > len(BEFORE) + PADS  # past the screenshot


def test_a_new_screenshot_after_the_change_is_fed_from_the_splice(policy):
    bank = SessionBank()
    _bank_turn(bank, FIRST_PROMPT, REPLY, [(IMAGE_A, PADS)], policy=policy)
    second = [*FIRST_PROMPT, *RESENT, *NEW_TURN[:9], *_image(), *NEW_TURN[9:]]
    # The restore resumes past the first screenshot; the suffix reads the
    # second one's rows from the splice, after the first one's.
    _check(
        bank, second, [(IMAGE_A, PADS), (IMAGE_B, PADS)], policy=policy,
        cached=len(FIRST_PROMPT),
    )


def test_other_pixels_in_the_old_screenshot_never_restore_past_it(policy):
    bank = SessionBank()
    _bank_turn(bank, FIRST_PROMPT, REPLY, [(IMAGE_A, PADS)], policy=policy)
    second = [*FIRST_PROMPT, *REPLY, *NEW_TURN]
    # The same ids as the banked turn, other pixels: the keyed match stops at
    # the first row, so the newest checkpoint under it is the pre-image one.
    _check(bank, second, [(IMAGE_A_OTHER_PIXELS, PADS)], policy=policy, cached=40)


def test_the_same_bytes_at_another_grid_part_inside_the_image(policy):
    bank = SessionBank()
    _bank_turn(bank, FIRST_PROMPT, REPLY, [(IMAGE_A, PADS)], policy=policy)
    # The same digest expanded to two more rows (another preprocessor): the
    # keyed rows agree for six rows and part inside the image. The restore
    # steps back before the image.
    second = [*BEFORE, *_image(PADS + 2), *AFTER, *REPLY, *NEW_TURN]
    _check(bank, second, [(IMAGE_A, PADS + 2)], policy=policy, cached=40)


def test_an_edit_before_the_screenshot_resumes_under_the_edit(policy):
    bank = SessionBank()
    _bank_turn(bank, FIRST_PROMPT, REPLY, [(IMAGE_A, PADS)], policy=policy)
    edited = list(BEFORE)
    edited[20] = (edited[20] + 1) % 14
    second = [*edited, *_image(), *AFTER, *REPLY, *NEW_TURN]
    _check(bank, second, [(IMAGE_A, PADS)], policy=policy, cached=16)


def test_an_edit_after_the_screenshot_resumes_under_the_edit(policy):
    """The client re-renders something after the screenshot but before the
    reply (a tool result it trimmed). The restore resumes at the checkpoint
    under the edit, past the screenshot, instead of re-reading from the
    screenshot on (fails on the old code: 40, the checkpoint before it)."""

    bank = SessionBank()
    _bank_turn(bank, FIRST_PROMPT, REPLY, [(IMAGE_A, PADS)], policy=policy)
    edited = list(AFTER)
    edited[10] = (edited[10] + 1) % 14  # position 43 + 6 + 10 = 59
    second = [*BEFORE, *_image(), *edited, *REPLY, *NEW_TURN]
    _check(bank, second, [(IMAGE_A, PADS)], policy=policy, cached=56)


def test_an_edit_between_two_screenshots_resumes_past_the_first(policy):
    """Two screenshots: an edit in the text between them resumes past the
    first one, under the edit, and the second one is fed from the splice."""

    bank = SessionBank()
    between = AFTER[:12]
    prompt = [*BEFORE, *_image(), *between, *_image(), *AFTER[12:]]
    images = [(IMAGE_A, PADS), (IMAGE_B, PADS)]
    # Images at [43, 49) and [61, 67): chunk ends 48 and 64 fall inside; the
    # last forward of the 79-token prompt ends at 78.
    _bank_turn(
        bank, prompt, REPLY, images, policy=policy,
        expect=[8, 16, 24, 32, 40, 56, 72, 78, len(prompt)],
    )
    edited = list(between)
    edited[9] = (edited[9] + 1) % 14  # position 43 + 6 + 9 = 58
    second = [*BEFORE, *_image(), *edited, *_image(), *AFTER[12:], *REPLY, *NEW_TURN]
    _check(bank, second, images, policy=policy, cached=56)
