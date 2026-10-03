"""An image prompt keeps recurrent checkpoints before and after its images.

Until 2026-09-30 a prefill that carried an image captured no recurrent
checkpoints at all (the cold and exact-restore lanes built no sink for it).
On a hybrid model a later turn can only resume at a checkpoint, so when it
changed anything after an image (a Pi screenshot turn whose tool call parts
from the stored reply) nothing was restorable and the whole conversation was
read again: 123K to 138K tokens cold on 2026-09-29.

At first only the text before the first image was restorable: the
near-prefix lane matched raw ids and stopped at the first image pad, so an
image prompt kept the checkpoints at or before its first image. The lane now
matches the content-keyed view and resumes past whole images, so an image
prompt keeps every checkpoint outside its images (never one inside, where a
restore would resume with part of an image's rows), still recorded only
where its plain chunk grid already ends a forward, and reports how close to
the first image the newest one before it sits. These tests drive the real
restore_or_prefill_prompt_state
on a real SessionBank with a toy hybrid model (an attention layer and a
recurrent running sum, logits exact small-integer sums over every input row,
image rows included), for both prefill loops:

* the prompt keeps checkpoints outside its image, none inside it, and its
  coverage report says so (fails on the old code: no checkpoints);
* keeping them does not change the image prefill: same forwards, same logits;
* the next turn, changed after the image, restores at the pre-image anchor
  and its logits equal a cold prefill's (fails on the old code: cold);
* a turn changed before the image restores at the newest anchor at or below
  its match, never past it;
* a turn that extends the image prompt restores it exactly through the image
  and banks an entry that still carries the anchors (fails on the old code:
  none).

CPU-sized: 16-id vocabulary, no model pack, no tower.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import pytest
from mlx_lm.models.cache import ArraysCache, KVCache

from mtplx.generation import restore_or_prefill_prompt_state
from mtplx.mtp_patch import MTPContract
from mtplx.runtime import MTPLXRuntime
from mtplx.session_bank import SessionBank
from mtplx.vision.splice import VisionSplice, vision_bank_key_ids

VOCAB = 16
PAD = 15
PADS = 6
IMAGE_START = 43  # the first image-pad position: 43 text tokens before it
CHUNK = 8

_MIX = mx.array(
    [[((i * 5 + j * 3) % 11) - 5 for j in range(VOCAB)] for i in range(VOCAB)],
    dtype=mx.float32,
)


class _OneHot:
    def __call__(self, ids):
        return mx.eye(VOCAB, dtype=mx.float32)[ids]


class HybridToy:
    """Causal toy hybrid. Layer 0 is attention (a KVCache of the input
    rows); layer 1 is recurrent (an ArraysCache holding the running sum of
    the input rows, which cannot be trimmed, like a GDN state). Logits mix
    both, so a restore whose KV or recurrent state came from another
    position, or from other pixels, cannot reproduce a cold prefill."""

    def __init__(self) -> None:
        self.model = SimpleNamespace(embed_tokens=_OneHot())
        self.forwards: list[int] = []

    def make_cache(self):
        return [KVCache(), ArraysCache(size=1)]

    def make_mtp_cache(self):
        return []

    def mtp_update_cache(self, hidden_states, next_token_ids, **_kwargs):
        return hidden_states

    def __call__(
        self,
        input_ids,
        *,
        cache=None,
        return_hidden: bool = False,
        hidden_variant: str | None = None,
        emit_logits: bool = True,
        logits_keep: int | None = None,
        input_embeddings=None,
    ):
        del hidden_variant
        batch, length = int(input_ids.shape[0]), int(input_ids.shape[1])
        self.forwards.append(length)
        rows = (
            input_embeddings
            if input_embeddings is not None
            else mx.eye(VOCAB, dtype=mx.float32)[input_ids]
        )
        keys, _values = cache[0].update_and_fetch(
            rows[:, None, :, :], rows[:, None, :, :]
        )
        carried = cache[1][0]
        if carried is None:
            carried = mx.zeros((batch, VOCAB), dtype=mx.float32)
        running = carried[:, None, :] + mx.cumsum(rows, axis=1)
        cache[1][0] = running[:, -1, :]
        hidden = mx.zeros((batch, length, 2), dtype=mx.float32)
        if not emit_logits:
            return (None, hidden) if return_hidden else None
        attended = mx.cumsum(keys[:, 0, :, :], axis=1)[:, -length:, :]
        logits = (attended + 2 * running + running * running) @ _MIX
        # logits_keep trims the logits only; the hidden rows all come back,
        # as the draft head's history needs them.
        keep = length if logits_keep is None else min(length, max(1, int(logits_keep)))
        logits = logits[:, -keep:, :]
        if return_hidden:
            return logits, hidden
        return logits


def _runtime() -> MTPLXRuntime:
    return MTPLXRuntime(
        model=HybridToy(),
        tokenizer=SimpleNamespace(),
        model_path=Path("models/pre-image-anchors"),
        mtp_enabled=True,
        contract=MTPContract(),
    )


def _splice(digest: int) -> VisionSplice:
    rows = [
        [float((digest % 9973 + 31 * row + column) % 5) for column in range(VOCAB)]
        for row in range(PADS)
    ]
    return VisionSplice(
        image_pad_token_id=PAD,
        embeddings=mx.array(rows, dtype=mx.float32),
        image_digests=(digest,),
        pad_counts=(PADS,),
    )


IMAGE = 0x5C4EE45A0F7B1D23
TEXT_BEFORE = [(7 * i + 3) % 14 for i in range(IMAGE_START)]
AFTER_ONE = [(5 * i + 1) % 14 for i in range(24)]
AFTER_TWO = [(3 * i + 8) % 14 for i in range(30)]
assert AFTER_ONE[0] != AFTER_TWO[0]


def _prompt(before: list[int], after: list[int]) -> list[int]:
    return [*before, *([PAD] * PADS), *after]


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
    for name in (
        "MTPLX_GDN_BOUNDARY_CAPTURE",
        "MTPLX_GDN_BOUNDARY_MAX",
        "MTPLX_MTP_HISTORY_POLICY",
        "MTPLX_SESSION_NEAR_PREFIX_RESTORE",
    ):
        monkeypatch.delenv(name, raising=False)
    return request.param


def _state(rt, prompt, *, policy, bank=None):
    return restore_or_prefill_prompt_state(
        rt,
        list(prompt),
        mtp_history_policy=policy,
        session_bank=bank,
        vision_splice=_splice(IMAGE),
        store_prefix_snapshot=bank is not None,
    )


def _positions(records) -> list[int]:
    return [int(record[0]) for record in records]


# The plain chunk ends of the 73-token prompt, but 48: it falls inside the
# image's rows [43, 49).
OUTSIDE_THE_IMAGE = [8, 16, 24, 32, 40, 56, 64, 72]


def test_an_image_prompt_keeps_checkpoints_outside_its_image(policy):
    bank = SessionBank()
    rt = _runtime()
    state = _state(rt, _prompt(TEXT_BEFORE, AFTER_ONE), policy=policy, bank=bank)
    # Chunk ends before and after the image, none inside it.
    assert _positions(state.gdn_boundaries) == OUTSIDE_THE_IMAGE
    assert state.checkpoint_coverage["anchors"] == OUTSIDE_THE_IMAGE
    assert state.checkpoint_coverage["pre_image"] == {
        "first_image_start": IMAGE_START,
        "anchor": 40,
        "tokens_to_reread": 3,
    }
    # The entry banked under the content-keyed ids carries them.
    keyed = vision_bank_key_ids(list(state.token_prefix), _splice(IMAGE))
    entry = bank.longest_prefix(keyed)
    assert entry is not None and entry.prefix_len == len(keyed)
    assert _positions(entry.gdn_boundaries) == OUTSIDE_THE_IMAGE


def test_keeping_them_does_not_change_the_image_prefill(policy):
    prompt = _prompt(TEXT_BEFORE, AFTER_ONE)
    plain_rt = _runtime()
    plain = _state(plain_rt, prompt, policy=policy)
    banked_rt = _runtime()
    banked = _state(banked_rt, prompt, policy=policy, bank=SessionBank())
    assert banked_rt.model.forwards == plain_rt.model.forwards
    assert banked.logits.tolist() == plain.logits.tolist()
    assert banked.gdn_boundaries and not plain.gdn_boundaries


def test_a_turn_changed_after_the_image_restores_at_the_pre_image_anchor(policy):
    bank = SessionBank()
    _state(_runtime(), _prompt(TEXT_BEFORE, AFTER_ONE), policy=policy, bank=bank)
    second = _prompt(TEXT_BEFORE, AFTER_TWO)
    warm = _state(_runtime(), second, policy=policy, bank=bank)
    assert warm.cached_tokens == 40, (warm.restore_mode, warm.cache_miss_reason)
    assert warm.suffix_tokens == len(second) - 40
    cold = _state(_runtime(), second, policy=policy)
    assert warm.logits.tolist() == cold.logits.tolist()
    # The restored turn keeps the anchors it inherited, still none past the
    # image, and reports them.
    assert _positions(warm.gdn_boundaries) == [8, 16, 24, 32, 40]
    assert warm.checkpoint_coverage["pre_image"]["anchor"] == 40


def test_a_turn_changed_before_the_image_never_restores_past_its_match(policy):
    bank = SessionBank()
    _state(_runtime(), _prompt(TEXT_BEFORE, AFTER_ONE), policy=policy, bank=bank)
    edited = list(TEXT_BEFORE)
    edited[20] = (edited[20] + 1) % 14
    second = _prompt(edited, AFTER_ONE)
    warm = _state(_runtime(), second, policy=policy, bank=bank)
    assert warm.cached_tokens == 16
    cold = _state(_runtime(), second, policy=policy)
    assert warm.logits.tolist() == cold.logits.tolist()


def test_a_turn_that_extends_the_image_prompt_carries_its_anchors(policy):
    """The next turn of an agent session extends the stored prompt, image
    included: the exact restore serves it through the image, and the entry
    it banks must still carry the pre-image anchors, or the turn after it
    has nothing to fall back to."""

    bank = SessionBank()
    first = _prompt(TEXT_BEFORE, AFTER_ONE)
    _state(_runtime(), first, policy=policy, bank=bank)
    extended = [*first, *AFTER_TWO]
    warm = _state(_runtime(), extended, policy=policy, bank=bank)
    assert warm.cached_tokens == len(first)
    assert _positions(warm.gdn_boundaries) == OUTSIDE_THE_IMAGE
    cold = _state(_runtime(), extended, policy=policy)
    assert warm.logits.tolist() == cold.logits.tolist()
    keyed = vision_bank_key_ids(extended, _splice(IMAGE))
    assert _positions(bank.longest_prefix(keyed).gdn_boundaries) == OUTSIDE_THE_IMAGE
