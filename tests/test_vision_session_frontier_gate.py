"""Vision session frontier: image turns commit the content-keyed view.

The engine-session frontier used to hold raw token ids, which carry no
image identity: every image pad shares one vocab id, so a committed vision
frontier let a later request with DIFFERENT pixels but identical pad ids
restore another image's KV. The pillar gate's correctness sentinel
(``different_image_alias_blocked``) caught exactly that once the F39
frontier fix made these histories commit, and image histories stopped
committing at all (2026-08-17). That froze every image session at its last
text turn: on 2026-09-29 every Pi turn after a screenshot re-read 123K to
138K tokens.

An image history now commits the content-keyed view the bank keys it by
(``vision_bank_key_ids``). Proven here on the F39 harness pattern
(``test_final_committed_frontier_byte_skip``):

  1. a vision history advances the frontier in the keyed view, in both the
     stored arm and the oversized-projection arm; raw pad ids never equal
     it;
  2. the bank store proceeds as before, keyed by content surrogates;
  3. the same ids with DIFFERENT pixels never match the frontier or the
     entry past the first pad, so such a request can never adopt this
     conversation's KV at or past the image (the alias-blocking property);
  4. either vision session switch off keeps the frontier where it was;
  5. text-only histories keep the F39 contract.

CPU-only: tiny deterministic model, real ``SessionBank``; no model
packs, no GPU.
"""

from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import pytest
from mlx_lm.models.cache import KVCache

from mtplx.engine_session import EngineSession, EngineSessionManager
from mtplx.mtp_patch import MTPContract
from mtplx.runtime import MTPLXRuntime
from mtplx.server import openai as oa
from mtplx.session_bank import SessionBank
from mtplx.vision.splice import vision_bank_key_ids

VOCAB = 32
PAD = 31
# Text tokens stay in [0, 30) so PAD occurrences are exactly the ones we
# place: 40 text tokens, one 6-pad image, 12 trailing text tokens.
TEXT_PREFIX = [(i * 5 + 3) % 30 for i in range(40)]
TEXT_TAIL = [(i * 7 + 1) % 30 for i in range(12)]
PAD_COUNT = 6
VISION_HISTORY = TEXT_PREFIX + [PAD] * PAD_COUNT + TEXT_TAIL
TEXT_HISTORY = TEXT_PREFIX + TEXT_TAIL
BLUE_DIGEST = 0x1122334455667788
RED_DIGEST = 0x99AABBCCDDEEFF00
POLICY = "vision-frontier-gate-policy"

_MIX = mx.array(
    [[((i * 7 + j * 13) % 31) - 15 for j in range(VOCAB)] for i in range(VOCAB)],
    dtype=mx.float32,
)


def _splice(digest: int) -> SimpleNamespace:
    """Minimal stand-in carrying exactly the content-identity surface
    ``vision_bank_key_ids`` and the postcommit store read."""

    return SimpleNamespace(
        image_digests=[digest],
        pad_counts=[PAD_COUNT],
        image_pad_token_id=PAD,
    )


class _Tokenizer:
    def decode(self, tokens, **_kwargs):
        return "".join(f"<{int(token)}>" for token in tokens)


class HistoryCountModel:
    """Causal toy model (F39 harness): logits are exact integer sums of
    the history through a fixed mixing matrix."""

    def __init__(self):
        self.calls: list[int] = []

    def make_cache(self):
        return [KVCache()]

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
    ):
        del hidden_variant
        batch, length = int(input_ids.shape[0]), int(input_ids.shape[1])
        self.calls.append(length)
        onehot = mx.eye(VOCAB, dtype=mx.float32)[input_ids]
        entry = cache[0]
        keys, _values = entry.update_and_fetch(
            onehot[:, None, :, :], onehot[:, None, :, :]
        )
        hidden = mx.zeros((batch, length, 2), dtype=mx.float32)
        if not emit_logits:
            return (None, hidden) if return_hidden else None
        counts = mx.cumsum(keys[:, 0, :, :], axis=1)
        counts = counts[:, -length:, :]
        logits = counts @ _MIX
        keep = length if logits_keep is None else min(length, max(1, int(logits_keep)))
        logits = logits[:, -keep:, :]
        if return_hidden:
            return logits, hidden[:, -keep:, :]
        return logits


def _runtime() -> MTPLXRuntime:
    return MTPLXRuntime(
        model=HistoryCountModel(),
        tokenizer=_Tokenizer(),
        model_path=Path("models/vision-frontier-gate"),
        mtp_enabled=False,
        contract=MTPContract(),
    )


class _ForegroundState:
    """Minimal ServerState stand-in for _store_retokenized_history_snapshot."""

    def __init__(self, runtime: MTPLXRuntime, bank: SessionBank) -> None:
        self.runtime = runtime
        self.sessions = SimpleNamespace(bank=bank)
        self.lock = threading.Lock()
        self.template_hash = None
        self.draft_head_identity = None

    def begin_foreground(self) -> None:
        pass

    def end_foreground(self) -> None:
        pass


def _patch_history(
    monkeypatch: pytest.MonkeyPatch, history_ids: list[int], splice
) -> None:
    monkeypatch.setattr(
        oa,
        "_history_ids_for_postcommit",
        lambda *_args, **_kwargs: (list(history_ids), splice),
    )
    if splice is not None:
        # The store's committed-history prefill feeds the splice's
        # embedding rows into the model; the toy model is id-driven, and
        # nothing in this file asserts KV content — only keying and
        # frontier behavior. Strip the vision kwarg so the real prefill
        # runs on ids while the store's keying still sees the splice.
        real_prefill = oa.restore_or_prefill_prompt_state

        def _prefill_without_vision(*args, **kwargs):
            kwargs.pop("vision_splice", None)
            return real_prefill(*args, **kwargs)

        monkeypatch.setattr(
            oa, "restore_or_prefill_prompt_state", _prefill_without_vision
        )


def _run_postcommit(state: _ForegroundState, session: EngineSession | None) -> dict:
    return oa._store_retokenized_history_snapshot(
        state,
        session_id="vision-gate",
        messages=[],
        assistant_content="turn answer",
        thinking_enabled=False,
        policy_fingerprint=POLICY,
        session=session,
        expected_session_revision=(
            session.revision if session is not None else None
        ),
        keep_live_ref=True,
    )


def _bank() -> SessionBank:
    return SessionBank(max_entries=8, max_bytes=1 << 30, per_session_max_bytes=1 << 30)


def test_vision_history_commits_the_keyed_frontier_and_banks_the_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime()
    bank = _bank()
    state = _ForegroundState(runtime, bank)
    session = EngineSession("vision-gate")
    _patch_history(monkeypatch, VISION_HISTORY, _splice(BLUE_DIGEST))

    outcome = _run_postcommit(state, session)

    blue_keys = vision_bank_key_ids(VISION_HISTORY, _splice(BLUE_DIGEST))
    assert outcome["stored"] is True, outcome
    assert outcome["session_commit"] == {
        "committed": True,
        "reason": "committed_retokenized_prefix",
        "prefix_len": len(VISION_HISTORY),
    }
    # The frontier holds the keyed view, never the raw pad ids.
    assert list(session.committed_token_ids) == blue_keys
    assert PAD not in session.committed_token_ids

    # The entry is content-keyed: raw pad ids cannot find it, the
    # surrogate view finds it at full length.
    assert bank.longest_prefix(VISION_HISTORY) is None
    entry = bank.longest_prefix(blue_keys)
    assert entry is not None
    assert entry.prefix_len == len(VISION_HISTORY)


def test_different_pixels_never_match_the_frontier_past_the_image(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The pillar alias guarantee, at every place the frontier is read: a
    request with other pixels behind the same ids never adopts this
    conversation's KV at or past the image."""

    runtime = _runtime()
    bank = _bank()
    state = _ForegroundState(runtime, bank)
    sessions = EngineSessionManager(bank=bank)
    session = sessions.get_or_create("vision-gate")
    state.sessions = SimpleNamespace(
        bank=bank,
        longest_prefix_session=sessions.longest_prefix_session,
        pending_near_prefix_session=sessions.pending_near_prefix_session,
        best_common_prefix_session=sessions.best_common_prefix_session,
    )
    _patch_history(monkeypatch, VISION_HISTORY, _splice(BLUE_DIGEST))
    assert _run_postcommit(state, session)["stored"] is True

    red_keys = vision_bank_key_ids(VISION_HISTORY, _splice(RED_DIGEST))
    blue_keys = vision_bank_key_ids(VISION_HISTORY, _splice(BLUE_DIGEST))
    first_pad = len(TEXT_PREFIX)
    assert red_keys[:first_pad] == blue_keys[:first_pad]
    assert red_keys[first_pad] != blue_keys[first_pad]
    # Neither the raw ids nor the other pixels' view extend the frontier.
    assert sessions.longest_prefix_session(red_keys + [1]) is None
    assert sessions.longest_prefix_session(VISION_HISTORY + [1]) is None
    assert sessions.longest_prefix_session(blue_keys + [1]) is session
    # Shared-prefix identity may adopt the session, but only for the text
    # before the image.
    shared, matched = sessions.best_common_prefix_session(red_keys + [1])
    assert shared is None or matched <= first_pad
    # The admission's live credit stops before the image too, and the bank
    # holds nothing for the other pixels past the text.
    assert oa._live_session_prefix_tokens(state, red_keys + [1], bank) <= first_pad
    assert bank.longest_prefix(red_keys) is None


def test_oversized_vision_projection_also_commits_the_keyed_frontier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gate site 1: the oversized-projection arm (which F39 taught to
    commit the frontier without storing) commits the keyed view too."""

    runtime = _runtime()
    bank = _bank()
    state = _ForegroundState(runtime, bank)
    session = EngineSession("vision-gate")
    _patch_history(monkeypatch, VISION_HISTORY, _splice(BLUE_DIGEST))
    monkeypatch.setattr(
        oa,
        "_estimate_retokenized_snapshot_nbytes",
        lambda *_args, **_kwargs: (1 << 40, 0),
        raising=False,
    )
    # Whatever arm the projection helper takes, the frontier is keyed.
    outcome = _run_postcommit(state, session)
    commit = outcome.get("session_commit")
    assert commit is not None, outcome
    assert commit["committed"] is True
    assert list(session.committed_token_ids) == vision_bank_key_ids(
        VISION_HISTORY, _splice(BLUE_DIGEST)
    )


@pytest.mark.parametrize(
    "switch", ["MTPLX_VISION_SESSION_RESTORE", "MTPLX_VISION_SESSION_CACHE"]
)
def test_a_vision_switch_off_keeps_the_frontier_where_it_was(
    monkeypatch: pytest.MonkeyPatch, switch: str
) -> None:
    runtime = _runtime()
    bank = _bank()
    state = _ForegroundState(runtime, bank)
    session = EngineSession("vision-gate")
    _patch_history(monkeypatch, VISION_HISTORY, _splice(BLUE_DIGEST))
    monkeypatch.setenv(switch, "0")

    outcome = _run_postcommit(state, session)

    assert outcome["session_commit"] == {
        "committed": False,
        "reason": "vision_session_frontier_skip",
        "prefix_len": 0,
    }
    assert list(session.committed_token_ids) == []
    assert oa._session_frontier_ids(VISION_HISTORY, _splice(BLUE_DIGEST)) is None


def test_the_foreground_frontier_is_the_keyed_view_of_an_image_prompt() -> None:
    """The ids both foreground ``session.commit`` sites and the prompt-prefix
    commit hand the session."""

    assert oa._session_frontier_ids(TEXT_HISTORY, None) == TEXT_HISTORY
    keyed = oa._session_frontier_ids(VISION_HISTORY, _splice(BLUE_DIGEST))
    assert keyed == vision_bank_key_ids(VISION_HISTORY, _splice(BLUE_DIGEST))
    assert PAD not in keyed
    # A splice without content identity commits nothing.
    broken = SimpleNamespace(image_digests=[], pad_counts=[], image_pad_token_id=PAD)
    assert oa._session_frontier_ids(VISION_HISTORY, broken) is None


def test_text_only_history_still_commits_frontier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Control: the F39 contract for text sessions is untouched."""

    runtime = _runtime()
    bank = _bank()
    state = _ForegroundState(runtime, bank)
    session = EngineSession("vision-gate")
    _patch_history(monkeypatch, TEXT_HISTORY, None)

    outcome = _run_postcommit(state, session)

    assert outcome["stored"] is True, outcome
    assert outcome["session_commit"] == {
        "committed": True,
        "reason": "committed_retokenized_prefix",
        "prefix_len": len(TEXT_HISTORY),
    }
    assert list(session.committed_token_ids) == [int(t) for t in TEXT_HISTORY]


def test_vision_keying_failure_stays_conservative(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A splice without content identity must neither store nor commit
    (the legacy bypass shape, never a raw-id entry)."""

    runtime = _runtime()
    bank = _bank()
    state = _ForegroundState(runtime, bank)
    session = EngineSession("vision-gate")
    broken = SimpleNamespace(
        image_digests=[], pad_counts=[], image_pad_token_id=PAD
    )
    _patch_history(monkeypatch, VISION_HISTORY, broken)

    outcome = _run_postcommit(state, session)

    assert outcome["stored"] is False
    assert outcome["reason"] == "vision_keying_failed"
    assert len(bank) == 0
    assert list(session.committed_token_ids) == []


def test_the_stable_prefix_is_counted_in_the_ids_the_model_reads() -> None:
    """The encoder counts the stable prefix (the tokens before a transient
    trailing hint) with one placeholder per image; the prompt-prefix commit
    and the prefill's chunk edge read it against the expanded ids."""

    text = [1, 2, PAD, 3, 4, PAD, 5, 6, 7]
    expanded = oa._expand_image_pads(text, image_pad_id=PAD, pad_counts=[4, 2])
    for text_position in range(len(text) + 1):
        served = oa._expanded_position(
            expanded, image_pad_id=PAD, pad_counts=[4, 2], text_position=text_position
        )
        # The same boundary: what precedes it expands to what precedes it.
        assert expanded[:served] == oa._expand_image_pads(
            text[:text_position],
            image_pad_id=PAD,
            pad_counts=[4, 2][: text[:text_position].count(PAD)],
        )
    assert oa._expanded_position(
        expanded, image_pad_id=PAD, pad_counts=[4, 2], text_position=len(text) + 1
    ) is None


@pytest.mark.parametrize("stream", [False, True])
def test_both_foreground_commits_hand_the_session_the_keyed_prompt(monkeypatch, stream):
    """The two foreground ``session.commit`` sites of ``chat_completions``
    (the non-stream one and the streaming one after a stored final commit)
    commit the keyed view of an image prompt plus the generated ids."""

    from fastapi.testclient import TestClient
    from test_server_openai import _fake_streaming_generation, _fake_streaming_session_state

    from mtplx.server.openai import create_app
    from mtplx.vision.splice import is_image_key

    state = _fake_streaming_session_state()
    state.args.enable_thinking = False
    state._vision_spec_cache = SimpleNamespace(image_token_id=999999)
    splice = SimpleNamespace(
        image_pad_token_id=999999, image_digests=[123], pad_counts=[2], total_rows=2,
    )
    materialized: list[list[int]] = []

    def materialize(_state, _images, ids, **_kwargs):
        expanded = [*ids, 999999, 999999]
        materialized.append(expanded)
        return expanded, splice

    monkeypatch.setattr(oa, "_vision_extract_and_flatten", lambda messages: (messages, [object()]))
    monkeypatch.setattr(oa, "_materialize_vision_splice", materialize)
    monkeypatch.setattr(
        oa,
        "_store_generation_final_history_snapshot",
        lambda *_args, **_kwargs: {"stored": True, "mode": "generation_final_exact", "nbytes": 0},
    )
    fake_generation = _fake_streaming_generation("done")

    def generation_with_a_final_state(*args, **kwargs):
        # A final state is what routes the streaming turn to its stored
        # generation-final commit (the store itself is stubbed above).
        return {**fake_generation(*args, **kwargs), "_final_state": SimpleNamespace()}

    monkeypatch.setattr(oa, "_run_generation", generation_with_a_final_state)

    with TestClient(create_app(state)) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={"x-mtplx-session-id": "vision-commit"},
            json={
                "messages": [{"role": "user", "content": "Look."}],
                "max_tokens": 16,
                "stream": stream,
                "enable_thinking": False,
            },
        )
    assert response.status_code == 200, response.text
    committed = list(state.sessions.peek("vision-commit").committed_token_ids)
    served = materialized[0]  # the prompt; a later one is the postcommit's history
    assert committed == [
        *vision_bank_key_ids(served, splice),
        *[ord(char) for char in "done"],
    ]
    assert 999999 not in committed
    assert is_image_key(committed[len(served) - 1])
