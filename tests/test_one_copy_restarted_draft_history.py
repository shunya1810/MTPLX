"""The turn after a long answer resumes it (the 2026-10-02 Pi session).

Generation bounds the draft head's history: once one answer has appended
``MTPLX_MTP_HISTORY_LIVE_RESET_THRESHOLD`` rows to it (16,384 by default) it
starts a fresh history, which from then on holds only the rows written since.
The one-copy store banks the finished answer as a lease on that live history,
and the next turn's restore asked the history for one row per prefix token but
the last. It had far fewer, so the lease was refused, the cold restore that
followed served an older, much shorter prefix from the SSD and released the
lease, and the turn read everything after it again. In the founder's session a
32,350-token answer (the only answer of 87 requests that restarted its draft
history) was followed by a 51,649-token re-read and a 44 s first token, and
the request log gave no reason: ``cache_miss_reason`` and ``reread.cause``
were empty and ``live_frontier_hit`` read true.

Pinned here:

- a lease whose answer restarted its draft history is taken in place by the
  conversation's next turn with that history as it was, and one the next
  answer decoded past is rewound to it;
- a history trimmed behind its own lease is still refused and the lease kept;
- on the tiny Flash-Next pack, banked through the server's own
  generation-final helpers, the next turn resumes the whole answer, proposes
  the same drafts and decodes what the copying store decodes;
- when the RAM lane refuses a conversation's saved state, the re-read says how
  far the history matched, where it resumed and why, and the frontier receipt
  says that a partial restore did not reach the committed stream.
"""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import numpy as np

from mtplx.cache_state import snapshot_cache
from mtplx.prefill_plan import explain_reread, reread_facts
from mtplx.server import openai
from mtplx.session_bank import SessionBank
from test_one_copy_conversation import (
    RUNTIME,
    _advance,
    _bits,
    _conversation,
    _prompt_lease,
    _qsa,
)
from tests.test_mtp_history_settled import _PROMPT, _generate
from tests.test_qsa_index_block_writes_evaluated import (  # noqa: F401 - tiny_pack is a fixture
    _same_bits,
    tiny_pack,
)


def _answer_after_a_restart(tokens: int, rows: int, *, capacity: int = 96):
    """The live caches an answer hands over after restarting its draft
    history: the trunk holds ``tokens`` tokens, the history the ``rows``
    written since the restart."""

    trunk, _full_history = _conversation(tokens, capacity)
    return trunk, [_qsa(rows, capacity, seed=5)]


def _bank_the_answer(bank: SessionBank, trunk, history, tokens: list[int]):
    """The generation-final put of a one-copy runtime
    (server ``_generation_final_bank_metadata``): a lease that holds the
    draft history by reference."""

    return bank.put(
        runtime=RUNTIME, token_ids=tokens, cache=trunk, logits=mx.zeros((1, 16)),
        hidden=mx.ones((1, 1, 8)), keep_live_ref=True, session_id="s",
        hidden_variant="post_norm", mtp_history_policy="committed",
        mtp_history_snapshot=None, mtp_history_cache_ref=history,
        snapshot_epoch=len(tokens), mtp_snapshot_epoch=len(tokens),
    )


def _bank() -> SessionBank:
    return SessionBank(max_bytes=1 << 30, per_session_max_bytes=1 << 30)


# -- the session bank ------------------------------------------------------------


def test_the_next_turn_takes_the_lease_of_an_answer_that_restarted_its_history(
    monkeypatch,
):
    monkeypatch.delenv("MTPLX_ONE_COPY", raising=False)
    bank = _bank()
    tokens = list(range(72))
    trunk, history = _answer_after_a_restart(72, 9)
    rows_bits = _bits(history[0].kv.keys[..., :9, :])
    entry = _bank_the_answer(bank, trunk, history, tokens)
    assert entry.live_ref_only and entry.lease_mtp_offset == 9
    restored = bank.restore(
        RUNTIME, tokens + [900, 901, 902], mode="reference", session_id="s",
        hidden_variant="post_norm", mtp_history_policy="committed",
    )
    # Before: None (no_snapshot_coverage), the next turn read all 72 again.
    assert restored is not None, bank.last_miss_reason
    assert restored.restore_mode == "reference_lease" and restored.cache_source == "ram"
    assert restored.cache is trunk and trunk[1].offset == 72
    # The draft history comes back as the answer left it.
    assert restored.mtp_history_cache is history and history[0].offset == 9
    assert np.array_equal(_bits(history[0].kv.keys[..., :9, :]), rows_bits)
    assert entry.cache_ref is None and entry.mtp_history_cache_ref is None


def test_a_restarted_history_the_next_answer_ran_past_rewinds_to_its_lease(monkeypatch):
    monkeypatch.delenv("MTPLX_ONE_COPY", raising=False)
    bank = _bank()
    prompt = list(range(40))
    trunk, history = _answer_after_a_restart(40, 9)
    entry = _prompt_lease(bank, trunk, history, prompt)
    assert entry.lease_mtp_offset == 9
    _advance(trunk, history, 6)  # an answer decodes past it and is never committed
    assert trunk[1].offset == 46 and history[0].offset == 15
    restored = bank.restore(RUNTIME, prompt + [900, 901], mode="reference", session_id="s")
    assert restored is not None, bank.last_miss_reason
    assert restored.restore_mode == "reference_lease"
    assert trunk[1].offset == 40 and history[0].offset == 9


def test_a_restarted_history_trimmed_behind_its_lease_is_still_refused(monkeypatch):
    monkeypatch.delenv("MTPLX_ONE_COPY", raising=False)
    bank = _bank()
    tokens = list(range(72))
    trunk, history = _answer_after_a_restart(72, 9)
    entry = _bank_the_answer(bank, trunk, history, tokens)
    history[0].kv.offset = 5  # something trimmed the history behind the lease
    assert bank.restore(RUNTIME, tokens + [900], mode="reference", session_id="s") is None
    assert bank.last_miss_reason == "no_snapshot_coverage"
    assert entry.cache_ref is trunk and entry.mtp_history_cache_ref is history
    assert trunk[1].offset == 72 and history[0].offset == 5


# -- generation on the tiny Flash-Next pack ---------------------------------------

# The bound counts the rows the commits append, not the rows the drafts
# staged (most of them, with the capture commit): a 40-token answer made six
# appends and never reached a bound of 8. A bound of 4 over 60 tokens restarts
# the history (the precondition the test asserts first).
_RESTART = {"MTPLX_MTP_HISTORY_LIVE_RESET_THRESHOLD": "4"}


def _commit_like_the_server(run, prompt: list[int], bank: SessionBank, session_id: str):
    """Bank a finished turn through the server's generation-final helpers."""

    state = SimpleNamespace(
        runtime=run.rt,
        backend_descriptor=SimpleNamespace(
            backend_id="qwen4_exp", mtp_history_policy="committed"
        ),
    )
    final = run.out.final_state
    assert final is not None and final.safe_to_commit
    values = openai._generation_final_bank_values(
        state, final, prompt_ids=prompt,
        final_token_ids=list(prompt) + list(run.out.tokens),
    )
    token_ids = values.pop("token_ids")
    metadata = openai._generation_final_bank_metadata(
        state, final, token_count=len(token_ids)
    )
    entry = bank.put(
        runtime=run.rt, token_ids=token_ids, keep_live_ref=True,
        session_id=session_id, snapshot_epoch=len(token_ids), **values, **metadata,
    )
    assert entry is not None
    return token_ids, entry


def _two_turns(pack, monkeypatch, *, one_copy: bool):
    """An answer long enough to restart its draft history, then the next turn."""

    monkeypatch.setenv("MTPLX_ONE_COPY", "1" if one_copy else "0")
    bank = SessionBank()
    first = _generate(
        pack, monkeypatch, max_tokens=60, env=_RESTART, session_bank=bank,
        session_id="s", session_restore_mode="reference",
        commit_prompt_state_to_bank=True,
    )
    banked, entry = _commit_like_the_server(first, _PROMPT, bank, "s")
    shape = (entry.live_ref_only, entry.lease_mtp_offset, entry.mtp_history_snapshot)
    warm = _generate(
        pack, monkeypatch, max_tokens=24, env=_RESTART,
        prompt=banked + [41, 43, 45], session_bank=bank, session_id="s",
        session_restore_mode="reference",
    )
    return first, banked, shape, warm


def test_the_turn_after_an_answer_that_restarted_its_history_resumes_it(
    tiny_pack, monkeypatch  # noqa: F811 - the imported fixture
):
    first, banked, (lease, lease_rows, _snapshot), warm = _two_turns(
        tiny_pack, monkeypatch, one_copy=True
    )
    assert first.out.stats.mtp_history_live_resets > 0
    # The answer's history holds fewer rows than the conversation has tokens.
    assert lease and lease_rows is not None and lease_rows < len(banked) - 1
    # Before: cached_tokens 0, the whole conversation prefilled again.
    assert warm.out.stats.cached_tokens == len(banked), warm.out.stats.cache_miss_reason
    assert warm.out.stats.session_restore_mode == "reference_lease"

    copy_first, copy_banked, (copy_lease, _rows, copy_snapshot), copy = _two_turns(
        tiny_pack, monkeypatch, one_copy=False
    )
    assert not copy_lease and copy_snapshot is not None
    assert copy_banked == banked
    assert copy.out.stats.cached_tokens == len(banked)
    # Both stores resume the same draft history, so the draft head proposes
    # the same drafts from it: a restored history with other rows, or other
    # positions, would change what is drafted and accepted before it changed
    # any committed token.
    assert warm.out.stats.drafted_tokens > 0
    for key in (
        "drafted_tokens",
        "accepted_drafts",
        "accepted_by_depth",
        "mtp_history_position_base",
    ):
        ours, theirs = getattr(warm.out.stats, key), getattr(copy.out.stats, key)
        assert ours == theirs, (key, ours, theirs)
    # Both stores resume the same history and decode the same turn.
    assert list(warm.out.tokens) == list(copy.out.tokens)
    assert _same_bits(
        warm.out.final_state.final_logits, copy.out.final_state.final_logits
    )


# -- what the receipts say -----------------------------------------------------------


class _OlderPrefixOnDisk:
    """An SSD tier that holds one older, shorter prefix of the conversation."""

    enabled = True
    last_miss_reason = None

    def __init__(self, record):
        self.record = record

    def lookup(self, token_ids, **_identity):
        n = len(self.record.token_ids)
        if tuple(int(token) for token in token_ids[:n]) != self.record.token_ids:
            return None
        return self.record


def test_a_refused_saved_state_explains_the_reread_after_an_older_ssd_restore(
    monkeypatch,
):
    monkeypatch.delenv("MTPLX_ONE_COPY", raising=False)
    bank = _bank()
    older, _history = _conversation(24, 64, seed=30)
    bank.cold_tier = _OlderPrefixOnDisk(
        SimpleNamespace(
            token_ids=tuple(range(24)), cache_snapshot=snapshot_cache(older),
            logits=mx.zeros((1, 16)), hidden=None, mtp_history_snapshot=None,
            metadata={"session_id": "s"}, nbytes=0, restore_s=0.0,
            has_recurrent=True, gdn_boundaries=[], gdn_boundary_loader=None,
        )
    )
    tokens = list(range(40))
    trunk, history = _conversation(40, 64)
    bank.put(
        runtime=RUNTIME, token_ids=tokens, cache=trunk, logits=mx.zeros((1, 16)),
        hidden=None, keep_live_ref=True, session_id="s", mtp_history_cache_ref=history,
        snapshot_epoch=40, mtp_snapshot_epoch=40,
    )
    history[0].kv.offset = 10  # a history the lease cannot be served with
    prompt = tokens + [900, 901]
    restored = bank.restore(RUNTIME, prompt, mode="reference", session_id="s")
    assert restored is not None and restored.cache_source == "ssd"
    assert restored.entry.prefix_len == 24
    # The cold restore released the refused lease of the session.
    assert tuple(tokens) not in bank._entries
    assert bank.eviction_log[-1]["reason"] == "superseded_session_lease"

    explanation = explain_reread(
        reread_facts(
            session_bank=bank, session_id="s", bank_ids=prompt,
            restore_point=24, source="ssd",
        ),
        session_served_before=True,
    )
    # Before: matched 24, no resume limit, "Reading 18 new tokens".
    assert explanation["history_matched_tokens"] == 40
    assert explanation["restore_point_tokens"] == 24
    assert explanation["resume_limit"] == "saved_state"
    assert explanation["resume_limit_at_token"] == 24
    assert explanation["ram_miss_reason"] == "no_snapshot_coverage"
    assert explanation["text"] == (
        "Resuming from the saved state at token 24. "
        "Restored 24 tokens from the SSD cache. Re-reading 18 tokens."
    )
    assert bank.to_dict()["last_ram_miss_reason"] == "no_snapshot_coverage"


def test_a_reread_that_only_extends_the_saved_state_names_no_refusal(monkeypatch):
    monkeypatch.delenv("MTPLX_ONE_COPY", raising=False)
    bank = _bank()
    tokens = list(range(40))
    trunk, history = _conversation(40, 64)
    bank.put(
        runtime=RUNTIME, token_ids=tokens, cache=trunk, logits=mx.zeros((1, 16)),
        hidden=None, keep_live_ref=True, session_id="s", mtp_history_cache_ref=history,
        snapshot_epoch=40, mtp_snapshot_epoch=40,
    )
    prompt = tokens + [900, 901]
    restored = bank.restore(RUNTIME, prompt, mode="reference", session_id="s")
    assert restored is not None and restored.restore_mode == "reference_lease"
    explanation = explain_reread(
        reread_facts(
            session_bank=bank, session_id="s", bank_ids=prompt,
            restore_point=40, source="ram",
        ),
        session_served_before=True,
    )
    assert "ram_miss_reason" not in explanation
    assert explanation["resume_limit"] is None
    assert explanation["text"] == "Reading 2 new tokens."


def test_the_frontier_receipt_says_whether_the_restore_reached_the_committed_stream():
    observability = {
        "live_frontier_result_turn": True,
        "live_frontier_assistant_tool_call_count": 3,
        "live_frontier_tool_result_count": 3,
        "live_frontier_unknown_tool_result_count": 0,
        "committed_reasoning_canonicalization": {"applied": True, "committed_len": 57_560},
    }

    def fields(cached_tokens: int) -> dict:
        return openai._live_frontier_envelope_fields(
            request_observability=observability,
            session_cache_hit=True,
            session_restore_mode="ssd_clone",
            cache_miss_reason=None,
            session_keep_live_ref=True,
            cached_tokens=cached_tokens,
        )

    # The founder's turn: 5,949 of a 57,560-token committed stream restored.
    partial = fields(5_949)
    assert partial["live_frontier_hit"] is True
    assert partial["live_frontier_extended"] is False
    assert partial["live_frontier_committed_len"] == 57_560
    assert fields(57_560)["live_frontier_extended"] is True
