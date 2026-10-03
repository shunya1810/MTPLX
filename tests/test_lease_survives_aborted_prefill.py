"""A conversation's only copy survives a prefill that stops part-way.

The one-copy store keeps a conversation as a lease on its live cache, with no
snapshot beside it, and a warm prefill takes that lease. A prefill that stops
early (the client cancels while a long prompt is re-read, or a postcommit
yields to the next request, which Pi's agent loop does between most turns)
used to drop the lease with the request: the next turn re-read the whole
conversation. The copying store hid this, because the snapshot it kept beside
every lease survived. Now the lease goes back to the bank at the point it was
restored at, with the recurrent state of that point, and the next request
restores there and writes exactly the tokens it would have written.
"""

from __future__ import annotations

import pytest

import mtplx.generation as generation
from mtplx.generation import PostcommitAbort
from mtplx.session_bank import SessionBank, _lease_advance
from test_one_copy_conversation import _one_turn
from test_qwen4_fixed_m4_capacity_bucket import NATIVE, SEED, lane, pack  # noqa: F401


def _runtime(pack, lane):
    from mtplx.qwen4_fixed_verify import install_qwen4_fixed_verify_route

    smoke, model = pack
    lane.setenv("MTPLX_ONE_COPY", "1")
    lane.setenv("MTPLX_COMPILED_VERIFY", "1")
    lane.setenv("MTPLX_QSA_GATHER", "1")
    lane.setenv("MTPLX_QSA_GATHER_MIN_CONTEXT", "16")
    lane.setenv("MTPLX_QWEN4_FIXED_M4_CAPACITY_BUCKET", "512")
    lane.setenv("MTPLX_SUSTAINED_PREFILL", "1")
    lane.setenv("MTPLX_PREFILL_CHUNK_SIZE", "32")
    # Chunk even short suffixes, so a stop can land between chunks.
    lane.setenv("MTPLX_SMALL_SUFFIX_FUSED_MAX", "0")
    rt = smoke._tiny_runtime(model)
    install_qwen4_fixed_verify_route(rt)
    return rt, generation._resolve_runtime_base_hidden_variant(rt, None)


def _abort_after(calls: int):
    seen = {"calls": 0}

    def check() -> bool:
        seen["calls"] += 1
        return seen["calls"] >= calls

    return check, seen


def _aborted_turn(rt, bank, prompt, *, abort_at: int):
    check, seen = _abort_after(abort_at)
    with pytest.raises(PostcommitAbort):
        generation.generate_mtpk(
            rt, list(prompt), max_tokens=24, sampler=NATIVE, draft_sampler=NATIVE,
            speculative_depth=3, seed=SEED + 1, mtp_cache_policy="persistent",
            mtp_history_policy="committed", verify_strategy="batched",
            stop_token_ids=set(), capture_final_state=True, session_bank=bank,
            session_id="s", session_restore_mode="reference",
            commit_prompt_state_to_bank=True, abort_check=check,
        )
    return seen["calls"]


def _prompt(length: int, salt: int = 0) -> list[int]:
    return [(7 * i + 3 + salt) % 128 for i in range(length)]


def test_a_cancelled_warm_prefill_gives_the_conversation_back(pack, lane):
    rt, hidden_variant = _runtime(pack, lane)
    bank = SessionBank(max_bytes=1 << 32, per_session_max_bytes=1 << 32)
    first_tokens, first_entry, _ = _one_turn(rt, bank, _prompt(48), 0, True, hidden_variant)
    conversation = list(first_entry.token_ids)
    prompt = conversation + _prompt(200, salt=5)

    # The client cancels while the 200 new tokens are read, two chunks in.
    _aborted_turn(rt, bank, prompt, abort_at=14)
    returned = bank.longest_prefix(prompt)
    assert returned is not None and returned.token_ids == tuple(conversation)
    assert returned.live_ref_only and returned.cache_ref is not None
    assert _lease_advance(returned) > 0  # the prefill wrote past it before it stopped
    assert bank.restore_plan(prompt)["reuse_tokens"] == len(conversation)

    # The retry restores the whole conversation and writes what an
    # uninterrupted turn writes.
    retried, _entry, result = _one_turn(rt, bank, prompt, 1, True, hidden_variant)
    assert result.stats.cached_tokens == len(conversation)
    assert result.stats.session_restore_mode == "reference_lease"

    reference_bank = SessionBank(max_bytes=1 << 32, per_session_max_bytes=1 << 32)
    _one_turn(rt, reference_bank, _prompt(48), 0, True, hidden_variant)
    expected, _entry, _result = _one_turn(rt, reference_bank, prompt, 1, True, hidden_variant)
    assert retried == expected


def test_a_yielding_boundary_restore_gives_the_conversation_back(pack, lane):
    """The postcommit's shape: the next history diverges inside the last
    answer, so the restore rewinds the lease to the prompt's anchor and reads
    the rest; the next request preempts it part-way."""

    rt, hidden_variant = _runtime(pack, lane)
    # The block lane at this tiny scale (production: 256-token blocks, 4,096
    # matched tokens at least).
    lane.setenv("MTPLX_SESSION_PREFIX_BLOCK_SIZE", "16")
    lane.setenv("MTPLX_SESSION_BLOCK_PREFIX_MIN_MATCH_TOKENS", "16")
    bank = SessionBank(max_bytes=1 << 32, per_session_max_bytes=1 << 32)
    first_prompt = _prompt(48)
    _tokens, final_entry, _result = _one_turn(rt, bank, first_prompt, 0, True, hidden_variant)
    served = list(final_entry.token_ids)
    # The client's rendering of the answer differs from the served tokens
    # after its first few tokens.
    history = served[: len(first_prompt) + 3] + _prompt(160, salt=9)

    _aborted_turn(rt, bank, history, abort_at=14)
    returned = bank.longest_prefix(history)
    assert returned is not None
    assert returned.token_ids == tuple(first_prompt)
    assert returned.cache_ref is not None
    assert _lease_advance(returned) > 0
    assert bank.restore_plan(history)["reuse_tokens"] == len(first_prompt)

    retried, _entry, result = _one_turn(rt, bank, history, 1, True, hidden_variant)
    assert result.stats.cached_tokens == len(first_prompt)

    reference_bank = SessionBank(max_bytes=1 << 32, per_session_max_bytes=1 << 32)
    _one_turn(rt, reference_bank, first_prompt, 0, True, hidden_variant)
    expected, _entry, _result = _one_turn(rt, reference_bank, history, 1, True, hidden_variant)
    assert retried == expected


def test_a_returned_lease_never_serves_its_own_length(pack, lane):
    """It has no logits for its last position, so only prompts that extend
    it restore from it."""

    rt, hidden_variant = _runtime(pack, lane)
    bank = SessionBank(max_bytes=1 << 32, per_session_max_bytes=1 << 32)
    _first, first_entry, _ = _one_turn(rt, bank, _prompt(48), 0, True, hidden_variant)
    conversation = list(first_entry.token_ids)
    _aborted_turn(rt, bank, conversation + _prompt(200, salt=5), abort_at=14)
    returned = bank.longest_prefix(conversation + [1])
    assert returned is not None and returned.extension_only
    assert bank.restore_plan(conversation)["reuse_tokens"] < len(conversation)
    assert bank.restore(rt, conversation, mode="reference", session_id="s") is None
    assert returned.cache_ref is not None
