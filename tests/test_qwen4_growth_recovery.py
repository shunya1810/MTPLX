"""A refused copy window preserves the answer; interrupted growth preserves its lease.

These are the remaining 2026-10-01 capacity-review witnesses, exercised on
the real generation loop with tiny random weights. Copy proposals are
controlled to reach the failing boundary, not used as quality measurements.
"""

from __future__ import annotations

import asyncio

import pytest

import mtplx.generation as generation
import mtplx.graphbank as graphbank
from mtplx.session_bank import _cache_kv_offset
from test_one_copy_reserve_resize import SEED, _Session, _same_turn, _walk
from test_one_copy_reserve_resize import lane as lane, tiny as tiny
from test_qwen4_fixed_m4_growth_layerwise import _second_growth_write_raises


@pytest.mark.parametrize("error", [asyncio.CancelledError, KeyboardInterrupt, ValueError])
def test_interrupted_growth_returns_the_prompt_lease(tiny, lane, error):
    session, reference = _Session(tiny, lane), _Session(tiny, lane)
    for item in (session, reference):
        item.opening()
    lane.setenv("MTPLX_COMPILED_VERIFY_GROWTH_RESERVE", "4")
    calls = []
    prompt = list(session.tokens)
    failure = error("interrupted while allocating the second layer")
    with pytest.raises(error) as raised:
        session.turn([], patches=[
            lambda patch: _second_growth_write_raises(patch, failure, calls),
        ])
    assert raised.value is failure
    assert len(calls) == 2
    retry = session.turn([], prompt=prompt, seed=SEED + len(prompt))
    uninterrupted = reference.turn([])
    assert retry.stats.cached_tokens == len(prompt)
    assert retry.stats.session_restore_mode == "reference_lease"
    _same_turn(retry, uninterrupted)


def _copy_turn(session, *, failure=None, max_tokens=100):
    """Force real copy rounds across a dense bank's capacity boundary."""
    walk = _walk()
    cursor = 50
    streamed, reservations, accepted, allocations = [], [], [], []

    def force_copy(patch):
        nonlocal cursor
        real_sample = generation._sample_from_logits
        real_accept = generation._point_mass_block_accept
        real_reserve = graphbank.CompiledVerifyBank.reserve_fixed_m4_window
        real_generate = generation.generate_mtpk

        def stream(*args, **kwargs):
            kwargs["token_callback"] = lambda tokens: streamed.extend(tokens)
            return real_generate(*args, **kwargs)

        def primary(*args, **kwargs):
            nonlocal cursor
            _, distribution = real_sample(*args, **kwargs)
            token = walk[cursor]
            cursor += 1
            return token, distribution

        def accept(logits, block, sampler, rng):
            nonlocal cursor
            real_accept(logits, block, sampler, rng)
            assert list(block) == walk[cursor:cursor + len(block)]
            accepted.append(len(block))
            cursor += len(block)
            return len(block), None

        def reserve(bank, cache, **kwargs):
            if bank._fixed_m4_dispatch is not None and kwargs.get("window_tokens", 4) > 4:
                reservations.append(kwargs["window_tokens"])
                if failure == "refusal":
                    bank.capacity_plan.admit_growth = lambda need: False
            return real_reserve(bank, cache, **kwargs)

        if failure == "allocation":
            _second_growth_write_raises(
                patch, RuntimeError("[metal::malloc] Unable to allocate memory"), allocations,
            )
        patch.setattr(generation, "generate_mtpk", stream)
        patch.setattr(generation, "_sample_from_logits", primary)
        patch.setattr(generation, "_point_mass_block_accept", accept)
        patch.setattr(graphbank.CompiledVerifyBank, "reserve_fixed_m4_window", reserve)

    result = session.turn(walk[45:50], max_tokens=max_tokens, patches=[force_copy])
    assert accepted and reservations, "must exercise actual copy forwards and reservations"
    assert result.tokens == streamed
    if failure == "allocation":
        assert len(allocations) == 2
    return result


@pytest.mark.parametrize("failure", ["refusal", "allocation"])
@pytest.mark.parametrize("compiled_copy", [False, True])
@pytest.mark.parametrize("reserve", [16, 48], ids=["probation-copy", "full-copy"])
def test_refused_copy_keeps_the_stream_and_next_turn(tiny, lane, failure, compiled_copy, reserve):
    lane.setenv("MTPLX_QSA_GATHER_MIN_CONTEXT", "16384")
    lane.setenv("MTPLX_CONTEXT_COPY", "1")
    lane.setenv("MTPLX_FAMILY_CAPTURE_COMMIT", "1")
    lane.setenv("MTPLX_COMPILED_VERIFY_GROWTH_RESERVE", str(reserve))
    lane.setenv("MTPLX_FIXED_M4_COPY_WINDOWS", "1" if compiled_copy else "0")
    session = _Session(tiny, lane)
    session.turn(_walk()[:-2], max_tokens=8)

    ended = _copy_turn(session, failure=failure)
    stop = ended.stats.memory_stop
    assert stop is not None and stop["reason"] == "fixed_m4_growth_refused"
    assert stop["completion_tokens"] == len(ended.tokens)
    assert 0 < len(ended.tokens) < 100
    assert any("memory_stop" in event for event in ended.stats.events)
    if failure == "allocation":
        assert "Unable to allocate" in stop["allocation_error"]
    committed = list(session.tokens)
    lease = session.bank.longest_prefix(committed)
    assert lease is not None and _cache_kv_offset(lease.cache_ref) == len(committed)

    # A normal continuation, not a fresh prompt: every committed token is
    # reused after the refused window and its already-streamed primary.
    after = session.turn([9, 10, 11, 12])
    assert after.stats.session_restore_mode == "reference_lease"
    assert after.stats.cached_tokens == len(committed)
    assert after.stats.memory_stop is None
    assert len(after.tokens) == 24


def test_copy_refusal_commits_the_same_prefix_as_a_normal_stop(tiny, lane):
    lane.setenv("MTPLX_QSA_GATHER_MIN_CONTEXT", "16384")
    lane.setenv("MTPLX_CONTEXT_COPY", "1")
    lane.setenv("MTPLX_FAMILY_CAPTURE_COMMIT", "1")
    lane.setenv("MTPLX_COMPILED_VERIFY_GROWTH_RESERVE", "16")
    refused, reference = _Session(tiny, lane), _Session(tiny, lane)
    for session in (refused, reference):
        session.turn(_walk()[:-2], max_tokens=8)
    ended = _copy_turn(refused, failure="refusal")
    normal = _copy_turn(reference, max_tokens=len(ended.tokens))
    assert normal.stats.memory_stop is None
    _same_turn(ended, normal)
    _same_turn(refused.turn([9, 10, 11, 12]), reference.turn([9, 10, 11, 12]))


def test_partial_handback_is_never_published(tiny, lane):
    """A second failure must not make a partially converted lease reusable."""
    session = _Session(tiny, lane)
    session.opening()
    prompt = list(session.tokens)
    lane.setenv("MTPLX_COMPILED_VERIFY_GROWTH_RESERVE", "4")
    real = graphbank.TensorOffsetQSACache.demote
    calls = []
    failure = asyncio.CancelledError("cancel during growth")

    def fail_second(entry, **kwargs):
        calls.append(entry)
        if len(calls) == 2:
            raise RuntimeError("GPU unavailable while returning cache")
        return real(entry, **kwargs)

    def inject(patch):
        _second_growth_write_raises(patch, failure)
        patch.setattr(graphbank.TensorOffsetQSACache, "demote", fail_second)

    with pytest.raises(asyncio.CancelledError) as caught:
        session.turn([], patches=[inject])
    assert caught.value is failure
    assert len(calls) == 2
    lease = session.bank.longest_prefix(prompt)
    assert lease is None or _cache_kv_offset(lease.cache_ref) is None, (
        "partially converted cache looks reusable"
    )


class _Cancelled(Exception):
    """The client's cancel, raised out of the token callback."""


def test_a_cancel_whose_hand_back_fails_never_publishes_a_partial_lease(tiny, lane):
    """The client-cancel twin of the case above (one_copy.hand_back_on_raise).

    Converted in place, the layers before a failed demotion became stock
    containers while the rest stayed adapters, and the session bank read the
    stock layers' offset as the whole lease's: a half-converted conversation
    looked reusable.
    """
    session = _Session(tiny, lane)
    session.opening()
    suffix = [9, 10, 11, 12]
    prompt = list(session.tokens) + suffix
    real_demote = graphbank.TensorOffsetQSACache.demote
    real_generate = generation.generate_mtpk
    calls = []

    def fail_second(entry, **kwargs):
        calls.append(entry)
        if len(calls) == 2:
            raise RuntimeError("GPU unavailable while returning cache")
        return real_demote(entry, **kwargs)

    def cancel_after_six_tokens(*args, **kwargs):
        streamed = []

        def callback(tokens):
            streamed.extend(tokens)
            if len(streamed) >= 6:
                raise _Cancelled()

        kwargs["token_callback"] = callback
        return real_generate(*args, **kwargs)

    def inject(patch):
        patch.setattr(generation, "generate_mtpk", cancel_after_six_tokens)
        patch.setattr(graphbank.TensorOffsetQSACache, "demote", fail_second)

    with pytest.raises(_Cancelled):
        session.turn(suffix, patches=[inject])
    assert len(calls) == 2
    [lease] = [
        entry for key, entry in session.bank._entries.items() if len(key) == len(prompt)
    ]
    assert _cache_kv_offset(lease.cache_ref) is None, "partially converted cache looks reusable"
