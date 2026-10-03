"""An answer never fails mid-stream because the fast path's bank cannot grow.

On 2026-09-29 two answers of a Pi session died mid-stream with "insufficient
memory": the fixed-M4 verifier's bank had to grow, the growth admission
refused, and the MemoryError reached the client after a minute of streamed
text. The bank now grows at the top of each decode round, before the round
samples, drafts or writes anything, with exactly the reservation the round's
verify made before (so every forward sees the capacity it saw before). If
the memory is not there even after the session bank gave way, the answer
ends at that boundary: finish_reason "length", ``stats.memory_stop`` saying
why, every committed token whole.
"""

from __future__ import annotations

import pytest

import mtplx.generation as generation
import mtplx.graphbank as graphbank
from mtplx.session_bank import _cache_kv_offset
from test_qwen4_fixed_m4_capacity_bucket import NATIVE, SEED, lane, pack  # noqa: F401

MAX_TOKENS = 320


def _prompt(length: int) -> list[int]:
    return [(i * i * 31 + 7 * i + 3) % 127 for i in range(length)]


def _generate(pack, lane, *, prompt, gather, round_reserve=True, grants=None):
    """One answer on the compiled fixed-M4 lane, its bank growing mid-answer.

    ``grants`` is how many growth admissions succeed before every later one
    is refused (None: all succeed): each growth asks for its bill, then for
    each layer's new banks. Returns the result and the bytes asked.
    """

    from mtplx.qwen4_fixed_verify import install_qwen4_fixed_verify_route

    smoke, model = pack
    lane.setenv("MTPLX_COMPILED_VERIFY", "1")
    lane.setenv("MTPLX_QSA_GATHER", "1")
    lane.setenv("MTPLX_QSA_GATHER_MIN_CONTEXT", "16" if gather else "16384")
    lane.setenv("MTPLX_QWEN4_FIXED_M4_CAPACITY_BUCKET", "256")
    lane.setenv("MTPLX_COMPILED_VERIFY_GROWTH_RESERVE", "16")
    rt = smoke._tiny_runtime(model)
    install_qwen4_fixed_verify_route(rt)
    asked: list[int] = []
    with lane.context() as patch:
        if not round_reserve:
            patch.setattr(
                graphbank.CompiledVerifyBank,
                "reserve_fixed_m4_round",
                lambda self, cache, **kwargs: None,
            )

        def admit(_rt, need, **_kwargs):
            asked.append(int(need))
            return grants is None or len(asked) <= grants

        patch.setattr(generation, "_qwen4_fixed_m4_layer_fits", admit)
        result = generation.generate_mtpk(
            rt, list(prompt), max_tokens=MAX_TOKENS, sampler=NATIVE, draft_sampler=NATIVE,
            speculative_depth=3, seed=SEED, mtp_cache_policy="persistent",
            mtp_history_policy="committed", verify_strategy="batched",
            stop_token_ids=set(), capture_final_state=True,
        )
    return result, asked


@pytest.mark.parametrize("gather", [False, True], ids=["dense", "gather"])
def test_growing_at_the_top_of_the_round_changes_no_token(pack, lane, gather):
    prompt = _prompt(1000)
    before, asked_before = _generate(pack, lane, prompt=prompt, gather=gather, round_reserve=False)
    after, asked_after = _generate(pack, lane, prompt=prompt, gather=gather)
    assert after.tokens == before.tokens
    # The bank grew during the answer, at the same widths.
    assert asked_after and asked_after == asked_before
    assert after.stats.memory_stop is None
    assert after.stats.graphbank["compiled_verify"]["fixed_m4"]["installed"]


@pytest.mark.parametrize("gather", [False, True], ids=["dense", "gather"])
def test_a_copy_window_is_reserved_ahead_only_where_capacity_changes_no_value(
    pack, lane, gather
):
    """The copy lane's block rounds (up to 25 rows at the default K) reserve
    their own window inside the round. On the rows-gather lane the round's
    reservation covers that window too, so a refusal still comes between
    rounds; on the dense lane, where the capacity is the attention's key
    length, the round reserves only what its verify reserved before."""

    from mtplx.qwen4_fixed_verify import install_qwen4_fixed_verify_route

    import mlx.core as mx

    smoke, model = pack
    lane.setenv("MTPLX_QSA_GATHER", "1")
    lane.setenv("MTPLX_QSA_GATHER_MIN_CONTEXT", "16" if gather else "16384")
    lane.setenv("MTPLX_QWEN4_FIXED_M4_CAPACITY_BUCKET", "256")
    lane.setenv("MTPLX_COMPILED_VERIFY_GROWTH_RESERVE", "16")
    rt = smoke._tiny_runtime(model)
    install_qwen4_fixed_verify_route(rt)
    prompt = _prompt(1000)
    cache = model.make_cache()
    rt.forward_ar(mx.array([prompt]), cache=cache, return_hidden=True)
    plan = graphbank.FixedM4CapacityPlan.for_request(MAX_TOKENS, runtime=rt)
    bank = graphbank.CompiledVerifyBank(
        rt, max_verify_len=4, request_max_tokens=MAX_TOKENS, capacity_plan=plan
    )
    bank.install_fixed_m4(cache, prompt_ids=prompt, hidden_variant=None)
    qsa = next(e for e in cache if isinstance(e, graphbank.TensorOffsetQSACache))
    capacity = int(qsa.dense_capacity)
    # Four rows still fit; a 25-row window does not.
    committed = capacity - 1000 - 10
    bank.reserve_fixed_m4_round(cache, committed_count=committed, copy_window=25)
    grew = int(qsa.dense_capacity) > capacity
    assert grew is gather


@pytest.mark.parametrize("gather", [False, True], ids=["dense", "gather"])
@pytest.mark.parametrize("grants", [0, 1])
def test_a_refused_growth_ends_the_answer_between_rounds(pack, lane, gather, grants):
    prompt = _prompt(1000)
    full, _ = _generate(pack, lane, prompt=prompt, gather=gather)
    ended, asked = _generate(pack, lane, prompt=prompt, gather=gather, grants=grants)
    assert asked
    assert ended.finish_reason == "length"
    stop = ended.stats.memory_stop
    assert stop is not None and stop["reason"] == "fixed_m4_growth_refused"
    assert stop["completion_tokens"] == len(ended.tokens)
    assert 0 < len(ended.tokens) < MAX_TOKENS
    # Every token it streamed is the token the unrefused answer streamed.
    assert ended.tokens == full.tokens[: len(ended.tokens)]
    assert any("memory_stop" in event for event in ended.stats.events)
    # Its final state is whole: committed, it holds the prompt and every
    # token; otherwise the bank is told not to take it.
    final = ended.final_state
    assert final is not None
    if final.safe_to_commit:
        assert _cache_kv_offset(final.final_trunk_cache) == len(prompt) + len(ended.tokens)


def test_the_client_is_told_in_plain_words_and_only_when_it_happened():
    import mtplx.server.openai as srv

    stop = {
        "reason": "fixed_m4_growth_refused",
        "capacity_tokens": 131_072,
        "required_tokens": 131_076,
        "requested_rows": 147_456,
        "completion_tokens": 18_204,
    }
    public = srv._public_mtplx_stats({"stats": {"memory_stop": stop}})
    assert public["memory_stop"]["completion_tokens"] == 18_204
    message = public["memory_stop"]["message"]
    assert "18,204 tokens" in message and "memory" in message
    assert "memory_stop" not in srv._public_mtplx_stats({"stats": {"memory_stop": None}})
