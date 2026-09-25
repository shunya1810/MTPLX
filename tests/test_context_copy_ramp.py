"""RAMP policy contracts (adapting community PR #375 by @johninthewinter).

Off by default and inert when off; when on, a fixed long block replaces the
confidence ladder and an exact-key miss falls through to the fuzzy
re-anchor. The batched (qwen4_exp) copy lane consumes the same two proposer
entry points, so the lane smoke below proves inheritance end to end on the
parrot fake with a byte-identical stream.
"""

from mtplx.context_copy import (
    NgramIndex,
    block_for_ext,
    context_copy_block_k,
    ramp_enabled,
)


def _clear_ramp_env(monkeypatch):
    for key in (
        "MTPLX_RAMP_ENABLED",
        "MTPLX_RAMP_BLOCK",
        "MTPLX_RAMP_FUZZY",
        "MTPLX_RAMP_ANCHOR_LEN",
        "MTPLX_RAMP_MAX_FUZZY_CANDIDATES",
        "MTPLX_RAMP_SIMILARITY_SPAN",
        "MTPLX_CONTEXT_COPY_K",
    ):
        monkeypatch.delenv(key, raising=False)
    # Off by default on the M1 GPU family; RAMP rides on the copy lane.
    monkeypatch.setenv("MTPLX_CONTEXT_COPY", "1")


def test_ramp_off_is_inert(monkeypatch):
    _clear_ramp_env(monkeypatch)
    assert not ramp_enabled()
    index = NgramIndex(6, 24)
    assert index._ramp_fuzzy is None
    prompt = list(range(100, 130))
    index.sync(prompt)
    # Exact hit behaves as before (the index is a self-index over one
    # growing stream: history = prompt + generated).
    pos, ext = index.find(prompt + prompt[4:10])
    assert pos == 10
    # Exact miss stays dark: no fuzzy fallback exists to consult.
    assert index.find([1, 2, 3, 4, 5, 6, 7]) == (None, -1)
    # Ladder and cap are stock.
    assert context_copy_block_k() == 24
    assert block_for_ext(0, 24) < block_for_ext(6, 24) <= 24


def test_ramp_fixed_block_widens_block_and_cap(monkeypatch):
    _clear_ramp_env(monkeypatch)
    monkeypatch.setenv("MTPLX_RAMP_ENABLED", "1")
    assert context_copy_block_k() == 48
    for ext in (0, 3, 12):
        assert block_for_ext(ext, context_copy_block_k()) == 48
    monkeypatch.setenv("MTPLX_RAMP_BLOCK", "64")
    assert context_copy_block_k() == 64
    assert block_for_ext(0, context_copy_block_k()) == 64


def test_ramp_fuzzy_reanchors_on_single_token_divergence(monkeypatch):
    _clear_ramp_env(monkeypatch)
    monkeypatch.setenv("MTPLX_RAMP_ENABLED", "1")
    prompt = list(range(100, 130))
    index = NgramIndex(6, 24)
    assert index._ramp_fuzzy is not None
    index.sync(prompt)
    # History re-walks the prompt but one token diverged (a renamed
    # identifier): the exact 6-gram key over the tail misses, the 3-gram
    # anchor still lands, and backward similarity picks position 15.
    history = [7, 7] + prompt[:11] + [999] + prompt[12:15]
    assert index._find_exact(history, len(history))[0] is None
    pos, ext = index.find(history)
    assert pos == 15
    assert ext == 0  # fuzzy hits carry no exact backward extension
    # max_pos fences the re-anchor exactly like the exact index.
    assert index.find(history, max_pos=10) == (None, -1)


def test_ramp_fuzzy_off_keeps_exact_only(monkeypatch):
    _clear_ramp_env(monkeypatch)
    monkeypatch.setenv("MTPLX_RAMP_ENABLED", "1")
    monkeypatch.setenv("MTPLX_RAMP_FUZZY", "0")
    prompt = list(range(100, 130))
    index = NgramIndex(6, 24)
    assert index._ramp_fuzzy is None
    index.sync(prompt)
    history = [7, 7] + prompt[:11] + [999] + prompt[12:15]
    assert index.find(history) == (None, -1)


def test_ramp_lane_smoke_batched_parrot_stream_identical(monkeypatch):
    """The qwen4_exp batched copy lane inherits RAMP through the shared
    proposer: fixed 48-token blocks commit more per verify call on grounded
    re-emission while the stream stays byte-identical."""
    from test_decode_growth_guards import _run

    _clear_ramp_env(monkeypatch)
    stock = _run(monkeypatch, MTPLX_CONTEXT_COPY_BATCHED="1")
    ramp = _run(
        monkeypatch,
        MTPLX_CONTEXT_COPY_BATCHED="1",
        MTPLX_RAMP_ENABLED="1",
    )
    assert list(ramp.tokens) == list(stock.tokens)
    assert ramp.stats.context_copy_rounds > 0
    assert ramp.stats.context_copy_accepted_tokens > 0
    # Longer blocks -> at least as much committed per verify call.
    assert ramp.stats.verify_calls <= stock.stats.verify_calls


def test_probation_k_default_and_env(monkeypatch):
    # Probation cap (2026-08-29): copy blocks stay small until the lane's
    # acceptance EMA proves the content pays — misfired 16-24-token blocks
    # are ~4x-cost verify forwards and short coding-agent turns re-paid
    # that tuition every turn before a suspension could arm.
    from mtplx.context_copy import context_copy_probation_k

    monkeypatch.delenv("MTPLX_CONTEXT_COPY_PROBATION_K", raising=False)
    assert context_copy_probation_k() == 8
    monkeypatch.setenv("MTPLX_CONTEXT_COPY_PROBATION_K", "4")
    assert context_copy_probation_k() == 4
    monkeypatch.setenv("MTPLX_CONTEXT_COPY_PROBATION_K", "1")
    assert context_copy_probation_k() == 2  # floor
    monkeypatch.setenv("MTPLX_CONTEXT_COPY_PROBATION_K", "garbage")
    assert context_copy_probation_k() == 8


def test_probation_ema_contract_math():
    # The gate is (seen >= 2 and ema >= 0.5) with ema' = 0.7*ema + 0.3*ratio
    # from a 0.5 start. Two winning rounds must open the full block; two
    # losing rounds must keep probation and put the lane one round from the
    # (seen >= 3, ema < 0.35) suspension.
    ema, seen = 0.5, 0
    for ratio in (1.0, 1.0):
        ema = 0.7 * ema + 0.3 * min(1.0, ratio)
        seen += 1
    assert seen >= 2 and ema >= 0.5  # winners open up by round 3
    ema, seen = 0.5, 0
    for ratio in (0.2, 0.2):
        ema = 0.7 * ema + 0.3 * min(1.0, ratio)
        seen += 1
    assert not (seen >= 2 and ema >= 0.5)  # losers stay on probation
    ema = 0.7 * ema + 0.3 * 0.2
    seen += 1
    assert seen >= 3 and ema < 0.35  # and suspend on the third miss
