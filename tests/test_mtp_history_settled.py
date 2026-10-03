"""The draft history holds at most one round's writes unevaluated, on every path (#544).

The committed draft history is written twice a round: the round's first draft
stages its primary row, and the commit appends the verified rows. With
``MTPLX_LAZY_MTP_HISTORY_APPEND`` the append is left for the next draft
forward, which reads the history and evaluates it together with itself. Many
supported paths leave writes that no forward reads: a draft on a fresh
per-step cache, a proposal served from the correction cache (its draft logits
are never read), a rejected proposal whose correction the capture commit
defers (the commit then appends nothing), a target-prefix draft substitution,
a context-copy round, and the final pending-token commit, whose state rebase
installs a new history with a lazy prefill append. Stacked writes keep every
round's rows alive, and on a QSA draft head the Metal shared event of each
round that computed them; the session bank snapshots whatever the generation
hands over. ``generate_mtpk`` therefore schedules the history at every commit,
empty or not, and once more when it ends.

Pinned here on the tiny Flash-Next pack, path by path, with the capture
commit on as the Flash-Next server runs it (and off too for rejected
correction-cache proposals):
- when an append starts, the history holds no unevaluated work;
- when a round's first draft starts, the history holds nothing beyond what
  the latest append left, and nothing at all after a round that appended
  nothing;
- the history the generation hands over holds nothing unevaluated;
- scheduling changes no bit: tokens, final logits and the handed-over history
  match eager appends, and a bank snapshot taken while the prompt's history
  append was still lazy keeps its bits through the settle and the writes
  after it; so does a warm turn that restores a banked turn and continues it.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import mlx.core as mx
import pytest

import mtplx.context_copy as context_copy
import mtplx.generation as generation
import mtplx.graphbank as graphbank
from mtplx.cache_state import snapshot_cache
from mtplx.session_bank import SessionBank
from tests.test_qsa_index_block_writes_evaluated import (  # noqa: F401 - tiny_pack is a fixture
    _pending_ops,
    _qsa_entries,
    _qsa_leaves,
    _same_bits,
    tiny_pack,
)

_PROMPT = [3, 5, 7, 9, 11, 13] + list(range(20, 40))
# Every token of the tiny vocabulary is followed by its successor, so the
# prompt correction cache serves every depth-one proposal.
_SUCCESSORS = [token % 128 for token in range(130)]
_APPEND = generation._append_mtp_history
_RESET = {
    "MTPLX_COMPILED_VERIFY",
    "MTPLX_STATE_REBASE_EVERY",
    "MTPLX_FAMILY_CAPTURE_COMMIT",
    "MTPLX_SUSTAINED_PREFILL",
    "MTPLX_MTP_HISTORY_LIVE_RESET_THRESHOLD",
    "MTPLX_MTP_HISTORY_MATERIALIZE_EVERY",
    "MTPLX_CONTEXT_COPY_TARGET_PREFIX",
    "MTPLX_LAZY_MTP_HISTORY_APPEND",
    "MTPLX_QSA_MTP_PRECOMPUTE",
}


def _backlog(cache) -> int:
    """Unevaluated operations anywhere in a draft history cache."""

    return sum(_pending_ops(leaf) for leaf in _qsa_leaves(_qsa_entries(cache)))


def _generate(
    tiny_pack,
    monkeypatch,
    *,
    max_tokens: int,
    prompt: list[int] = _PROMPT,
    env: dict[str, str] | None = None,
    lazy: bool = True,
    greedy: bool = False,
    capture_commit: bool = True,
    copy_streak: bool = False,
    **kwargs,
):
    """One generation, recording the history's backlog as it goes.

    ``appends``: the backlog as each decode-round append starts (a prefill
    append of the prompt or of a restored suffix is not one; the suffix's are
    evaluated as they are made). ``rounds``: at each
    round's first draft, the backlog of the history in use and what the
    appends to it since the previous round's first draft left (zero when
    there were none, as after a commit that appended nothing or a live reset).
    ``history``: the history the generation hands over.
    """

    from mtplx.qwen4_fixed_verify import install_qwen4_fixed_verify_route
    from mtplx.sampling import SamplerConfig

    smoke, model = tiny_pack
    for name in tuple(os.environ):
        if name.startswith("MTPLX_QWEN4_") or name in _RESET:
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MTPLX_COMPILED_VERIFY", "1")
    monkeypatch.setenv("MTPLX_FAMILY_CAPTURE_COMMIT", "1" if capture_commit else "0")
    monkeypatch.setenv("MTPLX_LAZY_MTP_HISTORY_APPEND", "1" if lazy else "0")
    for name, value in (env or {}).items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(graphbank, "_compiled_verify_bits_gate_ok", lambda _rt: True)
    if copy_streak:
        # Propose the prompt's first block every round: the tiny random model
        # rejects it, so the lane runs its probation rounds back to back.
        monkeypatch.setattr(
            context_copy.NgramIndex, "find", lambda self, history, max_pos=None: (0, 64)
        )
    sampler = SamplerConfig(temperature=0.0) if greedy else SamplerConfig(
        temperature=0.6, top_p=0.95, top_k=20
    )
    options = dict(
        max_tokens=max_tokens,
        sampler=sampler,
        draft_sampler=sampler,
        speculative_depth=3,
        seed=1234,
        mtp_cache_policy="persistent",
        mtp_history_policy="committed",
        verify_strategy="batched",
        stop_token_ids=set(),
        capture_final_state=True,
    )
    options.update(kwargs)
    # A persistent draft cache is the committed history itself; a fresh one
    # is not, and the history is then the cache the appends write.
    drafts_on_history = options["mtp_cache_policy"] == "persistent"
    appends: list[int] = []
    rounds: list[tuple[int, int]] = []
    current: list = [None]
    left: dict[int, int] = {}

    def recording(rt, mtp_cache, *args, **append_kwargs):
        if append_kwargs.get("phase") == "ar_decode":
            appends.append(_backlog(mtp_cache))
        current[0] = mtp_cache
        elapsed = _APPEND(rt, mtp_cache, *args, **append_kwargs)
        left[id(mtp_cache)] = _backlog(mtp_cache)
        return elapsed

    monkeypatch.setattr(generation, "_append_mtp_history", recording)
    rt = smoke._tiny_runtime(model)
    install_qwen4_fixed_verify_route(rt)
    draft = rt.draft_mtp

    def observed_draft(*args, mtp_depth=None, **draft_kwargs):
        if drafts_on_history:
            current[0] = draft_kwargs.get("mtp_cache")
        if mtp_depth == 1 and current[0] is not None:
            rounds.append((_backlog(current[0]), left.get(id(current[0]), 0)))
            left.clear()
        return draft(*args, mtp_depth=mtp_depth, **draft_kwargs)

    rt.draft_mtp = observed_draft
    out = generation.generate_mtpk(rt, list(prompt), **options)
    assert len(out.tokens) == max_tokens
    history = (
        out.final_state.final_committed_mtp_cache
        if out.final_state is not None
        else current[0]
    )
    return SimpleNamespace(out=out, appends=appends, rounds=rounds, history=history, rt=rt)


def _assert_bounded(run) -> None:
    """Nothing stacks: every append and every round starts from what the bound allows."""

    assert max(run.appends) == 0, run.appends
    assert all(backlog <= left for backlog, left in run.rounds), run.rounds
    assert _backlog(run.history) == 0


def _assert_same_decode(lazy, eager) -> None:
    assert list(lazy.out.tokens) == list(eager.out.tokens)
    assert _same_bits(lazy.out.final_state.final_logits, eager.out.final_state.final_logits)
    ours, theirs = _qsa_leaves(_qsa_entries(lazy.history)), _qsa_leaves(_qsa_entries(eager.history))
    assert ours and len(ours) == len(theirs)
    assert all(_same_bits(a, b) for a, b in zip(ours, theirs))


def _both(tiny_pack, monkeypatch, **kwargs):
    lazy = _generate(tiny_pack, monkeypatch, **kwargs)
    eager = _generate(tiny_pack, monkeypatch, lazy=False, **kwargs)
    _assert_same_decode(lazy, eager)
    return lazy


def test_drafts_on_fresh_caches_never_leave_the_committed_history_pending(
    tiny_pack, monkeypatch
):
    run = _both(tiny_pack, monkeypatch, max_tokens=40, mtp_cache_policy="fresh")
    # Every draft ran on a cache of its own, so no draft read the history.
    assert run.out.stats.drafted_tokens > 0 and len(run.appends) > 3
    _assert_bounded(run)


def test_correction_cache_proposals_never_leave_the_history_pending(tiny_pack, monkeypatch):
    run = _both(
        tiny_pack,
        monkeypatch,
        max_tokens=24,
        prompt=_SUCCESSORS,
        speculative_depth=1,
        prompt_correction_cache=True,
        prompt_correction_cache_min_depth=1,
    )
    assert run.out.stats.online_correction_cache["hits"] > 3, run.out.stats.online_correction_cache
    assert len(run.rounds) > 3
    _assert_bounded(run)


@pytest.mark.parametrize("capture", [True, False], ids=["capture", "no-capture"])
@pytest.mark.parametrize("replay", [False, True], ids=["direct", "qsa-replay"])
@pytest.mark.parametrize("greedy", [False, True], ids=["sampled", "greedy"])
def test_rejected_correction_cache_proposals_leave_nothing_stacked(
    tiny_pack, monkeypatch, greedy, replay, capture
):
    # The tiny random model rejects the served successors. The capture commit
    # leaves each correction to the next round, so those rounds append
    # nothing to the history their draft staged a row in; without it the
    # commit appends the correction.
    run = _generate(
        tiny_pack,
        monkeypatch,
        max_tokens=24,
        prompt=_SUCCESSORS,
        greedy=greedy,
        capture_commit=capture,
        env={"MTPLX_QSA_MTP_PRECOMPUTE": "1"} if replay else None,
        speculative_depth=1,
        prompt_correction_cache=True,
        prompt_correction_cache_min_depth=1,
    )
    rejected_hits = [
        draft
        for event in run.out.stats.events
        for draft in (event.get("drafts") or () if isinstance(event, dict) else ())
        if (draft.get("online_correction_cache") or {}).get("hit")
        and draft.get("accepted") is False
    ]
    assert len(rejected_hits) > 3
    if capture:
        # Rounds after a commit that appended nothing: the history must hold
        # nothing at all when the next draft starts.
        assert sum(1 for _now, left in run.rounds if left == 0) > 3, run.rounds
    _assert_bounded(run)


def test_a_live_reset_leaves_no_history_pending(tiny_pack, monkeypatch):
    run = _both(
        tiny_pack,
        monkeypatch,
        max_tokens=60,
        env={"MTPLX_MTP_HISTORY_LIVE_RESET_THRESHOLD": "8"},
    )
    assert run.out.stats.mtp_history_live_resets > 0
    _assert_bounded(run)


def test_a_rebase_inside_the_final_commit_hands_over_no_pending_history(
    tiny_pack, monkeypatch
):
    # The final pending-token commit rebases the state, and a prefill without
    # the sustained path gives the new history a lazy append of the prompt.
    run = _both(
        tiny_pack,
        monkeypatch,
        max_tokens=1,
        env={"MTPLX_SUSTAINED_PREFILL": "0", "MTPLX_STATE_REBASE_EVERY": "1"},
    )
    assert run.out.stats.state_rebase_events > 0
    assert run.out.final_state.safe_to_commit
    assert _backlog(run.history) == 0


def test_target_prefix_draft_substitution_leaves_no_history_pending(tiny_pack, monkeypatch):
    run = _generate(
        tiny_pack,
        monkeypatch,
        max_tokens=24,
        copy_streak=True,
        verify_strategy="target_prefix",
        speculative_depth=1,
        env={"MTPLX_CONTEXT_COPY_TARGET_PREFIX": "1"},
    )
    # The copy match served the depth-one draft; the draft head never ran.
    assert run.out.stats.context_copy_drafted_tokens > 0
    _assert_bounded(run)


def test_a_generation_without_a_final_state_leaves_no_pending_history(tiny_pack, monkeypatch):
    run = _generate(
        tiny_pack,
        monkeypatch,
        max_tokens=40,
        copy_streak=True,
        capture_final_state=False,
    )
    assert run.out.final_state is None and run.history is not None
    _assert_bounded(run)


def _saved_arrays(entry) -> list[mx.array]:
    return generation._tree_mx_arrays(
        [entry.cache_snapshot, entry.mtp_history_snapshot, entry.logits, entry.hidden]
    )


def _bank_then_restore(tiny_pack, monkeypatch, *, lazy: bool, one_copy: bool = False):
    monkeypatch.setenv("MTPLX_ONE_COPY", "1" if one_copy else "0")
    bank = SessionBank()
    first = _generate(
        tiny_pack,
        monkeypatch,
        max_tokens=40,
        lazy=lazy,
        copy_streak=True,
        session_bank=bank,
        session_id="snapshot-before-settle",
        commit_prompt_state_to_bank=True,
    )
    entries = list(bank._entries.values())
    if one_copy:
        # The one-copy store banks the prompt as a lease on the live history
        # (mtplx/one_copy.py); the warm turn rewinds it to the prompt's end.
        assert len(entries) == 1 and entries[0].live_ref_only
        assert entries[0].mtp_history_cache_ref is not None
        assert entries[0].mtp_history_snapshot is None
        warm = _generate(
            tiny_pack,
            monkeypatch,
            max_tokens=24,
            lazy=lazy,
            session_bank=bank,
            session_id="snapshot-before-settle",
        )
        return first, [], warm
    assert len(entries) == 1 and entries[0].mtp_history_snapshot is not None
    # Read after the generation settled the history and wrote past it.
    saved = _saved_arrays(entries[0])
    mx.eval(saved)
    warm = _generate(
        tiny_pack,
        monkeypatch,
        max_tokens=24,
        lazy=lazy,
        session_bank=bank,
        session_id="snapshot-before-settle",
    )
    return first, saved, warm


def test_a_bank_snapshot_taken_before_the_settle_keeps_its_bits_and_restores(
    tiny_pack, monkeypatch
):
    lazy_first, lazy_saved, lazy_warm = _bank_then_restore(tiny_pack, monkeypatch, lazy=True)
    eager_first, eager_saved, eager_warm = _bank_then_restore(tiny_pack, monkeypatch, lazy=False)
    assert lazy_warm.out.stats.cached_tokens > 0, "the warm turn restored from the bank"
    _assert_bounded(lazy_warm)
    assert list(lazy_first.out.tokens) == list(eager_first.out.tokens)
    assert len(lazy_saved) == len(eager_saved)
    assert all(_same_bits(a, b) for a, b in zip(lazy_saved, eager_saved))
    _assert_same_decode(lazy_warm, eager_warm)


def test_a_prompt_lease_taken_before_the_settle_restores_like_the_snapshot(
    tiny_pack, monkeypatch
):
    """The one-copy store's prompt lease serves the warm turn the snapshot served."""

    lazy_first, _, lazy_warm = _bank_then_restore(
        tiny_pack, monkeypatch, lazy=True, one_copy=True
    )
    eager_first, _, eager_warm = _bank_then_restore(
        tiny_pack, monkeypatch, lazy=False, one_copy=True
    )
    _copy_first, _saved, copy_warm = _bank_then_restore(tiny_pack, monkeypatch, lazy=True)
    assert lazy_warm.out.stats.cached_tokens == copy_warm.out.stats.cached_tokens > 0
    _assert_bounded(lazy_warm)
    assert list(lazy_first.out.tokens) == list(eager_first.out.tokens)
    _assert_same_decode(lazy_warm, eager_warm)
    # Across the two stores the buffers differ in size (the lease keeps its
    # capacity), so the draft history is compared on its valid rows.
    assert list(lazy_warm.out.tokens) == list(copy_warm.out.tokens)
    assert _same_bits(
        lazy_warm.out.final_state.final_logits, copy_warm.out.final_state.final_logits
    )
    ours = [leaf for entry in _qsa_entries(lazy_warm.history) for leaf in entry.state]
    theirs = [leaf for entry in _qsa_entries(copy_warm.history) for leaf in entry.state]
    assert ours and len(ours) == len(theirs)
    assert all(_same_bits(a, b) for a, b in zip(ours, theirs))


def _postcommit(bank, run, prompt, session_id) -> list[int]:
    """Bank a finished turn the way the server's generation-final commit does."""

    final = run.out.final_state
    assert final.safe_to_commit
    tokens = list(prompt) + list(run.out.tokens)
    entry = bank.put(
        runtime=run.rt,
        token_ids=tokens,
        cache=final.final_trunk_cache,
        logits=final.final_logits,
        hidden=final.final_hidden,
        hidden_variant="post_norm",
        session_id=session_id,
        mtp_history_policy="committed",
        mtp_history_snapshot=snapshot_cache(final.final_committed_mtp_cache),
        mtp_snapshot_epoch=len(tokens),
        snapshot_epoch=len(tokens),
    )
    assert entry is not None
    return tokens


def _continue_banked_turn(tiny_pack, monkeypatch, *, lazy: bool):
    bank = SessionBank()
    first = _generate(tiny_pack, monkeypatch, max_tokens=24, lazy=lazy, copy_streak=True)
    banked = _postcommit(bank, first, _PROMPT, "continuation")
    # The next user turn: everything banked, then new tokens to prefill.
    warm = _generate(
        tiny_pack,
        monkeypatch,
        max_tokens=24,
        lazy=lazy,
        prompt=banked + [41, 43, 45],
        session_bank=bank,
        session_id="continuation",
    )
    return banked, warm


def test_a_warm_turn_continuing_a_banked_turn_leaves_nothing_stacked(tiny_pack, monkeypatch):
    banked, lazy_warm = _continue_banked_turn(tiny_pack, monkeypatch, lazy=True)
    _banked, eager_warm = _continue_banked_turn(tiny_pack, monkeypatch, lazy=False)
    # The restore took the banked turn; the suffix after it was prefilled.
    assert lazy_warm.out.stats.cached_tokens == len(banked), lazy_warm.out.stats.cached_tokens
    _assert_bounded(lazy_warm)
    _assert_same_decode(lazy_warm, eager_warm)
