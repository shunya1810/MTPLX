"""PX.4: older GPUs get a fallback, not a failed request.

The Flash-Next fixed-M4 installed dispatch had no try/except: a kernel that
an older GPU refuses to build or dispatch failed the request. The first
dispatch of the process now runs under a guard that restores the pre-round
state, retires the lane to the eager verifier for the process, prints one
line and records the demotion. No model, no GPU work: the bank is built
bare, the compiled function is a stub that raises, and the state leaves are
plain Python objects.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from mtplx import demotions, generation, graphbank


class _Leaf:
    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<leaf {self.name}>"


def _qsa_entry(tag: str):
    kv = SimpleNamespace(
        cache=[_Leaf(f"{tag}.k"), _Leaf(f"{tag}.v"), _Leaf(f"{tag}.offset")],
        rollback_state=[_Leaf(f"{tag}.rb0"), _Leaf(f"{tag}.rb1")],
    )
    return SimpleNamespace(kv=kv, raw_keys=_Leaf(f"{tag}.raw"), pooled=_Leaf(f"{tag}.pooled"))


def _gdn_entry(tag: str):
    return SimpleNamespace(cache=[_Leaf(f"{tag}.conv"), _Leaf(f"{tag}.state")])


def _bank(fn, *, rope_args=()):
    qsa = _qsa_entry("qsa0")
    gdn = _gdn_entry("gdn0")
    bank = object.__new__(graphbank.CompiledVerifyBank)
    bank.stats = {"calls": 0, "compiled_calls": 0, "fallback_calls": 0, "buckets": {}}
    bank._held_state_refs = []
    bank._clear_shadow_leaf_refs = lambda: None
    bank.reserve_fixed_m4_window = lambda *_a, **_k: None
    bank._runtime_forward = lambda ids, **_kw: ("eager_logits", "eager_hidden", {"eager": True})
    bank._fixed_m4_dispatch = {
        "boundary": "none",
        "donate": True,
        "state_plan": (
            (graphbank.VERIFY_SPEC_KIND_QSA, qsa, 5),
            (graphbank.VERIFY_SPEC_KIND_GDN, gdn, 2),
        ),
        "prepare_aux": lambda *_a: "aux",
        "fn": fn,
        "host": None,
        "capture_leaves": 6,
        "capture_plan": ((gdn, 0, 6),),
        # What install_fixed_m4 binds for the request's positions: nothing
        # for a text request, the rotary delta for an image request.
        "rope_args": tuple(rope_args),
        "rope_delta": None,
        "parity2": False,
    }
    return bank, qsa, gdn


def _good_fn(_input_ids, _aux, *state_in):
    captures = tuple(_Leaf(f"cap{i}") for i in range(6))
    state_out = tuple(_Leaf(f"out{i}") for i in range(len(state_in)))
    return ("logits", "hidden", *captures, *state_out)


def _call(bank):
    return bank.forward_fixed_m4(
        SimpleNamespace(shape=(1, 4)),
        host_input_ids=[1, 2, 3, 4],
        completion_tokens=[1],
        committed_count=0,
        cache=[],
        hidden_variant=None,
    )


@pytest.fixture(autouse=True)
def _fresh_lane(monkeypatch):
    monkeypatch.setitem(graphbank._FIXED_M4_LANE, "proven", False)
    monkeypatch.setitem(graphbank._FIXED_M4_LANE, "retired", None)
    monkeypatch.setattr(graphbank.mx, "async_eval", lambda *_a, **_k: None)
    monkeypatch.setattr(graphbank.mx, "eval", lambda *_a, **_k: None)
    demotions.reset()
    yield
    demotions.reset()


def test_a_kernel_that_raises_retires_the_lane_and_the_round_runs_eager(capsys):
    def refused(*_args):
        raise RuntimeError(
            "[metal::Device] Thread group size (1024) is greater than the "
            "maximum allowed (896)"
        )

    bank, qsa, gdn = _bank(refused)
    before = (list(qsa.kv.cache), qsa.raw_keys, qsa.pooled, list(qsa.kv.rollback_state), list(gdn.cache))

    logits, hidden, captures = _call(bank)

    # The request got an answer from the eager forward, not an exception.
    assert (logits, hidden) == ("eager_logits", "eager_hidden")
    assert captures == {"eager": True}
    # The pre-round state is back, leaf for leaf (identity, not equality).
    assert all(a is b for a, b in zip(qsa.kv.cache, before[0]))
    assert qsa.raw_keys is before[1] and qsa.pooled is before[2]
    assert all(a is b for a, b in zip(qsa.kv.rollback_state, before[3]))
    assert all(a is b for a, b in zip(gdn.cache, before[4]))
    assert not hasattr(gdn, "_mtplx_verify_rows")
    # Retired for the process, said once, counted.
    assert "896" in graphbank.fixed_m4_lane_retired_reason()
    assert graphbank._FIXED_M4_LANE["proven"] is False
    assert bank.last_dispatch_kind == "eager"
    assert bank.last_fallback_reason == "fixed_m4_dispatch_retired"
    assert bank.stats["compiled_calls"] == 0
    snap = demotions.snapshot()
    assert snap["counts"]["fixed_m4_dispatch_retired"] == 1
    assert "896" in snap["reasons"]["fixed_m4_dispatch_retired"]
    out = capsys.readouterr().out
    assert out.count("Flash-Next compiled verifier could not dispatch") == 1
    assert "output is unchanged" in out


def test_a_failure_at_evaluation_time_is_caught_too(monkeypatch):
    # Metal builds pipelines when the graph is encoded, not when it is traced.
    bank, qsa, gdn = _bank(_good_fn)

    def boom(*_a, **_k):
        raise RuntimeError("Unable to build metal library from source")

    monkeypatch.setattr(graphbank.mx, "eval", boom)
    before_k = qsa.kv.cache[0]
    before_gdn = list(gdn.cache)
    logits, _hidden, _captures = _call(bank)
    assert logits == "eager_logits"
    # The replay had already rebound the state to its outputs (donating
    # route); the guard put the inputs back.
    assert qsa.kv.cache[0] is before_k
    assert all(a is b for a, b in zip(gdn.cache, before_gdn))
    assert not hasattr(gdn, "_mtplx_verify_rows")
    assert bank.stats["compiled_calls"] == 0
    assert graphbank.fixed_m4_lane_retired_reason().startswith("RuntimeError")


def test_later_rounds_and_later_requests_stay_eager_without_retrying(capsys):
    calls = []

    def refused(*_args):
        calls.append(1)
        raise RuntimeError("kernel refused")

    bank, _qsa, _gdn = _bank(refused)
    _call(bank)
    _call(bank)
    _call(bank)
    assert calls == [1]  # never dispatched again
    assert demotions.counts()["fixed_m4_dispatch_retired"] == 1
    assert demotions.counts()["fixed_m4_uncompiled_round"] == 3
    assert capsys.readouterr().out.count("could not dispatch") == 1
    # The next request never builds the bank: the construction gate says no.
    receipt: dict = {}
    rt = SimpleNamespace(qwen4_fixed_m4_compiled_verify=True)
    assert not generation._qwen4_fixed_m4_compiled_verify_requested(
        rt,
        verify_strategy="batched",
        compiled_mode="on",
        max_tokens=64,
        cached_tokens=0,
        prompt_tokens=100,
        speculative_depth=3,
        receipt=receipt,
    )
    assert receipt["reason"] == "dispatch_retired"
    assert demotions.counts()["fixed_m4_lane_skipped"] == 1


def test_a_clean_first_dispatch_proves_the_lane_and_later_rounds_are_unguarded(monkeypatch):
    bank, qsa, gdn = _bank(_good_fn)
    evals = []
    monkeypatch.setattr(graphbank.mx, "eval", lambda *a, **_k: evals.append(a))
    logits, hidden, captures = _call(bank)
    assert (logits, hidden, captures) == ("logits", "hidden", {})
    assert graphbank._FIXED_M4_LANE["proven"] is True
    assert graphbank.fixed_m4_lane_retired_reason() is None
    assert len(evals) == 1  # the guard's one blocking eval
    assert qsa.kv.cache[0].name == "out0"
    assert gdn._mtplx_verify_rows[0].name == "cap0"
    assert bank.stats["compiled_calls"] == 1
    # Round two takes today's replay verbatim: no guard, no extra eval.
    proved = []
    monkeypatch.setattr(
        bank, "_prove_fixed_m4_dispatch", lambda *a, **k: proved.append(1)
    )
    _call(bank)
    assert proved == [] and len(evals) == 1
    assert bank.stats["compiled_calls"] == 2
    assert demotions.snapshot()["total"] == 0


# ---------------------------------------------------------------------------
# Image requests: the rotary delta is a graph input of the replay.
# ---------------------------------------------------------------------------


def _recording_fn(seen):
    def fn(input_ids, *args):
        seen.append((input_ids, args))
        return _good_fn(input_ids, args[0], *args[-7:])

    return fn


def test_the_replay_passes_the_delta_between_the_auxiliary_and_the_state():
    """``verify_step(input_ids, aux, rope_delta, *state)``: the trace unpacks
    its inputs by position, so the order is the contract."""

    seen = []
    delta = _Leaf("rope_delta")
    bank, qsa, gdn = _bank(_recording_fn(seen), rope_args=(delta,))
    state_before = [*qsa.kv.cache, qsa.raw_keys, qsa.pooled, *gdn.cache]
    _call(bank)
    (input_ids, args), = seen
    assert input_ids.shape == (1, 4)
    assert args[0] == "aux"
    assert args[1] is delta
    assert len(args) == 2 + len(state_before)
    assert all(a is b for a, b in zip(args[2:], state_before))


def test_a_text_request_replays_with_the_inputs_it_always_had():
    seen = []
    bank, qsa, gdn = _bank(_recording_fn(seen))
    state_before = [*qsa.kv.cache, qsa.raw_keys, qsa.pooled, *gdn.cache]
    _call(bank)
    (_input_ids, args), = seen
    assert args[0] == "aux"
    assert len(args) == 1 + len(state_before)
    assert all(a is b for a, b in zip(args[1:], state_before))


def test_the_delta_is_bound_once_as_one_int32_value():
    import mlx.core as mx

    bank = object.__new__(graphbank.CompiledVerifyBank)
    bank._rope_delta = None
    bank._rope_delta_value = None
    bank._adopt_rope_delta(None)
    assert bank._rope_args() == ()
    assert bank._verify_key(4, None, 0) == (4, "", 0, 0)

    bank._adopt_rope_delta(-990)
    (delta,) = bank._rope_args()
    assert delta.dtype == mx.int32 and tuple(delta.shape) == (1,)
    assert bank._rope_delta_value == -990
    # The key says THAT the step takes a delta, never which one: every image
    # request of the process shares one trace.
    assert bank._verify_key(4, None, 1) == (4, "", 1, 1)
    bank._adopt_rope_delta(-990)  # the same request saying it again
    assert bank._rope_args()[0] is delta
    with pytest.raises(ValueError, match="a request has one delta"):
        bank._adopt_rope_delta(-4)
    # The rope kernels take one int32 value; anything else is refused where
    # the request is admitted, not inside a trace.
    for bad in (mx.array([1], dtype=mx.int64), mx.array([1, 2], dtype=mx.int32), 1.5):
        with pytest.raises(TypeError):
            graphbank.as_rope_delta(bad)


def test_install_stays_closed_to_the_eager_authoritative_parity_mode():
    bank = object.__new__(graphbank.CompiledVerifyBank)
    bank.strict_no_fallback = True
    bank.parity = True
    bank.parity2 = False
    with pytest.raises(ValueError, match="parity2"):
        bank.install_fixed_m4([], prompt_ids=[1], hidden_variant=None)
