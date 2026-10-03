"""A shared compiled verify program holds a bounded number of traces (#546).

MLX keeps every trace of a compiled function, one per input-shape signature,
until the function object is destroyed. The fixed-M4 verify state is sized
from the prompt, so an agent session traced a new shape nearly every turn, and
a program shared for the life of the process grew host memory without bound
(8.58 MiB per trace on a tiny-width model with Flash-Next's 48 layers).

Pinned here:
- a program that has traced its limit is replaced at the next lookup and goes
  with the last bank that still holds it, so no more than the limit of traces
  is ever reachable from the registry;
- a bank that holds a replaced program finishes on it, and a shape that
  program traced replays there without a new trace;
- the bank that replaces a program traces its own shapes, including a shape
  the old program had traced;
- every trace runs on the calling bank's containers and counts in its stats,
  even when another bank looked the program up after it;
- replacing programs changes no token: a growing session decodes the same
  tokens whether every lookup replaces the program or none does.
"""

from __future__ import annotations

import gc
import weakref
from types import SimpleNamespace

import mlx.core as mx
import pytest

import mtplx.generation as generation
from mtplx import graphbank
from tests.test_qsa_index_block_writes_evaluated import (  # noqa: F401 - tiny_pack is a fixture
    _same_bits,
    tiny_pack,
)

_KEY = (4, "", 0, 0)
_IDS = mx.array([[1, 2, 3, 4]], dtype=mx.int32)


class Runtime:
    pass


def _forward(input_ids, *, cache, return_hidden, hidden_variant, compiled_aux=None):
    """A verify forward over one state leaf: every value is exact in fp32."""

    state = cache[0].cache[0]
    new_state = state * 0.5 + input_ids.astype(mx.float32).sum()
    cache[0].cache[0] = new_state
    return new_state.sum(axis=-1, keepdims=True), new_state[..., :1], {}


def _bank(runtime):
    bank = object.__new__(graphbank.CompiledVerifyBank)
    bank.runtime = runtime
    bank.capture_backend = "linear_gdn_from_conv_tape"
    bank._capture_layout_override = ()
    bank._extra_capture_layout = ()
    bank._prepare_compiled_aux = None
    bank._spec = [(0, "state", 1)]
    bank._shadow = [SimpleNamespace(cache=[None])]
    bank._compiled = {}
    bank._program_hosts = {}
    bank.stats = {"traces": 0, "shared_programs_retired": 0}
    bank._runtime_forward = _forward
    return bank


def _call(bank, width: int, fn=None):
    """One dispatch, checked against the eager forward.

    ``fn`` is a program the caller looked up; by default the bank's own
    lookup, which a dispatch makes before every call.
    """

    if fn is None:
        fn = bank._verify_program(_KEY, 4, None)
    state = mx.arange(width, dtype=mx.float32).reshape(1, width)
    out = fn(_IDS, state)
    shadow = [SimpleNamespace(cache=[state])]
    logits, hidden, _captures = _forward(
        _IDS, cache=shadow, return_hidden=True, hidden_variant=None
    )
    reference = (logits, hidden, shadow[0].cache[0])
    assert len(out) == len(reference)
    assert all(mx.array_equal(a, b).item() for a, b in zip(out, reference))
    return fn


@pytest.fixture
def programs(monkeypatch):
    """Weak references to the body of every verify program compiled here."""

    monkeypatch.setattr(graphbank, "_SHARED_VERIFY_STEPS", {})
    monkeypatch.setenv("MTPLX_COMPILED_VERIFY_SHARED_TRACES", "1")
    monkeypatch.setenv("MTPLX_COMPILED_VERIFY_TRACES_PER_PROGRAM", "2")
    made: list[weakref.ref] = []
    original = graphbank.CompiledVerifyBank._make_verify_step

    def recording(self, *args, **kwargs):
        step = original(self, *args, **kwargs)

        def program_body(*inputs):
            return step(*inputs)

        made.append(weakref.ref(program_body))
        return program_body

    monkeypatch.setattr(graphbank.CompiledVerifyBank, "_make_verify_step", recording)
    return made


def test_a_program_that_traced_its_limit_is_replaced_and_freed(programs):
    runtime = Runtime()
    retired = 0
    for width in (8, 16, 24, 32, 40):
        bank = _bank(runtime)
        # A bank's first dispatch takes the program from the registry.
        _call(bank, width, fn=bank._shared_or_new_verify_step(_KEY, 4, None))
        assert bank.stats["traces"] == 1
        retired += bank.stats["shared_programs_retired"]
        del bank
    gc.collect()
    # Two traces per program: the third and the fifth lookups replaced it.
    assert (len(programs), retired) == (3, 2)
    assert [ref() is not None for ref in programs] == [False, False, True]


def test_banks_on_a_replaced_program_finish_on_it_and_the_next_bank_traces_fresh(programs):
    runtime = Runtime()
    a, b = _bank(runtime), _bank(runtime)
    first = _call(a, 8)
    assert _call(b, 16) is first
    assert (a.stats["traces"], b.stats["traces"]) == (1, 1)
    # b looked the program up after a; a's next new shape still traces on a's
    # own containers and counts in a's stats.
    assert _call(a, 24) is first
    assert (a.stats["traces"], b.stats["traces"]) == (2, 1)

    c = _bank(runtime)
    replacement = _call(c, 8)
    # Three traces against a limit of two: c's lookup replaced the program,
    # and c traced width 8 itself although the old program holds that shape.
    assert replacement is not first and len(programs) == 2
    assert (c.stats["shared_programs_retired"], c.stats["traces"]) == (1, 1)

    # a finishes on the old program; the shape it traced replays there.
    assert _call(a, 8) is first and a.stats["traces"] == 2
    assert programs[0]() is not None
    del a, b, first
    gc.collect()
    assert programs[0]() is None and programs[1]() is not None


def _session(tiny_pack, monkeypatch, limit: int):
    from mtplx.qwen4_fixed_verify import install_qwen4_fixed_verify_route
    from mtplx.sampling import SamplerConfig

    smoke, model = tiny_pack
    monkeypatch.setattr(graphbank, "_SHARED_VERIFY_STEPS", {})
    monkeypatch.setattr(graphbank, "_compiled_verify_bits_gate_ok", lambda _rt: True)
    monkeypatch.setenv("MTPLX_COMPILED_VERIFY", "1")
    monkeypatch.setenv("MTPLX_COMPILED_VERIFY_SHARED_TRACES", "1")
    monkeypatch.setenv("MTPLX_COMPILED_VERIFY_TRACES_PER_PROGRAM", str(limit))
    rt = smoke._tiny_runtime(model)
    install_qwen4_fixed_verify_route(rt)
    sampler = SamplerConfig(temperature=0.6, top_p=0.95, top_k=20)
    turns = []
    for turn in range(4):
        prompt = [3 + (i * 7) % 100 for i in range(40 + 24 * turn)]
        turns.append(
            generation.generate_mtpk(
                rt,
                prompt,
                max_tokens=24,
                sampler=sampler,
                draft_sampler=sampler,
                speculative_depth=3,
                seed=77 + turn,
                mtp_cache_policy="persistent",
                mtp_history_policy="committed",
                verify_strategy="batched",
                stop_token_ids=set(),
                capture_final_state=True,
            )
        )
    return turns


def test_replacing_programs_changes_no_token(tiny_pack, monkeypatch):
    replaced = _session(tiny_pack, monkeypatch, limit=1)
    kept = _session(tiny_pack, monkeypatch, limit=1000)
    reports = [turn.stats.graphbank["compiled_verify"] for turn in replaced + kept]
    assert all(r["compiled_calls"] > 0 and r["fallback_calls"] == 0 for r in reports), reports
    # One trace per turn in both sessions: a new shape each turn, and under
    # the limit of one also a new program each turn.
    assert [r["traces"] for r in reports] == [1] * 8
    # Every turn after the first replaced the program under the limit of one,
    # and none did under the other.
    assert [r["shared_programs_retired"] for r in reports] == [0, 1, 1, 1] + [0] * 4
    for ours, theirs in zip(replaced, kept):
        assert list(ours.tokens) == list(theirs.tokens)
        assert _same_bits(ours.final_state.final_logits, theirs.final_state.final_logits)
