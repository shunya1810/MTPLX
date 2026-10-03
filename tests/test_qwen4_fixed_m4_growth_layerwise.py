"""An installed fixed-M4 bank grows one layer at a time, billed at its peak (mtplx/graphbank.py).

The 2026-10-01 review of the capacity fix found that an installed bank's
growth during an answer (``CompiledVerifyBank.reserve_fixed_m4_window``)
asked the memory admission once, for one layer's new banks, then grew every
layer with its old banks swapped out before the new ones were allocated and
a block of zeros allocated for every pad: on three tiny layers it admitted
1,212,416 bytes, peaked at 3,031,048 and kept 1,818,624. An allocation that
failed partway left the verifier's adapters in the conversation's lease, so
the session bank refused the lease and the retry ran cold.

Now the growth asks for its bill before any leaf changes (the rows the
layers grown before the peak keep, plus one layer's new banks), then for
each layer's new banks before that layer is written; each layer's banks are
written, then swapped in, and its old banks are let go before the next layer
is admitted. A refusal, or an allocation the memory refuses partway, leaves
every layer whole and ends the answer between rounds with its memory stop,
as a refused growth always has, and the lease stays usable. This file pins
it with tiny arrays and with the tiny Flash-Next pack.
"""

from __future__ import annotations

import gc
import json
import os
import subprocess
import sys

import mlx.core as mx
import numpy as np
import pytest

import mtplx.generation as generation
import mtplx.graphbank as graphbank
from mtplx.graphbank import FixedM4CapacityPlan, FixedM4GrowthRefused, TensorOffsetQSACache
from test_one_copy_reserve_resize import (  # noqa: F401
    SEED,
    _compiled,
    _layer,
    _same_rounds,
    _same_turn,
    _Session,
    gpu,
    lane,
    tiny,
)
from test_qwen4_fixed_m4_capacity_bucket import _bits

# Three tiny layers holding 4,000 tokens in 4,096 rows, growing to 8,192:
# keys and values (2 x 2 heads x 16 x bf16 a row), raw index keys (8 x bf16
# a row), pooled index keys (8 x bf16 a block of four rows).
NEW = 8192 * (2 * 2 * 16 * 2 + 8 * 2) + 2048 * 8 * 2
OLD = 4096 * (2 * 2 * 16 * 2 + 8 * 2) + 1024 * 8 * 2
GAIN = NEW - OLD
BILL = 2 * GAIN + NEW


class _Done(Exception):
    """Raised where the bank would rebuild its program after a growth."""


def _installed(seeds=(21, 22, 23)):
    """Three installed banks the stock layers were adopted into, 4,096 rows each."""

    banks = []
    for seed in seeds:
        stock = _layer(4000, 4096, seed=seed)
        stock.pooled_f32_t = None
        bank = TensorOffsetQSACache.from_qsa_cache(stock, reserve_tokens=96)
        assert bank.promotion == "adopted" and bank.capacity == 4096
        banks.append(bank)
    for bank in banks:
        mx.eval(*bank.state_leaves)
    return banks


def _verify_bank(banks, admit):
    """The installed dispatch of a verify bank over ``banks``, nothing compiled."""

    bank = graphbank.CompiledVerifyBank.__new__(graphbank.CompiledVerifyBank)
    bank._fixed_m4_dispatch = {
        "base_offset": 4000, "dense_capacity": 4096, "growth_tokens": 4096,
        "capacity_limit": None, "route_transition_at": None, "qsa_entries": banks,
    }
    bank._held_state_refs = []
    bank._clear_shadow_leaf_refs = lambda: None

    def rebuild(_cache):
        raise _Done()

    bank._ensure_shadow = rebuild
    bank.capacity_plan = FixedM4CapacityPlan(1024, 0, admit_growth=admit)
    return bank


def _gather_lane() -> None:
    os.environ["MTPLX_QSA_GATHER"] = "1"
    os.environ["MTPLX_QSA_GATHER_MIN_CONTEXT"] = "16"
    for name in ("MTPLX_QSA_SCORE_TILE_ROWS", "MTPLX_QSA_M4_FUSED_KV_GATHER"):
        os.environ.pop(name, None)


@pytest.fixture()
def gathering(monkeypatch):
    monkeypatch.setenv("MTPLX_QSA_GATHER", "1")
    monkeypatch.setenv("MTPLX_QSA_GATHER_MIN_CONTEXT", "16")
    for name in ("MTPLX_QSA_SCORE_TILE_ROWS", "MTPLX_QSA_M4_FUSED_KV_GATHER"):
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


# -- the bank's growth, tiny arrays --------------------------------------------------


def _growth_transient() -> dict:
    """What the allocator holds while three installed banks grow from 4,096 rows to 8,192.

    Run by test_the_growth_holds_its_bill_and_no_more in a process of its own.
    Every buffer is a whole number of pages, so the allocator's counts match
    the bill byte for byte.
    """

    _gather_lane()
    banks = _installed()
    asked = []
    gc.collect()
    mx.synchronize()
    start = mx.get_active_memory()
    mx.reset_peak_memory()

    def admit(need):
        asked.append([need, mx.get_active_memory() - start])
        return True

    try:
        _verify_bank(banks, admit).reserve_fixed_m4_window(banks, committed_count=100)
    except _Done:
        pass
    return {
        "asked": asked,
        "capacities": [bank.capacity for bank in banks],
        "peak": mx.get_peak_memory() - start,
        "end": mx.get_active_memory() - start,
    }


@pytest.mark.skipif(not mx.metal.is_available(), reason="measures the Metal allocator")
def test_the_growth_holds_its_bill_and_no_more():
    """The review's witness: one admission of 1,212,416 bytes, a 3,031,048
    peak and 1,818,624 kept. Now the bill before any leaf changes, then each
    layer's new banks, and the peak is the bill."""

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        item for item in (root, env.get("PYTHONPATH", "")) if item
    )
    done = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "--isolated", "_growth_transient"],
        cwd=root, env=env, capture_output=True, text=True, timeout=300, check=False,
    )
    assert done.returncode == 0, f"{done.stdout}\n{done.stderr}"
    got = json.loads(done.stdout.strip().splitlines()[-1])
    print(got)
    assert got["capacities"] == [8192] * 3
    # The bill, then each layer's new banks with the layers before it grown
    # and their old banks gone.
    assert got["asked"] == [[BILL, 0], [NEW, 0], [NEW, GAIN], [NEW, 2 * GAIN]]
    assert BILL == 2_424_832
    # The peak is the bill (the last layer's new banks beside its old ones);
    # what is over it is the zero each pad is broadcast from.
    assert 0 <= got["peak"] - BILL < 4096
    assert got["end"] == 3 * GAIN


def test_a_bill_over_the_line_changes_no_leaf(gathering):
    """The review's room, one layer's new banks: the growth is refused whole."""

    banks = _installed()
    leaves = [bank.state_leaves for bank in banks]
    asked = []

    def admit(need):
        asked.append(need)
        return need <= NEW + 64

    with pytest.raises(FixedM4GrowthRefused) as refused:
        _verify_bank(banks, admit).reserve_fixed_m4_window(banks, committed_count=100)
    # The bucketed bill, then the unbucketed one (the same here: no bucket).
    assert asked == [BILL, BILL]
    assert refused.value.receipt["bill_bytes"] == BILL
    assert refused.value.receipt["reason"] == "fixed_m4_growth_refused"
    for bank, old in zip(banks, leaves):
        assert bank.capacity == 4096
        assert all(a is b for a, b in zip(bank.state_leaves, old))


def _grown_like_before(old, rows, axis):
    """What the growth wrote before the review: the rows, then a block of zeros."""

    shape = list(old.shape)
    shape[axis] = rows - int(old.shape[axis])
    return mx.concatenate([old, mx.zeros(tuple(shape), dtype=old.dtype)], axis=axis)


def _expected(bank_leaves):
    keys, values, _offset, raw, pooled = bank_leaves
    return [
        _bits(_grown_like_before(keys, 8192, 2)),
        _bits(_grown_like_before(values, 8192, 2)),
        _bits(_grown_like_before(raw, 8192, 1)),
        _bits(_grown_like_before(pooled, 2048, 1)),
    ]


def _grown_bits(bank):
    return [_bits(bank.kv.keys), _bits(bank.kv.values), _bits(bank.raw_keys), _bits(bank.pooled)]


def test_grown_banks_hold_what_the_growth_wrote_before(gathering):
    banks = _installed()
    want = [_expected(bank.state_leaves) for bank in banks]
    offsets = [int(bank.kv.offset.item()) for bank in banks]
    try:
        _verify_bank(banks, lambda need: True).reserve_fixed_m4_window(banks, committed_count=100)
    except _Done:
        pass
    for bank, expected, offset in zip(banks, want, offsets):
        assert bank.capacity == bank.dense_capacity == 8192
        assert int(bank.kv.offset.item()) == offset
        assert bank.kv._granted is True and bank.kv.growth_after_grant is False
        for got, wanted in zip(_grown_bits(bank), expected):
            assert got.shape == wanted.shape and np.array_equal(got, wanted)


def test_a_layer_refused_partway_leaves_every_layer_whole(gathering):
    """Memory taken after the bill was admitted: the second layer is refused."""

    banks = _installed()
    want = _expected(banks[0].state_leaves)
    leaves = [bank.state_leaves for bank in banks[1:]]
    asked = []

    def admit(need):
        asked.append(need)
        return len(asked) < 3

    verify = _verify_bank(banks, admit)
    with pytest.raises(FixedM4GrowthRefused) as refused:
        verify.reserve_fixed_m4_window(banks, committed_count=100)
    assert asked == [BILL, NEW, NEW]
    assert refused.value.receipt["layers_left"] == 2
    # The first layer grew whole; the others kept their banks.
    assert banks[0].capacity == 8192
    assert all(np.array_equal(g, w) for g, w in zip(_grown_bits(banks[0]), want))
    for bank, old in zip(banks[1:], leaves):
        assert bank.capacity == 4096
        assert all(a is b for a, b in zip(bank.state_leaves, old))
    # The dispatch still describes the capacity every layer holds.
    assert verify._fixed_m4_dispatch["dense_capacity"] == 4096


def _second_growth_write_raises(monkeypatch, error, calls=None):
    """``mx.eval`` raises ``error`` while the second layer's new banks are written."""

    real_ensure = TensorOffsetQSACache.ensure_capacity
    real_eval = mx.eval
    calls = [] if calls is None else calls

    def ensuring(self, needed):
        if int(needed) <= self.capacity:
            return real_ensure(self, needed)
        calls.append(int(needed))
        if len(calls) != 2:
            return real_ensure(self, needed)

        def failing(*arrays):
            raise error

        monkeypatch.setattr(mx, "eval", failing)
        try:
            return real_ensure(self, needed)
        finally:
            monkeypatch.setattr(mx, "eval", real_eval)

    monkeypatch.setattr(TensorOffsetQSACache, "ensure_capacity", ensuring)
    return calls


def test_an_allocation_the_memory_refuses_partway_leaves_every_layer_whole(gathering):
    banks = _installed()
    want = _expected(banks[0].state_leaves)
    leaves = [bank.state_leaves for bank in banks[1:]]
    calls = _second_growth_write_raises(
        gathering, RuntimeError("[metal::malloc] Unable to allocate 1212416 bytes"),
    )
    with pytest.raises(FixedM4GrowthRefused) as refused:
        _verify_bank(banks, lambda need: True).reserve_fixed_m4_window(banks, committed_count=100)
    assert calls == [8192, 8192]
    receipt = refused.value.receipt
    assert receipt["layers_left"] == 2
    assert "Unable to allocate" in receipt["allocation_error"]
    assert banks[0].capacity == banks[0].dense_capacity == 8192
    assert all(np.array_equal(g, w) for g, w in zip(_grown_bits(banks[0]), want))
    # The layer that failed is as it was, its dense width included.
    for bank, old in zip(banks[1:], leaves):
        assert bank.capacity == bank.dense_capacity == 4096
        assert all(a is b for a, b in zip(bank.state_leaves, old))


@pytest.mark.parametrize(
    "error", [RuntimeError("[metal] shape mismatch"), ValueError("bad rows")],
    ids=["runtime_error", "value_error"],
)
def test_other_failures_in_a_growth_reach_the_caller(gathering, error):
    banks = _installed()
    leaves = banks[1].state_leaves
    _second_growth_write_raises(gathering, error)
    with pytest.raises(type(error)):
        _verify_bank(banks, lambda need: True).reserve_fixed_m4_window(banks, committed_count=100)
    assert banks[1].capacity == 4096
    assert all(a is b for a, b in zip(banks[1].state_leaves, leaves))


# -- through generate_mtpk on the tiny Flash-Next pack -------------------------------
#
# The conversation of test_one_copy_reserve_resize: 508 tokens in 512-row
# buffers. A turn with the same prompt and a 4-token reserve adopts them as
# a 512-row bank, which grows to 1,024 rows during its 24-token answer.


@gpu
def test_a_bank_that_grows_mid_answer_gives_the_answer_of_one_that_never_grows(tiny, lane):
    grows, roomy = _Session(tiny, lane), _Session(tiny, lane)
    for session in (grows, roomy):
        session.opening()
    with lane.context() as patch:
        patch.setenv("MTPLX_COMPILED_VERIFY_GROWTH_RESERVE", "4")
        grown = grows.turn([], max_tokens=24)
    whole = roomy.turn([], max_tokens=24)
    _compiled(grown)
    _compiled(whole)
    assert grown.bank["fixed_m4_capacity_transitions"] >= 1
    assert whole.bank["fixed_m4_capacity_transitions"] == 0
    assert grown.stats.memory_stop is None
    # The rows-gather lane: the bank's capacity changes no value.
    _same_rounds(
        [{k: v for k, v in r.items() if k != "capacity"} for r in grown.rounds],
        [{k: v for k, v in r.items() if k != "capacity"} for r in whole.rounds],
    )
    _same_turn(grown, whole)


@gpu
def test_an_allocation_failure_while_the_bank_grows_ends_the_answer_and_keeps_the_lease(
    tiny, lane,
):
    """The review's witness: the second layer's growth fails to allocate during
    the answer. The error left the request and the verifier's adapters stayed
    in the conversation's lease, which the session bank then refused: the
    conversation went on cold. Now the answer ends at that round with its
    memory stop, exactly as when the admission refuses that layer, and the
    conversation goes on warm from it."""

    failing, refused, uninterrupted = (_Session(tiny, lane) for _ in range(3))
    for each in (failing, refused, uninterrupted):
        each.opening()
    error = RuntimeError("[metal::malloc] Unable to allocate 172032 bytes")
    calls: list[int] = []
    asked: list[int] = []

    def refuse_the_second_layer(patch):
        def layer_fits(_rt, need, **_kwargs):
            asked.append(int(need))
            return len(asked) != 3  # the bill, the first layer, then no

        patch.setattr(generation, "_qwen4_fixed_m4_layer_fits", layer_fits)

    with lane.context() as patch:
        patch.setenv("MTPLX_COMPILED_VERIFY_GROWTH_RESERVE", "4")
        ended = failing.turn([], max_tokens=24, patches=[
            lambda inner: _second_growth_write_raises(inner, error, calls),
        ])
        stopped = refused.turn([], max_tokens=24, patches=[refuse_the_second_layer])
        full = uninterrupted.turn([], max_tokens=24)
    assert calls[:2] == [calls[0]] * 2 and len(asked) == 3
    stop = ended.stats.memory_stop
    assert stop is not None and stop["reason"] == "fixed_m4_growth_refused"
    assert stop["layers_left"] == 1 and "Unable to allocate" in stop["allocation_error"]
    assert stop["completion_tokens"] == len(ended.tokens)
    assert stopped.stats.memory_stop["layers_left"] == 1
    assert 0 < len(ended.tokens) < 24
    # Every token it streamed is the token the unrefused answer streamed, and
    # it stopped where the refused layer stopped the answer.
    assert ended.tokens == stopped.tokens == full.tokens[: len(ended.tokens)]
    _same_turn(ended, stopped)
    # The stopped answer was committed with stock containers: the next turn
    # restores the whole conversation and matches the refused one's.
    after, after_refused = failing.turn([9, 10, 11, 12]), refused.turn([9, 10, 11, 12])
    assert after.stats.session_restore_mode == "reference_lease"
    assert after.stats.cached_tokens == len(after.prompt) - 4 == 508 + len(ended.tokens)
    assert after_refused.stats.cached_tokens == after.stats.cached_tokens
    _compiled(after)
    _same_rounds(after.rounds, after_refused.rounds)
    _same_turn(after, after_refused)


if __name__ == "__main__" and sys.argv[1:] == ["--isolated", "_growth_transient"]:
    print(json.dumps(_growth_transient()))
