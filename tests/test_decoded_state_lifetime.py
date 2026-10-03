"""Decoded and prefill state goes when its last reference does.

A helper that walks a tree of arrays with a closure that calls itself is a
reference cycle: the closure holds itself and whatever list it fills, so
every array it collected stays alive until Python's cyclic collector runs.
On 2026-10-01 the SSD decoder's walker kept a rejected 4.17 GB candidate
active through the cold prefill after it (the review of 425ffc58). These
tests run with the collector off: only reference counting may free them.
"""

from __future__ import annotations

import gc
import weakref
from types import SimpleNamespace

import pytest

import mtplx.cache_bank.codec as codec
from mtplx.cache_state import CacheSnapshot


class _Leaf:
    """Stands in for an mx.array: weak-referenceable, holds nothing."""


@pytest.fixture
def no_cyclic_collector():
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if was_enabled:
            gc.enable()


def _payload(leaf):
    return SimpleNamespace(
        cache_snapshot=CacheSnapshot(states=([leaf],), meta_states=(None,)),
        logits={"rows": leaf},
        hidden=None,
        mtp_history_snapshot=None,
        gdn_boundaries=(),
    )


def test_a_rejected_decode_is_freed_without_a_collection(monkeypatch, no_cyclic_collector):
    evaluated = []
    monkeypatch.setattr(
        codec, "mx", SimpleNamespace(array=_Leaf, eval=lambda *a: evaluated.append(len(a)))
    )

    def decode_then_reject():
        leaf = _Leaf()
        codec._eval_decoded_arrays(_payload(leaf))
        return weakref.ref(leaf)

    ref = decode_then_reject()
    assert evaluated == [2]
    assert ref() is None


def test_a_decode_that_fails_to_evaluate_is_freed_with_its_error(monkeypatch, no_cyclic_collector):
    def refuse(*arrays):
        raise RuntimeError("[metal::malloc] Unable to allocate")

    monkeypatch.setattr(codec, "mx", SimpleNamespace(array=_Leaf, eval=refuse))

    def decode():
        leaf = _Leaf()
        ref = weakref.ref(leaf)
        try:
            codec._eval_decoded_arrays(_payload(leaf))
        except RuntimeError:
            pass
        return ref

    assert decode()() is None


def test_the_prefill_midloop_walk_is_freed_without_a_collection(monkeypatch, no_cyclic_collector):
    import mtplx.models.qwen4_exp as qwen4

    class _Recurrent:
        def __init__(self, leaf):
            self.cache = [leaf, [leaf]]

    monkeypatch.setattr(qwen4, "mx", SimpleNamespace(array=_Leaf))
    monkeypatch.setattr(qwen4, "ArraysCache", _Recurrent)

    def name_then_replace():
        leaf = _Leaf()
        entry = _Recurrent(leaf)
        found = qwen4._midloop_state_arrays([entry, object()])
        assert found == [leaf, leaf]
        # The next chunk's states replace this one's on the cache entry.
        entry.cache = []
        return weakref.ref(leaf)

    assert name_then_replace()() is None
