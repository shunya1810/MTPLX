"""Shared compiled programs must not own a completed request's cache buffers."""

import gc
import weakref
from types import SimpleNamespace

from mtplx import graphbank


class Runtime:
    pass


def _bank(runtime, marker):
    bank = object.__new__(graphbank.CompiledVerifyBank)
    bank.runtime = runtime
    bank.capture_backend = "linear_gdn_from_conv_tape"
    bank._capture_layout_override = ()
    bank._extra_capture_layout = ()
    bank._prepare_compiled_aux = None
    bank._spec = []
    bank._shadow = []
    bank._program_hosts = {}
    bank.stats = {"traces": 0}
    bank._runtime_forward = lambda *args, **kwargs: (marker, "hidden", {})
    return bank


def test_shared_compiled_trace_does_not_own_completed_bank(monkeypatch):
    # The retention defect is Python ownership, independent of MLX execution.
    monkeypatch.setattr(graphbank.mx, "compile", lambda fn: fn)
    monkeypatch.setattr(graphbank, "_SHARED_VERIFY_STEPS", {})
    monkeypatch.setenv("MTPLX_COMPILED_VERIFY_SHARED_TRACES", "1")
    runtime = Runtime()
    first = _bank(runtime, "first")
    reference = weakref.ref(first)
    fn = first._shared_or_new_verify_step((4, "default", 8192, 0), 4, None)
    assert fn(SimpleNamespace(shape=(1, 4))) == ("first", "hidden")
    del first
    gc.collect()
    assert reference() is None, "process-global trace retains a completed request's shadow cache"

    second = _bank(runtime, "second")
    second_fn = second._shared_or_new_verify_step((4, "default", 8192, 0), 4, None)
    assert second_fn is fn
    assert second_fn(SimpleNamespace(shape=(1, 4))) == ("second", "hidden")
    assert second.stats["traces"] == 1


def test_unshared_callable_does_not_create_a_bank_cycle(monkeypatch):
    monkeypatch.setattr(graphbank.mx, "compile", lambda fn: fn)
    monkeypatch.setenv("MTPLX_COMPILED_VERIFY_SHARED_TRACES", "0")
    bank = _bank(Runtime(), "unshared")
    reference = weakref.ref(bank)
    fn = bank._shared_or_new_verify_step((4, "default", 8192, 0), 4, None)
    assert fn(SimpleNamespace(shape=(1, 4))) == ("unshared", "hidden")
    del bank
    gc.collect()
    assert reference() is None


def test_model_unload_releases_its_shared_programs(monkeypatch):
    monkeypatch.setattr(graphbank.mx, "compile", lambda fn: fn)
    monkeypatch.setattr(graphbank, "_SHARED_VERIFY_STEPS", {})
    monkeypatch.setenv("MTPLX_COMPILED_VERIFY_SHARED_TRACES", "1")
    runtime = Runtime()
    bank = _bank(runtime, "model")
    fn = bank._shared_or_new_verify_step((4, "default", 8192, 0), 4, None)
    program = weakref.ref(fn)
    del fn, bank
    gc.collect()
    assert program() is not None, "A live model should retain its reusable program"
    del runtime
    gc.collect()
    assert not graphbank._SHARED_VERIFY_STEPS, "Unloaded models leave permanent compiled entries"
    assert program() is None
