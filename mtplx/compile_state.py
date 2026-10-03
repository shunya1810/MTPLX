"""Shared 'inside a compiled forward trace' flag (issue #51, 70 tps goal).

A dependency-free home for one bit of state so `compiled_forward` (which sets it)
and the model forwards (which read it) can agree without an import cycle.

The model decode loop keeps the GPU fed during Python graph-build by calling
`mx.async_eval` every N layers (the submit cadence). Inside an `mx.compile`
trace that call is (a) illegal — "[async_eval] Not allowed inside a graph
transformation" — and (b) pointless, because the whole reason to compile is to
replace the per-layer Python walk with a single traced submission. So while a
compiled forward is tracing/replaying, the forward checks `compile_trace_active()`
and suppresses those scheduling-only host-syncs. Kernel math and ordering are
unchanged; this only removes an eval whose sole job was to paper over the
graph-build stall that compilation eliminates outright.
"""

from __future__ import annotations

import contextlib
from contextvars import ContextVar
from typing import Hashable, Iterator

_COMPILE_TRACE_ACTIVE = False


def compile_trace_active() -> bool:
    """True while a compiled AR forward is executing (and while it traces)."""
    return _COMPILE_TRACE_ACTIVE


@contextlib.contextmanager
def compile_trace() -> Iterator[None]:
    """Mark the enclosed block as running inside a compiled forward."""
    global _COMPILE_TRACE_ACTIVE
    previous = _COMPILE_TRACE_ACTIVE
    _COMPILE_TRACE_ACTIVE = True
    try:
        yield
    finally:
        _COMPILE_TRACE_ACTIVE = previous


# The Python body of a compiled step runs only while ``mx.compile`` traces it;
# a replay runs the recorded graph and no Python. Diagnostics that record host
# facts from inside the model forward (the KV attention record, issue #526)
# read this to label a record as a trace rather than as the dispatch that
# later fails. A ContextVar, so a trace on the model thread never labels a
# call on another thread.
_IN_COMPILED_STEP_BODY: ContextVar[bool] = ContextVar(
    "mtplx_in_compiled_step_body", default=False
)


def in_compiled_step_body() -> bool:
    """True inside a compiled step's Python body, i.e. while it is traced."""
    return _IN_COMPILED_STEP_BODY.get()


@contextlib.contextmanager
def compiled_step_body() -> Iterator[None]:
    """Wrap the Python body a compiled step hands to ``mx.compile``."""
    token = _IN_COMPILED_STEP_BODY.set(True)
    try:
        yield
    finally:
        _IN_COMPILED_STEP_BODY.reset(token)


# The specialization a compiled dispatcher is calling: the compiled callable
# and the input shapes that select its graph. A trace records under it (the
# KV attention record, issue #526), so a later replay of the same graph can be
# matched to its own trace and never to another width's.
_DISPATCH_IDENTITY: ContextVar[Hashable | None] = ContextVar(
    "mtplx_compiled_dispatch_identity", default=None
)


def current_dispatch_identity() -> Hashable | None:
    """The specialization of the compiled call in progress, if any."""
    return _DISPATCH_IDENTITY.get()


@contextlib.contextmanager
def compiled_dispatch_scope(identity: Hashable) -> Iterator[None]:
    """Wrap one call into a compiled step (its trace on a first call, else a replay).

    Also restores the step-body flag when the call returns or raises. A traced
    body clears the flag on its way out, except when MLX abandons the traced
    Python frame: ``np.asarray`` on a tracer raises MLX's refusal as a C++
    exception through numpy's buffer protocol, and no ``finally`` between it
    and the compiled call runs. This frame does unwind (the exception reaches
    it as a Python exception), so the flag cannot stay set for every later
    call on the thread.
    """

    flag = _IN_COMPILED_STEP_BODY.get()
    token = _DISPATCH_IDENTITY.set(identity)
    try:
        yield
    finally:
        _DISPATCH_IDENTITY.reset(token)
        _IN_COMPILED_STEP_BODY.set(flag)


_COMPILE_TRACE_REFUSAL = "during function transformations"


def is_compile_trace_error(exc: BaseException) -> bool:
    """True for MLX's refusal to read an array while ``mx.compile`` traces it.

    ``.item()`` and ``mx.eval`` on a tracer raise ValueError("[eval]
    Attempting to eval an array during function transformations like compile
    or vmap is not allowed."). The tensor-offset adapters and the attention
    diagnostic tell a traced call from an eager one by that refusal alone; any
    other ValueError is a real fault and propagates.
    tests/test_promoted_paged_capacity_526.py pins the text against the
    installed MLX, so a reworded refusal fails there rather than turning every
    traced call into an error.
    """

    return isinstance(exc, ValueError) and _COMPILE_TRACE_REFUSAL in str(exc)
