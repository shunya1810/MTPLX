"""MLX process settings MTPLX needs in place before the Metal device exists.

MLX reads ``MLX_MAX_MB_PER_BUFFER`` once, when it creates its Metal device, and
from then on closes the open command buffer every time the distinct buffers the
buffer has touched add up to more than that many MiB (50 on a Max).  The rule
is there to bound the temporaries one command buffer can pin.  It also counts
persistent state.  A sparse-attention layer's key bank and value bank are
135 MB each at a 128K context, so every write into them and every gather out
of them closes a command buffer: about four per layer, twelve layers, every
decode round.

Receipt (2026-09-20, Flash-Next, M5 Max, 128K context, same tree, same night):
verify forward 45 to 52 ms per round with MLX's 50 MiB rule, 30.0 ms with the
rule lifted (27.5 ms at 4K); decode 45.6 to 54.3 tok/s against 65.9.  A Metal
timeline with one op per command buffer shows the cost as one stalled
submission of about 1 ms per sparse-attention layer per round.  At 4K and 16K
the banks are under the limit and nothing changes.

It is NOT on by default, because prefill temporaries are exactly what the rule
bounds.  Same night, founder's cell order (4K, 16K, 64K, 128K in one process),
rule lifted: process peak 103.7 GB at 16K against 92.0 GB, 103.9 GB at 64K
against 95.6 GB, and the decode gain at 128K shrinks to 45.6 -> 49.4 tok/s
once the process is that close to its memory limit.  With 1,024 MiB: 128K
decode 59.2 tok/s from a fresh process, peak 103.2 GB.  MLX reads the value
once per process, so it cannot be raised for decode and lowered for prefill;
until it can, this is an operator's choice for decode-heavy long-context
serving on a Mac with memory to spare, and the engine leaves MLX's default
alone.

M1 GPU family (2026-09-27, opt-s12): the default is on there, paired with
the prefill-side bound it was missing. Unset, an M1 host (CPU brand "Apple
M1", or ``MTPLX_M1_LONG_CONTEXT=1``) gets 1,000 MiB and 150 ops per command
buffer plus ``MTPLX_PREFILL_LAYER_EVAL_EVERY=4`` (an eval every 4 layers of an
eager prefill forward closes the buffer, so prefill transients stay bounded).
On an M1 Max, Qwen3.8-27B: the 4-row verify went from ~103 command buffers to
~8, verify 85 -> 78.5 ms at 2K and 99 -> 93 ms at 32K, decode +4-11%
(2K/32K/64K); the first-turn peak footprint stayed at 25.0/29.0/31.7 GB
(default 25.4/29.1/31.3) where the buffer bound alone had reached 40.5/44.3
GB; cold prefill +1-2%; identical text. ``MTPLX_M1_LONG_CONTEXT=0`` restores
MLX's defaults.

``MTPLX_MLX_COMMAND_BUFFER_MB`` sets it: a number of MiB (``mlx``, ``0`` or
``off`` and an unset variable all leave MLX's own default alone).  A value the operator set for
MLX directly (``MLX_MAX_MB_PER_BUFFER``) always wins.  This module imports
nothing from MLX and must stay that way: it runs from ``mtplx/__init__.py``
so that it is in place before the first GPU call of the process.
"""

from __future__ import annotations

import os

MLX_ENV = "MLX_MAX_MB_PER_BUFFER"
MLX_OPS_ENV = "MLX_MAX_OPS_PER_BUFFER"
OVERRIDE_ENV = "MTPLX_MLX_COMMAND_BUFFER_MB"
PREFILL_EVAL_ENV = "MTPLX_PREFILL_LAYER_EVAL_EVERY"
M1_GATE_ENV = "MTPLX_M1_LONG_CONTEXT"
M1_COMMAND_BUFFER_MB = 1000
M1_COMMAND_BUFFER_OPS = 150
M1_PREFILL_LAYER_EVAL_EVERY = 4
#: None leaves MLX's own default in place (see the receipts above).
DEFAULT_COMMAND_BUFFER_MB: int | None = None

_LEAVE_ALONE = frozenset({"0", "off", "false", "no", "mlx", "default"})

#: What this process was told, for /health: MLX has no getter for the value.
_APPLIED: dict[str, object] = {"value_mb": None, "source": "unset"}


def _host_is_m1() -> bool:
    """CPU brand "Apple M1*" via sysctl (no MLX import: the device must not exist yet)."""

    try:
        import ctypes
        import ctypes.util

        libc = ctypes.CDLL(ctypes.util.find_library("c"))
        size = ctypes.c_size_t(0)
        name = b"machdep.cpu.brand_string"
        if libc.sysctlbyname(name, None, ctypes.byref(size), None, 0) != 0 or not size.value:
            return False
        buf = ctypes.create_string_buffer(size.value)
        if libc.sysctlbyname(name, buf, ctypes.byref(size), None, 0) != 0:
            return False
        return buf.value.decode("utf-8", "replace").startswith("Apple M1")
    except Exception:
        return False


def m1_family_process(env=None) -> bool:
    """``MTPLX_M1_LONG_CONTEXT`` 1/0 wins; unset, the host CPU brand decides."""

    source = os.environ if env is None else env
    raw = str(source.get(M1_GATE_ENV, "")).strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return _host_is_m1()


def resolve_command_buffer_mb(env=None) -> tuple[int | None, str]:
    """``(MiB or None, source)`` for this environment; None leaves MLX alone."""

    source = os.environ if env is None else env
    direct = str(source.get(MLX_ENV, "")).strip()
    if direct:
        try:
            return int(direct), "operator:" + MLX_ENV
        except ValueError:
            return None, "operator:" + MLX_ENV + ":unparsed"
    raw = str(source.get(OVERRIDE_ENV, "")).strip().lower()
    if raw in _LEAVE_ALONE and raw:
        return None, "override:mlx_default"
    if raw:
        try:
            value = int(raw)
        except ValueError:
            return DEFAULT_COMMAND_BUFFER_MB, "default:unparsed_override"
        if value <= 0:
            return None, "override:mlx_default"
        return value, "override:" + OVERRIDE_ENV
    if m1_family_process(source):
        return M1_COMMAND_BUFFER_MB, "default:m1_family"
    if DEFAULT_COMMAND_BUFFER_MB is None:
        return None, "default:mlx_default"
    return DEFAULT_COMMAND_BUFFER_MB, "default"


def apply_mlx_process_defaults(env=None) -> dict[str, object]:
    """Put the command-buffer bound in the environment MLX will read.

    Idempotent, and it never overwrites a value the operator gave MLX.
    """

    target = os.environ if env is None else env
    value, source = resolve_command_buffer_mb(target)
    if value is not None and not source.startswith("operator:"):
        target[MLX_ENV] = str(int(value))
    if source == "default:m1_family":
        # The op bound and the prefill-side eval travel with the M1 default;
        # an operator's own value for either stays.
        if not str(target.get(MLX_OPS_ENV, "")).strip():
            target[MLX_OPS_ENV] = str(M1_COMMAND_BUFFER_OPS)
        if not str(target.get(PREFILL_EVAL_ENV, "")).strip():
            target[PREFILL_EVAL_ENV] = str(M1_PREFILL_LAYER_EVAL_EVERY)
    if env is None:
        _APPLIED.update({"value_mb": value, "source": source})
    return {"value_mb": value, "source": source}


def applied_command_buffer_mb() -> dict[str, object]:
    return dict(_APPLIED)


__all__ = [
    "DEFAULT_COMMAND_BUFFER_MB",
    "MLX_ENV",
    "MLX_OPS_ENV",
    "OVERRIDE_ENV",
    "PREFILL_EVAL_ENV",
    "m1_family_process",
    "applied_command_buffer_mb",
    "apply_mlx_process_defaults",
    "resolve_command_buffer_mb",
]
