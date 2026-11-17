"""Page-residency-aware row gathers for the PLE n-gram sidecar.

Pure Python + NumPy + ctypes: no MLX import, nothing here runs on the GPU,
and every function is safe on a worker thread (the memmaps are read-only and
``mincore``/``madvise`` go through ``ctypes.CDLL``, which releases the GIL).

Why this module exists
----------------------
``_SidecarGather.prepare_rows_np`` opens every big gather with a threaded
``os.pread`` warm pass, then does the real read as one NumPy fancy index over
the memmaps.  On a page-cache-warm table the warm pass is the whole cost:

===============================  ==========  =========
formulation (32,768 rows)        pread warm  memmap fx
===============================  ==========  =========
shipped (step<=64, 512 tasks)      164.8 ms    0.62 ms
**no warm, memmap only**           **0 ms**    0.44 ms
===============================  ==========  =========

That is ~5 us of GIL-contended Python per ``os.pread``, times three maps times
the chunk's unique rows -- the syscall count IS the cost, so no batching or
page-dedup formulation moves it (measured J/W4).

Dropping the warm pass unconditionally is not safe.  Demand-faulting an mmap
is FLAT at 1.40 GiB/s however many threads touch it, while pooled ``pread``
saturates at 12.9 GiB/s (measured 2026-07-11, `mmap-willneed-unwired`): on a
COLD table the fancy index degenerates into serial faults *with the GIL held*,
which stalls the generation thread as well.  The 2026-08-26 receipt puts that
at ~36 s serial vs ~7.5 s pooled for a cold 100k-token prefill.

So the choice is measured, not assumed: :func:`warm_decision` samples the rows
this gather will actually read and asks ``mincore(2)`` whether their pages are
already in core.  A false "cold" costs nothing (it takes the shipped pread
path); a false "warm" is the expensive mistake, so the threshold is high and
an unavailable/erroring probe answers "pread".
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
from functools import lru_cache

__all__ = [
    "ENV_FLAG",
    "MADVISE_ENV",
    "PREWARM_AT_LOAD_ENV",
    "PREWARM_ENV",
    "PREWARM_CHUNK_BYTES",
    "MINCORE_INCORE",
    "PAGE_SIZE",
    "RESIDENT_FRACTION_THRESHOLD",
    "SAMPLE_ROWS",
    "enabled",
    "gather_matrices",
    "last_prewarm",
    "madvise_choice",
    "mode_text",
    "apply_prewarm_choice",
    "cold_map_names",
    "prewarm_at_load_enabled",
    "prewarm_file",
    "prewarm_skipped",
    "prewarm_source",
    "record_prewarm",
    "free_memory_bytes",
    "parse_prewarm_mode",
    "plan_hot_runs",
    "plan_small_maps_first",
    "prewarm_mode_setting",
    "prewarm_prefix",
    "resolve_budget",
    "run_prewarm",
    "set_prewarm_reservation",
    "load_hotness_order",
    "hotness_path_for",
    "format_prewarm_plan",
    "format_prewarm_result",
    "prewarm_reservation",
    "read_runs",
    "resident_fraction",
    "touch_rows",
    "warm_decision",
]

ENV_FLAG = "MTPLX_QWEN4_PLE_FIRST_GATHER_EARLY"

_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"", "0", "false", "no", "off"})

PAGE_SIZE = os.sysconf("SC_PAGESIZE")
MINCORE_INCORE = 0x1

#: How many of the gather's rows to probe.  ~1.9 us per probe on the M5 Max
#: (measured 2026-09-02), so 256 probes is ~0.5 ms against the 165 ms warm
#: pass it decides -- and 256 draws put the sampling error well under the
#: margin between the two regimes this has to tell apart (a prewarmed table
#: probes at 1.00, a cold one at 0.00; there is no realistic middle).
SAMPLE_ROWS = 256

#: Accept the vectorised path only at essentially full residency.  At 1% cold
#: the fancy index pays ~650 serial faults (~40 ms) with the GIL held; below
#: that the pread pool is strictly better, and taking it is never wrong.
RESIDENT_FRACTION_THRESHOLD = 0.99


@lru_cache(maxsize=1)
def enabled() -> bool:
    """Resolve :data:`ENV_FLAG` once.  Unparseable values raise."""

    raw = (os.environ.get(ENV_FLAG) or "").strip().lower()
    if raw in _FALSE:
        return False
    if raw in _TRUE:
        return True
    accepted = sorted((_TRUE | _FALSE) - {""})
    raise ValueError(
        f"{ENV_FLAG} must be one of {accepted}, got {os.environ.get(ENV_FLAG)!r}"
    )


class _Libc:
    """``mincore(2)`` through ctypes, resolved once and never re-raised."""

    _instance: "_Libc | None" = None
    _resolved = False

    def __init__(self) -> None:
        name = ctypes.util.find_library("c")
        if name is None:
            raise OSError("cannot locate libc")
        lib = ctypes.CDLL(name, use_errno=True)
        lib.mincore.restype = ctypes.c_int
        lib.mincore.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_char_p,
        ]
        self.lib = lib

    @classmethod
    def get(cls) -> "_Libc | None":
        if not cls._resolved:
            cls._resolved = True
            try:
                cls._instance = cls()
            except Exception:
                cls._instance = None
        return cls._instance


def _row_address(memmap, row: int) -> tuple[int, int]:
    """(byte address, byte length) of one row of a 2-D numpy memmap."""

    stride = int(memmap.strides[0])
    return int(memmap.ctypes.data) + int(row) * stride, stride


def resident_fraction(memmap, rows, *, sample: int = SAMPLE_ROWS):
    """Fraction of the sampled rows' pages already in core, or None.

    ``rows`` is any 1-D integer array of row indices.  The sample is a fixed
    stride through it rather than a random draw: the answer must not move
    run to run for the same gather, or an A/B could not attribute a delta to
    the code instead of to the sampler.
    """

    libc = _Libc.get()
    if libc is None:
        return None
    total = int(len(rows))
    if total <= 0:
        return None
    step = max(1, total // max(1, int(sample)))
    pages = 0
    resident = 0
    buffer = ctypes.create_string_buffer(8)
    try:
        for i in range(0, total, step):
            address, length = _row_address(memmap, int(rows[i]))
            start = (address // PAGE_SIZE) * PAGE_SIZE
            end = -(-(address + length) // PAGE_SIZE) * PAGE_SIZE
            span = end - start
            count = span // PAGE_SIZE
            if count > len(buffer):
                buffer = ctypes.create_string_buffer(count)
            if libc.lib.mincore(ctypes.c_void_p(start), span, buffer) != 0:
                return None
            pages += count
            resident += sum(
                1 for byte in buffer.raw[:count] if byte & MINCORE_INCORE
            )
    except Exception:
        return None
    if pages <= 0:
        return None
    return resident / pages


def warm_decision(memmaps, rows, *, sample: int = SAMPLE_ROWS):
    """``("vectorized" | "pread", fraction_or_None)`` for this gather.

    Every map is probed, and the WORST answers: the gather reads all three,
    so one cold map is a cold gather.
    """

    worst = None
    for memmap in memmaps:
        fraction = resident_fraction(memmap, rows, sample=sample)
        if fraction is None:
            return "pread", None
        worst = fraction if worst is None else min(worst, fraction)
    if worst is None:
        return "pread", None
    if worst >= RESIDENT_FRACTION_THRESHOLD:
        return "vectorized", worst
    return "pread", worst


def cold_map_names(memmaps: dict, rows, *, sample: int = SAMPLE_ROWS):
    """Names of the maps this gather would fault on, or None when unknown.

    The warm pass is one ``pread`` per row PER MAP, and its cost is the syscall
    count, not the I/O (module docstring).  Residency is a per-map fact: once
    the pre-read has filled the two small maps (scales and biases, 3 GiB each)
    they stay in core, and only the 24 GiB weight map is cold for a text the
    engine has not seen.  Warming just the cold maps is a third of the
    syscalls.  An empty list means every map is in core (the vectorised
    gather needs no warm pass at all); None means residency could not be read
    and the caller warms everything, as before.
    """

    cold = []
    for name, memmap in memmaps.items():
        fraction = resident_fraction(memmap, rows, sample=sample)
        if fraction is None:
            return None
        if fraction < RESIDENT_FRACTION_THRESHOLD:
            cold.append(name)
    return cold


def gather_matrices(memmaps: dict, uniq, inverse, names) -> dict:
    """The row read itself -- one NumPy fancy index per map, then un-unique.

    This is the SAME expression the shipped gather runs (`_rows_matrices` and
    `prepare_rows_np` both call it), so "vectorised" and "pread" differ only
    in whether the pages were warmed first: the bytes are identical by
    construction, not by comparison.
    """

    import numpy as np

    return {
        name: np.ascontiguousarray(memmaps[name][uniq])[inverse]
        for name in names
    }


#: macOS fcntl command: ask the kernel to read a byte range ahead of use (struct radvisory
#: { off_t ra_offset; int ra_count; }). The call returns at once; a later read of the page
#: waits for the I/O already in flight instead of starting a fault of its own.
F_RDADVISE = 44
ADVISE_ENV = "MTPLX_NGRAM_ADVISE"


def advise_enabled() -> bool:
    """MTPLX_NGRAM_ADVISE (default on, macOS only): large cold row gathers are read ahead
    with F_RDADVISE from one thread instead of a pread a row from a thread pool. Cold
    n-gram rows (a 3.2e8 x 90 B table on the SSD) cost 3.4 us a row this way, 8.9-11.3
    through the 16-thread pread pool (its threads contend: advisories issued from the pool
    are no faster than the pool's reads)."""
    import sys

    if sys.platform != "darwin":
        return False
    raw = (os.environ.get(ADVISE_ENV) or "").strip().lower()
    return raw not in {"0", "off", "false", "no"}


def advise_rows(fd: int, metas, rows) -> int:
    """Read-ahead advisories for the rows' bytes (``metas``: (base offset, row bytes) per
    map). Returns the number issued; raises OSError when the kernel refuses (the caller
    falls back to reading the rows)."""
    import fcntl

    import numpy as np

    rec = np.zeros(len(rows), np.dtype([("off", "<i8"), ("cnt", "<i4"), ("pad", "<i4")]))
    issued = 0
    rows64 = np.asarray(rows, dtype=np.int64)
    for base, row_bytes in metas:
        rec["off"] = base + rows64 * int(row_bytes)
        rec["cnt"] = int(row_bytes)
        buf = rec.tobytes()
        for i in range(0, len(buf), 16):
            fcntl.fcntl(fd, F_RDADVISE, buf[i : i + 16])
        issued += len(rows64)
    return issued


def touch_rows(memmaps, rows, *, block: int = 1 << 18) -> int:
    """Fault ``rows`` of every map into the page cache, ascending, in blocks.

    The madvise(WILLNEED) equivalent for a hash-scattered row set: WILLNEED
    over the whole mapping would be readahead on rows this prompt never
    touches, and per-row advice is the syscall count the pread loop already
    proved is the cost.  Reading the rows themselves in ASCENDING order is
    what the kernel can actually coalesce, and it leaves exactly the pages a
    later chunk's fancy index wants.

    Returns the number of rows touched.  Blocked so the temporary copy stays
    bounded (a 262k-token prompt hashes to 4.2M rows = ~420 MB if read whole).
    """

    import numpy as np

    ordered = np.asarray(rows, dtype=np.int64).reshape(-1)
    if ordered.size == 0:
        return 0
    ordered = np.sort(ordered)
    step = max(1, int(block))
    for memmap in memmaps:
        for start in range(0, ordered.size, step):
            chunk = ordered[start : start + step]
            # `.sum()` keeps the read from being optimised away and costs one
            # pass over bytes already in L2; the point is the fault, not the
            # value.
            int(np.asarray(memmap[chunk]).view(np.uint8).sum(dtype=np.int64))
    return int(ordered.size)


# ---------------------------------------------------------------------------
# Mapping advice, and the load-time sequential prewarm
# ---------------------------------------------------------------------------

MADVISE_ENV = "MTPLX_QWEN4_NGRAM_MADVISE"

#: The official knob.  The n-gram pre-read is not an experiment any more: it
#: is on by default and it is what `mtplx serve --ngram-prewarm /
#: --no-ngram-prewarm` sets, so it lives in the MTPLX_* namespace with the
#: other runtime keys (mtplx/profiles.py MODEL_RUNTIME_ENV_OVERRIDE_KEYS)
#: rather than in the PR #391 benchmark namespace.
PREWARM_ENV = "MTPLX_NGRAM_PREWARM"

#: Deprecated alias, honoured with a one-line warning so branches and scripts
#: that already set it keep working.
PREWARM_AT_LOAD_ENV = "MTPLX_QWEN4_NGRAM_PREWARM_AT_LOAD"

#: 64 MiB, the same unit the PR #391 driver's ``--prewarm-ngram-table`` uses, so
#: the two receipts' GiB/s are directly comparable.
PREWARM_CHUNK_BYTES = 64 * 1024 * 1024

MADV_NORMAL = 0
MADV_RANDOM = 1
MADV_SEQUENTIAL = 2

_ADVICE = {
    "normal": MADV_NORMAL,
    "random": MADV_RANDOM,
    "sequential": MADV_SEQUENTIAL,
}


def madvise_choice() -> tuple[str, int]:
    """Which ``madvise`` advice the n-gram maps should carry, and why.

    The shipped advice is ``MADV_RANDOM``, on the argument that the rows are
    hash-scattered so readahead around a fault is wasted IO.  That argument is
    about MAPPING FAULTS only -- ``pread(2)`` does not consult it at all, so it
    never governed the pread pool the comment credits it to -- and under
    MTPLX_QWEN4_PLE_FIRST_GATHER_EARLY the mapping faults are exactly the two
    cases readahead helps: the ascending-order pre-touch of the whole prompt's
    rows, and the vectorised gather's residual faults on a partly-cold table.
    So the flag flips it to the kernel default.

    ``MTPLX_QWEN4_NGRAM_MADVISE=random|normal|sequential`` overrides either
    way, so the choice is A/B-able without a code change.  It is applied once,
    at construction, and recorded on the sidecar.
    """

    raw = (os.environ.get(MADVISE_ENV) or "").strip().lower()
    if raw:
        if raw not in _ADVICE:
            raise ValueError(
                f"{MADVISE_ENV} must be one of {sorted(_ADVICE)}, got {raw!r}"
            )
        return raw, _ADVICE[raw]
    name = "normal" if enabled() else "random"
    return name, _ADVICE[name]


def _parse_bool_env(key: str):
    """``True``/``False`` for a set key, ``None`` for an unset one; else raise."""

    raw = os.environ.get(key)
    if raw is None:
        return None
    text = raw.strip().lower()
    if text == "":
        return None
    if text in _FALSE:
        return False
    if text in _TRUE:
        return True
    accepted = sorted((_TRUE | _FALSE) - {""})
    raise ValueError(f"{key} must be one of {accepted}, got {raw!r}")


_DEPRECATION_WARNED = False


def prewarm_source() -> tuple[bool, str]:
    """``(enabled, source)`` for the n-gram pre-read, and what decided it.

    Precedence: :data:`PREWARM_ENV` (which is what the CLI stamps, so
    ``--ngram-prewarm`` wins over a shell-set value), then the deprecated
    PR #391 alias, then the default.

    Default ON, in ``auto`` mode.  The as-found page-cache state is what
    production actually serves at, and a benchmark harness reads the table
    itself (the PR #391 driver's ``--prewarm-ngram-table``) so its receipts
    never showed the difference -- while the daemon, which had no equivalent,
    did: the w22 window measured a 1.9 s vs 4.4 s first prefill chunk,
    perfectly concordant with the prewarm read's own throughput, and cold
    sidecar rows cost 56 vs 68.8 tok/s on decode.
    """

    mode, source = prewarm_mode_setting()
    return mode != "off", source


def _warn_deprecated_alias() -> None:
    global _DEPRECATION_WARNED

    if _DEPRECATION_WARNED:
        return
    _DEPRECATION_WARNED = True
    try:
        print(
            f"[mtplx] {PREWARM_AT_LOAD_ENV} is deprecated; "
            f"use {PREWARM_ENV} (or mtplx serve --ngram-prewarm off)",
            flush=True,
        )
    except (OSError, ValueError):
        pass


def prewarm_at_load_enabled() -> bool:
    """Whether to read the n-gram table sequentially at model load."""

    return prewarm_source()[0]


def mode_text(mode) -> str:
    """The canonical spelling of a parsed mode, for stamping into the env."""

    if isinstance(mode, tuple) and mode and mode[0] == "bytes":
        return f"{int(mode[1]) / 1024**3:.6g}gib"
    return str(mode)


def apply_prewarm_choice(cli_value) -> str:
    """Stamp an explicit CLI choice into :data:`PREWARM_ENV`; return the source.

    ``None`` means the flag was not given, so the environment (or the
    default) decides.  Stamping is what makes "CLI wins over env" true for
    the whole process, including the model load that happens later and any
    subprocess.  Booleans are accepted for the older ``--no-ngram-prewarm``
    spelling and mean ``all`` / ``off``.
    """

    if cli_value is None:
        return prewarm_mode_setting()[1]
    if isinstance(cli_value, bool):
        cli_value = "all" if cli_value else "off"
    mode = parse_prewarm_mode(cli_value, key="--ngram-prewarm")
    if mode is None:
        return prewarm_mode_setting()[1]
    os.environ[PREWARM_ENV] = mode_text(mode)
    return "cli"


def prewarm_file(path, *, chunk_bytes: int = PREWARM_CHUNK_BYTES) -> dict:
    """Read a whole file sequentially into the page cache; return the receipt.

    Sequential ``read(2)`` into ONE reusable buffer: the unified buffer cache
    is shared between the vnode's read path and every mapping of it, so pages
    landed this way are the pages ``mincore`` then reports resident through
    the memmaps -- which is what makes the vectorised gather engage at all.
    """

    import time

    size = os.path.getsize(path)
    buffer = memoryview(bytearray(int(chunk_bytes)))
    started = time.perf_counter()
    total = 0
    with open(path, "rb", buffering=0) as handle:
        while True:
            read = handle.readinto(buffer)
            if not read:
                break
            total += read
    elapsed = time.perf_counter() - started
    return {
        "path": str(path),
        "bytes": int(total),
        "file_bytes": int(size),
        "complete": bool(total == size),
        "seconds": float(elapsed),
        "chunk_bytes": int(chunk_bytes),
        "gib_per_s": (total / 1024**3) / elapsed if elapsed > 0 else None,
        "skipped_reason": None,
    }


def prewarm_skipped(reason: str) -> dict:
    """The same receipt shape for a prewarm that did not run.

    One shape either way: a reader that has to tell ``None`` (the field was
    never written) from "it ran and read nothing" is a reader that will get it
    wrong.
    """

    return {
        "path": None,
        "bytes": 0,
        "file_bytes": None,
        "complete": False,
        "seconds": 0.0,
        "chunk_bytes": None,
        "gib_per_s": None,
        "skipped_reason": str(reason),
    }


#: The last n-gram pre-read this process performed, published by
#: ``_SidecarGather.__init__`` and read by ``/health``.  A module-level fact
#: rather than a walk from the server down to the sidecar: the server would
#: have to know the model's internal shape to find it, and that walk is
#: exactly what made the PLE lookahead inert on 2026-09-01.
_LAST_PREWARM: dict = {}


def record_prewarm(receipt: dict, *, enabled: bool, source: str) -> dict:
    """Publish one pre-read receipt for ``/health`` and return it."""

    published = dict(receipt)
    published["enabled"] = bool(enabled)
    published["source"] = str(source)
    _LAST_PREWARM.clear()
    _LAST_PREWARM.update(published)
    return published


def last_prewarm() -> dict:
    """The published pre-read receipt, or an unknown-shaped one before load."""

    if _LAST_PREWARM:
        return dict(_LAST_PREWARM)
    enabled, source = prewarm_source()
    unknown = prewarm_skipped("no_model_loaded")
    unknown["enabled"] = bool(enabled)
    unknown["source"] = str(source)
    return unknown


# ---------------------------------------------------------------------------
# How much of the table to pre-read, and in what order
# ---------------------------------------------------------------------------

ORDER_ENV = "MTPLX_NGRAM_PREWARM_ORDER"

#: The name a model directory may carry next to ``ngram-table.safetensors``.
HOTNESS_FILENAME = "ngram-hotness.npy"

#: Headroom left untouched by an ``auto`` budget.  The pre-read competes with
#: the KV cache, the session bank and macOS itself for the same unified pool,
#: and being wrong here is a swap storm rather than a slow first token, so the
#: margin is a documented constant instead of a fraction: 6 GiB is roughly the
#: working set macOS keeps for the window server and the compositor plus one
#: 4 GiB MLX allocator cache round.
AUTO_MARGIN_BYTES = 6 * 1024**3

#: Modes with no numeric argument.
PREWARM_MODES = ("auto", "all", "off")

_LEGACY_TRUE = frozenset({"true", "yes", "on"})
_LEGACY_FALSE = frozenset({"false", "no", "none", "off"})
_GIB_SUFFIXES = ("gib", "gb", "g")


def parse_prewarm_mode(text, *, key: str = "MTPLX_NGRAM_PREWARM"):
    """``"auto" | "all" | "off" | ("bytes", n)`` from one setting's value.

    Grammar: ``auto`` (default), ``all``, ``off``, or a byte budget written
    as a plain number of GiB (``12``) or with a unit (``12GiB``, ``12G``).
    ``true``/``yes``/``on`` mean ``all`` and ``false``/``no``/``none`` mean
    ``off``, because that is what the boolean spelling used to mean; ``0``
    means ``off`` for the same reason a zero budget does.

    A bare ``1`` is ONE GiB, not "on".  The number grammar has to be
    consistent to be usable at all, and the setting is new enough
    (unreleased) that no one can be relying on the boolean reading of it --
    the deprecated PR #391 alias, which only ever had a boolean reading, is
    parsed separately by :func:`_parse_bool_env`.
    """

    raw = "" if text is None else str(text).strip().lower()
    if raw == "":
        return None
    if raw in PREWARM_MODES:
        return raw
    if raw in _LEGACY_TRUE:
        return "all"
    if raw in _LEGACY_FALSE:
        return "off"
    number = raw
    for suffix in _GIB_SUFFIXES:
        if number.endswith(suffix):
            number = number[: -len(suffix)].strip()
            break
    try:
        gib = float(number)
    except ValueError:
        raise ValueError(
            f"{key} must be one of {list(PREWARM_MODES)}, or a budget in GiB "
            f'(for example "12" or "12GiB"), got {text!r}'
        ) from None
    if gib < 0:
        raise ValueError(f"{key} budget must not be negative, got {text!r}")
    if gib == 0:
        return "off"
    return ("bytes", int(gib * 1024**3))


def free_memory_bytes() -> tuple[int, str]:
    """Reclaimable RAM right now, and how it was measured.

    ``vm_stat`` free + inactive + purgeable, per the plan's own definition.
    Two caveats, stated rather than hidden: Darwin counts purgeable pages
    inside active/inactive as well, so the sum is slightly optimistic, and
    ``speculative`` (file-cache readahead) is deliberately excluded even
    though it is reclaimable -- it is largely the very page cache a pre-read
    would be competing with.  :data:`AUTO_MARGIN_BYTES` absorbs both.
    """

    import re
    import subprocess

    try:
        out = subprocess.run(
            ["/usr/bin/vm_stat"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout
    except Exception as error:
        return 0, f"unavailable: {error!r}"
    header = re.search(r"page size of (\d+) bytes", out)
    page = int(header.group(1)) if header else PAGE_SIZE
    counts = {}
    for line in out.splitlines():
        match = re.match(r'^"?Pages ([a-z ]+?)"?:\s+(\d+)\.', line.strip())
        if match:
            counts[match.group(1).strip()] = int(match.group(2))
    wanted = ("free", "inactive", "purgeable")
    if not all(name in counts for name in wanted):
        return 0, f"unavailable: vm_stat missing {sorted(set(wanted) - set(counts))}"
    total = sum(counts[name] for name in wanted) * page
    return int(total), "vm_stat(free+inactive+purgeable)"


def resolve_budget(
    mode,
    *,
    table_bytes: int,
    free_bytes: int,
    reserved_bytes: int,
    margin_bytes: int = AUTO_MARGIN_BYTES,
) -> dict:
    """Bytes of the table to pre-read, with every input that decided it."""

    table_bytes = max(0, int(table_bytes))
    decision = {
        "mode": mode if isinstance(mode, str) else "bytes",
        "table_bytes": table_bytes,
        "free_bytes": int(free_bytes),
        "reserved_bytes": int(reserved_bytes),
        "margin_bytes": int(margin_bytes),
        "requested_bytes": None,
    }
    if mode == "off":
        decision["budget_bytes"] = 0
        return decision
    if mode == "all":
        decision["budget_bytes"] = table_bytes
        return decision
    if isinstance(mode, tuple) and mode and mode[0] == "bytes":
        decision["requested_bytes"] = int(mode[1])
        decision["budget_bytes"] = min(table_bytes, int(mode[1]))
        return decision
    if mode != "auto":
        raise ValueError(f"unknown prewarm mode {mode!r}")
    headroom = int(free_bytes) - int(reserved_bytes) - int(margin_bytes)
    decision["headroom_bytes"] = headroom
    decision["budget_bytes"] = max(0, min(table_bytes, headroom))
    return decision


def load_hotness_order(path):
    """Row ids in descending gather frequency, or None.

    The file is a plain ``.npy`` of int64 row ids, most-gathered first --
    built by ``tests/ngram_row_hotness.py``.  Unreadable or
    wrong-shaped files are ignored (the prefix order still warms something);
    they are never a reason a model fails to load.
    """

    import numpy as np

    if path is None:
        return None
    try:
        rows = np.load(str(path), allow_pickle=False)
    except Exception:
        return None
    rows = np.asarray(rows).reshape(-1)
    if rows.size == 0 or not np.issubdtype(rows.dtype, np.integer):
        return None
    return rows.astype(np.int64, copy=False)


def hotness_path_for(table_path, override=None):
    """Where the hotness file for this table lives, or None."""

    from pathlib import Path

    if override:
        candidate = Path(str(override))
        return candidate if candidate.exists() else None
    candidate = Path(str(table_path)).parent / HOTNESS_FILENAME
    return candidate if candidate.exists() else None


def _page_runs(offsets, lengths, *, page: int = PAGE_SIZE):
    """Coalesce byte ranges into sorted, page-aligned (offset, length) runs."""

    import numpy as np

    starts = (np.asarray(offsets, dtype=np.int64) // page) * page
    ends = -(-(np.asarray(offsets, dtype=np.int64) + np.asarray(lengths, dtype=np.int64)) // page) * page
    order = np.argsort(starts, kind="stable")
    starts, ends = starts[order], ends[order]
    runs = []
    run_start, run_end = int(starts[0]), int(ends[0])
    for start, end in zip(starts[1:].tolist(), ends[1:].tolist()):
        if start <= run_end:
            run_end = max(run_end, end)
        else:
            runs.append((run_start, run_end - run_start))
            run_start, run_end = start, end
    runs.append((run_start, run_end - run_start))
    return runs


def plan_hot_runs(rows, row_meta, budget_bytes: int, *, page: int = PAGE_SIZE):
    """Page-aligned read runs for the hottest ``rows`` that fit ``budget_bytes``.

    Returns ``(runs, rows_taken)``.

    At a given budget both orders warm the SAME number of pages -- a 16 KiB
    page holds ~200 of the table's ~100 byte rows, and the rows are
    hash-scattered, so a hot row costs a whole page either way.  What the
    hotness order changes is WHICH pages: the ones the model will actually
    gather, instead of whichever ones happen to sit at the front of the file.
    The price is random reads instead of sequential ones, so the caller logs
    the run count and the achieved GiB/s rather than assuming.

    The row count that fills the budget is SEARCHED, not estimated: coalescing
    makes the bytes-per-row wildly non-linear (adjacent hot rows share a page,
    scattered ones do not), and a single estimate left 97% of the budget
    unspent in the first cut of this function.
    """

    import numpy as np

    rows = np.asarray(rows, dtype=np.int64).reshape(-1)
    row_meta = tuple(row_meta)
    if rows.size == 0 or budget_bytes <= 0 or not row_meta:
        return [], 0

    def _cost(take: int):
        head = rows[:take]
        offsets = np.concatenate(
            [base + head * row_bytes for base, row_bytes in row_meta]
        )
        lengths = np.concatenate(
            [
                np.full(head.size, row_bytes, dtype=np.int64)
                for _, row_bytes in row_meta
            ]
        )
        runs = _page_runs(offsets, lengths, page=page)
        return runs, sum(length for _, length in runs)

    # One row must fit, or there is nothing to plan.
    runs, total = _cost(1)
    if total > budget_bytes:
        return [], 0
    # Double until it does not fit (or the rows run out), then bisect.
    low, best = 1, (runs, 1)
    high = None
    take = 2
    while take <= rows.size:
        runs, total = _cost(take)
        if total > budget_bytes:
            high = take
            break
        low, best = take, (runs, take)
        take *= 2
    if high is None:
        return best[0], best[1]
    while high - low > 1:
        mid = (low + high) // 2
        runs, total = _cost(mid)
        if total > budget_bytes:
            high = mid
        else:
            low, best = mid, (runs, mid)
    return best[0], best[1]


def plan_small_maps_first(
    map_extents,
    budget_bytes: int,
    *,
    page: int = PAGE_SIZE,
    chunk_bytes: int = PREWARM_CHUNK_BYTES,
):
    """Read runs that fill whole maps in ascending size order.

    Returns ``(runs, maps_complete)``.

    Row ids are hash-uniform, so a gather of never-seen rows costs one cold
    page per row in every map that is not resident.  A byte of budget spent
    on a map of ``size`` bytes therefore avoids ``1 / size`` cold reads per
    row, which makes the smallest map the best purchase and the file prefix
    nearly the worst one: the shipped table is laid out weight (23.8 GiB),
    scales (3.0 GiB), biases (3.0 GiB), so a 6.5 GiB prefix covers 27% of the
    weights and nothing else (2.73 cold reads per new row), while the same
    budget covers scales and biases completely (0.98 cold reads per new row).

    Runs are cut at ``chunk_bytes`` so the pread pool can take them side by
    side; the last map the budget reaches gets a prefix of what is left.
    """

    extents = sorted(
        ((int(offset), int(nbytes)) for offset, nbytes in map_extents if int(nbytes) > 0),
        key=lambda extent: (extent[1], extent[0]),
    )
    remaining = int(budget_bytes)
    runs: list[tuple[int, int]] = []
    complete = 0
    for offset, nbytes in extents:
        if remaining < page:
            break
        start = (offset // page) * page
        end = -(-(offset + nbytes) // page) * page
        take = min(end - start, (remaining // page) * page)
        if take <= 0:
            break
        if take >= end - start:
            complete += 1
        cursor = start
        while cursor < start + take:
            length = min(int(chunk_bytes), start + take - cursor)
            runs.append((cursor, length))
            cursor += length
        remaining -= take
    return runs, complete


def read_runs(fd: int, runs, *, submit=None) -> int:
    """Fault ``runs`` into the page cache with ``pread``; return bytes read."""

    total = 0
    chunk = 8 * 1024 * 1024

    def _read(run):
        offset, length = run
        read = 0
        while read < length:
            block = os.pread(fd, min(chunk, length - read), offset + read)
            if not block:
                break
            read += len(block)
        return read

    if submit is None:
        for run in runs:
            total += _read(run)
        return total
    futures = [submit(_read, run) for run in runs]
    for future in futures:
        total += int(future.result() or 0)
    return total


def prewarm_prefix(path, budget_bytes: int, *, chunk_bytes: int = PREWARM_CHUNK_BYTES) -> int:
    """Sequentially read the first ``budget_bytes`` of ``path``; return bytes."""

    if budget_bytes <= 0:
        return 0
    buffer = memoryview(bytearray(int(chunk_bytes)))
    total = 0
    with open(path, "rb", buffering=0) as handle:
        while total < budget_bytes:
            want = min(int(chunk_bytes), int(budget_bytes) - total)
            read = handle.readinto(buffer[:want])
            if not read:
                break
            total += read
    return total


#: KV/session reservation the server publishes before the model load, so the
#: ``auto`` budget can subtract memory that is spoken for but not yet
#: allocated.  Module-level for the same reason `last_prewarm` is: the sidecar
#: is five getattrs below the server and must not have to walk back up.
_RESERVATION: dict = {"bytes": 0, "source": "unset"}


def estimate_engine_growth_bytes(model_dir) -> tuple[int, str]:
    """Bytes the engine may still take after the weights: its budget minus the weights on disk.

    The pre-read fills the page cache, and the kernel evicts that cache as the
    engine grows toward its budget on a long prompt.  A pre-read larger than
    what survives that growth is wasted at best; at 206k on a 128 GB Mac it
    left the kernel with no free pages (2026-09-03 crash receipt).  So the
    reservation the pre-read subtracts is the engine's whole remaining growth,
    not the KV estimate alone: budget (the allocator cap when one is set,
    otherwise the machine's engine envelope) minus the weights the pack loads
    (every safetensors file except the streamed n-gram table).

    Returns ``(0, reason)`` when an input is unknown; the caller then keeps the
    KV estimate.
    """

    from pathlib import Path

    try:
        from mtplx.memory_plan import (
            NGRAM_TABLE_FILENAME,
            detect_total_ram_bytes,
            usable_engine_bytes,
        )

        total = detect_total_ram_bytes()
        if not total:
            return 0, "ram_unknown"
        budget = int(usable_engine_bytes(int(total)))
        cap = os.environ.get("MTPLX_MEMORY_LIMIT_BYTES", "").strip()
        if cap.isdigit() and int(cap) > 0:
            budget = int(cap)
        weights = 0
        for f in Path(model_dir).glob("*.safetensors"):
            if f.name == NGRAM_TABLE_FILENAME:
                continue
            weights += f.stat().st_size
        if weights <= 0:
            return 0, "weights_unknown"
        growth = max(0, budget - weights)
        return growth, f"engine_growth(budget={budget // 1024**3}GiB,weights={weights // 1024**3}GiB)"
    except Exception as error:
        return 0, f"unavailable: {error!r}"


def set_prewarm_reservation(reserved_bytes: int, source: str) -> None:
    _RESERVATION["bytes"] = max(0, int(reserved_bytes))
    _RESERVATION["source"] = str(source)


def prewarm_reservation() -> tuple[int, str]:
    return int(_RESERVATION["bytes"]), str(_RESERVATION["source"])


def prewarm_mode_setting() -> tuple[object, str]:
    """``(mode, source)`` for the pre-read, honouring the whole precedence.

    :data:`PREWARM_ENV` (which the CLI stamps) then the deprecated boolean
    alias then the default ``auto``.
    """

    mode = parse_prewarm_mode(os.environ.get(PREWARM_ENV), key=PREWARM_ENV)
    if mode is not None:
        return mode, "env"
    legacy = _parse_bool_env(PREWARM_AT_LOAD_ENV)
    if legacy is not None:
        _warn_deprecated_alias()
        return ("all" if legacy else "off"), "deprecated_env"
    return "auto", "default"


def run_prewarm(
    *,
    table_path,
    row_meta=(),
    fd=None,
    submit=None,
    order_override=None,
    map_extents=(),
) -> dict:
    """Pre-read as much of the n-gram table as the budget allows.

    Returns the full receipt; never raises.  The order is the hotness file
    when one exists (rows the model actually gathers, read as coalesced
    page-aligned runs).  Without one, a caller that names the byte extents
    of the table's maps gets the smallest maps first
    (:func:`plan_small_maps_first`), and anyone else gets the file prefix --
    at the same budget all three warm the same number of pages, so the only
    question is which ones.
    """

    import time

    mode, source = prewarm_mode_setting()
    try:
        table_bytes = os.path.getsize(table_path)
    except OSError as error:
        receipt = prewarm_skipped(repr(error))
        receipt.update({"mode": "unknown", "source": source})
        return receipt
    free_bytes, free_source = free_memory_bytes()
    reserved_bytes, reserved_source = prewarm_reservation()
    plan = resolve_budget(
        mode,
        table_bytes=table_bytes,
        free_bytes=free_bytes,
        reserved_bytes=reserved_bytes,
    )
    plan.update(
        {
            "source": source,
            "free_source": free_source,
            "reserved_source": reserved_source,
            "path": str(table_path),
            "file_bytes": table_bytes,
        }
    )
    budget = int(plan["budget_bytes"])
    if budget <= 0:
        reason = "disabled" if mode == "off" else "no_headroom"
        receipt = prewarm_skipped(reason)
        receipt.update(plan)
        receipt["order"] = "none"
        receipt["warmed_bytes"] = 0
        receipt["bytes"] = 0
        receipt["runs"] = 0
        return receipt

    if order_override is None:
        order_override = os.environ.get(ORDER_ENV) or None
    hotness = load_hotness_order(hotness_path_for(table_path, order_override))
    started = time.perf_counter()
    order = "prefix"
    runs = 0
    try:
        # A budget that covers the table has nothing to prioritise, and the
        # sequential read is strictly faster than the same pages fetched at
        # random, so full coverage always takes the prefix path.
        if budget >= table_bytes:
            hotness = None
        if hotness is not None and fd is not None and row_meta:
            plan_runs, rows_taken = plan_hot_runs(hotness, row_meta, budget)
            if plan_runs:
                order = "hotness"
                runs = len(plan_runs)
                warmed = read_runs(fd, plan_runs, submit=submit)
                plan["hot_rows"] = int(rows_taken)
            else:
                warmed = prewarm_prefix(table_path, budget)
        elif budget < table_bytes and fd is not None and len(tuple(map_extents)) > 1:
            plan_runs, maps_complete = plan_small_maps_first(map_extents, budget)
            order = "small_maps_first"
            runs = len(plan_runs)
            warmed = read_runs(fd, plan_runs, submit=submit)
            plan["maps_complete"] = int(maps_complete)
        else:
            warmed = prewarm_prefix(table_path, budget)
    except OSError as error:
        receipt = prewarm_skipped(repr(error))
        receipt.update(plan)
        receipt["order"] = order
        receipt["warmed_bytes"] = 0
        receipt["bytes"] = 0
        receipt["runs"] = runs
        return receipt
    elapsed = time.perf_counter() - started

    receipt = dict(plan)
    receipt.update(
        {
            "bytes": int(warmed),
            "warmed_bytes": int(warmed),
            "complete": bool(warmed >= table_bytes),
            "seconds": float(elapsed),
            "chunk_bytes": PREWARM_CHUNK_BYTES,
            "gib_per_s": (warmed / 1024**3) / elapsed if elapsed > 0 else None,
            "skipped_reason": None,
            "order": order,
            "runs": runs,
        }
    )
    return receipt


def format_prewarm_plan(receipt: dict) -> str:
    """The decision line: every input that chose the budget."""

    gib = 1024**3

    def _g(key):
        value = receipt.get(key)
        return "?" if value is None else f"{value / gib:.1f}"

    return (
        "[mtplx] n-gram table pre-read plan: "
        f"mode={receipt.get('mode')} source={receipt.get('source')} "
        f"table={_g('table_bytes')} GiB free={_g('free_bytes')} GiB "
        f"reserved={_g('reserved_bytes')} GiB margin={_g('margin_bytes')} GiB "
        f"budget={_g('budget_bytes')} GiB order={receipt.get('order', 'none')}"
    )


def format_prewarm_result(receipt: dict) -> str:
    """The result line, in the shape the startup log has always had."""

    if receipt.get("skipped_reason"):
        return (
            "[mtplx] n-gram table pre-read skipped: "
            f"{receipt['skipped_reason']}"
        )
    rate = receipt.get("gib_per_s")
    return (
        "[mtplx] n-gram table pre-read "
        f"{receipt.get('warmed_bytes', 0) / 1024**3:.1f} GiB in "
        f"{receipt.get('seconds', 0.0):.1f} s "
        f"({'?' if rate is None else f'{rate:.1f}'} GiB/s)"
    )


def estimate_kv_reservation_bytes(model_dir, context_tokens: int) -> tuple[int, str]:
    """KV+aux bytes the server will reserve for ``context_tokens``, from disk.

    The server's own ``MemoryPlan`` is the authority, but it is built ~350
    lines AFTER the model load -- and the pre-read happens INSIDE that load,
    so the number cannot be read from it.  What CAN be read before the load is
    every input the plan derives it from: `mtplx.memory_plan` is runtime-free
    and computes bytes/token straight out of ``config.json``.  So this is the
    same arithmetic on the same inputs, not a second policy:

        (dense KV bytes/token * quant factor + QSA aux bytes/token) * tokens

    Returns ``(0, reason)`` whenever any input is unknown; the caller then
    leans on :data:`AUTO_MARGIN_BYTES` instead of inventing a number.
    """

    import json
    from pathlib import Path

    tokens = int(context_tokens or 0)
    if tokens <= 0:
        return 0, "no_context_window"
    try:
        from mtplx.memory_plan import (
            KV_QUANT_BYTE_FACTOR,
            dense_kv_bytes_per_token_from_config,
            qsa_aux_bytes_per_token_from_config,
        )

        config = json.loads((Path(model_dir) / "config.json").read_text("utf-8"))
    except Exception as error:
        return 0, f"unavailable: {error!r}"
    per_token = dense_kv_bytes_per_token_from_config(config)
    if not per_token:
        return 0, "config_uninformative"
    quant = (
        os.environ.get("MTPLX_VLLM_METAL_PAGED_KV_QUANT")
        or os.environ.get("MTPLX_PAGED_KV_QUANT")
        or "off"
    ).strip().lower()
    factor = KV_QUANT_BYTE_FACTOR.get(quant, 1.0)
    aux = qsa_aux_bytes_per_token_from_config(config) or 0
    total = int(tokens * (per_token * factor + aux))
    return total, f"config(kv={quant},tokens={tokens})"
