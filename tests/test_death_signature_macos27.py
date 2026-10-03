"""The death signature on macOS 27: early compression of idle pages with a
large clean file cache is not thrash, while every recorded death shape still
trips. Readings are replayed byte for byte where the receipt has them."""

from __future__ import annotations

import pytest

from mtplx import system_memory as sm

GB = 1_000_000_000
GIB = 1024**3
MIB = 1024**2
RAM = 137_438_953_472  # 128 GB


def _reading(
    *,
    free: int,
    file_backed: int,
    wired: int,
    compressor: int,
    swap: int,
    level: int = 20,
    at_s: float = 0.0,
    total: int = RAM,
) -> sm.SystemMemory:
    return sm.SystemMemory(
        available_bytes=free + file_backed,
        total_bytes=total,
        level_percent=level,
        free_bytes=free,
        file_backed_bytes=file_backed,
        wired_bytes=wired,
        compressor_bytes=compressor,
        swap_used_bytes=swap,
        monotonic_s=at_s,
    )


# E1, 2026-09-29, macOS 27.0.1, 50de43bb serving Flash-Next Optimized Speed:
# a healthy 65,536-token cold prefill aborted as "death_signature" twice
# (bench/e1-base-long-0929-0422/00-base0927/server.log), swap flat both times.
E1_FIRST = (
    _reading(
        free=6_469_861_376,
        file_backed=20_122_599_424,
        wired=96_834_748_416,
        compressor=5_389_058_048,
        swap=274_268_160,
        level=23,
        at_s=0.0,
    ),
    _reading(
        free=2_414_084_096,
        file_backed=24_996_462_592,
        wired=94_539_022_336,
        compressor=6_238_814_208,
        swap=274_268_160,
        level=24,
        at_s=2.915,
    ),
)
E1_SECOND = (
    _reading(
        free=2_058_387_456,
        file_backed=24_076_009_472,
        wired=95_925_157_888,
        compressor=5_661_065_216,
        swap=240_713_728,
        level=24,
        at_s=0.0,
    ),
    _reading(
        free=1_445_068_800,
        file_backed=23_204_462_592,
        wired=95_912_755_200,
        compressor=6_464_536_576,
        swap=240_713_728,
        level=23,
        at_s=2.827,
    ),
)


@pytest.mark.parametrize("pair", [E1_FIRST, E1_SECOND], ids=["e1-2.4GB-free", "e1-1.45GB-free"])
def test_macos27_early_compression_with_a_big_file_cache_is_not_death(pair):
    before, after = pair
    # The readings are what tripped on 50de43bb: free under the abort floor
    # while the compressor grew faster than 256 MiB/s.
    _shed, abort = sm.reading_floors(after)
    assert after.free_bytes < abort
    growth = after.compressor_bytes - before.compressor_bytes
    assert growth / (after.monotonic_s - before.monotonic_s) >= 256 * MIB

    assert not sm.memory_thrashing(after, before)
    assert sm.system_pressure_level(after, previous=before) == 1


# E2d, 2026-09-29 05:0x, the agent replay on Flash-Next at the product
# defaults: both arms were refused mid-prefill as "death_signature" with
# 17.6 to 18.2 GB of file-backed pages and swap flat (the 507 bodies in
# receipts/ttft/replay-diag-{A-50de43bb,B-ttft-32c84b08}). B had the 512 MiB
# starved line: free 0.16 GB was under it.
E2D_B_32K = (
    _reading(
        free=158_171_136,
        file_backed=18_891_063_296,
        wired=96_960_315_392,
        compressor=5_821_431_808,
        swap=98_369_536,
        level=24,
        at_s=0.0,
    ),
    _reading(
        free=162_988_032,
        file_backed=18_221_350_912,
        wired=97_474_232_320,
        compressor=6_334_103_552,
        swap=98_369_536,
        level=23,
        at_s=1.313,
    ),
)
E2D_A_61K = (
    _reading(
        free=309_362_688,
        file_backed=19_672_399_872,
        wired=95_837_110_272,
        compressor=6_564_511_744,
        swap=105_840_640,
        level=24,
        at_s=0.0,
    ),
    _reading(
        free=579_796_992,
        file_backed=17_627_742_208,
        wired=96_829_997_056,
        compressor=7_125_827_584,
        swap=105_840_640,
        level=23,
        at_s=2.024,
    ),
)

# The replay's own vm_stat rows (vm.jsonl, one a second) around each arm's
# lowest free reading: (t, free, speculative, purgeable, file-backed, wired
# pages; compressor and swap bytes). The lowest is 7,004 free pages
# (109 MiB), 1.75 times the kernel's 4,000-page target.
_E2D_ROWS = {
    "B": [
        (6341.617, 133289, 2834, 1266, 1186209, 5809804, 5820809216, 98366914),
        (6342.638, 180124, 3202, 1601, 1193448, 3854064, 5820874752, 98366914),
        (6343.662, 24652, 3528, 1665, 1206579, 5900105, 5820874752, 98366914),
        (6344.673, 229405, 2654, 69, 1171270, 5788627, 5821431808, 98366914),
        (6345.689, 7295, 2751, 56, 1169331, 5940731, 5821431808, 98366914),
        (6346.702, 7163, 2487, 127, 1156319, 5953635, 5821431808, 98366914),
        (6347.722, 7004, 2279, 117, 1119274, 5967192, 6190350336, 98366914),
        (6348.355, 408561, 2264, 66, 1116426, 5654157, 5871878144, 98366914),
    ],
    "A": [
        (6042.618, 235072, 326, 1051, 1329846, 5710556, 6404964352, 105843261),
        (6043.634, 8170, 211, 1026, 1298127, 5977558, 6404964352, 105843261),
        (6044.654, 333698, 255, 1040, 1284211, 5679303, 6404964352, 105843261),
        (6045.673, 103685, 450, 1054, 1284503, 5855497, 6404849664, 105843261),
        (6046.690, 94360, 449, 1140, 1284531, 5862754, 6404849664, 105843261),
        (6047.708, 7735, 284, 1140, 1224271, 6001354, 6404849664, 105843261),
        (6048.721, 7415, 364, 1328, 1213949, 5956212, 6404833280, 105843261),
        (6049.738, 27173, 358, 1361, 1111808, 6092399, 6404833280, 105843261),
        (6050.748, 23089, 467, 1365, 1111996, 5977378, 6404833280, 105843261),
    ],
}


def _vm_row_reading(row) -> sm.SystemMemory:
    t, free, speculative, purgeable, file_backed, wired, compressor, swap = row
    page = 16384
    return _reading(
        free=(free + purgeable) * page,
        file_backed=max(0, file_backed - speculative) * page,
        wired=wired * page,
        compressor=compressor,
        swap=swap,
        at_s=t,
    )


def test_the_kernel_free_target_is_the_starved_line():
    # vm_page_free_target pages of hw.pagesize: 4,000 x 16 KiB on this Mac.
    assert 16 * MIB <= sm.starved_free_bytes() <= 256 * MIB


@pytest.mark.parametrize(
    "pair", [E2D_B_32K, E2D_A_61K], ids=["e2d-B-32k-0.16GB-free", "e2d-A-61k-0.58GB-free"]
)
def test_e2d_replay_readings_are_not_death(pair):
    before, after = pair
    shed, abort = sm.reading_floors(after)
    assert after.free_bytes < abort and after.available_bytes >= shed
    growth = after.compressor_bytes - before.compressor_bytes
    assert growth / (after.monotonic_s - before.monotonic_s) >= 256 * MIB
    assert after.free_bytes > sm.starved_free_bytes()

    assert not sm.memory_thrashing(after, before)
    assert sm.system_pressure_level(after, previous=before) == 1


@pytest.mark.parametrize("arm", ["B", "A"])
def test_e2d_vm_rows_never_trip_through_the_window(arm):
    window = sm.ReadingWindow()
    for row in _E2D_ROWS[arm]:
        reading = _vm_row_reading(row)
        assert not sm.memory_thrashing(reading, window.readings()), row
        window.add(reading)


def test_the_same_rows_trip_once_free_pages_fall_under_the_kernel_target():
    # The B rows with the last one's free pages at the 09-23 panic's 878:
    # the compressor step that was benign above the target is death under it.
    rows = list(_E2D_ROWS["B"][:-1])
    last = list(rows[-1])
    last[1] = 878
    rows[-1] = tuple(last)
    window = sm.ReadingWindow()
    tripped = False
    for row in rows:
        reading = _vm_row_reading(row)
        tripped = tripped or sm.memory_thrashing(reading, window.readings())
        window.add(reading)
    assert tripped


def test_macos27_reading_with_swap_growth_still_trips():
    before, after = E1_FIRST
    swapping = _reading(
        free=after.free_bytes,
        file_backed=after.file_backed_bytes,
        wired=after.wired_bytes,
        compressor=after.compressor_bytes,
        swap=before.swap_used_bytes + 512 * MIB,
        at_s=after.monotonic_s,
    )
    assert sm.memory_thrashing(swapping, before)


def test_the_0923_panic_trips_whatever_the_file_cache():
    # Panic memoryStatus 2026-09-23: 878 free pages (14 MB), 57.7 GB
    # compressed, one process at 103 GiB. The file cache was not recorded:
    # both a small and a large one must trip.
    for file_backed in (1 * GIB, 30 * GIB):
        before = _reading(
            free=900 * MIB,
            file_backed=file_backed,
            wired=90 * GIB,
            compressor=int(55.0 * GB),
            swap=2 * GIB,
            at_s=0.0,
        )
        after = _reading(
            free=878 * 16384,
            file_backed=file_backed,
            wired=90 * GIB,
            compressor=int(57.7 * GB),
            swap=2 * GIB,
            at_s=2.0,
        )
        assert sm.memory_thrashing(after, before), file_backed


def test_the_0903_panic_with_the_ngram_pre_read_in_the_file_cache_trips():
    # 2026-09-03: 110 GiB engine budget plus a 23.4 GiB n-gram pre-read (clean
    # file pages) plus a 206K prefill; free RAM 0.0 GB before the watchdog.
    before = _reading(
        free=400 * MIB, file_backed=int(23.4 * GIB), wired=100 * GIB,
        compressor=8 * GIB, swap=0, at_s=0.0,
    )
    after = _reading(
        free=16 * MIB, file_backed=int(23.4 * GIB), wired=100 * GIB,
        compressor=9 * GIB, swap=0, at_s=2.0,
    )
    assert sm.memory_thrashing(after, before)


def test_the_field_report_freeze_shape_trips():
    # 2026-09-26 field report: free 0.1 to 0.5 GiB, 2 GiB of file cache,
    # 92.2 GiB wired, the compressor growing 10 GiB in 30 s.
    before = _reading(
        free=int(0.4 * GIB), file_backed=2 * GIB, wired=int(92.2 * GIB),
        compressor=int(13.4 * GIB), swap=0, at_s=0.0,
    )
    after = _reading(
        free=int(0.3 * GIB), file_backed=2 * GIB, wired=int(92.2 * GIB),
        compressor=int(14.7 * GIB), swap=0, at_s=4.0,
    )
    assert sm.memory_thrashing(after, before)


def test_a_thin_supply_keeps_the_abort_floor_sensitivity():
    # Freeze 3 of the field report had 5.7 GiB free two seconds before the
    # Mac stopped, with little file cache. Once free pages fall under the
    # abort floor with the supply under the shed floor, compressor growth
    # trips exactly as before, well above the starved line.
    before = _reading(
        free=int(5.6 * GIB), file_backed=2 * GIB, wired=88 * GIB,
        compressor=4 * GIB, swap=0, at_s=0.0,
    )
    after = _reading(
        free=5 * GIB, file_backed=2 * GIB, wired=88 * GIB,
        compressor=5 * GIB, swap=0, at_s=2.0,
    )
    shed, abort = sm.reading_floors(after)
    assert after.free_bytes < abort and after.available_bytes < shed
    assert after.free_bytes > 512 * MIB
    assert sm.memory_thrashing(after, before)


# ---------------------------------------------------------------------------
# Runaway compression inside one prefill, and the per-chunk check that stops it

from types import SimpleNamespace  # noqa: E402

import mtplx.server.openai as srv  # noqa: E402


def test_the_runaway_line_is_an_eighth_of_ram():
    assert sm.compressor_runaway_bytes(128 * GIB) == 16 * GIB
    assert sm.compressor_runaway_bytes(64 * GIB) == 8 * GIB
    assert sm.compressor_runaway_bytes(16 * GIB) == 4 * GIB


def _prefill_guard(monkeypatch, readings):
    sequence = iter(readings)
    last = [None]

    def reader():
        try:
            last[0] = next(sequence)
        except StopIteration:
            pass
        return last[0]

    monkeypatch.setattr(sm, "_reader", reader)
    monkeypatch.setattr(srv, "_PREFILL_SYSTEM_CHECK_INTERVAL_S", 0.0)
    monkeypatch.setattr(
        srv,
        "_mlx_memory_stats_live",
        lambda: {"ok": True, "active_memory_bytes": 80 * GIB, "cache_memory_bytes": 0},
    )
    monkeypatch.setattr(srv, "phys_footprint_bytes", lambda *a, **k: 0)
    state = SimpleNamespace(dashboard=SimpleNamespace(), allow_swap=False)
    return srv._PrefillSystemGuard(state, chunk_reserve_bytes=2 * GIB)


def _trajectory(*, rate_gb_s: float, seconds: float, step_s: float = 1.0):
    """A cold prefill on the E1 Mac: the compressor grows at ``rate_gb_s``
    from 5.5 GB while 20+ GB of clean file cache keeps the supply healthy,
    free pages sit near 2 GB and swap stays flat."""

    out = []
    t = 0.0
    while t <= seconds + 1e-9:
        out.append(
            _reading(
                free=int(2.0 * GB),
                file_backed=int(21.0 * GB),
                wired=int(94.5 * GB),
                compressor=int(5.5 * GB + rate_gb_s * GB * t),
                swap=int(0.25 * GB),
                at_s=t,
            )
        )
        t += step_s
    return out


def test_the_2120_131k_runaway_is_stopped_well_before_27_gb(monkeypatch):
    # 2.12.0's 131,072-token cold prefill: 27 GB compressed in about a
    # minute with 20.6 GB still available and swap flat (run/guard.log
    # 04:29:54). At 0.45 GB/s the check stops it once the growth passes
    # 16 GiB, about 38 s in.
    readings = _trajectory(rate_gb_s=0.45, seconds=70)
    guard = _prefill_guard(monkeypatch, readings)
    tripped_at = None
    for reading in readings:
        if guard():
            tripped_at = reading.monotonic_s
            break
    assert tripped_at is not None
    assert guard.tripped["reason"] == "compressor_runaway"
    grown = (
        guard.tripped["system_memory"]["compressor_bytes"]
        - guard.tripped["previous_system_memory"]["compressor_bytes"]
    )
    assert 16 * GIB <= grown < 20 * GB
    assert tripped_at < 45.0
    error = srv._prefill_system_abort_exception(SimpleNamespace(), guard.tripped)
    assert error.status_code == 507
    assert "compressed" in error.detail["message"]


def test_a_served_64k_cold_prefill_is_not_stopped(monkeypatch):
    # The same Mac's 64K cold prefills: about 0.3 GB/s for 42 s, 12.6 GB in
    # all (E1's readings: 0.85 GB in 2.9 s), which 2.12.0 served safely.
    readings = _trajectory(rate_gb_s=0.3, seconds=42)
    guard = _prefill_guard(monkeypatch, readings)
    assert not any(guard() for _ in readings)
    assert guard.tripped is None


def test_e1_readings_pass_the_per_chunk_check(monkeypatch):
    for before, after in (E1_FIRST, E1_SECOND):
        guard = _prefill_guard(monkeypatch, [before, after])
        assert guard() is False
        assert guard() is False, guard.tripped


# ---------------------------------------------------------------------------
# The admission gives back the allocator pool before it narrows the chunk


def test_the_pool_goes_back_before_the_chunk_narrows(monkeypatch):
    from test_memguard_admission import _flash_next_state, _install, _Machine, _manager

    # The served profile's chunked prefill (the widths the admission prices).
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", "1")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE", "auto")
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL_LAYOUT", "auto")
    monkeypatch.setenv("MTPLX_SUSTAINED_DENSE_DECODE_MAX_CONTEXT", "131072")
    monkeypatch.setattr(srv, "_record_guard_event", lambda state, payload: None)
    manager = _manager()
    machine = _Machine(manager.bank, base_gib=80.0, cache_gib=0.0, host_gib=1.0)
    _install(monkeypatch, machine)
    supply = {"base": 0}

    def read():
        pool_given_back = pool_bytes - machine.cache
        available = supply["base"] + pool_given_back
        return sm.SystemMemory(
            available_bytes=available,
            total_bytes=RAM,
            level_percent=22,
            free_bytes=int(0.13 * GB),
            file_backed_bytes=available - int(0.13 * GB),
            wired_bytes=int(94.7 * GB),
            compressor_bytes=int(10.7 * GB),
            swap_used_bytes=int(0.28 * GB),
            monotonic_s=0.0,
        )

    monkeypatch.setattr(srv, "_read_system_memory", read)
    prompt = list(range(16_384))

    # Price both widths on a tight Mac with an empty pool (E1's 16K cell).
    pool_bytes = 0
    supply["base"] = 12 * GIB
    probe = srv._prefill_admission_shed(
        _flash_next_state(manager),
        prompt_ids=prompt,
        session_bank=manager.bank,
        session_id=None,
        prefill_chunk_tokens=4096,
    )
    assert probe is not None and probe["prefill_chunk_requested"] == 4096
    growth = {int(k): int(v) for k, v in probe["growth_by_chunk"].items()}
    shed_4096, _abort = sm.admission_floors(read(), growth[4096])
    assert growth[2048] < growth[4096] - GIB

    # The wide chunk is short by 1 GiB on the Mac's line and the pool holds 4.
    pool_bytes = 4 * GIB
    machine.cache = pool_bytes
    supply["base"] = growth[4096] + shed_4096 - GIB
    receipt = srv._prefill_admission_shed(
        _flash_next_state(manager),
        prompt_ids=prompt,
        session_bank=manager.bank,
        session_id=None,
        prefill_chunk_tokens=4096,
    )

    # On 50de43bb: "narrower_prefill_chunk" at 2,048 rows, the pool untouched.
    assert machine.cache == 0
    assert receipt["action"] == "prefill_admission_pool_clear"
    assert receipt["reclamation_steps"] == ["allocator_pool"]
    assert receipt["prefill_chunk_tokens"] == 4096
    assert receipt["prefill_chunk_requested"] == 4096


def test_a_pool_clear_that_fails_is_reported_not_dropped(monkeypatch):
    # The review of 4c9da1ba: when mx.clear_cache() raised here and the
    # narrower chunk already fit, the error was dropped: the receipt read as
    # an ordinary narrowing and the guard's health stayed clean.
    import mlx.core as mx

    from test_memguard_admission import _flash_next_state, _install, _Machine, _manager

    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", "1")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE", "auto")
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL_LAYOUT", "auto")
    monkeypatch.setenv("MTPLX_SUSTAINED_DENSE_DECODE_MAX_CONTEXT", "131072")
    monkeypatch.setattr(srv, "_record_guard_event", lambda state, payload: None)
    manager = _manager()
    machine = _Machine(manager.bank, base_gib=80.0, cache_gib=0.0, host_gib=1.0)
    _install(monkeypatch, machine)
    supply = {"base": 0}

    def read():
        return sm.SystemMemory(
            available_bytes=supply["base"],
            total_bytes=RAM,
            level_percent=22,
            free_bytes=int(0.13 * GB),
            file_backed_bytes=supply["base"] - int(0.13 * GB),
            wired_bytes=int(94.7 * GB),
            compressor_bytes=int(10.7 * GB),
            swap_used_bytes=int(0.28 * GB),
            monotonic_s=0.0,
        )

    monkeypatch.setattr(srv, "_read_system_memory", read)
    prompt = list(range(16_384))
    supply["base"] = 12 * GIB
    probe = srv._prefill_admission_shed(
        _flash_next_state(manager),
        prompt_ids=prompt,
        session_bank=manager.bank,
        session_id=None,
        prefill_chunk_tokens=4096,
    )
    growth = {int(k): int(v) for k, v in probe["growth_by_chunk"].items()}
    shed_4096, _abort = sm.admission_floors(read(), growth[4096])

    def failing_clear():
        raise RuntimeError("clear_cache failed")

    monkeypatch.setattr(mx, "clear_cache", failing_clear)
    machine.cache = 4 * GIB
    supply["base"] = growth[4096] + shed_4096 - GIB
    state = _flash_next_state(manager)
    receipt = srv._prefill_admission_shed(
        state,
        prompt_ids=prompt,
        session_bank=manager.bank,
        session_id=None,
        prefill_chunk_tokens=4096,
    )

    assert receipt["prefill_chunk_tokens"] == 2048
    assert receipt["cache_cleared"] is False
    assert "clear_cache failed" in receipt["cache_clear_error"]
    assert receipt["guard_degraded"] is True
    health = srv._memory_guard_health(state)
    assert health["guard_degraded"] is True
    assert any(
        row["where"] == "prefill_admission_reclamation" for row in health["degraded"]
    )


# ---------------------------------------------------------------------------
# Before the per-chunk check refuses a request, the engine gives back its own
# reusable memory and reads the Mac again (E2d, 2026-09-29: a request was
# refused while idle session snapshots and the allocator pool were held).


def _shed_guard(monkeypatch, readings_by_bank_bytes, *, pool_bytes=0):
    """A per-chunk check on a real SessionBank: the Mac's reading follows
    what the bank still holds (``readings_by_bank_bytes(bank_bytes)``)."""

    from pathlib import Path

    from mtplx.engine_session import EngineSessionManager
    from mtplx.session_bank import SessionBank

    manager = EngineSessionManager(
        bank=SessionBank(max_entries=64, max_bytes=60 * GIB, per_session_max_bytes=30 * GIB),
        idle_ttl_s=3600,
    )
    runtime = SimpleNamespace(model_path=Path("models/example"), mtp_enabled=True)
    for session_id, tokens, nbytes in (
        ("idle-16k-a", range(1, 16_385), 3 * GIB),
        ("idle-16k-b", range(20_000, 36_384), 3 * GIB),
        ("generating", range(50_000, 82_768), 6 * GIB),
    ):
        entry = manager.bank.put(
            runtime=runtime,
            token_ids=list(tokens),
            cache=[],
            logits=None,
            hidden=None,
            session_id=session_id,
            nbytes_override=nbytes,
        )
        assert entry is not None
    generating = manager.get_or_create("generating")
    assert generating.try_begin_generation()
    pool = {"bytes": int(pool_bytes)}
    clock = {"t": 0.0}

    def read():
        clock["t"] += 1.0
        return readings_by_bank_bytes(manager.bank.total_nbytes, pool["bytes"], clock["t"])

    def clear_cache():
        pool["bytes"] = 0

    import mlx.core as mx

    monkeypatch.setattr(mx, "clear_cache", clear_cache)
    monkeypatch.setattr(srv, "_read_system_memory", read)
    monkeypatch.setattr(srv, "_PREFILL_SYSTEM_CHECK_INTERVAL_S", 0.0)
    monkeypatch.setattr(
        srv,
        "_mlx_memory_stats_live",
        lambda: {
            "ok": True,
            "active_memory_bytes": 80 * GIB,
            "cache_memory_bytes": pool["bytes"],
        },
    )
    monkeypatch.setattr(srv, "phys_footprint_bytes", lambda *a, **k: 0)
    state = SimpleNamespace(dashboard=SimpleNamespace(), allow_swap=False, sessions=manager)
    guard = srv._PrefillSystemGuard(state, chunk_reserve_bytes=2 * GIB)
    return guard, manager, generating


def _supply_reading(available, *, free=None, compressor=int(6.0 * GB), t=0.0):
    free = int(0.2 * GB) if free is None else int(free)
    return _reading(
        free=free,
        file_backed=max(0, int(available) - free),
        wired=int(94.0 * GB),
        compressor=compressor,
        swap=int(0.1 * GB),
        at_s=t,
    )


def test_idle_state_goes_back_before_a_request_is_refused(monkeypatch):
    # Under the abort floor with the pool and two idle 16K sessions held;
    # what the release gives back reaches the Mac's supply.
    def read(bank_bytes, pool_bytes, t):
        freed = 12 * GIB - bank_bytes
        return _supply_reading(3 * GIB + freed, t=t)

    guard, manager, generating = _shed_guard(monkeypatch, read, pool_bytes=GIB)
    try:
        assert guard() is False
    finally:
        generating.end_generation()
    assert guard.tripped is None
    assert guard.shed["request_continued"] is True
    assert guard.shed["reason_before"] == "under_abort_floor"
    assert guard.shed["pool_bytes"] == GIB
    assert set(guard.shed["released_sessions"]) <= {"idle-16k-a", "idle-16k-b"}
    assert guard.shed["released_bytes"] >= 3 * GIB
    # Never the conversation that is generating.
    assert manager.bank.has_session_entries("generating")


def test_a_mac_still_short_after_the_shed_is_refused_with_its_receipt(monkeypatch):
    def read(bank_bytes, pool_bytes, t):
        return _supply_reading(2 * GIB, t=t)

    guard, manager, generating = _shed_guard(monkeypatch, read, pool_bytes=GIB)
    try:
        assert guard() is True
    finally:
        generating.end_generation()
    tripped = guard.tripped
    assert tripped["reason"] == "under_abort_floor"
    shed = tripped["shed_before_abort"]
    assert shed["request_continued"] is False
    assert shed["pool_bytes"] == GIB
    assert shed["released_bytes"] >= 3 * GIB
    assert manager.bank.has_session_entries("generating")
    # One shed per request: the next check does not release again.
    assert guard() is True


def test_starved_free_pages_that_recover_after_the_shed_let_the_request_go_on(
    monkeypatch,
):
    # Free pages under the kernel's target while the compressor grows fast:
    # the death signature. The idle sessions' 6 GiB go back to the free
    # list, the next reading is above the target, and the request goes on.
    def read(bank_bytes, pool_bytes, t):
        freed = 12 * GIB - bank_bytes
        free = 16 * MIB + freed
        compressor = int(6.0 * GB) + (int(0.6 * GB) if t >= 2.0 else 0)
        return _supply_reading(20 * GIB + freed, free=free, compressor=compressor, t=t)

    guard, manager, generating = _shed_guard(monkeypatch, read)
    try:
        assert guard() is False  # first reading: nothing to compare yet
        assert guard() is False
    finally:
        generating.end_generation()
    assert guard.shed["reason_before"] == "death_signature"
    assert guard.shed["request_continued"] is True
    assert guard.shed["released_bytes"] >= 3 * GIB


# ---------------------------------------------------------------------------
# The 2026-09-29 review of 4c9da1ba: the relaxed signature must still catch a
# dangerous Mac. Runaway compression is measured across a run of requests,
# and a Mac already holding a quarter of its RAM compressed (or swapping at
# free pages under the kernel's target) is stopped however slowly it got
# there. Each of these passed every check on 4c9da1ba and 3863e9d8.

import json  # noqa: E402
from pathlib import Path  # noqa: E402


def _requests_sharing_a_server(monkeypatch, requests):
    """Run each request's readings through its own per-chunk check, all on
    one server (one ``CompressorEpisode``). Returns (request index, reading,
    tripped receipt) for the first trip, or None."""

    state = SimpleNamespace(dashboard=SimpleNamespace(), allow_swap=False)
    current = {"reading": None}
    monkeypatch.setattr(sm, "_reader", lambda: current["reading"])
    monkeypatch.setattr(srv, "_PREFILL_SYSTEM_CHECK_INTERVAL_S", 0.0)
    monkeypatch.setattr(
        srv,
        "_mlx_memory_stats_live",
        lambda: {"ok": True, "active_memory_bytes": 80 * GIB, "cache_memory_bytes": 0},
    )
    monkeypatch.setattr(srv, "phys_footprint_bytes", lambda *a, **k: 0)
    for index, readings in enumerate(requests):
        guard = srv._PrefillSystemGuard(state, chunk_reserve_bytes=2 * GIB)
        for reading in readings:
            current["reading"] = reading
            if guard():
                return index, reading, guard.tripped
    return None


def _compressing_request(*, start_s, start_gib, grow_gib=10, seconds=32, free=GIB,
                         file_backed=16 * GIB, swap=int(0.1 * GB)):
    """One request's prefill that compresses ``grow_gib`` over ``seconds``
    at 1 GiB free and 16 GiB of file cache, swap flat (the review's shape)."""

    return [
        _reading(
            free=free,
            file_backed=file_backed,
            wired=88 * GIB,
            compressor=int((start_gib + grow_gib * i / seconds) * GIB),
            swap=swap,
            at_s=start_s + i,
        )
        for i in range(seconds + 1)
    ]


def test_five_requests_that_each_compress_10_gib_are_stopped_before_the_second_ends(
    monkeypatch,
):
    # 5 GiB compressed at the start, each request adds 10 GiB in 32 s with a
    # 2 s turn between them. On 4c9da1ba none of the five tripped and the
    # Mac reached 55 GiB compressed.
    requests = []
    start_s = 0.0
    for n in range(5):
        requests.append(_compressing_request(start_s=start_s, start_gib=5 + 10 * n))
        start_s += 34.0
    trip = _requests_sharing_a_server(monkeypatch, requests)
    assert trip is not None
    index, reading, tripped = trip
    assert index == 1
    assert tripped["reason"] == "compressor_runaway"
    assert reading.compressor_bytes <= 5 * GIB + 17 * GIB
    grown = (
        tripped["system_memory"]["compressor_bytes"]
        - tripped["previous_system_memory"]["compressor_bytes"]
    )
    assert grown >= 16 * GIB
    error = srv._prefill_system_abort_exception(SimpleNamespace(), tripped)
    assert error.status_code == 507
    assert "of prefills" in error.detail["message"]


def test_the_same_requests_spread_over_the_afternoon_meet_the_full_line(monkeypatch):
    # Requests ten minutes apart each start a new run (the desktop's own
    # compression between them is not charged to one request), so the
    # quarter-of-RAM line is what stops the third, at 32 GiB compressed.
    requests = [
        _compressing_request(start_s=600.0 * n, start_gib=5 + 10 * n) for n in range(5)
    ]
    trip = _requests_sharing_a_server(monkeypatch, requests)
    assert trip is not None
    index, reading, tripped = trip
    assert index == 2
    assert tripped["reason"] == "compressor_full"
    assert 32 * GIB <= reading.compressor_bytes < 33 * GIB
    error = srv._prefill_system_abort_exception(SimpleNamespace(), tripped)
    assert error.status_code == 507
    assert "already holds" in error.detail["message"]


def test_a_run_is_charged_its_net_compression_not_every_requests_sum(monkeypatch):
    # Each request compresses 10 GiB and the kernel gives 9 back before the
    # next: 40 GiB compressed in all, 13 GiB net, nothing trips.
    requests = []
    for n in range(4):
        requests.append(_compressing_request(start_s=40.0 * n, start_gib=5 + n))
    assert _requests_sharing_a_server(monkeypatch, requests) is None


def test_compression_given_back_lowers_the_mark():
    episode = sm.CompressorEpisode()
    first = _reading(free=GIB, file_backed=16 * GIB, wired=88 * GIB,
                     compressor=10 * GIB, swap=0, at_s=0.0)
    lower = _reading(free=GIB, file_backed=16 * GIB, wired=88 * GIB,
                     compressor=4 * GIB, swap=0, at_s=40.0)
    higher = _reading(free=GIB, file_backed=16 * GIB, wired=88 * GIB,
                      compressor=21 * GIB, swap=0, at_s=70.0)
    assert episode.note(first) is first
    assert episode.note(lower) is lower
    assert episode.note(higher) is lower
    assert sm.compressor_runaway(higher, lower)


@pytest.mark.parametrize("file_backed", [1 * GIB, 30 * GIB])
def test_878_free_pages_with_58_gb_compressed_and_swap_creeping_is_stopped(
    monkeypatch, file_backed
):
    # The review's second counterexample on the real per-chunk check: both
    # calls returned False on 4c9da1ba.
    readings = [
        _reading(
            free=878 * 16384,
            file_backed=file_backed,
            wired=90 * GIB,
            compressor=58 * GB,
            swap=2 * GIB + 160 * MIB * i,
            at_s=10.0 * i,
        )
        for i in range(2)
    ]
    trip = _requests_sharing_a_server(monkeypatch, [readings])
    assert trip is not None
    _index, _reading_at, tripped = trip
    assert tripped["reason"] == "compressor_full"
    # The pressure loop reads it as critical from one reading.
    assert sm.system_pressure_level(readings[0]) == 4


def test_starved_free_pages_with_swap_creeping_is_the_death_signature():
    # Under the full line, slow swap growth at free pages under the kernel's
    # target still trips: 160 MiB in 10 s.
    before = _reading(
        free=878 * 16384, file_backed=20 * GIB, wired=90 * GIB,
        compressor=10 * GIB, swap=2 * GIB, at_s=0.0,
    )
    after = _reading(
        free=878 * 16384, file_backed=20 * GIB, wired=90 * GIB,
        compressor=10 * GIB, swap=2 * GIB + 160 * MIB, at_s=10.0,
    )
    assert not sm.compressor_full(after)
    assert sm.memory_thrashing(after, before)
    # Above the kernel's target the same creep is not (the E2d replays read
    # 109 MiB free at their lowest, with swap flat).
    above = [
        _reading(
            free=200 * MIB, file_backed=20 * GIB, wired=90 * GIB,
            compressor=10 * GIB, swap=r.swap_used_bytes, at_s=r.monotonic_s,
        )
        for r in (before, after)
    ]
    assert not sm.memory_thrashing(above[1], above[0])


def test_the_full_line_is_a_quarter_of_ram_and_never_under_8_gib():
    assert sm.compressor_full_bytes(128 * GIB) == 32 * GIB
    assert sm.compressor_full_bytes(64 * GIB) == 16 * GIB
    assert sm.compressor_full_bytes(16 * GIB) == 8 * GIB
    healthy = _reading(
        free=int(0.5 * GB), file_backed=18 * GIB, wired=91 * GIB,
        compressor=int(7.9 * GB), swap=int(0.1 * GB),
    )
    assert not sm.compressor_full(healthy)
    # Free pages above the abort floor: not full, whatever is compressed.
    roomy = _reading(
        free=12 * GIB, file_backed=18 * GIB, wired=60 * GIB,
        compressor=40 * GIB, swap=0,
    )
    assert not sm.compressor_full(roomy)


def test_a_steady_heavy_desktop_is_not_full_but_one_still_compressing_is():
    # 40 GiB compressed on a 128 GB Mac with 1 GiB free (under the abort
    # floor, above the kernel's target), nothing moving: a heavy desktop
    # that copes, not refused and not critical.
    steady = [
        _reading(free=GIB, file_backed=10 * GIB, wired=70 * GIB,
                 compressor=40 * GIB, swap=int(0.5 * GB), at_s=float(t))
        for t in (0.0, 5.0, 10.0)
    ]
    assert not sm.compressor_full(steady[-1], steady[:-1])
    assert sm.system_pressure_level(steady[-1], steady[:-1]) < 4
    # The same Mac while a request compresses 512 MiB more in 5 s, or swaps
    # 64 MiB: still losing ground at a quarter of RAM compressed.
    compressing = _reading(free=GIB, file_backed=10 * GIB, wired=70 * GIB,
                           compressor=40 * GIB + 512 * MIB, swap=int(0.5 * GB),
                           at_s=15.0)
    swapping = _reading(free=GIB, file_backed=10 * GIB, wired=70 * GIB,
                        compressor=40 * GIB, swap=int(0.5 * GB) + 64 * MIB,
                        at_s=15.0)
    for moving in (compressing, swapping):
        assert sm.compressor_full(moving, steady)
        assert sm.system_pressure_level(moving, steady) == 4


def _e2d_rows():
    path = Path(__file__).parent / "fixtures" / "e2d_replay_vm_rows.json"
    data = json.loads(path.read_text())
    total = int(data["total_bytes"])
    out = {}
    for arm, rows in data["arms"].items():
        out[arm] = [
            _reading(
                free=int(free), file_backed=int(file_backed), wired=int(wired),
                compressor=int(compressor), swap=int(swap), at_s=float(t), total=total,
            )
            for t, free, file_backed, wired, compressor, swap in rows
        ]
    return out


@pytest.mark.parametrize("arm", ["A", "B"])
def test_the_whole_e2d_replay_passes_every_mac_line_as_one_run(arm):
    # Every vm_stat row of each arm's replay (one a second, 106 and 146
    # rows), read as one run of prefills: no death signature, no runaway,
    # not full.
    episode = sm.CompressorEpisode()
    window = sm.ReadingWindow()
    for reading in _e2d_rows()[arm]:
        base = episode.note(reading)
        assert not sm.memory_thrashing(reading, window.readings()), reading
        assert not sm.compressor_runaway(reading, base), reading
        assert not sm.compressor_full(reading), reading
        assert sm.system_pressure_level(reading, window.readings()) < 4
        window.add(reading)


def test_a_new_run_starts_after_five_quiet_minutes():
    episode = sm.CompressorEpisode()
    first = _reading(free=GIB, file_backed=16 * GIB, wired=88 * GIB,
                     compressor=5 * GIB, swap=0, at_s=0.0)
    later = _reading(free=GIB, file_backed=16 * GIB, wired=88 * GIB,
                     compressor=20 * GIB, swap=0, at_s=301.0)
    assert episode.note(first) is first
    assert episode.note(later) is later
    soon = _reading(free=GIB, file_backed=16 * GIB, wired=88 * GIB,
                    compressor=30 * GIB, swap=0, at_s=400.0)
    assert episode.note(soon) is later
    assert not sm.compressor_runaway(soon, later)


def _each_request_on_one_server(monkeypatch, requests):
    """Like _requests_sharing_a_server, but every request runs: returns, per
    request, None when it was served or (reading, reason) where it stopped."""

    state = SimpleNamespace(dashboard=SimpleNamespace(), allow_swap=False)
    current = {"reading": None}
    monkeypatch.setattr(sm, "_reader", lambda: current["reading"])
    monkeypatch.setattr(srv, "_PREFILL_SYSTEM_CHECK_INTERVAL_S", 0.0)
    monkeypatch.setattr(
        srv,
        "_mlx_memory_stats_live",
        lambda: {"ok": True, "active_memory_bytes": 80 * GIB, "cache_memory_bytes": 0},
    )
    monkeypatch.setattr(srv, "phys_footprint_bytes", lambda *a, **k: 0)
    outcomes = []
    for readings in requests:
        guard = srv._PrefillSystemGuard(state, chunk_reserve_bytes=2 * GIB)
        outcome = None
        for reading in readings:
            current["reading"] = reading
            if guard():
                outcome = (reading, guard.tripped["reason"])
                break
        outcomes.append(outcome)
    return outcomes


def test_a_refusal_ends_the_run_and_the_full_line_bounds_the_retries(monkeypatch):
    # The second request is refused at 21.25 GiB. Its retry starts a new run
    # there instead of being refused at its first chunk for as long as the
    # pages stay compressed (a client retrying inside five minutes would
    # otherwise never be served), and a Mac that keeps compressing meets the
    # quarter-of-RAM line: the fourth request stops at 32 GiB.
    refused_at = 15 + 10 * 20 / 32
    requests = [
        _compressing_request(start_s=0.0, start_gib=5),
        _compressing_request(start_s=34.0, start_gib=15),
        _compressing_request(start_s=68.0, start_gib=refused_at),
        _compressing_request(start_s=102.0, start_gib=refused_at + 10),
    ]
    outcomes = _each_request_on_one_server(monkeypatch, requests)
    assert outcomes[0] is None
    reading, reason = outcomes[1]
    assert reason == "compressor_runaway"
    assert reading.compressor_bytes == int(refused_at * GIB)
    assert outcomes[2] is None
    reading, reason = outcomes[3]
    assert reason == "compressor_full"
    assert 32 * GIB <= reading.compressor_bytes < 33 * GIB


def test_each_prefill_records_what_the_mac_did_for_calibration(monkeypatch):
    # The lines above are calibrated on one Mac. Every prefill's receipt
    # now carries the compressor's physical occupancy at its first and last
    # check, its largest net growth over five seconds or more (not a single
    # step between two readings), the lowest free pages and swap growth.
    readings = _trajectory(rate_gb_s=0.3, seconds=12)
    guard = _prefill_guard(monkeypatch, readings)
    for _ in readings:
        assert guard() is False
    trajectory = guard.trajectory()
    assert trajectory["checks"] == len(readings)
    assert trajectory["interval_s"] == pytest.approx(12.0)
    assert trajectory["compressor_start_bytes"] == int(5.5 * GB)
    assert trajectory["compressor_growth_bytes"] == pytest.approx(0.3 * GB * 12, rel=1e-6)
    assert trajectory["compressor_growth_5s_max_bytes_per_s"] == pytest.approx(0.3 * GB, rel=1e-3)
    assert trajectory["episode_growth_max_bytes"] == trajectory["compressor_growth_bytes"]
    assert trajectory["free_min_bytes"] == int(2.0 * GB)
    assert trajectory["swap_growth_bytes"] == 0
