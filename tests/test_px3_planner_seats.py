"""Planner replay on simulated seats (PX.3, 2026-09-18).

The fixture is the read-only planner replay of the 2026-09-18 policy audit
(`04_memory_plan_table.py`) turned into a test: pack constants measured from
disk that night, the server's Metal-cap arithmetic, and the same two calls
into ``plan_memory`` the server makes. No MLX model, no GPU.

Pack constants (bytes on disk, 2026-09-18):

* Flash-Next Optimized Speed: 83,037,597,915 B of weight files (77.33 GiB,
  the n-gram table excluded), 32,000,154,008 B streamed table, KV 24,576
  B/token, QSA aux 7,872 B/token, dense-prefill transient 104,448 B/token.
* Qwen3.8-27B Optimized Speed: 20,683,241,105 B (19.26 GiB), KV 65,536.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from mtplx.memory_plan import GIB, plan_memory
from mtplx.server import openai

MODEL_MAX = 262_144

FLASH_NEXT = {
    "weights": 83_037_597_915,
    "table": 32_000_154_008,
    "kv": 24_576,
    "aux": 7_872,
    "dense_transient": 104_448,
    "resident_floor_family": True,
}
DENSE_27B = {
    "weights": 20_683_241_105,
    "table": 0,
    "kv": 65_536,
    "aux": 0,
    "dense_transient": 0,
    "resident_floor_family": False,
}


def _fake_mx():
    metal = SimpleNamespace(is_available=lambda: True)
    return SimpleNamespace(
        metal=metal,
        set_memory_limit=lambda value: None,
        set_wired_limit=lambda value: None,
    )


def _seat(pack: dict, ram_gb: int, *, sparse_prefill: bool = True):
    """(caps, fit plan, served default window) for one simulated seat."""

    ram = ram_gb * GIB
    floor = None
    if pack["resident_floor_family"]:
        floor = pack["weights"] + openai._resident_floor_margin_bytes(ram)
    caps = openai._apply_metal_memory_caps(
        mx_module=_fake_mx(), total_ram_bytes=ram, minimum_resident_bytes=floor
    )
    if not caps.get("applied"):
        return caps, None, None
    fit = plan_memory(
        total_ram_bytes=ram,
        model_weights_bytes=pack["weights"],
        ngram_table_streamed_bytes=pack["table"],
        kv_bytes_per_token=pack["kv"],
        model_max_context=MODEL_MAX,
        usable_bytes_override=caps["memory_limit_bytes"],
        usable_bytes_explicit=caps.get("memory_limit_source") == "env",
        resident_floor_bytes=caps.get("minimum_resident_bytes"),
        aux_bytes_per_token=pack["aux"],
        prefill_transient_bytes_per_token=(
            0 if sparse_prefill else pack["dense_transient"]
        ),
    )
    from mtplx.backends.descriptors import NATIVE_CONTRACT_DESCRIPTOR

    window = openai._select_backend_context_window(
        NATIVE_CONTRACT_DESCRIPTOR,
        model_max=MODEL_MAX,
        requested=None,
        machine_fit=openai._machine_fit_for_default_window(fit),
    )
    return caps, fit, window


@pytest.fixture(autouse=True)
def _no_operator_caps(monkeypatch):
    monkeypatch.delenv("MTPLX_MEMORY_LIMIT_BYTES", raising=False)
    monkeypatch.delenv("MTPLX_WIRED_LIMIT_BYTES", raising=False)


# --- PX.3(a): "does not fit" serves the floor window, never the model max ---


@pytest.mark.parametrize("ram_gb", [16, 24])
def test_27b_that_does_not_fit_serves_the_floor_window(ram_gb):
    _caps, fit, window = _seat(DENSE_27B, ram_gb)
    assert fit.available and not fit.model_fits
    assert window == 4_096
    assert window != MODEL_MAX


def test_machine_fit_is_zero_only_when_the_plan_is_unavailable():
    unavailable = plan_memory(total_ram_bytes=None, model_weights_bytes=1)
    assert openai._machine_fit_for_default_window(unavailable) == 0
    assert openai._machine_fit_for_default_window(None) == 0


def test_explicit_window_and_allow_swap_still_win_over_the_floor():
    from mtplx.backends.descriptors import NATIVE_CONTRACT_DESCRIPTOR

    _caps, fit, _window = _seat(DENSE_27B, 24)
    floor = openai._machine_fit_for_default_window(fit)
    assert floor == 4_096
    # Explicit --context-window always wins (the plan warns, never refuses).
    assert (
        openai._select_backend_context_window(
            NATIVE_CONTRACT_DESCRIPTOR,
            model_max=MODEL_MAX,
            requested=32_768,
            machine_fit=floor,
        )
        == 32_768
    )
    # --allow-swap passes machine_fit=0 and keeps the model maximum.
    assert (
        openai._select_backend_context_window(
            NATIVE_CONTRACT_DESCRIPTOR,
            model_max=MODEL_MAX,
            requested=None,
            machine_fit=0,
        )
        == MODEL_MAX
    )


@pytest.mark.parametrize(
    ("ram_gb", "expected"),
    [(36, 57_344), (48, 204_800), (64, MODEL_MAX), (96, MODEL_MAX), (128, MODEL_MAX)],
)
def test_27b_seats_that_fit_are_unchanged(ram_gb, expected):
    _caps, fit, window = _seat(DENSE_27B, ram_gb)
    assert fit.model_fits
    assert window == expected


# --- PX.3(c): Flash-Next on 96 GB: catalog, planner and the fixed-M4 gate ----


def test_flash_next_96g_envelope_is_ram_minus_the_system_reserve():
    caps, fit, window = _seat(FLASH_NEXT, 96)
    # 96 GiB - 12 GiB reserve. The bare floor (weights + 3 GiB = 80.3 GiB)
    # put the admission line (0.97 of the limit) under the resident set.
    assert caps["memory_limit_bytes"] == 84 * GIB
    assert caps["memory_limit_source"] == "resident_floor"
    # The wired cap stays at the floor: about 16 GiB stays unwired (#400).
    assert caps["wired_limit_bytes"] == caps["minimum_resident_bytes"]
    assert fit.usable_bytes == 84 * GIB
    assert fit.usable_source == "resident_floor"
    assert fit.model_fits
    assert fit.context_machine_bound
    assert window == 86_016
    assert not any("does not fit" in note for note in fit.notes)


def test_flash_next_96g_without_sparse_prefill_gets_a_smaller_honest_window():
    # M1 to M4 from pip or Homebrew: the dense indexer lane prices 104,448
    # extra bytes per context token.
    _caps, fit, window = _seat(FLASH_NEXT, 96, sparse_prefill=False)
    assert fit.model_fits
    assert window == 20_480


def test_flash_next_96g_catalog_planner_and_fixed_m4_gate_agree():
    from mtplx.model_catalog import catalog_model_with_id, evaluate_feasibility

    caps, fit, window = _seat(FLASH_NEXT, 96)
    verdict = evaluate_feasibility(
        catalog_model_with_id("flash-next-optimized-speed"),
        chip_tier="modern",
        ram_gib=96.0,
    )
    assert verdict.ok  # the catalog offers the pack on this seat
    assert fit.model_fits  # the planner admits it
    # The fixed-M4 memory gate admits a typical agent turn: resident weights
    # plus 32K tokens of KV, aux and promotion stay under 0.97 of the limit.
    line = int(caps["memory_limit_bytes"] * 0.97)
    promotion_per_token = 28_416  # generation._qwen4_fixed_m4_promotion_bytes_per_token
    context = 32_768
    live = FLASH_NEXT["weights"] + context * (FLASH_NEXT["kv"] + FLASH_NEXT["aux"])
    assert live + (context + 1_024) * promotion_per_token <= line
    # Under the old limit (the bare floor) the same turn was over the line.
    old_line = int(caps["minimum_resident_bytes"] * 0.97)
    assert live + (context + 1_024) * promotion_per_token > old_line


def test_flash_next_128g_desktop_seat():
    # G5 (09-29): 90 GiB on a 128 GB desktop (96 GiB before), which still
    # covers Flash-Next's 83.3 GiB resident floor, so the floor does not lift it.
    caps, fit, window = _seat(FLASH_NEXT, 128)
    assert caps["memory_limit_bytes"] == 90 * GIB
    assert caps["memory_limit_source"] == "default"
    assert caps["wired_limit_bytes"] == caps["minimum_resident_bytes"]
    assert fit.usable_bytes == 90 * GIB
    assert fit.usable_source == "formula"
    assert window == MODEL_MAX
    _caps, _fit, dense_window = _seat(FLASH_NEXT, 128, sparse_prefill=False)
    # The dense-prefill lane prices the indexer's per-token peak, so the six
    # GiB the desktop keeps cost it context: 114,688 at 96 GiB, 65,536 at 90.
    assert dense_window == 65_536


@pytest.mark.parametrize("ram_gb", [16, 24, 36, 48, 64])
def test_flash_next_still_refuses_seats_below_96g(ram_gb):
    caps, fit, window = _seat(FLASH_NEXT, ram_gb)
    assert caps["applied"] is False
    assert caps["reason"] == "insufficient_ram"
    assert fit is None and window is None


def test_simulated_96g_seat_plans_like_the_real_one():
    # --memory-budget 96G on the 128 GB box: the caps were configured for the
    # real machine (96 GiB limit, floor with the 6 GiB margin); the plan must
    # still budget the 96 GB envelope.
    real_floor = FLASH_NEXT["weights"] + openai._resident_floor_margin_bytes(128 * GIB)
    plan = plan_memory(
        total_ram_bytes=128 * GIB,
        memory_budget_bytes=96 * GIB,
        model_weights_bytes=FLASH_NEXT["weights"],
        kv_bytes_per_token=FLASH_NEXT["kv"],
        aux_bytes_per_token=FLASH_NEXT["aux"],
        model_max_context=MODEL_MAX,
        usable_bytes_override=96 * GIB,
        resident_floor_bytes=real_floor,
    )
    assert plan.usable_bytes == 84 * GIB
    assert plan.model_fits
    assert plan.context_window_fit == 86_016


def test_27b_envelope_never_reads_a_resident_floor():
    for ram_gb in (16, 24, 36, 48, 64, 96, 128):
        caps, fit, _window = _seat(DENSE_27B, ram_gb)
        assert caps["memory_limit_source"] == "default"
        assert fit.usable_source == "formula"
        assert "minimum_resident_bytes" not in caps


def test_48g_seat_plan_is_unchanged():
    _caps, fit, window = _seat(DENSE_27B, 48)
    assert window == 204_800
    assert fit.usable_bytes == 36 * GIB
    final = plan_memory(
        total_ram_bytes=48 * GIB,
        model_weights_bytes=DENSE_27B["weights"],
        kv_bytes_per_token=DENSE_27B["kv"],
        model_max_context=MODEL_MAX,
        usable_bytes_override=36 * GIB,
        dense_decode_ceiling=131_072,
    )
    assert round(final.bank_idle_max_bytes / GIB, 1) == 13.7
    # The under-load figure reserves the whole 204,800-token window (#525);
    # it reserved the 131,072-token dense ceiling and read 5.7 GiB before.
    assert final.kv_reserve_tokens == 204_800
    assert round(final.bank_steady_bytes / GIB, 1) == 1.2
