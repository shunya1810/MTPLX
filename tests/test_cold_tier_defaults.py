"""SSD cold-tier defaults accepted from PR #496 (Dizzler7).

The RAM-tiered SSD cap stays 32 GiB on a 64 GB Mac unless the disk has room
to spare (>= 150 GiB free lifts it to the 100 GiB big-machine default); at
write time the effective cap is still bounded by the free disk (normally a
quarter of the space the tier can use), so the lift only matters on disks
with >= 400 GiB free. The hourly SSD write budget default
is 128 GiB (was 64): a deep multi-turn session dedupes most blocks, but the
first snapshot of several long conversations in one hour could exceed 64.
"""

from __future__ import annotations

import os
from collections import namedtuple

from mtplx.cache_bank import cold_tier

GIB = 1024**3
_Usage = namedtuple("_Usage", "total used free")


def _pin(monkeypatch, *, ram_gib: int, free_gib: int) -> None:
    monkeypatch.setattr(cold_tier, "detect_total_ram_bytes", lambda: ram_gib * GIB)
    monkeypatch.setattr(
        cold_tier.shutil,
        "disk_usage",
        lambda _path: _Usage(2000 * GIB, (2000 - free_gib) * GIB, free_gib * GIB),
    )


def test_64gb_mac_keeps_32gib_cap_on_a_tight_disk(monkeypatch):
    _pin(monkeypatch, ram_gib=64, free_gib=120)
    assert cold_tier.default_cold_tier_max_bytes() == 32 * GIB


def test_64gb_mac_gets_the_big_cap_with_150gib_free(monkeypatch):
    _pin(monkeypatch, ram_gib=64, free_gib=150)
    assert cold_tier.default_cold_tier_max_bytes() == cold_tier.DEFAULT_COLD_TIER_MAX_BYTES


def test_64gb_mac_keeps_32gib_when_the_disk_cannot_be_read(monkeypatch):
    monkeypatch.setattr(cold_tier, "detect_total_ram_bytes", lambda: 64 * GIB)

    def boom(_path):
        raise OSError("no statvfs")

    monkeypatch.setattr(cold_tier.shutil, "disk_usage", boom)
    assert cold_tier.default_cold_tier_max_bytes() == 32 * GIB


def test_smaller_tiers_are_unchanged(monkeypatch):
    _pin(monkeypatch, ram_gib=32, free_gib=900)
    assert cold_tier.default_cold_tier_max_bytes() == 24 * GIB
    _pin(monkeypatch, ram_gib=16, free_gib=900)
    assert cold_tier.default_cold_tier_max_bytes() == 16 * GIB


def test_hourly_write_budget_default_is_128gib(monkeypatch, tmp_path):
    monkeypatch.delenv("MTPLX_SSD_WRITE_BUDGET_PER_HOUR", raising=False)
    tier = cold_tier.SessionBankColdTier(base_dir=tmp_path / "bank", mode="on")
    try:
        assert tier.stats()["write_budget_per_hour_bytes"] == 128 * GIB
    finally:
        tier.close()
    assert os.environ.get("MTPLX_SSD_WRITE_BUDGET_PER_HOUR") is None


# --- PX.3(d), 2026-09-18: bare `mtplx serve` reaches the RAM-tiered cap -------
#
# The serve parsers defaulted the cap to the literal "100GB", so the tiering
# above was unreachable unless a launcher passed "auto" (the app does): a
# 16 GB Mac serving from the terminal got a 100 GB session store.


def _serve_namespace(monkeypatch, tmp_path, argv):
    from mtplx.server import openai

    monkeypatch.delenv("MTPLX_SSD_SESSION_CACHE_MAX_SIZE", raising=False)
    monkeypatch.delenv("MTPLX_SSD_SESSION_CACHE", raising=False)
    return openai.parse_args(["--model", str(tmp_path / "model"), *argv])


def test_serve_parser_default_cap_is_auto(monkeypatch, tmp_path):
    args = _serve_namespace(monkeypatch, tmp_path, [])
    assert args.ssd_session_cache_max_size == "auto"


def test_public_cli_serve_default_cap_is_auto():
    from mtplx.cli import build_parser

    args = build_parser().parse_args(["serve", "--model", "models/example"])
    assert args.ssd_session_cache_max_size == "auto"


def test_auto_cap_resolves_through_the_ram_tiers(monkeypatch):
    for ram_gib, expected in ((16, 16), (32, 24), (64, 32), (128, 100)):
        _pin(monkeypatch, ram_gib=ram_gib, free_gib=100)
        assert (
            cold_tier.parse_size_bytes("auto", cold_tier.default_cold_tier_max_bytes())
            == expected * GIB
        )


def test_an_explicit_cap_still_wins(monkeypatch, tmp_path):
    args = _serve_namespace(
        monkeypatch, tmp_path, ["--ssd-session-cache-max-size", "48GB"]
    )
    _pin(monkeypatch, ram_gib=16, free_gib=900)
    assert (
        cold_tier.parse_size_bytes(
            args.ssd_session_cache_max_size, cold_tier.default_cold_tier_max_bytes()
        )
        == 48 * GIB
    )
