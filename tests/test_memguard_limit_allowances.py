"""Lowering MTPLX_MEMORY_LIMIT_BYTES lowers every allowance sized from the Mac.

Item (b) of the review of 9c96dd9c: the wired limit is clamped to the
allocation limit and the plan's budgets take an explicit limit as the engine
budget, but the MLX allocator's cache limit stayed at its RAM tier and the
host allowance at its RAM share, so an operator who lowered the limit to
48 GiB on a 128 GB Mac still let 8 GiB of pooled buffers and 8 GiB of host
memory ride outside it. Both are a twelfth of the default limit (a sixteenth
of the Mac); an explicit limit now bounds them at a twelfth of itself.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import mtplx.server.openai as srv

GIB = 1024**3


@pytest.fixture(autouse=True)
def _a_128gb_mac(monkeypatch):
    monkeypatch.setattr(srv, "_total_ram_bytes", lambda: 128 * GIB)
    monkeypatch.delenv("MTPLX_MEMORY_LIMIT_BYTES", raising=False)
    monkeypatch.delenv("MTPLX_MLX_CACHE_LIMIT", raising=False)
    monkeypatch.delenv("MTPLX_MEMORY_BUDGET", raising=False)
    monkeypatch.delenv("MTPLX_HOST_MEMORY_ALLOWANCE_BYTES", raising=False)


def _caps(limit_gib: float, source: str) -> SimpleNamespace:
    return SimpleNamespace(
        metal_memory_caps={
            "memory_limit_bytes": int(limit_gib * GIB),
            "memory_limit_source": source,
            "total_ram_bytes": 128 * GIB,
        },
        memory_budget_bytes=None,
    )


class TestTheCacheLimit:
    def test_the_default_keeps_its_ram_tier(self):
        assert srv._default_mlx_cache_limit_bytes(None) == 8 * GIB

    @pytest.mark.parametrize("limit_gib, cache_gib", [(96, 8), (90, 7.5), (48, 4), (12, 1)])
    def test_an_explicit_limit_bounds_it(self, limit_gib, cache_gib):
        assert srv._default_mlx_cache_limit_bytes(
            None, explicit_limit=int(limit_gib * GIB)
        ) == int(cache_gib * GIB)

    def test_the_server_reads_the_operators_limit(self, monkeypatch):
        import mlx.core as mx

        applied: list[int] = []
        monkeypatch.setattr(mx, "set_cache_limit", lambda value: applied.append(value) or 0)
        monkeypatch.setenv("MTPLX_MEMORY_LIMIT_BYTES", "48G")
        status = srv._configure_mlx_cache_limit(
            SimpleNamespace(mlx_cache_limit=None, memory_budget=None)
        )
        assert applied == [4 * GIB]
        assert status["limit_bytes"] == 4 * GIB
        assert status["source"] == "ram_tier_default_bounded_by_memory_limit"

    def test_an_explicit_cache_limit_still_wins(self, monkeypatch):
        import mlx.core as mx

        applied: list[int] = []
        monkeypatch.setattr(mx, "set_cache_limit", lambda value: applied.append(value) or 0)
        monkeypatch.setenv("MTPLX_MEMORY_LIMIT_BYTES", "48G")
        srv._configure_mlx_cache_limit(
            SimpleNamespace(mlx_cache_limit="6G", memory_budget=None)
        )
        assert applied == [6 * GIB]


class TestTheHostAllowance:
    def test_the_default_limit_keeps_the_seats_share(self):
        assert srv._host_memory_allowance_bytes(_caps(96, "default")) == 8 * GIB

    # Never under the 4 GiB a small seat's own model holds outside MLX
    # (2026-10-02): a 6 GiB limit lowers the share, not the daemon's floor.
    @pytest.mark.parametrize("limit_gib, allowance_gib", [(96, 8), (90, 7.5), (48, 4), (6, 4)])
    def test_an_explicit_limit_bounds_it(self, limit_gib, allowance_gib):
        assert srv._host_memory_allowance_bytes(_caps(limit_gib, "env")) == int(
            allowance_gib * GIB
        )

    def test_an_explicit_allowance_still_wins(self, monkeypatch):
        monkeypatch.setenv("MTPLX_HOST_MEMORY_ALLOWANCE_BYTES", "10G")
        assert srv._host_memory_allowance_bytes(_caps(48, "env")) == 10 * GIB
