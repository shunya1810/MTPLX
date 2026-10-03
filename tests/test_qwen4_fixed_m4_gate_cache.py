"""The compiled fixed-M4 lane's memory gate does not count reclaimable cache as live (2026-09-03 W3)."""

from __future__ import annotations

from types import SimpleNamespace

import mtplx.generation as gen
import mtplx.graphbank as graphbank

GB = 1_000_000_000


def _gate(monkeypatch, *, active, cache_before, cache_after, limit=103_079_215_104, per_token=28_416, prompt=99_792):
    calls = {"released": 0}
    state = {"cache": cache_before}

    def live():
        return active + state["cache"]

    def release():
        calls["released"] += 1
        freed = state["cache"] - cache_after
        state["cache"] = cache_after
        return freed

    monkeypatch.setattr(gen, "_mlx_live_memory_bytes", live)
    monkeypatch.setattr(gen, "_mlx_release_allocator_cache", release)
    monkeypatch.setattr(gen, "_qwen4_fixed_m4_promotion_bytes_per_token", lambda rt: per_token)
    monkeypatch.setattr(gen, "_metal_memory_limit_bytes", lambda rt: limit)
    monkeypatch.setattr(graphbank, "_fixed_m4_initial_growth_reserve", lambda: 4096)
    announced = []
    monkeypatch.setattr(gen, "_announce_qwen4_fixed_m4_skip", announced.append)
    monkeypatch.delenv("MTPLX_QWEN4_FIXED_M4_MAX_CONTEXT", raising=False)
    fits = gen._qwen4_fixed_m4_lane_fits(SimpleNamespace(), prompt_tokens=prompt)
    return fits, calls["released"], announced


def test_the_w3_receipt_is_admitted_once_the_cache_is_released(monkeypatch) -> None:
    # 2026-09-03 05:14 PDT: active ~84 GB + 19 GB of cache read as 103.0 GB live
    # against the 100.0 GB line; releasing the cache admits the 2.9 GB promotion.
    fits, releases, announced = _gate(monkeypatch, active=84 * GB, cache_before=19 * GB, cache_after=0)
    assert fits is True
    assert releases == 1
    assert announced == []


def test_the_common_case_never_touches_the_cache(monkeypatch) -> None:
    fits, releases, announced = _gate(monkeypatch, active=80 * GB, cache_before=5 * GB, cache_after=0)
    assert fits is True
    assert releases == 0
    assert announced == []


def test_a_genuinely_full_machine_still_skips_and_says_what_was_released(monkeypatch) -> None:
    fits, releases, announced = _gate(monkeypatch, active=100 * GB, cache_before=3 * GB, cache_after=0)
    assert fits is False
    assert releases == 1
    assert len(announced) == 1
    assert "allocator cache released: 3.0 GB" in announced[0]
    assert "over the 100.0 GB line" in announced[0]


def test_nothing_cached_means_no_second_read(monkeypatch) -> None:
    fits, releases, announced = _gate(monkeypatch, active=101 * GB, cache_before=0, cache_after=0)
    assert fits is False
    assert releases == 1  # asked, nothing to release
    assert "released" not in announced[0]


def test_shallow_depth_does_not_promote_or_evict(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("a depth with no compiled forward must not enter memory admission")

    monkeypatch.setattr(gen, "_qwen4_fixed_m4_lane_fits", unexpected)
    for depth in (1, 2):
        receipt = {}
        assert not gen._qwen4_fixed_m4_compiled_verify_requested(
            SimpleNamespace(qwen4_fixed_m4_compiled_verify=True),
            verify_strategy="batched", compiled_mode="on", max_tokens=10000,
            cached_tokens=0, prompt_tokens=128000, speculative_depth=depth,
            receipt=receipt,
        )
        assert receipt["reason"] == "depth_below_compiled_window"


def test_idle_bank_yields_but_logical_eviction_is_not_proof_of_freed_memory(monkeypatch):
    state = {"live": 99 * GB}
    monkeypatch.setattr(gen, "_mlx_live_memory_bytes", lambda: state["live"])
    monkeypatch.setattr(gen, "_mlx_release_allocator_cache", lambda: 0)
    monkeypatch.setattr(gen, "_qwen4_fixed_m4_promotion_bytes_per_token", lambda rt: 28416)
    monkeypatch.setattr(gen, "_metal_memory_limit_bytes", lambda rt: 103079215104)
    monkeypatch.setattr(graphbank, "_fixed_m4_initial_growth_reserve", lambda: 4096)
    monkeypatch.delenv("MTPLX_QWEN4_FIXED_M4_MAX_CONTEXT", raising=False)

    class Bank:
        total_nbytes = 8 * GB
        physical_reclaim = False

        def shrink_for_admission(self, target, *, protect_tokens, reason):
            assert protect_tokens == [1, 2, 3]
            assert reason == "fixed_m4_admission"
            self.total_nbytes = target
            if self.physical_reclaim:
                state["live"] = 90 * GB
            return 1, 0

    bank = Bank()
    assert not gen._qwen4_fixed_m4_lane_fits(SimpleNamespace(), prompt_tokens=128000,
                                          session_bank=bank, prompt_ids=[1, 2, 3])
    bank.physical_reclaim = True
    receipt = {}
    assert gen._qwen4_fixed_m4_lane_fits(SimpleNamespace(), prompt_tokens=128000,
                                      session_bank=bank, prompt_ids=[1, 2, 3], receipt=receipt)
    assert receipt["live_bytes_after"] == 90 * GB
