"""The MLX buffer pool keeps its full bound while model work runs and goes
back to macOS once the model owner has had no work for its idle grace.

2026-09-29, 128 GB M5 Max, Flash-Next Optimized Speed, A B B A: a 2 GiB pool
bound (3863e9d8) cost 6 to 10% of the 16K and 64K prefill rate when the
prompt followed earlier requests in the same boot. The bound is back at its
full size, and the idle footprint comes from returning the pool when the
owner goes quiet (after a request and the postcommits and SSD encodes that
follow it) instead of from a small bound.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

from mtplx.model_scheduler import ModelWorkScheduler
from mtplx.server import openai


def _scheduler(grace_s: float) -> tuple[ModelWorkScheduler, list[dict]]:
    scheduler = ModelWorkScheduler(
        name="test-owner-idle", idle_grace_s=0.0, persistence_quiet_grace_s=0.05
    )
    calls: list[dict] = []

    def hook():
        calls.append({"thread": threading.get_ident(), "at": time.monotonic()})
        return {"cleared": True}

    scheduler.owner_idle_grace_s = grace_s
    scheduler.on_owner_idle = hook
    return scheduler, calls


def _wait_for(predicate, timeout_s: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def test_the_owner_returns_the_pool_once_after_its_work_goes_quiet():
    scheduler, calls = _scheduler(0.05)
    try:
        owner = scheduler.submit_foreground(threading.get_ident).result(timeout=2)
        done_at = time.monotonic()
        assert _wait_for(lambda: len(calls) == 1)
        assert calls[0]["thread"] == owner
        assert calls[0]["at"] - done_at >= 0.04
        # Nothing ran since: no second return while the owner stays idle.
        time.sleep(0.2)
        assert len(calls) == 1
        stats = scheduler.stats()["owner_idle"]
        assert stats["turns"] == 1 and stats["errors"] == 0
        assert stats["last"]["result"] == {"cleared": True}
    finally:
        scheduler.shutdown(wait=True, cancel_futures=True)


def test_a_gap_shorter_than_the_grace_keeps_the_pool():
    # A request's restore and its generation arrive as separate items with
    # 100-200 ms of handler Python between them: the pool stays for both.
    scheduler, calls = _scheduler(0.5)
    try:
        scheduler.submit_foreground(lambda: None).result(timeout=2)
        time.sleep(0.15)
        scheduler.submit_foreground(lambda: None).result(timeout=2)
        assert calls == []
        assert _wait_for(lambda: len(calls) == 1)
    finally:
        scheduler.shutdown(wait=True, cancel_futures=True)


def test_background_work_after_a_request_is_followed_by_one_return():
    scheduler, calls = _scheduler(0.05)
    ran: list[str] = []
    try:
        scheduler.submit_foreground(lambda: ran.append("request")).result(timeout=2)
        scheduler.submit_idle_persistence(lambda: ran.append("ssd")).result(timeout=2)
        assert _wait_for(lambda: len(calls) >= 1)
        time.sleep(0.2)
        assert ran == ["request", "ssd"]
        # One return after the last of them (or one after each when the
        # persistence grace outlasted the owner's): never while work runs.
        assert 1 <= len(calls) <= 2
    finally:
        scheduler.shutdown(wait=True, cancel_futures=True)


def test_a_return_that_raises_is_counted_and_the_owner_keeps_serving():
    scheduler, _calls = _scheduler(0.02)

    def failing():
        raise RuntimeError("clear failed")

    scheduler.on_owner_idle = failing
    try:
        scheduler.submit_foreground(lambda: None).result(timeout=2)
        assert _wait_for(lambda: scheduler.stats()["owner_idle"]["errors"] == 1)
        assert scheduler.submit_foreground(lambda: 7).result(timeout=2) == 7
        last = scheduler.stats()["owner_idle"]["last"]
        assert "clear failed" in last["error"]
    finally:
        scheduler.shutdown(wait=True, cancel_futures=True)


def test_no_hook_means_no_idle_turns():
    scheduler = ModelWorkScheduler(name="test-owner-idle-none", idle_grace_s=0.0)
    scheduler.owner_idle_grace_s = 0.01
    try:
        scheduler.submit_foreground(lambda: None).result(timeout=2)
        time.sleep(0.1)
        assert scheduler.stats()["owner_idle"]["turns"] == 0
    finally:
        scheduler.shutdown(wait=True, cancel_futures=True)


def test_the_server_wires_the_return_and_clears_only_a_held_pool(monkeypatch):
    import mlx.core as mx

    monkeypatch.delenv("MTPLX_CLEAR_CACHE_AFTER_REQUEST", raising=False)
    pool = {"bytes": 3 * 1024**3}
    cleared: list[int] = []
    monkeypatch.setattr(mx, "get_cache_memory", lambda: pool["bytes"])
    monkeypatch.setattr(mx, "synchronize", lambda *a, **k: None)

    def clear_cache():
        cleared.append(pool["bytes"])
        pool["bytes"] = 0

    monkeypatch.setattr(mx, "clear_cache", clear_cache)
    state = SimpleNamespace(lock=threading.Lock())
    scheduler = SimpleNamespace(on_owner_idle=None)
    openai._wire_owner_idle_pool_return(state, scheduler)

    receipt = scheduler.on_owner_idle()
    assert receipt["cleared"] is True
    assert receipt["reason"] == "owner_idle"
    assert receipt["pool_bytes"] == 3 * 1024**3
    assert cleared == [3 * 1024**3]
    # An empty pool is left alone (no synchronize, no clear).
    assert scheduler.on_owner_idle() == {
        "cleared": False,
        "reason": "pool_empty",
        "pool_bytes": 0,
    }
    assert cleared == [3 * 1024**3]
    # The operator's switch keeps the pool.
    pool["bytes"] = 1024**3
    monkeypatch.setenv("MTPLX_CLEAR_CACHE_AFTER_REQUEST", "off")
    assert scheduler.on_owner_idle()["cleared"] is False
    assert cleared == [3 * 1024**3]


@pytest.mark.parametrize("ram_gib, pool_gib", [(16, 2), (48, 4), (96, 6), (128, 8)])
def test_the_pool_bound_is_back_at_its_full_size(monkeypatch, ram_gib, pool_gib):
    monkeypatch.setattr(openai, "_total_ram_bytes", lambda: ram_gib * 1024**3)
    assert openai._default_mlx_cache_limit_bytes() == pool_gib * 1024**3
