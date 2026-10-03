"""The prefill width is chosen after the admission reclaims memory.

On 2026-09-29 (128 GB, Flash-Next, 90 GiB limit) the wide-chunk gate ran
before the prefill admission: it saw 89.70 GB live plus a 7.65 GB bill and
refused 4,096 rows, then the admission reclaimed memory down to 84.28 GB,
where 4,096 would have fitted, and the cold 123K prompts ran at 2,048 rows
(about 7 s slower each). The wide widths are now candidates the admission
prices with the rest of the request, with the one bill that also decides
admission, and the width it settles on after reclamation is the one that
runs. Without an admission price the live-memory gate still decides.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")

from test_api_benchmark_contracts import _envelope_client, _fake_generation_output
from test_prefill_admission_shed import GIB, LIMIT, _Bank, _Entry, _state

import mtplx.generation as generation
import mtplx.server.openai as srv
from mtplx.server.prefill_safety import settle_wide_prefill_chunk

PROMPT = list(range(40_000))
THRESHOLD = int(LIMIT * srv._PREFILL_ADMISSION_PRESSURE_FRACTION)


@pytest.fixture(autouse=True)
def _served_shape(monkeypatch):
    # The served profiles prefill in chunks, and the process footprint is
    # the allocator's own reading here.
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", "1")
    monkeypatch.setattr(srv, "phys_footprint_bytes", lambda *_a, **_k: 0)


def _growth_by_width(monkeypatch) -> dict[int, int]:
    """What the admission's own bill charges this prompt at each width."""

    monkeypatch.setattr(
        srv,
        "_mlx_memory_stats_live",
        lambda: {"ok": True, "active_memory_bytes": 95 * GIB, "cache_memory_bytes": 0},
    )
    receipt = srv._prefill_admission_shed(
        _state(), prompt_ids=PROMPT, session_bank=None, session_id=None,
        wide_prefill_rungs=[4096],
    )
    return {int(width): int(bytes_) for width, bytes_ in receipt["growth_by_chunk"].items()}


def _admit_with_idle_entry(monkeypatch, idle_bytes: int, *, over_narrow: int):
    """Live memory such that even the 2,048-row plan is ``over_narrow`` bytes
    past the line until an idle stranger entry of ``idle_bytes`` is evicted."""

    growth = _growth_by_width(monkeypatch)
    assert growth[4096] > growth[2048]
    bank = _Bank([_Entry(range(90_000, 90_100), "idle-stranger", idle_bytes)])
    engine_before = THRESHOLD - growth[2048] + over_narrow
    base = engine_before - idle_bytes
    monkeypatch.setattr(
        srv,
        "_mlx_memory_stats_live",
        lambda: {
            "ok": True,
            "active_memory_bytes": int(base + bank.total_nbytes),
            "cache_memory_bytes": 0,
        },
    )
    pricing: dict = {}
    receipt = srv._prefill_admission_shed(
        _state(), prompt_ids=PROMPT, session_bank=bank, session_id="pi",
        wide_prefill_rungs=[4096], pricing=pricing,
    )
    return growth, bank, receipt, pricing


def test_a_wide_chunk_is_admitted_once_reclamation_made_room(monkeypatch):
    growth = _growth_by_width(monkeypatch)
    # Evicting the stranger frees more than the wide chunk costs over the
    # narrow one: after reclamation 4,096 rows fit clear of the line.
    idle = growth[4096] - growth[2048] + 2 * GIB
    _growth, bank, receipt, pricing = _admit_with_idle_entry(
        monkeypatch, idle, over_narrow=GIB
    )

    assert "lru_idle_entries" in receipt["reclamation_steps"]
    assert bank.entries == []
    assert receipt["prefill_chunk_requested"] == 4096
    assert receipt["prefill_chunk_tokens"] == 4096
    assert pricing["growth"]["prefill_chunk_tokens"] == 4096
    assert "refused" not in receipt


def test_the_narrow_chunk_remains_when_the_wide_one_still_does_not_fit(monkeypatch):
    # The stranger covers the narrow plan's shortfall and no more.
    _growth, bank, receipt, pricing = _admit_with_idle_entry(
        monkeypatch, 2 * GIB, over_narrow=GIB
    )

    assert "lru_idle_entries" in receipt["reclamation_steps"]
    assert receipt["prefill_chunk_tokens"] == 2048
    assert pricing["growth"]["prefill_chunk_tokens"] == 2048
    assert "refused" not in receipt


def test_the_settled_width_is_the_one_that_runs():
    rungs = [4096]
    receipt: dict = {}
    assert settle_wide_prefill_chunk(
        None, prompt_tokens=40_000, rungs=rungs,
        pricing={"growth": {"prefill_chunk_tokens": 4096, "growth_bytes": 7}},
        receipt=receipt,
    ) == 4096
    assert receipt["granted"] is True
    assert receipt["decided_by"] == "prefill_admission"

    receipt = {}
    assert settle_wide_prefill_chunk(
        None, prompt_tokens=40_000, rungs=rungs,
        pricing={"growth": {"prefill_chunk_tokens": 2048, "growth_bytes": 7}},
        receipt=receipt,
    ) is None
    assert receipt["granted"] is False


def test_without_an_admission_price_the_live_memory_gate_decides(monkeypatch):
    calls: list[int] = []

    def gate(_runtime, *, prompt_tokens, receipt=None):
        calls.append(prompt_tokens)
        if receipt is not None:
            receipt["granted"] = True
        return 4096

    monkeypatch.setattr(generation, "qwen4_wide_prefill_chunk_tokens", gate)
    receipt: dict = {}
    assert settle_wide_prefill_chunk(
        None, prompt_tokens=40_000, rungs=[4096], pricing={}, receipt=receipt
    ) == 4096
    assert calls == [40_000]
    assert receipt["decided_by"] == "live_memory_gate"


def _serve_one(monkeypatch, *, settled_width: int | None, gate_width: int | None):
    """One request through the real _run_generation. The live-memory gate
    answers ``gate_width``; the admission settles on ``settled_width`` (None:
    it priced nothing). Returns the chunk override generation ran under and
    the rungs the admission was offered."""

    monkeypatch.setenv("MTPLX_QWEN4_PREFILL_WIDE_CHUNK", "4096")
    seen: dict = {}
    output = _fake_generation_output()

    def fake_generate(*args, **kwargs):
        seen["chunk"] = generation._PREFILL_CHUNK_SIZE_OVERRIDE.get()
        return output(*args, **kwargs)

    def fake_admission(*_args, pricing=None, wide_prefill_rungs=(), **_kwargs):
        seen["rungs"] = list(wide_prefill_rungs)
        if settled_width is not None and pricing is not None:
            pricing["growth"] = {
                "prefill_chunk_tokens": settled_width,
                "growth_bytes": 1,
                "chunk_bytes": 1,
            }
        return None

    monkeypatch.setattr(
        generation,
        "qwen4_wide_prefill_chunk_tokens",
        lambda *_a, **_k: gate_width,
    )
    client, state = _envelope_client(
        monkeypatch, generator=fake_generate, prompt_tokens=40_000
    )
    state.context_window = 262_144
    monkeypatch.setattr(srv, "_prefill_admission_shed", fake_admission)
    response = client.post(
        "/v1/chat/completions",
        headers={"x-mtplx-cache-mode": "bypass"},
        json={"messages": [{"role": "user", "content": "Build it."}], "max_tokens": 8},
    )
    assert response.status_code == 200
    return seen


def test_the_request_runs_the_width_the_admission_settled_after_reclaim(monkeypatch):
    # The 09-29 shape: the live-memory gate refuses before reclamation, the
    # admission reclaims and settles on 4,096. The old path never offered
    # 4,096 to the admission and ran the 2,048 plan.
    seen = _serve_one(monkeypatch, settled_width=4096, gate_width=None)

    assert seen["chunk"] == 4096
    assert seen["rungs"] == [4096]


def test_the_profile_plan_runs_when_the_admission_settles_narrow(monkeypatch):
    # The gate would grant 4,096, but the admission's full bill says it does
    # not fit: the admission's answer stands.
    seen = _serve_one(monkeypatch, settled_width=2048, gate_width=4096)

    assert seen["chunk"] is None
    assert seen["rungs"] == [4096]


def test_the_gate_still_decides_when_nothing_priced_the_request(monkeypatch):
    seen = _serve_one(monkeypatch, settled_width=None, gate_width=4096)

    assert seen["chunk"] == 4096
