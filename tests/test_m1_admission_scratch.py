"""M1 family: the admission bills an MMA prefill chunk below the fused-SDPA reading.

The dense prefill bill was measured with fused SDPA attention, which builds the
score tensor. On the M1 family every prefill chunk runs on the MMA kernel, which
does not; on an M1 Max 64 GB (Qwen3.8-27B, 2026-10-03) a 2,048-row chunk at
~100K tokens of context held 2.05 GB against the 3.15 GB billed. The bill keeps
70% of the dense figure there, and the full figure wherever the MMA prefill does
not serve every chunk.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import mtplx.server.openai as srv
import tests.test_memguard_admission as base
from tests.test_memguard_admission import GIB, Q27_KV, _q27_runtime


@pytest.fixture
def bill():
    state = SimpleNamespace(runtime=_q27_runtime())
    geometry = base.TestGrowthModel._geometry(base.TestGrowthModel(), live_bytes_per_token=Q27_KV)

    def run(rows: int) -> tuple[int, str]:
        return srv._admission_scratch_bytes(state, rows=rows, prompt_tokens=99_355, geometry=geometry)

    return run


def _upstream(rows: int) -> int:
    mlp_row = 2 * (5_120 + 3 * 17_408)
    return int(0.75 * GIB + 10 * mlp_row * rows)


def test_m1_mma_prefill_bills_seventy_percent(bill, monkeypatch):
    monkeypatch.setenv("MTPLX_M1_LONG_CONTEXT", "1")
    for name in ("MTPLX_GQA_MMA_PREFILL", "MTPLX_GQA_MMA_PREFILL_MIN_PREFIX", "MTPLX_M1_MMA_PREFILL_SCRATCH_FRACTION"):
        monkeypatch.delenv(name, raising=False)
    scratch, source = bill(2048)
    assert source == "geometry_measured_m1_mma"
    assert scratch == int(_upstream(2048) * 0.70)
    assert 2.05 * 1e9 < scratch < 2.3 * 1e9  # over the 2.05 GB reading
    # Never under the fixed part a dense forward always holds.
    assert bill(16)[0] >= srv._DENSE_PREFILL_FIXED_BYTES


@pytest.mark.parametrize(
    "env",
    [
        {"MTPLX_M1_LONG_CONTEXT": "0"},
        {"MTPLX_M1_LONG_CONTEXT": "1", "MTPLX_GQA_MMA_PREFILL": "0"},
        {"MTPLX_M1_LONG_CONTEXT": "1", "MTPLX_GQA_MMA_PREFILL_MIN_PREFIX": "49152"},
        {"MTPLX_M1_LONG_CONTEXT": "1", "MTPLX_M1_MMA_PREFILL_SCRATCH_FRACTION": "1"},
    ],
)
def test_full_bill_without_the_mma_prefill(bill, monkeypatch, env):
    for name in ("MTPLX_GQA_MMA_PREFILL", "MTPLX_GQA_MMA_PREFILL_MIN_PREFIX", "MTPLX_M1_MMA_PREFILL_SCRATCH_FRACTION"):
        monkeypatch.delenv(name, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    scratch, source = bill(2048)
    if env.get("MTPLX_M1_MMA_PREFILL_SCRATCH_FRACTION") == "1":
        assert source == "geometry_measured_m1_mma" and scratch == _upstream(2048)
    else:
        assert source == "geometry_measured" and scratch == _upstream(2048)
