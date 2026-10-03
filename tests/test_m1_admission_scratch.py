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


# -- the M1 build's quantized direct restore ------------------------------------

from mtplx.cache_state import COMPACT_KV_FORMAT, COMPACT_KV_FORMAT_KEY  # noqa: E402

LIVE_W = 65_536 + 4_096  # 27B fp16 KV row + the MTP head's history
PAGED_W = 34_816 + 4_096  # q8 pages + the history


def _q8_geometry():
    return srv._AdmissionGeometry(
        live_bytes_per_token=LIVE_W,
        paged_bytes_per_token=PAGED_W,
        context_transient_bytes_per_token=0,
        flat_transient_bytes=3 * GIB,
        weights_bytes=19 * GIB,
        aux_bytes_per_token=4_096,
        kv_quantization="q8",
    )


def _compact_source(bits=8):
    state = {COMPACT_KV_FORMAT_KEY: COMPACT_KV_FORMAT, "bits": bits}
    return SimpleNamespace(cache_snapshot=SimpleNamespace(states=(None, state, None, state)))


def test_a_compact_q8_restore_is_priced_at_page_width(monkeypatch):
    monkeypatch.setenv("MTPLX_M1_LONG_CONTEXT", "1")
    geometry = _q8_geometry()
    assert srv._m1_compact_direct_restore(_compact_source(8), geometry)
    kwargs = dict(
        prompt_tokens=112_318, reused_tokens=112_299, restore_copies_prefix=True,
        layout="contiguous_then_repage", source_layout="contiguous_then_repage",
        output_tokens=16_384, publish=False, scratch_bytes=1 * GIB,
    )
    upstream = srv._admission_growth(geometry, **kwargs)
    direct = srv._admission_growth(geometry, **kwargs, compact_direct=True)
    assert upstream["repage_copy_bytes"] > 0
    assert direct["repage_copy_bytes"] == 0 and direct["quant_working_bytes"] == 0
    # The restored prefix, the suffix and the answer's reserve, all at page width.
    assert direct["live_prefill_bytes"] == (112_318 + 16_384) * PAGED_W
    assert direct["growth_bytes"] < upstream["growth_bytes"] / 2


@pytest.mark.parametrize(
    "env,source,quant",
    [
        ({"MTPLX_M1_LONG_CONTEXT": "0"}, _compact_source(8), "q8"),
        ({"MTPLX_M1_LONG_CONTEXT": "1", "MTPLX_SESSION_BANK_COMPACT_DIRECT": "0"}, _compact_source(8), "q8"),
        ({"MTPLX_M1_LONG_CONTEXT": "1", "MTPLX_KV_QUANT_MMA_PREFILL": "0"}, _compact_source(8), "q8"),
        ({"MTPLX_M1_LONG_CONTEXT": "1"}, _compact_source(4), "q8"),
        ({"MTPLX_M1_LONG_CONTEXT": "1"}, SimpleNamespace(cache_snapshot=SimpleNamespace(states=(None,))), "q8"),
        ({"MTPLX_M1_LONG_CONTEXT": "1"}, _compact_source(8), "off"),
    ],
)
def test_the_direct_path_needs_every_condition(monkeypatch, env, source, quant):
    for name in ("MTPLX_SESSION_BANK_COMPACT_DIRECT", "MTPLX_KV_QUANT_MMA_PREFILL", "MTPLX_GQA_MMA_PREFILL"):
        monkeypatch.delenv(name, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    geometry = SimpleNamespace(kv_quantization=quant)
    assert srv._m1_compact_direct_restore(source, geometry) is False
