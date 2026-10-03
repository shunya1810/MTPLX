"""Flash-Next without the sparse prefill lane is charged its dense intermediates once.

The review of 9c96dd9c (finding 8): the admission added the planner's context
transient (the QSA indexer's dense-lane chain, 12.75 B per row and context
token over four layers, calibrated at 2,048 rows: 104,448 B a context token)
to the itemized QSA bill, which already carries the dense attention and
indexer intermediates for the rows the forward runs. On a Mac without the
sparse lane, a 195-token suffix on a 131K conversation was charged 12.75 GiB
it would never allocate. The sparse-lane fixtures stamp the planner term to
zero and could not see it.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import mtplx.server.openai as srv
import mtplx.system_memory as sm
from mtplx.generation import _qwen4_wide_prefill_need
from mtplx.memory_plan import qsa_prefill_transient_bytes_per_token_from_config
from tests.test_memguard_admission import (
    FN_AUX,
    FN_KV,
    FN_WEIGHTS,
    GIB,
    _flash_next_runtime,
    _install,
    _Machine,
    _manager,
    _state,
)

PLANNER_TRANSIENT = 104_448


@pytest.fixture(autouse=True)
def _dense_flash_next(monkeypatch):
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", "1")
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL_LAYOUT", "auto")
    monkeypatch.setenv("MTPLX_SUSTAINED_DENSE_DECODE_MAX_CONTEXT", "262144")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE", "auto")
    import mtplx.models.qwen4_exp as qwen4

    # No tensor units: the sparse prefill lane is off, every forward dense.
    monkeypatch.setattr(qwen4, "_qsa_prefill_enabled", lambda: False)


def _state_without_the_sparse_lane():
    plan = SimpleNamespace(
        available=True,
        kv_bytes_per_token=FN_KV,
        kv_bytes_per_token_effective=FN_KV,
        aux_bytes_per_token=FN_AUX,
        prefill_transient_bytes_per_token=PLANNER_TRANSIENT,
        runtime_transients_bytes=3 * GIB,
        model_weights_bytes=FN_WEIGHTS,
    )
    return _state(
        _manager(), plan=plan, runtime=_flash_next_runtime(), limit_gib=96, total_gib=128
    )


def test_the_planner_term_is_what_the_review_says():
    config = {
        "text_config": {
            "indexer_n_heads": 4,
            "layer_types": ["full_attention"] * 12,
            "num_key_value_heads": 2,
            "head_dim": 256,
        }
    }
    assert qsa_prefill_transient_bytes_per_token_from_config(config) == PLANNER_TRANSIENT
    assert round(131_072 * PLANNER_TRANSIENT / GIB, 2) == 12.75


def test_a_short_suffix_is_charged_the_itemized_bill_alone():
    state = _state_without_the_sparse_lane()
    geometry = srv._admission_geometry(state)
    scratch, source = srv._admission_scratch_bytes(
        state, rows=195, prompt_tokens=131_072, geometry=geometry
    )
    assert source == "qsa_itemized"
    per_token = srv._admission_context_transient_per_token(
        geometry, rows=195, scratch_source=source
    )
    assert per_token == 0
    growth = srv._admission_growth(
        geometry,
        prompt_tokens=131_072,
        reused_tokens=131_072 - 195,
        restore_copies_prefix=False,
        layout="contiguous_dense_decode",
        source_layout="contiguous_dense_decode",
        output_tokens=0,
        publish=False,
        scratch_bytes=scratch,
        context_transient_bytes_per_token=per_token,
    )
    assert growth["context_transient_bytes"] == 0
    # The itemized bill carries the dense scores for the 195 rows it runs.
    bill = _qwen4_wide_prefill_need(
        state.runtime,
        rows=195,
        prompt_tokens=131_072,
        per_token=geometry.live_bytes_per_token,
    )
    assert bill["dense_attention_keys"] == 131_072
    assert bill["dense_attention_bytes"] > 0


def test_anything_else_scales_the_planner_term_to_its_rows():
    state = _state_without_the_sparse_lane()
    geometry = srv._admission_geometry(state)
    assert srv._admission_context_transient_per_token(
        geometry, rows=195, scratch_source="qsa_flat_bill"
    ) == PLANNER_TRANSIENT * 195 // 2048
    assert srv._admission_context_transient_per_token(
        geometry, rows=2048, scratch_source="geometry"
    ) == PLANNER_TRANSIENT


def test_the_admission_prices_it_that_way(monkeypatch):
    state = _state_without_the_sparse_lane()
    monkeypatch.setattr(srv, "_record_guard_event", lambda state, payload: None)
    monkeypatch.setattr(
        sm,
        "_reader",
        lambda: sm.SystemMemory(
            available_bytes=60 * GIB,
            total_bytes=128 * GIB,
            level_percent=50,
            free_bytes=50 * GIB,
            file_backed_bytes=10 * GIB,
            wired_bytes=40 * GIB,
            compressor_bytes=GIB,
            swap_used_bytes=0,
        ),
    )
    _install(
        monkeypatch,
        _Machine(state.sessions.bank, base_gib=80.0, cache_gib=0.0, host_gib=1.0),
    )
    pricing: dict = {}
    srv._prefill_admission_shed(
        state,
        prompt_ids=list(range(4_000)),
        session_bank=None,
        session_id=None,
        prefill_chunk_tokens=None,
        pricing=pricing,
    )
    growth = pricing.get("growth")
    assert growth is not None
    assert growth["scratch_source"] == "qsa_itemized"
    assert growth["context_transient_bytes"] == 0
