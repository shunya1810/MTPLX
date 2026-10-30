"""Flash-Next's own prefill width (2026-09-18).

The family's settings block stamps a 4,096-row prefill chunk and a
16,384-token sparse attention crossover on tensor-unit GPUs only; the wide
chunk is granted per request against live memory and a refusal is the
2,048-row plan that ships today, counted in the demotion ledger.  An
operator's explicit chunk or crossover always wins.
"""

from __future__ import annotations

import argparse
import json

import pytest

from mtplx import generation
from mtplx.models import qwen4_exp
from mtplx.server import openai as oa
from tests.test_env_flag_parsing import (
    _FLASH_NEXT_LANE_KEYS,
    _flash_next_fixed_m4_config,
    _flash_next_quantization,
)

PREFILL_KEYS = (
    "MTPLX_QWEN4_PREFILL_WIDE_CHUNK",
    "MTPLX_QSA_PREFILL_WIDE_MIN_CONTEXT",
)
GIB = 2**30


def _lane_args(tmp_path, monkeypatch):
    for key in (
        *_FLASH_NEXT_LANE_KEYS,
        *PREFILL_KEYS,
        "MTPLX_QSA_GATHER",
        "MTPLX_FUSED_GATE_UP",
    ):
        monkeypatch.delenv(key, raising=False)
    model = tmp_path / "flash-next"
    model.mkdir()
    config = _flash_next_fixed_m4_config()
    config["quantization"] = _flash_next_quantization(lm_head_bits=8, stage3=True)
    (model / "config.json").write_text(json.dumps(config))
    return argparse.Namespace(
        model=str(model),
        verify_strategy="batched",
        generation_mode="mtp",
        scheduler_mode="serial",
    )


def test_prefill_width_is_stamped_on_tensor_unit_gpus(tmp_path, monkeypatch):
    from mtplx.profiles import normalize_runtime_env_overrides

    args = _lane_args(tmp_path, monkeypatch)
    monkeypatch.setattr(oa, "_qwen4_tensor_unit_gpu", lambda: True)
    overrides = oa._server_runtime_env_overrides(args, {})
    assert overrides["MTPLX_QWEN4_PREFILL_WIDE_CHUNK"] == "4096"
    assert overrides["MTPLX_QSA_PREFILL_WIDE_MIN_CONTEXT"] == "16384"
    # The general crossovers are not touched: a warm turn's short suffix
    # keeps the 32,768 its own A/B chose.
    assert "MTPLX_QSA_PREFILL_MIN_CONTEXT" not in overrides
    assert "MTPLX_QSA_PREFILL_FLASH_MIN_CONTEXT" not in overrides
    assert normalize_runtime_env_overrides(overrides) == overrides


def test_other_gpus_keep_the_portable_prefill_values(tmp_path, monkeypatch):
    args = _lane_args(tmp_path, monkeypatch)
    monkeypatch.setattr(oa, "_qwen4_tensor_unit_gpu", lambda: False)
    overrides = oa._server_runtime_env_overrides(args, {})
    for key in PREFILL_KEYS:
        assert key not in overrides, key


@pytest.mark.parametrize("key", PREFILL_KEYS)
def test_an_operator_export_wins_over_the_stamp(tmp_path, monkeypatch, key):
    args = _lane_args(tmp_path, monkeypatch)
    monkeypatch.setattr(oa, "_qwen4_tensor_unit_gpu", lambda: True)
    monkeypatch.setenv(key, "0")
    overrides = oa._server_runtime_env_overrides(args, {})
    assert key not in overrides


def test_gpu_detector_follows_the_family_fallback_switch(monkeypatch):
    monkeypatch.setenv("MTPLX_FORCE_GPU_FAMILY_FALLBACK", "1")
    assert oa._qwen4_tensor_unit_gpu() is False


def _memory(monkeypatch, *, limit, live, per_token=28_416, released=0):
    for name in ("MTPLX_PREFILL_CHUNK_SIZE_DENSE", "MTPLX_PREFILL_CHUNK_SIZE_REPAGE"):
        monkeypatch.delenv(name, raising=False)
    state = {"live": live}

    def release():
        state["live"] -= released
        return released

    monkeypatch.setattr(generation, "_metal_memory_limit_bytes", lambda rt: limit)
    monkeypatch.setattr(generation, "_mlx_live_memory_bytes", lambda: state["live"])
    monkeypatch.setattr(
        generation, "_qwen4_fixed_m4_promotion_bytes_per_token", lambda rt: per_token
    )
    monkeypatch.setattr(generation, "_mlx_release_allocator_cache", release)


def test_the_wide_chunk_is_off_until_the_lane_names_a_width(monkeypatch):
    monkeypatch.delenv("MTPLX_QWEN4_PREFILL_WIDE_CHUNK", raising=False)
    monkeypatch.delenv("MTPLX_PREFILL_CHUNK_SIZE", raising=False)
    _memory(monkeypatch, limit=110 * GIB, live=85 * GIB)
    assert generation.qwen4_wide_prefill_chunk_tokens(None, prompt_tokens=65536) is None


def test_the_wide_chunk_is_granted_while_memory_allows(monkeypatch):
    monkeypatch.setenv("MTPLX_QWEN4_PREFILL_WIDE_CHUNK", "4096")
    monkeypatch.delenv("MTPLX_PREFILL_CHUNK_SIZE", raising=False)
    _memory(monkeypatch, limit=110 * GIB, live=85 * GIB)
    receipt: dict = {}
    assert (
        generation.qwen4_wide_prefill_chunk_tokens(
            None, prompt_tokens=65536, receipt=receipt
        )
        == 4096
    )
    assert receipt["granted"] is True
    assert receipt["need_bytes"] == 65536 * 28_416 + 8 * GIB
    assert receipt["threshold_bytes"] == int(110 * GIB * 0.97)


def test_a_tight_machine_keeps_the_2048_row_plan(monkeypatch):
    monkeypatch.setenv("MTPLX_QWEN4_PREFILL_WIDE_CHUNK", "4096")
    monkeypatch.delenv("MTPLX_PREFILL_CHUNK_SIZE", raising=False)
    _memory(monkeypatch, limit=96 * GIB, live=85 * GIB)
    receipt: dict = {}
    assert (
        generation.qwen4_wide_prefill_chunk_tokens(
            None, prompt_tokens=131072, receipt=receipt
        )
        is None
    )
    assert receipt["granted"] is False


def test_a_refusal_is_counted_and_a_grant_is_not(monkeypatch):
    # Nothing goes slow in silence: /health, the request log and
    # `mtplx doctor --explain` read this ledger.
    from mtplx import demotions

    kind = "qwen4_wide_prefill_chunk_refused"
    assert kind in demotions.KINDS
    monkeypatch.setenv("MTPLX_QWEN4_PREFILL_WIDE_CHUNK", "4096")
    monkeypatch.delenv("MTPLX_PREFILL_CHUNK_SIZE", raising=False)
    _memory(monkeypatch, limit=110 * GIB, live=85 * GIB)
    before = demotions.mark()
    assert generation.qwen4_wide_prefill_chunk_tokens(None, prompt_tokens=65536) == 4096
    assert demotions.since(before).get(kind, 0) == 0
    _memory(monkeypatch, limit=96 * GIB, live=85 * GIB)
    assert generation.qwen4_wide_prefill_chunk_tokens(None, prompt_tokens=131072) is None
    assert demotions.since(before)[kind] == 1


def test_the_allocator_cache_is_released_before_a_refusal(monkeypatch):
    monkeypatch.setenv("MTPLX_QWEN4_PREFILL_WIDE_CHUNK", "4096")
    monkeypatch.delenv("MTPLX_PREFILL_CHUNK_SIZE", raising=False)
    _memory(monkeypatch, limit=110 * GIB, live=97 * GIB, released=10 * GIB)
    assert generation.qwen4_wide_prefill_chunk_tokens(None, prompt_tokens=16384) == 4096


def test_short_prompts_and_pinned_chunks_are_left_alone(monkeypatch):
    monkeypatch.setenv("MTPLX_QWEN4_PREFILL_WIDE_CHUNK", "4096")
    monkeypatch.delenv("MTPLX_PREFILL_CHUNK_SIZE", raising=False)
    _memory(monkeypatch, limit=110 * GIB, live=85 * GIB)
    assert generation.qwen4_wide_prefill_chunk_tokens(None, prompt_tokens=2048) is None
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE", "2048")
    assert generation.qwen4_wide_prefill_chunk_tokens(None, prompt_tokens=65536) is None


def test_the_profile_stamped_chunk_knobs_are_not_an_operator_pin(monkeypatch):
    """Every profile exports auto / 2048 / 2048; only a moved knob is a pin."""

    _memory(monkeypatch, limit=110 * GIB, live=85 * GIB)
    monkeypatch.setenv("MTPLX_QWEN4_PREFILL_WIDE_CHUNK", "4096")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE", "auto")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE_DENSE", "2048")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE_REPAGE", "2048")
    assert generation.qwen4_wide_prefill_chunk_tokens(None, prompt_tokens=65536) == 4096
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE_DENSE", "1024")
    assert generation.qwen4_wide_prefill_chunk_tokens(None, prompt_tokens=65536) is None


def test_an_unknown_memory_limit_does_not_refuse(monkeypatch):
    monkeypatch.setenv("MTPLX_QWEN4_PREFILL_WIDE_CHUNK", "4096")
    monkeypatch.delenv("MTPLX_PREFILL_CHUNK_SIZE", raising=False)
    _memory(monkeypatch, limit=0, live=85 * GIB)
    assert generation.qwen4_wide_prefill_chunk_tokens(None, prompt_tokens=65536) == 4096


def test_the_compiled_selector_accepts_the_canonical_and_the_wide_width(monkeypatch):
    monkeypatch.delenv("MTPLX_QSA_PREFILL_COMPILE_ROWS", raising=False)
    monkeypatch.delenv("MTPLX_QWEN4_PREFILL_WIDE_CHUNK", raising=False)
    assert qwen4_exp._qsa_prefill_compile_row_set() == (2048,)
    monkeypatch.setenv("MTPLX_QWEN4_PREFILL_WIDE_CHUNK", "4096")
    assert qwen4_exp._qsa_prefill_compile_row_set() == (2048, 4096)
    monkeypatch.setenv("MTPLX_QWEN4_PREFILL_WIDE_CHUNK", "junk")
    assert qwen4_exp._qsa_prefill_compile_row_set() == (2048,)


def test_the_lowered_crossover_applies_to_wide_forwards_only(monkeypatch):
    for key in (
        "MTPLX_QSA_PREFILL_MIN_CONTEXT",
        "MTPLX_QSA_PREFILL_FLASH_MIN_CONTEXT",
        "MTPLX_QSA_PREFILL_WIDE_MIN_CONTEXT",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("MTPLX_M1_LONG_CONTEXT", "0")
    assert qwen4_exp._qsa_prefill_crossover(4096, 32768) == 32768
    monkeypatch.setenv("MTPLX_QSA_PREFILL_WIDE_MIN_CONTEXT", "16384")
    assert qwen4_exp._qsa_prefill_crossover(4096, 32768) == 16384
    assert qwen4_exp._qsa_prefill_crossover(2048, 32768) == 16384
    assert qwen4_exp._qsa_prefill_crossover(2047, 32768) == 32768
    assert qwen4_exp._qsa_prefill_crossover(64, 32768) == 32768
    # It can only lower the crossover, never raise an operator's lower one.
    assert qwen4_exp._qsa_prefill_crossover(4096, 8192) == 8192
    monkeypatch.setenv("MTPLX_QSA_PREFILL_WIDE_MIN_CONTEXT", "0")
    assert qwen4_exp._qsa_prefill_crossover(4096, 32768) == 32768


def test_the_m1_family_defaults_wide_forwards_to_its_16k_crossover(monkeypatch):
    for key in ("MTPLX_QSA_PREFILL_WIDE_MIN_CONTEXT", "MTPLX_QSA_PREFILL_DIRECT_MIN_CONTEXT"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("MTPLX_M1_LONG_CONTEXT", "1")
    assert qwen4_exp._qsa_prefill_crossover(4096, 32768) == 16384
    assert qwen4_exp._qsa_prefill_crossover(2048, 32768) == 16384
    assert qwen4_exp._qsa_prefill_crossover(2047, 32768) == 32768
    assert qwen4_exp._qsa_prefill_direct_min_context() == 16384
    # An operator's export still wins.
    monkeypatch.setenv("MTPLX_QSA_PREFILL_WIDE_MIN_CONTEXT", "0")
    assert qwen4_exp._qsa_prefill_crossover(4096, 32768) == 32768
    monkeypatch.setenv("MTPLX_M1_LONG_CONTEXT", "0")
    monkeypatch.delenv("MTPLX_QSA_PREFILL_WIDE_MIN_CONTEXT")
    assert qwen4_exp._qsa_prefill_direct_min_context() == 32768


def test_the_measured_128_gb_seat(monkeypatch):
    """96 GiB engine budget, 88.3 GB live: granted through 64K, refused at 128K
    (measured peaks 98.8 and 100.0 GB against a 103.1 GB budget)."""

    monkeypatch.setenv("MTPLX_QWEN4_PREFILL_WIDE_CHUNK", "4096")
    monkeypatch.delenv("MTPLX_PREFILL_CHUNK_SIZE", raising=False)
    monkeypatch.delenv("MTPLX_QWEN4_PREFILL_WIDE_PRESSURE", raising=False)
    _memory(monkeypatch, limit=96 * GIB, live=int(88.3e9))
    assert generation.qwen4_wide_prefill_chunk_tokens(None, prompt_tokens=4061) == 4096
    assert generation.qwen4_wide_prefill_chunk_tokens(None, prompt_tokens=65502) == 4096
    assert generation.qwen4_wide_prefill_chunk_tokens(None, prompt_tokens=131039) is None
    monkeypatch.setenv("MTPLX_QWEN4_PREFILL_WIDE_PRESSURE", "0.99")
    assert generation.qwen4_wide_prefill_chunk_tokens(None, prompt_tokens=131039) == 4096


# --- The itemized bill (2026-09-20) -----------------------------------------
#
# Friday's flat 8 GiB was three things: pre-conv streams the recurrent layers
# keep alive to the end of a forward (3.1 GB at 4,096 rows; the model's
# mid-loop evals let each one die with its layer), the score matrix stock
# attention materializes below the sparse crossover (3.2 GB at 4,096 rows and
# 16K keys) and the forward's own intermediates (1.8 GB).  None of them grows
# with the prompt, so a flat charge on top of the KV refused the 4,096-row
# chunk exactly where it pays most: 64K after a few earlier cells (live
# 89.9 GB + need 10.5 over the 100.0 GB line) and every 128K prompt.
#
# Measured on that seat with the states named in the chunk eval and ONE eval
# per forward (n3-pfx2-memfix, both prompts forced onto 4,096 rows): process
# peak 8.5 GB over the request's start at 64K and 9.6 GB at 128K, 100.8 GB
# against the 100.0 GB line.  The bill without mid-loop evals is 10.5 and
# 11.0 GB, so it refuses both, as it should.


class _FlashNextArgs:
    layer_types = (["linear_attention"] * 3 + ["full_attention"]) * 12
    num_attention_heads = 24
    num_key_value_heads = 2
    head_dim = 256
    hidden_size = 2560
    hc_count = 4
    linear_num_key_heads = 16
    linear_key_head_dim = 128
    linear_num_value_heads = 48
    linear_value_head_dim = 128
    ple_layer_ids = [2]
    indexer_head_dim = 128
    indexer_compress_ratio = 4


class _FlashNextText:
    args = _FlashNextArgs()


class _FlashNextModel:
    language_model = _FlashNextText()


class _FlashNextRuntime:
    model = _FlashNextModel()
    mtp_enabled = True


BILL_KEYS = (
    "MTPLX_QWEN4_PREFILL_WIDE_BILL",
    "MTPLX_QWEN4_PREFILL_MIDLOOP_EVAL",
    "MTPLX_PREFILL_EVAL_RECURRENT_STATE",
    "MTPLX_QSA_PREFILL_MIN_CONTEXT",
    "MTPLX_QSA_PREFILL_FLASH_MIN_CONTEXT",
    "MTPLX_QWEN4_PREFILL_WIDE_PRESSURE",
    "MTPLX_PREFILL_CHUNK_SIZE",
)


def _itemized(monkeypatch, *, sparse_lane=True, wide="4096", crossover="16384"):
    for key in BILL_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("MTPLX_QWEN4_PREFILL_WIDE_CHUNK", wide)
    monkeypatch.setenv("MTPLX_QSA_PREFILL_WIDE_MIN_CONTEXT", crossover)
    monkeypatch.setattr(qwen4_exp, "_qsa_prefill_enabled", lambda: sparse_lane)
    return _FlashNextRuntime()


def test_the_geometry_the_bill_is_made_of():
    geometry = generation._qwen4_prefill_geometry(_FlashNextRuntime())
    assert geometry == {
        "heads": 24,
        "n_qsa": 12,
        "hidden": 2560,
        # 36 GDN layers of a 10,240-wide conv stream and one PLE layer's
        # 4 x 2,560 window, in bf16: 3.1 GB at 4,096 rows.
        "pinned_bytes_per_row": 2 * (36 * 10_240 + 10_240),
    }
    assert round(geometry["pinned_bytes_per_row"] * 4096 / 1e9, 1) == 3.1
    assert generation._qwen4_prefill_geometry(None) is None
    assert generation._qwen4_fixed_m4_promotion_bytes_per_token(_FlashNextRuntime()) == 28_416


def test_the_bill_names_each_term(monkeypatch):
    rt = _itemized(monkeypatch)
    bill = generation._qwen4_wide_prefill_need(
        rt, rows=4096, prompt_tokens=131_039, per_token=28_416
    )
    per_token = 28_416 + 28_416 // 12 + 2 * 2560
    assert per_token == 35_904  # measured 33.7 KB a token on the 128K trace
    assert bill["kv_bytes"] == 131_039 * per_token
    assert bill["forward_bytes"] == 4096 * 512 * 1024
    # A dense forward sees fewer than crossover + rows keys.
    assert bill["dense_attention_keys"] == 16_384 + 4096
    assert bill["dense_attention_bytes"] == 4096 * 20_480 * (2 * 24 + 7)
    assert bill["pinned_stream_bytes"] == 0
    # The larger of two moments, never their sum: the last dense forward
    # (short KV, the score matrix, half a forward) against the last forward
    # of the prompt (all the KV, one forward).
    dense_moment = 20_480 * per_token + bill["dense_attention_bytes"] + bill["forward_bytes"] // 2
    last_moment = bill["kv_bytes"] + bill["forward_bytes"]
    assert bill["need_bytes"] == max(dense_moment, last_moment) + GIB
    assert bill["transient_bytes"] == bill["need_bytes"] - bill["kv_bytes"]
    assert 7.5e9 < bill["need_bytes"] < 8.2e9


def test_tonights_refusals_become_grants_and_the_margin_is_stated(monkeypatch):
    """128 GB M5 Max, 2026-09-20, the founder's cell order: 64K met 89.9 GB
    live and 128K 90.8 GB, both refused by the flat charge (need 10.5 and
    12.3 GB against the 100.0 GB line).  With the model's mid-loop evals the
    pre-conv streams are no longer part of the peak."""

    rt = _itemized(monkeypatch)
    _memory(monkeypatch, limit=96 * GIB, live=int(89.9e9))
    receipt: dict = {}
    assert (
        generation.qwen4_wide_prefill_chunk_tokens(rt, prompt_tokens=65_502, receipt=receipt)
        == 4096
    )
    assert receipt["granted_chunk_tokens"] == 4096
    assert receipt["dense_attention_bytes"] > receipt["kv_bytes"]
    _memory(monkeypatch, limit=96 * GIB, live=int(90.8e9))
    receipt = {}
    assert (
        generation.qwen4_wide_prefill_chunk_tokens(rt, prompt_tokens=131_039, receipt=receipt)
        == 4096
    )
    # Under the line with 1 to 2 GB to spare, not more: the bill is not slack.
    spare = receipt["threshold_bytes"] - receipt["live_bytes"] - receipt["need_bytes"]
    assert 1.0e9 < spare < 2.0e9
    # 262K on the same seat is still refused: the KV alone is 9.4 GB.
    assert generation.qwen4_wide_prefill_chunk_tokens(rt, prompt_tokens=262_000) is None


def test_the_flat_bill_is_the_rollback(monkeypatch):
    rt = _itemized(monkeypatch)
    monkeypatch.setenv("MTPLX_QWEN4_PREFILL_WIDE_BILL", "flat")
    _memory(monkeypatch, limit=96 * GIB, live=int(90.8e9))
    receipt: dict = {}
    assert (
        generation.qwen4_wide_prefill_chunk_tokens(rt, prompt_tokens=131_039, receipt=receipt)
        is None
    )
    assert receipt["need_bytes"] == 131_039 * 28_416 + 8 * GIB
    assert "dense_attention_bytes" not in receipt


def test_a_mac_without_the_sparse_lane_is_billed_the_whole_score_matrix(monkeypatch):
    rt = _itemized(monkeypatch, sparse_lane=False)
    bill = generation._qwen4_wide_prefill_need(
        rt, rows=4096, prompt_tokens=65_502, per_token=28_416
    )
    assert bill["dense_attention_keys"] == 65_502
    assert bill["dense_attention_bytes"] == 4096 * 65_502 * (2 * 24 + 7)
    assert bill["need_bytes"] > 17e9
    _memory(monkeypatch, limit=96 * GIB, live=int(84e9))
    assert generation.qwen4_wide_prefill_chunk_tokens(rt, prompt_tokens=65_502) is None
    # The same seat with the lane on has 16 GB of room and needs 7.
    rt = _itemized(monkeypatch, sparse_lane=True)
    assert generation.qwen4_wide_prefill_chunk_tokens(rt, prompt_tokens=65_502) == 4096


def test_a_lower_crossover_is_billed_less(monkeypatch):
    rt = _itemized(monkeypatch, crossover="8192")
    low = generation._qwen4_wide_prefill_need(
        rt, rows=4096, prompt_tokens=65_502, per_token=28_416
    )
    rt = _itemized(monkeypatch, crossover="16384")
    high = generation._qwen4_wide_prefill_need(
        rt, rows=4096, prompt_tokens=65_502, per_token=28_416
    )
    assert low["dense_attention_keys"] == 8192 + 4096
    assert low["dense_attention_bytes"] < high["dense_attention_bytes"]
    assert low["need_bytes"] <= high["need_bytes"]


def test_one_eval_per_forward_is_billed_its_pinned_streams(monkeypatch):
    rt = _itemized(monkeypatch)
    free = generation._qwen4_wide_prefill_need(
        rt, rows=4096, prompt_tokens=65_502, per_token=28_416
    )
    assert free["pinned_stream_bytes"] == 0
    monkeypatch.setenv("MTPLX_QWEN4_PREFILL_MIDLOOP_EVAL", "0")
    pinned = generation._qwen4_wide_prefill_need(
        rt, rows=4096, prompt_tokens=65_502, per_token=28_416
    )
    assert pinned["pinned_stream_bytes"] == 4096 * 2 * 37 * 10_240
    assert pinned["need_bytes"] == free["need_bytes"] + pinned["pinned_stream_bytes"]
    # Naming the states in the chunk eval frees them between chunks only, so
    # that switch does not move the bill.
    monkeypatch.setenv("MTPLX_PREFILL_EVAL_RECURRENT_STATE", "0")
    assert (
        generation._qwen4_wide_prefill_need(
            rt, rows=4096, prompt_tokens=65_502, per_token=28_416
        )
        == pinned
    )


def test_the_bill_without_mid_loop_evals_covers_the_measured_peaks(monkeypatch):
    """n3-pfx2-memfix: 8.5 GB over the request's start at 64K, 9.6 GB at 128K,
    on 4,096-row chunks with one eval per forward."""

    rt = _itemized(monkeypatch)
    monkeypatch.setenv("MTPLX_QWEN4_PREFILL_MIDLOOP_EVAL", "0")
    for prompt, measured in ((65_502, 8.5e9), (131_039, 9.6e9)):
        bill = generation._qwen4_wide_prefill_need(
            rt, rows=4096, prompt_tokens=prompt, per_token=28_416
        )
        assert measured < bill["need_bytes"] < measured + 2.5e9
    # So tonight's two seats are refused without them, as the peaks say.
    _memory(monkeypatch, limit=96 * GIB, live=int(89.9e9))
    assert generation.qwen4_wide_prefill_chunk_tokens(rt, prompt_tokens=65_502) is None
    _memory(monkeypatch, limit=96 * GIB, live=int(90.8e9))
    assert generation.qwen4_wide_prefill_chunk_tokens(rt, prompt_tokens=131_039) is None


def test_the_rungs_are_the_stamped_width_then_powers_of_two_down_to_4096():
    assert generation._wide_prefill_rungs(4096) == [4096]
    assert generation._wide_prefill_rungs(8192) == [8192, 4096]
    assert generation._wide_prefill_rungs(16_384) == [16_384, 8192, 4096]
    assert generation._wide_prefill_rungs(6000) == [6000, 4096]
    assert generation._wide_prefill_rungs(3000) == [3000]


def test_a_stamp_of_8192_degrades_to_4096_before_it_degrades_to_a_refusal(monkeypatch):
    rt = _itemized(monkeypatch, wide="8192")
    wide_bill = generation._qwen4_wide_prefill_need(
        rt, rows=8192, prompt_tokens=65_502, per_token=28_416
    )
    narrow_bill = generation._qwen4_wide_prefill_need(
        rt, rows=4096, prompt_tokens=65_502, per_token=28_416
    )
    assert wide_bill["need_bytes"] > narrow_bill["need_bytes"] + 4e9
    line = int(96 * GIB * 0.97)
    # Room for the 8,192-row bill: granted as stamped.
    _memory(monkeypatch, limit=96 * GIB, live=line - wide_bill["need_bytes"] - 1)
    receipt: dict = {}
    assert (
        generation.qwen4_wide_prefill_chunk_tokens(rt, prompt_tokens=65_502, receipt=receipt)
        == 8192
    )
    assert receipt["wide_chunk_tokens"] == 8192 and receipt["granted_chunk_tokens"] == 8192
    # Room for 4,096 only: the next rung, and no refusal is counted.
    from mtplx import demotions

    before = demotions.mark()
    _memory(monkeypatch, limit=96 * GIB, live=line - narrow_bill["need_bytes"] - 1)
    receipt = {}
    assert (
        generation.qwen4_wide_prefill_chunk_tokens(rt, prompt_tokens=65_502, receipt=receipt)
        == 4096
    )
    assert receipt["granted"] is True and receipt["granted_chunk_tokens"] == 4096
    assert receipt["need_bytes"] == narrow_bill["need_bytes"]
    assert demotions.since(before).get("qwen4_wide_prefill_chunk_refused", 0) == 0
    # Room for neither: the 2,048-row plan, counted.
    _memory(monkeypatch, limit=96 * GIB, live=line - narrow_bill["need_bytes"] + 1)
    receipt = {}
    assert generation.qwen4_wide_prefill_chunk_tokens(rt, prompt_tokens=65_502, receipt=receipt) is None
    assert receipt["granted"] is False and receipt["granted_chunk_tokens"] == 0
    assert demotions.since(before)["qwen4_wide_prefill_chunk_refused"] == 1


def test_a_rung_the_prompt_cannot_fill_is_not_tried(monkeypatch):
    rt = _itemized(monkeypatch, wide="8192")
    _memory(monkeypatch, limit=110 * GIB, live=85 * GIB)
    # 4,061 tokens are one 4,096-row forward either way; 8,192 buys nothing
    # and would only be billed more.
    assert generation.qwen4_wide_prefill_chunk_tokens(rt, prompt_tokens=4061) == 4096
    assert generation.qwen4_wide_prefill_chunk_tokens(rt, prompt_tokens=4097) == 8192
    assert generation.qwen4_wide_prefill_chunk_tokens(rt, prompt_tokens=2048) is None


def test_the_allocator_cache_is_released_once_across_the_rungs(monkeypatch):
    rt = _itemized(monkeypatch, wide="8192")
    calls = {"n": 0}
    _memory(monkeypatch, limit=96 * GIB, live=int(99e9), released=int(12e9))
    release = generation._mlx_release_allocator_cache

    def counted():
        calls["n"] += 1
        return release()

    monkeypatch.setattr(generation, "_mlx_release_allocator_cache", counted)
    assert generation.qwen4_wide_prefill_chunk_tokens(rt, prompt_tokens=65_502) == 4096
    assert calls["n"] == 1
