"""The fixed-M4 verify bank's capacity, and what the rows-gather lane's outputs depend on.

A compiled function keeps one trace per input-shape signature for as long as
it lives (MLX's compile cache), and the fixed-M4 bank's capacity is part of
that signature. The bank was sized prompt + reserve rounded to 256 tokens, so
every agent turn at 16K or more arrived with a capacity the process had not
seen and paid a fresh trace. On the rows-gather lane the capacity is now
rounded up to a bucket (``MTPLX_QWEN4_FIXED_M4_CAPACITY_BUCKET``, 8,192 rows
by default), so a growing session keeps one trace until it crosses a bucket
edge, and the lane's memory gate prices the bucketed bank. That is only
allowed because there the verify outputs do not depend on the capacity.

On the rows-gather lane (16,384 tokens of KV and more) they must not: each
row attends over its own gathered rows and the index scores are per block, so
the padded tail never enters a value. This file proves it on the compiled
verifier itself, with the tiny random pack of
``scripts/qwen4exp_mtp_tiny_smoke.py`` in bfloat16 on the GPU: the same
sampled request at two capacities, logits, hidden states, captures and every
state leaf bit for bit, round by round.

The dense lane is not capacity-invariant in general. It reduces over the
whole bank through MLX's vector SDPA, and MLX 0.32.2 picks that kernel and its
number of key blocks from the key length, which is the capacity
(``mlx/backend/metal/scaled_dot_product_attention.cpp`` 875-877 and 487-517).
One Flash-Next-geometry attention layer, capacity 3,524 against 8,192, differs
in 53 to 55% of its bfloat16 outputs per verify step under the base/Pro
dispatch class; capacity 9,024 against 16,384 differs in 58 to 59% under the
Ultra class (and in neither case under the Max class this suite runs on).
So the dense lane keeps its exact capacity.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import mlx.utils
import numpy as np
import pytest

import mtplx.generation as generation
import mtplx.graphbank as graphbank
from mtplx import demotions
from mtplx.sampling import SamplerConfig

_SMOKE = Path(__file__).resolve().parents[1] / "scripts" / "qwen4exp_mtp_tiny_smoke.py"
NATIVE = SamplerConfig(temperature=0.6, top_p=0.95, top_k=20)
MAX_TOKENS = 40
SEED = 1234
PROMPT = [3, 5, 7, 9, 11, 13] + list(range(20, 54))  # 40 tokens: 10 pooled blocks


def _smoke():
    spec = importlib.util.spec_from_file_location("qwen4exp_mtp_tiny_smoke", _SMOKE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def pack():
    """The tiny pack in bf16 on the GPU, shaped for the fixed-M4 lane (one PLE layer)."""

    if not mx.metal.is_available():
        pytest.skip("bfloat16 expert gathers need the GPU; the CPU has no exact full-model lane")
    import mlx_lm.models.cache as cache_module

    import mtplx.models.qwen4_exp as qwen4_exp
    from mtplx.models.qwen4_exp import Model, ModelArgs, Qwen4ExpMTP
    from mtplx.mtp_patch import validate_mtp_support

    smoke = _smoke()
    prev = mx.default_device()
    mx.set_default_device(mx.gpu)
    # The verify bank looks the ArraysCache class up when it is called; build
    # the model with the same class a runtime load earlier in the session
    # may have swapped in.
    previous_arrays_cache = qwen4_exp.ArraysCache
    qwen4_exp.ArraysCache = cache_module.ArraysCache
    mx.random.seed(0)
    args = dataclasses.replace(
        smoke._tiny_text_args(),
        head_dim=32,
        indexer_head_dim=32,
        indexer_compress_ratio=4,
        ple_layer_ids=[1],
        rope_parameters={
            "mrope_interleaved": True,
            "mrope_section": [2, 1, 1],
            "partial_rotary_factor": 0.25,
            "rope_theta": 10000000,
            "rope_type": "default",
        },
    )
    model = Model(ModelArgs(model_type="qwen4_exp", text_config=dataclasses.asdict(args)))
    model.language_model.mtp = Qwen4ExpMTP(model.language_model.args)
    model.update(
        mlx.utils.tree_map(
            lambda p: p.astype(mx.bfloat16) if p.dtype == mx.float32 else p,
            model.parameters(),
        )
    )
    model.eval()
    mx.eval(model.parameters())
    assert validate_mtp_support(model)
    yield smoke, model
    qwen4_exp.ArraysCache = previous_arrays_cache
    mx.set_default_device(prev)


@pytest.fixture()
def lane(monkeypatch):
    # One request's route must not depend on what the shell exported.
    import os

    for name in tuple(os.environ):
        if name.startswith("MTPLX_QWEN4_") or name.startswith("MTPLX_QSA_") or name in {
            "MTPLX_COMPILED_VERIFY",
            "MTPLX_COMPILED_VERIFY_GROWTH_RESERVE",
            "MTPLX_STATE_REBASE_EVERY",
            "MTPLX_FAMILY_CAPTURE_COMMIT",
        }:
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MTPLX_CONTEXT_COPY", "0")
    monkeypatch.setattr(graphbank, "_compiled_verify_bits_gate_ok", lambda _rt: True)
    demotions.reset()
    yield monkeypatch
    demotions.reset()


def _bits(value) -> np.ndarray:
    """Exact bit patterns, so equality is equality and NaN compares too."""

    if value.dtype in (mx.bfloat16, mx.float16):
        return np.array(value.view(mx.uint16))
    if value.dtype == mx.float32:
        return np.array(value.view(mx.uint32))
    return np.array(value)


def _recorded_rounds(monkeypatch) -> list[dict]:
    """Everything one installed fixed-M4 replay produced, round by round."""

    rounds: list[dict] = []
    real = graphbank.CompiledVerifyBank._forward_installed_fixed_m4

    def recording(self, input_ids, host_input_ids, completion_tokens, committed_count, cache):
        logits, hidden, extra = real(
            self, input_ids, host_input_ids, completion_tokens, committed_count, cache
        )
        dispatch = self._fixed_m4_dispatch
        record: dict[str, object] = {
            "capacity": int(dispatch["capacity"]),
            "logits": _bits(logits),
            "hidden": _bits(hidden),
        }
        for n, (kind, entry, n_leaves) in enumerate(dispatch["state_plan"]):
            if kind == graphbank.VERIFY_SPEC_KIND_QSA:
                end = int(entry.size())
                record[f"qsa{n}.rows_gather"] = bool(entry.fixed_rows_gather)
                record[f"qsa{n}.end"] = end
                record[f"qsa{n}.keys"] = _bits(entry.kv.cache[0][..., :end, :])
                record[f"qsa{n}.values"] = _bits(entry.kv.cache[1][..., :end, :])
                record[f"qsa{n}.raw"] = _bits(entry.raw_keys[:, :end])
                record[f"qsa{n}.pooled"] = _bits(entry.pooled[:, : end // entry.ratio])
            else:
                for slot in range(n_leaves):
                    record[f"state{n}.{slot}"] = _bits(entry.cache[slot])
        for n, (entry, _start, _count) in enumerate(dispatch["capture_plan"]):
            for slot, leaf in enumerate(getattr(entry, "_mtplx_verify_rows", ()) or ()):
                record[f"capture{n}.rows{slot}"] = _bits(leaf)
            for slot, leaf in enumerate(getattr(entry, "_mtplx_verify_ple", ()) or ()):
                record[f"capture{n}.ple{slot}"] = _bits(leaf)
        rounds.append(record)
        return logits, hidden, extra

    monkeypatch.setattr(graphbank.CompiledVerifyBank, "_forward_installed_fixed_m4", recording)
    return rounds


def _run(pack, monkeypatch, *, extra_rows: int):
    """One sampled request on the compiled fixed-M4 lane, the bank ``extra_rows`` larger."""

    from mtplx.qwen4_fixed_verify import install_qwen4_fixed_verify_route

    smoke, model = pack
    real_rule = graphbank.TensorOffsetQSACache._bank_capacity

    def rule(needed, ratio, kv_step, **kwargs):
        return real_rule(needed, ratio, kv_step, **kwargs) + extra_rows

    with monkeypatch.context() as patch:
        patch.setattr(graphbank.TensorOffsetQSACache, "_bank_capacity", staticmethod(rule))
        rounds = _recorded_rounds(patch)
        patch.setenv("MTPLX_COMPILED_VERIFY", "1")
        rt = smoke._tiny_runtime(model)
        install_qwen4_fixed_verify_route(rt)
        result = generation.generate_mtpk(
            rt,
            list(PROMPT),
            max_tokens=MAX_TOKENS,
            sampler=NATIVE,
            draft_sampler=NATIVE,
            speculative_depth=3,
            seed=SEED,
            mtp_cache_policy="persistent",
            mtp_history_policy="committed",
            verify_strategy="batched",
            stop_token_ids=set(),
        )
    return result, rounds


def _assert_same_rounds(narrow: list[dict], wide: list[dict], *, bucket: int) -> None:
    assert len(narrow) == len(wide) >= 8
    for index, (a, b) in enumerate(zip(narrow, wide)):
        assert b["capacity"] == a["capacity"] + bucket, index
        assert a.keys() == b.keys(), index
        for name in a:
            if name == "capacity":
                continue
            left, right = a[name], b[name]
            if isinstance(left, np.ndarray):
                assert left.shape == right.shape, (index, name)
                assert np.array_equal(left, right), (index, name)
            else:
                assert left == right, (index, name)


@pytest.mark.parametrize("extra_rows", [256, 16_384])
def test_rows_gather_verify_does_not_depend_on_the_capacity(pack, lane, extra_rows):
    """The invariant a capacity bucket rests on, on the compiled fixed-M4 verifier.

    256 rows is one K/V step; 16,384 puts the wide bank past the rows-gather
    switch length while the narrow one stays below it.
    """

    lane.setenv("MTPLX_QSA_GATHER", "1")
    lane.setenv("MTPLX_QSA_GATHER_MIN_CONTEXT", "16")  # the tiny prompt is past it
    narrow, narrow_rounds = _run(pack, lane, extra_rows=0)
    wide, wide_rounds = _run(pack, lane, extra_rows=extra_rows)

    if extra_rows == 16_384:
        assert narrow_rounds[0]["capacity"] < 16_384 < wide_rounds[0]["capacity"]
    assert all(r["qsa1.rows_gather"] for r in narrow_rounds + wide_rounds)
    _assert_same_rounds(narrow_rounds, wide_rounds, bucket=extra_rows)
    assert list(wide.tokens) == list(narrow.tokens)
    report = (wide.stats.graphbank or {}).get("compiled_verify") or {}
    assert report["compiled_calls"] == len(wide_rounds) and report["fallback_calls"] == 0



def _prompt(tokens: int) -> list[int]:
    return [(7 * i + 3) % 128 for i in range(tokens)]


def _session(pack, monkeypatch, *, bucket: int, prompts, max_tokens: int = 12):
    """A growing session on one runtime: per turn, the bank capacity and the verify traces it paid."""

    from mtplx.qwen4_fixed_verify import install_qwen4_fixed_verify_route

    smoke, model = pack
    monkeypatch.setenv("MTPLX_COMPILED_VERIFY", "1")
    monkeypatch.setenv("MTPLX_QWEN4_FIXED_M4_CAPACITY_BUCKET", str(bucket))
    rt = smoke._tiny_runtime(model)
    install_qwen4_fixed_verify_route(rt)
    turns = []
    for tokens in prompts:
        result = generation.generate_mtpk(
            rt,
            _prompt(tokens),
            max_tokens=max_tokens,
            sampler=NATIVE,
            draft_sampler=NATIVE,
            speculative_depth=3,
            seed=SEED,
            mtp_cache_policy="persistent",
            mtp_history_policy="committed",
            verify_strategy="batched",
            stop_token_ids=set(),
        )
        report = result.stats.graphbank["compiled_verify"]
        assert report["fallback_calls"] == 0 and report["compiled_calls"] > 0
        turns.append((report["fixed_m4"]["capacity"], report["traces"]))
    return turns



# -- the rule ---------------------------------------------------------------


def test_the_bucket_rounds_a_rows_gather_capacity_to_its_edge():
    rule = graphbank.TensorOffsetQSACache._bank_capacity
    for needed in (1, 17_408, 24_576, 24_577, 41_024, 99_999, 262_143, 1_001_024):
        cap = rule(needed, 4, 256, rows_gather=True, bucket=8192)
        assert cap % 8192 == 0
        assert needed <= cap < needed + 8192
        # Never below what the 256-row step alone gives, never a whole bucket over it.
        step = rule(needed, 4, 256, rows_gather=True)
        assert step <= cap <= step + 8192 - 256
    # A bucket off the step rounds up to the step; 0 is the step alone.
    assert rule(1, 4, 256, rows_gather=True, bucket=3000) == 3072
    assert rule(3073, 4, 256, rows_gather=True, bucket=3000) == 6144
    assert rule(1, 4, 256, rows_gather=True, bucket=100) == 256
    for needed in (1, 257, 17_408, 41_024):
        assert rule(needed, 4, 256, rows_gather=True, bucket=0) == rule(
            needed, 4, 256, rows_gather=True
        )


def test_the_dense_lane_ignores_the_bucket():
    rule = graphbank.TensorOffsetQSACache._bank_capacity
    for needed in (45, 4_577, 9_024, 16_866):
        assert rule(needed, 4, 256, rows_gather=False, bucket=8192) == 4 * math.ceil(needed / 4)


def test_the_bucket_setting(monkeypatch):
    monkeypatch.delenv("MTPLX_QWEN4_FIXED_M4_CAPACITY_BUCKET", raising=False)
    assert graphbank._fixed_m4_capacity_bucket() == 8192
    for raw, bucket in (("0", 0), ("4096", 4096), ("-5", 0), ("junk", 8192), (" ", 8192)):
        monkeypatch.setenv("MTPLX_QWEN4_FIXED_M4_CAPACITY_BUCKET", raw)
        assert graphbank._fixed_m4_capacity_bucket() == bucket, raw


def test_only_the_strict_fixed_m4_bank_takes_the_bucket(monkeypatch):
    monkeypatch.setattr(graphbank, "_compiled_verify_bits_gate_ok", lambda _rt: True)
    monkeypatch.setenv("MTPLX_QWEN4_FIXED_M4_CAPACITY_BUCKET", "4096")
    strict = graphbank.CompiledVerifyBank(
        SimpleNamespace(qwen4_fixed_m4_compiled_verify=True), max_verify_len=4
    )
    generic = graphbank.CompiledVerifyBank(SimpleNamespace(), max_verify_len=4)
    assert strict.fixed_m4_capacity_bucket == 4096
    assert generic.fixed_m4_capacity_bucket == 0
    assert strict.fixed_m4_capacity_receipt() is None  # nothing installed yet


# -- the bank ---------------------------------------------------------------


@pytest.fixture()
def cpu_layer():
    from mtplx.models.qwen4_exp import Attention, TextArgs

    prev = mx.default_device()
    mx.set_default_device(mx.cpu)
    mx.random.seed(3)
    layer = Attention(
        TextArgs(
            hidden_size=64,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            indexer_n_heads=2,
            indexer_kv_heads=1,
            indexer_head_dim=16,
            indexer_budget=8,
            indexer_compress_ratio=2,
        )
    )
    layer.update(mlx.utils.tree_map(lambda p: p.astype(mx.bfloat16), layer.parameters()))
    mx.eval(layer.parameters())
    yield layer
    mx.set_default_device(prev)


def _prefilled_entry(layer, tokens: int = 13):
    from mtplx.models.qwen4_exp import QSACache

    entry = QSACache(compress_ratio=layer.indexer.ratio)
    mx.random.seed(2)
    layer(mx.random.normal((1, tokens, 64)).astype(mx.bfloat16), entry)
    return entry


def test_a_rows_gather_bank_is_built_and_grown_on_bucket_edges(cpu_layer, monkeypatch):
    monkeypatch.setenv("MTPLX_QSA_GATHER", "1")
    monkeypatch.setenv("MTPLX_QSA_GATHER_MIN_CONTEXT", "8")
    monkeypatch.delenv("MTPLX_QSA_M4_FUSED_KV_GATHER", raising=False)
    Bank = graphbank.TensorOffsetQSACache

    bank = Bank.from_qsa_cache(_prefilled_entry(cpu_layer), reserve_tokens=32, capacity_bucket=1024)
    assert bank.fixed_rows_gather and bank.capacity_bucket == 1024
    assert bank.capacity == 1024
    assert int(bank.kv.keys.shape[2]) == int(bank.kv.values.shape[2]) == 1024
    assert int(bank.pooled.shape[1]) == 1024 // cpu_layer.indexer.ratio
    assert bank.ensure_capacity(1024) is False
    assert bank.ensure_capacity(1025) is True
    assert bank.capacity == 2048 and int(bank.raw_keys.shape[1]) == 2048

    # Through the promotion entry point: only a caller that passes a bucket gets one.
    plain = [_prefilled_entry(cpu_layer)]
    graphbank.promote_kv_cache_offsets(plain, reserve_tokens=4, initial_reserve_tokens=32)
    bucketed = [_prefilled_entry(cpu_layer)]
    graphbank.promote_kv_cache_offsets(
        bucketed, reserve_tokens=4, initial_reserve_tokens=32, qsa_capacity_bucket=1024
    )
    assert plain[0].capacity == 256 and plain[0].capacity_bucket == 0
    assert bucketed[0].capacity == 1024 and bucketed[0].capacity_bucket == 1024


def test_a_dense_bank_keeps_its_exact_capacity_under_a_bucket(cpu_layer, monkeypatch):
    monkeypatch.delenv("MTPLX_QSA_GATHER", raising=False)
    monkeypatch.delenv("MTPLX_QSA_GATHER_MIN_CONTEXT", raising=False)

    bank = graphbank.TensorOffsetQSACache.from_qsa_cache(
        _prefilled_entry(cpu_layer), reserve_tokens=32, capacity_bucket=1024
    )
    assert bank.fixed_rows_gather is False
    assert bank.capacity == 46  # 13 + 32 on the ratio, as without a bucket
    assert bank.ensure_capacity(47) is True
    assert bank.capacity == 48


# -- admission ----------------------------------------------------------------

GB = 1_000_000_000
PER_TOKEN = 28_416  # Flash-Next: 12 QSA layers x (2,048 + 256 + 64) bytes


def _admit(monkeypatch, *, prompt: int, live: int, limit: int = 100 * GB):
    monkeypatch.setattr(generation, "_mlx_live_memory_bytes", lambda: live)
    monkeypatch.setattr(generation, "_mlx_release_allocator_cache", lambda: 0)
    monkeypatch.setattr(generation, "_qwen4_fixed_m4_promotion_bytes_per_token", lambda rt: PER_TOKEN)
    monkeypatch.setattr(generation, "_metal_memory_limit_bytes", lambda rt: limit)
    monkeypatch.setattr(generation, "_announce_qwen4_fixed_m4_skip", lambda message: None)
    monkeypatch.delenv("MTPLX_QWEN4_FIXED_M4_MAX_CONTEXT", raising=False)
    monkeypatch.delenv("MTPLX_COMPILED_VERIFY_GROWTH_RESERVE", raising=False)
    receipt: dict = {}
    fits = generation._qwen4_fixed_m4_lane_fits(SimpleNamespace(), prompt_tokens=prompt, receipt=receipt)
    return fits, receipt


def test_admission_prices_the_bucketed_bank(monkeypatch):
    monkeypatch.setenv("MTPLX_QSA_GATHER", "1")
    monkeypatch.delenv("MTPLX_QSA_GATHER_MIN_CONTEXT", raising=False)
    monkeypatch.delenv("MTPLX_QWEN4_FIXED_M4_CAPACITY_BUCKET", raising=False)
    prompt = 40_000  # rows-gather lane; 41,024 rows with the 1,024-row reserve
    line = int(100 * GB * 0.97)
    bucketed_need = 49_152 * PER_TOKEN
    step_need = 41_216 * PER_TOKEN

    # Enough room for the parent's bank: preserve its compiled route without
    # evicting a cached session merely to buy bucket slack.
    live = line - (step_need + bucketed_need) // 2
    fits, receipt = _admit(monkeypatch, prompt=prompt, live=live)
    assert fits is True
    assert receipt["promotion_rows"] == 41_216
    assert receipt["promotion_bytes"] == step_need
    assert receipt["capacity_bucket"] == 0

    # The same machine with the bucket off admits the request, priced on the step.
    monkeypatch.setenv("MTPLX_QWEN4_FIXED_M4_CAPACITY_BUCKET", "0")
    fits, receipt = _admit(monkeypatch, prompt=prompt, live=live)
    assert fits is True
    assert receipt["promotion_rows"] == 41_216
    assert receipt["promotion_bytes"] == step_need

    # Room for the bucketed bank: admitted.
    monkeypatch.delenv("MTPLX_QWEN4_FIXED_M4_CAPACITY_BUCKET")
    fits, receipt = _admit(monkeypatch, prompt=prompt, live=line - bucketed_need)
    assert fits is True and receipt["promotion_rows"] == 49_152


def test_admission_below_the_rows_gather_switch_is_priced_as_before(monkeypatch):
    monkeypatch.setenv("MTPLX_QSA_GATHER", "1")
    monkeypatch.delenv("MTPLX_QSA_GATHER_MIN_CONTEXT", raising=False)
    monkeypatch.delenv("MTPLX_QWEN4_FIXED_M4_CAPACITY_BUCKET", raising=False)
    fits, receipt = _admit(monkeypatch, prompt=8_000, live=10 * GB)
    assert fits is True
    assert receipt["promotion_rows"] == 9_024  # dense lane: prompt + reserve on the ratio
    assert receipt["promotion_bytes"] == 9_024 * PER_TOKEN


# -- a growing session on the compiled lane ---------------------------------

SESSION = (40, 340, 640, 940, 1240, 1540)  # prompt tokens per turn; the tiny K/V step is 256


def test_a_growing_session_pays_one_trace_per_bucket_not_per_turn(pack, lane):
    lane.setenv("MTPLX_QSA_GATHER", "1")
    lane.setenv("MTPLX_QSA_GATHER_MIN_CONTEXT", "16")

    old = _session(pack, lane, bucket=0, prompts=SESSION)
    new = _session(pack, lane, bucket=1024, prompts=SESSION)

    # The 256-row step: a capacity the process has not traced on every turn.
    assert [capacity for capacity, _ in old] == [256, 512, 768, 1024, 1280, 1792]
    assert [traces for _, traces in old] == [1, 1, 1, 1, 1, 1]
    # The bucket: one trace per bucket edge crossed.
    assert [capacity for capacity, _ in new] == [1024, 1024, 1024, 1024, 2048, 2048]
    assert [traces for _, traces in new] == [1, 0, 0, 0, 1, 0]


def test_the_dense_lane_keeps_one_trace_per_turn(pack, lane):
    lane.setenv("MTPLX_QSA_GATHER", "0")

    turns = _session(pack, lane, bucket=1024, prompts=SESSION[:3])

    # prompt + (12 tokens + 4 headroom) on the ratio: the bucket does not apply.
    assert [capacity for capacity, _ in turns] == [56, 356, 656]
    assert [traces for _, traces in turns] == [1, 1, 1]


def test_the_bucketed_bank_gives_the_same_bits_as_the_step(pack, lane):
    """The parity above, through the policy itself: bucket off against bucket on."""

    lane.setenv("MTPLX_QSA_GATHER", "1")
    lane.setenv("MTPLX_QSA_GATHER_MIN_CONTEXT", "16")
    lane.setenv("MTPLX_QWEN4_FIXED_M4_CAPACITY_BUCKET", "0")
    step, step_rounds = _run(pack, lane, extra_rows=0)
    lane.setenv("MTPLX_QWEN4_FIXED_M4_CAPACITY_BUCKET", "1024")
    bucketed, bucketed_rounds = _run(pack, lane, extra_rows=0)

    assert step_rounds[0]["capacity"] == 256 and bucketed_rounds[0]["capacity"] == 1024
    _assert_same_rounds(step_rounds, bucketed_rounds, bucket=768)
    assert list(bucketed.tokens) == list(step.tokens)


def test_the_route_tape_logs_the_bucket_and_its_transitions(pack, lane):
    from mtplx.qwen4_fixed_verify import install_qwen4_fixed_verify_route
    from mtplx.route_tape import set_route_tape_sink

    lane.setenv("MTPLX_QSA_GATHER", "1")
    lane.setenv("MTPLX_QSA_GATHER_MIN_CONTEXT", "16")
    lane.setenv("MTPLX_QWEN4_FIXED_M4_CAPACITY_BUCKET", "512")
    # A 16-row first grant, so the request outgrows its first bucket.
    lane.setenv("MTPLX_COMPILED_VERIFY_GROWTH_RESERVE", "16")
    lane.setenv("MTPLX_COMPILED_VERIFY", "1")
    lane.setenv("MTPLX_ROUTE_TAPE", "1")
    lane.delenv("MTPLX_ROUTE_TAPE_JSONL", raising=False)
    smoke, model = pack
    rt = smoke._tiny_runtime(model)
    install_qwen4_fixed_verify_route(rt)
    rows: list[dict] = []
    set_route_tape_sink(rows.append)
    try:
        result = generation.generate_mtpk(
            rt,
            list(PROMPT),
            max_tokens=600,
            sampler=NATIVE,
            draft_sampler=NATIVE,
            speculative_depth=3,
            seed=SEED,
            mtp_cache_policy="persistent",
            mtp_history_policy="committed",
            verify_strategy="batched",
            stop_token_ids=set(),
        )
    finally:
        set_route_tape_sink(None)

    report = result.stats.graphbank["compiled_verify"]
    header = next(row for row in rows if row["name"] == "header")["attrs"]
    rounds = [row["attrs"] for row in rows if row["name"] == "round"]
    transitions = [r["fixed_m4_capacity"] for r in rounds if r["fixed_m4_capacity"]]

    assert header["fixed_m4"] == {
        "capacity": 512,
        "capacity_bucket": 512,
        "rows_gather": True,
        "base_offset": len(PROMPT),
    }
    assert transitions == [{"from": 512, "to": 1024, "bucket": 512, "rows_gather": True}]
    assert report["fixed_m4_capacity_transitions"] == 1
    assert report["fixed_m4"]["capacity"] == 1024
    assert report["fixed_m4"]["capacity_bucket"] == 512 and report["fixed_m4"]["rows_gather"]
    # Every trace the request paid is on the tape, on the round that paid it:
    # the first round and the first round at the new capacity.
    assert sum(r["verify_traces"] for r in rounds) == report["traces"] == 2
    traced = [i for i, r in enumerate(rounds) if r["verify_traces"]]
    grown = next(i for i, r in enumerate(rounds) if r["fixed_m4_capacity"])
    assert traced == [0, grown]
