"""Capacity buckets preserve every consumer of a fixed QSA bank."""

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

import mtplx.generation as generation
import mtplx.graphbank as graphbank
from mtplx.cache_state import restore_cache, snapshot_cache, snapshot_untrimmable_cache_lazy
from mtplx.models.qwen4_exp import QSACache
from test_qwen4_fixed_m4_capacity_bucket import (
    NATIVE, PROMPT, SEED, _bits, _prefilled_entry, _prompt, cpu_layer, lane, pack,
)


def _leaves(value):
    if isinstance(value, mx.array):
        return [_bits(value)]
    if isinstance(value, (tuple, list)):
        return [leaf for item in value for leaf in _leaves(item)]
    if isinstance(value, dict):
        return [leaf for item in value.values() for leaf in _leaves(item)]
    return []


def _state(cache):
    result = []
    for entry in cache:
        if isinstance(entry, graphbank.TensorOffsetQSACache):
            end = entry.size()
            result.extend(_leaves((entry.kv.keys[:, :, :end], entry.kv.values[:, :, :end],
                                   entry.raw_keys[:, :end], entry.pooled[:, :end // entry.ratio])))
        else:
            result.extend(_leaves(entry.state))
    return result


def _equal(left, right):
    assert len(left) == len(right)
    for i, (a, b) in enumerate(zip(left, right)):
        assert a.shape == b.shape, (i, a.shape, b.shape)
        assert np.array_equal(a, b), i


@pytest.mark.parametrize("tile", [0, 1, 2, 4])
def test_dense_consumers_keep_the_parent_sdpa_width(cpu_layer, lane, tile):
    import mtplx.models.qwen4_exp as qwen4

    lane.setenv("MTPLX_QSA_GATHER", "1")
    lane.setenv("MTPLX_QSA_GATHER_MIN_CONTEXT", "8")
    lane.setenv("MTPLX_QSA_SCORE_TILE_ROWS", str(tile))
    banks = [graphbank.TensorOffsetQSACache.from_qsa_cache(
        _prefilled_entry(cpu_layer), reserve_tokens=32, capacity_bucket=bucket,
    ) for bucket in (0, 1024)]
    if tile:
        assert banks[1].capacity == banks[0].capacity == 256
    else:
        assert [bank.capacity for bank in banks] == [256, 1024]
    widths = []
    sdpa = qwen4._verify_sdpa

    def observe(q, k, v, **kwargs):
        widths.append(k.shape[2])
        return sdpa(q, k, v, **kwargs)

    lane.setattr(qwen4, "_verify_sdpa", observe)
    for rows in (4, 1, 3, 9, 1):
        mx.random.seed(91 + rows)
        x = mx.random.normal((1, rows, 64)).astype(mx.bfloat16)
        outputs = [cpu_layer(x, bank) for bank in banks]
        _equal(_leaves(outputs[0]), _leaves(outputs[1]))
        if rows == 1 or 0 < tile < rows:
            # A behavioral check on the actual SDPA arguments, independent
            # of whether this device happens to round both widths equally.
            assert widths[-2:] == [256, 256]
        _equal(_state([banks[0]]), _state([banks[1]]))


def test_short_request_admission_prices_the_constructed_bank(lane):
    lane.setenv("MTPLX_QSA_GATHER", "1")
    lane.setenv("MTPLX_QWEN4_FIXED_M4_CAPACITY_BUCKET", "8192")
    args = SimpleNamespace(layer_types=["full_attention"], num_key_value_heads=2,
                           head_dim=16, indexer_head_dim=16, indexer_compress_ratio=4)
    rt = SimpleNamespace(model=SimpleNamespace(args=args), qwen4_fixed_m4_compiled_verify=True)
    lane.setattr(generation, "_mlx_live_memory_bytes", lambda: 0)
    lane.setattr(generation, "_metal_memory_limit_bytes", lambda rt: 2**30)
    plan = graphbank.FixedM4CapacityPlan.for_request(32)
    receipt = {}
    assert generation._qwen4_fixed_m4_compiled_verify_requested(
        rt, verify_strategy="batched", compiled_mode="on", max_tokens=32,
        cached_tokens=0, prompt_tokens=24_000, receipt=receipt, capacity_plan=plan,
    )
    entry = QSACache(4)
    entry.kv.keys = mx.zeros((1, 2, 24_000, 16), dtype=mx.bfloat16)
    entry.kv.values = mx.zeros_like(entry.kv.keys)
    entry.kv.offset = 24_000
    entry.raw_keys = mx.zeros((1, 24_000, 16), dtype=mx.bfloat16)
    entry.pooled = mx.zeros((1, 6_000, 16), dtype=mx.bfloat16)
    entry.pooled_len = 6_000
    bank = graphbank.TensorOffsetQSACache.from_qsa_cache(
        entry, reserve_tokens=plan.reserve_tokens, capacity_plan=plan,
    )
    assert receipt["reserve_tokens"] == 36
    assert bank.capacity == receipt["promotion_rows"] == 24_576
    assert receipt["promotion_bytes"] == bank.nbytes - bank.offset.nbytes


def test_short_request_gate_passes_its_real_budget_to_admission(lane):
    """Fails behaviorally on 50de43bb: its gate charges the 1,024-row reserve."""
    lane.setenv("MTPLX_QSA_GATHER", "1")
    lane.setenv("MTPLX_QWEN4_FIXED_M4_CAPACITY_BUCKET", "0")
    lane.setattr(generation, "_mlx_live_memory_bytes", lambda: 0)
    lane.setattr(generation, "_metal_memory_limit_bytes", lambda rt: 10**9)
    lane.setattr(generation, "_qwen4_fixed_m4_promotion_bytes_per_token", lambda rt: 100)
    receipt = {}
    assert generation._qwen4_fixed_m4_compiled_verify_requested(
        SimpleNamespace(qwen4_fixed_m4_compiled_verify=True),
        verify_strategy="batched", compiled_mode="on", max_tokens=32,
        cached_tokens=0, prompt_tokens=24_000, receipt=receipt,
    )
    assert receipt["promotion_bytes"] == 24_064 * 100


def _partial_session(pack, patch, bucket):
    from mtplx.qwen4_fixed_verify import install_qwen4_fixed_verify_route

    smoke, model = pack
    patch.setenv("MTPLX_QWEN4_FIXED_M4_CAPACITY_BUCKET", str(bucket))
    rt = smoke._tiny_runtime(model)
    install_qwen4_fixed_verify_route(rt)
    cache = model.make_cache()
    rt.forward_ar(mx.array([PROMPT]), cache=cache, return_hidden=True)
    bank = graphbank.CompiledVerifyBank(rt, max_verify_len=4, request_max_tokens=96)
    bank.install_fixed_m4(cache, prompt_ids=PROMPT, hidden_variant=None)
    records, completion, capacities = [], [], []
    for i in range(20):
        ids = [(i * 13 + j * 7) % 128 for j in range(4)]
        snap = snapshot_untrimmable_cache_lazy(cache)
        logits, hidden, _ = bank.forward_fixed_m4(
            mx.array([ids]), host_input_ids=ids, completion_tokens=completion,
            committed_count=len(completion), cache=cache,
        )
        records.extend(_leaves((logits, hidden)))
        for entry in cache:
            records.extend(_leaves(getattr(entry, "_mtplx_verify_rows", ())))
            records.extend(_leaves(getattr(entry, "_mtplx_verify_ple", ())))
        keep = 1 + i % 4
        assert model.language_model.model.commit_verified_window(
            cache, snap.states, keep_tokens=keep, verified_tokens=4,
        )
        completion.extend(ids[:keep])
        records.extend(_state(cache))
        qsa = next(entry for entry in cache if isinstance(entry, graphbank.TensorOffsetQSACache))
        capacities.append((qsa.capacity, qsa.dense_capacity, qsa.fixed_rows_gather))
    # Same one-row call as generation's final-pending capture, before demotion.
    bank.reserve_fixed_m4_window(
        cache, committed_count=len(completion), window_tokens=1, final_capture=True,
    )
    records.extend(_leaves(rt.forward_ar(mx.array([[17]]), cache=cache, return_hidden=True)))
    records.extend(_state(cache))
    bank.demote(cache)
    snapshot = snapshot_cache(cache)
    restored = model.make_cache()
    restore_cache(restored, snapshot)
    records.extend(_state(restored))
    records.extend(_leaves(rt.forward_ar(mx.array([[21, 23, 25]]), cache=restored, return_hidden=True)))
    records.extend(_state(restored))
    return records, capacities


def test_partial_acceptance_restore_and_dense_to_gather_growth_are_identical(pack, lane):
    lane.setenv("MTPLX_QSA_GATHER", "1")
    lane.setenv("MTPLX_QSA_GATHER_MIN_CONTEXT", "48")
    lane.setenv("MTPLX_COMPILED_VERIFY_GROWTH_RESERVE", "4")
    narrow, narrow_caps = _partial_session(pack, lane, 0)
    wide, wide_caps = _partial_session(pack, lane, 1024)
    assert not narrow_caps[0][2] and narrow_caps[-1][2]
    assert any(a[0] != b[0] for a, b in zip(narrow_caps, wide_caps))
    assert [a[1:] for a in narrow_caps] == [b[1:] for b in wide_caps]
    _equal(narrow, wide)


def _captured_session(pack, patch, bucket):
    from mtplx.qwen4_fixed_verify import install_qwen4_fixed_verify_route
    from mtplx.session_bank import SessionBank

    smoke, model = pack
    patch.setenv("MTPLX_COMPILED_VERIFY", "1")
    patch.setenv("MTPLX_QWEN4_FIXED_M4_CAPACITY_BUCKET", str(bucket))
    rt = smoke._tiny_runtime(model)
    install_qwen4_fixed_verify_route(rt)
    bank = SessionBank(max_bytes=2**28, per_session_max_bytes=2**28)
    pending_calls = []
    forward = rt.forward_ar

    def observe(ids, **kwargs):
        if ids.shape[1] == 1 and any(
            isinstance(entry, graphbank.TensorOffsetQSACache)
            for entry in kwargs.get("cache", ())
        ):
            pending_calls.append(True)
        return forward(ids, **kwargs)

    patch.setattr(rt, "forward_ar", observe)
    prompt = list(PROMPT)
    records, tokens, restored_counts = [], [], []
    for turn in range(2):
        result = generation.generate_mtpk(
            rt, prompt, max_tokens=40, sampler=NATIVE, draft_sampler=NATIVE,
            speculative_depth=3, seed=SEED, mtp_cache_policy="persistent",
            mtp_history_policy="committed", verify_strategy="batched", stop_token_ids=set(),
            capture_final_state=True, session_bank=bank,
        )
        final = result.final_state
        assert final is not None and final.safe_to_commit
        records.extend(_leaves((final.final_logits, final.final_hidden)))
        records.extend(_state(final.final_trunk_cache))
        records.extend(_state(final.final_committed_mtp_cache))
        tokens.append(list(result.tokens))
        restored_counts.append(result.stats.cached_tokens)
        prefix = prompt + list(result.tokens)
        assert bank.put(
            runtime=rt, token_ids=prefix, cache=final.final_trunk_cache,
            logits=final.final_logits, hidden=final.final_hidden,
            hidden_variant=generation._resolve_runtime_base_hidden_variant(rt, None),
            mtp_history_policy="committed",
            mtp_history_snapshot=snapshot_cache(final.final_committed_mtp_cache),
        ) is not None
        prompt = prefix + [31, 37, 41, 43]
    assert restored_counts[0] == 0 and restored_counts[1] > 0
    assert pending_calls, "must exercise the one-row forward before bank demotion"
    return records, tokens


def test_final_pending_capture_and_next_session_bank_turn_are_identical(pack, lane):
    lane.setenv("MTPLX_QSA_GATHER", "1")
    lane.setenv("MTPLX_QSA_GATHER_MIN_CONTEXT", "16")
    narrow, narrow_tokens = _captured_session(pack, lane, 0)
    wide, wide_tokens = _captured_session(pack, lane, 1024)
    assert narrow_tokens == wide_tokens
    _equal(narrow, wide)


def _extend_qsa_history(cache, tokens):
    for entry in cache:
        if not isinstance(entry, QSACache):
            continue
        keys, values, raw, pooled = entry.state

        def repeat(value, rows, axis):
            repeats = [1] * value.ndim
            repeats[axis] = (rows + value.shape[axis] - 1) // value.shape[axis]
            return graphbank.TensorOffsetQSACache._fixed_bank(
                mx.tile(value, repeats), rows, axis,
            )

        entry.state = (repeat(keys, tokens, 2), repeat(values, tokens, 2),
                       repeat(raw, tokens, 1), repeat(pooled, tokens // entry.ratio, 1))


def _grant_boundary_session(pack, patch, bucket, *, final_keep=4, deny_growth=False):
    """Run from synthetic 16K history to a stop bonus or correction.

    Only the cold history and accept coins are controlled. Verification,
    the final one-row forward, publication, and the next turn use the real
    implementations. No caller reserves on behalf of final capture.
    """
    from dataclasses import replace

    from mtplx.qwen4_fixed_verify import install_qwen4_fixed_verify_route
    from mtplx.session_bank import SessionBank

    smoke, model = pack
    patch.setenv("MTPLX_COMPILED_VERIFY", "1")
    patch.setenv("MTPLX_COMPILED_VERIFY_GROWTH_RESERVE", "1024")
    patch.setenv("MTPLX_QWEN4_FIXED_M4_CAPACITY_BUCKET", str(bucket))
    patch.setenv("MTPLX_BATCH_TARGET_ARRAYS", "0")
    patch.setenv("MTPLX_BATCH_TARGET_DISTS", "0")
    patch.setenv("MTPLX_DROP_EVENTS", "0")
    if final_keep < 4:
        patch.setenv("MTPLX_FAMILY_CAPTURE_COMMIT", "1")
    rt = smoke._tiny_runtime(model)
    install_qwen4_fixed_verify_route(rt)
    session = SessionBank(max_bytes=2**28, per_session_max_bytes=2**28)
    restore_or_prefill = generation.restore_or_prefill_prompt_state
    prompt = _prompt(16_384)

    def prefill(runtime, ids, **kwargs):
        if ids != prompt:
            return restore_or_prefill(runtime, ids, **kwargs)
        state = restore_or_prefill(runtime, PROMPT, **kwargs)
        _extend_qsa_history(state.trunk_cache, len(ids))
        return replace(state, token_prefix=tuple(ids), suffix_tokens=len(ids))

    patch.setattr(generation, "restore_or_prefill_prompt_state", prefill)
    default_rng = np.random.default_rng

    class AcceptAll:
        def __init__(self, seed):
            self.delegate = default_rng(seed)
            self.final_draws = 0

        def random(self, *args, **kwargs):
            if len(windows) == 256 and final_keep < 4:
                self.final_draws += 1
                if self.final_draws == final_keep:
                    return float("inf")  # force the final rejection, even p == 1
            return 0.0  # force all three draft accepts, including p == 0

        def __getattr__(self, name):
            return getattr(self.delegate, name)

    patch.setattr(np.random, "default_rng", AcceptAll)
    real_forward = rt.forward_ar
    real_verify = graphbank.CompiledVerifyBank._forward_installed_fixed_m4
    windows, pending = [], []
    stop = 127
    sample = generation._sample_from_logits
    sample_distribution = generation.sample_from_distribution
    is_stop = generation._is_stop
    bonus_sampled = False

    def sample_at_boundary(*args, **kwargs):
        nonlocal bonus_sampled
        token, distribution = sample(*args, **kwargs)
        # The sampler call after the 256th accepted M4 window is the bonus.
        if len(windows) == 256:
            token = stop
            bonus_sampled = True
        elif token == stop:
            token = 126
        return token, distribution

    def correction_at_boundary(*args, **kwargs):
        nonlocal bonus_sampled
        if len(windows) == 256 and final_keep < 4:
            bonus_sampled = True
            return stop
        return sample_distribution(*args, **kwargs)

    if deny_growth:
        install = graphbank.CompiledVerifyBank.install_fixed_m4

        def install_with_no_growth(self, *args, **kwargs):
            install(self, *args, **kwargs)
            self.capacity_plan.admit_growth = lambda need: False

        patch.setattr(graphbank.CompiledVerifyBank, "install_fixed_m4", install_with_no_growth)

    def verify(self, input_ids, host_input_ids, completion_tokens, committed_count, cache):
        result = real_verify(self, input_ids, host_input_ids, completion_tokens,
                             committed_count, cache)
        qsa = next(e for e in cache if isinstance(e, graphbank.TensorOffsetQSACache))
        windows.append((qsa.size(), qsa.dense_capacity, qsa.capacity))
        return result

    def forward(ids, **kwargs):
        qsa = next((e for e in kwargs.get("cache", ())
                    if isinstance(e, graphbank.TensorOffsetQSACache)), None)
        if qsa is not None and ids.shape[1] == 1:
            pending.append((qsa.size(), qsa.dense_capacity, qsa.capacity))
        return real_forward(ids, **kwargs)

    patch.setattr(generation, "_sample_from_logits", sample_at_boundary)
    patch.setattr(generation, "sample_from_distribution", correction_at_boundary)
    patch.setattr(generation, "_is_stop", lambda token, stops: bonus_sampled and is_stop(token, stops))
    patch.setattr(graphbank.CompiledVerifyBank, "_forward_installed_fixed_m4", verify)
    patch.setattr(rt, "forward_ar", forward)
    first = generation.generate_mtpk(
        rt, prompt, max_tokens=1032, sampler=NATIVE, draft_sampler=NATIVE,
        speculative_depth=3, seed=SEED, mtp_cache_policy="persistent",
        mtp_history_policy="committed", verify_strategy="batched",
        stop_token_ids={stop}, capture_final_state=True, session_bank=session,
    )
    final = first.final_state
    assert len(windows) == 256 and windows[-1][:2] == (17_408, 17_408)
    assert windows[-1][2] == (24_576 if bucket else 17_408)
    final_offset = 17_404 + final_keep
    assert len(first.tokens) == 1021 + final_keep and first.tokens[-1] == stop
    assert first.finish_reason == "stop"
    assert first.stats.finish_stop_origin == ("bonus" if final_keep == 4 else "residual_correction")
    if final_keep < 4:
        assert first.stats.deferred_correction_repairs == 1
    assert final is not None and final.safe_to_commit
    assert pending[-1][0] == final_offset
    if final_keep < 4:
        assert pending[-1][1] == 17_408, "a final row inside the grant must not widen attention"
    else:
        assert pending[-1][1] >= 17_409, "final capture must reserve its attention row"
    qsa = next(e for e in final.final_trunk_cache if isinstance(e, QSACache))
    assert qsa.offset == qsa.state[0].shape[2] == final_offset + 1
    restored = model.make_cache()
    restore_cache(restored, snapshot_cache(final.final_trunk_cache))
    assert next(e.offset for e in restored if isinstance(e, QSACache)) == final_offset + 1
    records = _leaves((final.final_logits, final.final_hidden)) + _state(restored)
    prefix = prompt + list(first.tokens)
    assert session.put(
        runtime=rt, token_ids=prefix, cache=final.final_trunk_cache,
        logits=final.final_logits, hidden=final.final_hidden,
        hidden_variant=generation._resolve_runtime_base_hidden_variant(rt, None),
        mtp_history_policy="committed",
        mtp_history_snapshot=snapshot_cache(final.final_committed_mtp_cache),
    ) is not None
    # Restore normal sampling for the next turn; the cached boundary is real.
    patch.setattr(np.random, "default_rng", default_rng)
    patch.setattr(generation, "_sample_from_logits", sample)
    patch.setattr(generation, "sample_from_distribution", sample_distribution)
    patch.setattr(generation, "_is_stop", is_stop)
    second = generation.generate_mtpk(
        rt, prefix + [31, 37, 41, 43], max_tokens=16,
        sampler=NATIVE, draft_sampler=NATIVE, speculative_depth=3, seed=SEED,
        mtp_cache_policy="persistent", mtp_history_policy="committed",
        verify_strategy="batched", stop_token_ids=set(), capture_final_state=True,
        session_bank=session,
    )
    assert second.stats.cached_tokens == final_offset + 1
    assert second.final_state.safe_to_commit
    records += _leaves((second.final_state.final_logits, second.final_state.final_hidden))
    records += _state(second.final_state.final_trunk_cache)
    records += _state(second.final_state.final_committed_mtp_cache)
    return list(first.tokens), list(second.tokens), second.stats.cached_tokens, records


def test_grant_boundary_capture_and_next_session_bank_turn(pack, lane):
    lane.setenv("MTPLX_QSA_GATHER", "1")
    with lane.context() as patch:
        wide = _grant_boundary_session(pack, patch, 8192)
    with lane.context() as patch:
        narrow = _grant_boundary_session(pack, patch, 0)
    assert narrow[:3] == wide[:3]
    _equal(narrow[3], wide[3])


@pytest.mark.parametrize("final_keep", [1, 2, 3])
def test_final_correction_inside_grant_keeps_width_and_restored_bits(pack, lane, final_keep):
    lane.setenv("MTPLX_QSA_GATHER", "1")
    with lane.context() as patch:
        wide = _grant_boundary_session(pack, patch, 8192, final_keep=final_keep)
    with lane.context() as patch:
        narrow = _grant_boundary_session(pack, patch, 0, final_keep=final_keep)
    assert narrow[:3] == wide[:3]
    _equal(narrow[3], wide[3])


def test_final_correction_inside_grant_survives_denied_growth(pack, lane):
    lane.setenv("MTPLX_QSA_GATHER", "1")
    _grant_boundary_session(pack, lane, 0, final_keep=3, deny_growth=True)


def _lazy_bonus_boundary_session(pack, patch, bucket):
    from dataclasses import replace

    from mtplx.qwen4_fixed_verify import install_qwen4_fixed_verify_route
    from mtplx.session_bank import SessionBank

    for name, value in {
        "MTPLX_COMPILED_VERIFY": "1",
        "MTPLX_COMPILED_VERIFY_GROWTH_RESERVE": "4",
        "MTPLX_QWEN4_FIXED_M4_CAPACITY_BUCKET": str(bucket),
        "MTPLX_LAZY_BONUS_VERIFY": "1",
        "MTPLX_LAZY_BONUS_VERIFY_MIN_DEPTH": "1",
        "MTPLX_LAZY_TARGET_DISTRIBUTIONS": "0",
        "MTPLX_BATCH_TARGET_ARRAYS": "0",
        "MTPLX_BATCH_TARGET_DISTS": "0",
        "MTPLX_DROP_EVENTS": "0",
        "MTPLX_LATE_DEPTH_SWITCH_AFTER_TOKENS": "5",
        "MTPLX_LATE_DEPTH_BEFORE": "3",
        "MTPLX_LATE_DEPTH_AFTER": "1",
    }.items():
        patch.setenv(name, value)
    smoke, model = pack
    rt = smoke._tiny_runtime(model)
    install_qwen4_fixed_verify_route(rt)
    session = SessionBank(max_bytes=2**28, per_session_max_bytes=2**28)
    prompt = _prompt(17_403)
    restore_or_prefill = generation.restore_or_prefill_prompt_state

    def prefill(runtime, ids, **kwargs):
        if ids != prompt:
            return restore_or_prefill(runtime, ids, **kwargs)
        state = restore_or_prefill(runtime, PROMPT, **kwargs)
        _extend_qsa_history(state.trunk_cache, len(ids))
        return replace(state, token_prefix=tuple(ids), suffix_tokens=len(ids))

    default_rng = np.random.default_rng

    class AcceptAll:
        def __init__(self, seed):
            self.delegate = default_rng(seed)

        def random(self, *args, **kwargs):
            return 0.0

        def __getattr__(self, name):
            return getattr(self.delegate, name)

    records, windows, writes = [], [], []
    real_verify = graphbank.CompiledVerifyBank.forward_ar_capture
    real_forward = rt.forward_ar

    def verify(self, ids, **kwargs):
        cache = kwargs["cache"]
        qsa = next(e for e in cache if isinstance(e, graphbank.TensorOffsetQSACache))
        windows.append((ids.shape[1], qsa.size(), qsa.dense_capacity, qsa.capacity))
        result = real_verify(self, ids, **kwargs)
        records.extend(_leaves(result[:2]) + _state(cache))
        return result

    def forward(ids, **kwargs):
        cache = kwargs.get("cache", ())
        qsa = next((e for e in cache if isinstance(e, graphbank.TensorOffsetQSACache)), None)
        if qsa is not None and ids.shape[1] == 1:
            start, width = qsa.size(), qsa.attention_capacity(1)
            writes.append((start, width))
            assert width > start, "lazy bonus write must see its own row"
        result = real_forward(ids, **kwargs)
        if qsa is not None:
            records.extend(_leaves(result) + _state(cache))
        return result

    patch.setattr(generation, "restore_or_prefill_prompt_state", prefill)
    patch.setattr(np.random, "default_rng", AcceptAll)
    patch.setattr(graphbank.CompiledVerifyBank, "forward_ar_capture", verify)
    patch.setattr(rt, "forward_ar", forward)
    options = dict(max_tokens=8, sampler=NATIVE, draft_sampler=NATIVE,
                   speculative_depth=3, seed=SEED, mtp_cache_policy="persistent",
                   mtp_history_policy="committed", verify_strategy="batched",
                   stop_token_ids=set(), capture_final_state=True, session_bank=session)
    first = generation.generate_mtpk(rt, prompt, **options)
    # The second round grows the bank at its top (reserve_fixed_m4_round),
    # with the reservation its one-row verify used to make on entry, so the
    # verify is called with the grown bank; every write sees what it saw.
    assert windows[:2] == [(3, 17_403, 17_408, 24_576 if bucket else 17_408),
                           (1, 17_407, 17_664, 24_576 if bucket else 17_664)]
    assert (17_408, 17_664) in writes
    assert first.stats.events[1]["lazy_bonus_verify"]["enabled"]
    final = first.final_state
    assert final is not None and final.safe_to_commit
    records.extend(_leaves((final.final_logits, final.final_hidden)))
    records.extend(_state(final.final_trunk_cache) + _state(final.final_committed_mtp_cache))
    prefix = prompt + list(first.tokens)
    assert session.put(
        runtime=rt, token_ids=prefix, cache=final.final_trunk_cache,
        logits=final.final_logits, hidden=final.final_hidden,
        hidden_variant=generation._resolve_runtime_base_hidden_variant(rt, None),
        mtp_history_policy="committed",
        mtp_history_snapshot=snapshot_cache(final.final_committed_mtp_cache),
    ) is not None
    patch.setattr(np.random, "default_rng", default_rng)
    second = generation.generate_mtpk(rt, prefix + [31, 37, 41, 43], **options)
    assert second.stats.cached_tokens == len(prefix)
    final = second.final_state
    assert final is not None and final.safe_to_commit
    records.extend(_leaves((final.final_logits, final.final_hidden)))
    records.extend(_state(final.final_trunk_cache) + _state(final.final_committed_mtp_cache))
    return list(first.tokens), list(second.tokens), second.stats.cached_tokens, records


def test_one_row_lazy_bonus_keeps_second_write_and_restored_bits(pack, lane):
    lane.setenv("MTPLX_QSA_GATHER", "1")
    with lane.context() as patch:
        wide = _lazy_bonus_boundary_session(pack, patch, 8192)
    with lane.context() as patch:
        narrow = _lazy_bonus_boundary_session(pack, patch, 0)
    assert narrow[:3] == wide[:3]
    _equal(narrow[3], wide[3])


@pytest.mark.parametrize("allow_step", [False, True])
def test_growth_is_admitted_before_any_leaf_changes(pack, lane, allow_step):
    from mtplx.qwen4_fixed_verify import install_qwen4_fixed_verify_route

    smoke, model = pack
    lane.setenv("MTPLX_QSA_GATHER", "1")
    lane.setenv("MTPLX_QSA_GATHER_MIN_CONTEXT", "16")
    lane.setenv("MTPLX_QWEN4_FIXED_M4_CAPACITY_BUCKET", "512")
    lane.setenv("MTPLX_COMPILED_VERIFY_GROWTH_RESERVE", "16")
    rt = smoke._tiny_runtime(model)
    install_qwen4_fixed_verify_route(rt)
    cache = model.make_cache()
    rt.forward_ar(mx.array([PROMPT]), cache=cache, return_hidden=True)
    plan = graphbank.FixedM4CapacityPlan.for_request(1000, runtime=rt)
    bank = graphbank.CompiledVerifyBank(rt, max_verify_len=4, request_max_tokens=1000,
                                        capacity_plan=plan)
    bank.install_fixed_m4(cache, prompt_ids=PROMPT, hidden_variant=None)
    banks = [entry for entry in cache if isinstance(entry, graphbank.TensorOffsetQSACache)]
    qsa = banks[0]
    refs = [bank_.state_leaves for bank_ in banks]
    requests = []

    def admit(need):
        # Bytes asked, and how many layers had changed a leaf when asked.
        changed = sum(
            not all(a is b for a, b in zip(ref, bank_.state_leaves))
            for ref, bank_ in zip(refs, banks)
        )
        requests.append((need, changed))
        if len(requests) == 1:
            return False  # the bucketed growth's bill
        if len(requests) == 2:
            return allow_step  # the unbucketed growth's bill
        return True  # each layer's new banks

    def bank_bytes(rows):
        """One layer's banks at ``rows`` rows (pooled: one block per four)."""

        per_row = sum(
            int(leaf.nbytes) // int(leaf.shape[axis])
            for leaf, axis in ((qsa.kv.keys, 2), (qsa.kv.values, 2), (qsa.raw_keys, 1))
        )
        per_block = int(qsa.pooled.nbytes) // int(qsa.pooled.shape[1])
        return rows * per_row + rows // 4 * per_block

    def bill(rows):
        """The layers before the last keep their gain; the last is written beside its old banks."""

        return (len(banks) - 1) * (bank_bytes(rows) - bank_bytes(512)) + bank_bytes(rows)

    plan.admit_growth = admit
    assert qsa.capacity == 512 and qsa.dense_capacity == 256
    if allow_step:
        bank.reserve_fixed_m4_window(cache, committed_count=469)
        assert qsa.capacity == qsa.dense_capacity == 768
        assert plan.bucket == qsa.capacity_bucket == 0
        # Then each layer's new banks, asked before that layer changed.
        assert requests[2:] == [(bank_bytes(768), layer) for layer in range(len(banks))]
    else:
        with pytest.raises(MemoryError, match="growth exceeds memory admission"):
            bank.reserve_fixed_m4_window(cache, committed_count=469)
        assert qsa.capacity == 512 and qsa.dense_capacity == 256
        assert all(a is b for a, b in zip(refs[0], qsa.state_leaves))
        assert len(requests) == 2
    # The bills at the bucketed width, then the unbucketed one, before any
    # leaf changed.
    assert requests[:2] == [(bill(1024), 0), (bill(768), 0)]


@pytest.mark.parametrize("offset", [16_384, 63_000])
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("poison", [float("nan"), 60_000.0], ids=["nan", "large"])
def test_production_fused_gather_and_one_row_capture_keep_identical_bits(lane, offset, dtype, poison):
    from mtplx.models.qwen4_exp import Attention, TextArgs
    import mlx.utils

    if not mx.metal.is_available():
        pytest.skip("production fused gather requires Metal")
    lane.setenv("MTPLX_QSA_GATHER", "1")
    lane.setenv("MTPLX_QSA_M4_FUSED_KV_GATHER", "1" if dtype == mx.bfloat16 else "0")
    lane.setenv("MTPLX_QWEN4_FIXED_M4_VERIFY", "1")
    previous = mx.default_device()
    mx.set_default_device(mx.gpu)
    try:
        mx.random.seed(28)
        layer = Attention(TextArgs(
            hidden_size=64, num_hidden_layers=4, num_attention_heads=24,
            num_key_value_heads=2, head_dim=256, indexer_n_heads=4,
            indexer_kv_heads=1, indexer_head_dim=128, indexer_budget=2048,
            indexer_compress_ratio=4,
        ))
        layer.update(mlx.utils.tree_map(lambda p: p.astype(dtype), layer.parameters()))
        entry = QSACache(4)
        entry.indexer_budget = 2048
        entry.kv.keys = (mx.random.normal((1, 2, offset, 256)) * 0.25).astype(dtype)
        entry.kv.values = (mx.random.normal((1, 2, offset, 256)) * 0.5).astype(dtype)
        entry.kv.offset = offset
        entry.raw_keys = mx.random.normal((1, offset, 128)).astype(dtype)
        entry.pooled = mx.random.normal((1, offset // 4, 128)).astype(dtype)
        entry.pooled_len = offset // 4
        banks = [graphbank.TensorOffsetQSACache.from_qsa_cache(
            entry, reserve_tokens=1024, capacity_bucket=bucket,
        ) for bucket in (0, 8192, 8192)]
        assert banks[0].capacity != banks[1].capacity
        assert all(bank.fused_rows_gather_kv_m4 == (dtype == mx.bfloat16) for bank in banks)
        poisoned = banks[2]
        # Poison every row added by bucketing, in every owned array. The
        # original step-rounded consumer width remains a separate control.
        start = poisoned.dense_capacity
        poisoned.kv.keys[:, :, start:] = poison
        poisoned.kv.values[:, :, start:] = poison
        poisoned.raw_keys[:, start:] = poison
        poisoned.pooled[:, start // 4:] = poison
        for rows in (4, 1, 4, 1):
            x = mx.random.normal((1, rows, 64)).astype(dtype)
            out = [layer(x, bank) for bank in banks]
            for other in (1, 2):
                assert bool(mx.all(mx.isfinite(out[other])).item())
                _equal(_leaves(out[0]), _leaves(out[other]))
                _equal(_state([banks[0]]), _state([banks[other]]))
            if rows == 4:
                for bank in banks:
                    bank.trim(3)
    finally:
        mx.set_default_device(previous)


def test_early_gather_keeps_the_parent_capacity_for_the_request(cpu_layer, lane):
    """A lowered gather threshold must not widen k_eff and gathered SDPA."""
    lane.setenv("MTPLX_QSA_GATHER", "1")
    lane.setenv("MTPLX_QSA_GATHER_MIN_CONTEXT", "8")
    cpu_layer.indexer.budget = 512
    cpu_layer.indexer.block_topk = 256  # ratio 2; larger than the first bank
    banks = [graphbank.TensorOffsetQSACache.from_qsa_cache(
        _prefilled_entry(cpu_layer), reserve_tokens=32, capacity_bucket=bucket,
    ) for bucket in (0, 1024)]
    assert [bank.capacity for bank in banks] == [256, 256]
    for grow in (False, True):
        if grow:
            for bank in banks:
                bank.ensure_capacity(257)
            assert [bank.capacity for bank in banks] == [512, 512]
        x = mx.random.normal((1, 4, 64)).astype(mx.bfloat16)
        out = [cpu_layer(x, bank) for bank in banks]
        _equal(_leaves(out[0]), _leaves(out[1]))
        _equal(_state([banks[0]]), _state([banks[1]]))


def test_admission_respects_a_larger_selection_budget(lane):
    lane.setenv("MTPLX_QSA_GATHER", "1")
    lane.setenv("MTPLX_QWEN4_FIXED_M4_CAPACITY_BUCKET", "8192")
    rt = SimpleNamespace(model=SimpleNamespace(args=SimpleNamespace(
        indexer_budget=32_768, indexer_compress_ratio=4,
    )))
    plan = graphbank.FixedM4CapacityPlan.for_request(32, runtime=rt)
    assert plan.rows(16_384, 4) == 16_640  # both allocation and admission keep k_eff
    assert plan.bucket == 0
    plan = graphbank.FixedM4CapacityPlan.for_request(32, runtime=rt)  # the next request
    assert plan.rows(32_768, 4) == 40_960
