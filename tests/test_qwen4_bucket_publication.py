"""Published bucketed QSA state owns only the bytes the session bank charges."""

import gc
import time
import weakref

import mlx.core as mx
import numpy as np
import pytest

import mtplx.generation as generation
from mtplx.graphbank import CompiledVerifyBank, TensorOffsetQSACache
from mtplx.models.qwen4_exp import QSACache
from mtplx.session_bank import SessionBank
from test_qwen4_fixed_m4_capacity_bucket import NATIVE, PROMPT, SEED, _bits, lane, pack


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
def test_bucket_demotion_releases_slack_before_lazy_session_publication(pack, lane, dtype):
    lane.setenv("MTPLX_QSA_GATHER", "1")
    lane.setenv("MTPLX_SESSION_LAZY_SNAPSHOT", "1")
    smoke, model = pack
    runtime = smoke._tiny_runtime(model)
    # One production-shaped attention layer. A 16K prompt grants 17,408
    # rows inside a 24,576-row bucket; the bank must not retain its slack.
    offset = 16_384
    logical_bytes = offset * (2 * 2 * 256 + 128 + 128 // 4) * 2
    session = SessionBank(max_bytes=logical_bytes * 3 // 2,
                          per_session_max_bytes=logical_bytes * 3 // 2)
    mx.synchronize()
    gc.collect()
    before = mx.get_active_memory()

    def publish(token):
        entry = QSACache(4)
        entry.indexer_budget = 2048
        entry.kv.keys = mx.full((1, 2, offset, 256), 0.25, dtype=dtype)
        entry.kv.values = mx.full((1, 2, offset, 256), -0.5, dtype=dtype)
        entry.kv.offset = offset
        entry.raw_keys = mx.full((1, offset, 128), 0.75, dtype=dtype)
        entry.pooled = mx.full((1, offset // 4, 128), -0.25, dtype=dtype)
        entry.pooled_len = offset // 4
        promoted = TensorOffsetQSACache.from_qsa_cache(
            entry, reserve_tokens=1024, capacity_bucket=8192,
        )
        assert promoted.capacity == 24_576 and promoted.dense_capacity == 17_408
        mx.eval(*promoted.state_leaves)
        started = time.perf_counter()
        demoted = promoted.demote()
        mx.eval(*demoted.state)
        elapsed = time.perf_counter() - started
        for original, published in zip(entry.state, demoted.state):
            assert np.array_equal(_bits(original), _bits(published))
        saved = session.put(
            runtime=runtime, token_ids=[token] * offset, cache=[demoted],
            logits=mx.zeros((1, 128)), hidden=mx.zeros((1, 1, 64)),
        )
        assert saved is not None
        mx.eval(*saved.cache_snapshot.states[0])
        return elapsed

    for token in (31, 37):
        elapsed = publish(token)
        gc.collect()
        mx.synchronize()
        retained = mx.get_active_memory() - before
        print(f"publication dtype={dtype} rows={offset} time_ms={elapsed * 1000:.3f} "
              f"retained={retained} charged={session.total_nbytes}")
        # A second independent session must evict the first on the real
        # budget, and leave at most allocator page rounding unaccounted.
        assert len(session) == 1
        assert retained <= session.total_nbytes + 1024 * 1024, (
            "published views retained uncharged bucket storage", retained, session.total_nbytes,
        )


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("offset", [16_384, 65_536])
def test_installed_bank_releases_each_layer_during_publication(pack, lane, dtype, offset):
    """Measure the peak while an actual installed bank is still alive.

    Three production-shaped QSA layers distinguish one-layer copy scratch
    from retaining the entire old bank alongside its compact replacement.
    The installed plan and shadow are real; no forward uses these test shapes.
    """
    from mtplx.qwen4_fixed_verify import install_qwen4_fixed_verify_route

    lane.setenv("MTPLX_QSA_GATHER", "1")
    lane.setenv("MTPLX_QWEN4_FIXED_M4_CAPACITY_BUCKET", "8192")
    smoke, model = pack
    runtime = smoke._tiny_runtime(model)
    install_qwen4_fixed_verify_route(runtime)
    cache = model.make_cache()
    runtime.forward_ar(mx.array([PROMPT]), cache=cache, return_hidden=True)

    def entry():
        qsa = QSACache(4)
        qsa.indexer_budget = 2048
        qsa.state = (
            mx.full((1, 2, offset, 256), 0.25, dtype=dtype),
            mx.full((1, 2, offset, 256), -0.5, dtype=dtype),
            mx.full((1, offset, 128), 0.75, dtype=dtype),
            mx.full((1, offset // 4, 128), -0.25, dtype=dtype),
        )
        return qsa

    cache = [e for e in cache if not isinstance(e, QSACache)] + [entry() for _ in range(3)]
    bank = CompiledVerifyBank(runtime, max_verify_len=4, request_max_tokens=1024)
    bank.install_fixed_m4(cache, prompt_ids=[31] * offset, hidden_variant=None)
    assert len(bank._fixed_m4_dispatch["qsa_entries"]) == 3
    mx.eval(*(leaf for e in cache for leaf in
              (e.state_leaves if isinstance(e, TensorOffsetQSACache) else e.state)))
    refs = [weakref.ref(e) for e in cache if isinstance(e, TensorOffsetQSACache)]
    gc.collect()
    mx.synchronize()
    live_before = mx.get_active_memory()
    bucket_bytes = sum(e.nbytes for e in cache if isinstance(e, TensorOffsetQSACache))
    logical_bytes = offset * (2 * 2 * 256 + 128 + 128 // 4) * 2
    mx.reset_peak_memory()
    assert bank.demote(cache) == 3
    mx.synchronize()
    peak = mx.get_peak_memory()
    retained = mx.get_active_memory()
    print(f"installed publication dtype={dtype} rows={offset} before={live_before} "
          f"peak={peak} retained={retained} logical_per_layer={logical_bytes}")
    assert peak <= live_before + logical_bytes + 2 * 1024**2, "compaction retained prior layers"
    assert retained <= live_before - bucket_bytes + 3 * logical_bytes + 2 * 1024**2
    assert all(ref() is None for ref in refs), "installed dispatch still owns demoted adapters"
    assert bank._fixed_m4_dispatch is None
    for qsa in (e for e in cache if isinstance(e, QSACache)):
        for leaf, value in zip(qsa.state, (0.25, -0.5, 0.75, -0.25)):
            assert bool(mx.all(leaf == value).item())


@pytest.mark.parametrize("one_copy", [False, True])
@pytest.mark.parametrize("capture", [False, True])
def test_generation_compacts_only_published_state_and_reports_demotion(pack, lane, capture, one_copy):
    """The copying store compacts published state; the one-copy store never copies.

    With the one-copy store (mtplx/one_copy.py) the session bank keeps the
    conversation as a lease on the bank's own buffers, so demotion hands them
    back whole whether or not the state is published.
    """

    from mtplx.qwen4_fixed_verify import install_qwen4_fixed_verify_route

    smoke, model = pack
    lane.setenv("MTPLX_ONE_COPY", "1" if one_copy else "0")
    runtime = smoke._tiny_runtime(model)
    install_qwen4_fixed_verify_route(runtime)
    lane.setenv("MTPLX_COMPILED_VERIFY", "1")
    lane.setenv("MTPLX_QSA_GATHER", "1")
    lane.setenv("MTPLX_QSA_GATHER_MIN_CONTEXT", "8")
    lane.setenv("MTPLX_QWEN4_FIXED_M4_CAPACITY_BUCKET", "1024")
    copies = []
    demote = TensorOffsetQSACache.demote
    asarray = mx.asarray
    clock = time.perf_counter
    clock_offset = 0.0

    def copy(*args, **kwargs):
        if kwargs.get("copy"):
            copies.append(True)
        return asarray(*args, **kwargs)

    def observed_demote(self, *args, **kwargs):
        nonlocal clock_offset
        assert self.capacity > self.dense_capacity
        with lane.context() as patch:
            patch.setattr(mx, "asarray", copy)
            result = demote(self, *args, **kwargs)
        clock_offset += 3.25  # deterministic cost, without sleeping
        return result

    lane.setattr(TensorOffsetQSACache, "demote", observed_demote)
    lane.setattr(time, "perf_counter", lambda: clock() + clock_offset)
    result = generation.generate_mtpk(
        runtime, PROMPT, max_tokens=16, sampler=NATIVE, draft_sampler=NATIVE,
        speculative_depth=3, seed=SEED, mtp_cache_policy="persistent",
        mtp_history_policy="committed", verify_strategy="batched",
        stop_token_ids=set(), capture_final_state=capture,
    )
    report = result.stats.graphbank["compiled_verify"]
    assert report["fixed_m4"]["installed"]
    if one_copy:
        assert copies == [], "the one-copy store never copies at demotion"
    else:
        assert len(copies) == (4 if capture else 0), "unpublished state must not be copied"
    assert report.get("demote_time_s", 0) >= 3.25
    if capture:
        assert result.final_state.safe_to_commit
        if one_copy:
            qsa = next(e for e in result.final_state.final_trunk_cache if isinstance(e, QSACache))
            assert qsa.kv.keys.shape[2] > qsa.offset, "the bank's capacity comes back whole"
    else:
        assert result.final_state is None
