"""SessionBank snapshots of quantized paged KV keep the stored precision."""

import mlx.core as mx
import pytest
from mlx_lm.models.cache import KVCache

from mtplx.cache_state import (
    VllmMetalPagedKVCache,
    is_compact_kv_state,
    restore_cache,
    snapshot_cache,
    snapshot_cache_lazy_hybrid,
)
from mtplx.kv_quant import PagedKVQuantConfig
from mtplx.session_bank import _snapshot_nbytes


def _quant_cache(bits: int, tokens: int = 45) -> VllmMetalPagedKVCache:
    cache = VllmMetalPagedKVCache(
        block_size=16,
        num_blocks=8,
        kv_quant_config=PagedKVQuantConfig(mode=f"q{bits}"),
    )
    keys = mx.random.normal((1, 4, tokens, 64), key=mx.random.key(1)).astype(mx.float16)
    values = mx.random.normal((1, 4, tokens, 64), key=mx.random.key(2)).astype(mx.float16)
    cache.update_and_fetch(keys, values)
    return cache


@pytest.mark.parametrize("bits", [8, 4])
@pytest.mark.parametrize("take", [snapshot_cache, snapshot_cache_lazy_hybrid])
def test_compact_snapshot_restores_the_dequantized_state(bits, take):
    cache = _quant_cache(bits)
    expected_k, expected_v = cache.state
    snapshot = take([cache])
    state = snapshot.states[0]
    assert is_compact_kv_state(state)
    assert state["keys"].shape[2] == 45
    dense_nbytes = expected_k.nbytes + expected_v.nbytes
    assert _snapshot_nbytes(snapshot) < dense_nbytes

    target = KVCache()
    restore_cache([target], snapshot, restore_meta_state=False)
    assert target.offset == 45
    assert mx.array_equal(target.keys[..., :45, :], expected_k).item()
    assert mx.array_equal(target.values[..., :45, :], expected_v).item()

    paged = VllmMetalPagedKVCache(
        block_size=16,
        num_blocks=8,
        kv_quant_config=PagedKVQuantConfig(mode=f"q{bits}"),
    )
    restore_cache([paged], snapshot)
    restored_k, restored_v = paged.state
    assert paged.offset == 45
    assert mx.allclose(restored_k, expected_k, atol=1e-2).item()
    assert mx.allclose(restored_v, expected_v, atol=1e-2).item()


def test_snapshot_is_isolated_from_later_writes():
    cache = _quant_cache(8, tokens=20)
    snapshot = snapshot_cache_lazy_hybrid([cache])
    before = mx.array(snapshot.states[0]["keys"])
    more = mx.ones((1, 4, 3, 64), dtype=mx.float16)
    cache.update_and_fetch(more, more)
    mx.eval(cache.key_cache)
    assert mx.array_equal(snapshot.states[0]["keys"], before).item()
    assert snapshot.states[0]["keys"].shape[2] == 20


def test_compact_snapshot_can_be_turned_off(monkeypatch):
    monkeypatch.setenv("MTPLX_SESSION_BANK_COMPACT_KV", "0")
    cache = _quant_cache(8)
    state = snapshot_cache([cache]).states[0]
    assert not is_compact_kv_state(state)


def test_unquantized_paged_cache_keeps_the_dense_state():
    cache = VllmMetalPagedKVCache(block_size=16, num_blocks=8)
    keys = mx.random.normal((1, 4, 10, 64)).astype(mx.float16)
    cache.update_and_fetch(keys, keys)
    assert not is_compact_kv_state(snapshot_cache([cache]).states[0])


def test_compact_state_round_trips_the_ssd_codec_and_prefix_decode():
    from mtplx.cache_bank.codec import decode_payload, decode_payload_prefix, encode_payload

    cache = _quant_cache(8, tokens=100)
    expected_k, _ = cache.state
    snapshot = snapshot_cache([cache])
    encoded = encode_payload(
        cache_snapshot=snapshot,
        logits=None,
        hidden=None,
        mtp_history_snapshot=None,
        gdn_boundaries=None,
        has_recurrent=False,
        block_size=16,
    )
    full = decode_payload(encoded.spec, encoded.tensors.__getitem__)
    state = full.cache_snapshot.states[0]
    assert is_compact_kv_state(state)
    target = KVCache()
    restore_cache([target], full.cache_snapshot, restore_meta_state=False)
    assert mx.array_equal(target.keys[..., :100, :], expected_k).item()

    prefix = decode_payload_prefix(
        encoded.spec,
        encoded.tensors.__getitem__,
        cache_prefix_len=64,
        boundary_prefix_len=None,
    )
    state = prefix.cache_snapshot.states[0]
    assert is_compact_kv_state(state)
    assert state["keys"].shape[2] >= 64
    target = KVCache()
    restore_cache([target], prefix.cache_snapshot, restore_meta_state=False)
    assert mx.array_equal(target.keys[..., :64, :], expected_k[..., :64, :]).item()


@pytest.mark.parametrize("pages", ["0", "1"])
def test_tensor_offset_adapter_snapshot_matches_the_paged_one(pages, monkeypatch):
    from mtplx.cache_state import TensorOffsetQuantizedPagedKVCache

    monkeypatch.setenv("MTPLX_KV_QUANT_PAGES_ADAPTER", pages)
    monkeypatch.setenv("MTPLX_GQA_MMA", "1")
    cache = _quant_cache(8, tokens=37)
    expected_k, expected_v = cache.state
    adapter = TensorOffsetQuantizedPagedKVCache.from_paged_cache(cache)
    assert adapter.layout == ("pages" if pages == "1" else "bank")
    state = snapshot_cache([adapter]).states[0]
    assert is_compact_kv_state(state)
    assert state["keys"].shape[2] == 37
    target = KVCache()
    restore_cache([target], snapshot_cache([adapter]), restore_meta_state=False)
    assert mx.array_equal(target.keys[..., :37, :], expected_k).item()
    assert mx.array_equal(target.values[..., :37, :], expected_v).item()


def _pages_equal(a: VllmMetalPagedKVCache, b: VllmMetalPagedKVCache, rows: int) -> bool:
    def flat(buf):
        return buf.reshape(-1, int(buf.shape[2]), int(buf.shape[3]))[:rows]

    return all(
        mx.array_equal(flat(x), flat(y)).item()
        for x, y in (
            (a.key_cache, b.key_cache),
            (a.value_cache, b.value_cache),
            (a.key_scale_cache, b.key_scale_cache),
            (a.value_scale_cache, b.value_scale_cache),
        )
    )


@pytest.mark.parametrize("bits", [8, 4])
def test_compact_state_loads_into_quantized_pages_without_dense(bits):
    cache = _quant_cache(bits, tokens=45)
    snapshot = snapshot_cache([cache])
    target = [
        VllmMetalPagedKVCache(
            block_size=16, num_blocks=2, kv_quant_config=PagedKVQuantConfig(mode=f"q{bits}")
        )
    ]
    restore_cache(target, snapshot)
    assert target[0].offset == 45
    assert target[0].capacity >= 45
    assert _pages_equal(target[0], cache, 45)
    assert target[0]._dequant_memo is None


def test_contiguous_target_becomes_quantized_pages_for_a_matching_request(monkeypatch):
    monkeypatch.setenv("MTPLX_PAGED_KV_QUANT", "q8")
    monkeypatch.delenv("MTPLX_VLLM_METAL_PAGED_KV_QUANT", raising=False)
    cache = _quant_cache(8, tokens=45)
    snapshot = snapshot_cache([cache])
    target = [KVCache()]
    restore_cache(target, snapshot)
    assert isinstance(target[0], VllmMetalPagedKVCache)
    assert target[0].kv_quant and target[0].offset == 45
    assert _pages_equal(target[0], cache, 45)
    # The restored pages keep serving writes.
    more = mx.ones((1, 4, 3, 64), dtype=mx.float16)
    target[0].update_and_fetch(more, more)
    assert target[0].offset == 48


@pytest.mark.parametrize("mode,direct", [("off", "1"), ("q4", "1"), ("q8", "0")])
def test_contiguous_target_stays_dense_otherwise(monkeypatch, mode, direct):
    monkeypatch.setenv("MTPLX_PAGED_KV_QUANT", mode)
    monkeypatch.delenv("MTPLX_VLLM_METAL_PAGED_KV_QUANT", raising=False)
    monkeypatch.setenv("MTPLX_SESSION_BANK_COMPACT_DIRECT", direct)
    cache = _quant_cache(8, tokens=45)
    expected_k, _ = cache.state
    snapshot = snapshot_cache([cache])
    target = [KVCache()]
    restore_cache(target, snapshot, restore_meta_state=False)
    assert isinstance(target[0], KVCache)
    assert mx.array_equal(target[0].keys[..., :45, :], expected_k).item()
