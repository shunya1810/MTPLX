"""A leased paged cache grows by what its pages cannot hold, not by its rows.

The review of 9c96dd9c (finding 2): the admission inferred a paged lease from
the token count and charged the new rows plus the output reservation. The
cache itself grows only when its allocated capacity cannot hold the prompt
(``_write_tail``) or the request's reservation at the repage
(``install_vllm_metal_paged_attention_kv_cache``), and then by 1.5 times,
clamped to the serving window (``_grow_to_capacity``). On the unquantized
27B, a 196,608-token capacity holding 196,607 tokens grows to 262,144 for a
196,609-token prompt: 4 GiB of pages, charged as three rows. A lease whose
pages already hold the prompt and the reservation was charged the
reservation again.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import mtplx.cache_state as cache_state
import mtplx.server.openai as srv
import mtplx.system_memory as sm
from tests.test_memguard_admission import (
    GIB,
    Q27_KV,
    Q27_WEIGHTS,
    _install,
    _Machine,
    _manager,
    _put,
    _q27_runtime,
    _state,
)


@pytest.fixture(autouse=True)
def _served_profile(monkeypatch):
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", "1")
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL_LAYOUT", "auto")
    monkeypatch.setenv("MTPLX_SUSTAINED_DENSE_DECODE_MAX_CONTEXT", "131072")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE", "auto")
    monkeypatch.setenv("MTPLX_DYNAMIC_PAGED_KV", "1")
    monkeypatch.setenv("MTPLX_CONTEXT_WINDOW_TOKENS", "262144")
    monkeypatch.delenv("MTPLX_VLLM_METAL_PAGED_NUM_BLOCKS", raising=False)
    monkeypatch.delenv("MTPLX_DYNAMIC_PAGED_KV_MIN_BLOCKS", raising=False)
    monkeypatch.delenv("MTPLX_DYNAMIC_PAGED_KV_MARGIN", raising=False)
    monkeypatch.delenv("MTPLX_PAGED_KV_QUANT", raising=False)
    monkeypatch.delenv("MTPLX_VLLM_METAL_PAGED_KV_QUANT", raising=False)
    monkeypatch.delenv("MTPLX_HOST_MEMORY_ALLOWANCE_BYTES", raising=False)
    monkeypatch.setattr(srv, "_record_guard_event", lambda state, payload: None)


def _geometry():
    return srv._AdmissionGeometry(
        live_bytes_per_token=Q27_KV,
        paged_bytes_per_token=Q27_KV,
        context_transient_bytes_per_token=0,
        flat_transient_bytes=3 * GIB,
        weights_bytes=Q27_WEIGHTS,
    )


def _lease(capacity: int, layers: int = 16) -> dict:
    return {
        "paged": True,
        "capacity_tokens": capacity,
        "paged_layers": layers,
        "block_size": 16,
    }


def _grow(lease, *, prompt: int, reused: int, output: int):
    return srv._admission_growth(
        _geometry(),
        prompt_tokens=prompt,
        reused_tokens=reused,
        restore_copies_prefix=False,
        layout=(
            "contiguous_then_repage" if prompt > 131_072 else "contiguous_dense_decode"
        ),
        source_layout="contiguous_then_repage",
        output_tokens=output,
        publish=False,
        scratch_bytes=2 * GIB,
        lease=lease,
    )


class TestTheCachesOwnRule:
    def test_it_grows_half_again_clamped_to_the_window(self):
        grown = cache_state.paged_grown_blocks
        assert grown(12_288, 196_609, 16) == 16_384
        # A larger requirement still wins over the window.
        assert grown(12_288, 300_000, 16) == 18_750

    def test_the_prompt_then_the_reservation(self):
        paged_lease_capacity_after = cache_state.paged_lease_capacity_after
        paged_grown_blocks = cache_state.paged_grown_blocks
        # The prefill's writes need 196,609 slots; the repage asks for the
        # prompt, the reservation and a 128-token margin.
        assert (
            paged_lease_capacity_after(
                196_608, 16, prompt_tokens=196_609, reserved_tokens=196_610, repages=True
            )
            == 262_144
        )
        # Pages with room stay as they are.
        assert (
            paged_lease_capacity_after(
                262_144, 16, prompt_tokens=150_000, reserved_tokens=166_384, repages=True
            )
            == 262_144
        )
        # A prompt that fits, a reservation that does not: the repage grows it.
        assert (
            paged_lease_capacity_after(
                147_456, 16, prompt_tokens=140_000, reserved_tokens=156_384, repages=True
            )
            == 16 * paged_grown_blocks(9_216, 156_384 + 128, 16)
        )


class TestLeaseGrowth:
    def test_a_full_lease_grows_by_its_geometric_step(self):
        growth = _grow(_lease(196_608), prompt=196_609, reused=196_607, output=1)
        assert growth["lease_capacity_after_tokens"] == 262_144
        assert growth["live_prefill_bytes"] == 65_536 * Q27_KV == 4 * GIB
        # One layer's pages sit beside their grown copy at a time.
        assert growth["lease_grow_transient_bytes"] == 262_144 * Q27_KV // 16
        assert growth["prefill_end_bytes"] == 4 * GIB + GIB + 2 * GIB

    def test_a_lease_with_room_adds_no_pages(self):
        growth = _grow(_lease(262_144), prompt=150_000, reused=148_000, output=16_384)
        assert growth["lease_capacity_after_tokens"] == 262_144
        assert growth["live_prefill_bytes"] == 0
        assert growth["lease_grow_transient_bytes"] == 0
        # The forward's scratch is all it needs; the old model added the
        # rows and the reservation, 1.2 GB.
        assert growth["prefill_end_bytes"] == 2 * GIB

    def test_an_unreadable_lease_keeps_the_row_estimate(self):
        growth = _grow(None, prompt=150_000, reused=148_000, output=16_384)
        assert growth["lease_capacity_after_tokens"] is None
        assert growth["live_prefill_bytes"] == (2_000 + 16_384) * Q27_KV

    def test_a_contiguous_lease_is_not_priced_as_pages(self):
        """The token count said the source was repaged; its cache says it is
        contiguous. Extended in place at full width: its rows, no page
        reservation."""

        growth = _grow({"paged": False}, prompt=100_000, reused=98_000, output=16_384)
        assert growth["live_prefill_bytes"] == 2_000 * Q27_KV
        assert growth["output_reserve_bytes"] == 0


class _PagedLayer:
    def __init__(self, blocks: int) -> None:
        self.allocated_blocks = blocks
        self.block_size = 16

    @property
    def capacity(self) -> int:
        return self.allocated_blocks * self.block_size


class TestTheAdmissionReadsTheLease:
    def test_the_shape_comes_off_the_live_cache(self):
        entry = SimpleNamespace(
            cache_ref=[SimpleNamespace(keys=None)] * 48 + [_PagedLayer(12_288)] * 16
        )
        # These layers hold no quantized working copy (no q8 mirror, no q4
        # bank), so a lease of them extends none.
        assert srv._lease_cache_shape(entry) == {**_lease(196_608), "working_rows": 0}
        assert srv._lease_cache_shape(SimpleNamespace(cache_ref=None)) is None
        assert srv._lease_cache_shape(
            SimpleNamespace(cache_ref=[SimpleNamespace(keys=None)])
        ) == {"paged": False}

    def test_a_warm_turn_on_a_lease_with_room_is_charged_no_pages(self, monkeypatch):
        """A 148,000-token conversation leased (not copied) for a 2,000-token
        turn: its 262,144 slots hold the prompt and the reservation."""

        manager = _manager(max_bytes=200 * GIB, per_session_max_bytes=200 * GIB)
        prompt = list(range(150_000))
        entry = _put(manager.bank, prompt[:148_000], session_id="long", row_bytes=Q27_KV)
        entry.cache_ref = [_PagedLayer(16_384)] * 16
        entry.lazy_kv = False
        plan = SimpleNamespace(
            available=True,
            kv_bytes_per_token=Q27_KV,
            kv_bytes_per_token_effective=Q27_KV,
            aux_bytes_per_token=0,
            prefill_transient_bytes_per_token=0,
            runtime_transients_bytes=3 * GIB,
            model_weights_bytes=Q27_WEIGHTS,
        )
        state = _state(manager, plan=plan, runtime=_q27_runtime(), limit_gib=96, total_gib=128)
        monkeypatch.setattr(
            sm,
            "_reader",
            # Too tight for the cheap worst case (the whole prompt cold), so
            # the bank is asked what the restore reads.
            lambda: sm.SystemMemory(
                available_bytes=25 * GIB,
                total_bytes=128 * GIB,
                level_percent=50,
                free_bytes=15 * GIB,
                file_backed_bytes=10 * GIB,
                wired_bytes=40 * GIB,
                compressor_bytes=GIB,
                swap_used_bytes=0,
            ),
        )
        _install(monkeypatch, _Machine(manager.bank, base_gib=30.0, cache_gib=0.0, host_gib=1.0))
        pricing: dict = {}
        srv._prefill_admission_shed(
            state,
            prompt_ids=prompt,
            session_bank=manager.bank,
            session_id="long",
            max_new_tokens=16_384,
            prefill_chunk_tokens=None,
            restore_mode="reference",
            pricing=pricing,
        )
        growth = pricing["growth"]
        assert growth["reused_tokens"] == 148_000
        # 54e01d1f charged the 2,000 rows and the 16,384-token reservation.
        assert growth["live_prefill_bytes"] == 0
        assert growth["lease_capacity_tokens"] == 262_144
        assert growth["lease_capacity_after_tokens"] == 262_144
