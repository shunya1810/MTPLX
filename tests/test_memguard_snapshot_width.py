"""Quantized KV is snapshotted at full width, and the MTP head's history counts.

The review of 9c96dd9c (finding 3): the decode-start bill priced the copy of a
banked prompt at the paged width, but ``snapshot_cache_lazy_hybrid`` reads each
cache's ``state``, and a quantized paged cache's ``state`` dequantizes
(``VllmMetalPagedKVCache._dequant_active_arrays``): q4 returns fresh
full-width arrays, q8 returns views of its bf16 mirror, which decode's first
write then copies. A 250K-token 27B snapshot is 15.3 GiB, not the 4.9 GiB of
its q4 pages. Decode also keeps a working copy beside the pages (the q8
mirror at full width, the q4 head-major bank), the per-session cap was
checked at the paged width, and the 27B's committed MTP-history cache
(4,096 B a token) was in no term at all.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import mtplx.memory_plan as memory_plan
import mtplx.server.openai as srv
import mtplx.system_memory as sm
from mtplx.memory_plan import plan_memory
from tests.test_memguard_admission import (
    FN_KV,
    GIB,
    Q27_KV,
    Q27_TEXT_CONFIG,
    Q27_WEIGHTS,
    _install,
    _Machine,
    _manager,
    _q27_runtime,
    _state,
)

Q4 = int(Q27_KV * 0.30)
Q8 = int(Q27_KV * 0.55)
MTP_HISTORY = 1 * 2 * 4 * 256 * 2


@pytest.fixture(autouse=True)
def _served_profile(monkeypatch):
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", "1")
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL_LAYOUT", "auto")
    monkeypatch.setenv("MTPLX_SUSTAINED_DENSE_DECODE_MAX_CONTEXT", "131072")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE", "auto")
    monkeypatch.delenv("MTPLX_HOST_MEMORY_ALLOWANCE_BYTES", raising=False)
    monkeypatch.setattr(srv, "_record_guard_event", lambda state, payload: None)


def _geometry(quant: str):
    """What the admission reads off a real plan with this KV quantization,
    and the 27B's attention shape (24 query heads, 4 KV heads, 256 dims:
    the q8 kernel takes 5 query rows, the q4 kernel 8)."""

    plan = plan_memory(
        total_ram_bytes=128 * GIB,
        model_weights_bytes=Q27_WEIGHTS,
        kv_bytes_per_token=Q27_KV,
        kv_quantization=quant,
        model_max_context=262_144,
    )
    assert plan.kv_bytes_per_token_effective == {"off": Q27_KV, "q8": Q8, "q4": Q4}[quant]
    return srv._admission_geometry(
        SimpleNamespace(memory_plan=plan, runtime=_q27_runtime())
    )


def _cold(geometry, prompt: int, *, publish: bool = True):
    return srv._admission_growth(
        geometry,
        prompt_tokens=prompt,
        reused_tokens=0,
        restore_copies_prefix=True,
        layout="contiguous_then_repage",
        source_layout=None,
        output_tokens=16_386,
        publish=publish,
        scratch_bytes=3 * GIB,
    )


class TestQuantizedSnapshot:
    def test_a_q4_snapshot_is_the_prompt_at_full_width(self):
        growth = _cold(_geometry("q4"), 250_000)
        assert growth["publish_copy_bytes"] == 250_000 * Q27_KV
        assert round(growth["publish_copy_bytes"] / GIB, 1) == 15.3
        # Not the 4.9 GiB of its q4 pages.
        assert growth["publish_copy_bytes"] > 3 * (250_000 * Q4)

    def test_q4_decode_builds_its_head_major_bank_beside_the_pages(self):
        """The kernel route builds the bank at decode's first call (the
        prompt's 250,000 rows) and regrows it at the next to the pages'
        capacity (266,386 rows), the first buffer alive while the second is
        filled (``_quant_bank_arrays``)."""

        growth = _cold(_geometry("q4"), 250_000, publish=False)
        assert growth["quant_working_bytes"] == (250_000 + 266_386) * Q4
        assert growth["quant_working"]["route"] == "kernel"
        assert growth["quant_working"]["rows"] == 250_000 + 266_386
        assert growth["decode_start_bytes"] == (
            (250_000 + 16_386) * Q4 + (250_000 + 266_386) * Q4
        )

    def test_q8_decode_on_the_kernel_route_builds_no_mirror(self):
        """A 100K prompt latches the q8 kernel, whose calls read the pages:
        no bf16 mirror, where the old bill charged 6.1 GiB of one."""

        growth = _cold(_geometry("q8"), 100_000, publish=False)
        assert growth["quant_working_bytes"] == 0
        assert growth["quant_working"]["route"] == "kernel"
        growth = _cold(_geometry("q8"), 100_000)
        assert growth["publish_copy_bytes"] == 100_000 * Q27_KV

    def test_q8_decode_builds_the_mirror_for_calls_the_kernel_declines(self):
        # A six-row verify (MTP depth 5) is wider than the kernel's five.
        growth = srv._admission_growth(
            _geometry("q8"),
            prompt_tokens=100_000,
            reused_tokens=0,
            restore_copies_prefix=True,
            layout="contiguous_then_repage",
            source_layout=None,
            output_tokens=16_386,
            publish=False,
            scratch_bytes=3 * GIB,
            verify_tokens=6,
        )
        assert growth["quant_working"]["builds"] == "q8_bf16_mirror"
        assert growth["quant_working"]["rows"] == 100_006 + 116_386
        assert growth["quant_working_bytes"] == (100_006 + 116_386) * Q27_KV
        # Under the kernel's threshold the dequant route builds it too.
        short = _cold(_geometry("q8"), 800, publish=False)
        assert short["quant_working"]["route"] == "dequant"
        assert short["quant_working_bytes"] > 800 * Q27_KV

    def test_plain_pages_are_copied_at_their_own_width(self):
        growth = _cold(_geometry("off"), 150_000)
        assert growth["quant_working_bytes"] == 0
        assert growth["publish_copy_bytes"] == (150_000 + 16_386) * Q27_KV

    def test_the_per_session_cap_is_checked_at_the_snapshots_width(self, monkeypatch):
        """200K tokens under q4: 3.9 GB of pages, 13.1 GB of snapshot. A
        10 GiB cap refuses that snapshot at the put, so nothing is banked
        and nothing is copied; the paged width said it fitted."""

        monkeypatch.setenv("MTPLX_PAGED_KV_QUANT", "q4")
        plan = plan_memory(
            total_ram_bytes=128 * GIB,
            model_weights_bytes=Q27_WEIGHTS,
            kv_bytes_per_token=Q27_KV,
            kv_quantization="q4",
            model_max_context=262_144,
        )
        manager = _manager(max_bytes=60 * GIB, per_session_max_bytes=10 * GIB)
        state = _state(manager, plan=plan, runtime=_q27_runtime(), limit_gib=96, total_gib=128)
        monkeypatch.setattr(
            sm,
            "_reader",
            lambda: sm.SystemMemory(
                available_bytes=80 * GIB,
                total_bytes=128 * GIB,
                level_percent=60,
                free_bytes=70 * GIB,
                file_backed_bytes=10 * GIB,
                wired_bytes=30 * GIB,
                compressor_bytes=GIB,
                swap_used_bytes=0,
            ),
        )
        _install(monkeypatch, _Machine(manager.bank, base_gib=22.0, cache_gib=0.0, host_gib=1.0))
        pricing: dict = {}
        srv._prefill_admission_shed(
            state,
            prompt_ids=list(range(200_000)),
            session_bank=manager.bank,
            session_id="deep",
            max_new_tokens=16_384,
            prefill_chunk_tokens=None,
            pricing=pricing,
        )
        assert 200_000 * Q4 < 10 * GIB < 200_000 * Q27_KV
        assert pricing["growth"]["publish_copy_bytes"] == 0


def _lease(geometry, *, prompt: int, working_rows: int, capacity: int = 196_608):
    """A warm turn that extends a leased paged cache in place by one token."""

    return srv._admission_growth(
        geometry,
        prompt_tokens=prompt,
        reused_tokens=prompt - 1,
        restore_copies_prefix=False,
        layout="contiguous_then_repage",
        source_layout="contiguous_then_repage",
        output_tokens=16_386,
        publish=False,
        scratch_bytes=3 * GIB,
        lease={
            "paged": True,
            "capacity_tokens": capacity,
            "paged_layers": 16,
            "block_size": 16,
            "working_rows": working_rows,
        },
    )


class TestTheWorkingCopyOfALease:
    """The review of 23a94abf (finding 3): the whole prefix's working copy
    was charged for every quantized request, whether the copy was already
    resident or would never be built."""

    def test_the_geometry_carries_the_attention_shape(self):
        assert _geometry("q4").attention_shape == (24, 4, 256, 256)

    def test_a_q4_lease_whose_bank_holds_the_reach_adds_nothing(self):
        """150K tokens with spare capacity, the bank already allocated past
        the request's reach (150,000 + 16,386): it was charged 2.75 GiB."""

        growth = _lease(_geometry("q4"), prompt=150_000, working_rows=175_000)
        assert growth["quant_working_bytes"] == 0
        assert round(150_000 * Q4 / GIB, 2) == 2.75

    def test_a_q4_lease_whose_bank_must_grow_adds_the_new_buffer(self):
        growth = _lease(_geometry("q4"), prompt=150_000, working_rows=150_000)
        # min(capacity, 1.5 x 150,000) rows, the old buffer already resident.
        assert growth["quant_working_bytes"] == 196_608 * Q4
        assert growth["quant_working"]["rows"] == 196_608

    def test_a_one_token_q8_extension_on_the_kernel_route_needs_no_mirror(self):
        """It was charged 9.16 GiB of bf16 mirror."""

        growth = _lease(_geometry("q8"), prompt=150_000, working_rows=0)
        assert growth["quant_working_bytes"] == 0
        assert growth["quant_working"]["route"] == "kernel"
        assert round(150_000 * Q27_KV / GIB, 2) == 9.16


class TestTheCachesOwnRule:
    def test_the_growth_rule_matches_the_caches(self):
        from mtplx.cache_state import kv_quant_working_copy_peak_rows

        # A new copy: the prompt's rows, then the capacity beside them.
        assert kv_quant_working_copy_peak_rows(
            existing_rows=0, first_offset=1_000, last_offset=1_100,
            capacity_rows=1_200, block_size=16,
        ) == 1_000 + 1_200
        # Already past the reach: nothing.
        assert kv_quant_working_copy_peak_rows(
            existing_rows=2_000, first_offset=1_000, last_offset=1_900,
            capacity_rows=4_000, block_size=16,
        ) == 0
        # Several regrows (100 -> 150 -> 225 -> 337 -> 505 rows): the last
        # old-plus-new pair, less the 100 rows already resident.
        assert kv_quant_working_copy_peak_rows(
            existing_rows=100, first_offset=101, last_offset=400,
            capacity_rows=10_000, block_size=16,
        ) == 337 + 505 - 100

    def test_the_route_rule_is_the_caches(self):
        from mtplx.cache_state import kv_quant_decode_route

        shape = (24, 4, 256, 256)
        assert kv_quant_decode_route(8, route_offset=1_024, attention_shape=shape) == "kernel"
        assert kv_quant_decode_route(8, route_offset=1_023, attention_shape=shape) == "dequant"
        assert kv_quant_decode_route(4, route_offset=4_096, attention_shape=shape) == "kernel"
        # The q4 kernel takes head dims of 64, 128 or 256 only.
        assert (
            kv_quant_decode_route(4, route_offset=4_096, attention_shape=(24, 4, 192, 192))
            == "dequant"
        )


class TestMtpHistory:
    def test_the_27b_head_keeps_4096_bytes_a_token(self):
        config = {"text_config": dict(Q27_TEXT_CONFIG, mtp_num_hidden_layers=1)}
        assert (
            memory_plan.mtp_history_bytes_per_token_from_config(config)
            == MTP_HISTORY
            == 4_096
        )

    def test_families_whose_aux_already_counts_it_or_that_have_none(self):
        mtp_history_bytes_per_token_from_config = (
            memory_plan.mtp_history_bytes_per_token_from_config
        )
        flash_next = {
            "text_config": {
                "indexer_n_heads": 4,
                "layer_types": ["full_attention"] * 12,
                "num_key_value_heads": 2,
                "head_dim": 256,
                "mtp_num_hidden_layers": 1,
            }
        }
        assert mtp_history_bytes_per_token_from_config(flash_next) == 0
        assert mtp_history_bytes_per_token_from_config({"text_config": Q27_TEXT_CONFIG}) == 0
        assert mtp_history_bytes_per_token_from_config(None) == 0

    def test_the_plan_carries_it_and_its_fit_is_unchanged(self):
        common = dict(
            total_ram_bytes=48 * GIB,
            model_weights_bytes=Q27_WEIGHTS,
            kv_bytes_per_token=Q27_KV,
            model_max_context=262_144,
        )
        without = plan_memory(**common)
        with_history = plan_memory(**common, mtp_history_bytes_per_token=MTP_HISTORY)
        assert with_history.mtp_history_bytes_per_token == MTP_HISTORY
        assert with_history.to_dict()["mtp_history_bytes_per_token"] == MTP_HISTORY
        assert with_history.context_window_fit == without.context_window_fit

    def test_the_admission_prices_every_row_with_it(self):
        plan = SimpleNamespace(
            available=True,
            kv_bytes_per_token=Q27_KV,
            kv_bytes_per_token_effective=Q27_KV,
            aux_bytes_per_token=0,
            mtp_history_bytes_per_token=MTP_HISTORY,
            prefill_transient_bytes_per_token=0,
            runtime_transients_bytes=3 * GIB,
            model_weights_bytes=Q27_WEIGHTS,
            kv_quantization="off",
        )
        geometry = srv._admission_geometry(SimpleNamespace(memory_plan=plan, runtime=None))
        assert geometry.live_bytes_per_token == Q27_KV + MTP_HISTORY
        assert geometry.paged_bytes_per_token == Q27_KV + MTP_HISTORY
        assert geometry.aux_bytes_per_token == MTP_HISTORY
        # A QSA family's aux already has its MTP head; nothing is added.
        qsa = SimpleNamespace(**{**vars(plan), "aux_bytes_per_token": 7_872,
                                 "mtp_history_bytes_per_token": 0,
                                 "kv_bytes_per_token": FN_KV,
                                 "kv_bytes_per_token_effective": FN_KV})
        assert srv._admission_geometry(
            SimpleNamespace(memory_plan=qsa, runtime=None)
        ).live_bytes_per_token == FN_KV + 7_872

    def test_the_server_hands_it_to_the_plan(self):
        import inspect

        src = inspect.getsource(srv.ServerState.__init__)
        assert '"mtp_history_bytes_per_token": (' in src
        assert "_plan_mtp_history_from_config(_plan_model_config)" in src
