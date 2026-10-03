"""The admission prices what each backend runs, not what the chunk settings say.

The review of 9c96dd9c (finding 1): Gemma 4 was priced as a chunked prefill,
but ``_gemma4_prefill_prompt`` forwards every uncached token in one call; its
runtime has no ``model.args``, so its scratch fell to the flat per-row bill;
and neither Gemma branch handed the request's abort check on, so the
per-chunk memory check never ran there. Reading the backend's code gives
three more ways the generic model misread it:

* after a prefill forward every sliding layer holds all of the new rows
  (``Gemma4RollbackRotatingKVCache`` trims on the next update), but once
  decode runs a token keeps only the full-attention layers' KV: 81,920 B on
  the 31B against the planner's 983,040 for all 60 layers, which the restore
  and the pre-decode clone were charged at;
* the sliding layers build a rows x (cached window + rows) boolean mask for
  one forward (4.3 GB for a 65,536-token cold prompt);
* the prompt cache is cloned before decode whenever a session bank is
  present, whatever store-on-prefill says.

Plus the audit of the 4B: the generic loop, its own geometry, the profile's
chunk. No model is loaded: the runtimes carry the real configs.

Since the chunked prefill (the 31B's cold peaks, 2026-09-27: 25 / 43 / 92 GiB
at 6,026 / 12,026 / 24,026 tokens, one forward's score blocks growing with the
square of the prompt), Gemma 4 forwards the uncached rows in chunks of the
house prefill width and answers the admission with them: a chunk's rows at the
width the chunked prefill keeps (the full-attention KV and the drafter's row
of the sliding KV, 98,304 B a token on the 31B), the chunk's widest score block
(the full-attention layers attend every cached key: rows x prompt), and the
sliding caches as they end a prefill (the window before the last chunk plus
its rows). ``MTPLX_GEMMA4_PREFILL_CHUNK_TOKENS=whole`` restores the one
forward, priced as before.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import mtplx.backends.gemma4_assistant as gemma4
import mtplx.generation as generation
import mtplx.server.openai as srv
import mtplx.system_memory as sm
from mtplx.memory_plan import dense_kv_bytes_per_token_from_config
from tests.test_memguard_admission import (
    GIB,
    _install,
    _Machine,
    _manager,
    _put,
    _state,
)

# google/gemma-4-31B-it's text_config (the Gemma 4 Optimized Speed target).
GEMMA31B_TEXT = dict(
    model_type="gemma4_text",
    hidden_size=5376,
    num_hidden_layers=60,
    intermediate_size=21504,
    num_attention_heads=32,
    num_key_value_heads=16,
    head_dim=256,
    global_head_dim=512,
    num_global_key_value_heads=4,
    sliding_window=1024,
    num_kv_shared_layers=0,
    use_double_wide_mlp=False,
    attention_k_eq_v=True,
    hidden_size_per_layer_input=0,
    enable_moe_block=False,
    max_position_embeddings=262_144,
    vocab_size=262_144,
    layer_types=(["sliding_attention"] * 5 + ["full_attention"]) * 10,
)
GEMMA_PLANNED_KV = dense_kv_bytes_per_token_from_config({"text_config": GEMMA31B_TEXT})
GEMMA_RESIDENT = 10 * 2 * 4 * 512 * 2
GEMMA_WINDOWS = 50 * 2 * 16 * 256 * 2 * 1024
# The chunked prefill: the drafter's row of the last sliding layer's KV beside
# the full-attention KV, and the sliding caches at the window before the last
# 2,048-row chunk plus that chunk.
GEMMA_DRAFTER_ROW = 2 * 16 * 256 * 2
GEMMA_CHUNKED_ROW = GEMMA_RESIDENT + GEMMA_DRAFTER_ROW
GEMMA_CHUNK_WINDOWS = 50 * 2 * 16 * 256 * 2 * (1023 + 2048)
# One row of all 50 sliding caches: what a window costs a row it keeps.
GEMMA_WINDOW_ROW = 50 * 2 * 16 * 256 * 2
GEMMA_WHOLE = "MTPLX_GEMMA4_PREFILL_CHUNK_TOKENS"
GEMMA_ROW = 2 * (5376 + 3 * 21504)  # the MLP, the widest layer
GEMMA_WEIGHTS = int(17.5 * GIB)

# Qwen3.5-4B's text_config (the 4B Optimized Speed pack's config.json).
Q4B_TEXT = dict(
    model_type="qwen3_5_text",
    hidden_size=2560,
    intermediate_size=9216,
    num_hidden_layers=32,
    num_attention_heads=16,
    num_key_value_heads=4,
    head_dim=256,
    linear_num_key_heads=16,
    linear_key_head_dim=128,
    linear_num_value_heads=32,
    linear_value_head_dim=128,
    linear_conv_kernel_dim=4,
    full_attention_interval=4,
    max_position_embeddings=262_144,
    vocab_size=248_320,
    rms_norm_eps=1e-6,
    tie_word_embeddings=True,
)


@pytest.fixture(autouse=True)
def _served_profile(monkeypatch):
    # The served profiles prefill in 2,048-row chunks with the auto layout.
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", "1")
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL_LAYOUT", "auto")
    monkeypatch.setenv("MTPLX_SUSTAINED_DENSE_DECODE_MAX_CONTEXT", "131072")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE", "auto")
    monkeypatch.delenv(GEMMA_WHOLE, raising=False)
    monkeypatch.delenv("MTPLX_PAGED_KV_QUANT", raising=False)
    monkeypatch.delenv("MTPLX_VLLM_METAL_PAGED_KV_QUANT", raising=False)
    monkeypatch.delenv("MTPLX_HOST_MEMORY_ALLOWANCE_BYTES", raising=False)
    monkeypatch.setattr(srv, "_record_guard_event", lambda state, payload: None)


def _gemma_text_args():
    from mlx_lm.models.gemma4_text import ModelArgs

    return ModelArgs.from_dict(GEMMA31B_TEXT)


def _gemma_runtime():
    """The backend's runtime class with the target's text model reduced to
    its config: every admission question goes through the real methods."""

    runtime = object.__new__(gemma4.Gemma4AssistantRuntime)
    runtime.target = SimpleNamespace(
        text_model=SimpleNamespace(config=_gemma_text_args())
    )
    runtime.model_path = Path("models/gemma4-31b/target")
    runtime.mtp_enabled = True
    runtime.backend_id = gemma4.BACKEND_NAME
    return runtime


def _gemma_state(manager, *, limit_gib: float = 96):
    plan = SimpleNamespace(
        available=True,
        kv_bytes_per_token=GEMMA_PLANNED_KV,
        kv_bytes_per_token_effective=GEMMA_PLANNED_KV,
        aux_bytes_per_token=0,
        prefill_transient_bytes_per_token=0,
        runtime_transients_bytes=3 * GIB,
        model_weights_bytes=GEMMA_WEIGHTS,
    )
    return _state(
        manager, plan=plan, runtime=_gemma_runtime(), limit_gib=limit_gib, total_gib=128
    )


def _roomy(monkeypatch, manager):
    monkeypatch.setattr(
        sm,
        "_reader",
        lambda: sm.SystemMemory(
            available_bytes=90 * GIB,
            total_bytes=128 * GIB,
            level_percent=70,
            free_bytes=80 * GIB,
            file_backed_bytes=10 * GIB,
            wired_bytes=20 * GIB,
            compressor_bytes=GIB,
            swap_used_bytes=0,
        ),
    )
    _install(monkeypatch, _Machine(manager.bank, base_gib=18.0, cache_gib=0.0, host_gib=1.0))


def _gemma_scratch(rows: int, cached: int) -> int:
    """Its layers' activations, the fixed part and its measured attention:
    the chunked families' 3 GiB per 2,048 rows does not describe Gemma's
    forward. The widest score block is a full-attention layer's, which
    attends every cached key: rows x (cached + rows)."""

    per_row = srv._ADMISSION_LIVE_LAYERS * GEMMA_ROW
    fixed = max(srv._ADMISSION_FIXED_FLOOR_BYTES, 3 * GIB - per_row * 2048)
    pairs = rows * (cached + rows) if rows > 1 else 0
    return fixed + per_row * rows + gemma4.GEMMA4_PREFILL_PAIR_BYTES * pairs


class TestGemmaGeometry:
    def test_its_text_config_is_read_through_its_runtime(self):
        """The runtime has no ``model``: its adapter holds the text model.
        The widest layer on the 31B is the MLP (the full-attention layers:
        512-wide heads, 4 KV heads, one array for K and V)."""

        runtime = _gemma_runtime()
        args = srv._runtime_text_args(runtime)
        assert args is not None and args.hidden_size == 5376
        assert srv._forward_row_bytes(args) == GEMMA_ROW == 139_776
        assert not srv._runtime_has_qsa_indexer(runtime)

    def test_the_full_attention_layers_have_their_own_width(self):
        # With a narrow MLP the full-attention layers are the widest:
        # 2 x (5,376 + 2 x 16,384 + 4 x 512 + 16,384 + 5,376).
        args = _gemma_text_args()
        args.intermediate_size = 1024
        assert srv._forward_row_bytes(args) == 2 * (5376 + 2 * 16384 + 4 * 512 + 16384 + 5376)

    def test_what_its_caches_keep(self):
        args = _gemma_text_args()
        assert GEMMA_PLANNED_KV == 983_040
        assert gemma4.gemma4_resident_kv_bytes_per_token(args) == GEMMA_RESIDENT == 81_920
        assert gemma4.gemma4_window_cache_bytes(args) == GEMMA_WINDOWS == 838_860_800
        assert gemma4.gemma4_drafter_window_kv_bytes_per_token(args) == GEMMA_DRAFTER_ROW
        assert gemma4.gemma4_window_cache_bytes(args, 1023 + 2048) == GEMMA_CHUNK_WINDOWS
        pairs = gemma4.gemma4_prefill_attention_pairs
        assert pairs(args, 65_536, 0) == 65_536 * 65_536
        # The full-attention layers' 512-wide heads have no fused prefill
        # attention in MLX: their scores are materialized inside the window
        # too.
        assert pairs(args, 1024, 0) == 1024 * 1024
        # The full-attention layers attend every cached key: a warm suffix
        # (or a prefill chunk) builds rows x (cached + rows), not the
        # sliding layers' rows x (window + rows).
        assert pairs(args, 600, 30_000) == 600 * 30_600
        assert pairs(args, 2048, 30_000) == 2048 * 32_048
        assert gemma4.gemma4_prefill_attention_bytes(args, 600, 30_000) == 80 * 600 * 30_600
        # A sliding-only model keeps to its window.
        args.layer_types = ["sliding_attention"] * 60
        assert pairs(args, 600, 30_000) == 600 * (1023 + 600)

    def test_what_its_runtime_answers(self, monkeypatch):
        """Chunked (the default): the prefill keeps a chunk's rows at the
        full-attention KV plus the drafter's sliding row, and the windows at
        the window before the last chunk plus its rows. Whole: what the
        whole-prompt forward was priced at."""

        runtime = _gemma_runtime()
        assert runtime.prefill_kv_bytes_per_token() == GEMMA_CHUNKED_ROW == 98_304
        assert runtime.resident_kv_bytes_per_token() == GEMMA_CHUNKED_ROW
        assert runtime.window_cache_bytes() == GEMMA_CHUNK_WINDOWS
        monkeypatch.setenv(GEMMA_WHOLE, "whole")
        assert runtime.prefill_kv_bytes_per_token() is None
        assert runtime.resident_kv_bytes_per_token() == GEMMA_RESIDENT
        assert runtime.window_cache_bytes() == GEMMA_WINDOWS


class TestGemmaCalibration:
    """The cold prefill peaks measured on the 31B (4-bit, 128 GB M5 Max,
    2026-09-27; MLX allocator peak above the 16.5 GiB warm baseline)."""

    MEASURED_GIB = {6_026: 8.7, 12_026: 26.5, 24_026: 75.6}

    def _bill(self, tokens: int) -> int:
        return tokens * GEMMA_PLANNED_KV + _gemma_scratch(tokens, 0)

    def test_the_bill_bounds_every_measured_peak(self):
        for tokens, peak in self.MEASURED_GIB.items():
            assert self._bill(tokens) >= peak * GIB, tokens

    def test_it_is_tightest_where_it_matters(self):
        # Within 5 percent at 24K, where a miss would freeze the Mac; the
        # small prompts carry the fixed part and more margin.
        assert self._bill(24_026) <= 1.06 * 75.6 * GIB
        assert self._bill(12_026) <= 1.15 * 26.5 * GIB
        assert self._bill(6_026) <= 1.55 * 8.7 * GIB

    def test_a_32k_cold_prompt_does_not_fit_a_128gb_mac(self):
        assert self._bill(32_768) > 128 * GIB


class TestGemmaAdmission:
    def test_a_cold_prompt_is_priced_at_its_chunk(self, monkeypatch):
        """16,384 cold tokens run as eight 2,048-row forwards: the bill is a
        chunk's rows, the last chunk's score block (2,048 x 16,384), every
        row at the chunked prefill's width and the windows as it leaves
        them: 9.3 GiB, where the one forward it replaced (16,384 x 16,384
        pairs, 983,040 B a row) is priced at 45.5 GiB."""

        manager = _manager()
        _roomy(monkeypatch, manager)
        pricing: dict = {}
        srv._prefill_admission_shed(
            _gemma_state(manager),
            prompt_ids=list(range(16_384)),
            session_bank=manager.bank,
            session_id="gemma",
            prefill_chunk_tokens=None,
            restore_mode="clone",
            pricing=pricing,
        )
        growth = pricing["growth"]
        scratch = _gemma_scratch(2048, 16_384 - 2048)
        assert growth["prefill_chunk_tokens"] == 2048
        assert growth["scratch_rows"] == 2048
        assert growth["scratch_source"] == "geometry_calibrated+attention"
        assert growth["scratch_bytes"] == scratch
        assert growth["layout"] == "contiguous_dense_decode"
        assert growth["repage_copy_bytes"] == 0
        assert growth["chunk_bytes"] == 2048 * GEMMA_CHUNKED_ROW + scratch
        assert growth["live_prefill_bytes"] == 16_384 * GEMMA_CHUNKED_ROW + GEMMA_CHUNK_WINDOWS
        assert growth["publish_copy_bytes"] == 16_384 * GEMMA_CHUNKED_ROW + GEMMA_CHUNK_WINDOWS
        assert growth["growth_bytes"] < 10 * GIB
        geometry = srv._admission_geometry(_gemma_state(manager))
        assert geometry.live_bytes_per_token == GEMMA_CHUNKED_ROW
        assert geometry.prefill_fixed_bytes == GEMMA_CHUNK_WINDOWS

    def test_a_rewrite_deeper_than_the_last_chunk_is_priced_cold(self, monkeypatch):
        """The review of 808a11e2: a banked 24,026-token prompt whose client
        keeps the first 20,000 tokens and replaces the tail with 600 was
        priced as a 20,000-token restore, while the restore could not trim
        the chunked prefill's sliding caches that far and ran all 20,600
        tokens cold. The bank's plan now knows how far an entry's caches
        reach (``restore_floor_tokens``), so on a Mac where the cold bill
        does not fit beside what is resident the admission sheds for it
        instead of admitting the request at the warm bill."""

        manager = _manager()
        _roomy(monkeypatch, manager)
        banked = list(range(24_026))
        entry = _put(manager.bank, banked, session_id="gemma", row_bytes=GEMMA_CHUNKED_ROW)
        # What a chunked prefill leaves: 1,023 + 2,048 rows in each sliding
        # cache, trimmable back until one 1,024-row window remains.
        entry.restore_floor_tokens = 24_026 - (1023 + 2048 - 1024)
        _install(monkeypatch, _Machine(manager.bank, base_gib=81.0, cache_gib=0.0, host_gib=1.0))
        pricing: dict = {}
        receipt = srv._prefill_admission_shed(
            _gemma_state(manager),
            prompt_ids=banked[:20_000] + list(range(10**6, 10**6 + 600)),
            session_bank=manager.bank,
            session_id="gemma",
            prefill_chunk_tokens=None,
            restore_mode="clone",
            pricing=pricing,
        )
        growth = pricing["growth"]
        assert growth["reused_tokens"] == 0 and growth["miss_tokens"] == 20_600
        assert growth["live_prefill_bytes"] == 20_600 * GEMMA_CHUNKED_ROW + GEMMA_CHUNK_WINDOWS
        assert growth["scratch_bytes"] == _gemma_scratch(2048, 20_600 - 2048)
        assert receipt is not None and receipt["reusable_prefix_tokens"] == 0
        assert not receipt.get("refused")

    def test_the_whole_prompt_forward_is_priced_as_one_forward(self, monkeypatch):
        """``MTPLX_GEMMA4_PREFILL_CHUNK_TOKENS=whole`` (the 2.12.0 prefill):
        16,384 cold tokens in one forward, with its score block, every
        layer's rows, and no repage."""

        monkeypatch.setenv(GEMMA_WHOLE, "whole")
        manager = _manager()
        _roomy(monkeypatch, manager)
        pricing: dict = {}
        srv._prefill_admission_shed(
            _gemma_state(manager),
            prompt_ids=list(range(16_384)),
            session_bank=manager.bank,
            session_id="gemma",
            prefill_chunk_tokens=None,
            restore_mode="clone",
            pricing=pricing,
        )
        growth = pricing["growth"]
        assert growth["prefill_chunk_tokens"] is None
        assert growth["scratch_rows"] == 16_384
        assert growth["scratch_source"] == "geometry_calibrated+attention"
        assert growth["scratch_calibration"] == gemma4.GEMMA4_PREFILL_CALIBRATION
        assert growth["scratch_bytes"] == _gemma_scratch(16_384, 0)
        assert growth["layout"] == "contiguous_dense_decode"
        assert growth["repage_copy_bytes"] == 0
        assert growth["chunk_bytes"] == 16_384 * GEMMA_PLANNED_KV + _gemma_scratch(16_384, 0)
        # Every layer's row, the sliding layers' among them: the windows are
        # inside the planner's width, as 2.12.0 priced them.
        assert growth["live_prefill_bytes"] == 16_384 * GEMMA_PLANNED_KV
        # The pre-decode clone copies what decode keeps, not every layer.
        assert growth["publish_copy_bytes"] == 16_384 * GEMMA_RESIDENT + GEMMA_WINDOWS
        geometry = srv._admission_geometry(_gemma_state(manager))
        assert geometry.live_bytes_per_token == GEMMA_PLANNED_KV

    def test_the_whole_prompt_path_is_charged_the_attention_it_builds(self, monkeypatch):
        """``whole`` restores the one forward and its every-layer row width,
        not 2.12.0's attention charge. Both paths materialize the
        full-attention layers' scores (their 512-wide heads never fuse:
        ``evidence/gemma-chunked-prefill/sdpa-routing-probe.txt``), which
        2.12.0 charged only through the sliding mask: nothing for a
        1,024-token cold prompt (80 MiB here) and 77,904,000 bytes for a
        600-token turn over 30,000 cached tokens (1,468,800,000 here, the
        full-attention block). Its figures would admit that warm turn
        1.4 GB short, so ``whole`` keeps the corrected charge."""

        monkeypatch.setenv(GEMMA_WHOLE, "whole")
        runtime = _gemma_runtime()
        assert runtime.prefill_forward_widths(30_600, None) == [None]
        assert runtime.prefill_attention_bytes(1024, 0) == 80 * 1024 * 1024
        assert runtime.prefill_attention_bytes(600, 30_000) == 80 * 600 * 30_600
        assert 80 * 600 * 30_600 == 1_468_800_000
        # 2.12.0: the sliding mask alone, rows x (cached window + rows), and
        # nothing inside the window.
        assert 80 * 600 * (1023 + 600) == 77_904_000
        # For a cold prompt past the window the two charges agree: 16,384 x
        # 16,384 pairs either way (the cold parity test above).
        assert runtime.prefill_attention_bytes(16_384, 0) == 80 * 16_384 * 16_384

    def test_a_warm_turn_is_charged_what_the_restored_cache_keeps(self):
        """A 30,000-token conversation restored by clone for a 600-token
        turn: the restore copies the full-attention KV, the drafter's
        sliding row and the windows (5.1 GB), not 30,000 rows of every layer
        (29.5 GB)."""

        geometry = srv._admission_geometry(_gemma_state(_manager()))
        assert geometry.resident_width == GEMMA_CHUNKED_ROW
        assert geometry.live_bytes_per_token == GEMMA_CHUNKED_ROW
        growth = srv._admission_growth(
            geometry,
            prompt_tokens=30_600,
            reused_tokens=30_000,
            restore_copies_prefix=True,
            layout="contiguous_dense_decode",
            source_layout=None,
            output_tokens=512,
            publish=True,
            scratch_bytes=_gemma_scratch(600, 30_000),
        )
        assert growth["restore_copy_bytes"] == 30_000 * GEMMA_CHUNKED_ROW + GEMMA_CHUNK_WINDOWS
        assert growth["live_prefill_bytes"] == (
            30_000 * GEMMA_CHUNKED_ROW + GEMMA_CHUNK_WINDOWS + 600 * GEMMA_CHUNKED_ROW
        )
        # The pre-decode clone keeps the windows as the 600-row suffix left
        # them: the kept window and the suffix.
        assert growth["publish_copy_bytes"] == (
            30_600 * GEMMA_CHUNKED_ROW + GEMMA_WINDOW_ROW * (1023 + 600)
        )
        assert growth["growth_bytes"] < 12 * GIB

    def test_a_wider_requested_chunk_is_priced_with_its_wider_windows(self, monkeypatch):
        """The review of 808a11e2: the geometry read the default width, so a
        request at 4,096 rows a forward was billed the windows a 2,048-row
        prefill leaves (1,023 + 2,048 rows a sliding layer) instead of the
        1,023 + 4,096 its own chunks leave: 1.56 GiB short."""

        manager = _manager()
        _roomy(monkeypatch, manager)
        pricing: dict = {}
        srv._prefill_admission_shed(
            _gemma_state(manager),
            prompt_ids=list(range(16_384)),
            session_bank=manager.bank,
            session_id="gemma",
            prefill_chunk_tokens=4096,
            restore_mode="clone",
            pricing=pricing,
        )
        growth = pricing["growth"]
        assert growth["prefill_chunk_tokens"] == 4096
        assert growth["scratch_bytes"] == _gemma_scratch(4096, 16_384 - 4096)
        assert growth["live_prefill_bytes"] == (
            16_384 * GEMMA_CHUNKED_ROW + GEMMA_WINDOW_ROW * (1023 + 4096)
        )
        assert growth["live_prefill_bytes"] - (
            16_384 * GEMMA_CHUNKED_ROW + GEMMA_CHUNK_WINDOWS
        ) == int(1.5625 * GIB)

    def test_a_restore_copies_the_windows_its_entry_holds(self, monkeypatch):
        """The restored cache is the banked entry's, whatever this request's
        width: an entry a 4,096-row chunk left holds 1,023 + 4,096 rows in
        each sliding cache, and a clone restore copies them all
        (``SessionBankEntry.window_nbytes``). On a Mac where the cold bill
        does not fit the admission prices the warm restore it plans."""

        manager = _manager()
        _roomy(monkeypatch, manager)
        banked = list(range(30_000))
        entry = _put(manager.bank, banked, session_id="gemma", row_bytes=GEMMA_CHUNKED_ROW)
        entry.window_nbytes = GEMMA_WINDOW_ROW * (1023 + 4096)
        _install(monkeypatch, _Machine(manager.bank, base_gib=78.0, cache_gib=0.0, host_gib=1.0))
        pricing: dict = {}
        srv._prefill_admission_shed(
            _gemma_state(manager),
            prompt_ids=banked + list(range(10**6, 10**6 + 600)),
            session_bank=manager.bank,
            session_id="gemma",
            prefill_chunk_tokens=None,
            restore_mode="clone",
            pricing=pricing,
        )
        growth = pricing["growth"]
        assert growth["reused_tokens"] == 30_000
        assert growth["restore_copy_bytes"] == (
            30_000 * GEMMA_CHUNKED_ROW + GEMMA_WINDOW_ROW * (1023 + 4096)
        )
        assert growth["live_prefill_bytes"] == growth["restore_copy_bytes"] + 600 * GEMMA_CHUNKED_ROW

    def test_skipping_store_on_prefill_saves_nothing_here(self, monkeypatch):
        """The clone does not follow store-on-prefill: turning it off must
        not read as room."""

        monkeypatch.setenv("MTPLX_SESSION_STORE_ON_PREFILL", "0")
        manager = _manager()
        _roomy(monkeypatch, manager)
        pricing: dict = {}
        srv._prefill_admission_shed(
            _gemma_state(manager),
            prompt_ids=list(range(4_096)),
            session_bank=manager.bank,
            session_id="gemma",
            prefill_chunk_tokens=None,
            restore_mode="clone",
            pricing=pricing,
        )
        assert pricing["growth"]["publish_copy_bytes"] == (
            4_096 * GEMMA_CHUNKED_ROW + GEMMA_CHUNK_WINDOWS
        )

    def test_a_65k_cold_prompt_fits_in_chunks(self, monkeypatch):
        """65,536 cold tokens, refused as one forward on any Mac (below), fit
        a 96 GiB limit in 2,048-row chunks: 6 GiB of rows, 2.3 GiB of
        windows and a 10 GiB score block in the last chunk."""

        manager = _manager()
        _roomy(monkeypatch, manager)
        pricing: dict = {}
        receipt = srv._prefill_admission_shed(
            _gemma_state(manager),
            prompt_ids=list(range(65_536)),
            session_bank=manager.bank,
            session_id="gemma",
            prefill_chunk_tokens=None,
            restore_mode="clone",
            pricing=pricing,
        )
        assert receipt is None
        growth = pricing["growth"]
        assert growth["prefill_chunk_tokens"] == 2048
        assert growth["scratch_bytes"] == _gemma_scratch(2048, 65_536 - 2048)
        assert growth["growth_bytes"] < 30 * GIB

    def test_a_prompt_the_backend_cannot_hold_is_refused_before_it_starts(
        self, monkeypatch
    ):
        """65,536 cold tokens in one forward (``whole``): every layer holds
        all of them when the forward ends (64 GB at 983,040 B a token) plus
        the forward's scratch and its score block. Past a 96 GiB limit on
        any Mac: refused up front instead of running into the wall."""

        monkeypatch.setenv(GEMMA_WHOLE, "whole")
        manager = _manager()
        _roomy(monkeypatch, manager)
        receipt = srv._prefill_admission_shed(
            _gemma_state(manager),
            prompt_ids=list(range(65_536)),
            session_bank=manager.bank,
            session_id="gemma",
            prefill_chunk_tokens=None,
            restore_mode="clone",
        )
        assert receipt["refused"] is True
        assert receipt["refusal_reason"] == "projected_over_limit_after_reclamation"
        assert receipt["prefill_chunk_tokens"] is None
        assert receipt["growth"]["scratch_rows"] == 65_536


class TestGemmaAbortSite:
    def _runtime(self):
        return SimpleNamespace(
            model_path=Path("models/gemma4"),
            mtp_enabled=True,
            make_cache=lambda: [SimpleNamespace(offset=0)],
        )

    def _counting_check(self, trip_on: int):
        calls = []

        def check() -> bool:
            calls.append(1)
            return len(calls) >= trip_on

        return check, calls

    def _fake_forward(self, monkeypatch):
        forwards = []

        class _Rows:
            def __getitem__(self, _key):
                return self

        def fake_prefill(_runtime, prompt_ids, *, cache, phase, abort_check=None):
            forwards.append(len(prompt_ids))
            return (
                SimpleNamespace(
                    logits=_Rows(),
                    hidden=_Rows(),
                    shared_kv_states={},
                    cache_offset=len(prompt_ids),
                ),
                0.0,
            )

        monkeypatch.setattr(gemma4, "_gemma4_prefill_prompt", fake_prefill)
        return forwards

    def test_it_stops_before_the_forward_allocates(self, monkeypatch):
        forwards = self._fake_forward(monkeypatch)
        check, calls = self._counting_check(trip_on=2)
        with pytest.raises(generation.PostcommitAbort):
            gemma4._restore_or_prefill_gemma4_prompt(
                self._runtime(),
                list(range(64)),
                require_shared_kv=True,
                abort_check=check,
            )
        assert forwards == []
        assert len(calls) == 2

    def test_it_runs_when_nothing_trips(self, monkeypatch):
        forwards = self._fake_forward(monkeypatch)
        check, calls = self._counting_check(trip_on=99)
        state = gemma4._restore_or_prefill_gemma4_prompt(
            self._runtime(),
            list(range(64)),
            require_shared_kv=True,
            abort_check=check,
        )
        assert forwards == [64]
        assert len(calls) == 3
        assert state.suffix_tokens == 64

    def test_it_stops_after_the_forward_before_decode(self, monkeypatch):
        forwards = self._fake_forward(monkeypatch)
        check, calls = self._counting_check(trip_on=3)
        with pytest.raises(generation.PostcommitAbort):
            gemma4._restore_or_prefill_gemma4_prompt(
                self._runtime(),
                list(range(64)),
                require_shared_kv=True,
                abort_check=check,
            )
        assert forwards == [64]
        assert len(calls) == 3

    def test_both_gemma_branches_hand_the_check_on(self, monkeypatch):
        """generate_ar and generate_mtpk route a Gemma runtime to the
        backend's own loops; both used to drop ``abort_check``."""

        seen: dict[str, object] = {}

        def fake_ar(rt, prompt_ids, **kwargs):
            seen["ar"] = kwargs.get("abort_check")
            return "ar"

        def fake_assistant(rt, prompt_ids, **kwargs):
            seen["mtp"] = kwargs.get("abort_check")
            return "mtp"

        monkeypatch.setattr(gemma4, "generate_gemma4_ar", fake_ar)
        monkeypatch.setattr(gemma4, "generate_gemma4_assistant", fake_assistant)
        runtime = SimpleNamespace(
            backend_id="gemma4_assistant",
            config=SimpleNamespace(draft_block_size=4),
        )

        def check() -> bool:
            return False

        sampler = generation.SamplerConfig(temperature=0.0)
        assert (
            generation.generate_ar(
                runtime, [1, 2, 3], max_tokens=4, sampler=sampler, abort_check=check
            )
            == "ar"
        )
        assert (
            generation.generate_mtpk(
                runtime,
                [1, 2, 3],
                max_tokens=4,
                sampler=sampler,
                speculative_depth=3,
                abort_check=check,
            )
            == "mtp"
        )
        assert seen == {"ar": check, "mtp": check}


class _Resident(_Machine):
    """The allocator double with what the request's own arrays hold
    (``rows``): a Gemma forward leaves every prompt row resident."""

    rows = 0

    def active(self) -> int:
        return super().active() + int(self.rows)


class TestGemmaAfterItsForward:
    """The review of 23a94abf (finding 2): the per-chunk check Gemma asks
    after its one forward reserved that forward again. 24,026 cold tokens
    are billed 79.45 GiB beside 16.5 GiB of weights, which fits a 96 GiB
    limit; after the forward the rows it wrote (22 GiB) are resident, and
    38.5 + 79.45 GiB over the limit refused a prefill that had fit. After
    the forward only decode start's clone is still to come."""

    PROMPT = 24_026

    def _setup(self, monkeypatch, *, base_gib: float):
        # These cases describe the one-forward backend: run the Gemma prefill in its
        # ``whole`` mode (the 2.12.0 prefill, priced as 2.12.0 priced it). The chunked
        # default prices each chunk and checks between chunks instead.
        monkeypatch.setenv(GEMMA_WHOLE, "whole")
        monkeypatch.setattr(srv, "_PREFILL_SYSTEM_CHECK_INTERVAL_S", 0.0)
        manager = _manager()
        supply = [90 * GIB]
        monkeypatch.setattr(
            sm,
            "_reader",
            lambda: sm.SystemMemory(
                available_bytes=int(supply[0]),
                total_bytes=128 * GIB,
                level_percent=70,
                free_bytes=int(supply[0]) - 10 * GIB,
                file_backed_bytes=10 * GIB,
                wired_bytes=20 * GIB,
                compressor_bytes=GIB,
                swap_used_bytes=0,
            ),
        )
        machine = _Resident(manager.bank, base_gib=base_gib, cache_gib=0.0, host_gib=1.0)
        _install(monkeypatch, machine)
        state = _gemma_state(manager)
        pricing: dict = {}
        receipt = srv._prefill_admission_shed(
            state,
            prompt_ids=list(range(self.PROMPT)),
            session_bank=manager.bank,
            session_id="gemma",
            prefill_chunk_tokens=None,
            restore_mode="clone",
            pricing=pricing,
        )
        # The check as _run_generation arms it.
        priced = pricing.get("growth")
        reserve = srv._prefill_chunk_reserve_bytes(
            state, prompt_tokens=self.PROMPT, chunk_tokens=None, priced=priced
        )
        plan_fn = getattr(srv, "_prefill_after_forward_plan", None)
        plan = (
            plan_fn(state, prompt_tokens=self.PROMPT, chunk_tokens=None, priced=priced)
            if callable(plan_fn)
            else {}
        )
        guard = srv._PrefillSystemGuard(state, chunk_reserve_bytes=reserve, **plan)
        return machine, supply, receipt, pricing, guard

    def _prefill(self, monkeypatch, machine, supply, guard):
        rows = self.PROMPT * GEMMA_PLANNED_KV
        forwards = []

        class _Rows:
            def __getitem__(self, _key):
                return self

        def forward(_runtime, prompt_ids, *, cache, phase, abort_check=None, **_kwargs):
            # Every layer keeps every row until decode trims the windows;
            # the forward's scratch is back in the pool or the Mac by now.
            forwards.append(len(prompt_ids))
            machine.rows = rows
            supply[0] -= rows
            return (
                SimpleNamespace(
                    logits=_Rows(), hidden=_Rows(), shared_kv_states={},
                    cache_offset=len(prompt_ids),
                ),
                0.0,
            )

        monkeypatch.setattr(gemma4, "_gemma4_prefill_prompt", forward)
        runtime = SimpleNamespace(
            model_path=Path("models/gemma4"),
            mtp_enabled=True,
            make_cache=lambda: [SimpleNamespace(offset=0)],
        )
        return forwards, lambda: gemma4._restore_or_prefill_gemma4_prompt(
            runtime,
            list(range(self.PROMPT)),
            require_shared_kv=True,
            abort_check=guard,
        )

    def test_a_forward_that_fit_is_not_refused_after_it_ran(self, monkeypatch):
        machine, supply, receipt, pricing, guard = self._setup(monkeypatch, base_gib=16.5)
        growth = pricing["growth"]
        assert round((16.5 * GIB + growth["growth_bytes"]) / GIB, 2) == 95.95
        assert receipt is None or receipt.get("refused") is not True
        forwards, prefill = self._prefill(monkeypatch, machine, supply, guard)
        state = prefill()
        assert forwards == [self.PROMPT]
        assert guard.tripped is None
        assert state.suffix_tokens == self.PROMPT
        # After the forward it reserved decode start's clone, not the forward.
        assert guard.prefill_done_by == "forward_rows_resident"
        assert guard.after_prefill_reserve_bytes == growth["publish_copy_bytes"]
        assert guard.after_prefill_reserve_bytes < 3 * GIB

    def test_the_forward_is_still_reserved_before_it_runs(self, monkeypatch):
        """Half a GiB more of anything else and the same forward no longer
        fits: the check before it stops the request, nothing allocated."""

        machine, supply, _receipt, _pricing, guard = self._setup(monkeypatch, base_gib=17.0)
        forwards, prefill = self._prefill(monkeypatch, machine, supply, guard)
        with pytest.raises(generation.PostcommitAbort):
            prefill()
        assert forwards == []
        assert guard.tripped["reason"] == "engine_limit"
        assert guard.tripped["reserve_after_prefill"] is False


class TestTheGenericLoopsProgress:
    def test_the_last_chunk_releases_the_forward_reservation(self, monkeypatch):
        """The generic loop reports each chunk before its post-chunk check;
        after the last one the check reserves what is still to come."""

        monkeypatch.setattr(srv, "_PREFILL_SYSTEM_CHECK_INTERVAL_S", 0.0)
        monkeypatch.setattr(sm, "_reader", lambda: None)
        manager = _manager()
        _install(monkeypatch, _Machine(manager.bank, base_gib=90.0, cache_gib=0.0, host_gib=1.0))
        state = _gemma_state(manager)
        guard = srv._PrefillSystemGuard(
            state, chunk_reserve_bytes=10 * GIB, after_prefill_reserve_bytes=GIB
        )
        guard.note_prefill_progress({"phase": "chunk", "tokens_done": 4_096, "tokens_total": 8_192})
        assert guard() is True
        assert guard.tripped["chunk_reserve_bytes"] == 10 * GIB
        guard = srv._PrefillSystemGuard(
            state, chunk_reserve_bytes=10 * GIB, after_prefill_reserve_bytes=GIB
        )
        guard.note_prefill_progress({"phase": "chunk", "tokens_done": 8_192, "tokens_total": 8_192})
        assert guard() is False
        assert guard.prefill_done_by == "prefill_progress"


class TestTheGenericLoop:
    def test_it_forwards_the_whole_prompt_without_sustained_prefill(self, monkeypatch):
        monkeypatch.delenv("MTPLX_SUSTAINED_PREFILL", raising=False)
        assert generation.prefill_forward_widths(SimpleNamespace(), 30_000, 4096) == [None]

    def test_it_runs_the_request_chunk_then_the_profiles(self):
        runtime = SimpleNamespace()
        assert generation.prefill_forward_widths(runtime, 30_000, 4096) == [4096, 2048]
        assert generation.prefill_forward_widths(runtime, 30_000, None) == [2048]
        assert generation.prefill_forward_widths(runtime, 30_000, 1024) == [1024]
        assert generation.prefill_cache_layout(runtime, 30_000) == "contiguous_dense_decode"
        assert generation.prefill_cache_layout(runtime, 200_000) == "contiguous_then_repage"

    def test_a_backend_answers_for_itself(self, monkeypatch):
        """Gemma 4 chunks whatever the sustained-prefill switch says: the
        request's width, then its default; ``whole`` is one forward."""

        runtime = _gemma_runtime()
        monkeypatch.delenv("MTPLX_SUSTAINED_PREFILL", raising=False)
        assert generation.prefill_forward_widths(runtime, 30_000, 4096) == [4096, 2048]
        assert generation.prefill_forward_widths(runtime, 30_000, None) == [2048]
        assert generation.prefill_forward_widths(runtime, 30_000, 1024) == [1024]
        assert generation.prefill_cache_layout(runtime, 200_000) == "contiguous_dense_decode"
        monkeypatch.setenv(GEMMA_WHOLE, "whole")
        assert generation.prefill_forward_widths(runtime, 30_000, 4096) == [None]


class TestThe4B:
    """Qwen3.5-4B runs the generic loop: the profile's 2,048-row chunk,
    its own geometry (the gated-delta layers are its widest), the planner's
    KV width, no backend terms."""

    def _runtime(self):
        from mlx_lm.models.qwen3_5 import TextModelArgs

        return SimpleNamespace(
            model=SimpleNamespace(
                language_model=SimpleNamespace(args=TextModelArgs.from_dict(Q4B_TEXT))
            ),
            mtp_enabled=True,
            model_path=Path("models/qwen3.5-4b"),
        )

    def test_its_row_is_read_from_its_own_config(self):
        args = srv._runtime_text_args(self._runtime())
        # Gated delta: 2 x (2,560 + 4,096 + 8,192 + 4,096 + 4,096 + 2,560)
        # + 4 x (4,096 + 8,192); wider than its MLP (60,416) and attention.
        assert srv._forward_row_bytes(args) == 100_352

    def test_it_is_priced_at_the_profile_chunk(self, monkeypatch):
        kv = dense_kv_bytes_per_token_from_config({"text_config": Q4B_TEXT})
        assert kv == 8 * 2 * 4 * 256 * 2
        plan = SimpleNamespace(
            available=True,
            kv_bytes_per_token=kv,
            kv_bytes_per_token_effective=kv,
            aux_bytes_per_token=0,
            prefill_transient_bytes_per_token=0,
            runtime_transients_bytes=3 * GIB,
            model_weights_bytes=int(2.6 * GIB),
        )
        manager = _manager()
        _roomy(monkeypatch, manager)
        # The 4B's own seat: its weights and a few GiB of headroom under the
        # 12 GiB limit, so the request is admitted at the profile's chunk.
        _install(monkeypatch, _Machine(manager.bank, base_gib=2.6, cache_gib=0.0, host_gib=1.0))
        state = _state(manager, plan=plan, runtime=self._runtime(), limit_gib=12, total_gib=16)
        pricing: dict = {}
        srv._prefill_admission_shed(
            state,
            prompt_ids=list(range(20_000)),
            session_bank=manager.bank,
            session_id="small",
            prefill_chunk_tokens=None,
            restore_mode="clone",
            pricing=pricing,
        )
        growth = pricing["growth"]
        # A dense family: what was measured on it (2026-10-02, the 4B
        # through mtplx serve: 1.55 to 1.65 GiB at 2,048 rows from 7K to 49K
        # of context), 0.75 GiB plus ten MLP rows (2 x (2,560 + 3 x 9,216))
        # a row at this width: 1.90 GiB. 2.12.1 charged the 27B's 3 GiB.
        args = srv._runtime_text_args(self._runtime())
        assert growth["prefill_chunk_tokens"] == 2048
        assert growth["scratch_rows"] == 2048
        assert growth["scratch_source"] == "geometry_measured"
        assert growth["scratch_bytes"] == srv._dense_prefill_bill(args, 2048)
        assert growth["scratch_bytes"] == 0.75 * GIB + 10 * 2 * (2_560 + 3 * 9_216) * 2048
        assert 1.65 * GIB < growth["scratch_bytes"] < 1.91 * GIB
        # A 59-token prompt (0.19 GiB measured; 2.12.1 charged 2.26 GiB).
        assert 0.19 * GIB < srv._dense_prefill_bill(args, 59) < 0.4 * GIB
        assert growth["publish_copy_bytes"] == 20_000 * kv
