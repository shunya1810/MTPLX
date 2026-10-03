"""#393 memory math: QSA per-token terms in the plan, geometric cache growth.

The shipped plan priced qwen4_exp like a dense family: KV bytes only, flat
3 GiB transient headroom. The real serve carries per-token QSA bookkeeping
(raw indexer keys, pooled block keys, the fp32-transposed mirror, the MTP
head's uncounted KV) and a prefill transient that scales with the full
token count — so a 128 GB machine "fit" 262K on paper and wedged at ~119 GB
in practice. These tests pin the new terms and their effect on the fit, and
the QSA cache's doubling growth (the old fixed +256-row growth full-copied
the buffer every step: Θ(N²) memcpy, ~34 GB of copy traffic per layer over
a 262K decode).
"""

from __future__ import annotations

import mlx.core as mx

from mtplx.memory_plan import (
    plan_memory,
    qsa_aux_bytes_per_token_from_config,
    qsa_prefill_transient_bytes_per_token_from_config,
)
from mtplx.models.qwen4_exp import QSACache

GIB = 1024**3

# Real-pack geometry (qwen4_exp): 36 linear + 12 full layers, idx_dim 128,
# compress ratio 4, kv 4 x 128.
QSA_CONFIG = {
    "text_config": {
        "indexer_n_heads": 4,
        "indexer_head_dim": 128,
        "indexer_compress_ratio": 4,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "layer_types": ["linear_attention"] * 36 + ["full_attention"] * 12,
    }
}

DENSE_CONFIG = {"num_key_value_heads": 8, "head_dim": 128}


class TestQsaPlanTerms:
    def test_aux_bytes_for_real_pack_geometry(self):
        # 13 QSA caches (12 layers + MTP head) x 448 B + 2048 B MTP-head KV.
        assert qsa_aux_bytes_per_token_from_config(QSA_CONFIG) == 7872

    def test_transient_bytes_at_default_chunk(self):
        assert (
            qsa_prefill_transient_bytes_per_token_from_config(
                QSA_CONFIG, chunk_size=2048
            )
            == 104448
        )

    def test_dense_families_unpriced(self):
        assert qsa_aux_bytes_per_token_from_config(DENSE_CONFIG) == 0
        assert (
            qsa_prefill_transient_bytes_per_token_from_config(DENSE_CONFIG)
            == 0
        )
        assert qsa_aux_bytes_per_token_from_config(None) == 0

    def test_terms_shrink_the_fit_and_flag_overcommit(self):
        # The #393 machine: 128 GB RAM, ~83 GiB 4-bit Flash-Next pack,
        # 262K requested. model_max set high so neither fit saturates at
        # the model cap and the per-token ratio is visible.
        base = dict(
            total_ram_bytes=128 * GIB,
            model_weights_bytes=83 * GIB,
            kv_bytes_per_token=24576,
            model_max_context=1_048_576,
            requested_context=262144,
            # The #393 machine ran at the then-default 96 GiB limit; pinned so
            # this test stays about the per-token terms, not the default.
            usable_bytes_override=96 * GIB,
            usable_bytes_explicit=True,
        )
        dense_priced = plan_memory(**base)
        qsa_priced = plan_memory(
            **base,
            aux_bytes_per_token=7872,
            prefill_transient_bytes_per_token=104448,
        )
        assert qsa_priced.context_window_fit < dense_priced.context_window_fit
        # The per-token cost is ~5.6x KV alone; the honest fit must drop by
        # more than 4x, not by a rounding margin.
        assert (
            qsa_priced.context_window_fit * 4
            < dense_priced.context_window_fit
        )
        # KV-only pricing said 262K fits (the shipped lie); the honest
        # terms flag exactly that request as overcommitted — which is what
        # arms the request-time 507 in the server.
        assert not dense_priced.context_overcommitted
        assert qsa_priced.context_overcommitted
        assert qsa_priced.context_window_resolved == 262144
        # Sanity band for the honest fit on this machine class (audit
        # estimate 90-152K; alignment granularity 2048).
        assert 64 * 1024 <= qsa_priced.context_window_fit <= 160 * 1024
        d = qsa_priced.to_dict()
        assert d["aux_bytes_per_token"] == 7872
        assert d["prefill_transient_bytes_per_token"] == 104448


class TestQsaCacheGeometricGrowth:
    def test_raw_growth_is_doubling_not_per_step(self):
        cache = QSACache(4)
        caps: list[int] = []
        rows = 0
        for i in range(40):
            keys = mx.full((1, 100, 8), i, dtype=mx.bfloat16)
            cache.write_raw(keys)
            cache.kv.offset += 100
            rows += 100
            assert cache.raw_keys is not None
            if not caps or cache.raw_keys.shape[1] != caps[-1]:
                caps.append(cache.raw_keys.shape[1])
        # 4000 rows under the old fixed +256 growth = 16 distinct
        # capacities; doubling reaches it in <= 6 (256..4096).
        assert len(caps) <= 6, caps
        assert all(b >= 2 * a for a, b in zip(caps, caps[1:])), caps
        # Content preserved across every regrow.
        got = cache.raw_keys[:, :rows, :]
        for i in range(40):
            block = got[:, i * 100 : (i + 1) * 100, :]
            assert mx.all(block == i).item(), f"chunk {i} lost in regrow"

    def test_pooled_growth_keeps_fp32_mirror_content_equal(self):
        cache = QSACache(4)
        nb = 0
        for i in range(12):
            blocks = mx.full((1, 64, 8), i + 1, dtype=mx.bfloat16)
            cache.write_pooled(blocks, nb, nb + 64)
            nb += 64
        view = cache.pooled_f32_view(nb)
        assert view.shape == (1, 1, 8, nb)
        expect = mx.swapaxes(
            cache.pooled[:, :nb, :].astype(mx.float32), 1, 2
        )[:, None]
        assert mx.array_equal(view, expect).item()
