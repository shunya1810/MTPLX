"""M1-family memory defaults: MLX cache limit capped at 1 GiB, 2 bank entries per session,
MMA prefill attention from the first chunk.

Explicit settings win over the MTPLX_M1_LONG_CONTEXT gate; other machines keep
the RAM tiers and 3 entries.
"""

from mtplx import attention_split, session_bank
from mtplx.server import openai

GIB = 1024**3


def _ram(monkeypatch, total):
    monkeypatch.setattr(openai, "_total_ram_bytes", lambda: total)


def test_cache_limit_capped_on_m1_family(monkeypatch):
    monkeypatch.setenv("MTPLX_M1_LONG_CONTEXT", "1")
    for total, expected in ((32 * GIB, 1 * GIB), (64 * GIB, 1 * GIB), (128 * GIB, 1 * GIB)):
        _ram(monkeypatch, total)
        assert openai._default_mlx_cache_limit_bytes() == expected


def test_cache_limit_tiers_elsewhere(monkeypatch):
    monkeypatch.setenv("MTPLX_M1_LONG_CONTEXT", "0")
    for total, expected in ((32 * GIB, 2 * GIB), (64 * GIB, 4 * GIB), (96 * GIB, 6 * GIB), (128 * GIB, 8 * GIB)):
        _ram(monkeypatch, total)
        assert openai._default_mlx_cache_limit_bytes() == expected


def test_cache_limit_budget_rule_ignores_the_gate(monkeypatch):
    for gate in ("0", "1"):
        monkeypatch.setenv("MTPLX_M1_LONG_CONTEXT", gate)
        assert openai._default_mlx_cache_limit_bytes(48 * GIB) == 6 * GIB


def test_unknown_ram_leaves_mlx_default(monkeypatch):
    monkeypatch.setenv("MTPLX_M1_LONG_CONTEXT", "1")
    _ram(monkeypatch, None)
    assert openai._default_mlx_cache_limit_bytes() is None


def test_per_session_entries_follow_the_gate(monkeypatch):
    monkeypatch.delenv("MTPLX_SESSION_BANK_PER_SESSION_MAX_ENTRIES", raising=False)
    monkeypatch.setenv("MTPLX_M1_LONG_CONTEXT", "1")
    assert session_bank._per_session_max_entries() == 2
    monkeypatch.setenv("MTPLX_M1_LONG_CONTEXT", "0")
    assert session_bank._per_session_max_entries() == 3


def test_explicit_per_session_entries_win(monkeypatch):
    for gate in ("0", "1"):
        monkeypatch.setenv("MTPLX_M1_LONG_CONTEXT", gate)
        for value, expected in (("1", 1), ("3", 3), ("0", 0)):
            monkeypatch.setenv("MTPLX_SESSION_BANK_PER_SESSION_MAX_ENTRIES", value)
            assert session_bank._per_session_max_entries() == expected


def test_mma_prefill_min_prefix_follows_the_gate(monkeypatch):
    monkeypatch.delenv("MTPLX_GQA_MMA_PREFILL_MIN_PREFIX", raising=False)
    monkeypatch.setenv("MTPLX_M1_LONG_CONTEXT", "1")
    assert attention_split._gqa_mma_prefill_min_prefix() == 0
    monkeypatch.setenv("MTPLX_M1_LONG_CONTEXT", "0")
    assert attention_split._gqa_mma_prefill_min_prefix() == 49152


def test_explicit_mma_prefill_min_prefix_wins(monkeypatch):
    for gate in ("0", "1"):
        monkeypatch.setenv("MTPLX_M1_LONG_CONTEXT", gate)
        monkeypatch.setenv("MTPLX_GQA_MMA_PREFILL_MIN_PREFIX", "16384")
        assert attention_split._gqa_mma_prefill_min_prefix() == 16384
