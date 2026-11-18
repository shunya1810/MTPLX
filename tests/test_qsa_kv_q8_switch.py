"""How int8 QSA KV is switched on: the serve flag, an explicit env, and the prefill reader that goes with it."""

import mtplx.models.qwen4_exp as Q


def _clear(monkeypatch):
    for name in ("MTPLX_QSA_KV_Q8", "MTPLX_PAGED_KV_QUANT", "MTPLX_QSA_PREFILL_ROWS_MMA"):
        monkeypatch.delenv(name, raising=False)


def test_off_by_default(monkeypatch):
    _clear(monkeypatch)
    assert not Q.qsa_kv_q8_enabled()
    assert type(Q.QSACache().kv) is Q.QSAKVCache


def test_the_serve_flag_turns_it_on(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("MTPLX_PAGED_KV_QUANT", "q8")
    assert Q.qsa_kv_q8_enabled()
    assert type(Q.QSACache().kv) is Q.QSAKVCacheQ8


def test_the_explicit_switch_wins_over_the_serve_flag(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("MTPLX_PAGED_KV_QUANT", "q8")
    monkeypatch.setenv("MTPLX_QSA_KV_Q8", "0")
    assert not Q.qsa_kv_q8_enabled()
    monkeypatch.setenv("MTPLX_PAGED_KV_QUANT", "off")
    monkeypatch.setenv("MTPLX_QSA_KV_Q8", "1")
    assert Q.qsa_kv_q8_enabled()


def test_the_prefill_rows_reader_follows_the_cache_type(monkeypatch):
    _clear(monkeypatch)
    assert not Q._qsa_prefill_rows_mma_enabled(False)
    assert Q._qsa_prefill_rows_mma_enabled(True)
    monkeypatch.setenv("MTPLX_QSA_PREFILL_ROWS_MMA", "0")
    assert not Q._qsa_prefill_rows_mma_enabled(True)
    monkeypatch.setenv("MTPLX_QSA_PREFILL_ROWS_MMA", "1")
    assert Q._qsa_prefill_rows_mma_enabled(False)
