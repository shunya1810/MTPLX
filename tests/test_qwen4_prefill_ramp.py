"""Flash-Next prefill ramp (generation._wide_prefill_ramp_spans): forwards wider
than the admitted chunk while the prompt's KV is still unwritten."""

from __future__ import annotations

import pytest

from mtplx import generation as G


@pytest.fixture
def wide(monkeypatch):
    monkeypatch.setenv("MTPLX_QWEN4_PREFILL_WIDE_CHUNK", "4096")
    for name in ("MTPLX_QWEN4_PREFILL_RAMP", "MTPLX_QWEN4_PREFILL_RAMP_KV_BYTES",
                 "MTPLX_PREFILL_CHUNK_SIZE", "MTPLX_PREFILL_CHUNK_SIZE_DENSE",
                 "MTPLX_PREFILL_CHUNK_SIZE_REPAGE"):
        monkeypatch.delenv(name, raising=False)


def _check_cover(spans, n):
    assert spans[0][0] == 0 and spans[-1][1] == n
    assert all(a[1] == b[0] for a, b in zip(spans, spans[1:]))


def test_ramp_keeps_every_forward_under_the_admitted_peak(wide):
    n, chunk, kv = 196653, 512, 36864
    spans = G._iter_prefill_chunk_spans(n, chunk_size=chunk)
    _check_cover(spans, n)
    widths = [e - s for s, e in spans]
    assert max(widths) == 4096 and widths.count(4096) > 30
    # widths never grow again once they narrow
    assert all(a >= b for a, b in zip(widths, widths[1:-1]))
    for s, e in spans:
        w = e - s
        if w > chunk:
            assert (n - e) * kv >= (w - chunk) * G._WIDE_PREFILL_FORWARD_BYTES_PER_ROW


def test_ramp_off_switch_pins_and_small_prompts(wide, monkeypatch):
    flat = [(0, 2048), (2048, 4096), (4096, 5000)]
    assert G._iter_prefill_chunk_spans(5000, chunk_size=2048) == flat  # too short to ramp
    assert G._wide_prefill_ramp_spans(100000, 4096) is None  # already the wide width
    monkeypatch.setenv("MTPLX_QWEN4_PREFILL_RAMP", "0")
    assert G._wide_prefill_ramp_spans(100000, 512) is None
    monkeypatch.delenv("MTPLX_QWEN4_PREFILL_RAMP")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE", "1024")  # an operator's pinned chunk
    assert G._wide_prefill_ramp_spans(100000, 512) is None


def test_ramp_respects_mandatory_edges(wide):
    spans = G._iter_prefill_chunk_spans(100000, chunk_size=1024, mandatory_edges=(5000,))
    _check_cover(spans, 100000)
    assert any(e == 5000 for _s, e in spans)


def test_lean_long_prefill_scope_drops_hc_packs_and_restores_pool(monkeypatch):
    import mlx.core as mx

    class M:
        pass

    a, b = M(), M()
    object.__setattr__(a, "_hc_q8_pack", (mx.zeros((4,)),))
    b._hc_q8_pack = None

    class Model:
        def named_modules(self):
            return [("a", a), ("b", b)]

    class RT:
        model = Model()

    before = mx.set_cache_limit(1 << 30)
    try:
        with G._lean_long_prefill_scope(RT(), 1000):  # short prompt: untouched
            assert a._hc_q8_pack is not None
        with G._lean_long_prefill_scope(RT(), 200000):
            assert a._hc_q8_pack is None
            assert mx.set_cache_limit(128 << 20) == 128 << 20
        assert mx.set_cache_limit(1 << 30) == 1 << 30
    finally:
        mx.set_cache_limit(before)
