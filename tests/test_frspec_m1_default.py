"""FR-Spec M1-family default: unset MTPLX_FRSPEC_DRAFT turns the pruned draft head on
(Japanese-aware built-in list, legacy swap) on M1 only; explicit settings keep their meaning."""

from mtplx import frspec_draft as fs


def _clear(monkeypatch):
    for k in ("MTPLX_FRSPEC_DRAFT", "MTPLX_FRSPEC_VOCAB", "MTPLX_FRSPEC_LEGACY", "MTPLX_FRSPEC_N"):
        monkeypatch.delenv(k, raising=False)


def test_m1_default_on(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("MTPLX_M1_LONG_CONTEXT", "1")
    assert fs.frspec_enabled() is True
    assert fs.frspec_explicitly_requested() is False
    assert fs.frspec_legacy_enabled() is True
    ids = fs.load_frspec_ids()
    assert ids is not None and len(ids) == 103887


def test_off_elsewhere(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("MTPLX_M1_LONG_CONTEXT", "0")
    assert fs.frspec_enabled() is False
    assert fs.frspec_legacy_enabled() is False
    assert fs.load_frspec_ids() is None


def test_explicit_settings_win(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("MTPLX_M1_LONG_CONTEXT", "1")
    monkeypatch.setenv("MTPLX_FRSPEC_DRAFT", "0")
    assert fs.frspec_enabled() is False
    assert fs.frspec_legacy_enabled() is False
    monkeypatch.setenv("MTPLX_FRSPEC_DRAFT", "1")
    monkeypatch.setenv("MTPLX_FRSPEC_VOCAB", "builtin:qwen38-code-64k")
    assert fs.frspec_enabled() is True and fs.frspec_explicitly_requested() is True
    assert fs.frspec_legacy_enabled() is False     # explicit FR-Spec keeps its own legacy switch
    assert len(fs.load_frspec_ids()) == 65536


def test_japanese_list_contains_the_code_list():
    import numpy as np

    code = set(np.load(fs._BUILTIN_VOCABS["qwen38-code-64k"]).reshape(-1).tolist())
    ja = set(np.load(fs._BUILTIN_VOCABS["qwen38-code64k-ja"]).reshape(-1).tolist())
    assert code <= ja and len(ja) == 103887
