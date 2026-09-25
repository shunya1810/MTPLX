"""context_copy_enabled(): explicit MTPLX_CONTEXT_COPY wins; unset follows the M1-family gate."""

from mtplx.context_copy import context_copy_enabled


def test_unset_is_off_on_m1_family(monkeypatch):
    monkeypatch.delenv("MTPLX_CONTEXT_COPY", raising=False)
    monkeypatch.setenv("MTPLX_M1_LONG_CONTEXT", "1")
    assert context_copy_enabled() is False


def test_unset_is_on_elsewhere(monkeypatch):
    monkeypatch.delenv("MTPLX_CONTEXT_COPY", raising=False)
    monkeypatch.setenv("MTPLX_M1_LONG_CONTEXT", "0")
    assert context_copy_enabled() is True


def test_explicit_value_wins_over_the_gate(monkeypatch):
    for gate in ("0", "1"):
        monkeypatch.setenv("MTPLX_M1_LONG_CONTEXT", gate)
        monkeypatch.setenv("MTPLX_CONTEXT_COPY", "1")
        assert context_copy_enabled() is True
        for off in ("0", "false", "off"):
            monkeypatch.setenv("MTPLX_CONTEXT_COPY", off)
            assert context_copy_enabled() is False
