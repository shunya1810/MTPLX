"""Wiring and parity tests for load-time projection fusion (PR #316 port).

Everything runs on the CPU stream: fusion is weight surgery plus slicing, and
the parity claim here is that each member serves exactly its own rows. The
Metal-side claims (bitwise identity of the fused kernel at M<=4, the raised
fp16 window) are measured live and gated by the identity/selfcheck corpus,
not unit-testable off-device.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest

from mtplx.proj_fusion import (
    FUSE_ENV,
    FusedProjectionMember,
    configure_fused_projections,
    fused_projection_stats,
    requested_groups,
)

K = 128
GROUP_SIZE = 32
BITS = 4


@pytest.fixture(autouse=True)
def _cpu_stream():
    with mx.stream(mx.cpu):
        yield


def _qlinear(n: int, seed: int) -> nn.QuantizedLinear:
    mx.random.seed(seed)
    lin = nn.Linear(K, n, bias=False)
    return nn.QuantizedLinear.from_linear(lin, group_size=GROUP_SIZE, bits=BITS)


class _Gdn(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_proj_qkv = _qlinear(96, 1)
        self.in_proj_z = _qlinear(64, 2)
        self.in_proj_b = _qlinear(8, 3)
        self.in_proj_a = _qlinear(8, 4)


class _Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = _qlinear(64, 5)
        self.k_proj = _qlinear(32, 6)
        self.v_proj = _qlinear(32, 7)


class _Mlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = _qlinear(64, 8)
        self.up_proj = _qlinear(64, 9)


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.gdn = _Gdn()
        self.attn = _Attn()
        self.mlp = _Mlp()


_GROUPS = {
    "gdn": ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"),
    "attn": ("q_proj", "k_proj", "v_proj"),
    "mlp": ("gate_proj", "up_proj"),
}


def _originals(model: _Model) -> dict[str, list[nn.QuantizedLinear]]:
    return {
        owner: [getattr(getattr(model, owner), n) for n in names]
        for owner, names in _GROUPS.items()
    }


def test_env_parsing(monkeypatch):
    monkeypatch.delenv(FUSE_ENV, raising=False)
    monkeypatch.setenv("MTPLX_M1_LONG_CONTEXT", "0")
    assert requested_groups() == set()
    monkeypatch.setenv("MTPLX_M1_LONG_CONTEXT", "1")
    assert requested_groups() == {"gdn", "attn"}
    monkeypatch.setenv(FUSE_ENV, "0")
    assert requested_groups() == set()
    monkeypatch.setenv(FUSE_ENV, "1")
    assert requested_groups() == {"gdn", "attn"}
    monkeypatch.setenv(FUSE_ENV, "all")
    assert requested_groups() == {"gdn", "attn", "mlp"}
    monkeypatch.setenv(FUSE_ENV, "mlp, attn")
    assert requested_groups() == {"attn", "mlp"}
    monkeypatch.setenv(FUSE_ENV, "off")
    assert requested_groups() == set()


def test_all_groups_fuse_and_members_stay_quantized_linear(monkeypatch):
    monkeypatch.setenv(FUSE_ENV, "all")
    monkeypatch.delenv("MTPLX_PACKED_PROJ_CONCATS", raising=False)
    model = _Model()
    stats = configure_fused_projections(model)
    assert (stats["gdn"], stats["attn"], stats["mlp"]) == (1, 1, 1)
    assert stats["skipped"] == 0
    assert stats["freed_bytes"] > 0
    for owner, names in _GROUPS.items():
        for name in names:
            member = getattr(getattr(model, owner), name)
            assert isinstance(member, FusedProjectionMember)
            assert isinstance(member, nn.QuantizedLinear)


def test_member_outputs_match_unfused_exactly(monkeypatch):
    monkeypatch.setenv(FUSE_ENV, "all")
    monkeypatch.delenv("MTPLX_PACKED_PROJ_CONCATS", raising=False)
    model = _Model()
    originals = _originals(model)
    configure_fused_projections(model)
    for rows in (1, 4):
        for owner, names in _GROUPS.items():
            x = mx.random.normal((rows, K))
            for name, original in zip(names, originals[owner]):
                member = getattr(getattr(model, owner), name)
                assert mx.array_equal(member(x), original(x)), (
                    f"{owner}.{name} rows={rows} diverged from unfused"
                )


def test_hub_runs_one_fused_dispatch_per_distinct_input(monkeypatch):
    monkeypatch.setenv(FUSE_ENV, "attn")
    monkeypatch.delenv("MTPLX_PACKED_PROJ_CONCATS", raising=False)
    model = _Model()
    configure_fused_projections(model)
    x = mx.random.normal((2, K))
    for name in _GROUPS["attn"]:
        getattr(model.attn, name)(x)
    stats = fused_projection_stats()
    assert stats["fused_dispatches"] == 1
    assert stats["member_calls"] == 3
    y = mx.random.normal((2, K))
    for name in _GROUPS["attn"]:
        getattr(model.attn, name)(y)
    assert fused_projection_stats()["fused_dispatches"] == 2


def test_rows_above_window_take_the_unfused_lane(monkeypatch):
    monkeypatch.setenv(FUSE_ENV, "mlp")
    monkeypatch.delenv("MTPLX_PACKED_PROJ_CONCATS", raising=False)
    model = _Model()
    originals = _originals(model)
    configure_fused_projections(model)
    x = mx.random.normal((8, K))
    for name, original in zip(_GROUPS["mlp"], originals["mlp"]):
        member = getattr(model.mlp, name)
        assert mx.array_equal(member(x), original(x))
    stats = fused_projection_stats()
    assert stats["unfused_lane_calls"] == 2
    assert stats["fused_dispatches"] == 0


def test_refuses_to_stack_on_packed_concats(monkeypatch):
    monkeypatch.setenv(FUSE_ENV, "all")
    monkeypatch.setenv("MTPLX_PACKED_PROJ_CONCATS", "1")
    model = _Model()
    stats = configure_fused_projections(model)
    assert stats["enabled"] is False
    assert (stats["gdn"], stats["attn"], stats["mlp"]) == (0, 0, 0)
    assert any("PACKED_PROJ" in r for r in stats["skip_reasons"])
    assert isinstance(model.attn.q_proj, nn.QuantizedLinear)
    assert not isinstance(model.attn.q_proj, FusedProjectionMember)


def test_packed_concats_refuses_under_proj_fusion(monkeypatch):
    pytest.importorskip("mlx_lm.models.qwen3_next")
    from mtplx.packed_concats import COUNTERS, install_qwen3_next_packed_concats

    monkeypatch.setenv("MTPLX_PACKED_PROJ_CONCATS", "1")
    monkeypatch.setenv(FUSE_ENV, "all")
    before = COUNTERS.get("refused_proj_fusion", 0)
    assert install_qwen3_next_packed_concats(_Model()) is None
    assert COUNTERS.get("refused_proj_fusion", 0) == before + 1
