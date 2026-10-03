from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from mtplx.server import openai


GiB = 1024**3


def _fake_mx(*, top_level: bool = True, system_wired: int | None = None):
    """``system_wired`` mimics a stock macOS wired limit: MLX reports it as
    ``max_recommended_working_set_size`` and refuses a larger wired limit."""
    calls: list[tuple[str, int]] = []

    def _wired(name):
        def set_wired_limit(value):
            if system_wired is not None and int(value) > system_wired:
                raise ValueError(
                    "Setting a wired limit larger than the maximum working set "
                    "size is not allowed."
                )
            calls.append((name, int(value)))

        return set_wired_limit

    metal = SimpleNamespace(
        is_available=lambda: True,
        set_memory_limit=lambda value: calls.append(("metal_memory", int(value))),
        set_wired_limit=_wired("metal_wired"),
    )
    mx = SimpleNamespace(metal=metal)
    if top_level:
        mx.set_memory_limit = lambda value: calls.append(("memory", int(value)))
        mx.set_wired_limit = _wired("wired")
    if system_wired is not None:
        mx.device_info = lambda: {"max_recommended_working_set_size": system_wired}
    return mx, calls


def test_parse_metal_memory_size_bytes_accepts_suffixes_and_fallbacks():
    assert openai._parse_metal_memory_size_bytes("64G", 1) == 64 * GiB
    assert openai._parse_metal_memory_size_bytes("1.5T", 1) == int(1.5 * 1024**4)
    assert openai._parse_metal_memory_size_bytes("512M", 1) == 512 * 1024**2
    assert openai._parse_metal_memory_size_bytes("bad", 123) == 123
    assert openai._parse_metal_memory_size_bytes("", 456) == 456


def test_detect_total_ram_uses_psutil_when_available(monkeypatch):
    fake_psutil = SimpleNamespace(
        virtual_memory=lambda: SimpleNamespace(total=128 * GiB)
    )
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)

    total, source = openai._detect_total_ram_bytes_for_metal_caps()

    assert total == 128 * GiB
    assert source == "psutil"


def test_detect_total_ram_falls_back_to_sysctl_on_macos(monkeypatch):
    monkeypatch.setitem(sys.modules, "psutil", None)
    monkeypatch.setattr(openai.sys, "platform", "darwin")
    monkeypatch.setattr(
        openai.subprocess,
        "check_output",
        lambda *_args, **_kwargs: str(64 * GiB),
    )

    total, source = openai._detect_total_ram_bytes_for_metal_caps()

    assert total == 64 * GiB
    assert source == "sysctl_hw_memsize"


def test_apply_metal_memory_caps_uses_top_level_mlx_apis(monkeypatch):
    mx, calls = _fake_mx(top_level=True)
    monkeypatch.setenv("MTPLX_MEMORY_LIMIT_BYTES", "64G")
    monkeypatch.setenv("MTPLX_WIRED_LIMIT_BYTES", "48G")

    result = openai._apply_metal_memory_caps(
        mx_module=mx,
        total_ram_bytes=128 * GiB,
    )

    assert result["applied"] is True
    assert result["memory_limit_bytes"] == 64 * GiB
    assert result["wired_limit_bytes"] == 48 * GiB
    assert result["memory_limit_api"] == "mx.set_memory_limit"
    assert result["memory_limit_source"] == "env"
    assert result["wired_limit_api"] == "mx.set_wired_limit"
    assert calls == [("memory", 64 * GiB), ("wired", 48 * GiB)]


def test_apply_metal_memory_caps_caps_large_unified_memory_defaults(monkeypatch):
    mx, calls = _fake_mx(top_level=True)
    monkeypatch.delenv("MTPLX_MEMORY_LIMIT_BYTES", raising=False)
    monkeypatch.delenv("MTPLX_WIRED_LIMIT_BYTES", raising=False)

    result = openai._apply_metal_memory_caps(
        mx_module=mx,
        total_ram_bytes=512 * GiB,
    )

    assert result["applied"] is True
    assert result["memory_limit_bytes"] == 192 * GiB
    assert result["memory_limit_source"] == "default"
    assert result["wired_limit_bytes"] == 160 * GiB
    assert calls == [("memory", 192 * GiB), ("wired", 160 * GiB)]


def test_apply_metal_memory_caps_preserves_128g_defaults(monkeypatch):
    mx, calls = _fake_mx(top_level=True)
    monkeypatch.delenv("MTPLX_MEMORY_LIMIT_BYTES", raising=False)
    monkeypatch.delenv("MTPLX_WIRED_LIMIT_BYTES", raising=False)

    result = openai._apply_metal_memory_caps(
        mx_module=mx,
        total_ram_bytes=128 * GiB,
    )

    # G5 (09-29): a 128 GB desktop leaves 38 GiB outside the engine, 90 GiB
    # rather than the 75% rule's 96 GiB; the wired default is unchanged.
    assert result["applied"] is True
    assert result["memory_limit_bytes"] == 90 * GiB
    assert result["wired_limit_bytes"] == int(128 * GiB * 0.60)
    assert calls == [("memory", 90 * GiB), ("wired", int(128 * GiB * 0.60))]


def test_apply_metal_memory_caps_raises_default_wired_floor_for_laguna(
    monkeypatch,
):
    from mtplx.models.laguna_config import LAGUNA_S_2_1_MIN_RESIDENT_BYTES

    mx, calls = _fake_mx(top_level=True)
    monkeypatch.delenv("MTPLX_MEMORY_LIMIT_BYTES", raising=False)
    monkeypatch.delenv("MTPLX_WIRED_LIMIT_BYTES", raising=False)

    result = openai._apply_metal_memory_caps(
        mx_module=mx,
        total_ram_bytes=96 * GiB,
        minimum_resident_bytes=LAGUNA_S_2_1_MIN_RESIDENT_BYTES,
    )

    assert result["applied"] is True
    assert result["memory_limit_bytes"] == 72 * GiB
    assert result["wired_limit_bytes"] == LAGUNA_S_2_1_MIN_RESIDENT_BYTES
    assert calls == [
        ("memory", 72 * GiB),
        ("wired", LAGUNA_S_2_1_MIN_RESIDENT_BYTES),
    ]


def test_apply_metal_memory_caps_rejects_insufficient_ram_for_laguna(
    monkeypatch,
):
    from mtplx.models.laguna_config import LAGUNA_S_2_1_MIN_RESIDENT_BYTES

    mx, calls = _fake_mx(top_level=True)
    monkeypatch.delenv("MTPLX_MEMORY_LIMIT_BYTES", raising=False)
    monkeypatch.delenv("MTPLX_WIRED_LIMIT_BYTES", raising=False)

    result = openai._apply_metal_memory_caps(
        mx_module=mx,
        total_ram_bytes=64 * GiB,
        minimum_resident_bytes=LAGUNA_S_2_1_MIN_RESIDENT_BYTES,
    )

    assert result["applied"] is False
    assert result["reason"] == "insufficient_ram"
    assert result["minimum_resident_bytes"] == LAGUNA_S_2_1_MIN_RESIDENT_BYTES
    assert calls == []


def test_laguna_explicit_context_must_fit_active_metal_cap(monkeypatch):
    from mtplx.backends.descriptors import LAGUNA_AR_DESCRIPTOR
    from mtplx.models.laguna_config import LAGUNA_S_2_1_MIN_RESIDENT_BYTES

    mx, _calls = _fake_mx(top_level=True)
    monkeypatch.delenv("MTPLX_MEMORY_LIMIT_BYTES", raising=False)
    monkeypatch.delenv("MTPLX_WIRED_LIMIT_BYTES", raising=False)
    caps = openai._apply_metal_memory_caps(
        mx_module=mx,
        total_ram_bytes=128 * GiB,
        minimum_resident_bytes=LAGUNA_S_2_1_MIN_RESIDENT_BYTES,
    )

    openai._validate_backend_context_memory_budget(
        LAGUNA_AR_DESCRIPTOR,
        caps,
        None,
    )
    with pytest.raises(RuntimeError, match="context window 1,048,576"):
        openai._validate_backend_context_memory_budget(
            LAGUNA_AR_DESCRIPTOR,
            caps,
            1_048_576,
        )


def test_laguna_server_uses_safe_default_but_preserves_explicit_context():
    from mtplx.backends.descriptors import LAGUNA_AR_DESCRIPTOR

    assert (
        openai._select_backend_context_window(
            LAGUNA_AR_DESCRIPTOR,
            model_max=1_048_576,
            requested=None,
        )
        == 32_768
    )
    assert (
        openai._select_backend_context_window(
            LAGUNA_AR_DESCRIPTOR,
            model_max=1_048_576,
            requested=65_536,
        )
        == 65_536
    )


def test_prefill_preflight_accepts_qwen_yarn_million_token_window(
    monkeypatch,
    tmp_path,
):
    model = tmp_path / "qwen-yarn"
    model.mkdir()
    (model / "config.json").write_text(
        '{"text_config":{"max_position_embeddings":1048576}}',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        openai,
        "_apply_metal_memory_caps",
        lambda: {"applied": True, "memory_limit_bytes": 110 * GiB},
    )

    receipt = openai.apply_memory_caps_preflight(
        entry="bench.prefill_ladder",
        model=str(model),
        contexts=[524_288, 1_048_576],
    )
    assert receipt["model_context_window"] == 1_048_576
    assert receipt["requested_contexts"] == [524_288, 1_048_576]

    with pytest.raises(ValueError, match="1,048,577 tokens exceeds"):
        openai.apply_memory_caps_preflight(
            entry="bench.prefill_ladder",
            model=str(model),
            contexts=[1_048_577],
        )


def test_apply_metal_memory_caps_falls_back_to_deprecated_metal_apis(monkeypatch):
    mx, calls = _fake_mx(top_level=False)
    monkeypatch.setenv("MTPLX_MEMORY_LIMIT_BYTES", "32G")
    monkeypatch.setenv("MTPLX_WIRED_LIMIT_BYTES", "64G")

    result = openai._apply_metal_memory_caps(
        mx_module=mx,
        total_ram_bytes=128 * GiB,
    )

    assert result["applied"] is True
    assert result["wired_limit_clamped_to_memory_limit"] is True
    assert result["memory_limit_bytes"] == 32 * GiB
    assert result["wired_limit_bytes"] == 32 * GiB
    assert result["memory_limit_api"] == "mx.metal.set_memory_limit"
    assert result["wired_limit_api"] == "mx.metal.set_wired_limit"
    assert calls == [("metal_memory", 32 * GiB), ("metal_wired", 32 * GiB)]


def test_apply_metal_memory_caps_skips_when_ram_unknown_without_overrides(
    monkeypatch,
):
    mx, calls = _fake_mx(top_level=True)
    monkeypatch.delenv("MTPLX_MEMORY_LIMIT_BYTES", raising=False)
    monkeypatch.delenv("MTPLX_WIRED_LIMIT_BYTES", raising=False)

    result = openai._apply_metal_memory_caps(mx_module=mx, total_ram_bytes=0)

    assert result == {"applied": False, "reason": "ram_unknown"}
    assert calls == []


# Issue #400: the flat 16 GiB reserve + 6 GiB floor margin refused the
# Flash-Next Optimized Speed pack on 96 GB Macs by exactly its own margin
# (77.3 + 6 + 16 = 99.3 > 96) while the machine demonstrably serves it
# with ~16 GiB left unwired. Both terms scale below 128 GB now; 128 GB+
# behavior is unchanged.


def test_system_reserve_scales_below_128g_and_holds_above():
    assert openai._metal_system_reserve_bytes(256 * GiB) == 16 * GiB
    assert openai._metal_system_reserve_bytes(128 * GiB) == 16 * GiB
    assert openai._metal_system_reserve_bytes(96 * GiB) == 12 * GiB
    assert openai._metal_system_reserve_bytes(64 * GiB) == 8 * GiB
    assert openai._metal_system_reserve_bytes(32 * GiB) == 8 * GiB


def test_resident_floor_margin_scales_below_112g_and_holds_above():
    assert openai._resident_floor_margin_bytes(None) == 6 * GiB
    assert openai._resident_floor_margin_bytes(256 * GiB) == 6 * GiB
    assert openai._resident_floor_margin_bytes(128 * GiB) == 6 * GiB
    assert openai._resident_floor_margin_bytes(96 * GiB) == 3 * GiB
    assert openai._resident_floor_margin_bytes(64 * GiB) == 2 * GiB


def test_flash_next_optimized_speed_pack_admits_on_96g(monkeypatch):
    # The #400 receipt machine: 77.3 GiB of weight files on a 96 GB M2
    # Max. floor = weights + margin(96G) = 80.3; reserve(96G) = 12;
    # 92.3 <= 96 admits, and the wired cap rises to the floor.
    mx, calls = _fake_mx(top_level=True)
    monkeypatch.delenv("MTPLX_MEMORY_LIMIT_BYTES", raising=False)
    monkeypatch.delenv("MTPLX_WIRED_LIMIT_BYTES", raising=False)
    weights = int(77.3 * GiB)
    floor = weights + openai._resident_floor_margin_bytes(96 * GiB)

    result = openai._apply_metal_memory_caps(
        mx_module=mx,
        total_ram_bytes=96 * GiB,
        minimum_resident_bytes=floor,
    )

    assert result["applied"] is True
    assert result["wired_limit_bytes"] == floor
    assert result["minimum_resident_bytes"] == floor
    # The unwired remainder stays at/above the scaled system reserve.
    assert 96 * GiB - floor >= openai._metal_system_reserve_bytes(96 * GiB)


def test_oversized_pack_still_refuses_on_96g(monkeypatch):
    mx, calls = _fake_mx(top_level=True)
    monkeypatch.delenv("MTPLX_MEMORY_LIMIT_BYTES", raising=False)
    monkeypatch.delenv("MTPLX_WIRED_LIMIT_BYTES", raising=False)

    result = openai._apply_metal_memory_caps(
        mx_module=mx,
        total_ram_bytes=96 * GiB,
        minimum_resident_bytes=90 * GiB,
    )

    assert result["applied"] is False
    assert result["reason"] == "insufficient_ram"
    assert result["minimum_system_reserve_bytes"] == 12 * GiB
    assert calls == []


# A stock Mac's wired limit (iogpu.wired_limit_mb, about 75% of RAM) can sit
# under a family's wired floor. MLX refuses the larger request, and before
# the clamp that refusal left the model with no wiring and no keepalive.


def test_wired_floor_above_the_system_limit_is_clamped_not_dropped(monkeypatch):
    # Flash-Next Speed on a stock 96 GB Mac: floor 80.3 GiB, system 72 GiB.
    mx, calls = _fake_mx(top_level=True, system_wired=72 * GiB)
    monkeypatch.delenv("MTPLX_MEMORY_LIMIT_BYTES", raising=False)
    monkeypatch.delenv("MTPLX_WIRED_LIMIT_BYTES", raising=False)
    floor = int(77.3 * GiB) + openai._resident_floor_margin_bytes(96 * GiB)

    result = openai._apply_metal_memory_caps(
        mx_module=mx,
        total_ram_bytes=96 * GiB,
        minimum_resident_bytes=floor,
    )

    assert result["applied"] is True
    assert "wired_limit_error" not in result
    assert result["wired_limit_bytes"] == 72 * GiB
    assert result["wired_limit_requested_bytes"] == floor
    assert result["system_wired_limit_bytes"] == 72 * GiB
    assert calls == [("memory", 84 * GiB), ("wired", 72 * GiB)]


def test_keepalive_arms_at_the_clamped_wired_limit(monkeypatch):
    mx, _calls = _fake_mx(top_level=True, system_wired=72 * GiB)
    monkeypatch.delenv("MTPLX_MEMORY_LIMIT_BYTES", raising=False)
    monkeypatch.delenv("MTPLX_WIRED_LIMIT_BYTES", raising=False)
    monkeypatch.delenv("MTPLX_GPU_KEEPALIVE", raising=False)
    monkeypatch.setattr(openai, "_make_gpu_residency_touch", lambda: (lambda: None))
    armed = []
    state = SimpleNamespace(
        metal_memory_caps=openai._apply_metal_memory_caps(
            mx_module=mx,
            total_ram_bytes=96 * GiB,
            minimum_resident_bytes=int(80.3 * GiB),
        ),
        model_scheduler=SimpleNamespace(
            arm_idle_keepalive=lambda touch, **kw: armed.append(kw)
        ),
    )

    receipt = openai._arm_gpu_keepalive(state)

    assert receipt["enabled"] is True
    assert receipt["wired_limit_bytes"] == 72 * GiB
    assert len(armed) == 1


def test_wired_limit_under_the_system_limit_is_untouched(monkeypatch):
    # Flash-Next Speed on a stock 128 GB Mac: floor 83.3 GiB, system 96 GiB.
    mx, calls = _fake_mx(top_level=True, system_wired=96 * GiB)
    monkeypatch.delenv("MTPLX_MEMORY_LIMIT_BYTES", raising=False)
    monkeypatch.delenv("MTPLX_WIRED_LIMIT_BYTES", raising=False)
    floor = int(77.3 * GiB) + openai._resident_floor_margin_bytes(128 * GiB)

    result = openai._apply_metal_memory_caps(
        mx_module=mx,
        total_ram_bytes=128 * GiB,
        minimum_resident_bytes=floor,
    )

    assert result["wired_limit_bytes"] == floor
    assert "wired_limit_requested_bytes" not in result
    # The 128 GB desktop default (90 GiB, G5) still covers the 83.3 GiB floor.
    assert calls == [("memory", 90 * GiB), ("wired", floor)]
