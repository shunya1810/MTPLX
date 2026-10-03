"""The memory limit default on 128 GB desktops (founder decision G5, 09-29)
and ``--memory-limit`` with its ``max`` choice for headless servers (#548).

09-27 receipts on the final guard, 128 GB M5 Max with 16 GB of other apps:
at the 96 GiB limit (the 75% rule) a compaction-size request was refused with
a 507 to keep the Mac alive; at 90 GiB the same request was served. From
128 GB up the default therefore leaves 38 GiB outside the engine.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from mtplx.memory_plan import GIB, max_engine_bytes, plan_memory, usable_engine_bytes
from mtplx.server import openai


@pytest.fixture(autouse=True)
def _restore_limit_env(monkeypatch):
    # The code under test writes MTPLX_MEMORY_LIMIT_BYTES; setenv records the
    # original state so teardown restores it (delenv on an absent variable
    # records nothing and would let the write leak into later tests).
    for name in ("MTPLX_MEMORY_LIMIT_BYTES", "MTPLX_WIRED_LIMIT_BYTES"):
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)


def _fake_mx():
    calls: list[tuple[str, int]] = []
    mx = SimpleNamespace(
        metal=SimpleNamespace(is_available=lambda: True),
        set_memory_limit=lambda value: calls.append(("memory", int(value))),
        set_wired_limit=lambda value: calls.append(("wired", int(value))),
        device_info=lambda: {},
    )
    return mx, calls


@pytest.mark.parametrize(
    "ram_gib, limit_gib",
    [
        (36, 27),  # 75% rule, unchanged
        (64, 48),  # unchanged
        (96, 72),  # unchanged
        (128, 90),  # G5: 96 -> 90 on the 128 GB desktop
        (192, 144),  # the 75% rule already leaves 48 GiB
        (512, 192),  # the 192 GiB cap
    ],
)
def test_default_limit_leaves_a_desktop_room_from_128_gb(ram_gib, limit_gib):
    assert usable_engine_bytes(ram_gib * GIB) == limit_gib * GIB


def test_metal_caps_and_planner_read_the_same_default(monkeypatch):
    monkeypatch.delenv("MTPLX_MEMORY_LIMIT_BYTES", raising=False)
    monkeypatch.delenv("MTPLX_WIRED_LIMIT_BYTES", raising=False)
    mx, calls = _fake_mx()

    caps = openai._apply_metal_memory_caps(mx_module=mx, total_ram_bytes=128 * GIB)
    plan = plan_memory(
        total_ram_bytes=128 * GIB,
        model_weights_bytes=20 * GIB,
        kv_bytes_per_token=65536,
        model_max_context=262144,
        usable_bytes_override=caps["memory_limit_bytes"],
    )

    assert caps["memory_limit_bytes"] == 90 * GIB
    assert caps["memory_limit_source"] == "default"
    assert plan.usable_bytes == 90 * GIB


@pytest.mark.parametrize(
    "ram_gib, max_gib",
    [(128, 112), (96, 84), (64, 56), (256, 240)],
)
def test_max_is_everything_outside_the_system_reserve(ram_gib, max_gib):
    assert max_engine_bytes(ram_gib * GIB) == max_gib * GIB


@pytest.mark.parametrize("ram_gib, max_gib", [(8, 8), (16, 12), (24, 18), (32, 24)])
def test_max_is_never_under_the_default_on_small_macs(ram_gib, max_gib):
    # The review of 4c9da1ba: 'max' read 8/12/18 GiB on 8/16/24 GB Macs
    # against a documented 8 GiB reserve. Those Macs' default (75% of RAM,
    # at least 8 GiB) already reaches past the reserve, and 'max' asks for
    # more than the default, never less: it is the default there.
    total = ram_gib * GIB
    assert max_engine_bytes(total) == usable_engine_bytes(total) == max_gib * GIB


@pytest.mark.parametrize(
    "floor_gib, limit_gib, source",
    [
        (83, 90, "default"),  # Flash-Next Optimized Speed: 77.3 GiB + 6
        (92, 96, "resident_floor"),  # 86 GiB of weights + 6: the 75% rule, as before
        (96, 96, "resident_floor"),
        (100, 112, "resident_floor"),  # above the 75% rule: unchanged since #400
    ],
)
def test_a_model_larger_than_the_flagship_keeps_the_limit_it_had(
    monkeypatch, floor_gib, limit_gib, source
):
    # The review of 4c9da1ba: lowering the 128 GB default to 90 GiB turned a
    # 92 GiB resident floor into the whole envelope outside the system
    # reserve (112 GiB, source resident_floor), where 50de43bb gave it 96.
    # The desktop headroom now gives way only back to the 75% rule.
    monkeypatch.delenv("MTPLX_MEMORY_LIMIT_BYTES", raising=False)
    monkeypatch.delenv("MTPLX_WIRED_LIMIT_BYTES", raising=False)
    mx, _calls = _fake_mx()

    caps = openai._apply_metal_memory_caps(
        mx_module=mx,
        total_ram_bytes=128 * GIB,
        minimum_resident_bytes=floor_gib * GIB,
    )
    plan = plan_memory(
        total_ram_bytes=128 * GIB,
        model_weights_bytes=(floor_gib - 6) * GIB,
        kv_bytes_per_token=65536,
        model_max_context=262144,
        resident_floor_bytes=floor_gib * GIB,
    )

    assert caps["memory_limit_bytes"] == limit_gib * GIB
    assert caps["memory_limit_source"] == source
    assert plan.usable_bytes == limit_gib * GIB


def test_memory_limit_max_sets_the_engine_limit(monkeypatch):
    monkeypatch.delenv("MTPLX_MEMORY_LIMIT_BYTES", raising=False)
    monkeypatch.delenv("MTPLX_WIRED_LIMIT_BYTES", raising=False)
    monkeypatch.setattr(openai, "_total_ram_bytes", lambda: 128 * GIB)
    mx, calls = _fake_mx()

    openai.apply_memory_limit_setting("max")
    caps = openai._apply_metal_memory_caps(mx_module=mx, total_ram_bytes=128 * GIB)

    assert openai.os.environ["MTPLX_MEMORY_LIMIT_BYTES"] == str(112 * GIB)
    assert caps["memory_limit_bytes"] == 112 * GIB
    assert caps["memory_limit_source"] == "env"
    # Every later reader sees plain bytes.
    assert openai._explicit_memory_limit_bytes() == 112 * GIB


def test_env_max_from_a_launcher_resolves_the_same_way(monkeypatch):
    # The app and other launchers set the variable, not the flag.
    monkeypatch.setenv("MTPLX_MEMORY_LIMIT_BYTES", "max")
    monkeypatch.delenv("MTPLX_WIRED_LIMIT_BYTES", raising=False)
    monkeypatch.setattr(openai, "_total_ram_bytes", lambda: 128 * GIB)
    mx, _calls = _fake_mx()

    caps = openai._apply_metal_memory_caps(mx_module=mx, total_ram_bytes=128 * GIB)

    assert caps["memory_limit_bytes"] == 112 * GIB


def test_memory_limit_size_and_refusal(monkeypatch):
    monkeypatch.delenv("MTPLX_MEMORY_LIMIT_BYTES", raising=False)

    openai.apply_memory_limit_setting("96G")
    assert openai.os.environ["MTPLX_MEMORY_LIMIT_BYTES"] == str(96 * GIB)

    with pytest.raises(ValueError):
        openai.apply_memory_limit_setting("lots")


def test_flag_parses_on_the_module_and_the_public_serve():
    from mtplx.cli import build_parser

    assert openai.parse_args(["--model", "m"]).memory_limit is None
    assert openai.parse_args(["--model", "m", "--memory-limit", "max"]).memory_limit == "max"
    parser = build_parser()
    assert parser.parse_args(["serve"]).memory_limit is None
    assert parser.parse_args(["serve", "--memory-limit", "90G"]).memory_limit == "90G"


def test_public_serve_forwards_memory_limit(monkeypatch):
    from mtplx.commands import public

    calls: dict = {}
    monkeypatch.setattr(
        public, "_resolve_runtime_model_path", lambda model, cache_dir=None: (model, None)
    )
    monkeypatch.setattr(
        public,
        "_model_gate",
        lambda model, unsafe_force_unverified=False, yes=False: (
            {"compatibility": {"tier": "verified", "can_run": True, "exit_code": 0}},
            None,
        ),
    )
    monkeypatch.setattr(public, "_port_is_busy", lambda host, port: False)

    def fake_execvpe(_executable, cmd, _env):
        calls["cmd"] = cmd
        raise SystemExit(0)

    monkeypatch.setattr(public.os, "execvpe", fake_execvpe)

    def run(memory_limit):
        args = SimpleNamespace(
            command="serve",
            model="models/example",
            model_id="mtplx-example",
            cache_dir=None,
            profile="sustained",
            unsafe_force_unverified=False,
            yes=True,
            host="127.0.0.1",
            port=8000,
            depth=3,
            no_mtp=False,
            stock_ar=False,
            api_key="mtplx-local",
            rate_limit=0,
            stream_interval=1,
            context_window=None,
            allow_swap=False,
            memory_limit=memory_limit,
            max_response_tokens=None,
            temperature=0.6,
            top_p=0.95,
            reasoning_parser="qwen3",
            stats_footer=False,
            warmup_tokens=0,
            strict_warmup=False,
            strict_fast_path=False,
            quickstart_pi=False,
            max=False,
            _cli_flags=set(),
        )
        with pytest.raises(SystemExit):
            public.cmd_serve_public(args)
        return list(calls["cmd"])

    with_flag = run("max")
    assert with_flag[with_flag.index("--memory-limit") + 1] == "max"
    assert "--memory-limit" not in run(None)
