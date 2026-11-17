"""`mtplx serve --ngram-prewarm`: CLI/env precedence and the /health field.

CPU-only.  ``mtplx.cli`` imports no MLX, so its parser is exercised for real;
``mtplx/server/openai.py`` and ``mtplx/commands/public.py`` do, so the two
functions under test there are compiled out of the shipped source instead of
imported -- the same trick the PLE lookahead tests use.
"""

from __future__ import annotations

import argparse
import ast
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from mtplx import ple_row_gather as row_gather


ROOT = Path(__file__).resolve().parents[1]
SERVER_TEXT = (ROOT / "mtplx" / "server" / "openai.py").read_text("utf-8")
PUBLIC_TEXT = (ROOT / "mtplx" / "commands" / "public.py").read_text("utf-8")
CLI_TEXT = (ROOT / "mtplx" / "cli.py").read_text("utf-8")


def _compile_function(source: str, name: str, namespace: dict | None = None):
    """One top-level function out of a module that imports MLX."""

    node = next(
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.FunctionDef) and n.name == name
    )
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    scope: dict = {"Any": Any, "os": os, "argparse": argparse}
    scope.update(namespace or {})
    exec(compile(module, f"<{name}>", "exec"), scope)
    return scope[name]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(row_gather.PREWARM_ENV, raising=False)
    monkeypatch.delenv(row_gather.PREWARM_AT_LOAD_ENV, raising=False)
    row_gather._LAST_PREWARM.clear()
    yield
    row_gather._LAST_PREWARM.clear()


# --------------------------------------------------------------------------
# The CLI surface
# --------------------------------------------------------------------------


@pytest.mark.parametrize("command", ["serve", "start"])
def test_both_flows_accept_the_flag_and_default_to_not_given(command):
    from mtplx.cli import build_parser

    parser = build_parser()
    assert parser.parse_args([command]).ngram_prewarm is None
    for value in ("auto", "all", "off", "12", "12GiB"):
        parsed = parser.parse_args([command, "--ngram-prewarm", value])
        assert parsed.ngram_prewarm == value
    assert parser.parse_args([command, "--no-ngram-prewarm"]).ngram_prewarm == "off"
    order = parser.parse_args([command, "--ngram-prewarm-order", "/tmp/h.npy"])
    assert order.ngram_prewarm_order == "/tmp/h.npy"


def test_the_flag_is_declared_once_for_both_flows():
    """Two copies is how one of them silently drifts."""

    assert CLI_TEXT.count('"--ngram-prewarm",') == 1
    assert "_add_ngram_prewarm_args(serve_p)" in CLI_TEXT
    assert "_add_ngram_prewarm_args(start_flow_p)" in CLI_TEXT


def test_the_flag_is_forwarded_to_the_server_child():
    """`mtplx serve` rebuilds the child argv explicitly: unforwarded is unheard."""

    assert 'cmd.extend(["--ngram-prewarm", str(ngram_prewarm)])' in PUBLIC_TEXT
    assert 'cmd.extend(["--ngram-prewarm-order", str(ngram_prewarm_order)])' in (
        PUBLIC_TEXT
    )
    # ...and quickstart builds its serve namespace field by field, so the
    # attribute has to be in the forwarding table too.
    policy = PUBLIC_TEXT.split("def _with_server_policy_args", 1)[1]
    policy = policy.split("\ndef ", 1)[0]
    assert '("ngram_prewarm", None),' in policy
    assert '("ngram_prewarm_order", None),' in policy


# --------------------------------------------------------------------------
# Precedence: CLI > env > deprecated alias > default(on)
# --------------------------------------------------------------------------


def _apply(args_value, monkeypatch=None):
    apply = _compile_function(SERVER_TEXT, "_apply_ngram_prewarm_choice")
    return apply(SimpleNamespace(ngram_prewarm=args_value))


def test_no_flag_and_no_env_is_auto_by_default():
    assert _apply(None) == "default"
    assert row_gather.prewarm_mode_setting() == ("auto", "default")
    assert row_gather.PREWARM_ENV not in os.environ


def test_env_alone_decides_when_the_flag_is_absent(monkeypatch):
    monkeypatch.setenv(row_gather.PREWARM_ENV, "off")
    assert _apply(None) == "env"
    assert row_gather.prewarm_at_load_enabled() is False


@pytest.mark.parametrize(
    "env_value, cli_value, expected_mode",
    [
        ("off", "all", "all"),
        ("all", "off", "off"),
        (None, "auto", "auto"),
        ("all", "12", ("bytes", 12 * 1024**3)),
        ("off", True, "all"),
        ("all", False, "off"),
    ],
)
def test_the_cli_flag_wins_over_the_environment(
    monkeypatch, env_value, cli_value, expected_mode
):
    if env_value is not None:
        monkeypatch.setenv(row_gather.PREWARM_ENV, env_value)
    assert _apply(cli_value) == "cli"
    assert row_gather.prewarm_mode_setting() == (expected_mode, "env")


def test_the_deprecated_alias_is_honoured_below_the_official_key(monkeypatch):
    monkeypatch.setenv(row_gather.PREWARM_AT_LOAD_ENV, "0")
    monkeypatch.setattr(row_gather, "_DEPRECATION_WARNED", False)
    assert _apply(None) == "deprecated_env"
    assert row_gather.prewarm_mode_setting() == ("off", "deprecated_env")
    # An explicit flag still overrules it.
    assert _apply("all") == "cli"
    assert row_gather.prewarm_mode_setting() == ("all", "env")


def test_a_broken_knob_never_breaks_startup(monkeypatch):
    monkeypatch.setenv(row_gather.PREWARM_ENV, "sometimes")
    source = _apply(None)
    assert source.startswith("unavailable: ")


def test_the_server_stamps_the_choice_before_the_model_loads():
    """After apply_profile_env, before the load that performs the read."""

    state = SERVER_TEXT.split("self.mlx_cache_limit_status = _configure_mlx", 1)[1]
    state = state.split("self.runtime = self.model_scheduler.submit_foreground", 1)[0]
    assert "self.ngram_prewarm_source = _apply_ngram_prewarm_choice(args)" in state
    order = SERVER_TEXT.index("self.ngram_prewarm_source = ")
    assert SERVER_TEXT.index("            apply_profile_env(") < order


# --------------------------------------------------------------------------
# /health
# --------------------------------------------------------------------------


def _payload():
    return _compile_function(SERVER_TEXT, "_ngram_prewarm_health_payload")()


def test_health_reports_the_pre_read_before_any_model_is_loaded():
    payload = _payload()
    assert payload["enabled"] is True
    assert payload["warmed_bytes"] == 0
    assert payload["gib_per_s"] is None
    assert payload["skipped_reason"] == "no_model_loaded"
    assert payload["source"] == "default"


def test_health_reports_a_completed_pre_read(tmp_path):
    table = tmp_path / "ngram-table.safetensors"
    table.write_bytes(os.urandom(512 * 1024))
    row_gather.record_prewarm(
        row_gather.prewarm_file(table, chunk_bytes=64 * 1024),
        enabled=True,
        source="cli",
    )
    payload = _payload()
    assert set(payload) == {
        "enabled",
        "mode",
        "order",
        "table_bytes",
        "budget_bytes",
        "warmed_bytes",
        "seconds",
        "gib_per_s",
        "free_bytes",
        "reserved_bytes",
        "margin_bytes",
        "source",
        "skipped_reason",
    }
    assert payload["enabled"] is True
    assert payload["warmed_bytes"] == table.stat().st_size
    assert payload["source"] == "cli"
    assert payload["skipped_reason"] is None
    assert payload["seconds"] > 0
    assert payload["gib_per_s"] > 0


def test_health_names_the_reason_when_the_pre_read_was_skipped():
    row_gather.record_prewarm(
        row_gather.prewarm_skipped("disabled"), enabled=False, source="env"
    )
    payload = _payload()
    assert payload["enabled"] is False
    assert payload["skipped_reason"] == "disabled"
    assert payload["warmed_bytes"] == 0


def test_health_exposes_the_field_and_reads_it_without_walking_the_model():
    assert '"ngram_prewarm": _ngram_prewarm_health_payload(),' in SERVER_TEXT
    body = SERVER_TEXT.split("def _ngram_prewarm_health_payload", 1)[1]
    body = body.split("\ndef ", 1)[0]
    assert "from mtplx.ple_row_gather import last_prewarm" in body
    assert "_ple_stage_idx" not in body
    assert "ngram_embedding" not in body


def test_the_startup_line_is_emitted_once_from_the_load_path():
    row_text = (ROOT / "mtplx" / "ple_row_gather.py").read_text("utf-8")
    assert row_text.count('"[mtplx] n-gram table pre-read "') == 1
    model_text = (ROOT / "mtplx" / "models" / "qwen4_exp.py").read_text("utf-8")
    assert "format_prewarm_plan(receipt)" in model_text
    assert "format_prewarm_result(receipt)" in model_text
    # The server does not print a second copy: the load path is the only
    # place that knows the numbers, and it runs in-process during startup.
    assert '_startup_line("[mtplx] n-gram table' not in SERVER_TEXT
    assert "n-gram table pre-read " not in SERVER_TEXT.replace(
        "n-gram table pre-read actually did", ""
    )


# --------------------------------------------------------------------------
# Budget arithmetic
# --------------------------------------------------------------------------

GIB = 1024**3


@pytest.mark.parametrize(
    "text, expected",
    [
        ("auto", "auto"),
        ("all", "all"),
        ("off", "off"),
        ("true", "all"),
        ("no", "off"),
        ("0", "off"),
        ("12", ("bytes", 12 * GIB)),
        ("12GiB", ("bytes", 12 * GIB)),
        ("12gb", ("bytes", 12 * GIB)),
        ("0.5g", ("bytes", GIB // 2)),
        ("", None),
        (None, None),
    ],
)
def test_the_mode_grammar(text, expected):
    assert row_gather.parse_prewarm_mode(text) == expected


@pytest.mark.parametrize("bad", ["sometimes", "-4", "12mb", "auto-ish"])
def test_a_bad_mode_is_named_not_guessed(bad):
    with pytest.raises(ValueError) as excinfo:
        row_gather.parse_prewarm_mode(bad)
    assert "MTPLX_NGRAM_PREWARM" in str(excinfo.value)


def test_a_bare_one_is_one_gibibyte_not_on():
    """The number grammar has to be consistent to be usable."""

    assert row_gather.parse_prewarm_mode("1") == ("bytes", GIB)
    # ...while the deprecated boolean-only alias keeps its boolean reading.
    assert row_gather._parse_bool_env.__doc__


def _budget(mode, table=32, free=60, reserved=8):
    return row_gather.resolve_budget(
        mode,
        table_bytes=table * GIB,
        free_bytes=free * GIB,
        reserved_bytes=reserved * GIB,
    )


def test_off_warms_nothing_and_all_warms_the_table():
    assert _budget("off")["budget_bytes"] == 0
    assert _budget("all")["budget_bytes"] == 32 * GIB
    # Even when the machine plainly cannot hold it: "all" is an instruction.
    assert _budget("all", free=4)["budget_bytes"] == 32 * GIB


def test_an_explicit_budget_is_clamped_to_the_table():
    assert _budget(("bytes", 8 * GIB))["budget_bytes"] == 8 * GIB
    plan = _budget(("bytes", 99 * GIB))
    assert plan["budget_bytes"] == 32 * GIB
    assert plan["requested_bytes"] == 99 * GIB


def test_auto_is_free_minus_reservation_minus_margin():
    plan = _budget("auto", free=60, reserved=8)
    assert plan["margin_bytes"] == row_gather.AUTO_MARGIN_BYTES
    assert plan["headroom_bytes"] == (60 - 8) * GIB - row_gather.AUTO_MARGIN_BYTES
    assert plan["budget_bytes"] == 32 * GIB  # the whole table fits


def test_auto_warms_only_what_fits():
    plan = _budget("auto", table=32, free=30, reserved=8)
    assert plan["budget_bytes"] == (30 - 8) * GIB - row_gather.AUTO_MARGIN_BYTES
    assert plan["budget_bytes"] < 32 * GIB


def test_auto_declines_rather_than_going_negative():
    """The 128 GB case: 85 GB wired weights leave no room for a 32 GB table."""

    plan = _budget("auto", table=32, free=10, reserved=8)
    assert plan["headroom_bytes"] < 0
    assert plan["budget_bytes"] == 0


def test_free_memory_is_measured_and_says_how():
    total, source = row_gather.free_memory_bytes()
    assert total > 0
    assert source == "vm_stat(free+inactive+purgeable)"


def test_a_missing_context_window_reserves_nothing_rather_than_guessing():
    assert row_gather.estimate_kv_reservation_bytes("/nonexistent", 0) == (
        0,
        "no_context_window",
    )
    reserved, reason = row_gather.estimate_kv_reservation_bytes("/nonexistent", 8192)
    assert reserved == 0
    assert reason.startswith("unavailable: ")


def test_the_reservation_the_server_publishes_is_what_auto_subtracts():
    row_gather.set_prewarm_reservation(7 * GIB, "test")
    assert row_gather.prewarm_reservation() == (7 * GIB, "test")
    row_gather.set_prewarm_reservation(0, "unset")


def test_the_server_publishes_the_reservation_before_the_load():
    body = SERVER_TEXT.split("def _publish_ngram_prewarm_reservation", 1)[1]
    body = body.split("\ndef ", 1)[0]
    assert "estimate_kv_reservation_bytes" in body
    assert "set_prewarm_reservation(reserved, source)" in body
    order = SERVER_TEXT.index("self.ngram_prewarm_reservation = ")
    assert order < SERVER_TEXT.index("self.runtime = self.model_scheduler")


# --------------------------------------------------------------------------
# Ordering
# --------------------------------------------------------------------------


def _table(tmp_path, size=8 << 20):
    table = tmp_path / "ngram-table.safetensors"
    table.write_bytes(os.urandom(size))
    return table


ROW_META = ((0, 80), (4 << 20, 10), (6 << 20, 10))


def test_a_partial_budget_without_a_hotness_file_reads_the_prefix(tmp_path):
    import numpy as np  # noqa: F401  (kept symmetric with the hotness case)

    table = _table(tmp_path)
    os.environ[row_gather.PREWARM_ENV] = "0.002"
    fd = os.open(str(table), os.O_RDONLY)
    try:
        receipt = row_gather.run_prewarm(
            table_path=table, row_meta=ROW_META, fd=fd
        )
    finally:
        os.close(fd)
    assert receipt["order"] == "prefix"
    assert receipt["warmed_bytes"] == receipt["budget_bytes"]
    assert receipt["skipped_reason"] is None


def test_a_partial_budget_with_a_hotness_file_reads_the_hot_rows(tmp_path):
    import numpy as np

    table = _table(tmp_path)
    np.save(tmp_path / row_gather.HOTNESS_FILENAME, np.arange(0, 40_000, 7, np.int64))
    os.environ[row_gather.PREWARM_ENV] = "0.002"
    fd = os.open(str(table), os.O_RDONLY)
    try:
        receipt = row_gather.run_prewarm(
            table_path=table, row_meta=ROW_META, fd=fd
        )
    finally:
        os.close(fd)
    assert receipt["order"] == "hotness"
    assert receipt["hot_rows"] > 0
    assert 0 < receipt["warmed_bytes"] <= receipt["budget_bytes"]
    # The search must actually fill the budget, not leave most of it unspent.
    assert receipt["warmed_bytes"] > 0.9 * receipt["budget_bytes"]


def test_a_full_budget_ignores_the_hotness_file(tmp_path):
    """Nothing to prioritise, and sequential beats the same pages at random."""

    import numpy as np

    table = _table(tmp_path)
    np.save(tmp_path / row_gather.HOTNESS_FILENAME, np.arange(0, 40_000, 7, np.int64))
    os.environ[row_gather.PREWARM_ENV] = "all"
    fd = os.open(str(table), os.O_RDONLY)
    try:
        receipt = row_gather.run_prewarm(
            table_path=table, row_meta=ROW_META, fd=fd
        )
    finally:
        os.close(fd)
    assert receipt["order"] == "prefix"
    assert receipt["warmed_bytes"] == table.stat().st_size


def test_an_explicit_order_path_overrides_the_model_directory(tmp_path):
    import numpy as np

    table = _table(tmp_path)
    elsewhere = tmp_path / "elsewhere.npy"
    np.save(elsewhere, np.arange(0, 20_000, 3, np.int64))
    assert row_gather.hotness_path_for(table) is None
    assert row_gather.hotness_path_for(table, elsewhere) == elsewhere
    # A path that does not exist is not an error, just no ordering.
    assert row_gather.hotness_path_for(table, tmp_path / "nope.npy") is None


def test_an_unreadable_hotness_file_is_ignored_not_fatal(tmp_path):
    broken = tmp_path / "broken.npy"
    broken.write_bytes(b"not a numpy file")
    assert row_gather.load_hotness_order(broken) is None
    assert row_gather.load_hotness_order(None) is None


def test_hot_runs_are_coalesced_and_page_aligned():
    import numpy as np

    page = row_gather.PAGE_SIZE
    rows = np.arange(64, dtype=np.int64)  # adjacent rows: one page, one run
    runs, taken = row_gather.plan_hot_runs(rows, ((0, 80),), 10 * page)
    assert taken == 64
    assert len(runs) == 1
    offset, length = runs[0]
    assert offset % page == 0 and length % page == 0


def test_hot_runs_return_nothing_when_even_one_row_will_not_fit():
    import numpy as np

    runs, taken = row_gather.plan_hot_runs(np.array([5], np.int64), ((0, 80),), 16)
    assert (runs, taken) == ([], 0)
    assert row_gather.plan_hot_runs(np.array([], np.int64), ((0, 80),), 1 << 20) == (
        [],
        0,
    )


def test_the_order_env_reaches_the_pre_read():
    body = (ROOT / "mtplx" / "ple_row_gather.py").read_text("utf-8")
    assert 'ORDER_ENV = "MTPLX_NGRAM_PREWARM_ORDER"' in body
    assert "os.environ.get(ORDER_ENV)" in body
    assert 'os.environ["MTPLX_NGRAM_PREWARM_ORDER"] = str(order)' in SERVER_TEXT


# --------------------------------------------------------------------------
# Smallest maps first (2026-09-20)
# --------------------------------------------------------------------------

# The shipped geometry in miniature: weight, scales, biases, in file order.
MAP_EXTENTS = ((0, 4 << 20), (4 << 20, 2 << 20), (6 << 20, 2 << 20))


def test_small_maps_are_planned_before_the_large_one():
    page = row_gather.PAGE_SIZE
    runs, complete = row_gather.plan_small_maps_first(
        MAP_EXTENTS, 5 << 20, chunk_bytes=1 << 20
    )
    assert complete == 2
    covered = sorted(runs)
    # Both 2 MiB maps in full, then the first 1 MiB of the weight map.
    assert sum(length for _, length in runs) == 5 << 20
    assert all(offset % page == 0 and length % page == 0 for offset, length in runs)
    small = [run for run in covered if run[0] >= 4 << 20]
    assert sum(length for _, length in small) == 4 << 20
    large = [run for run in covered if run[0] < 4 << 20]
    assert large == [(0, 1 << 20)]
    # Cut for the pread pool: no run is larger than the chunk.
    assert max(length for _, length in runs) <= 1 << 20


def test_small_maps_plan_never_exceeds_the_budget_and_handles_nothing():
    runs, complete = row_gather.plan_small_maps_first(MAP_EXTENTS, 3 << 20)
    assert complete == 1
    assert sum(length for _, length in runs) <= 3 << 20
    assert row_gather.plan_small_maps_first(MAP_EXTENTS, 0) == ([], 0)
    assert row_gather.plan_small_maps_first((), 1 << 20) == ([], 0)


def test_a_partial_budget_with_map_extents_reads_the_small_maps_first(tmp_path):
    table = _table(tmp_path)
    os.environ[row_gather.PREWARM_ENV] = "0.0045"  # 4.6 MiB of an 8 MiB table
    fd = os.open(str(table), os.O_RDONLY)
    try:
        receipt = row_gather.run_prewarm(
            table_path=table, row_meta=ROW_META, fd=fd, map_extents=MAP_EXTENTS
        )
    finally:
        os.close(fd)
    assert receipt["order"] == "small_maps_first"
    assert receipt["maps_complete"] == 2
    assert 4 << 20 <= receipt["warmed_bytes"] <= receipt["budget_bytes"]
    assert receipt["skipped_reason"] is None


def test_a_full_budget_with_map_extents_still_reads_sequentially(tmp_path):
    table = _table(tmp_path)
    os.environ[row_gather.PREWARM_ENV] = "all"
    fd = os.open(str(table), os.O_RDONLY)
    try:
        receipt = row_gather.run_prewarm(
            table_path=table, row_meta=ROW_META, fd=fd, map_extents=MAP_EXTENTS
        )
    finally:
        os.close(fd)
    assert receipt["order"] == "prefix"
    assert receipt["warmed_bytes"] == table.stat().st_size


def test_a_hotness_file_still_outranks_the_map_order(tmp_path):
    import numpy as np

    table = _table(tmp_path)
    np.save(tmp_path / row_gather.HOTNESS_FILENAME, np.arange(0, 40_000, 7, np.int64))
    os.environ[row_gather.PREWARM_ENV] = "0.002"
    fd = os.open(str(table), os.O_RDONLY)
    try:
        receipt = row_gather.run_prewarm(
            table_path=table, row_meta=ROW_META, fd=fd, map_extents=MAP_EXTENTS
        )
    finally:
        os.close(fd)
    assert receipt["order"] == "hotness"


def test_the_sidecar_names_its_map_extents_to_the_pre_read():
    body = (ROOT / "mtplx" / "models" / "qwen4_exp.py").read_text("utf-8")
    call = body[body.index("receipt = run_prewarm(") :]
    call = call[: call.index("\n        )\n") + 1]
    assert "map_extents=tuple(" in call
    assert 'self.row_layout == "planar"' in call


def test_advise_rows_issues_one_advisory_a_row_per_map(tmp_path, monkeypatch):
    """ple_row_gather.advise_rows hands the kernel one 16-byte radvisory record
    (offset, count) per row and map; on macOS the real call succeeds on a plain file."""
    import fcntl
    import struct
    import sys

    import numpy as np

    from mtplx import ple_row_gather as PG

    calls = []
    monkeypatch.setattr(fcntl, "fcntl", lambda fd, cmd, arg: calls.append((fd, cmd, bytes(arg))) or arg)
    n = PG.advise_rows(7, [(192, 90), (1000, 16)], np.array([3, 10, 4_000_000], dtype=np.int64))
    assert n == 6 and len(calls) == 6
    assert all(c[0] == 7 and c[1] == PG.F_RDADVISE == 44 and len(c[2]) == 16 for c in calls)
    assert [struct.unpack("<qi", c[2][:12]) for c in calls[:3]] == [(192 + 3 * 90, 90), (192 + 10 * 90, 90), (192 + 4_000_000 * 90, 90)]
    assert [struct.unpack("<qi", c[2][:12]) for c in calls[3:]] == [(1000 + 3 * 16, 16), (1000 + 10 * 16, 16), (1000 + 4_000_000 * 16, 16)]
    monkeypatch.undo()
    if sys.platform == "darwin":
        path = tmp_path / "table.bin"
        path.write_bytes(b"\0" * (1 << 20))
        fd = __import__("os").open(str(path), __import__("os").O_RDONLY)
        try:
            assert PG.advise_rows(fd, [(0, 90)], np.array([0, 5, 9000])) == 3
        finally:
            __import__("os").close(fd)


def test_advise_switch(monkeypatch):
    import sys

    from mtplx import ple_row_gather as PG

    monkeypatch.delenv(PG.ADVISE_ENV, raising=False)
    assert PG.advise_enabled() == (sys.platform == "darwin")
    monkeypatch.setenv(PG.ADVISE_ENV, "0")
    assert not PG.advise_enabled()
