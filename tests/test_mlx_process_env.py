"""The MLX command-buffer bound is in the environment before MLX can read it."""

from __future__ import annotations

import ast
from pathlib import Path

from mtplx import mlx_process_env as penv

ROOT = Path(__file__).resolve().parents[1]


def test_by_default_mlx_keeps_its_own_rule():
    """Lifting the rule costs 6 to 12 GB of prefill peak, so it is opt-in off M1."""

    env: dict[str, str] = {penv.M1_GATE_ENV: "0"}
    receipt = penv.apply_mlx_process_defaults(env)
    assert penv.MLX_ENV not in env
    assert penv.MLX_OPS_ENV not in env and penv.PREFILL_EVAL_ENV not in env
    assert receipt == {"value_mb": None, "source": "default:mlx_default"}


def test_m1_family_leaves_mlx_alone_by_default():
    """The raised MiB bound slows M1 prefill 2-21% (opt-s18), so it is opt-in there too."""

    env: dict[str, str] = {penv.M1_GATE_ENV: "1"}
    receipt = penv.apply_mlx_process_defaults(env)
    assert receipt == {"value_mb": None, "source": "default:mlx_default"}
    assert penv.MLX_ENV not in env
    assert penv.MLX_OPS_ENV not in env and penv.PREFILL_EVAL_ENV not in env


def test_m1_opt_in_pairs_the_buffer_bound_with_the_prefill_eval():
    env: dict[str, str] = {penv.M1_GATE_ENV: "1", penv.OVERRIDE_ENV: "1000"}
    receipt = penv.apply_mlx_process_defaults(env)
    assert receipt == {"value_mb": 1000, "source": "override:" + penv.OVERRIDE_ENV}
    assert env[penv.MLX_ENV] == "1000"
    assert env[penv.MLX_OPS_ENV] == "150"
    assert env[penv.PREFILL_EVAL_ENV] == "4"
    env = {penv.M1_GATE_ENV: "0", penv.OVERRIDE_ENV: "1000"}
    penv.apply_mlx_process_defaults(env)
    assert penv.MLX_OPS_ENV not in env and penv.PREFILL_EVAL_ENV not in env


def test_m1_opt_in_keeps_operator_values():
    env = {penv.M1_GATE_ENV: "1", penv.OVERRIDE_ENV: "1000", penv.MLX_OPS_ENV: "40", penv.PREFILL_EVAL_ENV: "0"}
    penv.apply_mlx_process_defaults(env)
    assert env[penv.MLX_OPS_ENV] == "40" and env[penv.PREFILL_EVAL_ENV] == "0"
    env = {penv.M1_GATE_ENV: "1", penv.MLX_ENV: "64"}
    receipt = penv.apply_mlx_process_defaults(env)
    assert receipt["source"] == "operator:" + penv.MLX_ENV
    assert penv.MLX_OPS_ENV not in env


def test_an_operator_value_for_mlx_always_wins():
    env = {penv.MLX_ENV: "64", penv.OVERRIDE_ENV: "2048"}
    receipt = penv.apply_mlx_process_defaults(env)
    assert env[penv.MLX_ENV] == "64"
    assert receipt == {"value_mb": 64, "source": "operator:" + penv.MLX_ENV}


def test_the_override_sets_a_number_or_leaves_mlx_alone():
    env = {penv.M1_GATE_ENV: "0", penv.OVERRIDE_ENV: "400"}
    assert penv.apply_mlx_process_defaults(env)["value_mb"] == 400
    assert env[penv.MLX_ENV] == "400"
    for spelling in ("0", "off", "mlx", "default", "-5"):
        env = {penv.OVERRIDE_ENV: spelling}
        receipt = penv.apply_mlx_process_defaults(env)
        assert receipt["value_mb"] is None
        assert penv.MLX_ENV not in env


def test_a_broken_override_falls_back_to_the_default_and_says_so():
    env = {penv.OVERRIDE_ENV: "lots"}
    receipt = penv.apply_mlx_process_defaults(env)
    assert penv.MLX_ENV not in env
    assert receipt["source"] == "default:unparsed_override"


def test_applying_twice_changes_nothing():
    env: dict[str, str] = {penv.OVERRIDE_ENV: "1024"}
    penv.apply_mlx_process_defaults(env)
    first = dict(env)
    penv.apply_mlx_process_defaults(env)
    assert env == first


def test_the_module_cannot_wake_mlx_and_runs_first_in_the_package():
    tree = ast.parse((ROOT / "mtplx" / "mlx_process_env.py").read_text("utf-8"))
    imported = {
        (node.module or "") if isinstance(node, ast.ImportFrom) else alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert not any(name.split(".")[0] == "mlx" for name in imported)
    init = (ROOT / "mtplx" / "__init__.py").read_text("utf-8")
    assert init.index("apply_mlx_process_defaults()") < init.index("def __getattr__")
    body = ast.parse(init).body
    first_import = next(n for n in body if isinstance(n, ast.ImportFrom) and n.level == 1)
    assert first_import.module == "mlx_process_env"
