from __future__ import annotations

import argparse
import os
import subprocess
import sys

import pytest

from mtplx.config import apply_user_config, load_user_config
from mtplx.constants import DEFAULT_RUNTIME_MODEL_DIR
from mtplx.profiles import DEFAULT_HF_MODEL_ID, DEFAULT_PROFILE_NAME


def test_load_user_config_reads_runtime_defaults(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text(
        'model = "mtplx/example"\n'
        f'model_dir = "{tmp_path / "models"}"\n'
        f'model_dirs = ["{tmp_path / "archive"}", "{tmp_path / "external"}"]\n'
        'profile = "exact"\n'
        'thermal_control = "none"\n',
        encoding="utf-8",
    )

    loaded = load_user_config(config)

    assert loaded.exists is True
    assert loaded.model == "mtplx/example"
    assert loaded.model_dir == str(tmp_path / "models")
    assert loaded.model_dirs == (
        str(tmp_path / "archive"),
        str(tmp_path / "external"),
    )
    assert loaded.profile == "exact"
    assert loaded.thermal_control == "none"


def test_load_user_config_reads_mtp_batch_numerics(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text(
        'mtp_batch_numerics = "balanced"\n',
        encoding="utf-8",
    )

    loaded = load_user_config(config)

    assert loaded.mtp_batch_numerics == "balanced"


def test_apply_user_config_respects_explicit_mtp_batch_numerics(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text(
        'mtp_batch_numerics = "balanced"\n',
        encoding="utf-8",
    )
    args = argparse.Namespace(
        command="serve",
        mtp_batch_numerics="throughput",
        _cli_flags={"mtp-batch-numerics"},
    )

    apply_user_config(args, config_path=config)

    assert args.mtp_batch_numerics == "throughput"


def test_apply_user_config_fills_mtp_batch_numerics_default(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text(
        'mtp_batch_numerics = "balanced"\n',
        encoding="utf-8",
    )
    args = argparse.Namespace(
        command="serve",
        mtp_batch_numerics="throughput",
        _cli_flags=set(),
    )

    apply_user_config(args, config_path=config)

    assert args.mtp_batch_numerics == "balanced"


def test_apply_user_config_fills_runtime_defaults(tmp_path):
    config = tmp_path / "config.toml"
    model_dir = tmp_path / "models"
    config.write_text(
        'model = "mtplx/example"\n'
        f'model_dir = "{model_dir}"\n'
        'profile = "exact"\n',
        encoding="utf-8",
    )
    args = argparse.Namespace(
        command="run",
        model=str(DEFAULT_RUNTIME_MODEL_DIR),
        cache_dir=None,
        profile=DEFAULT_PROFILE_NAME,
        _cli_flags=set(),
    )

    loaded = apply_user_config(args, config_path=config)

    assert loaded.exists is True
    assert args.model == "mtplx/example"
    assert args.cache_dir == str(model_dir)
    assert args.profile == "exact"
    assert args.mtplx_config["path"] == str(config)


def test_apply_user_config_fills_and_preserves_model_search_dirs(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text(
        'model_dirs = ["/models/archive", "/models/external"]\n',
        encoding="utf-8",
    )
    args = argparse.Namespace(
        command="serve",
        model=str(DEFAULT_RUNTIME_MODEL_DIR),
        cache_dir=None,
        model_search_dirs=None,
        profile=DEFAULT_PROFILE_NAME,
        _cli_flags=set(),
    )

    apply_user_config(args, config_path=config)
    assert args.model_search_dirs == ["/models/archive", "/models/external"]

    args.model_search_dirs = ["/models/explicit"]
    args._cli_flags = {"model-search-dir"}
    apply_user_config(args, config_path=config)
    assert args.model_search_dirs == ["/models/explicit"]


def test_apply_user_config_sets_forge_model_root(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text(
        'model_dir = "/models/primary"\n'
        'model_dirs = ["/models/archive", "/models/external"]\n',
        encoding="utf-8",
    )
    args = argparse.Namespace(command="forge", _cli_flags=set())

    apply_user_config(args, config_path=config)

    assert args.model_root == "/models/primary"


def test_apply_user_config_preserves_explicit_forge_model_root(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text('model_dir = "/models/config"\n', encoding="utf-8")
    args = argparse.Namespace(
        command="forge",
        forge_action="build",
        model_root="/models/operation",
        _cli_flags={"model-root"},
    )

    apply_user_config(args, config_path=config)

    assert args.model_root == "/models/operation"


@pytest.mark.parametrize(
    "configured,ignored",
    [
        # The default ids early `mtplx init` builds wrote: the old default,
        # not a choice, so the command's own default applies.
        ("mtplx/Qwen3.6-27B-MTPLX-GDN8-Speed4-CyanKiwiMTP", True),
        ("Youssofal/Qwen3.6-27B-MTPLX-Optimized", True),
        # MTPLX never wrote a path into config.toml, so a folder there is the
        # user's whatever its name: the CLI's old built-in default folders,
        # and a <publisher>/<repo> copy named after a retired id (#573).
        ("models/Qwen3.6-27B-MTPLX-Optimized-Speed", False),
        ("models/Qwen3.6-27B-MTPLX-GDN8-Speed4", False),
        ("/Users/example/models/Qwen3.6-27B-MTPLX-Optimized-Speed", False),
        ("~/models/Qwen3.6-27B-MTPLX-GDN8-Speed4", False),
        ("/Users/example/AI/models/Youssofal/Qwen3.6-27B-MTPLX-Optimized", False),
    ],
)
def test_apply_user_config_ignores_only_the_default_ids_old_init_builds_wrote(
    tmp_path, configured, ignored
):
    config = tmp_path / "config.toml"
    config.write_text(f'model = "{configured}"\n', encoding="utf-8")
    args = argparse.Namespace(
        command="quickstart",
        model=DEFAULT_HF_MODEL_ID,
        cache_dir=None,
        profile=DEFAULT_PROFILE_NAME,
        _cli_flags=set(),
    )

    apply_user_config(args, config_path=config)

    assert args.model == (DEFAULT_HF_MODEL_ID if ignored else configured)


def test_apply_user_config_preserves_explicit_runtime_values(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text(
        'model = "mtplx/example"\n'
        'model_dir = "/tmp/mtplx-models"\n'
        'profile = "exact"\n',
        encoding="utf-8",
    )
    args = argparse.Namespace(
        command="serve",
        model="models/local",
        cache_dir="/tmp/explicit-cache",
        profile="performance-cold",
        _cli_flags={"model", "cache-dir", "profile"},
    )

    apply_user_config(args, config_path=config)

    assert args.model == "models/local"
    assert args.cache_dir == "/tmp/explicit-cache"
    assert args.profile == "performance-cold"


def test_apply_user_config_preserves_explicit_default_like_model(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text(
        'model = "mtplx/example"\n'
        'profile = "exact"\n',
        encoding="utf-8",
    )
    explicit_model = "/Users/example/models/Qwen3.6-27B-MTPLX-Optimized-Speed"
    args = argparse.Namespace(
        command="serve",
        model=explicit_model,
        cache_dir=None,
        profile=DEFAULT_PROFILE_NAME,
        _cli_flags={"model"},
    )

    apply_user_config(args, config_path=config)

    assert args.model == explicit_model


def test_apply_user_config_fills_tune_model_defaults(tmp_path):
    config = tmp_path / "config.toml"
    model_dir = tmp_path / "models"
    config.write_text(
        'model = "mtplx/example"\n'
        f'model_dir = "{model_dir}"\n'
        'profile = "exact"\n',
        encoding="utf-8",
    )
    args = argparse.Namespace(
        command="tune",
        model=str(DEFAULT_RUNTIME_MODEL_DIR),
        cache_dir=None,
        profile="performance-cold",
        _cli_flags=set(),
    )

    apply_user_config(args, config_path=config)

    assert args.model == "mtplx/example"
    assert args.cache_dir == str(model_dir)
    assert args.profile == "performance-cold"


def test_apply_user_config_fills_retrieval_defaults(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text(
        'embedding_models = ["org/embed"]\n'
        'reranker_models = ["org/rank=fast-rank"]\n'
        "retrieval_max_resident = 3\n"
        "retrieval_trust_remote_code = true\n",
        encoding="utf-8",
    )

    loaded = load_user_config(config)
    assert loaded.embedding_models == ("org/embed",)
    assert loaded.retrieval_trust_remote_code is True

    args = argparse.Namespace(
        command="serve",
        model=str(DEFAULT_RUNTIME_MODEL_DIR),
        cache_dir=None,
        profile=DEFAULT_PROFILE_NAME,
        embedding_model=[],
        reranker_model=[],
        retrieval_max_resident=2,
        retrieval_trust_remote_code=False,
        _cli_flags=set(),
    )
    apply_user_config(args, config_path=config)

    assert tuple(args.embedding_model) == ("org/embed",)
    assert tuple(args.reranker_model) == ("org/rank=fast-rank",)
    assert args.retrieval_max_resident == 3
    assert args.retrieval_trust_remote_code is True


def test_an_explicit_trust_flag_beats_the_config_file(tmp_path):
    """Trust granted in config must not silently override an explicit CLI no.

    A user who runs with the flag omitted after setting the config key gets
    the config value — but one who explicitly typed the flag spelling into
    _cli_flags keeps their command line.
    """
    config = tmp_path / "config.toml"
    config.write_text("retrieval_trust_remote_code = true\n", encoding="utf-8")
    args = argparse.Namespace(
        command="serve",
        model=str(DEFAULT_RUNTIME_MODEL_DIR),
        cache_dir=None,
        profile=DEFAULT_PROFILE_NAME,
        retrieval_trust_remote_code=False,
        _cli_flags={"retrieval-trust-remote-code"},
    )

    apply_user_config(args, config_path=config)

    assert args.retrieval_trust_remote_code is False


def test_apply_user_config_fills_bench_tune_model_defaults(tmp_path):
    config = tmp_path / "config.toml"
    model_dir = tmp_path / "models"
    config.write_text(
        'model = "mtplx/example"\n'
        f'model_dir = "{model_dir}"\n',
        encoding="utf-8",
    )
    args = argparse.Namespace(
        command="bench",
        bench_action="tune",
        model=str(DEFAULT_RUNTIME_MODEL_DIR),
        cache_dir=None,
        profile=None,
        _cli_flags=set(),
    )

    apply_user_config(args, config_path=config)

    assert args.model == "mtplx/example"
    assert args.cache_dir == str(model_dir)


def test_malformed_user_config_does_not_crash_any_command(tmp_path, monkeypatch, capsys):
    # load_user_config runs on EVERY CLI dispatch. A truncated or hand-mangled
    # config.toml used to raise TOMLDecodeError straight through `mtplx status`,
    # `doctor`, and `stop`, so one bad file bricked the whole CLI.
    config = tmp_path / "config.toml"
    config.write_text('model = "unterminated\nprofile =\n', encoding="utf-8")

    loaded = load_user_config(config)

    assert loaded.exists is False
    assert loaded.path == config
    assert loaded.model is None
    assert loaded.profile is None
    warning = capsys.readouterr().err
    assert str(config) in warning
    assert "mtplx config show" in warning
    assert warning.count("\n") == 1

    # And the same file survives a real command dispatch rather than tracebacking.
    from mtplx.cli import main

    monkeypatch.setenv("MTPLX_CONFIG", str(config))
    assert main(["status", "--json"]) == 0
    assert str(config) in capsys.readouterr().err


def test_malformed_config_value_degrades_to_default(tmp_path, capsys):
    # A bad individual value degrades to that key's default with the same
    # one-line warning; it must not take the rest of the config down with it.
    config = tmp_path / "config.toml"
    config.write_text(
        'model = "mtplx/example"\npaged_kv_quantization = "not-a-mode"\n',
        encoding="utf-8",
    )

    loaded = load_user_config(config)

    assert loaded.exists is True
    assert loaded.model == "mtplx/example"
    assert loaded.paged_kv_quantization is None
    warning = capsys.readouterr().err
    assert str(config) in warning
    assert "paged_kv_quantization" in warning


def test_badly_typed_values_degrade_only_their_own_key(tmp_path, capsys):
    # The TOML parses, so the parse guard above never sees these: the typed
    # converters used to raise ValueError straight out of load_user_config
    # and every command exited 1 with a traceback. Each bad key must degrade
    # to its own default with one warning while the good keys still apply.
    config = tmp_path / "config.toml"
    config.write_text(
        "\n".join(
            [
                'model = "mtplx/example"',
                "top_k = 20",
                'context_window = "64k"',
                'experimental_mtp_cohorts = "maybe"',
                'temperature = "warm"',
                "embedding_models = 5",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    loaded = load_user_config(config)

    assert loaded.exists is True
    assert loaded.model == "mtplx/example"
    assert loaded.top_k == 20
    assert loaded.context_window is None
    assert loaded.experimental_mtp_cohorts is None
    assert loaded.temperature is None
    assert loaded.embedding_models == ()
    warnings = capsys.readouterr().err.splitlines()
    assert len(warnings) == 4
    assert all(str(config) in line for line in warnings)
    joined = "\n".join(warnings)
    assert f"ignoring context_window in {config}: expected a whole number, got '64k'" in joined
    assert f"ignoring experimental_mtp_cohorts in {config}: expected boolean value, got 'maybe'" in joined
    assert f"ignoring temperature in {config}: expected a number, got 'warm'" in joined
    assert f"ignoring embedding_models in {config}: expected a list of model names, got 5" in joined
    assert "fix or remove that line" in warnings[0]
    assert "Traceback" not in joined


def test_badly_typed_value_does_not_brick_command_dispatch(tmp_path, monkeypatch, capsys):
    from mtplx.cli import main

    config = tmp_path / "config.toml"
    config.write_text('context_window = "64k"\n', encoding="utf-8")
    monkeypatch.setenv("MTPLX_CONFIG", str(config))

    assert main(["status", "--json"]) == 0
    assert "ignoring context_window in" in capsys.readouterr().err
    assert main(["config", "show"]) == 0
    captured = capsys.readouterr()
    assert "context_window: None" in captured.out
    assert "ignoring context_window in" in captured.err


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("context_window", "64k", "context_window must be a whole number, got '64k'"),
        ("temperature", "warm", "temperature must be a number, got 'warm'"),
        ("experimental_mtp_cohorts", "maybe", "experimental_mtp_cohorts must be true or false"),
        ("profile", "fastest", "unknown MTPLX profile 'fastest'"),
        ("paged_kv_quantization", "q3", "unsupported paged KV quantization mode 'q3'"),
    ],
)
def test_config_set_refuses_a_bad_value_and_writes_nothing(tmp_path, monkeypatch, key, value, message):
    # `mtplx config set context_window 64k` used to traceback on int() and,
    # worse, is how the badly typed file above could be produced in the first
    # place. It must refuse with one plain line and leave the file untouched.
    from mtplx.cli import main

    config = tmp_path / "config.toml"
    monkeypatch.setenv("MTPLX_CONFIG", str(config))

    with pytest.raises(SystemExit) as excinfo:
        main(["config", "set", key, value])

    assert excinfo.value.code != 0
    assert message in str(excinfo.value.code)
    assert not config.exists()


def test_config_set_bad_value_exits_nonzero_without_a_traceback(tmp_path):
    # The real console path: the message is one stderr line, status 1.
    config = tmp_path / "config.toml"
    proc = subprocess.run(
        [sys.executable, "-m", "mtplx.cli", "config", "set", "context_window", "64k"],
        env={**os.environ, "MTPLX_CONFIG": str(config)},
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert proc.returncode == 1
    assert "Traceback" not in proc.stderr
    assert proc.stderr.strip() == "context_window must be a whole number, got '64k'"
    assert not config.exists()


def test_config_set_good_values_round_trip(tmp_path, monkeypatch, capsys):
    from mtplx.cli import main

    config = tmp_path / "config.toml"
    monkeypatch.setenv("MTPLX_CONFIG", str(config))

    assert main(["config", "set", "context_window", "65536"]) == 0
    assert main(["config", "set", "experimental_mtp_cohorts", "yes"]) == 0
    assert main(["config", "set", "temperature", "0.6"]) == 0
    capsys.readouterr()

    loaded = load_user_config(config)

    assert loaded.context_window == 65536
    assert loaded.experimental_mtp_cohorts is True
    assert loaded.temperature == 0.6
    assert capsys.readouterr().err == ""
