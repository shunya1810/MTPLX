"""`mtplx start` keeps the model the user chose (issue #573).

The start wizard saves its choice and offers it again with "Use the same
configuration?". Reuse re-resolved every saved model that looked like a
verified default: a local folder named like the default pack (a copy kept by
another app, a folder on another drive) and catalog packs that are, or once
were, a default somewhere (Bare Speed, Bonsai, the 27B on a 256 GB Mac). The
choice was swapped for this Mac's default, the Welcome back panel showed that
repo id, and Yes started a 20 GB download of a model already on disk. Only a
user who took the row marked as this Mac's verified default, for the copy the
default resolver loads, follows it when it moves; the same model reached
another way (Local folder, a catalog row, a custom repo, the app's model,
config.toml, a copy in another folder) stays as chosen.

These tests run `mtplx start` through `mtplx.cli.main` up to the launch,
with the Hugging Face preflight and pull, the busy-port probe, the fan-pin
recovery, the model gate and the server launch stubbed: no network, no port,
no model load.
"""

from __future__ import annotations

import builtins
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mtplx import cli, default_models, hf_loader, thermal
from mtplx.commands import public
from mtplx.constants import DEFAULT_RUNTIME_MODEL_DIR
from mtplx.profiles import (
    BONSAI_OPTIMIZED_SPEED_HF_MODEL_ID,
    QWEN38_OPTIMIZED_SPEED_HF_MODEL_ID,
)
from mtplx.ui import onboarding

DEFAULT_PACK = "Qwen3.8-27B-MTPLX-Optimized-Speed"


def _model_folder(path: Path) -> Path:
    """A folder the picker, the default resolver and the scan accept."""

    path.mkdir(parents=True)
    (path / "config.json").write_text(
        json.dumps({"architectures": ["Qwen3_5ForConditionalGeneration"]}),
        encoding="utf-8",
    )
    (path / "model.safetensors").write_bytes(b"weights")
    (path / "mtp.safetensors").write_bytes(b"head")
    return path


def _answer(monkeypatch, *answers) -> list:
    """Feed ``input`` in order and return the queue that is left.

    ``row:<title>`` picks a numbered row by the title a user reads (row
    numbers move with the catalog and the RAM tier); ``KeyboardInterrupt``
    presses Ctrl-C.
    """

    queue = list(answers)
    panels: list[list[tuple[str, str, str]]] = []
    real_panel = onboarding._choice_panel

    def panel(*, heading, options, intro=None, border_style="cyan"):
        panels.append(list(options))
        real_panel(heading=heading, options=options, intro=intro, border_style=border_style)

    def fake_input(prompt=""):
        assert queue, f"unexpected prompt {prompt!r}"
        answer = queue.pop(0)
        if answer is KeyboardInterrupt:
            raise KeyboardInterrupt
        if not answer.startswith("row:"):
            return answer
        rows = panels[-1] if panels else []
        for number, headline, _detail in rows:
            if answer[4:] in headline:
                return number
        raise AssertionError(f"{answer[4:]!r} not offered at {prompt!r}: {rows}")

    monkeypatch.setattr(onboarding, "_choice_panel", panel)
    monkeypatch.setattr(builtins, "input", fake_input)
    return queue


@pytest.fixture()
def mac(tmp_path, monkeypatch):
    """A Mac with its own home folder, an empty MTPLX model library, no
    config.toml yet (``mac.config``) and a pinned chip; ``mac.memory_gib``
    picks the RAM tier. The default resolver also probes fixed folders under
    the home, so the home moves too."""

    home = (tmp_path / "home").resolve()
    library = home / ".mtplx" / "models"
    library.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("MTPLX_MODEL_DIR", str(library))
    monkeypatch.setenv("MTPLX_QUICKSTART_STATE", str(home / ".mtplx" / "quickstart.json"))
    monkeypatch.setenv("MTPLX_CONFIG", str(home / ".mtplx" / "config.toml"))
    for name in (
        default_models.DEFAULT_MODEL_VARIANT_ENV,
        default_models.SPEED_MODEL_ENV,
        default_models.QWEN38_BARE_SPEED_MODEL_ENV,
        default_models.QWEN38_OPTIMIZED_SPEED_MODEL_ENV,
        default_models.QUALITY_MODEL_ENV,
        "MTPLX_MODEL_DIRS",
    ):
        monkeypatch.delenv(name, raising=False)
    machine = SimpleNamespace(
        home=home,
        library=library,
        config=home / ".mtplx" / "config.toml",
        memory_gib=64.0,
    )
    monkeypatch.setattr(
        default_models,
        "detect_apple_silicon",
        lambda: {
            "apple_silicon_generation": "m3",
            "chip": "Apple M3 Max",
            "memory_gib": machine.memory_gib,
        },
    )
    # Plain-text panels: one line per row, nothing wrapped by terminal width.
    monkeypatch.setattr(onboarding, "_console", lambda: None)
    return machine


@pytest.fixture()
def start(monkeypatch):
    """Run `mtplx start` for real up to the launch; record downloads and launches."""

    record = SimpleNamespace(downloads=[], launched=[])

    def pull_model(repo_id, **_kwargs):
        record.downloads.append(repo_id)
        raise RuntimeError("tests never download")

    def launch(_args, *, runtime_model, inspection):
        record.launched.append(str(runtime_model))
        return 0

    monkeypatch.setattr(hf_loader, "pull_model", pull_model)
    monkeypatch.setattr(
        public,
        "inspect_model",
        lambda ref: SimpleNamespace(
            to_dict=lambda: {"model": ref, "compatibility": {"can_run": True}}
        ),
    )
    monkeypatch.setattr(public, "_quickstart_autoselect_busy_port", lambda *a, **k: None)
    monkeypatch.setattr(thermal, "check_and_recover_stale_max", dict)
    monkeypatch.setattr(public, "_model_gate", lambda _model, **_k: ({}, None))
    monkeypatch.setattr(public, "_apply_runtime_compatibility_mode", lambda *a, **k: None)
    monkeypatch.setattr(public, "_apply_model_contract_depth_default", lambda *a, **k: None)
    monkeypatch.setattr(public, "_apply_backend_serve_defaults", lambda *a, **k: None)
    monkeypatch.setattr(public, "_quickstart_apply_tuned_depth", lambda *a, **k: None)
    monkeypatch.setattr(public, "_quickstart_run_openwebui", launch)

    def run(*answers, argv=()) -> int:
        left = _answer(monkeypatch, *answers)
        # An interactive terminal, set on the streams capsys has installed.
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        monkeypatch.setattr("sys.stdout.isatty", lambda: True)
        code = cli.main(["start", *argv])
        assert left == [], f"prompts never asked: {left}"
        return code

    record.run = run
    return record


def test_a_local_folder_pick_is_reused_on_every_start_without_a_download(
    mac, start, capsys
):
    # #573: a <publisher>/<repo name> copy of this Mac's default pack in
    # another app's model folder, outside MTPLX's model library.
    folder = _model_folder(mac.home / "AI" / "models" / "Youssofal" / DEFAULT_PACK)
    shown = f"~/AI/models/Youssofal/{DEFAULT_PACK}"

    assert start.run(
        "row:Local folder",
        "~/AI/models",
        f"row:{DEFAULT_PACK}",
        "row:Auto",
        "row:Web UI",
        "row:No",
    ) == 0
    capsys.readouterr()
    for _ in range(2):
        assert start.run("Y") == 0
        out = capsys.readouterr().out
        assert "Last time you used:" in out
        assert f"Model:     {shown}\n" in out
        assert f"[1/4] Checking model: {folder}\n" in out

    assert start.launched == [str(folder)] * 3
    assert start.downloads == []
    assert onboarding.load_state()["model"] == str(folder)


def test_a_saved_folder_stamped_by_the_old_setup_stays_local(mac, start):
    """The state #573 users still have if they have not answered Yes yet."""

    folder = _model_folder(mac.home / "Models" / DEFAULT_PACK)
    onboarding.save_state(
        {
            "model": str(folder),
            "profile": "auto",
            "max": False,
            "target": "openwebui",
            "open_dashboard": False,
            # The old setup stamped this Mac's default on any default-named pick.
            "model_selection": default_models.select_default_model().to_dict(),
        }
    )

    assert start.run("Y") == 0

    assert start.launched == [str(folder)]
    assert start.downloads == []


def test_taking_the_verified_default_still_follows_the_default_when_it_moves(
    mac, start, capsys
):
    installed = _model_folder(mac.library / f"Youssofal--{DEFAULT_PACK}")
    assert start.run(
        "row:verified default", "row:Auto", "row:Web UI", "row:No"
    ) == 0
    assert onboarding.load_state()["model_selection"]["model"] == str(installed)

    # A Mac in the 256 GB tier gets Flash-Next as its default: a stand-in for
    # the catalog moving the default this user took.
    mac.memory_gib = 256.0
    flash_next = _model_folder(
        mac.library / "Youssofal--Qwen3.8-Flash-Next-MTPLX-Optimized-Speed"
    )
    capsys.readouterr()
    assert start.run("Y") == 0

    out = capsys.readouterr().out
    assert f"moved here from ~/.mtplx/models/Youssofal--{DEFAULT_PACK}" in out
    assert start.launched == [str(installed), str(flash_next)]
    assert start.downloads == []
    saved = onboarding.load_state()
    assert saved["model"] == saved["model_selection"]["model"] == str(flash_next)


def test_a_local_folder_pick_of_the_defaults_own_copy_keeps_that_copy(mac, start):
    """Local folder lands on the very folder the default row would: still a
    pick, so a new default for this Mac does not replace it."""

    chosen = _model_folder(mac.library / f"Youssofal--{DEFAULT_PACK}")
    assert start.run(
        "row:Local folder", str(chosen), "row:Auto", "row:Web UI", "row:No"
    ) == 0
    assert onboarding.load_state()["model_selection"] is None

    mac.memory_gib = 256.0
    _model_folder(mac.library / "Youssofal--Qwen3.8-Flash-Next-MTPLX-Optimized-Speed")
    assert start.run("Y") == 0

    assert start.launched == [str(chosen)] * 2
    assert start.downloads == []


def test_the_default_row_for_a_copy_the_resolver_cannot_see_keeps_that_copy(
    mac, start, capsys
):
    """A --model-search-dir folder (or config.toml's model_dirs) fills the
    picker but not the default resolver, so the verified default row there
    shows a copy that following would replace with a download."""

    shelf = mac.home / "Models"
    copy = _model_folder(shelf / f"Youssofal--{DEFAULT_PACK}")
    argv = ("--model-search-dir", str(shelf))
    assert start.run(
        "row:verified default", "row:Auto", "row:Web UI", "row:No", argv=argv
    ) == 0
    assert onboarding.load_state()["model_selection"] is None

    capsys.readouterr()
    assert start.run("Y", argv=argv) == 0

    assert f"Model:     ~/Models/Youssofal--{DEFAULT_PACK}\n" in capsys.readouterr().out
    assert start.launched == [str(copy)] * 2
    assert start.downloads == []


def test_accepting_the_default_repo_id_as_a_custom_repo_is_a_pick(mac, monkeypatch):
    # Enter at the repo prompt accepts the suggested id, this Mac's default.
    _answer(monkeypatch, "row:Custom Hugging Face repo", "", "row:Auto", "row:CLI")
    first = onboarding.run_quickstart_flow(fresh=True)
    assert first["model"] == QWEN38_OPTIMIZED_SPEED_HF_MODEL_ID
    assert first["model_selection"] is None

    mac.memory_gib = 256.0  # the default moves to Flash-Next
    _answer(monkeypatch, "Y")
    assert onboarding.run_quickstart_flow()["model"] == QWEN38_OPTIMIZED_SPEED_HF_MODEL_ID


@pytest.mark.parametrize(
    "memory_gib,row,installed,expected",
    [
        # Bare Speed was briefly the 3.8 default; picking it now is a choice.
        (64.0, "Qwen 3.8 27B Bare Speed", "Youssofal--Qwen3.8-27B-MTPLX-Bare-Speed", None),
        # 256 GB Macs default to Flash-Next; the 27B is the user's pick.
        (256.0, "Qwen 3.8 27B Optimized Speed", None, QWEN38_OPTIMIZED_SPEED_HF_MODEL_ID),
        # Bonsai is the 16-31 GB default; on 64 GB it is the user's pick.
        (64.0, "Bonsai 2 27B Optimized Speed", None, BONSAI_OPTIMIZED_SPEED_HF_MODEL_ID),
    ],
)
def test_a_catalog_pick_that_is_not_this_macs_default_is_kept(
    mac, monkeypatch, memory_gib, row, installed, expected
):
    mac.memory_gib = memory_gib
    if installed:
        expected = str(_model_folder(mac.library / installed))
    _answer(monkeypatch, f"row:{row}", "row:Auto", "row:CLI")
    assert onboarding.run_quickstart_flow(fresh=True)["model"] == expected

    _answer(monkeypatch, "Y")
    again = onboarding.run_quickstart_flow()

    assert again["model"] == expected
    assert "previous_default_model" not in again


def test_same_as_the_app_keeps_the_apps_model(mac, monkeypatch, tmp_path):
    mac.memory_gib = 256.0  # default: Flash-Next
    app_settings = tmp_path / "app-settings.json"
    app_settings.write_text(
        json.dumps({"model": QWEN38_OPTIMIZED_SPEED_HF_MODEL_ID}), encoding="utf-8"
    )
    monkeypatch.setenv("MTPLX_APP_SETTINGS_PATH", str(app_settings))
    onboarding.save_state(
        {"model": "someone/other-model", "profile": "auto", "max": False, "target": "cli"}
    )
    _answer(monkeypatch, "row:Same as the MTPLX app")
    assert onboarding.run_quickstart_flow()["model"] == QWEN38_OPTIMIZED_SPEED_HF_MODEL_ID

    _answer(monkeypatch, "")  # Enter on the next start
    assert onboarding.run_quickstart_flow()["model"] == QWEN38_OPTIMIZED_SPEED_HF_MODEL_ID


@pytest.mark.parametrize("name", [DEFAULT_PACK, "my-qwen-build"])
def test_a_missing_saved_folder_is_named_and_never_swapped_for_a_download(
    mac, start, capsys, name
):
    drive = mac.home / "T7"
    folder = _model_folder(drive / name)
    assert start.run(
        "row:Local folder", str(folder), "row:Auto", "row:Web UI", "row:No"
    ) == 0

    unplugged = drive.rename(mac.home / "T7-unplugged")
    capsys.readouterr()
    assert start.run(KeyboardInterrupt) == 130  # Ctrl-C at the model screen

    out = capsys.readouterr().out
    assert f"The model folder from last time is not available: ~/T7/{name}" in out
    assert "Last time you used:" not in out
    assert start.downloads == []
    assert onboarding.load_state()["model"] == str(folder)

    unplugged.rename(drive)
    assert start.run("Y") == 0
    assert start.launched == [str(folder)] * 2
    assert start.downloads == []


def test_a_missing_library_folder_picked_by_hand_is_named(mac, start, capsys):
    chosen = _model_folder(mac.library / f"Youssofal--{DEFAULT_PACK}")
    assert start.run(
        "row:Local folder", str(chosen), "row:Auto", "row:Web UI", "row:No"
    ) == 0
    chosen.rename(mac.home / "moved-away")
    capsys.readouterr()
    assert start.run(KeyboardInterrupt) == 130

    out = capsys.readouterr().out
    assert (
        "The model folder from last time is not available: "
        f"~/.mtplx/models/Youssofal--{DEFAULT_PACK}"
    ) in out
    assert "Last time you used:" not in out
    assert start.downloads == []
    assert onboarding.load_state()["model"] == str(chosen)


def test_a_folder_is_never_taken_for_the_default_by_its_name(mac):
    own_copy = _model_folder(mac.library / f"Youssofal--{DEFAULT_PACK}")
    assert default_models.select_default_model().model == str(own_copy)

    for folder in (
        own_copy,
        # A former default's folder, still installed or long gone.
        mac.library / "Qwen3.8-27B-MTPLX-Bare-Speed",
        mac.home / "AI" / "models" / "Youssofal" / DEFAULT_PACK,
        Path("/Volumes/T7") / f"Youssofal--{DEFAULT_PACK}",
    ):
        assert not default_models.is_verified_default_model_ref(str(folder))
    # The repo id `mtplx setup` writes into config.toml still follows.
    assert default_models.is_verified_default_model_ref(QWEN38_OPTIMIZED_SPEED_HF_MODEL_ID)


@pytest.mark.parametrize(
    "where",
    [
        ("AI", "models", "Youssofal", DEFAULT_PACK),
        (".mtplx", "models", "Youssofal--Qwen3.8-27B-MTPLX-Bare-Speed"),
        (".mtplx", "models", f"Youssofal--{DEFAULT_PACK}"),
    ],
)
def test_a_configured_folder_is_used_as_configured(mac, monkeypatch, where):
    """`mtplx config set model <folder>`: start without the wizard (--yes, no
    terminal) loads the folder, and the wizard offers it as a choice. MTPLX
    writes a default into config.toml only as a repo id, so a folder there
    is the user's."""

    folder = _model_folder(mac.home.joinpath(*where))
    args = SimpleNamespace(model=str(folder), _model_explicit=False)
    assert public._quickstart_current_model(args) == str(folder)

    _answer(monkeypatch, "row:Use your configured model")
    assert onboarding.screen_model(configured=str(folder), installed=[]) == str(folder)


@pytest.mark.parametrize("relative", [True, False], ids=["relative", "absolute"])
def test_config_toml_loads_a_folder_named_like_the_old_built_in_default(
    mac, start, monkeypatch, relative
):
    """A real config.toml read by `mtplx start --yes`. The folder carries the
    name the CLI used as its own --model default until 2026-05-15, and this
    Mac's default is installed, so a swap would load without a download."""

    chosen = _model_folder(mac.home / DEFAULT_RUNTIME_MODEL_DIR)
    _model_folder(mac.library / f"Youssofal--{DEFAULT_PACK}")
    monkeypatch.chdir(mac.home)
    ref = str(DEFAULT_RUNTIME_MODEL_DIR) if relative else str(chosen)
    mac.config.write_text(f"model = {json.dumps(ref)}\n", encoding="utf-8")

    assert start.run(argv=("--yes",)) == 0

    # A relative setting stays relative, as typed: the same folder from here.
    assert [Path(path).resolve() for path in start.launched] == [chosen]
    assert start.downloads == []


@pytest.mark.parametrize("relative", [True, False], ids=["relative", "absolute"])
def test_config_toml_naming_a_missing_folder_asks_before_anything_loads(
    mac, start, monkeypatch, capsys, relative
):
    missing = mac.home / DEFAULT_RUNTIME_MODEL_DIR
    _model_folder(mac.library / f"Youssofal--{DEFAULT_PACK}")
    monkeypatch.chdir(mac.home)
    ref = str(DEFAULT_RUNTIME_MODEL_DIR) if relative else str(missing)
    mac.config.write_text(f"model = {json.dumps(ref)}\n", encoding="utf-8")

    # The start names the folder, then asks before downloading the default.
    assert start.run("n", argv=("--yes",)) == 1

    assert f"[1/4] Checking model: {ref}\n" in capsys.readouterr().out
    assert start.launched == []
    assert start.downloads == []


def test_the_welcome_back_panel_shows_a_long_folder_path_whole(monkeypatch):
    from rich.console import Console

    rendered = io.StringIO()
    monkeypatch.setattr(
        onboarding,
        "_console",
        lambda: Console(file=rendered, width=80, color_system=None),
    )
    monkeypatch.setattr(builtins, "input", lambda _prompt="": "")
    folder = f"/Volumes/External/shared-models/Youssofal/{DEFAULT_PACK}"

    assert onboarding.confirm_same_as_last(
        {"model": folder, "profile": "auto", "target": "openwebui"}
    )

    text = rendered.getvalue()
    assert "…" not in text
    assert folder in "".join(text.replace("│", "").split())
