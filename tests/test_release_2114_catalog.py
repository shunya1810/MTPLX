"""Release identities and the shared app/CLI recommendation contract (CPU only)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from mtplx.artifacts import _hf_repo_id_from_ref
from mtplx.backends.descriptors import model_family_from_inspection, reasoning_policy_for_model
from mtplx.commands.public import _model_ref_from_public_model_id
from mtplx.default_models import public_model_id_for_ref
from mtplx.model_catalog import catalog_model_matching, catalog_model_with_id

NEW_MODELS = (
    ("flash-next-optimized-quality", "qwen4_exp"),
    ("bonsai-2-27b-optimized-speed", "qwen3_8"),
)


@pytest.mark.parametrize("catalog_id,family", NEW_MODELS)
def test_new_pack_identity_round_trips(catalog_id, family, tmp_path):
    model = catalog_model_with_id(catalog_id)
    public_id = f"mtplx-{catalog_id}"
    refs = [catalog_id, model.hf_model_id, *[a for a in model.aliases if " " not in a]]
    for ref in refs:
        assert catalog_model_matching(ref) == model
        assert public_model_id_for_ref(ref) == public_id
        assert _model_ref_from_public_model_id(ref) == model.hf_model_id
        assert _hf_repo_id_from_ref(ref) == model.hf_model_id
        assert model_family_from_inspection(model_ref=ref) == family
    for name in {Path(model.hf_model_id).name, *[a for a in model.aliases if "MTPLX" in a]}:
        local = tmp_path / name
        local.mkdir()
        assert catalog_model_matching(local) == model
        assert public_model_id_for_ref(local) == public_id
        assert model_family_from_inspection(model_ref=str(local)) == family
    # A derivative does not acquire a first-party served identity.
    derivative = Path(model.hf_model_id).name + "-third-party"
    assert public_model_id_for_ref(derivative) != public_id
    assert catalog_model_matching(derivative) is None


def test_legacy_bonsai_runtime_identity_and_family(tmp_path):
    pack = tmp_path / "renamed-pack"
    pack.mkdir()
    (pack / "mtplx_runtime.json").write_text(json.dumps({
        "public_model_id": "mtplx-bonsai-38-27b-optimized-speed",
        "arch_id": "qwen3-next-mtp",
    }))
    assert public_model_id_for_ref(pack) == "mtplx-bonsai-2-27b-optimized-speed"
    assert model_family_from_inspection(model_ref=str(pack)) == "qwen3_8"
    codec = reasoning_policy_for_model(model_ref=str(pack))
    assert codec.effort_levels == ("xhigh", "medium")
    assert codec.default_effort == "medium"
    assert codec.agent_effort == "medium"


@pytest.mark.parametrize("ref", [
    "Youssofal/Ternary-Bonsai-2-27B-MTPLX-Optimized-Speed",
    "prism-ml/Ternary-Bonsai-2-27B-mlx-2bit",
    "mtplx-bonsai-2-27b-optimized-speed",
    "Bonsai-3.8-27B-MTPLX-Optimized-Speed",
])
def test_bonsai_reasoning_excludes_unsupported_low(ref):
    from mtplx.backends.descriptors import model_controls_for_descriptor, QWEN3_NEXT_DESCRIPTOR

    codec = reasoning_policy_for_model(ref)
    assert codec.effort_levels == ("xhigh", "medium")
    assert codec.default_effort == "medium"
    controls = model_controls_for_descriptor(QWEN3_NEXT_DESCRIPTOR, model_ref=ref)
    assert controls["reasoning"]["effort_levels"] == ["xhigh", "medium"]
    assert controls["reasoning"]["default_effort"] == "medium"

MATRIX = json.loads((Path(__file__).parent / "fixtures/release_2114_recommendations.json").read_text())


@pytest.mark.parametrize("row", MATRIX, ids=lambda r: f"{r['tier']}-{r['ram_gib']}")
def test_app_cli_ram_matrix(row, monkeypatch):
    from mtplx import default_models as defaults
    from mtplx.model_catalog import recommended_catalog_ids, recommended_models
    from mtplx.ui import onboarding

    monkeypatch.setenv(defaults.QWEN38_OPTIMIZED_SPEED_MODEL_ENV, "off")
    monkeypatch.setenv(defaults.SPEED_MODEL_ENV, "off")
    monkeypatch.delenv(defaults.DEFAULT_MODEL_VARIANT_ENV, raising=False)
    monkeypatch.setattr(defaults, "_QWEN38_OPTIMIZED_SPEED_FP16_LOCAL_CANDIDATES", ())
    args = dict(memory_gib=row["ram_gib"], chip_tier=row["tier"])
    assert recommended_catalog_ids(**args) == row["raw"]
    assert [m.id for m in recommended_models(**args)] == row["visible"]
    hardware = dict(apple_silicon_generation="m2" if row["tier"] == "legacy" else "m5", memory_gib=row["ram_gib"])
    if row["default"] is None:
        with pytest.raises(defaults.DefaultModelUnavailable):
            defaults.select_default_model(hardware=hardware)
        return
    selection = defaults.select_default_model(hardware=hardware)
    assert catalog_model_matching(selection.hf_model).id == row["default"]
    monkeypatch.setattr(onboarding, "_verified_default_selection", lambda: selection)
    panels = []
    monkeypatch.setattr(onboarding, "_step_panel", lambda **kw: panels.extend(kw["options"]))
    monkeypatch.setattr(onboarding, "_prompt_choice", lambda *args, **kw: kw["default"])
    assert onboarding.screen_model(installed=[]) == selection.model
    offered = [title.split("  ·")[0] for _, title, _ in panels[:-2]]
    expected = [catalog_model_with_id(i).display_name for i in row["visible"]]
    # Older CLI labels omit a space in Qwen3.5; the actual catalog identities agree.
    assert [x.replace("Qwen3.5", "Qwen 3.5") for x in offered] == expected


def test_bonsai_bound_matches_swift_and_can_move_to_24(monkeypatch):
    import re
    from mtplx import model_catalog as catalog
    from mtplx.default_models import select_default_model
    swift = Path("apps/MTPLXApp/Sources/MTPLXAppCore/Models/MTPLXModelOption.swift").read_text()
    bound = float(re.search(r"bonsaiRecommendationMinGiB: Double = ([0-9.]+)", swift)[1])
    assert bound == catalog.BONSAI_RECOMMENDATION_MIN_GIB == 16
    monkeypatch.setattr(catalog, "BONSAI_RECOMMENDATION_MIN_GIB", 24)
    # Below a moved bound Bonsai follows the 9B-class picks, so MiMo (always
    # right ahead of the Qwen 3.5 9B) is the app's first pick and the CLI
    # default alike.
    for ram, winner in [(16, "mimo-v26-qwen-9b-optimized-speed"), (18, "mimo-v26-qwen-9b-optimized-speed"), (24, "bonsai-2-27b-optimized-speed")]:
        offered = catalog.recommended_catalog_ids(memory_gib=ram, chip_tier="modern")
        assert offered[0] == winner
        assert "bonsai-2-27b-optimized-speed" in offered
        assert catalog_model_matching(select_default_model(hardware={"apple_silicon_generation": "m5", "memory_gib": ram}).hf_model).id == winner


@pytest.mark.parametrize("ram,peak,verdict", [(256, 170.7, "tight_fit"), (256, 170.6, "recommended"), (16, 10.7, "tight_fit"), (16, 10.6, "recommended")])
def test_badge_safety_boundaries(ram, peak, verdict):
    from dataclasses import replace
    from mtplx.model_catalog import evaluate_feasibility
    model = replace(catalog_model_with_id("bonsai-2-27b-optimized-speed"), peak_memory_gib=peak)
    assert evaluate_feasibility(model, chip_tier="modern", ram_gib=ram, disk_free_gib=1000).verdict == verdict


def test_download_disk_rule_is_the_mtplx_pull_rule_in_python_and_swift(monkeypatch, tmp_path):
    import re
    from types import SimpleNamespace
    from mtplx import hf_loader
    from mtplx.model_catalog import DOWNLOAD_HEADROOM_GIB, evaluate_feasibility
    swift = Path("apps/MTPLXApp/Sources/MTPLXAppCore/Onboarding/ModelFeasibility.swift").read_text()
    assert float(re.search(r"downloadHeadroomGiB: Double = ([0-9.]+)", swift)[1]) == DOWNLOAD_HEADROOM_GIB == 5
    quality = catalog_model_with_id("flash-next-optimized-quality")
    needs = quality.size_bytes / 1024**3 + DOWNLOAD_HEADROOM_GIB  # 163.3 GiB; the old 2.5x rule asked 395.7
    short = evaluate_feasibility(quality, chip_tier="modern", ram_gib=256, disk_free_gib=needs - 0.01)
    assert short.verdict == "insufficient_disk" and short.needs_gib == pytest.approx(needs)
    assert evaluate_feasibility(quality, chip_tier="modern", ram_gib=256, disk_free_gib=needs).ok
    # A paused pull needs only its remaining bytes.
    half = quality.size_bytes // 2
    resume_free = (quality.size_bytes - half) / 1024**3 + DOWNLOAD_HEADROOM_GIB
    assert evaluate_feasibility(quality, chip_tier="modern", ram_gib=256, disk_free_gib=resume_free, downloaded_bytes=half).ok
    assert not evaluate_feasibility(quality, chip_tier="modern", ram_gib=256, disk_free_gib=resume_free).ok
    # The engine's pull check draws the same line.
    required = quality.size_bytes + int(DOWNLOAD_HEADROOM_GIB * 1024**3)
    monkeypatch.setattr(hf_loader.shutil, "disk_usage", lambda _root: SimpleNamespace(free=required - 1))
    with pytest.raises(RuntimeError, match="insufficient free disk space"):
        hf_loader._require_download_disk_headroom(tmp_path, total_bytes=quality.size_bytes, started_size_bytes=0)
    monkeypatch.setattr(hf_loader.shutil, "disk_usage", lambda _root: SimpleNamespace(free=required))
    hf_loader._require_download_disk_headroom(tmp_path, total_bytes=quality.size_bytes, started_size_bytes=0)

@pytest.mark.parametrize("name,ram,catalog_id", [
    ("Bonsai-3.8-27B-MTPLX-Optimized-Speed", 16, "bonsai-2-27b-optimized-speed"),
    # 128 GiB Macs run this pack by choice; it must not be swapped for the 27B.
    ("Qwen3.8-Flash-Next-MTPLX-Optimized-Speed", 256, "flash-next-optimized-speed"),
])
def test_default_reuses_complete_library_pack(name, ram, catalog_id, monkeypatch, tmp_path):
    from mtplx.default_models import is_verified_default_model_ref, select_default_model
    root = tmp_path / "library"
    pack = root / name
    pack.mkdir(parents=True)
    (pack / "config.json").write_text("{}")
    (pack / "model.safetensors").write_bytes(b"test weights")
    (pack / "mtp.safetensors").write_bytes(b"test head")
    monkeypatch.setenv("MTPLX_MODEL_DIR", str(root))
    selection = select_default_model(hardware={"apple_silicon_generation": "m5", "memory_gib": ram})
    assert selection.model == str(pack)
    assert catalog_model_matching(selection.hf_model).id == catalog_id
    # Even the default's own copy is not "the default" by its folder alone:
    # start setup records a default pick, so a folder never gets swapped (#573).
    assert not is_verified_default_model_ref(pack)
    assert not is_verified_default_model_ref(catalog_model_with_id("flash-next-optimized-quality").hf_model_id)

@pytest.mark.parametrize("catalog_id,family", NEW_MODELS)
def test_new_packs_do_not_receive_unmeasured_turbo_promotion(catalog_id, family):
    from types import SimpleNamespace
    from mtplx.commands.public import _apply_model_default_profile
    model = catalog_model_with_id(catalog_id)
    args = SimpleNamespace(model=model.hf_model_id, profile="sustained", _cli_flags=set())
    assert not _apply_model_default_profile(args, f"mtplx-{catalog_id}")
    assert args.profile == "sustained"


def test_bonsai_builder_stamps_catalog_identity_family_and_engine_floor():
    from scripts import build_bonsai_mtplx_pack as builder
    pack = catalog_model_with_id("bonsai-2-27b-optimized-speed")
    assert builder.PACK_NAME == Path(pack.hf_model_id).name
    assert builder.HF_REPO == pack.hf_model_id
    assert builder.PUBLIC_MODEL_ID == f"mtplx-{pack.id}"
    contract = builder.build_runtime_contract(mtplx_version="2.11.3", provenance={}, head=None)
    assert contract["model_family"] == "qwen3_8"
    assert contract["public_model_id"] == builder.PUBLIC_MODEL_ID
    assert contract["min_engine_version"] == "2.12.0"
    card = builder.render_card(source_sha="test-digest", head_note="test head")
    assert f"--model {pack.hf_model_id}" in card
    assert "Bonsai 2 27B" in card and "Prism ML" in card
    assert "2.12.0 or newer" in card
