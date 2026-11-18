"""Family-marker collision fences (F21/F22, 2.8 charlatan-defensibility).

F21: stock Qwen3 sizes collide with the Qwen 3.8 version token. In
``Qwen/Qwen3-8B`` the ``-8B`` is a parameter count, not a version, but the
plain ``qwen3-8`` substring marker claimed it for the qwen3_8 family — wrong
sampler (temp 1.0 vs stock 0.6), wrong draft temperature, wrong reasoning
codec, and MTPLX takes the accuracy blame on someone else's model. Boundary
rule: the version token ``qwen3[._-]?8`` immediately followed by a digit-run
ending in ``b`` is a size (8B, 80B), never the 3.8 family.

F22: a renamed or symlinked directory must classify by what the artifact SAYS
it is (mtplx_runtime.json forge provenance), not what the folder is called.
Provenance outranks the resolved path, which outranks the ref as spelled.

NOTE: no family marker in any test name here — pytest bakes the test name
into tmp_path, and a "qwen38" in it would taint every resolved path.
"""

from __future__ import annotations

import json

import pytest

from mtplx.backends.descriptors import (
    QWEN3_NEXT_DESCRIPTOR,
    model_family_from_inspection,
    sampler_defaults_for_model,
)

# Stock control: a marker-free stock Qwen3 size. Whatever lane stock Qwen3
# rides today, the colliding sizes below must ride the same one.
STOCK_CONTROL = "Qwen/Qwen3-32B"


# ------------------------------------------------------------------- F21 fences


@pytest.mark.parametrize(
    "ref",
    [
        "Qwen3.8-27B",
        "qwen3-8-27b",
        "Qwen3_8-27B",
        "Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed",
        "qwen3.8",
    ],
)
def test_version_token_refs_keep_the_family(ref: str) -> None:
    assert (
        model_family_from_inspection(
            model_ref=ref, descriptor=QWEN3_NEXT_DESCRIPTOR
        )
        == "qwen3_8"
    )


@pytest.mark.parametrize(
    "ref",
    [
        "Qwen/Qwen3.8-Flash-Next",
        "Qwen/Qwen3.8-Flash-Next-FP8",
        "qwen3.8-flash-next",
        "Qwen4-Preview",
        "qwen4-moe",
    ],
)
def test_next_generation_preview_refs_do_not_claim_the_version_family(ref: str) -> None:
    # The Qwen4-generation previews carry "3.8" in their public names but
    # are a different architecture generation with their own contract; the
    # dense-27B qwen3_8 behavior contract must never claim them by name
    # (the F21 collision class, pre-armed for the Flash-Next release).
    assert (
        model_family_from_inspection(
            model_ref=ref, descriptor=QWEN3_NEXT_DESCRIPTOR
        )
        != "qwen3_8"
    )


@pytest.mark.parametrize(
    "ref",
    [
        "Qwen/Qwen3-8B",
        "qwen3-8b",
        "Qwen3-80B",
    ],
)
def test_stock_size_refs_stay_in_the_stock_lane(ref: str) -> None:
    stock = model_family_from_inspection(
        model_ref=STOCK_CONTROL, descriptor=QWEN3_NEXT_DESCRIPTOR
    )
    resolved = model_family_from_inspection(
        model_ref=ref, descriptor=QWEN3_NEXT_DESCRIPTOR
    )
    assert resolved != "qwen3_8"
    assert resolved == stock == "qwen3_6"  # the shared qwen lane default
    # Same holds with no descriptor at all.
    assert model_family_from_inspection(
        model_ref=ref
    ) == model_family_from_inspection(model_ref=STOCK_CONTROL)


def test_stock_size_ref_keeps_stock_sampler_defaults() -> None:
    # The published-benchmark blame vector: stock Qwen3-8B served with the
    # 3.8 thinking sampler (temp 1.0) looks noisy next to llama.cpp on the
    # identical model. Stock refs must keep the stock lane defaults.
    sampler = sampler_defaults_for_model(
        "Qwen/Qwen3-8B", None, QWEN3_NEXT_DESCRIPTOR
    )
    assert (sampler.temperature, sampler.top_p, sampler.top_k) == (0.6, 0.95, 20)


def test_stock_size_local_dir_stays_in_the_stock_lane(tmp_path) -> None:
    # A local download keeps the size boundary through the resolved-path lane.
    stock_dir = tmp_path / "Qwen3-8B"
    stock_dir.mkdir()
    assert (
        model_family_from_inspection(
            model_ref=str(stock_dir), descriptor=QWEN3_NEXT_DESCRIPTOR
        )
        == "qwen3_6"
    )


# ------------------------------------------------------------------- F22 fences


def _write_runtime_json(model_dir, trunk: str) -> None:
    model_dir.mkdir()
    (model_dir / "mtplx_runtime.json").write_text(
        json.dumps(
            {
                "base_trunk": trunk,
                "forge_provenance": {"forge_inputs": {"trunk_path": trunk}},
            }
        )
    )


def test_provenance_beats_misleading_dir_name(tmp_path) -> None:
    # Family is a behavior contract: a renamed dir classifies by what the
    # artifact says, not what the folder is called — in both directions.
    disguised = tmp_path / "Qwen3.6-27B-holding"
    _write_runtime_json(disguised, "/models/Qwen--Qwen3.8-27B")
    assert (
        model_family_from_inspection(
            model_ref=str(disguised), descriptor=QWEN3_NEXT_DESCRIPTOR
        )
        == "qwen3_8"
    )

    mislabeled = tmp_path / "Qwen3.8-27B-mislabeled"
    _write_runtime_json(mislabeled, "/models/Qwen--Qwen3.6-27B")
    assert (
        model_family_from_inspection(
            model_ref=str(mislabeled), descriptor=QWEN3_NEXT_DESCRIPTOR
        )
        == "qwen3_6"
    )


def test_marker_fallback_without_provenance(tmp_path) -> None:
    # No mtplx_runtime.json: the path marker still decides, for a local dir
    # and for a plain non-existent ref alike.
    local = tmp_path / "Qwen3.8-27B-local"
    local.mkdir()
    assert (
        model_family_from_inspection(
            model_ref=str(local), descriptor=QWEN3_NEXT_DESCRIPTOR
        )
        == "qwen3_8"
    )
    assert (
        model_family_from_inspection(
            model_ref="Youssofal/Qwen3.8-27B-MTPLX-Bare-Speed",
            descriptor=QWEN3_NEXT_DESCRIPTOR,
        )
        == "qwen3_8"
    )


# --------------------------------------------------- KV-quant policy per family


def test_kv_quant_policy_flash_next_offers_q8_through_its_own_int8_qsa_kv() -> None:
    # m1.18: Flash-Next's QSA attention has an int8 KV lane of its own
    # (QSAKVCacheQ8). The paged installer still never converts the QSA caches,
    # so q8 is served by the model's cache class, and q4 is not offered.
    from mtplx.backends.descriptors import kv_quant_policy_for_model

    policy = kv_quant_policy_for_model(
        model_ref="Qwen3.8-Flash-Next-MTPLX-Bare-Speed"
    )
    assert policy.supported
    assert policy.to_dict()["modes"] == ["off", "q8"]


def test_kv_quant_policy_27b_keeps_the_validated_modes() -> None:
    # The dense-27B contract keeps the validated q8/q4 paged lane (memo +
    # 2-pass q8 kernel, F29 routing) — the Flash-Next branch must not have
    # narrowed it.
    from mtplx.backends.descriptors import kv_quant_policy_for_model

    policy = kv_quant_policy_for_model(
        model_ref="Youssofal/Qwen3.8-27B-MTPLX-Bare-Speed"
    )
    assert policy.supported
    assert policy.to_dict()["modes"] == ["off", "q8", "q4"]
    assert policy.to_dict()["disabled_reason"] is None


def test_config_model_type_boot_sampler_is_the_family_law(tmp_path) -> None:
    """The qwen4_exp daemon boot sampler is the family law, whatever the stamp.

    After the app stopped forwarding the coding-target 0.6 pin (2026-08-28),
    a zero-flag daemon boot is the only thing standing between a Flash-Next
    launch and the generic 0.6 parser default. The daemon resolves the family
    sampler `mtplx serve` injects and /health reports, so a stamped pack, a
    pack without the stamp (the parser default used to stand) and a stale
    stamp all boot at 1.0/0.95/20. The folder name carries no family marker:
    config.json's model_type is what names the family here.
    """
    import json

    from mtplx.server.openai import _model_declared_sampler_defaults

    family_law = {"temperature": 1.0, "top_p": 0.95, "top_k": 20}
    model = tmp_path / "pack"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"model_type": "qwen4_exp"}))
    stale = {"temperature": 0.6, "top_p": 0.95, "top_k": 20}
    for runtime in ({"sampler": family_law}, {}, {"sampler": stale}):
        (model / "mtplx_runtime.json").write_text(json.dumps(runtime))
        assert _model_declared_sampler_defaults(str(model)) == family_law, runtime
