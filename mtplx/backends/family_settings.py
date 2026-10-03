"""Every model family owns its model-tuned settings (PX.0, 2026-09-18).

The shared profile keeps what is true for every model (memory plan, paging,
safety). A number that was tuned ON a model lives here, in that model's
block, with where it came from. Flash-Next (released 2026-08-26) was serving
on constants tuned on the dense 27B months earlier: the block records the
owner and the receipt of every such value, so a measured number has exactly
one place to go. The first two went in on 2026-09-18 (the wide prefill chunk
and the sparse attention crossover for wide forwards), both measured on an M5
Max and therefore stamped on tensor-unit GPUs only (``requires``).

HOW TO CHANGE A FLASH-NEXT VALUE (the main session's one-number edit):
edit the ``value=`` of the entry in ``QWEN4_EXP_SETTINGS`` below, set
``source=MEASURED`` and put the receipt id (cell folder or MEASUREMENTS.md
line) in ``receipt=``. Nothing else: the server stamps the matching env key
for that family at startup (``family_env_stamp``), an operator export still
wins, the user's ``--prefill-chunk-tokens`` still wins over everything, and
``tests/test_family_settings.py`` fails if a family serves a model-tuned key
without a receipt or an explicit "inherited, unmeasured" tag.

Two keys are never typed: KV bytes per token and the prewarm head geometry
are DERIVED from the served model's ``config.json`` (``derived_geometry``).

Pure data and arithmetic: no MLX import, no server import.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

MEASURED = "measured"
INHERITED = "inherited-27B-era, unmeasured"
FAMILY_OWN = "family-own, measured at one chunk width"
DERIVED = "derived from config.json"
SHARED_DEFAULT = "shared default, conservative"

# A capability the Mac must have before a value is stamped. A value measured
# on one GPU class is that class's value: everything else keeps the engine
# default (plan section 4A, rule 2).
TENSOR_UNIT_GPU = "tensor_unit_gpu"

# Sources that satisfy "this family has its own receipt or says it has none".
ACCOUNTED_SOURCES = frozenset({MEASURED, INHERITED, FAMILY_OWN, DERIVED})

# The model-tuned keys. A value that depends on which model is served belongs
# in this list; a family block must carry every one of them.
MODEL_TUNED_KEYS: tuple[str, ...] = (
    "prefill_chunk_tokens",
    "prefill_wide_chunk_tokens",
    "qsa_prefill_compile_rows",
    "qsa_prefill_wide_min_context",
    "qsa_prefill_score_mb",
    "prefill_cleanup_every",
    "decode_clear_every",
    "decode_clear_context_threshold",
    "kv_bytes_per_token",
    "prewarm_geometry",
    "depth_policy_priors",
    "copy_lane",
    "first_verify_reserve_tokens",
    "compiled_verify_depths",
)


@dataclass(frozen=True)
class ModelTunedSetting:
    key: str
    value: Any
    source: str
    receipt: str
    # Env keys the engine reads this value from (the stamp target). Empty for
    # values that are derived, or consumed through the launch resolver.
    env: tuple[str, ...] = ()
    # For a dict value: (field, env key) pairs, stamped field by field.
    env_by_field: tuple[tuple[str, str], ...] = ()
    # What the engine does when nothing is stamped. The stamp only carries a
    # key whose family value differs from this, so a block that matches the
    # engine default changes nothing at all.
    engine_default: Any = None
    # Capability this Mac must report before the value is stamped ("" = any
    # Mac). Without it the engine default serves, which is what shipped before
    # the value was measured.
    requires: str = ""

    def to_dict(self) -> dict[str, Any]:
        value = self.value
        if isinstance(value, (frozenset, set, tuple)):
            value = sorted(value) if isinstance(value, (frozenset, set)) else list(value)
        row = {
            "key": self.key,
            "value": value,
            "source": self.source,
            "receipt": self.receipt,
            "env": list(self.env),
        }
        if self.requires:
            row["requires"] = self.requires
        return row


def _block(*settings: ModelTunedSetting) -> dict[str, ModelTunedSetting]:
    return {setting.key: setting for setting in settings}


_FOLLOWS_CHUNK = "follows prefill_chunk_tokens"

_EV_PRIORS_27B = {
    # Server parser values (openai.py --adaptive-ev-*), recalibrated
    # 2026-08-07 on the 27B Speed-V2 at 13 to 18K. The CLI start parser
    # still carried the older 4.8 / 6.0 ms pair the server calls unreachable.
    "accept_priors": (0.92, 0.64, 0.32),
    "baseline_tok_s": 40.0,
    "draft_cost_ms": 2.0,
    "verify_extra_cost_ms": 1.5,
    "margin": 0.10,
    "min_extra_accept": 0.18,
    "warmup_cycles": 4,
    "explore_every": 32,
    "base_depth": 2,
    "min_depth": 1,
}

_COPY_LANE_ENV = (
    ("block_k", "MTPLX_CONTEXT_COPY_K"),
    ("probation_k", "MTPLX_CONTEXT_COPY_PROBATION_K"),
    ("ngram_min", "MTPLX_CONTEXT_COPY_NGMIN"),
    ("ngram_max", "MTPLX_CONTEXT_COPY_NGMAX"),
)
_COPY_LANE_27B = {
    # context_copy.py defaults: block K 24 (probation 8), n-gram key 6 to 10.
    # The EMA engage / suspend pair 0.5 / 0.35 is a literal in generation.py
    # (three sites near lines 10420, 10766, 11040), not an env: recorded here
    # so its owner is visible, changed there.
    "block_k": 24,
    "probation_k": 8,
    "ngram_min": 6,
    "ngram_max": 10,
    "ema_engage": 0.5,
    "ema_suspend": 0.35,
}

# Flash-Next's OWN copies: edit these two for Flash-Next. The 27B block and
# the engine defaults keep reading the 27B dictionaries above.
_EV_PRIORS_FLASH_NEXT = dict(_EV_PRIORS_27B)
_COPY_LANE_FLASH_NEXT = dict(_COPY_LANE_27B)


# --- Qwen3.8-Flash-Next (qwen4_exp) -----------------------------------------
#
# Every entry marked INHERITED was tuned on the dense 27B before this model
# existed. Each has its own cell in the 2026-09-18 plan (section 3.6 table);
# write the measured value here when the cell closes.
QWEN4_EXP_SETTINGS: dict[str, ModelTunedSetting] = _block(
    ModelTunedSetting(
        key="prefill_chunk_tokens",
        value=2048,
        source=INHERITED,
        receipt=(
            "set 2026-05-07/09 for the dense models (4956b5ae8, 7ac0740a1). "
            "Cell P2.2 closed on 2026-09-18 for tensor-unit GPUs only (see "
            "prefill_wide_chunk_tokens); this stays the plan every other Mac "
            "runs and the one a refused wide grant falls back to"
        ),
        env=("MTPLX_PREFILL_CHUNK_SIZE_DENSE", "MTPLX_PREFILL_CHUNK_SIZE_REPAGE"),
        engine_default=2048,
    ),
    ModelTunedSetting(
        key="prefill_wide_chunk_tokens",
        value=4096,
        source=MEASURED,
        receipt=(
            "2026-09-18, M5 Max 128 GB, cold prompts, one session "
            "(overnight-20260918 cells pfx-default4 against pfx-dense-2k): 4K "
            "810 to 1,250 tok/s, 16K 1,208 to 1,340, 64K 1,002 to 1,126 with "
            "the tail ladder; 8,192 ties on speed with 3.4 GB more peak. "
            "Granted per request by the prefill admission after its "
            "reclamation (prefill_safety.settle_wide_prefill_chunk), else the "
            "2,048 plan"
        ),
        env=("MTPLX_QWEN4_PREFILL_WIDE_CHUNK",),
        engine_default=0,
        requires=TENSOR_UNIT_GPU,
    ),
    ModelTunedSetting(
        key="qsa_prefill_compile_rows",
        value=_FOLLOWS_CHUNK,
        source=INHERITED,
        receipt=(
            "tied to the chunk width (qwen4_exp._qsa_prefill_compile_rows); "
            "an armed wide chunk earns the second captured width "
            "(_qsa_prefill_compile_row_set). Moves with P2.2"
        ),
        env=("MTPLX_QSA_PREFILL_COMPILE_ROWS",),
        engine_default=2048,
    ),
    ModelTunedSetting(
        key="qsa_prefill_wide_min_context",
        value=16384,
        source=MEASURED,
        receipt=(
            "2026-09-18, M5 Max: 4,096-row forwards on the masked dense lane "
            "run 1,132 tok/s at 16K of history, 817 at 20K, 695 at 28K, the "
            "block-sparse lane holds 1,100 to 1,165; armed from 8K it loses "
            "(1,317 against 1,433). Forwards under 2,048 rows keep the 32,768 "
            "crossover their own A/B chose"
        ),
        env=("MTPLX_QSA_PREFILL_WIDE_MIN_CONTEXT",),
        engine_default=0,
        requires=TENSOR_UNIT_GPU,
    ),
    ModelTunedSetting(
        key="qsa_prefill_score_mb",
        value=128,
        source=INHERITED,
        receipt="origin not stated in code (mlx-serve uses 256 MB). Moves with P2.2",
        env=("MTPLX_QSA_PREFILL_SCORE_MB",),
        engine_default=128,
    ),
    ModelTunedSetting(
        key="prefill_cleanup_every",
        value="auto",
        source=INHERITED,
        receipt=(
            "auto = every 4 chunks dense, 2 repage: A/B 2026-07-05 on the 27B "
            "(generation._prefill_chunk_cache_cleanup_every). Cell P2.3"
        ),
        env=("MTPLX_PREFILL_CHUNK_CACHE_CLEANUP_EVERY",),
        engine_default="auto",
    ),
    ModelTunedSetting(
        key="decode_clear_every",
        value=1024,
        source=INHERITED,
        receipt=(
            "synchronize + clear every 1,024 tokens: 2026-07-16 at 33K on the "
            "27B; Flash-Next rounds are faster, so one sync is a larger share. "
            "Cell P5.3"
        ),
        env=("MTPLX_CLEAR_CACHE_EVERY_LONG_CONTEXT",),
        engine_default=1024,
    ),
    ModelTunedSetting(
        key="decode_clear_context_threshold",
        value=16384,
        source=INHERITED,
        receipt="engages from 16,384 live tokens: same 27B receipt as decode_clear_every. Cell P5.3",
        env=("MTPLX_CLEAR_CACHE_EVERY_CONTEXT_THRESHOLD",),
        engine_default=16384,
    ),
    ModelTunedSetting(
        key="kv_bytes_per_token",
        value=DERIVED,
        source=DERIVED,
        receipt="12 full-attention layers x K+V x 2 KV heads x 256 x bf16 = 24,576 (memory_plan)",
    ),
    ModelTunedSetting(
        key="prewarm_geometry",
        value=DERIVED,
        source=DERIVED,
        receipt="2 KV heads, 24 query heads, head dim 256 from config.json (the hard-coded prewarm is the 27B's 4 / 24 / 256)",
    ),
    ModelTunedSetting(
        key="depth_policy_priors",
        value=_EV_PRIORS_FLASH_NEXT,
        source=INHERITED,
        receipt=(
            "accept 0.92 / 0.64 / 0.32, 40 tok/s baseline, 2.0 / 1.5 ms costs: "
            "27B Speed-V2 at 13 to 18K (2026-08-07). Unused while "
            "compiled_verify_depths is {3}. Cell P3.A"
        ),
    ),
    ModelTunedSetting(
        key="copy_lane",
        value=_COPY_LANE_FLASH_NEXT,
        source=INHERITED,
        receipt="6-gram key, block 8 to 24, EMA 0.5 / 0.35: tuned on the 27B on 2026-08-25 and 08-28. Cell P3.7(b)",
        env_by_field=_COPY_LANE_ENV,
        engine_default=_COPY_LANE_27B,
    ),
    ModelTunedSetting(
        key="first_verify_reserve_tokens",
        value=1024,
        source=FAMILY_OWN,
        receipt=(
            "fixed-M4 initial reserve 1,024 then doubling to 16,384 per step "
            "(graphbank._fixed_m4_initial_growth_reserve), measured at "
            "2,048-row chunks only. Cell 5.3"
        ),
        env=("MTPLX_COMPILED_VERIFY_GROWTH_RESERVE",),
        engine_default=1024,
    ),
    ModelTunedSetting(
        key="compiled_verify_depths",
        value=frozenset({3}),
        source=FAMILY_OWN,
        receipt=(
            "only verify width 4 (draft depth 3) has a compiled route "
            "(graphbank.install_fixed_m4); every other depth runs the eager "
            "verifier. Widen this set when more widths are compiled and the "
            "adaptive depth policy lifts by itself"
        ),
    ),
)


# --- Qwen3.8-27B (qwen3_8) ---------------------------------------------------
#
# Today's values with their dates and receipts. Nothing here moves tonight.
QWEN3_8_SETTINGS: dict[str, ModelTunedSetting] = _block(
    ModelTunedSetting(
        key="prefill_chunk_tokens",
        value=2048,
        source=MEASURED,
        receipt="2026-05-07/09 (4956b5ae8, 7ac0740a1), profiles.SUSTAINED_PREFILL_ENV",
        env=("MTPLX_PREFILL_CHUNK_SIZE_DENSE", "MTPLX_PREFILL_CHUNK_SIZE_REPAGE"),
        engine_default=2048,
    ),
    ModelTunedSetting(
        key="prefill_wide_chunk_tokens",
        value=None,
        source=INHERITED,
        receipt=(
            "no wide-chunk receipt on this family; the 2026-09-18 pair moved "
            "+5% at 4K and +1% at 16K from the tail ladder alone. Its own "
            "width ladder is queue item 5a2"
        ),
    ),
    ModelTunedSetting(
        key="qsa_prefill_compile_rows",
        value=None,
        source=DERIVED,
        receipt="not applicable: no QSA layers in this family (config.json)",
    ),
    ModelTunedSetting(
        key="qsa_prefill_wide_min_context",
        value=None,
        source=DERIVED,
        receipt="not applicable: no QSA layers in this family (config.json)",
    ),
    ModelTunedSetting(
        key="qsa_prefill_score_mb",
        value=None,
        source=DERIVED,
        receipt="not applicable: no QSA layers in this family (config.json)",
    ),
    ModelTunedSetting(
        key="prefill_cleanup_every",
        value="auto",
        source=MEASURED,
        receipt=(
            "auto = every 4 chunks dense, 2 repage: A/B 2026-07-05, fresh "
            "daemon per arm, max fans (16k 565 to 682, 128k 294 to 315 tok/s)"
        ),
        env=("MTPLX_PREFILL_CHUNK_CACHE_CLEANUP_EVERY",),
        engine_default="auto",
    ),
    ModelTunedSetting(
        key="decode_clear_every",
        value=1024,
        source=MEASURED,
        receipt="2026-07-16 at 33K: 256 to 1,024 removed a 3.8 percent decode tax",
        env=("MTPLX_CLEAR_CACHE_EVERY_LONG_CONTEXT",),
        engine_default=1024,
    ),
    ModelTunedSetting(
        key="decode_clear_context_threshold",
        value=16384,
        source=MEASURED,
        receipt="lowered 98,304 to 16,384 for the 16 to 40K agent regime (generation._clear_cache_every)",
        env=("MTPLX_CLEAR_CACHE_EVERY_CONTEXT_THRESHOLD",),
        engine_default=16384,
    ),
    ModelTunedSetting(
        key="kv_bytes_per_token",
        value=DERIVED,
        source=DERIVED,
        receipt="16 full-attention layers x K+V x 4 KV heads x 256 x bf16 = 65,536 (memory_plan)",
    ),
    ModelTunedSetting(
        key="prewarm_geometry",
        value=DERIVED,
        source=DERIVED,
        receipt="4 KV heads, 24 query heads, head dim 256 from config.json",
    ),
    ModelTunedSetting(
        key="depth_policy_priors",
        value=_EV_PRIORS_27B,
        source=MEASURED,
        receipt="M5 Max, 27B Speed-V2, 13 to 18K, recalibrated 2026-08-07 (openai.py --adaptive-ev-*)",
    ),
    ModelTunedSetting(
        key="copy_lane",
        value=_COPY_LANE_27B,
        source=MEASURED,
        receipt="2026-08-25 and 2026-08-28 copy-lane receipts (context_copy.py)",
        env_by_field=_COPY_LANE_ENV,
        engine_default=_COPY_LANE_27B,
    ),
    ModelTunedSetting(
        key="first_verify_reserve_tokens",
        value=512,
        source=MEASURED,
        receipt="2.4.0 regression fix 2026-07-31: sized for a 40 to 500 token agent tool round",
        env=("MTPLX_COMPILED_VERIFY_GROWTH_RESERVE",),
        engine_default=512,
    ),
    ModelTunedSetting(
        key="compiled_verify_depths",
        value=None,
        source=MEASURED,
        receipt="the generic compiled bank serves every verify length up to 6 rows, so every depth this family allows is compiled",
    ),
)


FAMILY_SETTINGS: dict[str, dict[str, ModelTunedSetting]] = {
    "qwen4_exp": QWEN4_EXP_SETTINGS,
    "qwen3_8": QWEN3_8_SETTINGS,
}


def family_settings(family: str | None) -> dict[str, ModelTunedSetting] | None:
    """The family's own block, or None for a family that has none yet."""

    return FAMILY_SETTINGS.get(str(family or ""))


def resolved_value(family: str | None, key: str) -> Any:
    """The value a family serves for ``key`` (placeholders resolved)."""

    block = family_settings(family)
    if block is None or key not in block:
        return None
    value = block[key].value
    if value == _FOLLOWS_CHUNK:
        return block["prefill_chunk_tokens"].value
    return value


def compiled_verify_depths(family: str | None) -> frozenset[int] | None:
    """Draft depths whose verify has a compiled route; None means all."""

    value = resolved_value(family, "compiled_verify_depths")
    return frozenset(int(item) for item in value) if value else None


def family_env_stamp(
    family: str | None, *, capabilities: Iterable[str] = ()
) -> dict[str, str]:
    """Env keys the server stamps for this family at startup.

    Only a value that differs from the engine's built-in default is carried,
    so a block that matches today's defaults stamps nothing and cannot change
    behavior. A value that names a capability (``requires``) is carried only
    when the caller reports it in ``capabilities``, so a Mac without it keeps
    the engine default. The caller keeps the usual rule: an operator export
    wins.
    """

    block = family_settings(family)
    if block is None:
        return {}
    have = frozenset(str(item) for item in capabilities)
    stamp: dict[str, str] = {}
    for setting in block.values():
        if setting.requires and setting.requires not in have:
            continue
        if setting.env_by_field and isinstance(setting.value, dict):
            defaults = setting.engine_default if isinstance(setting.engine_default, dict) else {}
            for field_name, env_key in setting.env_by_field:
                field_value = setting.value.get(field_name)
                if field_value is not None and field_value != defaults.get(field_name):
                    stamp[env_key] = str(field_value)
            continue
        if not setting.env:
            continue
        value = resolved_value(family, setting.key)
        if value is None or value == setting.engine_default:
            continue
        for env_key in setting.env:
            stamp[env_key] = str(value)
    return stamp


def derived_geometry(config: dict | None) -> dict[str, Any]:
    """KV bytes per token and the prewarm head geometry, from config.json."""

    from mtplx.memory_plan import dense_kv_bytes_per_token_from_config

    text: dict[str, Any] = {}
    if isinstance(config, dict):
        nested = config.get("text_config")
        text = nested if isinstance(nested, dict) else config
    kv_heads = text.get("num_key_value_heads")
    query_heads = text.get("num_attention_heads")
    head_dim = text.get("head_dim")
    if (
        head_dim is None
        and isinstance(text.get("hidden_size"), int)
        and isinstance(query_heads, int)
        and query_heads > 0
    ):
        head_dim = text["hidden_size"] // query_heads
    return {
        "kv_bytes_per_token": dense_kv_bytes_per_token_from_config(config),
        "kv_heads": kv_heads if isinstance(kv_heads, int) else None,
        "query_heads": query_heads if isinstance(query_heads, int) else None,
        "head_dim": head_dim if isinstance(head_dim, int) else None,
    }


def explain_rows(family: str | None, config: dict | None = None) -> list[dict[str, Any]]:
    """Rows for ``mtplx doctor --explain``: key, value, source, receipt."""

    block = family_settings(family)
    if block is None:
        return [
            {
                "key": key,
                "value": "shared profile value",
                "source": SHARED_DEFAULT,
                "receipt": "this family has no block of its own yet",
            }
            for key in MODEL_TUNED_KEYS
        ]
    geometry = derived_geometry(config) if config is not None else {}
    rows = []
    for key in MODEL_TUNED_KEYS:
        row = block[key].to_dict()
        row["value"] = resolved_value(family, key)
        if isinstance(row["value"], frozenset):
            row["value"] = sorted(row["value"])
        if key == "kv_bytes_per_token" and geometry.get("kv_bytes_per_token"):
            row["value"] = geometry["kv_bytes_per_token"]
        if key == "prewarm_geometry" and geometry.get("kv_heads"):
            row["value"] = {
                name: geometry[name] for name in ("kv_heads", "query_heads", "head_dim")
            }
        rows.append(row)
    return rows
