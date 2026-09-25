"""Every trunk forward of a Flash-Next image request ropes at the image positions.

``qwen4_exp`` attention reads the request's M-RoPE state (the prompt position
table and the decode delta) from a context variable. With nothing armed it
ropes at the raw cache index. Every row past the prompt of an image request
belongs at ``index + delta``.

Builds 2.10.1 to 2.11.3 opened that scope by hand around ONE forward of the
decode loop, the main verify. Copy rounds and their single-row repairs, the
repair forward, the lazy bonus commit and the final pending commit roped at
the raw index, so the K rows and the pooled indexer keys they wrote sat
``|delta|`` positions away from every other row of the conversation, in the
live cache and in every banked entry (one 1,024-token image: about 990).

A tiny random qwen4_exp model with its MTP head runs ``generate_mtpk`` end to
end here, with a synthetic image whose delta is -12. No pack and no tower.

* the spy tests: every trunk forward sees the request's position state, on
  every lane the tiny model can reach, and the draft head is left alone;
* the bit-for-bit tests: every K row in the final cache equals the row an
  independent rope at the request's position produces from the same layer
  input, and the whole final state equals a reference run whose runtime opens
  the scope around every trunk forward itself;
* the structural test: every trunk forward call site in ``generate_mtpk``
  sits inside the one scope helper, which also covers the capture lane the
  tiny model cannot run;
* the bank tests: an entry keyed the way older builds keyed it cannot be
  restored past its image, and text entries restore exactly as before.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from dataclasses import asdict
from pathlib import Path
from typing import Any, NamedTuple

import mlx.core as mx
import numpy as np
import pytest

from mtplx import generation
from mtplx.attention_context import (
    current_attention_phase,
    current_model_forward_kind,
    vision_rope,
    vision_rope_state,
)
from mtplx.generation import generate_mtpk
from mtplx.models.qwen4_exp import (
    Attention,
    Model,
    ModelArgs,
    QSACache,
    Qwen4ExpMTP,
    TextArgs,
    _apply_partial_rope,
    _mrope_cos_sin,
)
from mtplx.mtp_patch import MTPContract, validate_mtp_support
from mtplx.runtime import MTPLXRuntime
from mtplx.sampling import SamplerConfig
from mtplx.session_bank import SessionBank
from mtplx.vision.mrope import build_mrope_positions
from mtplx.vision.splice import _BANK_KEY_FLAG, VisionSplice, vision_bank_key_ids

VOCAB = 64
PAD = VOCAB - 1
HIDDEN = 64
# One 8 x 8 patch image: 4 x 4 tokens after the 2 x 2 merge, which occupy
# max(1, 4, 4) = 4 positions, so every later row sits 12 positions early.
GRID = (1, 8, 8)
IMAGE_TOKENS = 16
DELTA = -12
IMAGE_DIGEST = 0xC0FFEE
# The prompt ends on ANCHOR and holds the pair (ANCHOR, v) for every text id
# v. Whatever token the model samples first, the last two tokens of the
# stream then occur earlier in the prompt, so the first decode round is a
# copy round (with the n-gram floor lowered to 2 below).
ANCHOR = 5
HEAD = [3, 4, 6, 7]
COPY_BLOCK = 8  # the probation block length
MAX_TOKENS = 24

GREEDY = SamplerConfig(temperature=0.0, top_p=1.0, top_k=0)
SAMPLED = SamplerConfig(temperature=0.6, top_p=0.95, top_k=20)

_LANE_ENV = (
    "MTPLX_CONTEXT_COPY",
    "MTPLX_CONTEXT_COPY_BATCHED",
    "MTPLX_CONTEXT_COPY_TARGET_PREFIX",
    "MTPLX_CONTEXT_COPY_K",
    "MTPLX_CONTEXT_COPY_PROBATION_K",
    "MTPLX_CONTEXT_COPY_NGMAX",
    "MTPLX_CONTEXT_COPY_MINEXT",
    "MTPLX_RAMP_ENABLED",
    "MTPLX_FAMILY_CAPTURE_COMMIT",
    "MTPLX_LAZY_BONUS_VERIFY",
    "MTPLX_LAZY_BONUS_VERIFY_MIN_DEPTH",
    "MTPLX_LAZY_TARGET_DISTRIBUTIONS",
    "MTPLX_LAZY_VERIFY_LOGITS",
    "MTPLX_SKIP_VERIFY_SNAPSHOT",
    "MTPLX_COMPILED_VERIFY",
    "MTPLX_STATE_REBASE_EVERY",
    "MTPLX_QWEN4_VISION_QSA",
    "MTPLX_QSA_MTP_PRECOMPUTE",
    "MTPLX_DROP_EVENTS",
    "MTPLX_MTP_HISTORY_POLICY",
    "MTPLX_MTP_HISTORY_LIVE_RESET_THRESHOLD",
    "MTPLX_MTP_POSITION_MODE",
)


@pytest.fixture(autouse=True)
def _lanes(monkeypatch):
    for name in _LANE_ENV:
        monkeypatch.delenv(name, raising=False)
    # Copy rounds are on by default except on the M1 GPU family, so turn them
    # on explicitly; the n-gram floor is lowered so a 145-token prompt can
    # force one.
    monkeypatch.setenv("MTPLX_CONTEXT_COPY", "1")
    monkeypatch.setenv("MTPLX_CONTEXT_COPY_NGMIN", "2")


def _tiny_args() -> TextArgs:
    return TextArgs(
        hidden_size=HIDDEN,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        vocab_size=VOCAB,
        layer_types=["linear_attention", "full_attention"],
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        # mlx-lm's GPU gated-delta kernel needs Dk >= 32.
        linear_key_head_dim=32,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=32,
        hc_count=2,
        hc_lowrank=16,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=32,
        indexer_budget=8,
        indexer_compress_ratio=2,
        ple_layer_ids=[],
        ple_embed_dim=HIDDEN,
        ngram_vocab_size_base=128,
        heads_per_ngram=2,
        # The pack's contract, scaled down: interleaved axes over the four
        # rotary pairs of a 32-wide head at partial factor 0.25.
        rope_parameters={
            "mrope_interleaved": True,
            "mrope_section": [2, 1, 1],
            "partial_rotary_factor": 0.25,
            "rope_theta": 10_000.0,
            "rope_type": "default",
        },
        eos_token_id=0,
    )


class _Tokenizer:
    eos_token_id = None
    eos_token_ids: set[int] = set()  # noqa: RUF012

    def decode(self, tokens, **_kwargs):
        return " ".join(str(int(token)) for token in tokens)


def _build_model(seed: int = 7) -> Model:
    mx.random.seed(seed)
    model = Model(ModelArgs(model_type="qwen4_exp", text_config=asdict(_tiny_args())))
    model.language_model.mtp = Qwen4ExpMTP(model.language_model.args)
    model.eval()
    mx.eval(model.parameters())
    assert validate_mtp_support(model)
    return model


@pytest.fixture()
def model() -> Model:
    return _build_model()


def _runtime(model: Model) -> MTPLXRuntime:
    return MTPLXRuntime(
        model=model,
        tokenizer=_Tokenizer(),
        model_path=Path("."),
        mtp_enabled=True,
        contract=MTPContract(),
    )


def _prompt() -> list[int]:
    pairs: list[int] = []
    for token in range(VOCAB - 1):  # every text id; never the image pad
        pairs.extend([ANCHOR, token])
    return [*HEAD, *([PAD] * IMAGE_TOKENS), *pairs, ANCHOR]


PROMPT = _prompt()
FIRST_IMAGE = PROMPT.index(PAD)


def _table(prompt=PROMPT) -> np.ndarray:
    built = build_mrope_positions(
        list(prompt), image_token_id=PAD, image_grids=[GRID], spatial_merge_size=2
    )
    assert built is not None
    table, delta = built
    assert delta == DELTA
    return table


def _splice(prompt=PROMPT, *, roped: bool = True) -> VisionSplice:
    """The serve layer's splice for ``prompt``. A fresh one per run: prefill
    consumes its row queue, and the table covers exactly that prompt."""
    rows = mx.array(
        np.random.default_rng(11).standard_normal((IMAGE_TOKENS, HIDDEN)).astype(np.float32)
    )
    return VisionSplice(
        image_pad_token_id=PAD,
        embeddings=rows,
        image_digests=(IMAGE_DIGEST,),
        pad_counts=(IMAGE_TOKENS,),
        image_grids=(GRID,),
        mrope_table=mx.array(_table(prompt)) if roped else None,
        mrope_delta=DELTA if roped else 0,
    )


def _generate(rt, *, splice, prompt=PROMPT, sampler=GREEDY, max_tokens=MAX_TOKENS, **kwargs):
    return generate_mtpk(
        rt,
        list(prompt),
        max_tokens=max_tokens,
        sampler=sampler,
        draft_sampler=sampler,
        speculative_depth=3,
        seed=5,
        mtp_cache_policy="persistent",
        mtp_history_policy="committed",
        verify_strategy="batched",
        stop_token_ids=set(),
        vision_splice=splice,
        capture_final_state=True,
        **kwargs,
    )


# -- spies ----------------------------------------------------------------------


class _Seen(NamedTuple):
    surface: str  # "trunk" or "draft"
    phase: str
    kind: str
    rope: Any  # vision_rope_state() at call time
    rows: int
    offset: int | None  # the QSA cache offset the forward starts at


def _qsa_offset(cache) -> int | None:
    for entry in cache or ():
        if isinstance(entry, QSACache):
            return int(entry.offset)
    return None


def _spy_runtime(model: Model) -> tuple[MTPLXRuntime, list[_Seen]]:
    """A runtime whose trunk and draft-head surfaces record what they see."""
    rt = _runtime(model)
    seen: list[_Seen] = []

    def spy(name: str, surface: str) -> None:
        real = getattr(rt, name)

        def spied(*args, **kwargs):
            if surface == "trunk":
                ids = args[0]
                rows, offset = int(ids.shape[1]), _qsa_offset(kwargs.get("cache"))
            else:
                rows, offset = int(args[0].shape[1]), _qsa_offset(kwargs.get("mtp_cache"))
            seen.append(
                _Seen(
                    surface,
                    current_attention_phase(),
                    current_model_forward_kind(),
                    vision_rope_state(),
                    rows,
                    offset,
                )
            )
            return real(*args, **kwargs)

        setattr(rt, name, spied)

    spy("forward_ar", "trunk")
    spy("forward_ar_capture", "trunk")
    spy("draft_mtp", "draft")
    spy("update_mtp_cache", "draft")
    return rt, seen


def _oracle_draft_head(model: Model, full_sequence: list[int]) -> None:
    """Make every draft correct, so rounds accept all of their drafts.

    The real draft head still runs (its cache, its rope calls); only the
    logits it returns become a one-hot on the token the target will produce.
    Draft row r pairs hidden r with token r + 1 and predicts token r + 2.
    """
    real = model.mtp_forward

    def oracle(hidden_states, next_token_ids, **kwargs):
        cache = kwargs.get("mtp_cache")
        row = int(cache[0].offset) if cache else 0
        out = real(hidden_states, next_token_ids, **kwargs)
        logits, hidden = out if kwargs.get("return_hidden") else (out, None)
        targets = [
            full_sequence[min(row + i + 2, len(full_sequence) - 1)]
            for i in range(int(logits.shape[1]))
        ]
        peaked = mx.zeros_like(logits)
        peaked[0, mx.arange(len(targets)), mx.array(targets)] = 50.0
        return (peaked, hidden) if kwargs.get("return_hidden") else peaked

    model.mtp_forward = oracle


LANES = ("family_commit", "rollback_reforward", "lazy_bonus_commit")


def _arm_lane(lane: str, monkeypatch, model: Model) -> None:
    """Steer the request onto one set of decode-loop trunk forwards.

    family_commit      the Flash-Next serving default: a rejected window is
                       committed by the model (no re-forward), so the copy
                       round's kept rows and the final commit are the rows at
                       stake.
    rollback_reforward the engine default without the family env: the copy
                       round cannot commit a prefix (single-row repair) and
                       every rejected round re-forwards its committed tokens.
    lazy_bonus_commit  MTPLX_LAZY_BONUS_VERIFY with every draft accepted: the
                       omitted last draft is committed by its own forward.
    """
    if lane == "rollback_reforward":
        monkeypatch.setenv("MTPLX_FAMILY_CAPTURE_COMMIT", "0")
        return
    monkeypatch.setenv("MTPLX_FAMILY_CAPTURE_COMMIT", "1")
    if lane == "lazy_bonus_commit":
        monkeypatch.setenv("MTPLX_LAZY_BONUS_VERIFY", "1")
        # The continuation the target produces, from a run whose runtime
        # scopes every trunk forward itself (so it holds on any tree).
        reference = _generate(_reference_runtime(model, _splice()), splice=_splice())
        _oracle_draft_head(model, PROMPT + list(reference.tokens))


def _reference_runtime(model: Model, splice: VisionSplice) -> MTPLXRuntime:
    """Opens the request's position scope around EVERY trunk forward itself.

    Whatever scope generation opens or forgets, each trunk forward of this
    runtime ropes at the request's positions; the draft head is untouched. It
    runs the same forwards in the same order and shapes as the run under
    test, so the two final states are comparable bit for bit.
    """
    rt = _runtime(model)
    real = rt.forward_ar

    def scoped(*args, **kwargs):
        with vision_rope(splice.mrope_table, splice.mrope_delta):
            return real(*args, **kwargs)

    rt.forward_ar = scoped
    return rt


def _lane_receipts(lane: str, out, trunk: list[_Seen]) -> None:
    """The request really ran the forwards this lane is about."""
    decode = [call for call in trunk if call.phase == "decode_verify"]
    events = [event for event in out.stats.events if isinstance(event, dict)]
    n = len(PROMPT)
    assert out.tokens[0] != PAD, (
        "the first sampled token is the image pad, the one id with no pair in "
        "the prompt, so no copy round was forced: build the model from another seed"
    )
    # The first decode round is the forced copy round: primary + block.
    assert decode[0].rows == 1 + COPY_BLOCK and decode[0].offset == n
    assert any(call.kind == "target_verify" and call.rows <= 4 for call in decode)
    # The final pending commit: one row, at the last committed index.
    assert out.final_state.safe_to_commit
    assert (decode[-1].rows, decode[-1].offset) == (1, n + len(out.tokens) - 1)
    if lane == "family_commit":
        assert out.stats.context_copy_rounds >= 1
        assert any(event.get("capture_repair") == "captured_prefix_commit" for event in events)
        assert not any(call.kind == "repair" for call in decode)
    elif lane == "rollback_reforward":
        assert {"disabled": "no_per_position_commit"} in [
            event.get("context_copy") for event in events
        ]
        assert (decode[1].rows, decode[1].offset) == (1, n)  # the single-row repair
        assert any(call.kind == "repair" for call in decode)
    else:
        assert out.stats.lazy_bonus_verify_calls >= 1
        assert any(
            "bonus_commit_forward_s" in (event.get("lazy_bonus_verify") or {})
            for event in events
        )


# -- 1. every trunk forward sees the scope; the draft head is left alone ----------


def test_scope_helper_arms_the_phase_and_the_positions():
    assert vision_rope_state() is None
    with generation._decode_trunk_scope(None):
        # A text request: the verify phase, and nothing else.
        assert current_attention_phase() == "decode_verify"
        assert vision_rope_state() is None
    splice = _splice()
    with generation._decode_trunk_scope(splice):
        assert current_attention_phase() == "decode_verify"
        table, delta = vision_rope_state()
        assert table is splice.mrope_table and delta == DELTA
    assert vision_rope_state() is None
    assert current_attention_phase() == "unknown"
    # An image request whose table could not be built ropes sequentially on
    # every forward, as before: no state to arm.
    with generation._decode_trunk_scope(_splice(roped=False)):
        assert vision_rope_state() is None


@pytest.mark.parametrize("lane", LANES)
def test_every_trunk_forward_of_an_image_request_sees_the_position_scope(
    lane, model, monkeypatch
):
    _arm_lane(lane, monkeypatch, model)
    rt, seen = _spy_runtime(model)
    splice = _splice()
    out = _generate(rt, splice=splice)
    assert len(out.tokens) == MAX_TOKENS

    trunk = [call for call in seen if call.surface == "trunk"]
    _lane_receipts(lane, out, trunk)
    unscoped = [call for call in trunk if call.rope is None]
    assert not unscoped, (
        f"{len(unscoped)} of {len(trunk)} trunk forwards roped at the raw cache "
        f"index: {[(call.phase, call.kind, call.rows, call.offset) for call in unscoped]}"
    )
    for call in trunk:
        table, delta = call.rope
        assert table is splice.mrope_table and delta == DELTA
    # The scope closes with the request.
    assert vision_rope_state() is None and current_attention_phase() == "unknown"


@pytest.mark.parametrize("lane", LANES)
def test_the_decode_scope_never_covers_a_draft_head_forward(lane, model, monkeypatch):
    """What this fix deliberately leaves as it was.

    The draft head shares the attention class, so a scope around the whole
    request would re-rope its decode rows too. It cannot break exactness (the
    verify decides every token); its positions move acceptance only, and they
    change in their own measured step. Prefill builds the draft history
    inside the prompt scope, as before.
    """
    _arm_lane(lane, monkeypatch, model)
    rt, seen = _spy_runtime(model)
    _generate(rt, splice=_splice())
    draft = [call for call in seen if call.surface == "draft"]
    in_prefill = [call for call in draft if call.phase == "prefill"]
    in_decode = [call for call in draft if call.phase != "prefill"]
    assert in_prefill and in_decode
    assert all(call.rope is not None for call in in_prefill)
    assert all(call.rope is None for call in in_decode)
    assert all(call.phase != "decode_verify" for call in in_decode)


def test_a_text_request_never_sees_a_position_state(model, monkeypatch):
    _arm_lane("rollback_reforward", monkeypatch, model)
    rt, seen = _spy_runtime(model)
    text_prompt = [token for token in PROMPT if token != PAD]
    out = _generate(rt, splice=None, prompt=text_prompt)
    assert len(out.tokens) == MAX_TOKENS
    decode = [call for call in seen if call.surface == "trunk" and call.phase == "decode_verify"]
    assert decode and out.stats.context_copy_rounds + out.stats.verify_calls >= 2
    assert all(call.rope is None for call in seen)


# -- 2. bit for bit ---------------------------------------------------------------


def _request_positions(start: int, rows: int) -> mx.array:
    """[3, rows] positions of rows start.. from the request's own definition:
    the table inside the prompt, index + delta on all three axes after it."""
    table = _table()
    index = np.arange(start, start + rows)
    positions = np.broadcast_to(index + DELTA, (3, rows)).copy()
    inside = index < table.shape[1]
    positions[:, inside] = table[:, index[inside]]
    return mx.array(positions.astype(np.int32))


def _rope_keys(attn: Attention, x: mx.array, positions: mx.array) -> mx.array:
    """The K rows ``attn`` writes for layer input ``x`` at ``positions``:
    the layer's own projection and norm, then a rope built here."""
    batch, rows, _ = x.shape
    keys = attn.k_norm(attn.k_proj(x).reshape(batch, rows, attn.n_kv_heads, -1))
    cos, sin = _mrope_cos_sin(positions, attn._inv_freq, attn._mrope_axes)
    return _apply_partial_rope(keys, cos, sin).transpose(0, 2, 1, 3)


def _record_trunk_attention(model: Model, monkeypatch) -> list[tuple[Attention, mx.array, int]]:
    """(layer, layer input, first cache index) of every trunk attention call."""
    trunk = {id(layer.self_attn) for layer in model.layers if not layer.is_linear}
    calls: list[tuple[Attention, mx.array, int]] = []
    real = Attention.__call__

    def recording(self, x, cache):
        if id(self) in trunk:
            calls.append((self, x, int(cache.offset)))
        return real(self, x, cache)

    monkeypatch.setattr(Attention, "__call__", recording)
    return calls


def _cache_leaves(cache) -> list[mx.array]:
    leaves: list[mx.array] = []
    for entry in cache:
        state = entry.state
        leaves.extend(leaf for leaf in state if isinstance(leaf, mx.array))
    return leaves


@pytest.mark.parametrize(
    ("lane", "sampler"),
    [
        ("family_commit", GREEDY),
        ("family_commit", SAMPLED),
        ("rollback_reforward", GREEDY),
        ("rollback_reforward", SAMPLED),
        # The oracle draft head is a greedy construction.
        ("lazy_bonus_commit", GREEDY),
    ],
    ids=lambda value: value if isinstance(value, str) else f"t{value.temperature}",
)
def test_every_cached_row_is_roped_at_the_request_position_bit_for_bit(
    lane, sampler, model, monkeypatch
):
    _arm_lane(lane, monkeypatch, model)
    calls = _record_trunk_attention(model, monkeypatch)
    out = _generate(_runtime(model), splice=_splice(), sampler=sampler)
    assert len(out.tokens) == MAX_TOKENS and out.final_state.safe_to_commit
    # A copy round ran (the rollback lane runs one, then retires the lane).
    assert (
        out.stats.context_copy_rounds >= 1
        or out.stats.context_copy_disabled_reason == "no_per_position_commit"
    )
    total = len(PROMPT) + len(out.tokens)

    qsa = [entry for entry in out.final_state.final_trunk_cache if isinstance(entry, QSACache)]
    attns = [layer.self_attn for layer in model.layers if not layer.is_linear]
    assert len(qsa) == len(attns) == 1
    cache, attn = qsa[0], attns[0]
    assert cache.offset == total

    # The cache is positional: the last forward that covered an index wrote
    # the row that is there now (an earlier writer was trimmed away first).
    writer: dict[int, tuple[mx.array, int]] = {}
    for layer, x, start in calls:
        assert layer is attn
        for row in range(int(x.shape[1])):
            writer[start + row] = (x, start)
    assert set(range(total)) <= set(writer)

    expected_rows: dict[int, mx.array] = {}
    at_raw_index: dict[int, mx.array] = {}
    computed: dict[int, tuple[mx.array, mx.array]] = {}
    for index in range(total):
        x, start = writer[index]
        if id(x) not in computed:
            rows = int(x.shape[1])
            raw = mx.broadcast_to(mx.arange(start, start + rows, dtype=mx.int32)[None], (3, rows))
            computed[id(x)] = (
                _rope_keys(attn, x, _request_positions(start, rows)),
                _rope_keys(attn, x, raw),
            )
        expected, raw_roped = computed[id(x)]
        expected_rows[index] = expected[:, :, index - start, :]
        at_raw_index[index] = raw_roped[:, :, index - start, :]

    keys = cache.kv.keys[:, :, :total, :]
    wrong = [
        index
        for index in range(total)
        if not mx.array_equal(keys[:, :, index, :], expected_rows[index]).item()
    ]
    assert not wrong, (
        f"{len(wrong)} K rows are not at the request's position; decode rows among "
        f"them: {[index - len(PROMPT) for index in wrong if index >= len(PROMPT)]}"
    )
    # The comparison can tell the two positions apart: no row past the image
    # equals what the raw index would have produced.
    assert not any(
        mx.array_equal(keys[:, :, index, :], at_raw_index[index]).item()
        for index in range(FIRST_IMAGE + IMAGE_TOKENS, total)
    )

    # The whole final state (K, V, the indexer's raw and pooled keys, the
    # recurrent states) against the run that scopes every trunk forward itself.
    reference = _generate(
        _reference_runtime(model, _splice()), splice=_splice(), sampler=sampler
    )
    assert out.tokens == reference.tokens
    ours = _cache_leaves(out.final_state.final_trunk_cache)
    theirs = _cache_leaves(reference.final_state.final_trunk_cache)
    assert len(ours) == len(theirs) >= 4
    for mine, other in zip(ours, theirs):
        assert mine.shape == other.shape and mine.dtype == other.dtype
        assert mx.array_equal(mine, other).item()
    assert cache.pooled_len == total // cache.ratio  # pooled keys were written in decode
    assert mx.array_equal(out.final_state.final_logits, reference.final_state.final_logits).item()


# -- 3. structure: one scope helper, every trunk forward inside it ------------------

_TRUNK_FORWARD_CALLS = frozenset(
    {
        "forward_ar",
        "forward_ar_capture",
        "forward_fixed_m4",
        "verify_m2",
        "verify_m2_rebased",
        "verify_m3",
        "verify_m3_rebased",
    }
)


def _called_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    if isinstance(node.func, ast.Name):
        return node.func.id
    return None


def test_every_trunk_forward_call_site_sits_inside_the_scope_helper():
    """The capture lane and the compiled routes cannot run on the tiny model;
    their call sites are held to the same rule by where they stand."""
    source = textwrap.dedent(inspect.getsource(generation.generate_mtpk))
    tree = ast.parse(source)
    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent

    def in_trunk_scope(node: ast.AST) -> bool:
        while node in parents:
            node = parents[node]
            if isinstance(node, ast.With) and any(
                isinstance(item.context_expr, ast.Call)
                and _called_name(item.context_expr) == "_decode_trunk_scope"
                for item in node.items
            ):
                return True
        return False

    sites = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (_called_name(node) in _TRUNK_FORWARD_CALLS or _called_name(node) == "capture_forward")
    ]
    # Today's census: the main verify and its compiled, graph-bank and A3B
    # routes, both copy lanes with their repairs, the repair forward, the
    # lazy bonus commit and the final pending commit.
    assert len(sites) >= 20
    assert {_called_name(node) for node in sites} == {*_TRUNK_FORWARD_CALLS, "capture_forward"}
    outside = [(_called_name(node), node.lineno) for node in sites if not in_trunk_scope(node)]
    assert not outside, f"trunk forwards outside _decode_trunk_scope: {outside}"

    # The verify phase is only ever entered through the helper, so a new
    # forward written after the old pattern cannot miss the positions.
    bare = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and _called_name(node) == "attention_phase"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "decode_verify"
    ]
    assert not bare, f'bare attention_phase("decode_verify") in generate_mtpk: {bare}'
    scopes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _called_name(node) == "_decode_trunk_scope"
    ]
    assert len(scopes) >= 8  # today's census; a new trunk forward adds one
    for node in scopes:
        assert [ast.unparse(arg) for arg in node.args] == ["vision_splice"]


# -- 4. the session bank --------------------------------------------------------------


def _bank_turn(model, bank: SessionBank, prompt: list[int], *, image: bool = True):
    return _generate(
        _runtime(model),
        splice=_splice(prompt) if image else None,
        prompt=prompt,
        max_tokens=6,
        session_bank=bank,
        session_id="conversation",
        session_template_hash="template",
        session_draft_head_identity="draft-head",
        session_policy_fingerprint="policy",
        commit_prompt_state_to_bank=True,
    )


def test_an_entry_banked_by_an_older_build_cannot_be_restored_past_its_image(
    model, monkeypatch
):
    import mtplx.vision.splice as splice_module

    bank = SessionBank()
    second = [*PROMPT, 9, 10, 11, 12]
    third = [*second, 13, 14]
    fourth = [*third, 15, 16]

    # Builds up to 2.11.3 keyed an image row by its pixels and row index
    # only, whatever positions its rows were roped at.
    with monkeypatch.context() as older_build:
        older_build.setattr(
            splice_module, "_position_scheme_salts", lambda _splice, images: [0] * images
        )
        older_key = vision_bank_key_ids(PROMPT, _splice())
        assert _bank_turn(model, bank, PROMPT).stats.cached_tokens == 0
        # Control: under that key the next turn restores through the image.
        assert _bank_turn(model, bank, second).stats.cached_tokens == len(PROMPT)

    # This build: the rows those entries hold past the prompt may be
    # misplaced, so nothing at or past the image may come back from them.
    key = vision_bank_key_ids(PROMPT, _splice())
    pads = [index for index, token in enumerate(PROMPT) if token == PAD]
    assert all(key[index] != older_key[index] for index in pads)
    assert all(key[index] & _BANK_KEY_FLAG for index in pads)
    assert [key[i] for i in range(len(PROMPT)) if i not in pads] == [
        older_key[i] for i in range(len(PROMPT)) if i not in pads
    ]
    assert _bank_turn(model, bank, third).stats.cached_tokens <= FIRST_IMAGE
    # The entries this build writes restore warm, through the image.
    assert _bank_turn(model, bank, fourth).stats.cached_tokens == len(third)
    # Nothing was deleted to get there: the older entries are still banked
    # and age out through normal eviction.
    assert bank.longest_prefix(older_key) is not None


def test_text_entries_are_keyed_and_restored_exactly_as_before(model, monkeypatch):
    import mtplx.vision.splice as splice_module

    def never_for_text(*_args, **_kwargs):
        raise AssertionError("a text request reached the vision key derivation")

    # The salt lives in the vision key function, which a text request never
    # calls: its key cannot have moved.
    monkeypatch.setattr(splice_module, "vision_bank_key_ids", never_for_text)
    bank = SessionBank()
    text = [token for token in PROMPT if token != PAD]
    assert _bank_turn(model, bank, text, image=False).stats.cached_tokens == 0
    # A text request is keyed by its token ids, untouched.
    entry = bank.longest_prefix(text)
    assert entry is not None and list(entry.token_ids) == text
    longer = [*text, 9, 10, 11, 12]
    assert _bank_turn(model, bank, longer, image=False).stats.cached_tokens == len(text)
