"""Session/canonicalization hardening smalls (audit F11 #5/#6/#8/#9/P2).

Covers: the repair prompts preserving committed reasoning, the
transient trailing-sentinel registry, the system-suffix conversion of the
no-tools/post-tool contracts, the burst-pinned date line, single-call
session resolution, reasoning_effort ladder mapping, and the warmup
prefill-chunk env override.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from mtplx.server import openai as oa

OPENAI_PY = Path(oa.__file__)


# --- repair-encode preserves committed reasoning (x3 sites) ---------------


def test_every_repair_prompt_keeps_committed_reasoning():
    """The stream's recovery passes build their prompts from the gate's
    canonical messages; a re-encode without allow_committed_reasoning=True
    drops the substituted think and re-poisons what canonicalization just
    fixed (audit F11 #5). Since 2026-09-30 both re-generations take their
    prompt from served_retry_prompt_ids (the served ids, then the repair
    turn), whose two renders must carry the flag, and the reasoning-completion
    repair extends the served ids without re-encoding. AST-pinned so a
    repair site that re-encodes on its own, or a builder render without the
    flag, fails this test."""
    tree = ast.parse(OPENAI_PY.read_text())

    def called(call: ast.Call) -> str:
        return getattr(call.func, "id", getattr(call.func, "attr", ""))

    def has_flag(call: ast.Call) -> bool:
        return any(
            keyword.arg == "allow_committed_reasoning"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is True
            for keyword in call.keywords
        )

    builders = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "served_retry_prompt_ids"
    ]
    assert len(builders) == 1, f"expected one retry prompt builder, found {len(builders)}"
    renders = [
        node
        for node in ast.walk(builders[0])
        if isinstance(node, ast.Call) and called(node) == "_encode_messages"
    ]
    assert len(renders) == 2, (
        f"expected the builder's two renders, found {[n.lineno for n in renders]}"
    )
    missing = [node.lineno for node in renders if not has_flag(node)]
    assert not missing, (
        f"builder renders missing allow_committed_reasoning=True at lines {missing}"
    )

    sources: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        names: set[str] = set()
        for target in node.targets:
            elements = target.elts if isinstance(target, ast.Tuple) else [target]
            names.update(
                element.id for element in elements if isinstance(element, ast.Name)
            )
        if "repair_prompt_ids" not in names:
            continue
        value = node.value
        sources.append(
            (
                node.lineno,
                called(value) if isinstance(value, ast.Call) else type(value).__name__,
            )
        )
    assert sorted(name for _line, name in sources) == [
        "_reasoning_completion_repair_prompt_ids",
        "served_retry_prompt_ids",
        "served_retry_prompt_ids",
    ], f"repair prompts must come from the served-ids builder, found {sources}"


# --- sentinel-registry trailing boundary (x3 sentinels) -------------------


def _rendered_with_trailing_user(sentinel_text: str) -> str:
    return (
        "<|im_start|>system\nsys<|im_end|>\n"
        "<|im_start|>user\nreal question<|im_end|>\n"
        f"<|im_start|>user\n{sentinel_text}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n"
    )


@pytest.mark.parametrize(
    "sentinel_builder",
    [
        oa._mtplx_tool_result_continuation_hint_text,
        oa._mtplx_read_only_force_answer_contract_text,
        oa._mtplx_pi_convergence_contract_text,
    ],
    ids=["continuation_hint", "read_only_force_answer", "pi_convergence"],
)
def test_trailing_boundary_detects_each_registry_sentinel(sentinel_builder):
    rendered = _rendered_with_trailing_user(sentinel_builder())
    boundary = oa._trailing_tool_hint_char_boundary(rendered)
    assert boundary is not None
    # The boundary sits immediately AFTER the injected turn's <|im_start|>.
    expected = rendered.rindex("<|im_start|>user\n" + sentinel_builder()[:48])
    assert boundary == expected + len("<|im_start|>")


def test_trailing_boundary_rejects_non_tail_sentinel():
    hint = oa._mtplx_tool_result_continuation_hint_text()
    rendered = (
        "<|im_start|>system\nsys<|im_end|>\n"
        f"<|im_start|>user\n{hint}<|im_end|>\n"
        "<|im_start|>assistant\nanswer<|im_end|>\n"
        "<|im_start|>user\nnew question<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n"
    )
    assert oa._trailing_tool_hint_char_boundary(rendered) is None


# --- system-suffix contracts: msg0 stable across flips (#6) ---------------


def _msgs(*contents: str) -> list:
    roles = ["system", "user"]
    return [
        oa.ChatMessage(role=roles[min(i, 1)], content=content)
        for i, content in enumerate(contents)
    ]


@pytest.mark.parametrize(
    "with_contract, sentinel",
    [
        (oa._with_mtplx_no_tool_contract, oa._MTPLX_NO_TOOL_CONTRACT_SENTINEL),
        (
            oa._with_mtplx_post_tool_answer_contract,
            oa._MTPLX_POST_TOOL_ANSWER_SENTINEL,
        ),
    ],
    ids=["no_tools", "post_tool_answer"],
)
def test_contract_is_pure_suffix_msg0_stable(with_contract, sentinel):
    base = _msgs("client system prompt", "the question")
    updated = with_contract(list(base))
    # msg0 (and every pre-existing message) byte-stable: the contract flip
    # must not rewrite the prompt prefix the session bank has already
    # committed (audit F11 #6 — the old splice re-prefilled the whole
    # transcript cold on every flip).
    assert [
        (m.role, oa._content_to_text(m.content)) for m in updated[: len(base)]
    ] == [(m.role, oa._content_to_text(m.content)) for m in base]
    assert len(updated) == len(base) + 1
    tail = updated[-1]
    assert str(tail.role) == "user"
    assert sentinel in oa._content_to_text(tail.content)
    # Dedup: applying twice appends once.
    again = with_contract(list(updated))
    assert len(again) == len(updated)


def test_transient_suffix_flag_covers_new_suffix_contracts():
    source = (OPENAI_PY.parent / "request_policy.py").read_text()
    anchor = source.index("transient_suffix_contract_active = bool(")
    window = source[anchor : anchor + 500]
    assert "no_tools_contract_active" in window
    assert "post_tool_answer_contract_active" in window


# --- date-pin stability (P2) ----------------------------------------------


def test_date_line_pinned_across_midnight_within_burst(monkeypatch):
    clock = {"day": "August 16, 2026", "mono": 1000.0}
    monkeypatch.setattr(oa.time, "strftime", lambda fmt: clock["day"])
    monkeypatch.setattr(oa.time, "monotonic", lambda: clock["mono"])
    monkeypatch.setitem(oa._DATE_LINE_PIN, "day", None)
    monkeypatch.setitem(oa._DATE_LINE_PIN, "last_use_monotonic", None)

    first = oa._current_date_line()
    assert "August 16, 2026" in first
    # Midnight passes mid-burst (requests 30s apart): bytes must not move.
    clock["day"] = "August 17, 2026"
    clock["mono"] += 30.0
    assert oa._current_date_line() == first
    # Still mid-burst an hour later, as long as no idle gap ever exceeded
    # the refresh window.
    clock["mono"] += 60.0
    assert oa._current_date_line() == first
    # After a real idle window the pin refreshes to today.
    clock["mono"] += oa._DATE_LINE_IDLE_REFRESH_S + 1.0
    assert "August 17, 2026" in oa._current_date_line()


# --- resolve_session_id single call (P2) ----------------------------------


def test_gate_uses_preresolved_session_id(monkeypatch):
    committed = list(range(100, 200))
    committed_text = (
        "<|im_start|>user\nhi<|im_end|>\n"
        "<|im_start|>assistant\n<think>\nReal think.\n</think>\n\n"
        "The answer.<|im_end|>\n"
    )
    resolve_calls: list[int] = []
    session = SimpleNamespace(committed_token_ids=tuple(committed))
    sessions = SimpleNamespace(
        resolve_session_id=lambda **kw: resolve_calls.append(1) or ("s1", "x"),
        peek=lambda sid: session,
    )
    state = SimpleNamespace(
        args=SimpleNamespace(strip_assistant_reasoning_history=False),
        sessions=sessions,
        runtime=SimpleNamespace(
            tokenizer=SimpleNamespace(decode=lambda ids: committed_text)
        ),
    )
    monkeypatch.setattr(oa, "_reasoning_history_scoped_active", lambda state: False)
    monkeypatch.setattr(
        oa, "_encode_messages", lambda tokenizer, msgs, **kw: committed[:80]
    )
    request = oa.ChatCompletionRequest(
        model="m",
        messages=[
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "The answer."},
            {"role": "user", "content": "next"},
        ],
    )
    result = oa._maybe_canonicalize_committed_reasoning(
        state,
        messages=request.messages,
        prompt_ids=committed[:10] + [1],
        headers={},
        metadata={},
        request=request,
        thinking_enabled=True,
        reasoning_effort="medium",
        tools=None,
        tool_choice=None,
        tool_prompt_mode="hybrid",
        template_observability={},
        session_id="s1",
    )
    assert result is not None
    assert resolve_calls == [], (
        "a pre-resolved session id must skip the gate's own resolution"
    )


def test_chat_endpoint_resolves_session_once():
    """Source pin for the endpoint seam: the prologue resolves once into
    resolved_session_id, the gate consumes it, and the adoption step reuses
    it instead of resolving again. The adoption block sits AFTER the
    canonicalization call (the canon-after-wait early sweep, 2026-08-21,
    added an earlier `if resolved_session_id is not None:` for the pending
    postcommit wait — anchor past the canon call to keep pinning adoption)."""
    source = OPENAI_PY.read_text()
    assert "session_id=resolved_session_id," in source
    canon_call = source.index("_canonicalized = _maybe_canonicalize_committed_reasoning(")
    adoption = source.index("if resolved_session_id is not None:", canon_call)
    window = source[adoption : adoption + 700]
    assert "resolved_session_source" in window
    assert "else:" in window


# --- reasoning_effort mapping (#8) ----------------------------------------


def _effort_state(levels, default):
    return SimpleNamespace(
        args=SimpleNamespace(reasoning_effort=None),
    ), SimpleNamespace(effort_levels=levels, default_effort=default)


@pytest.mark.parametrize(
    "levels, default, requested, expected",
    [
        # Qwen 3.8 shape: no literal "high" tier -> nearest declared UP.
        (("xhigh", "medium", "low"), "medium", "high", "xhigh"),
        (("xhigh", "medium", "low"), "medium", "low", "low"),
        (("xhigh", "medium", "low"), "medium", "medium", "medium"),
        (("xhigh", "medium", "low"), "medium", "xhigh", "xhigh"),
        # Family with a real "high" tier: the literal tier wins.
        (("low", "medium", "high"), "medium", "high", "high"),
        # Nothing above the request -> nearest declared below.
        (("low", "medium", "high"), "medium", "xhigh", "high"),
        (("low", "medium"), "low", "high", "medium"),
    ],
)
def test_reasoning_effort_maps_to_declared_ladder(
    monkeypatch, levels, default, requested, expected
):
    state, codec = _effort_state(levels, default)
    monkeypatch.setattr(oa, "_reasoning_codec_for_state", lambda s: codec)
    resolved = oa._reasoning_effort_for_state(
        state, thinking_enabled=True, request_effort=requested
    )
    assert resolved == expected


def test_reasoning_effort_junk_is_a_400(monkeypatch):
    state, codec = _effort_state(("xhigh", "medium", "low"), "medium")
    monkeypatch.setattr(oa, "_reasoning_codec_for_state", lambda s: codec)
    with pytest.raises(oa.HTTPException) as excinfo:
        oa._reasoning_effort_for_state(
            state, thinking_enabled=True, request_effort="banana"
        )
    assert excinfo.value.status_code == 400


def test_reasoning_effort_auto_and_server_defaults(monkeypatch):
    state, codec = _effort_state(("xhigh", "medium", "low"), "medium")
    monkeypatch.setattr(oa, "_reasoning_codec_for_state", lambda s: codec)
    assert (
        oa._reasoning_effort_for_state(
            state, thinking_enabled=True, request_effort="auto"
        )
        == "medium"
    )
    assert (
        oa._reasoning_effort_for_state(state, thinking_enabled=True) == "medium"
    )
    assert (
        oa._reasoning_effort_for_state(state, thinking_enabled=False) is None
    )


# --- warmup-chunk env (#9) ------------------------------------------------


def test_warmup_prefill_chunk_env_override(monkeypatch):
    stub = SimpleNamespace(
        WARMUP_PREFILL_CHUNK_TOKENS=oa._BackgroundWarmup.WARMUP_PREFILL_CHUNK_TOKENS
    )
    resolve = oa._BackgroundWarmup._warmup_prefill_chunk_tokens
    monkeypatch.delenv("MTPLX_WARMUP_PREFILL_CHUNK", raising=False)
    assert resolve(stub) == 256, "default must stay the measured 2026-07-31 fence"
    monkeypatch.setenv("MTPLX_WARMUP_PREFILL_CHUNK", "512")
    assert resolve(stub) == 512
    monkeypatch.setenv("MTPLX_WARMUP_PREFILL_CHUNK", "not-a-number")
    assert resolve(stub) == 256
    monkeypatch.setenv("MTPLX_WARMUP_PREFILL_CHUNK", "-8")
    assert resolve(stub) == 1, "nonpositive values clamp to a sane floor"


def test_warmup_default_constant_unchanged():
    assert oa._BackgroundWarmup.WARMUP_PREFILL_CHUNK_TOKENS == 256


# --- postcommit call sites thread the committed stream (#3 plumbing) ------


def test_history_ids_for_postcommit_accepts_committed_stream():
    signature = inspect.signature(oa._history_ids_for_postcommit)
    assert "committed_stream_ids" in signature.parameters
    signature = inspect.signature(oa._store_retokenized_history_snapshot)
    assert "committed_stream_ids" in signature.parameters
    signature = inspect.signature(oa._schedule_idle_postcommit_snapshot)
    assert "committed_stream_ids" in signature.parameters
