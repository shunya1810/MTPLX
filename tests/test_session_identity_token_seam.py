"""A resent history that diverges from the session at a token-boundary seam keeps the session.

omp, 2026-10-03: an 81K-token thinking turn came back inside a 100K-token prompt.
Somewhere early in the thinking the model had sampled a run the tokenizer encodes
differently (same bytes), so the raw common prefix with the session ended a few
thousand tokens into the turn: under the 25 % fraction rule, past the turn-boundary
probe window. The request forked a fresh anon id, the committed-token splice had no
session to read from, and the turn was prefilled again (984 s TTFT). The token_seam
rule adopts the session when the divergence lies past a recorded turn prompt and the
server's tokenizer confirms the streams re-synchronise on the same text.
"""

from __future__ import annotations

import random
from types import SimpleNamespace

from mtplx.engine_session import EngineSessionManager
import mtplx.server.openai as srv


def _tokens(seed: int, count: int) -> list[int]:
    rng = random.Random(seed)
    return [rng.randrange(5, 50_000) for _ in range(count)]


def _resolve(manager: EngineSessionManager, prompt: list[int]) -> tuple[str, str, dict]:
    diagnostic: dict = {}
    session_id, source = manager.resolve_session_id(prompt_ids=prompt, diagnostic_out=diagnostic)
    return session_id, source, diagnostic


def _thinking_session(manager: EngineSessionManager):
    prompt = _tokens(1, 19_000)
    thinking = _tokens(2, 80_000)
    session_id, _, _ = _resolve(manager, prompt)
    session = manager.get_or_create(session_id)
    assert session.commit(prompt_ids=prompt, generated_ids=thinking, finish_reason="tool_calls").committed
    return session, prompt, thinking


def _resent(prompt: list[int], thinking: list[int], seam_at: int) -> list[int]:
    # The client's rendering of the same bytes: one token split in two at seam_at.
    return prompt + thinking[:seam_at] + [60_001, 60_002] + thinking[seam_at + 1 :] + _tokens(3, 1_000)


def test_a_seam_early_in_a_long_thinking_turn_keeps_the_session() -> None:
    manager = EngineSessionManager()
    session, prompt, thinking = _thinking_session(manager)
    calls = []

    def seam(tokens, committed, at):
        calls.append(at)
        return True

    manager.seam_resync = seam
    resent = _resent(prompt, thinking, 5_000)
    resolved, source, diagnostic = _resolve(manager, resent)
    assert resolved == session.session_id
    assert source == "common_prefix_reuse"
    assert diagnostic["reuse_rule"] == "token_seam"
    assert diagnostic["matched_prefix_len"] == 19_000 + 5_000
    assert calls == [24_000]
    assert session.last_seam_reuse_at == 24_000


def test_without_a_seam_check_the_request_forks_as_before() -> None:
    manager = EngineSessionManager()
    session, prompt, thinking = _thinking_session(manager)
    resolved, source, _ = _resolve(manager, _resent(prompt, thinking, 5_000))
    assert resolved != session.session_id
    assert source == "new"


def test_a_real_edit_forks() -> None:
    manager = EngineSessionManager()
    session, prompt, thinking = _thinking_session(manager)
    manager.seam_resync = lambda tokens, committed, at: False
    resolved, source, _ = _resolve(manager, _resent(prompt, thinking, 5_000))
    assert resolved != session.session_id and source == "new"


def test_a_divergence_inside_the_prompt_is_not_a_seam() -> None:
    """Another conversation sharing the system prompt and part of the first message."""
    manager = EngineSessionManager()
    session, prompt, thinking = _thinking_session(manager)
    manager.seam_resync = lambda tokens, committed, at: True
    other = prompt[:15_000] + [60_003] + _tokens(4, 90_000)
    resolved, source, _ = _resolve(manager, other)
    assert resolved != session.session_id and source == "new"


class _Tok:
    """Decodes ids through a table; 900 and 901 together spell what 77 spells."""

    table = {77: '"Nothing', 900: '"', 901: "Nothing"}

    def decode(self, ids):
        return "".join(self.table.get(int(i), f"<{int(i)}>") for i in ids)


def test_the_server_seam_check_reads_the_text() -> None:
    state = SimpleNamespace(runtime=SimpleNamespace(tokenizer=_Tok()))
    committed = [1, 2, 3, 77, 4, 5, 6]
    seam = [1, 2, 3, 900, 901, 4, 5, 6, 8]
    edit = [1, 2, 3, 900, 902, 4, 5, 6, 8]
    assert srv._session_seam_resync(state, seam, committed, 3) is True
    assert srv._session_seam_resync(state, edit, committed, 3) is False
