"""Context-copy GenerationStats counters: probes, rounds, accepted blocks/tokens,
suspend/backoff state, public-envelope exposure (#151 follow-up).

Deterministic next-token stub models drive generate_mtpk on CPU:
- a mod-VOCAB cycle whose prompt continuation always agrees with the model
  (full-accept copy rounds),
- a non-repeating ramp whose tail never matches a prompt gram (probe misses),
- a "trap" cycle whose prompt continuation always disagrees (zero-acceptance
  rounds driving the EMA into suspension + exponential backoff).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np

from mtplx.generation import GenerationStats, generate_mtpk
from mtplx.mtp_patch import MTPContract
from mtplx.runtime import MTPLXRuntime
from mtplx.sampling import SamplerConfig


class _Tokenizer:
    def decode(self, tokens, **_kwargs):
        return "".join(f"<{int(token)}>" for token in tokens)


class _ScriptedModel:
    """Deterministic automaton: after token t the model wants next_map(t)."""

    def __init__(self, vocab: int, next_map):
        self.vocab = vocab
        self.next_map = next_map
        self.mtp = SimpleNamespace(_mtplx_lora_targets=[])

    def make_cache(self):
        return []

    def make_mtp_cache(self):
        return []

    def mtp_update_cache(self, hidden_states, next_token_ids, **_kwargs):
        return hidden_states

    def _logits_for(self, last_tokens: list[int]) -> mx.array:
        rows = []
        for token in last_tokens:
            row = [0.0] * self.vocab
            row[self.next_map(int(token)) % self.vocab] = 10.0
            rows.append(row)
        return mx.array([rows], dtype=mx.float32)

    def __call__(
        self,
        input_ids,
        *,
        cache=None,
        return_hidden: bool = False,
        hidden_variant: str | None = None,
        emit_logits: bool = True,
        logits_keep: int | None = None,
    ):
        tokens = [int(token) for token in np.asarray(input_ids).reshape(-1)]
        keep = len(tokens) if logits_keep is None else min(len(tokens), max(1, int(logits_keep)))
        logits = self._logits_for(tokens[-keep:]) if emit_logits else None
        hidden = mx.zeros((1, len(tokens), 2), dtype=mx.float32)
        if not emit_logits:
            return (None, hidden) if return_hidden else None
        if return_hidden:
            return logits, hidden
        return logits

    def mtp_forward(
        self,
        hidden_states,
        next_token_ids,
        *,
        mtp_cache=None,
        concat_order=None,
        return_hidden: bool = False,
        mtp_hidden_variant: str | None = None,
        position_offset=None,
    ):
        tokens = [int(token) for token in np.asarray(next_token_ids).reshape(-1)]
        logits = self._logits_for(tokens)
        hidden = mx.zeros((1, len(tokens), 2), dtype=mx.float32)
        if return_hidden:
            return logits, hidden
        return logits


def _runtime(model: _ScriptedModel) -> MTPLXRuntime:
    return MTPLXRuntime(
        model=model,
        tokenizer=_Tokenizer(),
        model_path=Path("tiny-copy"),
        mtp_enabled=True,
        contract=MTPContract(),
    )


def _mtpk(model: _ScriptedModel, prompt: list[int], max_tokens: int):
    return generate_mtpk(
        _runtime(model),
        prompt,
        max_tokens=max_tokens,
        sampler=SamplerConfig(temperature=0.0, top_p=1.0, top_k=0),
        speculative_depth=1,
        seed=0,
        stop_token_ids=set(),
        verify_strategy="capture_commit",
    )


def _clean_env(monkeypatch) -> None:
    for name in (
        "MTPLX_CONTEXT_COPY",
        "MTPLX_CONTEXT_COPY_K",
        "MTPLX_CONTEXT_COPY_NGMIN",
        "MTPLX_CONTEXT_COPY_NGMAX",
        "MTPLX_CONTEXT_COPY_MINEXT",
        "MTPLX_SKIP_VERIFY_SNAPSHOT",
        # Same reason MTPLX_SKIP_VERIFY_SNAPSHOT is here: both are turbo/
        # sustained profile env (profiles.py), and apply_profile_env writes
        # os.environ for real, so any earlier test in the process that ran a
        # CLI lane through a profile leaves them set. These tests read
        # stats.events, and generation.py gates event recording on
        # MTPLX_DROP_EVENTS, so a leaked "1" silently empties every block
        # assertion here (order-dependent failure, e.g. after
        # tests/test_public_cli.py).
        "MTPLX_DROP_EVENTS",
    ):
        monkeypatch.delenv(name, raising=False)
    # The lane is off by default on the M1 GPU family (context_copy_enabled);
    # these tests exercise the lane itself, so turn it on explicitly.
    monkeypatch.setenv("MTPLX_CONTEXT_COPY", "1")


# --- full-accept copy rounds (mod-8 cycle, prompt continuation agrees) ---


def test_full_accept_rounds_count_probes_rounds_blocks_tokens(monkeypatch):
    _clean_env(monkeypatch)
    out = _mtpk(_ScriptedModel(8, lambda t: t + 1), [0, 1, 2, 3, 4, 5, 6, 7, 0], max_tokens=80)
    stats = out.stats
    assert stats.context_copy_active is True
    assert stats.context_copy_disabled_reason is None
    assert stats.context_copy_rounds >= 2
    # Early cycles probe before the generated tail re-enters a prompt gram.
    assert stats.context_copy_probes > stats.context_copy_rounds
    # The prompt continuation is exactly what the model wants: every proposed
    # block verifies in full.
    assert stats.context_copy_accepted_blocks == stats.context_copy_rounds
    assert stats.context_copy_drafted_tokens == stats.context_copy_accepted_tokens
    assert stats.context_copy_accepted_tokens > 0
    assert stats.context_copy_suspensions == 0
    assert stats.context_copy_suspended is False
    assert stats.context_copy_backoff_tokens == 64


def test_kill_switch_disables_and_output_is_byte_identical(monkeypatch):
    _clean_env(monkeypatch)
    on = _mtpk(_ScriptedModel(8, lambda t: t + 1), [0, 1, 2, 3, 4, 5, 6, 7, 0], max_tokens=80)
    monkeypatch.setenv("MTPLX_CONTEXT_COPY", "0")
    off = _mtpk(_ScriptedModel(8, lambda t: t + 1), [0, 1, 2, 3, 4, 5, 6, 7, 0], max_tokens=80)
    assert off.stats.context_copy_active is False
    assert off.stats.context_copy_probes == 0
    assert off.stats.context_copy_rounds == 0
    assert off.stats.context_copy_backoff_tokens == 0
    assert list(on.tokens) == list(off.tokens)


# --- probe misses (non-repeating ramp, tail never matches a prompt gram) ---


def test_probe_misses_count_probes_but_no_rounds(monkeypatch):
    _clean_env(monkeypatch)
    out = _mtpk(_ScriptedModel(64, lambda t: t + 1), [0, 1, 2, 3, 4, 5], max_tokens=40)
    stats = out.stats
    assert stats.context_copy_active is True
    assert stats.context_copy_probes > 0
    assert stats.context_copy_rounds == 0
    assert stats.context_copy_drafted_tokens == 0
    assert stats.context_copy_accepted_blocks == 0
    assert stats.context_copy_accepted_tokens == 0
    assert stats.context_copy_suspensions == 0


# --- zero-acceptance rounds -> suspension with exponential backoff ---


def _trap_next(t: int) -> int:
    # Cycle 10..15; every token outside the cycle re-enters it at 10. The
    # prompt continuation after the (10..15) gram is 99, which the model
    # never produces, so every copy round verifies 0 tokens.
    if 10 <= t < 15:
        return t + 1
    return 10


def test_zero_acceptance_rounds_drive_suspension_and_backoff(monkeypatch):
    _clean_env(monkeypatch)
    # Every rotation of the 10..15 cycle appears in the prompt, each followed
    # by a token the model never produces: whatever the generated tail's
    # alignment, the probe finds a match and the copy block verifies 0 tokens.
    cycle = [10, 11, 12, 13, 14, 15]
    prompt = []
    for i in range(6):
        prompt += cycle[i:] + cycle[:i] + [99 - i]
    out = _mtpk(_ScriptedModel(128, _trap_next), prompt, max_tokens=200)
    stats = out.stats
    assert stats.context_copy_active is True
    assert stats.context_copy_rounds >= 4
    assert stats.context_copy_drafted_tokens > 0
    assert stats.context_copy_accepted_blocks == 0
    assert stats.context_copy_accepted_tokens == 0
    assert stats.context_copy_suspensions >= 1
    # Each suspension doubles the retry backoff from its 64-token floor.
    assert stats.context_copy_backoff_tokens >= 128


# --- defaults + public envelope exposure ---


def test_generation_stats_defaults_serialize_context_copy_fields():
    stats = GenerationStats(mode="ar", generated_tokens=0, elapsed_s=0.0, tok_s=0.0)
    payload = stats.to_dict()
    assert payload["context_copy_active"] is False
    assert payload["context_copy_probes"] == 0
    assert payload["context_copy_rounds"] == 0
    assert payload["context_copy_drafted_tokens"] == 0
    assert payload["context_copy_accepted_blocks"] == 0
    assert payload["context_copy_accepted_tokens"] == 0
    assert payload["context_copy_suspensions"] == 0
    assert payload["context_copy_suspended"] is False
    assert payload["context_copy_backoff_tokens"] == 0
    assert payload["context_copy_disabled_reason"] is None


# --- temperature path: probability-ratio acceptance over copy blocks ---


def _mtpk_sampled(
    model: _ScriptedModel,
    prompt: list[int],
    max_tokens: int,
    *,
    temperature: float,
    seed: int,
):
    return generate_mtpk(
        _runtime(model),
        prompt,
        max_tokens=max_tokens,
        sampler=SamplerConfig(temperature=temperature, top_p=1.0, top_k=0),
        speculative_depth=1,
        seed=seed,
        stop_token_ids=set(),
        verify_strategy="capture_commit",
    )


def test_temperature_peaked_model_copy_rounds_fire_and_match_greedy(monkeypatch):
    # Peaked scripted logits (10.0 vs 0.0) make sampling at temp 0.6
    # deterministic in practice, so the copy-on stream must equal the
    # copy-off stream token for token while copy rounds actually fire.
    _clean_env(monkeypatch)
    on = _mtpk_sampled(
        _ScriptedModel(8, lambda t: t + 1),
        [0, 1, 2, 3, 4, 5, 6, 7, 0],
        max_tokens=80,
        temperature=0.6,
        seed=7,
    )
    monkeypatch.setenv("MTPLX_CONTEXT_COPY", "0")
    off = _mtpk_sampled(
        _ScriptedModel(8, lambda t: t + 1),
        [0, 1, 2, 3, 4, 5, 6, 7, 0],
        max_tokens=80,
        temperature=0.6,
        seed=7,
    )
    assert on.stats.context_copy_active is True
    assert on.stats.context_copy_rounds >= 2
    assert on.stats.context_copy_accepted_tokens > 0
    assert list(on.tokens) == list(off.tokens)


def test_temperature_trap_model_rejections_emit_residual_corrections(monkeypatch):
    # The trap prompt proposes a continuation (99) the model never wants;
    # under temperature every fired copy round must reject and emit a
    # correction sampled from the residual, which with peaked logits is the
    # model's own next token, so the stream still matches the copy-off arm.
    _clean_env(monkeypatch)
    cycle = [10, 11, 12, 13, 14, 15]
    prompt = []
    for i in range(6):
        prompt += cycle[i:] + cycle[:i] + [99 - i]
    on = _mtpk_sampled(
        _ScriptedModel(128, _trap_next),
        prompt,
        max_tokens=200,
        temperature=0.6,
        seed=11,
    )
    monkeypatch.setenv("MTPLX_CONTEXT_COPY", "0")
    off = _mtpk_sampled(
        _ScriptedModel(128, _trap_next),
        prompt,
        max_tokens=200,
        temperature=0.6,
        seed=11,
    )
    assert on.stats.context_copy_rounds >= 1
    assert on.stats.context_copy_accepted_tokens == 0
    assert on.stats.context_copy_suspensions >= 1
    assert list(on.tokens) == list(off.tokens)


class _SoftModel(_ScriptedModel):
    """60/40 coin between tokens 2 and 3 at every position (temp 1.0)."""

    def __init__(self):
        super().__init__(4, lambda t: 2)

    def _logits_for(self, last_tokens):
        import math

        rows = []
        for _ in last_tokens:
            row = [-1e9, -1e9, math.log(0.6), math.log(0.4)]
            rows.append(row)
        return mx.array([rows], dtype=mx.float32)


def test_temperature_copy_preserves_the_sampling_distribution(monkeypatch):
    # The sharp exactness check: the prompt is a run of 2s, so whenever the
    # generated tail re-enters a six-2 run the copy path proposes more 2s as
    # a point-mass draft. Exact speculative sampling must leave the law of
    # every emitted token at the model's own 60/40 regardless. Compare the
    # marginal frequency of token 2 across many seeded runs, copy on vs off,
    # and check the copy-armed conditional (the token right after a >=6-run
    # of 2s, where copy rounds actually fire) stays at ~0.6.
    _clean_env(monkeypatch)
    prompt = [2] * 8
    seeds = list(range(240))

    def run_arm() -> tuple[list[int], int]:
        emitted: list[int] = []
        copy_rounds = 0
        for seed in seeds:
            out = _mtpk_sampled(
                _SoftModel(), prompt, max_tokens=24, temperature=1.0, seed=seed
            )
            emitted.extend(int(t) for t in out.tokens)
            copy_rounds += out.stats.context_copy_rounds
        return emitted, copy_rounds

    on_tokens, on_rounds = run_arm()
    monkeypatch.setenv("MTPLX_CONTEXT_COPY", "0")
    off_tokens, off_rounds = run_arm()

    assert on_rounds > 0
    assert off_rounds == 0

    def freq2(tokens: list[int]) -> float:
        return sum(1 for t in tokens if t == 2) / max(1, len(tokens))

    # Marginal law matches between arms (both should sit near 0.6).
    assert abs(freq2(on_tokens) - freq2(off_tokens)) < 0.03
    assert abs(freq2(on_tokens) - 0.6) < 0.03

    # Conditional law at copy-armed positions: right after six 2s in a row
    # the next token is exactly where point-mass acceptance operates.
    def conditional_freq2(tokens_per_seed: list[list[int]]) -> float:
        hits = total = 0
        for run in tokens_per_seed:
            streak = 0
            for token in run:
                if streak >= 6:
                    total += 1
                    if token == 2:
                        hits += 1
                streak = streak + 1 if token == 2 else 0
        return hits / max(1, total)

    _clean_env(monkeypatch)
    per_seed_on = [
        [int(t) for t in _mtpk_sampled(_SoftModel(), prompt, max_tokens=24, temperature=1.0, seed=s).tokens]
        for s in seeds
    ]
    conditional = conditional_freq2(per_seed_on)
    assert abs(conditional - 0.6) < 0.05


def test_public_mtplx_stats_expose_context_copy_counters():
    from mtplx.server.openai import PUBLIC_MTPLX_STATS_KEYS, _public_mtplx_stats

    keys = {
        "context_copy_active",
        "context_copy_probes",
        "context_copy_rounds",
        "context_copy_drafted_tokens",
        "context_copy_accepted_blocks",
        "context_copy_accepted_tokens",
        "context_copy_suspensions",
        "context_copy_suspended",
        "context_copy_backoff_tokens",
        "context_copy_disabled_reason",
    }
    assert keys <= set(PUBLIC_MTPLX_STATS_KEYS)
    generated = {"stats": {key: 1 for key in keys}}
    public = _public_mtplx_stats(generated)
    assert keys <= set(public)


# --- stop-token boundary (copy round must never accept/commit past a stop) ---


def _mtpk_stop(
    model: _ScriptedModel,
    prompt: list[int],
    max_tokens: int,
    *,
    stop: set[int],
    temperature: float = 0.0,
):
    return generate_mtpk(
        _runtime(model),
        prompt,
        max_tokens=max_tokens,
        sampler=SamplerConfig(temperature=temperature, top_p=1.0, top_k=0),
        speculative_depth=1,
        seed=0,
        stop_token_ids=stop,
        verify_strategy="capture_commit",
        capture_final_state=True,
    )


# Two full mod-8 cycles: every rotation is a prompt gram, so the generated
# tail re-enters the prompt immediately and the copy block proposed from the
# second cycle contains the stop token (5) mid-block.
_TWO_CYCLE_PROMPT = [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0]


def _assert_stop_boundary(out) -> None:
    # The copy lane must actually have fired, or this test proves nothing.
    assert out.stats.context_copy_rounds >= 1
    # Emitted tokens end exactly at the first stop token.
    assert list(out.tokens) == [1, 2, 3, 4, 5]
    assert out.finish_reason == "stop"
    final = out.final_state
    assert final is not None
    assert final.generated_token_ids == tuple(out.tokens)
    assert final.safe_to_commit is True
    # Cold-replay contract: the stored final logits row must be the model's
    # next-token distribution AFTER the stop token (next_map(5) == 6), not a
    # row selected past the stop from the rest of the accepted copy block.
    next_after_stop = int(mx.argmax(final.final_logits[0]).item())
    assert next_after_stop == 6
    # Acceptance counters must not count tokens past the stop: the block from
    # the second cycle is [2, 3, 4, 5, 6, 7, 0] and only [2, 3, 4, 5] commits.
    assert out.stats.context_copy_accepted_tokens <= 4


def test_greedy_copy_round_stops_at_accepted_stop_token(monkeypatch):
    _clean_env(monkeypatch)
    out = _mtpk_stop(
        _ScriptedModel(8, lambda t: t + 1), _TWO_CYCLE_PROMPT, 80, stop={5}
    )
    _assert_stop_boundary(out)


def test_sampled_copy_round_stops_at_accepted_stop_token(monkeypatch):
    _clean_env(monkeypatch)
    # Peaked one-hot logits: probability-ratio acceptance approaches 1 for the
    # copied token, so the sampled branch walks the same copy block as greedy
    # and must cap acceptance at the stop the same way.
    out = _mtpk_stop(
        _ScriptedModel(8, lambda t: t + 1),
        _TWO_CYCLE_PROMPT,
        80,
        stop={5},
        temperature=0.6,
    )
    _assert_stop_boundary(out)


# --- prompt-boundary contract (copy blocks never slice generated output) ---


def test_match_ending_at_prompt_edge_never_fires_a_self_copy_round(monkeypatch):
    _clean_env(monkeypatch)
    # The only prompt gram is (0..5) whose continuation starts exactly at the
    # prompt edge: there is no prompt continuation to copy. Firing anyway
    # (the pre-fix behavior) would slice the block out of the model's own
    # generated output — exactly the self-repetition the module contract
    # excludes.
    out = _mtpk(_ScriptedModel(8, lambda t: t + 1), [0, 1, 2, 3, 4, 5], max_tokens=40)
    assert out.stats.context_copy_probes > 0
    assert out.stats.context_copy_rounds == 0
    assert out.stats.context_copy_drafted_tokens == 0


def test_copy_blocks_are_capped_at_the_prompt_boundary(monkeypatch):
    _clean_env(monkeypatch)
    out = _mtpk(_ScriptedModel(8, lambda t: t + 1), _TWO_CYCLE_PROMPT, max_tokens=60)
    stats = out.stats
    assert stats.context_copy_rounds >= 1
    copy_events = [
        event["context_copy"]
        for event in stats.events
        if isinstance(event, dict) and "context_copy" in event
    ]
    blocks = [event["block"] for event in copy_events if "block" in event]
    assert blocks, "copy rounds fired but recorded no block events"
    # Continuation positions in this prompt leave at most 7 prompt tokens; the
    # confidence ladder alone would propose 8+ (up to 24), so any block longer
    # than the prompt remainder proves a slice into generated output.
    assert max(blocks) <= 7
