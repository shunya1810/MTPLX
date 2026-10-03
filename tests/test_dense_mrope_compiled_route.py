"""Dense-path image requests on the compiled verify route.

The four-layer synthetic ``qwen3_5`` model of ``tests/dense_mrope_synth.py``
with its draft head, through ``generate_mtpk`` with the compiled verify bank
on. The bank owns the request's rotary delta next to its logical offset
(``mtplx/rope_origin.py``); every container it promotes rotates at
``offset + delta`` through the product's attention hook, and the verify
trace takes the delta as a graph input. No pack is loaded.

Pinned, against the eager route with the position scope open:

* at the layer, the bank-owned origin writes the same key rows bit for bit,
  and so does the parity instrument's reference (the host table resolved on
  the bank's own buffers), through a rollback;
* the generated tokens, greedy and sampled at native-like settings with a
  fixed seed, through rejected windows and a mid-request growth demotion;
* two requests with different deltas replay one trace, a text request keeps
  the text trace, and the request record says which route ran;
* the parity instrument compares every compiled round and finds nothing;
* the refused settings keep the eager verifier, for every image request,
  and say why; the session bank restores an image turn the compiled route
  banked.
"""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import pytest

from mtplx import demotions, generation
from mtplx.attention_split import configure_split_full_attention
from mtplx.dense_mrope import (
    configure_dense_mrope,
    dense_mrope_scope,
    dense_mrope_state,
    host_positions_for_tensor_offsets,
)
from mtplx.generation import generate_mtpk
from mtplx.graphbank import (
    TensorOffsetKVCache,
    promote_kv_cache_offsets,
    stamp_rope_delta,
)
from mtplx.mtp_patch import MTPContract
from mtplx.runtime import MTPLXRuntime
from mtplx.sampling import SamplerConfig
from mtplx.session_bank import SessionBank
from mtplx.vision.splice import VisionSplice
from tests import dense_mrope_synth as synth
from tests.test_dense_mrope_generation import (
    LOGIT_NOISE,
    _teacher_forced_gaps,
    _Tokenizer,
)

PAD, GRID, PROMPT, DELTA = synth.PAD, synth.GRID, synth.PROMPT, synth.DELTA
# A second image shape: a 2 x 2 block (four tokens) advances the position by
# max(1, 2, 2) = 2, so the request's delta is 2 - 4 = -2, not -3.
GRID2 = (1, 4, 4)
PROMPT2 = [5, 6, 7, PAD, PAD, PAD, PAD, 8, 9, 10, 11, 12]
DELTA2 = -2
TEXT = [(7 * i + 3) % 97 for i in range(16)]

GREEDY = SamplerConfig(temperature=0.0, top_p=1.0, top_k=20)
# The 27B pack's own settings (mtplx_runtime.json): temperature 0.6, top_p
# 0.95, top_k 20, and a draft sampler at temperature 1.0.
NATIVE = SamplerConfig(temperature=0.6, top_p=0.95, top_k=20)
NATIVE_DRAFT = SamplerConfig(temperature=1.0, top_p=0.95, top_k=20)


@pytest.fixture(autouse=True, params=["route-off", "turbo-route"])
def _clean(request, monkeypatch):
    # Tests of the compiled route explicitly opt in; the default has its
    # own product-level test below.
    monkeypatch.setenv("MTPLX_DENSE_VISION_COMPILED_VERIFY", "1")
    # Both attention configurations. "turbo-route" is the turbo profile's
    # (the profile that turns the compiled verifier on): every full-attention
    # forward takes MTPLX's route. With the route off, an image request's
    # compiled leg (containers that own a rotary origin) takes MTPLX's route
    # while its parity reference (containers without one) is mlx-lm's stock
    # forward outside verify; on verify both go through attention_gate, else
    # the two lowerings of the gate's sigmoid differ in float32 (hidden on an
    # M5 by the TF32 o_proj GEMM, not on the M1 to M4 kernels).
    if request.param == "turbo-route":
        monkeypatch.setenv("MTPLX_GQA_PACKED_SDPA", "1")
    for name in (
        *(() if request.param == "turbo-route" else ("MTPLX_GQA_PACKED_SDPA",)),
        "MTPLX_SPLIT_FULL_ATTN",
        "MTPLX_SDPA_2PASS",
        "MTPLX_VLLM_METAL_PAGED_ATTN",
        "MTPLX_BLOCKWISE_ATTN",
        "MTPLX_DENSE_MROPE",
        "MTPLX_DENSE_MROPE_STRICT",
        "MTPLX_COMPILED_VERIFY",
        "MTPLX_COMPILED_VERIFY_GROWTH_RESERVE",
        "MTPLX_QWEN4_VISION_COMPILED_VERIFY",
        "MTPLX_STATE_REBASE_EVERY",
        "MTPLX_MTP_HISTORY_POLICY",
        "MTPLX_MTP_POSITION_MODE",
    ):
        monkeypatch.delenv(name, raising=False)
    demotions.reset()
    yield
    demotions.reset()


def _install(model):
    # What runtime.load does to every model: the attention hook first (a
    # cache that owns a rotary origin rotates at it, everything else is the
    # stock forward), then the image position adapter.
    configure_split_full_attention(model)
    assert configure_dense_mrope(model, synth.pack_config()).installed
    return model


@pytest.fixture()
def rig(tmp_path):
    model = _install(synth.model_with_draft_head(tmp_path, seed=8, tie=False))
    rt = MTPLXRuntime(
        model=model,
        tokenizer=_Tokenizer(),
        model_path=tmp_path,
        mtp_enabled=True,
        contract=MTPContract(),
    )
    mx.random.seed(3)
    rows = mx.random.normal((6, 64)) * 0.05
    rows2 = mx.random.normal((4, 64)) * 0.05
    mx.eval(rows, rows2)
    return SimpleNamespace(rt=rt, model=model, rows=rows, rows2=rows2)


def _state_for(prompt, grids=(GRID,)):
    return synth.position_state(prompt, grids=grids)


def _splice(rig, prompt, *, armed=True, grids=(GRID,), digest=1):
    rows = rig.rows if grids == (GRID,) else rig.rows2
    return VisionSplice(
        image_pad_token_id=PAD,
        embeddings=rows,
        image_digests=(digest,),
        pad_counts=(int(rows.shape[0]),),
        image_grids=tuple(grids),
        dense_mrope=_state_for(prompt, grids) if armed else None,
    )


_ARMED = object()  # the default splice: the prompt's image at grid positions


def _generate(rig, prompt=PROMPT, *, splice=_ARMED, sampler=GREEDY, depth=3, max_tokens=12, **kwargs):
    """``splice=None`` is a text request; the default arms the prompt's image."""
    kwargs.setdefault("verify_strategy", "capture_commit")
    kwargs.setdefault("mtp_history_policy", "committed")
    return generate_mtpk(
        rig.rt,
        list(prompt),
        max_tokens=max_tokens,
        sampler=sampler,
        speculative_depth=depth,
        stop_token_ids=set(),
        vision_splice=_splice(rig, prompt) if splice is _ARMED else splice,
        **kwargs,
    )


def _bank(out):
    return out.stats.graphbank["compiled_verify"]


def _assert_on_the_delta_trace(out, delta):
    bank = _bank(out)
    assert bank["compiled_calls"] >= 1 and bank["fallback_calls"] == 0
    assert bank["rope_delta_input"] is True and bank["rope_delta"] == delta
    assert bank["compiled_keys"] and all(k.endswith(":rope_delta") for k in bank["compiled_keys"])
    assert out.stats.compiled_verify_admission["reason"] == "admitted"
    assert out.stats.compiled_verify_admission["rope_delta"] == delta


# -- the layer -----------------------------------------------------------------


def _clone_cache(cache):
    from mlx_lm.models.cache import KVCache

    clone = []
    for entry in cache:
        if isinstance(entry, KVCache):
            twin = KVCache()
            twin.keys = mx.array(entry.keys)
            twin.values = mx.array(entry.values)
            twin.offset = int(entry.offset)
            twin.step = entry.step
        else:
            twin = type(entry)(len(entry.cache))
            for slot, leaf in enumerate(entry.cache):
                twin[slot] = None if leaf is None else mx.array(leaf)
        clone.append(twin)
    leaves = []
    for entry in clone:
        leaves.extend(entry.cache if hasattr(entry, "cache") else [entry.keys, entry.values])
    mx.eval([leaf for leaf in leaves if leaf is not None])
    return clone


def _full_attention_rows(cache, model, start, length):
    rows = []
    for layer, entry in zip(model.model.layers, cache):
        if layer.is_linear:
            continue
        keys = entry.cache[0] if isinstance(entry, TensorOffsetKVCache) else entry.keys
        values = entry.cache[1] if isinstance(entry, TensorOffsetKVCache) else entry.values
        rows.append((keys[:, :, start : start + length], values[:, :, start : start + length]))
    return rows


def test_the_bank_owned_origin_writes_the_eager_routes_rows_at_the_layer():
    """Three routes over the same pre-decode state: (A) the eager route, a stock
    cache with the scope open; (B) the compiled routes' containers stamped with
    the delta; (C) the parity instrument's reference, the same containers with
    no delta and the host table resolved. B and C write A's key rows bit for
    bit and agree bit for bit with each other; A's outputs differ from theirs
    only by the padded reduction of the fixed buffers, a text-lane property."""
    model = _install(synth.text_model(seed=7))
    n = len(PROMPT)
    state = _state_for(PROMPT)
    with dense_mrope_scope(state):
        stock = model.make_cache()
        mx.eval(model(mx.array([PROMPT]), cache=stock))
    a, b, c = _clone_cache(stock), _clone_cache(stock), _clone_cache(stock)
    for cache in (b, c):
        promoted, failures = promote_kv_cache_offsets(cache, reserve_tokens=64)
        assert promoted == 2 and not failures
    assert stamp_rope_delta(b, DELTA) == 2

    def decode(cache, ids, *, host_plan=False):
        scope = host_positions_for_tensor_offsets() if host_plan else dense_mrope_scope(state)
        with dense_mrope_scope(state), scope:
            logits = model(mx.array([ids]), cache=cache)
        mx.eval(logits)
        return logits

    first = [20, 21, 22]
    out_a, out_b, out_c = decode(a, first), decode(b, first), decode(c, first, host_plan=True)
    for (ka, va), (kb, vb), (kc, vc) in zip(
        _full_attention_rows(a, model, n, 3),
        _full_attention_rows(b, model, n, 3),
        _full_attention_rows(c, model, n, 3),
    ):
        assert mx.array_equal(ka, kb).item() and mx.array_equal(ka, kc).item()
        assert mx.array_equal(va, vb).item() and mx.array_equal(va, vc).item()
    assert mx.array_equal(out_b, out_c).item()
    assert float(mx.max(mx.abs(out_a - out_b)).item()) < 1e-5
    # The positions really are the delta's: the same rows on a delta-less
    # container are far away.
    plain = _clone_cache(stock)
    promote_kv_cache_offsets(plain, reserve_tokens=64)
    demotions.reset()
    with dense_mrope_scope(state):
        wrong = model(mx.array([first]), cache=plain)
    assert float(mx.max(mx.abs(wrong - out_b)).item()) > 1e-2
    assert demotions.snapshot()["counts"]["vision_mrope_tensor_offset_call"] >= 1

    # A rejected window: roll the three back and write other rows.
    for cache in (a, b, c):
        for layer, entry in zip(model.model.layers, cache):
            if not layer.is_linear:
                entry.trim(3)
    second = [30, 31, 32, 33]
    out_a, out_b, out_c = decode(a, second), decode(b, second), decode(c, second, host_plan=True)
    for (ka, _va), (kb, _vb), (kc, _vc) in zip(
        _full_attention_rows(a, model, n, 4),
        _full_attention_rows(b, model, n, 4),
        _full_attention_rows(c, model, n, 4),
    ):
        assert mx.array_equal(ka, kb).item() and mx.array_equal(ka, kc).item()
    assert mx.array_equal(out_b, out_c).item()
    assert float(mx.max(mx.abs(out_a - out_b)).item()) < 1e-5
    assert dense_mrope_state() is None


# -- the generation loop ---------------------------------------------------------


@pytest.mark.parametrize(
    ("sampler", "draft_sampler"),
    [(GREEDY, None), (NATIVE, NATIVE_DRAFT)],
    ids=["greedy", "native_sampled"],
)
def test_compiled_route_generates_the_scoped_eager_routes_tokens(rig, monkeypatch, sampler, draft_sampler):
    eager = _generate(rig, sampler=sampler, draft_sampler=draft_sampler, seed=7, max_tokens=16)
    assert "compiled_verify" not in eager.stats.graphbank
    monkeypatch.setenv("MTPLX_COMPILED_VERIFY", "1")
    compiled = _generate(rig, sampler=sampler, draft_sampler=draft_sampler, seed=7, max_tokens=16)
    assert compiled.tokens == eager.tokens and len(compiled.tokens) == 16
    _assert_on_the_delta_trace(compiled, DELTA)
    assert _teacher_forced_gaps(rig, PROMPT, compiled.tokens).max() < LOGIT_NOISE or sampler is NATIVE
    assert demotions.snapshot()["total"] == 0 and compiled.stats.demotions == {}
    assert dense_mrope_state() is None


def test_two_deltas_replay_one_trace_and_a_text_request_keeps_the_text_trace(rig, monkeypatch):
    eager_1 = _generate(rig)
    eager_2 = _generate(rig, PROMPT2, splice=_splice(rig, PROMPT2, grids=(GRID2,), digest=2))
    eager_text = _generate(rig, TEXT, splice=None)
    monkeypatch.setenv("MTPLX_COMPILED_VERIFY", "1")

    first = _generate(rig)
    assert first.tokens == eager_1.tokens
    _assert_on_the_delta_trace(first, DELTA)
    assert _bank(first)["traces"] >= 1

    second = _generate(rig, PROMPT2, splice=_splice(rig, PROMPT2, grids=(GRID2,), digest=2))
    assert second.tokens == eager_2.tokens
    _assert_on_the_delta_trace(second, DELTA2)
    # The delta is a graph INPUT: the second request replays the first one's
    # trace with its own value, and traces nothing.
    assert _bank(second)["traces"] == 0

    text = _generate(rig, TEXT, splice=None)
    assert text.tokens == eager_text.tokens
    bank = _bank(text)
    assert bank["compiled_calls"] >= 1 and bank["fallback_calls"] == 0
    assert bank["rope_delta_input"] is False and bank["rope_delta"] is None
    assert not any(k.endswith(":rope_delta") for k in bank["compiled_keys"])
    assert text.stats.compiled_verify_admission == {
        "family": "dense",
        "positions": "text",
        "rope_delta": None,
        "images": 0,
        "engaged": True,
        "reason": "admitted",
    }
    assert demotions.snapshot()["total"] == 0


def test_the_parity_instrument_compares_every_compiled_round_and_finds_nothing(rig, monkeypatch):
    monkeypatch.setenv("MTPLX_COMPILED_VERIFY", "1")
    on = _generate(rig, sampler=NATIVE, draft_sampler=NATIVE_DRAFT, seed=11, max_tokens=16)
    monkeypatch.setenv("MTPLX_COMPILED_VERIFY", "parity2")
    out = _generate(rig, sampler=NATIVE, draft_sampler=NATIVE_DRAFT, seed=11, max_tokens=16)
    assert out.tokens == on.tokens
    bank = _bank(out)
    assert bank["mode"] == "parity2" and bank["compiled_calls"] >= 1
    record = bank["parity2"]
    # An exact verdict needs compiled dispatches behind it: every compiled
    # round was compared, and none was left without a reference.
    assert record["rounds"] == bank["compiled_calls"] == bank["parity2_calls"]
    assert record["divergent_rounds"] == 0 and bank["parity2_divergent_calls"] == 0
    assert record["reference_scope_missing_rounds"] == 0
    assert record["positions"] == "vision_delta" and record["rope_delta"] == DELTA
    assert record["logits_max_abs_diff"] == 0.0 and record["logits_max_kl"] == 0.0
    assert record["hidden_max_abs_diff"] == 0.0 and record["state_max_abs_diff"] == 0.0
    assert record["first_divergence"] is None and bank["parity2_first_divergence"] is None
    assert record["compared_leaves_per_round"] > 2  # logits, hidden, and every leaf
    assert demotions.snapshot()["total"] == 0


def test_the_parity_instrument_needs_the_request_scope_for_its_reference(rig, monkeypatch):
    """A caller with no scope open gets its rounds counted, never compared:
    the reference would otherwise be a text-positioned cache, and a wrong
    delta would read as a divergence of the lane (or a right one as exact,
    were the reference to read the bank's own delta)."""
    from mtplx.graphbank import CompiledVerifyBank

    monkeypatch.setenv("MTPLX_COMPILED_VERIFY", "parity2")
    real = CompiledVerifyBank._parity2_check

    def scopeless(self, *args, **kwargs):
        from mtplx import dense_mrope

        token = dense_mrope._STATE.set(None)
        try:
            return real(self, *args, **kwargs)
        finally:
            dense_mrope._STATE.reset(token)

    monkeypatch.setattr(CompiledVerifyBank, "_parity2_check", scopeless)
    out = _generate(rig)
    record = _bank(out)["parity2"]
    assert record["rounds"] == 0 and record["divergent_rounds"] == 0
    assert record["reference_scope_missing_rounds"] == _bank(out)["compiled_calls"] >= 1


def _wrong_draft_head(rig):
    """Every draft is wrong, so every window is rejected and rolled back."""
    real = rig.model.mtp_forward

    def wrong(hidden_states, next_token_ids, **kwargs):
        out = real(hidden_states, next_token_ids, **kwargs)
        logits, hidden = out if kwargs.get("return_hidden") else (out, None)
        peaked = mx.zeros_like(logits) - 50.0
        target = mx.argmax(logits, axis=-1)
        rows = mx.arange(int(logits.shape[1]))
        peaked[0, rows, (target[0] + 1) % int(logits.shape[-1])] = 50.0
        return (peaked, hidden) if kwargs.get("return_hidden") else peaked

    rig.model.mtp_forward = wrong


def test_rollback_after_every_rejected_window_stays_exact(rig, monkeypatch):
    _wrong_draft_head(rig)
    eager = _generate(rig, max_tokens=10)
    # Every window rejected: one committed token per verify round (the first
    # token is the prefill's), so the bank rolled back on every round.
    assert sum(eager.stats.accepted_by_depth) == 0
    assert eager.stats.verify_calls >= len(eager.tokens) - 1 >= 9
    monkeypatch.setenv("MTPLX_COMPILED_VERIFY", "1")
    out = _generate(rig, max_tokens=10)
    assert out.tokens == eager.tokens
    assert sum(out.stats.accepted_by_depth) == 0
    assert out.stats.verify_calls == eager.stats.verify_calls
    _assert_on_the_delta_trace(out, DELTA)
    assert demotions.snapshot()["total"] == 0


def test_a_growth_demotion_mid_request_keeps_the_route_exact(rig, monkeypatch):
    # A small grant keeps the bank on the stock 256-row buffer; a request that
    # runs past it grows a granted leaf, and the bank demotes to stock
    # containers for the rest of the request.
    eager = _generate(rig, max_tokens=260)
    assert len(eager.tokens) == 260
    monkeypatch.setenv("MTPLX_COMPILED_VERIFY", "1")
    monkeypatch.setenv("MTPLX_COMPILED_VERIFY_GROWTH_RESERVE", "4")
    out = _generate(rig, max_tokens=260)
    assert out.tokens == eager.tokens
    bank = _bank(out)
    assert bank["compiled_calls"] >= 1 and bank["growth_demotions"] >= 1
    assert "growth_budget_exhausted" in bank["fallback_reasons"]
    assert bank["rope_delta_input"] is True
    # Demoted containers are stock again: no tensor offset reaches any route
    # without an origin, and nothing else is counted.
    assert demotions.snapshot()["counts"]["vision_mrope_tensor_offset_call"] == 0
    assert set(out.stats.demotions) <= {"compiled_verify_growth_demotion"}


@pytest.mark.parametrize(
    ("env", "value", "refusal"),
    [
        ("MTPLX_QWEN4_VISION_COMPILED_VERIFY", "0", "vision_kill_switch"),
        ("MTPLX_STATE_REBASE_EVERY", "64", "vision_state_rebase"),
    ],
)
def test_refused_settings_keep_every_image_request_eager_and_say_why(rig, monkeypatch, env, value, refusal):
    eager = _generate(rig)
    monkeypatch.setenv("MTPLX_COMPILED_VERIFY", "1")
    monkeypatch.setenv(env, value)
    built: list[bool] = []
    real_bank = generation.CompiledVerifyBank

    def counting_bank(*args, **kwargs):
        built.append(True)
        return real_bank(*args, **kwargs)

    monkeypatch.setattr(generation, "CompiledVerifyBank", counting_bank)

    for armed in (True, False):  # a sequential image request is refused too
        demotions.reset()
        out = _generate(rig, splice=_splice(rig, PROMPT, armed=armed))
        assert built == []
        assert "compiled_verify" not in out.stats.graphbank
        if armed:
            assert out.tokens == eager.tokens
        record = out.stats.compiled_verify_admission
        assert record["engaged"] is False and record["reason"] == refusal
        assert record["positions"] == ("vision_delta" if armed else "vision_sequential")
        assert record["rope_delta"] == (DELTA if armed else None)
        snap = demotions.snapshot()
        assert snap["counts"]["vision_request_eager_verify"] == (1 if armed else 0)
        if armed:
            assert snap["reasons"]["vision_request_eager_verify"] == (
                generation._VISION_COMPILED_VERIFY_REFUSALS[refusal]
            )
    # A text request never notices the switch.
    text = _generate(rig, TEXT, splice=None)
    assert built == [True] and _bank(text)["compiled_calls"] >= 1
    assert text.stats.compiled_verify_admission["reason"] == "admitted"


def test_the_admission_verdicts():
    admit = generation._dense_vision_compiled_verify_admission
    assert admit(None, PROMPT) == {"positions": "text", "rope_delta": None, "images": 0, "refusal": None}
    armed = SimpleNamespace(dense_mrope=_state_for(PROMPT), image_pad_token_id=PAD, pad_counts=(6,))
    sequential = SimpleNamespace(dense_mrope=None, image_pad_token_id=PAD, pad_counts=(6,))
    assert admit(armed, PROMPT) == {
        "positions": "vision_delta", "rope_delta": DELTA, "images": 1, "refusal": None
    }
    assert admit(sequential, PROMPT) == {
        "positions": "vision_sequential", "rope_delta": None, "images": 1, "refusal": None
    }
    # The switches come before the sequential answer: every image request.
    for env, refusal in (
        ("MTPLX_QWEN4_VISION_COMPILED_VERIFY", "vision_kill_switch"),
        ("MTPLX_STATE_REBASE_EVERY", "vision_state_rebase"),
    ):
        with pytest.MonkeyPatch.context() as patch:
            patch.setenv(env, "0" if refusal == "vision_kill_switch" else "8")
            assert admit(armed, PROMPT)["refusal"] == refusal
            assert admit(sequential, PROMPT)["refusal"] == refusal
            assert admit(None, PROMPT)["refusal"] is None
    # The compiled draft core owns the delta only while the draft head is
    # row-aligned with the prompt (rows past the last image row).
    assert generation._dense_draft_rope_delta(None) is None
    state = _state_for(PROMPT)
    assert generation._dense_draft_rope_delta(state) == DELTA
    state.mtp_aligned = False
    assert generation._dense_draft_rope_delta(state) is None


@pytest.mark.parametrize("armed", [True, False])
def test_dense_image_compilation_is_on_by_default(rig, monkeypatch, armed):
    # No setting: a dense image request is admitted like a Flash-Next one.
    monkeypatch.setenv("MTPLX_COMPILED_VERIFY", "1")
    monkeypatch.delenv("MTPLX_DENSE_VISION_COMPILED_VERIFY", raising=False)
    result = _generate(rig, splice=_splice(rig, PROMPT, armed=armed))
    assert result.stats.compiled_verify_admission["reason"] != "vision_dense_kill_switch"
    assert "opt_in" not in str(result.stats.compiled_verify_admission["reason"])


@pytest.mark.parametrize("armed", [True, False])
def test_dense_image_compilation_has_its_own_kill_switch(rig, monkeypatch, armed):
    monkeypatch.setenv("MTPLX_COMPILED_VERIFY", "1")
    monkeypatch.setenv("MTPLX_DENSE_VISION_COMPILED_VERIFY", "0")
    result = _generate(rig, splice=_splice(rig, PROMPT, armed=armed))
    assert result.stats.compiled_verify_admission["reason"] == "vision_dense_kill_switch"
    assert result.stats.compiled_verify_admission["engaged"] is False
    assert not (result.stats.graphbank or {}).get("compiled_verify")


def test_dense_kill_switch_does_not_change_other_family_admission(monkeypatch):
    monkeypatch.setenv("MTPLX_DENSE_VISION_COMPILED_VERIFY", "0")
    splice = SimpleNamespace(dense_mrope=None, image_pad_token_id=PAD, pad_counts=(6,))
    admit = generation._dense_vision_compiled_verify_admission
    assert admit(splice, PROMPT, dense_model=False)["refusal"] is None
    assert admit(None, TEXT)["refusal"] is None
    assert admit(splice, PROMPT, dense_model=True)["refusal"] == "vision_dense_kill_switch"


def test_the_session_bank_restores_an_image_turn_the_compiled_route_banked(rig, monkeypatch):
    monkeypatch.setenv("MTPLX_COMPILED_VERIFY", "1")
    bank = SessionBank()

    def turn(prompt, *, splice):
        return _generate(
            rig,
            prompt,
            splice=splice,
            depth=2,
            max_tokens=8,
            session_bank=bank,
            session_id="s",
            session_template_hash="t",
            session_draft_head_identity="d",
            session_policy_fingerprint="p",
            commit_prompt_state_to_bank=True,
        )

    first = turn(PROMPT, splice=_splice(rig, PROMPT))
    assert first.stats.cached_tokens == 0
    _assert_on_the_delta_trace(first, DELTA)
    longer = list(PROMPT) + [20, 21, 22, 23]
    warm = turn(longer, splice=_splice(rig, longer))
    assert warm.stats.cached_tokens == len(PROMPT)
    _assert_on_the_delta_trace(warm, DELTA)
    cold = _generate(rig, longer, splice=_splice(rig, longer), depth=2, max_tokens=8)
    assert warm.tokens == cold.tokens
    monkeypatch.delenv("MTPLX_COMPILED_VERIFY")
    eager = _generate(rig, longer, splice=_splice(rig, longer), depth=2, max_tokens=8)
    assert cold.tokens == eager.tokens
    assert warm.stats.demotions == {} and demotions.snapshot()["total"] == 0
