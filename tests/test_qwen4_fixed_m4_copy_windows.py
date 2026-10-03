"""Compiled copy windows on the Flash-Next fixed-M4 bank (MTPLX_FIXED_M4_COPY_WINDOWS=1).

A context-copy block round forwards the primary and a block of prompt tokens:
9, 13, 17 or 25 rows with the default caps. The fixed-M4 lane compiled only its
four-row verify, so every block round ran the eager forward over the promoted
bank. With the opt-in, a full-length block replays the four-row lane's verify
step traced at its width (``CompiledVerifyBank.replay_fixed_m4_copy_window``).

On the tiny random pack of ``scripts/qwen4exp_mtp_tiny_smoke.py`` (bfloat16, on
the GPU) this file pins:

* exactness: for the same bank state and inputs a compiled window gives the
  eager window's logits, hidden rows, captures and state bit for bit, at every
  ladder width, through partial and full accepts and across a bucket growth;
  on the rows-gather lane through the product entry and on the dense lane
  through the replay itself; with the serve lane's op diet and verify glue;
  and with gate values planted where MLX's compiled and eager sigmoids differ;
* one copy: after a rejection the replay writes the conversation's K, V and
  raw banks in place, because its pre-dispatch evaluation is handed only the
  leaves a commit can leave pending, never a bank;
* compile cost: one trace per width per bank capacity;
* the policy: full-length blocks on a bucketed rows-gather bank below the first
  prefill-width route, with every refusal counted;
* a program that fails to trace retires copy windows for the process, and the
  round runs eager over untouched state;
* through ``generate_mtpk``: copy rounds replay compiled with the opt-in and
  eager without it, with every round's logits and the final state bit-equal,
  and parity2 comparing every compiled window with the eager verifier.

On 4976d4b6, before this lane, every test here fails: the bank had no copy
window entry, the replay took only the four-row program (a wider window raised
"compiled verify length mismatch" inside the trace), and the host n-gram
auxiliary always returned four rows.
"""

from __future__ import annotations

import os
from functools import partial
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

import mtplx.generation as generation
import mtplx.graphbank as graphbank
from mtplx import demotions, runtime_options
from mtplx.attention_context import attention_phase, model_forward_kind
from mtplx.cache_state import snapshot_untrimmable_cache_lazy
from mtplx.context_copy import ladder_widths
from mtplx.sampling import SamplerConfig
from test_fixed_m4_dispatch_guard import _bank as _stub_bank
from test_fixed_m4_dispatch_guard import _good_fn
from test_qwen4_fixed_m4_verify_exactness import (
    GATE,
    YARN,
    _constant_projection,
    _plant_attention_gate,
    _runtime,
)

gpu = pytest.mark.skipif(
    not mx.metal.is_available(), reason="bfloat16 expert gathers need the GPU"
)

NATIVE = SamplerConfig(temperature=0.6, top_p=0.95, top_k=20)
LADDER = frozenset({9, 13, 17, 25})
# 440 tokens: a 512-row bucket at install, grown to 1,024 rows mid-plan.
PROMPT = [(7 * i + 3) % 128 for i in range(440)]
# (window width, rows kept): four-row verify rounds between copy windows of
# every ladder width, with rejections, partial and full accepts.
PLAN = (
    (4, 2), (4, 4), (9, 3), (4, 1), (13, 13), (4, 3), (17, 5), (25, 2),
    (4, 4), (25, 25), (9, 1), (13, 7), (4, 2), (17, 17), (25, 11), (4, 4),
)


@pytest.fixture(autouse=True)
def lane(monkeypatch):
    import mlx_lm.models.cache as cache_module

    import mtplx.models.qwen4_exp as qwen4_exp

    for name in tuple(os.environ):
        if name.startswith(("MTPLX_QWEN4_", "MTPLX_QSA_", "MTPLX_CONTEXT_COPY", "MTPLX_RAMP_")) or name in {
            "MTPLX_COMPILED_VERIFY",
            "MTPLX_COMPILED_VERIFY_GROWTH_RESERVE",
            "MTPLX_STATE_REBASE_EVERY",
            "MTPLX_FAMILY_CAPTURE_COMMIT",
            "MTPLX_FIXED_M4_COPY_WINDOWS",
        }:
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MTPLX_CONTEXT_COPY", "0")
    # Import-frozen switches of the serve lane: off unless a test arms them.
    monkeypatch.setattr(runtime_options, "_QWEN4_OPDIET", False)
    monkeypatch.setattr(runtime_options, "_QWEN4_VERIFY_GLUE", False)
    monkeypatch.setattr(qwen4_exp, "ArraysCache", cache_module.ArraysCache)
    monkeypatch.setattr(graphbank, "_compiled_verify_bits_gate_ok", lambda _rt: True)
    # Process-wide lane state starts fresh in every test.
    monkeypatch.setitem(graphbank._FIXED_M4_LANE, "proven", False)
    monkeypatch.setitem(graphbank._FIXED_M4_LANE, "retired", None)
    monkeypatch.setitem(graphbank._FIXED_M4_COPY_WINDOWS, "proven", set())
    monkeypatch.setitem(graphbank._FIXED_M4_COPY_WINDOWS, "retired", None)
    demotions.reset()
    yield monkeypatch
    demotions.reset()


def _rows_gather(lane) -> None:
    lane.setenv("MTPLX_QSA_GATHER", "1")
    lane.setenv("MTPLX_QSA_GATHER_MIN_CONTEXT", "16")
    lane.setenv("MTPLX_QWEN4_FIXED_M4_CAPACITY_BUCKET", "512")
    lane.setenv("MTPLX_COMPILED_VERIFY_GROWTH_RESERVE", "16")


def _serve_switches(lane) -> None:
    """The Flash-Next serve lane's op diet, verify glue and rows-gather width."""

    lane.setattr(runtime_options, "_QWEN4_OPDIET", True)
    lane.setattr(runtime_options, "_QWEN4_OPDIET_SELECTED", frozenset(runtime_options.QWEN4_OPDIET_ITEMS))
    lane.setattr(runtime_options, "_QWEN4_VERIFY_GLUE", True)
    lane.setattr(
        runtime_options, "_QWEN4_VERIFY_GLUE_SELECTED", frozenset({"qsa_rope", "qsa_rope_idx"})
    )
    lane.setenv("MTPLX_QSA_GATHER_MAX_ROWS", "32")


def _bits(value) -> np.ndarray:
    """Exact bit patterns, so equality is equality and NaN compares too."""

    if value.dtype in (mx.bfloat16, mx.float16):
        return np.array(value.view(mx.uint16))
    if value.dtype == mx.float32:
        return np.array(value.view(mx.uint32))
    return np.array(value)


def _address(value: mx.array) -> int:
    mx.eval(value)
    view = np.asarray(value.view(mx.uint8), copy=False)
    address = int(view.__array_interface__["data"][0])
    del view
    return address


def _qsa(cache):
    return [entry for entry in cache if isinstance(entry, graphbank.TensorOffsetQSACache)]


def _qsa_addresses(cache) -> dict[str, int]:
    return {
        f"{name}[{layer}]": _address(leaf)
        for layer, entry in enumerate(_qsa(cache))
        for name, leaf in (
            ("keys", entry.kv.keys),
            ("values", entry.kv.values),
            ("raw", entry.raw_keys),
            ("pooled", entry.pooled),
        )
    }


def _captures(cache) -> dict[str, np.ndarray]:
    out = {}
    for index, entry in enumerate(cache):
        for name in ("_mtplx_verify_rows", "_mtplx_verify_ple"):
            for slot, leaf in enumerate(getattr(entry, name, None) or ()):
                out[f"capture[{index}].{name}.{slot}"] = _bits(leaf)
    return out


def _state(cache) -> dict[str, np.ndarray]:
    out = {}
    for index, entry in enumerate(cache):
        if isinstance(entry, graphbank.TensorOffsetQSACache):
            end = entry.size()
            out[f"state[{index}].offset"] = np.array(end)
            out[f"state[{index}].keys"] = _bits(entry.kv.keys[:, :, :end])
            out[f"state[{index}].values"] = _bits(entry.kv.values[:, :, :end])
            out[f"state[{index}].raw"] = _bits(entry.raw_keys[:, :end])
            out[f"state[{index}].pooled"] = _bits(entry.pooled[:, : end // entry.ratio])
            continue
        for slot, leaf in enumerate(entry.state):
            if isinstance(leaf, mx.array):
                out[f"state[{index}].{slot}"] = _bits(leaf)
    return out


def _session(rt, *, route, plan=PLAN, read_state=True):
    """``plan`` on one fresh request: every round's outputs as raw bits.

    Four-row rounds replay through ``forward_fixed_m4`` in every session. A
    copy window runs ``route``: "compiled" (the product entry), "replay" (the
    replay itself, the only way the dense lane reaches it) or "eager" (the copy
    round's eager forward, the reference). Each round is taken under the copy
    round's scopes and committed through the family capture-commit, as the
    batched lane commits it. Also returns the bank and, for every compiled
    window, the QSA buffers that moved to a new allocation (a copy).

    ``read_state=False`` reads only what the decode loop reads between rounds
    (each round's logits, for sampling): reading the state after a commit
    would evaluate the recurrent state the commit left pending, which the
    decode loop never does and which hides the copy the in-place test is for.
    """

    model = rt.model
    hidden_variant = generation._resolve_runtime_base_hidden_variant(rt, None)
    cache = model.make_cache()
    rt.forward_ar(mx.array([PROMPT]), cache=cache, return_hidden=True)
    bank = graphbank.CompiledVerifyBank(rt, max_verify_len=4, request_max_tokens=4096)
    bank.install_fixed_m4(cache, prompt_ids=PROMPT, hidden_variant=hidden_variant)
    completion: list[int] = []
    rounds, moved = [], []
    for step, (width, keep) in enumerate(plan):
        ids = [(step * 13 + j * 7 + 1) % 128 for j in range(width)]
        before = snapshot_untrimmable_cache_lazy(cache)
        bank.reserve_fixed_m4_window(cache, committed_count=len(completion), window_tokens=width)
        addresses = _qsa_addresses(cache)
        with (
            attention_phase("decode_verify"),
            model_forward_kind("target_verify"),
            model.verify_capture_scope(),
        ):
            if width == 4:
                logits, hidden, _ = bank.forward_fixed_m4(
                    mx.array([ids]), host_input_ids=ids, completion_tokens=completion,
                    committed_count=len(completion), cache=cache,
                )
            elif route == "compiled":
                replayed = bank.replay_fixed_m4_copy_window(
                    mx.array([ids]), widths=LADDER, host_input_ids=ids,
                    completion_tokens=completion, committed_count=len(completion), cache=cache,
                )
                assert replayed is not None, bank.stats.get("fixed_m4_copy_windows")
                logits, hidden = replayed
            elif route == "replay":
                logits, hidden, _ = bank._forward_installed_fixed_m4(
                    mx.array([ids]), ids, completion, len(completion), cache
                )
            else:
                logits, hidden = rt.forward_ar(
                    mx.array([ids]), cache=cache, return_hidden=True,
                    hidden_variant=hidden_variant,
                )
        record = {
            "width": width,
            "capacity": _qsa(cache)[0].capacity,
            "logits": _bits(logits),
            **({"hidden": _bits(hidden), **_captures(cache)} if read_state else {}),
        }
        if width != 4 and route != "eager":
            after = _qsa_addresses(cache)
            moved.append({name for name, address in addresses.items() if after[name] != address})
        assert model.language_model.model.commit_verified_window(
            cache, before.states, keep_tokens=keep, verified_tokens=width,
        )
        completion.extend(ids[:keep])
        if read_state:
            record.update(_state(cache))
        rounds.append(record)
    return rounds, bank, moved


def _assert_same_rounds(candidate: list[dict], reference: list[dict]) -> None:
    assert len(candidate) == len(reference) > 0
    for index, (got, want) in enumerate(zip(candidate, reference)):
        assert got.keys() == want.keys(), index
        for name, value in want.items():
            if isinstance(value, np.ndarray):
                assert got[name].shape == value.shape, (index, got["width"], name)
                differing = int(np.count_nonzero(got[name] != value))
                assert differing == 0, (index, got["width"], name, differing)
            else:
                assert got[name] == value, (index, name)


# -- exactness -----------------------------------------------------------------


@gpu
@pytest.mark.parametrize("switches", ["default", "serve"])
def test_every_ladder_width_is_the_eager_window_bit_for_bit(lane, switches):
    _rows_gather(lane)
    if switches == "serve":
        _serve_switches(lane)
    rt = _runtime()
    compiled, bank, moved = _session(rt, route="compiled")
    eager, _, _ = _session(rt, route="eager")
    _assert_same_rounds(compiled, eager)
    # The plan crossed a bucket edge, and every width ran on both buckets'
    # banks or on one of them; nothing fell back.
    capacities = [record["capacity"] for record in compiled]
    assert capacities[0] == 512 and capacities[-1] == 1024
    report = bank.to_dict()["fixed_m4_copy_windows"]
    assert report["compiled"] == {"9": 2, "13": 2, "17": 2, "25": 3}
    assert report["eager"] == {} and report["retired"] is None


@gpu
@pytest.mark.parametrize(
    "site", ["attention gate", "hyper-connection inject", "shared-expert gate", "yarn amplitude"]
)
def test_planted_gate_values_give_the_eager_window(lane, site):
    """The values where MLX 0.32.2's fused and standalone kernels part ways
    (see tests/test_qwen4_fixed_m4_verify_exactness.py), at every width."""

    _rows_gather(lane)
    rt = _runtime(rope=YARN) if site == "yarn amplitude" else _runtime()
    args = rt.model.language_model.args
    layers = rt.model.language_model.model.layers
    if site == "attention gate":
        _plant_attention_gate(rt, lane, GATE)
    elif site == "hyper-connection inject":
        width = args.hc_count * args.hidden_size
        for layer in layers:
            for connection in (layer.attn_hyper_connection, layer.mlp_hyper_connection):
                inject = _constant_projection(width, args.hc_count, GATE * args.hc_count)
                lane.setattr(connection, "block_inject_weight", inject)
    elif site == "shared-expert gate":
        for layer in layers:
            lane.setattr(layer.mlp, "shared_expert_gate", _constant_projection(args.hidden_size, 1, GATE))
            down = layer.mlp.switch_mlp.down_proj
            zero = mx.zeros_like(down.weight)
            mx.eval(zero)
            lane.setattr(down, "weight", zero)
    compiled, _, _ = _session(rt, route="compiled")
    eager, _, _ = _session(rt, route="eager")
    _assert_same_rounds(compiled, eager)


@gpu
def test_the_replay_is_exact_on_the_dense_lane_which_the_product_keeps_eager(lane):
    lane.setenv("MTPLX_QSA_GATHER", "0")
    lane.setenv("MTPLX_COMPILED_VERIFY_GROWTH_RESERVE", "16")
    rt = _runtime()
    replayed, bank, moved = _session(rt, route="replay")
    eager, _, _ = _session(rt, route="eager")
    _assert_same_rounds(replayed, eager)
    # The dense bank keeps its exact capacity (the SDPA width is arithmetic),
    # so it changes as the answer grows and every change is a new trace of
    # every width: the product keeps this lane's copy rounds eager.
    assert len({record["capacity"] for record in replayed}) > 2
    assert bank.fixed_m4_copy_window_refusal(9) == "capacity_not_bucketed"


# -- one copy ------------------------------------------------------------------------

# Every copy window follows a rejection, whose commit leaves the recurrent state
# pending: the case in which the replay's pre-dispatch evaluation used to hold
# the banks and the window copied K and V instead of writing them in place.
AFTER_REJECTIONS = (
    (4, 2), (9, 3), (4, 1), (13, 2), (4, 3), (17, 5), (4, 1), (25, 3),
    (9, 1), (13, 4), (17, 2), (25, 7), (4, 2), (25, 11), (9, 2), (4, 4),
)


@gpu
def test_copy_windows_write_the_conversation_in_place(lane):
    """At Flash-Next's QSA buffer geometry (K and V [1, 2, capacity, 256],
    index keys 128 wide), where the capacity rule keeps every bank donatable,
    no window moves a K, V or raw bank. (The pooled bank's per-round copy is
    the four-row lane's too and is not asserted either way here.)
    """

    _rows_gather(lane)
    rt = _runtime(head_dim=256, indexer_head_dim=128)
    _rounds, bank, moved = _session(rt, route="compiled", plan=AFTER_REJECTIONS, read_state=False)
    assert len(moved) == sum(width != 4 for width, _keep in AFTER_REJECTIONS)
    copies = [
        (window, sorted(name for name in names if not name.startswith("pooled")))
        for window, names in enumerate(moved)
    ]
    assert all(not names for _window, names in copies), copies
    assert bank.to_dict()["fixed_m4_copy_windows"]["eager"] == {}


def test_a_window_hands_its_pending_evaluation_no_bank(lane):
    """Why the windows above write in place, without the GPU's timing.

    Whether a held bank is copied depends on whether the pending evaluation
    has completed by the time the step's writes are encoded, so the test
    above can pass on a lucky run with the old call. What the evaluation is
    handed does not depend on timing: a copy window hands over the auxiliary
    and the leaves a commit can leave pending (the QSA offset and the
    recurrent state), never a K, V, raw or pooled bank. A verify round keeps
    the call it shipped with.
    """

    handed = []
    lane.setattr(graphbank.mx, "async_eval", lambda *arrays: handed.append(arrays))
    lane.setattr(graphbank.mx, "eval", lambda *_a, **_k: None)
    lane.setitem(graphbank._FIXED_M4_LANE, "proven", True)
    first_call = {}
    for width in (4, 9):
        bank, _qsa, _gdn = _copy_stub()
        bank._fixed_m4_dispatch["boundary"] = "both"
        handed.clear()
        bank._forward_installed_fixed_m4(
            SimpleNamespace(shape=(1, width)), list(range(width)), [1, 2], 2, []
        )
        first_call[width] = [getattr(leaf, "name", leaf) for leaf in handed[0]]
    assert first_call[9] == ["aux", "qsa0.offset", "gdn0.conv", "gdn0.state"]
    assert first_call[4] == [
        "aux", "qsa0.k", "qsa0.v", "qsa0.offset", "qsa0.raw", "qsa0.pooled",
        "gdn0.conv", "gdn0.state",
    ]


# -- compile cost ------------------------------------------------------------------


@gpu
def test_each_width_traces_once_per_bank_capacity(lane):
    _rows_gather(lane)
    rounds, bank, _ = _session(_runtime(), route="compiled")
    ran_at: dict[str, set[int]] = {}
    for record in rounds:
        ran_at.setdefault(str(record["width"]), set()).add(record["capacity"])
    report = bank.to_dict()["fixed_m4_copy_windows"]
    copy_widths = {width: capacities for width, capacities in ran_at.items() if width != "4"}
    assert report["traces"] == {width: len(capacities) for width, capacities in copy_widths.items()}
    # The request's whole bill: one trace per width and capacity, the
    # four-row program's included.
    assert bank.stats["traces"] == sum(len(capacities) for capacities in ran_at.values())


# -- the policy -------------------------------------------------------------------


def _copy_stub(program=_good_fn, *, bucket=512, rows_gather=True):
    """A bare bank (no model, no GPU) with a fixed-M4 plan and a copy program."""

    bank, qsa, gdn = _stub_bank(_good_fn)
    qsa.fixed_rows_gather = rows_gather
    bank._fixed_m4_dispatch.update(
        capacity_bucket=bucket, qsa_entries=(qsa,), hidden_variant=None
    )
    bank.stats["traces"] = 0
    bank._fixed_m4_program = lambda width: ((_good_fn, None) if width == 4 else (program, None))
    return bank, qsa, gdn


def _window(bank, width, widths=LADDER):
    ids = list(range(width))
    return bank.replay_fixed_m4_copy_window(
        SimpleNamespace(shape=(1, width)), widths=widths, host_input_ids=ids,
        completion_tokens=[1, 2], committed_count=2, cache=[],
    )


@pytest.fixture()
def stub_mx(lane):
    lane.setattr(graphbank.mx, "async_eval", lambda *_a, **_k: None)
    lane.setattr(graphbank.mx, "eval", lambda *_a, **_k: None)
    return lane


def test_the_opt_in_is_off_by_default(lane):
    assert graphbank.fixed_m4_copy_windows_enabled() is False
    lane.setenv("MTPLX_FIXED_M4_COPY_WINDOWS", "1")
    assert graphbank.fixed_m4_copy_windows_enabled() is True


def test_the_ladder_widths_follow_the_caps(lane):
    assert ladder_widths((24, 8)) == LADDER  # MTPLX_CONTEXT_COPY_K 24, probation 8
    assert ladder_widths((16, 8)) == {9, 13, 17}
    assert ladder_widths((32, 8)) == {9, 13, 17, 25, 33}
    lane.setenv("MTPLX_RAMP_ENABLED", "1")
    lane.setenv("MTPLX_RAMP_BLOCK", "48")
    assert ladder_widths((48, 8)) == {49}


def test_windows_serve_only_below_the_first_prefill_route(lane):
    from mtplx.models import qwen4_exp

    thresholds = (
        qwen4_exp._HC_COMPILE_MIN_ROWS,
        qwen4_exp._GDN_GATED_NORM_MIN_ROWS,
        qwen4_exp._GDN_PREFILL_PREWORK_MIN_ROWS,
        qwen4_exp._MOE_PREFILL_COMBINE_MIN_ROWS,
        qwen4_exp._QSA_DENSE_BAND_SDPA_MIN_ROWS,
    )
    assert qwen4_exp.verify_route_max_rows() == min(thresholds) - 1 == 31
    assert max(LADDER) <= qwen4_exp.verify_route_max_rows()


def test_every_refusal_is_counted_and_leaves_the_round_to_the_eager_forward(stub_mx):
    lane = stub_mx
    bank, qsa, _gdn = _copy_stub()
    original = list(qsa.kv.cache)
    lane.setitem(graphbank._FIXED_M4_LANE, "proven", False)
    assert _window(bank, 9) is None  # the four-row lane has not proven itself
    lane.setitem(graphbank._FIXED_M4_LANE, "proven", True)
    assert _window(bank, 10) is None  # a cut-short block keeps the eager forward
    assert _window(bank, 33, widths=ladder_widths((32, 8))) is None  # a prefill-width route
    bank._fixed_m4_dispatch["capacity_bucket"] = 0
    assert _window(bank, 9) is None  # admission took the bucket away
    bank._fixed_m4_dispatch["capacity_bucket"] = 512
    qsa.fixed_rows_gather = False
    assert _window(bank, 9) is None  # the dense lane
    lane.setitem(graphbank._FIXED_M4_LANE, "retired", "RuntimeError: kernel refused")
    qsa.fixed_rows_gather = True
    assert _window(bank, 9) is None
    assert bank.stats["fixed_m4_copy_windows"]["eager"] == {
        "lane_not_proven": 1,
        "width_off_the_ladder": 1,
        "width_past_verify_routes": 1,
        "capacity_not_bucketed": 2,
        "lane_retired": 1,
    }
    assert all(a is b for a, b in zip(qsa.kv.cache, original))
    assert bank.stats["compiled_calls"] == 0


def test_a_clean_window_replays_and_later_windows_skip_the_trace_check(stub_mx):
    lane = stub_mx
    lane.setitem(graphbank._FIXED_M4_LANE, "proven", True)
    traced = []

    def program(input_ids, *args):
        traced.append(tuple(input_ids.shape))
        return _good_fn(input_ids, *args)

    bank, qsa, gdn = _copy_stub(program)
    first = _window(bank, 13)
    second = _window(bank, 13)
    assert first == second == ("logits", "hidden")
    # One call to trace on the live inputs, then one replay per window.
    assert traced == [(1, 13), (1, 13), (1, 13)]
    assert graphbank._FIXED_M4_COPY_WINDOWS["proven"] == {13}
    assert bank.stats["fixed_m4_copy_windows"]["compiled"] == {"13": 2}
    assert bank.stats["compiled_calls"] == 2 and bank.last_dispatch_route("ccopy_bank") == "ccopy_bank"
    # The replay installed its outputs like a verify round.
    assert qsa.kv.cache[0].name == "out0" and gdn._mtplx_verify_rows[0].name == "cap0"
    assert "fixed_m4_copy_window_host_s" in bank.stats and "fixed_m4_host_s" not in bank.stats


def test_a_program_that_fails_to_trace_retires_copy_windows_and_the_round_runs_eager(stub_mx, capsys):
    lane = stub_mx
    lane.setitem(graphbank._FIXED_M4_LANE, "proven", True)
    calls = []

    def refused(*_args):
        calls.append(1)
        raise RuntimeError("[compile] cannot read an array while it is traced")

    bank, qsa, gdn = _copy_stub(refused)
    before = (list(qsa.kv.cache), qsa.raw_keys, qsa.pooled, list(gdn.cache))
    for _ in range(3):
        assert _window(bank, 25) is None
    assert calls == [1]  # traced once; never dispatched again
    # The state the round found, leaf for leaf: nothing was rebound.
    assert all(a is b for a, b in zip(qsa.kv.cache, before[0]))
    assert qsa.raw_keys is before[1] and qsa.pooled is before[2]
    assert all(a is b for a, b in zip(gdn.cache, before[3]))
    assert not hasattr(gdn, "_mtplx_verify_rows")
    assert "cannot read an array" in graphbank.fixed_m4_copy_windows_retired_reason()
    # The four-row lane is not touched.
    assert graphbank.fixed_m4_lane_retired_reason() is None
    assert bank.stats["fixed_m4_copy_windows"]["eager"] == {"copy_windows_retired": 3}
    assert bank.stats["compiled_calls"] == 0
    assert graphbank.fixed_m4_copy_windows_retired_reason().startswith("RuntimeError")
    assert demotions.counts()["fixed_m4_copy_windows_retired"] == 1
    out = capsys.readouterr().out
    assert out.count("compiled copy windows could not dispatch") == 1
    assert "output is unchanged" in out


# -- the host n-gram auxiliary -------------------------------------------------------


def test_the_sidecar_auxiliary_returns_one_row_per_window_token():
    """The staged sidecar lane (every shipping Flash-Next pack) at copy widths.

    The rows of a position depend on that token and the two before it, so a
    window's first four rows are the four-row window's rows.
    """

    from mtplx.models.qwen4_exp import _ngram_rows_np
    from mtplx.qwen4_fixed_verify import _FixedM4SidecarAux

    heads, row_dim = 4, 3
    rows = partial(
        _ngram_rows_np,
        mult=np.array([3, 5, 7], dtype=np.int64),
        sizes=np.array([11, 13, 17, 19], dtype=np.int64),
        offs=np.array([0, 11, 24, 41], dtype=np.int64),
        eos=0,
        ngram_size=3,
        heads_per_ngram=2,
    )

    def gather(flat):
        # Each gathered row carries its own row id, so the output says which
        # n-gram row landed where.
        return np.repeat(np.asarray(flat, dtype=np.float32)[:, None], row_dim, axis=1)

    aux = _FixedM4SidecarAux(
        prompt_tail=(5, 6), rows=rows, gather=gather, output_dim=heads * row_dim
    )
    window = [9, 10, 11, 12, 13, 14, 15, 16, 17]
    wide = aux(None, window, [5, 6], 2)
    narrow = aux(None, window[:4], [5, 6], 2)
    assert wide.shape == (1, 9, heads * row_dim)
    assert narrow.shape == (1, 4, heads * row_dim)
    np.testing.assert_array_equal(wide[:, :4], narrow)
    expected, _history = rows(np.asarray([window]), np.asarray([[5, 6]]))
    np.testing.assert_array_equal(wide.reshape(9, heads, row_dim)[..., 0], expected[0])


# -- through generate_mtpk -----------------------------------------------------------


def _walk():
    """A prompt made of a 300-token walk plus a quote of walk[45:50].

    The first primary continues the quote (walk[50]), so the history's tail
    matches the walk once, at 51; every later round copies the walk onward.
    """

    rng = np.random.default_rng(7)
    walk = [int(token) for token in rng.integers(1, 128, size=300)]
    return walk, walk + walk[45:50]


def _forced_copy_run(rt, lane, *, copy_windows: bool, mode: str = "1"):
    """generate_mtpk where every round is a copy round on the batched lane.

    The primary and each copy round's acceptance are forced to continue the
    walk, after the real samplers ran (so the generator is drawn as in a real
    run): odd rounds accept the whole block, even rounds half of it and take
    the next walk token as the correction. Returns the result and every copy
    round's logits rows as raw bits.
    """

    walk, prompt = _walk()
    lane.setenv("MTPLX_CONTEXT_COPY", "1")
    lane.setenv("MTPLX_FAMILY_CAPTURE_COMMIT", "1")
    lane.setenv("MTPLX_COMPILED_VERIFY", mode)
    lane.setenv("MTPLX_QSA_GATHER_MAX_ROWS", "32")  # the serve lane's width
    if copy_windows:
        lane.setenv("MTPLX_FIXED_M4_COPY_WINDOWS", "1")
    else:
        lane.delenv("MTPLX_FIXED_M4_COPY_WINDOWS", raising=False)
    # Every round is a copy round, so no four-row round proves the lane in
    # this request; its own proof is tests/test_fixed_m4_dispatch_guard.py.
    lane.setitem(graphbank._FIXED_M4_LANE, "proven", True)
    cursor = [50]
    copy_rounds: list[tuple[int, np.ndarray]] = []
    real_sample = generation._sample_from_logits
    real_accept = generation._point_mass_block_accept

    def primary(*args, **kwargs):
        _token, distribution = real_sample(*args, **kwargs)
        token = walk[cursor[0]]
        cursor[0] += 1
        return token, distribution

    def accept(block_logits, block, sampler, rng):
        real_accept(block_logits, block, sampler, rng)
        assert list(block) == walk[cursor[0] : cursor[0] + len(block)]
        copy_rounds.append((len(block) + 1, _bits(block_logits)))
        keep = len(block) if len(copy_rounds) % 2 else len(block) // 2
        cursor[0] += keep
        if keep == len(block):
            return keep, None
        cursor[0] += 1
        return keep, walk[cursor[0] - 1]

    with lane.context() as patch:
        patch.setattr(generation, "_sample_from_logits", primary)
        patch.setattr(generation, "_point_mass_block_accept", accept)
        result = generation.generate_mtpk(
            rt, list(prompt), max_tokens=96, sampler=NATIVE, draft_sampler=NATIVE,
            speculative_depth=3, seed=1234, mtp_cache_policy="persistent",
            mtp_history_policy="committed", verify_strategy="batched",
            stop_token_ids=set(), capture_final_state=True,
        )
    return result, copy_rounds


def _final_bits(result) -> list[np.ndarray]:
    final = result.final_state
    assert final is not None and final.safe_to_commit
    leaves = [_bits(final.final_logits), _bits(final.final_hidden)]
    for cache in (final.final_trunk_cache, final.final_committed_mtp_cache):
        for entry in cache:
            for leaf in entry.state:
                if isinstance(leaf, mx.array):
                    leaves.append(_bits(leaf))
    return leaves


def _copy_routes(result) -> list[tuple[int, str]]:
    return [
        (int(event["verify_width"]), event["verify_route"])
        for event in result.stats.events
        if (event.get("context_copy") or {}).get("lane") == "batched"
    ]


@gpu
def test_generation_replays_ladder_windows_and_matches_the_eager_run(lane):
    _rows_gather(lane)
    rt = _runtime()
    off, off_rounds = _forced_copy_run(rt, lane, copy_windows=False)
    on, on_rounds = _forced_copy_run(rt, lane, copy_windows=True)

    assert list(on.tokens) == list(off.tokens)
    assert [width for width, _ in on_rounds] == [width for width, _ in off_rounds]
    for (width, got), (_, want) in zip(on_rounds, off_rounds):
        assert got.shape == want.shape and np.array_equal(got, want), width
    final_on, final_off = _final_bits(on), _final_bits(off)
    assert len(final_on) == len(final_off)
    assert all(np.array_equal(a, b) for a, b in zip(final_on, final_off))

    routes_on, routes_off = _copy_routes(on), _copy_routes(off)
    widths = [width for width, _ in routes_on]
    assert {9, 25} <= set(widths) and len(widths) == len(on_rounds)
    assert routes_off == [(width, "ccopy_block") for width in widths]
    assert routes_on == [
        (width, "ccopy_bank" if width in LADDER else "ccopy_block") for width in widths
    ]
    report = on.stats.graphbank["compiled_verify"]["fixed_m4_copy_windows"]
    assert sum(report["compiled"].values()) == sum(width in LADDER for width in widths)
    assert report["eager"] == (
        {"width_off_the_ladder": sum(width not in LADDER for width in widths)}
        if any(width not in LADDER for width in widths)
        else {}
    )
    assert "fixed_m4_copy_windows" not in off.stats.graphbank["compiled_verify"]
    assert on.stats.context_copy_disabled_reason is None


@gpu
def test_parity2_compares_every_compiled_window_with_the_eager_verifier(lane):
    _rows_gather(lane)
    result, rounds = _forced_copy_run(_runtime(), lane, copy_windows=True, mode="parity2")
    bank = result.stats.graphbank["compiled_verify"]
    record = bank["fixed_m4_parity2"]
    compiled = {str(width): count for width, count in bank["fixed_m4_copy_windows"]["compiled"].items()}
    assert set(compiled) >= {"9", "25"}
    assert record["rounds_by_width"] == compiled
    assert record["compiled_rounds"] == sum(compiled.values()) == bank["compiled_calls"]
    assert record["divergent_rounds"] == 0, record["first_divergence"]
    for name in ("logits_max_abs_diff", "hidden_max_abs_diff", "state_max_abs_diff", "capture_max_abs_diff"):
        assert record[name] == 0.0, (name, record[name])
    assert len(rounds) >= sum(compiled.values())
