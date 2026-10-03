"""Flash-Next's compiled fixed-M4 verifier against its eager verifier, round by round.

The tiny random pack of ``scripts/qwen4exp_mtp_tiny_smoke.py`` (trunk and MTP
head, bfloat16, one PLE layer, an M-RoPE contract: what the fixed-M4 lane
needs), through the whole generation loop with MTPLX_COMPILED_VERIFY=parity2:
the compiled lane stays authoritative and every round's logits, hidden
states, recurrent states and captures are compared with the eager verifier.
Each case plants one value where MLX 0.32.2's compiled graph and its eager
kernels part ways:

* the attention output gate at -6.84375, where the fused bfloat16 sigmoid
  (fast exp) and the standalone one (precise exp) differ, on the dense-mask
  and the rows-gather lanes;
* the same value at the two other sigmoids the verify trace fuses: the
  hyper-connection inject gate (after its divide by hc_count) and the
  shared-expert gate of the MoE block;
* a static-YaRN amplitude (factor 4: 0.1 * ln 4 + 1 = 1.1386294), which a
  fused kernel would hold as a 7-significant-digit constant.

The tiny pack's bfloat16 streams round a moved amplitude away before it
reaches the logits, so the YaRN case here guards the route while
tests/test_qwen4_yarn_amplitude.py is the one that fails on a 7-digit
amplitude.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import math
import os
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.utils
import numpy as np
import pytest

import mtplx.generation as generation
import mtplx.graphbank as graphbank
from mtplx import demotions
from mtplx.sampling import SamplerConfig

pytestmark = pytest.mark.skipif(
    not mx.metal.is_available(), reason="the compiled verifier's parity is a GPU property"
)

NATIVE = SamplerConfig(temperature=0.6, top_p=0.95, top_k=20)
PROMPT = [3, 5, 7, 9, 11, 13] + list(range(20, 54))
MAX_TOKENS = 24
GATE = -6.84375
YARN = {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 16}


def _smoke():
    path = Path(__file__).resolve().parents[1] / "scripts" / "qwen4exp_mtp_tiny_smoke.py"
    spec = importlib.util.spec_from_file_location("qwen4exp_mtp_tiny_smoke", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def _lane(monkeypatch):
    import mlx_lm.models.cache as cache_module

    import mtplx.models.qwen4_exp as qwen4_exp

    for name in tuple(os.environ):
        if name.startswith("MTPLX_QWEN4_") or name.startswith("MTPLX_QSA_GATHER") or name in {
            "MTPLX_COMPILED_VERIFY",
            "MTPLX_STATE_REBASE_EVERY",
            "MTPLX_FAMILY_CAPTURE_COMMIT",
        }:
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MTPLX_CONTEXT_COPY", "0")
    # A runtime load earlier in the session swaps mlx-lm's ArraysCache for the
    # leak-free class; the bank looks the class up per call.
    monkeypatch.setattr(qwen4_exp, "ArraysCache", cache_module.ArraysCache)
    monkeypatch.setattr(graphbank, "_compiled_verify_bits_gate_ok", lambda _rt: True)
    demotions.reset()
    yield
    demotions.reset()


def _runtime(*, dtype=mx.bfloat16, rope=None, **overrides):
    from mtplx.models.qwen4_exp import Model, ModelArgs, Qwen4ExpMTP
    from mtplx.mtp_patch import validate_mtp_support
    from mtplx.qwen4_fixed_verify import install_qwen4_fixed_verify_route

    smoke = _smoke()
    mx.random.seed(0)
    fields = {
        "head_dim": 32,
        "indexer_head_dim": 32,
        "indexer_compress_ratio": 4,
        "ple_layer_ids": [1],
        "rope_parameters": {
            "mrope_interleaved": True,
            "mrope_section": [2, 1, 1],
            "partial_rotary_factor": 0.25,
            "rope_theta": 10000000,
            "rope_type": "default",
            **(rope or {}),
        },
    }
    fields.update(overrides)
    args = dataclasses.replace(smoke._tiny_text_args(), **fields)
    model = Model(ModelArgs(model_type="qwen4_exp", text_config=dataclasses.asdict(args)))
    model.language_model.mtp = Qwen4ExpMTP(model.language_model.args)
    if dtype != mx.float32:
        model.update(
            mlx.utils.tree_map(
                lambda p: p.astype(dtype) if p.dtype == mx.float32 else p,
                model.parameters(),
            )
        )
    model.eval()
    mx.eval(model.parameters())
    assert validate_mtp_support(model)
    rt = smoke._tiny_runtime(model)
    install_qwen4_fixed_verify_route(rt)
    return rt


def _generate(rt, mode, monkeypatch):
    monkeypatch.setenv("MTPLX_COMPILED_VERIFY", mode)
    return generation.generate_mtpk(
        rt,
        list(PROMPT),
        max_tokens=MAX_TOKENS,
        sampler=NATIVE,
        draft_sampler=NATIVE,
        speculative_depth=3,
        seed=1234,
        mtp_cache_policy="persistent",
        mtp_history_policy="committed",
        verify_strategy="batched",
        stop_token_ids=set(),
    )


def _bank(result) -> dict:
    return (result.stats.graphbank or {}).get("compiled_verify") or {}


def _assert_every_round_exact(result):
    bank = _bank(result)
    record = bank["fixed_m4_parity2"]
    assert record["compiled_rounds"] == bank["compiled_calls"] > 0, record
    assert record["divergent_rounds"] == 0, record["first_divergence"]
    for name in (
        "logits_max_abs_diff",
        "hidden_max_abs_diff",
        "state_max_abs_diff",
        "capture_max_abs_diff",
    ):
        assert record[name] == 0.0, (name, record[name])


def _plant_attention_gate(rt, monkeypatch, value):
    """Every attention layer's gate half of q_proj reads ``value``."""

    planted = 0
    for layer in rt.model.language_model.model.layers:
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            continue
        real, heads = attn.q_proj, attn.n_heads

        def q_proj(x, real=real, heads=heads):
            out = real(x)
            queries, _gate = mx.split(out.reshape(*out.shape[:-1], heads, -1), 2, axis=-1)
            gate = mx.full(queries.shape, value, dtype=out.dtype)
            return mx.concatenate([queries, gate], axis=-1).reshape(out.shape)

        monkeypatch.setattr(attn, "q_proj", q_proj)
        planted += 1
    assert planted


def _constant_projection(in_dims: int, out_dims: int, value: float) -> nn.Linear:
    """A bfloat16 projection that outputs ``value`` in every row.

    Zero weight and a bias: the value comes out of the projection's own
    kernel, so the fused sigmoid downstream reads it from memory, as it reads
    a real projection's output.
    """

    projection = nn.Linear(in_dims, out_dims, bias=True)
    projection.weight = mx.zeros((out_dims, in_dims), dtype=mx.bfloat16)
    projection.bias = mx.full((out_dims,), value, dtype=mx.bfloat16)
    mx.eval(projection.parameters())
    return projection


@pytest.mark.parametrize("lane", ["dense-mask", "rows-gather"])
def test_the_attention_gate_matches_eager_every_round(monkeypatch, lane):
    if lane == "rows-gather":
        monkeypatch.setenv("MTPLX_QSA_GATHER", "1")
        monkeypatch.setenv("MTPLX_QSA_GATHER_MIN_CONTEXT", "1")
    rt = _runtime()
    _plant_attention_gate(rt, monkeypatch, GATE)
    result = _generate(rt, "parity2", monkeypatch)
    assert result.stats.fixed_m4_admission["reason"] == "admitted"
    _assert_every_round_exact(result)


@pytest.mark.parametrize("site", ["hyper-connection inject", "shared-expert gate"])
def test_the_inject_and_shared_expert_gates_match_eager_every_round(monkeypatch, site):
    rt = _runtime()
    args = rt.model.language_model.args
    width = args.hc_count * args.hidden_size
    planted = 0
    for layer in rt.model.language_model.model.layers:
        if site == "shared-expert gate":
            gate = _constant_projection(args.hidden_size, 1, GATE)
            monkeypatch.setattr(layer.mlp, "shared_expert_gate", gate)
            # Zero routed experts: the block's output is the gated shared
            # expert alone, and a moved gate is not rounded away in the sum.
            down = layer.mlp.switch_mlp.down_proj
            zero = mx.zeros_like(down.weight)
            mx.eval(zero)
            monkeypatch.setattr(down, "weight", zero)
            planted += 1
            continue
        for connection in (layer.attn_hyper_connection, layer.mlp_hyper_connection):
            # inject = 2 * sigmoid(logits / hc_count): the sigmoid reads GATE.
            inject = _constant_projection(width, args.hc_count, GATE * args.hc_count)
            monkeypatch.setattr(connection, "block_inject_weight", inject)
            planted += 1
    assert planted
    result = _generate(rt, "parity2", monkeypatch)
    assert result.stats.fixed_m4_admission["reason"] == "admitted"
    _assert_every_round_exact(result)


def test_a_yarn_amplitude_matches_eager_every_round(monkeypatch):
    from mtplx.models.qwen4_exp import _rope_inv_freq_and_scaling_for

    rt = _runtime(rope=YARN)
    _inv_freq, scaling = _rope_inv_freq_and_scaling_for(rt.model.language_model.args)
    exact = np.float32(scaling)
    assert scaling == pytest.approx(0.1 * math.log(4.0) + 1.0)
    assert np.float32(float(f"{float(exact):.7g}")) != exact  # a 7-digit constant would move it
    result = _generate(rt, "parity2", monkeypatch)
    _assert_every_round_exact(result)


def _hc_read_served_here() -> bool:
    """The hyper-connection read's device route on this GPU (M1, M2 and an
    Ultra keep the stock chain; single-die generation 15 and newer take it)."""

    from mtplx.kernels import hc_verify_read
    from mtplx.nax_detect import gpu_architecture

    return bool(hc_verify_read.device_route(gpu_architecture())[0])


def test_the_hyper_connection_read_engages_in_the_compiled_verifier(monkeypatch):
    # The route install proves the verify-width read on this GPU; the
    # compiled verifier then traces it into every round, and every round
    # still equals the eager verifier. Where the device route keeps the
    # stock chain, the install says so and nothing engages.
    from mtplx.kernels import hc_verify_read

    rt = _runtime()
    report = rt._mtplx_hc_verify_read
    traced_before = hc_verify_read.engagement()["traces"]
    result = _generate(rt, "parity2", monkeypatch)
    assert result.stats.fixed_m4_admission["reason"] == "admitted"
    _assert_every_round_exact(result)
    if not _hc_read_served_here():
        assert not report["installed"], report
        assert hc_verify_read.engagement()["traces"] == traced_before
        return
    assert report["installed"], report
    assert report["rows"] == (4,)
    assert hc_verify_read.engagement()["traces"] > traced_before
    assert 4 in hc_verify_read.engagement()["engaged_rows"]


def test_the_qsa_verify_selection_engages_in_the_compiled_verifier(monkeypatch):
    # The route install proves the fixed bank's two-kernel QSA selection on
    # this GPU; the compiled verifier traces it into every round (the tiny
    # pack keeps 2 of its blocks per query, so the selection is real), and
    # every round still equals the eager verifier.
    from mtplx.kernels import qsa_verify_select

    rt = _runtime()
    report = rt._mtplx_qsa_verify_select
    assert report["installed"], report
    assert report["rows"] == (4,)
    traced_before = qsa_verify_select.engagement()["traces"]
    result = _generate(rt, "parity2", monkeypatch)
    assert result.stats.fixed_m4_admission["reason"] == "admitted"
    _assert_every_round_exact(result)
    assert qsa_verify_select.engagement()["traces"] > traced_before
    assert (4, "dense") in qsa_verify_select.engagement()["engaged"]


def _raw_bytes(a: mx.array) -> np.ndarray:
    """The array's storage as unsigned integers: equality here is bit equality
    (a signed zero or a NaN payload counts as a difference)."""

    if a.dtype in (mx.bfloat16, mx.float16):
        return np.array(mx.view(a, mx.uint16))
    host = np.array(a)
    return host.reshape(-1).view(np.uint8) if host.dtype.kind in "fc" else host


def _record_compiled_rounds(monkeypatch) -> list[dict]:
    """Every compiled round's outputs as the verifier returned them: logits,
    hidden state, captures and each cache leaf, stored as raw bytes."""

    rounds: list[dict] = []
    real = graphbank.CompiledVerifyBank._fixed_m4_parity2_compare

    def spy(self, dispatch, clone, input_ids, **kw):
        if kw["dispatch_kind"] == "compiled":
            leaves = {"logits": kw["candidate_logits"], "hidden": kw["candidate_hidden"]}
            for name, value in kw["candidate_captures"].items():
                leaves[f"capture:{name}"] = value
            for index, value in enumerate(kw["candidate_state"]):
                leaves[f"state:{index}"] = value
            arrays = {k: v for k, v in leaves.items() if isinstance(v, mx.array)}
            mx.eval(list(arrays.values()))
            rounds.append({k: (v.dtype, tuple(v.shape), _raw_bytes(v)) for k, v in arrays.items()})
        return real(self, dispatch, clone, input_ids, **kw)

    monkeypatch.setattr(graphbank.CompiledVerifyBank, "_fixed_m4_parity2_compare", spy)
    return rounds


@pytest.mark.parametrize("index_head", [32, 128], ids=["index-head-32", "index-head-128"])
def test_the_compiled_verifier_with_the_kernels_equals_its_parent_bit_for_bit(monkeypatch, index_head):
    # The parent is the same compiled verifier with the verify-width
    # hyper-connection read, the QSA selection and the pooled-key row switched
    # off at install: the stock chains traced into the compiled body. Every
    # compiled round's logits, hidden state, captures and cache leaves must
    # match it in their raw bits, and so must the tokens. A 128-wide index head
    # (Flash-Next's) brings the pooled-key row in; the tiny pack's 32-wide one
    # keeps it out.
    from mtplx.kernels import hc_verify_read, qsa_pooled_row, qsa_verify_select

    rounds = _record_compiled_rounds(monkeypatch)
    for module in (hc_verify_read, qsa_verify_select, qsa_pooled_row):
        monkeypatch.setenv(module.ENV, "0")
    parent_rt = _runtime(indexer_head_dim=index_head)
    assert not parent_rt._mtplx_hc_verify_read["installed"]
    assert not parent_rt._mtplx_qsa_verify_select["installed"]
    assert not parent_rt._mtplx_qsa_pooled_row["installed"]
    before = {m: m.engagement()["traces"] for m in (hc_verify_read, qsa_verify_select, qsa_pooled_row)}
    parent = _generate(parent_rt, "parity2", monkeypatch)
    for module, count in before.items():
        assert module.engagement()["traces"] == count
    parent_rounds = list(rounds)
    rounds.clear()

    for module in (hc_verify_read, qsa_verify_select, qsa_pooled_row):
        monkeypatch.delenv(module.ENV)
    rt = _runtime(indexer_head_dim=index_head)
    candidate = _generate(rt, "parity2", monkeypatch)
    assert qsa_verify_select.engagement()["traces"] > before[qsa_verify_select]
    if _hc_read_served_here():
        assert hc_verify_read.engagement()["traces"] > before[hc_verify_read]
    if index_head == 128:
        assert rt._mtplx_qsa_pooled_row["installed"], rt._mtplx_qsa_pooled_row
        assert qsa_pooled_row.engagement()["traces"] > before[qsa_pooled_row]
    else:
        assert not rt._mtplx_qsa_pooled_row["installed"]

    assert candidate.tokens == parent.tokens
    assert len(rounds) == len(parent_rounds) > 0
    for index, (want, got) in enumerate(zip(parent_rounds, rounds)):
        assert got.keys() == want.keys(), index
        for name, (dtype, shape, bits) in want.items():
            got_dtype, got_shape, got_bits = got[name]
            assert (got_dtype, got_shape) == (dtype, shape), (index, name)
            differing = int(np.count_nonzero(got_bits != bits))
            assert differing == 0, (index, name, differing)
