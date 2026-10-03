"""The compiled verifier and the eager verifier read the same evaluated weights.

``mx.compile`` treats an evaluated array it meets while tracing as a
constant, but it walks into an array that is still a lazy graph and replays
that graph on every call of the compiled function, inside fused JIT kernels.
A fused kernel does not reproduce the precompiled kernels bit for bit: MLX
0.32.2 prints its scalar constants with 7 significant digits and resolves the
unqualified ``metal::exp``/``metal::log``/``metal::pow`` in its sigmoid,
erfinv, power and logaddexp ops to the fast variants. A weight left lazy
would therefore reach the compiled verifier with different low bits than the
eager verifier, and be recomputed every round. A checkpoint load evaluates
the parameters (mlx_lm loads with ``lazy=False``) and the eager prefill
evaluates every array the forward reads, so the product's first verify trace
must find nothing lazy. The compiled-verify toy runtime once missed exactly
this; float32 GEMM on a Metal 4 tensor-unit GPU reads TF32 inputs and hid it
on M5, while M1 to M4 read all 23 mantissa bits.

Checked here:

* the first verify trace of a fresh runtime, on a request that is compiled
  from its first round, finds every array reachable from the model evaluated;
  every compiled round's logits, hidden state, captures and state equal the
  eager verifier's (parity2), and a separate fresh runtime run eagerly
  produces the same tokens: the
  synthetic four-layer ``qwen3_5`` model of ``tests/dense_mrope_synth.py``;
  the synthetic Ternary Bonsai 2 pack of ``tests/prism_hadamard_synth.py``
  through ``runtime.load``, the shipping loader; and Flash-Next's fixed-M4
  verify lane on the tiny pack of ``scripts/qwen4exp_mtp_tiny_smoke.py``;
* parity mode (eager authoritative, abort on the first bit mismatch) passes
  every compiled round of the synthetic ``qwen3_5`` model in float32,
  bfloat16, float16 and the 4-bit affine layout of the shipping packs, with
  and without the turbo profile's verify-shaped quantized matmul kernels,
  greedy and sampled;
* at the GDN key head size of every shipping pack (128), a float32 stream
  is sent to the eager verifier at admission (its key scale does not survive
  a fused kernel's 7-digit constants), while bfloat16, float16 and 4-bit over
  either stay compiled and exact.

Run natively for the M5 kernels and with ``MLX_METAL_GPU_ARCH=applegpu_g16s
MTPLX_FORCE_GPU_FAMILY_FALLBACK=1`` for the M1 to M4 kernels.
"""

from __future__ import annotations

import dataclasses
import functools
import importlib.util
import io
import types
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import pytest

from mtplx import demotions
from mtplx.attention_split import configure_split_full_attention
from mtplx.generation import generate_mtpk
from mtplx.graphbank import CompiledVerifyBank
from mtplx.mtp_patch import MTPContract
from mtplx.runtime import MTPLXRuntime
from mtplx.sampling import SamplerConfig
from tests import dense_mrope_synth as synth
from tests.test_dense_mrope_generation import _Tokenizer

GREEDY = SamplerConfig(temperature=0.0, top_p=1.0, top_k=20)
# The 27B pack's own settings, with a draft sampler at temperature 1.0.
NATIVE = SamplerConfig(temperature=0.6, top_p=0.95, top_k=20)
NATIVE_DRAFT = SamplerConfig(temperature=1.0, top_p=0.95, top_k=20)
TEXT = [(7 * i + 3) % 97 for i in range(16)]
SHIPPING_GDN_KEY_HEAD = 128

metal = pytest.mark.skipif(
    not mx.metal.is_available(), reason="the compiled verifier's parity is a GPU property"
)


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    # The attention route of the turbo profile, the profile that turns the
    # compiled verifier on (configure_split_full_attention reads it).
    monkeypatch.setenv("MTPLX_GQA_PACKED_SDPA", "1")
    for name in (
        "MTPLX_COMPILED_VERIFY",
        "MTPLX_COMPILED_VERIFY_FORCE",
        "MTPLX_COMPILED_VERIFY_GROWTH_RESERVE",
        "MTPLX_MTP_HISTORY_POLICY",
        "MTPLX_MTP_POSITION_MODE",
        "MTPLX_STATE_REBASE_EVERY",
    ):
        monkeypatch.delenv(name, raising=False)
    demotions.reset()
    yield
    demotions.reset()


_ATOMS = (str, bytes, bytearray, int, float, complex, bool, type, types.ModuleType)


def _children(value, path: str):
    if isinstance(value, dict):
        for key, item in list(value.items()):
            yield item, f"{path}.{key}"
    elif isinstance(value, (list, tuple, set, frozenset)):
        for index, item in enumerate(value):
            yield item, f"{path}[{index}]"
    if isinstance(value, types.MethodType):
        yield value.__self__, f"{path}.__self__"
        yield value.__func__, f"{path}.__func__"
    if isinstance(value, types.FunctionType):
        for index, cell in enumerate(value.__closure__ or ()):
            try:
                yield cell.cell_contents, f"{path}.<cell {index}>"
            except ValueError:  # an empty cell
                pass
        for index, item in enumerate(value.__defaults__ or ()):
            yield item, f"{path}.<default {index}>"
    if isinstance(value, functools.partial):
        yield value.func, f"{path}.func"
        yield value.args, f"{path}.args"
        yield value.keywords, f"{path}.keywords"
    try:
        attributes = vars(value)
    except TypeError:
        attributes = {}
    for key, item in list(attributes.items()):
        yield item, f"{path}.{key}"
    for klass in type(value).__mro__:
        slots = klass.__dict__.get("__slots__", ())
        for name in (slots,) if isinstance(slots, str) else slots:
            if name in ("__dict__", "__weakref__"):
                continue
            if name.startswith("__") and not name.endswith("__"):
                name = f"_{klass.__name__.lstrip('_')}{name}"
            try:
                yield getattr(value, name), f"{path}.{name}"
            except Exception:  # an unset slot
                continue


def _unevaluated_arrays(root) -> list[str]:
    """Paths of the arrays reachable from ``root`` that are still lazy graphs.

    Walks containers, module entries and attributes (private keys included,
    where ``parameters()`` stops), ``__slots__`` holders (the dense image
    position adapter is one), attribute holders from any library, and the
    cells, defaults and targets of closures, bound methods and partials kept
    as attributes. Classes, Python modules and function globals are not
    walked. MLX has no public "is evaluated" query; ``mx.export_to_dot``
    writes a primitive node (``shape=rectangle``) for every unevaluated array
    it reaches and only source nodes for an evaluated one.
    """

    found: list[str] = []
    seen: set[int] = set()
    stack = [(root, "model")]
    while stack:
        value, path = stack.pop()
        if isinstance(value, mx.array):
            dot = io.StringIO()
            mx.export_to_dot(dot, value)
            if "shape=rectangle" in dot.getvalue():
                found.append(path)
            continue
        if value is None or isinstance(value, _ATOMS) or id(value) in seen:
            continue
        seen.add(id(value))
        stack.extend(_children(value, path))
    return sorted(found)


def _first_trace_spy(monkeypatch) -> list[list[str]]:
    """The unevaluated arrays at every verify step the bank builds."""

    snapshots: list[list[str]] = []
    real = CompiledVerifyBank._make_verify_step

    def spying_make_verify_step(self, *args, **kwargs):
        snapshots.append(_unevaluated_arrays(self.runtime.model))
        return real(self, *args, **kwargs)

    monkeypatch.setattr(CompiledVerifyBank, "_make_verify_step", spying_make_verify_step)
    return snapshots


def _runtime(tmp_path, variant: str, **overrides) -> MTPLXRuntime:
    tmp_path.mkdir(parents=True, exist_ok=True)
    model = synth.model_with_draft_head(tmp_path, seed=8, tie=False, **overrides)
    if variant in ("bfloat16", "q4-bfloat16"):
        model.set_dtype(mx.bfloat16)
    elif variant in ("float16", "q4-float16"):
        model.set_dtype(mx.float16)
    if variant.startswith("q4"):
        nn.quantize(
            model,
            group_size=64,
            bits=4,
            class_predicate=lambda _path, module: isinstance(module, nn.Linear),
        )
    mx.eval(model.parameters())
    # What runtime.load installs on a gated-attention dense model.
    configure_split_full_attention(model)
    return MTPLXRuntime(
        model=model,
        tokenizer=_Tokenizer(),
        model_path=tmp_path,
        mtp_enabled=True,
        contract=MTPContract(),
    )


def _generate(rt, *, sampler=GREEDY, draft_sampler=None, max_tokens=16, prompt=TEXT):
    return generate_mtpk(
        rt,
        list(prompt),
        max_tokens=max_tokens,
        sampler=sampler,
        draft_sampler=draft_sampler,
        seed=11,
        speculative_depth=3,
        stop_token_ids=set(),
        verify_strategy="capture_commit",
        mtp_history_policy="committed",
    )


def _bank(out) -> dict:
    return out.stats.graphbank["compiled_verify"]


def _assert_every_round_exact(bank: dict) -> None:
    """parity2 compared every compiled round with the eager verifier: all equal."""

    assert bank["compiled_calls"] >= 1 and bank["fallback_calls"] == 0, bank
    assert bank["parity2_calls"] == bank["compiled_calls"], bank
    assert bank["parity2_divergent_calls"] == 0, bank.get("parity2_first_divergence")


def test_the_lazy_array_detector_sees_every_holder_kind():
    # The check below is only as good as its detector: a lazy array under a
    # private key, in a list, in a __slots__ holder, on a plain object from
    # any library, in a closure cell, a partial or a bound method must be
    # reported, and nothing once evaluated.
    class Slotted:
        __slots__ = ("_table", "inner")

        def __init__(self, table, inner):
            self._table = table
            self.inner = inner

    class Holder:  # not an mtplx type
        def __init__(self, value):
            self.value = value

    Holder.__module__ = "somelib.holders"

    class Owner(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = mx.ones((2, 2))
            self._table = mx.arange(4) * 0.5
            self.parts = [mx.zeros((3,)) + 1.0]
            captured = mx.ones((2,)) * 3.0
            self.adapter = Slotted(mx.ones((2,)) - 1.0, Holder(mx.ones((2,)) + 2.0))
            self.fn = lambda x: x * captured
            self.part = functools.partial(max, mx.ones((2,)) * 5.0)
            self.method = Holder(mx.ones((2,)) * 7.0).__init__

    owner = Owner()
    mx.eval(owner.weight)
    assert _unevaluated_arrays(owner) == sorted(
        [
            "model._table",
            "model.adapter._table",
            "model.adapter.inner.value",
            "model.fn.<cell 0>",
            "model.method.__self__.value",
            "model.part.args[0]",
            "model.parts[0]",
        ]
    )
    mx.eval(
        owner._table,
        owner.parts,
        owner.adapter._table,
        owner.adapter.inner.value,
        owner.fn.__closure__[0].cell_contents,
        owner.part.args,
        owner.method.__self__.value,
    )
    assert _unevaluated_arrays(owner) == []


@metal
def test_a_fresh_runtime_traces_its_first_verify_step_with_every_array_evaluated(
    tmp_path, monkeypatch
):
    snapshots = _first_trace_spy(monkeypatch)
    # parity2: the compiled verifier runs first and stays authoritative, and
    # every round's logits, hidden state, captures and state are compared
    # with the eager verifier's.
    monkeypatch.setenv("MTPLX_COMPILED_VERIFY", "parity2")
    compiled = _generate(_runtime(tmp_path / "compiled", "q4-bfloat16"), sampler=NATIVE,
                         draft_sampler=NATIVE_DRAFT)
    monkeypatch.setenv("MTPLX_COMPILED_VERIFY", "0")
    eager = _generate(_runtime(tmp_path / "eager", "q4-bfloat16"), sampler=NATIVE,
                      draft_sampler=NATIVE_DRAFT)

    _assert_every_round_exact(_bank(compiled))
    assert snapshots, "no compiled verify step was built"
    assert all(paths == [] for paths in snapshots), snapshots
    assert compiled.tokens == eager.tokens


@metal
def test_the_shipping_loader_hands_the_first_trace_evaluated_arrays(tmp_path, monkeypatch):
    """Ternary Bonsai 2's layout (GDN key head 128, float16) through runtime.load."""
    from mtplx import runtime
    from tests import prism_hadamard_synth

    pack = prism_hadamard_synth.build_synthetic_pack(tmp_path / "pack")
    prism_hadamard_synth.write_synthetic_mtp_sidecar(pack)
    prompt = [(5 * i + 1) % 300 for i in range(24)]
    snapshots = _first_trace_spy(monkeypatch)
    monkeypatch.setenv("MTPLX_COMPILED_VERIFY", "parity2")
    compiled = _generate(runtime.load(pack.path, mtp=True), sampler=NATIVE,
                         draft_sampler=NATIVE_DRAFT, prompt=prompt)
    monkeypatch.setenv("MTPLX_COMPILED_VERIFY", "0")
    eager = _generate(runtime.load(pack.path, mtp=True), sampler=NATIVE,
                      draft_sampler=NATIVE_DRAFT, prompt=prompt)

    bank = _bank(compiled)
    assert not bank.get("permanent_eager"), bank
    _assert_every_round_exact(bank)
    assert snapshots and all(paths == [] for paths in snapshots), snapshots
    assert compiled.tokens == eager.tokens


def _flash_next_tiny_runtime():
    """The tiny pack of the Flash-Next compiled-route tests, built fresh."""
    import mlx.utils

    from mtplx.models.qwen4_exp import Model, ModelArgs, Qwen4ExpMTP
    from mtplx.mtp_patch import validate_mtp_support
    from mtplx.qwen4_fixed_verify import install_qwen4_fixed_verify_route

    path = Path(__file__).resolve().parents[1] / "scripts" / "qwen4exp_mtp_tiny_smoke.py"
    spec = importlib.util.spec_from_file_location("qwen4exp_mtp_tiny_smoke", path)
    smoke = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(smoke)
    mx.random.seed(0)
    args = dataclasses.replace(
        smoke._tiny_text_args(),
        head_dim=32,
        indexer_head_dim=32,
        indexer_compress_ratio=4,
        ple_layer_ids=[1],
        rope_parameters={
            "mrope_interleaved": True,
            "mrope_section": [2, 1, 1],
            "partial_rotary_factor": 0.25,
            "rope_theta": 10000000,
            "rope_type": "default",
        },
    )
    model = Model(ModelArgs(model_type="qwen4_exp", text_config=dataclasses.asdict(args)))
    model.language_model.mtp = Qwen4ExpMTP(model.language_model.args)
    model.update(
        mlx.utils.tree_map(
            lambda p: p.astype(mx.bfloat16) if p.dtype == mx.float32 else p,
            model.parameters(),
        )
    )
    model.eval()
    mx.eval(model.parameters())
    assert validate_mtp_support(model)
    rt = smoke._tiny_runtime(model)
    install_qwen4_fixed_verify_route(rt)
    return rt


@metal
def test_flash_next_fixed_m4_lane_traces_its_first_verify_step_with_every_array_evaluated(
    monkeypatch,
):
    import os

    import mlx_lm.models.cache as cache_module

    import mtplx.graphbank as graphbank
    import mtplx.models.qwen4_exp as qwen4_exp

    for name in tuple(os.environ):
        if name.startswith("MTPLX_QWEN4_") or name == "MTPLX_FAMILY_CAPTURE_COMMIT":
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MTPLX_CONTEXT_COPY", "0")
    # A runtime load earlier in the session swaps mlx-lm's ArraysCache for the
    # leak-free class; the bank looks the class up per call (see the
    # compiled-route tests of this lane).
    monkeypatch.setattr(qwen4_exp, "ArraysCache", cache_module.ArraysCache)
    monkeypatch.setattr(graphbank, "_compiled_verify_bits_gate_ok", lambda _rt: True)
    prompt = [3, 5, 7, 9, 11, 13] + list(range(20, 54))

    def run(mode):
        monkeypatch.setenv("MTPLX_COMPILED_VERIFY", mode)
        return generate_mtpk(
            _flash_next_tiny_runtime(),
            list(prompt),
            max_tokens=40,
            sampler=NATIVE,
            draft_sampler=NATIVE,
            speculative_depth=3,
            seed=1234,
            mtp_cache_policy="persistent",
            mtp_history_policy="committed",
            verify_strategy="batched",
            stop_token_ids=set(),
        )

    snapshots = _first_trace_spy(monkeypatch)
    # parity2 is the installed lane's instrument: compiled authoritative, every
    # round's logits, hidden state, recurrent state and captures compared.
    compiled = run("parity2")
    eager = run("0")
    bank = _bank(compiled)
    (key,) = bank["compiled_keys"]  # the fixed-M4 lane's one text trace
    assert key.startswith("m4:"), key
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
    assert snapshots and all(paths == [] for paths in snapshots), snapshots
    assert compiled.tokens == eager.tokens


@metal
@pytest.mark.parametrize(
    "variant",
    [
        "float32",
        "bfloat16",
        "float16",
        "q4-bfloat16",
        "q4-float16",
        "q4-bfloat16+turbo-kernels",
        "q4-float16+turbo-kernels",
    ],
)
@pytest.mark.parametrize(
    ("sampler", "draft_sampler"),
    [(GREEDY, None), (NATIVE, NATIVE_DRAFT)],
    ids=["greedy", "native"],
)
def test_parity_mode_passes_every_round_on_the_product_architecture(
    tmp_path, monkeypatch, variant, sampler, draft_sampler
):
    from mtplx import nax_verify

    dtype_variant, _, kernels = variant.partition("+")
    routed: list[int] = []
    if kernels:
        # The turbo profile's verify-shaped quantized matmul route for sampled
        # rounds: here the plain-SIMD K-split kernel for the 4-row windows
        # (the synthetic K is too small for the M5-only 16-row NAX tile).
        monkeypatch.setenv("MTPLX_NAX_VERIFY", "1")
        monkeypatch.setenv("MTPLX_NAX_M4_IMPL", "vk_k")
        real_m4 = nax_verify.nax_qmm_m4

        def counting_m4(*args, **kwargs):
            routed.append(1)
            return real_m4(*args, **kwargs)

        monkeypatch.setattr(nax_verify, "nax_qmm_m4", counting_m4)
        assert nax_verify.install_nax_qlinear_patch()["installed"]
    try:
        rt = _runtime(tmp_path, dtype_variant)
        monkeypatch.setenv("MTPLX_COMPILED_VERIFY", "1")
        on = _generate(rt, sampler=sampler, draft_sampler=draft_sampler)
        monkeypatch.setenv("MTPLX_COMPILED_VERIFY", "parity")
        # A mismatch raises CompiledVerifyParityError out of the stream.
        checked = _generate(rt, sampler=sampler, draft_sampler=draft_sampler)
    finally:
        if kernels:
            # A process-global class patch: never leak it into other tests.
            nax_verify.uninstall_nax_qlinear_patch()

    bank = _bank(checked)
    assert bank["compiled_calls"] >= 1 and bank["fallback_calls"] == 0
    assert bank["parity_checks"] == bank["compiled_calls"]
    assert bank["parity_failures"] == 0
    assert checked.tokens == on.tokens
    if kernels and sampler is NATIVE:
        assert routed, "the K-split verify kernel never served a sampled round"


@metal
@pytest.mark.parametrize("variant", ["float32", "bfloat16", "float16", "q4-bfloat16", "q4-float16"])
def test_the_shipping_gdn_key_head_admits_every_stream_but_float32(tmp_path, monkeypatch, variant):
    """GDN key head 128: float32 verifies eagerly, the served dtypes stay compiled and exact."""

    geometry = {
        "linear_key_head_dim": SHIPPING_GDN_KEY_HEAD,
        "linear_value_head_dim": SHIPPING_GDN_KEY_HEAD,
    }
    rt = _runtime(tmp_path, variant, **geometry)
    monkeypatch.setenv("MTPLX_COMPILED_VERIFY", "1")
    on = _generate(rt, sampler=NATIVE, draft_sampler=NATIVE_DRAFT)
    bank = _bank(on)
    if variant == "float32":
        assert bank["permanent_eager"] is True
        assert bank["permanent_eager_reason"] == "float32_gdn_key_scale:head_k_dim=128"
        assert bank["compiled_calls"] == 0
        return
    assert bank["permanent_eager"] is False and bank["compiled_calls"] >= 1
    monkeypatch.setenv("MTPLX_COMPILED_VERIFY", "parity")
    checked = _generate(rt, sampler=NATIVE, draft_sampler=NATIVE_DRAFT)
    bank = _bank(checked)
    assert bank["parity_checks"] == bank["compiled_calls"] >= 1
    assert bank["parity_failures"] == 0
    assert checked.tokens == on.tokens
