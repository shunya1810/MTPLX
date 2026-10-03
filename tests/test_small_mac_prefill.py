"""2.12.2: the prefill admission on 8, 16 and 32 GB Macs.

2.12.1 refused ordinary prompts on the small seats that 2.12.0 served
(emulated 2026-10-02 on an M5 Max at each seat's limit): Bonsai 2 27B at the
16 GB limit refused 2,595 and 6,835-token prompts, the 27B at the 32 GB
limit refused 6,835 tokens, and the 4B at the 8 GB limit needed more than
3.2 GiB free for an 18-token prompt. These tests pin the changes against
what was measured that night through ``mtplx serve``:

* a dense family's prefill is billed what was measured on it, not the 27B's
  whole-request reserve per 2,048 rows (``_dense_prefill_bill``);
* the admission may run a dense family's prefill at 1,024 or 512 rows
  before it refuses, and a narrower chunk is chosen only where it costs
  less (``_admission_narrow_widths``);
* a 512-row chunk pays for the context it reads (``_prefill_forward_bill``).

The host-memory allowance (4 GiB on these seats) is in
test_host_memory_allowance.py.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import mtplx.server.openai as srv
import mtplx.system_memory as sm
from tests.test_memguard_admission import (
    GIB,
    _flash_next_runtime,
    _install,
    _Machine,
    _manager,
    _q27_runtime,
    _q27_text_args,
    _state,
)
from tests.test_memguard_backend_contract import Q4B_TEXT

BONSAI_KV = 65_536
BONSAI_HISTORY = 4_096
BONSAI_WEIGHTS = int(8.23 * GIB)
Q4B_KV = 8 * 2 * 4 * 256 * 2


@pytest.fixture(autouse=True)
def _served_profile(monkeypatch):
    # The served profiles prefill in 2,048-row chunks with the auto layout.
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", "1")
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL_LAYOUT", "auto")
    monkeypatch.setenv("MTPLX_SUSTAINED_DENSE_DECODE_MAX_CONTEXT", "131072")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE", "auto")
    monkeypatch.delenv("MTPLX_PAGED_KV_QUANT", raising=False)
    monkeypatch.delenv("MTPLX_VLLM_METAL_PAGED_KV_QUANT", raising=False)
    monkeypatch.delenv("MTPLX_HOST_MEMORY_ALLOWANCE_BYTES", raising=False)
    monkeypatch.delenv("MTPLX_SYSTEM_MEMORY_ABORT_FLOOR_BYTES", raising=False)
    monkeypatch.delenv("MTPLX_SYSTEM_MEMORY_SHED_FLOOR_BYTES", raising=False)
    monkeypatch.delenv("MTPLX_SYSTEM_MEMORY_REHEARSAL_AVAILABLE_BYTES", raising=False)
    monkeypatch.setattr(srv, "_record_guard_event", lambda state, payload: None)


def _q4b_runtime():
    from mlx_lm.models.qwen3_5 import TextModelArgs

    return SimpleNamespace(
        model=SimpleNamespace(
            language_model=SimpleNamespace(args=TextModelArgs.from_dict(Q4B_TEXT))
        ),
        mtp_enabled=True,
        model_path=Path("models/qwen3.5-4b"),
    )


def _plan(*, kv: int, history: int, weights: int):
    return SimpleNamespace(
        available=True,
        kv_bytes_per_token=kv,
        kv_bytes_per_token_effective=kv,
        aux_bytes_per_token=0,
        mtp_history_bytes_per_token=history,
        prefill_transient_bytes_per_token=0,
        runtime_transients_bytes=3 * GIB,
        model_weights_bytes=weights,
    )


def _mac(monkeypatch, *, total_gib: float, available_gib: float, wired_gib: float):
    """What macOS reports: its RAM, the free and file-backed supply, and
    what is wired. The floors follow from them (abort: the largest of 1 GiB,
    2.5% of RAM and a sixteenth of the wired memory; shed: twice that)."""

    monkeypatch.setattr(
        sm,
        "_reader",
        lambda: sm.SystemMemory(
            available_bytes=int(available_gib * GIB),
            total_bytes=int(total_gib * GIB),
            level_percent=int(100 * available_gib / total_gib),
            free_bytes=int(available_gib * GIB) // 2,
            file_backed_bytes=int(available_gib * GIB) // 2,
            wired_bytes=int(wired_gib * GIB),
            compressor_bytes=0,
            swap_used_bytes=0,
        ),
    )


def _bonsai_16gb(monkeypatch, *, available_gib: float, host_gib: float = 2.9):
    manager = _manager(max_bytes=GIB // 2, per_session_max_bytes=GIB)
    _install(
        monkeypatch,
        _Machine(manager.bank, base_gib=BONSAI_WEIGHTS / GIB, cache_gib=0.0, host_gib=host_gib),
    )
    _mac(monkeypatch, total_gib=16, available_gib=available_gib, wired_gib=9)
    plan = _plan(kv=BONSAI_KV, history=BONSAI_HISTORY, weights=BONSAI_WEIGHTS)
    return manager, _state(manager, plan=plan, runtime=_q27_runtime(), limit_gib=12, total_gib=16)


def _q4b_8gb(monkeypatch, *, available_gib: float):
    manager = _manager(max_bytes=2 * GIB, per_session_max_bytes=GIB)
    _install(monkeypatch, _Machine(manager.bank, base_gib=2.4, cache_gib=0.0, host_gib=0.5))
    _mac(monkeypatch, total_gib=8, available_gib=available_gib, wired_gib=3.5)
    plan = _plan(kv=Q4B_KV, history=BONSAI_HISTORY, weights=int(2.37 * GIB))
    return manager, _state(manager, plan=plan, runtime=_q4b_runtime(), limit_gib=8, total_gib=8)


def _admit(state, manager, prompt_tokens: int, session_id: str = "agent"):
    session = manager.get_or_create(session_id)
    assert session.try_begin_generation()
    pricing: dict = {}
    try:
        receipt = srv._prefill_admission_shed(
            state,
            prompt_ids=list(range(1_000_000, 1_000_000 + prompt_tokens)),
            session_bank=manager.bank,
            session_id=session_id,
            pricing=pricing,
        )
    finally:
        session.end_generation()
    return receipt, pricing.get("growth") or {}


# --------------------------------------------------------------------------
# Which widths a family may narrow to
# --------------------------------------------------------------------------


def test_a_dense_family_may_narrow_to_1024_and_512_rows():
    for runtime in (_q27_runtime(), _q4b_runtime()):
        ladder = srv._admission_prefill_widths(runtime, 8_000, None)
        assert ladder == [2048]
        assert srv._admission_narrow_widths(runtime, ladder) == [1024, 512]


def test_a_family_with_routed_experts_narrows_to_1024_only():
    # Its 512-row chunks were not measured.
    args = SimpleNamespace(
        hidden_size=2048,
        num_attention_heads=16,
        intermediate_size=6144,
        num_experts=256,
        num_experts_per_tok=8,
        moe_intermediate_size=512,
    )
    runtime = SimpleNamespace(model=SimpleNamespace(args=args))
    assert srv._admission_narrow_widths(runtime, [2048]) == [1024]


def test_flash_next_and_backends_with_their_own_widths_keep_their_ladders():
    assert srv._admission_narrow_widths(_flash_next_runtime(), [4096, 2048]) == []
    own = SimpleNamespace(prefill_forward_widths=lambda prompt, requested: [2048, 1024])
    assert srv._admission_narrow_widths(own, [2048, 1024]) == []
    # A prompt forwarded whole has no chunk to narrow.
    assert srv._admission_narrow_widths(_q27_runtime(), [None]) == []
    assert srv._admission_narrow_widths(None, [2048]) == []


# --------------------------------------------------------------------------
# What a chunk costs
# --------------------------------------------------------------------------


def _bill(width: int, prompt_tokens: int) -> dict:
    state = SimpleNamespace(runtime=_q27_runtime())
    geometry = srv._AdmissionGeometry(
        BONSAI_KV + BONSAI_HISTORY, BONSAI_KV + BONSAI_HISTORY, 0, 3 * GIB, 0
    )
    return srv._prefill_forward_bill(
        state, new_tokens=prompt_tokens, width=width, prompt_tokens=prompt_tokens, geometry=geometry
    )


def test_from_1024_rows_up_a_chunk_does_not_pay_for_the_context():
    """Bonsai 2 27B, measured: 1.70 to 1.72 GiB at 1,024 rows and 2.69 to
    2.72 at 2,048, from 2.4K to 8K tokens of context."""

    for width, low, high in ((1024, 1.72, 1.85), (2048, 2.72, 2.95)):
        short, long = _bill(width, 2_600), _bill(width, 30_000)
        assert short["scratch"] == long["scratch"]
        assert short["source"] == "geometry_measured"
        assert low * GIB < short["scratch"] < high * GIB


def test_a_512_row_chunk_pays_for_the_context_it_reads():
    """Bonsai 2 27B at 512 rows, measured: 1.21 GiB at 2.4K tokens, 1.48 at
    7K, 1.53 at 8K. The bill adds a live row per token of context, so it
    costs less than a 1,024-row chunk for a short prompt and more past
    about 8K tokens, where the admission never picks it."""

    short = _bill(512, 2_600)
    assert short["source"] == "geometry_measured+context"
    assert 1.21 * GIB < short["scratch"] < _bill(1024, 2_600)["scratch"]
    at_8k = _bill(512, 8_000)["scratch"]
    assert 1.53 * GIB < at_8k < 1.9 * GIB
    assert _bill(512, 30_000)["scratch"] > _bill(1024, 30_000)["scratch"]


# --------------------------------------------------------------------------
# The admission on the seats 2.12.1 refused
# --------------------------------------------------------------------------


@pytest.mark.parametrize("prompt_tokens", [2_595, 6_835])
def test_bonsai_on_a_16gb_mac_serves_what_2121_refused(monkeypatch, prompt_tokens):
    """2.12.1: 507 at both sizes (9.5 GiB held plus a 3.4 GiB chunk past
    the 12 GiB limit; 12.1 GiB projected, 1.1 GiB of it host memory past a
    1 GiB allowance). 2.12.0 served both."""

    manager, state = _bonsai_16gb(monkeypatch, available_gib=6)
    receipt, growth = _admit(state, manager, prompt_tokens)
    assert receipt is None or not receipt.get("refused")
    assert growth["scratch_source"].startswith("geometry_measured")
    if receipt is not None:
        assert receipt["host_overhang_charged_bytes"] == 0
        assert receipt["projected_bytes_after"] <= int(12 * GIB * 0.97)


def test_a_tight_16gb_mac_runs_a_short_prompt_at_512_rows(monkeypatch):
    """3 GiB free, the 1 GiB abort floor: a 2,600-token prompt fits only at
    512 rows (about 1.8 GiB of growth there, 2.2 at 1,024). 2.12.1 refused
    it; at 1,024 rows it would leave 0.8 GiB, under the floor."""

    manager, state = _bonsai_16gb(monkeypatch, available_gib=3)
    receipt, growth = _admit(state, manager, 2_600)
    assert receipt is not None
    assert not receipt.get("refused")
    assert receipt["prefill_chunk_tokens"] == 512
    assert growth["prefill_chunk_tokens"] == 512


def test_a_short_prompt_is_not_narrowed_for_nothing(monkeypatch):
    """A 300-token prompt is one forward at any width, so 2,048 and 1,024
    rows cost the same and 512 rows a little more (its context term). On a
    Mac short of its shed floor but above its abort floor the request runs,
    at the profile's width, not at a narrower one that would only be
    slower."""

    manager, state = _q4b_8gb(monkeypatch, available_gib=2.2)
    receipt, growth = _admit(state, manager, 300)
    assert receipt is None or not receipt.get("refused")
    assert growth["prefill_chunk_tokens"] == 2048


@pytest.mark.parametrize("prompt_tokens", [18, 2_500, 10_000])
def test_the_4b_on_an_8gb_mac_with_3gib_free(monkeypatch, prompt_tokens):
    """2.12.1 refused all three with 3 GiB free (an 18-token prompt priced
    at 2.3 GiB of growth); the matrix of 2026-10-03 served all three."""

    manager, state = _q4b_8gb(monkeypatch, available_gib=3)
    receipt, _growth = _admit(state, manager, prompt_tokens)
    assert receipt is None or not receipt.get("refused")


def test_what_does_not_fit_is_still_refused(monkeypatch):
    """1.2 GiB free on an 8 GB Mac: a 10,000-token prompt grows by more
    than the Mac can give without going under its 1 GiB abort floor."""

    manager, state = _q4b_8gb(monkeypatch, available_gib=1.2)
    receipt, _growth = _admit(state, manager, 10_000)
    assert receipt is not None and receipt["refused"] is True
