"""The prefill admission's bill for the one-copy store (mtplx/one_copy.py).

The one-copy store changed three things the admission prices:

- a prompt is published as a lease on the live cache plus one recurrent
  anchor, so decode's first write copies nothing: the bill charges the
  anchor, not a second conversation (Flash-Next at 100K tokens: 116 MB where
  it charged 3.2 GB), and the admission no longer drops that publication to
  make room, because it is what restores a cancelled or edited answer;
- the prefill writes the attention buffers at the verifier bank's rows (the
  answer's reserve rounded up to the capacity bucket), so those rows are
  charged with the prompt;
- a lease is its conversation's only copy: its own session takes it in
  place, and a request of another session is served a copy, which the bill
  charges.
"""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import pytest
from mlx_lm.models.cache import ArraysCache

import mtplx.server.openai as srv
from mtplx.graphbank import FixedM4CapacityPlan
from mtplx.models.qwen4_exp import QSACache
from mtplx.one_copy import anchor_nbytes, prefill_slack_rows, prompt_lease_fields
from test_memguard_admission import (  # noqa: F401  (_served_profile: autouse)
    FN_ROW,
    FN_WEIGHTS,
    GIB,
    _flash_next_state,
    _install,
    _Machine,
    _manager,
    _put,
    _served_profile,
)

# One Flash-Next recurrent anchor: 36 GDN layers of 48 x 128 x 128 fp32
# state, their conv tails and the PLE tail (PRO-REVIEW-2-code.md 5.2).
ANCHOR = 115_642_384
_ANSWER_FLOOR = srv._ANSWER_ROOM_FLOOR_TOKENS


def _geometry(**overrides):
    values = dict(
        live_bytes_per_token=FN_ROW,
        paged_bytes_per_token=FN_ROW,
        context_transient_bytes_per_token=0,
        flat_transient_bytes=3 * GIB,
        weights_bytes=FN_WEIGHTS,
    )
    values.update(overrides)
    return srv._AdmissionGeometry(**values)


def _growth(**overrides):
    values = dict(
        prompt_tokens=100_000,
        reused_tokens=0,
        restore_copies_prefix=True,
        layout="contiguous_dense_decode",
        source_layout=None,
        output_tokens=0,
        publish=True,
        scratch_bytes=3 * GIB,
    )
    values.update(overrides)
    return srv._admission_growth(_geometry(), **values)


@pytest.fixture
def one_copy_on(monkeypatch):
    monkeypatch.delenv("MTPLX_ONE_COPY", raising=False)
    monkeypatch.setenv("MTPLX_QSA_GATHER", "1")
    monkeypatch.setenv("MTPLX_QSA_GATHER_MIN_CONTEXT", "16384")


def _one_copy_runtime(runtime):
    runtime.qwen4_fixed_m4_compiled_verify = True
    return runtime


class TestTheBill:
    def test_a_publication_is_charged_its_anchor_not_a_copy(self):
        copying = _growth()
        one_copy = _growth(publish_bytes=ANCHOR)
        rows = 100_000 * FN_ROW
        assert copying["publish_copy_bytes"] == rows
        assert one_copy["publish_copy_bytes"] == ANCHOR
        assert one_copy["decode_start_bytes"] == rows + ANCHOR
        assert copying["decode_start_bytes"] - one_copy["decode_start_bytes"] == rows - ANCHOR

    def test_no_publication_is_charged_nothing(self):
        assert _growth(publish=False, publish_bytes=ANCHOR)["publish_copy_bytes"] == 0

    def test_the_prefill_slack_is_charged_with_the_prompt(self):
        growth = _growth(slack_rows=9_000, publish_bytes=ANCHOR)
        assert growth["slack_bytes"] == 9_000 * FN_ROW
        assert growth["live_prefill_bytes"] == 109_000 * FN_ROW
        assert growth["prefill_end_bytes"] == 109_000 * FN_ROW + 3 * GIB

    def test_a_full_hit_prefills_nothing_and_allocates_no_slack(self):
        growth = _growth(
            reused_tokens=100_000,
            restore_copies_prefix=False,
            source_layout="contiguous_dense_decode",
            slack_rows=9_000,
            publish_bytes=ANCHOR,
        )
        assert growth["slack_bytes"] == 0
        assert growth["live_prefill_bytes"] == 0


class TestLeaseOwnership:
    def _lease(self, session_id):
        return SimpleNamespace(
            cache_ref=object(),
            live_ref_only=True,
            session_id=session_id,
            lazy_kv=False,
            snapshot_settled_at=None,
        )

    def test_its_own_session_takes_it_in_place(self):
        entry = self._lease("pi-main")
        assert srv._admission_restore_copies_prefix(entry, "reference", "pi-main") is False

    def test_another_session_is_charged_the_copy_it_is_served(self):
        entry = self._lease("pi-main")
        assert srv._admission_restore_copies_prefix(entry, "reference", "pi-subagent") is True

    def test_without_session_ids_the_lease_is_taken_as_before(self):
        assert srv._admission_restore_copies_prefix(self._lease("pi-main"), "reference") is False
        assert srv._admission_restore_copies_prefix(self._lease(None), "reference", "x") is False

    def test_a_clone_restore_always_copies(self):
        entry = self._lease("pi-main")
        assert srv._admission_restore_copies_prefix(entry, "clone", "pi-main") is True


class TestTheRuntimeAnswers:
    def test_the_first_prompt_lease_measures_the_anchor(self):
        runtime = SimpleNamespace()
        assert anchor_nbytes(runtime) is None
        recurrent = ArraysCache(size=2)
        recurrent[0] = mx.zeros((1, 3, 40), dtype=mx.bfloat16)
        recurrent[1] = mx.zeros((1, 4, 16, 16), dtype=mx.float32)
        cache = [recurrent, QSACache(4)]
        fields = prompt_lease_fields(
            cache,
            committed_mtp_cache=None,
            hidden=None,
            prompt_len=10,
            boundaries=[],
            runtime=runtime,
        )
        assert anchor_nbytes(runtime) == 1 * 3 * 40 * 2 + 1 * 4 * 16 * 16 * 4
        [(position, snapshot, _hidden)] = fields["gdn_boundaries"]
        assert position == 10
        assert snapshot.states[1] is None

    def test_the_slack_is_the_bank_rows_past_the_prompt(self, one_copy_on):
        runtime = _one_copy_runtime(_flash_next_state(_manager()).runtime)
        slack = prefill_slack_rows(runtime, 100_000, 32_768)
        plan = FixedM4CapacityPlan.for_request(32_768, runtime=runtime)
        assert slack == plan.rows(100_000, 4, 256) - 100_000
        assert 0 < slack <= plan.bucket + plan.reserve_tokens

    def test_no_slack_below_the_gather_lane_or_with_the_store_off(self, one_copy_on, monkeypatch):
        runtime = _one_copy_runtime(_flash_next_state(_manager()).runtime)
        assert prefill_slack_rows(runtime, 8_000, 32_768) == 0
        monkeypatch.setenv("MTPLX_ONE_COPY", "0")
        assert prefill_slack_rows(runtime, 100_000, 32_768) == 0


class TestTheAdmission:
    """A warm Pi turn: 3,000 new tokens on the session's 120,000-token
    conversation, the engine near its line (96 GiB limit, 93.1 GiB line).

    The copying store banked the conversation as a lazy snapshot beside the
    live cache, so the restore copies it on the first write and the prompt's
    publication copies it again at decode: 7.4 GiB of growth. The admission
    drops the publication to fit, so a cancelled answer can no longer
    restore at this prompt. The one-copy store leases the live cache, so the
    same turn grows by its new rows, the bucket slack and one anchor (2.3 GiB,
    most of it the prefill's scratch), and is admitted without touching
    anything."""

    CONVERSATION = 120_000

    def _admit(self, monkeypatch, *, one_copy: bool, base_gib: float):
        manager = _manager()
        conversation = list(range(self.CONVERSATION))
        entry = _put(
            manager.bank, conversation, session_id="pi", row_bytes=FN_ROW, live_cache=True
        )
        if one_copy:
            # What a one-copy put records: a lease on the live cache and no
            # snapshot (session_bank.put, "one_copy_lease").
            entry.live_ref_only = True
            entry.lazy_kv = False
            entry.lease_pinned_nbytes = entry.nbytes
            entry.nbytes = 0
        else:
            monkeypatch.setenv("MTPLX_ONE_COPY", "0")
        state = _flash_next_state(manager)
        _one_copy_runtime(state.runtime)
        state.runtime.one_copy_anchor_nbytes = ANCHOR
        session = manager.get_or_create("pi")
        assert session.try_begin_generation()
        machine = _Machine(manager.bank, base_gib=base_gib, cache_gib=0.0, host_gib=0.0)
        _install(monkeypatch, machine)
        pricing: dict = {}
        try:
            receipt = srv._prefill_admission_shed(
                state,
                prompt_ids=conversation + list(range(5_000_000, 5_003_000)),
                session_bank=manager.bank,
                session_id="pi",
                commit_prompt_prefix=True,
                max_new_tokens=32_768,
                pricing=pricing,
            )
        finally:
            session.end_generation()
        return receipt, pricing["growth"], manager

    def test_the_one_copy_turn_is_admitted_untouched(self, monkeypatch, one_copy_on):
        receipt, growth, manager = self._admit(monkeypatch, one_copy=True, base_gib=86.5)
        assert receipt is None
        assert growth["restore_copy_bytes"] == 0
        assert growth["publish_copy_bytes"] == ANCHOR
        assert 0 < growth["slack_bytes"] <= (8_192 + 1_024) * FN_ROW
        assert growth["growth_bytes"] < 3 * GIB
        assert manager.bank.has_session_entries("pi")

    def test_the_copying_store_drops_the_publication_for_the_same_turn(
        self, monkeypatch, one_copy_on
    ):
        receipt, growth, _manager_ = self._admit(monkeypatch, one_copy=False, base_gib=86.5)
        assert receipt is not None and receipt.get("refused") is not True
        # The first bill: the restore's copy, the new rows and the scratch,
        # then the publication's copy at decode.
        [first] = receipt["growth_by_chunk"].values()
        assert first > 7 * GIB
        assert receipt["prompt_publish_skipped"] is True
        assert growth["restore_copy_bytes"] == self.CONVERSATION * FN_ROW
        assert growth["publish_copy_bytes"] == 0


class TestTheAnswersRoom:
    """Before its prefill, a request's longest answer (its max_tokens inside
    the served window) is priced against what the engine line leaves after
    the prefill (96 GiB limit, 93.1 GiB line) plus what the bank would give
    back. Other conversations are counted, not released: the growth releases
    them only if the answer really grows into their room.

    A 100K prompt asking for 32,768 tokens: the prefill already allocated
    9,000 spare rows, so the answer may add 23,771 rows (0.72 GiB) plus one
    layer's growth beside the old one (0.29 GiB)."""

    def _state(self, *, idle_tokens=0, window=262_144):
        manager = _manager()
        if idle_tokens:
            # Another conversation: it shares nothing with the prompt, so the
            # prompt cannot restore from it.
            _put(
                manager.bank,
                range(1_000_000, 1_000_000 + idle_tokens),
                session_id="idle",
                row_bytes=FN_ROW,
            )
        state = _flash_next_state(manager)
        _one_copy_runtime(state.runtime)
        args = state.runtime.model.args
        args.num_key_value_heads = 2
        args.head_dim = 256
        args.indexer_head_dim = 128
        state.context_window = window
        return state, manager

    def _room(self, monkeypatch, state, manager, *, base_gib, prompt=100_000,
              max_new_tokens=32_768, slack_rows=9_000):
        machine = _Machine(manager.bank, base_gib=base_gib, cache_gib=0.0, host_gib=0.0)
        _install(monkeypatch, machine)
        growth = {"slack_bytes": slack_rows * FN_ROW, "decode_start_bytes": 2 * GIB}
        return srv._answer_room(
            state,
            prompt_ids=list(range(prompt)),
            max_new_tokens=max_new_tokens,
            mtp_depth=3,
            session_bank=manager.bank,
            session_id="pi",
            growth=growth,
        )

    def test_an_answer_that_fits_passes_untouched(self, monkeypatch, one_copy_on):
        state, manager = self._state()
        assert self._room(monkeypatch, state, manager, base_gib=80.0) is None

    def test_the_bank_counts_as_room_and_is_not_released(self, monkeypatch, one_copy_on):
        # 88.5 GiB plus an idle 114K conversation (3.45 GiB) plus the prefill
        # leaves no room under the line, but releasing that conversation
        # would, if the answer ever grew that long. Nothing is released now.
        state, manager = self._state(idle_tokens=114_191)
        assert self._room(monkeypatch, state, manager, base_gib=88.5) is None
        assert manager.bank.has_session_entries("idle")
        # Without it the same Mac caps the answer.
        state, manager = self._state()
        capped = self._room(monkeypatch, state, manager, base_gib=88.5 + 3.45)
        assert capped is not None and capped.get("answer_token_cap")

    def test_a_squeezed_mac_caps_the_answer_it_cannot_hold(self, monkeypatch, one_copy_on):
        state, manager = self._state()
        receipt = self._room(monkeypatch, state, manager, base_gib=90.5)
        assert receipt is not None and not receipt.get("refused")
        cap = receipt["answer_token_cap"]
        assert 9_000 < cap < 32_768
        # What the cap adds past the spare rows fits the room with the growth.
        grown = (cap + 3 - receipt["slack_rows"]) * FN_ROW
        assert grown <= receipt["room_bytes"]

    def test_no_room_for_a_short_answer_is_refused_before_prefill(
        self, monkeypatch, one_copy_on
    ):
        # A full hit prefills nothing, so there are no spare rows either.
        state, manager = self._state()
        receipt = self._room(monkeypatch, state, manager, base_gib=91.0, slack_rows=0)
        assert receipt is not None and receipt["refused"] is True
        error = srv._answer_room_refusal(receipt)
        assert error.status_code == 507
        assert error.detail["code"] == "insufficient_memory"
        assert "refused before prefill" in error.detail["message"]

    def test_the_window_bounds_the_answer(self, monkeypatch, one_copy_on):
        # 250K of history in a 262K window: the answer can add 12,144 rows,
        # whatever max_tokens asks. The same Mac without a known window
        # would have to cap a 131,072-token answer.
        state, manager = self._state()
        assert self._room(
            monkeypatch, state, manager, base_gib=89.6, prompt=250_000, max_new_tokens=131_072
        ) is None
        state, manager = self._state(window=0)
        capped = self._room(
            monkeypatch, state, manager, base_gib=89.6, prompt=250_000, max_new_tokens=131_072
        )
        assert capped is not None and capped["answer_rows"] == 131_075

    def test_other_stores_are_not_checked(self, monkeypatch, one_copy_on):
        monkeypatch.setenv("MTPLX_ONE_COPY", "0")
        state, manager = self._state()
        assert self._room(monkeypatch, state, manager, base_gib=91.0, slack_rows=0) is None
