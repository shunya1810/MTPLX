"""Generation-final bank values carry the prompt's GDN boundary records."""

from types import SimpleNamespace

from mtplx.generation import GenerationFinalState
from mtplx.server import openai as server


def _final_state(boundaries):
    return GenerationFinalState(
        final_trunk_cache=["cache"],
        final_logits="logits",
        final_hidden=None,
        final_committed_mtp_cache=None,
        generated_token_ids=(7, 8),
        safe_to_commit=True,
        finish_reason="length",
        prompt_gdn_boundaries=boundaries,
    )


def _values(final_state, prompt_ids, final_token_ids):
    state = SimpleNamespace()
    original = server._bank_backend_id
    server._bank_backend_id = lambda _state: "qwen"
    try:
        return server._generation_final_bank_values(
            state, final_state, prompt_ids=prompt_ids, final_token_ids=final_token_ids
        )
    finally:
        server._bank_backend_id = original


def test_prompt_boundaries_ride_the_generation_final_entry():
    records = [(2, "snap2", None), (4, "snap4", None)]
    values = _values(_final_state(records), [1, 2, 3, 4], [1, 2, 3, 4, 7, 8])
    assert values["gdn_boundaries"] == records
    assert values["token_ids"] == [1, 2, 3, 4, 7, 8]


def test_only_boundaries_inside_the_shared_prefix_ride_along():
    # Upstream 2.12.3 rule: a record at p is the state after the first p
    # tokens, valid for any history that starts with them.
    records = [(2, "snap2", None), (4, "snap4", None)]
    values = _values(_final_state(records), [1, 2, 3, 4], [1, 2, 9, 4, 7, 8])
    assert values["gdn_boundaries"] == [(2, "snap2", None)]


def test_no_boundaries_when_the_banked_tokens_part_before_the_first_record():
    records = [(2, "snap2", None)]
    values = _values(_final_state(records), [1, 2, 3, 4], [1, 9, 3, 4, 7, 8])
    assert "gdn_boundaries" not in values


def test_no_boundaries_key_without_records():
    values = _values(_final_state(None), [1, 2, 3], [1, 2, 3, 7])
    assert "gdn_boundaries" not in values
