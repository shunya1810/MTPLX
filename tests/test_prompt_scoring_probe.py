"""The queued HTTP gate fails closed on invalid scores and excessive drift."""

from copy import deepcopy

import pytest

from scripts.check_prompt_scoring import check_repeat, checked_scores, compare


def _response(width, rows=3):
    return {
        "mtplx_stats": {"prefill_chunk_tokens": width},
        "usage": {"prompt_tokens": rows},
        "choices": [{"logprobs": {
            "token_ids": [7] * rows,
            "token_logprobs": [None] + [-0.25] * (rows - 1),
            "top_logprobs": [{}] + [{"a": -0.25, "b": -1.5} for _ in range(rows - 1)],
        }}],
    }


def _receipt(width):
    return {"width": width, "head": "candidate", "pack": "gemma", "cases": {
        label: {"first": _response(width, rows), "repeated": _response(width, rows)}
        for label, rows in (("short", 3), ("long", 2050))
    }}


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_probe_rejects_nonfinite_top_scores(value):
    response = _response(256)
    response["choices"][0]["logprobs"]["top_logprobs"][1]["a"] = value
    with pytest.raises(AssertionError, match="finite top scores"):
        checked_scores(response, width=256)


def test_probe_rejects_changed_repeat():
    first = _response(256)
    repeated = deepcopy(first)
    repeated["choices"][0]["logprobs"]["token_logprobs"][1] += 1e-7
    with pytest.raises(AssertionError, match="repeated scoring changed"):
        check_repeat(first, repeated, width=256)


def test_probe_enforces_cross_width_bound_and_reports_exactness():
    left, right = _receipt(256), _receipt(512)
    assert compare(left, right)["cases"]["long"]["exactly_equal_across_widths"]
    for arm in ("first", "repeated"):
        right["cases"]["long"][arm]["choices"][0]["logprobs"]["token_logprobs"][1] += 0.006
    with pytest.raises(AssertionError, match="absolute logprob delta"):
        compare(left, right)


def test_probe_requires_the_admitted_width_and_distinct_widths():
    with pytest.raises(AssertionError, match="admitted width"):
        checked_scores(_response(128), width=256)
    with pytest.raises(AssertionError, match="two different admitted widths"):
        compare(_receipt(256), _receipt(256))
