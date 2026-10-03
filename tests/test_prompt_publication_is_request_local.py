"""A request's decision to skip publishing its prompt snapshot stays its own.

The prefill admission can decide that this request must not bank a copy of
its prompt (the copy decode's first write makes is what crosses the memory
line). The server used to carry that decision by setting
MTPLX_SESSION_STORE_ON_PREFILL=0 in the process environment around the whole
generation, where every other reader of the operator's switch (a postcommit,
the next admission's pricing, any thread) saw it too, and where two holders
restoring their snapshots out of order could leave it behind. The decision
now travels as the request's own ``store_prefix_snapshot`` argument, and the
environment variable stays what it was: the operator's kill switch.
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient
from test_api_benchmark_contracts import _envelope_client, _fake_generation_output
from test_generation_sustained import TinyModel, _runtime

import mtplx.generation as generation
from mtplx.server import openai

ENV = "MTPLX_SESSION_STORE_ON_PREFILL"


class _RecordingBank:
    last_miss_reason = "new_session"

    def __init__(self):
        self.puts: list[dict] = []

    def restore(self, *_args, **_kwargs):
        return None

    def longest_prefix(self, *_args, **_kwargs):
        return None

    def put(self, **kwargs):
        self.puts.append(kwargs)
        return SimpleNamespace(
            prefix_len=len(kwargs["token_ids"]), nbytes=123, token_hash="h"
        )


def _cold_prefill(store_prefix_snapshot):
    bank = _RecordingBank()
    state = generation.restore_or_prefill_prompt_state(
        _runtime(TinyModel()),
        [0, 1, 2, 3, 4, 5],
        mtp_history_policy="cycle",
        session_bank=bank,
        store_prefix_snapshot=store_prefix_snapshot,
    )
    return bank, state.prefill_store_snapshot


class TestPromptStateReceipt:
    """A tiny synthetic model through the real prompt-state builder."""

    @pytest.fixture(autouse=True)
    def _small_suffix_counts(self, monkeypatch):
        monkeypatch.setenv("MTPLX_SESSION_STORE_ON_PREFILL_MIN_SUFFIX", "1")

    def test_default_publishes(self, monkeypatch):
        monkeypatch.delenv(ENV, raising=False)
        bank, receipt = _cold_prefill(None)
        assert len(bank.puts) == 1
        assert receipt["stored"] is True

    def test_a_request_skip_is_named_as_such(self, monkeypatch):
        monkeypatch.delenv(ENV, raising=False)
        bank, receipt = _cold_prefill(False)
        assert bank.puts == []
        assert receipt == {"stored": False, "skip_reason": "skipped_for_request"}

    def test_the_operator_kill_switch_is_still_the_environment(self, monkeypatch):
        monkeypatch.setenv(ENV, "0")
        bank, receipt = _cold_prefill(None)
        assert bank.puts == []
        assert receipt == {"stored": False, "skip_reason": "disabled"}


def _client(monkeypatch, decisions: list[bool], *, raise_on: set[int] = frozenset()):
    """A TestClient over the real _run_generation. ``decisions[i]`` is
    whether request i's admission skips its prompt publication; request i
    raises inside generation when i is in ``raise_on``. Every generation
    records its argument, the environment it saw, and what another thread
    reading the operator's switch saw at the same moment."""

    seen: list[dict] = []
    calls = {"admission": 0, "generation": 0}
    output = _fake_generation_output()

    def fake_admission(*_args, **_kwargs):
        index = calls["admission"]
        calls["admission"] += 1
        if decisions[index]:
            return {"action": "prefill_admission_shed", "prompt_publish_skipped": True}
        return None

    def fake_generate(*args, **kwargs):
        index = calls["generation"]
        calls["generation"] += 1
        other_thread: list[bool] = []
        reader = threading.Thread(
            target=lambda: other_thread.append(generation._store_on_prefill_env_enabled())
        )
        reader.start()
        reader.join()
        seen.append(
            {
                "argument": kwargs.get("store_prefix_snapshot", "absent"),
                "environment": os.environ.get(ENV),
                "other_thread_enabled": other_thread[0],
            }
        )
        if index in raise_on:
            raise RuntimeError("generation failed")
        return output(*args, **kwargs)

    client, _state = _envelope_client(monkeypatch, generator=fake_generate)
    monkeypatch.setattr(openai, "_prefill_admission_shed", fake_admission)
    return client, seen


def _post(client, *, mode: str | None = None):
    body = {
        "messages": [{"role": "user", "content": "Say OK."}],
        "max_tokens": 8,
    }
    if mode is not None:
        body["generation_mode"] = mode
    return client.post(
        "/v1/chat/completions", headers={"x-mtplx-cache-mode": "bypass"}, json=body
    )


@pytest.fixture
def operator_default(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)


def test_interleaved_requests_carry_their_own_decisions(monkeypatch, operator_default):
    client, seen = _client(monkeypatch, [True, False, True, False])

    for _ in range(4):
        assert _post(client).status_code == 200

    assert [row["argument"] for row in seen] == [False, None, False, None]
    # Nothing a request decides reaches the process environment, and a
    # reader on another thread keeps seeing the operator's own setting.
    assert [row["environment"] for row in seen] == [None] * 4
    assert [row["other_thread_enabled"] for row in seen] == [True] * 4
    assert os.environ.get(ENV) is None


def test_a_failed_generation_leaves_nothing_behind(monkeypatch, operator_default):
    client, seen = _client(monkeypatch, [True, False], raise_on={0})

    # The test client re-raises the server's error for the failed request.
    with pytest.raises(RuntimeError, match="generation failed"):
        _post(client)
    second = _post(client)

    assert second.status_code == 200
    assert [row["argument"] for row in seen] == [False, None]
    assert os.environ.get(ENV) is None


def test_a_cancelled_generation_leaves_nothing_behind(monkeypatch, operator_default):
    seen: list[dict] = []
    decisions = [True, False]
    calls = {"admission": 0, "generation": 0}
    output = _fake_generation_output()

    def fake_admission(*_args, **_kwargs):
        index = calls["admission"]
        calls["admission"] += 1
        return (
            {"action": "prefill_admission_shed", "prompt_publish_skipped": True}
            if decisions[index]
            else None
        )

    def fake_generate(*args, **kwargs):
        index = calls["generation"]
        calls["generation"] += 1
        seen.append(
            {
                "argument": kwargs.get("store_prefix_snapshot", "absent"),
                "environment": os.environ.get(ENV),
            }
        )
        if index == 0:
            raise openai._StreamCancelled("client disconnected during prefill")
        return output(*args, **kwargs)

    client, _state = _envelope_client(monkeypatch, generator=fake_generate)
    monkeypatch.setattr(openai, "_prefill_admission_shed", fake_admission)

    _post(client)
    assert _post(client).status_code == 200

    assert [row["argument"] for row in seen] == [False, None]
    assert [row["environment"] for row in seen] == [None, None]
    assert os.environ.get(ENV) is None


def test_the_operator_kill_switch_still_reaches_every_request(monkeypatch):
    monkeypatch.setenv(ENV, "0")
    client, seen = _client(monkeypatch, [True, False])

    for _ in range(2):
        assert _post(client).status_code == 200

    # The request argument only ever narrows publication; the operator's
    # switch is untouched and every reader still sees it off.
    assert [row["argument"] for row in seen] == [False, None]
    assert [row["environment"] for row in seen] == ["0", "0"]
    assert [row["other_thread_enabled"] for row in seen] == [False, False]


def test_the_ar_lane_carries_the_decision_too(monkeypatch, operator_default):
    client, seen = _client(monkeypatch, [True, False])

    for _ in range(2):
        assert _post(client, mode="ar").status_code == 200

    assert [row["argument"] for row in seen] == [False, None]
    assert [row["environment"] for row in seen] == [None, None]
