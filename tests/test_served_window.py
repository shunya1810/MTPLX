"""The window clients configure: the one this server executes.

The app wrote 262,144 into Pi's contextWindow and maxTokens from settings on
2026-09-29 while the engine could not serve that. /health now publishes
``execution_window``, from one function, and the app configures Pi and
OpenCode from it. The window is also the answer ceiling clients advertise:
no smaller answer share is published (builds from 59288061 published half the
window, and Pi stopped every answer there while the prompt was short).
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(__file__))

from mtplx.memory_plan import MemoryPlan
from mtplx.server.served_window import served_execution_window


def _state(*, window, fit=None, available=True, allow_swap=False):
    plan = None
    if fit is not None or not available:
        plan = MemoryPlan(available=available, context_window_fit=int(fit or 0))
    return SimpleNamespace(context_window=window, memory_plan=plan, allow_swap=allow_swap)


def test_an_explicit_window_above_the_machine_fit_is_not_what_clients_get():
    served = served_execution_window(_state(window=262_144, fit=98_304))
    assert served["tokens"] == 98_304
    assert served["basis"] == "machine_fit"
    assert served["configured_tokens"] == 262_144
    assert served["machine_fit_tokens"] == 98_304


def test_a_window_inside_the_fit_is_served_as_configured():
    served = served_execution_window(_state(window=131_072, fit=262_144))
    assert served["tokens"] == 131_072
    assert served["basis"] == "configured_window"


def test_allow_swap_keeps_the_operators_window():
    served = served_execution_window(_state(window=262_144, fit=98_304, allow_swap=True))
    assert served["tokens"] == 262_144
    assert served["allow_swap"] is True


def test_an_unavailable_plan_leaves_the_resolved_window():
    served = served_execution_window(_state(window=65_536, available=False))
    assert served["tokens"] == 65_536
    assert served["machine_fit_tokens"] is None
    served = served_execution_window(SimpleNamespace(context_window=32_768))
    assert served["tokens"] == 32_768


def test_the_whole_window_is_the_answer_ceiling_at_the_boundaries():
    # The server caps each answer to the memory actually free (_answer_room
    # in mtplx/server/openai.py); a published share of the window would be a
    # second cap that every client applies whether the memory is there or not.
    for window in (4_096, 32_768, 262_144):
        served = served_execution_window(_state(window=window, fit=262_144))
        assert served["tokens"] == window
        assert "answer_tokens" not in served


def test_cli_handoff_gives_pi_and_opencode_the_served_window(tmp_path, monkeypatch):
    """`mtplx start pi` and `mtplx start opencode` write the client's config
    before the model loads, from the model's window or --context-window
    (262,144 here). Right before the server opens the client, its MTPLX
    entries take served_execution_window (32,768 on this machine fit), the
    value the app configures both clients from, and the next launch's early
    write keeps it instead of flipping back."""
    import json

    from mtplx.opencode import write_opencode_config
    from mtplx.pi import write_pi_models_config
    from mtplx.server.openai import _sync_launched_client_window

    pi_path = tmp_path / "pi" / "models.json"
    opencode_path = tmp_path / "opencode" / "opencode.json"
    monkeypatch.setenv("MTPLX_PI_MODELS_JSON", str(pi_path))
    monkeypatch.setenv("MTPLX_OPENCODE_CONFIG", str(opencode_path))
    model_id = "mtplx-qwen38-27b-optimized-speed"
    base_url = "http://127.0.0.1:8000/v1"

    def early_writes():
        write_pi_models_config(
            base_url=base_url, model_id=model_id, context_window=262_144, keep_window=True
        )
        write_opencode_config(
            base_url=base_url, model_id=model_id, context_window=262_144, keep_window=True
        )

    def client_windows():
        pi = json.loads(pi_path.read_text())["providers"]["mtplx"]["models"][0]
        limit = json.loads(opencode_path.read_text())["provider"]["mtplx"]["models"][
            model_id
        ]["limit"]
        return (pi["contextWindow"], pi["maxTokens"]), limit

    early_writes()
    assert client_windows() == ((262_144, 262_144), {"context": 262_144, "output": 32_000})
    state = _state(window=262_144, fit=32_768)
    state.args = SimpleNamespace(
        launch_pi=True, launch_opencode=True, model_id=model_id, max_response_tokens=None
    )
    _sync_launched_client_window(state)
    served = served_execution_window(state)["tokens"]
    assert served == 32_768
    assert client_windows() == ((served, served), {"context": served, "output": 16_384})
    early_writes()
    assert client_windows() == ((served, served), {"context": served, "output": 16_384})


def test_health_publishes_the_execution_window():
    from fastapi.testclient import TestClient
    from test_server_openai import _fake_state
    from mtplx.server.openai import create_app

    state = _fake_state()
    state.context_window = 262_144
    state.memory_plan = MemoryPlan(available=True, context_window_fit=98_304)
    payload = TestClient(create_app(state)).get("/health").json()
    assert payload["context_window"] == 262_144
    assert payload["execution_window"]["tokens"] == 98_304
    assert payload["execution_window"]["basis"] == "machine_fit"
    assert "answer_tokens" not in payload["execution_window"]
