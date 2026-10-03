"""The context window this server can execute, as clients should see it.

The app used to write the window from its settings (262,144) into Pi and
OpenCode even when the engine could not serve a conversation that long
(2026-09-29). ``served_execution_window`` is the one place that answers how
long a conversation, prompt plus answer, this server executes. ``/health``
publishes the answer as ``execution_window`` and the app configures Pi and
OpenCode from it.

The window is also the answer ceiling clients advertise: Pi's ``maxTokens``
is the whole window (Pi clamps each request to the room its prompt leaves),
and the server caps each answer to the memory actually free (``_answer_room``
in ``mtplx/server/openai.py``), so no fixed share of the window is held back
from an answer. OpenCode's ``limit.output`` is its own reply reserve
(``mtplx.opencode.opencode_output_limit``).

Today the answer comes from what the server already computes: the resolved
serving window, bounded by the memory planner's machine fit
(``memory_plan.plan_memory``) unless the operator chose ``--allow-swap``. A
qualified per-machine profile can replace the body of this function later
without touching any client.
"""

from __future__ import annotations

from typing import Any


def served_execution_window(state: Any) -> dict[str, Any]:
    """The conversation length this server executes.

    ``tokens`` is the resolved window (``--context-window``, else the plan's
    default) bounded by the memory plan's machine fit, the largest window
    whose state fits this Mac's engine budget. An explicit window above the
    fit is still accepted by the server (it warns and sheds caches), but it
    is not a length clients should plan to use. ``--allow-swap`` means the
    operator accepts swap past the fit, so the resolved window stands.
    """

    configured = max(0, int(getattr(state, "context_window", 0) or 0))
    allow_swap = bool(getattr(state, "allow_swap", False))
    plan = getattr(state, "memory_plan", None)
    fit = 0
    if plan is not None and bool(getattr(plan, "available", False)):
        fit = max(0, int(getattr(plan, "context_window_fit", 0) or 0))
    tokens = configured
    basis = "configured_window"
    if configured > 0 and fit > 0 and fit < configured and not allow_swap:
        tokens = fit
        basis = "machine_fit"
    return {
        "tokens": int(tokens),
        "basis": basis,
        "configured_tokens": int(configured),
        "machine_fit_tokens": int(fit) if fit > 0 else None,
        "allow_swap": allow_swap,
    }
