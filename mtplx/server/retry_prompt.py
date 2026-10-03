"""Prompt ids for a server-side re-generation of a turn.

A recovery pass (the tool-fed empty retry, the opt-in read-only force-answer
retry) asks the model again with one more user turn after the conversation.
Its prompt must be the ids the first pass was served, up to where the served
generation prompt begins, then the new turn. The served ids carry what a
fresh render of the messages does not: the committed-token splice (the
model's own tokenization of its earlier turns), the committed reasoning
substitution, the expanded image pads the request's vision rows were built
for, and any close the server appended after the generation prompt. A fresh
render loses all of it. On 2026-09-29 a retry prompt built that way diverged
from the session's KV at token 16,371 of 46,861 and restored 4,096 tokens.

The new turn's own ids come from the chat template, so this stays template
agnostic: two renders of the same messages with the same arguments, one with
the turn appended, share everything up to the point where the appended turn
takes over, and the rest of the plain render (the generation prompt, and any
history a template re-renders once a user turn follows it) is what the
appended turn replaces. The served ids must end with exactly that part; if
they do not (the splice, a substitution or an image expansion reached into
it), no exact retry prompt exists and the caller must not re-generate.
"""

from __future__ import annotations

from typing import Any, Sequence

from mtplx.session_bank import common_prefix_len


def served_prompt_with_appended_turn(
    served_ids: Sequence[int],
    plain_ids: Sequence[int],
    appended_ids: Sequence[int],
    *,
    served_suffix_ids: Sequence[int] = (),
) -> tuple[list[int] | None, dict[str, Any]]:
    """The served prompt with the appended turn in place of its generation
    prompt, or None when the served ids do not end the way the template does.

    ``plain_ids`` is the template render of the messages the first pass was
    served, generation prompt included; ``appended_ids`` the render of the
    same messages plus the new turn, with the same encode arguments.
    ``served_suffix_ids`` are ids the server added after the generation
    prompt (the visible-working close); they stay at the end.

    The receipt says how many served ids the retry reuses and how many it
    adds, or why there is no exact retry prompt.
    """

    served = [int(token) for token in served_ids]
    plain = [int(token) for token in plain_ids]
    appended = [int(token) for token in appended_ids]
    suffix = [int(token) for token in served_suffix_ids]
    shared = common_prefix_len(plain, appended)
    replaced = plain[shared:]
    tail = replaced + suffix
    receipt: dict[str, Any] = {
        "served_tokens": len(served),
        "replaced_tokens": len(replaced),
        "appended_turn_tokens": len(appended) - shared,
    }
    if len(tail) > len(served) or served[len(served) - len(tail) :] != tail:
        receipt["reason"] = "served_prompt_tail_differs_from_template"
        return None, receipt
    kept = len(served) - len(tail)
    receipt["reused_served_tokens"] = kept
    return [*served[:kept], *appended[shared:], *suffix], receipt
