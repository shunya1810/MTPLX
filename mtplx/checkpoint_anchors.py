"""Which recurrent checkpoints a conversation keeps, and why.

A checkpoint (a "boundary record" ``(position, recurrent snapshot, hidden)``)
is the recurrent state (GDN matrix state, conv and PLE tails) after the first
``position`` tokens of a prompt, captured during prefill. A hybrid model can
only resume at a checkpoint at or below the matched prefix: the KV is trimmed
to it and the rest of the prompt is prefilled again. Each one costs about
115.6 MB on Flash-Next, so a conversation keeps a few.

Prefill captures one at every chunk edge (and at the tail ladder's rungs).
Until 2026-09-30 the list was thinned back to eight after every capture,
keeping one record per power-of-two distance from the NEWEST record. Thinning
again each time the newest record moved eroded the middle: on 2026-09-29 a
16,371-token match restored at 4,096 (B-cache 2.6 reproduced it with the real
function). The kept set is now chosen from anchors that do not move as the
frontier advances, in this order:

* the prompt-end anchor: the newest checkpoint at least ``tail_backoff``
  tokens before the prompt end, where the next agent turn diverges (the
  reply after it is re-rendered);
* for an image prompt, the pre-image anchor: the newest checkpoint at or
  before its first image, where a turn that changes that image resumes;
* the restore point this prefill resumed from (the previous prompt's end);
* the stable prompt-prefix edge (the turn the tool-continuation hint rides);
* the newest checkpoint;
* a fixed absolute grid: the first checkpoint in every ``grid_tokens`` cell,
  thinned by halving its resolution when the budget cannot hold every cell
  (every 16,384 tokens, then 32,768, ...), so a kept grid anchor keeps its
  position and a restore point once covered stays covered at the resolution
  the budget affords;
* then the newest remaining checkpoints while the budget lasts.

A checkpoint's grid rank depends only on its cell and the cell of the
checkpoint before it, and dropping a checkpoint that shares its cell with an
earlier one never changes a later one's rank. So within one prefill,
trimming after every capture keeps exactly what trimming once at the end
would, and the order captures arrive in cannot erode the set (tested on
random prefills). Across turns a dropped checkpoint stays dropped: a session
that grew long and then rewound keeps the grid its longest prompt could
afford, never less.

Retention is bounded by bytes, passed in by the caller (``budget_bytes``, the
session bank's ``checkpoint_budget_bytes``). Without one it is what today's
memory plan prices: ``MTPLX_GDN_BOUNDARY_MAX`` (8) checkpoints of the model at
hand. A restore never starts past the proven matched prefix: selection
(``SessionBankEntry.recurrent_boundary_at_or_below``) only ever returns a
checkpoint at or below it, and ``image_spans`` keeps every checkpoint of an
image prompt out of its images (a restore never resumes with part of an
image's rows, ``vision.splice.inside_image``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Iterable

#: Cell width of the absolute grid, in tokens.
GRID_TOKENS = 8192
#: Checkpoints kept when the caller gives no byte budget (the count the
#: prefill memory plan has always priced: eight boundary snapshots).
DEFAULT_CHECKPOINT_COUNT = 8
#: Where the prompt-end anchor sits, in tokens before the prompt end
#: (MTPLX_GDN_BOUNDARY_TAIL_BACKOFF's default, where the tail ladder puts it).
DEFAULT_TAIL_BACKOFF = 64

_FIRST_LEVEL = 1 << 30  # the first checkpoint represents every grid cell


def default_checkpoint_count() -> int:
    """``MTPLX_GDN_BOUNDARY_MAX``: the no-budget retention, in checkpoints."""

    raw = os.environ.get("MTPLX_GDN_BOUNDARY_MAX", str(DEFAULT_CHECKPOINT_COUNT))
    try:
        return max(2, int(raw))
    except (TypeError, ValueError):
        return DEFAULT_CHECKPOINT_COUNT


def default_tail_backoff() -> int:
    """``MTPLX_GDN_BOUNDARY_TAIL_BACKOFF``: how far before the prompt end the
    prefill's tail ladder puts its top rung, the prompt-end anchor. The one
    parse of it, so the prefill and the session bank agree on the anchor."""

    raw = os.environ.get("MTPLX_GDN_BOUNDARY_TAIL_BACKOFF", str(DEFAULT_TAIL_BACKOFF))
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return DEFAULT_TAIL_BACKOFF


@dataclass(frozen=True)
class AnchorPlan:
    """What one prefill knows about its checkpoints before capturing any.

    ``prompt_end``: the last position this prefill can checkpoint (the prompt
    length minus one; None when unknown, and the newest checkpoint stands in).
    ``restore_point``: where the prefill resumed from a stored entry.
    ``stable_prefix``: the stable prompt-prefix edge, when the prompt has one.
    ``ceiling``: no checkpoint past this position.
    ``image_spans``: ``[start, end)`` of each image's rows in an image prompt;
    no checkpoint strictly inside one. Image prompts kept none past their
    first image until the restores could resume past a whole image (the
    content-keyed near-prefix lane, 2026-09-30); a checkpoint between or
    after images is a state a cold prefill of the same keyed prefix reaches
    exactly.
    """

    budget_bytes: int | None = None
    record_count: int | None = None
    prompt_end: int | None = None
    tail_backoff: int = DEFAULT_TAIL_BACKOFF
    restore_point: int | None = None
    stable_prefix: int | None = None
    ceiling: int | None = None
    image_spans: tuple[tuple[int, int], ...] = ()
    grid_tokens: int = GRID_TOKENS

    def admits(self, position: int) -> bool:
        position = int(position)
        if self.ceiling is not None and position > int(self.ceiling):
            return False
        return not any(
            int(start) < position < int(end) for start, end in self.image_spans
        )

    @property
    def first_image_start(self) -> int | None:
        return int(self.image_spans[0][0]) if self.image_spans else None


def _tree_nbytes(value: Any) -> int:
    nbytes = getattr(value, "nbytes", None)
    if isinstance(nbytes, int) and not isinstance(value, (bytes, bytearray)):
        return int(nbytes)
    states = getattr(value, "states", None)
    if states is not None and hasattr(value, "meta_states"):
        return _tree_nbytes(states) + _tree_nbytes(value.meta_states)
    if isinstance(value, dict):
        return sum(_tree_nbytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tree_nbytes(item) for item in value)
    return 0


def record_nbytes(record: Any) -> int:
    """Bytes one checkpoint holds (its snapshot and hidden row); at least 1,
    so records without arrays still count as one each."""

    snapshot = record[1] if len(record) > 1 else None
    hidden = record[2] if len(record) > 2 else None
    return max(1, _tree_nbytes(snapshot) + _tree_nbytes(hidden))


def budget_bytes_for(records: Iterable[Any], plan: AnchorPlan) -> int:
    """The plan's byte budget, or ``record_count`` checkpoints of the
    largest record present when the caller gave none."""

    if plan.budget_bytes is not None:
        return max(0, int(plan.budget_bytes))
    count = plan.record_count if plan.record_count is not None else default_checkpoint_count()
    largest = max((record_nbytes(record) for record in records), default=1)
    return max(0, int(count)) * largest


def _grid_levels(positions: list[int], grid_tokens: int) -> list[int]:
    """Per ascending position: the coarsest grid level whose cell it is the
    first checkpoint of, or -1 when an earlier checkpoint shares its finest
    cell. Cells of level L are ``grid_tokens * 2**L`` wide and nest, so the
    level is the highest bit where its cell index differs from the previous
    checkpoint's."""

    grid = max(1, int(grid_tokens))
    levels: list[int] = []
    previous: int | None = None
    for position in positions:
        cell = int(position) // grid
        if previous is None:
            levels.append(_FIRST_LEVEL)
        elif cell == previous:
            levels.append(-1)
        else:
            levels.append((cell ^ previous).bit_length() - 1)
        previous = cell
    return levels


def retain_checkpoints(records: Iterable[Any], plan: AnchorPlan) -> list:
    """The checkpoints to keep, ascending by position, within the budget."""

    by_position: dict[int, Any] = {}
    for record in records:
        position = int(record[0])
        if position <= 0 or not plan.admits(position):
            continue
        # A later capture of the same position replaces the earlier one.
        by_position[position] = record
    if not by_position:
        return []
    positions = sorted(by_position)
    newest = positions[-1]
    end = int(plan.prompt_end) if plan.prompt_end is not None else newest
    limit = end - max(0, int(plan.tail_backoff))
    if plan.ceiling is not None:
        limit = min(limit, int(plan.ceiling))
    below_limit = [position for position in positions if position <= limit]
    first_image = plan.first_image_start
    pre_image = (
        [position for position in positions if position <= first_image]
        if first_image is not None
        else []
    )
    protected: list[int] = []
    for position in (
        below_limit[-1] if below_limit else None,
        pre_image[-1] if pre_image else None,
        plan.restore_point,
        plan.stable_prefix,
        newest,
    ):
        if position is None:
            continue
        position = int(position)
        if position in by_position and position not in protected:
            protected.append(position)
    levels = dict(zip(positions, _grid_levels(positions, plan.grid_tokens)))
    rest = sorted(
        (position for position in positions if position not in protected),
        key=lambda position: (levels[position] < 0, -levels[position], -position),
    )
    budget = budget_bytes_for(by_position.values(), plan)
    kept: list[int] = []
    spent = 0
    for position in (*protected, *rest):
        size = record_nbytes(by_position[position])
        if spent + size > budget:
            continue
        kept.append(position)
        spent += size
    return [by_position[position] for position in sorted(kept)]


class CheckpointSink(list):
    """A prefill's checkpoint list that knows its plan and keeps its budget.

    It is a list, so the prefill loops, ``PromptState`` and the session bank
    take it as one. ``cuts_forwards=False`` marks a sink that may only record
    where the forwards already end: the prefill must not add a chunk edge, a
    tail ladder or an in-forward capture for it. Image prompts get one, so
    their prefill stays exactly what it was before they kept checkpoints.
    """

    def __init__(
        self,
        records: Iterable[Any] = (),
        *,
        plan: AnchorPlan,
        cuts_forwards: bool = True,
    ) -> None:
        super().__init__(records)
        self.plan = plan
        self.cuts_forwards = bool(cuts_forwards)

    def admits(self, position: int) -> bool:
        return self.plan.admits(position)

    def retain(self) -> None:
        self[:] = retain_checkpoints(self, self.plan)


def checkpoint_coverage(
    records: Iterable[Any],
    *,
    budget_bytes: int | None = None,
    first_image_start: int | None = None,
) -> dict[str, Any]:
    """What a prompt's checkpoints cover, for the request receipt.

    With an image prompt, ``pre_image`` says how close to the first image the
    newest checkpoint before it sits: the most a later turn that changes
    something after the image can restore without re-reading the text.
    """

    records = list(records or ())
    positions = sorted(int(record[0]) for record in records)
    total = sum(record_nbytes(record) for record in records)
    report: dict[str, Any] = {
        "anchors": positions,
        "count": len(positions),
        "bytes": int(total),
        "budget_bytes": (
            int(budget_bytes)
            if budget_bytes is not None
            else budget_bytes_for(records, AnchorPlan())
        ),
    }
    if first_image_start is not None:
        start = int(first_image_start)
        before = [position for position in positions if position <= start]
        anchor = before[-1] if before else None
        report["pre_image"] = {
            "first_image_start": start,
            "anchor": anchor,
            "tokens_to_reread": start - (anchor or 0),
        }
    return report
