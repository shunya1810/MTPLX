"""Generic MoE expert-layout adapters.

Some Hugging Face checkpoints store MoE experts as numbered modules:

    layers.N.mlp.experts.E.gate_proj.weight

MLX's switch-MoE layers load the same experts stacked under:

    layers.N.mlp.switch_mlp.gate_proj.weight

This module owns that translation once so Forge, runtime MTP injection, and
future model families do not grow model-by-model key patches.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


StackFn = Callable[[list[Any]], Any]

EXPERT_COUNT_CONFIG_KEYS = (
    "num_experts",
    "n_routed_experts",
    "n_experts",
    "moe_num_experts",
    "num_local_experts",
    "num_moe_experts",
)


@dataclass(frozen=True)
class NumberedExpertKey:
    source_key: str
    expert_index: int
    module_prefix: str
    leaf: str

    @property
    def output_key(self) -> str:
        return f"{self.module_prefix}.{self.leaf}"


def parse_numbered_expert_key(key: str) -> NumberedExpertKey | None:
    """Return parsed numbered-expert metadata, or None for ordinary keys."""
    text = str(key)
    marker = ".experts."
    if marker not in text:
        return None
    before, after = text.split(marker, 1)
    parts = after.split(".")
    if len(parts) < 3 or not parts[0].isdigit():
        return None
    expert_index = int(parts[0])
    projection = parts[1]
    leaf = ".".join(parts[2:])
    if not projection or not leaf:
        return None
    return NumberedExpertKey(
        source_key=text,
        expert_index=expert_index,
        module_prefix=f"{before}.switch_mlp.{projection}",
        leaf=leaf,
    )


def num_experts_from_config(config: dict[str, Any]) -> int:
    tcfg = config.get("text_config", config) if isinstance(config, dict) else {}
    for key in EXPERT_COUNT_CONFIG_KEYS:
        try:
            value = tcfg.get(key) if isinstance(tcfg, dict) else None
            if value is None:
                value = config.get(key)
            if value:
                return int(value)
        except Exception:
            continue
    return 0


def _default_stack(values: list[Any]) -> Any:
    import mlx.core as mx

    return mx.stack(values, axis=0)


def stack_numbered_experts(
    weights: dict[str, Any],
    *,
    num_experts: int | None = None,
    stack_fn: StackFn | None = None,
    strict: bool = False,
) -> dict[str, Any]:
    """Stack complete numbered expert leaves into switch-MoE leaves.

    Ordinary keys pass through unchanged. In strict mode, any partial numbered
    expert group raises instead of leaking keys the MLX model cannot load.
    """
    stack = stack_fn or _default_stack
    passthrough: dict[str, Any] = {}
    grouped: dict[tuple[str, str], dict[int, tuple[str, Any]]] = {}
    for key, value in weights.items():
        parsed = parse_numbered_expert_key(key)
        if parsed is None:
            passthrough[key] = value
            continue
        grouped.setdefault((parsed.module_prefix, parsed.leaf), {})[parsed.expert_index] = (
            parsed.source_key,
            value,
        )

    if not grouped:
        return dict(weights)

    result = dict(passthrough)
    incomplete: list[str] = []
    for (module_prefix, leaf), experts in sorted(grouped.items()):
        expected = int(num_experts or (max(experts) + 1))
        expected_indexes = set(range(expected))
        present_indexes = set(experts)
        if present_indexes == expected_indexes:
            result[f"{module_prefix}.{leaf}"] = stack(
                [experts[index][1] for index in range(expected)]
            )
            continue
        missing = sorted(expected_indexes - present_indexes)
        label = f"{module_prefix}.{leaf}"
        if missing:
            label += f" missing experts {missing[:8]}"
            if len(missing) > 8:
                label += f" (+{len(missing) - 8} more)"
        incomplete.append(label)
        if not strict:
            for _expert_index, (source_key, value) in sorted(experts.items()):
                result[source_key] = value

    if incomplete and strict:
        raise ValueError(
            "incomplete numbered MoE expert groups: " + "; ".join(incomplete[:8])
        )
    return result


class NumberedExpertAccumulator:
    """Streaming accumulator for shard-by-shard expert stacking."""

    def __init__(self, *, num_experts: int | None = None, stack_fn: StackFn | None = None) -> None:
        self.num_experts = num_experts
        self.stack = stack_fn or _default_stack
        self._groups: dict[tuple[str, str], dict[int, tuple[str, Any]]] = {}

    def add(self, key: str, value: Any) -> bool:
        parsed = parse_numbered_expert_key(key)
        if parsed is None:
            return False
        self._groups.setdefault((parsed.module_prefix, parsed.leaf), {})[
            parsed.expert_index
        ] = (parsed.source_key, value)
        return True

    def flush_complete(self) -> dict[str, Any]:
        if self.num_experts is None:
            return {}
        complete: dict[str, Any] = {}
        for group_key, experts in list(self._groups.items()):
            expected = int(self.num_experts)
            if set(experts) != set(range(expected)):
                continue
            module_prefix, leaf = group_key
            complete[f"{module_prefix}.{leaf}"] = self.stack(
                [experts[index][1] for index in range(expected)]
            )
            del self._groups[group_key]
        return complete

    def flush_remaining(self, *, strict: bool = False) -> dict[str, Any]:
        if not self._groups:
            return {}
        pending: dict[str, Any] = {}
        for experts in self._groups.values():
            for _expert_index, (source_key, value) in experts.items():
                pending[source_key] = value
        self._groups.clear()
        return stack_numbered_experts(
            pending,
            num_experts=self.num_experts,
            stack_fn=self.stack,
            strict=strict,
        )


_FUSED_GATE_UP = ".mlp.experts.gate_up_proj"
_FUSED_DOWN = ".mlp.experts.down_proj"


def _fused_orientation(shape: tuple[int, ...], hidden: int, *, gate_up: bool) -> str | None:
    """``"linear"`` or ``"bmm"`` when a fused tensor's shape names its layout.

    Hub Linear layout: gate_up ``[E, 2*inter, hidden]``, down ``[E, hidden,
    inter]``; transformers bmm layout: gate_up ``[E, hidden, 2*inter]``, down
    ``[E, inter, hidden]``. A square tensor (``2*inter == hidden`` for
    gate_up, ``inter == hidden`` for down) fits both and names neither.
    """
    if len(shape) != 3 or (shape[1] == hidden) == (shape[2] == hidden):
        return None
    if gate_up:
        return "bmm" if shape[1] == hidden else "linear"
    return "linear" if shape[1] == hidden else "bmm"


def split_fused_experts(weights: dict[str, Any], *, hidden_size: int) -> dict[str, Any]:
    """Map fused ``experts.gate_up_proj`` / ``experts.down_proj`` onto switch-MoE leaves.

    Official Qwen3.5/3.6 MoE checkpoints store each MoE block's routed experts
    as two fused tensors, the MTP layer included:

        layers.N.mlp.experts.gate_up_proj   [E, 2*inter, hidden]
        layers.N.mlp.experts.down_proj      [E, hidden, inter]

    (or the transformers bmm layout ``[E, hidden, 2*inter]`` / ``[E, inter,
    hidden]``). MLX's switch-MoE layers load them as ``switch_mlp.{gate,up,
    down}_proj.weight``. Without this mapping ``load_weights(strict=False)``
    drops both keys and the routed experts keep their random init. Gate is the
    first half of the fused projection, as in transformers. Other keys pass
    through unchanged. Every loader and the forge sidecar writer use this one
    mapping.

    A block's two tensors share one layout, and at most one of them can be
    square (``2*inter == hidden`` and ``inter == hidden`` exclude each
    other), so the block's tensors decide it together; a block whose layout
    they cannot name, or name two ways, is refused rather than guessed. The
    leaves are made contiguous, the layout the numbered-expert stack
    produces, so a fused head computes exactly what the same head saved as
    numbered experts computes.
    """
    if not any(key.endswith((_FUSED_GATE_UP, _FUSED_DOWN)) for key in weights):
        return weights
    import mlx.core as mx

    hidden = int(hidden_size)
    orientations: dict[str, set[str]] = {}
    shapes: dict[str, list[tuple[int, ...]]] = {}
    for key, value in weights.items():
        for suffix, gate_up in ((_FUSED_GATE_UP, True), (_FUSED_DOWN, False)):
            if key.endswith(suffix):
                prefix = key[: -len(suffix)]
                shape = tuple(int(dim) for dim in value.shape)
                shapes.setdefault(prefix, []).append(shape)
                found = orientations.setdefault(prefix, set())
                orientation = _fused_orientation(shape, hidden, gate_up=gate_up)
                if orientation is not None:
                    found.add(orientation)
    for prefix, found in orientations.items():
        if len(found) != 1:
            raise ValueError(
                f"cannot tell the fused expert layout of {prefix}.mlp.experts: "
                f"shapes {shapes[prefix]} at hidden size {hidden}"
            )
    result: dict[str, Any] = {}
    for key, value in weights.items():
        if key.endswith(_FUSED_GATE_UP):
            prefix = key[: -len(_FUSED_GATE_UP)]
            if orientations[prefix] == {"bmm"}:
                gate, up = mx.split(value, 2, axis=-1)
                gate, up = gate.swapaxes(1, 2), up.swapaxes(1, 2)
            else:
                gate, up = mx.split(value, 2, axis=1)
            result[f"{prefix}.mlp.switch_mlp.gate_proj.weight"] = mx.contiguous(gate)
            result[f"{prefix}.mlp.switch_mlp.up_proj.weight"] = mx.contiguous(up)
        elif key.endswith(_FUSED_DOWN):
            prefix = key[: -len(_FUSED_DOWN)]
            if orientations[prefix] == {"bmm"}:
                value = value.swapaxes(1, 2)
            result[f"{prefix}.mlp.switch_mlp.down_proj.weight"] = mx.contiguous(value)
        else:
            result[key] = value
    return result
