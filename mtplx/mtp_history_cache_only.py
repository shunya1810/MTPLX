"""Draft-head history appends in a prefill loop evaluate the cache only.

A prefill loop appends every chunk to the MTP draft head's history so the
first draft step after the prompt can attend to it. ``_append_mtp_history``
then evaluated the head's output hidden, which nothing reads: the append only
exists for the keys and values it writes. On mlx-lm's Qwen3.5/3.6 head (one
full-attention decoder layer, ``mtp_patch._mtp_core``) the cache write comes
first (input norms, ``fc``, the q/k/v projections, rope), the attention over
the whole history, ``o_proj``, the routed and shared experts and the final
norm come after it and only feed the unused hidden.

MLX computes lazily, so naming the cache arrays in the eval instead of the
hidden runs exactly the kernels behind the cache write, on the same inputs:
the cache gets the same bits and the rest of the layer is never dispatched.
No model code changes (Flash-Next does the same inside its own head, commit
9760a15c).

Prefill phase only, and only for plain mlx-lm ``KVCache`` entries (the head's
own cache as ``make_mtp_cache`` builds it); any other cache kind, decode, and
every caller outside a prefill loop keep evaluating the hidden.

On by default: the cache is bit-identical to the full layer pass and the first
draft step after the prompt is unchanged (tests/test_mtp_history_cache_only.py),
so there is nothing to trade.  ``MTPLX_MTP_HISTORY_CACHE_ONLY=0`` evaluates the
hidden again.
"""

from __future__ import annotations

import os
from typing import Any

ENV = "MTPLX_MTP_HISTORY_CACHE_ONLY"


def mtp_history_cache_only_enabled() -> bool:
    """On unless ``MTPLX_MTP_HISTORY_CACHE_ONLY`` is set to a false value."""

    return os.environ.get(ENV, "").strip().lower() not in {"0", "false", "no", "off"}


def mtp_history_cache_arrays(mtp_cache: Any) -> list | None:
    """The arrays a history append leaves on ``mtp_cache``, or None when some
    entry is not a plain ``KVCache`` (the caller then evaluates the hidden)."""

    from mlx_lm.models.cache import KVCache

    if not mtp_cache:
        return None
    arrays: list = []
    for entry in mtp_cache:
        if type(entry) is not KVCache or entry.keys is None or entry.values is None:
            return None
        arrays.extend((entry.keys, entry.values))
    return arrays
