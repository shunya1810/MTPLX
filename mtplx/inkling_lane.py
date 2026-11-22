"""Which Inkling 2 Mini implementation serves a model folder.

Upstream MTPLX 2.13 ships its own Inkling runtime (``mtplx.models.inkling``:
its fast engine, the DSpark draft head, the FP16 packs for M1 / M2). This
fork had ported the model before (``mtplx.models.inkling2_loop``, an AR lane
with M1 kernels, backend ``inkling2_ar``). Both load ``model_type
inkling2_loop``; ``fork_lane`` decides (the fork's lane on the M1 family,
upstream's runtime elsewhere).
"""

from __future__ import annotations

import os


def fork_lane() -> bool:
    """``MTPLX_INKLING2_FORK_LANE`` 1/0 forces; unset, the fork's lane on the
    M1 family (``MTPLX_M1_LONG_CONTEXT`` / the host CPU, as every M1 default).

    On an M1 Max (2026-11-22, Optimized Speed FP16 pack, server, temperature
    0) upstream 2.13.0's runtime failed requests with "Thread group size (512)
    is greater than the maximum allowed threads per threadgroup (448)" (2K on
    every run, 16K / 32K on pristine 2.13.0); where it answered it read 16K /
    32K at 408 / 377 tok/s and decoded 29.2 / 25.9 tok/s. The fork's lane on
    the official weights: 478 / 522 / 490 tok/s and 48.2 / 44.7 / 41.6 tok/s
    at 2K / 16K / 32K, every request answered. It also reads upstream's packs
    (their 8-bit projections load as they are).
    """
    raw = (os.environ.get("MTPLX_INKLING2_FORK_LANE") or "").strip().lower()
    if raw in {"1", "true", "on", "yes"}:
        return True
    if raw in {"0", "false", "off", "no"}:
        return False
    try:
        from .mlx_process_env import m1_family_process

        return bool(m1_family_process())
    except Exception:
        return False
