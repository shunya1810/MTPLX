"""A float32 Flash-Next stream verifies eagerly instead of refusing the request.

Compiled-verify admission sends a float32 stream whose GDN key scale a fused
kernel would rewrite (head 128: 128**-0.5 written with 7 significant digits)
to the eager verifier. On the fixed-M4 lane that must happen before the lane
is installed: an installed lane may not fall back, so a demoted bank used to
end the request with "qwen4 fixed-M4 installation refused: permanent_eager".
The tiny Flash-Next pack of tests/test_qwen4_fixed_m4_verify_exactness.py,
kept in float32 with GDN key head 128.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from mtplx import demotions
from tests.test_qwen4_fixed_m4_verify_exactness import (  # noqa: F401 (the autouse fixture)
    MAX_TOKENS,
    _generate,
    _lane,
    _runtime,
)

pytestmark = pytest.mark.skipif(
    not mx.metal.is_available(), reason="the Flash-Next tiny pack's expert gathers need the GPU"
)


def test_a_float32_stream_with_gdn_key_head_128_verifies_eagerly(monkeypatch):
    fixed = _generate(
        _runtime(dtype=mx.float32, linear_key_head_dim=128), "1", monkeypatch
    )
    eager = _generate(
        _runtime(dtype=mx.float32, linear_key_head_dim=128), "0", monkeypatch
    )
    admission = fixed.stats.fixed_m4_admission
    assert admission["engaged"] is False
    assert admission["reason"] == "float32_gdn_key_scale"
    assert demotions.counts().get("fixed_m4_lane_skipped", 0) >= 1
    assert "128**-0.5" in demotions.snapshot()["reasons"]["fixed_m4_lane_skipped"]
    assert fixed.tokens == eager.tokens and len(fixed.tokens) == MAX_TOKENS
