"""In-forward GDN boundary captures for mlx-lm qwen3_5 layers (qwen35_inforward_boundary)."""

import mlx.core as mx
import numpy as np
import pytest

q5 = pytest.importorskip("mlx_lm.models.qwen3_5")
from mlx_lm.models.cache import ArraysCache, KVCache

from mtplx import qwen35_inforward_boundary as ifb


def _gdn():
    args = q5.TextModelArgs(
        model_type="qwen3_5",
        hidden_size=64,
        num_hidden_layers=2,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=128,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_conv_kernel_dim=4,
        head_dim=16,
    )
    mx.random.seed(7)
    layer = q5.GatedDeltaNet(args)
    layer.eval()
    return layer


def _state(cache):
    mx.eval(cache[0], cache[1])
    return np.array(cache[0]), np.array(cache[1])


def test_segmented_call_matches_forwards_cut_at_the_offsets():
    gdn = _gdn()
    x = mx.random.normal((1, 40, 64))
    edges = (9, 24)

    cut = ArraysCache(size=2)
    reference = {}
    outs = []
    prev = 0
    for edge in (*edges, 40):
        outs.append(q5.GatedDeltaNet.__call__(gdn, x[:, prev:edge], None, cut))
        reference[edge] = _state(cut)
        prev = edge
    cut_out = np.array(mx.concatenate(outs, axis=1))

    whole = ArraysCache(size=2)
    out = ifb._call_with_captures(gdn, x, whole, edges)
    captured = ifb.take_boundary_captures([whole, KVCache()])
    assert sorted(captured) == list(edges)
    for edge in edges:
        states = captured[edge]
        assert states[1] is None
        conv, delta = states[0]
        np.testing.assert_allclose(np.array(conv), reference[edge][0], rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(np.array(delta), reference[edge][1], rtol=1e-4, atol=1e-5)
    final = _state(whole)
    np.testing.assert_allclose(final[0], reference[40][0], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(final[1], reference[40][1], rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(np.array(out), cut_out, rtol=1e-3, atol=1e-4)
    assert whole.offset == cut.offset if hasattr(whole, "offset") else True
    # the captures were popped
    assert ifb.take_boundary_captures([whole, KVCache()]) == {}


def test_a_partial_capture_is_not_a_boundary():
    a, b = ArraysCache(size=2), ArraysCache(size=2)
    ifb._record(a, 5, (mx.zeros((1,)), mx.zeros((1,))))
    ifb._record(a, 8, (mx.zeros((1,)), mx.zeros((1,))))
    ifb._record(b, 8, (mx.zeros((1,)), mx.zeros((1,))))
    assert sorted(ifb.take_boundary_captures([a, KVCache(), b])) == [8]


def test_scope_arms_and_disarms():
    assert ifb.capture_offsets() == ()
    with ifb.boundary_capture_scope((12, 0, 3, 12)):
        assert ifb.capture_offsets() == (3, 12)
    assert ifb.capture_offsets() == ()
