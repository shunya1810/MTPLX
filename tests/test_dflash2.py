"""mtplx/dflash2.py on a tiny random model: the sliding-window context, incremental
updates, the dynamic conv and the candidate selector. (The real checkpoint is checked
against a torch transcription in m1max-longctx-bench-work/fn-s8/dflash2_parity_*.py.)"""

from __future__ import annotations

import numpy as np
import pytest
import mlx.core as mx
from mlx.utils import tree_map

from mtplx.dflash2 import DFlash2Config, DFlash2Draft, GroupedDynamicCausalConv

CFG = DFlash2Config(
    hidden_size=64, num_layers=2, num_heads=4, num_kv_heads=2, head_dim=16, intermediate_size=96,
    sliding_window=8, block_size=4, conv_kernel_size=2, conv_group_size=8, selector_rank=8,
    selector_top_k=4, mask_token_id=3, target_layer_ids=(1, 2, 3), vocab_size=50,
)


def _model(seed: int = 0) -> DFlash2Draft:
    mx.random.seed(seed)
    model = DFlash2Draft(CFG, dtype=mx.float32)

    def init(p):
        return mx.random.normal(p.shape) * (0.3 if p.ndim >= 2 else 0.1) + (1.0 if p.ndim == 1 else 0.0)

    model.update(tree_map(init, model.parameters()))
    mx.eval(model.parameters())
    return model


def _taps(n: int, seed: int = 1):
    mx.random.seed(seed)
    return mx.random.normal((n, len(CFG.target_layer_ids) * CFG.hidden_size))


def test_conv_matches_a_loop():
    mx.random.seed(2)
    conv = GroupedDynamicCausalConv(64, 2, 8)
    conv.base_kernel = mx.random.normal(conv.base_kernel.shape)
    conv.kernel_projection.weight = mx.random.normal(conv.kernel_projection.weight.shape) * 0.2
    x = mx.random.normal((5, 64))
    mixed, dyn = conv.prepare(x)
    xn = np.array(x); dn = np.array(conv.kernel_projection(x)).reshape(5, 2, 2, 8)
    base = np.array(conv.base_kernel)
    want = np.zeros((5, 8, 8))
    for t in range(5):
        for o in range(2):
            if t - o < 0:
                continue
            blocks = xn[t - o].reshape(8, 8)
            want[t] += (base[0, o].reshape(8, 8) + dn[t, 0, o][:, None]) * blocks
    np.testing.assert_allclose(np.array(mixed), want.reshape(5, 64), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.array(dyn), dn[:, 1], rtol=1e-6)


def test_incremental_context_equals_one_shot():
    model = _model()
    taps = _taps(7)
    one, parts = model.new_context(), model.new_context()
    model.update_context(one, taps, 100)
    model.update_context(parts, taps[:3], 100)
    model.update_context(parts, taps[3:], 103)
    for i in range(CFG.num_layers):
        np.testing.assert_array_equal(np.array(one.k[i]), np.array(parts.k[i]))
        np.testing.assert_array_equal(np.array(one.v[i]), np.array(parts.v[i]))
    assert (one.pos0, one.length) == (parts.pos0, parts.length) == (100, 7)


def test_window_keeps_the_last_rows_and_the_block_never_sees_older_ones():
    model = _model()
    taps = _taps(20)
    full, tail = model.new_context(), model.new_context()
    model.update_context(full, taps, 50)
    assert (full.length, full.pos0) == (CFG.sliding_window, 50 + 20 - CFG.sliding_window)
    model.update_context(tail, taps[-CFG.sliding_window:], 50 + 20 - CFG.sliding_window)
    mx.random.seed(5)
    emb = mx.random.normal((CFG.block_size, CFG.hidden_size))
    a = model.block_hidden(full, emb, 70)
    b = model.block_hidden(tail, emb, 70)
    np.testing.assert_array_equal(np.array(a), np.array(b))
    # the block starts right after the held run, so the run's oldest row (position 62) is
    # exactly a window (8) away from the anchor: no block row reaches it; the next one (63) is
    # 7 away from row 0 and is read by it
    oldest = model.new_context()
    model.update_context(oldest, mx.concatenate([taps[:12], taps[12:13] + 5.0, taps[13:]], axis=0), 50)
    np.testing.assert_array_equal(np.array(model.block_hidden(oldest, emb, 70)), np.array(a))
    second = model.new_context()
    model.update_context(second, mx.concatenate([taps[:13], taps[13:14] + 5.0, taps[14:]], axis=0), 50)
    assert np.abs(np.array(model.block_hidden(second, emb, 70))[0] - np.array(a)[0]).max() > 1e-4


def test_non_contiguous_rows_start_a_new_run():
    model = _model()
    ctx = model.new_context()
    model.update_context(ctx, _taps(4), 10)
    model.update_context(ctx, _taps(3, seed=9), 40)
    assert (ctx.pos0, ctx.length) == (40, 3)


def _head(vocab: int = 50):
    mx.random.seed(11)
    table = mx.random.normal((vocab, CFG.hidden_size)) * 0.3

    def embed(ids):
        return table[ids]

    def head(hidden):
        return hidden @ table.T, mx.arange(vocab, dtype=mx.int32)

    return embed, head


def test_propose_is_deterministic_and_returns_candidate_picks():
    model = _model()
    ctx = model.new_context()
    model.update_context(ctx, _taps(6), 20)
    embed, head = _head()
    t1, p1, c1 = model.propose(ctx, 7, 26, embed, head, temperature=0.0)
    t2, _, _ = model.propose(ctx, 7, 26, embed, head, temperature=0.0)
    mx.eval(t1, t2, c1)
    assert p1 is None and t1.shape == (CFG.block_size - 1,) and c1.shape == (CFG.block_size - 1, CFG.selector_top_k)
    assert np.array_equal(np.array(t1), np.array(t2))
    cands = np.array(c1)
    for row, token in enumerate(np.array(t1)):
        assert token in cands[row]
        assert len(set(cands[row].tolist())) == CFG.selector_top_k


def test_sampled_proposal_gives_distributions_over_the_candidates():
    model = _model()
    ctx = model.new_context()
    model.update_context(ctx, _taps(6), 20)
    embed, head = _head()
    tokens, probs, cands = model.propose(ctx, 7, 26, embed, head, temperature=1.0, key=mx.random.key(0))
    mx.eval(tokens, probs, cands)
    p = np.array(probs)
    assert p.shape == (CFG.block_size - 1, CFG.selector_top_k)
    np.testing.assert_allclose(p.sum(-1), 1.0, rtol=1e-5)
    assert all(t in c for t, c in zip(np.array(tokens), np.array(cands)))


def test_config_reads_the_checkpoint_directory(tmp_path):
    import json

    (tmp_path / "config.json").write_text(json.dumps({
        "hidden_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
        "intermediate_size": 96, "sliding_window": 8, "vocab_size": 50, "rms_norm_eps": 1e-6,
        "rope_parameters": {"rope_theta": 10000},
        "dflash_config": {"block_size": 4, "conv_kernel_size": 2, "conv_group_size": 8, "selector_rank": 8,
                          "selector_top_k": 4, "mask_token_id": 3, "target_layer_ids": [1, 2, 3]},
    }))
    cfg = DFlash2Config.from_dir(tmp_path)
    assert cfg == DFlash2Config(**{**CFG.__dict__, "target_layer_ids": (1, 2, 3)})
