"""DSpark drafter module: a tiny random drafter builds context keys, runs a
block forward and the recurrent selection end to end."""

import unittest

import mlx.core as mx
import mlx.nn as nn

from mtplx.models.inkling2_dspark import DSpark, DSparkConfig

CFG = {
    "hidden_size": 64, "num_hidden_layers": 3, "num_attention_heads": 4, "num_key_value_heads": 2,
    "head_dim": 16, "intermediate_size": 128, "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
    "local_layer_ids": [0, 2], "sliding_window_size": 8, "block_size": 4, "markov_rank": 8,
    "markov_head_type": "rnn_hidden", "target_layer_ids": [1, 2, 3],
}
V = 50


class TestDSpark(unittest.TestCase):
    def test_block_and_selection(self):
        mx.random.seed(0)
        draft = DSpark(DSparkConfig.from_dict(CFG), V)
        from mlx.utils import tree_flatten, tree_unflatten

        draft.update(tree_unflatten([(k, mx.random.normal(v.shape) * 0.1) for k, v in tree_flatten(draft.parameters())]))
        draft.dtype = mx.float32
        head = nn.Linear(64, V, bias=False)
        draft.fold_head(head)
        C = 12
        feats = mx.random.normal((1, C, 3 * 64))
        ctx = [draft.context_kv(i, feats, 0) for i in range(3)]
        self.assertEqual(ctx[0][0].shape, (1, 2, C, 16))
        anchor = mx.array([7])
        h = draft.block_forward(mx.random.normal((1, 1, 64)), ctx, C)
        self.assertEqual(h.shape, (1, 4, 64))
        feat = draft.head_feature(feats[:, -1:])[:, 0]
        toks = draft.select_greedy(h[:, 1:], head(h[:, 1:]), anchor, feat, 30.0)
        self.assertEqual(toks.shape, (3,))
        self.assertTrue(((toks >= 0) & (toks < V)).all().item())


if __name__ == "__main__":
    unittest.main()
