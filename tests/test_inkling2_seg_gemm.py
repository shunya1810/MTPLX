"""Expert-segmented GEMM: the same values as mx.gather_qmm on sorted rows,
NVFP4 and MXFP8, uneven expert runs (some empty, one longer than a tile)."""

import unittest

import mlx.core as mx
import numpy as np

from mtplx.kernels.inkling2_seg_gemm import seg_gemm, tile_table


class TestSegGemm(unittest.TestCase):
    def test_matches_gather_qmm(self):
        E, N, K = 8, 128, 64
        counts = [0, 3, 70, 1, 0, 33, 5, 9]
        idx = mx.array(np.repeat(np.arange(E), counts).astype(np.uint32))
        P = int(idx.shape[0])
        for mode, gs, bits in (("nvfp4", 16, 4), ("mxfp8", 32, 8)):
            mx.random.seed(bits)
            w = (mx.random.normal((E, N, K)) * 0.05).astype(mx.float16)
            wq, sc = mx.quantize(w, group_size=gs, bits=bits, mode=mode)[:2]
            x = mx.random.normal((P, K)).astype(mx.float16)
            ref = mx.gather_qmm(x[:, None, :], wq, sc, None, rhs_indices=idx, transpose=True,
                                group_size=gs, bits=bits, mode=mode, sorted_indices=True)[:, 0]
            got = seg_gemm(x, wq, sc, tile_table(idx, E), bits=bits, group_size=gs)
            err = mx.abs(got.astype(mx.float32) - ref.astype(mx.float32)).max().item()
            self.assertLess(err, 1e-3, mode)

    def test_wide_tiles_match_gather_qmm(self):
        """64-row tiles (experts averaging 256+ routed rows, a 4,096-row
        chunk at top-8 of 128): the same values as the sorted gather."""
        from mtplx.kernels import inkling2_seg_gemm as sg

        E, N, K = 4, 128, 64
        counts = [0, 300, 517, 300]
        self.assertEqual(sg.tile_rows(sum(counts), E), 64)
        self.assertEqual(sg.tile_rows(255 * E, E), 32)
        idx = mx.array(np.repeat(np.arange(E), counts).astype(np.uint32))
        P = int(idx.shape[0])
        mx.random.seed(5)
        w = (mx.random.normal((E, N, K)) * 0.05).astype(mx.float16)
        wq, sc = mx.quantize(w, group_size=16, bits=4, mode="nvfp4")[:2]
        x = mx.random.normal((P, K)).astype(mx.float16)
        ref = mx.gather_qmm(x[:, None, :], wq, sc, None, rhs_indices=idx, transpose=True,
                            group_size=16, bits=4, mode="nvfp4", sorted_indices=True)[:, 0]
        got = seg_gemm(x, wq, sc, tile_table(idx, E), bits=4, group_size=16)
        self.assertEqual(mx.abs(got.astype(mx.float32) - ref.astype(mx.float32)).max().item(), 0.0)


if __name__ == "__main__":
    unittest.main()


class TestSteelAttention(unittest.TestCase):
    def test_matches_relative_attention(self):
        from mtplx.kernels.inkling2_kernels import relative_attention
        from mtplx.kernels.inkling2_steel_attn import steel_relative_attention

        for L, S, window, start in ((40, 40, 16, 0), (70, 130, 0, 5), (33, 90, 32, 0)):
            mx.random.seed(L + S)
            q = (mx.random.normal((1, L, 6, 128)) * 0.3).astype(mx.float16).transpose(0, 2, 1, 3)
            kb = (mx.random.normal((1, 2, S + 8, 128)) * 0.3).astype(mx.float16)
            vb = mx.random.normal((1, 2, S + 8, 128)).astype(mx.float16)
            rel = mx.random.normal((1, L, 6, 24)).astype(mx.float16)
            sinks = mx.random.normal((1, L, 6)).astype(mx.float16)
            kw = dict(scale=128 ** -0.5, window=window, key_start=mx.array([start], mx.int32))
            args = (q, kb[:, :, :S], vb[:, :, :S], rel, sinks)
            ref = relative_attention(*args, **kw).astype(mx.float32)
            got = steel_relative_attention(*args, **kw).astype(mx.float32)
            self.assertLess(mx.abs(got - ref).max().item(), 4e-3, f"L={L} S={S}")
