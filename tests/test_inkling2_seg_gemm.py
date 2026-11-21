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


if __name__ == "__main__":
    unittest.main()
