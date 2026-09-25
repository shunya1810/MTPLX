#!/usr/bin/env python3
"""One layer's causal prefill attention on sdpa_gqa_mma_prefill vs rows per threadgroup.

A chunk of L new queries after P cached keys (Qwen3.8: Hq 24, Hk 4, D 256),
dense fp16 K/V. ``block_positions`` query positions share one K/V stream, so
rows per threadgroup = GQA 6 x block_positions; the kernel re-reads the whole
prefix per block, so more rows means more math per byte read.

Reports ms per layer-chunk and effective TFLOPS (4 * Hq * D * sum of visible
keys over the L queries).
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import mlx.core as mx

from mtplx.kernels.sdpa_gqa_mma import sdpa_gqa_mma_prefill

HQ, HK, D = 24, 4, 256


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefixes", default="65536,131072,258048")
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--positions", default="4,5,6@8", help="block_positions[@simdgroups], comma list")
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()
    for prefix in [int(v) for v in args.prefixes.split(",")]:
        cap = prefix + args.chunk
        k = (mx.random.normal((1, HK, cap, D), key=mx.random.key(1)) * 0.5).astype(mx.float16)
        v = (mx.random.normal((1, HK, cap, D), key=mx.random.key(2)) * 0.5).astype(mx.float16)
        q = (mx.random.normal((1, HQ, args.chunk, D), key=mx.random.key(3)) * 0.5).astype(mx.float16)
        mx.eval(k, v, q)
        visible = args.chunk * prefix + args.chunk * (args.chunk + 1) / 2
        flops = 4.0 * HQ * D * visible
        ref = None
        for spec in args.positions.split(","):
            ql, _, nsg = spec.partition("@")
            ql, nsg = int(ql), int(nsg or 4)

            def run(ql=ql, nsg=nsg):
                return sdpa_gqa_mma_prefill(
                    queries=q, keys=k, values=v, prefix=prefix,
                    scale=D ** -0.5, num_kv_heads=HK, block_positions=ql,
                    simdgroups=nsg,
                )
            out = run()
            if out is None:
                print(json.dumps({"prefix": prefix, "block_positions": ql, "simdgroups": nsg, "bail": True}))
                continue
            mx.eval(out)
            if ref is None:
                ref = out
            err = float(mx.max(mx.abs(out.astype(mx.float32) - ref.astype(mx.float32))).item())
            ts = []
            for _ in range(args.reps):
                mx.synchronize()
                t = time.perf_counter()
                mx.eval(run())
                mx.synchronize()
                ts.append(time.perf_counter() - t)
            sec = statistics.median(ts)
            print(json.dumps({"prefix": prefix, "chunk": args.chunk, "block_positions": ql,
                              "simdgroups": nsg, "rows": 6 * ql, "ms": round(sec * 1e3, 1),
                              "tflops": round(flops / sec / 1e12, 2), "max_diff_vs_first": err}), flush=True)
        del k, v, q
        mx.clear_cache()


if __name__ == "__main__":
    main()
