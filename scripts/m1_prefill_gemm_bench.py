#!/usr/bin/env python3
"""Prefill-chunk matmul: quantized_matmul vs dequantize-then-fp16-GEMM on M1.

For each trunk weight shape (4-bit g32 and 8-bit g64 as in Qwen3.8-27B
Optimized-Speed) and chunk M, times over `layers` distinct weights so they
stream from DRAM:

- qmm:      mx.quantized_matmul(x, w, scales, biases)
- deq+gemm: w_fp16 = mx.dequantize(...); x @ w_fp16.T  (dequant per call)
- gemm:     x @ w_fp16.T with the dense weight prebuilt (upper bound)

Synthetic weights; timing only.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import mlx.core as mx

SHAPES = [
    ("mlp.gate/up 4b", 5120, 17408, 4, 32),
    ("mlp.down 4b", 17408, 5120, 4, 32),
    ("gdn.qkv 4b", 5120, 10240, 4, 32),
    ("gdn.out 8b", 6144, 5120, 8, 64),
    ("attn.q 4b", 5120, 12288, 4, 32),
]


def timed(fn, warm=1, reps=5):
    ts = []
    for r in range(warm + reps):
        mx.synchronize()
        t = time.perf_counter()
        mx.eval(fn())
        mx.synchronize()
        if r >= warm:
            ts.append(time.perf_counter() - t)
    return statistics.median(ts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ms", default="512,2048")
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    rows = []
    for name, k, n, bits, gs in SHAPES:
        ws = []
        for i in range(args.layers):
            wf = (mx.random.normal((n, k), key=mx.random.key(i)) * 0.02).astype(mx.float16)
            w, s, b = mx.quantize(wf, group_size=gs, bits=bits)
            mx.eval(w, s, b)
            ws.append((w, s, b))
            del wf
        dense = [mx.dequantize(w, s, b, group_size=gs, bits=bits) for w, s, b in ws]
        mx.eval(dense)
        for m in [int(v) for v in args.ms.split(",")]:
            x = (mx.random.normal((m, k)) * 0.1).astype(mx.float16)
            mx.eval(x)
            flops = 2.0 * m * k * n * args.layers

            def qmm():
                return [mx.quantized_matmul(x, w, s, b, transpose=True, group_size=gs, bits=bits) for w, s, b in ws]

            def deq_gemm():
                return [x @ mx.dequantize(w, s, b, group_size=gs, bits=bits).T for w, s, b in ws]

            def gemm():
                return [x @ d.T for d in dense]

            res = {}
            for lane, fn in (("qmm", qmm), ("deq_gemm", deq_gemm), ("gemm", gemm)):
                sec = timed(fn)
                res[lane] = {"ms_per_layer": sec * 1e3 / args.layers, "tflops": flops / sec / 1e12}
            row = {"shape": name, "k": k, "n": n, "bits": bits, "m": m, **{
                f"{lane}_{key}": round(val, 2) for lane, d in res.items() for key, val in d.items()}}
            rows.append(row)
            print(json.dumps(row), flush=True)
        del dense, ws
        mx.clear_cache()
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(rows, fh, indent=1)


if __name__ == "__main__":
    main()
