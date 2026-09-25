#!/usr/bin/env python3
"""Quantized-matmul cost per model shape and M (rows), stock vs the patched lane.

Loads the model under the turbo profile env and groups every trunk
QuantizedLinear by (kind, K, N, bits, group_size). For each group and M, one
sample runs the matmul once on EVERY module of the group (as a forward does)
and syncs once, so each weight is streamed from DRAM rather than from the
48 MB SLC. Two lanes per M:

- stock:   mx.quantized_matmul (MLX kernel choice)
- patched: the module's __call__ (turbo: nax_qmm_m4 at M=4, nax_qmm_m6 at M=6)

GB/s counts the packed weight plus scales and biases of all modules.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

MODEL = "/Users/nakano/.mtplx/models/Youssofal--Qwen3.8-27B-MTPLX-Optimized-Speed-FP16"


def kind_of(path: str) -> str:
    if "lm_head" in path or "embed_tokens" in path:
        return "lm_head"
    leaf = path.split(".")[-1]
    if ".mlp." in path:
        return f"mlp.{leaf}"
    if ".linear_attn." in path:
        return f"gdn.{leaf}"
    if ".self_attn." in path:
        return f"attn.{leaf}"
    return leaf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--ms", default="1,2,3,4,5,6,8,12,16")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument("--m4-impls", default="", help="comma list: sweep MTPLX_NAX_M4_IMPL at M=4 (patched lane only)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from mtplx.profiles import apply_profile_env

    apply_profile_env("turbo")
    import mlx.core as mx
    import mlx.nn as nn

    from mtplx.runtime import load

    rt = load(args.model, mtp=True)
    text_model = getattr(rt.model, "language_model", rt.model)
    groups: dict[tuple, list] = defaultdict(list)
    for path, mod in text_model.named_modules():
        if not isinstance(mod, nn.QuantizedLinear):
            continue
        if path.startswith("mtp") or ".mtp." in path:
            continue
        w = mod["weight"]
        n = int(w.shape[0])
        k = int(w.shape[1]) * 32 // int(mod.bits)
        groups[(kind_of(path), k, n, int(mod.bits), int(mod.group_size))].append(mod)
    # Tied embeddings: the lm_head is embed_tokens.as_linear.
    if text_model.args.tie_word_embeddings:
        emb = text_model.model.embed_tokens
        if isinstance(emb, nn.QuantizedEmbedding):
            n, kp = emb["weight"].shape
            groups[("lm_head", kp * 32 // emb.bits, n, emb.bits, emb.group_size)].append(emb)

    def nbytes(mod) -> int:
        total = mod["weight"].nbytes + mod["scales"].nbytes
        if "biases" in mod and mod["biases"] is not None:
            total += mod["biases"].nbytes
        return total

    def stock(mod, x):
        return mx.quantized_matmul(
            x, mod["weight"], scales=mod["scales"], biases=mod.get("biases") if hasattr(mod, "get") else mod["biases"],
            transpose=True, group_size=mod.group_size, bits=mod.bits, mode=getattr(mod, "mode", "affine"),
        )

    def patched(mod, x):
        if isinstance(mod, nn.QuantizedEmbedding):
            return mod.as_linear(x)
        return mod(x)

    rows = []
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ms = [int(v) for v in args.ms.split(",")]
    impls = [v for v in args.m4_impls.split(",") if v]
    if impls:
        import os

        ms = [4]
    for key, mods in sorted(groups.items()):
        kind, k, n, bits, gs = key
        total_bytes = sum(nbytes(m) for m in mods)
        for m in ms:
            x = (mx.random.normal((1, m, k)) * 0.1).astype(mx.float16)
            mx.eval(x)
            res = {}
            lanes = [("stock", stock), ("patched", patched)]
            if impls:
                lanes = [("stock", stock)]
                for impl in impls:
                    def run(mod, x, impl=impl):
                        os.environ["MTPLX_NAX_M4_IMPL"] = impl
                        return patched(mod, x)
                    lanes.append((impl, run))
            for lane, fn in lanes:
                samples = []
                for r in range(args.warmup + args.repeats):
                    mx.synchronize()
                    s = time.perf_counter()
                    outs = [fn(mod, x) for mod in mods]
                    mx.eval(outs)
                    mx.synchronize()
                    if r >= args.warmup:
                        samples.append(time.perf_counter() - s)
                med = statistics.median(samples)
                res[lane] = {"ms": med * 1e3, "gb_s": total_bytes / med / 1e9}
            row = {"kind": kind, "k": k, "n": n, "bits": bits, "group_size": gs, "modules": len(mods),
                   "bytes": total_bytes, "m": m, **{f"{l}_ms": v["ms"] for l, v in res.items()},
                   **{f"{l}_gb_s": v["gb_s"] for l, v in res.items()}}
            rows.append(row)
            print(json.dumps({kk: (round(vv, 2) if isinstance(vv, float) else vv) for kk, vv in row.items()}), flush=True)
            out_path.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
