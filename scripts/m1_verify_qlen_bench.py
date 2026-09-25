#!/usr/bin/env python3
"""Verify-forward cost vs q_len over a synthetic dense KV (in process, no server).

Loads the model under the turbo profile env, builds the contiguous dense
decode cache the server uses for fp16 KV (paged/owned KV off), fills the 16
full-attention KV caches with random rows up to --contexts and the 48 GDN
states with random values, then times one verify-shaped forward
(``forward_ar(..., return_hidden=True)``, eager) at each q_len. The cache is
rewound after every call, so each sample sees the same context.

Per-kind timings (``--parts``) run each kind alone across all its layers with
one sync at the end: full attention (16), GDN (48), MLP (64), final norm +
lm_head. ``other`` is full minus their sum (embed, norms, residuals, hidden
capture, dispatch).

Timing only: the KV and GDN contents are random, so logits are meaningless.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

MODEL = "/Users/nakano/.mtplx/models/Youssofal--Qwen3.8-27B-MTPLX-Optimized-Speed-FP16"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--contexts", default="8192,65536")
    ap.add_argument("--qlens", default="1,2,3,4,5,6,8,12,16")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument("--parts", action="store_true")
    ap.add_argument("--profile", default="turbo")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from mtplx.profiles import apply_profile_env

    apply_profile_env(args.profile)
    import mlx.core as mx

    from mtplx.generation import _make_target_prefill_cache
    from mtplx.runtime import load

    contexts = [int(x) for x in args.contexts.split(",")]
    qlens = [int(x) for x in args.qlens.split(",")]
    max_q = max(qlens)

    t0 = time.perf_counter()
    rt = load(args.model, mtp=True)
    load_s = time.perf_counter() - t0
    text_model = getattr(rt.model, "language_model", rt.model)
    inner = text_model.model
    layers = inner.layers
    fa_ids = [i for i, l in enumerate(layers) if not l.is_linear]
    gdn_ids = [i for i, l in enumerate(layers) if l.is_linear]

    from mlx_lm.models.base import create_attention_mask, create_ssm_mask

    out_rows: list[dict] = []
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "model": args.model,
        "profile": args.profile,
        "mlx": mx.__version__,
        "machine": platform.machine(),
        "device": mx.metal.device_info() if mx.metal.is_available() else None,
        "load_s": load_s,
        "env": {k: v for k, v in os.environ.items() if k.startswith(("MTPLX", "MLX"))},
    }

    for ctx in contexts:
        os.environ["MTPLX_CURRENT_PREFILL_CONTEXT_TOKENS"] = str(ctx)
        cache = _make_target_prefill_cache(rt)
        # A tiny real prefill so every cache owns correctly shaped buffers.
        mx.eval(rt.forward_ar(mx.array([list(range(1000, 1016))]), cache=cache))
        kv_cls = type(cache[fa_ids[0]]).__name__
        gdn_cls = type(cache[gdn_ids[0]]).__name__
        mx.random.seed(0)
        cap = ctx + max_q + 256
        for i in fa_ids:
            c = cache[i]
            _, h, _, d = c.keys.shape
            dt = c.keys.dtype
            c.keys = (mx.random.normal((1, h, cap, d)) * 0.5).astype(dt)
            c.values = (mx.random.normal((1, h, cap, d)) * 0.5).astype(dt)
            c.offset = ctx
            mx.eval(c.keys, c.values)
        gdn_saved = {}
        for i in gdn_ids:
            c = cache[i]
            conv, state = c[0], c[1]
            c[0] = (mx.random.normal(conv.shape) * 0.1).astype(conv.dtype)
            c[1] = (mx.random.normal(state.shape) * 0.01).astype(state.dtype)
            mx.eval(c[0], c[1])
            gdn_saved[i] = (c[0], c[1])

        def rewind() -> None:
            for i in fa_ids:
                cache[i].offset = ctx
            for i in gdn_ids:
                cache[i][0], cache[i][1] = gdn_saved[i]

        def timed(fn, n_warm, n_rep):
            samples = []
            for r in range(n_warm + n_rep):
                rewind()
                mx.synchronize()
                s = time.perf_counter()
                res = fn()
                mx.eval(res)
                mx.synchronize()
                e = time.perf_counter() - s
                if r >= n_warm:
                    samples.append(e)
            rewind()
            return samples

        mem_ctx = mx.get_active_memory() if hasattr(mx, "get_active_memory") else None
        for q in qlens:
            ids = mx.array([[(1000 + 37 * j) % 200000 for j in range(q)]])

            def full():
                out = rt.forward_ar(ids, cache=cache, return_hidden=True)
                return list(out) if isinstance(out, tuple) else out

            samples = timed(full, args.warmup, args.repeats)
            row = {
                "context": ctx,
                "q_len": q,
                "kv_cache": kv_cls,
                "gdn_cache": gdn_cls,
                "full_ms_median": statistics.median(samples) * 1e3,
                "full_ms_min": min(samples) * 1e3,
                "full_ms_samples": [s * 1e3 for s in samples],
            }
            if args.parts:
                x = (mx.random.normal((1, q, text_model.args.hidden_size)) * 0.1).astype(mx.float16)
                mx.eval(x)
                fa_mask = create_attention_mask(x, cache[fa_ids[0]])
                ssm_mask = create_ssm_mask(x, cache[gdn_ids[0]])

                def attn():
                    return [layers[i].self_attn(x, mask=fa_mask, cache=cache[i]) for i in fa_ids]

                def gdn():
                    return [layers[i].linear_attn(x, mask=ssm_mask, cache=cache[i]) for i in gdn_ids]

                def mlp():
                    return [l.mlp(x) for l in layers]

                def head():
                    y = inner.norm(x)
                    if text_model.args.tie_word_embeddings:
                        return inner.embed_tokens.as_linear(y)
                    return text_model.lm_head(y)

                parts = {}
                for name, fn in (("attn", attn), ("gdn", gdn), ("mlp", mlp), ("head", head)):
                    ps = timed(fn, args.warmup, args.repeats)
                    parts[name] = statistics.median(ps) * 1e3
                parts["other"] = row["full_ms_median"] - sum(parts.values())
                row["parts_ms"] = parts
            row["active_mem_gb"] = (mem_ctx or 0) / 1e9
            out_rows.append(row)
            print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in row.items() if k != "full_ms_samples"}), flush=True)
            out_path.write_text(json.dumps({"meta": meta, "rows": out_rows}, indent=1))
        del cache, gdn_saved
        mx.clear_cache()

    out_path.write_text(json.dumps({"meta": meta, "rows": out_rows}, indent=1))


if __name__ == "__main__":
    main()
