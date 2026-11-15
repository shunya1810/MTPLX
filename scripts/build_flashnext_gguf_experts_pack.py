#!/usr/bin/env python3
"""Point a converted Flash-Next pack's routed experts at another GGUF.

    python scripts/build_flashnext_gguf_experts_pack.py \
        --base ~/models/flashnext-ista-mtplx \
        --gguf ~/models/flashnext-unsloth-ud-q2_0/*.gguf \
        --out ~/models/flashnext-unsloth-mtplx

The new pack symlinks every file of --base (dense weights, MTP head,
tokenizer, n-gram table) and rewrites config.json so that
text_config.routed_experts_gguf reads each layer's gate/up/down experts from
--gguf in the blocks they are stored in ("format": "mixed", a "type" per
projection: Q2_0, IQ4_NL, IQ2_XXS or IQ2_S). qwen4_exp loads them into
GGUFMixedSwitchGLU (mtplx.kernels.moe_gguf_iq). Built for Unsloth's UD-Q2_0
GGUF, whose experts are IQ2_XXS / IQ2_S gate+up and Q2_0 / IQ4_NL down.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from convert_flashnext_ista_gguf import Gguf  # noqa: E402

SUPPORTED = {"gate_proj": ("IQ2_XXS", "IQ2_S", "Q2_0"), "up_proj": ("IQ2_XXS", "IQ2_S", "Q2_0"),
             "down_proj": ("Q2_0", "IQ4_NL")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--gguf", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    g = Gguf(sorted(args.gguf))
    cfg = json.load(open(args.base / "config.json"))
    tc = cfg["text_config"]
    n_layer = int(tc["num_hidden_layers"])
    layers = {}
    for il in range(n_layer):
        spec = {}
        for proj, t in (("gate_proj", "ffn_gate_exps"), ("up_proj", "ffn_up_exps"),
                        ("down_proj", "ffn_down_exps")):
            info = g.tensors[f"blk.{il}.{t}.weight"]
            if info["type"] not in SUPPORTED[proj]:
                raise SystemExit(f"layer {il} {proj}: unsupported type {info['type']}")
            cols, rows, n_exp = info["dims"]
            spec[proj] = {"path": str(Path(info["path"]).resolve()), "offset": info["offset"],
                          "experts": n_exp, "rows": rows, "cols": cols, "type": info["type"]}
        if spec["gate_proj"]["type"] != spec["up_proj"]["type"]:
            raise SystemExit(f"layer {il}: gate and up types differ")
        layers[str(il)] = spec
    tc["routed_experts_gguf"] = {"format": "mixed", "layers": layers}
    for key in ("quantization", "quantization_config"):
        if key in cfg:
            cfg[key] = {k: v for k, v in cfg[key].items() if ".switch_mlp." not in k}
    recipe = dict(cfg.get("mtplx_recipe") or {})
    recipe["routed_experts_source"] = sorted(str(p.resolve()) for p in args.gguf)
    cfg["mtplx_recipe"] = recipe

    args.out.mkdir(parents=True, exist_ok=True)
    for f in sorted(args.base.iterdir()):
        if f.name == "config.json" or f.name.endswith(".bak") or ".bak-" in f.name:
            continue
        dst = args.out / f.name
        if not dst.exists():
            os.symlink(f.resolve(), dst)
    json.dump(cfg, open(args.out / "config.json", "w"), indent=2)
    kinds = {}
    for spec in layers.values():
        key = "/".join(spec[p]["type"] for p in ("gate_proj", "up_proj", "down_proj"))
        kinds[key] = kinds.get(key, 0) + 1
    print(f"{args.out}: {n_layer} layers, expert types {kinds}")


if __name__ == "__main__":
    main()
