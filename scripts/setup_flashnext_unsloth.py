#!/usr/bin/env python3
"""Download and convert Qwen3.8-Flash-Next (Unsloth UD-Q2_0) for this fork.

    python scripts/setup_flashnext_unsloth.py            # into ~/models
    python scripts/setup_flashnext_unsloth.py --models-dir /Volumes/ssd/models

What it does (each step skips what is already there, so it can be re-run
after an interrupted download):

1. Downloads Unsloth's UD-Q2_0 GGUF (unsloth/Qwen3.8-Flash-Next-GGUF, three
   splits, 70.5 GB: metadata, the trunk, the n-gram table) into
   <models-dir>/Qwen3.8-Flash-Next-UD-Q2_0-GGUF.
2. Downloads the config, tokenizer, chat template and MTP head (1.7 GB) of
   Youssofal/Qwen3.8-Flash-Next-MTPLX-Optimized-Speed; the GGUF has no MTP
   head.
3. Converts with scripts/convert_flashnext_ista_gguf.py into
   <models-dir>/Qwen3.8-Flash-Next-UD-Q2_0-MTPLX (5.6 GB + the MTP head), in a few
   minutes on the CPU.
   Dense weights are requantized to MLX 8-bit; the routed experts and the
   n-gram table stay in the GGUF and are read from it, so keep the GGUF
   folder where it is.

Serve the result with
    mtplx serve --model <models-dir>/Qwen3.8-Flash-Next-UD-Q2_0-MTPLX
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

GGUF_REPO = "unsloth/Qwen3.8-Flash-Next-GGUF"
GGUF_FILES = {
    "UD-Q2_0/Qwen3.8-Flash-Next-UD-Q2_0-00001-of-00003.gguf": 10_946_618,
    "UD-Q2_0/Qwen3.8-Flash-Next-UD-Q2_0-00002-of-00003.gguf": 41_738_806_688,
    "UD-Q2_0/Qwen3.8-Flash-Next-UD-Q2_0-00003-of-00003.gguf": 28_800_138_432,
}
REF_REPO = "Youssofal/Qwen3.8-Flash-Next-MTPLX-Optimized-Speed"
REF_FILES = ("config.json", "mtplx_runtime.json", "tokenizer.json", "tokenizer_config.json",
             "vocab.json", "merges.txt", "chat_template.jinja", "generation_config.json", "LICENSE")
MTP_FILE = ("mtp.safetensors", 1_678_147_375)
PACK_BYTES = 6_000_000_000  # converted dense weights, with margin
GIB = 1 << 30


def _size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models-dir", type=Path, default=Path.home() / "models")
    ap.add_argument("--gguf-dir", type=Path, help="default: <models-dir>/Qwen3.8-Flash-Next-UD-Q2_0-GGUF")
    ap.add_argument("--out", type=Path, help="default: <models-dir>/Qwen3.8-Flash-Next-UD-Q2_0-MTPLX")
    ap.add_argument("--yes", action="store_true", help="do not ask before downloading")
    args = ap.parse_args()

    models = args.models_dir.expanduser()
    gguf_dir = (args.gguf_dir or models / "Qwen3.8-Flash-Next-UD-Q2_0-GGUF").expanduser()
    out = (args.out or models / "Qwen3.8-Flash-Next-UD-Q2_0-MTPLX").expanduser()
    ref_dir = out.parent / (out.name + ".ref")
    for d in (gguf_dir, out.parent):
        d.mkdir(parents=True, exist_ok=True)

    missing = sum(max(0, size - _size(gguf_dir / name)) for name, size in GGUF_FILES.items())
    missing += max(0, MTP_FILE[1] - _size(out / MTP_FILE[0]) - _size(ref_dir / MTP_FILE[0]))
    need = missing + (0 if (out / "config.json").exists() else PACK_BYTES)
    free = min(shutil.disk_usage(gguf_dir).free, shutil.disk_usage(out.parent).free)
    print(f"GGUF:  {gguf_dir}\nmodel: {out}")
    print(f"to download {missing / GIB:.1f} GiB, disk needed {need / GIB:.1f} GiB, free {free / GIB:.1f} GiB")
    if need > free:
        print("not enough free disk space", file=sys.stderr)
        return 1
    if missing and not args.yes:
        if input("download now? [y/N] ").strip().lower() not in ("y", "yes"):
            return 1

    from huggingface_hub import hf_hub_download

    for name, size in GGUF_FILES.items():
        if _size(gguf_dir / name) != size:
            print(f"downloading {name} ({size / GIB:.1f} GiB)", flush=True)
            hf_hub_download(GGUF_REPO, name, local_dir=gguf_dir)
        if _size(gguf_dir / name) != size:
            print(f"{name}: size {_size(gguf_dir / name)} != {size}", file=sys.stderr)
            return 1
    ref_dir.mkdir(parents=True, exist_ok=True)
    for name in REF_FILES:
        if not (ref_dir / name).exists():
            hf_hub_download(REF_REPO, name, local_dir=ref_dir)
    if _size(out / MTP_FILE[0]) != MTP_FILE[1] and _size(ref_dir / MTP_FILE[0]) != MTP_FILE[1]:
        print(f"downloading {MTP_FILE[0]} ({MTP_FILE[1] / GIB:.1f} GiB)", flush=True)
        hf_hub_download(REF_REPO, MTP_FILE[0], local_dir=ref_dir)

    out.mkdir(parents=True, exist_ok=True)
    if _size(out / MTP_FILE[0]) != MTP_FILE[1]:
        os.replace(ref_dir / MTP_FILE[0], out / MTP_FILE[0])
    if not (out / "config.json").exists():
        convert = Path(__file__).resolve().parent / "convert_flashnext_ista_gguf.py"
        cmd = [sys.executable, str(convert), "--gguf", *(str(gguf_dir / n) for n in GGUF_FILES),
               "--ref", str(ref_dir), "--out", str(out),
               "--source", f"{GGUF_REPO} UD-Q2_0", "--model-id", "mtplx-flash-next-unsloth-ud-q2_0"]
        print("converting (CPU, a few minutes)", flush=True)
        if subprocess.run(cmd).returncode != 0:
            return 1
    shutil.rmtree(ref_dir, ignore_errors=True)
    print(f"\ndone. Serve it with:\n  mtplx serve --model {out}\n"
          f"(the experts and the n-gram table are read from {gguf_dir}; keep it in place)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
