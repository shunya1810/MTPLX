# MTPLX for M1 Macs

[![Release](https://img.shields.io/github/v/release/shunya1810/MTPLX)](https://github.com/shunya1810/MTPLX/releases/latest)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)

English | [日本語](README.ja.md)

An unofficial fork of [MTPLX](https://github.com/youssofal/MTPLX) that makes it faster on Macs with an M1-family chip. It is MTPLX 2.12.2 plus changes that only switch on for M1-family GPUs. It targets Qwen3.5, Qwen3.6 and Qwen3.8 family models, mainly for long prompts and coding agents.

Like upstream, it decodes speculatively with the model's own MTP (multi-token prediction) heads, so the output distribution is the same as upstream's. It serves an OpenAI-compatible and an Anthropic-compatible API.

- **On an M3 or newer Mac, use [upstream MTPLX](https://github.com/youssofal/MTPLX).** This fork was not measured there.
- **You need the upstream MTPLX app.** This fork uses the app's Python runtime and the models you download in the app.
- This is an unofficial fork, not affiliated with the MTPLX project. The changes were developed with AI assistance (Claude) and measured on one MacBook Pro M1 Max (64 GB). Upstream's README is in [README.upstream.md](README.upstream.md).

## Key numbers

This fork and upstream MTPLX 2.12.2 (the version it is based on), measured on the same Mac under the same conditions. Bold is this fork.

### Prefill (tok/s)

How fast a prompt of about 8K, 32K, 64K and 128K tokens is read with no cache.

| Model | | 8K | 32K | 64K | 128K |
|---|---|---|---|---|---|
| Qwen3.6-35B-A3B | this fork | **1,065** | **1,028** | **771** | **499** |
|  | upstream 2.12.2 | 957 | 795 | 607 | 318 |
| Ornith-1.5-35B-A3B | this fork | **1,069** | **1,034** | **766** | **499** |
|  | upstream 2.12.2 | 954 | 795 | 606 | 317 |
| Qwen3.8-27B | this fork | **233** | **211** | **177** | **139** |
|  | upstream 2.12.2 | 150 | 139 | 121 | 87 |
| Huihui-Qwen3.8-27B-abliterated | this fork | **234** | **220** | **181** | **141** |
|  | upstream 2.12.2 | 150 | 139 | 123 | 87 |
| Swift-1.5-Qwen3.8-27b | this fork | **234** | **220** | **181** | **132** |
|  | upstream 2.12.2 | 150 | 139 | 123 | 86 |

### Decode (tok/s)

"Short prompts" is three short questions with thinking on, 1,500 tokens each. 8K to 128K is a 512-token answer after a prompt of that length has been read.

| Model | | short prompts | 8K | 32K | 64K | 128K |
|---|---|---|---|---|---|---|
| Qwen3.6-35B-A3B | this fork | **107.6** | **101.8** | **86.5** | **76.5** | **60.4** |
|  | upstream 2.12.2 | 58.0 | 53.4 | 46.8 | 39.1 | 24.9 |
| Ornith-1.5-35B-A3B | this fork | **110.5** | **106.8** | **94.6** | **81.9** | **63.8** |
|  | upstream 2.12.2 | 63.6 | 61.0 | 86.8 | 29.3 | 16.2 |
| Qwen3.8-27B | this fork | **32.5** | **30.3** | **30.2** | **27.4** | **22.8** |
|  | upstream 2.12.2 | 23.2 | 22.7 | 18.6 | 9.1 | 6.1 |
| Huihui-Qwen3.8-27B-abliterated | this fork | **32.0** | **33.9** | **30.1** | **27.8** | **23.3** |
|  | upstream 2.12.2 | 24.2 | 22.9 | 19.9 | 10.3 | 6.0 |
| Swift-1.5-Qwen3.8-27b | this fork | **31.5** | **32.3** | **29.0** | **27.0** | **16.9** |
|  | upstream 2.12.2 | 24.3 | 23.5 | 19.3 | 10.7 | 5.9 |

### Peak memory

The largest amount of memory MLX allocated while reading the prompt of each length and answering.

| Model | | 8K | 32K | 64K | 128K |
|---|---|---|---|---|---|
| Qwen3.6-35B-A3B | this fork | **32.6 GB** | **32.6 GB** | **32.6 GB** | **32.6 GB** |
|  | upstream 2.12.2 | 22.6 GB | 24.7 GB | 27.3 GB | 32.2 GB |
| Ornith-1.5-35B-A3B | this fork | **32.6 GB** | **32.6 GB** | **32.6 GB** | **32.6 GB** |
|  | upstream 2.12.2 | 23.9 GB | 26.0 GB | 28.6 GB | 33.5 GB |
| Qwen3.8-27B | this fork | **25.1 GB** | **27.4 GB** | **30.6 GB** | **35.5 GB** |
|  | upstream 2.12.2 | 25.5 GB | 29.7 GB | 36.1 GB | 45.0 GB |
| Huihui-Qwen3.8-27B-abliterated | this fork | **20.2 GB** | **22.9 GB** | **26.9 GB** | **35.0 GB** |
|  | upstream 2.12.2 | 21.0 GB | 25.2 GB | 31.6 GB | 41.0 GB |
| Swift-1.5-Qwen3.8-27b | this fork | **20.2 GB** | **22.9 GB** | **26.9 GB** | **35.0 GB** |
|  | upstream 2.12.2 | 21.0 GB | 25.2 GB | 31.6 GB | 41.3 GB |

### 246K prompt (non-repeating text)

From bench 2 of [apple-silicon-llm-bench](https://github.com/shunya1810/apple-silicon-llm-bench): 245,637 tokens of telemetry records, then three turns with thinking on (temperature 1.0, up to 4,096 tokens per turn).

| Model | | Prefill | Decode (turns 1 / 2 / 3) | Peak wired memory |
|---|---|---|---|---|
| Qwen3.6-35B-A3B | this fork | **253 tok/s** | **48.5 / 46.2 / 45.9** | **34.6 GB** |
|  | upstream 2.12.2 | 147 tok/s | 11.5 / 13.2 / 12.6 | 49.6 GB |
| Qwen3.8-27B | this fork | **80 tok/s** | **20.9 / 21.0 / 18.2** | **44.4 GB** |
|  | upstream 2.12.2 | stopped out of memory in turn 1 | — | 58.4 GB |

### Conditions

- Qwen3.6-35B-A3B and Qwen3.8-27B were measured on v2.12.2-m1.10 (2026-10-20); the other rows are from v2.12.2-m1.8.
- v2.12.2-m1.11 fixes the prefill attention over long histories, whose running sum was kept in fp16: on the 27B most cold prompts of 192K–256K ended with NaN, and from 32K on the answers of both models change a little. Prefill speed is unchanged; in bench 3 the decode at 32K–128K moved by a few percent, so the tables above were not re-measured. Bench 3 of [apple-silicon-llm-bench](https://github.com/shunya1810/apple-silicon-llm-bench) has 192K–256K after the fix (27B: prefill 78–90 tok/s, decode 17–21 tok/s).
- v2.12.2-m1.12 leaves these models' speed unchanged: against m1.11 on Qwen3.8-27B and Qwen3.6-35B-A3B, prefill and the time of one verify round at 1.6K tokens differed by less than 2%, so the tables were not re-measured.
- The 8K–128K prompts repeat one 11K-token HTML file. Read to about 256K (22 repeats), Qwen3.8-27B predicts its own next tokens poorly from the third repeat on (upstream MTPLX too), and the draft head, which reads the model's hidden states, then rarely matches, so 256K is shown with the non-repeating text above instead.
- MacBook Pro M1 Max 64 GB, macOS 26. Profile `turbo`, 256K context limit, 8-bit KV cache, temperature 0.6 with fixed seeds.
- Both engines use the MTP depth this fork runs fastest at (listed in [Models](#models)).
- The server was restarted for each length, so every prompt is read cold. Items measured more than once show the faster run (some runs came out slower, for example with a hot GPU).
- Upstream's 32K decode on Ornith-1.5-35B-A3B (86.8 tok/s) is higher than its other lengths because the answer quoted the file at length and upstream's context-copy copied it.
- This fork's peak on the Qwen3.6-35B-A3B family (32.6 GB) is reached while the model loads: the MoE weights are repacked then, which needs memory for a while. While answering it uses 21–24 GB (24.4 GB at 128K, against 26.2 GB on upstream).
- Swift-1.5-Qwen3.8-27b decodes more slowly at 128K than Huihui-Qwen3.8-27B-abliterated, which has the same layout (16.9 against 23.3 tok/s). Repeated runs agree; the cause is still being investigated.

## Models

### Measured models

| Model | Type | Quantization | Weights | MTP depth |
|---|---|---|---|---|
| [Youssofal/Qwen3.6-35B-A3B-MTPLX-Optimized-Speed-FP16](https://huggingface.co/Youssofal/Qwen3.6-35B-A3B-MTPLX-Optimized-Speed-FP16) | MoE (35B total, 3B per token) | 4-bit (groups of 64), some 8-bit | 21 GB | 1 |
| Ornith-1.5-35B-A3B ([ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) converted with MTPLX forge) | MoE (a Qwen3.6-35B-A3B derivative) | 4-bit (groups of 64), some 8-bit | 21 GB | 1 |
| [Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed-FP16](https://huggingface.co/Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed-FP16) | dense (27B) | 4-bit (groups of 32), some 8-bit | 20 GB | 3 |
| Huihui-Qwen3.8-27B-abliterated ([huihui-ai/Huihui-Qwen3.8-27B-abliterated](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-27B-abliterated) converted with MTPLX forge) | dense (27B) | 4-bit (groups of 64) | 16 GB | 3 |
| Swift-1.5-Qwen3.8-27b ([ukisai/Swift-1.5-Qwen3.8-27b](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27b) converted with MTPLX forge) | dense (27B, a Qwen3.8-27B derivative) | 4-bit (groups of 64) | 16 GB | 3 |

- 4-bit weights carry one set of scales per group of 32 or 64 values. Groups of 32 are more precise, but the weights are larger and decode is slower.
- [Youssofal/Qwen3.8-27B-MTPLX-Optimized-Quality-FP16](https://huggingface.co/Youssofal/Qwen3.8-27B-MTPLX-Optimized-Quality-FP16) (8-bit) also runs; its weights are about 9 GB larger and it decodes 10–20% slower than Speed.

### Other models

- Qwen3.5, 3.6 and 3.8 family models launch on `turbo` on M1 when no profile is given. The family is read from `model_type` in `config.json`, so fine-tunes and packs you forge yourself are included.
- Models with the same layout as Qwen3.6-35B-A3B (256 experts, 8 per token, hidden size 2048, 40 layers, 4-bit body in groups of 64) also get the A3B optimizations: decode rounds queued on the GPU back to back, context-copy for file rewrites, and more.
- `turbo`'s verify kernels cover 4-bit, 6-bit and 8-bit affine quantization. Other quantizations (3-bit, mxfp4, ternary and so on) compute those matmuls on the stock path.
- Models of other families launch on `sustained`, as upstream. Choosing `turbo` with `--profile turbo` or the profile setting in the app may enable the M1 changes for them; that combination is not measured.
- [Re-measure the MTP depth](#re-measure-the-mtp-depth) finds each model's fastest depth.

## Requirements

- An M1-family Mac (measured: M1 Max 64 GB, macOS 26)
- Xcode with the Metal toolchain (`xcrun metal` must work)
- The [MTPLX app](https://github.com/youssofal/MTPLX), opened once. Download models from the app's model screen.

## Install

```bash
git clone --branch v2.12.2-m1.12 https://github.com/shunya1810/MTPLX.git MTPLX-m1
cd MTPLX-m1
scripts/build-m1-native-extensions.sh   # about a minute; done when it prints "mfa_attn: ok", "ane_ffn: ok" and "qsa_kernels: ok"
```

The server also runs without the native extensions (the Neural Engine prefill and the M1 attention), with a slower prefill.

### Use it from the app

Switch the app's engine to this fork:

```bash
scripts/m1-app-use-checkout.sh install
```

Quit and reopen the app; it then runs this fork's engine, however it is opened (Dock, Finder and so on). The setting survives logging out. To go back, run `scripts/m1-app-use-checkout.sh uninstall` and reopen the app.

### Run it as a server

Quit the MTPLX app first (two engines do not fit in memory side by side), then start the server. Pass the folder of a model downloaded in the app to `--model`.

```bash
PYTHONPATH="$PWD" "$HOME/Library/Application Support/MTPLX/runtime-venv/bin/python" -m mtplx.cli serve \
  --model "$HOME/.mtplx/models/Youssofal--Qwen3.6-35B-A3B-MTPLX-Optimized-Speed-FP16" \
  --context-window 262144 --paged-kv-quantization q8
```

The server listens on `http://127.0.0.1:8000`. Check that it answers:

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model": "mtplx", "messages": [{"role": "user", "content": "Say hello."}], "max_tokens": 64}'
```

More: [docs/m1max-longctx/USAGE.md](docs/m1max-longctx/USAGE.md).

### Update to a new release

```bash
git fetch --tags
git checkout <new tag>                  # for example v2.12.2-m1.12
scripts/build-m1-native-extensions.sh   # needed only when the native extensions changed (the release notes say so)
```

If you use it from the app, reopen the app after updating.

## Use it with a coding agent

Any client that speaks the OpenAI or Anthropic API works.

- OpenCode: add a provider to `~/.config/opencode/opencode.json`:

  ```json
  {
    "provider": {
      "mtplx": {
        "npm": "@ai-sdk/openai-compatible",
        "name": "MTPLX (M1)",
        "options": { "baseURL": "http://127.0.0.1:8000/v1", "apiKey": "local", "timeout": false },
        "models": { "mtplx": { "name": "MTPLX", "reasoning": true, "tool_call": true,
                               "limit": { "context": 262144, "output": 32768 } } }
      }
    }
  }
  ```

- Claude Code: `ANTHROPIC_BASE_URL=http://127.0.0.1:8000 claude`

Qwen3.6 family models have no reasoning-effort levels: thinking is on or off. Qwen3.8 family models take `low` to `xhigh`.

## Re-measure the MTP depth

The MTP depth (how many tokens one draft looks ahead) that runs fastest differs between models. This script serves the model at depth 1, 2 and 3 in turn, measures decode speed and reports the fastest depth. It takes about 15 minutes per model.

```bash
scripts/m1-depth-sweep.py --model "$HOME/.mtplx/models/<model folder>"
```

`--apply-app` writes the result into the app's per-model setting. The app writes its settings back when it quits, so run it with the app closed.

## What this fork changes

- Prefill: part of the work runs on the Neural Engine alongside the GPU. Attention over the cached history uses flash-attention tuned for M1. Short additions to a long history (tool results and the like) read the 8-bit KV in place.
- Decode: fewer CPU–GPU round trips per round. On Qwen3.6-35B-A3B family models, the next round is queued on the GPU before the CPU reads the previous one. File rewrites copy the parts that match the prompt (context-copy, exact matches only).
- Memory: the KV cache is kept in 8-bit pages, and long prompts are prefilled straight into them (on dense models, prompts over 192K tokens are read with fp16 KV and repacked). Memory estimates follow values measured on this fork. A 256K-token conversation runs at the default settings on a 64 GB Mac.
- Defaults: Qwen3.5, 3.6 and 3.8 family models launch on `turbo`. A Qwen3.6-35B-A3B family model started from the app also takes the A3B path.
- Qwen4 groundwork (not yet tested on a real Qwen4 model): upstream's `qwen4_exp` path is written for Qwen3.8-Flash-Next (hidden size 2560, MoE, 24 query / 2 KV heads). In case a Qwen4 model uses this architecture at another size, it also takes dense layers, other widths and other head layouts. On M1, bf16 packs of this family run in fp16.
- Agent sessions (on any Mac): finding the same conversation again across history rewrites, keeping the cache when thinking is switched on or off, keeping whitespace in tool arguments, retrying a request when macOS stops a long GPU command, resuming from the deepest saved state, and more.

Every change with its measurements: [CHANGES.ja.md](docs/m1max-longctx/CHANGES.ja.md) (Japanese) and the [release notes](https://github.com/shunya1810/MTPLX/releases).

## Switches

Each M1 default can be turned off with an environment variable; `MTPLX_M1_LONG_CONTEXT=0` turns all of them off.

| Variable | Default on M1 | Turns off |
|---|---|---|
| `MTPLX_ANE_PREFILL=0` / `MTPLX_ANE_PREFILL_MOE=0` | on | the Neural Engine prefill (dense / MoE) |
| `MTPLX_MFA_PREFILL=0` | on | the flash-attention prefill |
| `MTPLX_QWEN4_SAMPLED_DRAFT_CHAIN=0` | on | building the draft on the GPU without host round trips |
| `MTPLX_M1_PAGED_PREFILL=0` | on from 32K | prefill straight into 8-bit pages |
| `MTPLX_M1_ENGINE_RAM_FRACTION=0.75` | 13/16 of memory | the engine's memory cap (set a smaller value) |
| `MTPLX_M1_TURBO_DEFAULT=0` | `turbo` for Qwen3.5, 3.6 and 3.8 families | the `turbo` default |
| `MTPLX_M1_FP16_CAST=0` | on for the `qwen4_exp` family | running a bf16 pack in fp16 (`=1` turns it on for other families; no effect on fp16 packs) |
| `MTPLX_M1_MOE_LANE=0` | on for MoE models | every MoE default (the two rows below included) |
| `MTPLX_M1_RESIDENT_DECODE=0` | on for Qwen3.6-35B-A3B family models | decode rounds queued on the GPU |
| `MTPLX_CONTEXT_COPY=0` | on for Qwen3.6-35B-A3B family models | context-copy |

## Known limitations

- Measured on one M1 Max 64 GB. M1 Pro, M1 Ultra and other memory sizes run the same code but are untested.
- The Neural Engine path uses Apple's private `AppleNeuralEngine.framework` and may stop working after a macOS update; prefill then falls back to the GPU.
- The Neural Engine prefill computes in int8, so it is not bit-identical to the GPU prefill (on Qwen3.6-35B-A3B, 98.25% of next tokens agree over 24,576 positions). `MTPLX_ANE_PREFILL=0` / `MTPLX_ANE_PREFILL_MOE=0` keep prefill on the GPU.
- Thinking models sometimes reason for a very long time on hard agent tasks. Reasoning length is not capped by default; on requests that carry tools, `MTPLX_THINKING_BUDGET=<tokens>` caps it.
- Some upstream tests assume upstream's defaults and fail with the M1 defaults on; most pass with `MTPLX_M1_LONG_CONTEXT=0`.

## License and credits

Apache-2.0, as upstream ([LICENSE](LICENSE), [NOTICE](NOTICE)). The Neural Engine client and FFN program layout are adapted from [Splash](https://github.com/incoai/splash) (`native_extensions/ane_ffn/NOTICE`); the flash-attention kernel follows [metal-flash-attention](https://github.com/philipturner/metal-flash-attention) (`native_extensions/mfa_attn/MFA_LICENSE.txt`).
