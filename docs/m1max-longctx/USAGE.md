# Running a server from this fork (m1max-longctx)

[日本語](USAGE.ja.md)

The `m1max-longctx` branch adds long-context changes for the M1 Max on top of upstream MTPLX (changes and measurements: [benchmark](../benchmarks/m1max-longctx/README.md)).
This page shows how to start this branch's server locally and use it.

## Relation to the MTPLX app

- The MTPLX app (`/Applications/MTPLX.app`) runs the upstream release bundled with the app, in its own Python environment (`~/Library/Application Support/MTPLX/runtime-venv`). **The app does not use this branch.**
- The app cannot attach to a server started outside it. If its configured port is taken, it starts its own server on another port.
- Swapping the app's runtime for this branch has not been tested, and the app may reinstall its bundled version, so it is not recommended.

To use this branch, quit the app and start the server as below. The server has its own browser chat page and an OpenAI-compatible API, so the app is not needed.
To go back to the app, stop this server and open the app.

## Requirements

- An Apple Silicon Mac (measured on an M1 Max 64GB, macOS 26)
- The MTPLX app, opened once so its runtime (mlx and the other dependencies) and the model are installed. The steps below use the app's runtime Python (as every measurement did)
- Model: `~/.mtplx/models/Youssofal--Qwen3.8-27B-MTPLX-Optimized-Speed-FP16` (downloaded from the app's model screen)

Installing the dependencies into your own venv has not been tested.

## Start

```bash
git clone -b m1max-longctx https://github.com/shunya1810/MTPLX.git
cd MTPLX
```

Quit the MTPLX app if it is running (the server uses the GPU and about 20 GB of memory; two cannot run side by side).

```bash
PYTHONPATH="$PWD" "$HOME/Library/Application Support/MTPLX/runtime-venv/bin/python" -m mtplx.cli serve \
  --model "$HOME/.mtplx/models/Youssofal--Qwen3.8-27B-MTPLX-Optimized-Speed-FP16" \
  --profile turbo --depth 3 --host 127.0.0.1 --port 8000 \
  --context-window 262144 --no-auth --yes
```

- `PYTHONPATH="$PWD"` makes the app's runtime Python load `mtplx` from this checkout
- The server is ready when it prints `MTPLX is ready.` (the model loads in ten seconds to a minute)
- Stop it with Ctrl-C

Main options:

| Option | Default | Notes |
|---|---|---|
| `--kv-quant` | unset (`auto` on the M1 family) | `auto`: q8 KV only for prompts of 131,072 tokens or more. An explicit `off` / `q8` / `q4` is used as given |
| `--context-window` | from the model | `262144` to use up to 256K |
| `--ssd-session-cache` | `on` | also saves the conversation cache to the SSD, so a restarted server resumes where it left off |
| `--max-tokens` | server default | cap per response |
| `--port` | 8000 | |

## Use

- Browser chat: <http://127.0.0.1:8000/>
- OpenAI-compatible API: `http://127.0.0.1:8000/v1` (no API key, since `--no-auth` is set; the server listens on localhost only)

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model": "mtplx", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 128}'
```

`mtplx.cli connect` prints settings for Open WebUI, OpenCode, Claude Code and others (run it with the same `PYTHONPATH`).

## What to expect with long contexts (M1 Max 64GB, measured)

- The first turn prefills the whole prompt: about 25 minutes at 128K and 75 minutes at 256K
- Later turns of the same conversation restore from the cache in seconds (128K about 2 s, 256K about 3.4 s) and prefill only the new tokens
- Peak memory on a 256K second turn is about 50 GB; other memory-heavy apps can push the Mac into swap
