# Running a server from this fork (m1max-longctx)

[日本語](USAGE.ja.md)

The `m1max-longctx` branch adds long-context changes for the M1 Max on top of upstream MTPLX (changes and measurements: [benchmark](../benchmarks/m1max-longctx/README.md)).
This page shows how to start this branch's server locally and use it.

## Two ways to use it

- **In the MTPLX app**: the app's developer hook (a source checkout as the engine) runs this branch behind the app's own UI (see "Run it in the app")
- **Server only**: start the server from the command line and use its own chat page or the OpenAI-compatible API (see "Server only")

Both use the Python of the app's runtime (`~/Library/Application Support/MTPLX/runtime-venv`) and load `mtplx` from this checkout.
Opening the app normally still runs the upstream release bundled with the app.

## Requirements

- An Apple Silicon Mac (measured on an M1 Max 64GB, macOS 26)
- The MTPLX app, opened once so its runtime (mlx and the other dependencies) and the model are installed. The steps below use the app's runtime Python (as every measurement did)
- Model: `~/.mtplx/models/Youssofal--Qwen3.8-27B-MTPLX-Optimized-Speed-FP16` (downloaded from the app's model screen)

Installing the dependencies into your own venv has not been tested.

```bash
git clone -b m1max-longctx https://github.com/shunya1810/MTPLX.git
cd MTPLX
```

Run the commands below from this `MTPLX` directory.

## Run it in the app

Given `MTPLX_APP_ALLOW_SOURCE_WRAPPER=1` and `MTPLX_APP_SOURCE_WRAPPER_PATH=<checkout>/bin/mtplx`, the app starts its engine through that wrapper ahead of its own runtime (`apps/MTPLXApp/Sources/MTPLXAppCore/Services/MTPLXCommandBuilder.swift`).
`bin/mtplx` runs this checkout's `mtplx` with the Python named by `MTPLX_RUNTIME_VENV_PY` in `~/Library/Application Support/MTPLX/runtime.env`.

[`scripts/open-mtplx-app-with-checkout.command`](../../scripts/open-mtplx-app-with-checkout.command) does all of it (it also runs from Finder with a double click):

1. writes `runtime.env` pointing at the app runtime's Python, if it does not exist
2. quits the app if it is running and stops any MTPLX server left behind (the app adopts a server from its previous run)
3. opens the app with the two variables

Then start the engine in the app. By hand:

```bash
printf 'MTPLX_RUNTIME_VENV_PY="%s"\n' "$HOME/Library/Application Support/MTPLX/runtime-venv/bin/python" \
  > "$HOME/Library/Application Support/MTPLX/runtime.env"
open -a /Applications/MTPLX.app \
  --env MTPLX_APP_ALLOW_SOURCE_WRAPPER=1 \
  --env MTPLX_APP_SOURCE_WRAPPER_PATH="$PWD/bin/mtplx"
```

To check: the engine process carries a `PYTHONPATH` pointing at this checkout when it runs this branch.

```bash
ps eww -p "$(pgrep -f mtplx.server.openai | head -1)" -o command= | grep -o 'PYTHONPATH=[^ ]*'
```

Notes:

- The app's settings (context window, SSD cache, KV quantization, ...) apply as usual. For 128K / 256K, raise the context window to 262,144 and turn the SSD cache on
- KV quantization: the app offers off / q8 / q4, no `auto`. The app passes nothing to the engine for off, so with this branch off becomes `auto` on the M1 family (q8 only for prompts of 131,072 tokens or more); read from the app's code, not tried from the UI. q8 means q8 at every length
- The engine runs the checkout's working tree: switching branches changes what the next engine start runs
- To go back, quit the app, delete `runtime.env`, and open the app normally

## Server only

Quit the MTPLX app if it is running (an engine uses the GPU and about 20 GB of memory; two cannot run side by side).

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
