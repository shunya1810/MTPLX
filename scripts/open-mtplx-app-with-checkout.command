#!/bin/zsh
# Open the MTPLX app with its engine running from the m1max-longctx checkout.
# Normal launches of the app keep using its bundled release.
set -eu
REPO="$(cd "$(dirname "$0")/.." && pwd)"
APP="/Applications/MTPLX.app"
RUNTIME_PY="$HOME/Library/Application Support/MTPLX/runtime-venv/bin/python"
RUNTIME_ENV="$HOME/Library/Application Support/MTPLX/runtime.env"

[[ -x "$REPO/bin/mtplx" ]] || { echo "not found: $REPO/bin/mtplx"; exit 1; }
[[ -x "$RUNTIME_PY" ]] || { echo "not found: $RUNTIME_PY (open the MTPLX app once first)"; exit 1; }
if [[ ! -f "$RUNTIME_ENV" ]]; then
  printf 'MTPLX_RUNTIME_VENV_PY="%s"\n' "$RUNTIME_PY" > "$RUNTIME_ENV"
fi

if pgrep -xq MTPLXApp; then
  osascript -e 'tell application id "com.youssofal.mtplx" to quit' || true
  for _ in {1..30}; do pgrep -xq MTPLXApp || break; sleep 1; done
fi
# The app adopts a daemon left from its previous run, which would keep the
# bundled engine; stop any MTPLX server so the app starts a fresh one.
if pgrep -f "mtplx.server.openai|mtplx.cli serve" >/dev/null; then
  echo "stopping the running MTPLX server"
  pkill -INT -f "mtplx.server.openai|mtplx.cli serve" || true
  for _ in {1..60}; do pgrep -f "mtplx.server.openai|mtplx.cli serve" >/dev/null || break; sleep 1; done
fi

echo "branch: $(git -C "$REPO" branch --show-current) @ $(git -C "$REPO" rev-parse --short HEAD)"
open -a "$APP" \
  --env MTPLX_APP_ALLOW_SOURCE_WRAPPER=1 \
  --env MTPLX_APP_SOURCE_WRAPPER_PATH="$REPO/bin/mtplx"
echo "MTPLX opened with the checkout engine. Start the engine in the app."
