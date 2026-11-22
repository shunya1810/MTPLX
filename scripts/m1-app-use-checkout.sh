#!/bin/zsh
# Make the MTPLX app run this checkout's engine however the app is opened (Dock, Finder, Spotlight).
#
#   scripts/m1-app-use-checkout.sh install     # from now on, and after every login
#   scripts/m1-app-use-checkout.sh uninstall   # back to the app's bundled engine
#   scripts/m1-app-use-checkout.sh status
#
# The app reads two environment variables when it starts its engine:
# MTPLX_APP_ALLOW_SOURCE_WRAPPER=1 and MTPLX_APP_SOURCE_WRAPPER_PATH=<checkout>/bin/mtplx.
# `install` sets them for the login session with `launchctl setenv` and adds a LaunchAgent that sets
# them again at every login. Quit and reopen the app afterwards: a running app keeps its old environment.
set -euo pipefail

repo="${0:A:h:h}"
label="io.github.shunya1810.mtplx-m1-checkout"
plist="$HOME/Library/LaunchAgents/$label.plist"
# MTPLX_M1_APP_WRAPPER: another engine wrapper (scripts/install-m1.sh --auto-update writes one that
# follows each release); default this checkout's bin/mtplx.
wrapper="${MTPLX_M1_APP_WRAPPER:-$repo/bin/mtplx}"

case "${1:-status}" in
  install)
    [[ -x "$wrapper" ]] || { echo "not found: $wrapper" >&2; exit 1; }
    mkdir -p "${plist:h}"
    cat > "$plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$label</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/sh</string><string>-c</string>
    <string>launchctl setenv MTPLX_APP_ALLOW_SOURCE_WRAPPER 1; launchctl setenv MTPLX_APP_SOURCE_WRAPPER_PATH '$wrapper'</string>
  </array>
  <key>RunAtLoad</key><true/>
</dict>
</plist>
PLIST
    launchctl bootout "gui/$(id -u)" "$plist" 2>/dev/null || true
    launchctl bootstrap "gui/$(id -u)" "$plist"
    launchctl setenv MTPLX_APP_ALLOW_SOURCE_WRAPPER 1
    launchctl setenv MTPLX_APP_SOURCE_WRAPPER_PATH "$wrapper"
    echo "installed: the MTPLX app now runs $wrapper (quit and reopen the app)"
    ;;
  uninstall)
    launchctl bootout "gui/$(id -u)" "$plist" 2>/dev/null || true
    rm -f "$plist"
    launchctl unsetenv MTPLX_APP_ALLOW_SOURCE_WRAPPER
    launchctl unsetenv MTPLX_APP_SOURCE_WRAPPER_PATH
    echo "uninstalled: the MTPLX app uses its bundled engine again (quit and reopen the app)"
    ;;
  status)
    echo "MTPLX_APP_ALLOW_SOURCE_WRAPPER=$(launchctl getenv MTPLX_APP_ALLOW_SOURCE_WRAPPER)"
    echo "MTPLX_APP_SOURCE_WRAPPER_PATH=$(launchctl getenv MTPLX_APP_SOURCE_WRAPPER_PATH)"
    [[ -f "$plist" ]] && echo "LaunchAgent: $plist" || echo "LaunchAgent: not installed"
    ;;
  *)
    echo "usage: $0 install|uninstall|status" >&2; exit 2
    ;;
esac
