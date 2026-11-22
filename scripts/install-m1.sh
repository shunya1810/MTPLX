#!/bin/zsh
# Install or update MTPLX for M1 Macs (this fork) and, if asked, make the MTPLX app run it.
#
#   curl -fsSL https://raw.githubusercontent.com/shunya1810/MTPLX/m1max-longctx/scripts/install-m1.sh | zsh -s -- --app --auto-update
#   scripts/install-m1.sh [--dir DIR] [--app] [--auto-update] [--no-build]
#   scripts/install-m1.sh --update-only        # what --auto-update runs before the app starts its engine
#   scripts/install-m1.sh --uninstall-app      # the app goes back to its bundled engine
#
# What it does:
#   1. checks for an Apple Silicon Mac, the MTPLX app's runtime (open the app once first) and, for the
#      build, Xcode's Metal toolchain;
#   2. clones the fork into DIR (default ~/MTPLX-m1), or, when run from a checkout, uses that checkout,
#      and moves it to the latest release (the m1max-longctx branch; a checkout with local changes is
#      left where it is);
#   3. builds the native extensions (about a minute) when they are missing, when their sources changed,
#      or when the app's MLX changed (an app update can bring a new one);
#   4. with --app, makes the MTPLX app run this checkout however it is opened; with --auto-update as
#      well, the app's engine moves to each new release when it starts (at most every 10 minutes).
#
# MTPLX_M1_REF (or the first line of ~/Library/Application Support/MTPLX/fork-app-ref) picks another
# ref than origin/m1max-longctx, for example a branch to try.

main() {
  setopt local_options err_exit no_unset pipe_fail
  local repo_url="https://github.com/shunya1810/MTPLX.git"
  local branch="m1max-longctx"
  local support="$HOME/Library/Application Support/MTPLX"
  local runtime_py="${MTPLX_RUNTIME_VENV_PY:-$support/runtime-venv/bin/python}"
  local dir="" app=0 auto=0 build=1 update_only=0 uninstall=0 quiet=0

  while (( $# )); do
    case "$1" in
      --dir) dir="$2"; shift 2 ;;
      --dir=*) dir="${1#--dir=}"; shift ;;
      --app) app=1; shift ;;
      --auto-update) auto=1; shift ;;
      --no-build) build=0; shift ;;
      --update-only) update_only=1; shift ;;
      --uninstall-app) uninstall=1; shift ;;
      --quiet) quiet=1; shift ;;
      -h|--help) sed -n '2,24p' "${0:A}" 2>/dev/null || true; return 0 ;;
      *) print -u2 "unknown option: $1"; return 2 ;;
    esac
  done
  say() { (( quiet )) || print -- "$@"; }
  ver() { git -C "$1" describe --tags --exact-match 2>/dev/null || git -C "$1" log -1 --format='%h, %cd' --date=short; }

  # A checkout of this fork that the script runs from, if any (piped through `zsh -s` it has none).
  local here="" repo
  [[ "${0:t}" == install-m1.sh ]] && here="${0:A:h:h}"
  if [[ -n "$here" && -d "$here/mtplx" && -e "$here/.git" ]]; then
    repo="$here"
  else
    repo="${dir:-$HOME/MTPLX-m1}"
  fi

  if (( uninstall )); then
    [[ -x "$repo/scripts/m1-app-use-checkout.sh" ]] && "$repo/scripts/m1-app-use-checkout.sh" uninstall
    rm -f "$support/mtplx-m1-app"
    return 0
  fi

  # 1. checks
  [[ "$(uname -m)" == arm64 ]] || { print -u2 "this fork needs an Apple Silicon Mac"; return 1; }
  local chip; chip=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || true)
  if [[ "$chip" != *M1* && "$chip" != *M2* ]] && (( ! update_only )); then
    say "note: $chip. This fork is tuned and measured on an M1 Max; on an M3 or newer, upstream MTPLX is the better choice."
  fi
  if [[ ! -x "$runtime_py" ]]; then
    print -u2 "The MTPLX app's runtime was not found ($runtime_py)."
    print -u2 "Install the MTPLX app (https://github.com/youssofal/MTPLX), open it once, then run this again."
    return 1
  fi
  if (( build )) && ! xcrun -f metal >/dev/null 2>&1; then
    print -u2 "Xcode's Metal toolchain was not found (xcrun metal). Install Xcode, then: xcodebuild -downloadComponent MetalToolchain"
    print -u2 "(or pass --no-build: the server runs without the native extensions, with a slower prefill)"
    return 1
  fi

  # 2. code
  local ref="${MTPLX_M1_REF:-}"
  [[ -z "$ref" && -f "$support/fork-app-ref" ]] && ref=$(head -1 "$support/fork-app-ref" | tr -d '[:space:]')
  ref="${ref:-origin/$branch}"
  if [[ ! -e "$repo/.git" ]]; then
    say "cloning into $repo ..."
    git clone -q --depth 1 --branch "$branch" "$repo_url" "$repo"
  fi
  local old new
  old=$(git -C "$repo" rev-parse HEAD)
  if [[ -n "$(git -C "$repo" status --porcelain --untracked-files=no)" ]]; then
    say "$repo has local changes: left at $(ver "$repo")"
  elif [[ "$ref" == origin/* ]]; then
    local depth=()
    [[ "$(git -C "$repo" rev-parse --is-shallow-repository)" == true ]] && depth=(--depth 1)
    if git -C "$repo" fetch -q "${depth[@]}" origin "${ref#origin/}" 2>/dev/null; then
      git -C "$repo" checkout -q --detach FETCH_HEAD
    else
      say "could not reach GitHub: staying at $(ver "$repo")"
    fi
  elif git -C "$repo" rev-parse -q --verify "$ref^{commit}" >/dev/null; then
    git -C "$repo" checkout -q --detach "$ref"
  elif git -C "$repo" fetch -q origin "$ref" 2>/dev/null; then
    git -C "$repo" checkout -q --detach FETCH_HEAD
  else
    say "ref $ref not found here or on GitHub: following origin/$branch"
    git -C "$repo" fetch -q origin "$branch" && git -C "$repo" checkout -q --detach FETCH_HEAD
  fi
  new=$(git -C "$repo" rev-parse HEAD)

  # 3. native extensions: missing, sources changed, or another MLX in the app's runtime
  if (( build )); then
    local mlx want stamp="$repo/.native-build-stamp"
    mlx=$("$runtime_py" -c 'import mlx.core as m; print(m.__version__)')
    want="mlx=$mlx src=$(git -C "$repo" rev-parse HEAD:native_extensions) script=$(git -C "$repo" rev-parse HEAD:scripts/build-m1-native-extensions.sh)"
    if [[ ! -f "$stamp" || "$(cat "$stamp")" != "$want" ]] \
       || ! ls "$repo"/native_extensions/mfa_attn/mtplx_mfa_attn/_ext*.so >/dev/null 2>&1; then
      say "building the native extensions (about a minute) ..."
      if (( quiet )); then
        "$repo/scripts/build-m1-native-extensions.sh" >/dev/null
      else
        "$repo/scripts/build-m1-native-extensions.sh"
      fi
      print -r -- "$want" > "$stamp"
    fi
  fi
  say "MTPLX for M1: $repo at $(ver "$repo")$([[ "$old" != "$new" ]] && print " (was ${old[1,8]})")"
  (( update_only )) && return 0

  # 4. the app
  if (( app )); then
    local wrapper="$repo/bin/mtplx"
    if (( auto )); then
      wrapper="${MTPLX_M1_APP_WRAPPER_DIR:-$support}/mtplx-m1-app"
      mkdir -p "${wrapper:h}"
      cat > "$wrapper" <<WRAPPER
#!/bin/zsh
# MTPLX app engine for MTPLX for M1 Macs (written by scripts/install-m1.sh --app --auto-update).
# Before the app starts its engine, move the checkout to the latest release (at most every 10 minutes).
repo='$repo'
case "\${1:-}" in
  serve|start|daemon)
    stamp="\$HOME/Library/Caches/MTPLX/mtplx-m1-last-update"
    if [[ ! -f "\$stamp" || -n \$(find "\$stamp" -mmin +10 2>/dev/null) ]]; then
      mkdir -p "\${stamp:h}"; touch "\$stamp"
      "\$repo/scripts/install-m1.sh" --update-only --quiet >>"\$HOME/Library/Caches/MTPLX/mtplx-m1-update.log" 2>&1 || true
    fi
    ;;
esac
export MTPLX_RUNTIME_SOURCE_SHADOW="\$repo/.no-runtime-shadow"
exec "\$repo/bin/mtplx" "\$@"
WRAPPER
      chmod +x "$wrapper"
    fi
    if [[ -n "${MTPLX_M1_NO_LAUNCHCTL:-}" ]]; then
      say "app wrapper written: $wrapper (MTPLX_M1_NO_LAUNCHCTL: not registered)"
    else
      MTPLX_M1_APP_WRAPPER="$wrapper" "$repo/scripts/m1-app-use-checkout.sh" install
    fi
  fi

  say ""
  say "Next:"
  (( app )) && say "  quit and reopen the MTPLX app; it now runs this fork$( (( auto )) && print ', and follows each new release')."
  say "  server:  $repo/bin/mtplx-m1 serve --model <a folder in ~/.mtplx/models>"
  say "  models:  download in the app, or $repo/bin/mtplx-m1 get inkling2 | flash-next"
  return 0
}
main "$@"; exit $?
