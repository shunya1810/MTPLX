#!/bin/zsh
# Build the M1 native extensions of this fork in place:
#   native_extensions/mfa_attn  metal-flash-attention prefill, command-buffer limits,
#                               in-place MTP KV append for the sampled draft chain
#   native_extensions/ane_ffn   Neural Engine prefill (FFN split between the GPU and the ANE)
#   native_extensions/qsa_kernels  Steel sparse attention for Qwen4-family (QSA) prefill at any
#                               head layout; the in-place build wins over the app runtime's copy
# Without them the server still runs: each feature falls back to the stock GPU path.
#
# Needs: the MTPLX app's runtime (~/Library/Application Support/MTPLX/runtime-venv),
# Xcode with the Metal toolchain (xcrun metal), internet access for pip.
set -eu
ROOT=${0:A:h:h}
RUNTIME=${MTPLX_RUNTIME_PYTHON:-"$HOME/Library/Application Support/MTPLX/runtime-venv/bin/python"}
VENV=${MTPLX_BUILD_VENV:-$ROOT/.build-venv}
MLX_VERSION=$("$RUNTIME" -c 'import mlx.core as m; print(m.__version__)')
# nanobind must be the version MLX itself was built with, or MLX arrays do not cross the boundary.
case $MLX_VERSION in
  0.32.*) NANOBIND=2.15.0 ;;
  *) echo "unknown MLX $MLX_VERSION: set MTPLX_NANOBIND_VERSION" >&2; NANOBIND=${MTPLX_NANOBIND_VERSION:?} ;;
esac
if [ ! -x "$VENV/bin/python" ]; then
  "$RUNTIME" -m venv "$VENV"
fi
"$VENV/bin/python" -m pip install -q "mlx==$MLX_VERSION" "nanobind==$NANOBIND" cmake setuptools
for ext in mfa_attn ane_ffn qsa_kernels; do
  cd "$ROOT/native_extensions/$ext"
  rm -rf build
  echo "building $ext ..."
  PATH="$VENV/bin:$PATH" "$VENV/bin/python" setup.py build_ext -j8 --inplace > "$ROOT/.build-$ext.log" 2>&1 \
    || { echo "build of $ext failed, see $ROOT/.build-$ext.log" >&2; exit 1; }
done
cd "$ROOT"
"$RUNTIME" -c "
import sys; sys.path.insert(0, '$ROOT')
from mtplx.kernels.mfa_prefill import _ext
e = _ext(); print('mfa_attn:', 'ok' if e is not None and hasattr(e, 'inplace_row_write') else 'NOT LOADED')
try:
    from mtplx import ane_prefill as ap
    r = ap._ext().unavailable(); print('ane_ffn:', 'ok' if not r else 'NOT AVAILABLE: ' + r)
except Exception as exc:
    print('ane_ffn: NOT LOADED', exc)
from mtplx.kernels import qsa_prefill_direct as q
print('qsa_kernels:', 'ok' if q._ext_max_gqa() else 'NOT LOADED (24/2-only or missing)', q.qsa_prefill_direct_build_info().get('built_against_mlx', ''))"
