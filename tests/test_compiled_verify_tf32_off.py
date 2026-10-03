"""The compiled-verify parity tests once more with MLX_ENABLE_TF32=0.

On a Metal 4 tensor-unit GPU (M5) MLX runs float32 matrix multiplies on the
tensor units with TF32 inputs by default. That rounding hides a compiled
verifier that differs from the eager one in the low bits of a float32 value
(a weight replayed from a lazy graph, a fused sigmoid, a constant printed with
7 digits), while the M1 to M4 kernels read all 23 mantissa bits and fail. With
TF32 off, an M5 multiplies float32 in full precision as they do, so this arm
catches that class on the release machine. The tests run in a subprocess
because MLX reads the variable once per process.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import mlx.core as mx
import pytest

ROOT = Path(__file__).resolve().parents[1]
PARITY_TESTS = (
    "tests/test_graphbank_compiled_verify.py",
    "tests/test_ccopy_bank_route.py",
    "tests/test_dense_mrope_compiled_route.py",
    "tests/test_compiled_verify_evaluated_weights.py",
    "tests/test_attention_gate_exactness.py",
    "tests/test_qsa_fixed_bank_score_scale.py",
    "tests/test_qwen4_ple_gate_exactness.py",
    "tests/test_float32_operand.py",
    "tests/test_qwen4_yarn_amplitude.py",
    "tests/test_qwen4_fixed_m4_verify_exactness.py",
    "tests/test_qwen4_fixed_m4_float32_admission.py",
    "tests/test_qwen4_rows_gather_scale.py",
    "tests/test_qwen4_verify_gate_sites.py",
)


@pytest.mark.skipif(not mx.metal.is_available(), reason="TF32 is a Metal GPU setting")
@pytest.mark.skipif(
    os.environ.get("MLX_ENABLE_TF32") == "0", reason="this process is the TF32-off arm already"
)
def test_compiled_verify_parity_holds_with_tf32_off():
    env = dict(os.environ, MLX_ENABLE_TF32="0")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "-W",
            "ignore::DeprecationWarning",
            *PARITY_TESTS,
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    tail = "\n".join(result.stdout.splitlines()[-40:])
    assert result.returncode == 0, f"{tail}\n{result.stderr[-2000:]}"
