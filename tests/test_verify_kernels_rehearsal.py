"""The verify-width kernels under the M1-M4 rehearsal switch, in a child process.

MLX reads ``MLX_METAL_GPU_ARCH`` once, when it creates its Metal device, so a
process that has already touched the GPU cannot change it. The child sets
``MLX_METAL_GPU_ARCH=applegpu_g16s`` and ``MTPLX_FORCE_GPU_FAMILY_FALLBACK=1``
(MLX's kernels without the tensor units plus our own fallback routes: the M4
path, on this Mac) and proves each kernel there against its stock parent, on
bit patterns:

* the hyper-connection verify read installs at widths 2 to 8 (its probe
  compares every width with the compiled stock chain);
* the QSA verify selection installs at widths 1 to 8, both lanes;
* the fixed bank's pooled-key row installs (its probe compares the bank);
* the one-dispatch routed-down tail equals its two-dispatch parent.

Every threadgroup these kernels launch is at most 256 threads (issue #400:
M1 and M2 cap some pipelines below 1,024), which the child also checks.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import mlx.core as mx
import pytest

ROOT = Path(__file__).resolve().parents[1]

def _physical_architecture() -> str:
    from mtplx.nax_detect import gpu_architecture

    return str(gpu_architecture() or "")


# The rehearsal is an M5 exercise: MLX_METAL_GPU_ARCH=applegpu_g16s runs the
# library's generation-16 kernels on generation-17 hardware. On any other GPU
# the tests of each kernel run on the device's own route instead.
pytestmark = [
    pytest.mark.skipif(not mx.metal.is_available(), reason="the rehearsal runs Metal kernels"),
    pytest.mark.skipif(
        os.environ.get("MLX_METAL_GPU_ARCH") is not None
        or not _physical_architecture().startswith("applegpu_g17"),
        reason="the M1-M4 rehearsal switch is exercised from an M5 (generation 17) process",
    ),
]

CHILD = r"""
import json
from types import SimpleNamespace

import mlx.core as mx
import numpy as np

from mtplx.kernels import hc_verify_read, qsa_verify_select
from mtplx.kernels import qwen4_m4_routed_down as down
from mtplx.models.qwen4_exp import GatedResidual, QSAIndexer, TextArgs
from mtplx.nax_detect import gpu_architecture

out = {"arch": gpu_architecture()}


def residual(combine, seed):
    args = SimpleNamespace(hc_count=4, hidden_size=2560, hc_lowrank=320, rms_norm_eps=1e-6)
    module = GatedResidual(args, use_combine=combine)
    keys = mx.random.split(mx.random.key(seed), 4)
    module.hc_norm.weight = (1.0 + 0.1 * mx.random.normal((10240,), key=keys[0])).astype(mx.bfloat16)
    module.input_mix_weight_down.weight = (0.01 * mx.random.normal((320, 10240), key=keys[1])).astype(mx.bfloat16)
    module.input_mix_weight_up.weight = (0.06 * mx.random.normal((10240, 320), key=keys[2])).astype(mx.bfloat16)
    if combine:
        module.block_inject_weight.weight = (0.01 * mx.random.normal((4, 10240), key=keys[3])).astype(mx.bfloat16)
    mx.eval(module.parameters())
    return module


model = SimpleNamespace(
    layers=[SimpleNamespace(attn_hyper_connection=residual(True, 1))],
    hyper_connection_mixer=residual(False, 2),
)
report = hc_verify_read.install(model, rows=(2, 3, 4, 5, 6, 7, 8))
out["hc"] = {"installed": report["installed"], "reason": report["disabled_reason"], "cases": report["probe_cases"]}

args = TextArgs.from_dict(
    {
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "vocab_size": 128,
        "layer_types": ["full_attention"] * 2,
        "rope_parameters": {"partial_rotary_factor": 0.25, "rope_theta": 10000000, "rope_type": "default"},
        "indexer_n_heads": 4,
        "indexer_kv_heads": 1,
        "indexer_head_dim": 128,
        "indexer_budget": 2048,
        "indexer_compress_ratio": 4,
    }
)
indexer = QSAIndexer(args)
report = qsa_verify_select.install(
    SimpleNamespace(layers=[SimpleNamespace(self_attn=SimpleNamespace(indexer=indexer))]),
    rows=(1, 2, 3, 4, 5, 6, 7, 8),
)
out["qsa"] = {"installed": report["installed"], "reason": report["disabled_reason"], "cases": report["probe_cases"]}

from mtplx.kernels import qsa_pooled_row

pooled_args = TextArgs.from_dict(
    {
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "vocab_size": 128,
        "layer_types": ["full_attention"] * 2,
        "rope_parameters": {"partial_rotary_factor": 0.25, "rope_theta": 10000000, "rope_type": "default"},
        "indexer_n_heads": 4,
        "indexer_kv_heads": 1,
        "indexer_head_dim": 128,
        "indexer_budget": 2048,
        "indexer_compress_ratio": 4,
    }
)
pooled_indexer = QSAIndexer(pooled_args)
pooled_indexer.k_layernorm.weight = pooled_indexer.k_layernorm.weight.astype(mx.bfloat16)
mx.eval(pooled_indexer.parameters(), pooled_indexer._inv_freq)
report = qsa_pooled_row.install(
    SimpleNamespace(layers=[SimpleNamespace(self_attn=SimpleNamespace(indexer=pooled_indexer))])
)
out["pooled"] = {"installed": report["installed"], "reason": report["disabled_reason"], "cases": report["probe_cases"]}

experts, hidden, inter, rows, top_k = 16, 2560, 640, 4, 10
keys = mx.random.split(mx.random.key(31), 8)
weights = mx.random.randint(-(2**31), 2**31 - 1, (experts, hidden, inter // 8), dtype=mx.int32, key=keys[0]).view(mx.uint32)
scales = (0.01 + 0.002 * mx.random.normal((experts, hidden, inter // 32), key=keys[1])).astype(mx.bfloat16)
biases = (-0.08 + 0.002 * mx.random.normal((experts, hidden, inter // 32), key=keys[2])).astype(mx.bfloat16)
routed_h = (0.3 * mx.random.normal((rows, top_k, inter), key=keys[3])).astype(mx.bfloat16)
shared_down = (0.2 * mx.random.normal((rows, hidden), key=keys[4])).astype(mx.bfloat16)
shared_factor = mx.random.uniform(0, 1, (rows,), key=keys[5]).astype(mx.bfloat16)
hyper = mx.random.normal((1, rows, 4 * hidden), key=keys[6]).astype(mx.bfloat16)
inject = mx.random.uniform(-2, 2, (1, rows, 4), key=keys[7]).astype(mx.bfloat16)
rng = np.random.default_rng(5)
ids = mx.array(np.stack([rng.choice(experts, top_k, replace=False) for _ in range(rows)]).astype(np.uint32))
scores = rng.random((rows, top_k)).astype(np.float32)
scores = mx.array(scores / scores.sum(axis=1, keepdims=True)).astype(mx.bfloat16)
call = (routed_h, weights, scales, biases, ids, scores, shared_down, shared_factor, hyper, inject)
want = down.bind_residual_tail_two_dispatch()(*call)
got = down.bind_residual_tail()(*call)
mx.eval(want, got)
out["down_mismatches"] = int((np.array(want.view(mx.uint16)) != np.array(got.view(mx.uint16))).sum())
out["down_threads"] = int(down.parallel_launch_geometry()[1][0])
print(json.dumps(out))
"""


def test_the_verify_kernels_hold_under_the_rehearsal_switch():
    env = dict(os.environ)
    env.update(
        MLX_METAL_GPU_ARCH="applegpu_g16s",
        MTPLX_FORCE_GPU_FAMILY_FALLBACK="1",
        MTPLX_SSD_SESSION_CACHE="off",
    )
    env.pop("MTPLX_QWEN4_HC_VERIFY_READ", None)
    env.pop("MTPLX_QWEN4_QSA_VERIFY_SELECT", None)
    env.pop("MTPLX_QWEN4_QSA_POOLED_ROW", None)
    proc = subprocess.run(
        [sys.executable, "-c", CHILD],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert proc.returncode == 0, proc.stderr[-4000:]
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert result["arch"] == "applegpu_g16s"
    assert result["hc"]["installed"], result["hc"]["reason"]
    assert result["hc"]["cases"] >= 1 + 3 * 7
    assert result["qsa"]["installed"], result["qsa"]["reason"]
    assert result["qsa"]["cases"] >= 1 + 6 * 8
    assert result["pooled"]["installed"], result["pooled"]["reason"]
    assert result["pooled"]["cases"] >= 12
    assert result["down_mismatches"] == 0
    assert result["down_threads"] <= 256
