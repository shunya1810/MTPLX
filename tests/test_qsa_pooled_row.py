"""The fixed QSA bank's pooled-key row in one kernel, against the compiled stock update.

``QSAIndexer._extend_pooled_fixed`` brings a fixed bank's pooled keys up to date
inside the compiled verifier: mean of the completed block's raw keys, the key
RMS norm, the rotation at the block's first position, a select on whether the
block completed, and the bank write. mtplx/kernels/qsa_pooled_row.py computes
the written row in one kernel. Every case builds the indexer at Flash-Next's
geometry (index head 128, compress ratio 4, a 64-wide partial rotation), traces
the stock update and the fused one inside a compiled step body, and compares
the whole updated bank as stored bits.
"""

from __future__ import annotations

import collections
import tempfile
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from mtplx.compile_state import compiled_step_body
from mtplx.kernels import qsa_pooled_row as row
from mtplx.models.qwen4_exp import QSAIndexer, TextArgs

pytestmark = pytest.mark.skipif(
    not mx.metal.is_available(), reason="the kernel and its compiled parent run on the GPU"
)

RATIO = 4
YARN = {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 16}


def _indexer(rope=None) -> QSAIndexer:
    args = TextArgs.from_dict(
        {
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "vocab_size": 128,
            "layer_types": ["full_attention"] * 2,
            "rope_parameters": {
                "partial_rotary_factor": 0.25,
                "rope_theta": 10000000,
                "rope_type": "default",
                **(rope or {}),
            },
            "indexer_n_heads": 4,
            "indexer_kv_heads": 1,
            "indexer_head_dim": 128,
            "indexer_budget": 2048,
            "indexer_compress_ratio": RATIO,
        }
    )
    built = QSAIndexer(args)
    built.k_layernorm.weight = (
        1.0 + 0.2 * mx.random.normal((128,), key=mx.random.key(3))
    ).astype(mx.bfloat16)
    mx.eval(built.parameters(), built._inv_freq)
    return built


def _model(indexer):
    return SimpleNamespace(layers=[SimpleNamespace(self_attn=SimpleNamespace(indexer=indexer))])


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    monkeypatch.delenv(row.ENV, raising=False)
    row.reset_for_tests()
    yield
    row.reset_for_tests()


@pytest.fixture(scope="module")
def indexer() -> QSAIndexer:
    return _indexer()


class _Bank:
    fixed_capacity = True

    def __init__(self, raw_keys, pooled, offset, rows, delta):
        self.raw_keys = raw_keys
        self.pooled = pooled
        self.offset = offset
        self._last_write_rows = rows
        self.rope_delta = delta


def _banks(blocks: int, seed: int, dtype=mx.bfloat16):
    keys = mx.random.split(mx.random.key(seed), 3)
    raw = (
        mx.random.normal((1, blocks * RATIO, 128), key=keys[0])
        * mx.exp(2.0 * mx.random.normal((1, blocks * RATIO, 128), key=keys[1]))
    ).astype(dtype)
    pooled = mx.random.normal((1, blocks, 128), key=keys[2]).astype(dtype)
    mx.eval(raw, pooled)
    return raw, pooled


def _update(indexer, raw, pooled, offset, rows, delta):
    def update(raw, pooled, offset, total):
        return indexer._extend_pooled_fixed(_Bank(raw, pooled, offset, rows, delta), total)

    with compiled_step_body():
        out = mx.compile(update)(
            raw, pooled, mx.array(offset, dtype=mx.int32), mx.array(offset + rows, dtype=mx.int32)
        )
    mx.eval(out)
    return out


def _bits(a: mx.array) -> np.ndarray:
    return np.array(mx.view(a, mx.uint16))


def _stock_then_fused(indexer, raw, pooled, offset, rows, delta):
    installed = dict(row._STATE)
    row._STATE["installed"] = False
    want = _update(indexer, raw, pooled, offset, rows, delta)
    row._STATE.update(installed)
    got = _update(indexer, raw, pooled, offset, rows, delta)
    return want, got


def test_install_proves_the_row(indexer):
    report = row.install(_model(indexer))
    assert report["installed"], report
    assert report["spelling"][0] in row.SUMS and report["spelling"][1] in row.ROTATIONS
    assert report["probe_cases"] >= 24
    assert report["traces"] == 0  # the probe's own traces are not engagements


def test_the_float32_stage_tells_the_spellings_apart(indexer):
    # At bfloat16 the final cast hides most one-ulp float32 differences, so
    # every spelling could pass a bfloat16-only probe. The float32 twin must
    # see the chosen spelling exact and at least one other spelling wrong.
    report = row.install(_model(indexer))
    assert report["installed"], report
    counts = report["float32_differing_values"]
    chosen = "{}/{}".format(*report["spelling"])
    assert counts[chosen] == 0, counts
    assert max(counts.values()) > 0, counts


@pytest.mark.parametrize(
    "offset, rows, delta",
    [
        (4 * 200 + 1, 1, None),  # inside a block: the bank keeps its row
        (4 * 300 - 1, 1, None),  # the last row of a block completes it
        (4 * 300, 4, None),  # a whole block in one window
        (4 * 411 - 2, 4, None),  # a window across a block boundary
        (4 * 411 - 2, 4, 13),  # the same with a rotary delta
        (4 * 511 - 4, 4, None),  # the last block of the bank
    ],
)
def test_every_case_matches_the_compiled_stock_update(indexer, offset, rows, delta):
    report = row.install(_model(indexer))
    assert report["installed"], report
    raw, pooled = _banks(512, seed=offset + (delta or 0))
    want, got = _stock_then_fused(indexer, raw, pooled, offset, rows, delta)
    np.testing.assert_array_equal(_bits(got), _bits(want))
    assert row.engagement()["traces"] >= 1


def test_a_yarn_amplitude_matches_the_compiled_stock_update():
    yarn = _indexer(YARN)
    assert yarn._rope_attention_scaling != 1.0
    report = row.install(_model(yarn))
    assert report["installed"], report
    raw, pooled = _banks(128, seed=5)
    want, got = _stock_then_fused(yarn, raw, pooled, 4 * 90 - 2, 4, None)
    np.testing.assert_array_equal(_bits(got), _bits(want))


def _graph(indexer, raw, pooled, offset, rows, tmp_path) -> collections.Counter:
    def update(raw, pooled, offset, total):
        return indexer._extend_pooled_fixed(_Bank(raw, pooled, offset, rows, None), total)

    with compiled_step_body():
        out = mx.compile(update)(
            raw, pooled, mx.array(offset, dtype=mx.int32), mx.array(offset + rows, dtype=mx.int32)
        )
    path = tempfile.mktemp(suffix=".dot", dir=tmp_path)
    mx.export_to_dot(path, out)
    names = [
        line.split('label ="')[1].split('"')[0]
        for line in open(path)
        if "shape=rectangle" in line
    ]
    views = {"Reshape", "Broadcast", "Slice", "Transpose", "ExpandDims", "Squeeze", "Flatten", "Unflatten", "AsStrided"}
    return collections.Counter(name for name in names if name not in views)


def test_the_update_is_one_kernel_before_the_bank_write(indexer, tmp_path):
    raw, pooled = _banks(256, seed=7)
    stock = _graph(indexer, raw, pooled, 4 * 100 - 2, 4, tmp_path)
    assert "CustomKernel" not in stock
    assert sum(stock.values()) >= 12, stock
    report = row.install(_model(indexer))
    assert report["installed"], report
    fused = _graph(indexer, raw, pooled, 4 * 100 - 2, 4, tmp_path)
    assert fused["CustomKernel"] == 1, fused
    assert fused["DynamicSliceUpdate"] == 1, fused
    # The position arithmetic around the kernel (the floor divisions and the
    # clamp) is the library's and its node count moves between MLX builds.
    assert sum(stock.values()) - sum(fused.values()) >= 10, (stock, fused)


def test_outside_a_compiled_body_the_stock_update_runs(indexer):
    assert row.install(_model(indexer))["installed"]
    raw, pooled = _banks(64, seed=8)
    out = indexer._extend_pooled_fixed(
        _Bank(raw, pooled, mx.array(4 * 30 - 2, dtype=mx.int32), 4, None),
        mx.array(4 * 30 + 2, dtype=mx.int32),
    )
    mx.eval(out)
    assert row.engagement()["traces"] == 0


def test_a_bank_of_another_dtype_keeps_the_stock_update(indexer):
    assert row.install(_model(indexer))["installed"]
    raw, pooled = _banks(64, seed=9, dtype=mx.float16)
    assert not row.serves(indexer, pooled, raw)


def test_the_probe_turns_the_row_off_on_any_difference(indexer, monkeypatch, capsys):
    real = row.pooled_row

    def one_bit_off(*args, **kwargs):
        out = real(*args, **kwargs)
        return mx.view(mx.view(out, mx.uint16) ^ mx.array(1, dtype=mx.uint16), out.dtype)

    monkeypatch.setattr(row, "pooled_row", one_bit_off)
    report = row.install(_model(indexer))
    assert not report["installed"]
    assert "no sum and rotation spelling" in report["disabled_reason"]
    assert "QSA pooled-key row off" in capsys.readouterr().out


def test_a_kernel_that_fails_to_build_leaves_the_stock_update(indexer, monkeypatch):
    def refuse(*_args, **_kwargs):
        raise RuntimeError("pipeline refused")

    monkeypatch.setattr(row, "_kernel", refuse)
    report = row.install(_model(indexer))
    assert not report["installed"]
    assert "pipeline refused" in report["disabled_reason"]


def test_the_switch_keeps_the_stock_update(indexer, monkeypatch):
    monkeypatch.setenv(row.ENV, "0")
    report = row.install(_model(indexer))
    assert not report["installed"]
    assert report["probe_cases"] == 0
