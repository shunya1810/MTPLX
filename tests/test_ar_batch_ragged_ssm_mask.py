"""Ragged batched prefill must not let right-pad tokens reach the GDN state.

mlx-lm's BatchGenerator builds a prompt batch by merging fresh per-sequence
caches (``ArraysCache.merge`` sets ``left_padding`` to zeros) and, when the
prompts differ in length, right-pads the shorter ones and calls
``prepare(lengths=..., right_padding=...)``. With both fields armed, the
stock ``make_mask`` returned the left-padding mask alone, which is all true,
so the pad tokens of the shorter rows went through ``gated_delta_update`` and
changed their recurrent state. The Qwen 3.x hybrids (``qwen3_5``,
``qwen3_5_moe``) build that mask through ``create_ssm_mask``, so a batched-AR
row prefilled beside a longer prompt decoded from a drifted state.

The sequence below is the lane's: merge, prepare, one padded forward,
finalize, then a ``[B, 1]`` decode step, compared with each prompt run alone.
It runs on the CPU device, where a batched float32 matmul matches B=1 to
rounding; on M5 GPUs the batched matmul route alone differs from B=1 by about
3e-4, which would hide the pad drift behind a looser bound.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from mtplx.arrays_cache_patch import install_arrays_cache_fix

LENS = (19, 7)  # row 1 is right-padded by 12 tokens
VOCAB = 97


@pytest.fixture
def cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)


def _qwen35_text_model(num_layers: int):
    from mlx_lm.models.qwen3_5 import TextModel, TextModelArgs

    args = TextModelArgs(
        model_type="qwen3_5_text",
        hidden_size=64,
        intermediate_size=96,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=VOCAB,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        full_attention_interval=4,
    )
    mx.random.seed(11)
    model = TextModel(args)
    mx.eval(model.parameters())
    return model


def _solo(model, prompts, decode_ids):
    rows = []
    for i, prompt in enumerate(prompts):
        cache = model.make_cache()
        model(prompt, cache=cache)
        logits = model(decode_ids[i : i + 1], cache=cache)
        mx.eval(logits)
        rows.append(logits)
    return rows


def _batched(model, prompts, decode_ids):
    width = max(LENS)
    padded = mx.concatenate(
        [
            mx.concatenate(
                [prompt, mx.zeros((1, width - prompt.shape[1]), mx.int32)], axis=1
            )
            for prompt in prompts
        ],
        axis=0,
    )
    fresh = [model.make_cache() for _ in prompts]
    caches = [type(per_layer[0]).merge(list(per_layer)) for per_layer in zip(*fresh)]
    for cache in caches:
        cache.prepare(lengths=list(LENS), right_padding=[width - n for n in LENS])
    model(padded, cache=caches)
    mx.eval([cache.state for cache in caches])
    for cache in caches:
        cache.finalize()
    logits = model(decode_ids, cache=caches)
    mx.eval(logits)
    return logits


# qwen3_5 puts a full-attention layer every fourth layer: 3 GDN + 1 and 6 GDN + 2.
@pytest.mark.parametrize("num_layers", [4, 8])
def test_ragged_batched_rows_match_solo_on_qwen35_hybrid(cpu, num_layers):
    install_arrays_cache_fix()
    model = _qwen35_text_model(num_layers)
    mx.random.seed(5)
    prompts = [mx.random.randint(0, VOCAB, (1, n)) for n in LENS]
    decode_ids = mx.random.randint(0, VOCAB, (len(LENS), 1))

    reference = _solo(model, prompts, decode_ids)
    batched = _batched(model, prompts, decode_ids)

    for row in range(len(LENS)):
        diff = float(mx.abs(batched[row : row + 1] - reference[row]).max())
        assert diff < 1e-5, f"row {row} (length {LENS[row]}) differs from solo by {diff}"


def test_make_mask_ands_left_padding_and_lengths():
    install_arrays_cache_fix()
    from mlx_lm.models.cache import ArraysCache

    merged = ArraysCache.merge([ArraysCache(size=2), ArraysCache(size=2)])
    assert merged.left_padding.tolist() == [0, 0]  # what merge of fresh caches sets
    merged.prepare(lengths=[5, 2], right_padding=[0, 3])
    assert merged.make_mask(5).tolist() == [
        [True] * 5,
        [True, True, False, False, False],
    ]
    merged.finalize()
    assert merged.make_mask(1) is None
