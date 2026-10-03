"""Context-copy block acceptance reads the device once per chunk of rows, exactly.

A copy round verifies ``[primary, c_0 .. c_{n-1}]`` and accepts the copied
tokens in order, each as a point-mass proposal (accept with the target's own
shaped probability, draw the correction from the residual on the first
rejection). The target distribution of each row used to be built and read
from the device one row at a time, a full-vocabulary support build plus a
host round trip per examined row, up to 24 per round on a 24-token block.

``_point_mass_block_accept`` reads the rows a chunk at a time (8 rows at the
248,320-entry vocabulary) and runs the host arithmetic only for the rows it
examines. These tests pin that the result is the per-row reader's, bit for
bit, on the Metal device: the same distributions, the same accepted count and
correction, the same generator state afterwards, and the same emitted stream
through the real decode loop on both copy lanes; that a row past the first
rejection can neither warn nor raise; and how many reads a block costs.
"""

from __future__ import annotations

import math
import warnings

import mlx.core as mx
import numpy as np
import pytest

import mtplx.generation as generation
from mtplx.fast_sampling import (
    _block_read_chunk_rows,
    sparse_distribution_from_mlx_logits,
    sparse_distribution_rows_from_mlx_logits,
    sparse_distributions_from_mlx_logits,
)
from mtplx.generation import _distribution_from_mlx_logits, _point_mass_block_accept
from mtplx.sampling import (
    NonFiniteLogitsError,
    SamplerConfig,
    SparseDistribution,
    acceptance_probability,
    residual_distribution,
    sample_from_distribution,
)

from test_context_copy_stats import _ScriptedModel, _clean_env, _runtime


FAMILY = SamplerConfig(temperature=1.0, top_p=0.95, top_k=20)
QWEN_VOCAB = 248_320
CONFIGS = (
    FAMILY,
    SamplerConfig(temperature=0.6, top_p=0.95, top_k=20),
    SamplerConfig(temperature=1.0, top_p=1.0, top_k=20),
    SamplerConfig(temperature=0.7, top_p=0.8, top_k=5),
)


@pytest.fixture()
def metal():
    """The bit-equality claims are about the Metal kernels: run them there."""

    if not mx.metal.is_available():
        pytest.skip("the claims under test are about the Metal kernels")
    previous = mx.default_device()
    mx.set_default_device(mx.gpu)
    assert mx.default_device() == mx.gpu
    yield
    mx.set_default_device(previous)


def _per_row_block_accept(block_logits, block, sampler, rng):
    """The copy lanes' accept loop before this change, verbatim in effect:
    one ``_distribution_from_mlx_logits`` device read per examined row."""

    vocab = int(block_logits.shape[-1])
    accepted = 0
    for index, drafted in enumerate(block):
        target_p = _distribution_from_mlx_logits(
            block_logits[index], sampler, token_counts=None
        )
        draft_q = SparseDistribution(
            np.array([int(drafted)], dtype=np.int64),
            np.array([1.0], dtype=np.float64),
            vocab,
        )
        accept_prob = acceptance_probability(target_p, draft_q, int(drafted))
        if float(rng.random()) <= accept_prob:
            accepted += 1
            continue
        return accepted, int(
            sample_from_distribution(residual_distribution(target_p, draft_q), rng)
        )
    return accepted, None


def _assert_same_distribution(left: SparseDistribution, right: SparseDistribution):
    assert left.vocab_size == right.vocab_size
    assert np.array_equal(left.token_ids, right.token_ids)
    # Bit equality, not closeness: the batched read must be the same float64s.
    assert left.probs.dtype == right.probs.dtype == np.float64
    assert np.array_equal(left.probs.view(np.uint64), right.probs.view(np.uint64))


def _special_rows(vocab: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Rows that reach the reader's rarely taken branches."""

    # Quantized to quarter steps: many exact ties at the top-k cutoff, so the
    # candidate superset spills and the exact tie-resolving path runs.
    tied = np.round(rng.normal(size=vocab) * 4.0) / 4.0
    # Every logit equal: the cutoff tie spans the whole vocabulary.
    flat = np.zeros(vocab)
    # Half the vocabulary masked to -inf (a grammar or suppress mask).
    masked = rng.normal(size=vocab)
    masked[rng.permutation(vocab)[: vocab // 2]] = -np.inf
    # One dominant token: the nucleus keeps a single entry.
    peaked = rng.normal(size=vocab)
    peaked[int(rng.integers(vocab))] = 40.0
    return [tied, flat, masked, peaked]


@pytest.mark.parametrize("config", CONFIGS, ids=lambda c: f"t{c.temperature}-p{c.top_p}-k{c.top_k}")
@pytest.mark.parametrize(
    ("vocab", "rows", "dtype"),
    [
        (QWEN_VOCAB, 16, mx.bfloat16),
        (QWEN_VOCAB, 24, mx.float32),
        (QWEN_VOCAB, 12, mx.float16),
        (1_000, 24, mx.float32),
        (1_000, 9, mx.bfloat16),
    ],
    ids=["qwen-vocab-16-bf16", "qwen-vocab-24-f32", "qwen-vocab-12-f16", "small-f32", "small-bf16"],
)
def test_rows_reader_is_the_single_row_reader_bit_for_bit(metal, config, vocab, rows, dtype):
    rng = np.random.default_rng(vocab + rows)
    logits = rng.normal(size=(rows, vocab)) * 3.0
    logits[: len(_special_rows(vocab, rng))] = np.stack(_special_rows(vocab, rng))
    block = mx.array(logits.astype(np.float32)).astype(dtype)

    batched = sparse_distribution_rows_from_mlx_logits(block, config)

    assert batched is not None and len(batched) == rows
    for index in range(rows):
        single = sparse_distribution_from_mlx_logits(block[index], config)
        _assert_same_distribution(batched[index], single)



@pytest.mark.parametrize(
    "vocab",
    [4_096, 4_097, 248_320],
    ids=["block-kernel-max", "looped-kernel-min", "flash-next-vocab"],
)
def test_device_normalizer_of_a_row_does_not_depend_on_the_block_height(metal, vocab):
    """The rows reader's bit equality rests on one device reduction.

    Everything else in ``_device_serial_support_arrays`` is elementwise, a
    gather, or a candidate superset whose ties the host resolves exactly; the
    full-vocabulary ``logsumexp`` normalizer is the only float reduction over
    a row. MLX picks its kernel by row length alone (the block kernel up to
    4,096 entries, the looped kernel above) and reduces each row in its own
    threadgroup (``mlx/backend/metal/logsumexp.cpp``), so a row's normalizer
    must be the same float whether it is reduced alone or inside a copy
    block. The heights are the copy lanes' block and verify widths.
    """

    rng = np.random.default_rng(vocab)
    logits = rng.normal(size=(25, vocab)) * 3.0
    logits[3, rng.permutation(vocab)[: vocab // 2]] = -np.inf
    logits[5, int(rng.integers(vocab))] = 40.0
    rows = mx.array(logits.astype(np.float32))
    single = []
    for index in range(25):
        value = mx.logsumexp(rows[index : index + 1], axis=-1, keepdims=True)
        mx.eval(value)
        single.append(np.asarray(value).reshape(-1)[0])
    single = np.array(single, dtype=np.float32)

    for height in (2, 8, 9, 12, 13, 16, 17, 24, 25):
        block = mx.logsumexp(rows[:height], axis=-1, keepdims=True)
        mx.eval(block)
        batched = np.asarray(block).reshape(-1)
        assert batched.dtype == np.float32
        assert np.array_equal(batched.view(np.uint32), single[:height].view(np.uint32)), height

def test_list_reader_keeps_its_contract():
    rng = np.random.default_rng(7)
    block = mx.array(rng.normal(size=(5, 300)).astype(np.float32))
    listed = sparse_distributions_from_mlx_logits(block, FAMILY)
    rows = sparse_distribution_rows_from_mlx_logits(block, FAMILY)
    assert listed is not None and rows is not None
    for index, dist in enumerate(listed):
        _assert_same_distribution(dist, rows[index])
    assert sparse_distributions_from_mlx_logits(block, SamplerConfig(temperature=0.0)) is None
    assert sparse_distribution_rows_from_mlx_logits(block, SamplerConfig(temperature=0.0)) is None
    assert (
        sparse_distribution_rows_from_mlx_logits(
            block, SamplerConfig(temperature=1.0, top_p=0.95, top_k=0)
        )
        is None
    )


def test_rows_reader_raises_on_a_non_finite_row_only_when_it_is_read():
    rng = np.random.default_rng(11)
    logits = rng.normal(size=(4, 64)).astype(np.float32)
    logits[2] = math.nan
    rows = sparse_distribution_rows_from_mlx_logits(mx.array(logits), FAMILY)

    assert rows is not None
    rows[0]
    rows[1]
    rows[3]
    with pytest.raises(NonFiniteLogitsError):
        rows[2]
    # The per-row reader raises the same error on that row.
    with pytest.raises(NonFiniteLogitsError):
        sparse_distribution_from_mlx_logits(mx.array(logits[2]), FAMILY)


# --- rows past the first rejection, and rows that are examined -------------


def _rejecting_block(bad_row: np.ndarray, *, width: int = 8, bad_index: int = 1):
    """Row 0 strongly prefers token 0 and the first copied token is 1,000,
    far outside its top-k: the copy is rejected at row 0 with certainty and
    the correction is drawn from row 0. ``bad_row`` sits at ``bad_index``,
    inside the same 8-row chunk, where the per-row loop never looks."""

    draws = np.random.default_rng(0)
    logits = draws.normal(size=(width + 1, QWEN_VOCAB)).astype(np.float32)
    logits[0, 0] = 30.0
    logits[bad_index] = bad_row
    block = [1000] + [int(t) for t in draws.integers(QWEN_VOCAB, size=width - 1)]
    block_logits = mx.array(logits).astype(mx.bfloat16)
    mx.eval(block_logits)
    return block_logits, block


def _same_outcome_and_state(block_logits, block, config):
    stock_rng = np.random.default_rng(0)
    new_rng = np.random.default_rng(0)
    stock = _per_row_block_accept(block_logits, block, config, stock_rng)
    new = _point_mass_block_accept(block_logits, block, config, new_rng)
    assert stock == new
    assert stock[0] == 0 and stock[1] == 0  # rejected at row 0, corrected to token 0
    assert new_rng.bit_generator.state == stock_rng.bit_generator.state


def test_an_unread_all_nan_row_cannot_fail_the_block_under_strict_warnings(metal):
    """Review scenario 1: an all-NaN row after a certain rejection. The
    per-row loop never examines it; with RuntimeWarning promoted to an
    error, the whole-block host arithmetic used to raise "All-NaN slice
    encountered" before the first draw."""

    block_logits, block = _rejecting_block(np.full(QWEN_VOCAB, np.nan, dtype=np.float32))
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        _same_outcome_and_state(block_logits, block, FAMILY)


def test_an_unread_minus_inf_row_cannot_fail_the_block_under_numpy_raise(metal):
    """Review scenario 2: top-p off (the max-subtraction branch) and an
    all -inf later row under ``np.seterr(invalid="raise")``: the
    whole-block subtraction used to raise "invalid value encountered in
    subtract"."""

    no_top_p = SamplerConfig(temperature=1.0, top_p=1.0, top_k=20)
    block_logits, block = _rejecting_block(
        np.full(QWEN_VOCAB, -np.inf, dtype=np.float32), bad_index=5
    )
    with np.errstate(invalid="raise"):
        _same_outcome_and_state(block_logits, block, no_top_p)


def test_an_examined_bad_row_fails_exactly_as_under_the_per_row_reader(metal):
    """The examined row keeps the per-row reader's behaviour: the host
    fallback's NonFiniteLogitsError by default, and under strict warnings
    the same RuntimeWarning from the same statement."""

    draws = np.random.default_rng(1)
    logits = draws.normal(size=(9, QWEN_VOCAB)).astype(np.float32)
    logits[0] = np.nan
    block_logits = mx.array(logits).astype(mx.bfloat16)
    block = [int(t) for t in draws.integers(QWEN_VOCAB, size=8)]

    for accept in (_per_row_block_accept, _point_mass_block_accept):
        with pytest.raises(NonFiniteLogitsError):
            accept(block_logits, block, FAMILY, np.random.default_rng(0))
    messages = []
    for accept in (_per_row_block_accept, _point_mass_block_accept):
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            with pytest.raises(RuntimeWarning) as raised:
                accept(block_logits, block, FAMILY, np.random.default_rng(0))
        messages.append(str(raised.value))
    assert messages[0] == messages[1]


def test_reads_are_one_per_chunk_at_the_qwen_vocabulary(metal, monkeypatch):
    """How many device reads a 24-token block costs, stated honestly: one per
    8-row chunk reached, where the per-row loop paid one per examined row."""

    assert _block_read_chunk_rows(QWEN_VOCAB) == 8  # 32 MiB // (16 B x 248,320)
    width = 24
    draws = np.random.default_rng(4)
    # Distinct background values: no tie at the top-k cutoff, so no row
    # needs the exact tie resolver's extra read.
    base = (draws.normal(size=(width + 1, QWEN_VOCAB)) * 0.1 - 4.0).astype(np.float32)
    tokens = [int(t) for t in draws.integers(QWEN_VOCAB, size=width)]

    reads = {"n": 0}
    real_eval = mx.eval

    def counting_eval(*args, **kwargs):
        reads["n"] += 1
        return real_eval(*args, **kwargs)

    def counted(accept, rejected_at):
        logits = base.copy()
        block = list(tokens)
        for row, token in enumerate(block):
            logits[row, token] = 30.0  # the target is certain of the copy
        if rejected_at is not None:
            block[rejected_at] = (tokens[rejected_at] + 1) % QWEN_VOCAB  # p = 0
        block_logits = mx.array(logits)
        real_eval(block_logits)
        with monkeypatch.context() as patch:
            patch.setattr(mx, "eval", counting_eval)
            reads["n"] = 0
            result = accept(block_logits, block, FAMILY, np.random.default_rng(0))
        return result, reads["n"]

    for rejected_at, stock_reads, chunk_reads in (
        (None, 24, 3),  # fully accepted: rows 0-23, chunks 0-2
        (0, 1, 1),  # rejected at the first row: one read either way
        (10, 11, 2),  # rejected in the second chunk
    ):
        stock, stock_n = counted(_per_row_block_accept, rejected_at)
        new, new_n = counted(_point_mass_block_accept, rejected_at)
        assert stock == new
        assert (stock_n, new_n) == (stock_reads, chunk_reads), rejected_at


@pytest.mark.parametrize("config", [FAMILY, CONFIGS[1]], ids=["t1.0", "t0.6"])
def test_qwen_vocabulary_blocks_match_the_per_row_loop_through_rejection(metal, config):
    """Production-width blocks that reject in every chunk position, with the
    correction and the generator state after it, and the next draw."""

    draws = np.random.default_rng(7)
    seen = set()
    for trial, (width, rejected_at) in enumerate(
        ((8, 0), (8, 7), (12, 8), (16, 13), (24, 3), (24, 17), (24, None), (16, None))
    ):
        logits = draws.normal(size=(width + 1, QWEN_VOCAB)).astype(np.float32)
        block = []
        for row in range(width):
            favourite = int(np.argmax(logits[row]))
            logits[row, favourite] += 9.0
            block.append(favourite)
        if rejected_at is not None:
            outsider = int(np.argmin(logits[rejected_at]))  # never in the top-k
            block[rejected_at] = outsider
        block_logits = mx.array(logits).astype(mx.bfloat16)
        seed = int(draws.integers(1 << 31))
        stock_rng = np.random.default_rng(seed)
        new_rng = np.random.default_rng(seed)

        stock = _per_row_block_accept(block_logits, block, config, stock_rng)
        new = _point_mass_block_accept(block_logits, block, config, new_rng)

        assert new == stock, trial
        assert new_rng.bit_generator.state == stock_rng.bit_generator.state, trial
        assert new_rng.random() == stock_rng.random(), trial
        seen.add("full" if stock[1] is None else "rejected")
    assert seen == {"full", "rejected"}


def _accept_block(rng: np.random.Generator, vocab: int, width: int):
    """A copy block scored by uncertain target rows.

    Half the trials copy the target's favourite token under a confident
    target (most of those blocks are accepted whole); the other half mix
    favourites, runners-up and outsiders under a flatter target (those reject
    somewhere and draw a correction)."""

    confident = rng.random() < 0.5
    logits = rng.normal(size=(width + 1, vocab)) * (2.0 if not confident else 1.0)
    block = []
    for row in range(width):
        favourite = int(np.argmax(logits[row]))
        if confident:
            logits[row, favourite] += 9.0
            block.append(favourite)
            continue
        choice = rng.random()
        if choice < 0.6:
            token = favourite
        elif choice < 0.9:
            token = int(np.argsort(-logits[row])[int(rng.integers(1, 6))])
        else:
            token = int(rng.integers(vocab))
        block.append(token)
    return mx.array(logits.astype(np.float32)).astype(mx.bfloat16), block


@pytest.mark.parametrize("config", CONFIGS, ids=lambda c: f"t{c.temperature}-p{c.top_p}-k{c.top_k}")
def test_block_accept_is_the_per_row_loop_draw_for_draw(metal, config):
    trials = np.random.default_rng(2026)
    outcomes = set()
    for trial in range(60):
        width = int(trials.integers(1, 25))
        logits, block = _accept_block(trials, vocab=512, width=width)
        seed = int(trials.integers(1 << 31))
        stock_rng = np.random.default_rng(seed)
        new_rng = np.random.default_rng(seed)

        stock = _per_row_block_accept(logits[0 : width + 1], block, config, stock_rng)
        new = _point_mass_block_accept(logits[0 : width + 1], block, config, new_rng)

        assert new == stock, f"trial {trial}"
        # Same number and order of draws: the request's generator is shared by
        # every later sample of the response.
        assert new_rng.bit_generator.state == stock_rng.bit_generator.state
        outcomes.add("full" if stock[1] is None else "rejected")
    assert outcomes == {"full", "rejected"}


def test_block_accept_reads_the_device_once_where_the_loop_read_it_per_row(monkeypatch):
    width = 8
    draws = np.random.default_rng(3)
    # Distinct background values: no tie at the top-k cutoff, so each row is
    # one support build and one read (a tie adds the exact resolver's read).
    logits = (draws.normal(size=(width + 1, 256)) * 0.1 - 4.0).astype(np.float32)
    block = [int(t) for t in draws.integers(256, size=width)]
    for row, token in enumerate(block):
        logits[row, token] = 30.0  # the target is certain: every copy accepts
    block_logits = mx.array(logits)
    mx.eval(block_logits)

    reads = {"n": 0}
    real_eval = mx.eval

    def counting_eval(*args, **kwargs):
        reads["n"] += 1
        return real_eval(*args, **kwargs)

    monkeypatch.setattr(mx, "eval", counting_eval)

    reads["n"] = 0
    stock = _per_row_block_accept(block_logits, block, FAMILY, np.random.default_rng(0))
    stock_reads = reads["n"]
    reads["n"] = 0
    new = _point_mass_block_accept(block_logits, block, FAMILY, np.random.default_rng(0))
    new_reads = reads["n"]

    assert stock == new == (width, None)
    assert stock_reads == width  # before: one device read per copied token
    assert new_reads == 1


def test_empty_block_accepts_nothing_and_draws_nothing():
    rng = np.random.default_rng(5)
    state = rng.bit_generator.state
    assert _point_mass_block_accept(mx.zeros((1, 16)), [], FAMILY, rng) == (0, None)
    assert rng.bit_generator.state == state


# --- the real decode loop, both copy lanes -------------------------------


class _PositionCounter:
    """A trimmable cache entry that only counts positions, so the batched copy
    lane can commit an accepted prefix by trimming (the automaton has no KV)."""

    def __init__(self) -> None:
        self.offset = 0

    def is_trimmable(self) -> bool:
        return True

    def trim(self, n: int) -> int:
        self.offset -= int(n)
        return int(n)


class _SoftScriptedModel(_ScriptedModel):
    """Like the stats tests' automaton, but the target is uncertain: the
    scripted successor carries a bounded lead over seeded noise, so sampled
    copy rounds accept some tokens, reject others and emit corrections."""

    def make_cache(self):
        return [_PositionCounter()]

    def __call__(self, input_ids, *, cache=None, **kwargs):
        if cache:
            cache[0].offset += int(np.asarray(input_ids).size)
        return super().__call__(input_ids, cache=cache, **kwargs)

    def _logits_for(self, last_tokens):
        rows = []
        for token in last_tokens:
            noise = np.random.default_rng(1000 + int(token)).normal(size=self.vocab)
            noise[self.next_map(int(token)) % self.vocab] += 6.0
            rows.append(noise.astype(np.float32))
        return mx.array(np.stack(rows)[None], dtype=mx.float32)


def _sampled_copy_run(verify_strategy: str, seed: int):
    model = _SoftScriptedModel(24, lambda t: t + 1)
    prompt = list(range(24)) * 3
    return generation.generate_mtpk(
        _runtime(model),
        prompt,
        max_tokens=160,
        sampler=FAMILY,
        speculative_depth=1,
        seed=seed,
        stop_token_ids=set(),
        verify_strategy=verify_strategy,
    )


@pytest.mark.parametrize("verify_strategy", ["capture_commit", "batched"])
def test_sampled_copy_lane_emits_the_per_row_readers_stream(monkeypatch, verify_strategy):
    _clean_env(monkeypatch)
    for seed in (0, 1, 2):
        batched = _sampled_copy_run(verify_strategy, seed)
        with monkeypatch.context() as per_row:
            # None sends every row through the per-row reader: the old loop.
            per_row.setattr(
                generation,
                "sparse_distribution_rows_from_mlx_logits",
                lambda *_args, **_kwargs: None,
            )
            stock = _sampled_copy_run(verify_strategy, seed)

        assert list(batched.tokens) == list(stock.tokens), f"seed {seed}"
        assert batched.stats.context_copy_rounds == stock.stats.context_copy_rounds
        assert (
            batched.stats.context_copy_accepted_tokens
            == stock.stats.context_copy_accepted_tokens
        )
        # The run must actually exercise sampled copy rounds with rejections,
        # or the comparison above proves nothing about the accept path.
        assert stock.stats.context_copy_rounds > 0
        assert (
            stock.stats.context_copy_accepted_tokens
            < stock.stats.context_copy_drafted_tokens
        )
