"""Per-segment encode memo: bounded, isolated, and token-identical.

The segmented chat encode is the concatenation of independent per-segment
encodes, so memoizing a segment's ids by (tokenizer, exact text) must never
change a single token. These tests pin that parity (stub tokenizers always,
the real Qwen3.6 tokenizer when it is cached locally) plus the memo's bounds,
off switch and observability.
"""

from __future__ import annotations

import json
import threading
import weakref
from pathlib import Path

import pytest

import mtplx.server.openai as oa
from mtplx.chat_encode_cache import ChatEncodeCache, ChatSegmentEncodeMemo


class GreedyTokenizer:
    """Longest-match tokenizer: a split point changes the ids, like BPE."""

    chat_template = "{{ messages }}"

    def __init__(self, vocab: list[str]):
        self.vocab = sorted(vocab, key=len, reverse=True)
        self.encode_calls = 0

    def encode(self, text, add_special_tokens=False):
        self.encode_calls += 1
        ids: list[int] = []
        at = 0
        while at < len(text):
            for piece in self.vocab:
                if piece and text.startswith(piece, at):
                    ids.append(1000 + self.vocab.index(piece))
                    at += len(piece)
                    break
            else:
                ids.append(ord(text[at]))
                at += 1
        return ids

    @property
    def vocab_size(self):
        return 1000

    @property
    def added_tokens_decoder(self):
        return dict(enumerate(self.vocab))

    def add_tokens(self, pieces):
        self.vocab = sorted([*self.vocab, *pieces], key=len, reverse=True)


class WrappedTokenizer:
    """Like mlx-lm's TokenizerWrapper: forwards attributes to the HF one."""

    def __init__(self, inner):
        self._tokenizer = inner

    def __getattr__(self, name):
        return getattr(self._tokenizer, name)


@pytest.fixture
def memo(monkeypatch):
    fresh = ChatSegmentEncodeMemo(max_tokens=200_000)
    monkeypatch.setattr(oa, "GLOBAL_CHAT_SEGMENT_MEMO", fresh)
    monkeypatch.delenv("MTPLX_CHAT_SEGMENT_MEMO", raising=False)
    return fresh


def _segmented(tok, text, boundaries, obs=None):
    return oa._encode_rendered_chat_text_segmented(
        tok, text, boundaries, template_observability=obs
    )


def _without_memo(monkeypatch, tok, text, boundaries):
    monkeypatch.setenv("MTPLX_CHAT_SEGMENT_MEMO", "off")
    try:
        return _segmented(tok, text, boundaries)
    finally:
        monkeypatch.delenv("MTPLX_CHAT_SEGMENT_MEMO")


# --- the memo itself -------------------------------------------------------


def test_hit_returns_a_fresh_copy(memo):
    memo.put("k", [1, 2, 3])
    first = memo.get("k")
    first.append(99)
    assert memo.get("k") == [1, 2, 3]
    assert memo.stats() == {"entries": 1, "tokens": 3, "hits": 2, "misses": 0}


def test_token_budget_evicts_least_recently_used():
    memo = ChatSegmentEncodeMemo(max_tokens=10)
    memo.put("a", [1] * 4)
    memo.put("b", [2] * 4)
    assert memo.get("a") is not None  # a is now most recent
    memo.put("c", [3] * 4)
    assert memo.get("b") is None
    assert memo.get("a") == [1] * 4
    assert memo.stats()["tokens"] == 8


def test_entry_cap_bounds_tiny_segments(monkeypatch):
    monkeypatch.setattr(ChatSegmentEncodeMemo, "MAX_ENTRIES", 3)
    memo = ChatSegmentEncodeMemo(max_tokens=1_000)
    for i in range(5):
        memo.put(f"k{i}", [i])
    assert memo.stats()["entries"] == 3
    assert memo.get("k0") is None


def test_oversized_segment_is_not_stored():
    memo = ChatSegmentEncodeMemo(max_tokens=4)
    memo.put("small", [1, 2])
    memo.put("big", [1, 2, 3, 4, 5])
    assert memo.get("big") is None
    assert memo.get("small") == [1, 2]


def test_storing_a_key_twice_counts_its_tokens_once():
    memo = ChatSegmentEncodeMemo(max_tokens=100)
    memo.put("k", [1, 2, 3])
    memo.put("k", [1, 2, 3])
    assert memo.stats()["tokens"] == 3


def test_token_budget_env_override(monkeypatch):
    monkeypatch.setenv("MTPLX_CHAT_SEGMENT_MEMO_TOKENS", "123")
    assert ChatSegmentEncodeMemo().max_tokens == 123
    monkeypatch.setenv("MTPLX_CHAT_SEGMENT_MEMO_TOKENS", "not-a-number")
    assert ChatSegmentEncodeMemo().max_tokens == ChatSegmentEncodeMemo.DEFAULT_MAX_TOKENS


def test_concurrent_use_stays_consistent():
    memo = ChatSegmentEncodeMemo(max_tokens=50)

    def worker(offset):
        for i in range(500):
            key = f"k{(i + offset) % 40}"
            if memo.get(key) is None:
                memo.put(key, [i % 7] * 3)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    stats = memo.stats()
    assert stats["tokens"] == 3 * stats["entries"] <= 50


# --- parity through the segmented encoder (stub tokenizer) ----------------

VOCAB = ["ab", "abc", "bc", "cd", "<s>", "\n\n", "é", "世界"]
TEXT = "<s>abcd\n\nabc</s>" + "bcabé世界" * 5 + "<s>ab\n\ncd" + "abc" * 50
BOUNDARIES = [5, 9, 20, 40, 70]


def test_memo_is_token_identical_to_plain_segmented_encode(memo, monkeypatch):
    tok = GreedyTokenizer(VOCAB)
    expected = _without_memo(monkeypatch, tok, TEXT, BOUNDARIES)
    cold = _segmented(tok, TEXT, BOUNDARIES)
    warm = _segmented(tok, TEXT, BOUNDARIES)
    assert cold == expected
    assert warm == expected


def test_growing_transcript_only_encodes_new_segments(memo, monkeypatch):
    tok = GreedyTokenizer(VOCAB)
    turn1, turn2 = TEXT[:40], TEXT
    _segmented(tok, turn1, [5, 9, 20])
    tok.encode_calls = 0
    obs: dict = {}
    ids = _segmented(tok, turn2, BOUNDARIES, obs)
    assert tok.encode_calls == 2
    assert ids == _without_memo(monkeypatch, tok, turn2, BOUNDARIES)
    # every turn-1 segment (its tail 20:40 included) is reused; 40:70 and
    # the new tail are the only segments tokenized
    assert obs["chat_segment_memo"]["hits"] == 4
    assert obs["chat_segment_memo"]["misses"] == 2
    assert obs["chat_segment_memo"]["reused_tokens"] == len(
        _without_memo(monkeypatch, tok, turn1, [5, 9, 20])
    )


def test_edit_in_earlier_history_is_not_served_stale(memo, monkeypatch):
    tok = GreedyTokenizer(VOCAB)
    _segmented(tok, TEXT, BOUNDARIES)
    edited = TEXT[:10] + "X" + TEXT[11:]
    obs: dict = {}
    ids = _segmented(tok, edited, BOUNDARIES, obs)
    assert ids == _without_memo(monkeypatch, tok, edited, BOUNDARIES)
    assert ids != _without_memo(monkeypatch, tok, TEXT, BOUNDARIES)
    assert obs["chat_segment_memo"]["misses"] == 1


def test_tokenizers_never_share_entries(memo, monkeypatch):
    plain = GreedyTokenizer([])
    merging = GreedyTokenizer(VOCAB)
    _segmented(plain, TEXT, BOUNDARIES)
    ids = _segmented(merging, TEXT, BOUNDARIES)
    assert ids == _without_memo(monkeypatch, merging, TEXT, BOUNDARIES)
    assert ids != _without_memo(monkeypatch, plain, TEXT, BOUNDARIES)


def test_tokens_added_in_place_are_not_served_stale(memo, monkeypatch):
    tok = GreedyTokenizer(VOCAB)
    _segmented(tok, TEXT, BOUNDARIES)
    tok.add_tokens(["abcab"])
    obs: dict = {}
    ids = _segmented(tok, TEXT, BOUNDARIES, obs)
    assert ids == _without_memo(monkeypatch, tok, TEXT, BOUNDARIES)
    assert obs["chat_segment_memo"]["hits"] == 0


def test_encoding_fingerprint_reads_through_a_wrapper():
    inner = GreedyTokenizer(VOCAB)
    wrapped = WrappedTokenizer(inner)
    before = oa._chat_tokenizer_encoding_fingerprint(wrapped)
    assert oa._chat_tokenizer_encoding_fingerprint(wrapped) == before
    inner.add_tokens(["abcab"])
    assert oa._chat_tokenizer_encoding_fingerprint(wrapped) != before
    assert isinstance(oa._chat_tokenizer_encoding_fingerprint(object()), str)


def test_wrapped_tokenizer_sees_tokens_added_to_the_inner_one(memo, monkeypatch):
    inner = GreedyTokenizer(VOCAB)
    wrapped = WrappedTokenizer(inner)
    _segmented(wrapped, TEXT, BOUNDARIES)
    inner.add_tokens(["abcab"])
    obs: dict = {}
    ids = _segmented(wrapped, TEXT, BOUNDARIES, obs)
    assert ids == _without_memo(monkeypatch, wrapped, TEXT, BOUNDARIES)
    assert obs["chat_segment_memo"]["hits"] == 0


def _single_word_ab_tokenizer():
    """A real fast tokenizer whose added token ``ab`` is whole-word only."""
    tokenizers = pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")
    backend = tokenizers.Tokenizer(
        tokenizers.models.WordLevel(
            vocab={"[UNK]": 0, "x": 1, "|": 2}, unk_token="[UNK]"
        )
    )
    backend.pre_tokenizer = tokenizers.pre_tokenizers.Split(
        pattern=tokenizers.Regex(r"\w|[^\w\s]|\s"), behavior="isolated"
    )
    backend.add_tokens(
        [tokenizers.AddedToken("ab", single_word=True, normalized=False)]
    )
    return transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]"
    )


def test_added_token_flag_change_is_not_served_stale(memo, monkeypatch):
    """Re-adding ``ab`` without single_word keeps the vocab size and the
    added-token count, but "xab" now splits as x + ab. The memo must not
    serve the ids it stored under the old flags."""
    from tokenizers import AddedToken

    tok = _single_word_ab_tokenizer()
    text, cut = "xab|xab", [4]
    assert _segmented(tok, text, cut) == [1, 0, 0, 2, 1, 0, 0]

    tok.add_tokens([AddedToken("ab", single_word=False, normalized=False)])
    assert (tok.vocab_size, len(tok.added_tokens_decoder)) == (3, 2)
    obs: dict = {}
    ids = _segmented(tok, text, cut, obs)
    assert ids == [1, 3, 2, 1, 3]
    assert ids == _without_memo(monkeypatch, tok, text, cut)
    assert obs["chat_segment_memo"]["hits"] == 0


def _special_marker_tokenizer(**kwargs):
    """A real fast tokenizer with ``<|im_start|>`` as a special token."""
    tokenizers = pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")
    vocab = {"[UNK]": 0}
    for word in ["abc", "hi", "<", "|", "im_start", ">", "\n"]:
        vocab[word] = len(vocab)
    backend = tokenizers.Tokenizer(
        tokenizers.models.WordLevel(vocab=vocab, unk_token="[UNK]")
    )
    backend.pre_tokenizer = tokenizers.pre_tokenizers.Split(
        pattern=tokenizers.Regex(r"\w+|[^\w\s]|\s"), behavior="isolated"
    )
    backend.add_special_tokens(
        [tokenizers.AddedToken("<|im_start|>", normalized=False, special=True)]
    )
    return transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", **kwargs
    )


def test_split_policy_change_is_not_served_stale(memo, monkeypatch):
    """Warm with split_special_tokens on, then turn it off. transformers
    copies the flag to the Rust tokenizer only on its next encode, so the
    Rust flag still says split; the key must see the transformers flag
    change on its own."""
    tok = _special_marker_tokenizer(split_special_tokens=True)
    text, cut = "<|im_start|>hi<|im_start|>abc", [14]
    split_ids = _segmented(tok, text, cut)
    assert 8 not in split_ids  # the marker went in as text

    tok.split_special_tokens = False
    assert tok._tokenizer.encode_special_tokens is True  # not synced yet
    obs: dict = {}
    ids = _segmented(tok, text, cut, obs)
    assert ids == [8, 2, 8, 1]
    assert obs["chat_segment_memo"]["hits"] == 0


class ChangingTokenizer(GreedyTokenizer):
    """Gains a token during its first encode call, as a shared tokenizer
    reconfigured by another thread would, and can drop it again."""

    def __init__(self, vocab, piece: str):
        super().__init__(vocab)
        self.piece = piece

    def encode(self, text, add_special_tokens=False):
        ids = super().encode(text, add_special_tokens=add_special_tokens)
        if self.encode_calls == 1:
            self.add_tokens([self.piece])
        return ids

    def remove_piece(self):
        self.vocab = [piece for piece in self.vocab if piece != self.piece]


def test_ids_are_filed_under_the_configuration_that_produced_them(
    memo, monkeypatch
):
    """The key is taken again after every miss. Segments encoded after the
    change are filed under the new configuration, so once the tokenizer is
    back to the old one none of them is served for it."""
    tok = ChangingTokenizer(VOCAB, piece="abcab")
    _segmented(tok, TEXT, BOUNDARIES)
    tok.remove_piece()
    obs: dict = {}
    ids = _segmented(tok, TEXT, BOUNDARIES, obs)
    assert ids == _without_memo(monkeypatch, tok, TEXT, BOUNDARIES)
    assert obs["chat_segment_memo"]["hits"] == 0


def _many_added_tokens_tokenizer(count: int):
    """A real fast tokenizer with ``count`` added tokens: the fingerprint
    walks all of them, so its cost grows with the count."""
    tokenizers = pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")
    vocab = {"[UNK]": 0}
    for word in ["user", "turn", "<", "|", "im_start", ">", "\n"]:
        vocab[word] = len(vocab)
    for index in range(256):
        vocab[f"w{index}"] = len(vocab)
    backend = tokenizers.Tokenizer(
        tokenizers.models.WordLevel(vocab=vocab, unk_token="[UNK]")
    )
    backend.pre_tokenizer = tokenizers.pre_tokenizers.Split(
        pattern=tokenizers.Regex(r"\w+|[^\w\s]|\s"), behavior="isolated"
    )
    backend.add_special_tokens(
        [tokenizers.AddedToken("<|im_start|>", normalized=False, special=True)]
    )
    backend.add_tokens(
        [
            tokenizers.AddedToken(f"<extra_{index}>", normalized=False)
            for index in range(count)
        ]
    )
    tok = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]"
    )
    tok.chat_template = "{% for m in messages %}{{ m['content'] }}{% endfor %}"
    return tok


@pytest.fixture
def fingerprint_calls(monkeypatch):
    calls = {"count": 0}
    real = oa._chat_tokenizer_encoding_fingerprint

    def counting(tokenizer):
        calls["count"] += 1
        return real(tokenizer)

    monkeypatch.setattr(oa, "_chat_tokenizer_encoding_fingerprint", counting)
    return calls


def test_fingerprint_is_taken_twice_per_encode_whatever_the_misses(
    memo, monkeypatch, fingerprint_calls
):
    """4,096 added tokens and 128 turns that all miss: the fingerprint is
    taken when the encode starts and once more before its ids are stored,
    not once per miss (that cost 1.1 s for 128 misses at 16,384 added
    tokens)."""
    tok = _many_added_tokens_tokenizer(4096)
    turns = [f"<|im_start|>user w{index % 256} turn\n" for index in range(128)]
    rendered = "".join(turns)
    boundaries = oa._chat_turn_boundaries(rendered)
    assert len(boundaries) == 127

    obs: dict = {}
    cold = _segmented(tok, rendered, boundaries, obs)
    assert obs["chat_segment_memo"]["misses"] == 128
    assert fingerprint_calls["count"] == 2

    fingerprint_calls["count"] = 0
    obs = {}
    warm = _segmented(tok, rendered, boundaries, obs)
    assert obs["chat_segment_memo"]["hits"] == 128
    assert fingerprint_calls["count"] == 1  # nothing new to store

    assert cold == warm == oa._encode_rendered_chat_text(tok, rendered)


def test_whole_request_encode_takes_two_fingerprints(
    memo, monkeypatch, fingerprint_calls
):
    """Through the request front: the whole-request key, the turn-cut proof
    and the segment keys share one fingerprint, and one more confirms it
    before anything is stored. A repeat is a whole-request hit: one."""
    monkeypatch.setattr(oa, "GLOBAL_CHAT_ENCODE_CACHE", ChatEncodeCache(max_entries=8))
    monkeypatch.setattr(oa, "_CHAT_TURN_SEGMENT_PROOFS", weakref.WeakKeyDictionary())
    tok = _many_added_tokens_tokenizer(4096)
    request = oa.ChatCompletionRequest(
        model="m",
        messages=[
            {"role": "user", "content": f"<|im_start|>user w{index} turn\n"}
            for index in range(64)
        ],
    )

    def encode(obs):
        return oa._encode_messages(
            tok,
            request.messages,
            enable_thinking=False,
            scoped_reasoning_history=True,
            template_observability=obs,
        )

    obs: dict = {}
    first = encode(obs)
    assert obs["chat_encode_cache"] == "miss"
    assert obs["chat_segment_memo"]["misses"] >= 64
    assert fingerprint_calls["count"] == 2

    fingerprint_calls["count"] = 0
    obs = {}
    assert encode(obs) == first
    assert obs["chat_encode_cache"] == "hit"
    assert fingerprint_calls["count"] == 1


def test_off_switch_bypasses_the_memo(memo, monkeypatch):
    monkeypatch.setenv("MTPLX_CHAT_SEGMENT_MEMO", "off")
    obs: dict = {}
    _segmented(GreedyTokenizer(VOCAB), TEXT, BOUNDARIES, obs)
    assert memo.stats() == {"entries": 0, "tokens": 0, "hits": 0, "misses": 0}
    assert "chat_segment_memo" not in obs


def test_unsegmented_encode_does_not_touch_the_memo(memo):
    obs: dict = {}
    _segmented(GreedyTokenizer(VOCAB), TEXT, [], obs)
    assert memo.stats()["entries"] == 0
    assert "chat_segment_memo" not in obs


# --- parity with the real tokenizer and chat template ---------------------

MODEL_DIR = (
    Path.home() / ".mtplx/models/Youssofal--Qwen3.6-35B-A3B-MTPLX-Optimized-Balance"
)
needs_real_tokenizer = pytest.mark.skipif(
    not (MODEL_DIR / "tokenizer.json").exists(),
    reason="Qwen3.6 model pack not cached locally",
)
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": name,
            "description": f"The {name} tool.",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
        },
    }
    for name in ("read", "bash")
]
TINY_PNG = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlE"
    "QVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


@pytest.fixture(scope="module")
def real_tok():
    from mtplx.runtime import _load_tokenizer_resilient

    config = json.loads((MODEL_DIR / "config.json").read_text())
    return _load_tokenizer_resilient(MODEL_DIR, config)


def _tool_turn(i: int, result: str) -> list[dict]:
    call = {
        "id": f"call_{i}",
        "type": "function",
        "function": {"name": "read", "arguments": json.dumps({"path": f"f{i}.py"})},
    }
    return [
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": f"Step {i}: read the next file. Ünïcödé 世界 🚀",
            "tool_calls": [call],
        },
        {"role": "tool", "tool_call_id": f"call_{i}", "content": result},
    ]


def _agent_transcript(turns: int) -> list[dict]:
    messages = [
        {"role": "system", "content": "You are a coding agent."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Fix the bug shown here."},
                {"type": "image_url", "image_url": {"url": TINY_PNG}},
            ],
        },
    ]
    for i in range(turns):
        if i in (2, 3):
            result = "same file content\n" * 20  # repeated segments
        elif i == 4:
            result = "def huge():\n    return 'x' * 80\n" * 4000  # very long segment
        else:
            result = f"def f{i}():\n    return {i}  # naïve café 日本語 \u200b\n" * 10
        messages.extend(_tool_turn(i, result))
    return messages


def _encode_messages(tok, messages, *, thinking, mode, obs=None):
    request = oa.ChatCompletionRequest(model="m", messages=messages)
    return oa._encode_messages(
        tok,
        request.messages,
        enable_thinking=thinking,
        reasoning_effort="medium",
        tools=TOOLS,
        tool_prompt_mode=mode,
        template_observability=obs if obs is not None else {},
    )


@needs_real_tokenizer
@pytest.mark.parametrize("mode", ["hybrid", "compact"])
@pytest.mark.parametrize("thinking", [True, False])
def test_real_growing_agent_transcript_is_token_identical(
    real_tok, memo, monkeypatch, mode, thinking
):
    monkeypatch.setenv("MTPLX_CHAT_ENCODE_CACHE", "off")
    for turns in range(1, 7):
        messages = _agent_transcript(turns)
        with_memo = _encode_messages(real_tok, messages, thinking=thinking, mode=mode)
        monkeypatch.setenv("MTPLX_CHAT_SEGMENT_MEMO", "off")
        without = _encode_messages(real_tok, messages, thinking=thinking, mode=mode)
        monkeypatch.delenv("MTPLX_CHAT_SEGMENT_MEMO")
        assert with_memo == without, f"diverged at {turns} turns"


@needs_real_tokenizer
def test_real_warm_turn_reuses_history_and_survives_an_edit(
    real_tok, memo, monkeypatch
):
    monkeypatch.setenv("MTPLX_CHAT_ENCODE_CACHE", "off")
    _encode_messages(real_tok, _agent_transcript(5), thinking=True, mode="hybrid")

    obs: dict = {}
    grown = _agent_transcript(6)
    ids = _encode_messages(real_tok, grown, thinking=True, mode="hybrid", obs=obs)
    assert obs["chat_segment_memo"]["hits"] >= 5
    assert obs["chat_segment_memo"]["misses"] == 1

    edited = _agent_transcript(6)
    edited[3]["content"] = "an earlier tool result, edited"
    obs = {}
    edited_ids = _encode_messages(real_tok, edited, thinking=True, mode="hybrid", obs=obs)
    monkeypatch.setenv("MTPLX_CHAT_SEGMENT_MEMO", "off")
    assert ids == _encode_messages(real_tok, grown, thinking=True, mode="hybrid")
    assert edited_ids == _encode_messages(
        real_tok, edited, thinking=True, mode="hybrid"
    )
    assert edited_ids != ids
    assert obs["chat_segment_memo"]["misses"] >= 1
