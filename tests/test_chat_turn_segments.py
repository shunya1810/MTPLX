"""Seam-less chat renders encode turn by turn, token-identical.

Scoped reasoning history renders plain chat without generation seams, so the
segment memo never applied there. Cutting before every ``<|im_start|>`` (an
atomic added token) must give exactly the single-call ids; these tests pin
that on the real Qwen3.6 tokenizer and chat template when it is cached
locally, the proof that gates the cut (with the two counterexamples the
marker metadata alone let through), the guards and the off switch.
"""

from __future__ import annotations

import json
import random
import weakref
from dataclasses import dataclass
from pathlib import Path

import pytest

import mtplx.server.openai as oa
from mtplx.chat_encode_cache import ChatSegmentEncodeMemo


@dataclass
class FakeAddedToken:
    content: str
    normalized: bool = False
    lstrip: bool = False
    rstrip: bool = False
    single_word: bool = False
    special: bool = True


class TurnTokenizer:
    """One id per character; counts encode calls."""

    def __init__(self, turn_open: FakeAddedToken | None):
        self.added = {} if turn_open is None else {1: turn_open}
        self.encode_calls = 0

    def encode(self, text, add_special_tokens=False):
        self.encode_calls += 1
        return [ord(char) for char in text]

    vocab_size = 1000

    @property
    def added_tokens_decoder(self):
        return self.added


class WrappedTokenizer:
    """Like mlx-lm's TokenizerWrapper: forwards attributes to the HF one."""

    def __init__(self, inner):
        self._tokenizer = inner

    def __getattr__(self, name):
        return getattr(self._tokenizer, name)


RENDER = (
    "<|im_start|>system\nsys<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n"
)


@pytest.fixture
def memo(monkeypatch):
    fresh = ChatSegmentEncodeMemo(max_tokens=2_000_000)
    monkeypatch.setattr(oa, "GLOBAL_CHAT_SEGMENT_MEMO", fresh)
    monkeypatch.setattr(oa, "_CHAT_TURN_SEGMENT_PROOFS", weakref.WeakKeyDictionary())
    monkeypatch.delenv("MTPLX_CHAT_SEGMENT_MEMO", raising=False)
    monkeypatch.delenv("MTPLX_CHAT_TURN_SEGMENTS", raising=False)
    return fresh


def test_boundaries_sit_before_every_turn_but_the_first():
    starts = [RENDER.find("<|im_start|>user"), RENDER.find("<|im_start|>assistant")]
    assert oa._chat_turn_boundaries(RENDER) == starts
    assert oa._chat_turn_boundaries("no turns here") == []


def _proof_encodes(tok) -> int:
    """Encode calls the one-time proof makes: each probe once whole, then
    once per piece between its cuts."""
    return sum(
        2 + len(oa._chat_turn_boundaries(probe))
        for probe in oa._chat_turn_segment_probes(tok)
    )


def test_turns_are_encoded_separately_and_memoized(memo):
    tok = TurnTokenizer(FakeAddedToken("<|im_start|>"))
    obs: dict = {}
    ids = oa._encode_rendered_chat_turns(tok, RENDER, obs)
    assert ids == [ord(char) for char in RENDER]
    # Cold: the proof runs first, then each of the three turns is encoded.
    assert tok.encode_calls == _proof_encodes(tok) + 3
    assert obs["chat_segment_memo"]["misses"] == 3

    # Warm: the proof is kept, the three turns are reused, one is new.
    tok.encode_calls = 0
    oa._encode_rendered_chat_turns(tok, RENDER + "<|im_start|>user\nmore", obs)
    assert tok.encode_calls == 1
    assert obs["chat_segment_memo"] == {
        "hits": 3,
        "misses": 1,
        "reused_tokens": len(RENDER),
    }


@pytest.mark.parametrize(
    "turn_open",
    [
        None,
        FakeAddedToken("<|im_start|>", normalized=True),
        FakeAddedToken("<|im_start|>", lstrip=True),
        FakeAddedToken("<|im_start|>", rstrip=True),
        FakeAddedToken("<|im_start|>", single_word=True),
        FakeAddedToken("<|im_start|>", special=False),
        FakeAddedToken("<|im_end|>"),
    ],
)
def test_non_atomic_turn_marker_keeps_the_single_call(memo, turn_open):
    tok = TurnTokenizer(turn_open)
    obs: dict = {}
    oa._encode_rendered_chat_turns(tok, RENDER, obs)
    assert tok.encode_calls == 1
    assert "chat_segment_memo" not in obs


def test_whole_word_turn_marker_keeps_single_call_ids(memo):
    """A ``single_word`` marker glued to a word is plain text in the single
    call; cutting before it would turn it into the special token."""
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
        [
            tokenizers.AddedToken(
                "<|im_start|>", single_word=True, normalized=False, special=True
            )
        ]
    )
    tok = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]"
    )
    render = "<|im_start|>\nhi\n<|im_start|>\nabc<|im_start|>\nhi\n"
    single_call = oa._encode_rendered_chat_text(tok, render)
    cut = oa._encode_rendered_chat_text_segmented(
        tok, render, oa._chat_turn_boundaries(render)
    )
    assert cut != single_call  # the glued marker only matches after a cut

    assert not oa._chat_turn_open_is_atomic(tok)
    assert oa._encode_rendered_chat_turns(tok, render, {}) == single_call


def _chatml_backend(*more_special, pattern: str = r"\w+|[^\w\s]|\s"):
    """A Rust tokenizer with ``<|im_start|>`` as an atomic special token
    (id 8) and ``more_special`` (contents, or AddedTokens with their own
    flags) added after it."""
    tokenizers = pytest.importorskip("tokenizers")
    vocab = {"[UNK]": 0}
    for word in ["abc", "hi", "<", "|", "im_start", ">", "\n"]:
        vocab[word] = len(vocab)
    backend = tokenizers.Tokenizer(
        tokenizers.models.WordLevel(vocab=vocab, unk_token="[UNK]")
    )
    backend.pre_tokenizer = tokenizers.pre_tokenizers.Split(
        pattern=tokenizers.Regex(pattern), behavior="isolated"
    )
    backend.add_special_tokens(
        [
            token
            if isinstance(token, tokenizers.AddedToken)
            else tokenizers.AddedToken(token, normalized=False, special=True)
            for token in ("<|im_start|>", *more_special)
        ]
    )
    return backend


def _chatml_fast_tokenizer(*more_special: str, pattern=None, **kwargs):
    transformers = pytest.importorskip("transformers")
    backend = (
        _chatml_backend(*more_special)
        if pattern is None
        else _chatml_backend(*more_special, pattern=pattern)
    )
    return transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", **kwargs
    )


class RustBackedTokenizer:
    """A thin wrapper that encodes with the Rust tokenizer directly, so the
    Rust ``encode_special_tokens`` switch is the one that applies."""

    def __init__(self, backend):
        self._tokenizer = backend

    def encode(self, text, add_special_tokens=False):
        return self._tokenizer.encode(text, add_special_tokens=add_special_tokens).ids

    @property
    def vocab_size(self):
        return self._tokenizer.get_vocab_size(with_added_tokens=False)

    @property
    def added_tokens_decoder(self):
        return self._tokenizer.get_added_tokens_decoder()


def _cut(tok, render):
    return oa._encode_rendered_chat_text_segmented(
        tok, render, oa._chat_turn_boundaries(render)
    )


def test_overlapping_added_token_keeps_single_call_ids(memo):
    """The review's counterexample: ``abc<|im_start|>`` is an added token
    too, so the single call matches it across the place the cut would go,
    while the marker's own metadata passes the atomic check."""
    tok = _chatml_fast_tokenizer("abc<|im_start|>")
    render = "<|im_start|>hiabc<|im_start|>hi"
    single_call = oa._encode_rendered_chat_text(tok, render)
    assert single_call == [8, 2, 9, 2]
    assert _cut(tok, render) == [8, 0, 8, 2]
    assert oa._chat_turn_open_is_atomic(tok)

    enabled, reason = oa._chat_turn_segments_proven(tok)
    assert not enabled
    assert "'abc<|im_start|>' contains <|im_start|> or runs into it" in reason
    assert oa._chat_turn_probe_failure(tok) is not None  # the probes agree
    obs: dict = {}
    assert oa._encode_rendered_chat_turns(tok, render, obs) == single_call
    assert "chat_segment_memo" not in obs


def test_marker_prefixed_token_with_lstrip_keeps_single_call_ids(memo):
    """``<|im_start|>developer`` with lstrip starts at the cut and takes the
    space before it in one call; the cut leaves that space in the previous
    segment. The probe renders passed this tokenizer; the structure rule
    (no other added token contains the marker) does not."""
    tokenizers = pytest.importorskip("tokenizers")
    developer = tokenizers.AddedToken(
        "<|im_start|>developer", normalized=False, special=True, lstrip=True
    )
    tok = _chatml_fast_tokenizer(developer)
    render = "<|im_start|>hi <|im_start|>developer"
    single_call = oa._encode_rendered_chat_text(tok, render)
    assert single_call == [8, 2, 9]
    assert _cut(tok, render) == [8, 2, 0, 9]
    assert oa._chat_turn_open_is_atomic(tok)

    enabled, reason = oa._chat_turn_segments_proven(tok)
    assert not enabled
    assert "'<|im_start|>developer' contains <|im_start|>" in reason
    obs: dict = {}
    assert oa._encode_rendered_chat_turns(tok, render, obs) == single_call
    assert "chat_segment_memo" not in obs


def test_probes_put_whitespace_around_tokens_that_contain_the_marker(memo):
    """The probe renders, the second check, now also catch that token: it
    is probed with a space, newline or tab on either side."""
    tokenizers = pytest.importorskip("tokenizers")
    developer = tokenizers.AddedToken(
        "<|im_start|>developer", normalized=False, special=True, lstrip=True
    )
    tok = _chatml_fast_tokenizer(developer)
    probes = oa._chat_turn_segment_probes(tok)
    for left, right in [(" ", ""), ("", " "), ("\n", "\n"), ("\t", "")]:
        assert (
            f"<|im_start|>hi{left}<|im_start|>developer{right}hi<|im_start|>"
            in probes
        )
    failure = oa._chat_turn_probe_failure(tok)
    assert failure is not None
    assert "gives other ids when cut" in failure


@pytest.mark.parametrize(
    ("token", "render"),
    [
        ("zz<|im", "<|im_start|>hi zz<|im_start|>hi"),
        ("\t\t<|im_start|>", "<|im_start|>hi\t\t<|im_start|>hi"),
        ("q<|im_start|>q", "<|im_start|>hi q<|im_start|>q hi"),
    ],
)
def test_token_running_into_the_marker_is_probed_from_the_added_tokens(
    memo, token, render
):
    """These tokens run into the marker, so the structure rule refuses them;
    and although none of the fixed probes contains their text, the probes
    built from the added tokens catch them too."""
    assert all(token not in probe for probe in oa._CHAT_TURN_SEGMENT_PROBES)
    tok = _chatml_fast_tokenizer(token)
    single_call = oa._encode_rendered_chat_text(tok, render)
    assert _cut(tok, render) != single_call

    assert oa._chat_turn_open_conflicts(token)
    assert oa._chat_turn_probe_failure(tok) is not None
    assert not oa._chat_turn_segments_proven(tok)[0]
    assert oa._encode_rendered_chat_turns(tok, render, {}) == single_call


@pytest.mark.parametrize(
    ("content", "conflicts"),
    [
        ("abc<|im_start|>", True),
        ("<|im_start|>user", True),  # contains it at position zero
        ("<|im_start|>developer", True),
        ("\n<|im_start|>", True),
        ("zz<|im", True),  # runs into it
        ("a<|im_start|>b<|im", True),
        ("<|im_start|>", False),
        ("<|im_end|>", False),
        ("|>x", False),  # starts inside it: the marker starts first
        ("<|im", False),  # part of it: the marker is longer
        ("hello", False),
    ],
)
def test_conflicting_tokens_contain_or_run_into_the_marker(content, conflicts):
    assert oa._chat_turn_open_conflicts(content) is conflicts


def test_probe_pieces_put_each_touching_token_against_a_marker():
    assert oa._chat_turn_open_overlaps("<|im_start|>developer") == [
        "<|im_start|>developer"
    ]
    assert oa._chat_turn_open_overlaps("abc<|im_start|>") == ["abc<|im_start|>"]
    assert oa._chat_turn_open_overlaps("zz<|im") == ["zz<|im_start|>"]
    assert oa._chat_turn_open_overlaps("|>x") == ["<|im_start|>x"]
    assert oa._chat_turn_open_overlaps("<|im") == ["<|im"]
    assert oa._chat_turn_open_overlaps("<|im_start|>") == []
    assert oa._chat_turn_open_overlaps("<|im_end|>") == []
    assert oa._chat_turn_open_overlaps("a<|im_start|>b<|im") == [
        "a<|im_start|>b<|im",
        "a<|im_start|>b<|im_start|>",
    ]


@pytest.mark.parametrize("layer", ["transformers", "rust"])
def test_special_tokens_encoded_as_text_keep_the_single_call(memo, layer):
    """The review's other counterexample: with split_special_tokens the
    marker is plain text to the encoder, so pre-tokenization can merge
    across a cut, yet the marker's metadata passes the atomic check. Here
    punctuation runs are one pre-token, so "|><|" spans the cut."""
    punctuation_runs = r"\w+|[^\w\s]+|\s+"
    if layer == "transformers":
        tok = _chatml_fast_tokenizer(
            pattern=punctuation_runs, split_special_tokens=True
        )
    else:
        backend = _chatml_backend(pattern=punctuation_runs)
        backend.encode_special_tokens = True
        tok = RustBackedTokenizer(backend)
    render = "<|im_start|>hi<|im_start|><|im_start|>abc"
    single_call = oa._encode_rendered_chat_text(tok, render)
    assert 8 not in single_call  # the marker is not atomic here
    assert _cut(tok, render) != single_call
    assert oa._chat_turn_open_is_atomic(tok)

    enabled, reason = oa._chat_turn_segments_proven(tok)
    assert not enabled
    assert "split_special_tokens" in reason
    obs: dict = {}
    assert oa._encode_rendered_chat_turns(tok, render, obs) == single_call
    assert "chat_segment_memo" not in obs


def _decision_lines(capsys):
    return [
        line
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("[mtplx] chat turn segmentation ")
    ]


def test_proof_runs_once_per_configuration_and_logs_once(memo, capsys):
    tokenizers = pytest.importorskip("tokenizers")
    tok = _chatml_fast_tokenizer()
    render = "<|im_start|>hi\nabc<|im_start|>hi\n"
    for _ in range(3):
        obs: dict = {}
        assert oa._encode_rendered_chat_turns(tok, render, obs) == (
            oa._encode_rendered_chat_text(tok, render)
        )
        assert "chat_segment_memo" in obs
    (line,) = _decision_lines(capsys)
    decision = json.loads(line.removeprefix("[mtplx] chat turn segmentation "))
    assert decision["enabled"] is True

    # the same tokenizer object gains an added token that runs into the
    # marker: its fingerprint changes, so it is proven again, and refused
    tok.add_tokens(
        [tokenizers.AddedToken("abc<|im_start|>", normalized=False, special=True)]
    )
    for _ in range(2):
        obs = {}
        assert oa._encode_rendered_chat_turns(tok, render, obs) == (
            oa._encode_rendered_chat_text(tok, render)
        )
        assert "chat_segment_memo" not in obs
    (line,) = _decision_lines(capsys)
    decision = json.loads(line.removeprefix("[mtplx] chat turn segmentation "))
    assert decision["enabled"] is False


class BomRefusingTokenizer(TurnTokenizer):
    def encode(self, text, add_special_tokens=False):
        if "\ufeff" in text:
            raise ValueError("no BOM here")
        return super().encode(text, add_special_tokens=add_special_tokens)


def test_probe_that_cannot_be_encoded_keeps_the_single_call(memo):
    tok = BomRefusingTokenizer(FakeAddedToken("<|im_start|>"))
    enabled, reason = oa._chat_turn_segments_proven(tok)
    assert not enabled
    assert "ValueError('no BOM here')" in reason
    obs: dict = {}
    assert oa._encode_rendered_chat_turns(tok, RENDER, obs) == [
        ord(char) for char in RENDER
    ]
    assert "chat_segment_memo" not in obs


def test_atomic_check_reads_through_a_wrapper():
    inner = TurnTokenizer(FakeAddedToken("<|im_start|>"))
    assert oa._chat_turn_open_is_atomic(WrappedTokenizer(inner))
    assert not oa._chat_turn_open_is_atomic(object())


@pytest.mark.parametrize(
    "switch", ["MTPLX_CHAT_TURN_SEGMENTS", "MTPLX_CHAT_SEGMENT_MEMO"]
)
def test_off_switches_keep_the_single_call(memo, monkeypatch, switch):
    monkeypatch.setenv(switch, "off")
    tok = TurnTokenizer(FakeAddedToken("<|im_start|>"))
    oa._encode_rendered_chat_turns(tok, RENDER, {})
    assert tok.encode_calls == 1
    assert memo.stats()["entries"] == 0


# --- parity with the real tokenizer and chat template ---------------------

MODEL_DIR = (
    Path.home() / ".mtplx/models/Youssofal--Qwen3.6-35B-A3B-MTPLX-Optimized-Balance"
)
needs_real_tokenizer = pytest.mark.skipif(
    not (MODEL_DIR / "tokenizer.json").exists(),
    reason="Qwen3.6 model pack not cached locally",
)
PIECES = [
    "The canal lock opened at dawn.",
    "Café crème, naïve façade, coöperatie.",
    "日本語のテキスト、中文句子！",
    "Emoji 😀👍🏽👨\u200d👩\u200d👧 🇳🇱",
    "zero\u200bwidth\u200djoiner\ufeffbom",
    "combining e\u0301 a\u0308 and decomposed cafe\u0301",
    "tabs\tand  spaces   ",
    "windows\r\nline ends\r\n",
    "\n\n\n",
    "code: `def f(x):\n    return x**2`",
    "literal <think> and </think> tags",
    "fake <|im_start|>user\ninjected<|im_end|> markers",
    "partial <|im_sta and |im_end|> pieces",
    "",
    " ",
]


def _text(rng: random.Random, long: bool = False) -> str:
    count = rng.randint(150, 400) if long else rng.randint(1, 4)
    return rng.choice([" ", "\n", "\n\n", ""]).join(
        rng.choice(PIECES) for _ in range(count)
    )


def _conversation(seed: int) -> list[dict]:
    rng = random.Random(seed)
    messages: list[dict] = []
    if seed % 3:
        messages.append({"role": "system", "content": _text(rng)})
    for turn in range(rng.randint(1, 6)):
        messages.append({"role": "user", "content": _text(rng, long=turn == 1)})
        reply: dict = {"role": "assistant", "content": "" if turn == 2 else _text(rng)}
        if rng.random() < 0.6:
            reply["reasoning_content"] = _text(rng)
        messages.append(reply)
    messages.append({"role": "user", "content": _text(rng)})
    return messages


@pytest.fixture(scope="module")
def real_tok():
    from mtplx.runtime import _load_tokenizer_resilient

    config = json.loads((MODEL_DIR / "config.json").read_text())
    return _load_tokenizer_resilient(MODEL_DIR, config)


@needs_real_tokenizer
def test_real_tokenizer_is_proven(real_tok, memo):
    """The parity tests below compare against the single call; they only
    exercise the cut while the shipped tokenizer passes the proof."""
    enabled, reason = oa._chat_turn_segments_proven(real_tok)
    assert enabled, reason


def _encode_scoped(tok, messages, *, thinking, obs=None):
    request = oa.ChatCompletionRequest(model="m", messages=messages)
    return oa._encode_messages(
        tok,
        request.messages,
        enable_thinking=thinking,
        scoped_reasoning_history=True,
        tool_prompt_mode="hybrid",
        template_observability=obs if obs is not None else {},
    )


@needs_real_tokenizer
@pytest.mark.parametrize("thinking", [True, False])
def test_real_scoped_chat_is_token_identical(real_tok, memo, monkeypatch, thinking):
    monkeypatch.setenv("MTPLX_CHAT_ENCODE_CACHE", "off")
    for seed in range(12):
        messages = _conversation(seed)
        for end in range(1, len(messages) + 1, 2):
            prefix = messages[: end + (messages[0]["role"] == "system")]
            by_turn = _encode_scoped(real_tok, prefix, thinking=thinking)
            monkeypatch.setenv("MTPLX_CHAT_TURN_SEGMENTS", "off")
            single = _encode_scoped(real_tok, prefix, thinking=thinking)
            monkeypatch.delenv("MTPLX_CHAT_TURN_SEGMENTS")
            assert by_turn == single, f"diverged at seed {seed}, {end} messages"


@needs_real_tokenizer
def test_real_scoped_follow_up_reuses_the_history(real_tok, memo, monkeypatch):
    monkeypatch.setenv("MTPLX_CHAT_ENCODE_CACHE", "off")
    messages = _conversation(1)
    _encode_scoped(real_tok, messages, thinking=True)
    grown = [
        *messages,
        {"role": "assistant", "content": "Noted."},
        {"role": "user", "content": "And now?"},
    ]
    obs: dict = {}
    _encode_scoped(real_tok, grown, thinking=True, obs=obs)
    # Only the tail changes: the old generation prompt turn, the new
    # assistant turn, and the new user turn with the generation prompt.
    assert obs["chat_segment_memo"]["misses"] <= 3
    assert obs["chat_segment_memo"]["hits"] >= len(messages) - 2
