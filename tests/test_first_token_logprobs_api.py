"""First-token logprobs on the regular generation routes.

/v1/completions (logprobs=K, no echo) and /v1/chat/completions
(logprobs=true + top_logprobs=K) return the raw next-token distribution of
the first generated token, and only for max_tokens=1 non-stream requests:
every other shape is a clear 400, never a silent response without logprobs.
The real ``_run_generation`` runs over faked generators (the golden-matrix
harness), so the result-dict plumbing is exercised end to end.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient
from test_api_benchmark_contracts import (
    BYPASS,
    _envelope_client,
    _fake_generation_output,
    _scoring_state,
)
from test_server_openai import _mtp_batch_dispatch_state

from mtplx.server import openai
from mtplx.server.openai import create_app

# Engine-shaped first-token result (mtplx.generation.FirstTokenLogprobs);
# strings are whatever the fake state's tokenizer decodes these ids to
# (chr(id): 79 is "O").
FIRST = SimpleNamespace(
    token_id=79,
    logprob=-0.25,
    top=((79, -0.25), (75, -1.5), (80, -3.0)),
)


def _logprobs_generator(calls: list[dict], first=FIRST, text: str | None = None):
    """One generated token, and the text and stats of that one token: the
    max_tokens=1 shape every logprobs request has."""
    base = _fake_generation_output()

    def generate(*args, **kwargs):
        calls.append(kwargs)
        out = base(*args, **kwargs)
        out.tokens = [first.token_id]
        out.text = chr(first.token_id) if text is None else text
        out.stats = dataclasses.replace(out.stats, generated_tokens=1)
        out.first_token_logprobs = first
        return out

    return generate


def _decode(state, token_id: int) -> str:
    return state.runtime.tokenizer.decode([token_id])


# --- /v1/completions ---------------------------------------------------------


def test_completions_first_token_logprobs_openai_shape(monkeypatch):
    calls: list[dict] = []
    client, state = _envelope_client(
        monkeypatch, generator=_logprobs_generator(calls)
    )

    response = client.post(
        "/v1/completions",
        headers=BYPASS,
        json={"prompt": "hi", "logprobs": 2, "max_tokens": 1},
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    logprobs = choice["logprobs"]
    sampled = _decode(state, 79)
    assert choice["text"] == sampled
    assert response.json()["usage"]["completion_tokens"] == 1
    assert logprobs["tokens"] == [sampled]
    assert logprobs["token_logprobs"] == [pytest.approx(-0.25)]
    assert logprobs["text_offset"] == [0]
    assert logprobs["token_ids"] == [79]
    top = logprobs["top_logprobs"][0]
    assert top[sampled] == pytest.approx(-0.25)
    assert top[_decode(state, 75)] == pytest.approx(-1.5)
    assert calls[0]["first_token_logprobs_top_k"] == 2


def test_completions_logprobs_zero_reports_sampled_token(monkeypatch):
    calls: list[dict] = []
    client, _state = _envelope_client(
        monkeypatch, generator=_logprobs_generator(calls)
    )

    response = client.post(
        "/v1/completions",
        headers=BYPASS,
        json={"prompt": "hi", "logprobs": 0, "max_tokens": 1},
    )

    assert response.status_code == 200
    assert calls[0]["first_token_logprobs_top_k"] == 0
    logprobs = response.json()["choices"][0]["logprobs"]
    assert logprobs["token_logprobs"] == [pytest.approx(-0.25)]


def test_logprobs_request_skips_blank_retries(monkeypatch):
    """A whitespace/stop first token is a valid classifier answer: it must
    not trigger the unseeded blank-retry loop (up to 4 full generations)."""

    calls: list[dict] = []
    generate = _logprobs_generator(calls)

    def blank(*args, **kwargs):
        out = generate(*args, **kwargs)
        out.text = ""
        return out

    client, _state = _envelope_client(monkeypatch, generator=blank)

    response = client.post(
        "/v1/completions",
        headers=BYPASS,
        json={"prompt": "hi", "logprobs": 1, "max_tokens": 1},
    )

    assert response.status_code == 200
    assert len(calls) == 1


@pytest.mark.parametrize(
    ("body_extra", "message_part"),
    [
        ({"max_tokens": 8}, "max_tokens to 1"),
        ({}, "max_tokens to 1"),
        ({"max_tokens": 1, "stream": True}, "stream"),
        ({"max_tokens": 1, "stop": ["\n"]}, "stop"),
        ({"max_tokens": 1, "logprobs": 10_000}, "MTPLX_PROMPT_LOGPROBS_MAX"),
    ],
)
def test_completions_unservable_logprobs_shapes_are_400(
    monkeypatch, body_extra, message_part
):
    def _explode(*_args, **_kwargs):
        raise AssertionError("an unservable logprobs request must not generate")

    monkeypatch.setattr(openai, "_run_generation_dispatched", _explode)
    client = TestClient(create_app(_scoring_state()))

    response = client.post(
        "/v1/completions",
        json={"prompt": "hi", "logprobs": 2, **body_extra},
    )

    assert response.status_code == 400
    assert message_part in response.json()["error"]["message"]


def test_completions_missing_engine_logprobs_is_loud(monkeypatch):
    """An engine lane that ran without producing logprobs is a server
    error, never a 200 that silently drops the requested field."""

    client, _state = _envelope_client(monkeypatch)

    response = client.post(
        "/v1/completions",
        headers=BYPASS,
        json={"prompt": "hi", "logprobs": 2, "max_tokens": 1},
    )

    assert response.status_code == 500
    assert "first-token logprobs" in response.json()["error"]["message"]


# --- /v1/chat/completions ----------------------------------------------------


def test_chat_first_token_logprobs_openai_shape(monkeypatch):
    calls: list[dict] = []
    client, state = _envelope_client(
        monkeypatch, generator=_logprobs_generator(calls)
    )

    response = client.post(
        "/v1/chat/completions",
        headers=BYPASS,
        json={
            "messages": [{"role": "user", "content": "label?"}],
            "max_tokens": 1,
            "logprobs": True,
            "top_logprobs": 3,
        },
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    logprobs = choice["logprobs"]
    assert logprobs["refusal"] is None
    [entry] = logprobs["content"]
    sampled = _decode(state, 79)
    assert choice["message"]["content"] == sampled
    assert response.json()["usage"]["completion_tokens"] == 1
    assert entry["token"] == sampled
    assert entry["logprob"] == pytest.approx(-0.25)
    # The fake state's tokenizer has no decoder to read bytes from, and a
    # token's bytes are never guessed from its decoded text.
    assert entry["bytes"] is None
    assert [alt["token"] for alt in entry["top_logprobs"]] == [
        _decode(state, token) for token, _value in FIRST.top
    ]
    assert [alt["logprob"] for alt in entry["top_logprobs"]] == [
        pytest.approx(value) for _token, value in FIRST.top
    ]
    assert calls[0]["first_token_logprobs_top_k"] == 3


def _byte_level_tokenizer():
    """A real byte-level BPE tokenizer with no merges: one token per byte,
    so "é" is two tokens that each decode to U+FFFD on their own."""
    tokenizers = pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")
    alphabet = sorted(tokenizers.pre_tokenizers.ByteLevel.alphabet())
    backend = tokenizers.Tokenizer(
        tokenizers.models.BPE(
            vocab={char: index for index, char in enumerate(alphabet)}, merges=[]
        )
    )
    backend.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(
        add_prefix_space=False
    )
    backend.decoder = tokenizers.decoders.ByteLevel()
    return transformers.PreTrainedTokenizerFast(tokenizer_object=backend)


@pytest.mark.parametrize(
    "lead_is_added", [False, True], ids=["vocabulary", "registered_as_added"]
)
def test_chat_logprobs_report_the_bytes_of_byte_level_tokens(
    monkeypatch, lead_is_added
):
    """The two halves of "é" both decode to U+FFFD, so their UTF-8 is the
    same three bytes. The response must carry each token's own byte, and
    the two alternatives must stay apart. Registering the lead piece "Ã"
    as an added token changes nothing: the ids and the decode stay the
    same, and so must the bytes (not the UTF-8 of "Ã", [195, 131])."""
    tokenizer = _byte_level_tokenizer()
    if lead_is_added:
        tokenizer.add_tokens(["\u00c3"])
    lead, trail = tokenizer.encode("é", add_special_tokens=False)
    assert tokenizer.decode([lead, trail]) == "é"
    [letter] = tokenizer.encode("a", add_special_tokens=False)
    first = SimpleNamespace(
        token_id=lead,
        logprob=-0.5,
        top=((lead, -0.5), (trail, -1.25), (letter, -2.0)),
    )
    calls: list[dict] = []
    client, state = _envelope_client(
        monkeypatch,
        generator=_logprobs_generator(
            calls, first=first, text=tokenizer.decode([lead])
        ),
    )
    state.runtime.tokenizer = tokenizer

    response = client.post(
        "/v1/chat/completions",
        headers=BYPASS,
        json={
            "messages": [{"role": "user", "content": "label?"}],
            "max_tokens": 1,
            "logprobs": True,
            "top_logprobs": 3,
        },
    )

    assert response.status_code == 200
    [entry] = response.json()["choices"][0]["logprobs"]["content"]
    assert entry["token"] == "\ufffd"
    assert entry["bytes"] == [0xC3]
    assert [alt["bytes"] for alt in entry["top_logprobs"]] == [[0xC3], [0xA9], [0x61]]
    assert bytes(entry["bytes"] + entry["top_logprobs"][1]["bytes"]).decode() == "é"


def test_chat_without_logprobs_has_no_logprobs_field(monkeypatch):
    calls: list[dict] = []
    client, _state = _envelope_client(
        monkeypatch, generator=_logprobs_generator(calls)
    )

    response = client.post(
        "/v1/chat/completions",
        headers=BYPASS,
        json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 1},
    )

    assert response.status_code == 200
    assert "logprobs" not in response.json()["choices"][0]
    assert calls[0]["first_token_logprobs_top_k"] is None


@pytest.mark.parametrize(
    ("body_extra", "message_part"),
    [
        ({"top_logprobs": 3}, "requires logprobs=true"),
        ({"logprobs": True}, "max_tokens to 1"),
        ({"logprobs": True, "max_completion_tokens": 4}, "max_tokens to 1"),
        ({"logprobs": True, "max_tokens": 1, "stream": True}, "stream"),
        ({"logprobs": True, "max_tokens": 1, "stop": "x"}, "stop"),
        ({"logprobs": True, "max_tokens": 1, "top_logprobs": -1}, ">= 0"),
    ],
)
def test_chat_unservable_logprobs_shapes_are_400(
    monkeypatch, body_extra, message_part
):
    def _explode(*_args, **_kwargs):
        raise AssertionError("an unservable logprobs request must not generate")

    monkeypatch.setattr(openai, "_run_generation_dispatched", _explode)
    client = TestClient(create_app(_scoring_state()))

    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], **body_extra},
    )

    assert response.status_code == 400
    assert message_part in response.json()["error"]["message"]


# --- lane routing ------------------------------------------------------------


def test_logprobs_requests_bypass_live_ar_batch(monkeypatch):
    state = _mtp_batch_dispatch_state()
    state.args.scheduler_mode = "ar_batch"
    state.ar_batch_service = SimpleNamespace(
        submit=lambda _job: pytest.fail("logprobs must not ride the AR batch")
    )
    seen: dict = {}

    def solo(*_args, **kwargs):
        seen.update(kwargs)
        return {"route": "solo"}

    monkeypatch.setattr(openai, "_run_generation", solo)

    generated = openai._run_generation_dispatched(
        state,
        [1],
        batch_key="test.logprobs",
        generation_mode="ar",
        first_token_logprobs_top_k=5,
    )

    assert generated == {"route": "solo"}
    assert seen["first_token_logprobs_top_k"] == 5
    lane = seen["request_observability"]
    assert lane["scheduler_lane"] == "solo_logprobs"
    assert lane["ar_batch_bypass_reason"] == "first_token_logprobs"


def test_mtp_batch_rejects_logprobs_without_solo_fallback(monkeypatch):
    state = _mtp_batch_dispatch_state()
    state.mtp_batch_service = SimpleNamespace(
        submit=lambda _job: pytest.fail("no submit")
    )
    monkeypatch.setattr(
        openai,
        "_run_generation",
        lambda *_args, **_kwargs: pytest.fail("mtp_batch logprobs cannot go solo"),
    )

    with pytest.raises(openai.MTPBatchRequestError, match="does not support logprobs"):
        openai._run_generation_dispatched(
            state,
            [1],
            batch_key="test.logprobs",
            generation_mode="mtp",
            first_token_logprobs_top_k=5,
        )


# --- token bytes -------------------------------------------------------------


def _sentencepiece_tokenizer():
    """A real SentencePiece-style tokenizer like Gemma's: "\u2581" for a
    space, <0xNN> byte tokens for anything outside the vocabulary."""
    tokenizers = pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")
    vocab = {"<unk>": 0}
    for byte in range(256):
        vocab[f"<0x{byte:02X}>"] = len(vocab)
    for piece in ("\u2581hello", "\u2581", "é"):
        vocab[piece] = len(vocab)
    backend = tokenizers.Tokenizer(
        tokenizers.models.BPE(
            vocab=vocab, merges=[], unk_token="<unk>", byte_fallback=True
        )
    )
    backend.pre_tokenizer = tokenizers.pre_tokenizers.Metaspace(
        prepend_scheme="never"
    )
    backend.decoder = tokenizers.decoders.Sequence(
        [
            tokenizers.decoders.Replace("\u2581", " "),
            tokenizers.decoders.ByteFallback(),
            tokenizers.decoders.Fuse(),
        ]
    )
    return transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="<unk>"
    )


@pytest.mark.parametrize("make", [_byte_level_tokenizer, _sentencepiece_tokenizer])
def test_token_bytes_spell_the_text_back(make):
    tokenizer = make()
    text = "é hello 😀"
    token_bytes = openai._token_bytes_reader(tokenizer)
    ids = tokenizer.encode(text, add_special_tokens=False)
    assert b"".join(bytes(token_bytes(token)) for token in ids) == text.encode()


def test_sentencepiece_byte_tokens_and_spaces():
    tokenizer = _sentencepiece_tokenizer()
    token_bytes = openai._token_bytes_reader(tokenizer)
    [emoji_lead, *_rest] = tokenizer.encode("😀", add_special_tokens=False)
    assert tokenizer.convert_ids_to_tokens(emoji_lead) == "<0xF0>"
    assert token_bytes(emoji_lead) == [0xF0]
    hello = tokenizer.convert_tokens_to_ids("\u2581hello")
    assert token_bytes(hello) == list(b" hello")


def test_added_tokens_are_their_content():
    tokenizer = _byte_level_tokenizer()
    tokenizer.add_special_tokens({"additional_special_tokens": ["<|im_end|>"]})
    [end] = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    assert openai._token_bytes_reader(tokenizer)(end) == list(b"<|im_end|>")


def test_byte_level_token_outside_the_byte_table_is_its_text():
    """The ByteLevel decoder passes a token with a character outside the
    byte table through unchanged; "｜" is one."""
    tokenizer = _byte_level_tokenizer()
    tokenizer.add_tokens(["｜x", "\u00c3｜"])
    token_bytes = openai._token_bytes_reader(tokenizer)
    for text in ("｜x", "\u00c3｜"):
        token = tokenizer.convert_tokens_to_ids(text)
        assert tokenizer.decode([token]) == text
        assert token_bytes(token) == list(text.encode("utf-8"))


def _pieces_tokenizer(decoder, pieces):
    """A real tokenizer over ``pieces`` with <0xNN> byte fallback tokens and
    the given decoder."""
    tokenizers = pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")
    vocab = {"<unk>": 0}
    for byte in range(256):
        vocab[f"<0x{byte:02X}>"] = len(vocab)
    for piece in pieces:
        vocab[piece] = len(vocab)
    backend = tokenizers.Tokenizer(
        tokenizers.models.BPE(
            vocab=vocab, merges=[], unk_token="<unk>", byte_fallback=True
        )
    )
    backend.decoder = decoder
    return transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="<unk>"
    )


def test_byte_fallback_alone_keeps_the_literal_space_mark():
    """Without a Replace or Metaspace step "\u2581" is not a space: the
    tokenizer decodes it as itself, so its bytes are its UTF-8."""
    tokenizers = pytest.importorskip("tokenizers")
    tokenizer = _pieces_tokenizer(
        tokenizers.decoders.ByteFallback(), ["\u2581", "\u2581hello"]
    )
    token_bytes = openai._token_bytes_reader(tokenizer)
    mark = tokenizer.convert_tokens_to_ids("\u2581")
    assert tokenizer.decode([mark]) == "\u2581"
    assert token_bytes(mark) == [0xE2, 0x96, 0x81]
    assert token_bytes(tokenizer.convert_tokens_to_ids("\u2581hello")) == list(
        "\u2581hello".encode()
    )
    assert token_bytes(tokenizer.convert_tokens_to_ids("<0xC3>")) == [0xC3]


def test_metaspace_space_is_its_own_replacement_character():
    """A Metaspace decoder with "_" as its replacement: "_" is the space and
    "\u2581" is an ordinary character."""
    tokenizers = pytest.importorskip("tokenizers")
    tokenizer = _pieces_tokenizer(
        tokenizers.decoders.Metaspace(replacement="_", prepend_scheme="always"),
        ["_hello", "\u2581"],
    )
    token_bytes = openai._token_bytes_reader(tokenizer)
    assert token_bytes(tokenizer.convert_tokens_to_ids("_hello")) == list(b" hello")
    assert token_bytes(tokenizer.convert_tokens_to_ids("\u2581")) == [0xE2, 0x96, 0x81]


def test_unread_decoders_report_null():
    """No Rust decoder to read (the fake state's tokenizer), or a decoder
    step not modelled here (WordPiece): the bytes are unknown, not guessed."""
    tokenizers = pytest.importorskip("tokenizers")
    fake = SimpleNamespace(decode=lambda ids, **_kwargs: "".join(map(chr, ids)))
    assert openai._token_bytes_reader(fake)(ord("a")) is None

    tokenizer = _pieces_tokenizer(tokenizers.decoders.WordPiece(), ["hello"])
    token = tokenizer.convert_tokens_to_ids("hello")
    assert openai._token_bytes_reader(tokenizer)(token) is None


MODELS = Path.home() / ".mtplx/models"


@pytest.mark.parametrize(
    "pack",
    [
        MODELS / "Youssofal--Qwen3.6-35B-A3B-MTPLX-Optimized-Balance",
        MODELS / "Youssofal--Gemma4-MTPLX-Optimized-Speed" / "target",
    ],
    ids=["qwen3.6-byte-level", "gemma4-sentencepiece"],
)
def test_real_pack_token_bytes_spell_the_text_back(pack):
    if not (pack / "tokenizer.json").exists():
        pytest.skip("model pack not cached locally")
    from mtplx.runtime import _load_tokenizer_resilient

    tokenizer = _load_tokenizer_resilient(
        pack, json.loads((pack / "config.json").read_text())
    )
    text = "Café naïve 日本語 😀👍🏽 \U0001f9ec\u0f00\U0001d11e zero\u200bwidth\r\n\tend"
    ids = tokenizer.encode(text, add_special_tokens=False)
    token_bytes = openai._token_bytes_reader(tokenizer)
    assert all(token_bytes(token) is not None for token in ids)
    assert b"".join(bytes(token_bytes(token)) for token in ids) == text.encode()
