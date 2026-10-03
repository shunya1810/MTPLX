"""Gemma 4 tokenizer detection without materializing the vocabulary.

``get_vocab()`` on a fast tokenizer builds a dict of the whole vocabulary on
every call (~47 ms for the 248k-token Qwen3.6 tokenizer), and the detection
ran on every chat request of a non-Gemma model. Fast tokenizers answer the
same membership question per token through the backend's ``token_to_id``.
"""

from __future__ import annotations

import pytest

from mtplx.chat_encoding import is_gemma4_tokenizer

tokenizers = pytest.importorskip("tokenizers")
transformers = pytest.importorskip("transformers")

GEMMA4_MARKERS = ["<|think|>", "<|channel>", "<channel|>", "<|turn>", "<turn|>"]


def _fast_tokenizer(extra_tokens: list[str]):
    vocab = {"[UNK]": 0, "hello": 1}
    for token in extra_tokens:
        vocab[token] = len(vocab)
    backend = tokenizers.Tokenizer(
        tokenizers.models.WordLevel(vocab=vocab, unk_token="[UNK]")
    )
    return transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]"
    )


def _forbid_get_vocab(monkeypatch, tokenizer):
    def _fail(*_args, **_kwargs):
        raise AssertionError("get_vocab() materializes the whole vocabulary")

    monkeypatch.setattr(tokenizer, "get_vocab", _fail)


def test_fast_gemma4_tokenizer_detected_without_get_vocab(monkeypatch):
    tokenizer = _fast_tokenizer(GEMMA4_MARKERS)
    _forbid_get_vocab(monkeypatch, tokenizer)

    assert is_gemma4_tokenizer(tokenizer) is True


def test_fast_non_gemma_tokenizer_rejected_without_get_vocab(monkeypatch):
    tokenizer = _fast_tokenizer(["<|im_start|>", "<think>"])
    _forbid_get_vocab(monkeypatch, tokenizer)

    assert is_gemma4_tokenizer(tokenizer) is False


def test_partial_marker_set_is_not_gemma4(monkeypatch):
    tokenizer = _fast_tokenizer(GEMMA4_MARKERS[:-1])
    _forbid_get_vocab(monkeypatch, tokenizer)

    assert is_gemma4_tokenizer(tokenizer) is False


def test_wrapped_fast_tokenizer_uses_inner_backend(monkeypatch):
    class Wrapper:  # mlx-lm TokenizerWrapper shape: the HF tokenizer inside
        def __init__(self, inner):
            self._tokenizer = inner

    inner = _fast_tokenizer(GEMMA4_MARKERS)
    _forbid_get_vocab(monkeypatch, inner)

    assert is_gemma4_tokenizer(Wrapper(inner)) is True


def test_tokenizer_without_backend_falls_back_to_get_vocab():
    class VocabOnly:
        def __init__(self, tokens):
            self._vocab = {token: i for i, token in enumerate(tokens)}

        def get_vocab(self):
            return dict(self._vocab)

    assert is_gemma4_tokenizer(VocabOnly(GEMMA4_MARKERS)) is True
    assert is_gemma4_tokenizer(VocabOnly(["<think>"])) is False
