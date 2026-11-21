"""inkling2 reasoning codec: Inkling 2 message channels -> think tags /
<tool_call> blocks, whole-text and streamed, plus the native-tool tokenizer
check the message normalizer uses."""

import json
import unittest

from mtplx.backends.descriptors import (
    INKLING2_AR_DESCRIPTOR,
    model_family_from_inspection,
    reasoning_policy_for_model,
    sampler_defaults_for_model,
)
from mtplx.chat_encoding import is_inkling2_tokenizer
from mtplx.reasoning_codecs import (
    Inkling2MarkupTranscoder,
    inkling2_to_think_markup,
    split_reasoning_text,
    stream_splitter_for_parser,
)

ANSWER = (
    "<|message_model|><|content_thinking|>Think A.<|end_message|>"
    "<|message_model|><|content_text|>Answer B.<|end_message|>"
    "<|content_model_end_sampling|>"
)
TOOL = (
    "<|message_model|><|content_thinking|>need tool<|end_message|>"
    "<|message_model|>get_weather<|content_invoke_tool_json|>"
    '{"name":"get_weather","args":{"city":"Tokyo"}}<|end_message|>'
    "<|content_model_end_sampling|>"
)


class _Backend:
    def __init__(self, vocab):
        self._vocab = vocab

    def token_to_id(self, token):
        return self._vocab.get(token)


class _Tok:
    def __init__(self, tokens):
        self.backend_tokenizer = _Backend({t: i for i, t in enumerate(tokens)})


class TestInkling2Codec(unittest.TestCase):
    def test_whole_text(self):
        self.assertEqual(inkling2_to_think_markup(ANSWER), "<think>Think A.</think>Answer B.")
        self.assertEqual(
            inkling2_to_think_markup(TOOL),
            '<think>need tool</think><tool_call>{"name":"get_weather","args":{"city":"Tokyo"}}</tool_call>',
        )
        self.assertEqual(inkling2_to_think_markup("plain answer"), "plain answer")

    def test_split(self):
        parts = split_reasoning_text(ANSWER, parser="inkling2", thinking_enabled=True)
        self.assertEqual((parts.reasoning, parts.content), ("Think A.", "Answer B."))
        off = split_reasoning_text(ANSWER, parser="inkling2", thinking_enabled=False)
        self.assertEqual(off.content, "Answer B.")

    def test_stream_matches_whole_text_at_any_chunking(self):
        for text in (ANSWER, TOOL):
            whole = inkling2_to_think_markup(text)
            for size in (1, 2, 5, 13, 64):
                tr = Inkling2MarkupTranscoder()
                out = "".join(tr.feed(text[i : i + size]) for i in range(0, len(text), size))
                out += tr.finish()
                self.assertEqual(out, whole, f"chunk {size}")

    def test_unterminated_thinking_closes(self):
        tr = Inkling2MarkupTranscoder()
        out = tr.feed("<|message_model|><|content_thinking|>still thinking") + tr.finish()
        self.assertEqual(out, "<think>still thinking</think>")

    def test_stream_splitter_fields(self):
        sp = stream_splitter_for_parser("inkling2", thinking_enabled=True)
        chunks = []
        for i in range(0, len(ANSWER), 7):
            chunks += sp.feed(ANSWER[i : i + 7])
        chunks += sp.finish()
        reasoning = "".join(t for f, t in chunks if f == "reasoning_content")
        content = "".join(t for f, t in chunks if f == "content")
        self.assertEqual((reasoning, content), ("Think A.", "Answer B."))

    def test_tool_payload_parses(self):
        body = inkling2_to_think_markup(TOOL).split("<tool_call>")[1].split("</tool_call>")[0]
        payload = json.loads(body)
        self.assertEqual((payload["name"], payload["args"]), ("get_weather", {"city": "Tokyo"}))

    def test_native_tool_tokenizer(self):
        ink = _Tok(["<|message_model|>", "<|message_tool|>", "<|content_invoke_tool_json|>", "<|end_message|>"])
        self.assertTrue(is_inkling2_tokenizer(ink))
        self.assertFalse(is_inkling2_tokenizer(_Tok(["<|im_start|>", "<|im_end|>"])))

    def test_family_policy(self):
        ref = "/models/inkling2-mini"
        self.assertEqual(model_family_from_inspection(model_ref=ref, descriptor=INKLING2_AR_DESCRIPTOR), "inkling2")
        codec = reasoning_policy_for_model(ref, None, INKLING2_AR_DESCRIPTOR)
        self.assertEqual(codec.parser, "inkling2")
        self.assertEqual(codec.default_effort, "high")
        sampler = sampler_defaults_for_model(ref, None, INKLING2_AR_DESCRIPTOR)
        self.assertEqual((sampler.temperature, sampler.top_p, sampler.top_k), (1.0, 1.0, 0))


if __name__ == "__main__":
    unittest.main()
