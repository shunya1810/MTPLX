"""Qwen XML tool parameters keep their payload whitespace; only the framing newlines go.

The template renders '<parameter=NAME>\\n' + value + '\\n</parameter>'. Every
surrounding whitespace used to be stripped: a written file lost its trailing
newline (or its first line's indentation), the history the client sent back no
longer matched what the model generated, and the session bank refused the turn's
snapshot (omp, 2026-10-04: the next call prefilled 74K tokens of thinking again).
"""

from __future__ import annotations

import json

import mtplx.server.omlx_bridge.tool_calling as tc
import mtplx.server.openai as srv

CONTENT = "  indented first line\n<html>\n</html>\n"


def _xml(value: str) -> str:
    return (
        "<tool_call>\n<function=write>\n<parameter=path>\nkyoto.html\n</parameter>\n"
        f"<parameter=content>\n{value}\n</parameter>\n<parameter=n>\n42\n</parameter>\n</function>\n</tool_call>"
    )


def test_the_final_parser_keeps_the_payload_whitespace():
    _text, calls, _err = tc._parse_xml_tool_calls(_xml(CONTENT))
    args = json.loads(calls[0]["function"]["arguments"])
    assert args == {"path": "kyoto.html", "content": CONTENT, "n": 42}


def test_single_line_values_stay_trimmed():
    assert tc._qwen_xml_param("\n kyoto.html \n") == "kyoto.html"
    assert srv._decode_tool_parameter_value(" npm run build\n") == "npm run build"


def test_the_decoder_keeps_multiline_strings_whole():
    assert srv._decode_tool_parameter_value(CONTENT) == CONTENT
    assert srv._decode_tool_parameter_value('{"a": 1}') == {"a": 1}
    assert srv._decode_tool_parameter_value("<string>abc</string>", schema={"type": "string"}) == "abc"


def test_the_stream_parser_drops_only_the_framing_newline_whatever_the_chunks():
    text = _xml(CONTENT)
    for size in (1, 3, 7, len(text)):
        parser = srv._QwenXMLToolCallStreamParser(tools=[{"type": "function", "function": {"name": "write", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}, "n": {"type": "integer"}}}}}])
        for i in range(0, len(text), size):
            parser.feed(text[i : i + size])
        parser.finish()
        calls = parser.tool_calls
        args = calls[0]["function"]["arguments"]
        args = json.loads(args) if isinstance(args, str) else args
        assert args["content"] == CONTENT, size
        assert args["path"] == "kyoto.html"
