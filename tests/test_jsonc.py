"""mtplx.jsonc: the JSON-with-comments reader used for OpenCode and Pi configs."""

from __future__ import annotations

import json

import pytest

from mtplx import jsonc

JSONC_CONFIG = """{
  // OpenCode's own example uses a trailing comma and comments.
  "$schema": "https://opencode.ai/config.json",
  "model": "anthropic/claude-sonnet-4-5", /* default model */
  "provider": {
    "lmstudio": {"name": "LM Studio", "options": {"baseURL": "http://localhost:1234/v1"},},
  },
  "keybinds": {"leader": "ctrl+x", "note": "a // is not a comment inside a string, nor is /* this */",},
}
"""


def test_loads_accepts_comments_and_trailing_commas():
    parsed = jsonc.loads(JSONC_CONFIG)
    assert parsed["model"] == "anthropic/claude-sonnet-4-5"
    assert parsed["provider"]["lmstudio"]["options"]["baseURL"] == "http://localhost:1234/v1"
    assert parsed["keybinds"]["note"] == "a // is not a comment inside a string, nor is /* this */"


def test_loads_is_plain_json_loads_for_strict_input():
    text = json.dumps({"a": [1, 2, {"b": "c,}"}], "d": "e"})
    assert jsonc.loads(text) == json.loads(text)


def test_loads_keeps_escaped_quotes_inside_strings():
    assert jsonc.loads(r'{"k": "say \"hi\" // not a comment",}') == {"k": 'say "hi" // not a comment'}


def test_loads_reports_errors_at_positions_in_the_original_text():
    text = '{\n  // comment\n  "a": 1,\n  "b": \n}\n'
    with pytest.raises(json.JSONDecodeError) as excinfo:
        jsonc.loads(text)
    # The blank-out keeps every offset, so the error points at the real line.
    assert excinfo.value.lineno == 5


def test_loads_accepts_a_byte_order_mark_as_pi_does():
    # Pi 0.85.1 strips a BOM from models.json before parsing it.
    assert jsonc.loads("\ufeff" + '{"providers": {}, // note\n}') == {"providers": {}}
    assert jsonc.loads("\ufeff{}") == {}


def test_loads_rejects_what_no_lenient_reader_accepts():
    for text in ("{bad json", "{'single': 'quotes'}", '{"a": 1 "b": 2}', ""):
        with pytest.raises(json.JSONDecodeError):
            jsonc.loads(text)


def test_load_config_file_returns_object_and_original_text(tmp_path):
    path = tmp_path / "opencode.json"
    path.write_text(JSONC_CONFIG, encoding="utf-8")
    parsed, text = jsonc.load_config_file(path)
    assert parsed["model"] == "anthropic/claude-sonnet-4-5"
    assert text == JSONC_CONFIG


def test_load_config_file_refuses_unparseable_text_with_path_and_position(tmp_path):
    path = tmp_path / "opencode.json"
    path.write_text('{\n  "provider": {\n    "x": }\n}\n', encoding="utf-8")
    with pytest.raises(jsonc.InvalidConfigFile) as excinfo:
        jsonc.load_config_file(path)
    message = str(excinfo.value)
    assert str(path) in message
    assert "line 3, column 10" in message
    assert excinfo.value.path == path


def test_load_config_file_refuses_a_non_object_top_level(tmp_path):
    path = tmp_path / "models.json"
    path.write_text("[1, 2, 3]\n", encoding="utf-8")
    with pytest.raises(jsonc.InvalidConfigFile, match="not an object"):
        jsonc.load_config_file(path)
