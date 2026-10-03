"""JSON with comments and trailing commas, as the client apps read it.

OpenCode parses every config file, ``opencode.json`` included, with
``jsonc-parser`` and ``allowTrailingComma`` (``packages/opencode/src/config/
parse.ts``); Pi strips ``//`` comments and trailing commas from
``models.json`` (``packages/coding-agent/src/utils/json.ts``). A file those
tools accept is a working config, not a broken one, so MTPLX reads it the same
way before merging its own provider into it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class InvalidConfigFile(ValueError):
    """A client config file MTPLX cannot read even with comments and trailing
    commas allowed. The file is left exactly as it was."""

    def __init__(self, path: Path, reason: str) -> None:
        self.path = Path(path)
        self.reason = reason
        super().__init__(f"{self.path} could not be read: {reason}.")


def loads(text: str) -> Any:
    """Parse JSON, also accepting ``//`` and ``/* */`` comments, trailing
    commas and a leading byte order mark (Pi strips one before parsing).
    Raises ``json.JSONDecodeError`` positioned in the original text.
    SYNC: ``ClientConfigFile`` in the app."""

    text = text.removeprefix("\ufeff")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    return json.loads(_blank_trailing_commas(_blank_comments(text)))


def load_config_file(path: str | Path) -> tuple[dict[str, Any], str]:
    """Read a client config file and return ``(object, original_text)``.

    Raises ``InvalidConfigFile`` when the text does not parse or its top-level
    value is not an object; ``OSError`` from reading propagates.
    """

    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")
    try:
        parsed = loads(text)
    except json.JSONDecodeError as exc:
        raise InvalidConfigFile(
            config_path, f"{exc.msg} at line {exc.lineno}, column {exc.colno}"
        ) from exc
    if not isinstance(parsed, dict):
        raise InvalidConfigFile(
            config_path, f"the top-level value is {type(parsed).__name__}, not an object"
        )
    return parsed, text


def _blank_comments(text: str) -> str:
    """Replace ``//`` and ``/* */`` comments outside strings with spaces,
    keeping newlines so every offset in the result matches the input."""

    out = list(text)
    n = len(text)
    i = 0
    in_string = False
    while i < n:
        ch = text[i]
        if in_string:
            if ch == "\\":
                i += 2
                continue
            if ch == '"':
                in_string = False
            i += 1
            continue
        if ch == '"':
            in_string = True
            i += 1
            continue
        if ch == "/" and i + 1 < n and text[i + 1] == "/":
            while i < n and text[i] != "\n":
                out[i] = " "
                i += 1
            continue
        if ch == "/" and i + 1 < n and text[i + 1] == "*":
            end = text.find("*/", i + 2)
            end = n if end < 0 else end + 2
            for k in range(i, end):
                if text[k] != "\n":
                    out[k] = " "
            i = end
            continue
        i += 1
    return "".join(out)


def _blank_trailing_commas(text: str) -> str:
    """Replace a comma that is followed only by whitespace and then ``}`` or
    ``]`` with a space, outside strings."""

    out = list(text)
    n = len(text)
    i = 0
    in_string = False
    while i < n:
        ch = text[i]
        if in_string:
            if ch == "\\":
                i += 2
                continue
            if ch == '"':
                in_string = False
            i += 1
            continue
        if ch == '"':
            in_string = True
        elif ch == ",":
            j = i + 1
            while j < n and text[j] in " \t\r\n":
                j += 1
            if j < n and text[j] in "}]":
                out[i] = " "
        i += 1
    return "".join(out)
