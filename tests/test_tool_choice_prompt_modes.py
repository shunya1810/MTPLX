"""Tool choice is a request constraint in every schema rendering mode."""

from copy import deepcopy

import pytest
from test_parallel_tool_prompt_policy import GemmaTokenizer
from test_server_openai import JSONToolPrefixTokenizer, _fake_state, _named_tool_schema

from mtplx.server import openai
from mtplx.server.request_policy import resolve_request_policy


@pytest.mark.parametrize("mode", ["native", "hybrid", "compact"])
@pytest.mark.parametrize("tokenizer_type", [GemmaTokenizer, JSONToolPrefixTokenizer])
@pytest.mark.parametrize("thinking", [False, True])
def test_tool_choice_changes_prompt_in_every_rendering_mode(
    monkeypatch, mode, tokenizer_type, thinking
):
    monkeypatch.setenv("MTPLX_AGENT_REWRITES", "off")
    tokenizer = tokenizer_type()
    messages = [
        openai.ChatMessage(role="system", content="Answer in French."),
        openai.ChatMessage(role="user", content="Read the file."),
    ]
    original = deepcopy(messages)
    tools = [_named_tool_schema("read"), _named_tool_schema("write")]
    choices = ["auto", "required", {"type": "function", "function": {"name": "read"}}]
    prompts = []
    observations = []
    for choice in choices:
        obs = {}
        ids = openai._encode_messages(
            tokenizer,
            messages,
            enable_thinking=thinking,
            tools=tools,
            tool_choice=choice,
            tool_prompt_mode=mode,
            template_observability=obs,
        )
        prompts.append(tokenizer.decode(ids))
        observations.append(obs)

    auto, required, named = prompts
    assert required != auto
    assert named != auto and named != required
    assert "This request requires one declared tool call" in required
    assert "This request requires the `read` tool call" in named
    head = openai._MTPLX_FORCED_TOOL_CHOICE_SENTINEL_HEAD
    assert head not in auto
    for prompt, obs in zip(prompts[1:], observations[1:]):
        assert prompt.count(head) == 1
        assert prompt.index(head) > prompt.index("Read the file.")
        assert "Answer in French." in prompt
        assert obs["forced_tool_choice_sentinel_injected"] is True
    # Per-turn choice must leave the reusable system/history prefix intact.
    prefix = required[:required.index(head)]
    assert named.startswith(prefix)
    assert messages == original


@pytest.mark.parametrize("client_hint", ["pi", "hermes"])
@pytest.mark.parametrize("tokenizer_type", [GemmaTokenizer, JSONToolPrefixTokenizer])
def test_declared_parallel_policy_preserves_choice_after_tool_result(
    monkeypatch, client_hint, tokenizer_type
):
    monkeypatch.setenv("MTPLX_AGENT_REWRITES", "off")
    state = _fake_state()
    tokenizer = tokenizer_type()
    request = openai.ChatCompletionRequest(
        messages=[
            openai.ChatMessage(role="system", content="Keep changes minimal."),
            openai.ChatMessage(role="user", content="Read then write the file."),
            openai.ChatMessage(role="assistant", content="", tool_calls=[{
                "id": "read_1", "type": "function",
                "function": {"name": "read", "arguments": "{}"},
            }]),
            openai.ChatMessage(role="tool", tool_call_id="read_1", content="File contents."),
        ],
        tools=[_named_tool_schema("read"), _named_tool_schema("write")],
        tool_choice={"type": "function", "function": {"name": "write"}},
        parallel_tool_calls=True,
    )
    policy = resolve_request_policy(
        state, request, headers={"x-mtplx-client": client_hint}, metadata={}, endpoint="chat"
    )
    ids = openai._encode_messages(
        tokenizer, request.messages, enable_thinking=False,
        tools=request.tools, tool_choice=request.tool_choice,
        tool_prompt_mode=policy.template_tool_prompt_mode,
    )
    rendered = tokenizer.decode(ids)

    assert "This request requires the `write` tool call" in rendered
    assert rendered.index("File contents.") < rendered.index("This request requires")
    assert "Keep changes minimal." in rendered
