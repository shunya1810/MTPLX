"""Declarative parallel-tool routing is shared by generation and counting."""

import json
from copy import deepcopy

import pytest
from fastapi.testclient import TestClient
from test_server_openai import (
    JSONToolPrefixTokenizer,
    _fake_state,
    _fake_streaming_generation,
    _named_tool_schema,
    _stream_payloads,
)

from mtplx.chat_encoding import encode_chat_messages
from mtplx.server import openai
from mtplx.server.openai import ChatCompletionRequest, ChatMessage
from mtplx.server.request_policy import resolve_request_policy


@pytest.mark.parametrize("endpoint", ["chat", "count_tokens"])
@pytest.mark.parametrize("tool_choice", ["auto", "none"])
def test_parallel_policy_keeps_generation_and_postcommit_in_the_same_mode(
    monkeypatch, endpoint, tool_choice
):
    monkeypatch.delenv("MTPLX_AGENT_REWRITES", raising=False)
    state = _fake_state()
    request = ChatCompletionRequest(
        messages=[ChatMessage(role="user", content="Read both files.")],
        tools=[_named_tool_schema("read")],
        tool_choice=tool_choice,
        parallel_tool_calls=True,
    )
    policy = resolve_request_policy(
        state,
        request,
        headers={"x-mtplx-client": "pi"},
        metadata={},
        endpoint=endpoint,
    )
    assert policy.template_tool_prompt_mode == "native"
    assert policy.postcommit_tool_prompt_mode == "native"


class GemmaTokenizer(JSONToolPrefixTokenizer):
    bos_token = "<bos>"
    has_tool_calling = False
    model_specific_special_tokens = {
        "think_token": "<|think|>",
        "soc_token": "<|channel>",
        "eoc_token": "<channel|>",
    }


@pytest.mark.parametrize("role", ["system", "developer"])
@pytest.mark.parametrize("thinking", [False, True])
def test_gemma_native_declaration_preserves_client_instruction(role, thinking):
    tokenizer = GemmaTokenizer()
    instruction = "Answer in French and never edit files."
    messages = [
        {"role": role, "content": [{"type": "text", "text": instruction}]},
        {"role": "user", "content": "Look up both cities."},
    ]
    original = deepcopy(messages)

    ids = encode_chat_messages(
        tokenizer,
        messages,
        enable_thinking=thinking,
        tools=[_named_tool_schema("lookup")],
    )
    rendered = tokenizer.decode(ids)

    assert instruction in rendered
    assert "Available tools:" in rendered
    assert rendered.count("<|turn>system\n") == 1
    assert "exactly one XML tool call" not in rendered
    assert messages == original


@pytest.mark.parametrize("client_hint", ["pi", "hermes", "opencode", None])
@pytest.mark.parametrize("tokenizer_type", [GemmaTokenizer, JSONToolPrefixTokenizer])
@pytest.mark.parametrize("stream", [False, True])
def test_parallel_endpoint_preserves_system_and_emits_sibling_calls(
    monkeypatch, client_hint, tokenizer_type, stream
):
    state = _fake_state()
    state.runtime.tokenizer = tokenizer_type()
    state.args.tool_prompt_mode = "native"
    state.args.stats_footer = False
    instruction = "Answer in French and never edit files."
    tool = _named_tool_schema("lookup")
    tool["function"]["parameters"] = {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    }
    output = "\n".join(
        f"<tool_call>\n<function=lookup>\n<parameter=city>{city}</parameter>"
        "\n</function>\n</tool_call>"
        for city in ("Oslo", "Lima")
    )
    generation = _fake_streaming_generation(output)
    prompts = []

    def generate(server_state, prompt_ids, **kwargs):
        prompts.append(server_state.runtime.tokenizer.decode(prompt_ids))
        return generation(server_state, prompt_ids, **kwargs)

    monkeypatch.setattr(openai, "_run_generation", generate)
    headers = {"x-mtplx-cache-mode": "bypass"}
    if client_hint:
        headers["x-mtplx-client"] = client_hint
    response = TestClient(openai.create_app(state)).post(
        "/v1/chat/completions",
        headers=headers,
        json={
            "messages": [
                {"role": "system", "content": instruction},
                {"role": "user", "content": "Look up Oslo and Lima."},
            ],
            "tools": [tool],
            "parallel_tool_calls": True,
            "max_tokens": 256,
            "stream": stream,
        },
    )

    assert response.status_code == 200
    assert len(prompts) == 1
    assert instruction in prompts[0]
    if stream:
        calls = {}
        for payload in _stream_payloads(response.text):
            for choice in payload.get("choices", []):
                for call in choice.get("delta", {}).get("tool_calls", []):
                    entry = calls.setdefault(call["index"], {"name": "", "arguments": ""})
                    for key, value in call.get("function", {}).items():
                        entry[key] += value
        functions = list(calls.values())
    else:
        functions = [
            call["function"]
            for call in response.json()["choices"][0]["message"]["tool_calls"]
        ]
    assert [function["name"] for function in functions] == ["lookup", "lookup"]
    assert [json.loads(function["arguments"]) for function in functions] == [
        {"city": "Oslo"}, {"city": "Lima"}
    ]
