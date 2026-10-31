"""Run both adapters through their real SDKs against a simulated HTTP server."""
import json

import anthropic
import httpx
import httpx2
import openai
import pytest

from app.agent.loop import tool_definitions
from app.llm.anthropic_client import AnthropicClient
from app.llm.base import LLMError
from app.llm.openai_client import OpenAIClient

USAGE = {"input_tokens": 11, "output_tokens": 7}


def anthropic_client_response(status, body):
    return httpx2.Response(status, json=body)


def anthropic_body(stop_reason="tool_use"):
    return {
        "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
        "stop_reason": stop_reason, "stop_sequence": None, "usage": USAGE,
        "content": [
            {"type": "thinking", "thinking": "", "signature": "sig"},
            {"type": "tool_use", "id": "t1", "name": "list_hubs", "input": {"region": "West"}},
        ],
    }


def anthropic_client(handler) -> AnthropicClient:
    client = AnthropicClient.__new__(AnthropicClient)
    client.model = "claude-opus-5-5"
    client.client = anthropic.Anthropic(
        api_key="test", http_client=httpx2.Client(transport=httpx2.MockTransport(handler)), max_retries=0,
    )
    return client


def test_anthropic_request_and_response():
    requests = []

    def handler(request):
        requests.append(request)
        return anthropic_client_response(200, anthropic_body())

    client = anthropic_client(handler)
    response = client.complete("sys", tool_definitions(), [{"role": "user", "content": "Hi"}])

    body = json.loads(requests[0].content)
    assert "tool_choice" not in body
    assert body["fallbacks"] == "default"
    assert all(tool["strict"] for tool in body["tools"])
    assert "server-side-fallback-2026-07-01" in requests[0].headers["anthropic-beta"]
    assert response.tool_calls[0].name == "list_hubs"
    assert response.tool_calls[0].arguments == {"region": "West"}
    assert response.native[0]["signature"] == "sig"

    history = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "text": "", "tool_calls": [], "provider": "anthropic", "native": response.native},
    ]
    client.complete("sys", tool_definitions(), history)
    replayed = json.loads(requests[1].content)["messages"][1]["content"]
    assert replayed[0] == {"type": "thinking", "thinking": "", "signature": "sig"}


def test_anthropic_refusal_is_reported():
    client = anthropic_client(lambda r: anthropic_client_response(200, anthropic_body("refusal")))
    with pytest.raises(LLMError, match="declined"):
        client.complete("sys", [], [{"role": "user", "content": "Hi"}])


def test_anthropic_status_error_is_reported():
    client = anthropic_client(lambda r: anthropic_client_response(401, {"error": {"message": "no"}}))
    with pytest.raises(LLMError, match="401"):
        client.complete("sys", [], [{"role": "user", "content": "Hi"}])


def openai_client(handler) -> OpenAIClient:
    client = OpenAIClient.__new__(OpenAIClient)
    client.model = "test-model"
    client.client = openai.OpenAI(
        api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(handler)), max_retries=0,
    )
    return client


def openai_body(arguments='{"region": "West"}'):
    return {
        "id": "c1", "object": "chat.completion", "created": 1, "model": "test-model",
        "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
        "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
            "role": "assistant", "content": None,
            "tool_calls": [{"id": "t1", "type": "function",
                            "function": {"name": "list_hubs", "arguments": arguments}}],
        }}],
    }


def test_openai_request_and_response():
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=openai_body())

    response = openai_client(handler).complete("sys", tool_definitions(), [{"role": "user", "content": "Hi"}])
    body = json.loads(requests[0].content)
    assert body["messages"][0]["role"] == "system"
    assert all(tool["function"]["strict"] for tool in body["tools"])
    assert response.tool_calls[0].arguments == {"region": "West"}
    assert response.input_tokens == 11


def test_openai_invalid_arguments_are_reported():
    client = openai_client(lambda r: httpx.Response(200, json=openai_body("{broken")))
    with pytest.raises(LLMError, match="invalid"):
        client.complete("sys", [], [{"role": "user", "content": "Hi"}])


@pytest.mark.parametrize("effort, expected", [("none", "none"), ("", None)])
def test_openai_reasoning_effort_is_only_sent_when_set(monkeypatch, effort, expected):
    monkeypatch.setenv("OPENAI_REASONING_EFFORT", effort)
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return httpx.Response(200, json=openai_body())

    openai_client(handler).complete("sys", tool_definitions(), [{"role": "user", "content": "Hi"}])
    assert seen.get("reasoning_effort") == expected


def test_anthropic_status_error_logs_the_reason_without_a_key(caplog):
    body = {"error": {"message": "Bad tools for key sk-ant-abc123"}}
    client = anthropic_client(lambda r: anthropic_client_response(400, body))
    with pytest.raises(LLMError), caplog.at_level("WARNING"):
        client.complete("sys", [], [{"role": "user", "content": "Hi"}])
    assert "Bad tools" in caplog.text and "sk-ant-abc123" not in caplog.text
