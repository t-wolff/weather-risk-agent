import json
import logging

import openai

from app.config import env
from app.llm.base import LLMError, LLMResponse, ToolCall, ToolDefinition, redact

logger = logging.getLogger(__name__)


def to_openai_messages(system: str, messages: list[dict]) -> list[dict]:
    out: list[dict] = [{"role": "system", "content": system}]
    for message in messages:
        if message["role"] == "user":
            out.append({"role": "user", "content": message["content"]})
        elif message["role"] == "assistant":
            entry: dict = {"role": "assistant", "content": message.get("text") or None}
            if message.get("tool_calls"):
                entry["tool_calls"] = [
                    {"id": c["id"], "type": "function",
                     "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}}
                    for c in message["tool_calls"]
                ]
            out.append(entry)
        else:
            out.append({"role": "tool", "tool_call_id": message["tool_call_id"], "content": message["content"]})
    return out


class OpenAIClient:
    provider = "openai"

    def __init__(self):
        self.model = env("OPENAI_MODEL")
        if not self.model:
            raise LLMError("Set OPENAI_MODEL to use the OpenAI provider.")
        self.client = openai.OpenAI()

    def complete(self, system: str, tools: list[ToolDefinition], messages: list[dict]) -> LLMResponse:
        effort = env("OPENAI_REASONING_EFFORT")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=to_openai_messages(system, messages),
                **({"reasoning_effort": effort} if effort else {}),
                tools=[
                    {"type": "function", "function": {
                        "name": t.name, "description": t.description, "parameters": t.schema, "strict": True,
                    }}
                    for t in tools
                ],
            )
        except openai.APIConnectionError as error:
            raise LLMError("Could not reach OpenAI.") from error
        except openai.RateLimitError as error:
            raise LLMError("OpenAI rate limit reached. Try again shortly.") from error
        except openai.APIStatusError as error:
            logger.warning("OpenAI returned %s: %s", error.status_code, redact(error.message))
            raise LLMError(f"OpenAI returned status {error.status_code}.") from error
        choice = response.choices[0]
        if choice.finish_reason == "length":
            raise LLMError("The model's reply was too long and was cut off.")
        calls = []
        for call in choice.message.tool_calls or []:
            try:
                arguments = json.loads(call.function.arguments)
            except json.JSONDecodeError as error:
                raise LLMError("The model returned invalid tool arguments.") from error
            calls.append(ToolCall(id=call.id, name=call.function.name, arguments=arguments))
        return LLMResponse(
            text=choice.message.content or "",
            tool_calls=calls,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
        )
