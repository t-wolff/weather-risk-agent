import logging

import anthropic

from app.config import env
from app.llm.base import LLMError, LLMResponse, ToolCall, ToolDefinition, redact

logger = logging.getLogger(__name__)

MAX_TOKENS = 16000


def to_anthropic_messages(messages: list[dict]) -> list[dict]:
    out: list[dict] = []
    for message in messages:
        if message["role"] == "user":
            out.append({"role": "user", "content": message["content"]})
        elif message["role"] == "assistant":
            out.append({"role": "assistant", "content": _assistant_content(message)})
        else:
            block = {
                "type": "tool_result",
                "tool_use_id": message["tool_call_id"],
                "content": message["content"],
                "is_error": message["is_error"],
            }
            # Results for one assistant turn must share a single user message.
            if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                out[-1]["content"].append(block)
            else:
                out.append({"role": "user", "content": [block]})
    return out


def _assistant_content(message: dict) -> list[dict]:
    # Native blocks keep thinking intact, which must be passed back unchanged.
    if message.get("provider") == "anthropic" and message.get("native"):
        return message["native"]
    content = [{"type": "text", "text": message["text"]}] if message.get("text") else []
    return content + [
        {"type": "tool_use", "id": c["id"], "name": c["name"], "input": c["arguments"]}
        for c in message.get("tool_calls", [])
    ]


class AnthropicClient:
    provider = "anthropic"

    def __init__(self):
        self.model = env("ANTHROPIC_MODEL", "claude-opus-5-5")
        self.client = anthropic.Anthropic()

    def complete(self, system: str, tools: list[ToolDefinition], messages: list[dict]) -> LLMResponse:
        try:
            response = self.client.beta.messages.create(
                model=self.model,
                max_tokens=MAX_TOKENS,
                system=system,
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                output_config={"effort": "medium"},
                tools=[
                    {"name": t.name, "description": t.description, "input_schema": t.schema, "strict": True}
                    for t in tools
                ],
                messages=to_anthropic_messages(messages),
            )
        except TypeError as error:
            # The SDK raises this when no credentials are configured.
            raise LLMError("No Anthropic credentials are set. Add ANTHROPIC_API_KEY to .env.") from error
        except anthropic.APIConnectionError as error:
            raise LLMError("Could not reach Anthropic.") from error
        except anthropic.RateLimitError as error:
            raise LLMError("Anthropic rate limit reached. Try again shortly.") from error
        except anthropic.APIStatusError as error:
            logger.warning("Anthropic returned %s: %s", error.status_code, redact(error.message))
            raise LLMError(f"Anthropic returned status {error.status_code}.") from error
        if response.stop_reason == "refusal":
            raise LLMError("The model declined this request.")
        if response.stop_reason == "max_tokens":
            raise LLMError("The model's reply was too long and was cut off.")
        return LLMResponse(
            text="".join(b.text for b in response.content if b.type == "text"),
            tool_calls=[
                ToolCall(id=b.id, name=b.name, arguments=b.input)
                for b in response.content
                if b.type == "tool_use"
            ],
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            native=[b.model_dump(exclude_none=True) for b in response.content],
        )
