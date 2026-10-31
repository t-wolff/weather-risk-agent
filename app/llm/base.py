"""Provider-neutral LLM interface.

Messages use one internal format:
  {"role": "user", "content": str}
  {"role": "assistant", "text": str, "tool_calls": [ToolCall dicts], "native": provider blocks, "provider": str}
  {"role": "tool", "tool_call_id": str, "name": str, "content": str, "is_error": bool}
"""
import re
from dataclasses import dataclass, field
from typing import Protocol


SECRET = re.compile(r"\b(?:sk|gsk)[-_][\w-]+")


def redact(text: str) -> str:
    """Provider messages can quote a key, so nothing reaches the logs without this."""
    return SECRET.sub("[redacted]", text)


class LLMError(Exception):
    """The provider failed or declined, and the turn can't continue."""


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    native: list | None = None


@dataclass
class ToolDefinition:
    name: str
    description: str
    schema: dict


class LLMClient(Protocol):
    provider: str
    model: str

    def complete(self, system: str, tools: list[ToolDefinition], messages: list[dict]) -> LLMResponse: ...
