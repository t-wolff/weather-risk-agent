"""The JSON contract between the LLM and our code."""
from typing import Literal

from pydantic import BaseModel, ConfigDict


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RankedHub(Strict):
    rank: int
    hub: str
    edd: float
    tier: str


class Citation(Strict):
    hub: str
    metric: str
    value: float
    tool_call_id: str


class AgentAnswer(Strict):
    answer_markdown: str
    reasoning_summary: str
    ranked_hubs: list[RankedHub]
    citations: list[Citation]
    assumptions: list[str]
    confidence: Literal["high", "medium", "low"]
    confidence_reason: str
    data_sources: list[str]
    out_of_scope: bool
    misuse_flag: Literal["none", "destructive", "abusive", "out_of_scope"]
    misuse_reason: str


def strict_schema(model: type[BaseModel]) -> dict:
    """JSON schema in the strict form both providers accept."""
    schema = model.model_json_schema()

    def clean(node):
        if isinstance(node, dict):
            node.pop("title", None)
            node.pop("default", None)
            if node.get("type") == "object" and "properties" in node:
                node["additionalProperties"] = False
                node["required"] = list(node["properties"])
            for key, value in node.items():
                if key != "properties":
                    clean(value)
            for value in node.get("properties", {}).values():
                clean(value)
        elif isinstance(node, list):
            for item in node:
                clean(item)

    clean(schema)
    return schema
