"""The agent loop: call the LLM, run tools, validate the final answer, record the turn."""
import json
import time
from dataclasses import asdict
from datetime import date

from pydantic import ValidationError

from app.agent import grounding
from app.agent.flags import resolve_flag
from app.agent.prompts import NUDGE, PROMPT_VERSION, SUBMIT_DESCRIPTION, SYSTEM_PROMPT
from app.agent.schemas import AgentAnswer, strict_schema
from app.config import scoring_config
from app.data import store
from app.llm.base import LLMClient, ToolDefinition
from app.tools.registry import TOOLS, run_tool

MAX_STEPS = 10
MAX_REPAIRS = 1
# History is resent on every LLM call in a turn, so it is capped. Roughly 15,000 tokens.
MAX_HISTORY_CHARS = 60_000
SUBMIT = "submit_answer"


def tool_definitions() -> list[ToolDefinition]:
    return [
        ToolDefinition(tool.name, tool.description, strict_schema(tool.input_model)) for tool in TOOLS
    ] + [ToolDefinition(SUBMIT, SUBMIT_DESCRIPTION, strict_schema(AgentAnswer))]


def _tool_message(call_id: str, name: str, result, is_error: bool) -> dict:
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "name": name,
        # Some models can't read the transport id, and citations must quote it.
        "content": json.dumps({"tool_call_id": call_id, "result": result}),
        "is_error": is_error,
    }


def _past_tool_results(messages: list[dict]) -> dict:
    """Tool results already in the conversation, so follow-ups can reuse them."""
    return {
        m["tool_call_id"]: json.loads(m["content"])
        for m in messages
        if m["role"] == "tool" and m["name"] != SUBMIT and not m["is_error"]
    }


def _fallback_answer(reason: str) -> AgentAnswer:
    return AgentAnswer(
        answer_markdown="I couldn't complete a reliable answer to that. Please try rephrasing it.",
        reasoning_summary=reason,
        ranked_hubs=[],
        citations=[],
        assumptions=[],
        confidence="low",
        confidence_reason=reason,
        data_sources=[],
        out_of_scope=False,
        misuse_flag="none",
        misuse_reason="",
    )


def recent_history(turns: list[dict]) -> list[dict]:
    """The newest whole turns that fit the budget, so tool calls keep their results."""
    kept, size = [], 0
    for turn in reversed(turns):
        size += len(json.dumps(turn["llm_messages"]))
        if size > MAX_HISTORY_CHARS:
            break
        kept = turn["llm_messages"] + kept
    return kept


def run_turn(conn, llm: LLMClient, conversation_id: str, history: list[dict], user_message: str,
             today: date | None = None) -> dict:
    """Run one user turn. `history` is the LLM messages from earlier turns."""
    system = SYSTEM_PROMPT.format(today=(today or date.today()).isoformat())
    tools = tool_definitions()
    new_messages: list[dict] = [{"role": "user", "content": user_message}]
    trace = {
        "provider": llm.provider,
        "model": llm.model,
        "prompt_version": PROMPT_VERSION,
        "method_version": scoring_config()["method_version"],
        "tool_calls": [],
        "skills_loaded": [],
        "input_tokens": 0,
        "output_tokens": 0,
        "llm_calls": 0,
        "repairs": 0,
        "nudges": 0,
    }
    answer = None
    check = {"passed": False, "problems": ["No answer submitted"], "warnings": []}

    for _ in range(MAX_STEPS):
        response = llm.complete(system, tools, history + new_messages)
        trace["llm_calls"] += 1
        trace["input_tokens"] += response.input_tokens
        trace["output_tokens"] += response.output_tokens
        new_messages.append({
            "role": "assistant",
            "text": response.text,
            "tool_calls": [asdict(call) for call in response.tool_calls],
            "native": response.native,
            "provider": llm.provider,
        })
        if not response.tool_calls:
            if trace["nudges"] >= MAX_REPAIRS:
                break
            trace["nudges"] += 1
            new_messages.append({"role": "user", "content": NUDGE})
            continue

        finished = False
        for call in response.tool_calls:
            if call.name != SUBMIT:
                started = time.perf_counter()
                result, is_error = run_tool(conn, call.name, call.arguments)
                trace["tool_calls"].append({
                    "order": len(trace["tool_calls"]) + 1,
                    "id": call.id,
                    "name": call.name,
                    "arguments": call.arguments,
                    "result": result,
                    "is_error": is_error,
                    "latency_ms": round(1000 * (time.perf_counter() - started)),
                })
                if call.name == "load_skill" and not is_error:
                    trace["skills_loaded"].append(
                        {"name": result["skill"], "version": result["version"]}
                    )
                new_messages.append(_tool_message(call.id, call.name, result, is_error))
                continue

            try:
                candidate = AgentAnswer.model_validate(call.arguments)
            except ValidationError as error:
                can_retry = trace["repairs"] < MAX_REPAIRS
                trace["repairs"] += 1
                result = {"error": "Answer failed schema validation", "details": error.errors()[:5]}
                new_messages.append(_tool_message(call.id, SUBMIT, json.loads(json.dumps(result, default=str)), True))
                finished = not can_retry
                continue
            answer = candidate
            check = grounding.check(candidate, _past_tool_results(history + new_messages))
            if check["passed"] or trace["repairs"] >= MAX_REPAIRS:
                new_messages.append(_tool_message(call.id, SUBMIT, {"status": "accepted"}, False))
                finished = True
            else:
                trace["repairs"] += 1
                new_messages.append(_tool_message(
                    call.id, SUBMIT,
                    {"error": grounding.repair_error(check),
                     "problems": check["problems"] + check["warnings"]},
                    True,
                ))
        if finished:
            break

    if answer is None:
        answer = _fallback_answer("The model did not produce a valid answer.")
    trace["grounding"] = check
    verified = check["passed"]
    flag = resolve_flag(user_message, answer)
    return {
        "conversation_id": conversation_id,
        "answer": answer.model_dump(),
        "verified": verified,
        "flag": flag,
        "trace": trace,
        "llm_messages": new_messages,
    }


def record_turn(conn, result: dict, user_message: str, ip_hash: str) -> None:
    store.add_turn(conn, {
        "conversation_id": result["conversation_id"],
        "user_message": user_message,
        "answer": {**result["answer"], "verified": result["verified"]},
        "trace": result["trace"],
        "llm_messages": result["llm_messages"],
        "flag": result["flag"]["flag"],
        "flag_reason": result["flag"]["reason"],
        "flag_source": result["flag"]["source"],
        "ip_hash": ip_hash,
    })
