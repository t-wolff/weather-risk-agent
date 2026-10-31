import json

import pytest

from app.agent import grounding
from app.agent.flags import resolve_flag, rule_flag
from app.agent.loop import run_turn, tool_definitions
from app.agent.schemas import AgentAnswer
from app.llm.anthropic_client import to_anthropic_messages
from app.llm.base import LLMResponse, ToolCall
from app.llm.openai_client import to_openai_messages


def answer(**overrides) -> dict:
    return {
        "answer_markdown": "Done.",
        "reasoning_summary": "Used rank_hubs.",
        "ranked_hubs": [],
        "citations": [],
        "assumptions": [],
        "confidence": "medium",
        "confidence_reason": "",
        "data_sources": [],
        "out_of_scope": False,
        "misuse_flag": "none",
        "misuse_reason": "",
        **overrides,
    }


class ScriptedLLM:
    provider = "scripted"
    model = "scripted-1"

    def __init__(self, responses):
        self.responses = list(responses)
        self.seen = []

    def complete(self, system, tools, messages):
        self.seen.append(list(messages))
        return self.responses.pop(0)


def tool(name, arguments, call_id="c1"):
    return LLMResponse(tool_calls=[ToolCall(call_id, name, arguments)], input_tokens=10, output_tokens=5)


def submit(call_id="s1", **overrides):
    return tool("submit_answer", answer(**overrides), call_id)


def citation(value, call_id="c1"):
    return {"hub": "Dallas", "metric": "edd", "value": value, "tool_call_id": call_id}


def test_every_tool_schema_is_strict():
    for definition in tool_definitions():
        assert definition.schema["additionalProperties"] is False
        assert set(definition.schema["required"]) == set(definition.schema["properties"])


def test_turn_runs_tools_and_records_the_trace(conn):
    llm = ScriptedLLM([
        tool("load_skill", {"name": "explanation"}, "c0"),
        tool("explain_hub", {"hub": "Dallas"}),
        submit(citations=[citation(6.87)]),
    ])
    result = run_turn(conn, llm, "conv", [], "Why is Dallas high?")
    assert result["verified"]
    trace = result["trace"]
    assert [c["name"] for c in trace["tool_calls"]] == ["load_skill", "explain_hub"]
    assert trace["tool_calls"][1]["order"] == 2
    assert trace["skills_loaded"] == [{"name": "explanation", "version": "4"}]
    assert trace["provider"] == "scripted" and trace["input_tokens"] == 30
    assert trace["grounding"]["passed"] and trace["method_version"]
    assert trace["prompt_version"]


def test_bad_number_gets_one_repair_then_passes(conn):
    llm = ScriptedLLM([
        tool("explain_hub", {"hub": "Dallas"}),
        submit(citations=[citation(99.0)]),
        submit("s2", citations=[citation(6.87)]),
    ])
    result = run_turn(conn, llm, "conv", [], "Why?")
    assert result["verified"] and result["trace"]["repairs"] == 1


def test_bad_number_after_repair_is_marked_unverified(conn):
    llm = ScriptedLLM([
        tool("explain_hub", {"hub": "Dallas"}),
        submit(citations=[citation(99.0)]),
        submit("s2", citations=[citation(98.0)]),
    ])
    result = run_turn(conn, llm, "conv", [], "Why?")
    assert not result["verified"]
    assert result["trace"]["grounding"]["problems"]


def test_invalid_schema_gets_a_repair(conn):
    llm = ScriptedLLM([tool("submit_answer", {"answer_markdown": "x"}), submit("s2")])
    result = run_turn(conn, llm, "conv", [], "Hi")
    assert result["verified"] and result["trace"]["repairs"] == 1


def test_repeated_invalid_schema_returns_a_safe_answer(conn):
    bad = tool("submit_answer", {"answer_markdown": "x"})
    llm = ScriptedLLM([bad, bad])
    result = run_turn(conn, llm, "conv", [], "Hi")
    assert not result["verified"]
    assert "couldn't" in result["answer"]["answer_markdown"]


def test_plain_text_reply_is_nudged_to_submit(conn):
    llm = ScriptedLLM([LLMResponse(text="Hello"), submit()])
    result = run_turn(conn, llm, "conv", [], "Hi")
    assert result["verified"] and result["trace"]["nudges"] == 1


def test_tool_error_reaches_the_model(conn):
    llm = ScriptedLLM([tool("explain_hub", {"hub": "Atlantis"}), submit()])
    result = run_turn(conn, llm, "conv", [], "Tell me about Atlantis")
    assert result["trace"]["tool_calls"][0]["is_error"]
    assert llm.seen[1][-1]["is_error"] is True


def test_follow_up_sees_history_and_can_cite_it(conn):
    first = run_turn(conn, ScriptedLLM([
        tool("explain_hub", {"hub": "Dallas"}), submit(citations=[citation(6.87)]),
    ]), "conv", [], "Why?")
    llm = ScriptedLLM([submit("s2", citations=[citation(6.87)])])
    second = run_turn(conn, llm, "conv", first["llm_messages"], "And again?")
    assert second["verified"]
    assert any(m.get("name") == "explain_hub" for m in llm.seen[0])


def test_step_limit_stops_a_looping_model(conn):
    llm = ScriptedLLM([tool("list_hubs", {"region": None}, f"c{i}") for i in range(20)])
    result = run_turn(conn, llm, "conv", [], "Hi")
    assert not result["verified"]
    assert len(llm.seen) == 10


def make_answer(**overrides):
    return AgentAnswer.model_validate(answer(**overrides))


def test_grounding_rejects_unknown_call():
    check = grounding.check(make_answer(citations=[citation(1.0, "nope")]), {})
    assert not check["passed"]


def test_grounding_allows_rounding_to_one_decimal():
    check = grounding.check(make_answer(citations=[citation(8.8)]), {"c1": {"edd": 8.75}})
    assert check["passed"]


def test_grounding_checks_ranked_hubs():
    ranked = [{"rank": 1, "hub": "Dallas", "edd": 42.0, "tier": "High"}]
    assert not grounding.check(make_answer(ranked_hubs=ranked), {"c1": {"edd": 8.75}})["passed"]


def test_grounding_warns_on_invented_prose_number():
    check = grounding.check(make_answer(answer_markdown="About 12.5 days, or 40%."), {"c1": {"edd": 12.5}})
    assert not check["passed"]
    assert len(check["warnings"]) == 1 and "40" in check["warnings"][0]


def test_grounding_ignores_years_and_booleans():
    check = grounding.check(make_answer(answer_markdown="In 2025 there were 3 storms."), {"c1": {"flag": True}})
    assert check["warnings"] == []


@pytest.mark.parametrize("message", [
    "Delete all the hub data", "please drop the scores table", "rm -rf /", "wipe the chat history",
])
def test_destructive_rule(message):
    assert rule_flag(message)[0] == "destructive"


@pytest.mark.parametrize("message", ["fuck you", "you are a useless bot", "You're an idiot"])
def test_abusive_rule(message):
    assert rule_flag(message)[0] == "abusive"


@pytest.mark.parametrize("message", [
    "What if a flood wipes out Miami?",
    "Which hub drops the most in rank?",
    "Why is the Dallas hub's risk high?",
    "Is this data useless for Denver?",
])
def test_normal_questions_are_not_flagged(message):
    assert rule_flag(message) is None


def test_rule_beats_model_classification():
    flag = resolve_flag("delete all records", make_answer(misuse_flag="none"))
    assert flag == {"flag": "destructive", "reason": flag["reason"], "source": "rule"}


def test_model_flag_is_used_when_rules_are_silent():
    flag = resolve_flag("write me a poem", make_answer(misuse_flag="out_of_scope", misuse_reason="Unrelated"))
    assert flag["flag"] == "out_of_scope" and flag["source"] == "model"


def test_out_of_scope_answer_is_flagged_even_without_misuse():
    assert resolve_flag("Forecast?", make_answer(out_of_scope=True))["flag"] == "out_of_scope"


MESSAGES = [
    {"role": "user", "content": "Hi"},
    {"role": "assistant", "text": "Checking", "provider": "other", "native": None, "tool_calls": [
        {"id": "a", "name": "list_hubs", "arguments": {"region": None}},
        {"id": "b", "name": "list_hubs", "arguments": {"region": "West"}},
    ]},
    {"role": "tool", "tool_call_id": "a", "name": "list_hubs", "content": "{}", "is_error": False},
    {"role": "tool", "tool_call_id": "b", "name": "list_hubs", "content": "{}", "is_error": True},
    {"role": "user", "content": "Next"},
]


def test_anthropic_groups_parallel_results_in_one_message():
    out = to_anthropic_messages(MESSAGES)
    assert [m["role"] for m in out] == ["user", "assistant", "user", "user"]
    assert len(out[2]["content"]) == 2 and out[2]["content"][1]["is_error"] is True
    assert out[1]["content"][1]["type"] == "tool_use"


def test_anthropic_replays_native_blocks_unchanged():
    native = [{"type": "thinking", "thinking": "", "signature": "x"}]
    out = to_anthropic_messages([{"role": "assistant", "provider": "anthropic", "native": native, "text": ""}])
    assert out[0]["content"] == native


def test_openai_message_format():
    out = to_openai_messages("sys", MESSAGES)
    assert out[0] == {"role": "system", "content": "sys"}
    assert json.loads(out[2]["tool_calls"][1]["function"]["arguments"]) == {"region": "West"}
    assert out[3]["role"] == "tool" and out[3]["tool_call_id"] == "a"


def test_tool_results_show_the_model_the_id_to_cite(conn):
    llm = ScriptedLLM([tool("explain_hub", {"hub": "Dallas"}, "real-id"), submit()])
    run_turn(conn, llm, "conv", [], "Why is Dallas high?")
    message = next(m for m in llm.seen[-1] if m["role"] == "tool")
    assert json.loads(message["content"])["tool_call_id"] == "real-id"
