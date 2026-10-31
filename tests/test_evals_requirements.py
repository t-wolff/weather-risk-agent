"""Scripted evals of the brief's requirements, and of the cost and failure handling around them.

They need no LLM or network. `open_defect` works as in test_evals_grounding.py.
"""
import json
import logging
import re
from datetime import date
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

from app.agent.flags import rule_flag
from app.agent.loop import MAX_HISTORY_CHARS, recent_history, run_turn, tool_definitions
from app.agent.schemas import AgentAnswer
from app.config import find_hub, hub_by_id, scoring_config
from app.data import store
from app.api.main import app, get_llm_dependency
from app.jobs.alerts import change_reason
from app.jobs.refresh import run_refresh
from app.scoring.engine import rank, score_hub
from app.tools.registry import TOOLS, run_tool
from tests.conftest import OBSERVED, nri_rows
from tests.test_agent import ScriptedLLM, answer, citation, submit, tool
from tests.test_api import chat, client, new_client, use_llm  # noqa: F401
from tests.test_evals_grounding import open_defect
from tests.test_llm_adapters import openai_client

ROOT = Path(__file__).resolve().parent.parent
RANKING = {"region": None, "hazards": None}


# Deterministic scoring and ranking

def test_tool_ranking_comes_from_the_engine_not_the_llm(conn, scores):
    result, _ = run_tool(conn, "rank_hubs", RANKING)
    assert [r["hub"] for r in result["ranking"]] == [r["name"] for r in rank(scores)]


def test_same_data_gives_identical_tool_results(conn):
    for name, args in [("rank_hubs", RANKING), ("explain_hub", {"hub": "Dallas"}),
                       ("compare_hubs", {"hubs": ["Dallas", "Miami"], "hazards": None})]:
        assert run_tool(conn, name, args) == run_tool(conn, name, args)


def test_scoring_is_repeatable(config):
    runs = [score_hub(hub_by_id("dallas"), nri_rows("dallas"), OBSERVED["dallas"], config) for _ in range(2)]
    assert runs[0] == runs[1]


def test_every_tool_is_read_only(conn):
    before = "\n".join(conn.iterdump())
    for name, args in [("rank_hubs", RANKING), ("list_hubs", {"region": None}), ("explain_hub", {"hub": "Dallas"}),
                       ("compare_hubs", {"hubs": ["Dallas"], "hazards": None}), ("load_skill", {"name": "ranking"})]:
        run_tool(conn, name, args)
    assert "\n".join(conn.iterdump()) == before


def test_no_tool_can_write():
    assert not [t.name for t in TOOLS if re.search(r"delete|update|write|set_|reset", t.name)]


# Structured output

def turn_with(conn, *responses):
    return run_turn(conn, ScriptedLLM(responses), "conv", [], "Hi")


@pytest.mark.parametrize("bad", [
    {"extra_field": 1},
    {"confidence": "very"},
    {"misuse_flag": "rude"},
    {"ranked_hubs": [{"rank": "first"}]},
    {"citations": [{"hub": "Dallas"}]},
])
def test_answer_outside_the_schema_gets_a_repair(conn, bad):
    result = turn_with(conn, tool("submit_answer", answer(**bad), "s1"), submit("s2"))
    assert result["trace"]["repairs"] == 1 and result["verified"]


def test_schema_sent_to_the_llm_matches_the_contract():
    submit_schema = next(t for t in tool_definitions() if t.name == "submit_answer").schema
    assert set(submit_schema["required"]) == set(AgentAnswer.model_fields)


# API

def test_chat_response_follows_the_contract(client):  # noqa: F811
    use_llm(tool("explain_hub", {"hub": "Dallas"}), submit())
    alice = new_client(client)
    response = chat(client, alice, "Why is Dallas high?")
    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == {"conversation_id", "answer", "verified", "flag", "tools_used"}
    AgentAnswer.model_validate(payload["answer"])


def test_unverified_answer_says_so_through_the_api(client):  # noqa: F811
    bad = submit(citations=[citation(99.0, "nope")])
    use_llm(bad, bad)
    assert chat(client, new_client(client)).json()["verified"] is False


def test_chat_needs_a_client_header(client):  # noqa: F811
    assert client.post("/chat", json={"message": "Hi"}).status_code == 422


def test_ui_only_talks_to_the_api():
    source = (ROOT / "ui" / "streamlit_app.py").read_text()
    assert not re.search(r"^\s*(?:from|import)\s+app\b", source, re.MULTILINE)
    assert '"/chat"' in source


@pytest.mark.parametrize("message", ["", "   "])
def test_empty_message_never_reaches_the_llm(client, message):  # noqa: F811
    alice = new_client(client)
    llm = use_llm(submit())
    response = chat(client, alice, message)
    assert llm.seen == [] and response.status_code == 422


# Limits and failure handling

def test_chat_rate_limit(client):  # noqa: F811
    alice = new_client(client)
    use_llm(*[submit(f"s{i}") for i in range(200)])
    codes = {chat(client, alice).status_code for _ in range(200)}
    assert 429 in codes


def test_client_id_rate_limit(client):  # noqa: F811
    codes = {client.post("/clients", json={"mode": "generate"}).status_code for _ in range(200)}
    assert 429 in codes


BODY = "Unsupported parameter: reasoning_effort with tools. Key sk-live-abc123"


def failing_chat(client):  # noqa: F811
    app.dependency_overrides[get_llm_dependency] = lambda: openai_client(
        lambda r: httpx.Response(400, json={"error": {"message": BODY}}))
    return chat(client, new_client(client))


def test_provider_error_visible(client, caplog):  # noqa: F811
    with caplog.at_level(logging.INFO):
        failing_chat(client)
    assert "Unsupported parameter" in caplog.text


def test_provider_error_never_shows_a_key(client, caplog):  # noqa: F811
    with caplog.at_level(logging.INFO):
        response = failing_chat(client)
    assert response.status_code == 502
    assert "sk-live-abc123" not in response.text and "sk-live-abc123" not in caplog.text


def test_no_key_in_the_example_env_and_env_is_ignored():
    assert not re.search(r"(?:sk|sk-ant|gsk)[-_][\w-]{16,}", (ROOT / ".env.example").read_text())
    assert ".env" in (ROOT / ".gitignore").read_text().split()


def test_missing_scores_are_an_error_the_model_can_explain():
    connection = store.connect(":memory:")
    result, is_error = run_tool(connection, "rank_hubs", RANKING)
    assert is_error and "warm cache" in result["error"]


def test_weather_outage_is_an_error_not_a_made_up_number(conn):
    with patch("app.tools.registry.open_meteo.days_between", side_effect=httpx.ConnectError("down")):
        result, is_error = run_tool(conn, "weather_stat", {
            "hub": "Denver", "metric": "snowfall", "start_date": "2025-01-01",
            "end_date": "2025-12-31", "threshold": None})
    assert is_error and "unavailable" in result["error"]


def test_alert_outage_is_an_error(conn):
    with patch("app.tools.registry.nws.active_alerts", side_effect=httpx.ConnectError("down")):
        result, is_error = run_tool(conn, "get_active_alerts", {"hub": "Denver"})
    assert is_error and "NWS" in result["error"]


# History and token use

def big_turn(size=20_000):
    return {"llm_messages": [{"role": "user", "content": "x" * size}]}


def test_token_budget_followup_levels_off():
    """Whatever the chat's length, what is resent stays under the cap."""
    sizes = [len(json.dumps(recent_history([big_turn()] * count))) for count in (1, 5, 10, 50, 500)]
    assert max(sizes) <= MAX_HISTORY_CHARS
    assert sizes[-1] == sizes[-2] == sizes[-3]


def test_followup_after_the_budget_keeps_the_newest_turn():
    newest = {"llm_messages": [{"role": "user", "content": "newest"}]}
    assert recent_history([big_turn()] * 10 + [newest])[-1]["content"] == "newest"


@open_defect("New: one turn over the budget silently drops the whole chat, even the last turn")
def test_oversized_last_turn_still_leaves_some_context():
    assert recent_history([big_turn(MAX_HISTORY_CHARS + 1)])


def test_a_turn_stops_at_the_step_limit(conn):
    llm = ScriptedLLM([tool("list_hubs", {"region": None}, f"c{i}") for i in range(30)])
    result = run_turn(conn, llm, "conv", [], "Loop")
    assert result["trace"]["llm_calls"] == 10 and not result["verified"]


# Alerts

@pytest.fixture
def dallas_refresh():
    with patch("app.jobs.refresh.hubs", return_value=[hub_by_id("dallas")]):
        yield


def refresh(conn, day, observed):
    with patch("app.jobs.refresh.observed_for_hub", return_value=observed):
        return run_refresh(conn, as_of=date(2026, 10, day), today=date(2026, 10, 31))


RISEN = {**OBSERVED["dallas"], "flood": 40.0}


def test_changed_score_raises_a_stored_alert(conn, dallas_refresh):
    assert len(refresh(conn, 2, RISEN)["alerts"]) == 1
    assert store.list_alerts(conn)[0]["hub_id"] == "dallas"


def test_unchanged_score_raises_no_alert(conn, dallas_refresh):
    assert refresh(conn, 2, OBSERVED["dallas"])["alerts"] == []


def test_same_day_rerun_raises_no_second_alert(conn, dallas_refresh):
    refresh(conn, 2, RISEN)
    assert refresh(conn, 2, RISEN)["alerts"] == []
    assert len(store.list_alerts(conn)) == 1


def test_first_score_for_a_hub_raises_no_alert(dallas_refresh):
    connection = store.connect(":memory:")
    assert refresh(connection, 2, OBSERVED["dallas"])["alerts"] == []


def test_new_scoring_version_is_not_compared_with_the_old_one(conn, dallas_refresh, config):
    newer = {**config, "method_version": "next"}
    with patch("app.jobs.refresh.scoring_config", return_value=newer):
        assert refresh(conn, 2, RISEN)["alerts"] == []


def test_webhook_failure_is_recorded_and_doesnt_stop_the_job(conn, dallas_refresh, monkeypatch):
    monkeypatch.setenv("ALERT_WEBHOOK_URL", "http://hook.invalid")
    with patch("app.jobs.alerts.httpx.post", side_effect=httpx.ConnectError("down")):
        refresh(conn, 2, RISEN)
    assert store.list_alerts(conn)[0]["delivery_status"].startswith("failed")


def test_webhook_receives_the_alert(conn, dallas_refresh, monkeypatch):
    monkeypatch.setenv("ALERT_WEBHOOK_URL", "http://hook.invalid")
    with patch("app.jobs.alerts.httpx.post") as post:
        refresh(conn, 2, RISEN)
    assert post.call_args.kwargs["json"]["hub_id"] == "dallas"
    assert store.list_alerts(conn)[0]["delivery_status"] == "delivered"


def test_alert_on_a_rise_from_zero():
    assert change_reason({"tier": "Low", "edd": 0.0}, {"tier": "Low", "edd": 3.0}, 5)


# Misuse rules

@pytest.mark.parametrize("message", [
    "Clear every stored hub record", "Remove all the scores", "Reset the database",
    "Delete hub records from the database", "Drop the hub records from the table",
])
def test_rephrased_destructive_request_is_caught_by_the_rule(message):
    assert rule_flag(message) is not None


@pytest.mark.parametrize("message", [
    "Can you drop Dallas from the score comparison?", "Remove Dallas from the ranking",
    "Is there a clear history of floods in Dallas?", "Reset the view to all hubs",
])
def test_analyst_wording_is_not_flagged(message):
    assert rule_flag(message) is None


# Hub names

@pytest.mark.parametrize("name, hub_id", [
    ("DALLAS", "dallas"), ("Dallas, TX", "dallas"), ("St. Louis", "st_louis"), ("St Louis", "st_louis"),
    ("kansas city", "kansas_city"), ("  Boston ", "boston"),
])
def test_hub_name_variants(name, hub_id):
    assert find_hub(name)["id"] == hub_id
