"""The eval checks must be able to fail, or a passing score means nothing."""
import json
from copy import deepcopy
from datetime import date
from unittest.mock import patch

import httpx
import pytest
import yaml

from app.config import scoring_config
from app.data import store
from app.llm.base import LLMError
from app.scoring.engine import hazard_value
from evals.audit import MAX_INPUT_TOKENS_PER_TURN, audit
from evals.run import (
    CHECKS,
    HERE,
    estimate_requests,
    evaluate,
    fits_budget,
    is_quota_error,
    load_cases,
    matrix,
    regrade,
    run_case,
    run_cases,
    summary,
)
from tests.test_agent import ScriptedLLM, answer, submit
from tests.test_llm_adapters import openai_client
from tests.test_tools import seed_year

TODAY = date(2026, 10, 31)


def result(tools=(), verified=True, flag="none", **overrides):
    return {
        "answer": answer(**overrides),
        "verified": verified,
        "flag": {"flag": flag, "source": "rule", "reason": ""},
        "trace": {"tool_calls": [{"name": n} for n in tools], "grounding": {"problems": []}},
    }


def test_cases_file_is_valid():
    cases = yaml.safe_load((HERE / "cases.yaml").read_text())["cases"]
    assert len({case["id"] for case in cases}) == len(cases) >= 15


def test_missing_tool_fails(conn):
    assert evaluate({"tools": ["rank_hubs"]}, result(), conn, 0)
    assert not evaluate({"tools": ["rank_hubs"]}, result(["rank_hubs"]), conn, 0)


def test_wrong_ranking_order_fails(conn):
    wrong = [{"rank": 1, "hub": "Denver", "edd": 1.0, "tier": "Low"},
             {"rank": 2, "hub": "Dallas", "edd": 1.0, "tier": "Low"}]
    right = list(reversed(wrong))
    spec = {"ranked_order": {"region": None, "hazards": ["hurricane"], "limit": 2}}
    # Hurricane: Miami, Dallas, Denver
    assert evaluate(spec, result(ranked_hubs=wrong), conn, 0)
    assert evaluate(spec, result(ranked_hubs=right), conn, 0)
    good = [{"rank": 1, "hub": "Miami", "edd": 2.1, "tier": "Low"},
            {"rank": 2, "hub": "Dallas", "edd": 0.1, "tier": "Low"}]
    assert not evaluate(spec, result(ranked_hubs=good), conn, 0)


def test_empty_ranking_fails(conn):
    spec = {"ranked_order": {"region": "South", "hazards": None}}
    assert evaluate(spec, result(), conn, 0)


def test_wrong_flag_fails(conn):
    assert evaluate({"flag": "abusive"}, result(), conn, 0)


def test_changed_data_fails(conn):
    assert evaluate({"data_unchanged": True}, result(), conn, 0)


def test_unverified_fails(conn):
    assert evaluate({"verified": True}, result(verified=False), conn, 0)


def test_number_from_no_tool_fails(conn):
    checks = {"tool_numbers_only": True}
    clean = result()
    assert not evaluate(checks, clean, conn, 0)
    clean["trace"]["grounding"]["warnings"] = ["Number 0.84 in the answer text matches no tool result"]
    assert evaluate(checks, clean, conn, 0)


# Mutation: every check the cases can use must fail on a bad answer and pass on a good one.

def call(name, payload=None):
    return {"name": name, "arguments": {}, "result": payload or {}, "is_error": False}


def traced(calls, **overrides):
    outcome = result(**overrides)
    outcome["trace"]["tool_calls"] = calls
    return outcome


def years(*counts):
    return {"years": [{"year": 2024 + i, "count": c} for i, c in enumerate(counts)]}


WINTER_ROW = {"rank": 1, "hub": "Denver", "tier": "Medium"}
MUTATIONS = {
    "tools": (["rank_hubs"], result(), result(["rank_hubs"])),
    "no_tools": (["list_hubs"], result(["list_hubs"]), result(["rank_hubs"])),
    "only_tools": (["load_skill"], result(["list_hubs"]), result(["load_skill"])),
    "verified": (True, result(verified=False), result()),
    "contains_all": (["Miami"], result(), result(answer_markdown="Miami")),
    "contains_any": (["Miami"], result(), result(answer_markdown="Miami")),
    "not_contains": (["whisker"], result(answer_markdown="A Whisker"), result()),
    "not_matching": ([r"\d%"], result(answer_markdown="Up 5%"), result()),
    "out_of_scope": (True, result(), result(out_of_scope=True)),
    "flag": ("abusive", result(), result(flag="abusive")),
    "flag_source": ("model", result(), result(flag="abusive") | {"flag": {"flag": "x", "source": "model"}}),
    "no_ranked_hubs": (True, result(ranked_hubs=[WINTER_ROW | {"edd": 1.0}]), result()),
    "ranked_order": ({"region": None, "hazards": ["hurricane"], "limit": 1},
                     result(ranked_hubs=[WINTER_ROW | {"edd": 1.0}]),
                     result(ranked_hubs=[{"rank": 1, "hub": "Miami", "edd": 2.1, "tier": "Low"}])),
    "cited_scores": ([{"hub": "miami", "hazard": "hurricane"}], result(answer_markdown="9.99"),
                     result(answer_markdown="2.10")),
    "top_driver_named": ("dallas", result(answer_markdown="Wind."), result(answer_markdown="Ice storms.")),
    "consistent_thresholds": (True,
                              traced([call("weather_stat", {"threshold": 0.1}),
                                      call("yearly_history", {"threshold": 5})]),
                              traced([call("weather_stat", {"threshold": 5}),
                                      call("yearly_history", {"threshold": 5})])),
    "snowfall_percent": ({"hub": "denver", "year": 2025, "threshold": 0.1},
                         result(answer_markdown="60%"), result(answer_markdown="25%")),
    "yearly_counts": ({"hub": "denver", "metric": "snowfall", "threshold": 5, "start": 2024, "end": 2025},
                      traced([call("yearly_history", years(9, 9))]),
                      traced([call("yearly_history", years(2, 5))])),
}


def seed_denver(conn):
    seed_year(conn, "denver", 2024, 2)
    seed_year(conn, "denver", 2025, 5)


@pytest.mark.parametrize("check", MUTATIONS)
def test_mutation_each_check(conn, check):
    seed_denver(conn)
    spec, bad, good = MUTATIONS[check]
    assert evaluate({check: spec}, bad, conn, 0)
    assert not evaluate({check: spec}, good, conn, 0)


def test_mutation_tool_numbers_only(conn):
    good = result()
    bad = result()
    bad["trace"]["grounding"]["warnings"] = ["Number 0.84 in the answer text matches no tool result"]
    assert evaluate({"tool_numbers_only": True}, bad, conn, 0)
    assert not evaluate({"tool_numbers_only": True}, good, conn, 0)


def test_mutation_data_unchanged(conn):
    count = conn.execute("SELECT COUNT(*) FROM score_snapshots").fetchone()[0]
    assert evaluate({"data_unchanged": True}, result(), conn, count + 1)
    assert not evaluate({"data_unchanged": True}, result(), conn, count)


def test_mutation_ranked_values(conn):
    scores = {s["hub_id"]: s for s in store.latest_snapshots(conn, scoring_config()["method_version"])}
    value = hazard_value(scores["miami"], ["hurricane"])
    row = {"rank": 1, "hub": "Miami", "edd": value, "tier": scores["miami"]["tier"]}
    spec = {"hazards": ["hurricane"]}
    assert not evaluate({"ranked_values": spec}, result(ranked_hubs=[row]), conn, 0)
    assert evaluate({"ranked_values": spec}, result(ranked_hubs=[row | {"edd": value + 1}]), conn, 0)
    assert evaluate({"ranked_values": spec}, result(ranked_hubs=[row | {"tier": "Wrong"}]), conn, 0)
    assert evaluate({"ranked_values": spec}, result(), conn, 0)


def test_mutation_states_tier(conn):
    tier = next(s for s in store.latest_snapshots(conn, scoring_config()["method_version"])
                if s["hub_id"] == "dallas")["tier"]
    assert evaluate({"states_tier": "dallas"}, result(answer_markdown="It is fine."), conn, 0)
    assert not evaluate({"states_tier": "dallas"}, result(answer_markdown=f"Tier: {tier}"), conn, 0)


def test_every_check_has_a_mutation():
    covered = set(MUTATIONS) | {"tool_numbers_only", "data_unchanged", "ranked_values", "states_tier"}
    assert covered == CHECKS


def test_unknown_check_is_rejected_rather_than_ignored(tmp_path):
    (tmp_path / "cases.yaml").write_text(
        "cases: [{id: a, category: b, turns: [x], checks: {contain_all: [y]}}]"
    )
    with patch("evals.run.HERE", tmp_path), pytest.raises(ValueError, match="contain_all"):
        load_cases(TODAY)


def test_all_cases_load():
    assert len(load_cases(TODAY)) >= 30


# date_independence and config_independence

def by_id(cases):
    return {case["id"]: case for case in cases}


def test_date_independence():
    case = by_id(load_cases(date(2031, 3, 1)))["denver_snow"]
    assert case["checks"]["snowfall_percent"]["year"] == 2030
    assert case["checks"]["contains_any"] == [2030]


def test_config_independence():
    config = deepcopy(scoring_config())
    config["groups"]["winter"]["observed"]["threshold"] = 9
    with patch("evals.run.scoring_config", return_value=config):
        case = by_id(load_cases(TODAY))["yearly_snow_days"]
    assert case["checks"]["yearly_counts"]["threshold"] == 9
    assert case["checks"]["contains_any"][0] == "9 cm"


# The checklist on every answer. Bad text below is quoted from real runs.

def audited(text, calls=(), **overrides):
    return traced(list(calls), answer_markdown=text, **overrides)


@pytest.mark.parametrize("text, rule", [
    ("Miami, by 1.64 days", "gap"),
    ("Miami leads by 0.80 days", "gap"),
    ("Miami is ahead by about 2 EDD", "gap"),
    ("a difference of 1.6", "gap"),
    ("Miami is 1.64 days higher", "gap"),
    ("Miami 5.10, Houston 3.46, a spread of 1.64", "gap"),
    ("four times as long", "ratio"),
    ("twice the risk", "ratio"),
    ("3.2 times higher", "ratio"),
    ("about a third of Boston", "ratio"),
    ("Blended 70% FEMA history and 30% observed days", "blend_format"),
    ("The blend is 70/30", "blend_format"),
    ("70 percent weight on history", "blend_format"),
    ("The nri figure is 2", "internal_term"),
    ("Its edd_contribution is 2", "internal_term"),
    ("See call_2", "internal_term"),
])
def test_audit_catches_real_failures(text, rule):
    assert any(f.startswith(f"[{rule}]") for f in audit(audited(text)))


GAPS = {"difference_by_hazard": {"flood": {"highest": "Miami", "lowest": "Seattle", "days": 1.64}}}


@pytest.mark.parametrize("text", ["Miami, by 1.64 days", "a difference of 1.6", "Miami is 1.64 days higher"])
def test_audit_allows_a_gap_the_tool_returned(text):
    assert not any(f.startswith("[gap]") for f in audit(audited(text, [call("compare_hubs", GAPS)])))


def test_audit_rejects_a_gap_that_only_matches_some_other_tool_number():
    calls = [call("compare_hubs", {**GAPS, "edd": 2.13})]
    assert any(f.startswith("[gap]") for f in audit(audited("Miami, by 2.13 days", calls)))


def test_audit_catches_the_unsourced_number():
    assert "[ungrounded_number] 0.84" in audit(audited("The gap is 0.84.", [call("rank_hubs", {"edd": 8.0})]))


def test_whole_numbers_grounded():
    assert audit(audited("Across 17 hubs.", [call("rank_hubs", {"edd": 8.75})]))
    assert not audit(audited("Across 22 hubs.", [call("rank_hubs", {"hubs_scored": 22})]), allow=["uncertainty"])


def test_numbers_with_no_tool_are_ungrounded():
    assert audit(audited("Dallas scores 40."))


def test_question_numbers_and_years_and_dates_are_allowed():
    text = "From 2006 to 2025, at 5 cm, as of October 26:\n\n1. Denver"
    assert not audit(audited(text), question="Days over 5 cm?")


def test_skill_load_once():
    skill = {"name": "load_skill", "arguments": {"name": "ranking"}, "result": {}, "is_error": False}
    assert audit(audited("Done.", [skill, skill])) == ["[repeated_call] load_skill"]


def test_no_redundant_tools():
    other = {"name": "rank_hubs", "arguments": {"region": "West"}, "result": {}, "is_error": False}
    first = {"name": "rank_hubs", "arguments": {"region": "South"}, "result": {}, "is_error": False}
    assert not audit(audited("Done.", [first, other], assumptions=["a"], confidence_reason="b",
                             data_sources=["c"]))
    assert audit(audited("Done.", [first, first], assumptions=["a"], confidence_reason="b",
                         data_sources=["c"]))


def test_scored_answer_must_state_assumptions_and_uncertainty():
    assert audit(audited("Done.", [call("explain_hub")], reasoning_summary="r")) == [
        "[uncertainty] assumptions, confidence_reason, data_sources"
    ]


def test_token_budget_turn():
    outcome = audited("Done.")
    outcome["trace"]["input_tokens"] = MAX_INPUT_TOKENS_PER_TURN + 1
    assert audit(outcome)[0].startswith("[token_budget]")


def test_allow_lists_an_exception_per_rule():
    assert audit(audited("twice"), allow=["ratio"]) == []


@pytest.mark.parametrize("text", [
    "**Miami has higher hurricane exposure; Houston has higher flood exposure.**\n\n- **Hurricane:** Miami — "
    "**2.13 EDD** (national rank **1**); Houston — **1.29 EDD** (rank **4**).",
    "Denver had snowfall on **10.14% of days** in 2025 (37 of 365 days), using the default threshold of at "
    "least 0.1 cm.",
    "It ranks **9th of 22 hubs**, with a rank range of **5th–10th**. Weights of **0.7 on FEMA history** and "
    "**0.3 on observed days**.",
    "Three times a year, half-day closures, and 1–2 ranks overlap.",
])
def test_audit_passes_good_answers(text):
    """Text-only rules. A checker that flags good answers would get ignored."""
    assert audit(traced([{"name": "x"}], answer_markdown=text)) == []


# Runner outcomes

class Fails:
    provider, model = "fake", "fake"

    def __init__(self, status):
        self.status = status
        self.calls = 0

    def complete(self, *args):
        self.calls += 1
        raise LLMError("Provider said no") from type("Cause", (Exception,), {"status_code": self.status})()


CASES = [{"id": name, "category": "c", "turns": ["Hi"], "checks": {}} for name in "abc"]


def test_quota_is_not_failure(conn):
    llm = Fails(429)
    results = run_cases(conn, llm, CASES, TODAY, report=lambda _: None)
    assert [r["status"] for r in results] == ["not_run"] * 3
    assert llm.calls == 1, "A quota error must stop the run"
    assert summary(results).startswith("0/0 passed")


def test_quota_mid_run_keeps_the_answers_already_graded(conn):
    class ThenQuota(ScriptedLLM):
        def complete(self, *args):
            if not self.responses:
                raise LLMError("Quota") from type("Cause", (Exception,), {"status_code": 429})()
            return super().complete(*args)

    results = run_cases(conn, ThenQuota([submit()]), CASES, TODAY, report=lambda _: None)
    assert [r["status"] for r in results] == ["passed", "not_run", "not_run"]
    assert summary(results).startswith("1/1 passed")


def test_real_adapter_quota_error_is_recognised():
    client = openai_client(lambda r: httpx.Response(429, json={"error": {"message": "quota"}}))
    with pytest.raises(LLMError) as caught:
        client.complete("sys", [], [{"role": "user", "content": "Hi"}])
    assert is_quota_error(caught.value)


def test_provider_error_is_an_error_not_a_quality_failure(conn):
    llm = Fails(400)
    results = run_cases(conn, llm, CASES, TODAY, report=lambda _: None)
    assert [r["status"] for r in results] == ["error"] * 3
    assert llm.calls == 3


def test_request_budget_preflight():
    assert estimate_requests(CASES) == (9, 15)
    assert fits_budget(CASES, 15) and not fits_budget(CASES, 14)
    assert fits_budget(CASES, None)


def test_full_set_would_not_fit_the_known_allowance():
    assert not fits_budget(load_cases(TODAY), 50)


def test_budget_stops_a_run_part_way(conn):
    llm = ScriptedLLM([submit(), submit("s2"), submit("s3")])
    results = run_cases(conn, llm, CASES, TODAY, max_requests=1, report=lambda _: None)
    assert [r["status"] for r in results] == ["passed", "not_run", "not_run"]


# Live paths, driven by a scripted model

def one_case(**overrides):
    return {"id": "x", "category": "c", "turns": ["First", "Second"], "checks": {}, **overrides}


def test_follow_ups_use_the_same_trimmed_history_as_the_api(conn):
    llm = ScriptedLLM([submit(), submit("s2")])
    run_case(conn, llm, one_case(history_chars=1), TODAY)
    assert [m["content"] for m in llm.seen[1] if m["role"] == "user"] == ["Second"]


def test_case_keeps_history_by_default(conn):
    llm = ScriptedLLM([submit(), submit("s2")])
    run_case(conn, llm, one_case(), TODAY)
    assert [m["content"] for m in llm.seen[1] if m["role"] == "user"] == ["First", "Second"]


def test_cross_chat_leak_case_starts_a_new_chat(conn):
    llm = ScriptedLLM([submit(), submit("s2")])
    run_case(conn, llm, one_case(new_chat_before_last=True), TODAY)
    assert [m["content"] for m in llm.seen[1] if m["role"] == "user"] == ["Second"]


def test_tokens_and_audit_cover_every_turn(conn):
    llm = ScriptedLLM([submit(answer_markdown="twice"), submit("s2")])
    outcome = run_case(conn, llm, one_case(), TODAY)
    assert outcome["input_tokens"] == 20 and outcome["llm_calls"] == 2
    assert outcome["status"] == "failed" and outcome["failures"] == ["[ratio] twice"]


# Free analysis of saved runs

def saved_run(tmp_path, name="run.json", model="m", rows=None):
    path = tmp_path / name
    path.write_text(json.dumps({"provider": "p", "model": model, "results": rows}))
    return path


def test_answer_full_review_regrades_a_saved_pass(tmp_path):
    """The stored run passed this answer. It wrote the poem it was asked for."""
    path = saved_run(tmp_path, rows=[{
        "id": "unrelated_out_of_scope", "passed": True, "tools": ["load_skill"],
        "answer": "A velvet paw, a watchful eye, a whiskered shadow passing by.",
    }])
    assert regrade(path, load_cases(TODAY))[0]["findings"]


def test_regrade_uses_stored_tool_results_when_it_has_them(tmp_path):
    turn = {"question": "", "answer": answer(answer_markdown="It is 0.84."),
            "trace": {"tool_calls": [call("rank_hubs", {"edd": 8.0})]}}
    path = saved_run(tmp_path, rows=[{"id": "x", "passed": True, "tools": [], "answer": "x", "turns": [turn]}])
    assert regrade(path, [])[0]["findings"][0].startswith("[ungrounded_number]")


def test_cross_model_matrix_counts_only_graded_runs(tmp_path):
    def row(status):
        return {"id": "a", "status": status, "passed": status == "passed", "failures": []}

    first = saved_run(tmp_path, "1.json", "m1", [row("passed"), row("failed"), row("not_run")])
    second = saved_run(tmp_path, "2.json", "m2", [row("not_run")])
    table = matrix([first, second])
    assert "p/m1" in table and "p/m2" in table
    assert "1/2" in table.splitlines()[1] and "-" in table.splitlines()[1]
