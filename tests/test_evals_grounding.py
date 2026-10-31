"""Scripted evals of number integrity, repair and skills. No LLM, so they cost nothing.

`open_defect` marks a behaviour that should hold and doesn't yet. Each is strict, so a fix turns it
into a failure until the marker is removed. Numbers refer to the failure report.
"""
import json
import re

import pytest

from app.agent import grounding
from app.agent.loop import run_turn
from app.config import scoring_config
from app.tools.registry import TOOLS_BY_NAME, load_skill, LoadSkillInput, run_tool, SKILLS_DIR
from tests.test_agent import ScriptedLLM, citation, make_answer, submit, tool
from tests.test_tools import seed_year


def open_defect(reason):
    return pytest.mark.xfail(strict=True, raises=AssertionError, reason=reason)


def explained(conn, hub="Dallas"):
    return run_tool(conn, "explain_hub", {"hub": hub})[0]


def turn(conn, *submissions, tool_args=("explain_hub", {"hub": "Dallas"})):
    llm = ScriptedLLM([tool(*tool_args), *submissions])
    return run_turn(conn, llm, "conv", [], "Question"), llm


def both(**overrides):
    """The same answer twice, so that a repair request can't change the outcome."""
    return submit(**overrides), submit("s2", **overrides)


# Grounding and repair

@pytest.mark.parametrize("bad_id", ["explain_hub", "call_2"])
def test_citation_tool_name_repair_names_the_id_problem(conn, bad_id):
    value = explained(conn)["edd"]
    _, llm = turn(conn, *both(citations=[citation(value, bad_id)]))
    repair = json.loads(llm.seen[2][-1]["content"])["result"]
    assert "Numbers" not in repair["error"]
    assert "id" in repair["error"].lower()


def test_citation_invented_id_requests_a_repair_and_stays_flagged(conn):
    value = explained(conn)["edd"]
    result, llm = turn(conn, *both(citations=[citation(value, "call_2")]))
    assert result["trace"]["repairs"] == 1 and len(llm.seen) == 3
    assert not result["verified"]


def test_wrong_number_repair_still_talks_about_numbers(conn):
    _, llm = turn(conn, submit(citations=[citation(99.0)]), submit("s2", citations=[citation(explained(conn)["edd"])]))
    assert "Numbers" in json.loads(llm.seen[2][-1]["content"])["result"]["error"]


def test_repair_recovers(conn):
    value = explained(conn)["edd"]
    result, _ = turn(conn, submit(citations=[citation(value, "explain_hub")]), submit("s2", citations=[citation(value)]))
    assert result["verified"] and result["trace"]["repairs"] == 1


def test_only_one_repair_is_spent(conn):
    result, llm = turn(conn, *both(citations=[citation(99.0)]))
    assert len(llm.seen) == 3 and not result["verified"]


# "Verified" must mean the answer is backed by its tools

def test_warning_blocks_verified(conn):
    """Decided rule: an unsourced number means not verified. It is the 0.84 case."""
    result, _ = turn(conn, *both(answer_markdown="Dallas leads by 0.84."))
    assert result["trace"]["grounding"]["warnings"]
    assert not result["verified"]


def test_uncited_answer_with_no_tools_is_not_verified(conn):
    llm = ScriptedLLM([*both(answer_markdown="The Dallas hub scores 40 expected closure days.")])
    assert not run_turn(conn, llm, "conv", [], "How is Dallas?")["verified"]


def test_out_of_scope_answer_may_mention_a_number_uncited(conn):
    llm = ScriptedLLM([submit(answer_markdown="I can't forecast the next 7 days.", out_of_scope=True)])
    assert run_turn(conn, llm, "conv", [], "Forecast?")["verified"]


@open_defect("Pattern: a citation can name any hub, and only the value is checked")
def test_citation_right_hub(conn):
    miami = explained(conn, "Miami")["edd"]
    wrong = {"hub": "Dallas", "metric": "edd", "value": miami, "tool_call_id": "c1"}
    result, _ = turn(conn, *both(citations=[wrong]), tool_args=("explain_hub", {"hub": "Miami"}))
    assert not result["verified"]


@open_defect("Pattern: a citation can name any metric, and only the value is checked")
def test_citation_right_metric(conn):
    threshold = float(scoring_config()["tiers"]["high"])
    result, _ = turn(conn, *both(citations=[citation(threshold)]))
    assert explained(conn)["edd"] != threshold
    assert not result["verified"]


@open_defect("Pattern: any matching value anywhere in any result counts as sourced")
def test_coincidental_value(conn):
    """Miami isn't in this tool result. Its figure only happens to equal the tier threshold."""
    text = f"Miami floods for {scoring_config()['tiers']['high']:.1f} days."
    result, _ = turn(conn, *both(answer_markdown=text))
    assert result["trace"]["grounding"]["warnings"]


@open_defect("Pattern: the app checks only decimals and percentages, so whole numbers pass")
def test_grounding_checks_whole_numbers():
    check = grounding.check(make_answer(answer_markdown="Dallas closes 17 days a year."), {"c1": {"edd": 8.75}})
    assert check["warnings"]


@open_defect("Pattern: ratio words carry no digits, so nothing checks them")
def test_grounding_checks_ratio_words():
    check = grounding.check(make_answer(answer_markdown="Dallas is four times as exposed."), {"c1": {"edd": 8.75}})
    assert check["warnings"]


def ranked(conn):
    return run_tool(conn, "rank_hubs", {"region": None, "hazards": None})[0]["ranking"]


def row(entry, rank=None, **overrides):
    return {"rank": rank or entry["rank"], "hub": entry["hub"], "edd": entry["edd"],
            "tier": entry["overall_tier"], **overrides}


RANK_TOOL = ("rank_hubs", {"region": None, "hazards": None})


def test_correct_ranking_is_verified(conn):
    rows = [row(entry) for entry in ranked(conn)]
    assert turn(conn, *both(ranked_hubs=rows), tool_args=RANK_TOOL)[0]["verified"]


@open_defect("New: the code never checks the model's order against the deterministic ranking")
def test_reordered_ranking_is_not_verified(conn):
    rows = [row(entry, rank=i) for i, entry in enumerate(reversed(ranked(conn)), start=1)]
    assert not turn(conn, *both(ranked_hubs=rows), tool_args=RANK_TOOL)[0]["verified"]


@open_defect("New: a ranked EDD can belong to a different hub")
def test_ranked_edd_must_belong_to_that_hub(conn):
    first, second, *_ = ranked(conn)
    rows = [row(first, edd=second["edd"]), row(second, edd=first["edd"])]
    assert not turn(conn, *both(ranked_hubs=rows), tool_args=RANK_TOOL)[0]["verified"]


@open_defect("New: the tier in ranked_hubs is never checked")
def test_ranked_tier_must_match_the_tool(conn):
    rows = [row(entry, tier="Made up") for entry in ranked(conn)]
    assert not turn(conn, *both(ranked_hubs=rows), tool_args=RANK_TOOL)[0]["verified"]


@open_defect("New: a ranked hub name is never checked against the tool result")
def test_ranked_hub_must_be_a_real_hub(conn):
    entry = ranked(conn)[0]
    rows = [row(entry, hub="Atlantis")]
    assert not turn(conn, *both(ranked_hubs=rows), tool_args=RANK_TOOL)[0]["verified"]


def test_citation_to_a_dropped_turn_is_unverified(conn):
    """Once history is trimmed, its results can't back a citation."""
    llm = ScriptedLLM([*both(citations=[citation(7.0, "gone")])])
    assert not run_turn(conn, llm, "conv", [], "And again?")["verified"]


# Labels

def test_no_share_language_in_tool_output(conn):
    """A field called a contribution to EDD should be the amount that reaches EDD."""
    result = explained(conn)
    blend = result["blend"]["nri"]
    for driver in result["top_nri_drivers"]:
        group = result["groups"][driver["group"]]
        if group["observed_days_per_year"] is None:
            continue
        nri_days = group["components"][driver["hazard"]]["nri_days"]
        assert driver.get("edd_contribution", blend * nri_days) == pytest.approx(blend * nri_days, abs=0.02)


# Skills against tools

def tool_outputs(conn):
    seed_year(conn, "denver", 2025, 5)
    return {
        "rank_hubs": run_tool(conn, "rank_hubs", {"region": None, "hazards": None})[0],
        "compare_hubs": run_tool(conn, "compare_hubs", {"hubs": ["Dallas", "Miami"], "hazards": None})[0],
        "explain_hub": explained(conn),
        "weather_stat": run_tool(conn, "weather_stat", {
            "hub": "Denver", "metric": "snowfall", "start_date": "2025-01-01",
            "end_date": "2025-12-31", "threshold": None})[0],
        "yearly_history": run_tool(conn, "yearly_history", {
            "hub": "Denver", "metric": "snow_days", "start_year": 2025, "end_year": 2025})[0],
    }


def keys_in(value) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {key for item in value.values() for key in keys_in(item)}
    if isinstance(value, list):
        return {key for item in value for key in keys_in(item)}
    return set()


SKILLS = sorted(path.stem for path in SKILLS_DIR.glob("*.md"))
# What each skill asks for in words, and the field that must supply it.
ASKS = [
    ("ranking", "rank ranges", "rank_hubs", "rank_range"),
    ("ranking", "catastrophic flags", "rank_hubs", "catastrophic_flags"),
    ("ranking", "`caveat`", "rank_hubs", "caveat"),
    ("comparison", "national rank", "compare_hubs", "national_rank_by_hazard"),
    ("comparison", "`caveat`", "compare_hubs", "caveat"),
    ("comparison", "`difference_by_hazard`", "compare_hubs", "difference_by_hazard"),
    ("explanation", "annual frequencies", "explain_hub", "annual_frequency"),
    ("explanation", "closure days before blending", "explain_hub", "pre_blend_days"),
    ("explanation", "`blend`", "explain_hub", "blend"),
    ("explanation", "tier thresholds", "explain_hub", "tier_thresholds"),
    ("explanation", "rank range", "explain_hub", "rank_range"),
    ("weather_statistics", "threshold", "weather_stat", "threshold"),
]


@pytest.mark.parametrize("skill", SKILLS)
def test_skill_outputs_exist(conn, skill):
    """Every field in backticks is a tool, a tool option or something a tool really returns."""
    outputs = tool_outputs(conn)
    real = set(TOOLS_BY_NAME) | {key for output in outputs.values() for key in keys_in(output)}
    real |= {"hurricane_declarations"} | set(scoring_config()["groups"])
    body = load_skill(None, LoadSkillInput(name=skill))["instructions"]
    assert [word for word in re.findall(r"`(\w+)`", body) if word not in real] == []


@pytest.mark.parametrize("skill, phrase, tool_name, field", ASKS)
def test_skill_asks_are_supplied_by_the_tool(conn, skill, phrase, tool_name, field):
    body = load_skill(None, LoadSkillInput(name=skill))["instructions"]
    assert phrase in body, "Update ASKS: the skill no longer says this"
    assert field in keys_in(tool_outputs(conn)[tool_name])


@pytest.mark.parametrize("skill", SKILLS)
@pytest.mark.parametrize("ask", ["by how much", "how much higher", "how many times", "percentage difference"])
def test_skill_never_asks_for_a_calculation(skill, ask):
    """No tool returns these, so asking for one pushes the model to do arithmetic."""
    assert ask not in load_skill(None, LoadSkillInput(name=skill))["instructions"].lower()


def test_only_gaps_between_hubs_are_returned_never_ratios(conn):
    assert "ratio" not in keys_in(tool_outputs(conn)["compare_hubs"])


def test_unsourced_number_gets_a_repair_that_can_clear_it(conn):
    result, llm = turn(conn, submit(answer_markdown="Dallas leads by 0.84."), submit("s2", answer_markdown="Dallas leads."))
    repair = json.loads(llm.seen[2][-1]["content"])["result"]
    assert any("0.84" in problem for problem in repair["problems"])
    assert result["verified"] and result["trace"]["repairs"] == 1
