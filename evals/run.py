"""Run the evaluation set against a live LLM. No LLM grades another LLM's output.

Outcomes are kept apart: `failed` is a quality failure, `error` is a provider or code fault, and
`not_run` means the case never got an answer, so it stays out of the pass rate.
"""
import argparse
import json
import re
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from unittest.mock import patch

import yaml

from app.agent.loop import MAX_HISTORY_CHARS, recent_history, run_turn
from app.config import hub_by_id, scoring_config
from app.data import open_meteo, store
from app.llm.base import LLMError
from app.llm.factory import get_llm
from app.scoring.engine import hazard_value, rank
from evals.audit import audit

HERE = Path(__file__).parent
# Stored runs took 3 to 5 requests a turn: skill, tools, submit, and sometimes one repair.
REQUESTS_PER_TURN = (3, 5)
CHECKS = {
    "tools", "no_tools", "only_tools", "verified", "contains_all", "contains_any", "not_contains",
    "not_matching", "tool_numbers_only", "out_of_scope", "flag", "flag_source", "no_ranked_hubs",
    "data_unchanged", "ranked_order", "ranked_values", "cited_scores", "snowfall_percent",
    "top_driver_named", "yearly_counts", "states_tier", "consistent_thresholds",
}
CASE_KEYS = {"id", "category", "turns", "checks", "allow", "history_chars", "new_chat_before_last"}


def load_cases(today: date) -> list[dict]:
    cases = yaml.safe_load((HERE / "cases.yaml").read_text())["cases"]
    for case in cases:
        unknown = (set(case) - CASE_KEYS) | (set(case["checks"]) - CHECKS)
        if unknown:
            raise ValueError(f"{case['id']} uses unknown keys {sorted(unknown)}, which would never be checked")
    return fill(cases, placeholders(today))


def placeholders(today: date) -> dict:
    """Dates and thresholds come from today and the config, so cases don't go stale."""
    return {
        "{last_year}": today.year - 1,
        "{this_year}": today.year,
        "{snow_threshold}": scoring_config()["groups"]["winter"]["observed"]["threshold"],
    }


def fill(node, values: dict):
    if isinstance(node, str):
        if node in values:
            return values[node]
        for key, value in values.items():
            node = node.replace(key, str(value))
        return node
    if isinstance(node, list):
        return [fill(item, values) for item in node]
    if isinstance(node, dict):
        return {key: fill(value, values) for key, value in node.items()}
    return node


def _all_calls(result: dict) -> list[dict]:
    return result["trace"]["tool_calls"]


def _independent_snow_percent(conn, hub_id: str, year: int, threshold: float) -> float:
    """Calculated in SQL, separately from the tool code it checks."""
    row = conn.execute(
        """SELECT 100.0 * SUM(json_extract(day.value, '$.snowfall') >= ?) / COUNT(*)
           FROM weather_cache w, json_each(w.days_json) day
           WHERE w.hub_id = ? AND w.year = ?""",
        (threshold, hub_id, year),
    ).fetchone()
    return row[0]


def _numbers_in_text(text: str) -> list[float]:
    return [float(n) for n in re.findall(r"\d+(?:\.\d+)?", text)]


def text_failures(checks: dict, text: str) -> list[str]:
    failures = []
    for word in checks.get("contains_all", []):
        if str(word) not in text:
            failures.append(f"Answer is missing '{word}'")
    if "contains_any" in checks and not any(str(w) in text for w in checks["contains_any"]):
        failures.append(f"Answer has none of {checks['contains_any']}")
    for word in checks.get("not_contains", []):
        if str(word).lower() in text.lower():
            failures.append(f"Answer should not contain '{word}'")
    for pattern in checks.get("not_matching", []):
        found = re.search(pattern, text, re.IGNORECASE)
        if found:
            failures.append(f"Answer should not say '{found.group()}'")
    return failures


def evaluate(checks: dict, result: dict, conn, before: int) -> list[str]:
    """Returns the failed checks. Pure apart from read-only queries."""
    failures = []
    answer = result["answer"]
    text = answer["answer_markdown"]
    called = [call["name"] for call in _all_calls(result)]
    scores = store.latest_snapshots(conn, scoring_config()["method_version"])

    for name in checks.get("tools", []):
        if name not in called:
            failures.append(f"Expected tool {name}, called {called}")
    for name in checks.get("no_tools", []):
        if name in called:
            failures.append(f"Should not have called {name}")
    if "only_tools" in checks and set(called) - set(checks["only_tools"]):
        failures.append(f"Unnecessary tools: {sorted(set(called) - set(checks['only_tools']))}")
    if "verified" in checks and result["verified"] != checks["verified"]:
        failures.append(f"verified={result['verified']}: {result['trace']['grounding']['problems']}")
    failures += text_failures(checks, text)
    if checks.get("tool_numbers_only") and result["trace"]["grounding"].get("warnings"):
        failures.append(f"Numbers from no tool: {result['trace']['grounding']['warnings']}")
    if "out_of_scope" in checks and answer["out_of_scope"] != checks["out_of_scope"]:
        failures.append(f"out_of_scope={answer['out_of_scope']}")
    if "flag" in checks and result["flag"]["flag"] != checks["flag"]:
        failures.append(f"flag={result['flag']['flag']}, expected {checks['flag']}")
    if "flag_source" in checks and result["flag"]["source"] != checks["flag_source"]:
        failures.append(f"flag source={result['flag']['source']}")
    if checks.get("no_ranked_hubs") and answer["ranked_hubs"]:
        failures.append("Ranked hubs for a place that isn't a hub")
    if checks.get("data_unchanged"):
        after = conn.execute("SELECT COUNT(*) FROM score_snapshots").fetchone()[0]
        if after != before:
            failures.append("Stored scores changed")

    if "ranked_order" in checks:
        spec = checks["ranked_order"]
        pool = [s for s in scores if not spec.get("region") or s["region"] == spec["region"]]
        expected = [row["name"] for row in rank(pool, spec.get("hazards"))]
        if spec.get("limit"):
            expected = expected[: spec["limit"]]
        actual = [row["hub"] for row in answer["ranked_hubs"]]
        if actual[: len(expected)] != expected[: len(actual)] or not actual:
            failures.append(f"Order {actual}, expected {expected}")

    if "ranked_values" in checks:
        spec = checks["ranked_values"]
        by_name = {s["name"]: s for s in scores}
        if not answer["ranked_hubs"]:
            failures.append("No ranked hubs to check")
        for row in answer["ranked_hubs"]:
            score = by_name.get(row["hub"])
            if not score:
                failures.append(f"{row['hub']} is not a scored hub")
                continue
            value = hazard_value(score, spec.get("hazards"))
            if abs(row["edd"] - value) > 0.051:
                failures.append(f"{row['hub']} edd {row['edd']}, expected {value:.2f}")
            if row["tier"] != score["tier"]:
                failures.append(f"{row['hub']} tier {row['tier']}, overall tier is {score['tier']}")

    if "states_tier" in checks:
        score = next(s for s in scores if s["hub_id"] == checks["states_tier"])
        if score["tier"].lower() not in text.lower():
            failures.append(f"Answer doesn't state the actual tier, {score['tier']}")

    if checks.get("consistent_thresholds"):
        thresholds = {
            call["result"].get("threshold") for call in _all_calls(result)
            if call["name"] in ("weather_stat", "yearly_history") and not call["is_error"]
            and call["result"].get("threshold") is not None
        }
        if len(thresholds) > 1:
            failures.append(f"Counts use different thresholds: {sorted(thresholds)}")

    if "cited_scores" in checks:
        by_id = {s["hub_id"]: s for s in scores}
        for spec in checks["cited_scores"]:
            value = hazard_value(by_id[spec["hub"]], [spec["hazard"]])
            if not any(abs(n - value) <= 0.051 for n in _numbers_in_text(text)):
                failures.append(f"Answer doesn't state {spec['hub']} {spec['hazard']} = {value:.2f}")

    if "snowfall_percent" in checks:
        spec = checks["snowfall_percent"]
        value = _independent_snow_percent(conn, spec["hub"], spec["year"], spec["threshold"])
        if not any(abs(n - value) <= 0.51 for n in _numbers_in_text(text)):
            failures.append(f"Answer doesn't state {value:.1f}%")

    if "top_driver_named" in checks:
        score = next(s for s in scores if s["hub_id"] == checks["top_driver_named"])
        component = max(
            ((c, d["nri_days"]) for g in score["groups"].values() for c, d in g["components"].items()),
            key=lambda item: item[1],
        )[0]
        if component.split("_")[0] not in text.lower():
            failures.append(f"Top driver {component} isn't named")
        if "hurricane" in text.lower() and "not" not in text.lower():
            failures.append("Attributes the score to hurricanes")

    if "yearly_counts" in checks:
        spec = checks["yearly_counts"]
        expected = {}
        for year in range(spec["start"], spec["end"] + 1):
            days = open_meteo.year_days(conn, hub_by_id(spec["hub"]), year)
            expected[year] = sum((d[spec["metric"]] or 0) >= spec["threshold"] for d in days)
        tool_rows = [c["result"] for c in _all_calls(result) if c["name"] == "yearly_history" and not c["is_error"]]
        actual = {row["year"]: row["count"] for row in tool_rows[-1]["years"]} if tool_rows else {}
        if actual != expected:
            failures.append("Yearly counts don't match the independent calculation")
    return failures


def run_case(conn, llm, case: dict, today: date) -> dict:
    """Follow-ups go through recent_history, as they do in the API."""
    before = conn.execute("SELECT COUNT(*) FROM score_snapshots").fetchone()[0]
    earlier: list[dict] = []
    turns = []
    with patch("app.agent.loop.MAX_HISTORY_CHARS", case.get("history_chars", MAX_HISTORY_CHARS)):
        for position, message in enumerate(case["turns"], start=1):
            if case.get("new_chat_before_last") and position == len(case["turns"]):
                earlier = []
            result = run_turn(conn, llm, f"eval-{case['id']}", recent_history(earlier), message, today)
            earlier.append({"llm_messages": result["llm_messages"]})
            turns.append({**result, "question": message})
    failures = evaluate(case["checks"], result, conn, before)
    for turn in turns:
        failures += audit(turn, turn["question"], case.get("allow", []))
    return {
        "id": case["id"],
        "category": case["category"],
        "status": "failed" if failures else "passed",
        "passed": not failures,
        "failures": failures,
        "answer": result["answer"]["answer_markdown"],
        "tools": [call["name"] for call in result["trace"]["tool_calls"]],
        "input_tokens": sum(t["trace"]["input_tokens"] for t in turns),
        "output_tokens": sum(t["trace"]["output_tokens"] for t in turns),
        "llm_calls": sum(t["trace"]["llm_calls"] for t in turns),
        # Everything --regrade needs to apply new rules later without spending quota.
        "turns": [{k: t[k] for k in ("question", "answer", "verified", "flag", "trace")} for t in turns],
    }


def is_quota_error(error: Exception) -> bool:
    return getattr(error.__cause__, "status_code", None) == 429


def unfinished(case: dict, status: str, reason: str) -> dict:
    return {"id": case["id"], "category": case["category"], "status": status, "passed": False,
            "failures": [reason], "tools": [], "input_tokens": 0, "output_tokens": 0, "llm_calls": 0}


def run_cases(conn, llm, cases: list[dict], today: date, max_requests: int | None = None,
              report=print) -> list[dict]:
    """Stops spending at the first quota error or at the budget. The rest are marked not_run."""
    results, used, stop = [], 0, None
    for case in cases:
        if max_requests is not None and used >= max_requests:
            stop = stop or f"Request budget of {max_requests} used"
        if stop:
            outcome = unfinished(case, "not_run", stop)
        else:
            try:
                outcome = run_case(conn, llm, case, today)
            except LLMError as error:
                if is_quota_error(error):
                    stop = f"Quota or rate limit: {error}"
                    outcome = unfinished(case, "not_run", stop)
                else:
                    outcome = unfinished(case, "error", f"LLMError: {error}")
            except Exception as error:
                outcome = unfinished(case, "error", f"{type(error).__name__}: {error}")
        used += outcome["llm_calls"]
        results.append(outcome)
        report(f"{outcome['status'].upper():<8} {outcome['category']:<14} {outcome['id']}")
        for failure in outcome["failures"]:
            report(f"      {failure}")
    return results


def summary(results: list[dict]) -> str:
    counts = {s: sum(r["status"] == s for r in results) for s in ("passed", "failed", "error", "not_run")}
    graded = counts["passed"] + counts["failed"]
    return (f"{counts['passed']}/{graded} passed. "
            f"{counts['error']} errored and {counts['not_run']} not run, none of them in the pass rate.")


def estimate_requests(cases: list[dict]) -> tuple[int, int]:
    turns = sum(len(case["turns"]) for case in cases)
    return turns * REQUESTS_PER_TURN[0], turns * REQUESTS_PER_TURN[1]


def fits_budget(cases: list[dict], max_requests: int | None) -> bool:
    return max_requests is None or estimate_requests(cases)[1] <= max_requests


def regrade(path: Path, cases: list[dict]) -> list[dict]:
    """Apply today's rules to a saved run. Runs saved before turns were stored get text rules only."""
    by_id = {case["id"]: case for case in cases}
    regraded = []
    for saved in json.loads(path.read_text())["results"]:
        case = by_id.get(saved["id"], {"checks": {}})
        if not saved.get("answer"):
            continue
        turns = saved.get("turns") or [{
            "question": "",
            "answer": {"answer_markdown": saved["answer"], "out_of_scope": False},
            "trace": {"tool_calls": [{"name": name} for name in saved["tools"]]},
        }]
        findings = []
        for turn in turns:
            findings += audit(turn, turn["question"], case.get("allow", []))
        findings += text_failures(case["checks"], turns[-1]["answer"]["answer_markdown"])
        regraded.append({"id": saved["id"], "was": saved.get("status", saved["passed"]), "findings": findings})
    return regraded


def matrix(paths: list[Path]) -> str:
    """Pass counts per case and model, from saved runs. Costs nothing."""
    table: dict[str, dict] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    models = []
    for path in paths:
        run = json.loads(path.read_text())
        model = f"{run['provider']}/{run['model']}"
        models += [model] if model not in models else []
        for row in run["results"]:
            cell = table[row["id"]][model]
            status = row.get("status")
            if status in (None, "passed", "failed") and not _is_old_quota_row(row):
                cell[0] += row["passed"]
                cell[1] += 1
    width = max(map(len, table), default=4) + 2
    lines = ["case".ljust(width) + "  ".join(models)]
    for case_id in sorted(table):
        cells = ["{}/{}".format(*table[case_id][m]) if table[case_id][m][1] else "-" for m in models]
        lines.append(case_id.ljust(width) + "  ".join(c.ljust(len(m)) for c, m in zip(cells, models)))
    return "\n".join(lines)


def _is_old_quota_row(row: dict) -> bool:
    return any("rate limit" in failure for failure in row["failures"])


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the evaluation set")
    parser.add_argument("--provider", help="anthropic or openai; defaults to LLM_PROVIDER")
    parser.add_argument("--case", nargs="+", help="Run these case ids only")
    parser.add_argument("--category", nargs="+", help="Run these categories only")
    parser.add_argument("--repeat", type=int, default=1, help="Run each case this many times")
    parser.add_argument("--max-requests", type=int, help="Refuse to start if the estimate exceeds this")
    parser.add_argument("--dry-run", action="store_true", help="Validate cases and show the estimate")
    parser.add_argument("--regrade", type=Path, help="Apply current rules to a saved run, with no LLM")
    parser.add_argument("--matrix", type=Path, nargs="+", help="Tabulate saved runs per model")
    args = parser.parse_args()

    if args.matrix:
        print(matrix(args.matrix))
        return 0

    today = date.today()
    cases = load_cases(today)
    if args.regrade:
        regraded = regrade(args.regrade, cases)
        for row in regraded:
            print(f"{'CLEAN' if not row['findings'] else 'FINDINGS':<9} {row['id']} (was {row['was']})")
            for finding in row["findings"]:
                print(f"      {finding}")
        return 1 if any(row["findings"] for row in regraded) else 0

    if args.case:
        cases = [case for case in cases if case["id"] in args.case]
    if args.category:
        cases = [case for case in cases if case["category"] in args.category]
    cases = cases * args.repeat
    low, high = estimate_requests(cases)
    print(f"{len(cases)} cases, estimated {low}-{high} LLM requests.")
    if args.dry_run:
        return 0
    if not fits_budget(cases, args.max_requests):
        print(f"Not started: the estimate could exceed --max-requests {args.max_requests}. "
              "Narrow the run with --case or --category.")
        return 2

    llm = get_llm(args.provider)
    conn = store.connect()
    results = run_cases(conn, llm, cases, today, args.max_requests)

    tokens_in = sum(r["input_tokens"] for r in results)
    tokens_out = sum(r["output_tokens"] for r in results)
    print(f"\n{summary(results)} {llm.provider} {llm.model}. Tokens: {tokens_in} in, {tokens_out} out. "
          f"Requests: {sum(r['llm_calls'] for r in results)}.")
    if args.repeat > 1:
        for case_id in dict.fromkeys(r["id"] for r in results):
            runs = [r for r in results if r["id"] == case_id and r["status"] in ("passed", "failed")]
            print(f"  {sum(r['passed'] for r in runs)}/{len(runs)}  {case_id}")
    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"{datetime.now():%Y%m%d-%H%M%S}-{llm.provider}.json"
    path.write_text(json.dumps({"provider": llm.provider, "model": llm.model, "results": results},
                               indent=2, default=str))
    print(f"Saved {path}")
    return 0 if all(r["status"] == "passed" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
