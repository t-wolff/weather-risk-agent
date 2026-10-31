"""Fixed checklist applied to every answer, so a case can't pass on narrow criteria alone.

Each rule has a name. A case opts out with `allow: [rule]`, which keeps exceptions visible.
"""
import json
import re

from app.agent.grounding import numbers_in
from app.config import scoring_config

SKILL_TOOL = "load_skill"
SCORED_TOOLS = {"rank_hubs", "compare_hubs", "explain_hub", "weather_stat", "yearly_history"}
# Worst turn measured so far is 24,212. This is a tripwire, not a tuned limit.
MAX_INPUT_TOKENS_PER_TURN = 30_000
ROUND_TO = 0.0051
# "a difference of 1.6" ends before its number, so look just past the match.
GAP_NUMBER_REACH = 12

GAP = re.compile(
    r"\bby\s+(?:about\s+|around\s+|roughly\s+|nearly\s+|almost\s+|just\s+|only\s+)?"
    r"\d+(?:\.\d+)?\s*(?:expected\s+)?(?:closure\s+)?(?:EDD|days?|points?|places?|positions?)\b"
    r"|\b(?:gap|difference|margin)\s+of\b"
    r"|\b\d+(?:\.\d+)?\s*(?:EDD|days?)?\s+(?:more|fewer)\s+(?:EDD|days?|closure)\b"
    r"|\b\d+(?:\.\d+)?\s*(?:EDD|days?)\s+(?:more|higher|lower|fewer|ahead|behind)\b",
    re.IGNORECASE,
)
RATIO = re.compile(
    r"\b(?:twice|double|triple|quadruple)\b|\bhalf\s+(?:as|of)\b"
    r"|\b(?:two|three|four|five|six|seven|eight|nine|ten|\d+(?:\.\d+)?)[\s-]*(?:times|x)"
    r"\s+(?:as|more|higher|greater|larger|longer|the|that)\b"
    r"|\b(?:a|one|two)[\s-](?:third|quarter|fifth)\b",
    re.IGNORECASE,
)
INTERNAL_TERM = re.compile(r"\bnri\b|\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b", re.IGNORECASE)
TWO_PLACES = re.compile(r"(?<![\w.])\d+\.\d{2}\b")
ANY_NUMBER = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?")
LIST_MARKER = re.compile(r"^\s*\d+\.\s", re.MULTILINE)
CALENDAR_DATE = re.compile(
    r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2}\b|\b\d{4}-\d{2}-\d{2}\b"
)
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def is_year(value: float) -> bool:
    return value == int(value) and 1900 <= value <= 2100


def tool_numbers(result: dict) -> list[float] | None:
    """None when the saved run has no tool results, which older runs don't."""
    calls = result["trace"]["tool_calls"]
    if calls and any("result" not in call for call in calls):
        return None
    return numbers_in([call["result"] for call in calls])


def _matches(value: float, numbers: list[float]) -> bool:
    return any(abs(value - number) <= 0.051 for number in numbers)


def ungrounded_numbers(text: str, numbers: list[float], question: str = "") -> list[str]:
    """Whole numbers count too. The app's own check only looks at decimals and percentages."""
    text = LIST_MARKER.sub(" ", CALENDAR_DATE.sub(" ", text))
    allowed = numbers + [float(n.replace(",", "")) for n in ANY_NUMBER.findall(question)]
    found = []
    for raw in ANY_NUMBER.findall(text):
        value = float(raw.replace(",", ""))
        if not is_year(value) and not _matches(value, allowed):
            found.append(raw)
    return found


def tool_gaps(calls: list[dict]) -> list[float]:
    """Only real gaps count: a value that matches some other tool number is still invented."""
    return [
        gap["days"]
        for call in calls if call["name"] == "compare_hubs" and isinstance(call.get("result"), dict)
        for gap in call["result"].get("difference_by_hazard", {}).values()
    ]


def unsourced_gaps(text: str, gaps: list[float]) -> list[str]:
    found = []
    for match in GAP.finditer(text):
        stated = ANY_NUMBER.search(text, match.start(), match.end() + GAP_NUMBER_REACH)
        if not stated or not _matches(float(stated.group().replace(",", "")), gaps):
            found.append(match.group())
    return found


def derived_numbers(text: str, numbers: list[float] | None) -> list[str]:
    """Two-place figures that equal one stated figure minus another and came from no tool."""
    stated = [float(n) for n in TWO_PLACES.findall(text)]
    found = []
    for candidate in stated:
        if numbers and _matches(candidate, numbers):
            continue
        for first in stated:
            for second in stated:
                if first > second and candidate not in (first, second) \
                        and abs(first - second - candidate) <= ROUND_TO:
                    found.append(f"{candidate} = {first} - {second}")
    return sorted(set(found))


def blend_as_percent(text: str) -> list[str]:
    weights = [round(100 * w) for w in scoring_config()["blend"].values()]
    pattern = "|".join(str(w) for w in weights)
    percent = re.compile(rf"\b(?:{pattern})\s*(?:%|percent)|\b(?:{pattern})\s*/\s*(?:{pattern})\b")
    return [
        sentence for sentence in SENTENCE_END.split(text)
        if re.search(r"blend|weight|observed|FEMA", sentence, re.IGNORECASE) and percent.search(sentence)
    ]


def repeated_calls(calls: list[dict]) -> list[str]:
    seen, repeats = set(), []
    for call in calls:
        key = (call["name"], json.dumps(call.get("arguments"), sort_keys=True))
        if key in seen:
            repeats.append(call["name"])
        seen.add(key)
    return repeats


def audit(result: dict, question: str = "", allow: tuple[str, ...] | list[str] = ()) -> list[str]:
    """Findings for one turn, each prefixed with its rule name."""
    answer = result["answer"]
    text = answer["answer_markdown"]
    calls = result["trace"]["tool_calls"]
    numbers = tool_numbers(result)
    findings: dict[str, list[str]] = {
        "gap": unsourced_gaps(text, tool_gaps(calls)) + derived_numbers(text, numbers),
        "ratio": [m.group() for m in RATIO.finditer(text)],
        "internal_term": [m.group() for m in INTERNAL_TERM.finditer(text)],
        "blend_format": blend_as_percent(text),
        "repeated_call": repeated_calls(calls),
    }
    if numbers is not None:
        findings["ungrounded_number"] = ungrounded_numbers(text, numbers, question)
    if SCORED_TOOLS & {call["name"] for call in calls} and not answer["out_of_scope"]:
        findings["uncertainty"] = [
            field for field in ("reasoning_summary", "assumptions", "confidence_reason", "data_sources")
            if field in answer and not answer[field]
        ]
    tokens = result["trace"].get("input_tokens", 0)
    if tokens > MAX_INPUT_TOKENS_PER_TURN:
        findings["token_budget"] = [f"{tokens} input tokens"]
    return [
        f"[{rule}] {', '.join(items)}"
        for rule, items in findings.items()
        if items and rule not in allow
    ]
