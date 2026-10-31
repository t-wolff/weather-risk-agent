"""Checks that the numbers in an answer came from tool results."""
import re

from app.agent.schemas import AgentAnswer

# Tool results are rounded to 2 places, and answers may round further to 1.
TOLERANCE = 0.051
UNKNOWN_CALL = "refers to unknown call"
PROSE_NUMBER = re.compile(r"(?<![\w.-])\d+(?:,\d{3})*\.\d+|\b\d+(?:\.\d+)?(?=%)")


def numbers_in(value) -> list[float]:
    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, dict):
        return [n for item in value.values() for n in numbers_in(item)]
    if isinstance(value, list):
        return [n for item in value for n in numbers_in(item)]
    return []


def _matches(value: float, numbers: list[float]) -> bool:
    return any(abs(value - number) <= TOLERANCE for number in numbers)


def check(answer: AgentAnswer, tool_results: dict[str, object]) -> dict:
    """`problems` are structured values that don't match. `warnings` are prose numbers from no tool.
    Either one fails the check, so an unsourced number can't show as verified."""
    problems = []
    for citation in answer.citations:
        if citation.tool_call_id not in tool_results:
            problems.append(f"Citation {citation.metric} {UNKNOWN_CALL} {citation.tool_call_id}")
        elif not _matches(citation.value, numbers_in(tool_results[citation.tool_call_id])):
            problems.append(
                f"Cited {citation.metric}={citation.value} for {citation.hub} is not in "
                f"result {citation.tool_call_id}"
            )
    if not answer.citations and not answer.out_of_scope and re.search(r"\d", answer.answer_markdown):
        problems.append("The answer states numbers but cites none")
    every_number = numbers_in(list(tool_results.values()))
    for ranked in answer.ranked_hubs:
        if not _matches(ranked.edd, every_number):
            problems.append(f"Ranked EDD {ranked.edd} for {ranked.hub} is not in any tool result")
    warnings = [
        f"Number {text} in the answer text matches no tool result"
        for text in PROSE_NUMBER.findall(answer.answer_markdown)
        if not _matches(float(text.replace(",", "")), every_number + [c.value for c in answer.citations])
    ]
    return {"passed": not problems and not warnings, "problems": problems, "warnings": warnings}


def repair_error(check_result: dict) -> str:
    if any(UNKNOWN_CALL in problem for problem in check_result["problems"]):
        return "A citation uses an unknown id. Copy tool_call_id from the tool result, then resubmit."
    return "Numbers don't match tool results. Fix them and resubmit."
