"""Misuse flags. Rules run first, and the model's classification fills the gap."""
import re

from app.agent.schemas import AgentAnswer

DESTRUCTIVE = re.compile(
    r"\b(delete|drop|wipe|erase|truncate|destroy|overwrite)\b.{0,40}"
    r"\b(data|database|db|tables?|records?|history|scores?|logs?|hubs?|conversations?|chats?)\b"
    # "clear" and "reset" are everyday words, so these need a determiner and a close target.
    r"|\b(clear|remove|reset)\s+(all|every|the|our|my)\b\s+(\w+\s+){0,2}"
    r"(data|database|db|tables?|records?|history|scores?|logs?|hubs?|conversations?|chats?)\b"
    r"|\brm\s+-rf\b|\bdrop\s+table\b",
    re.IGNORECASE,
)
# Taking a hub out of an answer changes nothing that is stored.
ANALYST_EDIT = re.compile(
    r"\b(drop|remove|exclude)\b.{0,40}\bfrom\s+((the|this|my|our)\s+)?(\w+\s+)?"
    r"(comparison|ranking|list|chart|results?|answer)\b",
    re.IGNORECASE,
)
ABUSIVE = re.compile(
    r"\b(fuck\s+you|shut\s+up|screw\s+you"
    r"|you(?:'re|\s+are)\s+(?:an?\s+)?(?:\w+\s+)?(?:idiot|moron|stupid|useless|garbage|trash))\b",
    re.IGNORECASE,
)


def rule_flag(message: str) -> tuple[str, str] | None:
    if DESTRUCTIVE.search(message) and not ANALYST_EDIT.search(message):
        return "destructive", "Message asks to delete or alter stored data"
    if ABUSIVE.search(message):
        return "abusive", "Message contains abusive language aimed at the assistant"
    return None


def resolve_flag(message: str, answer: AgentAnswer) -> dict:
    rule = rule_flag(message)
    if rule:
        return {"flag": rule[0], "reason": rule[1], "source": "rule"}
    if answer.misuse_flag != "none":
        return {"flag": answer.misuse_flag, "reason": answer.misuse_reason, "source": "model"}
    if answer.out_of_scope:
        return {"flag": "out_of_scope", "reason": answer.misuse_reason or "Marked out of scope",
                "source": "model"}
    return {"flag": "none", "reason": "", "source": "none"}
