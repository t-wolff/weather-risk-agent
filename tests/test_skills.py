"""Playbooks are prompts, so these guard what can be checked without an LLM."""
import re
from pathlib import Path

import pytest

from app.tools.registry import SKILLS_DIR, LoadSkillInput, load_skill

ROOT = Path(__file__).resolve().parent.parent
SOURCES = ["app/tools/registry.py", "app/scoring/engine.py", "config/scoring.yaml"]
SKILLS = [path.stem for path in SKILLS_DIR.glob("*.md")]


def body(name: str) -> str:
    return load_skill(None, LoadSkillInput(name=name))["instructions"]


def test_every_skill_file_can_be_loaded():
    assert sorted(SKILLS) == sorted(LoadSkillInput.model_fields["name"].annotation.__args__)


@pytest.mark.parametrize("name", SKILLS)
def test_skill_has_a_version(name):
    assert load_skill(None, LoadSkillInput(name=name))["version"].isdigit()


@pytest.mark.parametrize("name", SKILLS)
def test_skill_holds_no_numbers(name):
    """Values belong in tool results, or a config change leaves the playbook stale."""
    text = re.sub(r"^\d+\. ", "", body(name), flags=re.MULTILINE)
    assert not re.findall(r"\d+(?:[./]\d+)*", text)


@pytest.mark.parametrize("name", SKILLS)
def test_skill_names_only_real_tools_and_fields(name):
    source = "\n".join((ROOT / path).read_text() for path in SOURCES)
    assert [word for word in re.findall(r"`(\w+)`", body(name)) if word not in source] == []
