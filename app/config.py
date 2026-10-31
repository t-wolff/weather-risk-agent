import os
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

HISTORY_START_YEAR = 2006


def env(name: str, default: str = "") -> str:
    return os.environ.get(name) or default


def db_path() -> Path:
    path = Path(env("DB_PATH", "data/weather_risk.db"))
    return path if path.is_absolute() else ROOT / path


@lru_cache
def scoring_config() -> dict:
    return yaml.safe_load((ROOT / "config" / "scoring.yaml").read_text())


@lru_cache
def hubs() -> tuple[dict, ...]:
    return tuple(yaml.safe_load((ROOT / "config" / "hubs.yaml").read_text())["hubs"])


def hub_by_id(hub_id: str) -> dict | None:
    return next((h for h in hubs() if h["id"] == hub_id), None)


def find_hub(name: str) -> dict | None:
    """Match a hub by id or city name, ignoring case and a trailing state."""
    wanted = name.strip().lower().split(",")[0].strip().replace(" ", "_")
    return next(
        (h for h in hubs() if wanted in (h["id"], h["name"].lower().replace(" ", "_"))),
        None,
    )
