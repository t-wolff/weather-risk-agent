import os

import pytest

os.environ["APP_SECRET"] = "test-secret"
os.environ["ADMIN_TOKEN"] = "test-admin"
os.environ["ALERT_WEBHOOK_URL"] = ""
os.environ["TRUST_FORWARDED_FOR"] = "false"

from app.config import hub_by_id, scoring_config
from app.data import store
from app.scoring.engine import score_hub

NRI = {
    "dallas": {"WNTW": 4.0, "ISTM": 1.0, "HRCN": 0.01, "IFLD": 4.0, "SWND": 4.0, "TRND": 1.0, "HAIL": 10.0},
    "miami": {"HRCN": 0.3, "IFLD": 4.0, "CFLD": 2.0, "SWND": 1.0, "TRND": 1.0, "HAIL": 1.0},
    "denver": {"WNTW": 14.0, "IFLD": 1.0, "SWND": 1.0, "HAIL": 8.0},
}
OBSERVED = {
    "dallas": {"winter": 1.0, "flood": 2.0, "severe_storm": 1.0},
    "miami": {"winter": 0.0, "flood": 8.0, "severe_storm": 2.0},
    "denver": {"winter": 10.0, "flood": 0.0, "severe_storm": 3.0},
}


def nri_rows(hub_id: str) -> dict:
    return {
        field: {"annual_frequency": value, "loss_ratio_rating": None}
        for field, value in NRI[hub_id].items()
    }


@pytest.fixture
def config():
    return scoring_config()


@pytest.fixture
def scores(config):
    return [score_hub(hub_by_id(h), nri_rows(h), OBSERVED[h], config) for h in NRI]


@pytest.fixture
def conn(config, scores):
    connection = store.connect(":memory:")
    rows = [
        {"hub_id": hub, "nri_field": field, "annual_frequency": value, "loss_ratio_rating": None}
        for hub, fields in NRI.items()
        for field, value in fields.items()
    ]
    store.replace_nri(connection, rows)
    for score in scores:
        store.save_snapshot(connection, score, "2026-10-01", config["method_version"])
    return connection
