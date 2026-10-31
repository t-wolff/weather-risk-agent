import pytest

from app.config import hub_by_id
from app.jobs.alerts import change_reason
from app.scoring.engine import (
    catastrophic_flags, hazard_value, observed_days_per_year, rank, score_hub, tier, trend_per_decade,
)
from app.scoring.sensitivity import rank_ranges


def test_edd_matches_hand_calculation(scores):
    dallas = next(s for s in scores if s["hub_id"] == "dallas")
    # winter: nri 4*.25 + 1*2 = 3.0; 0.7*3.0 + 0.3*1 = 2.4
    assert dallas["groups"]["winter"]["edd"] == pytest.approx(2.4)
    # hurricane: 0.01*7, with no observed blend
    assert dallas["groups"]["hurricane"]["edd"] == pytest.approx(0.07)
    # flood: nri 4*.5 = 2.0; 0.7*2.0 + 0.3*2 = 2.0
    assert dallas["groups"]["flood"]["edd"] == pytest.approx(2.0)
    # severe: nri 1 + 1 + 1 = 3.0; 0.7*3.0 + 0.3*1 = 2.4
    assert dallas["groups"]["severe_storm"]["edd"] == pytest.approx(2.4)
    assert dallas["edd"] == pytest.approx(6.87)
    assert dallas["tier"] == "Medium"


def test_missing_nri_counts_as_zero(scores):
    denver = next(s for s in scores if s["hub_id"] == "denver")
    assert denver["groups"]["hurricane"]["edd"] == 0


def test_none_frequency_counts_as_zero(config):
    score = score_hub(hub_by_id("denver"), {"HRCN": {"annual_frequency": None}}, {}, config)
    assert score["edd"] == 0


@pytest.mark.parametrize("edd,expected", [(8, "High"), (7.99, "Medium"), (4, "Medium"), (3.99, "Low")])
def test_tier_boundaries(config, edd, expected):
    assert tier(edd, config) == expected


def test_observed_days_scale_to_a_year():
    days = [{"snowfall": 6.0}] * 10 + [{"snowfall": 0.0}] * 720 + [{"snowfall": None}] * 5
    assert observed_days_per_year(days, "snowfall", 5) == pytest.approx(10 / (730 / 365.25))


def test_observed_threshold_is_inclusive():
    assert observed_days_per_year([{"gust": 90.0}] * 365, "gust", 90) > 0


def test_rank_highest_first_and_ties_deterministic(scores):
    assert [row["rank"] for row in rank(scores)] == [1, 2, 3]
    values = [row["value"] for row in rank(scores)]
    assert values == sorted(values, reverse=True)
    tied = [{**scores[0], "hub_id": "b"}, {**scores[0], "hub_id": "a"}]
    assert [row["hub_id"] for row in rank(tied)] == ["a", "b"]


def test_hazard_value_for_group_and_component(scores):
    dallas = next(s for s in scores if s["hub_id"] == "dallas")
    assert hazard_value(dallas, ["hurricane"]) == pytest.approx(0.07)
    assert hazard_value(dallas, ["ice_storm"]) == pytest.approx(2.0)


def test_rank_ranges_contain_actual_rank(scores, config):
    ranges = rank_ranges(scores, config)
    for row in rank(scores):
        assert ranges[row["hub_id"]]["best"] <= row["rank"] <= ranges[row["hub_id"]]["worst"]


def test_rank_range_widens_when_an_assumption_flips_the_order(config):
    a = score_hub(hub_by_id("dallas"), {"WNTW": {"annual_frequency": 4.0}}, {}, config)
    b = score_hub(hub_by_id("miami"), {"HRCN": {"annual_frequency": 0.1}}, {}, config)
    # a = .7, b = .7: scaling either assumption flips them
    assert rank_ranges([a, b], config)["dallas"] == {"best": 1, "worst": 2}


def test_rescaled_edd_matches_a_full_rescore(config):
    from app.scoring import sensitivity
    nri = {"WNTW": {"annual_frequency": 4.0}, "HRCN": {"annual_frequency": 1.0}}
    base = score_hub(hub_by_id("dallas"), nri, {"winter": 2.0}, config)
    changed = sensitivity._rescale(base, "winter", "winter_weather", 1.5, config)
    # winter nri rises from 1.0 to 1.5, blended at 0.7
    assert changed["edd"] == pytest.approx(base["edd"] + 0.35)


def test_catastrophic_flags_use_fema_rating(config):
    nri = {
        "WFIR": {"loss_ratio_rating": "Very High"},
        "ERQK": {"loss_ratio_rating": "Relatively Low"},
    }
    assert catastrophic_flags(nri, config) == [{"hazard": "wildfire", "fema_rating": "Very High"}]


def test_trend_per_decade():
    assert trend_per_decade([(2000, 1.0), (2001, 2.0), (2002, 3.0)]) == pytest.approx(10)
    assert trend_per_decade([(2000, 1.0)]) is None


@pytest.mark.parametrize("old,new,fires", [
    ({"tier": "Medium", "edd": 7.9}, {"tier": "High", "edd": 8.0}, True),
    ({"tier": "High", "edd": 10.0}, {"tier": "High", "edd": 10.5}, True),
    ({"tier": "High", "edd": 10.0}, {"tier": "High", "edd": 9.5}, True),
    ({"tier": "High", "edd": 10.0}, {"tier": "High", "edd": 10.4}, False),
])
def test_alert_trigger(old, new, fires):
    assert bool(change_reason(old, new, 5)) is fires
