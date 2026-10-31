from unittest.mock import patch

import httpx
import pytest

from app.config import scoring_config
from app.data import open_meteo, store
from app.tools.registry import classify_location, run_tool


def place(name, country, population, admin1="State", lat=33.0, lon=-96.0):
    return {"name": name, "admin1": admin1, "country_code": country, "latitude": lat,
            "longitude": lon, "population": population}


def test_location_not_found():
    assert classify_location("Zzz", [])["status"] == "not_found"


def test_location_non_us():
    assert classify_location("Berlin", [place("Berlin", "DE", 3_000_000)])["status"] == "non_us"


def test_location_ambiguous_when_foreign_city_shares_a_us_name():
    results = [place("Paris", "FR", 2_000_000), place("Paris", "US", 24_000, "Texas")]
    outcome = classify_location("Paris", results)
    assert outcome["status"] == "ambiguous"
    assert "Paris, Texas, US" in outcome["options"]


def test_foreign_city_with_only_tiny_us_namesakes_is_non_us():
    results = [place("Berlin", "DE", 3_400_000), place("Berlin", "US", 10_000, "New Hampshire")]
    assert classify_location("Berlin", results)["status"] == "non_us"


def test_location_hub():
    assert classify_location("Dallas", [place("Dallas", "US", 1_300_000)])["hub"] == "Dallas"


def test_location_us_not_hub_names_nearest():
    outcome = classify_location("Fort Worth", [place("Fort Worth", "US", 900_000, "Texas", 32.75, -97.33)])
    assert outcome["status"] == "us_not_hub"
    assert outcome["nearest_hub"]["hub"] == "Dallas"


def test_geocoder_outage_falls_back_to_hub_list(conn):
    with patch("app.data.open_meteo.geocode", side_effect=httpx.ConnectError("down")):
        assert run_tool(conn, "resolve_location", {"name": "Denver"})[0]["status"] == "hub"
        assert run_tool(conn, "resolve_location", {"name": "Tulsa"})[0]["status"] == "unverified"


def test_rank_hubs_rounds_and_filters(conn):
    result, is_error = run_tool(conn, "rank_hubs", {"region": "South", "hazards": ["hurricane"]})
    assert not is_error
    assert [row["hub"] for row in result["ranking"]] == ["Miami", "Dallas"]
    assert result["ranking"][0]["edd"] == 2.1


def test_unknown_hazard_is_an_error_the_model_can_fix(conn):
    result, is_error = run_tool(conn, "rank_hubs", {"region": None, "hazards": ["locusts"]})
    assert is_error and "locusts" in result["error"]


def test_unknown_hub_is_an_error(conn):
    result, is_error = run_tool(conn, "explain_hub", {"hub": "Atlantis"})
    assert is_error


def test_explain_hub_lists_drivers(conn):
    result, _ = run_tool(conn, "explain_hub", {"hub": "dallas"})
    days = [d["pre_blend_days"] for d in result["top_nri_drivers"]]
    assert days == sorted(days, reverse=True)
    assert days[0] == 2.0
    assert result["hubs_scored"] == 3


def test_invalid_arguments_are_rejected(conn):
    _, is_error = run_tool(conn, "explain_hub", {"hub": "dallas", "extra": 1})
    assert is_error


def seed_year(conn, hub_id, year, snow_days):
    days = [
        {"date": f"{year}-01-{i + 1:02d}", "snowfall": 6.0 if i < snow_days else 0.0,
         "precipitation": 0.0, "gust": 10.0}
        for i in range(20)
    ]
    store.save_year(conn, hub_id, year, days, complete=True)


def test_weather_stat_percent(conn):
    seed_year(conn, "denver", 2025, 5)
    result, is_error = run_tool(conn, "weather_stat", {
        "hub": "Denver", "metric": "snowfall", "start_date": "2025-01-01",
        "end_date": "2025-12-31", "threshold": None,
    })
    assert not is_error
    assert result["percent_of_days"] == 25.0
    assert result["threshold"] == 0.1


def test_weather_stat_rejects_the_future(conn):
    _, is_error = run_tool(conn, "weather_stat", {
        "hub": "Denver", "metric": "snowfall", "start_date": "2099-01-01",
        "end_date": "2099-02-01", "threshold": None,
    })
    assert is_error


def test_yearly_history_counts_and_trend(conn):
    seed_year(conn, "denver", 2023, 2)
    seed_year(conn, "denver", 2024, 4)
    result, _ = run_tool(conn, "yearly_history", {
        "hub": "Denver", "metric": "snow_days", "start_year": 2023, "end_year": 2024,
    })
    assert [row["count"] for row in result["years"]] == [2, 4]
    assert result["trend_per_decade"] == 20.0


def test_hurricane_history_is_labelled_as_declarations(conn):
    store.replace_declarations(conn, "dallas", [{
        "hub_id": "dallas", "disaster_number": 1, "declaration_type": "EM", "incident_type": "Hurricane",
        "title": "HURRICANE RITA", "declaration_date": "2005-09-21",
    }])
    result, _ = run_tool(conn, "yearly_history", {
        "hub": "Dallas", "metric": "hurricane_declarations", "start_year": 2006, "end_year": 2006,
    })
    assert result["total"] == 0
    assert "not direct hits" in result["note"]


def test_finished_year_is_served_from_cache_without_a_fetch(conn):
    seed_year(conn, "denver", 2020, 1)
    with patch("app.data.open_meteo.fetch_days", side_effect=AssertionError("should not fetch")):
        assert len(open_meteo.year_days(conn, {"id": "denver"}, 2020)) == 20


def test_cache_miss_fetches_once_then_stores(conn):
    from datetime import date
    hub = {"id": "denver"}
    day = [{"date": "2021-01-01", "snowfall": 0.0, "precipitation": 0.0, "gust": 1.0}]
    with patch("app.data.open_meteo.fetch_days", return_value=day) as fetch:
        open_meteo.year_days(conn, hub, 2021, date(2026, 10, 31))
        open_meteo.year_days(conn, hub, 2021, date(2026, 10, 31))
    assert fetch.call_count == 1


def test_stale_current_year_survives_a_source_outage(conn):
    from datetime import date
    day = [{"date": "2026-01-01", "snowfall": 0.0, "precipitation": 0.0, "gust": 1.0}]
    store.save_year(conn, "denver", 2026, day, complete=False)
    conn.execute("UPDATE weather_cache SET fetched_at = '2020-01-01T00:00:00+00:00'")
    with patch("app.data.open_meteo.fetch_days", side_effect=httpx.ConnectError("down")):
        assert open_meteo.year_days(conn, {"id": "denver"}, 2026, date(2026, 10, 31)) == day


def test_compare_hubs_returns_gaps_from_the_rounded_figures(conn):
    result, _ = run_tool(conn, "compare_hubs", {"hubs": ["Dallas", "Miami", "Denver"], "hazards": None})
    gaps, views = result["difference_by_hazard"], result["comparison"]
    assert set(gaps) == {"overall", "winter", "hurricane", "flood", "severe_storm"}
    for hazard, gap in gaps.items():
        values = {v["hub"]: v["edd"] if hazard == "overall" else v["selected_hazards_edd"][hazard] for v in views}
        assert gap["highest"] == max(values, key=values.get) and gap["lowest"] == min(values, key=values.get)
        assert gap["days"] == round(max(values.values()) - min(values.values()), 2)


def test_compare_hubs_has_no_gaps_for_one_hub(conn):
    result, _ = run_tool(conn, "compare_hubs", {"hubs": ["Dallas"], "hazards": None})
    assert result["difference_by_hazard"] == {}
