"""Open-Meteo: daily weather history and geocoding, with a per-year cache."""
from datetime import date, datetime, timedelta, timezone

import httpx

from app.data import store

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
DAILY_FIELDS = {
    "snowfall": "snowfall_sum",
    "precipitation": "precipitation_sum",
    "gust": "wind_gusts_10m_max",
}
# The archive publishes a few days late, and the current year's cache is refreshed daily.
ARCHIVE_LAG_DAYS = 5
CURRENT_YEAR_MAX_AGE = timedelta(days=1)


def fetch_days(hub: dict, start: date, end: date) -> list[dict]:
    response = httpx.get(
        ARCHIVE_URL,
        params={
            "latitude": hub["lat"],
            "longitude": hub["lon"],
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "daily": ",".join(DAILY_FIELDS.values()),
            "timezone": "auto",
        },
        timeout=60,
    )
    response.raise_for_status()
    daily = response.json()["daily"]
    return [
        {"date": day, **{name: daily[field][i] for name, field in DAILY_FIELDS.items()}}
        for i, day in enumerate(daily["time"])
    ]


def year_days(conn, hub: dict, year: int, today: date | None = None) -> list[dict]:
    """Daily rows for one year, read from the cache and fetched only when needed."""
    today = today or date.today()
    cached = store.cached_year(conn, hub["id"], year)
    if cached and cached["complete"]:
        return cached["days"]
    if cached:
        age = datetime.now(timezone.utc) - datetime.fromisoformat(cached["fetched_at"])
        if age < CURRENT_YEAR_MAX_AGE:
            return cached["days"]
    last_available = today - timedelta(days=ARCHIVE_LAG_DAYS)
    end = min(date(year, 12, 31), last_available)
    if end < date(year, 1, 1):
        return []
    try:
        days = fetch_days(hub, date(year, 1, 1), end)
    except httpx.HTTPError:
        if cached:
            return cached["days"]
        raise
    store.save_year(conn, hub["id"], year, days, complete=end == date(year, 12, 31))
    return days


def days_between(conn, hub: dict, start: date, end: date, today: date | None = None) -> list[dict]:
    days = []
    for year in range(start.year, end.year + 1):
        days += year_days(conn, hub, year, today)
    first, last = start.isoformat(), end.isoformat()
    return [day for day in days if first <= day["date"] <= last]


def geocode(name: str) -> list[dict]:
    response = httpx.get(GEOCODE_URL, params={"name": name, "count": 10, "format": "json"}, timeout=15)
    response.raise_for_status()
    return [
        {
            "name": r["name"],
            "admin1": r.get("admin1"),
            "country_code": r.get("country_code"),
            "latitude": r["latitude"],
            "longitude": r["longitude"],
            "population": r.get("population") or 0,
        }
        for r in response.json().get("results", [])
    ]
