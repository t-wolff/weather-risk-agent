"""Warm the cache: NRI, 20 years of weather, declarations, then a first score."""
import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

import httpx

from app.config import HISTORY_START_YEAR, hubs
from app.data import nri, open_meteo, openfema, store
from app.jobs.refresh import run_refresh

RATE_LIMIT_WAIT_SECONDS = 65
RETRY_WAIT_SECONDS = 10
RATE_LIMIT_MAX_WAITS = 60


def warm_cache(conn, refresh_nri: bool = False) -> None:
    if refresh_nri or not nri.SNAPSHOT.exists():
        nri.fetch_snapshot()
    store.replace_nri(conn, nri.load_snapshot())
    print(f"NRI loaded for {len(hubs())} hubs")

    today = date.today()
    for hub in hubs():
        missing = [
            year for year in range(HISTORY_START_YEAR, today.year + 1)
            if not (store.cached_year(conn, hub["id"], year) or {}).get("complete")
        ]
        if missing:
            end = today - timedelta(days=open_meteo.ARCHIVE_LAG_DAYS)
            days = _fetch_waiting(hub, date(missing[0], 1, 1), end)
            by_year: dict[int, list] = {}
            for day in days:
                by_year.setdefault(int(day["date"][:4]), []).append(day)
            for year, rows in by_year.items():
                if year in missing:
                    store.save_year(conn, hub["id"], year, rows, complete=year < today.year)
        print(f"Weather ready: {hub['name']}")

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = pool.map(_declarations, hubs())
    for hub, rows in zip(hubs(), results):
        if rows is not None:
            store.replace_declarations(conn, hub["id"], rows)

    result = run_refresh(conn)
    print(f"Scored {result['hubs']} hubs as of {result['as_of']}")


def _fetch_waiting(hub: dict, start: date, end: date) -> list[dict]:
    """Open-Meteo's free tier limits request weight, so wait and resume rather than fail."""
    for _ in range(RATE_LIMIT_MAX_WAITS):
        try:
            return open_meteo.fetch_days(hub, start, end)
        except httpx.TransportError as error:
            print(f"Network problem on {hub['name']} ({type(error).__name__}), retrying")
            time.sleep(RETRY_WAIT_SECONDS)
        except httpx.HTTPStatusError as error:
            if error.response.status_code != 429:
                raise
            print(f"Rate limited on {hub['name']}, waiting {RATE_LIMIT_WAIT_SECONDS}s")
            time.sleep(RATE_LIMIT_WAIT_SECONDS)
    raise RuntimeError(f"Still rate limited on {hub['name']}; run the command again later")


def _declarations(hub: dict) -> list[dict] | None:
    try:
        return openfema.fetch_declarations(hub, HISTORY_START_YEAR)
    except httpx.HTTPError as error:
        print(f"Declarations unavailable for {hub['name']}: {error}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data for the agent")
    parser.add_argument("--refresh-nri", action="store_true", help="Rebuild the committed NRI snapshot")
    args = parser.parse_args()
    warm_cache(store.connect(), args.refresh_nri)
