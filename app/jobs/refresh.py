"""Daily job: refresh data, write score snapshots, raise alerts."""
import argparse
from datetime import date, timedelta

from app.config import hubs, scoring_config
from app.data import open_meteo, store
from app.jobs.alerts import change_reason, deliver
from app.scoring.engine import observed_days_per_year, score_hub


def observed_for_hub(conn, hub: dict, as_of: date, config: dict, today: date) -> dict[str, float]:
    start = as_of - timedelta(days=round(365.25 * config["observed_window_years"]))
    days = open_meteo.days_between(conn, hub, start, as_of, today)
    return {
        name: observed_days_per_year(days, group["observed"]["metric"], group["observed"]["threshold"])
        for name, group in config["groups"].items()
        if group["observed"]
    }


def run_refresh(conn, as_of: date | None = None, today: date | None = None) -> dict:
    config = scoring_config()
    today = today or date.today()
    as_of = as_of or today - timedelta(days=open_meteo.ARCHIVE_LAG_DAYS)
    version = config["method_version"]
    alerts = []
    for hub in hubs():
        nri = store.nri_for_hub(conn, hub["id"])
        score = score_hub(hub, nri, observed_for_hub(conn, hub, as_of, config, today), config)
        store.save_snapshot(conn, score, as_of.isoformat(), version)
        previous = store.previous_snapshot(conn, hub["id"], as_of.isoformat(), version)
        reason = previous and change_reason(previous, score, config["alerts"]["min_change_pct"])
        if not reason:
            continue
        alert = {
            "hub_id": hub["id"],
            "as_of": as_of.isoformat(),
            "old_edd": previous["edd"],
            "new_edd": score["edd"],
            "old_tier": previous["tier"],
            "new_tier": score["tier"],
            "reason": reason,
            "delivery_status": "stored",
        }
        if store.add_alert(conn, alert):
            deliver(conn, alert)
            alerts.append(alert)
    return {"as_of": as_of.isoformat(), "hubs": len(hubs()), "alerts": alerts}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Refresh data and scores")
    parser.add_argument("--as-of", type=date.fromisoformat, help="Score as of this date")
    args = parser.parse_args()
    result = run_refresh(store.connect(), args.as_of)
    print(f"Scored {result['hubs']} hubs as of {result['as_of']}; {len(result['alerts'])} alert(s)")
    for alert in result["alerts"]:
        print(f"  {alert['hub_id']}: {alert['reason']}")
