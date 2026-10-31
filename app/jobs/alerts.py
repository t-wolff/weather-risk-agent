"""Decide when a score change is worth an alert, and deliver it."""
import httpx

from app.config import env
from app.data import store


def change_reason(old: dict, new: dict, min_change_pct: float) -> str | None:
    """Pure trigger logic: a tier change, or an EDD move past the threshold."""
    if old["tier"] != new["tier"]:
        return f"Tier changed from {old['tier']} to {new['tier']}"
    if old["edd"] > 0:
        change = 100 * (new["edd"] - old["edd"]) / old["edd"]
        if abs(change) >= min_change_pct:
            return f"EDD moved {change:+.1f}%"
    elif new["edd"] > 0:
        return f"EDD rose from zero to {new['edd']:.1f}"
    return None


def deliver(conn, alert: dict) -> None:
    url = env("ALERT_WEBHOOK_URL")
    if not url:
        return
    try:
        httpx.post(url, json=alert, timeout=10).raise_for_status()
        status = "delivered"
    except httpx.HTTPError as error:
        status = f"failed: {type(error).__name__}"
    store.set_alert_delivery(conn, alert["hub_id"], alert["as_of"], status)
