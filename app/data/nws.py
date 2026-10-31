"""National Weather Service live alerts. Shown as context and never scored."""
import httpx

URL = "https://api.weather.gov/alerts/active"
HEADERS = {"User-Agent": "weather-risk-agent (prototype)", "Accept": "application/geo+json"}


def active_alerts(hub: dict) -> list[dict]:
    response = httpx.get(URL, params={"point": f"{hub['lat']},{hub['lon']}"}, headers=HEADERS, timeout=15)
    response.raise_for_status()
    return [
        {
            "event": feature["properties"].get("event"),
            "severity": feature["properties"].get("severity"),
            "headline": feature["properties"].get("headline"),
            "ends": feature["properties"].get("ends") or feature["properties"].get("expires"),
        }
        for feature in response.json().get("features", [])
    ]
