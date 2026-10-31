"""Read-only agent tools. Each returns JSON-ready data, and the LLM does no arithmetic."""
import math
from datetime import date
from pathlib import Path
from typing import Callable, Literal

import httpx
from pydantic import BaseModel, ConfigDict

from app.config import HISTORY_START_YEAR, find_hub, hubs, scoring_config
from app.data import nws, open_meteo, store
from app.scoring.engine import hazard_value, rank, trend_per_decade, valid_hazards
from app.scoring.sensitivity import rank_ranges

SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"
DECLARATION_NOTE = (
    "Federal declarations, not direct hits. Inland counties can appear for storms that "
    "struck elsewhere. These are context and are not used in scoring."
)
ROUNDING = 2
# A US town only makes a name ambiguous if it is at least this share of the foreign city's size.
AMBIGUOUS_POPULATION_SHARE = 0.01


class ToolError(Exception):
    """A problem the model can fix or explain, returned to it as an error result."""


class ToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ResolveLocationInput(ToolInput):
    name: str


class ListHubsInput(ToolInput):
    region: str | None


class RankHubsInput(ToolInput):
    region: str | None
    hazards: list[str] | None


class CompareHubsInput(ToolInput):
    hubs: list[str]
    hazards: list[str] | None


class ExplainHubInput(ToolInput):
    hub: str


class WeatherStatInput(ToolInput):
    hub: str
    metric: Literal["snowfall", "precipitation", "gust"]
    start_date: str
    end_date: str
    threshold: float | None


class YearlyHistoryInput(ToolInput):
    hub: str
    metric: Literal["snow_days", "heavy_rain_days", "high_gust_days", "hurricane_declarations"]
    start_year: int
    end_year: int


class ActiveAlertsInput(ToolInput):
    hub: str


class LoadSkillInput(ToolInput):
    name: Literal["ranking", "comparison", "explanation", "weather_statistics"]


WEATHER_UNITS = {"snowfall": "cm", "precipitation": "mm", "gust": "km/h"}
DEFAULT_THRESHOLDS = {"snowfall": 0.1, "precipitation": 1.0, "gust": 90.0}


def _rounded(value):
    if isinstance(value, float):
        return round(value, ROUNDING) if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _rounded(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_rounded(v) for v in value]
    return value


def _hub(name: str) -> dict:
    hub = find_hub(name)
    if not hub:
        raise ToolError(f"'{name}' is not one of our hubs. Use resolve_location or list_hubs.")
    return hub


def _scores(conn) -> list[dict]:
    scores = store.latest_snapshots(conn, scoring_config()["method_version"])
    if not scores:
        raise ToolError("No scores are available yet. An operator needs to run the warm cache.")
    return scores


def _check_hazards(hazards: list[str] | None) -> None:
    unknown = set(hazards or []) - valid_hazards(scoring_config())
    if unknown:
        raise ToolError(
            f"Unknown hazards {sorted(unknown)}. Valid: {sorted(valid_hazards(scoring_config()))}"
        )


def _distance_km(lat1, lon1, lat2, lon2) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    a = (
        math.sin((phi2 - phi1) / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    )
    return 6371 * 2 * math.asin(math.sqrt(a))


def _nearest_hub(lat: float, lon: float) -> dict:
    hub = min(hubs(), key=lambda h: _distance_km(lat, lon, h["lat"], h["lon"]))
    return {"hub": hub["name"], "distance_km": round(_distance_km(lat, lon, hub["lat"], hub["lon"]))}


def classify_location(name: str, results: list[dict]) -> dict:
    """US check first, then the hub list. Pure so that each outcome can be tested."""
    if not results:
        return {"status": "not_found", "query": name}
    us = [r for r in results if r["country_code"] == "US"]
    top = max(results, key=lambda r: r["population"])
    largest_us = max((r["population"] for r in us), default=0)
    if not us or (
        top["country_code"] != "US" and largest_us < AMBIGUOUS_POPULATION_SHARE * top["population"]
    ):
        return {"status": "non_us", "query": name, "country_code": top["country_code"]}
    if top["country_code"] != "US":
        options = sorted(us, key=lambda r: -r["population"])[:3]
        return {
            "status": "ambiguous",
            "query": name,
            "options": [f"{top['name']}, {top['country_code']}"]
            + [f"{r['name']}, {r['admin1']}, US" for r in options],
        }
    best = max(us, key=lambda r: r["population"])
    hub = find_hub(best["name"])
    if hub:
        return {"status": "hub", "query": name, "hub": hub["name"]}
    return {
        "status": "us_not_hub",
        "query": name,
        "location": f"{best['name']}, {best['admin1']}",
        "nearest_hub": _nearest_hub(best["latitude"], best["longitude"]),
    }


def resolve_location(conn, args: ResolveLocationInput) -> dict:
    try:
        return classify_location(args.name, open_meteo.geocode(args.name))
    except httpx.HTTPError:
        hub = find_hub(args.name)
        if hub:
            return {"status": "hub", "query": args.name, "hub": hub["name"]}
        return {
            "status": "unverified",
            "query": args.name,
            "note": "Geocoder unavailable, and this isn't one of our hubs.",
        }


def list_hubs(conn, args: ListHubsInput) -> dict:
    rows = [h for h in hubs() if not args.region or h["region"].lower() == args.region.lower()]
    return {"hubs": [{k: h[k] for k in ("name", "state", "region")} for h in rows]}


def _select(scores: list[dict], region: str | None) -> list[dict]:
    if not region:
        return scores
    selected = [s for s in scores if s["region"].lower() == region.lower()]
    if not selected:
        raise ToolError("Unknown region. Valid regions: Midwest, South, West, Northeast.")
    return selected


def rank_hubs(conn, args: RankHubsInput) -> dict:
    _check_hazards(args.hazards)
    scores = _select(_scores(conn), args.region)
    config = scoring_config()
    ranges = rank_ranges(scores, config, args.hazards)
    by_id = {s["hub_id"]: s for s in scores}
    return {
        "scope": {"region": args.region or "all", "hazards": args.hazards or "all"},
        "unit": "expected disruption days per year",
        "as_of": scores[0]["as_of"],
        "method_version": config["method_version"],
        "ranking": [
            {
                "rank": row["rank"],
                "hub": row["name"],
                "edd": row["value"],
                "overall_edd": by_id[row["hub_id"]]["edd"],
                "overall_tier": by_id[row["hub_id"]]["tier"],
                "rank_range": ranges[row["hub_id"]],
                "catastrophic_flags": by_id[row["hub_id"]]["catastrophic_flags"],
                "caveat": by_id[row["hub_id"]]["caveat"],
            }
            for row in rank(scores, args.hazards)
        ],
    }


def _hub_view(score: dict) -> dict:
    return {
        "hub": score["name"],
        "edd": score["edd"],
        "tier": score["tier"],
        "groups": score["groups"],
        "catastrophic_flags": score["catastrophic_flags"],
        "caveat": score["caveat"],
    }


def _gap(views: list[dict], value: Callable[[dict], float]) -> dict:
    """From the rounded figures, so the gap is exactly what a reader gets by subtracting."""
    highest, lowest = max(views, key=value), min(views, key=value)
    days = round(value(highest), ROUNDING) - round(value(lowest), ROUNDING)
    return {"highest": highest["hub"], "lowest": lowest["hub"], "days": days}


def compare_hubs(conn, args: CompareHubsInput) -> dict:
    _check_hazards(args.hazards)
    wanted = [_hub(name)["id"] for name in args.hubs]
    scores = _scores(conn)
    positions = {
        hazard: {row["hub_id"]: row["rank"] for row in rank(scores, [hazard])}
        for hazard in (args.hazards or list(scoring_config()["groups"]))
    }
    views = []
    for score in scores:
        if score["hub_id"] in wanted:
            view = _hub_view(score)
            view["selected_hazards_edd"] = {
                hazard: hazard_value(score, [hazard]) for hazard in positions
            }
            view["national_rank_by_hazard"] = {
                hazard: ranks[score["hub_id"]] for hazard, ranks in positions.items()
            }
            views.append(view)
    gaps = {}
    if len(views) > 1:
        gaps = {hazard: _gap(views, lambda v: v["selected_hazards_edd"][hazard]) for hazard in positions}
        gaps["overall"] = _gap(views, lambda v: v["edd"])
    return {"hubs_scored": len(scores), "as_of": scores[0]["as_of"], "comparison": views,
            "difference_by_hazard": gaps}


def explain_hub(conn, args: ExplainHubInput) -> dict:
    hub = _hub(args.hub)
    scores = _scores(conn)
    score = next(s for s in scores if s["hub_id"] == hub["id"])
    config = scoring_config()
    drivers = sorted(
        (
            {"hazard": component, "group": group_name, "pre_blend_days": detail["nri_days"]}
            for group_name, group in score["groups"].items()
            for component, detail in group["components"].items()
        ),
        key=lambda d: -d["pre_blend_days"],
    )
    overall = next(r for r in rank(scores) if r["hub_id"] == hub["id"])
    return {
        **_hub_view(score),
        "as_of": score["as_of"],
        "rank": overall["rank"],
        "hubs_scored": len(scores),
        "rank_range": rank_ranges(scores, config)[hub["id"]],
        "top_nri_drivers": drivers[:4],
        "tier_thresholds": config["tiers"],
        "blend": config["blend"],
        "method_version": config["method_version"],
    }


def weather_stat(conn, args: WeatherStatInput) -> dict:
    hub = _hub(args.hub)
    try:
        start, end = date.fromisoformat(args.start_date), date.fromisoformat(args.end_date)
    except ValueError as error:
        raise ToolError("Dates must be YYYY-MM-DD.") from error
    if end < start:
        raise ToolError("end_date is before start_date.")
    if start.year < HISTORY_START_YEAR:
        raise ToolError(f"History starts in {HISTORY_START_YEAR}.")
    if end > date.today():
        raise ToolError("end_date is in the future. This agent doesn't forecast.")
    try:
        days = open_meteo.days_between(conn, hub, start, end)
    except httpx.HTTPError as error:
        raise ToolError("Weather data isn't cached and the source is unavailable.") from error
    threshold = args.threshold if args.threshold is not None else DEFAULT_THRESHOLDS[args.metric]
    values = [day[args.metric] for day in days if day[args.metric] is not None]
    if not values:
        raise ToolError("No weather data is available for that period.")
    matching = sum(value >= threshold for value in values)
    return {
        "hub": hub["name"],
        "metric": args.metric,
        "unit": WEATHER_UNITS[args.metric],
        "start_date": days[0]["date"],
        "end_date": days[-1]["date"],
        "threshold": threshold,
        "days_with_data": len(values),
        "days_at_or_above_threshold": matching,
        "percent_of_days": 100 * matching / len(values),
        "total": sum(values),
        "maximum": max(values),
        "source": "Open-Meteo reanalysis (modelled, not station readings)",
    }


def yearly_history(conn, args: YearlyHistoryInput) -> dict:
    hub = _hub(args.hub)
    if args.start_year < HISTORY_START_YEAR or args.end_year < args.start_year:
        raise ToolError(f"Use a year range from {HISTORY_START_YEAR} onwards.")
    today = date.today()
    if args.end_year > today.year:
        raise ToolError("end_year is in the future.")
    if args.metric == "hurricane_declarations":
        rows = store.hurricane_declarations(conn, hub["id"], args.start_year, args.end_year)
        counts = {year: 0 for year in range(args.start_year, args.end_year + 1)}
        for row in rows:
            counts[int(row["declaration_date"][:4])] += 1
        return {
            "hub": hub["name"],
            "metric": args.metric,
            "note": DECLARATION_NOTE,
            "total": len(rows),
            "years": [{"year": y, "count": c} for y, c in counts.items()],
            "declarations": [
                {k: row[k] for k in ("declaration_date", "declaration_type", "title")}
                for row in rows
            ],
        }
    spec = {
        "snow_days": ("winter", "snowfall"),
        "heavy_rain_days": ("flood", "precipitation"),
        "high_gust_days": ("severe_storm", "gust"),
    }[args.metric]
    observed = scoring_config()["groups"][spec[0]]["observed"]
    years = []
    try:
        for year in range(args.start_year, args.end_year + 1):
            days = open_meteo.year_days(conn, hub, year, today)
            count = sum((d[spec[1]] or 0) >= observed["threshold"] for d in days)
            years.append({"year": year, "count": count, "partial": year == today.year})
    except httpx.HTTPError as error:
        raise ToolError("Weather data isn't cached and the source is unavailable.") from error
    full_years = [(row["year"], float(row["count"])) for row in years if not row["partial"]]
    counts = [count for _, count in full_years]
    return {
        "hub": hub["name"],
        "metric": args.metric,
        "threshold": observed["threshold"],
        "unit": WEATHER_UNITS[spec[1]],
        "years": years,
        "average_per_full_year": sum(counts) / len(counts) if counts else None,
        "trend_per_decade": trend_per_decade(full_years),
        "trend_note": "Simple descriptive slope with no significance test.",
    }


def get_active_alerts(conn, args: ActiveAlertsInput) -> dict:
    hub = _hub(args.hub)
    try:
        alerts = nws.active_alerts(hub)
    except httpx.HTTPError as error:
        raise ToolError("NWS is unavailable right now, so live alerts can't be checked.") from error
    return {"hub": hub["name"], "alerts": alerts, "note": "Live context only, not part of the score."}


def load_skill(conn, args: LoadSkillInput) -> dict:
    path = SKILLS_DIR / f"{args.name}.md"
    text = path.read_text()
    header, _, body = text.partition("\n\n")
    return {"skill": args.name, "version": header.removeprefix("version: ").strip(), "instructions": body}


class Tool(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str
    description: str
    input_model: type[ToolInput]
    run: Callable


TOOLS = [
    Tool(name="resolve_location", input_model=ResolveLocationInput, run=resolve_location,
         description="Check a place the user named: first whether it is in the US, then whether it is "
                     "one of our hubs. Use before answering about any place you aren't sure is a hub."),
    Tool(name="list_hubs", input_model=ListHubsInput, run=list_hubs,
         description="List hubs, optionally by region: Midwest, South, West or Northeast."),
    Tool(name="rank_hubs", input_model=RankHubsInput, run=rank_hubs,
         description="Rank hubs by Expected Disruption Days, with rank ranges. Optional region. Optional "
                     "hazards: groups (winter, hurricane, flood, severe_storm) or components (winter_weather, "
                     "ice_storm, inland_flood, coastal_flood, strong_wind, tornado, hail)."),
    Tool(name="compare_hubs", input_model=CompareHubsInput, run=compare_hubs,
         description="Compare named hubs in full, with each hub's national rank per hazard."),
    Tool(name="explain_hub", input_model=ExplainHubInput, run=explain_hub,
         description="Full score breakdown for one hub: drivers, rank range, flags and method."),
    Tool(name="weather_stat", input_model=WeatherStatInput, run=weather_stat,
         description="Historical statistic for a hub and date range, such as percent of days with snowfall. "
                     "Pass threshold as null for the default. Dates are YYYY-MM-DD."),
    Tool(name="yearly_history", input_model=YearlyHistoryInput, run=yearly_history,
         description=f"Counts per year from {HISTORY_START_YEAR}, with average and trend. Use "
                     "hurricane_declarations for hurricane history, which are federal declarations."),
    Tool(name="get_active_alerts", input_model=ActiveAlertsInput, run=get_active_alerts,
         description="Live NWS alerts for a hub. Context only, never part of the score."),
    Tool(name="load_skill", input_model=LoadSkillInput, run=load_skill,
         description="Load the playbook for the type of answer you are writing."),
]
TOOLS_BY_NAME = {tool.name: tool for tool in TOOLS}


def run_tool(conn, name: str, arguments: dict):
    """Returns (result, is_error)."""
    tool = TOOLS_BY_NAME.get(name)
    if not tool:
        return {"error": f"Unknown tool {name}"}, True
    try:
        return _rounded(tool.run(conn, tool.input_model.model_validate(arguments))), False
    except ToolError as error:
        return {"error": str(error)}, True
    except ValueError as error:
        return {"error": f"Invalid arguments: {error}"}, True
