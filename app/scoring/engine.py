"""Expected Disruption Days. Pure functions: no I/O and no LLM."""
import math


def observed_days_per_year(days: list[dict], metric: str, threshold: float) -> float:
    """Days at or above the threshold, per year of data."""
    values = [day[metric] for day in days if day.get(metric) is not None]
    if not values:
        return 0.0
    return sum(value >= threshold for value in values) / (len(values) / 365.25)


def tier(edd: float, config: dict) -> str:
    if edd >= config["tiers"]["high"]:
        return "High"
    return "Medium" if edd >= config["tiers"]["medium"] else "Low"


def catastrophic_flags(nri: dict[str, dict], config: dict) -> list[dict]:
    flags_config = config["catastrophic_flags"]
    flags = []
    for hazard, field in flags_config["hazards"].items():
        rating = (nri.get(field) or {}).get("loss_ratio_rating")
        if rating in flags_config["ratings"]:
            flags.append({"hazard": hazard, "fema_rating": rating})
    return flags


def score_hub(hub: dict, nri: dict[str, dict], observed: dict[str, float], config: dict) -> dict:
    """Score one hub. `observed` holds days per year by hazard group."""
    groups = {}
    for name, group in config["groups"].items():
        components = {}
        for component, spec in group["components"].items():
            frequency = (nri.get(spec["nri_field"]) or {}).get("annual_frequency") or 0.0
            components[component] = {
                "annual_frequency": frequency,
                "closure_days_per_event": spec["closure_days"],
                "nri_days": frequency * spec["closure_days"],
            }
        nri_days = sum(c["nri_days"] for c in components.values())
        if group["observed"]:
            observed_days = observed.get(name, 0.0)
            edd = config["blend"]["nri"] * nri_days + config["blend"]["observed"] * observed_days
        else:
            observed_days = None
            edd = nri_days
        groups[name] = {
            "edd": edd,
            "nri_days": nri_days,
            "observed_days_per_year": observed_days,
            "components": components,
        }
    total = sum(group["edd"] for group in groups.values())
    return {
        "hub_id": hub["id"],
        "name": hub["name"],
        "state": hub["state"],
        "region": hub["region"],
        "edd": total,
        "tier": tier(total, config),
        "groups": groups,
        "catastrophic_flags": catastrophic_flags(nri, config),
        "caveat": hub.get("caveat"),
    }


def hazard_value(score: dict, hazards: list[str] | None) -> float:
    """EDD limited to hazard groups or single components such as ice_storm."""
    if not hazards:
        return score["edd"]
    total = 0.0
    for hazard in hazards:
        if hazard in score["groups"]:
            total += score["groups"][hazard]["edd"]
            continue
        for group in score["groups"].values():
            if hazard in group["components"]:
                total += group["components"][hazard]["nri_days"]
    return total


def valid_hazards(config: dict) -> set[str]:
    groups = config["groups"]
    return set(groups) | {c for group in groups.values() for c in group["components"]}


def rank(scores: list[dict], hazards: list[str] | None = None) -> list[dict]:
    """Highest first. Ties break on hub id so that order is deterministic."""
    ordered = sorted(scores, key=lambda s: (-hazard_value(s, hazards), s["hub_id"]))
    return [
        {"rank": i, "hub_id": s["hub_id"], "name": s["name"], "value": hazard_value(s, hazards)}
        for i, s in enumerate(ordered, start=1)
    ]


def trend_per_decade(rows: list[tuple[int, float]]) -> float | None:
    """Least-squares slope, in units per decade."""
    if len(rows) < 2:
        return None
    mean_x = sum(x for x, _ in rows) / len(rows)
    mean_y = sum(y for _, y in rows) / len(rows)
    denominator = sum((x - mean_x) ** 2 for x, _ in rows)
    if math.isclose(denominator, 0):
        return None
    return 10 * sum((x - mean_x) * (y - mean_y) for x, y in rows) / denominator
