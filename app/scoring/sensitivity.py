"""Rank ranges: how far a hub moves when one closure-day assumption changes."""
from app.scoring.engine import rank


def _rescale(score: dict, group_name: str, component: str, factor: float, config: dict) -> dict:
    group = score["groups"][group_name]
    delta = group["components"][component]["nri_days"] * (factor - 1)
    has_observed = config["groups"][group_name]["observed"] is not None
    edd_delta = delta * (config["blend"]["nri"] if has_observed else 1)
    groups = {
        **score["groups"],
        group_name: {
            **group,
            "edd": group["edd"] + edd_delta,
            "components": {
                **group["components"],
                component: {
                    **group["components"][component],
                    "nri_days": group["components"][component]["nri_days"] * factor,
                },
            },
        },
    }
    return {**score, "edd": score["edd"] + edd_delta, "groups": groups}


def rank_ranges(scores: list[dict], config: dict, hazards: list[str] | None = None) -> dict:
    """Best and worst rank per hub across every one-at-a-time scenario."""
    positions = {row["hub_id"]: [row["rank"]] for row in rank(scores, hazards)}
    for group_name, group in config["groups"].items():
        for component in group["components"]:
            for factor in config["sensitivity_factors"]:
                changed = [_rescale(s, group_name, component, factor, config) for s in scores]
                for row in rank(changed, hazards):
                    positions[row["hub_id"]].append(row["rank"])
    return {hub: {"best": min(p), "worst": max(p)} for hub, p in positions.items()}
