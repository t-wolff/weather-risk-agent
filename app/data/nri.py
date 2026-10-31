"""FEMA National Risk Index: county hazard frequency and loss ratio ratings."""
import csv

import httpx

from app.config import ROOT, hubs, scoring_config

SERVICE = (
    "https://services.arcgis.com/XG15cJAlne2vxtgt/arcgis/rest/services/"
    "National_Risk_Index_Counties/FeatureServer/0/query"
)
SNAPSHOT = ROOT / "data" / "nri_snapshot.csv"


def nri_fields() -> list[str]:
    config = scoring_config()
    fields = [
        component["nri_field"]
        for group in config["groups"].values()
        for component in group["components"].values()
    ]
    return fields + list(config["catastrophic_flags"]["hazards"].values())


def fetch_snapshot() -> None:
    """Pull our counties from FEMA and write the CSV committed to the repo."""
    fips = {h["fips"]: h["id"] for h in hubs()}
    out_fields = ["STCOFIPS"] + [
        f"{field}_{suffix}" for field in nri_fields() for suffix in ("AFREQ", "HLRR")
    ]
    where = "STCOFIPS IN ({})".format(",".join(f"'{code}'" for code in fips))
    response = httpx.post(
        SERVICE,
        data={"where": where, "outFields": ",".join(out_fields),
              "returnGeometry": "false", "f": "json"},
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    if "error" in payload:
        raise RuntimeError(f"NRI query failed: {payload['error']}")
    rows = []
    for feature in payload["features"]:
        attributes = feature["attributes"]
        for field in nri_fields():
            rows.append({
                "hub_id": fips[attributes["STCOFIPS"]],
                "nri_field": field,
                "annual_frequency": attributes[f"{field}_AFREQ"],
                "loss_ratio_rating": attributes[f"{field}_HLRR"],
            })
    found = {row["hub_id"] for row in rows}
    missing = set(fips.values()) - found
    if missing:
        raise RuntimeError(f"NRI returned no data for: {sorted(missing)}")
    rows.sort(key=lambda r: (r["hub_id"], r["nri_field"]))
    with SNAPSHOT.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_snapshot() -> list[dict]:
    with SNAPSHOT.open(newline="") as f:
        return [
            {
                **row,
                "annual_frequency": float(row["annual_frequency"]) if row["annual_frequency"] else None,
                "loss_ratio_rating": row["loss_ratio_rating"] or None,
            }
            for row in csv.DictReader(f)
        ]
