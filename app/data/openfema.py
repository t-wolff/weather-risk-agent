"""OpenFEMA disaster declarations. Context only: never used in scoring."""
import httpx

URL = "https://www.fema.gov/api/open/v2/DisasterDeclarationsSummaries"


def fetch_declarations(hub: dict, start_year: int) -> list[dict]:
    state, county = hub["fips"][:2], hub["fips"][2:]
    response = httpx.get(
        URL,
        params={
            "$filter": (
                f"fipsStateCode eq '{state}' and fipsCountyCode eq '{county}' "
                f"and declarationDate ge '{start_year}-01-01T00:00:00.000Z'"
            ),
            "$top": 1000,
            "$select": "disasterNumber,declarationType,incidentType,declarationTitle,declarationDate",
        },
        timeout=60,
    )
    response.raise_for_status()
    return [
        {
            "hub_id": hub["id"],
            "disaster_number": row["disasterNumber"],
            "declaration_type": row["declarationType"],
            "incident_type": row["incidentType"],
            "title": row["declarationTitle"],
            "declaration_date": row["declarationDate"][:10],
        }
        for row in response.json()["DisasterDeclarationsSummaries"]
    ]
