# Weather Risk Intelligence Agent: Design

## Purpose

The company chooses a handful of hubs each year for resilience upgrades. This agent helps analysts see which hubs are most exposed to weather disruption and defend that choice.

One rule drives the design: **code computes every number, and the LLM only chooses tools and explains the results.** The ranking is the same whichever model is used, and it can be tested without one.

## Scope

| In scope | Out of scope |
|---|---|
| 22 illustrative hubs in US metros | The real hub list, throughput and financials |
| 4 scored hazard groups: winter, hurricane, flood, severe storm | Heat and drought, which rarely stop deliveries |
| Wildfire and earthquake flags, shown but not scored | Scoring those two: they are too rare for frequency × days to represent them |
| Ranking, comparison, explanation and follow-ups | Forecasting, dollar losses, ROI of specific upgrades |
| Yearly statistics for 2006–2025 | Non-US locations and US cities that aren't hubs (the agent says so) |
| Live NWS alerts as context | Live alerts changing the investment score |

## System architecture

```
 Streamlit chat UI ──HTTP──▶ FastAPI service
                              │
                              ▼
                        Agent loop ◀──▶ LLM adapter (Anthropic | OpenAI)
                              │              chosen by LLM_PROVIDER
                     tool calls│  ▲ validated JSON (AgentAnswer)
                              ▼  │
                        Tools (read-only)  ──▶ skills/*.md playbooks
                              ▼
                        Scoring engine (pure functions, no I/O or LLM)
                              ▼
                        SQLite ◀── fetch-once cache ◀── FEMA NRI, Open-Meteo, OpenFEMA
                              ▲
                        Daily job: refresh → score → compare → alert → webhook

 Live only: get_active_alerts ──▶ NWS (never feeds the score)
```

### Components

- **Streamlit UI:** a plain client of the API, with a start screen for the client ID, the chat, and tables and charts.
- **FastAPI service:** chat, hubs, scores, alerts and flag review endpoints.
- **Agent loop:** a hand-written tool loop with no framework, so it stays readable.
- **LLM adapters:** one small interface with Anthropic and OpenAI implementations.
- **Tools:** nine read-only tools. The LLM never does arithmetic.
- **Skills:** four versioned Markdown playbooks, for ranking, comparison, explanation and weather statistics. The agent loads them through `load_skill`, and each load is logged. They hold instructions only, never numbers.
- **Scoring engine:** pure functions, which import nothing from the agent or LLM layers.

### Request flow

1. The UI sends a message and client ID to `POST /chat`.
2. The agent loads that one conversation, never any other. Only its newest whole turns within a size budget reach the LLM.
3. The LLM calls tools, and code runs them and returns JSON.
4. The LLM finishes by calling `submit_answer`, whose schema is the `AgentAnswer` model.
5. Code validates the answer and checks its numbers against tool results.
6. The turn, its decision trace and any misuse flag are stored and returned.

### Structured output

`AgentAnswer` is a Pydantic model. It produces the JSON schema sent to the LLM and validates what comes back:

```
answer_markdown, reasoning_summary, ranked_hubs[], citations[{hub, metric, value, tool_call_id}],
assumptions[], confidence, confidence_reason, data_sources[], out_of_scope, misuse_flag, misuse_reason
```

Answers are enforced in two layers:

- **Schema validation:** an invalid answer gets one repair attempt.
- **Grounding check:** every cited value must match a tool result in the conversation. If it still fails after the repair attempt, the answer is shown with a visible `unverified` flag. Decimals and percentages in the prose that match no tool result get the same repair attempt and, if they remain, the same flag.

### Identity and isolation

There is no login. The start screen offers three options: enter an existing client ID, create one, or generate one.

- The client ID decides access and works like a password. Generated IDs are 128-bit random values, and custom IDs need at least 12 characters.
- Only a keyed hash of the ID is stored. It is never sent to the LLM or written to traces.
- Failed attempts are rate limited by IP.
- A keyed hash of the IP is stored with each turn for misuse review. IP doesn't control access, because shared networks would mix users and changing networks would lock them out.
- Every conversation query requires the client's hash, and another client's conversation returns 404.

### Decision trace and misuse flags

Each turn stores the model's `reasoning_summary`, every tool call with arguments, result, order and latency, the skills loaded, provider, model, tokens, prompt version, grounding outcome, repair attempts and scoring version. This records the model's stated reasoning and actions, not hidden provider reasoning.

Each turn also gets a flag of `none`, `destructive`, `abusive` or `out_of_scope`, with a reason and its source (rule or model). Deterministic rules run first, and the model's classification fills the gap. Flags are for review through `GET /admin/flags` and don't block users. All tools are read-only, so a destructive request has nothing to act on.

### Alerts

`run_refresh(as_of)` updates the data, writes a score snapshot and compares it with the previous one. An alert fires on a tier change or an EDD move of at least 5%. It is stored and POSTed to `ALERT_WEBHOOK_URL` if set. APScheduler runs this daily, and the CLI uses the same function. Scores move slowly by design, so `--as-of` backfills an earlier date to show a real change without faking data.

## Data sources and scoring

| Source | Use |
|---|---|
| FEMA National Risk Index (county) | Long-term hazard frequency and Historic Loss Ratio ratings |
| Open-Meteo Archive | Daily snowfall, precipitation and gusts |
| Open-Meteo Geocoding | US check and nearest hub |
| NWS active alerts | Live context |
| OpenFEMA declarations | Hurricane history as context, never scored |

Testing the data changed the design in three ways:

1. **FEMA's overall `RISK_SCORE` can't separate these hubs.** Sampled large metros all score 95–100, probably because it scales with population and building value. We use hazard frequency instead.
2. **Declaration counts don't measure direct hits.** Dallas County's 8 "Hurricane" declarations since 2005 cover 7 Gulf Coast storms, 6 are emergency declarations, and none included individual assistance. FEMA's hurricane frequency is 0.01 per year for Dallas and 0.30 for Miami.
3. **FEMA's percentile scores also reflect city size.** Chicago's earthquake score is in the 98th percentile with a frequency of 0.001 per year. Catastrophic flags therefore use FEMA's Historic Loss Ratio rating, at Relatively High or Very High.

Scoring the real data then exposed two problems, which are recorded here rather than hidden:

- **Coastal flood was overweighted.** At 1 closure day per event, Boston ranked first by a wide margin, because FEMA counts 7.4 coastal flood events a year there and many are likely minor. Version 1.1 uses 0.5, matching inland flood.
- **County data can misdescribe a city.** King County reports 26.9 winter weather events a year against 1.2 heavy snow days observed at the Seattle grid point, likely because the county reaches into the Cascades. The score isn't special-cased. The hub carries a visible caveat that the agent reports.

### KPI: Expected Disruption Days (EDD)

EDD estimates closure days per year. Analysts can reason about days directly, and unlike a relative index it doesn't change when a hub is added.

```
nri_days  = sum(annual_frequency × closure_days_per_event)
hazard    = 0.7 × nri_days + 0.3 × observed_days_per_year     (hurricane: nri_days only)
hub_edd   = sum of the 4 hazard groups
```

| Group | NRI hazard × closure days | Observed trigger, trailing 5 years |
|---|---|---|
| Winter | Winter weather × 0.25, ice storm × 2 | Snowfall ≥ 5 cm in a day |
| Hurricane | Hurricane × 7 | None |
| Flood | Inland flood × 0.5, coastal flood × 0.5 | Precipitation ≥ 50 mm in a day |
| Severe storm | Strong wind × 0.25, tornado × 1, hail × 0.1 | Gust ≥ 90 km/h |

Tiers are High ≥ 8, Medium 4–8 and Low < 4. All parameters live in `config/scoring.yaml` with a version number.

**Uncertainty:** each ranking is recomputed with each closure-day assumption at ±50%, and hubs report a rank range such as "#2, range 1–3".

**Known limitation:** EDD counts closure days, not damage, so rare but costly hurricanes weigh less than their financial impact would suggest. The hurricane breakdown is always available for that reason.

## Repository structure

```
weather-risk-agent/
├── README.md
├── docs/DESIGN.md
├── config/          hubs.yaml, scoring.yaml
├── data/            nri_snapshot.csv (committed), SQLite file (ignored)
├── app/
│   ├── api/         FastAPI routes and identity
│   ├── agent/       loop, prompts, schemas, grounding, flags
│   ├── llm/         base interface, Anthropic and OpenAI adapters
│   ├── skills/      ranking, comparison, explanation, weather_statistics
│   ├── tools/       tool functions and registry
│   ├── scoring/     engine, sensitivity (pure)
│   ├── data/        store, schema, NRI, Open-Meteo, OpenFEMA, NWS clients
│   └── jobs/        warm cache, refresh, alerts
├── ui/              Streamlit app
├── evals/           cases and runner
└── tests/           unit tests, no network
```

## Data storage: SQLite

| Table | Contents |
|---|---|
| `nri_hazards` | hub, hazard, annual frequency, loss ratio rating |
| `weather_cache` | hub, year, daily values as JSON, fetched-at time |
| `declarations` | hub, disaster, type, title, date |
| `score_snapshots` | hub, as-of date, method version, EDD, breakdown JSON |
| `alerts` | hub, old and new score, reason, delivery status |
| `clients` | hashed client ID, created-at time |
| `conversations` | conversation ID, hashed client ID, created-at time |
| `turns` | message, answer, trace JSON, LLM messages, flag, hashed IP |

**Why SQLite:** the data is small, with at most about 450 cache records plus scores and chats. Queries are simple lookups with no complex joins, there is one writer, and it runs from a clone with nothing to install.

**Why not the alternatives:**
- **MongoDB** is the closest alternative, and a better fit for nested traces and for redeploys. It lost because it adds Docker or Atlas setup, and running locally is the priority.
- **A vector database** has nothing to retrieve, because numbers should come from queries, not similarity search.
- **Postgres** adds operational work with no benefit at this size. It is the natural move for heavier multi-user use.

**Raw weather is fetched once, then cached per hub and year.** Finished years never change, so they stay valid and evals stay repeatable. The daily job refreshes the current year. Rankings read stored snapshots, so they don't trigger 22 API calls. `make warm-cache` prepares a demo that doesn't depend on third-party APIs. Open-Meteo's free tier limits request weight, so a first warm-up takes several minutes, waits out rate limits and resumes where it stopped. All access goes through `app/data/store.py`, so changing database touches one file.

**Deployment caveat:** hosted platforms often wipe local files, so a deployment needs a persistent disk or a re-run of the warm cache at startup.

## Assumptions and uncertainty

- Hubs are illustrative, assumed equal in size, and sit at the city centre. Elevation and drainage aren't captured.
- NRI data is county-level. Open-Meteo is modelled reanalysis on a coarse grid, not station readings, so snowfall may differ from local reports.
- NRI hazards count events differently, so frequencies aren't perfectly comparable across hazards.
- Closure days, thresholds and the 70/30 blend are judgement calls and need calibrating against real outage history.
- "Last year" means the previous calendar year. "Had snowfall" defaults to at least 0.1 cm, and the threshold appears in the answer.
- Network effects aren't modelled: a Gulf hurricane may delay freight into Dallas without touching the hub.
- Open-Meteo's free tier is for non-commercial use, so production would need a licence.
- The agent supports analysts and doesn't make the final investment decision.
