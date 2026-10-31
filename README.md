# Weather Risk Intelligence Agent

A conversational agent that helps logistics analysts decide which hubs are most exposed to weather disruption and should be prioritised for resilience investment.

The agent ranks 22 illustrative US hubs by **Expected Disruption Days (EDD)**, an estimate of closure days per year. Code computes every number, and the LLM only chooses tools and explains the results. The design is in [docs/DESIGN.md](docs/DESIGN.md).

## Run it

You need [uv](https://docs.astral.sh/uv/) and an Anthropic or OpenAI API key. uv installs Python 3.12 if needed.

```bash
cp .env.example .env      # add your API key
make install
make warm-cache           # first run takes several minutes, and stops and resumes on rate limits
make dev
```

Open http://localhost:8501. The API runs on http://127.0.0.1:8000, with interactive docs at `/docs`.

To use OpenAI, set `LLM_PROVIDER=openai`, `OPENAI_API_KEY` and `OPENAI_MODEL`.

## Questions to try

- Which hubs in the Midwest are most exposed to winter disruption?
- Compare Miami and Houston in terms of hurricane and flood exposure.
- What percentage of days in Denver last year had snowfall?
- Why is the Dallas hub's weather disruption risk high?
- How have snow days in Denver changed from 2006 to 2025?
- What about ice storms only? (as a follow-up)

## Test and evaluate

```bash
make test                  # no network or API key needed
make evals-free            # scripted evals: no API key, no quota, lists open defects
make evals                 # calls the live LLM and costs money
uv run python -m evals.run --provider openai
uv run python -m evals.run --case denver_snow compare_miami_houston
uv run python -m evals.run --category follow_up --repeat 3
uv run python -m evals.run --dry-run                 # validate cases and estimate requests
uv run python -m evals.run --max-requests 50         # refuse to start if the estimate is higher
uv run python -m evals.run --regrade evals/results/<run>.json   # apply current rules, no LLM
uv run python -m evals.run --matrix evals/results/*.json        # pass counts per model
```

The evals use deterministic checks only, with no LLM grading another LLM. Expected rankings come from the scoring engine, and weather figures are recalculated separately in SQL. Results are saved to `evals/results/`.

Every turn also goes through a fixed checklist in `evals/audit.py`, whatever the case set out to test. It covers calculated gaps, ratio words, internal field names, reformatted blend weights, numbers from no tool, repeated calls, missing assumptions and token use. `--dry-run` rejects any case that uses a check name the runner doesn't know.

Each live case ends as `passed`, `failed`, `error` or `not_run`. Only passed and failed count towards the pass rate. A quota error stops the run and marks the remaining cases `not_run`, so quota problems don't look like quality failures.

Open defects are written as `xfail(strict=True)` tests in `tests/test_evals_grounding.py` and `tests/test_evals_requirements.py`. They show as `xfailed`, and fixing one turns the test into a failure until its marker is removed. Run `uv run pytest --runxfail` to see them fail.

## Alerts

```bash
uv run python -m app.jobs.refresh                    # score now and raise any alerts
uv run python -m app.jobs.refresh --as-of 2024-01-01 # score an earlier date
```

An alert fires on a tier change or an EDD move of at least 5%. Alerts appear at `GET /alerts` and go to `ALERT_WEBHOOK_URL` if set. Set `ENABLE_SCHEDULER=true` to run the refresh daily inside the API process.

Scores move slowly by design. To see a real change, backfill an earlier date and then score the current date:

```bash
uv run python -m app.jobs.refresh --as-of 2023-01-01
uv run python -m app.jobs.refresh
curl http://127.0.0.1:8000/alerts
```

## Reviewing chats and misuse

Each turn stores the model's reasoning summary, tool calls, skills, tokens and grounding result. To list flagged turns, set `ADMIN_TOKEN` and call:

```bash
curl -H "X-Admin-Token: $ADMIN_TOKEN" "http://127.0.0.1:8000/admin/flags?flag=destructive"
```

## Key assumptions

- **The hubs are illustrative,** assumed equal in size and placed at city centres.
- **Closure days per event, thresholds and the 70/30 blend are judgement calls.** They live in `config/scoring.yaml`. Rank ranges show how much these assumptions matter.
- **EDD counts closure days, not damage,** so rare but costly hurricanes weigh less than their financial impact would suggest.
- **Weather data is modelled reanalysis, not station readings.** FEMA data is at county level.
- **Wildfire and earthquake are flags, not scores.** Heat, drought, forecasts and dollar losses are out of scope.
- **Client IDs work like passwords.** Anyone with an ID can read its chats.
- **Limits are per hour and held in memory:** 30 messages per client, 60 per IP and 20 new IDs per IP. They reset when the API restarts and aren't shared between API processes. If `TRUST_FORWARDED_FOR=true`, keep the API behind the UI or a proxy, because a caller reaching it directly could set their own IP.
- **Long chats lose their oldest turns from the agent's memory.** Only one chat reaches the LLM, and only its newest turns within a size budget. The full chat stays visible on screen.
- **Open-Meteo's free tier is for non-commercial use,** so production would need a licence.

See [docs/DESIGN.md](docs/DESIGN.md) for the full list.
