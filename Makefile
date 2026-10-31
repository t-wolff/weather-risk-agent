UV ?= uv

.PHONY: install warm-cache api ui dev test evals evals-free

install:
	$(UV) sync

warm-cache:
	$(UV) run python -m app.jobs.ingest

api:
	$(UV) run uvicorn app.api.main:app --port 8000

ui:
	$(UV) run streamlit run ui/streamlit_app.py

dev:
	$(MAKE) -j2 api ui

test:
	$(UV) run pytest -q

evals:
	$(UV) run python -m evals.run

evals-free:
	$(UV) run pytest -q -rx tests/test_evals.py tests/test_evals_grounding.py tests/test_evals_requirements.py
	$(UV) run python -m evals.run --dry-run
