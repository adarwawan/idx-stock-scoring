VENV   := .venv
PY     := $(VENV)/bin/python
PIP    := $(VENV)/bin/pip
STAMP  := $(VENV)/.installed

.DEFAULT_GOAL := help
.PHONY: help setup fetch refresh coverage notebook clean clean-cache

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  make %-12s %s\n", $$1, $$2}'

setup: $(STAMP) ## Create .venv and install dependencies

$(STAMP): requirements.txt
	test -d $(VENV) || python3 -m venv $(VENV)
	$(PIP) install -q -r requirements.txt
	touch $(STAMP)

fetch: $(STAMP) ## Fetch data, reusing caches that are still fresh
	$(PY) -W ignore fetch_data.py $(ARGS)

refresh: $(STAMP) ## Fetch data, ignoring all caches (~90s)
	$(PY) -W ignore fetch_data.py --refresh $(ARGS)

coverage: ## Show per-field data coverage from the last fetch
	@test -f data/coverage_latest.csv || { echo "No data yet. Run: make fetch"; exit 1; }
	@column -s, -t data/coverage_latest.csv

notebook: $(STAMP) ## Fetch fresh data, then open the scoring notebook
	$(MAKE) fetch
	$(VENV)/bin/jupyter lab idx_stock_scoring.ipynb

clean-cache: ## Delete cached Pasardana/Yahoo responses
	rm -rf data/cache

clean: clean-cache ## Delete cache and all generated data files
	rm -f data/idx_metrics_*.csv data/latest.csv data/coverage_latest.csv data/failures.csv
