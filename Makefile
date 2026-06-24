# Single-command reproduction. Core path needs no network and runs in seconds.
.PHONY: help results db web repro fetch clean serve

help:
	@echo "make results  - recompute artifacts/results.json from committed clean data"
	@echo "make db       - build data/trials.db and run the sql/ analyst queries"
	@echo "make web      - embed results.json into the web/index.html explainer"
	@echo "make repro    - results + db + web (full local reproduction, no network)"
	@echo "make serve    - serve the repo at http://localhost:8000 (open web/index.html)"
	@echo "make fetch    - OPTIONAL re-fetch from ClinicalTrials.gov + PubMed (~30 min, network)"
	@echo "make clean    - remove generated artifacts"

results:
	python src/run.py

db:
	python src/build_db.py

web:
	python src/build_web.py

serve:
	python -m http.server 8000

repro: results db web

fetch:
	python src/fetch_data.py && python src/clean_data.py

clean:
	rm -f artifacts/results.json data/trials.db
