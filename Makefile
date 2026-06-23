# Single-command reproduction. Core path needs no network and runs in seconds.
.PHONY: help results db repro fetch clean

help:
	@echo "make results  - recompute artifacts/results.json from committed clean data"
	@echo "make db       - build data/trials.db and run the sql/ analyst queries"
	@echo "make repro    - results + db (full local reproduction, no network)"
	@echo "make fetch    - OPTIONAL re-fetch from ClinicalTrials.gov + PubMed (~30 min, network)"
	@echo "make clean    - remove generated artifacts"

results:
	python src/run.py

db:
	python src/build_db.py

repro: results db

fetch:
	python src/fetch_data.py && python src/clean_data.py

clean:
	rm -f artifacts/results.json data/trials.db
