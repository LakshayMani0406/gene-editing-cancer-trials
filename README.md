# Predicting clinical-trial completion: a three-layer leakage audit

**0.8963 leaked → 0.7185 naive-clean → 0.6047 strict-temporal.**

Honest, prospective prediction of whether an oncology gene-editing trial will complete is barely better than a coin flip. That collapse, not the headline 0.90, is the finding.

I trained a model on ClinicalTrials.gov registration data to predict trial completion. The first version scored AUC 0.8963. I did not trust it, so I audited my own work twice, and each pass found a way the model was reading the answer off its own inputs.

**Leak 1: conduct-time features.** The 0.90 model used the actual enrolled patient count, whether results were posted, and the number of countries. None of those are known when a trial registers. A terminated trial mechanically ends up with low enrollment and no posted results, so the model was partly reading the outcome from features the outcome itself produces. Removing them drops AUC to 0.7185.

**Leak 2: the one hiding inside my own fix.** The 0.7185 still cheated, in a subtler way. My success label counted trials that were still recruiting or active as wins. **63.9% of every trial I had labelled a "success" had never actually completed; they simply had not failed yet.** Those trials are recent, so the model could guess the label from the calendar: start year on its own scores 0.678, about 82% of the 0.72 model's signal above chance. The "clean" model was mostly a recency detector. I redefined success as completed versus terminated, withdrawn, or suspended, dropped the in-progress trials, and tested on the future (train on older trials, test on newer) instead of a random split. That gives AUC 0.6047.

I report all three numbers on purpose. The 0.90 is what the data hands you if you are not careful. The 0.72 is what survives the obvious fix and still misleads. The 0.60 is the number I can defend, and it says trial completion is close to unpredictable from registration metadata. I would rather show that and explain it than ship the 0.90 and be wrong in production.

## See it

An interactive explainer lets you peel back each leak, watch the ROC curve fall, and move a decision-threshold cost slider to see what the model would do under asymmetric mistake costs.

- Live: https://lakshaymani0406.github.io/gene-editing-cancer-trials/web/ (enable GitHub Pages on the repo, deploy from `main`)
- Local: `make serve`, then open http://localhost:8000/web/index.html

The explainer is a single static HTML file. It reads `artifacts/results.json`, needs no server, no API key, and no per-user cost.

## Reproduce

```bash
pip install -r requirements.txt
python src/run.py        # or: make results
```

This runs in seconds from the committed clean data, recomputes the full arc, and writes `artifacts/results.json`. It cross-checks its own numbers against the prior pipeline and refuses to write if they drift. `make repro` also builds the SQL database and refreshes the explainer's embedded data.

Re-collecting raw data from the public APIs is optional and slow: `make fetch` (about 30 minutes, network).

## SQL layer

`sql/` holds five queries that run against a SQLite build of the clean data (`make db`). They cover the development funnel, completion by cancer type, the recency leak (query `03` reproduces the 63.9% finding directly), right-censoring in the strict label, and the industry-sponsor trend over time.

## Methodology

- **Data.** Interventional oncology gene-editing trials from ClinicalTrials.gov, pulled through the REST API v2 with 28 keyword queries and deduplicated by NCT ID. The cohort is keyword-defined, not expert-curated.
- **Model.** One random forest, fixed seed 42, held constant across the arc so each AUC drop is attributable to a single change: first the feature set, then the label, then the split. Features are one-hot and ordinal encoded with median and mode imputation inside a scikit-learn pipeline.
- **Feature sets.** Full (includes conduct-time features) versus registration-only (known at trial registration).
- **Label.** Bundled (completed, recruiting, or active counts as success) versus strict (completed is success, terminated/withdrawn/suspended is failure, in-progress trials dropped).
- **Validation.** Stratified random split versus temporal split (train on earlier start years, test on the most recent 20%).
- **Honest configuration.** Registration-only features, strict label, temporal split, reported as AUC 0.6047.

## Repository

```
src/        run.py (entry point), pipeline.py, build_db.py, build_web.py, fetch/clean
data/       committed clean CSV (the raw API dump is gitignored)
artifacts/  results.json (single writer, versioned)
sql/        five analyst queries
web/        index.html static explainer
legacy/     earlier exploratory scripts, kept for reference
```

## Limitations

- Observational registration metadata only. Completion status is an administrative outcome, not clinical efficacy.
- The honest model is near chance. The signal that survives the audit is weak, which is the point, not a defect to hide.
- The temporal test set is small (about 410 resolved trials), so the 0.6047 carries real uncertainty.
- Even the strict label keeps a mild time effect from right-censoring: completion takes years, so recently-resolved trials skew toward termination. A time-to-event model would handle that better and is future work.

## References

National Library of Medicine. (2025). *ClinicalTrials.gov* [Data set]. U.S. National Institutes of Health. https://clinicaltrials.gov

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, E. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825-2830.

Kaufman, S., Rosset, S., Perlich, C., & Stitelman, O. (2012). Leakage in data mining: Formulation, detection, and avoidance. *ACM Transactions on Knowledge Discovery from Data, 6*(4), Article 15. https://doi.org/10.1145/2382577.2382579

## Author

Lakshay Mani. MS Analytics, Northeastern University.
