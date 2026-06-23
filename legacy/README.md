# legacy/ — original exploratory scripts (superseded)

These are the first-pass scripts the project grew out of. They still run, but they
are **not** the maintained pipeline and are kept for reference only.

- The canonical results now come from `src/run.py` → `artifacts/results.json`.
- `ml_models.py` and `leakage_audit.py` here both used to write `outputs/leakage_audit.json`
  with **different** schemas (a real collision). That is why the live pipeline has a single
  writer. These two are kept only to show the original 11-model exploration and SHAP work.
- `dashboard.py`, `molecules.py`, `molecular_simulation.py`, `trial_embeddings.py`,
  `advanced_simulation.py`, `image_compare.py`, `patch_theme.py` produced the old
  `outputs/` HTML and figures. `image_compare.py` needs a runtime API key and is not
  deployable statically; it is parked, not part of the portfolio piece.

Outputs from these scripts land in `outputs/` (legacy), not `artifacts/` (canonical).
