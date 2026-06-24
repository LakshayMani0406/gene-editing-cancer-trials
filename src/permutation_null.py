"""
permutation_null.py: Phase 2 of the pre-registered study (docs/PREREGISTRATION.md, section 4.5).

Label-permutation null for the honest classifier (strict label, registration-only features,
temporal split, RandomForest seed 42). Shuffle the TRAIN labels 1000 times, refit, score the
real held-out test set, build the null AUC distribution, and compare the observed 0.6047.

Raw output only, no conclusions. Writes artifacts/permutation_null.json (checkpointed every 100).
Run: python src/permutation_null.py
"""
import warnings; warnings.filterwarnings("ignore")
import json, time
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

import pipeline as P

SEED = 42
N_PERM = 1000
ROOT = P.ROOT
CKPT = ROOT / "artifacts" / "permutation_null.json"
rng = np.random.default_rng(SEED)

# ── Honest config: strict label, registration-only, temporal split ────────────
d, y = P.load("strict")
cut = int(np.quantile(d["start_year"], 0.80))
feat = [c for c in P.NUM_REG + P.CAT_OHE + P.CAT_ORD if c in d.columns]
X = d[feat]
tr = (d["start_year"] < cut).values
te = (d["start_year"] >= cut).values
Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]

# Preprocessing does not use y, so fit it once and reuse. Only the forest is refit per shuffle.
prep = P._prep([c for c in P.NUM_REG if c in d.columns],
               [c for c in P.CAT_OHE if c in d.columns],
               [c for c in P.CAT_ORD if c in d.columns])
prep.fit(Xtr)
Xtr_t, Xte_t = prep.transform(Xtr), prep.transform(Xte)

def new_rf():  # identical to pipeline.rf() but n_jobs=-1 for speed (random_state fixed, so reproducible)
    return RandomForestClassifier(n_estimators=400, max_depth=8, min_samples_leaf=5,
                                  max_features=0.4, class_weight=None, random_state=SEED, n_jobs=-1)

# Observed (should reproduce the honest 0.6047)
m = new_rf(); m.fit(Xtr_t, ytr)
obs = float(roc_auc_score(yte, m.predict_proba(Xte_t)[:, 1]))
print(f"n_train={tr.sum()} n_test={te.sum()} cut_year={cut}")
print(f"observed AUC = {obs:.4f}  (honest config; expected ~0.6047)\n")

# ── Permutation loop ──────────────────────────────────────────────────────────
null = []
t0 = time.time()
for i in range(N_PERM):
    ys = ytr[rng.permutation(len(ytr))]
    m = new_rf(); m.fit(Xtr_t, ys)
    null.append(float(roc_auc_score(yte, m.predict_proba(Xte_t)[:, 1])))
    if (i + 1) % 100 == 0:
        a = np.array(null)
        json.dump({"status": "running", "n_done": i + 1, "observed_auc": round(obs, 4),
                   "null_mean": round(float(a.mean()), 4),
                   "null_ci95": [round(float(np.percentile(a, 2.5)), 4), round(float(np.percentile(a, 97.5)), 4)]},
                  open(CKPT, "w"), indent=2)
        print(f"  {i+1}/{N_PERM}  null_mean={a.mean():.4f}  elapsed={time.time()-t0:.0f}s")

a = np.array(null)
n_ge = int((a >= obs).sum())
pval = (n_ge + 1) / (N_PERM + 1)
lo, hi = float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))
result = {
    "phase": 2, "note": "permutation null, raw output, no conclusions",
    "config": "strict label, registration-only features, temporal split, RandomForest seed 42",
    "n_permutations": N_PERM, "seed": SEED, "temporal_cut_year": cut,
    "n_train": int(tr.sum()), "n_test": int(te.sum()), "test_base_rate": round(float(yte.mean()), 4),
    "observed_auc": round(obs, 4),
    "null_mean": round(float(a.mean()), 4), "null_std": round(float(a.std()), 4),
    "null_min": round(float(a.min()), 4), "null_max": round(float(a.max()), 4),
    "null_ci95": [round(lo, 4), round(hi, 4)],
    "n_null_ge_observed": n_ge,
    "empirical_p_one_sided": round(float(pval), 5),
    "p_formula": "(count(null >= observed) + 1) / (n_permutations + 1)",
    "observed_inside_null_95": bool(lo <= obs <= hi),
    "elapsed_seconds": round(time.time() - t0, 1),
}
json.dump(result, open(CKPT, "w"), indent=2)
print("\n" + json.dumps(result, indent=2))
print("\nRAW OUTPUT ONLY. No conclusions, no commit.")
