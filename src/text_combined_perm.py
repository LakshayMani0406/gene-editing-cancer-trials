"""
text_combined_perm.py: permutation null for the combined (structured + interpretable text) model.
Confirms whether the +0.05 AUC gain over structured is real or test-set noise.
Strict label, temporal split, RF seed 42. Writes artifacts/text_combined_permutation.json (checkpointed).
"""
import warnings; warnings.filterwarnings("ignore")
import json, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import pipeline as P

SEED = 42; ROOT = P.ROOT
CKPT = ROOT / "artifacts" / "text_combined_permutation.json"
feat = pd.read_csv(ROOT / "artifacts" / "text_features.csv")
TEXT_NUM = [c for c in feat.columns if c not in ("nct_id", "start_year", "label")]
d, y = P.load("strict")
d = d.merge(feat[["nct_id"] + TEXT_NUM], on="nct_id", how="left")
cut = int(np.quantile(d["start_year"], 0.80))
tr, te = (d["start_year"] < cut).values, (d["start_year"] >= cut).values

NUM = P.NUM_REG + TEXT_NUM
feats = [c for c in NUM + P.CAT_OHE + P.CAT_ORD if c in d.columns]
prep = P._prep(NUM, P.CAT_OHE, P.CAT_ORD); prep.fit(d[feats][tr])
Xtr, Xte = prep.transform(d[feats][tr]), prep.transform(d[feats][te])
ytr, yte = y[tr], y[te]

def rf(): return RandomForestClassifier(n_estimators=400, max_depth=8, min_samples_leaf=5,
                                        max_features=0.4, random_state=SEED, n_jobs=-1)
obs = float(roc_auc_score(yte, rf().fit(Xtr, ytr).predict_proba(Xte)[:, 1]))
print(f"observed combined AUC = {obs:.4f}  (n_train={tr.sum()} n_test={te.sum()})")
rng = np.random.default_rng(SEED); null = []; t0 = time.time()
for i in range(1000):
    ys = ytr[rng.permutation(len(ytr))]
    null.append(float(roc_auc_score(yte, rf().fit(Xtr, ys).predict_proba(Xte)[:, 1])))
    if (i + 1) % 200 == 0:
        a = np.array(null)
        json.dump({"status": "running", "n_done": i + 1, "observed": round(obs, 4),
                   "null_mean": round(float(a.mean()), 4)}, open(CKPT, "w"), indent=2)
        print(f"  {i+1}/1000  null_mean={a.mean():.4f}  elapsed={time.time()-t0:.0f}s")
a = np.array(null); n_ge = int((a >= obs).sum()); pval = (n_ge + 1) / 1001
lo, hi = float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))
out = {"phase": "T1 combined permutation null", "model": "structured + interpretable text",
       "config": "strict label, temporal split, RandomForest seed 42", "n_permutations": 1000,
       "n_train": int(tr.sum()), "n_test": int(te.sum()),
       "observed_auc": round(obs, 4), "null_mean": round(float(a.mean()), 4),
       "null_ci95": [round(lo, 4), round(hi, 4)], "n_null_ge_observed": n_ge,
       "empirical_p_one_sided": round(float(pval), 5), "observed_inside_null_95": bool(lo <= obs <= hi),
       "elapsed_seconds": round(time.time() - t0, 1)}
json.dump(out, open(CKPT, "w"), indent=2)
print(json.dumps(out, indent=2))
