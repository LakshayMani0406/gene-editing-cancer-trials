"""
text_t2_perm.py: permutation nulls for the T2 text-only and combined models, on FIRST-POSTED
features (interpretable + PCA-50 embeddings). Strict label, temporal split, RF seed 42, 1000 shuffles.
Writes artifacts/text_t2_permutation.json (checkpointed).
"""
import warnings; warnings.filterwarnings("ignore")
import json, re, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import pipeline as P

SEED = 42; ROOT = P.ROOT
CKPT = ROOT / "artifacts" / "text_t2_permutation.json"
FP = json.loads((ROOT / "raw" / "text_fields_firstposted.json").read_text())
valid = {n: r for n, r in FP.items() if not r.get("error") and r.get("eligibility", "").strip()}
BIO = ["mutation","mutant","biomarker","expression","positive","negative","her2","egfr","alk","ros1","braf","kras",
       "pd-l1","pd-1","brca","msi","tmb","cd19","bcma","genotype","allele","amplification","fusion","wild-type","overexpression"]
NUMRE = re.compile(r'(>=|<=|>|<|≥|≤|at least|no more than|greater than|less than|minimum|maximum)\s*\d+'
                   r'|\b\d+\s*(years?|months?|weeks?|days?|mg|ml|kg|%|cells|x\s*10)\b', re.I)
def syl(w):
    w = w.lower(); c = 0; p = False
    for ch in w:
        v = ch in "aeiouy"
        if v and not p: c += 1
        p = v
    return max(c - 1 if w.endswith("e") and c > 1 else c, 1)
def fk(t):
    s = [x for x in re.split(r'[.!?]+', t) if x.strip()]; w = re.findall(r"[A-Za-z]+", t)
    return 0.39 * (len(w)/len(s)) + 11.8 * (sum(syl(x) for x in w)/len(w)) - 15.59 if s and w else np.nan
def feats(r):
    e, bs = r.get("eligibility", ""), r.get("brief_summary", ""); el = e.lower()
    li, lx = el.find("inclusion"), el.find("exclusion"); b = lambda t: sum(1 for ln in t.splitlines() if ln.strip())
    inc, ex = (b(e[li:lx]), b(e[lx:])) if lx > li >= 0 else (b(e), 0)
    return {"elig_word_count": len(e.split()), "summary_word_count": len(bs.split()), "desc_word_count": len(r.get("detailed_description","").split()),
            "n_inclusion": inc, "n_exclusion": ex, "fk_grade_elig": fk(e), "fk_grade_summary": fk(bs),
            "n_numeric_constraints": len(NUMRE.findall(e)), "n_biomarker_terms": sum(el.count(t) for t in BIO)}

ncts = json.loads((ROOT / "artifacts" / "text_embeddings_order.json").read_text())
emb = np.load(ROOT / "artifacts" / "text_embeddings_firstposted.npy")
emb_by = {n: emb[i] for i, n in enumerate(ncts)}
INTERP = list(feats(valid[ncts[0]]).keys())
interp_df = pd.DataFrame({n: feats(valid[n]) for n in ncts}).T.reset_index().rename(columns={"index": "nct_id"})
interp_df[INTERP] = interp_df[INTERP].astype(float)

d, y = P.load("strict")
d = d.merge(interp_df[["nct_id"] + INTERP], on="nct_id", how="left")
cut = int(np.quantile(d["start_year"], 0.80))
tr, te = (d["start_year"] < cut).values, (d["start_year"] >= cut).values
EMB = np.vstack([emb_by.get(n, np.zeros(384)) for n in d["nct_id"]])
EMB50 = PCA(n_components=50, random_state=SEED).fit(EMB[tr]).transform(EMB)
EMB_COLS = [f"emb_{i}" for i in range(50)]
d[EMB_COLS] = EMB50

def rf(): return RandomForestClassifier(n_estimators=400, max_depth=8, min_samples_leaf=5,
                                        max_features=0.4, random_state=SEED, n_jobs=-1)
def perm_null(num, ohe, ordc, label):
    cols = [c for c in num + ohe + ordc if c in d.columns]
    prep = P._prep([c for c in num if c in d.columns], [c for c in ohe if c in d.columns], [c for c in ordc if c in d.columns])
    prep.fit(d[cols][tr]); Xtr, Xte = prep.transform(d[cols][tr]), prep.transform(d[cols][te])
    ytr, yte = y[tr], y[te]
    obs = float(roc_auc_score(yte, rf().fit(Xtr, ytr).predict_proba(Xte)[:, 1]))
    rng = np.random.default_rng(SEED); null = []; t0 = time.time()
    for i in range(1000):
        null.append(float(roc_auc_score(yte, rf().fit(Xtr, ytr[rng.permutation(len(ytr))]).predict_proba(Xte)[:, 1])))
        if (i + 1) % 250 == 0: print(f"  [{label}] {i+1}/1000 null_mean={np.mean(null):.4f} elapsed={time.time()-t0:.0f}s", flush=True)
    a = np.array(null); n_ge = int((a >= obs).sum())
    return {"observed_auc": round(obs, 4), "null_mean": round(float(a.mean()), 4),
            "null_ci95": [round(float(np.percentile(a, 2.5)), 4), round(float(np.percentile(a, 97.5)), 4)],
            "n_null_ge_observed": n_ge, "empirical_p_one_sided": round((n_ge + 1) / 1001, 5),
            "observed_inside_null_95": bool(np.percentile(a, 2.5) <= obs <= np.percentile(a, 97.5))}

out = {"phase": "T2 permutation nulls on first-posted text", "n_permutations": 1000,
       "n_train": int(tr.sum()), "n_test": int(te.sum())}
out["text_only_interp_plus_emb"] = perm_null(INTERP + EMB_COLS, [], [], "text_only")
CKPT.write_text(json.dumps(out, indent=2))
out["combined_structured_plus_text"] = perm_null(P.NUM_REG + INTERP + EMB_COLS, P.CAT_OHE, P.CAT_ORD, "combined")
CKPT.write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
