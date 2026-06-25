"""
text_stage1.py: Study 2, Phase T1 Stage 1 (interpretable text features only, no embeddings).

Builds the section 5.1a interpretable features from raw/text_fields.json, reports text-fidelity
gap, then evaluates structured-only vs text-only-interpretable vs combined-interpretable under the
strict label and temporal split, with a permutation null for the text-only model. Raw numbers only.

Run: python src/text_stage1.py   (after src/fetch_text.py)
"""
import warnings; warnings.filterwarnings("ignore")
import json, re, datetime
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

import pipeline as P

SEED = 42; np.random.seed(SEED)
ROOT = P.ROOT
TXT = json.loads((ROOT / "raw" / "text_fields.json").read_text())

# ── Text-fidelity gap (first-posted vs last-updated) ──────────────────────────
def pdate(s):
    if not s: return None
    for f in ("%Y-%m-%d", "%Y-%m", "%Y"):
        try: return datetime.datetime.strptime(s, f)
        except: pass
    return None
gaps = [( (b - a).days ) for v in TXT.values()
        for a in [pdate(v["first_posted"])] for b in [pdate(v["last_updated"])] if a and b]
gaps = np.array(gaps)

# ── Interpretable text features (prereg 5.1a) ─────────────────────────────────
BIO = ["mutation","mutant","biomarker","expression","positive","negative","her2","egfr","alk",
       "ros1","braf","kras","pd-l1","pd-1","brca","msi","tmb","cd19","bcma","genotype","allele",
       "amplification","fusion","wild-type","overexpression"]
NUMRE = re.compile(r'(>=|<=|>|<|≥|≤|at least|no more than|greater than|less than|minimum|maximum)\s*\d+'
                   r'|\b\d+\s*(years?|months?|weeks?|days?|mg|ml|kg|%|cells|x\s*10)\b', re.I)
VOWELS = "aeiouy"
def syllables(w):
    w = w.lower(); c = 0; prev = False
    for ch in w:
        v = ch in VOWELS
        if v and not prev: c += 1
        prev = v
    if w.endswith("e") and c > 1: c -= 1
    return max(c, 1)
def fk_grade(t):
    sents = [s for s in re.split(r'[.!?]+', t) if s.strip()]
    words = re.findall(r"[A-Za-z]+", t)
    if not sents or not words: return np.nan
    syl = sum(syllables(w) for w in words)
    return 0.39 * (len(words) / len(sents)) + 11.8 * (syl / len(words)) - 15.59
def features(rec):
    e, bs, dd = rec["eligibility"], rec["brief_summary"], rec["detailed_description"]
    el = e.lower()
    li, lx = el.find("inclusion"), el.find("exclusion")
    bullets = lambda t: sum(1 for ln in t.splitlines() if ln.strip())
    if lx > li >= 0:
        inc, ex = bullets(e[li:lx]), bullets(e[lx:])
    else:
        inc, ex = bullets(e), 0
    return {"elig_word_count": len(e.split()), "summary_word_count": len(bs.split()),
            "desc_word_count": len(dd.split()), "n_inclusion": inc, "n_exclusion": ex,
            "fk_grade_elig": fk_grade(e), "fk_grade_summary": fk_grade(bs),
            "n_numeric_constraints": len(NUMRE.findall(e)),
            "n_biomarker_terms": sum(el.count(t) for t in BIO)}

feat_df = pd.DataFrame({n: features(r) for n, r in TXT.items()}).T.reset_index().rename(columns={"index": "nct_id"})
TEXT_NUM = [c for c in feat_df.columns if c != "nct_id"]
feat_df[TEXT_NUM] = feat_df[TEXT_NUM].astype(float)

# ── Cohort: strict-resolved, joined to text (all have text) ───────────────────
d, y = P.load("strict")
d = d.merge(feat_df, on="nct_id", how="left")
cut = int(np.quantile(d["start_year"], 0.80))
tr, te = (d["start_year"] < cut).values, (d["start_year"] >= cut).values
cov_dd = sum(1 for v in TXT.values() if v["detailed_description"].strip()) / len(TXT)

(ROOT / "artifacts").mkdir(exist_ok=True)
d[["nct_id", "start_year"] + TEXT_NUM].assign(label=y).to_csv(ROOT / "artifacts" / "text_features.csv", index=False)

# ── Three models, identical cohort + split, RandomForest seed 42 ──────────────
struct  = P.fit_eval(d, y, P.NUM_REG, P.CAT_OHE, P.CAT_ORD, P.rf(), "temporal", cut)
txt     = P.fit_eval(d, y, TEXT_NUM,  [], [],               P.rf(), "temporal", cut, want_imp=True)
combo   = P.fit_eval(d, y, P.NUM_REG + TEXT_NUM, P.CAT_OHE, P.CAT_ORD, P.rf(), "temporal", cut)

# ── Permutation null for the text-only model ──────────────────────────────────
prep = P._prep(TEXT_NUM, [], []); prep.fit(d[TEXT_NUM][tr])
Xtr, Xte = prep.transform(d[TEXT_NUM][tr]), prep.transform(d[TEXT_NUM][te])
ytr, yte = y[tr], y[te]
def rf(): return RandomForestClassifier(n_estimators=400, max_depth=8, min_samples_leaf=5,
                                        max_features=0.4, random_state=SEED, n_jobs=-1)
obs = float(roc_auc_score(yte, rf().fit(Xtr, ytr).predict_proba(Xte)[:, 1]))
rng = np.random.default_rng(SEED); null = []
for i in range(1000):
    ys = ytr[rng.permutation(len(ytr))]
    null.append(float(roc_auc_score(yte, rf().fit(Xtr, ys).predict_proba(Xte)[:, 1])))
    if (i + 1) % 200 == 0: print(f"  perm {i+1}/1000  null_mean={np.mean(null):.4f}")
a = np.array(null); n_ge = int((a >= obs).sum()); pval = (n_ge + 1) / 1001
lo, hi = float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))

out = {
    "phase": "T1-stage1", "note": "interpretable text features only, no embeddings, raw output",
    "cohort": {"n": int(len(d)), "n_train": int(tr.sum()), "n_test": int(te.sum()),
               "cut_year": cut, "test_base_rate": round(float(yte.mean()), 4)},
    "text_coverage": {"eligibility": 1.0, "brief_summary": 1.0, "detailed_description": round(cov_dd, 4)},
    "fidelity_gap_days": {"n": int(len(gaps)), "median": int(np.median(gaps)), "p75": int(np.percentile(gaps, 75)),
                          "pct_gt_365": round(100 * (gaps > 365).mean(), 1), "pct_gt_1825": round(100 * (gaps > 1825).mean(), 1),
                          "note": "days between first-posted and last-updated; large gap means current text may differ from registration-time"},
    "auc": {"structured_only": struct["auc"], "text_only_interpretable": txt["auc"], "combined_interpretable": combo["auc"]},
    "delta_combined_minus_structured": round(combo["auc"] - struct["auc"], 4),
    "permutation_text_only": {"observed": round(obs, 4), "null_mean": round(float(a.mean()), 4),
                              "null_ci95": [round(lo, 4), round(hi, 4)], "n_null_ge_observed": n_ge,
                              "empirical_p_one_sided": round(float(pval), 5), "observed_inside_null_95": bool(lo <= obs <= hi)},
    "text_only_feature_importance": txt["importance"],
    "practical_line": 0.65,
    "crosses_practical": {"text_only": bool(txt["auc"] >= 0.65), "combined": bool(combo["auc"] >= 0.65)},
}
(ROOT / "artifacts" / "text_eval_stage1.json").write_text(json.dumps(out, indent=2))

print("\n" + "=" * 60)
print(f"cohort n={out['cohort']['n']}  train={out['cohort']['n_train']}  test={out['cohort']['n_test']}  cut={cut}")
print(f"detailed_description coverage: {cov_dd:.1%}")
print(f"fidelity gap (first-posted to last-updated): median={out['fidelity_gap_days']['median']}d  p75={out['fidelity_gap_days']['p75']}d  >365d={out['fidelity_gap_days']['pct_gt_365']}%  >5y={out['fidelity_gap_days']['pct_gt_1825']}%")
print(f"\nAUC  structured_only      = {struct['auc']:.4f}")
print(f"AUC  text_only_interp     = {txt['auc']:.4f}")
print(f"AUC  combined_interp      = {combo['auc']:.4f}   (delta vs structured = {out['delta_combined_minus_structured']:+.4f})")
print(f"\npermutation null (text-only): observed={obs:.4f}  null_mean={a.mean():.4f}  95%=[{lo:.4f},{hi:.4f}]  {n_ge}/1000>=obs  p={pval:.4f}  inside_null={out['permutation_text_only']['observed_inside_null_95']}")
print(f"\ntop text features: " + ", ".join(f"{f['feature']}={f['importance']}" for f in txt['importance'][:5]))
print("\nWrote artifacts/text_eval_stage1.json and artifacts/text_features.csv. RAW, no conclusions.")
