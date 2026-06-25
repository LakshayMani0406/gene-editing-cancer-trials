"""
text_embed.py: Study 2 embeddings on FIRST-POSTED text (registration-time, no leakage).

Embeds eligibility + brief_summary with all-MiniLM-L6-v2, PCA-50 (fit on train only),
then point AUCs decomposing structured / interpretable / embeddings / text-only / combined.
Permutation nulls are run separately (background). Run after src/fetch_text_firstposted.py.
"""
import warnings; warnings.filterwarnings("ignore")
import json, re, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.decomposition import PCA
import pipeline as P

ROOT = P.ROOT
FP = json.loads((ROOT / "raw" / "text_fields_firstposted.json").read_text())
valid = {n: r for n, r in FP.items() if not r.get("error") and r.get("eligibility", "").strip()}

# ── interpretable features (same definitions as src/text_stage1.py) ──────────
BIO = ["mutation","mutant","biomarker","expression","positive","negative","her2","egfr","alk","ros1",
       "braf","kras","pd-l1","pd-1","brca","msi","tmb","cd19","bcma","genotype","allele","amplification",
       "fusion","wild-type","overexpression"]
NUMRE = re.compile(r'(>=|<=|>|<|≥|≤|at least|no more than|greater than|less than|minimum|maximum)\s*\d+'
                   r'|\b\d+\s*(years?|months?|weeks?|days?|mg|ml|kg|%|cells|x\s*10)\b', re.I)
VOWELS = "aeiouy"
def syl(w):
    w = w.lower(); c = 0; p = False
    for ch in w:
        v = ch in VOWELS
        if v and not p: c += 1
        p = v
    return max(c - 1 if w.endswith("e") and c > 1 else c, 1)
def fk(t):
    s = [x for x in re.split(r'[.!?]+', t) if x.strip()]; w = re.findall(r"[A-Za-z]+", t)
    return 0.39 * (len(w) / len(s)) + 11.8 * (sum(syl(x) for x in w) / len(w)) - 15.59 if s and w else np.nan
def feats(r):
    e, bs, dd = r.get("eligibility", ""), r.get("brief_summary", ""), r.get("detailed_description", "")
    el = e.lower(); li, lx = el.find("inclusion"), el.find("exclusion")
    b = lambda t: sum(1 for ln in t.splitlines() if ln.strip())
    inc, ex = (b(e[li:lx]), b(e[lx:])) if lx > li >= 0 else (b(e), 0)
    return {"elig_word_count": len(e.split()), "summary_word_count": len(bs.split()), "desc_word_count": len(dd.split()),
            "n_inclusion": inc, "n_exclusion": ex, "fk_grade_elig": fk(e), "fk_grade_summary": fk(bs),
            "n_numeric_constraints": len(NUMRE.findall(e)), "n_biomarker_terms": sum(el.count(t) for t in BIO)}

ncts = list(valid.keys())
texts = [valid[n]["eligibility"] + " " + valid[n]["brief_summary"] for n in ncts]

from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
t0 = time.time(); smoke = model.encode(texts[:50], show_progress_bar=False)
print(f"smoke-test: embedded 50 -> shape {smoke.shape} in {time.time()-t0:.1f}s")
assert smoke.shape == (50, 384)
t0 = time.time(); emb = model.encode(texts, batch_size=64, show_progress_bar=False)
print(f"embedded {len(texts)} first-posted texts -> {emb.shape} in {time.time()-t0:.1f}s")
np.save(ROOT / "artifacts" / "text_embeddings_firstposted.npy", emb)
(ROOT / "artifacts" / "text_embeddings_order.json").write_text(json.dumps(ncts))

INTERP = list(feats(valid[ncts[0]]).keys())
interp_df = pd.DataFrame({n: feats(valid[n]) for n in ncts}).T.reset_index().rename(columns={"index": "nct_id"})
interp_df[INTERP] = interp_df[INTERP].astype(float)
emb_by_nct = {n: emb[i] for i, n in enumerate(ncts)}

d, y = P.load("strict")
d = d.merge(interp_df[["nct_id"] + INTERP], on="nct_id", how="left")
cut = int(np.quantile(d["start_year"], 0.80))
tr = (d["start_year"] < cut).values
EMB = np.vstack([emb_by_nct.get(n, np.zeros(384)) for n in d["nct_id"]])
pca = PCA(n_components=50, random_state=42).fit(EMB[tr])
EMB50 = pca.transform(EMB)
EMB_COLS = [f"emb_{i}" for i in range(50)]
d[EMB_COLS] = EMB50

def auc(num, ohe=P.CAT_OHE, ordc=P.CAT_ORD):
    return round(P.fit_eval(d, y, num, ohe, ordc, P.rf(), "temporal", cut)["auc"], 4)

res = {
    "phase": "T2 eval on first-posted text (embeddings + interpretable)",
    "embedding_model": "all-MiniLM-L6-v2", "embedding_dim": 384, "pca_components": 50,
    "pca_explained_variance": round(float(pca.explained_variance_ratio_.sum()), 3),
    "cohort_n": int(len(d)), "n_train": int(tr.sum()), "n_test": int((~tr).sum()), "cut_year": cut,
    "auc": {
        "structured_only": auc(P.NUM_REG),
        "interpretable_only": auc(INTERP, ohe=[], ordc=[]),
        "embeddings_only": auc(EMB_COLS, ohe=[], ordc=[]),
        "text_only_interp_plus_emb": auc(INTERP + EMB_COLS, ohe=[], ordc=[]),
        "combined_structured_plus_text": auc(P.NUM_REG + INTERP + EMB_COLS),
    },
    "practical_line": 0.65,
}
res["delta_combined_minus_structured"] = round(res["auc"]["combined_structured_plus_text"] - res["auc"]["structured_only"], 4)
res["crosses_practical"] = {k: bool(v >= 0.65) for k, v in res["auc"].items()}
(ROOT / "artifacts" / "text_t2_eval.json").write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))
