"""
text_firstposted_eval.py: definitive registration-time leakage test (Study 2).

Recomputes the interpretable text features on the FIRST-POSTED (registration-time) text,
measures how much the text drifted from registration to now, and re-runs structured /
text-only / combined under the strict label + temporal split. If the combined gain survives
on first-posted text, the signal is genuine registration-time; if it collapses to the
structured baseline, the current-text gain was post-registration-edit leakage.

Run: python src/text_firstposted_eval.py   (after src/fetch_text_firstposted.py)
"""
import warnings; warnings.filterwarnings("ignore")
import json, re
from pathlib import Path
import numpy as np, pandas as pd
import pipeline as P

ROOT = P.ROOT
FP = json.loads((ROOT / "raw" / "text_fields_firstposted.json").read_text())
CUR = json.loads((ROOT / "raw" / "text_fields.json").read_text())

# ── identical feature definitions to src/text_stage1.py ──────────────────────
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
    e, bs, dd = rec.get("eligibility", ""), rec.get("brief_summary", ""), rec.get("detailed_description", "")
    el = e.lower()
    li, lx = el.find("inclusion"), el.find("exclusion")
    bullets = lambda t: sum(1 for ln in t.splitlines() if ln.strip())
    inc, ex = (bullets(e[li:lx]), bullets(e[lx:])) if lx > li >= 0 else (bullets(e), 0)
    return {"elig_word_count": len(e.split()), "summary_word_count": len(bs.split()),
            "desc_word_count": len(dd.split()), "n_inclusion": inc, "n_exclusion": ex,
            "fk_grade_elig": fk_grade(e), "fk_grade_summary": fk_grade(bs),
            "n_numeric_constraints": len(NUMRE.findall(e)), "n_biomarker_terms": sum(el.count(t) for t in BIO)}

valid = {n: r for n, r in FP.items() if not r.get("error") and r.get("eligibility", "").strip()}
n_err = sum(1 for r in FP.values() if r.get("error"))
fp_feat = pd.DataFrame({n: features(r) for n, r in valid.items()}).T.reset_index().rename(columns={"index": "nct_id"})
TEXT_NUM = [c for c in fp_feat.columns if c != "nct_id"]
fp_feat[TEXT_NUM] = fp_feat[TEXT_NUM].astype(float)

# ── Drift: how different is current text from first-posted? ───────────────────
both = [n for n in valid if n in CUR and CUR[n].get("eligibility", "").strip()]
def wc(s): return len(s.split())
elig_changed = sum(1 for n in both if valid[n]["eligibility"].strip() != CUR[n]["eligibility"].strip())
sum_changed = sum(1 for n in both if valid[n]["brief_summary"].strip() != CUR[n]["brief_summary"].strip())
elig_wc_absdiff = [abs(wc(CUR[n]["eligibility"]) - wc(valid[n]["eligibility"])) for n in both]
elig_wc_reldiff = [abs(wc(CUR[n]["eligibility"]) - wc(valid[n]["eligibility"])) / max(wc(valid[n]["eligibility"]), 1) for n in both]
drift = {
    "n_compared": len(both),
    "pct_eligibility_text_changed": round(100 * elig_changed / len(both), 1),
    "pct_summary_text_changed": round(100 * sum_changed / len(both), 1),
    "median_elig_wordcount_abs_change": int(np.median(elig_wc_absdiff)),
    "median_elig_wordcount_rel_change_pct": round(100 * float(np.median(elig_wc_reldiff)), 1),
    "pct_elig_wordcount_changed_gt10pct": round(100 * np.mean([r > 0.10 for r in elig_wc_reldiff]), 1),
}

# ── Re-evaluate on first-posted features (same cohort, split, model) ──────────
d, y = P.load("strict")
d = d.merge(fp_feat[["nct_id"] + TEXT_NUM], on="nct_id", how="left")
cut = int(np.quantile(d["start_year"], 0.80))
def auc(num, ohe=P.CAT_OHE, ordc=P.CAT_ORD):
    return round(P.fit_eval(d, y, num, ohe, ordc, P.rf(), "temporal", cut)["auc"], 4)
struct = auc(P.NUM_REG)
txt = auc(TEXT_NUM, ohe=[], ordc=[])
combo = auc(P.NUM_REG + TEXT_NUM)

out = {
    "phase": "T1 definitive registration-time test (first-posted text)",
    "first_posted_coverage": {"n_valid": len(valid), "n_errors": n_err, "of_cohort": 1804},
    "text_drift_current_vs_firstposted": drift,
    "auc_on_firstposted_features": {"structured_only": struct, "text_only_interpretable": txt,
                                    "combined_interpretable": combo,
                                    "delta_combined_minus_structured": round(combo - struct, 4)},
    "reference_current_text": {"structured_only": 0.6047, "combined_interpretable": 0.6562,
                               "delta_combined_minus_structured": 0.0515},
    "crosses_practical_0.65": {"combined_firstposted": bool(combo >= 0.65)},
}
(ROOT / "artifacts" / "text_firstposted_eval.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
print(f"\nINTERPRETATION GUIDE: if combined_firstposted delta ~ +0.05 and crosses 0.65 -> gain is "
      f"registration-time (not leakage). If it collapses toward structured ({struct}) -> current-text gain was edit leakage.")
