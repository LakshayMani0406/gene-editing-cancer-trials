"""
text_leakage_check.py: is the combined +0.05 AUC gain registration-time signal or
post-registration text-edit leakage? (Study 2, pre-embedding gate.)

Update gap = last_updated - first_posted is a post-registration quantity. Tests:
  - gap-only AUC and structured+gap AUC: is the "text" gain just a proxy for update activity?
  - does the gap itself correlate with the outcome (a leakage channel)?
  - does the combined-minus-structured gain survive in low-gap (lower-leakage) trials?
  - correlation of the top text features with the gap.
Raw output only. Writes artifacts/text_leakage_check.json.
"""
import warnings; warnings.filterwarnings("ignore")
import json, datetime
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import pointbiserialr
import pipeline as P

ROOT = P.ROOT
TXT = json.loads((ROOT / "raw" / "text_fields.json").read_text())
feat = pd.read_csv(ROOT / "artifacts" / "text_features.csv")
TEXT_NUM = [c for c in feat.columns if c not in ("nct_id", "start_year", "label")]

def pdate(s):
    if not s: return None
    for f in ("%Y-%m-%d", "%Y-%m", "%Y"):
        try: return datetime.datetime.strptime(s, f)
        except: pass
    return None
gap = {n: ((pdate(v["last_updated"]) - pdate(v["first_posted"])).days
           if pdate(v["last_updated"]) and pdate(v["first_posted"]) else np.nan)
       for n, v in TXT.items()}
gap_df = pd.DataFrame({"nct_id": list(gap), "gap_days": list(gap.values())})

d, y = P.load("strict")
d = d.merge(feat[["nct_id"] + TEXT_NUM], on="nct_id", how="left").merge(gap_df, on="nct_id", how="left")
cut = int(np.quantile(d["start_year"], 0.80))

def auc(num, ohe=P.CAT_OHE, ordc=P.CAT_ORD, dd=d, yy=y, c=cut):
    try:
        return round(P.fit_eval(dd, yy, num, ohe, ordc, P.rf(), "temporal", c)["auc"], 4)
    except Exception as e:
        return f"NA ({type(e).__name__})"

res = {"phase": "T1 leakage check", "note": "is the combined gain registration-time or update-edit leakage"}

# 1) gap as a post-outcome channel
res["gap_correlates_with_outcome"] = {
    "pointbiserial_r": round(float(pointbiserialr(d["gap_days"].fillna(d["gap_days"].median()), y)[0]), 4),
    "median_gap_completed": int(np.nanmedian(d.loc[y == 1, "gap_days"])),
    "median_gap_failed": int(np.nanmedian(d.loc[y == 0, "gap_days"])),
}
# 2) does update activity reproduce the text gain?
res["auc"] = {
    "structured_only": auc(P.NUM_REG),
    "combined_structured_plus_text": auc(P.NUM_REG + TEXT_NUM),
    "gap_only": auc(["gap_days"], ohe=[], ordc=[]),
    "structured_plus_gap": auc(P.NUM_REG + ["gap_days"]),
    "structured_plus_text_plus_gap": auc(P.NUM_REG + TEXT_NUM + ["gap_days"]),
}
# 3) does the gain survive in low-gap (lower-leakage) trials?
med = float(np.nanmedian(d["gap_days"]))
res["median_gap_days"] = int(med)
for name, mask in [("low_gap_below_median", d["gap_days"] < med), ("high_gap_at_or_above_median", d["gap_days"] >= med)]:
    sub = d[mask].copy(); ysub = y[mask.values]
    csub = int(np.quantile(sub["start_year"], 0.80))
    s = auc(P.NUM_REG, dd=sub, yy=ysub, c=csub)
    cmb = auc(P.NUM_REG + TEXT_NUM, dd=sub, yy=ysub, c=csub)
    delta = (round(cmb - s, 4) if isinstance(s, float) and isinstance(cmb, float) else "NA")
    res.setdefault("gap_subsets", {})[name] = {"n": int(mask.sum()), "structured": s, "combined": cmb, "combined_minus_structured": delta}
# 4) which text features are most gap-correlated?
res["text_feature_gap_correlation"] = {
    c: round(float(d[[c, "gap_days"]].dropna().corr().iloc[0, 1]), 3) for c in TEXT_NUM
}

(ROOT / "artifacts" / "text_leakage_check.json").write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))
