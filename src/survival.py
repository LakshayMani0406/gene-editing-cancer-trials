"""
survival.py: Phase 1 of the pre-registered study (docs/PREREGISTRATION.md).

Competing-risks survival analysis of trial completion on registration-only features.
Writes raw numbers to artifacts/survival.json and KM/CIF plots to outputs/.
Conclusions are NOT written here; this produces raw output only.

Run: python src/survival.py   (after: pip install lifelines)
"""
import warnings; warnings.filterwarnings("ignore")
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lifelines import KaplanMeierFitter, CoxPHFitter, AalenJohansenFitter
from lifelines.statistics import multivariate_logrank_test, proportional_hazard_test

SEED = 42
np.random.seed(SEED)
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "crispr_trials_clean.csv"
(ROOT / "artifacts").mkdir(exist_ok=True)
(ROOT / "outputs").mkdir(exist_ok=True)

COMPLETED = {"COMPLETED"}
COMPETING = {"TERMINATED", "WITHDRAWN", "SUSPENDED"}
ACTIVE    = {"RECRUITING", "ACTIVE_NOT_RECRUITING"}

# ── Load and build the survival table ─────────────────────────────────────────
df = pd.read_csv(DATA)
n_raw = len(df)
df["start_dt"] = pd.to_datetime(df["start_date"], errors="coerce")
df["comp_dt"]  = pd.to_datetime(df["completion_date"], errors="coerce")

# Snapshot date for censoring active trials = data collection date (file timestamps, 2026-06-01).
# Deriving it from the max resolved completion date is unsafe: the cleaned data contains erroneous
# future completion dates (e.g. 2044) that would inflate active-trial follow-up by ~18 years.
SNAPSHOT = pd.Timestamp("2026-06-01")

def build_row(r):
    s = r["overall_status"]
    if s in COMPLETED:
        return pd.Series({"T": r["duration_months"], "event": 1, "code": 1})
    if s in COMPETING:
        return pd.Series({"T": r["duration_months"], "event": 0, "code": 2})
    if s in ACTIVE:
        t = (SNAPSHOT - r["start_dt"]).days / 30.44 if pd.notna(r["start_dt"]) else np.nan
        return pd.Series({"T": t, "event": 0, "code": 0})
    return pd.Series({"T": np.nan, "event": np.nan, "code": np.nan})  # UNKNOWN etc -> excluded

sv = df.join(df.apply(build_row, axis=1))

n_unknown = int((~df["overall_status"].isin(COMPLETED | COMPETING | ACTIVE)).sum())
future_resolved = (df["overall_status"].isin(COMPLETED | COMPETING) & (df["comp_dt"] > SNAPSHOT)).values
n_future = int(future_resolved.sum())
bad = sv["T"].isna() | (sv["T"] <= 0) | (sv["T"] > 480) | sv["code"].isna() | future_resolved
n_excluded = int(bad.sum())
n_bad_dur = n_excluded - n_unknown - n_future
sv = sv[~bad].copy()
sv["event"] = sv["event"].astype(int)
sv["code"]  = sv["code"].astype(int)

n_complete = int((sv["code"] == 1).sum())
n_compete  = int((sv["code"] == 2).sum())
n_censor   = int((sv["code"] == 0).sum())
print(f"Snapshot date for active censoring: {SNAPSHOT.date()}")
print(f"Cohort: {len(sv)} trials | completions(event)={n_complete} | competing={n_compete} | active-censored={n_censor}")
print(f"Excluded {n_excluded}: {n_unknown} UNKNOWN-status, {n_future} future-dated completions (data errors), {n_bad_dur} bad/missing duration\n")

out = {
    "phase": 1, "note": "Raw survival output, no conclusions. See docs/PREREGISTRATION.md.",
    "lifelines_version": __import__("lifelines").__version__,
    "cohort": {"n_raw": n_raw, "n_analyzed": int(len(sv)), "n_completion_events": n_complete,
               "n_competing": n_compete, "n_active_censored": n_censor, "n_excluded": n_excluded,
               "n_unknown_excluded": n_unknown, "n_future_completion_excluded": n_future,
               "n_bad_duration_excluded": n_bad_dur,
               "snapshot_date": str(SNAPSHOT.date()),
               "snapshot_note": "data collection date; resolved trials with completion dates after it excluded as data errors",
               "event_definition": "completion = event; competing (term/wd/susp) censored in cause-specific Cox; active censored at snapshot"},
}

T, E = sv["T"].values, sv["event"].values

# ── Kaplan-Meier (cause-specific completion) ──────────────────────────────────
kmf = KaplanMeierFitter()
kmf.fit(T, E, label="All trials")
HORIZONS = [12, 24, 36, 48, 60]
def incidence_at(fitter, t):  # 1 - S(t) = cause-specific completion incidence
    return float(1 - fitter.predict(t))
out["km_overall"] = {
    "median_completion_time_months": (None if not np.isfinite(kmf.median_survival_time_) else float(kmf.median_survival_time_)),
    "completion_incidence_at_months": {str(h): round(incidence_at(kmf, h), 4) for h in HORIZONS},
    "note": "1 - KM survival; cause-specific, upward biased vs the Aalen-Johansen CIF below",
}
plt.figure(figsize=(7, 5)); kmf.plot_survival_function(ci_show=True)
plt.title("Kaplan-Meier: time to completion (cause-specific)"); plt.xlabel("Months"); plt.ylabel("P(not yet completed)")
plt.tight_layout(); plt.savefig(ROOT / "outputs" / "km_overall.png", dpi=140); plt.close()

# ── KM stratified + log-rank ──────────────────────────────────────────────────
def strat(colname, fname, title, min_n=30):
    g = sv.copy()
    counts = g[colname].value_counts()
    keep = counts[counts >= min_n].index
    g = g[g[colname].isin(keep)]
    lr = multivariate_logrank_test(g["T"].values, g[colname].values, g["event"].values)
    plt.figure(figsize=(7.5, 5))
    km = KaplanMeierFitter()
    per = {}
    for lvl in sorted(g[colname].unique()):
        m = g[colname] == lvl
        km.fit(g.loc[m, "T"].values, g.loc[m, "event"].values, label=f"{lvl} (n={int(m.sum())})")
        km.plot_survival_function(ci_show=False)
        med = km.median_survival_time_
        per[str(lvl)] = {"n": int(m.sum()), "events": int(g.loc[m, "event"].sum()),
                         "median_months": (None if not np.isfinite(med) else float(med)),
                         "completion_incidence_60m": round(float(1 - km.predict(60)), 4)}
    plt.title(title); plt.xlabel("Months"); plt.ylabel("P(not yet completed)")
    plt.legend(fontsize=7); plt.tight_layout()
    plt.savefig(ROOT / "outputs" / fname, dpi=140); plt.close()
    return {"groups": per, "logrank": {"chi2": round(float(lr.test_statistic), 4),
            "p": float(lr.p_value), "df": int(lr.degrees_of_freedom)}}

out["km_by_phase"]   = strat("phase_clean", "km_by_phase.png", "Completion by trial phase")
out["km_by_sponsor"] = strat("sponsor_class_clean", "km_by_sponsor.png", "Completion by sponsor class")
print(f"Log-rank by phase:   chi2={out['km_by_phase']['logrank']['chi2']}  p={out['km_by_phase']['logrank']['p']:.2e}")
print(f"Log-rank by sponsor: chi2={out['km_by_sponsor']['logrank']['chi2']}  p={out['km_by_sponsor']['logrank']['p']:.2e}\n")

# ── Cox PH on registration-only features (de-duplicated to a full-rank design) ─
# The registered set has deterministic redundancies that make a Cox model non-identifiable.
# Pre-specified de-duplication (decided before reading any HR, removes only redundant info):
#   keep start_year (drop is_recent, trial_era);  keep tumor_category (drop is_hematologic, cancer_type).
PHASE_KEEP = {"Phase I", "Phase I/II", "Phase II", "Phase III", "Phase IV"}
d = sv.copy()
d["phase_grp"]   = d["phase_clean"].where(d["phase_clean"].isin(PHASE_KEEP), "Phase other")
d["start_year_z"] = (d["start_year"] - d["start_year"].mean()) / d["start_year"].std()
d["n_primary_outcomes"] = d["n_primary_outcomes"].fillna(d["n_primary_outcomes"].median())

def dummies(series, ref, prefix):
    dd = pd.get_dummies(series, prefix=prefix)
    refcol = f"{prefix}_{ref}"
    if refcol in dd.columns:
        dd = dd.drop(columns=[refcol])
    return dd.astype(float)

X = pd.concat([
    d[["start_year_z", "n_primary_outcomes"]].astype(float),
    dummies(d["phase_grp"], "Phase I", "phase"),
    dummies(d["sponsor_class_clean"], "Academic / Non-profit", "sponsor"),
    dummies(d["tumor_category"], "Solid Tumor", "tumor"),
], axis=1)
design = pd.concat([d[["T", "event"]].reset_index(drop=True), X.reset_index(drop=True)], axis=1).dropna()

penalizer = 0.0; cox_note = "no penalizer"
try:
    cph = CoxPHFitter(penalizer=penalizer); cph.fit(design, "T", "event")
except Exception as e:
    penalizer = 0.1; cox_note = f"ridge penalizer 0.1 used after convergence issue: {type(e).__name__}"
    cph = CoxPHFitter(penalizer=penalizer); cph.fit(design, "T", "event")

s = cph.summary  # has coef, exp(coef), exp(coef) lower/upper 95%, p
def bh(pvals):
    p = np.asarray(pvals, float); n = len(p); order = np.argsort(p)
    adj = np.empty(n); prev = 1.0
    for rank, idx in enumerate(reversed(order)):
        i = n - rank
        prev = min(prev, p[idx] * n / i); adj[idx] = min(prev, 1.0)
    return adj
pvals = s["p"].values
pbh = bh(pvals)
cox = {}
for (name, row), pb in zip(s.iterrows(), pbh):
    cox[name] = {"hr": round(float(row["exp(coef)"]), 4),
                 "ci_low": round(float(row["exp(coef) lower 95%"]), 4),
                 "ci_high": round(float(row["exp(coef) upper 95%"]), 4),
                 "p": float(row["p"]), "p_bh": float(pb),
                 "ci_excludes_1": bool(row["exp(coef) lower 95%"] > 1 or row["exp(coef) upper 95%"] < 1)}
lr_test = cph.log_likelihood_ratio_test()
out["cox"] = {
    "covariates": cox,
    "c_index": round(float(cph.concordance_index_), 4),
    "log_likelihood_ratio_test": {"stat": round(float(lr_test.test_statistic), 4), "p": float(lr_test.p_value)},
    "n": int(design.shape[0]), "n_events": int(design["event"].sum()),
    "penalizer": penalizer, "cox_note": cox_note,
    "references": {"phase": "Phase I", "sponsor": "Academic / Non-profit", "tumor": "Solid Tumor"},
    "start_year_note": "HR is per 1 SD of start_year (standardized)",
    "dropped_for_collinearity": ["is_recent", "trial_era", "is_hematologic", "cancer_type"],
}
n_excl1 = sum(v["ci_excludes_1"] for v in cox.values())
print(f"Cox: n={out['cox']['n']} events={out['cox']['n_events']} C-index={out['cox']['c_index']} ({cox_note})")
print(f"  covariates with 95% CI excluding 1: {n_excl1}/{len(cox)}")
for name, v in cox.items():
    flag = "*" if v["ci_excludes_1"] else " "
    print(f"   {flag} {name:<34} HR={v['hr']:.3f}  [{v['ci_low']:.3f}, {v['ci_high']:.3f}]  p={v['p']:.3g}  p_bh={v['p_bh']:.3g}")

# ── Proportional-hazards (Schoenfeld) check ──────────────────────────────────
try:
    ph = proportional_hazard_test(cph, design, time_transform="rank")
    ph_tab = ph.summary
    per_cov = {idx: {"stat": round(float(r["test_statistic"]), 4), "p": float(r["p"])} for idx, r in ph_tab.iterrows()}
    violations = [k for k, v in per_cov.items() if v["p"] < 0.05]
    out["ph_assumption"] = {"per_covariate": per_cov, "violations_p_lt_0.05": violations,
                            "holds_global": len(violations) == 0,
                            "method": "Schoenfeld residuals, rank time-transform (Grambsch-Therneau)"}
    print(f"\nPH check (Schoenfeld): {len(violations)} covariate(s) violate p<0.05 -> {violations if violations else 'none'}")
except Exception as e:
    out["ph_assumption"] = {"error": f"{type(e).__name__}: {e}"}
    print(f"\nPH check failed: {e}")

# ── Aalen-Johansen cumulative incidence (competing risks) ────────────────────
def cif_at(fitter, t):
    cd = fitter.cumulative_density_.iloc[:, 0]
    s = cd[cd.index <= t]
    return round(float(s.iloc[-1]), 4) if len(s) else 0.0
aj = {}
try:
    plt.figure(figsize=(7.5, 5))
    for code, label, fname in [(1, "Completion", None), (2, "Termination/withdrawal/suspension", None)]:
        ajf = AalenJohansenFitter(seed=SEED, calculate_variance=False)
        ajf.fit(sv["T"].values, sv["code"].values, event_of_interest=code)
        ajf.plot(label=label)
        aj[label] = {str(h): cif_at(ajf, h) for h in HORIZONS}
    plt.title("Aalen-Johansen cumulative incidence (competing risks)")
    plt.xlabel("Months"); plt.ylabel("Cumulative incidence"); plt.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(ROOT / "outputs" / "aalen_johansen_cif.png", dpi=140); plt.close()
    out["aalen_johansen_cif_by_month"] = aj
    print(f"\nAalen-Johansen CIF @60m: completion={aj['Completion']['60']}  competing={aj['Termination/withdrawal/suspension']['60']}")
except Exception as e:
    out["aalen_johansen_cif_by_month"] = {"error": f"{type(e).__name__}: {e}"}
    print(f"\nAalen-Johansen failed: {e}")

with open(ROOT / "artifacts" / "survival.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"\nWrote artifacts/survival.json and 4 plots to outputs/. RAW OUTPUT ONLY, no conclusions.")
