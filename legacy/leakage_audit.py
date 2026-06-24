"""
leakage_audit.py - Standalone Registration-Time Leakage Audit

Runs the same comparison as section 10 of ml_models.py, but as a clean
standalone process that does NOT import XGBoost or LightGBM. This avoids
the macOS OpenMP segfault that happens when running the audit after the
main pipeline.

Trains 3 sklearn-native models on both feature sets:
  - Logistic Regression
  - Random Forest (tuned params from main pipeline)
  - Gradient Boosting

Output: outputs/leakage_audit.json

Usage:
    python3 leakage_audit.py    (run after clean_data.py)
"""
import warnings; warnings.filterwarnings("ignore")
import json
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

np.random.seed(42)
Path("outputs").mkdir(exist_ok=True)

# ── Load data ────────────────────────────────────────────────────────────────
df_all = pd.read_csv("data/crispr_trials_clean.csv")
df = df_all[df_all["trial_outcome"].notna()].copy()
print(f"Loaded {len(df):,} labeled trials\n")

# Median-impute the numeric features
for c in ["is_hematologic","is_recent","log_enrollment",
          "n_countries","n_primary_outcomes","start_year"]:
    if c in df.columns:
        df[c] = df[c].fillna(df[c].median())

# ── Feature sets ─────────────────────────────────────────────────────────────
PHASE_ORDER = [["Early Phase I","Phase I","Phase I/II","Phase II",
                "Phase III","Phase IV","N/A","Unknown","PHASE2/PHASE3"]]

CAT_OHE = [c for c in ["cancer_type","tumor_category",
                        "sponsor_class_clean","trial_era"] if c in df.columns]
CAT_ORD = ["phase_clean"]

NUM_FULL = [c for c in ["log_enrollment","start_year","n_countries",
                         "n_primary_outcomes","is_hematologic","is_recent",
                         "results_available"] if c in df.columns]

NUM_REG = [c for c in ["start_year","n_primary_outcomes",
                        "is_hematologic","is_recent"] if c in df.columns]

DROPPED = sorted(set(NUM_FULL) - set(NUM_REG))

def build_prep(num_cols):
    return ColumnTransformer([
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc",  StandardScaler()),
        ]), num_cols),
        ("ohe", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("enc", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), CAT_OHE),
        ("ord", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("enc", OrdinalEncoder(categories=PHASE_ORDER,
                                   handle_unknown="use_encoded_value", unknown_value=-1)),
        ]), CAT_ORD),
    ], remainder="drop")

y = df["trial_outcome"].astype(int).values


def models_factory():
    """Three sklearn-native models. Same hyperparameters used in both runs."""
    return {
        "Logistic Regression":
            LogisticRegression(C=1.0, max_iter=2000,
                               class_weight="balanced", random_state=42),
        "Random Forest":
            RandomForestClassifier(n_estimators=400, max_depth=8,
                                   min_samples_leaf=5, max_features=0.4,
                                   class_weight=None, random_state=42, n_jobs=1),
        "Gradient Boosting":
            GradientBoostingClassifier(n_estimators=200, learning_rate=0.08,
                                       max_depth=4, random_state=42),
    }


def run_feature_set(num_cols, label):
    """Train all 3 models on this feature set, return {model: test AUC}."""
    feat_cols = num_cols + CAT_OHE + CAT_ORD
    X = df[feat_cols]
    prep = build_prep(num_cols)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    results = {}
    for name, clf in models_factory().items():
        pipe = Pipeline([("prep", prep), ("clf", clf)])
        pipe.fit(X_tr, y_tr)
        y_prob = pipe.predict_proba(X_te)[:, 1]
        auc = roc_auc_score(y_te, y_prob)
        results[name] = float(auc)
        print(f"  [{label:>4}]  {name:<22}  AUC = {auc:.4f}")
    return results


print("=" * 70)
print("  REGISTRATION-TIME LEAKAGE AUDIT  (standalone)")
print("=" * 70)
print()
print("  FULL features:")
print(f"    Numeric:  {NUM_FULL}")
print(f"    OHE:      {CAT_OHE}")
print(f"    Ordinal:  {CAT_ORD}")
print()
print("  REGISTRATION-only features:")
print(f"    Numeric:  {NUM_REG}")
print(f"    OHE:      {CAT_OHE}")
print(f"    Ordinal:  {CAT_ORD}")
print()
print(f"  Dropped (conduct-time): {DROPPED}")
print()

print("── Training on FULL feature set ──")
full_aucs = run_feature_set(NUM_FULL, "FULL")

print("\n── Training on REGISTRATION-only feature set ──")
reg_aucs = run_feature_set(NUM_REG, "REG")

# ── Comparison ──────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("  LEAKAGE AUDIT - SIDE BY SIDE")
print("=" * 70)
print()
print(f"  {'Model':<22} {'Full AUC':>10} {'Reg-only AUC':>14}  {'Δ AUC':>9}")
print(f"  {'-'*22} {'-'*10} {'-'*14}  {'-'*9}")

audit = {}
for name in full_aucs:
    delta = reg_aucs[name] - full_aucs[name]
    audit[name] = {
        "full_auc": full_aucs[name],
        "reg_auc":  reg_aucs[name],
        "delta":    float(delta),
    }
    print(f"  {name:<22}  {full_aucs[name]:.4f}        "
          f"{reg_aucs[name]:.4f}        {delta:+.4f}")

# ── Headline ────────────────────────────────────────────────────────────────
best_full_name = max(full_aucs, key=full_aucs.get)
best_full_auc  = full_aucs[best_full_name]
best_reg_name  = max(reg_aucs, key=reg_aucs.get)
best_reg_auc   = reg_aucs[best_reg_name]
drop = best_full_auc - best_reg_auc

# Also note the global best - pull from model_results.json if present
global_best_auc = best_full_auc
global_best_name = best_full_name
try:
    with open("outputs/model_results.json") as f:
        all_results = json.load(f)
    for name, r in all_results.items():
        if r.get("auc", 0) > global_best_auc:
            global_best_auc = r["auc"]
            global_best_name = name
except Exception:
    pass

print()
print("  ── HONEST HEADLINE NUMBERS ──")
print(f"     Global best (full features, all 11 models) : {global_best_auc:.4f}  ({global_best_name})")
print(f"     Best registration-only model               : {best_reg_auc:.4f}  ({best_reg_name})")
print(f"     AUC drop attributable to leakage           : {global_best_auc - best_reg_auc:+.4f}")
print()

if (global_best_auc - best_reg_auc) > 0.10:
    print("  → Substantial AUC drop. The headline number is driven heavily by")
    print("    conduct-time features. The registration-only model is the one")
    print("    that would be usable for prospective prediction at trial start.")
elif (global_best_auc - best_reg_auc) > 0.03:
    print("  → Moderate AUC drop. Conduct-time features contribute meaningfully")
    print("    but registration-only features still hold useful signal.")
else:
    print("  → Small AUC drop. Registration-time features alone carry most")
    print("    of the predictive signal - leakage concern is limited here.")
print()

# ── Save audit JSON ─────────────────────────────────────────────────────────
audit_out = {
    "feature_sets": {
        "full":            sorted(NUM_FULL + CAT_OHE + CAT_ORD),
        "registration":    sorted(NUM_REG + CAT_OHE + CAT_ORD),
        "dropped_for_reg": DROPPED,
    },
    "results": audit,
    "summary": {
        "global_best_full_auc":  float(global_best_auc),
        "global_best_full_model": global_best_name,
        "best_reg_auc":          float(best_reg_auc),
        "best_reg_model":        best_reg_name,
        "auc_drop":              float(global_best_auc - best_reg_auc),
    },
    "note": "Standalone audit using sklearn-native models only "
            "(no XGBoost/LightGBM) to avoid macOS OpenMP segfault.",
}
with open("outputs/leakage_audit.json", "w") as f:
    json.dump(audit_out, f, indent=2)
print(f"  Saved → outputs/leakage_audit.json")

# ── Append to best_model_report.txt ─────────────────────────────────────────
try:
    with open("outputs/best_model_report.txt", "a") as f:
        f.write("\n\n" + "=" * 70 + "\n")
        f.write("REGISTRATION-TIME LEAKAGE AUDIT\n")
        f.write("=" * 70 + "\n")
        f.write(f"Dropped conduct-time features: {DROPPED}\n\n")
        f.write(f"  {'Model':<22} {'Full AUC':>10} {'Reg AUC':>10} {'Δ AUC':>9}\n")
        for name, r in audit.items():
            f.write(f"  {name:<22}  {r['full_auc']:.4f}     "
                    f"{r['reg_auc']:.4f}     {r['delta']:+.4f}\n")
        f.write(f"\nGlobal best (full):  {global_best_auc:.4f}  ({global_best_name})\n")
        f.write(f"Best reg-only:       {best_reg_auc:.4f}  ({best_reg_name})\n")
        f.write(f"AUC drop:            {global_best_auc - best_reg_auc:+.4f}\n")
    print(f"  Appended → outputs/best_model_report.txt")
except Exception as e:
    print(f"  Could not update best_model_report.txt: {e}")

print()
print("Done.")
