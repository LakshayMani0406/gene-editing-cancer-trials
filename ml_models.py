"""
ml_models.py
------------
Trains and compares classifiers on the gene-editing cancer trial dataset.
Adds XGBoost, LightGBM, and SHAP explainability.

Section 10 (the LEAKAGE AUDIT) re-runs the top-3 models on a
"registration-time only" feature set to quantify how much of the headline
AUC is driven by features that are partly determined by trial conduct
(log_enrollment, results_available). This is the honest comparison.

Models:
  1. Logistic Regression
  2. SVM (RBF kernel)
  3. K-Nearest Neighbors
  4. Random Forest
  5. Gradient Boosting
  6. Extra Trees
  7. XGBoost
  8. LightGBM
  + Soft Voting Ensemble (top 3 by CV AUC)

Usage:
    python3 ml_models.py    (run after clean_data.py)

Install extras if needed:
    pip install xgboost lightgbm shap
"""

import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from time import time

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_validate,
    GridSearchCV, learning_curve, cross_val_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    ExtraTreesClassifier, VotingClassifier
)
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report, average_precision_score
)
from sklearn.inspection import permutation_importance

# Optional imports — degrade gracefully if not installed
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("  xgboost not found — skipping. Install: pip install xgboost")

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("  lightgbm not found — skipping. Install: pip install lightgbm")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("  shap not found — skipping. Install: pip install shap")

Path("outputs").mkdir(exist_ok=True)

# ── Styling ───────────────────────────────────────────────────────────────────
DARK_BG  = "#0a1628"; SURFACE  = "#0d1f3c"; BORDER   = "#1e3a5f"
TEXT_PRI = "#c0d8f0"; TEXT_SEC = "#8ab4d4"
CYAN  = "#00d4aa"; BLUE = "#4a9eff"; PURPLE = "#8b5cf6"
ORANGE = "#f59e0b"; RED = "#ef4444"; GREEN = "#22c55e"; PINK = "#ec4899"
LIME  = "#84cc16"; TEAL = "#14b8a6"
COLORS = [BLUE, PURPLE, ORANGE, CYAN, GREEN, RED, PINK, LIME, TEAL]

plt.rcParams.update({
    "figure.facecolor": DARK_BG, "axes.facecolor": SURFACE,
    "axes.edgecolor": BORDER,    "axes.labelcolor": TEXT_PRI,
    "xtick.color": TEXT_SEC,     "ytick.color": TEXT_SEC,
    "text.color": TEXT_PRI,      "grid.color": BORDER,
    "grid.alpha": 0.4,           "axes.titlesize": 11,
    "axes.labelsize": 9,
})

np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load & feature engineer
# ─────────────────────────────────────────────────────────────────────────────
df_all = pd.read_csv("data/crispr_trials_clean.csv")
df = df_all[df_all["trial_outcome"].notna()].copy()

print(f"Trials with outcome labels: {len(df):,}")
print(f"Class balance: {df['trial_outcome'].mean():.1%} success\n")

PHASE_ORDER = [["Early Phase I","Phase I","Phase I/II","Phase II",
                "Phase III","Phase IV","N/A","Unknown","PHASE2/PHASE3"]]

for c in ["is_hematologic","is_recent","log_enrollment",
          "n_countries","n_primary_outcomes","start_year"]:
    if c in df.columns:
        df[c] = df[c].fillna(df[c].median())

# ── FEATURE SETS ─────────────────────────────────────────────────────────────
# FULL: everything in the cleaned dataset — includes conduct-time features.
#       Reported as the headline number but acknowledged as upward-biased.
# REG:  registration-time only — features known when a trial is first
#       registered. No log_enrollment, no results_available.
# ─────────────────────────────────────────────────────────────────────────────

# Shared categorical features (declared at registration, stable)
CAT_OHE = [c for c in ["cancer_type","tumor_category",
                        "sponsor_class_clean","trial_era"] if c in df.columns]
CAT_ORD = ["phase_clean"]

# FULL feature set (current behaviour — used in section 2-9)
NUM_FULL = [c for c in ["log_enrollment","start_year","n_countries",
                         "n_primary_outcomes","is_hematologic","is_recent",
                         "results_available"] if c in df.columns]

# REGISTRATION-TIME feature set (used in section 10)
# Drops:
#   log_enrollment       — actual count is determined during trial conduct;
#                          terminated trials have small actuals → reverse causality
#   results_available    — only true for trials that completed AND posted results
#   n_countries          — sites can be added during conduct (borderline,
#                          but we exclude to be conservative)
NUM_REG = [c for c in ["start_year","n_primary_outcomes",
                        "is_hematologic","is_recent"] if c in df.columns]

NUM = NUM_FULL  # default — section 2-9 uses the full feature set

def build_preprocessor(num_cols, cat_ohe, cat_ord):
    return ColumnTransformer([
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc",  StandardScaler()),
        ]), num_cols),
        ("ohe", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("enc", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), cat_ohe),
        ("ord", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("enc", OrdinalEncoder(categories=PHASE_ORDER,
                                   handle_unknown="use_encoded_value", unknown_value=-1)),
        ]), cat_ord),
    ], remainder="drop")

preprocessor = build_preprocessor(NUM_FULL, CAT_OHE, CAT_ORD)

feat_cols = [c for c in NUM_FULL + CAT_OHE + CAT_ORD if c in df.columns]
X = df[feat_cols]
y = df["trial_outcome"].astype(int).values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)
print(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}")
print(f"Positive rate — Train: {y_train.mean():.1%}  |  Test: {y_test.mean():.1%}\n")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def make_pipe(clf, prep=None):
    return Pipeline([("prep", prep if prep is not None else preprocessor),
                     ("clf", clf)])


# ─────────────────────────────────────────────────────────────────────────────
# 2. Define all models
# ─────────────────────────────────────────────────────────────────────────────
base_models = {
    "Logistic Regression": make_pipe(
        LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced", random_state=42)
    ),
    "SVM (RBF)": make_pipe(
        SVC(kernel="rbf", C=1.0, probability=True, class_weight="balanced", random_state=42)
    ),
    "K-Nearest Neighbors": make_pipe(KNeighborsClassifier(n_neighbors=9)),
    "Random Forest": make_pipe(
        RandomForestClassifier(n_estimators=300, class_weight="balanced",
                               random_state=42, n_jobs=-1)
    ),
    "Gradient Boosting": make_pipe(
        GradientBoostingClassifier(n_estimators=200, learning_rate=0.08,
                                   max_depth=4, random_state=42)
    ),
    "Extra Trees": make_pipe(
        ExtraTreesClassifier(n_estimators=300, class_weight="balanced",
                             random_state=42, n_jobs=-1)
    ),
}

if HAS_XGB:
    base_models["XGBoost"] = make_pipe(
        XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=5,
                      subsample=0.8, colsample_bytree=0.8,
                      scale_pos_weight=(y_train==0).sum()/(y_train==1).sum(),
                      eval_metric="logloss", random_state=42,
                      verbosity=0, n_jobs=-1)
    )

if HAS_LGB:
    base_models["LightGBM"] = make_pipe(
        lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, max_depth=5,
                           subsample=0.8, colsample_bytree=0.8,
                           class_weight="balanced", random_state=42,
                           verbose=-1, n_jobs=-1)
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3. Baseline cross-validation
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("  BASELINE 5-FOLD CROSS-VALIDATION  (full feature set — includes conduct-time signal)")
print("=" * 70)
print(f"  {'Model':<28} {'Accuracy':>10} {'AUC-ROC':>10} {'F1':>10}  {'Time':>7}")
print(f"  {'-'*28} {'-'*10} {'-'*10} {'-'*10}  {'-'*7}")

cv_results = {}
for name, pipe in base_models.items():
    t0 = time()
    scores = cross_validate(pipe, X_train, y_train, cv=cv,
                            scoring=["accuracy","roc_auc","f1"],
                            return_train_score=False, n_jobs=-1)
    elapsed = time() - t0
    cv_results[name] = {k: scores[f"test_{k}"] for k in ["accuracy","roc_auc","f1"]}
    cv_results[name]["time"] = elapsed
    r = cv_results[name]
    print(f"  {name:<28}  {r['accuracy'].mean():.4f}±{r['accuracy'].std():.3f}"
          f"  {r['roc_auc'].mean():.4f}±{r['roc_auc'].std():.3f}"
          f"  {r['f1'].mean():.4f}±{r['f1'].std():.3f}"
          f"  {elapsed:>6.1f}s")

best_base = max(cv_results, key=lambda k: cv_results[k]["roc_auc"].mean())
print(f"\n  Best baseline: {best_base}  (AUC={cv_results[best_base]['roc_auc'].mean():.4f})")


# ─────────────────────────────────────────────────────────────────────────────
# 4. GridSearchCV on top models
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  GRIDSEARCHCV HYPERPARAMETER TUNING")
print("=" * 70)

rf_grid = {
    "clf__n_estimators":     [200, 400],
    "clf__max_depth":        [None, 8, 15],
    "clf__min_samples_leaf": [1, 5],
    "clf__max_features":     ["sqrt", 0.4],
    "clf__class_weight":     ["balanced", None],
}
print("  Tuning Random Forest...")
t0 = time()
rf_gs = GridSearchCV(
    make_pipe(RandomForestClassifier(random_state=42, n_jobs=-1)),
    rf_grid, cv=3, scoring="roc_auc", n_jobs=-1, refit=True, verbose=0
)
rf_gs.fit(X_train, y_train)
print(f"    Best CV AUC: {rf_gs.best_score_:.4f}  ({time()-t0:.1f}s)")
print(f"    Params: {rf_gs.best_params_}")

tuned_models = {"Random Forest (Tuned)": rf_gs.best_estimator_}

if HAS_XGB:
    xgb_grid = {
        "clf__n_estimators":    [200, 400],
        "clf__learning_rate":   [0.03, 0.05, 0.10],
        "clf__max_depth":       [3, 5, 7],
        "clf__subsample":       [0.7, 0.9],
    }
    print("  Tuning XGBoost...")
    t0 = time()
    xgb_gs = GridSearchCV(
        make_pipe(XGBClassifier(eval_metric="logloss", random_state=42,
                                verbosity=0, n_jobs=-1)),
        xgb_grid, cv=3, scoring="roc_auc", n_jobs=-1, refit=True, verbose=0
    )
    xgb_gs.fit(X_train, y_train)
    print(f"    Best CV AUC: {xgb_gs.best_score_:.4f}  ({time()-t0:.1f}s)")
    print(f"    Params: {xgb_gs.best_params_}")
    tuned_models["XGBoost (Tuned)"] = xgb_gs.best_estimator_


# ─────────────────────────────────────────────────────────────────────────────
# 5. Voting Ensemble
# ─────────────────────────────────────────────────────────────────────────────
print("\n  Building Soft Voting Ensemble (top 3 by CV AUC)...")
top3_names = sorted(cv_results, key=lambda k: cv_results[k]["roc_auc"].mean(), reverse=True)[:3]
print(f"    Using: {top3_names}")

for name in top3_names:
    base_models[name].fit(X_train, y_train)

ensemble = VotingClassifier(
    estimators=[(n.replace(" ","_"), base_models[n]) for n in top3_names],
    voting="soft"
)
ensemble.fit(X_train, y_train)
tuned_models["Soft Voting Ensemble"] = ensemble


# ─────────────────────────────────────────────────────────────────────────────
# 6. Holdout test evaluation
# ─────────────────────────────────────────────────────────────────────────────
all_eval = {**base_models, **tuned_models}
for name, pipe in base_models.items():
    if name not in top3_names:
        pipe.fit(X_train, y_train)

print("\n" + "=" * 70)
print("  HOLDOUT TEST SET EVALUATION  (full feature set)")
print("=" * 70)
print(f"  {'Model':<30} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'AUC':>7}")
print(f"  {'-'*30} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")

test_results = {}
for name, pipe in all_eval.items():
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]
    test_results[name] = {
        "acc":    accuracy_score(y_test, y_pred),
        "prec":   precision_score(y_test, y_pred, zero_division=0),
        "rec":    recall_score(y_test, y_pred, zero_division=0),
        "f1":     f1_score(y_test, y_pred, zero_division=0),
        "auc":    roc_auc_score(y_test, y_prob),
        "y_pred": y_pred, "y_prob": y_prob,
    }
    r = test_results[name]
    print(f"  {name:<30} {r['acc']:.4f}  {r['prec']:.4f}  {r['rec']:.4f}  "
          f"{r['f1']:.4f}  {r['auc']:.4f}")

best_name = max(test_results, key=lambda k: test_results[k]["auc"])
best = test_results[best_name]

print(f"\n  Best model (full features): {best_name}")
print(f"    Accuracy={best['acc']:.4f}  AUC={best['auc']:.4f}  F1={best['f1']:.4f}")
print(f"\n  NOTE: this AUC reflects the full feature set, which includes")
print(f"  features partly determined by trial conduct. See section 10 for")
print(f"  the registration-time-only audit.\n")

print("=" * 70)
print(f"  CLASSIFICATION REPORT — {best_name}  (full features)")
print("=" * 70)
print(classification_report(y_test, best["y_pred"],
                              target_names=["Terminated/Withdrawn","Completed/Active"]))

with open("outputs/best_model_report.txt", "w") as f:
    f.write(f"Best Model (FULL feature set): {best_name}\n{'='*70}\n\n")
    f.write("NOTE: This is the headline number reported in dashboards.\n")
    f.write("The full feature set includes conduct-time features\n")
    f.write("(log_enrollment, results_available). See section 10 of\n")
    f.write("ml_models.py for the registration-time-only comparison.\n\n")
    for k in ["acc","prec","rec","f1","auc"]:
        f.write(f"  {k.upper()}: {best[k]:.4f}\n")
    f.write("\nClassification Report:\n")
    f.write(classification_report(y_test, best["y_pred"],
                                   target_names=["Terminated/Withdrawn","Completed/Active"]))
    f.write("\n\nAll Models (FULL feature set):\n")
    for name, r in test_results.items():
        f.write(f"  {name:<30} acc={r['acc']:.4f}  auc={r['auc']:.4f}\n")

# Save full-feature results as JSON for dashboard (preserves old schema)
import json
results_for_dashboard = {
    name: {k: float(v) for k, v in r.items() if k not in ["y_pred","y_prob"]}
    for name, r in test_results.items()
}
with open("outputs/model_results.json", "w") as f:
    json.dump(results_for_dashboard, f, indent=2)
print("  Saved → outputs/best_model_report.txt")
print("  Saved → outputs/model_results.json")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Figure 5 — Model comparison
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(20, 7))
fig.suptitle("Model Comparison — All Models (full feature set)", fontsize=14, color="white")

ax = axes[0]
names = list(test_results.keys())
accs = [test_results[m]["acc"]*100 for m in names]
aucs = [test_results[m]["auc"]*100 for m in names]
x = np.arange(len(names)); w = 0.35
ax.bar(x-w/2, accs, w, color=CYAN,   edgecolor=BORDER, alpha=0.85, label="Accuracy")
ax.bar(x+w/2, aucs, w, color=ORANGE, edgecolor=BORDER, alpha=0.85, label="AUC-ROC")
ax.set_xticks(x); ax.set_xticklabels(names, rotation=40, ha="right", fontsize=7)
ax.set_ylabel("Score (%)"); ax.set_title("Accuracy & AUC-ROC Comparison")
ax.axhline(80, color=TEXT_SEC, linestyle=":", linewidth=1, alpha=0.5)
ax.legend(fontsize=9); ax.set_ylim(40, 105); ax.grid(True, axis="y", alpha=0.3)
bi = names.index(best_name)
ax.text(bi, aucs[bi]+1.5, "★", ha="center", color=ORANGE, fontsize=14)

ax = axes[1]
sorted_names = sorted(test_results, key=lambda k: test_results[k]["auc"], reverse=True)
for i, name in enumerate(sorted_names):
    r = test_results[name]
    fpr, tpr, _ = roc_curve(y_test, r["y_prob"])
    lw = 3 if name == best_name else 1.5
    ls = "-" if name == best_name else "--"
    ax.plot(fpr, tpr, color=COLORS[i % len(COLORS)], linewidth=lw, linestyle=ls,
            label=f"{name[:20]} ({r['auc']:.3f})")
ax.plot([0,1],[0,1], color=TEXT_SEC, linestyle=":", linewidth=1)
fpr_b, tpr_b, _ = roc_curve(y_test, best["y_prob"])
ax.fill_between(fpr_b, tpr_b, alpha=0.07, color=CYAN)
ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
ax.set_title("ROC Curves"); ax.legend(fontsize=6, loc="lower right")
ax.grid(True, alpha=0.3)

ax = axes[2]
top3 = sorted_names[:3]
for i, name in enumerate(top3):
    r = test_results[name]
    pr, rc, _ = precision_recall_curve(y_test, r["y_prob"])
    ap = average_precision_score(y_test, r["y_prob"])
    ax.plot(rc, pr, color=COLORS[i], linewidth=2.5 if i==0 else 1.5,
            label=f"{name[:20]} (AP={ap:.3f})")
    ax.fill_between(rc, pr, alpha=0.06, color=COLORS[i])
ax.axhline(y_test.mean(), color=TEXT_SEC, linestyle=":", linewidth=1,
           label=f"Baseline ({y_test.mean():.2f})")
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_title("Precision-Recall (Top 3)"); ax.legend(fontsize=7, loc="lower left")
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig("outputs/fig5_ml_comparison.png", dpi=150, bbox_inches="tight", facecolor=DARK_BG)
print("  Saved → outputs/fig5_ml_comparison.png")
plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 8. Figure 6 — Best model deep-dive
# ─────────────────────────────────────────────────────────────────────────────
interp_pipe = rf_gs.best_estimator_

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle(f"Deep Dive: {best_name} (full feature set)", fontsize=14, color="white")

# Confusion matrix
ax = axes[0, 0]
cm = confusion_matrix(y_test, best["y_pred"])
tn, fp, fn, tp = cm.ravel()
for i in range(2):
    for j in range(2):
        c = [[BLUE, RED],[ORANGE, CYAN]][i][j]
        ax.add_patch(plt.Rectangle([j-.5,i-.5],1,1,facecolor=c,alpha=0.35,
                                    edgecolor=BORDER,linewidth=2))
        pct = cm[i,j]/cm.sum()*100
        ax.text(j, i, f"{cm[i,j]}\n({pct:.1f}%)", ha="center", va="center",
                fontsize=12, fontweight="bold", color="white")
ax.set_xlim(-.5,1.5); ax.set_ylim(-.5,1.5)
ax.set_xticks([0,1]); ax.set_xticklabels(["Predicted\nFailure","Predicted\nSuccess"])
ax.set_yticks([0,1]); ax.set_yticklabels(["Actual\nFailure","Actual\nSuccess"])
tpr_v = tp/(tp+fn) if (tp+fn) else 0
tnr_v = tn/(tn+fp) if (tn+fp) else 0
ax.set_title(f"Confusion Matrix  |  TPR={tpr_v:.3f}  TNR={tnr_v:.3f}")

# Permutation importance
ax = axes[0, 1]
perm = permutation_importance(interp_pipe, X_test, y_test, n_repeats=10,
                               random_state=42, n_jobs=-1, scoring="roc_auc")
fi = pd.Series(perm.importances_mean, index=feat_cols).sort_values(ascending=False)
fi_pos = fi[fi > 0].head(12)
bar_colors = [CYAN if v == fi_pos.max() else BLUE for v in fi_pos.values]
bars = ax.barh(fi_pos.index[::-1], fi_pos.values[::-1],
               color=bar_colors[::-1], edgecolor=BORDER, height=0.65)
for bar, val in zip(bars, fi_pos.values[::-1]):
    ax.text(val+0.0005, bar.get_y()+bar.get_height()/2,
            f"{val:.4f}", va="center", fontsize=7.5)
ax.set_xlabel("Mean AUC Decrease"); ax.set_title("Feature Importance (Permutation, n=10)")
ax.grid(True, axis="x", alpha=0.3)

# Learning curve
ax = axes[1, 0]
train_sz, tr_sc, va_sc = learning_curve(
    interp_pipe, X_train, y_train, cv=cv,
    train_sizes=np.linspace(0.1,1.0,8), scoring="roc_auc", n_jobs=-1
)
tr_m = tr_sc.mean(axis=1); tr_s = tr_sc.std(axis=1)
va_m = va_sc.mean(axis=1); va_s = va_sc.std(axis=1)
ax.plot(train_sz, tr_m, color=ORANGE, linewidth=2, marker="o", markersize=4, label="Train AUC")
ax.plot(train_sz, va_m, color=CYAN,   linewidth=2, marker="s", markersize=4, label="CV AUC")
ax.fill_between(train_sz, tr_m-tr_s, tr_m+tr_s, alpha=0.1, color=ORANGE)
ax.fill_between(train_sz, va_m-va_s, va_m+va_s, alpha=0.1, color=CYAN)
ax.set_xlabel("Training Set Size"); ax.set_ylabel("AUC-ROC")
ax.set_title("Learning Curve"); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# Calibration curve
ax = axes[1, 1]
for name in sorted_names[:3]:
    r = test_results[name]
    frac_pos, mean_pred = calibration_curve(y_test, r["y_prob"], n_bins=8)
    ax.plot(mean_pred, frac_pos, marker="o", markersize=4, linewidth=2,
            label=name[:22])
ax.plot([0,1],[0,1], color=TEXT_SEC, linestyle="--", linewidth=1.5, label="Perfect calibration")
ax.set_xlabel("Mean Predicted Probability"); ax.set_ylabel("Fraction of Positives")
ax.set_title("Calibration Curves (Top 3 Models)"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig("outputs/fig6_best_model.png", dpi=150, bbox_inches="tight", facecolor=DARK_BG)
print("  Saved → outputs/fig6_best_model.png")
plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 9. SHAP Analysis
# ─────────────────────────────────────────────────────────────────────────────
if HAS_SHAP:
    print("\n" + "=" * 70)
    print("  SHAP EXPLAINABILITY ANALYSIS")
    print("=" * 70)

    # Use RF tuned — extract the fitted preprocessor + classifier
    prep_fitted = interp_pipe.named_steps["prep"]
    clf_fitted   = interp_pipe.named_steps["clf"]

    # Transform test set
    X_test_transformed = prep_fitted.transform(X_test)

    # SHAP TreeExplainer
    explainer = shap.TreeExplainer(clf_fitted)
    shap_values = explainer.shap_values(X_test_transformed)

    # For binary classification, shap_values can be:
    #   - a list [class0, class1] in older SHAP versions
    #   - a 3D ndarray (n_samples, n_features, n_classes) in newer SHAP
    if isinstance(shap_values, list):
        sv = shap_values[1]
    elif hasattr(shap_values, 'ndim') and shap_values.ndim == 3:
        sv = shap_values[:, :, 1]
    else:
        sv = shap_values

    # Get feature names after preprocessing
    try:
        num_features  = NUM_FULL
        ohe_features  = list(prep_fitted.named_transformers_["ohe"]
                             .named_steps["enc"].get_feature_names_out(CAT_OHE))
        ord_features  = CAT_ORD
        all_features  = num_features + ohe_features + ord_features
    except Exception:
        all_features = [f"feature_{i}" for i in range(X_test_transformed.shape[1])]

    # Pad/trim to match
    n_feat = X_test_transformed.shape[1]
    all_features = all_features[:n_feat]
    while len(all_features) < n_feat:
        all_features.append(f"feat_{len(all_features)}")

    # Figure 7 — SHAP summary + waterfall
    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    fig.suptitle("SHAP Explainability — Random Forest (Tuned, full features)", fontsize=14, color="white")

    # SHAP beeswarm (summary plot) — manual implementation
    ax = axes[0]
    mean_abs_shap = np.abs(sv).mean(axis=0).flatten()
    # Guard: clamp top_idx to actual sv feature dimension
    sv_n_features = sv.shape[1]
    mean_abs_shap = mean_abs_shap[:sv_n_features]
    top_idx  = np.argsort(mean_abs_shap)[-15:]
    top_idx  = [int(x) for x in top_idx if int(x) < len(all_features) and int(x) < sv_n_features]
    sv_top   = sv[:, top_idx]
    feat_top = [all_features[i] for i in top_idx]

    # Use sample of test points for clarity
    sample = min(200, len(sv_top))
    sv_s = sv_top[:sample]
    X_s  = X_test_transformed[:sample, :][:, top_idx]

    for j, (feat, idx) in enumerate(zip(feat_top, top_idx)):
        shap_vals = sv_s[:, j]
        feat_vals = X_s[:, j]
        # Normalize feature values for color
        vmin, vmax = feat_vals.min(), feat_vals.max()
        if vmax > vmin:
            normed = (feat_vals - vmin) / (vmax - vmin)
        else:
            normed = np.zeros_like(feat_vals)
        colors_shap = plt.cm.RdYlGn(normed)
        jitter = np.random.normal(0, 0.08, size=len(shap_vals))
        ax.scatter(shap_vals, np.full_like(shap_vals, j) + jitter,
                   c=colors_shap, alpha=0.4, s=12)

    ax.axvline(0, color=TEXT_SEC, linewidth=1, linestyle="--")
    ax.set_yticks(range(len(feat_top)))
    ax.set_yticklabels(feat_top, fontsize=8)
    ax.set_xlabel("SHAP Value (impact on model output)")
    ax.set_title("SHAP Beeswarm — Feature Impact on Trial Success\n(color = feature value: green=high, red=low)")
    ax.grid(True, axis="x", alpha=0.3)

    # Mean SHAP bar chart
    ax = axes[1]
    mean_shap_top = mean_abs_shap[top_idx]
    bar_cols = [CYAN if v == mean_shap_top.max() else BLUE for v in mean_shap_top]
    ax.barh(feat_top, mean_shap_top, color=bar_cols, edgecolor=BORDER, height=0.65)
    for i, val in enumerate(mean_shap_top):
        ax.text(val + 0.0002, i, f"{val:.4f}", va="center", fontsize=8)
    ax.set_xlabel("Mean |SHAP Value| (mean impact magnitude)")
    ax.set_title("Mean Absolute SHAP Values\n(top 15 features by importance)")
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    fig.savefig("outputs/fig7_shap.png", dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    print("  Saved → outputs/fig7_shap.png")
    plt.close()

    # Save SHAP values for dashboard
    shap_summary = pd.DataFrame({
        "feature": all_features,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False).head(15)
    shap_summary.to_csv("outputs/shap_summary.csv", index=False)
    print("  Saved → outputs/shap_summary.csv")

else:
    print("\n  SHAP skipped — install with: pip install shap")


# ═════════════════════════════════════════════════════════════════════════════
# 10. REGISTRATION-TIME LEAKAGE AUDIT
# ═════════════════════════════════════════════════════════════════════════════
# The full feature set includes signals partly determined by trial conduct
# (log_enrollment becomes the ACTUAL enrolled count, which for terminated
# trials is mechanically small; results_available is true only for trials
# that completed AND posted results). This inflates the AUC.
#
# Here we retrain the top-3 models on a registration-time-only feature set
# and report the AUC drop. This is the honest comparison.
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  10. REGISTRATION-TIME LEAKAGE AUDIT")
print("=" * 70)
print()
print("  The full feature set above includes features partly determined")
print("  by trial conduct (log_enrollment, results_available, n_countries).")
print("  These inflate AUC because terminated trials mechanically end up")
print("  with low enrollment and no posted results.")
print()
print("  This section retrains on registration-time-only features:")
print(f"    Numeric: {NUM_REG}")
print(f"    OHE   : {CAT_OHE}")
print(f"    Ordinal: {CAT_ORD}")
print()
print("  Dropped (conduct-time): log_enrollment, results_available, n_countries")
print()

# Build registration-time preprocessor and feature matrix
prep_reg  = build_preprocessor(NUM_REG, CAT_OHE, CAT_ORD)
feat_reg  = [c for c in NUM_REG + CAT_OHE + CAT_ORD if c in df.columns]
X_reg     = df[feat_reg]

X_tr_reg, X_te_reg, y_tr_reg, y_te_reg = train_test_split(
    X_reg, y, test_size=0.20, stratify=y, random_state=42
)

# Same top-3 models, same hyperparameters, registration-time features only
reg_models = {
    "Logistic Regression": Pipeline([("prep", prep_reg),
        ("clf", LogisticRegression(C=1.0, max_iter=2000,
                                   class_weight="balanced", random_state=42))]),
    "Random Forest": Pipeline([("prep", prep_reg),
        ("clf", RandomForestClassifier(n_estimators=400, max_depth=None,
                                       min_samples_leaf=1, max_features="sqrt",
                                       class_weight="balanced",
                                       random_state=42, n_jobs=1))]),
    "Gradient Boosting": Pipeline([("prep", prep_reg),
        ("clf", GradientBoostingClassifier(n_estimators=200, learning_rate=0.08,
                                           max_depth=4, random_state=42))]),
}
if HAS_XGB:
    reg_models["XGBoost"] = Pipeline([("prep", prep_reg),
        ("clf", XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=5,
                              subsample=0.8, colsample_bytree=0.8,
                              scale_pos_weight=(y_tr_reg==0).sum()/(y_tr_reg==1).sum(),
                              eval_metric="logloss", random_state=42,
                              verbosity=0, n_jobs=1))])

print(f"  {'Model':<22} {'Full AUC':>10} {'Reg-only AUC':>14} {'Δ AUC':>9}")
print(f"  {'-'*22} {'-'*10} {'-'*14} {'-'*9}")

audit = {}
for name, pipe in reg_models.items():
    # CV AUC on registration-only features
    cv_scores = cross_val_score(pipe, X_tr_reg, y_tr_reg, cv=cv,
                                 scoring="roc_auc", n_jobs=1)
    pipe.fit(X_tr_reg, y_tr_reg)
    y_prob_reg = pipe.predict_proba(X_te_reg)[:, 1]
    test_auc_reg = roc_auc_score(y_te_reg, y_prob_reg)
    # Match with the corresponding full-feature result
    match_name = name
    if name == "Random Forest":
        match_name = "Random Forest (Tuned)"
    elif name == "XGBoost":
        match_name = "XGBoost (Tuned)" if "XGBoost (Tuned)" in test_results else "XGBoost"
    full_auc = test_results.get(match_name, {"auc": np.nan})["auc"]
    delta = test_auc_reg - full_auc
    audit[name] = {
        "cv_auc_mean":  float(cv_scores.mean()),
        "cv_auc_std":   float(cv_scores.std()),
        "test_auc_reg": float(test_auc_reg),
        "test_auc_full": float(full_auc),
        "delta_auc":    float(delta),
    }
    print(f"  {name:<22}  {full_auc:.4f}        {test_auc_reg:.4f}      {delta:+.4f}")

# Best registration-time model
best_reg_name = max(audit, key=lambda k: audit[k]["test_auc_reg"])
best_reg_auc  = audit[best_reg_name]["test_auc_reg"]
best_full_auc = best["auc"]
auc_drop      = best_full_auc - best_reg_auc

print()
print(f"  ── HONEST HEADLINE NUMBERS ──")
print(f"     Full feature set     : {best_full_auc:.4f} AUC  ({best_name})")
print(f"     Registration-only    : {best_reg_auc:.4f} AUC  ({best_reg_name})")
print(f"     Drop from leakage    : {auc_drop:+.4f}")
print()
if auc_drop > 0.10:
    print("  → Substantial AUC drop. The headline number is driven heavily by")
    print("    conduct-time features. The registration-only model is the one")
    print("    that would be usable for prospective prediction at trial start.")
elif auc_drop > 0.03:
    print("  → Moderate AUC drop. Conduct-time features contribute meaningfully")
    print("    but registration-only features still hold useful signal.")
else:
    print("  → Small AUC drop. Registration-time features alone carry most")
    print("    of the predictive signal — leakage concern is limited here.")
print()

# Save audit results as separate JSON for dashboard / docs
audit_out = {
    "feature_sets": {
        "full":            sorted(feat_cols),
        "registration":    sorted(feat_reg),
        "dropped_for_reg": sorted(set(feat_cols) - set(feat_reg)),
    },
    "results": audit,
    "summary": {
        "best_full_auc":  float(best_full_auc),
        "best_full_model": best_name,
        "best_reg_auc":   float(best_reg_auc),
        "best_reg_model": best_reg_name,
        "auc_drop":       float(auc_drop),
    },
}
with open("outputs/leakage_audit.json", "w") as f:
    json.dump(audit_out, f, indent=2)
print("  Saved → outputs/leakage_audit.json")

# Append to best_model_report.txt
with open("outputs/best_model_report.txt", "a") as f:
    f.write("\n\n" + "=" * 70 + "\n")
    f.write("REGISTRATION-TIME LEAKAGE AUDIT\n")
    f.write("=" * 70 + "\n")
    f.write(f"Dropped conduct-time features: {sorted(set(feat_cols) - set(feat_reg))}\n\n")
    f.write(f"  {'Model':<22} {'Full AUC':>10} {'Reg-only AUC':>14} {'Δ AUC':>9}\n")
    for name, r in audit.items():
        f.write(f"  {name:<22}  {r['test_auc_full']:.4f}        "
                f"{r['test_auc_reg']:.4f}      {r['delta_auc']:+.4f}\n")
    f.write(f"\nHonest headline (registration-only): {best_reg_auc:.4f} AUC  ({best_reg_name})\n")


# ─────────────────────────────────────────────────────────────────────────────
# Final summary
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print(f"  ★  BEST MODEL (full features)      : {best_name}")
print(f"     AUC-ROC                          : {best['auc']:.4f}")
print(f"     ⚠  Includes conduct-time signal — see leakage audit")
print()
print(f"  ★  BEST MODEL (registration-only)  : {best_reg_name}")
print(f"     AUC-ROC                          : {best_reg_auc:.4f}")
print(f"     ✓  Honest prospective number — usable at trial start")
print("=" * 70)
print()
print("  Output files:")
for f in ["fig5_ml_comparison.png","fig6_best_model.png",
          "fig7_shap.png","best_model_report.txt","model_results.json",
          "leakage_audit.json"]:
    p = Path(f"outputs/{f}")
    if p.exists():
        print(f"    outputs/{f}")
print()
print("  Next: python3 dashboard.py  (builds interactive HTML dashboard)")
