"""
pipeline.py — canonical modeling logic for the trial-completion leakage study.

One model family (Random Forest) is held fixed across the three-number arc so the
only thing that changes between steps is (1) which features are allowed, (2) how the
success label is defined, and (3) how the train/test split is drawn. That isolates
each layer of leakage instead of confounding it with a model change.

sklearn-only on purpose: no XGBoost/LightGBM/torch, so the canonical run is fast,
reproducible, and free of the macOS OpenMP segfault that affected the old pipeline.
"""
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
CLEAN_CSV = ROOT / "data" / "crispr_trials_clean.csv"

PHASE_ORDER = [["Early Phase I", "Phase I", "Phase I/II", "Phase II",
                "Phase III", "Phase IV", "N/A", "Unknown", "PHASE2/PHASE3"]]

INPROG   = {"RECRUITING", "ACTIVE_NOT_RECRUITING"}      # "success" by default, never completed
RESOLVED = {"COMPLETED", "TERMINATED", "WITHDRAWN", "SUSPENDED"}

# ── Feature sets ──────────────────────────────────────────────────────────────
NUM_FULL = ["log_enrollment", "start_year", "n_countries", "n_primary_outcomes",
            "is_hematologic", "is_recent", "results_available"]
NUM_REG  = ["start_year", "n_primary_outcomes", "is_hematologic", "is_recent"]
CAT_OHE  = ["cancer_type", "tumor_category", "sponsor_class_clean", "trial_era"]
CAT_ORD  = ["phase_clean"]

# time-encoding features (calendar recency) — used to isolate how much signal is "just time"
TIME_NUM = ["start_year", "is_recent"]
TIME_CAT = ["trial_era"]
NUM_REG_NOTIME = ["n_primary_outcomes", "is_hematologic"]
CAT_NOTIME     = ["cancer_type", "tumor_category", "sponsor_class_clean"]

CONDUCT_LEAK_REASONS = {
    "results_available": "Post-outcome. True only for trials that completed AND posted results; "
                         "a terminated trial essentially never has this.",
    "log_enrollment":    "Conduct-time. Uses the actual enrolled count, which is mechanically small "
                         "for trials that terminated early. Reverse causality.",
    "n_countries":       "Conduct-time. Trial sites can be added after registration.",
}

IMPUTE_COLS = ["is_hematologic", "is_recent", "log_enrollment",
               "n_countries", "n_primary_outcomes", "start_year"]


def load(label="bundled"):
    """Return (dataframe, y). label='bundled' = current target; 'strict' = COMPLETED vs
    TERMINATED/WITHDRAWN/SUSPENDED with in-progress trials dropped."""
    df = pd.read_csv(CLEAN_CSV)
    if label == "bundled":
        d = df[df["trial_outcome"].notna()].copy()
        y = d["trial_outcome"].astype(int).values
    elif label == "strict":
        d = df[df["overall_status"].isin(RESOLVED)].copy()
        y = (d["overall_status"] == "COMPLETED").astype(int).values
    else:
        raise ValueError(label)
    for c in IMPUTE_COLS:
        if c in d.columns:
            d[c] = d[c].fillna(d[c].median())
    return d, y


def _prep(num, ohe, ordc):
    parts = []
    if num:
        parts.append(("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                                       ("sc", StandardScaler())]), num))
    if ohe:
        parts.append(("ohe", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                                        ("enc", OneHotEncoder(handle_unknown="ignore",
                                                              sparse_output=False))]), ohe))
    if ordc:
        parts.append(("ord", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                                        ("enc", OrdinalEncoder(categories=PHASE_ORDER,
                                            handle_unknown="use_encoded_value", unknown_value=-1))]), ordc))
    return ColumnTransformer(parts, remainder="drop")


def rf():
    return RandomForestClassifier(n_estimators=400, max_depth=8, min_samples_leaf=5,
                                  max_features=0.4, class_weight=None,
                                  random_state=SEED, n_jobs=1)

def lr():
    return LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced", random_state=SEED)

def gb():
    return GradientBoostingClassifier(n_estimators=200, learning_rate=0.08,
                                      max_depth=4, random_state=SEED)


def _clean_name(n):
    """num__start_year -> start_year ; ohe__cancer_type_Lymphoma -> cancer_type=Lymphoma"""
    for pre in ("num__", "ord__"):
        if n.startswith(pre):
            return n[len(pre):]
    if n.startswith("ohe__"):
        return n[len("ohe__"):]
    return n


def fit_eval(d, y, num, ohe, ordc, model, split="random", cut=None,
             want_roc=False, want_imp=False, roc_points=100):
    """Fit `model` on (num+ohe+ordc) features and return a metrics dict.
    split='random' = stratified 80/20; split='temporal' = train start_year<cut, test>=cut."""
    feat = [c for c in num + ohe + ordc if c in d.columns]
    X = d[feat]
    prep = _prep([c for c in num if c in d.columns],
                 [c for c in ohe if c in d.columns],
                 [c for c in ordc if c in d.columns])
    if split == "random":
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.20, stratify=y, random_state=SEED)
    elif split == "temporal":
        tr, te = d["start_year"] < cut, d["start_year"] >= cut
        Xtr, Xte, ytr, yte = X[tr.values], X[te.values], y[tr.values], y[te.values]
    else:
        raise ValueError(split)

    pipe = Pipeline([("prep", prep), ("clf", model)])
    pipe.fit(Xtr, ytr)
    prob = pipe.predict_proba(Xte)[:, 1]
    out = {
        "auc": round(float(roc_auc_score(yte, prob)), 4),
        "n_train": int(len(ytr)), "n_test": int(len(yte)),
        "test_base_rate": round(float(np.mean(yte)), 3),
        "split": split,
    }
    if split == "temporal":
        out["cut_year"] = int(cut)
    if want_roc:
        fpr, tpr, _ = roc_curve(yte, prob)
        idx = np.linspace(0, len(fpr) - 1, min(roc_points, len(fpr))).astype(int)
        out["roc"] = {"fpr": [round(float(x), 4) for x in fpr[idx]],
                      "tpr": [round(float(x), 4) for x in tpr[idx]]}
    if want_imp and hasattr(pipe.named_steps["clf"], "feature_importances_"):
        names = pipe.named_steps["prep"].get_feature_names_out()
        imp = pipe.named_steps["clf"].feature_importances_
        order = np.argsort(imp)[::-1][:12]
        out["importance"] = [{"feature": _clean_name(str(names[i])),
                              "importance": round(float(imp[i]), 4)} for i in order]
    return out


def label_composition():
    """Descriptive proof that the bundled success label is time-dependent by construction."""
    df = pd.read_csv(CLEAN_CSV)
    d = df[df["trial_outcome"].notna()].copy()
    pos = d[d["trial_outcome"] == 1]
    neg = d[d["trial_outcome"] == 0]
    by_status = {s: int(n) for s, n in pos["overall_status"].value_counts().items()}
    eras = []
    for era in ["1990s", "2000s", "2010-2014", "2015-2019", "2020+"]:
        g = d[d["trial_era"] == era]
        if len(g) == 0:
            continue
        pg = g[g["trial_outcome"] == 1]
        eras.append({
            "era": era, "n": int(len(g)),
            "success_rate": round(float(g["trial_outcome"].mean()), 3),
            "pct_positives_in_progress": round(float(pg["overall_status"].isin(INPROG).mean()), 3),
        })
    return {
        "n_labeled": int(len(d)),
        "n_positive": int(len(pos)),
        "positive_rate": round(float(d["trial_outcome"].mean()), 3),
        "positive_class_by_status": by_status,
        "inprogress_share_of_positives": round(float(pos["overall_status"].isin(INPROG).mean()), 3),
        "median_start_year": {
            "completed": int(pos[pos["overall_status"] == "COMPLETED"]["start_year"].median()),
            "in_progress": int(pos[pos["overall_status"].isin(INPROG)]["start_year"].median()),
            "failures": int(neg["start_year"].median()),
        },
        "by_era": eras,
    }
