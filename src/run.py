"""
run.py — single reproducible entry point for the trial-completion leakage study.

    python src/run.py

Loads the committed clean dataset, recomputes the full three-number leakage arc,
the conduct-time leakage audit, and the label/validation audit, then writes ONE
versioned artifact: artifacts/results.json. This is the only writer of that file.

The three-number arc (Random Forest held fixed; only features/label/split change):
    1. leaked          full features,  bundled label, random split
    2. naive_clean     reg-only feats,  bundled label, random split   (still cheats via recency)
    3. strict_temporal reg-only feats,  strict label,  temporal split (the honest answer)

Re-fetching/re-cleaning is a separate optional upstream step (src/fetch_data.py,
src/clean_data.py); this script starts from data/crispr_trials_clean.csv.
"""
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

import pipeline as P

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "results.json"

# Known-good values from the independent legacy pipeline (outputs/leakage_audit.json).
# run.py must reproduce these or the export is not trustworthy.
EXPECT = {"rf_full": 0.8963, "rf_reg": 0.7185, "lr_reg": 0.6662, "gb_reg": 0.7062}
TOL = 0.01


def build():
    np.random.seed(P.SEED)

    df_b, y_b = P.load("bundled")
    df_s, y_s = P.load("strict")
    cut_b = int(np.quantile(df_b["start_year"], 0.80))
    cut_s = int(np.quantile(df_s["start_year"], 0.80))

    # ── Label / validation audit (computed first so the arc text can cite real numbers) ──
    def block(d, y, cut):
        return {
            "reg_auc_random":   P.fit_eval(d, y, P.NUM_REG, P.CAT_OHE, P.CAT_ORD, P.rf(), "random")["auc"],
            "reg_auc_temporal": P.fit_eval(d, y, P.NUM_REG, P.CAT_OHE, P.CAT_ORD, P.rf(), "temporal", cut)["auc"],
            "time_only_auc":    P.fit_eval(d, y, P.TIME_NUM, [], P.TIME_CAT, P.rf(), "random")["auc"],
            "no_time_auc":      P.fit_eval(d, y, P.NUM_REG_NOTIME, P.CAT_NOTIME, P.CAT_ORD, P.rf(), "random")["auc"],
            "n": int(len(y)), "positive_rate": round(float(np.mean(y)), 3), "cut_year": int(cut),
        }
    label_audit = {"bundled": block(df_b, y_b, cut_b), "strict": block(df_s, y_s, cut_s)}
    time_only_b = label_audit["bundled"]["time_only_auc"]

    # ── Three-number arc (RF fixed) ──────────────────────────────────────────
    leaked = P.fit_eval(df_b, y_b, P.NUM_FULL, P.CAT_OHE, P.CAT_ORD, P.rf(),
                        split="random", want_roc=True, want_imp=True)
    naive  = P.fit_eval(df_b, y_b, P.NUM_REG,  P.CAT_OHE, P.CAT_ORD, P.rf(),
                        split="random", want_roc=True, want_imp=True)
    honest = P.fit_eval(df_s, y_s, P.NUM_REG,  P.CAT_OHE, P.CAT_ORD, P.rf(),
                        split="temporal", cut=cut_s, want_roc=True, want_imp=True)

    arc = [
        {"step": 1, "key": "leaked", "label": "Full model (leaked)",
         "auc": leaked["auc"], "model": "Random Forest", "target": "bundled", "split": "random",
         "still_cheating": "Uses results_available, log_enrollment and n_countries — features only "
                           "known after the trial runs. A terminated trial mechanically has low "
                           "enrollment and no posted results, so the model reads the outcome off its own inputs.",
         "n_test": leaked["n_test"], "test_base_rate": leaked["test_base_rate"],
         "roc": leaked["roc"], "importance": leaked["importance"]},
        {"step": 2, "key": "naive_clean", "label": "Registration-only (looks honest)",
         "auc": naive["auc"], "model": "Random Forest", "target": "bundled", "split": "random",
         "still_cheating": "Drops the obvious leaks but keeps the bundled success label, which counts "
                           "not-yet-failed recent trials as wins. Calendar recency now leaks the label: "
                           "time features alone score {:.3f}.".format(time_only_b),
         "n_test": naive["n_test"], "test_base_rate": naive["test_base_rate"],
         "roc": naive["roc"], "importance": naive["importance"]},
        {"step": 3, "key": "strict_temporal", "label": "Strict label + temporal split (honest)",
         "auc": honest["auc"], "model": "Random Forest", "target": "strict", "split": "temporal",
         "cut_year": honest["cut_year"],
         "still_cheating": "Nothing left to peel. Real completions vs terminations only, trained on "
                           "the past to predict the future. Near the 0.50 coin-flip: trial completion "
                           "is barely predictable from registration metadata, and that is the finding.",
         "n_test": honest["n_test"], "test_base_rate": honest["test_base_rate"],
         "roc": honest["roc"], "importance": honest["importance"]},
    ]

    # ── Conduct-time leakage audit (LR / RF / GB, full vs reg, bundled, random) ──
    models = {"Logistic Regression": P.lr, "Random Forest": P.rf, "Gradient Boosting": P.gb}
    leak_models = {}
    for name, mk in models.items():
        full = P.fit_eval(df_b, y_b, P.NUM_FULL, P.CAT_OHE, P.CAT_ORD, mk(), split="random")
        reg  = P.fit_eval(df_b, y_b, P.NUM_REG,  P.CAT_OHE, P.CAT_ORD, mk(), split="random")
        leak_models[name] = {"full_auc": full["auc"], "reg_auc": reg["auc"],
                             "delta_auc": round(reg["auc"] - full["auc"], 4)}

    comp = P.label_composition()

    results = {
        "schema_version": "1.0",
        "generated_by": "src/run.py",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "title": "Predicting oncology gene-editing trial completion: a three-layer leakage audit",
        "headline": {
            "arc_auc": [arc[0]["auc"], arc[1]["auc"], arc[2]["auc"]],
            "punchline": "Honest prospective prediction of trial completion is near chance "
                         "(AUC ~{:.2f}). The apparent skill was leakage at three layers.".format(arc[2]["auc"]),
        },
        "arc": arc,
        "callouts": {
            "inprogress_share_of_positives": comp["inprogress_share_of_positives"],
            "time_only_auc_bundled": label_audit["bundled"]["time_only_auc"],
            "no_time_auc_bundled": label_audit["bundled"]["no_time_auc"],
            "share_of_above_chance_from_time": round(
                (label_audit["bundled"]["time_only_auc"] - 0.5) / (arc[1]["auc"] - 0.5), 3),
        },
        "leakage_audit": {
            "feature_sets": {
                "full": sorted(P.NUM_FULL + P.CAT_OHE + P.CAT_ORD),
                "registration": sorted(P.NUM_REG + P.CAT_OHE + P.CAT_ORD),
                "dropped_for_reg": sorted(set(P.NUM_FULL) - set(P.NUM_REG)),
            },
            "leak_features": P.CONDUCT_LEAK_REASONS,
            "models": leak_models,
        },
        "label_audit": label_audit,
        "label_composition": comp,
        "leak_note": (
            "Three layers of leakage. (1) Conduct-time features (results_available, log_enrollment, "
            "n_countries) are determined while the trial runs, so they encode the outcome; removing "
            "them drops AUC ~0.90 -> ~0.72. (2) The bundled success label counts RECRUITING and "
            "ACTIVE_NOT_RECRUITING trials as successes, and {:.0%} of all 'successes' are these "
            "not-yet-completed trials, concentrated in recent years (median start 2023). So calendar "
            "recency predicts the label by construction; time features alone score {:.3f}. (3) A random "
            "split lets the model see the same eras in train and test. Using the strict label "
            "(COMPLETED vs TERMINATED/WITHDRAWN/SUSPENDED) with a temporal split (train on the past, "
            "test on the future) removes both, leaving AUC ~{:.2f}."
        ).format(comp["inprogress_share_of_positives"],
                 label_audit["bundled"]["time_only_auc"], arc[2]["auc"]),
    }
    return results


def verify(results):
    rf_full = results["leakage_audit"]["models"]["Random Forest"]["full_auc"]
    rf_reg  = results["leakage_audit"]["models"]["Random Forest"]["reg_auc"]
    lr_reg  = results["leakage_audit"]["models"]["Logistic Regression"]["reg_auc"]
    gb_reg  = results["leakage_audit"]["models"]["Gradient Boosting"]["reg_auc"]
    got = {"rf_full": rf_full, "rf_reg": rf_reg, "lr_reg": lr_reg, "gb_reg": gb_reg}
    print("\n  Verification vs known legacy values (outputs/leakage_audit.json):")
    ok = True
    for k, exp in EXPECT.items():
        d = abs(got[k] - exp)
        flag = "OK " if d <= TOL else "FAIL"
        ok = ok and d <= TOL
        print(f"    {k:<10} computed={got[k]:.4f}  expected={exp:.4f}  |Δ|={d:.4f}  [{flag}]")
    return ok


def main():
    print("=" * 70)
    print("  Building artifacts/results.json from data/crispr_trials_clean.csv")
    print("=" * 70)
    results = build()

    a = results["arc"]
    print("\n  Three-number arc (Random Forest, fixed):")
    for s in a:
        print(f"    {s['step']}. {s['label']:<38} AUC = {s['auc']:.4f}  "
              f"({s['target']} label, {s['split']} split)")
    c = results["callouts"]
    print(f"\n  Callouts: {c['inprogress_share_of_positives']:.1%} of 'successes' never completed; "
          f"time-only AUC = {c['time_only_auc_bundled']:.3f} "
          f"({c['share_of_above_chance_from_time']:.0%} of the naive-clean signal is just recency).")

    ok = verify(results)

    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  {'VERIFIED' if ok else 'WARNING: numbers drifted from legacy'} — "
          f"wrote {OUT.relative_to(ROOT)}")
    if not ok:
        raise SystemExit("Numbers drifted beyond tolerance; not trusting this export.")


if __name__ == "__main__":
    main()
