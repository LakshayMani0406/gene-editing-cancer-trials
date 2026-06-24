"""
statistical_tests.py
--------------------
Hypothesis testing on the real CRISPR cancer trial dataset.

Tests:
  1. Levene's test (variance homogeneity)
  2. One-Way ANOVA - enrollment by cancer type
  3. Kruskal-Wallis H - enrollment by cancer type (non-parametric)
  4. Tukey HSD post-hoc - pairwise enrollment comparisons
  5. Mann-Whitney U - hematologic vs solid enrollment
  6. Chi-Square - tumor category × trial outcome
  7. Chi-Square - phase × trial outcome
  8. Welch's t-test - enrollment: hematologic vs solid
  9. Point-biserial correlation - enrollment × trial outcome

Usage:
    python statistical_tests.py    (run after clean_data.py)
Output:
    outputs/fig4_stats.png
"""

import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from itertools import combinations
from scipy.stats import (
    levene, f_oneway, kruskal, mannwhitneyu,
    ttest_ind, chi2_contingency, pointbiserialr,
    norm, studentized_range
)
from pathlib import Path

Path("outputs").mkdir(exist_ok=True)

df = pd.read_csv("data/crispr_trials_clean.csv")
df_labeled = df[df["trial_outcome"].notna()].copy()

print(f"Loaded {len(df):,} records  |  {len(df_labeled):,} with outcome labels\n")

DARK_BG = "#0a1628"; SURFACE = "#0d1f3c"; BORDER = "#1e3a5f"
TEXT_PRI = "#c0d8f0"; TEXT_SEC = "#8ab4d4"
CYAN = "#00d4aa"; BLUE = "#4a9eff"; PURPLE = "#8b5cf6"
ORANGE = "#f59e0b"; RED = "#ef4444"; GREEN = "#22c55e"

plt.rcParams.update({
    "figure.facecolor": DARK_BG, "axes.facecolor": SURFACE,
    "axes.edgecolor": BORDER,    "axes.labelcolor": TEXT_PRI,
    "xtick.color": TEXT_SEC,     "ytick.color": TEXT_SEC,
    "text.color": TEXT_PRI,      "grid.color": BORDER,
})

def divider(title):
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)

def result_line(reject, alpha=0.05):
    tag = "✓ REJECT H₀  - SIGNIFICANT" if reject else "✗ FAIL TO REJECT H₀  - not significant"
    return f"  → {tag}  (α={alpha})"

def tukey_hsd(groups_dict, alpha=0.05):
    k = len(groups_dict)
    names = list(groups_dict.keys())
    ms_within = sum(((v - v.mean())**2).sum() for v in groups_dict.values())
    df_within  = sum(len(v) for v in groups_dict.values()) - k
    ms_within /= df_within
    records = []
    for n1, n2 in combinations(names, 2):
        g1, g2 = groups_dict[n1], groups_dict[n2]
        diff = g1.mean() - g2.mean()
        hn   = 2 / (1/len(g1) + 1/len(g2))
        q    = abs(diff) / (np.sqrt(ms_within) / np.sqrt(hn/2))
        try:
            p = float(1 - studentized_range.cdf(q, k, df_within))
        except Exception:
            p = 1.0
        records.append({"group1": n1, "group2": n2,
                         "meandiff": round(diff,2), "p_adj": round(p,5),
                         "reject": p < alpha})
    return pd.DataFrame(records)


# Use enrollment as the continuous outcome variable (log-transformed)
df_enr = df[df["enrollment_count"].notna() & (df["enrollment_count"] > 0)].copy()
df_enr["log_enrollment"] = np.log1p(df_enr["enrollment_count"])

# Group by cancer type - use min n>=2, fall back to tumor_category if too few groups
ct_counts = df_enr["cancer_type"].value_counts()
top_cancers = ct_counts[ct_counts >= 2].index.tolist()
groups = {ct: df_enr[df_enr["cancer_type"]==ct]["log_enrollment"].values
          for ct in top_cancers}

# Need at least 2 groups for ANOVA - fall back to tumor_category if needed
if len(groups) < 2:
    groups = {cat: df_enr[df_enr["tumor_category"]==cat]["log_enrollment"].values
              for cat in df_enr["tumor_category"].unique()
              if len(df_enr[df_enr["tumor_category"]==cat]) >= 2}
    print("  Note: Using tumor_category groups (too few samples per cancer type for ANOVA)")
hem_types = {"Leukemia (AML)","Leukemia (ALL)","Leukemia (CLL/CML)",
             "Lymphoma","Multiple Myeloma","Hematologic (General)"}
hem_enr = df_enr[df_enr["tumor_category"]=="Hematologic"]["log_enrollment"].values
sol_enr = df_enr[df_enr["tumor_category"]=="Solid Tumor"]["log_enrollment"].values

# ─────────────────────────────────────────────────────────────────────────────
# 1. Levene's Test
# ─────────────────────────────────────────────────────────────────────────────
divider("1. LEVENE'S TEST - Variance homogeneity of enrollment by cancer type")
if len(groups) >= 2:
    lev_stat, lev_p = levene(*groups.values())
    print(f"  W = {lev_stat:.4f},  p = {lev_p:.6f}")
    print(result_line(lev_p < 0.05))
    if lev_p < 0.05:
        print("  → Variances are heterogeneous - Welch's ANOVA / Kruskal-Wallis preferred.")
    else:
        print("  → Variances are homogeneous - standard ANOVA appropriate.")

# ─────────────────────────────────────────────────────────────────────────────
# 2. One-Way ANOVA
# ─────────────────────────────────────────────────────────────────────────────
divider("2. ONE-WAY ANOVA - Log-enrollment across cancer types")
f_stat, p_anova = f_oneway(*groups.values())
k   = len(groups)
N   = sum(len(v) for v in groups.values())
df_b, df_w = k-1, N-k
grand_mean = np.concatenate(list(groups.values())).mean()
ss_b = sum(len(v)*(v.mean()-grand_mean)**2 for v in groups.values())
ss_t = sum(((x-grand_mean)**2) for v in groups.values() for x in v)
eta_sq = ss_b / ss_t
print(f"  F({df_b}, {df_w}) = {f_stat:.4f},  p = {p_anova:.6f}")
print(f"  η² = {eta_sq:.4f}  ({'large' if eta_sq > 0.14 else 'medium' if eta_sq > 0.06 else 'small'} effect)")
print(result_line(p_anova < 0.05))

# ─────────────────────────────────────────────────────────────────────────────
# 3. Kruskal-Wallis
# ─────────────────────────────────────────────────────────────────────────────
divider("3. KRUSKAL-WALLIS H TEST - Non-parametric alternative")
h_stat, p_kruskal = kruskal(*groups.values())
print(f"  H({df_b}) = {h_stat:.4f},  p = {p_kruskal:.6f}")
print(result_line(p_kruskal < 0.05))

# ─────────────────────────────────────────────────────────────────────────────
# 4. Tukey HSD
# ─────────────────────────────────────────────────────────────────────────────
divider("4. TUKEY HSD POST-HOC - Pairwise log-enrollment comparisons")
tukey_df = tukey_hsd(groups)
sig_pairs = tukey_df[tukey_df["reject"]].sort_values("p_adj")
print(f"  Total pairwise comparisons: {len(tukey_df)}")
print(f"  Significant (α=0.05):       {len(sig_pairs)}")
if len(sig_pairs):
    print(f"\n  {'Group 1':<28} {'Group 2':<28} {'Δ log-enroll':>12} {'p-adj':>8}")
    print(f"  {'-'*28} {'-'*28} {'-'*12} {'-'*8}")
    for _, r in sig_pairs.head(15).iterrows():
        print(f"  {r['group1']:<28} {r['group2']:<28} {r['meandiff']:>+12.3f} {r['p_adj']:>8.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Mann-Whitney U - Hematologic vs Solid
# ─────────────────────────────────────────────────────────────────────────────
divider("5. MANN-WHITNEY U - Enrollment: Hematologic vs Solid Tumor")
u_stat, p_mwu = mannwhitneyu(hem_enr, sol_enr, alternative="two-sided")
z_val = norm.ppf(min(p_mwu/2, 1-1e-10))
r_eff = abs(z_val) / np.sqrt(len(hem_enr)+len(sol_enr))
print(f"  U = {u_stat:.1f},  p = {p_mwu:.6f}")
print(f"  Effect size r = {r_eff:.4f}  ({'large' if r_eff>0.5 else 'medium' if r_eff>0.3 else 'small'})")
print(f"  Hematologic: mean log-enroll={hem_enr.mean():.3f}  (raw median={np.expm1(np.median(hem_enr)):.0f})")
print(f"  Solid Tumor: mean log-enroll={sol_enr.mean():.3f}  (raw median={np.expm1(np.median(sol_enr)):.0f})")
print(result_line(p_mwu < 0.05))

# ─────────────────────────────────────────────────────────────────────────────
# 6. Chi-Square - Tumor category × Trial outcome
# ─────────────────────────────────────────────────────────────────────────────
divider("6. CHI-SQUARE - Tumor Category × Trial Outcome")
ct1 = pd.crosstab(df_labeled["tumor_category"], df_labeled["trial_outcome"])
print("  Observed (rows=category, cols=0=failure/1=success):")
print(ct1.to_string()); print()
chi2_1, p_chi2_1, dof1, _ = chi2_contingency(ct1)
v1 = np.sqrt(chi2_1 / (ct1.values.sum() * (min(ct1.shape)-1)))
print(f"  χ²({dof1}) = {chi2_1:.4f},  p = {p_chi2_1:.6f},  Cramér's V = {v1:.4f}")
print(result_line(p_chi2_1 < 0.05))

# ─────────────────────────────────────────────────────────────────────────────
# 7. Chi-Square - Phase × Trial outcome
# ─────────────────────────────────────────────────────────────────────────────
divider("7. CHI-SQUARE - Trial Phase × Trial Outcome")
ct2 = pd.crosstab(df_labeled["phase_clean"], df_labeled["trial_outcome"])
ct2_filtered = ct2[(ct2.sum(axis=1) >= 5)]   # only phases with >=5 trials
chi2_2, p_chi2_2, dof2, _ = chi2_contingency(ct2_filtered)
v2 = np.sqrt(chi2_2 / (ct2_filtered.values.sum() * (min(ct2_filtered.shape)-1)))
print(f"  χ²({dof2}) = {chi2_2:.4f},  p = {p_chi2_2:.6f},  Cramér's V = {v2:.4f}")
print(result_line(p_chi2_2 < 0.05))

# ─────────────────────────────────────────────────────────────────────────────
# 8. Welch's t-test
# ─────────────────────────────────────────────────────────────────────────────
divider("8. WELCH'S t-TEST - Enrollment: Hematologic vs Solid Tumor")
t_stat, p_ttest = ttest_ind(hem_enr, sol_enr, equal_var=False)
d = (hem_enr.mean()-sol_enr.mean()) / np.sqrt((hem_enr.std()**2+sol_enr.std()**2)/2)
print(f"  t = {t_stat:.4f},  p = {p_ttest:.6f},  Cohen's d = {d:.4f}")
print(result_line(p_ttest < 0.05))

# ─────────────────────────────────────────────────────────────────────────────
# 9. Point-biserial correlation
# ─────────────────────────────────────────────────────────────────────────────
divider("9. POINT-BISERIAL CORRELATION - Enrollment × Trial Outcome")
df_both = df_labeled[df_labeled["enrollment_count"].notna()].copy()
df_both["log_enrollment"] = np.log1p(df_both["enrollment_count"])
if len(df_both) > 10:
    rpb, p_rpb = pointbiserialr(df_both["trial_outcome"], df_both["log_enrollment"])
    print(f"  r_pb = {rpb:.4f},  p = {p_rpb:.6f}")
    print(result_line(p_rpb < 0.05))
    if rpb > 0:
        print("  → Larger trials (higher enrollment) tend to complete more often.")

# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  RESULTS SUMMARY")
print("=" * 65)
results_table = [
    ("Levene's Test",        lev_p,    "Enrollment variance homogeneity"),
    ("One-Way ANOVA",        p_anova,  f"Enrollment differs across cancer types  [η²={eta_sq:.3f}]"),
    ("Kruskal-Wallis",       p_kruskal,"Non-parametric enrollment comparison"),
    ("Mann-Whitney U",       p_mwu,    f"Hem vs Solid enrollment  [r={r_eff:.3f}]"),
    ("Welch's t-Test",       p_ttest,  f"Mean enrollment difference  [d={d:.3f}]"),
    ("Chi-Square (Cat×Out)", p_chi2_1, f"Category predicts outcome  [V={v1:.3f}]"),
    ("Chi-Square (Ph×Out)",  p_chi2_2, f"Phase predicts outcome  [V={v2:.3f}]"),
]
if len(df_both) > 10:
    results_table.append(("Point-Biserial Corr.", p_rpb, f"Enrollment × Outcome  [r={rpb:.3f}]"))

print(f"  {'Test':<26} {'p-value':>12}  {'Sig':>5}  Note")
print(f"  {'-'*26} {'-'*12}  {'-'*5}  {'-'*40}")
for name, pval, note in results_table:
    sig = "✓" if pval < 0.05 else "✗"
    print(f"  {name:<26} {pval:>12.6f}  {sig:>5}  {note}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Statistical Hypothesis Testing - Visual Summary", fontsize=14, color="white")

# 4a. Box plots: enrollment by cancer type
ax = axes[0, 0]
sorted_cancers = sorted(groups.keys(), key=lambda c: np.median(groups[c]), reverse=True)
data_plot = [groups[c] for c in sorted_cancers]
bp = ax.boxplot(data_plot, tick_labels=sorted_cancers, vert=False, patch_artist=True)
for patch, ct in zip(bp["boxes"], sorted_cancers):
    patch.set_facecolor(CYAN if ct in hem_types else BLUE); patch.set_alpha(0.7)
for m in bp["medians"]: m.set_color("white"); m.set_linewidth(2)
ax.set_xlabel("Log Enrollment"); ax.set_title(f"Enrollment by Cancer Type\nANOVA F={f_stat:.1f}, p={p_anova:.4f}, η²={eta_sq:.3f}")
ax.tick_params(axis="y", labelsize=7); ax.grid(True, axis="x", alpha=0.3)

# 4b. Outcome rate by tumor category
ax = axes[0, 1]
if len(df_labeled) > 0:
    cat_outcome = df_labeled.groupby("tumor_category")["trial_outcome"].agg(["mean","count"]).reset_index()
    cat_outcome.columns = ["category","success_rate","n"]
    colors_co = [CYAN if c=="Hematologic" else BLUE for c in cat_outcome["category"]]
    bars = ax.bar(cat_outcome["category"], cat_outcome["success_rate"]*100,
                  color=colors_co, edgecolor=BORDER, width=0.5)
    for bar, row in zip(bars, cat_outcome.itertuples()):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                f"{row.success_rate*100:.1f}%\n(n={row.n})", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("Completion Rate (%)"); ax.set_title(f"Completion by Category\nχ²={chi2_1:.2f}, p={p_chi2_1:.4f}, V={v1:.3f}")
    ax.set_ylim(0, 115); ax.grid(True, axis="y", alpha=0.3)

# 4c. p-value summary
ax = axes[1, 0]
test_names = [r[0] for r in results_table]
p_vals     = [r[1] for r in results_table]
bar_cols   = [GREEN if p < 0.05 else RED for p in p_vals]
log_ps     = [-np.log10(max(p, 1e-16)) for p in p_vals]
ax.barh(test_names, log_ps, color=bar_cols, edgecolor=BORDER, height=0.55, alpha=0.85)
ax.axvline(-np.log10(0.05), color=ORANGE, linestyle="--", linewidth=2, label="α=0.05 threshold")
for i, pv in enumerate(p_vals):
    ax.text(log_ps[i]+0.05, i, f"p={pv:.3f}" if pv >= 0.001 else f"p={pv:.2e}",
            va="center", fontsize=7.5)
ax.set_xlabel("−log₁₀(p-value)"); ax.set_title("Statistical Significance by Test")
ax.legend(fontsize=9); ax.grid(True, axis="x", alpha=0.3)
sig_p = mpatches.Patch(color=GREEN, alpha=0.85, label="Significant")
ns_p  = mpatches.Patch(color=RED,   alpha=0.85, label="Not significant")
ax.legend(handles=[sig_p, ns_p,
    mpatches.Patch(color=ORANGE, label="α=0.05")], fontsize=8)

# 4d. Contingency heatmap
ax = axes[1, 1]
if len(ct1) > 0:
    cv_vals = ct1.values
    row_labs = list(ct1.index)
    col_labs = ["Failure (0)", "Success (1)"] if 0 in ct1.columns and 1 in ct1.columns \
               else [str(c) for c in ct1.columns]
    im = ax.imshow(cv_vals, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(col_labs))); ax.set_xticklabels(col_labs)
    ax.set_yticks(range(len(row_labs))); ax.set_yticklabels(row_labs)
    for i in range(len(row_labs)):
        for j in range(len(col_labs)):
            total = cv_vals[i,:].sum()
            pct = cv_vals[i,j]/total*100 if total > 0 else 0
            ax.text(j, i, f"{cv_vals[i,j]}\n({pct:.0f}%)",
                    ha="center", va="center", fontsize=9, fontweight="bold",
                    color="white" if cv_vals[i,j] > cv_vals.max()*0.55 else TEXT_PRI)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"Category × Outcome\nχ²={chi2_1:.2f}, V={v1:.3f}")

plt.tight_layout()
fig.savefig("outputs/fig4_stats.png", dpi=150, bbox_inches="tight", facecolor=DARK_BG)
print(f"\n  Saved → outputs/fig4_stats.png")
plt.close()
print("  Next: python ml_models.py")
