"""
eda.py
------
Exploratory Data Analysis on the cleaned CRISPR cancer trial dataset.

Input:
    data/crispr_trials_clean.csv
    data/publication_trends_clean.csv

Output:
    outputs/fig1_trial_landscape.png
    outputs/fig2_temporal.png
    outputs/fig3_features.png

Usage:
    python eda.py     (run after clean_data.py)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

Path("outputs").mkdir(exist_ok=True)

df  = pd.read_csv("data/crispr_trials_clean.csv")
pub = pd.read_csv("data/publication_trends_clean.csv")

print(f"Loaded {len(df):,} clean trial records\n")

DARK_BG  = "#0a1628"; SURFACE  = "#0d1f3c"; BORDER   = "#1e3a5f"
TEXT_PRI = "#c0d8f0"; TEXT_SEC = "#8ab4d4"
CYAN  = "#00d4aa"; BLUE = "#4a9eff"; PURPLE = "#8b5cf6"
ORANGE = "#f59e0b"; RED = "#ef4444"; GREEN = "#22c55e"
PALETTE = [CYAN, BLUE, PURPLE, ORANGE, GREEN, RED,
           "#f97316","#a78bfa","#34d399","#fb7185","#60a5fa","#fbbf24"]

plt.rcParams.update({
    "figure.facecolor": DARK_BG, "axes.facecolor": SURFACE,
    "axes.edgecolor": BORDER,    "axes.labelcolor": TEXT_PRI,
    "xtick.color": TEXT_SEC,     "ytick.color": TEXT_SEC,
    "text.color": TEXT_PRI,      "grid.color": BORDER,
    "grid.alpha": 0.4,           "axes.titlesize": 11,
    "axes.labelsize": 9,
})

def save(path):
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    print(f"  Saved → {path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 - Trial Landscape
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("CRISPR Cancer Trials - Landscape  (ClinicalTrials.gov)",
             fontsize=14, color="white", y=1.01)

# 1a. Status pie
ax = axes[0, 0]
status_counts = df["overall_status"].value_counts()
status_colors = {"COMPLETED": CYAN, "RECRUITING": GREEN,
                 "ACTIVE_NOT_RECRUITING": BLUE, "TERMINATED": RED,
                 "WITHDRAWN": ORANGE, "SUSPENDED": PURPLE, "UNKNOWN": "#666"}
colors_s = [status_colors.get(s, "#888") for s in status_counts.index]
wedges, _, autotexts = ax.pie(
    status_counts.values, labels=None,
    colors=colors_s, autopct="%1.1f%%", pctdistance=0.80,
    startangle=140, wedgeprops={"edgecolor": DARK_BG, "linewidth": 1.5}
)
for at in autotexts:
    at.set_color("white"); at.set_fontsize(8)
ax.legend(wedges, [f"{s}  ({n:,})" for s, n in status_counts.items()],
          loc="lower left", fontsize=7, bbox_to_anchor=(-0.3, -0.1))
ax.set_title("Trial Status Distribution")

# 1b. Phase
ax = axes[0, 1]
phase_counts = df["phase_clean"].value_counts().head(8)
bars = ax.barh(phase_counts.index, phase_counts.values,
               color=PALETTE[:len(phase_counts)], edgecolor=BORDER, height=0.65)
for bar, val in zip(bars, phase_counts.values):
    ax.text(val + 0.5, bar.get_y() + bar.get_height()/2, str(val), va="center", fontsize=9)
ax.set_xlabel("Number of Trials"); ax.set_title("Trial Phase Distribution")
ax.grid(True, axis="x", alpha=0.3)

# 1c. Cancer type
ax = axes[1, 0]
ct_counts = df["cancer_type"].value_counts().head(15)
hem_types = {"Leukemia (AML)","Leukemia (ALL)","Leukemia (CLL/CML)",
             "Lymphoma","Multiple Myeloma","Hematologic (General)"}
bar_colors = [CYAN if t in hem_types else BLUE for t in ct_counts.index]
bars = ax.barh(ct_counts.index, ct_counts.values, color=bar_colors,
               edgecolor=BORDER, height=0.65)
for bar, val in zip(bars, ct_counts.values):
    ax.text(val + 0.3, bar.get_y() + bar.get_height()/2, str(val), va="center", fontsize=8)
ax.set_xlabel("Number of Trials"); ax.set_title("Cancer Type Distribution")
ax.tick_params(axis="y", labelsize=7.5)
ax.grid(True, axis="x", alpha=0.3)
hem_p = mpatches.Patch(color=CYAN, label="Hematologic")
sol_p = mpatches.Patch(color=BLUE, label="Solid Tumor")
ax.legend(handles=[hem_p, sol_p], fontsize=8, loc="lower right")

# 1d. Sponsor class
ax = axes[1, 1]
sp_counts = df["sponsor_class_clean"].value_counts()
ax.barh(sp_counts.index, sp_counts.values,
        color=PALETTE[:len(sp_counts)], edgecolor=BORDER, height=0.6)
for i, val in enumerate(sp_counts.values):
    ax.text(val + 0.3, i, str(val), va="center", fontsize=9)
ax.set_xlabel("Number of Trials"); ax.set_title("Lead Sponsor Class")
ax.grid(True, axis="x", alpha=0.3)

plt.tight_layout()
save("outputs/fig1_trial_landscape.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 - Temporal
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("CRISPR Cancer Research Trends Over Time", fontsize=14, color="white")

ax = axes[0]
df_year = df[df["start_year"].between(2015, 2024)].copy()
hem_by_yr = df_year[df_year["tumor_category"]=="Hematologic"].groupby("start_year").size()
sol_by_yr = df_year[df_year["tumor_category"]=="Solid Tumor"].groupby("start_year").size()
yrs = sorted(set(hem_by_yr.index) | set(sol_by_yr.index))
hem_vals = [hem_by_yr.get(y, 0) for y in yrs]
sol_vals = [sol_by_yr.get(y, 0) for y in yrs]
x = np.arange(len(yrs))
ax.bar(x, hem_vals, color=CYAN,   edgecolor=BORDER, label="Hematologic", alpha=0.85)
ax.bar(x, sol_vals, color=BLUE,   edgecolor=BORDER, label="Solid Tumor",  alpha=0.85, bottom=hem_vals)
ax.set_xticks(x); ax.set_xticklabels(yrs, rotation=45)
ax.set_xlabel("Start Year"); ax.set_ylabel("Number of Trials")
ax.set_title("New CRISPR Trials per Year")
ax.legend(fontsize=9); ax.grid(True, axis="y", alpha=0.3)

ax = axes[1]
cum_all = df_year.groupby("start_year").size().sort_index().cumsum()
cum_hem = df_year[df_year["tumor_category"]=="Hematologic"].groupby("start_year").size().sort_index().cumsum()
cum_sol = df_year[df_year["tumor_category"]=="Solid Tumor"].groupby("start_year").size().sort_index().cumsum()
ax.fill_between(cum_all.index, cum_all.values, alpha=0.12, color=ORANGE)
ax.plot(cum_all.index, cum_all.values, color=ORANGE, linewidth=2.5, marker="o", markersize=5, label="All Trials")
ax.plot(cum_hem.index, cum_hem.values, color=CYAN,   linewidth=2,   marker="s", markersize=4, label="Hematologic")
ax.plot(cum_sol.index, cum_sol.values, color=BLUE,   linewidth=2,   marker="^", markersize=4, label="Solid Tumor")
ax.set_xlabel("Year"); ax.set_ylabel("Cumulative Trials")
ax.set_title("Cumulative Trial Growth")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

ax = axes[2]
pub_cols = [c for c in pub.columns if c not in ("year","total") and not c.endswith("_yoy_pct")]
if pub_cols and len(pub) > 0:
    pub_plot = pub[pub["year"].between(2013, 2024)].sort_values("year")
    for col, color in zip(pub_cols, [CYAN, BLUE, PURPLE]):
        if col in pub_plot.columns and pub_plot[col].notna().any():
            ax.plot(pub_plot["year"], pub_plot[col], color=color, linewidth=2,
                    marker="o", markersize=4, label=col.replace("_", " ").title())
            ax.fill_between(pub_plot["year"], pub_plot[col], alpha=0.06, color=color)
    ax.set_xlabel("Year"); ax.set_ylabel("Publications")
    ax.set_title("PubMed Publications by Year")
    ax.legend(fontsize=8, loc="upper left"); ax.grid(True, alpha=0.3)
else:
    ax.text(0.5, 0.5, "PubMed data not available\nrun fetch_data.py first",
            ha="center", va="center", transform=ax.transAxes, fontsize=11, color=TEXT_SEC)
    ax.set_title("PubMed Publications")

plt.tight_layout()
save("outputs/fig2_temporal.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 - Feature Distributions
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Feature Distributions", fontsize=14, color="white")

ax = axes[0]
hem_enr = df[(df["tumor_category"]=="Hematologic") & df["enrollment_count"].notna()]["enrollment_count"]
sol_enr = df[(df["tumor_category"]=="Solid Tumor")  & df["enrollment_count"].notna()]["enrollment_count"]
bins = np.logspace(0, 4, 30)
ax.hist(hem_enr, bins=bins, color=CYAN, alpha=0.65, label=f"Hematologic (n={len(hem_enr)})")
ax.hist(sol_enr, bins=bins, color=BLUE, alpha=0.65, label=f"Solid Tumor (n={len(sol_enr)})")
ax.set_xscale("log")
ax.set_xlabel("Enrollment (log scale)"); ax.set_ylabel("Trials")
ax.set_title(f"Enrollment Distribution\nHem median={hem_enr.median():.0f}  |  Solid median={sol_enr.median():.0f}")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

ax = axes[1]
if "trial_outcome" in df.columns:
    phase_outcome = (
        df[df["trial_outcome"].notna()]
        .groupby("phase_clean")["trial_outcome"]
        .agg(["mean","count"])
        .reset_index()
        .sort_values("count", ascending=False)
        .head(8)
    )
    phase_outcome.columns = ["phase","success_rate","n"]
    colors_ph = [GREEN if r > 0.6 else ORANGE if r > 0.4 else RED
                 for r in phase_outcome["success_rate"]]
    bars = ax.barh(phase_outcome["phase"], phase_outcome["success_rate"]*100,
                   color=colors_ph, edgecolor=BORDER, height=0.65)
    for bar, row in zip(bars, phase_outcome.itertuples()):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f"{row.success_rate*100:.0f}%  (n={row.n})", va="center", fontsize=8)
    ax.axvline(50, color=TEXT_SEC, linestyle="--", linewidth=1)
    ax.set_xlabel("Trial Completion Rate (%)"); ax.set_title("Completion Rate by Phase")
    ax.set_xlim(0, 115); ax.grid(True, axis="x", alpha=0.3)

ax = axes[2]
num_cols = [c for c in ["enrollment_count","log_enrollment","duration_months",
                         "start_year","n_countries","n_primary_outcomes",
                         "trial_outcome","results_available"] if c in df.columns]
corr = df[num_cols].corr()
nice = {"enrollment_count":"Enrollment","log_enrollment":"Log Enroll.",
        "duration_months":"Duration","start_year":"Start Year",
        "n_countries":"# Countries","n_primary_outcomes":"# Outcomes",
        "trial_outcome":"Outcome","results_available":"Has Results"}
labels = [nice.get(c,c) for c in num_cols]
mask = np.triu(np.ones_like(corr), k=1).astype(bool)
cv = corr.copy(); cv[mask] = np.nan
im = ax.imshow(cv.values, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=7)
ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=7)
for i in range(len(labels)):
    for j in range(len(labels)):
        v = cv.values[i,j]
        if not np.isnan(v):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=7, color="black" if abs(v)<0.6 else "white")
plt.colorbar(im, ax=ax, label="Pearson r", fraction=0.046, pad=0.04)
ax.set_title("Feature Correlation Matrix")

plt.tight_layout()
save("outputs/fig3_features.png")

print()
print("=" * 55)
print("EDA SUMMARY")
print("=" * 55)
print(f"  Total trials:        {len(df):,}")
print(f"  Hematologic:         {(df['tumor_category']=='Hematologic').sum():,}")
print(f"  Solid Tumor:         {(df['tumor_category']=='Solid Tumor').sum():,}")
if "trial_outcome" in df.columns:
    n_labeled = df["trial_outcome"].notna().sum()
    print(f"  With outcome label:  {n_labeled:,}")
    print(f"  Completion rate:     {df['trial_outcome'].mean():.1%}")
print(f"  Median enrollment:   {df['enrollment_count'].median():.0f}")
print()
print("  Next: python statistical_tests.py")
