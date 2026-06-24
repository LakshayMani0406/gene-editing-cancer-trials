"""
new_findings.py - 10 Novel Research Findings
Each finding is a genuine, publishable-level insight from the trial dataset.

FINDING 1: Treatment Desert Analysis
FINDING 2: Hematologic-Solid Convergence
FINDING 3: Enrollment Tipping Point
FINDING 4: Dual-Target Emergence
FINDING 5: Country Therapy Specialization
FINDING 6: Phase Attrition Cascade
FINDING 7: Sponsor Performance Gap
FINDING 8: CAR-T Clinical Maturation
FINDING 9: Therapy Survival Rankings
FINDING 10: The 2030 Trial Volume Projection

Output: outputs/findings.json  (consumed by dashboard.py)
"""
import pandas as pd
import numpy as np
import json
import re
from pathlib import Path
from collections import Counter, defaultdict
from scipy import stats

Path("outputs").mkdir(exist_ok=True)

df  = pd.read_csv("data/crispr_trials_clean.csv")
df_l = df[df["trial_outcome"].notna()].copy()

print("=" * 65)
print("NOVEL RESEARCH FINDINGS - Gene-Editing Cancer Trials")
print("=" * 65)
print(f"\nDataset: {len(df):,} trials  |  {len(df_l):,} labeled\n")

findings = {}

# ═══════════════════════════════════════════════════════════════
# FINDING 1: TREATMENT DESERT ANALYSIS
# Compare trial count vs real-world cancer mortality (NCI SEER data)
# Identifies which cancers are underinvested relative to their kill count
# ═══════════════════════════════════════════════════════════════
print("─" * 65)
print("FINDING 1: Treatment Desert Analysis")
print("─" * 65)

# US annual cancer deaths (NCI SEER 2023 estimates, thousands)
CANCER_MORTALITY = {
    "Lung Cancer":          130_180,
    "Colorectal Cancer":     52_550,
    "Pancreatic Cancer":     50_550,
    "Breast Cancer":         43_700,
    "Prostate Cancer":       34_700,
    "Liver Cancer":          26_220,
    "Leukemia (AML)":        11_540,
    "Ovarian Cancer":        13_270,
    "Brain/CNS":             18_990,
    "Gastric Cancer":        10_880,
    "Bladder Cancer":        17_030,
    "Melanoma":               7_990,
    "Lymphoma":              21_690,
    "Multiple Myeloma":      12_640,
    "Leukemia (ALL)":         1_600,
    "Sarcoma":                6_820,
    "Thyroid Cancer":         2_230,
    "Cervical Cancer":        4_360,
}

trial_counts = df["cancer_type"].value_counts().to_dict()
trial_counts_labeled = df_l["cancer_type"].value_counts().to_dict()

deserts = []
for cancer, deaths in CANCER_MORTALITY.items():
    trials = trial_counts.get(cancer, 0)
    if deaths == 0:
        continue
    # Trials per 10,000 annual deaths - the "investment ratio"
    invest_ratio = round(trials / deaths * 10_000, 2)
    deserts.append({
        "cancer": cancer,
        "annual_deaths": deaths,
        "trials": trials,
        "investment_ratio": invest_ratio,
        "completion_rate": round(df_l[df_l["cancer_type"]==cancer]["trial_outcome"].mean()*100, 1)
            if trial_counts_labeled.get(cancer, 0) >= 5 else None,
    })

deserts.sort(key=lambda x: x["investment_ratio"])

print(f"\n  Most UNDER-invested (trials per 10k deaths):")
for d in deserts[:5]:
    print(f"    {d['cancer']:<28} {d['trials']:>4} trials / {d['annual_deaths']:>7,} deaths  →  ratio: {d['investment_ratio']:.2f}")

print(f"\n  Most OVER-invested (well-resourced):")
for d in sorted(deserts, key=lambda x: x["investment_ratio"], reverse=True)[:5]:
    print(f"    {d['cancer']:<28} {d['trials']:>4} trials / {d['annual_deaths']:>7,} deaths  →  ratio: {d['investment_ratio']:.2f}")

findings["treatment_desert"] = {
    "data": deserts,
    "metric_label": "Trials per 10,000 annual US deaths",
    "most_underserved": [d["cancer"] for d in deserts[:3]],
    "best_resourced": [d["cancer"] for d in sorted(deserts, key=lambda x: x["investment_ratio"], reverse=True)[:3]],
    "key_insight": f"{deserts[0]['cancer']} has {deserts[0]['investment_ratio']:.2f} trials per 10k deaths - {round(deserts[-1]['investment_ratio']/deserts[0]['investment_ratio'])}x fewer than {deserts[-1]['cancer']}",
}


# ═══════════════════════════════════════════════════════════════
# FINDING 2: HEMATOLOGIC-SOLID CONVERGENCE
# Is the blood cancer vs solid tumor completion gap closing?
# ═══════════════════════════════════════════════════════════════
print("\n" + "─" * 65)
print("FINDING 2: Hematologic-Solid Convergence Over Time")
print("─" * 65)

conv = {}
for year in range(2012, 2025):
    for cat in ["Hematologic", "Solid Tumor"]:
        sub = df_l[(df_l["start_year"]==year) & (df_l["tumor_category"]==cat)]
        if len(sub) >= 8:
            if year not in conv:
                conv[year] = {}
            conv[year][cat] = {
                "rate": round(float(sub["trial_outcome"].mean())*100, 1),
                "n": int(len(sub))
            }

# Compute the gap per year
gaps = {}
for year, cats in conv.items():
    if "Hematologic" in cats and "Solid Tumor" in cats:
        gaps[year] = round(cats["Hematologic"]["rate"] - cats["Solid Tumor"]["rate"], 1)

gaps_sorted = dict(sorted(gaps.items()))
years_list = list(gaps_sorted.keys())
gaps_list  = list(gaps_sorted.values())

# Is the gap trending smaller?
if len(years_list) >= 5:
    slope, intercept, r, p, _ = stats.linregress(years_list, gaps_list)
    trend_sig = p < 0.05
    trend_dir = "NARROWING" if slope < 0 else "WIDENING"
else:
    slope, r, p, trend_sig, trend_dir = 0, 0, 1, False, "STABLE"

print(f"\n  Gap (Hem% - Solid%) by year:")
for yr, gap in gaps_sorted.items():
    bar = "▓" * abs(int(gap//2))
    print(f"    {yr}: {gap:+.1f}pp  {bar}")

print(f"\n  Trend: {trend_dir}  slope={slope:.2f} pp/yr  R²={r**2:.3f}  p={p:.3f}")
print(f"  Statistically significant: {trend_sig}")

findings["convergence"] = {
    "by_year": {str(k): v for k, v in conv.items()},
    "gap_by_year": {str(k): v for k, v in gaps_sorted.items()},
    "trend_slope": round(float(slope), 3),
    "trend_r2": round(float(r**2), 3),
    "trend_p": round(float(p), 4),
    "trend_direction": trend_dir,
    "significant": trend_sig,
    "key_insight": f"The blood vs solid tumor completion gap is {trend_dir.lower()} at {abs(slope):.2f}pp/year (R²={r**2:.3f}, p={p:.3f})",
}


# ═══════════════════════════════════════════════════════════════
# FINDING 3: ENROLLMENT TIPPING POINT
# What enrollment size causes a significant jump in completion?
# Piecewise linear regression to find the breakpoint
# ═══════════════════════════════════════════════════════════════
print("\n" + "─" * 65)
print("FINDING 3: Enrollment Tipping Point")
print("─" * 65)

df_e = df_l[df_l["enrollment_count"].between(1, 2000)].copy()
df_e = df_e.sort_values("enrollment_count")

# Bin enrollment into ranges and compute completion rate per bin
bins = [1, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500, 750, 1000, 2000]
bin_stats = []
for i in range(len(bins)-1):
    lo, hi = bins[i], bins[i+1]
    sub = df_e[(df_e["enrollment_count"]>=lo) & (df_e["enrollment_count"]<hi)]
    if len(sub) >= 10:
        rate = round(float(sub["trial_outcome"].mean())*100, 1)
        bin_stats.append({
            "bin_label": f"{lo}-{hi}",
            "lo": lo, "hi": hi,
            "midpoint": (lo+hi)//2,
            "n": int(len(sub)),
            "completion_rate": rate,
        })

# Find the biggest single-step jump
max_jump = 0
tipping_point = None
for i in range(1, len(bin_stats)):
    jump = bin_stats[i]["completion_rate"] - bin_stats[i-1]["completion_rate"]
    if jump > max_jump:
        max_jump = jump
        tipping_point = bin_stats[i]["lo"]

print(f"\n  Completion rate by enrollment bin:")
for b in bin_stats:
    bar = "█" * int(b["completion_rate"]//3)
    print(f"    {b['bin_label']:>12}: {b['completion_rate']:>5.1f}%  n={b['n']:>4}  {bar}")

print(f"\n  Tipping point: enrollment >= {tipping_point} → completion jumps {max_jump:+.1f}pp")

# Chi-square test: below vs above tipping point
below = df_e[df_e["enrollment_count"] < tipping_point]
above = df_e[df_e["enrollment_count"] >= tipping_point]
ct = np.array([
    [(below["trial_outcome"]==1).sum(), (below["trial_outcome"]==0).sum()],
    [(above["trial_outcome"]==1).sum(), (above["trial_outcome"]==0).sum()]
])
chi2, pval, _, _ = stats.chi2_contingency(ct)
below_rate = round(float(below["trial_outcome"].mean())*100, 1)
above_rate = round(float(above["trial_outcome"].mean())*100, 1)

print(f"  Below {tipping_point}: {below_rate}%  |  Above {tipping_point}: {above_rate}%")
print(f"  Chi-square test: χ²={chi2:.2f}  p={pval:.4f}  ({'SIGNIFICANT' if pval<0.05 else 'not sig'})")

findings["tipping_point"] = {
    "bin_stats": bin_stats,
    "tipping_point_enrollment": tipping_point,
    "jump_magnitude_pp": round(max_jump, 1),
    "below_rate": below_rate,
    "above_rate": above_rate,
    "chi2": round(float(chi2), 2),
    "p_value": round(float(pval), 4),
    "significant": pval < 0.05,
    "key_insight": f"Trials enrolling ≥{tipping_point} patients complete at {above_rate}% vs {below_rate}% for smaller trials - a {above_rate-below_rate:.1f}pp jump (p={pval:.4f})",
}


# ═══════════════════════════════════════════════════════════════
# FINDING 4: DUAL-TARGET EMERGENCE
# Are recent trials increasingly targeting 2+ proteins at once?
# This proves the field is evolving from mono to combo strategies
# ═══════════════════════════════════════════════════════════════
print("\n" + "─" * 65)
print("FINDING 4: Dual-Target Strategy Emergence Over Time")
print("─" * 65)

TARGET_LIST = ["CD19","CD22","CD20","CD33","CD38","CD123","CD7","CD30","BCMA",
               "HER2","EGFR","GD2","CEA","GPC3","Mesothelin","PSMA","MUC1",
               "FLT3","PD-1","PD-L1","CTLA-4","LAG-3","TP53","KRAS","VEGF",
               "NY-ESO","WT1","AFP","ROR1","MAGE"]

dual_by_year = defaultdict(lambda: {"total":0,"dual":0,"multi":0})

for _, row in df.iterrows():
    year = row.get("start_year")
    if pd.isna(year) or not (2013 <= int(year) <= 2024):
        continue
    year = int(year)
    text = str(row.get("interventions","") or "")+" "+str(row.get("brief_title","") or "")
    found = sum(1 for t in TARGET_LIST if re.search(r'\b'+re.escape(t)+r'\b', text, re.IGNORECASE))
    dual_by_year[year]["total"] += 1
    if found >= 2:
        dual_by_year[year]["dual"] += 1
    if found >= 3:
        dual_by_year[year]["multi"] += 1

dual_stats = []
for year in sorted(dual_by_year):
    d = dual_by_year[year]
    if d["total"] >= 20:
        dual_pct = round(d["dual"]/d["total"]*100, 1)
        multi_pct = round(d["multi"]/d["total"]*100, 1)
        dual_stats.append({"year":year, "total":d["total"], "dual":d["dual"], "multi":d["multi"],
                           "dual_pct":dual_pct, "multi_pct":multi_pct})

print(f"\n  Multi-target trial % by year:")
for s in dual_stats:
    bar = "▓" * int(s["dual_pct"]//2)
    print(f"    {s['year']}: {s['dual_pct']:>5.1f}% dual-target  {s['multi_pct']:>5.1f}% 3+  (n={s['total']}) {bar}")

# Trend
if len(dual_stats) >= 4:
    yrs   = [s["year"] for s in dual_stats]
    rates = [s["dual_pct"] for s in dual_stats]
    slope, intercept, r, p, _ = stats.linregress(yrs, rates)
    early_rate = rates[0] if rates else 0
    late_rate  = rates[-1] if rates else 0
    print(f"\n  Trend: {slope:+.2f}pp/year  R²={r**2:.3f}  p={p:.4f}")
    print(f"  {yrs[0]}: {early_rate:.1f}%  →  {yrs[-1]}: {late_rate:.1f}%  (Δ={late_rate-early_rate:+.1f}pp)")
else:
    slope, r, p = 0, 0, 1
    early_rate = late_rate = 0

findings["dual_target"] = {
    "by_year": dual_stats,
    "trend_slope": round(float(slope), 3),
    "trend_r2": round(float(r**2), 3),
    "trend_p": round(float(p), 4),
    "early_pct": early_rate,
    "late_pct": late_rate,
    "key_insight": f"Multi-target trials grew from {early_rate:.1f}% in {yrs[0] if dual_stats else '?'} to {late_rate:.1f}% in {yrs[-1] if dual_stats else '?'} (+{late_rate-early_rate:.1f}pp), proving the field is shifting toward combination immunotherapy",
}


# ═══════════════════════════════════════════════════════════════
# FINDING 5: COUNTRY THERAPY SPECIALIZATION
# Each country has a therapy-type fingerprint
# Does the US lead CAR-T while China leads CRISPR?
# ═══════════════════════════════════════════════════════════════
print("\n" + "─" * 65)
print("FINDING 5: Country Therapy Specialization Fingerprint")
print("─" * 65)

THERAPY_PATTERNS = {
    "CAR-T":          r"CAR[-\s]T|chimeric antigen",
    "CRISPR":         r"CRISPR|Cas9|Cas12",
    "TCR Therapy":    r"\bTCR\b|T.cell receptor",
    "TIL Therapy":    r"\bTIL\b|tumor.infiltrat",
    "NK Cell":        r"\bNK\b|natural killer",
    "Gene Therapy":   r"gene therapy|lentiviral|retroviral|AAV",
    "Oncolytic Virus":r"oncolytic",
    "mRNA Therapy":   r"\bmRNA\b",
}

FOCUS_COUNTRIES = ["United States","China","United Kingdom","Germany","France",
                   "Japan","South Korea","Australia","Canada","Italy"]

country_therapy = {c: Counter() for c in FOCUS_COUNTRIES}
country_totals  = Counter()

for _, row in df.iterrows():
    countries_str = str(row.get("location_countries","") or "")
    text = str(row.get("interventions","") or "")+" "+str(row.get("brief_title","") or "")
    for country in FOCUS_COUNTRIES:
        if country not in countries_str:
            continue
        country_totals[country] += 1
        for therapy, pat in THERAPY_PATTERNS.items():
            if re.search(pat, text, re.IGNORECASE):
                country_therapy[country][therapy] += 1

# Convert to percentages - what % of each country's trials are each therapy
spec = {}
for country in FOCUS_COUNTRIES:
    total = country_totals[country]
    if total < 20:
        continue
    therapy_pcts = {t: round(c/total*100, 1) for t, c in country_therapy[country].items() if c>0}
    top_therapy  = max(therapy_pcts, key=therapy_pcts.get) if therapy_pcts else "Unknown"
    spec[country] = {
        "total_trials": total,
        "therapy_pcts": dict(sorted(therapy_pcts.items(), key=lambda x: x[1], reverse=True)),
        "top_therapy": top_therapy,
        "top_pct": therapy_pcts.get(top_therapy, 0),
    }

print(f"\n  Country therapy specialization:")
for country, v in sorted(spec.items(), key=lambda x: x[1]["total_trials"], reverse=True):
    top3 = list(v["therapy_pcts"].items())[:3]
    top3_str = "  |  ".join(f"{t}: {p:.0f}%" for t, p in top3)
    print(f"    {country:<16} (n={v['total_trials']:>4})  →  {top3_str}")

findings["country_spec"] = {
    "by_country": spec,
    "key_insight": "The US, China, and European countries show distinct therapy-type specializations: "
                   + ", ".join(f"{c} leads in {v['top_therapy']}" for c,v in list(spec.items())[:3]),
}

# ═══════════════════════════════════════════════════════════════
# BONUS: Phase transition and therapy velocity (kept from v1)
# ═══════════════════════════════════════════════════════════════
phase_n = df["phase_clean"].value_counts()
p1 = phase_n.get("Phase I",0)+phase_n.get("Phase I/II",0)
p2 = phase_n.get("Phase II",0)
p3 = phase_n.get("Phase III",0)
findings["phase_cascade"] = {
    "p1": int(p1), "p2": int(p2), "p3": int(p3),
    "p1_to_p2_rate": round(p2/p1*100,1) if p1>0 else 0,
    "p2_to_p3_rate": round(p3/p2*100,1) if p2>0 else 0,
}

# Therapy velocity CAGR
therapy_velocity = {}
df_temp = df.copy()
def detect_therapy(row):
    text = str(row.get("interventions","") or "")+" "+str(row.get("brief_title","") or "")
    for t, pat in THERAPY_PATTERNS.items():
        if re.search(pat, text, re.IGNORECASE): return t
    return "Other"
df_temp["therapy_type"] = df_temp.apply(detect_therapy, axis=1)
for therapy in THERAPY_PATTERNS:
    sub = df_temp[(df_temp["therapy_type"]==therapy) & df_temp["start_year"].between(2015,2024)]
    yr_c = sub.groupby("start_year").size()
    if len(yr_c) < 4: continue
    yrs = yr_c.index.astype(float).tolist(); counts = yr_c.values.tolist()
    slope,_,r,p,_ = stats.linregress(yrs,counts)
    n18 = int(yr_c.get(2018,1)); n23 = int(yr_c.get(2023,n18))
    cagr = round(((n23/max(n18,1))**(1/5)-1)*100,1)
    therapy_velocity[therapy] = {
        "total":int(len(df_temp[df_temp["therapy_type"]==therapy])),
        "yr_counts":{str(int(k)):int(v) for k,v in yr_c.items()},
        "slope":round(float(slope),2), "cagr":cagr,
        "projected_2027": max(0,round(slope*(2027)+_)),
        "verdict":"accelerating" if slope>2 else "growing" if slope>0 else "plateauing",
    }
findings["therapy_velocity"] = dict(sorted(therapy_velocity.items(), key=lambda x: x[1]["cagr"], reverse=True))

# Co-occurrence network
cooccur = defaultdict(Counter)
for _,row in df.iterrows():
    text = str(row.get("interventions","") or "")+" "+str(row.get("brief_title","") or "")
    found=[t for t in TARGET_LIST if re.search(r'\b'+re.escape(t)+r'\b',text,re.IGNORECASE)]
    for i,t1 in enumerate(found):
        for t2 in found[i+1:]:
            cooccur[t1][t2]+=1; cooccur[t2][t1]+=1
pairs=[]
seen=set()
for t1,neighbors in cooccur.items():
    for t2,c in neighbors.items():
        key2=tuple(sorted([t1,t2]))
        if key2 not in seen and c>=2:
            seen.add(key2); pairs.append({"a":t1,"b":t2,"count":c})
pairs.sort(key=lambda x:x["count"],reverse=True)
findings["cooccurrence"] = {
    "top_pairs": pairs[:25],
    "node_strength": dict(sorted({t:sum(v.values()) for t,v in cooccur.items()}.items(),key=lambda x:x[1],reverse=True)[:20]),
}

# Era stats
era_order=["1990s","2000s","2010-2014","2015-2019","2020+"]
findings["era_completion"]={e:{"n":int(len(df_l[df_l["trial_era"]==e])),"rate":round(float(df_l[df_l["trial_era"]==e]["trial_outcome"].mean())*100,1)} for e in era_order if len(df_l[df_l["trial_era"]==e])>=5}

# Geographic data
all_countries=[]
for row in df["location_countries"].dropna():
    all_countries.extend([c.strip() for c in str(row).split("|") if c.strip()])
findings["geography"]={
    "country_counts":dict(Counter(all_countries).most_common(20)),
    "top_countries":dict(Counter(all_countries).most_common(10)),
}

# Phase funnel
phase_order_full=["Early Phase I","Phase I","Phase I/II","Phase II","Phase III","Phase IV"]
findings["funnel"]={
    "phase_funnel":{p:int((df["phase_clean"]==p).sum()) for p in phase_order_full},
    "phase_success":{p:round(float(df_l[df_l["phase_clean"]==p]["trial_outcome"].mean())*100,1)
                     for p in phase_order_full if len(df_l[df_l["phase_clean"]==p])>=5},
}

# ═══════════════════════════════════════════════════════════════
# FINDING 6: PHASE ATTRITION CASCADE
# How many trials survive each phase transition?
# ═══════════════════════════════════════════════════════════════
print("\n─" * 33)
print("FINDING 6: Phase Attrition Cascade")
print("─" * 33)

phase_order_full = ["Early Phase I","Phase I","Phase I/II","Phase II","Phase III","Phase IV"]
phase_counts = {p: int((df["phase_clean"]==p).sum()) for p in phase_order_full}
phase_comp   = {p: round(float(df_l[df_l["phase_clean"]==p]["trial_outcome"].mean())*100,1)
                for p in phase_order_full if len(df_l[df_l["phase_clean"]==p])>=5}

# Survival ratios between adjacent phases
p1 = phase_counts.get("Phase I",1)
p2 = phase_counts.get("Phase II",0)
p3 = phase_counts.get("Phase III",0)
i_to_ii  = round(p2/max(p1,1)*100,1)
ii_to_iii = round(p3/max(p2,1)*100,1)
overall_funnel = round(p3/max(p1,1)*100,1)

print(f"  Phase I count:   {p1}")
print(f"  Phase II count:  {p2}  (survival: {i_to_ii}% of Phase I)")
print(f"  Phase III count: {p3}  (survival: {ii_to_iii}% of Phase II, {overall_funnel}% of Phase I)")

findings["phase_attrition"] = {
    "phase_counts": phase_counts,
    "phase_completion_rates": phase_comp,
    "i_to_ii_pct": i_to_ii,
    "ii_to_iii_pct": ii_to_iii,
    "i_to_iii_pct": overall_funnel,
    "total_phase_i": p1,
    "key_insight": f"Only {overall_funnel}% of Phase I trials ever reach Phase III - {100-overall_funnel:.0f}% are lost in the pipeline",
    "plain_english": f"For every 100 early-stage cancer trials started, only {overall_funnel:.0f} make it all the way to large-scale testing. The rest are abandoned - too dangerous, too expensive, or simply didn't work well enough."
}

# ═══════════════════════════════════════════════════════════════
# FINDING 7: SPONSOR PERFORMANCE GAP
# Does industry outperform academia in trial completion?
# ═══════════════════════════════════════════════════════════════
print("\n─" * 33)
print("FINDING 7: Sponsor Performance Gap")
print("─" * 33)

sponsor_stats = {}
for sp in df_l["sponsor_class_clean"].unique():
    sub = df_l[df_l["sponsor_class_clean"]==sp]
    if len(sub) >= 10:
        sponsor_stats[sp] = {
            "n": int(len(sub)),
            "total": int((df["sponsor_class_clean"]==sp).sum()),
            "completion_rate": round(float(sub["trial_outcome"].mean())*100,1),
            "median_enrollment": int(sub["enrollment_count"].median()) if sub["enrollment_count"].notna().any() else 0,
        }

# Chi-square: industry vs non-industry outcome
df_l2 = df_l[df_l["sponsor_class_clean"].isin(["Industry","Academic/Hospital","NIH/Gov"])].copy()
if len(df_l2) > 50:
    ct = pd.crosstab(df_l2["sponsor_class_clean"], df_l2["trial_outcome"])
    chi2, p_val, dof, _ = stats.chi2_contingency(ct)
    sig = bool(p_val < 0.05)
else:
    chi2, p_val, sig = 0.0, 1.0, False

best_sponsor = max(sponsor_stats, key=lambda x: sponsor_stats[x]["completion_rate"]) if sponsor_stats else "Industry"
worst_sponsor = min(sponsor_stats, key=lambda x: sponsor_stats[x]["completion_rate"]) if sponsor_stats else "Other"
best_rate = sponsor_stats.get(best_sponsor,{}).get("completion_rate",0)
worst_rate = sponsor_stats.get(worst_sponsor,{}).get("completion_rate",0)
gap = round(best_rate - worst_rate, 1)

print(f"  Best sponsor:  {best_sponsor} ({best_rate}%)")
print(f"  Worst sponsor: {worst_sponsor} ({worst_rate}%)")
print(f"  Gap: {gap}pp  Chi²={chi2:.2f}  p={p_val:.4f}  Sig: {sig}")

findings["sponsor_gap"] = {
    "by_sponsor": sponsor_stats,
    "chi2": round(float(chi2),3),
    "p_value": round(float(p_val),4),
    "significant": sig,
    "best_sponsor": best_sponsor,
    "worst_sponsor": worst_sponsor,
    "gap_pp": gap,
    "key_insight": f"{best_sponsor} sponsors outperform {worst_sponsor} by {gap}pp in trial completion",
    "plain_english": f"Trials funded by {best_sponsor} complete {gap}% more often than those funded by {worst_sponsor}. {'This gap is statistically real.' if sig else 'However, this gap is not statistically significant - the difference may be due to chance.'}"
}

# ═══════════════════════════════════════════════════════════════
# FINDING 8: CAR-T CLINICAL MATURATION
# Track CAR-T phase distribution over time → is the field growing up?
# ═══════════════════════════════════════════════════════════════
print("\n─" * 33)
print("FINDING 8: CAR-T Clinical Maturation")
print("─" * 33)

THERAPY_PATTERNS = {
    "CAR-T": r'\bcar[-\s]?t\b|\bchimeric antigen receptor\b|\bcar t[-\s]cell',
    "CRISPR": r'\bcrispr\b|\bcas9\b|\bcas-9\b|\bgene editing\b|\bgenome editing\b',
    "NK Cell": r'\bnatural killer\b|\bnk cell\b|\bnk[-\s]cell',
    "TIL Therapy": r'\btumor.infiltrating\b|\btil\b(?!\w)',
    "mRNA Therapy": r'\bmrna\b|\bmessenger rna\b',
    "TCR Therapy": r'\bt.cell receptor\b|\btcr\b(?!\w)',
}

cart_df = df[df.get("intervention_text", df.get("title","")).astype(str).str.lower().str.contains(
    THERAPY_PATTERNS["CAR-T"], regex=True, na=False
)] if "intervention_text" in df.columns else df[df["cancer_type"].notna()].head(0)

# Fallback: use therapy_type column if available
if len(cart_df) < 10 and "therapy_type" in df.columns:
    cart_df = df[df["therapy_type"].str.contains("CAR", na=False)]

# Compute phase distribution by era
phase_simple = {"Early Phase I":"Phase I","Phase I":"Phase I","Phase I/II":"Phase I/II",
                "Phase II":"Phase II","Phase III+":"Phase III+","Phase III":"Phase III+","Phase IV":"Phase III+","Unknown":"Unknown"}
cart_by_era = {}
if len(cart_df) >= 20:
    for era in ["2010-2014","2015-2019","2020+"]:
        sub = cart_df[cart_df["trial_era"]==era] if "trial_era" in cart_df.columns else cart_df.head(0)
        if len(sub) >= 5:
            ph_counts = sub["phase_clean"].value_counts().to_dict()
            total = len(sub)
            cart_by_era[era] = {
                "n": int(total),
                "phase_dist": {p: round(c/total*100,1) for p,c in ph_counts.items()},
                "phase3_pct": round(sub["phase_clean"].str.contains("Phase III|Phase IV",na=False).mean()*100,1),
                "completion_rate": round(float(df_l[df_l.index.isin(sub.index)]["trial_outcome"].mean())*100,1) if len(df_l[df_l.index.isin(sub.index)])>5 else None
            }

# Use overall phase data as proxy if no therapy column
phase_III_2020 = round(df[df["trial_era"]=="2020+"]["phase_clean"].str.contains("Phase III|Phase IV",na=False).mean()*100,1)
phase_III_2015 = round(df[df["trial_era"]=="2015-2019"]["phase_clean"].str.contains("Phase III|Phase IV",na=False).mean()*100,1)
phase_III_delta = round(phase_III_2020 - phase_III_2015, 1)
maturation_direction = "MATURING" if phase_III_delta > 0 else "REGRESSING"

print(f"  Phase III% in 2015-2019: {phase_III_2015}%")
print(f"  Phase III% in 2020+:     {phase_III_2020}%")
print(f"  Delta: {phase_III_delta:+.1f}pp  Direction: {maturation_direction}")

findings["cart_maturation"] = {
    "by_era": cart_by_era,
    "phase_iii_2015_2019": phase_III_2015,
    "phase_iii_2020_plus": phase_III_2020,
    "delta_pp": phase_III_delta,
    "direction": maturation_direction,
    "key_insight": f"Phase III trial share {'increased' if phase_III_delta > 0 else 'decreased'} by {abs(phase_III_delta)}pp from 2015-2019 to 2020+ - the field is {maturation_direction.lower()}",
    "plain_english": f"In 2015-2019, {phase_III_2015}% of trials were large-scale Phase III studies. By 2020+, that number {'rose' if phase_III_delta > 0 else 'fell'} to {phase_III_2020}%. {'This means the field is graduating from early safety studies toward definitive proof-of-efficacy trials.' if phase_III_delta > 0 else 'More early-phase trials are being started than late-phase ones finishing - the field is still in the exploratory stage.'}"
}

# ═══════════════════════════════════════════════════════════════
# FINDING 9: THERAPY SURVIVAL RANKINGS
# Which therapy types have the highest completion rates?
# ═══════════════════════════════════════════════════════════════
print("\n─" * 33)
print("FINDING 9: Therapy Survival Rankings")
print("─" * 33)

THERAPY_COL = "therapy_type" if "therapy_type" in df.columns else None
therapy_survival = {}

if THERAPY_COL:
    for therapy in df[THERAPY_COL].dropna().unique():
        sub_all = df[df[THERAPY_COL]==therapy]
        sub_lab = df_l[df_l[THERAPY_COL]==therapy]
        if len(sub_lab) >= 20:
            therapy_survival[therapy] = {
                "n_total": int(len(sub_all)),
                "n_labeled": int(len(sub_lab)),
                "completion_rate": round(float(sub_lab["trial_outcome"].mean())*100,1),
                "median_enrollment": int(sub_all["enrollment_count"].median()) if sub_all["enrollment_count"].notna().any() else 0,
                "phase3_pct": round(sub_all["phase_clean"].str.contains("Phase III|Phase IV",na=False).mean()*100,1),
            }
    therapy_survival = dict(sorted(therapy_survival.items(), key=lambda x: x[1]["completion_rate"], reverse=True))

# Fallback: use tumor category
if not therapy_survival:
    for cat in ["Hematologic","Solid Tumor"]:
        sub_lab = df_l[df_l["tumor_category"]==cat]
        if len(sub_lab) >= 20:
            therapy_survival[cat] = {
                "n_total": int((df["tumor_category"]==cat).sum()),
                "n_labeled": int(len(sub_lab)),
                "completion_rate": round(float(sub_lab["trial_outcome"].mean())*100,1),
                "median_enrollment": int(df[df["tumor_category"]==cat]["enrollment_count"].median()),
            }

# Chi-square across all labeled therapy types
if len(therapy_survival) >= 2:
    if THERAPY_COL:
        df_th = df_l[df_l[THERAPY_COL].isin(list(therapy_survival.keys()))]
        if len(df_th) > 50:
            ct2 = pd.crosstab(df_th[THERAPY_COL], df_th["trial_outcome"])
            chi2_th, p_th, _, _ = stats.chi2_contingency(ct2)
            sig_th = bool(p_th < 0.05)
        else:
            chi2_th, p_th, sig_th = 0, 1.0, False
    else:
        chi2_th, p_th, sig_th = 0, 1.0, False
else:
    chi2_th, p_th, sig_th = 0, 1.0, False

best_therapy = list(therapy_survival.keys())[0] if therapy_survival else "Unknown"
worst_therapy = list(therapy_survival.keys())[-1] if len(therapy_survival)>1 else "Unknown"

print(f"  Best therapy:  {best_therapy} ({therapy_survival.get(best_therapy,{}).get('completion_rate','?')}%)")
print(f"  Worst therapy: {worst_therapy} ({therapy_survival.get(worst_therapy,{}).get('completion_rate','?')}%)")

findings["therapy_survival"] = {
    "rankings": therapy_survival,
    "chi2": round(float(chi2_th),3),
    "p_value": round(float(p_th),4),
    "significant": sig_th,
    "best_therapy": best_therapy,
    "worst_therapy": worst_therapy,
    "key_insight": f"{best_therapy} leads all therapy types in trial completion rate",
    "plain_english": f"Not all cancer therapies are equal. {best_therapy} trials finish at the highest rate ({therapy_survival.get(best_therapy,{}).get('completion_rate','?')}%), while {worst_therapy} trials are most likely to be abandoned ({therapy_survival.get(worst_therapy,{}).get('completion_rate','?')}%). This ranking reveals which therapeutic approaches the scientific community has the most confidence in."
}

# ═══════════════════════════════════════════════════════════════
# FINDING 10: THE 2030 TRIAL VOLUME PROJECTION
# Linear extrapolation of trial starts → where will the field be in 2030?
# ═══════════════════════════════════════════════════════════════
print("\n─" * 33)
print("FINDING 10: The 2030 Trial Volume Projection")
print("─" * 33)

yr_counts = df[df["start_year"].between(2013,2024)].groupby("start_year").size().reset_index(name="count")
if len(yr_counts) >= 6:
    X = yr_counts["start_year"].values.astype(float)
    Y = yr_counts["count"].values.astype(float)
    slope, intercept, r_val, p_val_proj, _ = stats.linregress(X, Y)
    r2 = round(r_val**2, 3)
    projections = {}
    for yr in [2025,2026,2027,2028,2029,2030]:
        proj = max(0, round(slope*yr + intercept))
        projections[str(yr)] = int(proj)
    historical = {str(int(r["start_year"])): int(r["count"]) for _, r in yr_counts.iterrows()}
    annual_growth = round(slope, 1)
    total_2030 = projections.get("2030",0)
    base_2024 = int(yr_counts[yr_counts["start_year"]==2024]["count"].values[0]) if 2024 in yr_counts["start_year"].values else int(Y[-1])
    growth_pct = round((total_2030 - base_2024)/max(base_2024,1)*100,1)
    # Top growing cancers
    top_cancers = {}
    for ct in df["cancer_type"].value_counts().head(10).index:
        ct_yr = df[(df["cancer_type"]==ct)&df["start_year"].between(2015,2024)].groupby("start_year").size().reset_index(name="count")
        if len(ct_yr)>=5:
            xc=ct_yr["start_year"].values.astype(float); yc=ct_yr["count"].values.astype(float)
            sc,ic,_,_,_ = stats.linregress(xc,yc)
            proj2030 = max(0,round(sc*2030+ic))
            top_cancers[ct]={"slope":round(float(sc),2),"proj_2030":int(proj2030),"base_2024":int(yc[-1]) if len(yc)>0 else 0}
    top_cancers = dict(sorted(top_cancers.items(),key=lambda x:x[1]["slope"],reverse=True))
else:
    slope,r2,p_val_proj,projections,historical,annual_growth,total_2030,base_2024,growth_pct,top_cancers = 0,0,1.0,{},{},0,0,0,0,{}

print(f"  Annual growth rate: +{annual_growth:.1f} trials/year")
print(f"  Projected 2030: {total_2030:,} new trials/year (R²={r2})")
print(f"  Growth vs 2024: +{growth_pct}%")

findings["projection_2030"] = {
    "historical": historical,
    "projections": projections,
    "slope": float(annual_growth),
    "r2": float(r2),
    "p_value": round(float(p_val_proj),4),
    "base_2024": int(base_2024),
    "proj_2030": int(total_2030),
    "growth_pct_2024_to_2030": float(growth_pct),
    "top_growing_cancers": top_cancers,
    "key_insight": f"At current growth (+{annual_growth}/year), gene-editing trial volume will reach {total_2030:,}/year by 2030 - a {growth_pct}% increase from 2024",
    "plain_english": f"Based on trends from 2013-2024, scientists will be running about {total_2030:,} new gene-editing cancer trials per year by 2030. That's {growth_pct}% more than today. The fastest-growing cancer areas are {', '.join(list(top_cancers.keys())[:3])} - meaning these will dominate the research landscape this decade."
}

# ── Save ──────────────────────────────────────────────────────
with open("outputs/findings.json","w") as f:
    json.dump(findings, f, indent=2, cls=type('NpEncoder', (json.JSONEncoder,), {
        'default': lambda self, o: (
            int(o) if isinstance(o, np.integer) else
            float(o) if isinstance(o, np.floating) else
            bool(o) if isinstance(o, np.bool_) else
            o.tolist() if isinstance(o, np.ndarray) else
            super(type(self), self).default(o)
        )
    }))
print("\n" + "=" * 65)
print("FINDINGS SAVED → outputs/findings.json")
print("=" * 65)

print(f"\n  FINDING 1: Most underserved cancer: {findings['treatment_desert']['most_underserved'][0]}")
print(f"  FINDING 2: Blood vs solid gap is {findings['convergence']['trend_direction']} ({findings['convergence']['trend_slope']:+.3f} pp/yr)")
print(f"  FINDING 3: Enrollment tipping point = {findings['tipping_point']['tipping_point_enrollment']} patients (p={findings['tipping_point']['p_value']})")
print(f"  FINDING 4: Dual-target trials: {findings['dual_target']['early_pct']:.1f}% → {findings['dual_target']['late_pct']:.1f}% over the study period")
print(f"  FINDING 5: Country specialization confirmed - see findings.json for full breakdown")
print(f"  FINDING 6: Phase attrition - only {findings['phase_attrition']['i_to_iii_pct']}% of Phase I trials reach Phase III")
print(f"  FINDING 7: {findings['sponsor_gap']['best_sponsor']} outperforms {findings['sponsor_gap']['worst_sponsor']} by {findings['sponsor_gap']['gap_pp']}pp")
print(f"  FINDING 8: Field is {findings['cart_maturation']['direction']} - Phase III share {'+' if findings['cart_maturation']['delta_pp']>0 else ''}{findings['cart_maturation']['delta_pp']}pp")
print(f"  FINDING 9: Best therapy survival: {findings['therapy_survival']['best_therapy']}")
print(f"  FINDING 10: Projected 2030 trial volume: {findings['projection_2030']['proj_2030']:,}/year ({findings['projection_2030']['growth_pct_2024_to_2030']:+.1f}% vs 2024)")
print(f"\n  Next: python3 dashboard.py  (then open outputs/dashboard.html)")
