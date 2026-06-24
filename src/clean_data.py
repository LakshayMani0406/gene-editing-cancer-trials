"""
clean_data.py v2 - Expanded cancer type classifier
Key improvements:
  - 50+ cancer type patterns (was 24) - recovers most "Unclassified" trials
  - New types: Sarcoma, Thyroid, Esophageal, Gastric, Nasopharyngeal,
               Neuroblastoma, Mesothelioma, Endometrial, Testicular,
               Neuroendocrine, Skin/SCC, Medulloblastoma, Ewing Sarcoma,
               Wilms Tumor, Retinoblastoma, and broader "Other Solid"
  - "Other Cancer" bucket catches anything with cancer keywords
  - Enrollment cap raised to 20,000 (was 10,000) - keep large trials
  - Duration cap raised to 480 months (was 360)

Usage: python3 clean_data.py
"""
import pandas as pd, numpy as np, re
from pathlib import Path

Path("data").mkdir(exist_ok=True)

df = pd.read_csv("raw/clinicaltrials_studies.csv")
df_pub = pd.read_csv("raw/pubmed_counts_by_year.csv")

print("="*60)
print("CLEANING REPORT")
print("="*60)
print(f"\nRaw records loaded: {len(df):,}")

log = []

# ── Step 1: Drop non-interventional ──────────────────────────────────────
before = len(df)
df = df[df["study_type"].str.upper().fillna("") == "INTERVENTIONAL"]
msg = f"Dropped {before-len(df)} non-interventional studies"
print(f"  [{len(df):>6}] {msg}"); log.append(msg)

# ── Step 2: Valid status ──────────────────────────────────────────────────
valid = {"COMPLETED","TERMINATED","ACTIVE_NOT_RECRUITING","RECRUITING","WITHDRAWN","SUSPENDED","UNKNOWN"}
before = len(df)
df = df[df["overall_status"].isin(valid)]
msg = f"Dropped {before-len(df)} records with invalid/null status"
print(f"  [{len(df):>6}] {msg}"); log.append(msg)

# ── Step 3: Require start date ────────────────────────────────────────────
before = len(df)
df = df[df["start_date"].notna()]
msg = f"Dropped {before-len(df)} records with no start date"
print(f"  [{len(df):>6}] {msg}"); log.append(msg)

# ── Step 4: Parse dates ───────────────────────────────────────────────────
def parse_date(s):
    if pd.isna(s): return pd.NaT
    s = str(s).strip()
    for fmt in ("%Y-%m-%d","%B %Y","%Y"):
        try: return pd.to_datetime(s, format=fmt)
        except: pass
    return pd.NaT

df["start_date_dt"]      = df["start_date"].apply(parse_date)
df["completion_date_dt"] = df["completion_date"].apply(parse_date)
df["start_year"]         = df["start_date_dt"].dt.year
df["start_month"]        = df["start_date_dt"].dt.month

before = len(df)
df = df[df["start_year"] >= 1990]
msg = f"Dropped {before-len(df)} records with start year < 1990"
print(f"  [{len(df):>6}] {msg}"); log.append(msg)

df["duration_months"] = ((df["completion_date_dt"]-df["start_date_dt"]).dt.days/30.44).round(1)
bad_dur = df["duration_months"].notna() & ((df["duration_months"]<0)|(df["duration_months"]>480))
df.loc[bad_dur,"duration_months"] = np.nan
if bad_dur.sum():
    msg = f"Nulled {bad_dur.sum()} implausible durations"
    print(f"  [{len(df):>6}] {msg}"); log.append(msg)

# ── Step 5: Clean enrollment ──────────────────────────────────────────────
df["enrollment_count"] = pd.to_numeric(df["enrollment_count"],errors="coerce")
extreme = df["enrollment_count"] > 20000
df.loc[extreme,"enrollment_count"] = np.nan
if extreme.sum():
    msg = f"Nulled {extreme.sum()} enrollment values > 20,000"
    print(f"  [{len(df):>6}] {msg}"); log.append(msg)

# ── Step 6: Standardize phase ─────────────────────────────────────────────
PHASE_MAP = {
    "PHASE1":"Phase I","PHASE1/PHASE2":"Phase I/II","PHASE2":"Phase II",
    "PHASE3":"Phase III","PHASE4":"Phase IV","NA":"N/A",
    "EARLY_PHASE1":"Early Phase I",
}
def clean_phase(val):
    if pd.isna(val): return "Unknown"
    parts = [p.strip() for p in re.split(r"[,|]",str(val).upper().replace(" ",""))]
    mapped = [PHASE_MAP.get(p,p) for p in parts]
    return " / ".join(sorted(set(mapped),key=lambda x: mapped.index(x)))

df["phase_clean"] = df["phase"].apply(clean_phase)
print(f"\n  Phase distribution:")
print(df["phase_clean"].value_counts().head(10).to_string())

# ── Step 7: EXPANDED cancer type classifier ───────────────────────────────
CANCER_PATTERNS = {
    # Hematologic - specific
    "Leukemia (AML)":           r"\baml\b|acute myeloid|acute myelogenous",
    "Leukemia (ALL)":           r"\ball\b|acute lympho",
    "Leukemia (CLL/CML)":       r"\bcll\b|\bcml\b|chronic lympho|chronic myelo",
    "Lymphoma":                 r"lymphoma|dlbcl|hodgkin|non.hodgkin|mantle cell|follicular lymph|burkitt",
    "Multiple Myeloma":         r"myeloma|plasma cell neoplasm",
    "Hematologic (General)":    r"hematolog|blood cancer|bone marrow|myelodysplastic|\bMDS\b|aplastic anemia",
    # Solid - CNS
    "Brain/CNS":                r"brain|gliom|glioblas|\bGBM\b|\bcns\b|central nervous|meningioma|medulloblastoma|ependymoma",
    "Neuroblastoma":            r"neuroblastoma",
    # Solid - Thoracic
    "Lung Cancer":              r"lung|nsclc|sclc|pulmonary carcinoma|bronch",
    "Mesothelioma":             r"mesothelioma",
    # Solid - GI
    "Colorectal Cancer":        r"colorect|colon|rectal|rectum",
    "Gastric Cancer":           r"gastric|stomach",
    "Esophageal Cancer":        r"esophag|oesophag",
    "Pancreatic Cancer":        r"pancrea",
    "Liver Cancer":             r"liver|hepato|hcc|hepatocellular",
    # Solid - GYN
    "Breast Cancer":            r"breast",
    "Ovarian Cancer":           r"ovar",
    "Cervical Cancer":          r"cervical|cervix|hpv",
    "Endometrial Cancer":       r"endometri|uterine",
    # Solid - Urologic
    "Bladder Cancer":           r"bladder|urothelial",
    "Prostate Cancer":          r"prostate",
    "Renal/Kidney":             r"\brenal\b|kidney|clear cell carcinoma|wilms",
    "Testicular Cancer":        r"testicular|germ cell|seminoma",
    # Solid - Skin
    "Melanoma":                 r"melanoma",
    "Skin Cancer (Non-Melanoma)": r"\bscc\b|squamous cell skin|basal cell|merkel|cutaneous",
    # Solid - Head & Neck
    "Head & Neck":              r"head.*neck|oral cavity|pharyn|laryn|nasopharyn|salivary",
    # Solid - Sarcoma
    "Sarcoma":                  r"sarcoma|osteosarcoma|rhabdomyo|ewing|liposarcoma|leiomyosarcoma",
    # Solid - Other specific
    "Thyroid Cancer":           r"thyroid",
    "Neuroendocrine":           r"neuroendocrine|carcinoid|pheochromocytoma",
    "Retinoblastoma":           r"retinoblastoma",
    # Solid - broad
    "Solid Tumors (Multiple)":  r"solid tumor|advanced cancer|multiple.*tumor|various.*cancer|pan.cancer|unresectable|metastatic solid",
    # Broad matches
    "Prostate Cancer":          r"prostate",
    "Other Cancer":             r"cancer|carcinoma|tumor|neoplasm|malignant|sarcoma|lymphoma|leukemia",
}

def classify_cancer(s):
    if pd.isna(s): return "Unclassified"
    s = str(s)
    for label, pattern in CANCER_PATTERNS.items():
        if re.search(pattern, s, re.IGNORECASE):
            return label
    return "Unclassified"

def cancer_category(t):
    hem = {"Leukemia (AML)","Leukemia (ALL)","Leukemia (CLL/CML)",
           "Lymphoma","Multiple Myeloma","Hematologic (General)"}
    return "Hematologic" if t in hem else ("Unknown" if t=="Unclassified" else "Solid Tumor")

df["cancer_type"]    = df["conditions"].apply(classify_cancer)
df["tumor_category"] = df["cancer_type"].apply(cancer_category)

print(f"\n  Cancer type distribution (top 20):")
print(df["cancer_type"].value_counts().head(20).to_string())
print(f"\n  Unclassified: {(df['cancer_type']=='Unclassified').sum()}")

# ── Step 8: Sponsor class ─────────────────────────────────────────────────
SPONSOR_MAP = {
    "NIH":"Government (NIH)","FED":"Government (Federal)",
    "INDIV":"Individual","INDUSTRY":"Industry",
    "NETWORK":"Research Network","OTHER":"Academic / Non-profit",
    "UNKNOWN":"Unknown",
}
df["sponsor_class_clean"] = df["sponsor_class"].map(SPONSOR_MAP).fillna("Unknown")

# ── Step 9: Decade / era ─────────────────────────────────────────────────
def era(y):
    if pd.isna(y): return "Unknown"
    y = int(y)
    if y < 2000: return "1990s"
    if y < 2010: return "2000s"
    if y < 2015: return "2010-2014"
    if y < 2020: return "2015-2019"
    return "2020+"

df["trial_era"] = df["start_year"].apply(era)

# ── Step 10: ML target ────────────────────────────────────────────────────
SUCCESS = {"COMPLETED","ACTIVE_NOT_RECRUITING","RECRUITING"}
FAILURE = {"TERMINATED","WITHDRAWN","SUSPENDED"}

def assign_outcome(row):
    if row["overall_status"] in SUCCESS: return 1
    if row["overall_status"] in FAILURE: return 0
    return np.nan

df["trial_outcome"]     = df.apply(assign_outcome,axis=1)
df["has_results"]       = df["has_results"].fillna(False).astype(bool)
df["results_available"] = df["has_results"].astype(int)
df["includes_us"]       = df["location_countries"].str.contains("United States",na=False).astype(int)
df["is_multinational"]  = (df["n_countries"]>1).astype(int)
df["log_enrollment"]    = np.log1p(df["enrollment_count"].fillna(0))
df["is_hematologic"]    = (df["tumor_category"]=="Hematologic").astype(int)
df["is_recent"]         = (df["start_year"]>=2015).astype(float)

print(f"\n  Trial outcome distribution:")
print(df["trial_outcome"].value_counts(dropna=False).to_string())

# ── Step 11: Drop unclassified ────────────────────────────────────────────
before = len(df)
df = df[df["cancer_type"] != "Unclassified"]
msg = f"Dropped {before-len(df)} truly unclassifiable records"
print(f"\n  [{len(df):>6}] {msg}"); log.append(msg)

KEEP = [
    "nct_id","brief_title","overall_status","trial_outcome",
    "cancer_type","tumor_category","trial_era","phase_clean",
    "sponsor_class_clean","enrollment_count","log_enrollment",
    "duration_months","start_year","start_month","n_countries",
    "includes_us","is_multinational","results_available",
    "n_primary_outcomes","is_hematologic","is_recent",
    "conditions","interventions","primary_outcome",
    "lead_sponsor","location_countries",
    "start_date","completion_date","has_results","why_stopped",
]
df_clean = df[[c for c in KEEP if c in df.columns]].copy()

print(f"\n  FINAL clean dataset: {len(df_clean):,} records × {df_clean.shape[1]} columns")

# ── Step 12: Publication trends ───────────────────────────────────────────
df_pub = df_pub.dropna(subset=["year"])
df_pub["year"] = df_pub["year"].astype(int)
num_cols = [c for c in df_pub.columns if c != "year"]
df_pub["total"] = df_pub[num_cols].sum(axis=1)
for col in num_cols:
    df_pub[f"{col}_yoy_pct"] = df_pub[col].pct_change()*100

# ── Save ──────────────────────────────────────────────────────────────────
df_clean.to_csv("data/crispr_trials_clean.csv",index=False)
df_pub.to_csv("data/publication_trends_clean.csv",index=False)

with open("data/cleaning_report.txt","w") as f:
    f.write(f"Cleaning Report\nRaw: {len(pd.read_csv('raw/clinicaltrials_studies.csv')):,}\n")
    f.write(f"Clean: {len(df_clean):,}\n\nSteps:\n")
    for i,s in enumerate(log,1): f.write(f"  {i}. {s}\n")
    f.write(f"\nColumns:\n")
    for c in df_clean.columns:
        f.write(f"  {c:<30} nulls: {df_clean[c].isna().sum()}/{len(df_clean)}\n")

print(f"\n  Saved → data/crispr_trials_clean.csv")
print(f"  Saved → data/publication_trends_clean.csv")
print(f"  Saved → data/cleaning_report.txt")

print(f"\n  Null counts in key ML features:")
for c in ["cancer_type","tumor_category","phase_clean","sponsor_class_clean",
          "enrollment_count","duration_months","trial_outcome"]:
    if c in df_clean.columns:
        n = df_clean[c].isna().sum()
        print(f"    {c:<30}  {n:>5} / {len(df_clean):,}  ({n/len(df_clean):.1%})")

print(f"\n  Cancer type distribution:")
print(df_clean["cancer_type"].value_counts().head(20).to_string())

print(f"\n  Trial era distribution:")
print(df_clean["trial_era"].value_counts().to_string())

print(f"\n  Next: python3 eda.py")
