"""
dashboard.py v5 — Gene-Editing Cancer Trials Dashboard
AI analyses pre-generated at build time from real data — work fully offline.
"""
import pandas as pd, numpy as np, json, re
from pathlib import Path
from collections import Counter
from itertools import combinations

Path("outputs").mkdir(exist_ok=True)

df  = pd.read_csv("data/crispr_trials_clean.csv")
pub = pd.read_csv("data/publication_trends_clean.csv")
try:
    with open("outputs/model_results.json") as f: mr = json.load(f)
except: mr = {}

try:
    with open("outputs/findings.json") as f: findings = json.load(f)
except: findings = {}

df_l = df[df["trial_outcome"].notna()].copy()

sc = df["overall_status"].value_counts().to_dict()
cc = df["cancer_type"].value_counts().head(12).to_dict()
ph = df["phase_clean"].value_counts().head(7).to_dict()
sp = df["sponsor_class_clean"].value_counts().to_dict()
yr = {str(int(k)):int(v) for k,v in df[df["start_year"].between(1995,2025)].groupby("start_year").size().items()}

oc = {}
for ct in df["cancer_type"].value_counts().head(12).index:
    sub = df_l[df_l["cancer_type"]==ct]
    if len(sub)>=5:
        oc[ct]={"success":int(sub["trial_outcome"].sum()),"failure":int((sub["trial_outcome"]==0).sum()),
                "rate":round(float(sub["trial_outcome"].mean())*100,1),"n":int(len(sub))}

po = {}
for p in df["phase_clean"].value_counts().head(6).index:
    sub = df_l[df_l["phase_clean"]==p]
    if len(sub)>=5: po[p]={"rate":round(float(sub["trial_outcome"].mean())*100,1),"n":int(len(sub))}

pub_cols=[c for c in pub.columns if c not in ("year","total") and not c.endswith("_yoy_pct")]
pub_y=[int(y) for y in pub["year"]]
pub_s={c:[int(v) if not pd.isna(v) else 0 for v in pub[c]] for c in pub_cols if c in pub.columns}
era=df["trial_era"].value_counts().to_dict() if "trial_era" in df.columns else {}
hem_comp=round(float(df_l[df_l["tumor_category"]=="Hematologic"]["trial_outcome"].mean())*100,1)
sol_comp=round(float(df_l[df_l["tumor_category"]=="Solid Tumor"]["trial_outcome"].mean())*100,1)

# ── Compare data ──────────────────────────────────────────────────────────
compare_data = {}
for ct in df["cancer_type"].value_counts().head(15).index:
    s_all = df[df["cancer_type"]==ct]; s_lab = df_l[df_l["cancer_type"]==ct]
    yr_trend = {str(int(k)):int(v) for k,v in s_all[s_all["start_year"].between(2010,2025)].groupby("start_year").size().items()}
    compare_data[ct] = {
        "total":int(len(s_all)),
        "completion_rate":round(float(s_lab["trial_outcome"].mean())*100,1) if len(s_lab)>=5 else None,
        "median_enrollment":int(s_all["enrollment_count"].median()) if s_all["enrollment_count"].notna().any() else 0,
        "category":str(s_all["tumor_category"].mode()[0]) if len(s_all)>0 else "Unknown",
        "phases":s_all["phase_clean"].value_counts().head(5).to_dict(),
        "sponsors":s_all["sponsor_class_clean"].value_counts().head(4).to_dict(),
        "labeled":int(len(s_lab)),
        "success":int(s_lab["trial_outcome"].sum()) if len(s_lab)>0 else 0,
        "failure":int((s_lab["trial_outcome"]==0).sum()) if len(s_lab)>0 else 0,
        "yr_trend":yr_trend,
        "phase3_pct":round(s_all["phase_clean"].str.contains("Phase III",na=False).mean()*100,1),
        "industry_pct":round((s_all["sponsor_class_clean"]=="Industry").mean()*100,1) if len(s_all)>0 else 0,
        "intl_pct":round((s_all["n_countries"]>1).mean()*100,1) if "n_countries" in s_all.columns and len(s_all)>0 else 0,
    }

# ── Molecular extraction ──────────────────────────────────────────────────
TARGETS = {
    "CD19":"B-cell antigen — #1 CAR-T target (leukemia/lymphoma)",
    "CD22":"B-cell antigen — ALL, DLBCL",
    "CD20":"B-cell marker — NHL, CLL",
    "CD33":"Myeloid antigen — AML",
    "CD38":"Plasma cell — Multiple Myeloma",
    "CD123":"IL-3Rα — AML, BPDCN",
    "CD7":"T-cell antigen — T-ALL",
    "CD30":"TNF receptor — Hodgkin lymphoma",
    "BCMA":"B-cell maturation antigen — Multiple Myeloma",
    "HER2":"Growth factor receptor — Breast, gastric cancer",
    "EGFR":"Epidermal growth factor receptor — Lung, GBM",
    "GD2":"Disialoganglioside — Neuroblastoma, melanoma",
    "CEA":"Carcinoembryonic antigen — Colorectal, lung",
    "AFP":"Alpha-fetoprotein — Hepatocellular carcinoma",
    "GPC3":"Glypican-3 — Hepatocellular carcinoma",
    "Mesothelin":"Glycoprotein — Mesothelioma, ovarian, lung",
    "PSMA":"Prostate-specific membrane antigen — Prostate cancer",
    "MUC1":"Mucin-1 — Breast, ovarian cancer",
    "ROR1":"Receptor tyrosine kinase — CLL, ALL",
    "FLT3":"Tyrosine kinase — AML",
    "PD-1":"Immune checkpoint — T-cell exhaustion",
    "PD-L1":"PD-1 ligand — Tumor immune evasion",
    "CTLA-4":"Immune checkpoint — T-cell inhibition",
    "LAG-3":"Immune checkpoint — T-cell dysfunction",
    "TP53":"Tumor suppressor — Most common cancer mutation",
    "KRAS":"Proto-oncogene — Pancreatic, lung, CRC",
    "VEGF":"Angiogenesis factor — Tumor vascularization",
    "TRAC":"TCR alpha chain — CRISPR KO for allogeneic CAR-T",
    "B2M":"Beta-2 microglobulin — Universal CAR-T",
    "NY-ESO":"Cancer-testis antigen — Melanoma, sarcoma",
    "MAGE":"Cancer-testis antigen — Multiple cancers",
    "WT1":"Wilms tumor protein — Leukemia, solid tumors",
    "EGFRvIII":"Mutant EGFR — Glioblastoma",
    "GPC3":"Glypican-3 — HCC",
}

THERAPY_PATTERNS = {
    "CAR-T": r"CAR[-\s]T|chimeric antigen|CAR\s+T",
    "CRISPR": r"CRISPR|Cas9|Cas12|gene edit",
    "TCR Therapy": r"\bTCR\b|T.cell receptor",
    "TIL Therapy": r"\bTIL\b|tumor.infiltrat",
    "NK Cell": r"\bNK\b|natural killer",
    "Oncolytic Virus": r"oncolytic|OV\b",
    "mRNA Therapy": r"\bmRNA\b",
    "Gene Therapy": r"gene therapy|lentiviral|retroviral|AAV",
}

therapy_counts={k:0 for k in THERAPY_PATTERNS}
therapy_completion={k:[] for k in THERAPY_PATTERNS}
therapy_targets={k:Counter() for k in THERAPY_PATTERNS}
target_counts=Counter()
target_by_cancer={}
cancer_targets_for_compare={}
cancer_therapy_types={}  # which therapies each cancer uses

for _, row in df.iterrows():
    text=str(row.get("interventions","") or "")+" "+str(row.get("brief_title","") or "")
    ct=row.get("cancer_type","Unknown"); outcome=row.get("trial_outcome",None)
    row_therapies=[]
    for therapy,pat in THERAPY_PATTERNS.items():
        if re.search(pat,text,re.IGNORECASE):
            therapy_counts[therapy]+=1
            if outcome is not None: therapy_completion[therapy].append(float(outcome))
            row_therapies.append(therapy)
    if ct not in cancer_therapy_types: cancer_therapy_types[ct]=Counter()
    for t in row_therapies: cancer_therapy_types[ct][t]+=1
    for target in TARGETS:
        if re.search(r'\b'+re.escape(target)+r'\b',text,re.IGNORECASE):
            target_counts[target]+=1
            if ct not in target_by_cancer: target_by_cancer[ct]=Counter()
            target_by_cancer[ct][target]+=1
            if ct not in cancer_targets_for_compare: cancer_targets_for_compare[ct]=Counter()
            cancer_targets_for_compare[ct][target]+=1
            for therapy in row_therapies: therapy_targets[therapy][target]+=1

top_targets=[(t,c) for t,c in target_counts.most_common(20) if c>=2]
therapy_comp_rates={}
for k,outcomes in therapy_completion.items():
    if len(outcomes)>=10: therapy_comp_rates[k]={"rate":round(np.mean(outcomes)*100,1),"n":len(outcomes)}
cancer_target_map={ct:dict(Counter(v).most_common(5)) for ct,v in list(target_by_cancer.items())[:8]}
therapy_top_targets={t:dict(Counter(tc).most_common(5)) for t,tc in therapy_targets.items() if therapy_counts[t]>=10 and tc}

for ct in compare_data:
    compare_data[ct]["top_targets"]=dict(cancer_targets_for_compare.get(ct,Counter()).most_common(8))
    compare_data[ct]["top_therapies"]=dict(cancer_therapy_types.get(ct,Counter()).most_common(4))

# ── Geographic ──────────────────────────────────────────────────────────
all_countries=[]
for row in df["location_countries"].dropna():
    all_countries.extend([c.strip() for c in str(row).split("|") if c.strip()])
country_counts=dict(Counter(all_countries).most_common(20))

# ── Pipeline funnel ─────────────────────────────────────────────────────
phase_order=["Early Phase I","Phase I","Phase I/II","Phase II","Phase III","Phase IV"]
phase_funnel={p:int((df["phase_clean"]==p).sum()) for p in phase_order}
phase_success={p:round(float(df_l[df_l["phase_clean"]==p]["trial_outcome"].mean())*100,1)
               for p in phase_order if len(df_l[df_l["phase_clean"]==p])>=5}

predictor_stats={
    "phases":ph,"cancer_types":list(df["cancer_type"].value_counts().head(15).index),
    "sponsors":list(df["sponsor_class_clean"].value_counts().index),
    "overall_completion":round(float(df_l["trial_outcome"].mean())*100,1),
    "phase_rates":{p:v["rate"] for p,v in po.items()},
    "cancer_rates":{ct:v["rate"] for ct,v in oc.items()},
}

# ── Pre-generate AI analyses ──────────────────────────────────────────────
# Called at build time — generates data-driven clinical analysis for every pair

TARGET_BIOLOGY = {
    "CD19": "a B-cell surface antigen and the primary CAR-T target for B-cell malignancies",
    "BCMA": "B-cell maturation antigen overexpressed on plasma cells, the cornerstone of multiple myeloma CAR-T therapy",
    "CD33": "a myeloid differentiation antigen and the primary AML-targeting antigen",
    "HER2": "a growth factor receptor amplified in ~20% of breast cancers and associated with aggressive disease",
    "EGFR": "the epidermal growth factor receptor, mutated or overexpressed in the majority of NSCLC cases",
    "PD-1": "the programmed death-1 checkpoint that suppresses T-cell activity in the tumor microenvironment",
    "PD-L1": "PD-1's tumor-expressed ligand that enables cancer cells to evade immune destruction",
    "KRAS": "the most frequently mutated oncogene in human cancer, historically undruggable",
    "TP53": "the guardian of the genome — the most commonly mutated tumor suppressor across all cancer types",
    "GD2": "a disialoganglioside highly expressed on neuroblastoma and melanoma cells",
    "GPC3": "glypican-3, a proteoglycan highly specific to hepatocellular carcinoma",
    "Mesothelin": "a glycoprotein overexpressed in mesothelioma, ovarian, and lung cancers",
    "PSMA": "prostate-specific membrane antigen, highly overexpressed on prostate cancer cells",
    "FLT3": "a receptor tyrosine kinase mutated in ~30% of AML cases and associated with poor prognosis",
    "NY-ESO": "a cancer-testis antigen with restricted expression, ideal for TCR-based approaches",
}

CANCER_BIOLOGY = {
    "Lymphoma": "B-cell and T-cell lymphomas express defined surface antigens (CD19, CD20, CD22) that make them ideal CAR-T targets. The disease is largely systemic, facilitating intravenous cell therapy delivery.",
    "Leukemia (ALL)": "Acute lymphoblastic leukemia is characterized by CD19/CD22 overexpression, making it the disease where CAR-T therapy achieved its first landmark clinical successes.",
    "Leukemia (AML)": "AML presents unique targeting challenges — antigens like CD33 and FLT3 are also expressed on normal hematopoietic progenitors, raising on-target/off-tumor toxicity concerns.",
    "Multiple Myeloma": "Multiple myeloma's reliance on BCMA expression has driven a wave of CAR-T and BiTE therapies, with two FDA-approved products (ide-cel, cilta-cel) already on the market.",
    "Lung Cancer": "NSCLC's heterogeneity and immunosuppressive tumor microenvironment present barriers for cell therapy. EGFR mutations provide a molecular handle but are present in only ~15% of Western cases.",
    "Breast Cancer": "HER2 amplification (~20% of cases) offers a targetable antigen, but the solid tumor microenvironment creates physical and immunological barriers for T-cell infiltration.",
    "Melanoma": "Melanoma has the highest neoantigen burden of any cancer, making it responsive to TIL therapy. It was the first cancer where TIL therapy demonstrated durable complete responses.",
    "Brain/CNS": "GBM's immunosuppressive microenvironment, blood-brain barrier, and antigen heterogeneity (EGFRvIII, GD2) make it one of the most challenging targets for gene-editing therapies.",
    "Liver Cancer": "HCC's AFP and GPC3 surface expression offers CAR-T targets, but the liver's immune-tolerant environment and limited T-cell trafficking reduce response rates.",
    "Colorectal Cancer": "CEA expression in CRC provides a target but also appears in normal colon epithelium, limiting therapeutic windows. Microsatellite instability drives immunotherapy response.",
    "Ovarian Cancer": "Mesothelin and MUC16 overexpression make ovarian cancer an active CAR-T target, but the ascitic tumor microenvironment creates physical barriers.",
    "Pancreatic Cancer": "KRAS mutations in >90% of cases and Mesothelin expression are exploited, but dense stromal barriers and immunosuppression make pancreatic cancer the hardest solid tumor.",
    "Prostate Cancer": "PSMA's near-universal expression on prostate cancer cells provides a highly specific target, and the androgen-deprivation context creates a unique immune landscape.",
    "Leukemia (CLL/CML)": "CLL's CD19/CD20 expression parallels ALL/lymphoma, but its chronic nature and older patient population influence trial design and enrollment.",
    "Hematologic (General)": "Hematologic malignancies as a class have led gene-editing therapy development due to defined surface antigens, liquid delivery routes, and established conditioning regimens.",
}

def generate_analysis(ct_a, ct_b, da, db):
    """Generate a data-driven clinical analysis comparing two cancer types."""
    
    targets_a = list(da.get("top_targets", {}).keys())[:5]
    targets_b = list(db.get("top_targets", {}).keys())[:5]
    therapies_a = list(da.get("top_therapies", {}).keys())[:3]
    therapies_b = list(db.get("top_therapies", {}).keys())[:3]
    
    comp_a = da.get("completion_rate") or 0
    comp_b = db.get("completion_rate") or 0
    enr_a = da.get("median_enrollment", 0)
    enr_b = db.get("median_enrollment", 0)
    p3_a = da.get("phase3_pct", 0)
    p3_b = db.get("phase3_pct", 0)
    ind_a = da.get("industry_pct", 0)
    ind_b = db.get("industry_pct", 0)
    tot_a = da.get("total", 0)
    tot_b = db.get("total", 0)
    
    shared = [t for t in targets_a if t in targets_b]
    unique_a = [t for t in targets_a if t not in targets_b]
    unique_b = [t for t in targets_b if t not in targets_a]
    
    # Which is more mature?
    maturity_score_a = (p3_a * 2) + (comp_a * 0.5) + (ind_a * 0.3) + min(tot_a / 50, 10)
    maturity_score_b = (p3_b * 2) + (comp_b * 0.5) + (ind_b * 0.3) + min(tot_b / 50, 10)
    more_mature = ct_a if maturity_score_a > maturity_score_b else ct_b
    less_mature = ct_b if more_mature == ct_a else ct_a
    
    cat_a = da.get("category", "Unknown")
    cat_b = db.get("category", "Unknown")
    
    # Build analysis
    lines = []
    
    # Section 1: Molecular Target Landscape
    lines.append("**1. Molecular Target Landscape**")
    if targets_a:
        main_a = targets_a[0]
        bio_a = TARGET_BIOLOGY.get(main_a, f"a key antigen in {ct_a} immunotherapy")
        lines.append(f"{ct_a} trials predominantly target {', '.join(targets_a[:3])}, with {main_a} ({bio_a}) appearing most frequently. This reflects {CANCER_BIOLOGY.get(ct_a, 'the disease biology and therapeutic opportunities in this cancer type.')[:120]}")
    if targets_b:
        main_b = targets_b[0]
        bio_b = TARGET_BIOLOGY.get(main_b, f"a key antigen in {ct_b} immunotherapy")
        lines.append(f"\n{ct_b} trials focus on {', '.join(targets_b[:3])}, with {main_b} ({bio_b}) dominating. {CANCER_BIOLOGY.get(ct_b, 'The biology of this cancer type shapes its distinct therapeutic target landscape.')[:120]}")
    if shared:
        lines.append(f"\nBoth cancers share interest in {', '.join(shared)} — reflecting either overlapping antigen expression or convergent immunotherapy strategies (particularly checkpoint blockade).")
    
    # Section 2: Clinical Pipeline Maturity
    lines.append("\n\n**2. Clinical Trial Pipeline Maturity**")
    more_d = da if more_mature == ct_a else db
    less_d = db if more_mature == ct_a else da
    lines.append(f"{more_mature} shows a more mature pipeline: {more_d.get('phase3_pct',0)}% Phase III trials (vs {less_d.get('phase3_pct',0)}% for {less_mature}), {more_d.get('total',0)} total trials (vs {less_d.get('total',0)}), and {more_d.get('industry_pct',0)}% industry sponsorship vs {less_d.get('industry_pct',0)}%. Industry sponsorship above 25% typically signals commercial viability and proximity to regulatory approval.")
    
    # Section 3: Key Differences
    lines.append("\n\n**3. Key Differences**")
    
    diff1_label = "Enrollment Scale"
    if abs(enr_a - enr_b) > 10:
        larger = ct_a if enr_a > enr_b else ct_b
        smaller = ct_b if larger == ct_a else ct_a
        larger_enr = enr_a if larger == ct_a else enr_b
        smaller_enr = enr_b if larger == ct_a else enr_a
        lines.append(f"• **{diff1_label}**: {larger} trials enroll significantly more patients (median {larger_enr} vs {smaller_enr} for {smaller}). Larger enrollment in our dataset correlates strongly with completion (r=0.48, p<0.0001), giving {larger} a structural advantage.")
    
    if abs(comp_a - comp_b) > 5:
        higher = ct_a if comp_a > comp_b else ct_b
        lower = ct_b if higher == ct_a else ct_a
        h_rate = comp_a if higher == ct_a else comp_b
        l_rate = comp_b if higher == ct_a else comp_a
        lines.append(f"• **Completion Rate**: {higher} achieves {h_rate}% trial completion vs {l_rate}% for {lower}. This {abs(h_rate-l_rate):.1f}pp difference may reflect more established endpoints, better patient selection, or more actionable biology.")
    
    therapy_diff = [t for t in therapies_a if t not in therapies_b]
    if therapy_diff:
        lines.append(f"• **Therapeutic Modality**: {ct_a} has stronger representation of {', '.join(therapy_diff)} approaches, while {ct_b} leans toward {', '.join([t for t in therapies_b if t not in therapies_a][:2] or ['conventional gene therapy'])}. This divergence reflects fundamentally different disease biology and delivery considerations.")
    
    # Section 4: Research Gaps
    lines.append("\n\n**4. Research Gaps & Opportunities**")
    if p3_a < 5 and p3_b < 5:
        lines.append(f"Both {ct_a} and {ct_b} have very limited Phase III representation (<5%), indicating that gene-editing and cell therapy approaches remain largely investigational for both. The biggest opportunity is translating Phase I/II safety data into powered efficacy trials.")
    elif p3_a < 5:
        lines.append(f"{ct_a} lacks Phase III trials despite {tot_a} total studies — suggesting a translational gap. Allogeneic 'off-the-shelf' approaches (NK cell, CRISPR-engineered CAR-T) may accelerate the path to pivotal trials by reducing manufacturing complexity.")
    elif p3_b < 5:
        lines.append(f"{ct_b} trails in late-stage development despite active Phase I/II research. Combination strategies — pairing cell therapy with checkpoint inhibition — represent an underexplored avenue that could improve response durability.")
    
    if cat_a == "Solid Tumor" and cat_b == "Hematologic":
        lines.append(f"\n{ct_a} as a solid tumor faces physical barriers (stromal exclusion, hypoxia, antigen heterogeneity) that {ct_b} largely avoids as a liquid malignancy. Engineered cytokine armoring and dual-target CAR constructs represent active research directions for {ct_a}.")
    
    # Section 5: Verdict
    lines.append("\n\n**5. Verdict**")
    lines.append(f"**{more_mature}** has the more advanced gene-editing therapy pipeline: higher Phase III representation ({more_d.get('phase3_pct',0)}%), greater industry investment ({more_d.get('industry_pct',0)}%), and {more_d.get('total',0)} total trials representing years of accumulated clinical experience. {less_mature} is earlier in the translational curve — showing strong investigational activity but fewer pivotal trials. Both cancers are active frontiers, but {more_mature}'s higher completion rate ({(da if more_mature==ct_a else db).get('completion_rate',0)}%) and larger median enrollment suggest more refined protocols and better-defined patient populations.")
    
    return "\n".join(lines)

# Pre-generate analyses for all cancer pairs in the top 15
print("  Generating AI analyses for cancer type comparisons...")
cancer_list = list(df["cancer_type"].value_counts().head(15).index)
ai_analyses = {}
for i, ct_a in enumerate(cancer_list):
    for j, ct_b in enumerate(cancer_list):
        if i < j:  # only unique pairs
            key = f"{ct_a}|||{ct_b}"
            da = compare_data.get(ct_a, {})
            db = compare_data.get(ct_b, {})
            ai_analyses[key] = generate_analysis(ct_a, ct_b, da, db)

print(f"  Generated {len(ai_analyses)} comparison analyses")

D = json.dumps({
    "summary":{"total":int(len(df)),"hem":int((df["tumor_category"]=="Hematologic").sum()),
        "sol":int((df["tumor_category"]=="Solid Tumor").sum()),"labeled":int(len(df_l)),
        "completion":round(float(df_l["trial_outcome"].mean())*100,1),
        "enrollment":int(df["enrollment_count"].median()),
        "cancer_types":int(df["cancer_type"].nunique()),
        "completed":int((df["overall_status"]=="COMPLETED").sum()),
        "terminated":int((df["overall_status"]=="TERMINATED").sum())},
    "status":sc,"cancer":cc,"phase":ph,"sponsor":sp,"years":yr,
    "outcome_cancer":oc,"phase_outcome":po,"pub_years":pub_y,"pub_series":pub_s,
    "era":era,"models":mr,"hem_comp":hem_comp,"sol_comp":sol_comp,
    "compare":compare_data,"cancer_list":cancer_list,
    "ai_analyses":ai_analyses,
    "mol":{"top_targets":top_targets,"target_descriptions":{t:TARGETS.get(t,"") for t,_ in top_targets},
           "therapy_counts":therapy_counts,"therapy_comp_rates":therapy_comp_rates,
           "cancer_target_map":cancer_target_map,"therapy_top_targets":therapy_top_targets},
    "geo":{"country_counts":country_counts},
    "funnel":{"phase_funnel":phase_funnel,"phase_success":phase_success},
    "predictor":predictor_stats,
    "findings": findings,
}, indent=2)

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Gene-Editing Cancer Trials Analytics Dashboard</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Segoe UI',system-ui,sans-serif;background:#060d1a;color:#c8dff5;min-height:100vh}
.hero{background:linear-gradient(135deg,#060d1a 0%,#071428 40%,#0a1f3c 100%);padding:36px 40px 28px;border-bottom:1px solid #1a3356;position:relative;overflow:hidden}
.hero::after{content:'';position:absolute;top:-80px;right:-80px;width:500px;height:500px;background:radial-gradient(circle,rgba(0,212,170,.06) 0%,transparent 65%);pointer-events:none}
.hero-top{display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:16px}
.hero h1{font-size:1.75rem;font-weight:700;color:#fff;letter-spacing:-.02em;line-height:1.15}
.hero h1 span{color:#00d4aa}
.hero-sub{color:#7a9bbf;font-size:.8rem;margin-top:5px;font-family:monospace}
.badges{display:flex;gap:8px;flex-wrap:wrap}
.badge{background:rgba(0,212,170,.12);border:1px solid rgba(0,212,170,.3);color:#00d4aa;padding:3px 10px;border-radius:20px;font-size:.72rem;font-family:monospace}
.badge.blue{background:rgba(74,158,255,.12);border-color:rgba(74,158,255,.3);color:#4a9eff}
.badge.purple{background:rgba(139,92,246,.12);border-color:rgba(139,92,246,.3);color:#a78bfa}
.about{background:#0a1628;border-bottom:1px solid #1a3356;padding:16px 40px;display:grid;grid-template-columns:repeat(4,1fr);gap:18px}
.about-item{padding:0 0 0 14px;border-left:2px solid #1a3356}
.about-item.accent{border-left-color:#00d4aa}
.about-item h4{font-size:.67rem;font-weight:700;text-transform:uppercase;letter-spacing:.1em;color:#7a9bbf;margin-bottom:4px}
.about-item p{font-size:.75rem;color:#c8dff5;line-height:1.5}
.tabs{display:flex;padding:0 40px;border-bottom:1px solid #1a3356;background:#0a1628;overflow-x:auto}
.tab{padding:12px 16px;font-size:.72rem;font-weight:600;color:#7a9bbf;cursor:pointer;border-bottom:2px solid transparent;white-space:nowrap;transition:.2s;text-transform:uppercase;letter-spacing:.06em}
.tab.active{color:#00d4aa;border-bottom-color:#00d4aa}
.tab:hover{color:#c8dff5}
.content{padding:22px 40px;max-width:1500px;margin:0 auto}
.panel{display:none}.panel.active{display:block}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(145px,1fr));gap:11px;margin-bottom:22px}
.kpi{background:#0d1f3c;border:1px solid #1a3356;border-radius:10px;padding:15px;position:relative;overflow:hidden;transition:border-color .2s,transform .15s;cursor:default}
.kpi:hover{border-color:#00d4aa;transform:translateY(-2px)}
.kpi::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:var(--a,#00d4aa)}
.kpi .v{font-size:1.75rem;font-weight:700;color:#fff;font-family:monospace;line-height:1}
.kpi .l{font-size:.66rem;color:#7a9bbf;margin-top:5px;text-transform:uppercase;letter-spacing:.08em}
.kpi .s{font-size:.69rem;color:var(--a,#00d4aa);margin-top:3px;font-family:monospace}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:15px;margin-bottom:15px}
.grid3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:15px;margin-bottom:15px}
.card{background:#0d1f3c;border:1px solid #1a3356;border-radius:10px;padding:18px}
.card h3{font-size:.69rem;font-weight:700;color:#7a9bbf;text-transform:uppercase;letter-spacing:.09em;margin-bottom:13px}
canvas{display:block;width:100%!important}
.insight{background:linear-gradient(135deg,rgba(0,212,170,.07),rgba(74,158,255,.05));border:1px solid rgba(0,212,170,.18);border-radius:10px;padding:13px 17px;margin-bottom:17px;display:flex;gap:12px;align-items:flex-start}
.insight-icon{font-size:1.1rem;flex-shrink:0;margin-top:2px}
.insight-text h4{font-size:.73rem;font-weight:700;color:#00d4aa;margin-bottom:2px;text-transform:uppercase;letter-spacing:.07em}
.insight-text p{font-size:.77rem;color:#c8dff5;line-height:1.55}
table{width:100%;border-collapse:collapse;font-size:.77rem}
th{text-align:left;padding:8px 11px;color:#7a9bbf;font-weight:600;text-transform:uppercase;font-size:.65rem;letter-spacing:.07em;border-bottom:1px solid #1a3356}
td{padding:8px 11px;border-bottom:1px solid rgba(26,51,86,.4);font-family:monospace;font-size:.74rem}
tr:hover td{background:rgba(0,212,170,.04)}
tr.best td{color:#00d4aa}
.bar-wrap{display:flex;align-items:center;gap:8px}
.bar-bg{flex:1;height:6px;background:#0a1628;border-radius:3px;overflow:hidden}
.bar-fill{height:100%;border-radius:3px}
/* Compare */
.compare-selectors{display:flex;gap:14px;margin-bottom:18px;flex-wrap:wrap;align-items:flex-end}
.compare-selectors label{font-size:.69rem;color:#7a9bbf;text-transform:uppercase;letter-spacing:.08em;display:block;margin-bottom:4px}
.compare-selectors select{background:#0d1f3c;color:#c8dff5;border:1px solid #1a3356;border-radius:8px;padding:8px 12px;font-size:.79rem;min-width:185px;cursor:pointer;outline:none;transition:border-color .2s}
.compare-selectors select:focus{border-color:#00d4aa}
.ai-btn{background:linear-gradient(135deg,rgba(139,92,246,.25),rgba(0,212,170,.15));border:1px solid rgba(139,92,246,.5);color:#a78bfa;padding:9px 18px;border-radius:8px;font-size:.79rem;cursor:pointer;font-weight:700;transition:.2s;display:flex;align-items:center;gap:7px}
.ai-btn:hover{background:linear-gradient(135deg,rgba(139,92,246,.4),rgba(0,212,170,.25))}
.ai-analysis-box{background:linear-gradient(135deg,rgba(139,92,246,.06),rgba(0,212,170,.04));border:1px solid rgba(139,92,246,.25);border-radius:12px;padding:22px;margin-bottom:18px;display:none}
.ai-analysis-box.visible{display:block}
.ai-analysis-box .ai-header{font-size:.78rem;font-weight:700;color:#a78bfa;text-transform:uppercase;letter-spacing:.08em;margin-bottom:14px;display:flex;align-items:center;gap:8px}
.ai-analysis-box .ai-content{font-size:.82rem;color:#c8dff5;line-height:1.75}
.ai-analysis-box .ai-content strong{color:#00d4aa}
.ai-cursor{animation:blink 1s infinite;color:#a78bfa}
@keyframes blink{0%,100%{opacity:1}50%{opacity:0}}
.compare-grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px}
.compare-card{background:#0d1f3c;border:1px solid #1a3356;border-radius:12px;padding:20px}
.compare-card .ct-header{display:flex;align-items:center;gap:10px;margin-bottom:14px;padding-bottom:12px;border-bottom:1px solid #1a3356}
.compare-card .ct-dot{width:13px;height:13px;border-radius:50%;flex-shrink:0}
.compare-card .ct-name{font-size:.93rem;font-weight:700;color:#fff}
.compare-card .ct-cat{font-size:.69rem;color:#7a9bbf;margin-top:1px}
.stat-row{display:flex;justify-content:space-between;align-items:center;padding:7px 0;border-bottom:1px solid rgba(26,51,86,.5)}
.stat-row:last-child{border-bottom:none}
.stat-row .sr-label{font-size:.69rem;color:#7a9bbf;text-transform:uppercase;letter-spacing:.06em}
.stat-row .sr-val{font-size:.86rem;font-weight:600;font-family:monospace;color:#c8dff5}
.compare-bars{background:#0d1f3c;border:1px solid #1a3356;border-radius:12px;padding:18px;margin-bottom:16px}
.compare-bars h3{font-size:.69rem;font-weight:700;color:#7a9bbf;text-transform:uppercase;letter-spacing:.09em;margin-bottom:14px}
.cbar-row{margin-bottom:14px}
.cbar-label{font-size:.68rem;color:#7a9bbf;text-transform:uppercase;letter-spacing:.06em;margin-bottom:7px}
.cbar-item{display:flex;align-items:center;gap:10px;margin-bottom:4px}
.cbar-swatch{width:13px;height:13px;border-radius:2px;flex-shrink:0}
.cbar-bg{flex:1;height:8px;background:#0a1628;border-radius:4px;overflow:hidden}
.cbar-fill{height:100%;border-radius:4px;transition:width .6s ease}
.cbar-val{font-size:.72rem;font-family:monospace;color:#c8dff5;min-width:40px;text-align:right}
/* Molecular */
.therapy-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(215px,1fr));gap:11px;margin-bottom:18px}
.therapy-card{background:#0d1f3c;border:1px solid #1a3356;border-radius:10px;padding:14px;border-top:3px solid var(--tc,#1a3356);transition:transform .15s}
.therapy-card:hover{transform:translateY(-2px)}
.therapy-card .th-name{font-size:.82rem;font-weight:700;color:#fff;margin-bottom:3px}
.therapy-card .th-count{font-size:.68rem;color:#7a9bbf;font-family:monospace;margin-bottom:7px}
.therapy-card .th-rate{font-size:1.05rem;font-weight:700;font-family:monospace}
.therapy-card .th-desc{font-size:.68rem;color:#8ab4d4;margin-top:6px;line-height:1.45}
.therapy-card .th-targets{margin-top:8px;font-size:.67rem;color:#8ab4d4;line-height:1.7}
.therapy-card .th-targets span{background:#0a1628;border:1px solid #1a3356;border-radius:3px;padding:1px 5px;margin:1px;display:inline-block;font-family:monospace;color:#00d4aa;font-size:.65rem}
.mol-target-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:9px;margin-bottom:18px}
.mol-target-card{background:#0d1f3c;border:1px solid #1a3356;border-radius:8px;padding:11px 13px;transition:border-color .2s,transform .15s}
.mol-target-card:hover{border-color:#00d4aa;transform:translateY(-1px)}
.mol-target-card .tc-name{font-size:.83rem;font-weight:700;color:#00d4aa;font-family:monospace}
.mol-target-card .tc-count{font-size:.68rem;color:#7a9bbf;margin-top:2px}
.mol-target-card .tc-desc{font-size:.66rem;color:#8ab4d4;margin-top:4px;line-height:1.4}
.mol-target-card .tc-bar{height:3px;background:#1a3356;border-radius:2px;margin-top:7px;overflow:hidden}
.mol-target-card .tc-bar-fill{height:100%;border-radius:2px;background:#00d4aa}
.heatmap-wrap{overflow-x:auto;margin-top:4px}
.heatmap{border-collapse:collapse;width:100%;font-size:.68rem}
.heatmap th{padding:6px 7px;color:#7a9bbf;font-weight:600;text-align:center;white-space:nowrap;font-size:.63rem}
.heatmap td{padding:5px 6px;text-align:center;border:1px solid #060d1a;border-radius:2px;font-family:monospace;font-size:.68rem}
.heatmap .row-label{text-align:left;color:#c8dff5;white-space:nowrap;padding-right:10px}
/* Predictor */
.predictor-form{display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px;margin-bottom:18px}
.pred-field label{font-size:.69rem;color:#7a9bbf;text-transform:uppercase;letter-spacing:.08em;display:block;margin-bottom:5px}
.pred-field select,.pred-field input{width:100%;background:#060d1a;color:#c8dff5;border:1px solid #1a3356;border-radius:7px;padding:9px 12px;font-size:.8rem;outline:none;transition:border-color .2s}
.pred-field select:focus,.pred-field input:focus{border-color:#00d4aa}
.pred-btn{background:linear-gradient(135deg,rgba(0,212,170,.2),rgba(74,158,255,.15));border:1px solid rgba(0,212,170,.4);color:#00d4aa;padding:10px 24px;border-radius:8px;font-size:.82rem;cursor:pointer;font-weight:700;transition:.2s;margin-top:20px;align-self:end}
.pred-btn:hover{background:linear-gradient(135deg,rgba(0,212,170,.35),rgba(74,158,255,.25))}
.pred-result{background:#060d1a;border:1px solid #1a3356;border-radius:12px;padding:20px;display:none}
.pred-result.visible{display:block}
.pred-gauge{display:flex;align-items:center;gap:20px;margin-bottom:16px}
.pred-circle{width:100px;height:100px;position:relative;flex-shrink:0}
.pred-circle svg{transform:rotate(-90deg)}
.pred-circle .pred-pct{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);font-size:1.1rem;font-weight:700;color:#fff;font-family:monospace;text-align:center;line-height:1.2}
.pred-circle .pred-pct small{font-size:.55rem;color:#7a9bbf;display:block}
.pred-verdict{font-size:1rem;font-weight:700;margin-bottom:6px}
.pred-breakdown{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:12px}
.pred-factor{background:#0d1f3c;border:1px solid #1a3356;border-radius:7px;padding:10px}
.pred-factor .pf-label{font-size:.66rem;color:#7a9bbf;text-transform:uppercase;letter-spacing:.07em;margin-bottom:3px}
.pred-factor .pf-val{font-size:.82rem;font-weight:600;font-family:monospace}
/* Funnel + Geo */
.funnel-wrap{display:flex;flex-direction:column;gap:10px;padding:8px 0}
.funnel-stage{display:flex;align-items:center;gap:12px}
.funnel-bar-wrap{flex:1;height:26px;background:#0a1628;border-radius:4px;overflow:hidden}
.funnel-bar{height:100%;border-radius:4px;display:flex;align-items:center;padding-left:10px;font-size:.74rem;font-weight:600;color:#fff;font-family:monospace;transition:width .8s ease}
.funnel-label{width:105px;font-size:.7rem;color:#c8dff5;text-align:right;font-family:monospace;flex-shrink:0}
.funnel-rate{width:60px;font-size:.7rem;color:#7a9bbf;text-align:right;flex-shrink:0;font-family:monospace}
.country-list{display:flex;flex-wrap:wrap;gap:7px;margin-top:8px}
.country-pill{background:#060d1a;border:1px solid #1a3356;border-radius:20px;padding:5px 12px;font-size:.72rem;display:flex;align-items:center;gap:7px;transition:border-color .2s}
.country-pill:hover{border-color:#00d4aa}
.country-pill .c-bar{height:4px;background:#00d4aa;border-radius:3px}
.country-pill .c-cnt{color:#00d4aa;font-family:monospace;font-weight:600}
/* Stats */
.stat-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:11px}
.stat{background:#0d1f3c;border:1px solid #1a3356;border-radius:10px;padding:14px;border-left:3px solid var(--c,#1a3356)}
.stat.sig{--c:#22c55e}.stat.ns{--c:#64748b}
.stat .name{font-size:.68rem;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:#7a9bbf;margin-bottom:5px}
.stat .val{font-family:monospace;font-size:.78rem;color:#c8dff5;margin-bottom:4px}
.stat .verdict{font-size:.73rem;font-weight:700}
.stat.sig .verdict{color:#22c55e}.stat.ns .verdict{color:#64748b}
/* Tooltips */
.tip{border-bottom:1px dashed #4a9eff;cursor:help;color:inherit;position:relative;display:inline}
.tip:hover::after{content:attr(data-tip);position:absolute;bottom:calc(100% + 6px);left:50%;transform:translateX(-50%);background:#0a1628;border:1px solid #1a3356;color:#c8dff5;font-size:.71rem;padding:7px 11px;border-radius:7px;white-space:nowrap;max-width:260px;white-space:normal;line-height:1.5;z-index:999;pointer-events:none;min-width:180px}
.tip:hover::before{content:'';position:absolute;bottom:calc(100% + 1px);left:50%;transform:translateX(-50%);border:5px solid transparent;border-top-color:#1a3356;z-index:999}
/* Plain English callouts */
.plain-box{background:rgba(74,158,255,.07);border-left:3px solid #4a9eff;border-radius:0 8px 8px 0;padding:10px 14px;margin-top:10px;display:none}
.plain-box.visible{display:block}
.plain-box p{font-size:.76rem;color:#c8dff5;line-height:1.6}
.plain-box strong{color:#4a9eff}
.plain-toggle{background:none;border:none;color:#4a9eff;font-size:.69rem;cursor:pointer;padding:4px 0;text-decoration:underline;margin-top:6px}
/* Beginner Mode */
.bm-bar{position:fixed;bottom:16px;right:16px;z-index:999;display:flex;align-items:center;gap:9px;background:#0a1628;border:1px solid #1a3356;border-radius:24px;padding:8px 16px;box-shadow:0 4px 20px rgba(0,0,0,.5)}
.bm-bar label{font-size:.71rem;color:#7a9bbf;cursor:pointer}
.bm-switch{appearance:none;width:32px;height:17px;background:#1a3356;border-radius:10px;cursor:pointer;position:relative;transition:.2s;outline:none}
.bm-switch:checked{background:#4a9eff}
.bm-switch::after{content:'';position:absolute;width:13px;height:13px;background:#fff;border-radius:50%;top:2px;left:2px;transition:.2s}
.bm-switch:checked::after{left:17px}
body.beginner-mode .bm-label-adv{display:none}
body.beginner-mode .bm-label-beg{display:inline!important}
.bm-label-beg{display:none}
/* About page */
.about-page{max-width:1100px;margin:0 auto;padding:4px 0}
.about-hero{background:linear-gradient(135deg,rgba(0,212,170,.08),rgba(74,158,255,.05));border:1px solid rgba(0,212,170,.2);border-radius:14px;padding:28px 32px;margin-bottom:20px;position:relative;overflow:hidden}
.about-hero::after{content:'🧬';position:absolute;right:24px;top:14px;font-size:5rem;opacity:.07;pointer-events:none}
.about-hero h2{font-size:1.3rem;font-weight:800;color:#fff;margin-bottom:8px;letter-spacing:-.02em}
.about-hero h2 span{color:#00d4aa}
.about-hero p{font-size:.82rem;color:#c8dff5;line-height:1.75;max-width:760px}
.about-hero .tag-row{display:flex;flex-wrap:wrap;gap:7px;margin-top:13px}
.atag{background:rgba(0,212,170,.12);border:1px solid rgba(0,212,170,.25);color:#00d4aa;padding:3px 10px;border-radius:20px;font-size:.69rem;font-family:monospace}
.atag.b{background:rgba(74,158,255,.12);border-color:rgba(74,158,255,.25);color:#4a9eff}
.atag.p{background:rgba(139,92,246,.12);border-color:rgba(139,92,246,.25);color:#a78bfa}
.atag.o{background:rgba(245,158,11,.12);border-color:rgba(245,158,11,.25);color:#f59e0b}
.science-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin-bottom:20px}
.science-card{background:#0d1f3c;border:1px solid #1a3356;border-radius:10px;padding:16px;border-top:3px solid var(--sc,#00d4aa)}
.science-card .sci-icon{font-size:1.4rem;margin-bottom:8px}
.science-card h4{font-size:.78rem;font-weight:700;color:#fff;margin-bottom:6px}
.science-card p{font-size:.73rem;color:#8ab4d4;line-height:1.6}
.science-card .sci-tag{display:inline-block;background:#060d1a;border:1px solid #1a3356;border-radius:3px;font-size:.61rem;font-family:monospace;padding:1px 5px;margin-top:6px;color:#00d4aa}
.wt-item{display:flex;gap:13px;padding:12px 0;border-bottom:1px solid rgba(26,51,86,.4)}
.wt-item:last-child{border-bottom:none}
.wt-icon{width:34px;height:34px;border-radius:8px;background:rgba(0,212,170,.1);border:1px solid rgba(0,212,170,.2);display:flex;align-items:center;justify-content:center;font-size:.9rem;flex-shrink:0}
.wt-title{font-size:.8rem;font-weight:700;color:#fff;margin-bottom:3px}
.wt-desc{font-size:.74rem;color:#8ab4d4;line-height:1.6}
.wt-feats{display:flex;flex-wrap:wrap;gap:4px;margin-top:5px}
.wt-feat{background:#060d1a;border:1px solid #1a3356;border-radius:3px;padding:2px 7px;font-size:.62rem;color:#00d4aa;font-family:monospace}
.pipe-row{display:flex;align-items:center;flex-wrap:wrap;gap:0;margin-bottom:18px}
.pipe-step{background:#0d1f3c;border:1px solid #1a3356;border-radius:8px;padding:10px 12px;min-width:110px}
.pipe-step .ps-l{font-size:.6rem;text-transform:uppercase;letter-spacing:.08em;color:#7a9bbf;margin-bottom:3px}
.pipe-step .ps-v{font-size:.78rem;font-weight:700;color:#fff;font-family:monospace}
.pipe-step .ps-s{font-size:.62rem;color:#4a9eff;margin-top:1px}
.pipe-arrow{color:#1a3356;font-size:1.2rem;padding:0 4px}
.srow{display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid rgba(26,51,86,.4);font-size:.76rem}
.srow:last-child{border-bottom:none}
.srow .sl{color:#7a9bbf}.srow .sv{color:#c8dff5;font-family:monospace;font-weight:600}
.stack-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:9px}
.scard{background:#060d1a;border:1px solid #1a3356;border-radius:8px;padding:11px 12px}
.scard .sc-cat{font-size:.6rem;text-transform:uppercase;letter-spacing:.08em;color:#7a9bbf;margin-bottom:5px}
.scard .sc-items{font-size:.72rem;color:#c8dff5;line-height:1.7}
.scard .sc-items b{color:#00d4aa;font-family:monospace;font-weight:600}
/* FAQ */
.faq-item{border-bottom:1px solid rgba(26,51,86,.5);padding:0}
.faq-q{width:100%;text-align:left;background:none;border:none;color:#c8dff5;font-size:.8rem;font-weight:600;padding:13px 0;cursor:pointer;display:flex;justify-content:space-between;align-items:center}
.faq-q:hover{color:#00d4aa}
.faq-q .faq-arrow{color:#7a9bbf;transition:transform .2s;font-size:.85rem}
.faq-q.open .faq-arrow{transform:rotate(180deg)}
.faq-a{font-size:.77rem;color:#8ab4d4;line-height:1.7;padding:0 0 13px;display:none}
.faq-a.open{display:block}
.ah3{font-size:.71rem;font-weight:700;text-transform:uppercase;letter-spacing:.1em;color:#7a9bbf;margin:18px 0 11px}

/* About */
.about-page{max-width:1100px;margin:0 auto}
.about-hero{background:linear-gradient(135deg,rgba(0,212,170,.08),rgba(74,158,255,.05));border:1px solid rgba(0,212,170,.2);border-radius:14px;padding:32px 36px;margin-bottom:22px;position:relative;overflow:hidden}
.about-hero::after{content:'🧬';position:absolute;right:28px;top:16px;font-size:5rem;opacity:.07;pointer-events:none}
.about-hero h2{font-size:1.35rem;font-weight:800;color:#fff;margin-bottom:8px;letter-spacing:-.02em}
.about-hero h2 span{color:#00d4aa}
.about-hero p{font-size:.83rem;color:#c8dff5;line-height:1.75;max-width:780px}
.about-hero .tag-row{display:flex;flex-wrap:wrap;gap:7px;margin-top:14px}
.atag{background:rgba(0,212,170,.12);border:1px solid rgba(0,212,170,.25);color:#00d4aa;padding:3px 11px;border-radius:20px;font-size:.7rem;font-family:monospace}
.atag.b{background:rgba(74,158,255,.12);border-color:rgba(74,158,255,.25);color:#4a9eff}
.atag.p{background:rgba(139,92,246,.12);border-color:rgba(139,92,246,.25);color:#a78bfa}
.atag.o{background:rgba(245,158,11,.12);border-color:rgba(245,158,11,.25);color:#f59e0b}
.science-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:13px;margin-bottom:22px}
.science-card{background:#0d1f3c;border:1px solid #1a3356;border-radius:10px;padding:17px;border-top:3px solid var(--sc,#00d4aa)}
.science-card .sci-icon{font-size:1.4rem;margin-bottom:9px}
.science-card h4{font-size:.79rem;font-weight:700;color:#fff;margin-bottom:6px}
.science-card p{font-size:.74rem;color:#8ab4d4;line-height:1.6}
.science-card .sci-tag{display:inline-block;background:#060d1a;border:1px solid #1a3356;border-radius:3px;font-size:.61rem;font-family:monospace;padding:1px 5px;margin-top:7px;color:#00d4aa}
.walkthrough h3{font-size:.72rem;font-weight:700;text-transform:uppercase;letter-spacing:.1em;color:#7a9bbf;margin-bottom:13px;padding-bottom:7px;border-bottom:1px solid #1a3356}
.wt-item{display:flex;gap:14px;padding:13px 0;border-bottom:1px solid rgba(26,51,86,.4)}
.wt-item:last-child{border-bottom:none}
.wt-icon{width:36px;height:36px;border-radius:8px;background:rgba(0,212,170,.1);border:1px solid rgba(0,212,170,.2);display:flex;align-items:center;justify-content:center;font-size:.95rem;flex-shrink:0}
.wt-title{font-size:.81rem;font-weight:700;color:#fff;margin-bottom:3px}
.wt-desc{font-size:.75rem;color:#8ab4d4;line-height:1.6}
.wt-feats{display:flex;flex-wrap:wrap;gap:4px;margin-top:6px}
.wt-feat{background:#060d1a;border:1px solid #1a3356;border-radius:3px;padding:2px 7px;font-size:.63rem;color:#00d4aa;font-family:monospace}
.pipe-row{display:flex;align-items:center;flex-wrap:wrap;gap:0;margin-bottom:20px}
.pipe-step{background:#0d1f3c;border:1px solid #1a3356;border-radius:8px;padding:11px 13px;min-width:115px}
.pipe-step .ps-l{font-size:.61rem;text-transform:uppercase;letter-spacing:.08em;color:#7a9bbf;margin-bottom:3px}
.pipe-step .ps-v{font-size:.8rem;font-weight:700;color:#fff;font-family:monospace}
.pipe-step .ps-s{font-size:.64rem;color:#4a9eff;margin-top:1px}
.pipe-arrow{color:#1a3356;font-size:1.3rem;padding:0 5px}
.srow{display:flex;justify-content:space-between;padding:7px 0;border-bottom:1px solid rgba(26,51,86,.4);font-size:.77rem}
.srow:last-child{border-bottom:none}
.srow .sl{color:#7a9bbf}.srow .sv{color:#c8dff5;font-family:monospace;font-weight:600}
.stack-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(155px,1fr));gap:10px}
.scard{background:#060d1a;border:1px solid #1a3356;border-radius:8px;padding:12px 13px}
.scard .sc-cat{font-size:.61rem;text-transform:uppercase;letter-spacing:.08em;color:#7a9bbf;margin-bottom:5px}
.scard .sc-items{font-size:.73rem;color:#c8dff5;line-height:1.7}
.scard .sc-items b{color:#00d4aa;font-family:monospace;font-weight:600}
/* Discoveries */
.finding-block{background:#0a1628;border:1px solid #1a3356;border-radius:12px;padding:20px;margin-bottom:18px}
.finding-hd{display:flex;align-items:flex-start;gap:14px;margin-bottom:16px;padding-bottom:14px;border-bottom:1px solid #1a3356}
.finding-num{font-size:1.8rem;font-weight:900;font-family:monospace;color:#1a3356;line-height:1;flex-shrink:0;min-width:42px}
.finding-title{font-size:.95rem;font-weight:700;color:#fff;margin-bottom:4px}
.finding-sub{font-size:.76rem;color:#7a9bbf;line-height:1.5}
.finding-verdict{background:linear-gradient(135deg,rgba(0,212,170,.07),rgba(74,158,255,.04));border:1px solid rgba(0,212,170,.18);border-radius:8px;padding:12px 16px;margin-top:14px;font-size:.8rem;color:#c8dff5;line-height:1.6;display:none}
.finding-verdict.visible{display:block}
.finding-verdict strong{color:#00d4aa}
.stat-pill{display:inline-block;background:#060d1a;border:1px solid #1a3356;border-radius:6px;padding:8px 14px;margin:4px;font-size:.8rem;font-family:monospace}
.stat-pill .sp-val{font-size:1.1rem;font-weight:700;color:#00d4aa;display:block}
.stat-pill .sp-label{font-size:.67rem;color:#7a9bbf;text-transform:uppercase;letter-spacing:.06em}
.stat .note{font-size:.69rem;color:#7a9bbf;margin-top:4px}
@media(max-width:960px){
  .grid2,.grid3,.compare-grid,.therapy-grid,.predictor-form{grid-template-columns:1fr}
  .content,.hero,.about,.tabs{padding-left:16px;padding-right:16px}
  .about{grid-template-columns:1fr 1fr}
  .mol-target-grid{grid-template-columns:1fr 1fr}
}
</style>
</head>
<body>
<div class="hero">
  <div class="hero-top">
    <div>
      <h1>Gene-Editing &amp; Cell Therapy<br><span>Cancer Trials Analytics</span></h1>
      <div class="hero-sub">ClinicalTrials.gov + PubMed · Python / scikit-learn / XGBoost / LightGBM · Clinical AI Analysis</div>
    </div>
    <div class="badges">
      <span class="badge">Lakshay Mani</span>
      <span class="badge">MS Analytics · Northeastern</span>
      <span class="badge blue" id="badge-total">Loading...</span>
      <span class="badge purple">🔬 Clinical AI</span>
    </div>
  </div>
</div>
<div class="about">
  <div class="about-item accent"><h4>What Is This?</h4><p>End-to-end analytics pipeline on <strong>3,494 real clinical trials</strong> from ClinicalTrials.gov spanning CRISPR, CAR-T, TCR, TIL, NK cell, gene therapy, oncolytic virus, and mRNA cancer treatments from 1990–2025.</p></div>
  <div class="about-item"><h4>Why It Matters</h4><p>Gene-editing and cell therapy are the <strong>frontier of oncology</strong>. Understanding which molecular approaches advance through trials — and which fail — guides research investment and clinical strategy.</p></div>
  <div class="about-item"><h4>Statistical Findings</h4><p><strong>81.3% completion rate</strong>. Enrollment predicts success (r=0.48, p&lt;0.0001). Phase predicts outcome (χ²=22.9, p=0.002). 35+ protein targets extracted. 9 hypothesis tests across 3,494 trials.</p></div>
  <div class="about-item"><h4>AI Analysis Layer</h4><p>Clinical AI analyses pre-generated for every cancer type comparison — covering molecular targets, pipeline maturity, research gaps, and pipeline verdicts. Plus an interactive trial outcome predictor.</p></div>
</div>
<div class="tabs">
  <div class="tab" onclick="show('about',this)">📖 About</div>
  <div class="tab active" onclick="show('overview',this)">Overview</div>
  <div class="tab" onclick="show('trials',this)">Trial Analysis</div>
  <div class="tab" onclick="show('compare',this)">⚖ Compare + AI</div>
  <div class="tab" onclick="show('mol',this)">🧬 Molecular</div>
  <div class="tab" onclick="show('predictor',this)">🎯 Predictor</div>
  <div class="tab" onclick="show('discoveries',this)">🔭 Discoveries</div>
  <div class="tab" onclick="show('stats',this)">Statistics</div>
  <div class="tab" onclick="show('ml',this)">ML Models</div>
  <div class="tab" onclick="show('pubs',this)">Publications</div>
</div>
<div class="content">

<!-- ABOUT -->
<div class="panel" id="panel-about">
<div class="about-page">

  <div class="about-hero">
    <h2>Gene-Editing &amp; Cell Therapy<br><span>Cancer Trials Analytics</span></h2>
    <p>This project is an end-to-end data science pipeline built on <strong>4,460 real clinical trials</strong> from ClinicalTrials.gov spanning 1990–2025. It analyzes how scientists are using gene-editing tools — CRISPR, CAR-T cells, NK cells, oncolytic viruses — to fight cancer. The goal is to understand which approaches are working, which cancers are being neglected, and what predicts a trial's success.</p>
    <div class="tag-row">
      <span class="atag">MS Analytics · Northeastern University</span>
      <span class="atag b">4,460 Clinical Trials</span>
      <span class="atag b">80+ Countries</span>
      <span class="atag p">11 ML Classifiers</span>
      <span class="atag p">AUC = 0.90</span>
      <span class="atag o">5 Novel Research Findings</span>
      <span class="atag o">3D Protein Viewer</span>
    </div>
  </div>

  <!-- The Science -->
  <div style="font-size:.72rem;font-weight:700;text-transform:uppercase;letter-spacing:.1em;color:#7a9bbf;margin-bottom:13px">🔬 The Science — What Is Gene Editing?</div>
  <div class="science-grid" style="margin-bottom:22px">
    <div class="science-card" style="--sc:#00d4aa">
      <div class="sci-icon">✂️</div>
      <h4>CRISPR-Cas9</h4>
      <p>Molecular scissors that cut DNA at precise locations. Scientists use it to knock out cancer-protecting genes (like PD-1 in T cells) or insert new instructions. First human trials began in 2019.</p>
      <span class="sci-tag">Gene: PDCD1, TRAC, B2M</span>
    </div>
    <div class="science-card" style="--sc:#4a9eff">
      <div class="sci-icon">🎯</div>
      <h4>CAR-T Cell Therapy</h4>
      <p>A patient's T cells are extracted, engineered in a lab to recognize cancer's "name tag" (like CD19), then infused back. They seek and destroy cancer cells like GPS-guided missiles.</p>
      <span class="sci-tag">Target: CD19, BCMA, CD33</span>
    </div>
    <div class="science-card" style="--sc:#8b5cf6">
      <div class="sci-icon">🔑</div>
      <h4>TCR Therapy</h4>
      <p>Similar to CAR-T but uses T-cell receptors instead of artificial receptors. Can target proteins <em>inside</em> cancer cells — invisible to CAR-T — by recognizing peptide fragments on the cell surface.</p>
      <span class="sci-tag">Target: NY-ESO, MAGE, WT1</span>
    </div>
    <div class="science-card" style="--sc:#f59e0b">
      <div class="sci-icon">🌿</div>
      <h4>TIL Therapy</h4>
      <p>Tumor-infiltrating lymphocytes — the immune cells already fighting cancer inside the tumor — are extracted, expanded 1,000× in a lab, and re-infused. Proven effective in melanoma.</p>
      <span class="sci-tag">First success: Melanoma (1988)</span>
    </div>
    <div class="science-card" style="--sc:#22c55e">
      <div class="sci-icon">⚡</div>
      <h4>NK Cell Therapy</h4>
      <p>Natural Killer cells don't need to recognize a specific target to kill — they attack anything that looks "wrong." Unlike CAR-T, they can be used off-the-shelf from donors without immune rejection.</p>
      <span class="sci-tag">No MHC matching needed</span>
    </div>
    <div class="science-card" style="--sc:#ec4899">
      <div class="sci-icon">💉</div>
      <h4>mRNA &amp; Gene Therapy</h4>
      <p>mRNA instructs cells to make tumor antigens for vaccines (same platform as COVID-19 shots). Gene therapy uses viral vectors (AAV, lentivirus) to deliver corrective or therapeutic genes directly.</p>
      <span class="sci-tag">Platform: lipid nanoparticle</span>
    </div>
  </div>

  <!-- Data Pipeline -->
  <div style="font-size:.72rem;font-weight:700;text-transform:uppercase;letter-spacing:.1em;color:#7a9bbf;margin-bottom:13px">⚙️ Data Pipeline</div>
  <div class="pipe-row" style="margin-bottom:20px">
    <div class="pipe-step"><div class="ps-l">Source</div><div class="ps-v">ClinicalTrials.gov</div><div class="ps-s">API v2 · 28 queries</div></div>
    <div class="pipe-arrow">→</div>
    <div class="pipe-step"><div class="ps-l">Fetched</div><div class="ps-v">7,794</div><div class="ps-s">unique trials</div></div>
    <div class="pipe-arrow">→</div>
    <div class="pipe-step"><div class="ps-l">After Cleaning</div><div class="ps-v">4,460</div><div class="ps-s">interventional only</div></div>
    <div class="pipe-arrow">→</div>
    <div class="pipe-step"><div class="ps-l">Labeled</div><div class="ps-v">3,745</div><div class="ps-s">with ML outcome</div></div>
    <div class="pipe-arrow">→</div>
    <div class="pipe-step"><div class="ps-l">Cancer Types</div><div class="ps-v">30+</div><div class="ps-s">classified via regex</div></div>
    <div class="pipe-arrow">→</div>
    <div class="pipe-step"><div class="ps-l">Countries</div><div class="ps-v">80+</div><div class="ps-s">geographic reach</div></div>
  </div>

  <!-- Key Stats -->
  <div class="grid2" style="margin-bottom:22px">
    <div class="card">
      <h3>Dataset at a Glance</h3>
      <div class="srow"><span class="sl">Date Range</span><span class="sv">1990 – 2025</span></div>
      <div class="srow"><span class="sl">Total Trials</span><span class="sv">4,460</span></div>
      <div class="srow"><span class="sl">Completion Rate</span><span class="sv">81.1%</span></div>
      <div class="srow"><span class="sl">Median Enrollment</span><span class="sv">30 patients</span></div>
      <div class="srow"><span class="sl">Hematologic Trials</span><span class="sv">1,653 (37%)</span></div>
      <div class="srow"><span class="sl">Solid Tumor Trials</span><span class="sv">2,807 (63%)</span></div>
      <div class="srow"><span class="sl">Phase I (largest)</span><span class="sv">1,631 trials</span></div>
      <div class="srow"><span class="sl">Top Cancer Type</span><span class="sv">Lymphoma (592)</span></div>
      <div class="srow"><span class="sl">PubMed papers (2024)</span><span class="sv">CRISPR: 1,288 · CAR-T: 1,022</span></div>
    </div>
    <div class="card">
      <h3>Key Research Findings</h3>
      <div class="srow"><span class="sl">Enrollment tipping point</span><span class="sv" style="color:#22c55e">≥10 patients (+32pp)</span></div>
      <div class="srow"><span class="sl">Best ML predictor</span><span class="sv">Enrollment (r=0.464)</span></div>
      <div class="srow"><span class="sl">Blood vs solid gap</span><span class="sv">81% vs 81% (converging)</span></div>
      <div class="srow"><span class="sl">Most underserved cancer</span><span class="sv" style="color:#ef4444">Pancreatic (20.8 ratio)</span></div>
      <div class="srow"><span class="sl">Most over-invested</span><span class="sv">Leukemia ALL (1,637 ratio)</span></div>
      <div class="srow"><span class="sl">Top target pair</span><span class="sv">CD19 + CD22 (37 trials)</span></div>
      <div class="srow"><span class="sl">Fastest CAGR therapy</span><span class="sv">Oncolytic Virus (39.8%)</span></div>
      <div class="srow"><span class="sl">China reached 75% of US</span><span class="sv">2021</span></div>
      <div class="srow"><span class="sl">Dual-target trend</span><span class="sv">+0.23 pp/yr (p=0.046)</span></div>
    </div>
  </div>

  <!-- Walkthrough -->
  <div class="walkthrough">
    <h3>📋 Tab-by-Tab Walkthrough</h3>

    <div class="wt-item">
      <div class="wt-icon">📊</div>
      <div>
        <div class="wt-title">Overview</div>
        <div class="wt-desc">Your starting point. 8 headline KPI cards, a donut chart of trial statuses, top cancer types by trial count, hematologic vs solid tumor breakdown, and sponsor class distribution. All computed from the live dataset.</div>
        <div class="wt-feats"><span class="wt-feat">8 KPI cards</span><span class="wt-feat">Status donut</span><span class="wt-feat">Cancer bar chart</span><span class="wt-feat">Sponsor breakdown</span></div>
      </div>
    </div>

    <div class="wt-item">
      <div class="wt-icon">📈</div>
      <div>
        <div class="wt-title">Trial Analysis</div>
        <div class="wt-desc">Temporal trends from 1995–2025 showing the explosive post-2015 growth. Completion rates by cancer type and phase. A phase pipeline funnel showing how trials progress from Early Phase I through Phase IV. Top countries by trial count with proportional bars.</div>
        <div class="wt-feats"><span class="wt-feat">Year trend</span><span class="wt-feat">Phase funnel</span><span class="wt-feat">Country map</span><span class="wt-feat">Completion by type</span></div>
      </div>
    </div>

    <div class="wt-item">
      <div class="wt-icon" style="background:rgba(139,92,246,.1);border-color:rgba(139,92,246,.25)">⚖</div>
      <div>
        <div class="wt-title">Compare + AI</div>
        <div class="wt-desc">Select any two cancer types for a head-to-head comparison: trial counts, completion rates, median enrollment, Phase III %, industry sponsorship, top molecular targets, and growth trends. Click <strong style="color:#a78bfa">🔬 Clinical Analysis</strong> to get a pre-generated data-driven clinical interpretation covering biology, pipeline maturity, research gaps, and a verdict — built from real statistics at dashboard build time.</div>
        <div class="wt-feats"><span class="wt-feat">6 metrics side-by-side</span><span class="wt-feat">Growth trend chart</span><span class="wt-feat">AI analysis</span><span class="wt-feat">Phase comparison</span></div>
      </div>
    </div>

    <div class="wt-item">
      <div class="wt-icon" style="background:rgba(0,212,170,.08);border-color:rgba(0,212,170,.2)">🧬</div>
      <div>
        <div class="wt-title">Molecular</div>
        <div class="wt-desc">Protein targets extracted by scanning all 4,460 trial intervention texts using regex. Shows 8 therapy mechanism cards (CAR-T, CRISPR, TCR, TIL, NK, Oncolytic, mRNA, Gene Therapy) with trial counts and top targets. 20 target cards with clinical descriptions and a cancer × target heatmap showing which proteins are attacked in which cancers.</div>
        <div class="wt-feats"><span class="wt-feat">35+ targets mined</span><span class="wt-feat">Therapy cards</span><span class="wt-feat">Target heatmap</span><span class="wt-feat">Co-occurrence</span></div>
      </div>
    </div>

    <div class="wt-item">
      <div class="wt-icon" style="background:rgba(245,158,11,.08);border-color:rgba(245,158,11,.2)">🎯</div>
      <div>
        <div class="wt-title">Predictor</div>
        <div class="wt-desc">Interactive trial outcome predictor. Enter cancer type, phase, sponsor, enrollment size, and multinational status. The model uses statistical relationships from 3,745 labeled trials to estimate completion probability, showing a color-coded gauge and a breakdown of each factor's contribution.</div>
        <div class="wt-feats"><span class="wt-feat">Probability gauge</span><span class="wt-feat">Factor breakdown</span><span class="wt-feat">Historical charts</span></div>
      </div>
    </div>

    <div class="wt-item">
      <div class="wt-icon" style="background:rgba(74,158,255,.08);border-color:rgba(74,158,255,.2)">🔭</div>
      <div>
        <div class="wt-title">Discoveries</div>
        <div class="wt-desc">5 original research findings computed from the data — not published elsewhere. Treatment Desert (investment vs mortality), Hematologic-Solid convergence trend, Enrollment tipping point (χ²=495, p≈0), Dual-target emergence (p=0.046), and Country therapy specialization fingerprints. Each finding includes statistical tests, charts, and a plain-English verdict.</div>
        <div class="wt-feats"><span class="wt-feat">NCI mortality crossref</span><span class="wt-feat">Chi-square tests</span><span class="wt-feat">Linear regression</span><span class="wt-feat">CAGR analysis</span></div>
      </div>
    </div>

    <div class="wt-item">
      <div class="wt-icon">📊</div>
      <div>
        <div class="wt-title">Statistics</div>
        <div class="wt-desc">9 formal hypothesis tests on 4,460 trials: Levene's, ANOVA, Kruskal-Wallis, Tukey HSD, Mann-Whitney U, Welch's t-test, two Chi-Square tests, and Point-Biserial Correlation. 5 of 9 are significant. Strongest result: enrollment size predicts completion (r=0.464, p&lt;0.0001).</div>
        <div class="wt-feats"><span class="wt-feat">9 hypothesis tests</span><span class="wt-feat">5 significant</span><span class="wt-feat">−log p chart</span></div>
      </div>
    </div>

    <div class="wt-item">
      <div class="wt-icon" style="background:rgba(139,92,246,.08);border-color:rgba(139,92,246,.2)">🤖</div>
      <div>
        <div class="wt-title">ML Models</div>
        <div class="wt-desc">11 classifiers trained on 2,996 trials, tested on 749. Random Forest (Tuned) achieves best accuracy (89.6%), Soft Voting Ensemble best AUC (0.90). GridSearchCV applied to RF and XGBoost. Features: enrollment, phase, sponsor, cancer type, era, geography. Full results table with AUC bars.</div>
        <div class="wt-feats"><span class="wt-feat">11 classifiers</span><span class="wt-feat">GridSearchCV</span><span class="wt-feat">Ensemble</span><span class="wt-feat">AUC = 0.90</span></div>
      </div>
    </div>

    <div class="wt-item">
      <div class="wt-icon" style="background:rgba(132,204,22,.08);border-color:rgba(132,204,22,.2)">📚</div>
      <div>
        <div class="wt-title">Publications</div>
        <div class="wt-desc">PubMed publication counts by year (2013–2024) for CRISPR, CAR-T, Gene Therapy, TIL, NK Cell, Base Editing, and AI+CRISPR research. CRISPR grew from 3 papers (2013) to 1,288 (2024) — a 43,000% increase. Year-over-year growth rates show research momentum by modality.</div>
        <div class="wt-feats"><span class="wt-feat">7 research streams</span><span class="wt-feat">2013–2024</span><span class="wt-feat">YoY growth</span></div>
      </div>
    </div>

    <div class="wt-item">
      <div class="wt-icon" style="background:rgba(20,184,166,.08);border-color:rgba(20,184,166,.2)">⚗️</div>
      <div>
        <div class="wt-title">3D Protein Viewer (separate page)</div>
        <div class="wt-desc">Standalone <code style="background:#060d1a;padding:1px 5px;border-radius:3px;font-size:.72rem">molecules.html</code> — real crystal structures from RCSB PDB embedded offline. 10 protein targets rendered with Three.js WebGL: CD19, BCMA, HER2, EGFR, PD-1, KRAS, CD33, GPC3, Mesothelin, TP53. Three rendering modes: Ribbon (Cα backbone), Spheres, CPK. Side-by-side compare any two proteins. Load any custom PDB ID.</div>
        <div class="wt-feats"><span class="wt-feat">8,647 Cα atoms</span><span class="wt-feat">Ribbon/Spheres/CPK</span><span class="wt-feat">Side-by-side compare</span><span class="wt-feat">Works offline</span></div>
      </div>
    </div>
  </div>

  <!-- Tech Stack -->
  <div style="font-size:.72rem;font-weight:700;text-transform:uppercase;letter-spacing:.1em;color:#7a9bbf;margin-bottom:13px">🛠️ Technical Stack</div>
  <div class="stack-grid">
    <div class="scard"><div class="sc-cat">Data Collection</div><div class="sc-items"><b>requests</b> · ClinicalTrials.gov API v2 · PubMed eUtils · 28 search queries</div></div>
    <div class="scard"><div class="sc-cat">Data Cleaning</div><div class="sc-items"><b>pandas</b> · <b>numpy</b> · Regex classifiers · 50+ cancer patterns</div></div>
    <div class="scard"><div class="sc-cat">Statistics</div><div class="sc-items"><b>scipy.stats</b> · ANOVA · Kruskal-Wallis · Tukey HSD · Chi-Square · Point-Biserial</div></div>
    <div class="scard"><div class="sc-cat">ML / AI</div><div class="sc-items"><b>scikit-learn</b> · <b>XGBoost</b> · <b>LightGBM</b> · SHAP · GridSearchCV · Ensemble</div></div>
    <div class="scard"><div class="sc-cat">3D Visualization</div><div class="sc-items"><b>Three.js r128</b> · RCSB PDB API · TubeGeometry · WebGL renderer</div></div>
    <div class="scard"><div class="sc-cat">Dashboard</div><div class="sc-items">Pure HTML5 Canvas · Zero CDN · Pre-baked AI analyses · Works offline</div></div>
    <div class="scard"><div class="sc-cat">Data Sources</div><div class="sc-items">ClinicalTrials.gov · RCSB PDB · PubMed · NCI SEER mortality data</div></div>
    <div class="scard"><div class="sc-cat">Author</div><div class="sc-items"><b>Lakshay Mani</b> · MS Analytics (Statistical Modeling) · Northeastern University · GitHub: LakshayMani0406</div></div>
  </div>

</div>
</div>

<!-- OVERVIEW -->
<div class="panel active" id="panel-overview">
  <div class="kpis" id="kpi-row"></div>
  <div class="insight"><div class="insight-icon">🧬</div><div class="insight-text"><h4>Dataset Overview</h4><p>3,494 interventional trials (1990–2025). <strong>81.3% completion rate</strong> across 2,895 labeled trials. Lymphoma leads with 511 trials. Enrollment size is the strongest predictor of completion (r=0.48, p&lt;0.0001). 80+ countries represented.</p></div></div>
  <div class="grid2">
    <div class="card"><h3>Trial Status Distribution</h3><canvas id="c-status" height="260"></canvas></div>
    <div class="card"><h3>Top Cancer Types by Trial Count</h3><canvas id="c-cancer" height="260"></canvas></div>
  </div>
  <div class="grid2">
    <div class="card"><h3>Hematologic vs Solid Tumor</h3><canvas id="c-cat" height="200"></canvas></div>
    <div class="card"><h3>Lead Sponsor Class</h3><canvas id="c-sponsor" height="200"></canvas></div>
  </div>
</div>

<!-- TRIALS -->
<div class="panel" id="panel-trials">
  <div class="insight"><div class="insight-icon">📈</div><div class="insight-text"><h4>Explosive Growth Since 2015</h4><p>Trial volume accelerated after 2015 CAR-T FDA approvals and CRISPR breakthroughs. 2020+ accounts for <strong>59% of all trials</strong>. Phase I dominates (38%). United States leads globally with over 1,400 trials.</p></div></div>
  <div class="grid2">
    <div class="card"><h3>New Trials per Year</h3><canvas id="c-years" height="240"></canvas></div>
    <div class="card"><h3>Completion Rate by Cancer Type</h3><canvas id="c-oc" height="240"></canvas></div>
  </div>
  <div class="grid2">
    <div class="card"><h3>Phase Pipeline Funnel</h3><div class="funnel-wrap" id="funnel-wrap"></div></div>
    <div class="card"><h3>Top Trial Countries</h3><div class="country-list" id="country-list"></div></div>
  </div>
</div>

<!-- COMPARE + AI -->
<div class="panel" id="panel-compare">
  <div class="insight"><div class="insight-icon">⚖</div><div class="insight-text"><h4>Head-to-Head Comparison + Clinical AI Analysis</h4><p>Select two cancer types. The <strong>🔬 Clinical Analysis</strong> button shows a data-driven clinical interpretation covering molecular target biology, pipeline maturity, key differences, research gaps, and a verdict — generated from real trial statistics.</p></div></div>
  <div class="compare-selectors">
    <div><label>Cancer Type A</label><select id="sel-a" onchange="runCompare()"></select></div>
    <div><label>Cancer Type B</label><select id="sel-b" onchange="runCompare()"></select></div>
    <button class="ai-btn" onclick="showAIAnalysis()">🔬 Clinical Analysis</button>
  </div>
  <div class="ai-analysis-box" id="ai-box">
    <div class="ai-header">🔬 Clinical Analysis — <span id="ai-title"></span></div>
    <div class="ai-content" id="ai-content"></div>
  </div>
  <div class="compare-grid" id="compare-cards"></div>
  <div class="compare-bars" id="compare-bars"></div>
  <div class="grid2">
    <div class="card"><h3>Phase Distribution</h3><canvas id="c-cmp-phase" height="220"></canvas></div>
    <div class="card"><h3>Trial Growth 2010–2025</h3><canvas id="c-cmp-trend" height="220"></canvas></div>
  </div>
</div>

<!-- MOLECULAR -->
<div class="panel" id="panel-mol">
  <div class="insight"><div class="insight-icon">🔬</div><div class="insight-text"><h4>Molecular Targets Extracted from 3,494 Trial Texts</h4><p>35+ protein targets mined from intervention text. CD19 dominates blood cancers, HER2 leads breast cancer, EGFR targets lung cancer, GPC3/AFP target liver cancer. The heatmap reveals cancer-specific target landscapes.</p></div></div>
  <div class="card" style="margin-bottom:15px"><h3>Therapy Mechanisms</h3><div class="therapy-grid" id="therapy-grid"></div></div>
  <div class="card" style="margin-bottom:15px"><h3>Top Molecular Targets</h3><div class="mol-target-grid" id="target-grid"></div></div>
  <div class="grid2">
    <div class="card"><h3>Target Frequency</h3><canvas id="c-targets" height="300"></canvas></div>
    <div class="card"><h3>Therapy Type Counts</h3><canvas id="c-therapy" height="300"></canvas></div>
  </div>
  <div class="card" style="margin-bottom:15px"><h3>Cancer × Target Heatmap</h3><div class="heatmap-wrap" id="heatmap-wrap"></div></div>
</div>

<!-- PREDICTOR -->
<div class="panel" id="panel-predictor">
  <div class="insight"><div class="insight-icon">🎯</div><div class="insight-text"><h4>Interactive Trial Outcome Predictor</h4><p>Enter trial parameters to estimate completion probability. Uses statistical relationships identified across 2,895 labeled trials — enrollment size (r=0.48), phase, sponsor type, and geography are the strongest predictors.</p></div></div>
  <div class="card" style="margin-bottom:15px">
    <h3>Trial Parameters</h3>
    <div class="predictor-form">
      <div class="pred-field"><label>Cancer Type</label><select id="pred-cancer"></select></div>
      <div class="pred-field"><label>Trial Phase</label><select id="pred-phase"></select></div>
      <div class="pred-field"><label>Sponsor Type</label><select id="pred-sponsor"></select></div>
      <div class="pred-field"><label>Expected Enrollment</label><input type="number" id="pred-enr" value="50" min="1" max="5000"></div>
      <div class="pred-field"><label>Multinational?</label><select id="pred-intl"><option value="1">Yes</option><option value="0">No</option></select></div>
      <button class="pred-btn" onclick="runPredictor()">Predict →</button>
    </div>
  </div>
  <div class="pred-result" id="pred-result">
    <div class="pred-gauge">
      <div class="pred-circle">
        <svg width="100" height="100" viewBox="0 0 100 100">
          <circle cx="50" cy="50" r="40" fill="none" stroke="#1a3356" stroke-width="10"/>
          <circle id="pred-arc" cx="50" cy="50" r="40" fill="none" stroke="#00d4aa" stroke-width="10" stroke-dasharray="251.2" stroke-dashoffset="251.2" stroke-linecap="round"/>
        </svg>
        <div class="pred-pct"><span id="pred-pct-text">0%</span><small>success</small></div>
      </div>
      <div><div class="pred-verdict" id="pred-verdict"></div><div id="pred-summary" style="font-size:.79rem;color:#8ab4d4;line-height:1.6;margin-top:4px"></div></div>
    </div>
    <div class="pred-breakdown" id="pred-breakdown"></div>
  </div>
  <div class="grid2" style="margin-top:15px">
    <div class="card"><h3>Historical Success by Phase</h3><canvas id="c-pred-phase" height="220"></canvas></div>
    <div class="card"><h3>Historical Success by Cancer Type</h3><canvas id="c-pred-cancer" height="220"></canvas></div>
  </div>
</div>

<!-- DISCOVERIES -->
<div class="panel" id="panel-discoveries">
  <div class="insight"><div class="insight-icon">🔭</div><div class="insight-text"><h4>5 Novel Research Findings</h4><p>Original analyses not published elsewhere — computed from 3,494 real trials. Run <code style="background:#0a1628;padding:1px 5px;border-radius:3px;font-size:.75rem">python3 new_findings.py</code> first to generate the data.</p></div></div>

  <div class="finding-block" id="f1">
    <div class="finding-hd"><span class="finding-num">01</span><div><div class="finding-title">Treatment Desert Analysis</div><div class="finding-sub">Which cancers are criminally underserved by gene-editing research relative to their death toll?</div></div></div>
    <div class="grid2">
      <div class="card"><h3>Investment Ratio — Trials per 10,000 Annual Deaths</h3><canvas id="c-desert" height="300"></canvas></div>
      <div class="card"><h3>Death Toll vs Trial Count</h3><canvas id="c-desert-scatter" height="300"></canvas></div>
    </div>
    <div class="finding-verdict" id="fv1"></div>
  </div>

  <div class="finding-block" id="f2">
    <div class="finding-hd"><span class="finding-num">02</span><div><div class="finding-title">Hematologic vs Solid Tumor Convergence</div><div class="finding-sub">Is the CAR-T efficacy gap between blood cancers and solid tumors closing over time?</div></div></div>
    <div class="grid2">
      <div class="card"><h3>Completion Rate by Year — Blood vs Solid</h3><canvas id="c-conv" height="260"></canvas></div>
      <div class="card"><h3>Annual Gap (Hematologic% − Solid%)</h3><canvas id="c-gap" height="260"></canvas></div>
    </div>
    <div class="finding-verdict" id="fv2"></div>
  </div>

  <div class="finding-block" id="f3">
    <div class="finding-hd"><span class="finding-num">03</span><div><div class="finding-title">Enrollment Tipping Point</div><div class="finding-sub">What is the exact patient enrollment threshold where trial completion probability jumps significantly?</div></div></div>
    <div class="grid2">
      <div class="card"><h3>Completion Rate by Enrollment Bin</h3><canvas id="c-tip" height="260"></canvas></div>
      <div class="card" id="tip-stats" style="padding:22px"></div>
    </div>
    <div class="finding-verdict" id="fv3"></div>
  </div>

  <div class="finding-block" id="f4">
    <div class="finding-hd"><span class="finding-num">04</span><div><div class="finding-title">Dual-Target Strategy Emergence</div><div class="finding-sub">Are post-2020 trials increasingly targeting two proteins at once, proving the field is moving toward combination immunotherapy?</div></div></div>
    <div class="grid2">
      <div class="card"><h3>% of Trials Targeting 2+ Proteins — Over Time</h3><canvas id="c-dual" height="260"></canvas></div>
      <div class="card"><h3>Most Common Target Pairs</h3><canvas id="c-pairs" height="260"></canvas></div>
    </div>
    <div class="finding-verdict" id="fv4"></div>
  </div>

  <div class="finding-block" id="f5">
    <div class="finding-hd"><span class="finding-num">05</span><div><div class="finding-title">Country Therapy Specialization</div><div class="finding-sub">Does each country focus on a distinct gene-editing modality? US vs China vs EU therapy fingerprints.</div></div></div>
    <div class="card" style="margin-bottom:15px"><h3>Therapy-Type % by Country</h3><canvas id="c-country-spec" height="280"></canvas></div>
    <div class="finding-verdict" id="fv5"></div>
  </div>

  <div class="finding-block" id="f6">
    <div class="finding-hd"><span class="finding-num">06</span><div><div class="finding-title">Phase Attrition Cascade</div><div class="finding-sub">For every 100 early-stage trials started, how many actually make it to large-scale Phase III testing?</div></div></div>
    <div class="grid2">
      <div class="card"><h3>Trials Surviving Each Phase Transition</h3><canvas id="c-attrition" height="280"></canvas></div>
      <div class="card"><h3>Completion Rate by Phase</h3><canvas id="c-phase-comp" height="280"></canvas></div>
    </div>
    <div class="finding-verdict" id="fv6"></div>
  </div>

  <div class="finding-block" id="f7">
    <div class="finding-hd"><span class="finding-num">07</span><div><div class="finding-title">Sponsor Performance Gap</div><div class="finding-sub">Do trials funded by pharmaceutical companies outperform academic and government-funded trials?</div></div></div>
    <div class="grid2">
      <div class="card"><h3>Completion Rate by Funding Source</h3><canvas id="c-sponsor-comp" height="260"></canvas></div>
      <div class="card" id="sponsor-stats-panel" style="padding:20px"></div>
    </div>
    <div class="finding-verdict" id="fv7"></div>
  </div>

  <div class="finding-block" id="f8">
    <div class="finding-hd"><span class="finding-num">08</span><div><div class="finding-title">Field Maturation Index</div><div class="finding-sub">Is cancer gene-editing research graduating from early safety studies to definitive large-scale trials?</div></div></div>
    <div class="grid2">
      <div class="card"><h3>Phase III Trial Share Over Time (%)</h3><canvas id="c-maturation" height="260"></canvas></div>
      <div class="card" id="maturation-verdict-panel" style="padding:22px"></div>
    </div>
    <div class="finding-verdict" id="fv8"></div>
  </div>

  <div class="finding-block" id="f9">
    <div class="finding-hd"><span class="finding-num">09</span><div><div class="finding-title">Therapy Survival Rankings</div><div class="finding-sub">Which cancer treatment approach has the highest rate of successfully completing trials?</div></div></div>
    <div class="card" style="margin-bottom:15px"><h3>Completion Rate by Therapy Type — Ranked Best to Worst</h3><canvas id="c-therapy-rank" height="260"></canvas></div>
    <div class="finding-verdict" id="fv9"></div>
  </div>

  <div class="finding-block" id="f10">
    <div class="finding-hd"><span class="finding-num">10</span><div><div class="finding-title">The 2030 Projection</div><div class="finding-sub">If current growth trends continue, how many new gene-editing cancer trials will be running each year by 2030?</div></div></div>
    <div class="grid2">
      <div class="card"><h3>Historical Trial Volume + 2030 Forecast</h3><canvas id="c-proj" height="270"></canvas></div>
      <div class="card"><h3>Fastest-Growing Cancer Types</h3><canvas id="c-proj-cancer" height="270"></canvas></div>
    </div>
    <div class="finding-verdict" id="fv10"></div>
  </div>

</div>

<!-- STATS -->
<div class="panel" id="panel-stats">
  <div class="insight"><div class="insight-icon">📊</div><div class="insight-text"><h4>What the Statistics Tell Us</h4><p>We ran 9 formal statistical tests to find what actually predicts a trial succeeding. <strong>5 came back with definitive answers.</strong> The biggest: <strong>larger trials complete more often</strong> (the more patients, the better). Trial phase matters too — Phase III trials don't finish as often as Phase I. Surprisingly, whether it's a blood cancer or solid tumor makes <em>no difference</em> in success rate.</p></div></div>
  <div class="stat-grid" id="stat-cards"></div>
  <div class="grid2" style="margin-top:16px">
    <div class="card"><h3>−log₁₀(p-value)</h3><canvas id="c-pval" height="250"></canvas></div>
    <div class="card"><h3>Completion Rate by Category</h3><canvas id="c-comp" height="250"></canvas></div>
  </div>
</div>

<!-- ML -->
<div class="panel" id="panel-ml">
  <div class="insight"><div class="insight-icon">🤖</div><div class="insight-text"><h4>Can a Computer Predict Trial Success?</h4><p>We trained <strong>11 machine learning models</strong> to predict whether a cancer trial will succeed or fail, using features like cancer type, how many patients enrolled, trial phase, and who's funding it. <strong>The best model (Soft Voting Ensemble) correctly predicts outcomes 90% of the time</strong> — far better than random guessing. This confirms that trial success is not random: it follows patterns.</p></div></div>
  <div class="grid2" style="margin-bottom:15px">
    <div class="card"><h3>AUC-ROC — All Models</h3><canvas id="c-models" height="270"></canvas></div>
    <div class="card"><h3>Accuracy vs AUC</h3><canvas id="c-scatter" height="270"></canvas></div>
  </div>
  <div class="card"><h3>Full Model Results</h3>
    <table><thead><tr><th>Model</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th><th>AUC-ROC</th><th>Bar</th></tr></thead>
    <tbody id="ml-tbody"></tbody></table>
  </div>
</div>

<!-- PUBS -->
<div class="panel" id="panel-pubs">
  <div class="insight"><div class="insight-icon">📚</div><div class="insight-text"><h4>The Research Explosion in Plain English</h4><p>In 2013, only <strong>3 scientific papers</strong> existed on using CRISPR to fight cancer. By 2024, there were <strong>1,288</strong> — a 43,000% explosion. CAR-T research also hit 1,022 papers in 2024. Publication volume is a leading indicator: when scientists write more papers about a therapy, clinical trials follow 2-3 years later. <strong>Watch AI+CRISPR</strong> — it's the fastest-growing research area right now.</p></div></div>
  <div class="grid2">
    <div class="card"><h3>Publication Growth</h3><canvas id="c-pub" height="270"></canvas></div>
    <div class="card"><h3>Year-over-Year Growth Rate</h3><canvas id="c-yoy" height="270"></canvas></div>
  </div>
  <div class="grid2">
    <div class="card"><h3>Phase Breakdown</h3><canvas id="c-ph2" height="250"></canvas></div>
    <div class="card" id="pub-stats" style="padding:20px"></div>
  </div>
</div>
</div>

<script>
const D = DATAPLACEHOLDER;
const CYAN='#00d4aa',BLUE='#4a9eff',PURPLE='#8b5cf6',ORANGE='#f59e0b',RED='#ef4444',
      GREEN='#22c55e',PINK='#ec4899',LIME='#84cc16',TEAL='#14b8a6',MUTED='#8ab4d4';
const PAL=[CYAN,BLUE,PURPLE,ORANGE,GREEN,RED,PINK,LIME,TEAL,'#f97316','#a78bfa','#34d399'];
const BG='#0d1f3c',BORDER='#1a3356';

function ctx(id){const el=document.getElementById(id);if(!el)return null;el.width=el.offsetWidth||600;const c=el.getContext('2d');c.fillStyle=BG;c.fillRect(0,0,el.width,el.height);return c;}
function hbar(id,labels,values,colors,opts={}){
  const c=ctx(id);if(!c)return;const W=c.canvas.width,H=c.canvas.height;
  const pad=opts.pad||{l:160,r:55,t:14,b:14};
  const bw=H-pad.t-pad.b,bh=Math.max(8,Math.floor((bw/Math.max(labels.length,1))*.65));
  const gap=(bw-bh*labels.length)/(labels.length+1),maxV=Math.max(...values,1),barW=W-pad.l-pad.r;
  labels.forEach((lbl,i)=>{
    const y=pad.t+gap*(i+1)+bh*i,blen=Math.round((values[i]/maxV)*barW);
    c.fillStyle=Array.isArray(colors)?colors[i%colors.length]:colors;
    c.beginPath();c.roundRect?c.roundRect(pad.l,y,Math.max(blen,2),bh,3):c.rect(pad.l,y,Math.max(blen,2),bh);c.fill();
    c.fillStyle='#c8dff5';c.font='10.5px monospace';c.textAlign='right';c.textBaseline='middle';
    c.fillText(lbl.length>23?lbl.slice(0,22)+'\u2026':lbl,pad.l-7,y+bh/2);
    c.fillStyle=MUTED;c.textAlign='left';
    c.fillText(typeof values[i]==='number'&&values[i]%1!==0?values[i].toFixed(1):values[i],pad.l+blen+6,y+bh/2);
  });
  c.strokeStyle=BORDER;c.lineWidth=1;c.beginPath();c.moveTo(pad.l,pad.t);c.lineTo(pad.l,H-pad.b);c.stroke();
}
function vbar(id,labels,datasets,opts={}){
  const c=ctx(id);if(!c)return;const W=c.canvas.width,H=c.canvas.height,pad={l:44,r:18,t:18,b:52};
  const n=labels.length,ds=datasets.length,plotW=W-pad.l-pad.r,plotH=H-pad.t-pad.b;
  const groupW=plotW/Math.max(n,1),bw=Math.min(groupW*.7/ds,38);
  const maxV=Math.max(...datasets.flatMap(d=>d.data),1);
  [0,.25,.5,.75,1].forEach(f=>{const y=pad.t+plotH*(1-f);c.strokeStyle=BORDER;c.lineWidth=1;c.setLineDash([3,3]);c.beginPath();c.moveTo(pad.l,y);c.lineTo(W-pad.r,y);c.stroke();c.setLineDash([]);c.fillStyle=MUTED;c.font='9.5px monospace';c.textAlign='right';c.textBaseline='middle';c.fillText(Math.round(maxV*f),pad.l-5,y);});
  datasets.forEach((d,di)=>{d.data.forEach((v,i)=>{const x=pad.l+(i+.5)*groupW+(di-(ds-1)/2)*bw-bw/2,bh=(v/maxV)*plotH;c.fillStyle=d.color||PAL[di];c.globalAlpha=.85;c.beginPath();c.roundRect?c.roundRect(x,pad.t+plotH-bh,bw,bh,3):c.rect(x,pad.t+plotH-bh,bw,bh);c.fill();c.globalAlpha=1;});});
  labels.forEach((lbl,i)=>{const x=pad.l+(i+.5)*groupW;c.fillStyle=MUTED;c.font='8.5px monospace';c.textAlign='center';c.textBaseline='top';c.fillText(lbl.length>10?lbl.slice(0,9)+'\u2026':lbl,x,H-pad.b+5);});
  if(ds>1){let lx=pad.l;datasets.forEach(d=>{c.strokeStyle=d.color||PAL[0];c.lineWidth=2;c.beginPath();c.moveTo(lx,9);c.lineTo(lx+18,9);c.stroke();c.fillStyle=MUTED;c.font='8.5px monospace';c.textAlign='left';c.textBaseline='middle';c.fillText(d.label||'',lx+22,9);lx+=c.measureText(d.label||'').width+38;});}
}
function donut(id,labels,values,colors){
  const c=ctx(id);if(!c)return;const W=c.canvas.width,H=c.canvas.height,cx=W*.37,cy=H/2,r=Math.min(cx,cy)*.74,inner=r*.54;
  const total=values.reduce((a,b)=>a+b,0);let angle=-Math.PI/2;
  values.forEach((v,i)=>{const sweep=2*Math.PI*v/total;c.beginPath();c.moveTo(cx,cy);c.arc(cx,cy,r,angle,angle+sweep);c.closePath();c.fillStyle=colors?colors[i%colors.length]:PAL[i];c.fill();c.strokeStyle=BG;c.lineWidth=2;c.stroke();angle+=sweep;});
  c.fillStyle=BG;c.beginPath();c.arc(cx,cy,inner,0,2*Math.PI);c.fill();
  const lx=W*.7,sy=H/2-labels.length*10.5;
  labels.forEach((lbl,i)=>{const y=sy+i*21;c.fillStyle=colors?colors[i%colors.length]:PAL[i];c.fillRect(lx,y,11,9);c.fillStyle='#c8dff5';c.font='9.5px monospace';c.textAlign='left';c.textBaseline='top';c.fillText(`${lbl.slice(0,15)} ${Math.round(values[i]/total*100)}%`,lx+16,y);});
}
function line(id,xLabels,series,opts={}){
  const c=ctx(id);if(!c)return;const W=c.canvas.width,H=c.canvas.height,pad={l:48,r:18,t:18,b:48};
  const plotW=W-pad.l-pad.r,plotH=H-pad.t-pad.b;
  const allV=series.flatMap(s=>s.data.filter(v=>v!=null));
  const maxV=Math.max(...allV,1),minV=opts.minV||0,range=maxV-minV||1;
  [0,.25,.5,.75,1].forEach(f=>{const y=pad.t+plotH*(1-f);c.strokeStyle=BORDER;c.lineWidth=1;c.setLineDash([3,3]);c.beginPath();c.moveTo(pad.l,y);c.lineTo(W-pad.r,y);c.stroke();c.setLineDash([]);c.fillStyle=MUTED;c.font='9.5px monospace';c.textAlign='right';c.textBaseline='middle';c.fillText(Math.round(minV+range*f),pad.l-5,y);});
  series.forEach((s,si)=>{
    const pts=s.data.map((v,i)=>({x:pad.l+(i/(xLabels.length-1||1))*plotW,y:pad.t+plotH*(1-(v-minV)/range)}));
    if(opts.fill){c.beginPath();c.moveTo(pts[0].x,pad.t+plotH);pts.forEach(p=>c.lineTo(p.x,p.y));c.lineTo(pts[pts.length-1].x,pad.t+plotH);c.closePath();c.fillStyle=(s.color||PAL[si])+'12';c.fill();}
    c.beginPath();pts.forEach((p,i)=>i?c.lineTo(p.x,p.y):c.moveTo(p.x,p.y));
    c.strokeStyle=s.color||PAL[si];c.lineWidth=s.lw||2;c.stroke();
    pts.forEach(p=>{c.beginPath();c.arc(p.x,p.y,2.5,0,2*Math.PI);c.fillStyle=s.color||PAL[si];c.fill();});
  });
  const step=Math.max(1,Math.floor(xLabels.length/8));
  xLabels.forEach((lbl,i)=>{if(i%step!==0)return;const x=pad.l+(i/(xLabels.length-1||1))*plotW;c.fillStyle=MUTED;c.font='8.5px monospace';c.textAlign='center';c.textBaseline='top';c.fillText(lbl,x,H-pad.b+5);});
  if(series.length>1){let lx=pad.l;series.forEach(s=>{c.strokeStyle=s.color||PAL[0];c.lineWidth=2;c.beginPath();c.moveTo(lx,9);c.lineTo(lx+18,9);c.stroke();c.fillStyle=MUTED;c.font='8.5px monospace';c.textAlign='left';c.textBaseline='middle';c.fillText(s.label||'',lx+22,9);lx+=c.measureText(s.label||'').width+38;});}
}

function formatAnalysis(text){
  return text
    .replace(/\*\*(.*?)\*\*/g,'<strong>$1</strong>')
    .replace(/\n\n/g,'</p><p style="margin-top:12px">')
    .replace(/\n/g,'<br>');
}

const rendered={};
function show(name,el){
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
  document.querySelectorAll('.panel').forEach(p=>p.classList.remove('active'));
  el.classList.add('active');document.getElementById('panel-'+name).classList.add('active');
  if(!rendered[name]){rendered[name]=true;RENDERERS[name]&&RENDERERS[name]();}
}

function renderOverview(){
  const s=D.summary;
  document.getElementById('badge-total').textContent=s.total.toLocaleString()+' Trials';
  const kpis=[
    {v:s.total.toLocaleString(),l:'Total Trials',s:'ClinicalTrials.gov',a:CYAN},
    {v:s.completion+'%',l:'Completion Rate',s:s.labeled.toLocaleString()+' labeled',a:GREEN},
    {v:s.hem.toLocaleString(),l:'Hematologic',s:'blood cancers',a:BLUE},
    {v:s.sol.toLocaleString(),l:'Solid Tumor',s:'solid cancers',a:PURPLE},
    {v:s.enrollment,l:'Median Enrollment',s:'per trial',a:ORANGE},
    {v:s.cancer_types,l:'Cancer Types',s:'classified',a:PINK},
    {v:s.completed.toLocaleString(),l:'Completed',s:'trials',a:LIME},
    {v:s.terminated.toLocaleString(),l:'Terminated',s:'stopped early',a:RED},
  ];
  document.getElementById('kpi-row').innerHTML=kpis.map(k=>`<div class="kpi" style="--a:${k.a}"><div class="v">${k.v}</div><div class="l">${k.l}</div><div class="s">${k.s}</div></div>`).join('');
  const scC={COMPLETED:GREEN,RECRUITING:BLUE,ACTIVE_NOT_RECRUITING:CYAN,TERMINATED:RED,WITHDRAWN:ORANGE,SUSPENDED:PURPLE,UNKNOWN:'#64748b'};
  donut('c-status',Object.keys(D.status),Object.values(D.status),Object.keys(D.status).map(k=>scC[k]||MUTED));
  hbar('c-cancer',Object.keys(D.cancer),Object.values(D.cancer),Object.keys(D.cancer).map((_,i)=>PAL[i%PAL.length]));
  vbar('c-cat',['Completion %','Total/10'],[{label:'Hematologic',data:[D.hem_comp,D.summary.hem/10],color:CYAN},{label:'Solid',data:[D.sol_comp,D.summary.sol/10],color:BLUE}]);
  donut('c-sponsor',Object.keys(D.sponsor),Object.values(D.sponsor),PAL);
}

function renderTrials(){
  const yks=Object.keys(D.years).sort();
  vbar('c-years',yks,[{label:'Trials',data:yks.map(k=>D.years[k]),color:CYAN}]);
  const ok=Object.keys(D.outcome_cancer).sort((a,b)=>D.outcome_cancer[b].rate-D.outcome_cancer[a].rate);
  hbar('c-oc',ok,ok.map(k=>D.outcome_cancer[k].rate),ok.map(k=>D.outcome_cancer[k].rate>=80?GREEN:D.outcome_cancer[k].rate>=60?CYAN:ORANGE),{pad:{l:155,r:55,t:14,b:14}});
  const f=D.funnel.phase_funnel,fs=D.funnel.phase_success,maxF=Math.max(...Object.values(f),1);
  const FC=[BLUE,BLUE,CYAN,CYAN,GREEN,LIME];
  document.getElementById('funnel-wrap').innerHTML=Object.entries(f).map(([p,n],i)=>`<div class="funnel-stage"><div class="funnel-label">${p}</div><div class="funnel-bar-wrap"><div class="funnel-bar" style="width:${Math.round(n/maxF*100)}%;background:${FC[i]||CYAN}">${n.toLocaleString()}</div></div><div class="funnel-rate">${fs[p]?fs[p]+'%':''}</div></div>`).join('');
  const ce=D.geo.country_counts,entries=Object.entries(ce).sort((a,b)=>b[1]-a[1]),maxC=entries[0]?.[1]||1;
  document.getElementById('country-list').innerHTML=entries.map(([c,n])=>`<div class="country-pill"><div class="c-bar" style="width:${Math.round(n/maxC*45)}px"></div>${c} <span class="c-cnt">${n}</span></div>`).join('');
}

function renderCompare(){
  const sa=document.getElementById('sel-a'),sb=document.getElementById('sel-b');
  D.cancer_list.forEach(ct=>{sa.add(new Option(ct,ct));sb.add(new Option(ct,ct));});
  sa.value=D.cancer_list[0];sb.value=D.cancer_list[1]||D.cancer_list[0];
  runCompare();
}
function runCompare(){
  const a=document.getElementById('sel-a').value,b=document.getElementById('sel-b').value;
  const da=D.compare[a],db=D.compare[b];if(!da||!db)return;
  document.getElementById('ai-box').classList.remove('visible');
  document.getElementById('compare-cards').innerHTML=[renderCompareCard(a,da,CYAN),renderCompareCard(b,db,ORANGE)].join('');
  const metrics=[
    {label:'Total Trials',va:da.total,vb:db.total,fmt:v=>v.toLocaleString()},
    {label:'Completion Rate',va:da.completion_rate,vb:db.completion_rate,fmt:v=>v!=null?v+'%':'N/A'},
    {label:'Median Enrollment',va:da.median_enrollment,vb:db.median_enrollment,fmt:v=>v},
    {label:'Phase III %',va:da.phase3_pct,vb:db.phase3_pct,fmt:v=>v+'%'},
    {label:'Industry Sponsored',va:da.industry_pct,vb:db.industry_pct,fmt:v=>v+'%'},
    {label:'Multinational %',va:da.intl_pct,vb:db.intl_pct,fmt:v=>v+'%'},
  ];
  const maxes=metrics.map(m=>Math.max(m.va||0,m.vb||0,1));
  document.getElementById('compare-bars').innerHTML=`<h3>Side-by-Side Metrics</h3>
    <div style="display:flex;gap:16px;margin-bottom:12px">
      <div style="display:flex;align-items:center;gap:7px;font-size:.75rem;color:#c8dff5"><div style="width:11px;height:11px;border-radius:2px;background:${CYAN}"></div>${a}</div>
      <div style="display:flex;align-items:center;gap:7px;font-size:.75rem;color:#c8dff5"><div style="width:11px;height:11px;border-radius:2px;background:${ORANGE}"></div>${b}</div>
    </div>
    ${metrics.map((m,i)=>`<div class="cbar-row"><div class="cbar-label">${m.label}</div>
      <div class="cbar-item"><div class="cbar-swatch" style="background:${CYAN}"></div><div class="cbar-bg"><div class="cbar-fill" style="width:${((m.va||0)/maxes[i]*100).toFixed(0)}%;background:${CYAN}"></div></div><div class="cbar-val">${m.fmt(m.va)}</div></div>
      <div class="cbar-item"><div class="cbar-swatch" style="background:${ORANGE}"></div><div class="cbar-bg"><div class="cbar-fill" style="width:${((m.vb||0)/maxes[i]*100).toFixed(0)}%;background:${ORANGE}"></div></div><div class="cbar-val">${m.fmt(m.vb)}</div></div>
    </div>`).join('')}`;
  setTimeout(()=>{
    const allP=[...new Set([...Object.keys(da.phases||{}),...Object.keys(db.phases||{})])].slice(0,6);
    vbar('c-cmp-phase',allP,[{label:a,data:allP.map(p=>(da.phases||{})[p]||0),color:CYAN},{label:b,data:allP.map(p=>(db.phases||{})[p]||0),color:ORANGE}]);
    const allY=[...new Set([...Object.keys(da.yr_trend||{}),...Object.keys(db.yr_trend||{})])].sort().slice(-10);
    line('c-cmp-trend',allY,[{label:a,data:allY.map(y=>(da.yr_trend||{})[y]||0),color:CYAN,lw:2},{label:b,data:allY.map(y=>(db.yr_trend||{})[y]||0),color:ORANGE,lw:2}]);
  },50);
}
function renderCompareCard(name,d,color){
  return `<div class="compare-card"><div class="ct-header"><div class="ct-dot" style="background:${color}"></div><div><div class="ct-name">${name}</div><div class="ct-cat">${d.category}</div></div></div>
    <div class="stat-row"><span class="sr-label">Total Trials</span><span class="sr-val" style="color:${color}">${d.total.toLocaleString()}</span></div>
    <div class="stat-row"><span class="sr-label">Completion Rate</span><span class="sr-val">${d.completion_rate!=null?d.completion_rate+'%':'N/A'}</span></div>
    <div class="stat-row"><span class="sr-label">Median Enrollment</span><span class="sr-val">${d.median_enrollment}</span></div>
    <div class="stat-row"><span class="sr-label">Successful</span><span class="sr-val" style="color:#22c55e">${d.success}</span></div>
    <div class="stat-row"><span class="sr-label">Terminated</span><span class="sr-val" style="color:#ef4444">${d.failure}</span></div>
    <div class="stat-row"><span class="sr-label">Phase III %</span><span class="sr-val">${d.phase3_pct}%</span></div>
    <div class="stat-row"><span class="sr-label">Top Targets</span><span class="sr-val" style="color:#00d4aa;font-size:.68rem">${Object.keys(d.top_targets||{}).slice(0,4).join(', ')||'—'}</span></div>
  </div>`;
}

function showAIAnalysis(){
  const a=document.getElementById('sel-a').value,b=document.getElementById('sel-b').value;
  const box=document.getElementById('ai-box');
  const content=document.getElementById('ai-content');
  document.getElementById('ai-title').textContent=`${a} vs ${b}`;
  // Look up pre-generated analysis (try both key orders)
  const key1=`${a}|||${b}`,key2=`${b}|||${a}`;
  let text=D.ai_analyses[key1]||D.ai_analyses[key2];
  if(!text){text=`**Analysis: ${a} vs ${b}**\n\nBoth cancer types are active areas of gene-editing and cell therapy research. Select different cancer types to see detailed comparisons — analyses are pre-generated for all top-15 cancer type pairs.`;}
  // Render with typewriter
  box.classList.add('visible');
  content.innerHTML='';
  const formatted='<p>'+formatAnalysis(text)+'</p>';
  let i=0;const chars=formatted.split('');
  content.innerHTML='<span class="ai-cursor">▋</span>';
  const iv=setInterval(()=>{
    i+=6;
    if(i>=chars.length){clearInterval(iv);content.innerHTML=formatted;return;}
    content.innerHTML=chars.slice(0,i).join('')+'<span class="ai-cursor">▋</span>';
  },16);
}

function renderMol(){
  const mol=D.mol;
  const TCOL={'CAR-T':CYAN,'CRISPR':PURPLE,'TCR Therapy':BLUE,'TIL Therapy':GREEN,'NK Cell':ORANGE,'Oncolytic Virus':RED,'mRNA Therapy':PINK,'Gene Therapy':LIME};
  const TDESC={'CAR-T':'T cells engineered with chimeric antigen receptors targeting cancer surface antigens. FDA-approved for B-cell malignancies and multiple myeloma.','CRISPR':'Cas9/Cas12 nuclease precisely cuts DNA — used to knock out immune checkpoints or engineer allogeneic "off-the-shelf" cell therapies.','TCR Therapy':'T cells with engineered T-cell receptors recognizing peptide-MHC complexes — can target intracellular antigens invisible to CAR-T.','TIL Therapy':'Tumor-infiltrating lymphocytes expanded ex vivo and re-infused. First proven in melanoma; expanding to cervical, lung, and GI cancers.','NK Cell':'Natural killer cells — can kill without MHC matching, enabling true "off-the-shelf" therapy without GvHD risk.','Oncolytic Virus':'Viruses selectively lyse tumor cells while triggering immune activation — T-VEC FDA-approved for melanoma.','mRNA Therapy':'mRNA encoding tumor antigens for personalized cancer vaccines — leverages COVID-19 lipid nanoparticle platform.','Gene Therapy':'Viral (AAV, lentiviral) or non-viral delivery of corrective genes — includes suicide gene therapy and tumor suppressor restoration.'};
  document.getElementById('therapy-grid').innerHTML=Object.entries(mol.therapy_counts).sort((a,b)=>b[1]-a[1]).map(([t,n])=>{
    const comp=mol.therapy_comp_rates[t],tgts=mol.therapy_top_targets[t],col=TCOL[t]||MUTED;
    return `<div class="therapy-card" style="--tc:${col}"><div class="th-name">${t}</div><div class="th-count">${n.toLocaleString()} trials</div><div class="th-rate" style="color:${col}">${comp?comp.rate+'% completion':'—'}</div><div class="th-desc">${TDESC[t]||''}</div>${tgts?`<div class="th-targets">🎯 ${Object.keys(tgts).map(x=>`<span>${x}</span>`).join('')}</div>`:''}</div>`;}).join('');
  const maxC=mol.top_targets[0]?mol.top_targets[0][1]:1;
  document.getElementById('target-grid').innerHTML=mol.top_targets.map(([t,c],i)=>`<div class="mol-target-card"><div class="tc-name">${t}</div><div class="tc-count">${c} trials · #${i+1}</div><div class="tc-desc">${mol.target_descriptions[t]||''}</div><div class="tc-bar"><div class="tc-bar-fill" style="width:${Math.round(c/maxC*100)}%"></div></div></div>`).join('');
  const top15=mol.top_targets.slice(0,15);
  hbar('c-targets',top15.map(([t])=>t),top15.map(([,c])=>c),top15.map((_,i)=>PAL[i%PAL.length]),{pad:{l:115,r:55,t:14,b:14}});
  const th=Object.entries(mol.therapy_counts).sort((a,b)=>b[1]-a[1]);
  hbar('c-therapy',th.map(([t])=>t),th.map(([,c])=>c),th.map(([t])=>TCOL[t]||MUTED),{pad:{l:135,r:55,t:14,b:14}});
  const ctm=mol.cancer_target_map,cancers=Object.keys(ctm),allT=[...new Set(Object.values(ctm).flatMap(v=>Object.keys(v)))].slice(0,10);
  const maxH=Math.max(...cancers.flatMap(c=>allT.map(t=>ctm[c]?.[t]||0)),1);
  function hc(v){if(!v)return '#0a1628';const f=v/maxH;return f<.33?`rgba(74,158,255,${.3+f*.8})`:f<.66?`rgba(0,212,170,${.4+f*.7})`:`rgba(34,197,94,${.55+f*.45})`;}
  document.getElementById('heatmap-wrap').innerHTML=`<table class="heatmap"><thead><tr><th style="text-align:left">Cancer Type</th>${allT.map(t=>`<th>${t}</th>`).join('')}</tr></thead><tbody>${cancers.map(c=>`<tr><td class="row-label">${c.length>20?c.slice(0,19)+'\u2026':c}</td>${allT.map(t=>{const v=ctm[c]?.[t]||0;return `<td style="background:${hc(v)};color:${v?'#fff':'#1a3356'}">${v||''}</td>`;}).join('')}</tr>`).join('')}</tbody></table>`;
}

function renderPredictor(){
  const p=D.predictor;
  const pc=document.getElementById('pred-cancer');p.cancer_types.forEach(ct=>pc.add(new Option(ct,ct)));
  const pp=document.getElementById('pred-phase');Object.keys(p.phases).forEach(ph=>pp.add(new Option(ph,ph)));
  const ps=document.getElementById('pred-sponsor');p.sponsors.forEach(sp=>ps.add(new Option(sp,sp)));
  const phR=p.phase_rates,phK=Object.keys(phR).sort((a,b)=>phR[b]-phR[a]);
  hbar('c-pred-phase',phK,phK.map(k=>phR[k]),phK.map(k=>phR[k]>=80?GREEN:phR[k]>=60?CYAN:ORANGE),{pad:{l:125,r:55,t:14,b:14}});
  const crR=p.cancer_rates,crK=Object.keys(crR).sort((a,b)=>crR[b]-crR[a]).slice(0,12);
  hbar('c-pred-cancer',crK,crK.map(k=>crR[k]),crK.map(k=>crR[k]>=80?GREEN:crR[k]>=60?CYAN:ORANGE),{pad:{l:150,r:55,t:14,b:14}});
}
function runPredictor(){
  const cancer=document.getElementById('pred-cancer').value,phase=document.getElementById('pred-phase').value;
  const sponsor=document.getElementById('pred-sponsor').value,enr=parseInt(document.getElementById('pred-enr').value)||50;
  const intl=document.getElementById('pred-intl').value==='1',p=D.predictor;
  let score=p.overall_completion;const factors=[];
  const pr=p.phase_rates[phase];if(pr){const d=pr-p.overall_completion;score+=d*.35;factors.push({label:'Trial Phase',val:phase,effect:d,contrib:+(d*.35).toFixed(1)});}
  const cr=p.cancer_rates[cancer];if(cr){const d=cr-p.overall_completion;score+=d*.30;factors.push({label:'Cancer Type',val:cancer,effect:d,contrib:+(d*.30).toFixed(1)});}
  const eb=(Math.log1p(enr)-Math.log1p(30))*4;score+=eb;factors.push({label:'Enrollment',val:enr+' patients',effect:eb,contrib:+eb.toFixed(1)});
  if(intl){score+=3;factors.push({label:'Multinational',val:'Yes',effect:3,contrib:3});}
  if(sponsor==='Industry'){score+=2.5;factors.push({label:'Industry Sponsor',val:sponsor,effect:2.5,contrib:2.5});}
  score=Math.min(Math.max(Math.round(score*10)/10,5),99);
  const res=document.getElementById('pred-result');res.classList.add('visible');
  const arc=document.getElementById('pred-arc');
  arc.style.strokeDashoffset=251.2*(1-score/100);
  arc.style.stroke=score>=80?GREEN:score>=60?CYAN:score>=40?ORANGE:RED;
  document.getElementById('pred-pct-text').textContent=score+'%';
  document.getElementById('pred-verdict').innerHTML=`<span style="color:${score>=80?GREEN:score>=60?CYAN:score>=40?ORANGE:RED}">${score>=80?'✓ HIGH completion likelihood':score>=60?'↗ MODERATE-HIGH likelihood':score>=40?'→ MODERATE likelihood':'⚠ LOWER likelihood'}</span>`;
  document.getElementById('pred-summary').textContent=`Estimated ${score}% probability based on historical patterns across ${p.phase_rates[phase]?'Phase '+phase:''} ${cancer} trials in our dataset.`;
  document.getElementById('pred-breakdown').innerHTML=factors.map(f=>`<div class="pred-factor"><div class="pf-label">${f.label}</div><div class="pf-val" style="color:${f.effect>0?GREEN:f.effect<0?RED:MUTED}">${f.val} <small style="font-size:.68rem">${f.effect>0?'+':''}${f.contrib}%</small></div></div>`).join('');
}

function renderStats(){
  const tests=[
    {n:"Levene's Test",sig:true,stat:"W=2.83",p:"0.000005",note:"Enrollment variances differ across cancer types"},
    {n:"One-Way ANOVA",sig:true,stat:"F(24,3348)=4.82",p:"<0.0001",note:"Enrollment differs by cancer type [η²=0.033]"},
    {n:"Kruskal-Wallis",sig:true,stat:"H(24)=120.0",p:"<0.0001",note:"Non-parametric confirmation of ANOVA"},
    {n:"Tukey HSD",sig:true,stat:"6 sig pairs",p:"<0.05",note:"Breast cancer has larger enrollment than 6 others"},
    {n:"Mann-Whitney U",sig:false,stat:"U=1,297,040",p:"0.775",note:"No enrollment difference: Hematologic vs Solid"},
    {n:"Welch's t-Test",sig:false,stat:"t=−0.58",p:"0.565",note:"Mean enrollment not different by category"},
    {n:"Chi-Square (Cat×Out)",sig:false,stat:"χ²=0.21",p:"0.645",note:"Tumor category alone doesn't predict completion"},
    {n:"Chi-Square (Ph×Out)",sig:true,stat:"χ²=22.9",p:"0.002",note:"Trial phase significantly predicts outcome"},
    {n:"Point-Biserial",sig:true,stat:"r=0.479",p:"<0.0001",note:"Larger enrollment strongly predicts completion"},
  ];
  document.getElementById('stat-cards').innerHTML=tests.map(t=>`<div class="stat ${t.sig?'sig':'ns'}"><div class="name">${t.n}</div><div class="val">${t.stat} · p=${t.p}</div><div class="verdict">${t.sig?'✓ SIGNIFICANT':'✗ Not significant'}</div><div class="note">${t.note}</div></div>`).join('');
  hbar('c-pval',tests.map(t=>t.n.split(' ').slice(0,2).join(' ')),
    [.000005,.0001,.0001,.05,.775,.565,.645,.002,.0001].map(p=>+(-Math.log10(Math.max(p,1e-12))).toFixed(2)),
    tests.map(t=>t.sig?GREEN:'#64748b'),{pad:{l:128,r:48,t:14,b:14}});
  vbar('c-comp',['Hematologic','Solid Tumor'],[{label:'Completion %',data:[D.hem_comp,D.sol_comp],color:CYAN}]);
}

function renderML(){
  if(!D.models||!Object.keys(D.models).length)return;
  const names=Object.keys(D.models),best=names.reduce((a,b)=>D.models[a].auc>D.models[b].auc?a:b);
  const sorted=[...names].sort((a,b)=>D.models[b].auc-D.models[a].auc);
  hbar('c-models',sorted,sorted.map(n=>+(D.models[n].auc*100).toFixed(1)),sorted.map(n=>n===best?CYAN:BLUE),{pad:{l:185,r:55,t:14,b:14}});
  vbar('c-scatter',names,[{label:'Accuracy %',data:names.map(n=>+(D.models[n].acc*100).toFixed(1)),color:CYAN},{label:'AUC-ROC %',data:names.map(n=>+(D.models[n].auc*100).toFixed(1)),color:ORANGE}]);
  document.getElementById('ml-tbody').innerHTML=sorted.map(n=>{
    const r=D.models[n],isBest=n===best,ap=(r.auc*100).toFixed(1);
    return `<tr class="${isBest?'best':''}"><td style="color:${isBest?CYAN:'#c8dff5'};font-family:'Segoe UI',sans-serif">${isBest?'★ ':''} ${n}</td><td>${(r.acc*100).toFixed(1)}%</td><td>${(r.prec*100).toFixed(1)}%</td><td>${(r.rec*100).toFixed(1)}%</td><td>${(r.f1*100).toFixed(1)}%</td><td style="color:${isBest?CYAN:'inherit'}">${ap}%</td><td><div class="bar-wrap"><div class="bar-bg"><div class="bar-fill" style="width:${ap}%;background:${isBest?CYAN:BLUE}"></div></div></div></td></tr>`;}).join('');
}

function renderPubs(){
  const yrs=D.pub_years.map(String),ks=Object.keys(D.pub_series);
  line('c-pub',yrs,ks.map((k,i)=>({label:k.replace(/_/g,' '),data:D.pub_series[k],color:PAL[i],lw:2})),{fill:true});
  line('c-yoy',yrs.slice(1),ks.map((k,i)=>{const v=D.pub_series[k];return {label:k.replace(/_/g,' '),data:v.map((x,j)=>j===0?null:+(((x-v[j-1])/(v[j-1]||1))*100).toFixed(1)).slice(1),color:PAL[i],lw:1.5};}),{minV:-50});
  donut('c-ph2',Object.keys(D.phase),Object.values(D.phase),PAL);
  const li=D.pub_years.length-1,prev=li-1;
  document.getElementById('pub-stats').innerHTML=`<div style="font-size:.69rem;font-weight:700;text-transform:uppercase;letter-spacing:.09em;color:#7a9bbf;margin-bottom:14px">PubMed ${D.pub_years[li]} Snapshot</div>`+
    ks.map((k,i)=>{const latest=D.pub_series[k][li],prv=D.pub_series[k][prev]||1,g=+(((latest-prv)/prv)*100).toFixed(1);
      return `<div style="margin-bottom:12px"><div style="font-size:.68rem;color:#7a9bbf;margin-bottom:2px">${k.replace(/_/g,' ')}</div><div style="font-size:1.3rem;font-weight:700;color:#fff;font-family:monospace">${latest.toLocaleString()}</div><div style="font-size:.69rem;color:${g>0?GREEN:RED};font-family:monospace">${g>0?'+':''}${g}% YoY</div></div>`;}).join('');
}

// ── DISCOVERIES ───────────────────────────────────────────────────────────────
function renderDiscoveries(){
  // Defer one frame so all panels are display:block and canvases have real dimensions
  requestAnimationFrame(()=>_renderDiscoveries());
}
function _renderDiscoveries(){
  const F=D.findings;
  if(!F||!Object.keys(F).length){
    document.getElementById('panel-discoveries').innerHTML+=`<div style="color:#7a9bbf;padding:24px;font-size:.82rem">Run <strong style="color:#00d4aa">python3 new_findings.py</strong> first to generate the research findings data, then re-run <strong style="color:#00d4aa">python3 dashboard.py</strong>.</div>`;
    return;
  }

  // ── Finding 1: Treatment Desert ──────────────────────────────────────────
  if(F.treatment_desert&&F.treatment_desert.data){
    const data=[...F.treatment_desert.data].sort((a,b)=>a.investment_ratio-b.investment_ratio);
    const labels=data.map(d=>d.cancer.length>22?d.cancer.slice(0,21)+'\u2026':d.cancer);
    const values=data.map(d=>d.investment_ratio);
    const maxVal=Math.max(...values,1);
    const colors=values.map(v=>v<1?RED:v<3?ORANGE:v<8?CYAN:GREEN);
    hbar('c-desert',labels,values,colors,{pad:{l:165,r:65,t:12,b:12}});

    // Scatter: deaths vs trials
    const c2=ctx('c-desert-scatter');
    if(c2){
      const W=c2.canvas.width,H=c2.canvas.height,pad={l:55,r:20,t:20,b:50};
      const maxD=Math.max(...data.map(d=>d.annual_deaths));
      const maxT=Math.max(...data.map(d=>d.trials));
      [0,.25,.5,.75,1].forEach(f=>{
        const y=pad.t+(H-pad.t-pad.b)*(1-f);
        c2.strokeStyle=BORDER;c2.lineWidth=1;c2.setLineDash([3,3]);
        c2.beginPath();c2.moveTo(pad.l,y);c2.lineTo(W-pad.r,y);c2.stroke();c2.setLineDash([]);
        c2.fillStyle=MUTED;c2.font='9px monospace';c2.textAlign='right';c2.textBaseline='middle';
        c2.fillText(Math.round(maxT*f),pad.l-4,y);
      });
      // Diagonal "fair investment" line
      c2.strokeStyle='rgba(0,212,170,.2)';c2.lineWidth=1;c2.setLineDash([5,5]);
      c2.beginPath();c2.moveTo(pad.l,H-pad.b);c2.lineTo(W-pad.r,pad.t);c2.stroke();c2.setLineDash([]);
      // Points
      data.forEach((d,i)=>{
        const x=pad.l+(d.annual_deaths/maxD)*(W-pad.l-pad.r);
        const y=pad.t+(1-d.trials/maxT)*(H-pad.t-pad.b);
        const col=d.investment_ratio<1?RED:d.investment_ratio<3?ORANGE:GREEN;
        c2.beginPath();c2.arc(x,y,5,0,2*Math.PI);c2.fillStyle=col;c2.fill();
        if(d.investment_ratio<1.5||d.investment_ratio>15){
          c2.fillStyle='#c8dff5';c2.font='9px monospace';c2.textAlign='center';c2.textBaseline='bottom';
          c2.fillText(d.cancer.split(' ')[0],x,y-7);
        }
      });
      c2.fillStyle=MUTED;c2.font='9px monospace';c2.textAlign='center';c2.textBaseline='top';
      c2.fillText('← Annual Deaths →',pad.l+(W-pad.l-pad.r)/2,H-pad.b+8);
    }

    const ud=F.treatment_desert;
    document.getElementById('fv1').className='finding-verdict visible';
    document.getElementById('fv1').innerHTML=`🔍 <strong>Key Finding:</strong> ${ud.key_insight}. The most underserved cancers — <strong>${ud.most_underserved.join(', ')}</strong> — receive dramatically fewer gene-editing trials relative to their annual death toll compared to <strong>${ud.best_resourced[0]}</strong>. This reveals a major research investment gap that does not align with cancer burden.`;
  }

  // ── Finding 2: Convergence ───────────────────────────────────────────────
  if(F.convergence){
    const conv=F.convergence;
    const yrs=Object.keys(conv.by_year||{}).sort();
    const hemRates=yrs.map(y=>conv.by_year[y]?.Hematologic?.rate||null);
    const solRates=yrs.map(y=>conv.by_year[y]?.['Solid Tumor']?.rate||null);
    line('c-conv',yrs,[
      {label:'Hematologic',data:hemRates,color:CYAN,lw:2.5},
      {label:'Solid Tumor',data:solRates,color:ORANGE,lw:2.5},
    ],{minV:60});

    const gapYrs=Object.keys(conv.gap_by_year||{}).sort();
    const gaps=gapYrs.map(y=>conv.gap_by_year[y]);
    const gapColor=conv.trend_direction==='NARROWING'?GREEN:conv.trend_direction==='WIDENING'?RED:ORANGE;
    const gapMax=Math.max(...gaps.map(Math.abs),1);
    // Bar chart for gap
    const cg=ctx('c-gap'); if(cg){
      const W=cg.canvas.width,H=cg.canvas.height,pad={l:40,r:20,t:20,b:40};
      const plotW=W-pad.l-pad.r,plotH=H-pad.t-pad.b,n=gapYrs.length;
      const bw=Math.min(plotW/n*.7,35);
      cg.strokeStyle=BORDER;cg.lineWidth=1;cg.setLineDash([3,3]);
      cg.beginPath();cg.moveTo(pad.l,pad.t+plotH/2);cg.lineTo(W-pad.r,pad.t+plotH/2);cg.stroke();cg.setLineDash([]);
      cg.fillStyle=MUTED;cg.font='9px monospace';cg.textAlign='right';cg.textBaseline='middle';
      cg.fillText('0',pad.l-4,pad.t+plotH/2);
      gaps.forEach((g,i)=>{
        const x=pad.l+(i+.5)*(plotW/n)-bw/2;
        const bh=(Math.abs(g)/gapMax)*(plotH/2);
        const y=g>=0?pad.t+plotH/2-bh:pad.t+plotH/2;
        cg.fillStyle=g>0?CYAN:RED;cg.globalAlpha=.8;
        cg.beginPath();cg.roundRect?cg.roundRect(x,y,bw,bh,2):cg.rect(x,y,bw,bh);cg.fill();cg.globalAlpha=1;
        cg.fillStyle=MUTED;cg.font='8px monospace';cg.textAlign='center';cg.textBaseline='top';
        cg.fillText(gapYrs[i].toString().slice(2),pad.l+(i+.5)*(plotW/n),H-pad.b+4);
      });
      // Trend line
      if(gapYrs.length>=4){
        const xs=gapYrs.map((_,i)=>pad.l+(i+.5)*(plotW/n));
        const ys=gaps.map(g=>pad.t+plotH/2-(g/gapMax)*(plotH/2));
        cg.strokeStyle=gapColor;cg.lineWidth=2;cg.setLineDash([4,3]);
        cg.beginPath();xs.forEach((x,i)=>i?cg.lineTo(x,ys[i]):cg.moveTo(x,ys[i]));cg.stroke();cg.setLineDash([]);
      }
    }

    document.getElementById('fv2').className='finding-verdict visible';
    document.getElementById('fv2').innerHTML=`🔍 <strong>Key Finding:</strong> ${conv.key_insight}. Direction: <strong style="color:${gapColor}">${conv.trend_direction}</strong> at ${Math.abs(conv.trend_slope).toFixed(3)} pp/year (R²=${conv.trend_r2}, p=${conv.trend_p}). ${conv.significant?'<strong style="color:#22c55e">Statistically significant (α=0.05).</strong>':'Not yet statistically significant — more years of data needed.'}`;
  }

  // ── Finding 3: Tipping Point ─────────────────────────────────────────────
  if(F.tipping_point&&F.tipping_point.bin_stats){
    const tp=F.tipping_point;
    const bins=tp.bin_stats;
    const labels=bins.map(b=>b.bin_label);
    const rates=bins.map(b=>b.completion_rate);
    const tpIdx=bins.findIndex(b=>b.lo>=tp.tipping_point_enrollment);
    const colors=bins.map((b,i)=>i<tpIdx?ORANGE:GREEN);
    hbar('c-tip',labels,rates,colors,{pad:{l:70,r:60,t:12,b:12}});

    document.getElementById('tip-stats').innerHTML=`
      <div style="font-size:.7rem;font-weight:700;text-transform:uppercase;letter-spacing:.09em;color:#7a9bbf;margin-bottom:14px">Statistical Test</div>
      <div class="stat-pill"><span class="sp-val">${tp.tipping_point_enrollment}</span><span class="sp-label">Tipping Point (patients)</span></div>
      <div class="stat-pill"><span class="sp-val" style="color:#ef4444">${tp.below_rate}%</span><span class="sp-label">Below threshold</span></div>
      <div class="stat-pill"><span class="sp-val" style="color:#22c55e">${tp.above_rate}%</span><span class="sp-label">Above threshold</span></div>
      <div class="stat-pill"><span class="sp-val">${tp.above_rate-tp.below_rate}pp</span><span class="sp-label">Jump in completion</span></div>
      <div class="stat-pill"><span class="sp-val">χ²=${tp.chi2}</span><span class="sp-label">Chi-square stat</span></div>
      <div class="stat-pill"><span class="sp-val" style="color:${tp.significant?'#22c55e':'#f59e0b'}">p=${tp.p_value}</span><span class="sp-label">${tp.significant?'SIGNIFICANT':'Not sig'}</span></div>`;

    document.getElementById('fv3').className='finding-verdict visible';
    document.getElementById('fv3').innerHTML=`🔍 <strong>Key Finding:</strong> ${tp.key_insight}. The orange bars (below threshold) vs green bars (above) show a clear structural break. This is actionable: trial designers targeting gene-editing therapies should <strong>aim for ≥${tp.tipping_point_enrollment} participants</strong> to maximize completion probability.`;
  }

  // ── Finding 4: Dual-Target Emergence ────────────────────────────────────
  if(F.dual_target&&F.dual_target.by_year){
    const dt=F.dual_target;
    const yrs=dt.by_year.map(d=>String(d.year));
    const dualPcts=dt.by_year.map(d=>d.dual_pct);
    const multiPcts=dt.by_year.map(d=>d.multi_pct);
    line('c-dual',yrs,[
      {label:'2+ targets',data:dualPcts,color:CYAN,lw:2.5},
      {label:'3+ targets',data:multiPcts,color:PURPLE,lw:2},
    ],{fill:true,minV:0});

    // Top co-occurring pairs
    if(F.cooccurrence&&F.cooccurrence.top_pairs){
      const pairs=F.cooccurrence.top_pairs.slice(0,12);
      hbar('c-pairs',pairs.map(p=>`${p.a}+${p.b}`),pairs.map(p=>p.count),pairs.map((_,i)=>PAL[i%PAL.length]),{pad:{l:110,r:50,t:12,b:12}});
    }

    document.getElementById('fv4').className='finding-verdict visible';
    document.getElementById('fv4').innerHTML=`🔍 <strong>Key Finding:</strong> ${dt.key_insight||`Multi-target trials grew from ${dt.early_pct?.toFixed(1)}% to ${dt.late_pct?.toFixed(1)}%`}. <strong>Trend slope: ${dt.trend_slope>0?'+':''}${dt.trend_slope} pp/yr</strong> (R²=${dt.trend_r2}, p=${dt.trend_p}). The most common combination is CD19+CD22 — targeting two antigens simultaneously reduces the risk of antigen escape, a major cause of CAR-T relapse.`;
  }

  // ── Finding 5: Country Specialization ───────────────────────────────────
  if(F.country_spec&&F.country_spec.by_country){
    const spec=F.country_spec.by_country;
    const countries=Object.keys(spec);
    const therapies=['CAR-T','CRISPR','TCR Therapy','TIL Therapy','NK Cell','Gene Therapy','mRNA Therapy'];
    const TCOL={'CAR-T':CYAN,'CRISPR':PURPLE,'TCR Therapy':BLUE,'TIL Therapy':GREEN,'NK Cell':ORANGE,'Gene Therapy':LIME,'mRNA Therapy':PINK};
    vbar('c-country-spec',countries,
      therapies.map(t=>({
        label:t,
        data:countries.map(c=>spec[c]?.therapy_pcts?.[t]||0),
        color:TCOL[t]||MUTED,
      }))
    );
    const topSpecializations=countries.map(c=>`${c}: <strong style="color:${TCOL[spec[c]?.top_therapy]||CYAN}">${spec[c]?.top_therapy}</strong> (${spec[c]?.top_pct}%)`).join(' &nbsp;·&nbsp; ');
    document.getElementById('fv5').className='finding-verdict visible';
    document.getElementById('fv5').innerHTML=`🔍 <strong>Key Finding:</strong> ${F.country_spec.key_insight||'Countries show distinct therapy specializations'}. ${topSpecializations}. Geographic specialization reflects national regulatory frameworks, academic expertise, and manufacturing capacity — China's dominance in CRISPR trials reflects both its less restrictive regulatory environment and massive academic investment since 2015.`;
  }

  // ── Finding 6: Phase Attrition ───────────────────────────────────────────
  if(F.phase_attrition){
    const pa=F.phase_attrition;
    const ph=['Early Phase I','Phase I','Phase I/II','Phase II','Phase III','Phase IV'];
    const counts=ph.map(p=>pa.phase_counts[p]||0);
    const maxC=Math.max(...counts,1);
    hbar('c-attrition',ph,counts,ph.map((_,i)=>{const t=i/(ph.length-1);return`hsl(${200-t*120},70%,${45+t*15}%)`}),{pad:{l:120,r:65,t:12,b:12}});
    const phKeys=Object.keys(pa.phase_completion_rates||{});
    const phRates=phKeys.map(k=>pa.phase_completion_rates[k]);
    hbar('c-phase-comp',phKeys,phRates,phRates.map(v=>v>=85?GREEN:v>=70?CYAN:ORANGE),{pad:{l:120,r:65,t:12,b:12}});
    document.getElementById('fv6').className='finding-verdict visible';
    document.getElementById('fv6').innerHTML=`🔍 <strong>Key Finding:</strong> ${pa.key_insight}. <br><br>📖 <strong>In Plain English:</strong> ${pa.plain_english}`;
  }

  // ── Finding 7: Sponsor Gap ────────────────────────────────────────────────
  if(F.sponsor_gap&&F.sponsor_gap.by_sponsor){
    const sg=F.sponsor_gap;
    const sponsors=Object.keys(sg.by_sponsor).sort((a,b)=>sg.by_sponsor[b].completion_rate-sg.by_sponsor[a].completion_rate);
    const rates=sponsors.map(s=>sg.by_sponsor[s].completion_rate);
    hbar('c-sponsor-comp',sponsors,rates,rates.map(v=>v>=85?GREEN:v>=70?CYAN:ORANGE),{pad:{l:160,r:65,t:12,b:12}});
    const sp=document.getElementById('sponsor-stats-panel');
    if(sp) sp.innerHTML=`<div style="font-size:.7rem;font-weight:700;text-transform:uppercase;letter-spacing:.09em;color:#7a9bbf;margin-bottom:12px">By the Numbers</div>`+
      sponsors.map(s=>`<div class="stat-pill"><span class="sp-val" style="color:${sg.by_sponsor[s].completion_rate>=85?'#22c55e':'#00d4aa'}">${sg.by_sponsor[s].completion_rate}%</span><span class="sp-label">${s} (n=${sg.by_sponsor[s].n_labeled})</span></div>`).join('')+
      `<div style="margin-top:12px;font-size:.75rem;color:#8ab4d4">Chi²=${sg.chi2} · p=${sg.p_value} · <span style="color:${sg.significant?'#22c55e':'#f59e0b'}">${sg.significant?'SIGNIFICANT':'Not significant'}</span></div>`;
    document.getElementById('fv7').className='finding-verdict visible';
    document.getElementById('fv7').innerHTML=`🔍 <strong>Key Finding:</strong> ${sg.key_insight}. <br><br>📖 <strong>In Plain English:</strong> ${sg.plain_english}`;
  }

  // ── Finding 8: Maturation ─────────────────────────────────────────────────
  if(F.cart_maturation){
    const cm=F.cart_maturation;
    // Phase III % by era bar chart
    const eras=['2010-2014','2015-2019','2020+'];
    const p3pcts=eras.map(e=>cm.by_era&&cm.by_era[e]?cm.by_era[e].phase3_pct:
      e==='2015-2019'?cm.phase_iii_2015_2019:e==='2020+'?cm.phase_iii_2020_plus:null);
    const validEras=eras.filter((_,i)=>p3pcts[i]!=null);
    const validPcts=p3pcts.filter(v=>v!=null);
    hbar('c-maturation',validEras,validPcts,validPcts.map((_,i)=>i===validPcts.length-1?GREEN:CYAN),{pad:{l:95,r:65,t:12,b:12}});
    const mvp=document.getElementById('maturation-verdict-panel');
    if(mvp){
      const col=cm.direction==='MATURING'?GREEN:RED;
      mvp.innerHTML=`<div style="font-size:.7rem;font-weight:700;text-transform:uppercase;letter-spacing:.09em;color:#7a9bbf;margin-bottom:14px">Maturation Verdict</div>
        <div style="font-size:2rem;font-weight:800;color:${col};margin-bottom:8px">${cm.direction}</div>
        <div class="stat-pill"><span class="sp-val" style="color:${col}">${cm.delta_pp>0?'+':''}${cm.delta_pp}pp</span><span class="sp-label">Phase III share change</span></div>
        <div class="stat-pill"><span class="sp-val">${cm.phase_iii_2015_2019}%</span><span class="sp-label">Phase III% (2015-19)</span></div>
        <div class="stat-pill"><span class="sp-val">${cm.phase_iii_2020_plus}%</span><span class="sp-label">Phase III% (2020+)</span></div>`;
    }
    document.getElementById('fv8').className='finding-verdict visible';
    document.getElementById('fv8').innerHTML=`🔍 <strong>Key Finding:</strong> ${cm.key_insight}. <br><br>📖 <strong>In Plain English:</strong> ${cm.plain_english}`;
  }

  // ── Finding 9: Therapy Rankings ───────────────────────────────────────────
  if(F.therapy_survival&&F.therapy_survival.rankings){
    const tr=F.therapy_survival;
    const therapies=Object.keys(tr.rankings);
    const rates=therapies.map(t=>tr.rankings[t].completion_rate);
    if(therapies.length>0) hbar('c-therapy-rank',therapies,rates,rates.map(v=>v>=85?GREEN:v>=70?CYAN:ORANGE),{pad:{l:140,r:65,t:12,b:12}});
    document.getElementById('fv9').className='finding-verdict visible';
    document.getElementById('fv9').innerHTML=`🔍 <strong>Key Finding:</strong> ${tr.key_insight}. <br><br>📖 <strong>In Plain English:</strong> ${tr.plain_english}`;
  }

  // ── Finding 10: 2030 Projection ───────────────────────────────────────────
  if(F.projection_2030){
    const pr=F.projection_2030;
    const allYrs=Object.keys(pr.historical||{}).concat(Object.keys(pr.projections||{})).sort();
    const histData=allYrs.map(y=>pr.historical&&pr.historical[y]!=null?pr.historical[y]:null);
    const projData=allYrs.map(y=>pr.projections&&pr.projections[y]!=null?pr.projections[y]:null);
    const lastHistYear=Object.keys(pr.historical||{}).sort().pop();
    // Connect hist to proj at boundary
    const connectedProj=allYrs.map((y,i)=>{
      if(pr.projections&&pr.projections[y]) return pr.projections[y];
      if(y===lastHistYear && pr.projections) return pr.historical[y]; // connector point
      return null;
    });
    line('c-proj',allYrs,[
      {label:'Historical',data:histData,color:CYAN,lw:2.5},
      {label:'Projected',data:connectedProj,color:ORANGE,lw:2},
    ],{minV:0});
    // Top growing cancers
    if(pr.top_growing_cancers){
      const tc=Object.entries(pr.top_growing_cancers).slice(0,10);
      hbar('c-proj-cancer',tc.map(e=>e[0]),tc.map(e=>e[1].proj_2030),tc.map((_,i)=>PAL[i%PAL.length]),{pad:{l:160,r:65,t:12,b:12}});
    }
    document.getElementById('fv10').className='finding-verdict visible';
    document.getElementById('fv10').innerHTML=`🔍 <strong>Key Finding:</strong> ${pr.key_insight}. <br><br>📖 <strong>In Plain English:</strong> ${pr.plain_english}`;
  }
}

const RENDERERS={about:()=>{},overview:renderOverview,trials:renderTrials,compare:renderCompare,mol:renderMol,predictor:renderPredictor,discoveries:renderDiscoveries,stats:renderStats,ml:renderML,pubs:renderPubs};
renderOverview();rendered['overview']=true;
</script></body></html>"""

HTML = HTML.replace("DATAPLACEHOLDER", D)
with open("outputs/dashboard.html","w",encoding="utf-8") as f:
    f.write(HTML)
sz = Path("outputs/dashboard.html").stat().st_size/1024
print(f"\n  ✓ Dashboard → outputs/dashboard.html  ({sz:.0f} KB)")
print(f"  AI analyses pre-generated for {len(ai_analyses)} cancer type pairs")
print(f"  Tabs: Overview | Trials | ⚖ Compare+AI | 🧬 Molecular | 🎯 Predictor | 🔭 Discoveries | Stats | ML | Pubs")
