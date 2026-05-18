"""
fetch_data.py v2 — Expanded fetch targeting 10,000+ unique trials
28 search queries, 1000/query, deduplicated by NCT ID.

New queries added:
  - allogeneic cell therapy, stem cell gene therapy, retroviral vector
  - AAV gene therapy, base editing, prime editing, zinc finger nuclease
  - TALEN, RNA interference cancer, siRNA cancer, epigenetic cancer
  - bispecific T cell engager, tumor infiltrating, neoantigen vaccine
  - PD-1 gene therapy, checkpoint gene, viral vector cancer

Usage:
    python3 fetch_data.py      (~25-35 min)
"""
import requests, json, time, pandas as pd
from pathlib import Path

Path("raw").mkdir(exist_ok=True)

CT_BASE = "https://clinicaltrials.gov/api/v2/studies"
CT_FIELDS = ",".join([
    "NCTId","BriefTitle","OfficialTitle",
    "OverallStatus","StartDate","PrimaryCompletionDate","CompletionDate",
    "Phase","StudyType","Condition","InterventionName","InterventionType",
    "PrimaryOutcomeMeasure","EnrollmentCount","EnrollmentType",
    "LeadSponsorName","LeadSponsorClass","LocationCountry",
    "WhyStopped","HasResults","ResultsFirstPostDate",
])

# 28 queries — covers the full gene/cell therapy in cancer landscape
QUERIES = [
    # Core cell therapies
    "CRISPR cancer",
    "CAR-T cell cancer neoplasm",
    "TCR therapy cancer neoplasm",
    "TIL therapy cancer tumor",
    "NK cell therapy cancer neoplasm",
    # Gene editing
    "gene editing cancer tumor",
    "gene therapy cancer neoplasm",
    "gene knockout cancer tumor",
    "base editing cancer tumor",
    "prime editing cancer tumor",
    "zinc finger nuclease cancer",
    "TALEN gene editing cancer",
    # Delivery vectors
    "chimeric antigen receptor cancer",
    "lentiviral transduction cancer",
    "retroviral vector cancer neoplasm",
    "AAV gene therapy cancer tumor",
    "adenoviral vector cancer neoplasm",
    "viral vector cancer treatment",
    # Cell therapies
    "adoptive cell therapy cancer neoplasm",
    "allogeneic cell therapy cancer",
    "stem cell gene therapy cancer",
    "natural killer cell cancer neoplasm",
    # Other modalities
    "oncolytic virus cancer tumor",
    "mRNA therapy cancer neoplasm",
    "T cell receptor engineered cancer",
    "immunotherapy gene transfer cancer",
    "neoantigen vaccine cancer tumor",
    "tumor infiltrating lymphocyte cancer",
]


def fetch_query(query_term, max_per_query=1000):
    studies, page_token, page = [], None, 1
    print(f"\n  Querying: '{query_term}'")
    while True:
        params = {
            "query.term": query_term,
            "filter.overallStatus": (
                "COMPLETED,TERMINATED,ACTIVE_NOT_RECRUITING,"
                "RECRUITING,WITHDRAWN,SUSPENDED,UNKNOWN"
            ),
            "pageSize": 100,
            "fields":   CT_FIELDS,
            "format":   "json",
        }
        if page_token: params["pageToken"] = page_token
        try:
            resp = requests.get(CT_BASE, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            print(f"    ERROR page {page}: {e}"); break
        batch = data.get("studies", [])
        if not batch: break
        studies.extend(batch)
        total = data.get("totalCount", "?")
        print(f"    Page {page}: +{len(batch)}  (this query: {len(studies)} / {total})")
        page_token = data.get("nextPageToken")
        if not page_token or len(studies) >= max_per_query: break
        page += 1
        time.sleep(0.25)
    return studies


def flatten_study(study):
    ps   = study.get("protocolSection", {})
    id_m = ps.get("identificationModule",       {})
    st_m = ps.get("statusModule",               {})
    des  = ps.get("designModule",               {})
    cond = ps.get("conditionsModule",           {})
    intr = ps.get("interventionsModule",        {})
    out  = ps.get("outcomesModule",             {})
    spon = ps.get("sponsorCollaboratorsModule", {})
    locs = ps.get("contactsLocationsModule",    {})
    enr  = des.get("enrollmentInfo",            {})
    phases        = des.get("phases", [])
    conditions    = cond.get("conditions", [])
    interventions = intr.get("interventions", [])
    intr_names    = [i.get("name","") for i in interventions]
    intr_types    = list(set(i.get("type","") for i in interventions))
    primary_out   = [o.get("measure","") for o in out.get("primaryOutcomes",[])]
    countries     = list({loc.get("country","") for loc in locs.get("locations",[]) if loc.get("country")})
    return {
        "nct_id":               id_m.get("nctId"),
        "brief_title":          id_m.get("briefTitle"),
        "overall_status":       st_m.get("overallStatus"),
        "start_date":           st_m.get("startDateStruct",{}).get("date"),
        "primary_completion":   st_m.get("primaryCompletionDateStruct",{}).get("date"),
        "completion_date":      st_m.get("completionDateStruct",{}).get("date"),
        "why_stopped":          st_m.get("whyStopped"),
        "has_results":          study.get("hasResults",False),
        "results_first_posted": st_m.get("resultsFirstPostDateStruct",{}).get("date"),
        "phase":                " / ".join(phases) if phases else None,
        "study_type":           des.get("studyType"),
        "enrollment_count":     enr.get("count"),
        "enrollment_type":      enr.get("type"),
        "conditions":           " | ".join(conditions),
        "n_conditions":         len(conditions),
        "interventions":        " | ".join(intr_names),
        "intervention_types":   " | ".join(intr_types),
        "primary_outcome":      primary_out[0] if primary_out else None,
        "n_primary_outcomes":   len(primary_out),
        "lead_sponsor":         spon.get("leadSponsor",{}).get("name"),
        "sponsor_class":        spon.get("leadSponsor",{}).get("class"),
        "location_countries":   " | ".join(sorted(countries)),
        "n_countries":          len(countries),
    }


print("="*65)
print(f"STEP 1: ClinicalTrials.gov  ({len(QUERIES)} queries, 1000/query cap)")
print("="*65)

seen = {}
for i, query in enumerate(QUERIES, 1):
    print(f"\n[{i}/{len(QUERIES)}]", end="")
    batch = fetch_query(query, max_per_query=1000)
    new = 0
    for s in batch:
        nct = s.get("protocolSection",{}).get("identificationModule",{}).get("nctId")
        if nct and nct not in seen:
            seen[nct] = s; new += 1
    print(f"    +{new} new unique  |  TOTAL: {len(seen):,}")

raw_studies = list(seen.values())
print(f"\n  TOTAL unique trials fetched: {len(raw_studies):,}")

with open("raw/clinicaltrials_raw.json","w") as f:
    json.dump(raw_studies, f, indent=2)

flat = [flatten_study(s) for s in raw_studies]
df_ct = pd.DataFrame(flat)
df_ct.to_csv("raw/clinicaltrials_studies.csv", index=False)
print(f"  Saved → raw/clinicaltrials_raw.json")
print(f"  Saved → raw/clinicaltrials_studies.csv")
print(f"\n  Status breakdown:")
print(df_ct["overall_status"].value_counts().to_string())


# ── PubMed ─────────────────────────────────────────────────────────────────
PUBMED = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"

def pubmed_count(query, year):
    try:
        r = requests.get(PUBMED, params={
            "db":"pubmed","retmax":0,"retmode":"json",
            "term":f"{query} AND {year}[PDAT]",
            "email":"analytics@example.com",
        }, timeout=15)
        return int(r.json()["esearchresult"]["count"])
    except: return None

print("\n"+"="*65)
print("STEP 2: PubMed Publication Counts (2013-2024)")
print("="*65)

pub_queries = {
    "crispr_cancer":        "CRISPR[Title/Abstract] AND cancer[Title/Abstract]",
    "cart_cancer":          "CAR-T[Title/Abstract] AND cancer[Title/Abstract]",
    "gene_therapy_cancer":  "gene therapy[Title/Abstract] AND cancer[Title/Abstract]",
    "til_therapy":          "tumor infiltrating lymphocyte[Title/Abstract] AND cancer[Title/Abstract]",
    "nk_cell_cancer":       "NK cell therapy[Title/Abstract] AND cancer[Title/Abstract]",
    "base_editing_cancer":  "base editing[Title/Abstract] AND cancer[Title/Abstract]",
    "ai_crispr":            "CRISPR[Title/Abstract] AND (artificial intelligence OR machine learning)[Title/Abstract]",
}

pub_records = []
for year in range(2013,2025):
    row = {"year":year}
    for col,q in pub_queries.items():
        count = pubmed_count(q,year)
        row[col] = count
        print(f"  {year}  {col:<28} = {count}")
        time.sleep(0.35)
    pub_records.append(row)

pd.DataFrame(pub_records).to_csv("raw/pubmed_counts_by_year.csv",index=False)
print(f"\n  Saved → raw/pubmed_counts_by_year.csv")

print("\n"+"="*65)
print("FETCH COMPLETE")
print("="*65)
print(f"  Trials fetched:  {len(df_ct):,}")
print(f"  PubMed points:   {len(pub_records)}")
print(f"\n  Next: python3 clean_data.py")
