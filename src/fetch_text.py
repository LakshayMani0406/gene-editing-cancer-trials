"""
fetch_text.py: re-fetch registration text fields for the strict-resolved cohort.

The original fetch (fetch_data.py) used a field mask that omitted the eligibility and
description modules, so the local raw/ has no text. This pulls EligibilityModule +
DescriptionModule + StatusModule (first-posted / last-updated dates) for the strict-resolved
NCT IDs from the open ClinicalTrials.gov API v2 (no key), and caches to raw/text_fields.json.

Run: python src/fetch_text.py
"""
import json, time
from pathlib import Path
import requests
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESOLVED = {"COMPLETED", "TERMINATED", "WITHDRAWN", "SUSPENDED"}
API = "https://clinicaltrials.gov/api/v2/studies"
FIELDS = "IdentificationModule,EligibilityModule,DescriptionModule,StatusModule"

df = pd.read_csv(ROOT / "data" / "crispr_trials_clean.csv")
ncts = df[df["overall_status"].isin(RESOLVED)]["nct_id"].dropna().unique().tolist()
print(f"strict-resolved NCT IDs to fetch: {len(ncts)}")

out, B = {}, 50
for i in range(0, len(ncts), B):
    batch = ncts[i:i + B]
    params = {"filter.ids": ",".join(batch), "fields": FIELDS, "pageSize": B, "format": "json"}
    for attempt in range(3):
        try:
            r = requests.get(API, params=params, timeout=60)
            r.raise_for_status()
            break
        except Exception as e:
            if attempt == 2:
                raise
            time.sleep(1.5)
    for s in r.json().get("studies", []):
        ps = s.get("protocolSection", {})
        nct = ps.get("identificationModule", {}).get("nctId")
        if not nct:
            continue
        desc = ps.get("descriptionModule", {})
        st = ps.get("statusModule", {})
        out[nct] = {
            "eligibility": ps.get("eligibilityModule", {}).get("eligibilityCriteria", "") or "",
            "brief_summary": desc.get("briefSummary", "") or "",
            "detailed_description": desc.get("detailedDescription", "") or "",
            "first_posted": st.get("studyFirstPostDateStruct", {}).get("date"),
            "last_updated": st.get("lastUpdatePostDateStruct", {}).get("date"),
        }
    print(f"  {min(i + B, len(ncts))}/{len(ncts)} requested, {len(out)} returned", end="\r")
    time.sleep(0.3)

(ROOT / "raw" / "text_fields.json").write_text(json.dumps(out, indent=1))
print(f"\nSaved {len(out)} records to raw/text_fields.json")
he = sum(1 for v in out.values() if v["eligibility"].strip())
hb = sum(1 for v in out.values() if v["brief_summary"].strip())
print(f"non-empty eligibility: {he} ({he/len(ncts):.1%}) | brief_summary: {hb} ({hb/len(ncts):.1%})")
