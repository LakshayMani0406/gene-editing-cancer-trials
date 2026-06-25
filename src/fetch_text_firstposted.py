"""
fetch_text_firstposted.py: fetch the FIRST-POSTED (version 0) text for the strict cohort,
via the CT.gov history API (/api/int/studies/{nct}/history/0). This is the registration-time
text, used to test whether the combined text gain is genuine registration-time signal or an
artifact of post-registration edits. Caches to raw/text_fields_firstposted.json (resumable).
"""
import json, time
from pathlib import Path
import requests, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESOLVED = {"COMPLETED", "TERMINATED", "WITHDRAWN", "SUSPENDED"}
ncts = pd.read_csv(ROOT / "data" / "crispr_trials_clean.csv")
ncts = ncts[ncts["overall_status"].isin(RESOLVED)]["nct_id"].dropna().unique().tolist()
CACHE = ROOT / "raw" / "text_fields_firstposted.json"
out = json.loads(CACHE.read_text()) if CACHE.exists() else {}
HDR = {"accept": "application/json", "user-agent": "ct-research/1.0"}

t0, done = time.time(), 0
for i, nct in enumerate(ncts):
    if nct in out and "eligibility" in out[nct]:
        continue
    rec = None
    for _ in range(3):
        try:
            r = requests.get(f"https://clinicaltrials.gov/api/int/studies/{nct}/history/0",
                             timeout=30, headers=HDR)
            if r.status_code == 200:
                rec = r.json(); break
            time.sleep(1.0)
        except Exception:
            time.sleep(1.5)
    if rec is None:
        out[nct] = {"error": True}
    else:
        ps = rec.get("study", {}).get("protocolSection", {})
        st = ps.get("statusModule", {}) or {}
        out[nct] = {
            "eligibility": (ps.get("eligibilityModule", {}) or {}).get("eligibilityCriteria", "") or "",
            "brief_summary": (ps.get("descriptionModule", {}) or {}).get("briefSummary", "") or "",
            "detailed_description": (ps.get("descriptionModule", {}) or {}).get("detailedDescription", "") or "",
            "v0_first_post": (st.get("studyFirstPostDateStruct", {}) or {}).get("date"),
        }
    done += 1
    if done % 200 == 0:
        CACHE.write_text(json.dumps(out))
        print(f"  {i+1}/{len(ncts)} ({done} new) elapsed={time.time()-t0:.0f}s", flush=True)
    time.sleep(0.3)

CACHE.write_text(json.dumps(out))
he = sum(1 for v in out.values() if v.get("eligibility", "").strip())
hb = sum(1 for v in out.values() if v.get("brief_summary", "").strip())
err = sum(1 for v in out.values() if v.get("error"))
print(f"done: {len(out)} records | eligibility {he} | brief_summary {hb} | errors {err}", flush=True)
