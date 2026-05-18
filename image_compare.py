"""
image_compare.py — Biomarker Image Analysis System
Pharma-grade UI: clinical trial management aesthetic
All three modes: Single / Comparison / Cancer Type Match
"""
from pathlib import Path
import json

Path("outputs").mkdir(exist_ok=True)

DATASET_TOTAL = 4460

PROTEINS = {
    "CD19":       {"trials":854, "pct":19.1, "color":"#0ea5e9","fda":True, "cancer":"Lymphoma, Leukemia (ALL)","pdb":"6AL6"},
    "PD-1":       {"trials":450, "pct":10.1, "color":"#10b981","fda":True, "cancer":"Melanoma, Lung Cancer","pdb":"5IUS"},
    "CD33":       {"trials":364, "pct":8.2,  "color":"#3b82f6","fda":True, "cancer":"Leukemia (AML)","pdb":"5UCM"},
    "EGFR":       {"trials":353, "pct":7.9,  "color":"#f59e0b","fda":True, "cancer":"Lung Cancer, Brain/CNS","pdb":"2GS7"},
    "HER2":       {"trials":259, "pct":5.8,  "color":"#ec4899","fda":True, "cancer":"Breast Cancer, Gastric","pdb":"3PP0"},
    "BCMA":       {"trials":229, "pct":5.1,  "color":"#8b5cf6","fda":True, "cancer":"Multiple Myeloma","pdb":"1XU0"},
    "GPC3":       {"trials":129, "pct":2.9,  "color":"#84cc16","fda":False,"cancer":"Liver Cancer (HCC)","pdb":"5XQ2"},
    "KRAS":       {"trials":105, "pct":2.4,  "color":"#ef4444","fda":True, "cancer":"Pancreatic, Colorectal","pdb":"4OBE"},
    "TP53":       {"trials":95,  "pct":2.1,  "color":"#f97316","fda":False,"cancer":"Pan-cancer, Ovarian","pdb":"2OCJ"},
    "Mesothelin": {"trials":62,  "pct":1.4,  "color":"#14b8a6","fda":False,"cancer":"Mesothelioma, Ovarian","pdb":"3UAK"},
}

CANCERS = {
    "Lymphoma":           {"trials":592,"color":"#0ea5e9","therapy":"CAR-T, BiTE"},
    "Leukemia (AML)":     {"trials":364,"color":"#3b82f6","therapy":"CAR-T, ADC"},
    "Leukemia (ALL)":     {"trials":262,"color":"#10b981","therapy":"CAR-T, BiTE"},
    "Lung Cancer":        {"trials":353,"color":"#f59e0b","therapy":"TKI, Checkpoint"},
    "Melanoma":           {"trials":194,"color":"#8b5cf6","therapy":"Checkpoint, TIL"},
    "Multiple Myeloma":   {"trials":229,"color":"#7c3aed","therapy":"CAR-T, ADC"},
    "Breast Cancer":      {"trials":259,"color":"#ec4899","therapy":"ADC, CAR-T"},
    "Brain/CNS":          {"trials":145,"color":"#6366f1","therapy":"CAR-T, CRISPR"},
    "Liver Cancer (HCC)": {"trials":129,"color":"#84cc16","therapy":"CAR-T, BiTE"},
    "Colorectal Cancer":  {"trials":184,"color":"#fb923c","therapy":"Checkpoint, ADC"},
    "Pancreatic Cancer":  {"trials":105,"color":"#ef4444","therapy":"CRISPR, mRNA"},
    "Ovarian Cancer":     {"trials":98, "color":"#14b8a6","therapy":"CAR-T, ADC"},
}

P_JS  = json.dumps(PROTEINS, separators=(',',':'))
C_JS  = json.dumps(CANCERS,  separators=(',',':'))
DT_JS = str(DATASET_TOTAL)

PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>BioMarker Vision — Oncology Target Analysis</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root{
  --navy:#0f2044;--navy2:#162a52;--navy3:#1e3a6e;
  --white:#ffffff;--off:#f8fafc;--sheet:#f1f5f9;
  --border:#cbd5e1;--border2:#e2e8f0;
  --blue:#1d4ed8;--blue2:#2563eb;--blue3:#3b82f6;--blue-pale:#dbeafe;
  --green:#047857;--green2:#059669;--green-pale:#d1fae5;
  --amber:#b45309;--amber-pale:#fef3c7;
  --red:#dc2626;--red-pale:#fee2e2;
  --purple:#6d28d9;--purple-pale:#ede9fe;
  --slate:#475569;--slate2:#64748b;--slate3:#94a3b8;
  --text:#0f172a;--text2:#1e293b;--text3:#334155;
  --mono:'JetBrains Mono',monospace;
  --sans:'Inter',system-ui,sans-serif;
  --radius:6px;--radius2:10px;
  --shadow:0 1px 3px rgba(0,0,0,.08),0 1px 2px rgba(0,0,0,.05);
  --shadow2:0 4px 6px rgba(0,0,0,.05),0 2px 4px rgba(0,0,0,.04);
}
*{margin:0;padding:0;box-sizing:border-box}
html,body{min-height:100%;background:var(--sheet);color:var(--text);font-family:var(--sans);font-size:14px;line-height:1.5}

/* ─── Header ─── */
.hdr{
  background:linear-gradient(135deg,var(--navy) 0%,var(--navy3) 100%);
  border-bottom:3px solid var(--blue2);
  padding:0 32px;
  height:60px;
  display:flex;align-items:center;justify-content:space-between;
  position:sticky;top:0;z-index:100;
}
.hdr-left{display:flex;align-items:center;gap:14px}
.hdr-logo{
  width:32px;height:32px;border-radius:8px;
  background:linear-gradient(135deg,var(--blue2),var(--blue3));
  display:flex;align-items:center;justify-content:center;
  font-size:14px;color:white;font-weight:700;letter-spacing:-.5px;
  flex-shrink:0;
}
.hdr-title{font-size:.95rem;font-weight:600;color:#fff;letter-spacing:-.01em}
.hdr-sub{font-size:.65rem;color:#93c5fd;font-family:var(--mono);margin-top:1px}
.hdr-right{display:flex;align-items:center;gap:6px}
.hdr-pill{
  font-size:.62rem;font-family:var(--mono);padding:3px 9px;border-radius:20px;
  border:1px solid rgba(255,255,255,.15);color:rgba(255,255,255,.7);
  background:rgba(255,255,255,.07);
}
.hdr-link{
  font-size:.65rem;color:#93c5fd;text-decoration:none;
  border:1px solid rgba(147,197,253,.3);border-radius:4px;padding:3px 9px;
}
.hdr-link:hover{background:rgba(147,197,253,.1)}

/* ─── Tab nav ─── */
.tabnav{
  background:var(--white);border-bottom:1px solid var(--border);
  padding:0 32px;display:flex;align-items:center;gap:0;
}
.ntab{
  padding:14px 20px;font-size:.75rem;font-weight:500;cursor:pointer;
  color:var(--slate2);border-bottom:2px solid transparent;transition:.15s;
  display:flex;align-items:center;gap:6px;white-space:nowrap;
}
.ntab .dot{width:6px;height:6px;border-radius:50%;background:var(--border);transition:.15s}
.ntab.on{color:var(--blue2);border-bottom-color:var(--blue2)}
.ntab.on .dot{background:var(--blue2)}
.ntab:hover:not(.on){color:var(--text);border-bottom-color:var(--border)}
.nsep{color:var(--border2);padding:0 2px;font-size:.9rem;align-self:center}

/* ─── Main layout ─── */
.main-wrap{max-width:1400px;margin:0 auto;padding:24px 32px}

/* ─── Upload panels ─── */
.upload-row{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px}
.upload-panel{background:var(--white);border:1px solid var(--border);border-radius:var(--radius2);overflow:hidden;box-shadow:var(--shadow)}
.panel-hdr{
  padding:10px 16px;border-bottom:1px solid var(--border2);
  display:flex;align-items:center;gap:8px;
  background:var(--off);
}
.panel-letter{
  width:22px;height:22px;border-radius:4px;
  display:flex;align-items:center;justify-content:center;
  font-size:.7rem;font-weight:700;flex-shrink:0;
}
.la{background:var(--blue-pale);color:var(--blue2)}
.lb{background:var(--purple-pale);color:var(--purple)}
.panel-label{font-size:.72rem;font-weight:600;color:var(--text2);text-transform:uppercase;letter-spacing:.06em}
.panel-body{padding:12px}

.dz{
  border:1.5px dashed var(--border);border-radius:var(--radius);
  background:var(--off);min-height:150px;
  display:flex;flex-direction:column;align-items:center;justify-content:center;
  gap:6px;cursor:pointer;transition:.15s;padding:20px;text-align:center;
}
.dz:hover,.dz.over{border-color:var(--blue3);background:var(--blue-pale)}
.dz-icon{font-size:1.8rem;opacity:.3}
.dz-text{font-size:.75rem;color:var(--slate2);font-weight:500}
.dz-sub{font-size:.63rem;color:var(--slate3)}

.pw{display:none;position:relative;border-radius:var(--radius);overflow:hidden;background:#000;min-height:150px;align-items:center;justify-content:center}
.pimg{max-height:180px;max-width:100%;object-fit:contain;border-radius:4px}
.pbadge-row{position:absolute;top:6px;right:6px;display:flex;gap:3px}
.pbadge{background:rgba(0,0,0,.7);border:1px solid rgba(255,255,255,.15);border-radius:3px;font-size:.58rem;font-family:var(--mono);padding:1px 5px;color:#cbd5e1}
.clr{position:absolute;top:6px;left:6px;background:rgba(220,38,38,.85);border:none;color:#fff;font-size:.62rem;border-radius:3px;padding:2px 7px;cursor:pointer;font-weight:600}

/* ─── Action bar ─── */
.action-bar{
  background:var(--white);border:1px solid var(--border);border-radius:var(--radius2);
  padding:12px 16px;margin-bottom:20px;
  display:flex;align-items:center;gap:12px;flex-wrap:wrap;
  box-shadow:var(--shadow);
}
.key-section{display:flex;align-items:center;gap:8px;flex:1;min-width:260px}
.key-label{font-size:.65rem;font-weight:600;color:var(--slate2);text-transform:uppercase;letter-spacing:.06em;white-space:nowrap;font-family:var(--mono)}
.key-input{
  flex:1;background:var(--off);border:1px solid var(--border);border-radius:var(--radius);
  padding:7px 11px;font-size:.72rem;font-family:var(--mono);color:var(--text);outline:none;
  transition:.15s;
}
.key-input:focus{border-color:var(--blue3);box-shadow:0 0 0 3px rgba(59,130,246,.12)}
.show-btn{
  background:var(--white);border:1px solid var(--border);color:var(--slate2);
  font-size:.63rem;font-family:var(--mono);border-radius:4px;padding:5px 10px;cursor:pointer;
  white-space:nowrap;
}
.show-btn:hover{border-color:var(--blue3);color:var(--blue2)}
.run-btn{
  background:var(--blue2);border:none;color:#fff;
  font-size:.75rem;font-weight:600;padding:8px 20px;border-radius:var(--radius);
  cursor:pointer;transition:.15s;display:flex;align-items:center;gap:7px;white-space:nowrap;
}
.run-btn:hover:not(:disabled){background:var(--blue);box-shadow:0 0 0 3px rgba(37,99,235,.2)}
.run-btn:disabled{background:var(--slate3);cursor:not-allowed}
.run-icon{width:14px;height:14px;border-radius:50%;background:rgba(255,255,255,.25);display:flex;align-items:center;justify-content:center;font-size:.6rem}
.hint-text{font-size:.65rem;color:var(--slate3);font-family:var(--mono)}
.key-err{display:none;font-size:.62rem;color:var(--red);font-family:var(--mono)}

/* ─── Results ─── */
.results-wrap{min-height:340px}

/* Empty */
.empty-state{
  background:var(--white);border:1px solid var(--border);border-radius:var(--radius2);
  display:flex;flex-direction:column;align-items:center;justify-content:center;
  min-height:320px;gap:14px;text-align:center;padding:40px;
  box-shadow:var(--shadow);
}
.empty-icon{
  width:56px;height:56px;border-radius:14px;
  background:var(--blue-pale);
  display:flex;align-items:center;justify-content:center;font-size:1.5rem;
}
.empty-title{font-size:1rem;font-weight:600;color:var(--text2)}
.empty-sub{font-size:.75rem;color:var(--slate2);max-width:380px;line-height:1.7}
.empty-tags{display:flex;gap:6px;flex-wrap:wrap;justify-content:center;margin-top:4px}
.etag{background:var(--blue-pale);color:var(--blue2);font-size:.62rem;border-radius:20px;padding:2px 10px;font-weight:500}

/* Loading */
.loading-state{
  background:var(--white);border:1px solid var(--border);border-radius:var(--radius2);
  display:none;flex-direction:column;align-items:center;justify-content:center;
  min-height:320px;gap:16px;box-shadow:var(--shadow);
}
.load-spinner{
  width:40px;height:40px;border:3px solid var(--border2);border-top-color:var(--blue2);
  border-radius:50%;animation:spin .8s linear infinite;
}
@keyframes spin{to{transform:rotate(360deg)}}
.load-text{font-size:.8rem;color:var(--text2);font-weight:500}
.load-sub{font-size:.65rem;color:var(--slate3);font-family:var(--mono)}
.load-steps{display:flex;gap:8px;margin-top:4px}
.lstep{font-size:.6rem;color:var(--slate3);font-family:var(--mono);padding:2px 8px;border:1px solid var(--border2);border-radius:10px}
.lstep.active{background:var(--blue-pale);color:var(--blue2);border-color:var(--blue3)}

/* Results container */
#rout{display:none}

/* Summary bar */
.summary-bar{
  background:linear-gradient(135deg,var(--navy),var(--navy2));
  border-radius:var(--radius2);padding:14px 18px;margin-bottom:16px;
  color:#e2e8f0;font-size:.75rem;line-height:1.7;
}
.summary-bar strong{color:#fff;font-weight:600}
.sum-a{color:#7dd3fc;font-weight:600}
.sum-b{color:#c4b5fd;font-weight:600}

/* Two-column results */
.res-grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.res-panel{background:var(--white);border:1px solid var(--border);border-radius:var(--radius2);overflow:hidden;box-shadow:var(--shadow)}
.res-phead{padding:10px 16px;border-bottom:1px solid var(--border2);background:var(--off);display:flex;align-items:center;justify-content:space-between}
.res-ptitle{font-size:.7rem;font-weight:600;text-transform:uppercase;letter-spacing:.07em;color:var(--navy)}
.res-psub{font-size:.6rem;color:var(--slate3);font-family:var(--mono)}
.res-pbody{padding:10px 14px}

/* Rank rows */
.rr{display:flex;align-items:center;gap:8px;padding:6px 0;border-bottom:1px solid var(--border2)}
.rr:last-child{border-bottom:none}
.rr-n{font-size:.65rem;font-weight:700;font-family:var(--mono);width:20px;text-align:center;flex-shrink:0;color:var(--slate3)}
.rr-n.top{color:#d97706;background:var(--amber-pale);border-radius:3px;padding:1px 3px}
.rr-name{font-size:.72rem;font-weight:600;flex-shrink:0;width:92px;font-family:var(--mono)}
.rr-track{flex:1;background:var(--sheet);border-radius:2px;height:6px;overflow:hidden}
.rr-fill{height:100%;border-radius:2px;transition:width .65s cubic-bezier(.22,1,.36,1);width:0%}
.rr-pct{font-size:.65rem;font-family:var(--mono);font-weight:600;width:32px;text-align:right;flex-shrink:0}
.rr-trials{font-size:.58rem;font-family:var(--mono);color:var(--slate3);width:36px;text-align:right;flex-shrink:0}
.rr-logbar{width:36px;flex-shrink:0}
.rr-logtrack{height:3px;background:var(--sheet);border-radius:2px;overflow:hidden;margin-bottom:1px}
.rr-logfill{height:100%;border-radius:2px;opacity:.5}
.rr-pct-ds{font-size:.54rem;color:var(--slate3);font-family:var(--mono);text-align:right}
.fda-badge{
  font-size:.52rem;font-weight:600;padding:1px 5px;border-radius:3px;
  background:var(--green-pale);color:var(--green);border:1px solid #a7f3d0;
  flex-shrink:0;letter-spacing:.02em;
}

/* Detail cards */
.dc{background:var(--off);border:1px solid var(--border2);border-radius:var(--radius);padding:10px 12px;margin-bottom:8px;border-left:3px solid}
.dc:last-child{margin-bottom:0}
.dc-row{display:flex;justify-content:space-between;align-items:center;margin-bottom:3px}
.dc-name{font-size:.74rem;font-weight:600;font-family:var(--mono)}
.dc-chip{font-size:.58rem;font-family:var(--mono);padding:1px 6px;border-radius:3px;background:var(--sheet);color:var(--slate2);border:1px solid var(--border)}
.dc-cancer{font-size:.63rem;color:var(--slate2);margin-bottom:4px}
.dc-reason{font-size:.65rem;color:var(--text3);line-height:1.55;margin-bottom:5px;font-style:italic}
.dc-track{height:4px;background:var(--border2);border-radius:2px;overflow:hidden}
.dc-fill{height:100%;border-radius:2px;transition:width .65s cubic-bezier(.22,1,.36,1);width:0%}

/* Analysis box */
.analysis-box{
  background:var(--white);border:1px solid var(--border);border-radius:var(--radius2);
  padding:16px 18px;margin-top:16px;box-shadow:var(--shadow);
}
.analysis-title{
  font-size:.65rem;font-weight:600;text-transform:uppercase;letter-spacing:.07em;
  color:var(--navy);font-family:var(--mono);margin-bottom:8px;
  display:flex;align-items:center;gap:6px;
}
.analysis-title::before{content:'';width:3px;height:13px;background:var(--blue2);border-radius:2px;display:inline-block}
.analysis-text{font-size:.73rem;color:var(--text3);line-height:1.75}
.analysis-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:14px}

/* Comparison mode */
.cmp-table{width:100%;border-collapse:collapse}
.cmp-thead td{
  padding:8px 10px;font-size:.62rem;font-weight:600;text-transform:uppercase;
  letter-spacing:.07em;color:var(--slate2);border-bottom:2px solid var(--border);
  background:var(--off);
}
.cmp-row-tr{border-bottom:1px solid var(--border2)}
.cmp-row-tr:hover{background:var(--off)}
.cmp-row-tr td{padding:7px 10px;vertical-align:middle}
.cmp-pname{font-size:.72rem;font-weight:600;font-family:var(--mono);display:flex;align-items:center;gap:6px}
.cmp-colordot{width:8px;height:8px;border-radius:50%;flex-shrink:0}
.cmp-cell{display:flex;align-items:center;gap:6px}
.cmp-track{flex:1;height:6px;background:var(--sheet);border-radius:2px;overflow:hidden}
.cmp-fa{height:100%;background:#0ea5e9;border-radius:2px;transition:width .7s cubic-bezier(.22,1,.36,1);width:0%}
.cmp-fb{height:100%;background:#8b5cf6;border-radius:2px;transition:width .7s cubic-bezier(.22,1,.36,1);width:0%}
.cmp-pct{font-size:.62rem;font-family:var(--mono);font-weight:600;width:28px;flex-shrink:0}
.cmp-win{font-size:.56rem;font-family:var(--mono);padding:1px 5px;border-radius:3px;flex-shrink:0;font-weight:600}
.wa{background:var(--blue-pale);color:var(--blue2)}
.wb{background:var(--purple-pale);color:var(--purple)}
.wt{background:var(--sheet);color:var(--slate3)}

/* Diff section */
.diff-section{
  background:var(--white);border:1px solid var(--border);border-radius:var(--radius2);
  padding:14px 16px;margin-top:16px;box-shadow:var(--shadow);
}
.diff-title{font-size:.65rem;font-weight:600;text-transform:uppercase;letter-spacing:.07em;color:var(--amber);font-family:var(--mono);margin-bottom:10px;display:flex;align-items:center;gap:6px}
.diff-title::before{content:'';width:3px;height:13px;background:var(--amber);border-radius:2px;display:inline-block}
.diff-sub{font-size:.62rem;font-weight:600;font-family:var(--mono);margin-bottom:6px;margin-top:8px}
.diff-rr{display:flex;justify-content:space-between;align-items:center;gap:12px;padding:5px 0;border-bottom:1px solid var(--border2);font-size:.7rem}
.diff-rr:last-child{border-bottom:none}
.diff-name{font-family:var(--mono);font-weight:600}
.diff-delta{font-family:var(--mono);font-weight:600;font-size:.65rem}

/* History strip */
.hist-strip{
  background:var(--white);border:1px solid var(--border);border-radius:var(--radius2);
  padding:10px 16px;margin-top:16px;display:none;flex-direction:row;
  align-items:center;gap:10px;overflow-x:auto;box-shadow:var(--shadow);
}
.hist-lbl{font-size:.6rem;font-family:var(--mono);color:var(--slate3);text-transform:uppercase;letter-spacing:.07em;flex-shrink:0}
.hist-pair{display:flex;gap:4px;cursor:pointer;padding:3px;border-radius:5px;border:1px solid transparent;transition:.15s;flex-shrink:0}
.hist-pair:hover{border-color:var(--blue3);background:var(--blue-pale)}
.hist-img{width:36px;height:36px;object-fit:cover;border-radius:3px;border:1px solid var(--border2)}
.hist-vs{font-size:.55rem;color:var(--slate3);align-self:center;font-family:var(--mono);padding:0 2px}

@media(max-width:900px){
  .upload-row,.res-grid,.analysis-grid{grid-template-columns:1fr}
  .main-wrap{padding:16px}
  .hdr{padding:0 16px}
  .tabnav{padding:0 16px}
}
</style>
</head>
<body>

<!-- Header -->
<div class="hdr">
  <div class="hdr-left">
    <div class="hdr-logo">Bx</div>
    <div>
      <div class="hdr-title">BioMarker Vision &mdash; Oncology Target Analysis</div>
      <div class="hdr-sub">Gene-Editing &amp; Cell Therapy Trials &middot; n=4,460 &middot; Claude Vision AI &middot; Northeastern MS Analytics</div>
    </div>
  </div>
  <div class="hdr-right">
    <span class="hdr-pill">Lakshay Mani</span>
    <span class="hdr-pill">v2.1</span>
    <a href="dashboard.html" class="hdr-link">&larr; Dashboard</a>
    <a href="molecules.html" class="hdr-link">3D Viewer</a>
  </div>
</div>

<!-- Tab nav -->
<div class="tabnav">
  <div class="ntab on" id="tab-single" onclick="setMode('single')"><span class="dot"></span>Single Image</div>
  <div class="nsep">|</div>
  <div class="ntab" id="tab-compare" onclick="setMode('compare')"><span class="dot"></span>Side-by-Side Comparison</div>
  <div class="nsep">|</div>
  <div class="ntab" id="tab-cancer" onclick="setMode('cancer')"><span class="dot"></span>Cancer Type Match</div>
</div>

<div class="main-wrap">

<!-- Upload row -->
<div class="upload-row" id="upload-row">
  <div class="upload-panel">
    <div class="panel-hdr">
      <div class="panel-letter la">A</div>
      <div class="panel-label">Image A &mdash; Primary Specimen</div>
    </div>
    <div class="panel-body">
      <div class="dz" id="dz-a" onclick="document.getElementById('fi-a').click()">
        <div class="dz-icon">&#128247;</div>
        <div class="dz-text">Drop image or click to upload</div>
        <div class="dz-sub">H&amp;E slides &middot; protein structures &middot; microscopy &middot; Western blots</div>
      </div>
      <div class="pw" id="pw-a">
        <img class="pimg" id="pi-a" src="" alt="">
        <div class="pbadge-row"><span class="pbadge" id="ps-a"></span><span class="pbadge" id="pt-a"></span></div>
        <button class="clr" onclick="clearSlot('a')">&#10005; Remove</button>
      </div>
      <input type="file" id="fi-a" accept="image/*" style="display:none">
    </div>
  </div>
  <div class="upload-panel" id="slot-b">
    <div class="panel-hdr">
      <div class="panel-letter lb">B</div>
      <div class="panel-label">Image B &mdash; Comparative Specimen</div>
    </div>
    <div class="panel-body">
      <div class="dz" id="dz-b" onclick="document.getElementById('fi-b').click()">
        <div class="dz-icon">&#128247;</div>
        <div class="dz-text">Drop second image for comparison</div>
        <div class="dz-sub">Differential scoring across all 10 protein targets</div>
      </div>
      <div class="pw" id="pw-b">
        <img class="pimg" id="pi-b" src="" alt="">
        <div class="pbadge-row"><span class="pbadge" id="ps-b"></span><span class="pbadge" id="pt-b"></span></div>
        <button class="clr" onclick="clearSlot('b')">&#10005; Remove</button>
      </div>
      <input type="file" id="fi-b" accept="image/*" style="display:none">
    </div>
  </div>
</div>

<!-- Action bar -->
<div class="action-bar">
  <div class="key-section">
    <span class="key-label">API Key</span>
    <input class="key-input" type="password" id="api-key" placeholder="sk-ant-api03-..." autocomplete="off">
    <button class="show-btn" onclick="var f=document.getElementById('api-key');f.type=f.type==='password'?'text':'password';this.textContent=f.type==='password'?'Show':'Hide'">Show</button>
    <span class="key-err" id="key-err">API key required</span>
  </div>
  <button class="run-btn" id="abtn" onclick="runAnalysis()" disabled>
    <span class="run-icon">&#9654;</span>Run Analysis
  </button>
  <span class="hint-text" id="mhint">Upload an image to begin</span>
</div>

<!-- Results -->
<div class="results-wrap">
  <div class="empty-state" id="empty">
    <div class="empty-icon">&#129514;</div>
    <div class="empty-title">Ready for Biomarker Analysis</div>
    <div class="empty-sub">Upload a biological image to rank protein targets by AI-assessed relevance. Scores validated against 4,460 gene-editing clinical trials.</div>
    <div class="empty-tags">
      <span class="etag">H&amp;E Histology</span>
      <span class="etag">Protein Structures</span>
      <span class="etag">Microscopy</span>
      <span class="etag">Western Blots</span>
      <span class="etag">IHC Slides</span>
    </div>
  </div>
  <div class="loading-state" id="loading">
    <div class="load-spinner"></div>
    <div class="load-text">Analyzing specimen</div>
    <div class="load-sub">Claude Vision &middot; Oncology target classification</div>
    <div class="load-steps">
      <span class="lstep active">Image decode</span>
      <span class="lstep">Feature extraction</span>
      <span class="lstep">Target ranking</span>
    </div>
  </div>
  <div id="rout"></div>
</div>

<!-- History strip -->
<div class="hist-strip" id="hstrip">
  <span class="hist-lbl">Session History</span>
  <div id="hinner" style="display:flex;gap:8px"></div>
</div>

</div><!-- /main-wrap -->

<script>
var P = __PROTEINS__;
var C = __CANCERS__;
var DT = __DSTOTAL__;
var mode = 'single';
var imgA = null, imgB = null;
var hist = [];

function setMode(m){
  mode = m;
  ['single','compare','cancer'].forEach(function(t){document.getElementById('tab-'+t).classList.toggle('on',t===m);});
  document.getElementById('slot-b').style.display = (m==='compare') ? '' : 'none';
  document.getElementById('mhint').textContent =
    m==='single'  ? 'Ranks all 10 protein targets by image relevance' :
    m==='compare' ? 'Compare two images: differential scoring across protein targets' :
                    'Matches image to 12 cancer types with trial counts';
  checkReady();
}

['a','b'].forEach(function(s){
  var dz=document.getElementById('dz-'+s);
  dz.addEventListener('dragover',function(e){e.preventDefault();dz.classList.add('over');});
  dz.addEventListener('dragleave',function(){dz.classList.remove('over');});
  dz.addEventListener('drop',function(e){e.preventDefault();dz.classList.remove('over');var f=e.dataTransfer.files[0];if(f&&f.type.startsWith('image/'))loadFile(s,f);});
  document.getElementById('fi-'+s).addEventListener('change',function(e){if(e.target.files[0])loadFile(s,e.target.files[0]);});
});

function loadFile(slot,file){
  var reader=new FileReader();
  reader.onload=function(e){
    var obj={dataUrl:e.target.result,base64:e.target.result.split(',')[1],mime:file.type||'image/jpeg',name:file.name,size:(file.size/1024).toFixed(0)+' KB'};
    if(slot==='a') imgA=obj; else imgB=obj;
    document.getElementById('dz-'+slot).style.display='none';
    document.getElementById('pw-'+slot).style.display='flex';
    document.getElementById('pi-'+slot).src=obj.dataUrl;
    document.getElementById('ps-'+slot).textContent=obj.size;
    document.getElementById('pt-'+slot).textContent=obj.mime.split('/')[1].toUpperCase();
    checkReady();
  };
  reader.readAsDataURL(file);
}

function clearSlot(slot){
  if(slot==='a') imgA=null; else imgB=null;
  document.getElementById('dz-'+slot).style.display='flex';
  document.getElementById('pw-'+slot).style.display='none';
  document.getElementById('fi-'+slot).value='';
  checkReady();
}

function checkReady(){
  var ok=(mode==='compare')?(imgA&&imgB):imgA;
  document.getElementById('abtn').disabled=!ok;
  if(!ok){document.getElementById('empty').style.display='flex';document.getElementById('loading').style.display='none';document.getElementById('rout').style.display='none';}
}

function showLoading(){document.getElementById('empty').style.display='none';document.getElementById('loading').style.display='flex';document.getElementById('rout').style.display='none';}
function showOut(html){document.getElementById('empty').style.display='none';document.getElementById('loading').style.display='none';var r=document.getElementById('rout');r.style.display='block';r.innerHTML=html;tick();}
function tick(){requestAnimationFrame(function(){requestAnimationFrame(function(){document.querySelectorAll('[data-w]').forEach(function(el){el.style.width=el.dataset.w+'%';});});});}

function logScale(trials){return Math.round(Math.log(trials+1)/Math.log(DT+1)*100);}

function rankRow(n,name,score,color,trials,fda){
  var p=Math.max(0,Math.min(100,score||0));
  var ls=logScale(trials);
  var pctDS=((trials/DT)*100).toFixed(1);
  var isTop=n===1;
  return '<div class="rr">'+
    '<div class="rr-n'+(isTop?' top':'')+'">'+(isTop?'#1':n)+'</div>'+
    '<div class="rr-name" style="color:'+color+'">'+name+'</div>'+
    '<div class="rr-track"><div class="rr-fill" style="background:'+color+'" data-w="'+p+'"></div></div>'+
    '<div class="rr-pct" style="color:'+color+'">'+p+'%</div>'+
    '<div class="rr-trials">'+trials+'</div>'+
    '<div class="rr-logbar"><div class="rr-logtrack"><div class="rr-logfill" style="background:'+color+';width:'+ls+'%"></div></div><div class="rr-pct-ds">'+pctDS+'%</div></div>'+
    (fda?'<span class="fda-badge">FDA</span>':'<span style="width:32px;display:inline-block"></span>')+
    '</div>';
}

function detailCard(name,score,color,sub1,sub2,reason){
  var p=Math.max(0,Math.min(100,score||0));
  return '<div class="dc" style="border-left-color:'+color+'">'+
    '<div class="dc-row"><span class="dc-name" style="color:'+color+'">'+name+'</span><span class="dc-chip">'+sub1+'</span></div>'+
    '<div class="dc-cancer">'+sub2+'</div>'+
    (reason?'<div class="dc-reason">&ldquo;'+reason+'&rdquo;</div>':'')+
    '<div class="dc-track"><div class="dc-fill" style="background:'+color+'" data-w="'+p+'"></div></div>'+
    '</div>';
}

async function runAnalysis(){
  var key=document.getElementById('api-key').value.trim();
  if(!key){
    var ki=document.getElementById('api-key');ki.style.borderColor='var(--red)';ki.focus();
    var err=document.getElementById('key-err');err.style.display='inline';
    setTimeout(function(){ki.style.borderColor='';err.style.display='none';},3500);return;
  }
  if(!imgA){
    var msg=document.getElementById('mhint');var old=msg.textContent;
    msg.style.color='var(--red)';msg.textContent='Upload Image A first';
    setTimeout(function(){msg.style.color='';msg.textContent=old;},2500);return;
  }
  if(mode==='compare'&&!imgB){
    var msg=document.getElementById('mhint');var old=msg.textContent;
    msg.style.color='var(--red)';msg.textContent='Upload Image B for comparison mode';
    setTimeout(function(){msg.style.color='';msg.textContent=old;},2500);return;
  }
  showLoading();
  document.getElementById('abtn').disabled=true;

  var pList=Object.keys(P).map(function(k){return k+' ('+P[k].cancer+', '+P[k].trials+' trials)';}).join(', ');
  var cList=Object.keys(C).map(function(k){return k+' ('+C[k].trials+' trials)';}).join(', ');
  var prompt;

  if(mode==='single'){
    prompt='You are an expert structural biologist analyzing a biomedical image for a cancer research dashboard.\n\nRank ALL 10 protein targets by relevance to this image. Consider structural similarity, cancer type depicted, therapy type, domain patterns, markers, or any relevant connection.\n\nProtein targets: '+pList+'\n\nReturn ONLY valid JSON:\n{"image_type":"brief type","summary":"one sentence","explanation":"2-3 sentences explaining WHY top matches rank highest based on what you observe","rankings":[{"protein":"CD19","score":85,"reason":"one-line reason"},...all 10 sorted highest to lowest, scores 0-100]}';
  } else if(mode==='cancer'){
    prompt='You are an oncologist analyzing a medical image.\n\nRank ALL 12 cancer types by how relevant/likely this image is for each one. Consider: tissue type, cell morphology, staining patterns, disease context.\n\nCancer types: '+cList+'\n\nReturn ONLY valid JSON:\n{"image_type":"brief type","summary":"one sentence","explanation":"2-3 sentences","rankings":[{"cancer":"Lymphoma","score":85,"reason":"one-line"},...all 12 sorted highest to lowest, scores 0-100]}';
  } else {
    prompt='You are comparing TWO biomedical images. Image A is the FIRST image, Image B is the SECOND image.\n\nFor each of the 10 protein targets, give SEPARATE scores for Image A and Image B.\n\nProtein targets: '+pList+'\n\nReturn ONLY valid JSON:\n{"summary_a":"one sentence about Image A","summary_b":"one sentence about Image B","explanation_a":"structural features seen in A","explanation_b":"structural features seen in B","comparison":[{"protein":"CD19","score_a":85,"score_b":40,"reason":"why they differ"},...all 10]}';
  }

  var content=[{type:'image',source:{type:'base64',media_type:imgA.mime,data:imgA.base64}}];
  if(mode==='compare'&&imgB) content.push({type:'image',source:{type:'base64',media_type:imgB.mime,data:imgB.base64}});
  content.push({type:'text',text:prompt});

  try{
    var resp=await fetch('https://api.anthropic.com/v1/messages',{method:'POST',headers:{'Content-Type':'application/json','x-api-key':key,'anthropic-version':'2023-06-01','anthropic-dangerous-direct-browser-access':'true'},body:JSON.stringify({model:'claude-sonnet-4-5',max_tokens:1500,messages:[{role:'user',content:content}]})});
    var data=await resp.json();
    if(data.error) throw new Error(data.error.message);
    var raw=data.content.map(function(b){return b.text||'';}).join('').replace(/```json|```/g,'').trim();
    var result=JSON.parse(raw);
    render(result);
    addHist(result);
  }catch(e){
    showOut('<div style="background:var(--red-pale);border:1px solid #fca5a5;border-radius:var(--radius2);padding:16px;font-size:.74rem;color:var(--red)"><strong>Analysis Error</strong><br><br>'+e.message+'<br><br>Verify your API key at console.anthropic.com and ensure image is PNG, JPG, or WebP.</div>');
  }
  document.getElementById('abtn').disabled=false;
}

function render(result){
  if(mode==='compare') renderCompare(result);
  else if(mode==='cancer') renderCancer(result);
  else renderSingle(result);
}

function renderSingle(r){
  var rankings=(r.rankings||[]).slice().sort(function(a,b){return b.score-a.score;});
  var h='<div class="summary-bar"><strong>'+r.image_type+'</strong> &mdash; '+r.summary+'</div>';
  h+='<div class="res-grid">';
  h+='<div class="res-panel"><div class="res-phead"><span class="res-ptitle">Protein Target Rankings</span><span class="res-psub">n=4,460 trials dataset</span></div><div class="res-pbody">';
  rankings.forEach(function(ri,i){var pd=P[ri.protein]||{color:'#888',trials:0,fda:false};h+=rankRow(i+1,ri.protein,ri.score,pd.color,pd.trials,pd.fda);});
  h+='</div></div>';
  h+='<div class="res-panel"><div class="res-phead"><span class="res-ptitle">Top Match Detail</span><span class="res-psub">AI reasoning</span></div><div class="res-pbody">';
  rankings.slice(0,5).forEach(function(ri){var pd=P[ri.protein]||{color:'#888',trials:0,cancer:'?'};h+=detailCard(ri.protein,ri.score,pd.color,pd.trials+' trials',pd.cancer,ri.reason);});
  h+='</div></div></div>';
  if(r.explanation) h+='<div class="analysis-box"><div class="analysis-title">Structural Analysis</div><div class="analysis-text">'+r.explanation+'</div></div>';
  showOut(h);
}

function renderCancer(r){
  var rankings=(r.rankings||[]).slice().sort(function(a,b){return b.score-a.score;});
  var h='<div class="summary-bar"><strong>'+r.image_type+'</strong> &mdash; '+r.summary+'</div>';
  h+='<div class="res-grid">';
  h+='<div class="res-panel"><div class="res-phead"><span class="res-ptitle">Cancer Type Rankings</span><span class="res-psub">Image relevance score</span></div><div class="res-pbody">';
  rankings.forEach(function(ri,i){var cd=C[ri.cancer]||{color:'#888',trials:0};h+=rankRow(i+1,ri.cancer,ri.score,cd.color,cd.trials,false);});
  h+='</div></div>';
  h+='<div class="res-panel"><div class="res-phead"><span class="res-ptitle">Top Cancer Matches</span><span class="res-psub">AI classification reasoning</span></div><div class="res-pbody">';
  rankings.slice(0,5).forEach(function(ri){var cd=C[ri.cancer]||{color:'#888',trials:0,therapy:'?'};h+=detailCard(ri.cancer,ri.score,cd.color,cd.trials+' trials','Therapy: '+cd.therapy,ri.reason);});
  h+='</div></div></div>';
  if(r.explanation) h+='<div class="analysis-box"><div class="analysis-title">Pathology Analysis</div><div class="analysis-text">'+r.explanation+'</div></div>';
  showOut(h);
}

function renderCompare(r){
  var cmp=(r.comparison||[]).slice().sort(function(a,b){return Math.max(b.score_a,b.score_b)-Math.max(a.score_a,a.score_b);});
  var h='<div class="summary-bar"><span class="sum-a">Image A:</span> '+(r.summary_a||'')+'<br><span class="sum-b">Image B:</span> '+(r.summary_b||'')+'</div>';
  h+='<div class="res-panel"><div class="res-phead"><span class="res-ptitle">Differential Protein Scoring</span><span class="res-psub">Image A vs Image B</span></div><div class="res-pbody">';
  h+='<table class="cmp-table"><thead><tr class="cmp-thead"><td style="width:160px">Target</td><td style="color:#0ea5e9">&#9632; Image A</td><td style="color:#8b5cf6">&#9632; Image B</td></tr></thead><tbody>';
  var diffA=[],diffB=[];
  cmp.forEach(function(ri){
    var pd=P[ri.protein]||{color:'#888'};
    var sA=Math.max(0,Math.min(100,ri.score_a||0)),sB=Math.max(0,Math.min(100,ri.score_b||0));
    var delta=sA-sB;var wh='';
    if(Math.abs(delta)>=8){if(delta>0){wh='<span class="cmp-win wa">A +'+delta+'</span>';diffA.push({protein:ri.protein,delta:delta,color:pd.color});}else{wh='<span class="cmp-win wb">B +'+Math.abs(delta)+'</span>';diffB.push({protein:ri.protein,delta:Math.abs(delta),color:pd.color});}}else{wh='<span class="cmp-win wt">tie</span>';}
    h+='<tr class="cmp-row-tr"><td><div class="cmp-pname"><div class="cmp-colordot" style="background:'+pd.color+'"></div>'+ri.protein+'</div></td>';
    h+='<td><div class="cmp-cell"><div class="cmp-track"><div class="cmp-fa" data-w="'+sA+'"></div></div><div class="cmp-pct" style="color:#0ea5e9">'+sA+'%</div></div></td>';
    h+='<td><div class="cmp-cell">'+wh+'<div class="cmp-track"><div class="cmp-fb" data-w="'+sB+'"></div></div><div class="cmp-pct" style="color:#8b5cf6">'+sB+'%</div></div></td>';
    h+='</tr>';
  });
  h+='</tbody></table></div></div>';
  h+='<div class="diff-section"><div class="diff-title">Differential Summary (&Delta; &ge; 8pp)</div>';
  if(diffA.length){h+='<div class="diff-sub" style="color:#0ea5e9">Image A matches better:</div>';diffA.sort(function(a,b){return b.delta-a.delta;}).forEach(function(d){h+='<div class="diff-rr"><span class="diff-name" style="color:'+d.color+'">'+d.protein+'</span><span class="diff-delta" style="color:#0ea5e9">A advantage: +'+d.delta+'pp</span></div>';});}
  if(diffB.length){h+='<div class="diff-sub" style="color:#8b5cf6">Image B matches better:</div>';diffB.sort(function(a,b){return b.delta-a.delta;}).forEach(function(d){h+='<div class="diff-rr"><span class="diff-name" style="color:'+d.color+'">'+d.protein+'</span><span class="diff-delta" style="color:#8b5cf6">B advantage: +'+d.delta+'pp</span></div>';});}
  if(!diffA.length&&!diffB.length) h+='<div style="font-size:.72rem;color:var(--slate2)">Both specimens show equivalent protein target profiles across all markers.</div>';
  h+='</div>';
  if(r.explanation_a||r.explanation_b){h+='<div class="analysis-grid">';if(r.explanation_a)h+='<div class="analysis-box"><div class="analysis-title" style="color:#0ea5e9">Image A Analysis</div><div class="analysis-text">'+r.explanation_a+'</div></div>';if(r.explanation_b)h+='<div class="analysis-box"><div class="analysis-title" style="color:#8b5cf6">Image B Analysis</div><div class="analysis-text">'+r.explanation_b+'</div></div>';h+='</div>';}
  showOut(h);
}

function addHist(result){
  hist.unshift({imgA:imgA?imgA.dataUrl:null,imgB:(mode==='compare'&&imgB)?imgB.dataUrl:null,result:result,mode:mode});
  if(hist.length>8) hist.pop();
  var strip=document.getElementById('hstrip');var inner=document.getElementById('hinner');
  strip.style.display='flex';inner.innerHTML='';
  hist.forEach(function(h){
    var pair=document.createElement('div');pair.className='hist-pair';
    if(h.imgA){var ia=document.createElement('img');ia.className='hist-img';ia.src=h.imgA;pair.appendChild(ia);}
    if(h.imgB){var vs=document.createElement('div');vs.className='hist-vs';vs.textContent='vs';pair.appendChild(vs);var ib=document.createElement('img');ib.className='hist-img';ib.src=h.imgB;pair.appendChild(ib);}
    pair.onclick=(function(r,m){return function(){mode=m;render(r);};})(h.result,h.mode);
    inner.appendChild(pair);
  });
}

setMode('single');
</script>
</body>
</html>"""

PAGE = PAGE.replace("__PROTEINS__", P_JS).replace("__CANCERS__", C_JS).replace("__DSTOTAL__", DT_JS)

with open("outputs/image_compare.html","w",encoding="utf-8") as f:
    f.write(PAGE)

sz = Path("outputs/image_compare.html").stat().st_size / 1024
print(f"  outputs/image_compare.html  ({sz:.0f} KB)")
print(f"  Serve via: cd ~/Desktop/git2/outputs && python3 -m http.server 8080")
print(f"  Then open: http://localhost:8080/image_compare.html")
