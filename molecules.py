"""
molecules.py — 3D Protein Target Viewer
Zero template literals. Safari-safe. Sidebar loads before Three.js.
"""
import json, math, requests, time
from pathlib import Path

Path("outputs").mkdir(exist_ok=True)

PROTEINS = {
    "CD19":{"pdb":"6AL6","full_name":"CD19 B-Lymphocyte Antigen","gene":"CD19","cancer_types":["Lymphoma","Leukemia (ALL)","CLL"],"therapy_types":["CAR-T","BiTE"],"color":"#00d4aa","approved":True,"fda_drugs":["Tisagenlecleucel (Kymriah)","Axicabtagene (Yescarta)"],"description":"CD19 is the most clinically validated CAR-T target, expressed on over 95% of B-cell malignancies. The first two FDA-approved CAR-T therapies (2017) both target CD19.","mechanism":"CAR-T scFv binds CD19, triggering CD3-zeta and 4-1BB signaling, T cell activation, and cytotoxic killing.","binding_site":"Extracellular immunoglobulin-like domain","mw_kda":95,"trials_approx":592},
    "BCMA":{"pdb":"1XU0","full_name":"B-Cell Maturation Antigen (TNFRSF17)","gene":"TNFRSF17","cancer_types":["Multiple Myeloma"],"therapy_types":["CAR-T","ADC","BiTE"],"color":"#8b5cf6","approved":True,"fda_drugs":["Idecabtagene vicleucel (Abecma)","Ciltacabtagene autoleucel (Carvykti)"],"description":"BCMA is a TNF receptor family member overexpressed on plasma cells and myeloma cells. Two FDA-approved CAR-T products target BCMA for relapsed/refractory multiple myeloma.","mechanism":"BCMA mediates plasma cell survival via APRIL and BAFF; CAR-T binding triggers ADCC and direct cytolysis.","binding_site":"Cysteine-rich extracellular TNF receptor domain","mw_kda":20,"trials_approx":229},
    "HER2":{"pdb":"3PP0","full_name":"HER2 Human Epidermal Growth Factor Receptor 2","gene":"ERBB2","cancer_types":["Breast Cancer","Gastric Cancer","Ovarian Cancer"],"therapy_types":["CAR-T","ADC","mAb"],"color":"#ec4899","approved":True,"fda_drugs":["Trastuzumab (Herceptin)","Pertuzumab","T-DM1 (Kadcyla)"],"description":"HER2 is a receptor tyrosine kinase amplified in about 20% of breast cancers, driving constitutive RAS/MAPK and PI3K/AKT signaling.","mechanism":"Trastuzumab blocks domain IV, preventing dimerization; ADCs deliver cytotoxic payload directly.","binding_site":"Extracellular domains II and IV; intracellular kinase domain (ATP pocket)","mw_kda":138,"trials_approx":259},
    "EGFR":{"pdb":"2GS7","full_name":"Epidermal Growth Factor Receptor","gene":"EGFR","cancer_types":["Lung Cancer","Brain/CNS","Head and Neck"],"therapy_types":["CAR-T","TKI","mAb"],"color":"#f59e0b","approved":True,"fda_drugs":["Erlotinib","Osimertinib (Tagrisso)","Cetuximab (Erbitux)"],"description":"EGFR is mutated in about 15% of Western NSCLC and 50% of Asian NSCLC. The EGFRvIII deletion mutant is tumor-specific in GBM and an active CAR-T target.","mechanism":"Ligand binding triggers dimerization, kinase activation, and RAS/MAPK and PI3K/AKT signaling.","binding_site":"Ligand-binding extracellular domain; ATP-binding kinase domain (L858R, T790M hotspots)","mw_kda":134,"trials_approx":353},
    "PD-1":{"pdb":"5IUS","full_name":"Programmed Cell Death Protein 1","gene":"PDCD1","cancer_types":["Melanoma","Lung Cancer","Lymphoma"],"therapy_types":["Checkpoint Blockade","CRISPR KO"],"color":"#22c55e","approved":True,"fda_drugs":["Pembrolizumab (Keytruda)","Nivolumab (Opdivo)"],"description":"PD-1 is the primary immune checkpoint receptor on T cells. CRISPR knockout of PD-1 in CAR-T cells dramatically enhances anti-tumor potency.","mechanism":"PD-1:PD-L1 binding recruits SHP2, dephosphorylates ZAP70, and drives T cell exhaustion.","binding_site":"IgV-like extracellular domain; FG and BC loops at PD-L1 interface","mw_kda":32,"trials_approx":180},
    "KRAS":{"pdb":"4OBE","full_name":"KRAS Proto-Oncogene GTPase","gene":"KRAS","cancer_types":["Pancreatic Cancer","Colorectal Cancer","Lung Cancer"],"therapy_types":["mRNA Therapy","Gene Therapy","CRISPR"],"color":"#ef4444","approved":True,"fda_drugs":["Sotorasib (Lumakras) for KRAS G12C only"],"description":"KRAS is mutated in over 90% of pancreatic, 40% of colorectal, and 25% of lung cancers. Historically undruggable until G12C-specific covalent inhibitors were developed.","mechanism":"Mutant KRAS is constitutively GTP-bound, permanently activating RAF/MEK/ERK signaling.","binding_site":"GTP-binding pocket; G12 switch-I region (G12C creates unique druggable cysteine)","mw_kda":21,"trials_approx":105},
    "CD33":{"pdb":"5UCM","full_name":"CD33 Myeloid Cell Surface Antigen","gene":"CD33","cancer_types":["Leukemia (AML)"],"therapy_types":["CAR-T","ADC","BiTE"],"color":"#4a9eff","approved":True,"fda_drugs":["Gemtuzumab ozogamicin (Mylotarg)"],"description":"CD33 is a sialic acid-binding receptor on myeloid cells and AML blasts. Key challenge: CD33 is also expressed on normal myeloid progenitors.","mechanism":"ITIM-mediated inhibitory signaling; therapeutic exploitation via ADC toxin delivery or CAR-T cytolysis.","binding_site":"N-terminal V-set immunoglobulin-like lectin domain","mw_kda":67,"trials_approx":364},
    "Mesothelin":{"pdb":"3UAK","full_name":"Mesothelin GPI-anchored Glycoprotein","gene":"MSLN","cancer_types":["Mesothelioma","Ovarian Cancer","Lung Cancer"],"therapy_types":["CAR-T","ADC","mAb"],"color":"#14b8a6","approved":False,"fda_drugs":[],"description":"Mesothelin is a GPI-anchored glycoprotein overexpressed in mesothelioma, ovarian, and pancreatic cancers with minimal expression on normal cells.","mechanism":"Mesothelin-MUC16 interaction promotes tumor cell adhesion; CAR-T targeting triggers direct cytotoxic response.","binding_site":"N-terminal functional region (residues 296-359) involved in MUC16 binding","mw_kda":69,"trials_approx":62},
    "TP53":{"pdb":"2OCJ","full_name":"Tumor Protein p53 Guardian of the Genome","gene":"TP53","cancer_types":["Multiple Cancers","Ovarian","Colorectal"],"therapy_types":["Gene Therapy","mRNA Therapy","CRISPR Base Editing"],"color":"#f97316","approved":False,"fda_drugs":[],"description":"TP53 is mutated in about 50% of all human cancers, the most frequently altered gene in oncology. Restoration via base editing is an active research direction.","mechanism":"p53 tetramer binds response elements, activating CDKN1A/BAX/PUMA expression for growth arrest or apoptosis.","binding_site":"DNA-binding domain (R175, G245, R248, R249, R273, R282 hotspot mutations); tetramerization domain","mw_kda":44,"trials_approx":95},
    "GPC3":{"pdb":"5XQ2","full_name":"Glypican-3 Liver-Specific Proteoglycan","gene":"GPC3","cancer_types":["Liver Cancer (HCC)","Ovarian Cancer"],"therapy_types":["CAR-T","BiTE"],"color":"#84cc16","approved":False,"fda_drugs":[],"description":"GPC3 is a heparan sulfate proteoglycan overexpressed in hepatocellular carcinoma but absent in normal adult liver, providing high specificity for HCC targeting.","mechanism":"GPC3 promotes Wnt and Hedgehog signaling in HCC; surface expression enables specific CAR-T targeting.","binding_site":"Core protein extracellular domain; heparan sulfate attachment sites at Ser495/Ser509","mw_kda":66,"trials_approx":48},
}

def fetch_pdb(pdb_id):
    for url in ["https://files.rcsb.org/download/"+pdb_id.upper()+".pdb",
                "https://files.rcsb.org/view/"+pdb_id.upper()+".pdb"]:
        try:
            r = requests.get(url, timeout=30)
            if r.ok and "ATOM" in r.text: return r.text
        except: pass
    return None

def parse_ca(pdb_text, max_chains=3):
    chains = {}
    for line in pdb_text.split("\n"):
        if not line.startswith("ATOM"): continue
        if line[12:16].strip() != "CA": continue
        try:
            ch = line[21:22].strip() or "A"
            rs = int(line[22:26])
            x,y,z = float(line[30:38]),float(line[38:46]),float(line[46:54])
        except: continue
        if ch not in chains:
            if len(chains) >= max_chains: continue
            chains[ch] = []
        chains[ch].append([round(x,2),round(y,2),round(z,2),rs])
    for ch in chains:
        chains[ch].sort(key=lambda a:a[3])
        chains[ch] = [[a[0],a[1],a[2]] for a in chains[ch]]
    return chains

def make_helix(n=80):
    pts = []
    for i in range(n):
        t = i*0.4
        pts.append([round(math.cos(t)*8,2), round(t*2-30,2), round(math.sin(t)*8,2)])
    return {"A": pts}

print("="*55)
print("Downloading PDB structures from RCSB PDB...")
print("="*55)

for name, info in PROTEINS.items():
    pid = info["pdb"]
    print(f"  [{pid}] {name}...", end=" ", flush=True)
    text = fetch_pdb(pid)
    if text:
        chains = parse_ca(text)
        n = sum(len(v) for v in chains.values())
        PROTEINS[name]["coords"] = chains
        print(f"OK  {n} Ca atoms")
    else:
        PROTEINS[name]["coords"] = make_helix()
        print("fallback helix")
    time.sleep(0.3)

try:
    with open("outputs/molecular_insights.json") as f:
        mol_insights = json.load(f)
    print(f"\n  Loaded molecular_insights.json ({len(mol_insights.get('proteins',{}))} proteins)")
except:
    mol_insights = {}
    print("\n  No molecular_insights.json found")

for name in PROTEINS:
    PROTEINS[name]["simulation"] = mol_insights.get("proteins", {}).get(name)

P_JSON = json.dumps(PROTEINS, separators=(',',':'), ensure_ascii=True)
MI_JSON = json.dumps(mol_insights, separators=(',',':'), ensure_ascii=True)
P_JSON = P_JSON.replace("</", "<\\/")
MI_JSON = MI_JSON.replace("</", "<\\/")


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>3D Protein Target Viewer</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Segoe UI',system-ui,sans-serif;background:#060d1a;color:#c8dff5;overflow-x:hidden}
.hdr{background:linear-gradient(135deg,#060d1a,#0a1f3c);padding:18px 30px;border-bottom:1px solid #1a3356;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px}
.hdr h1{font-size:1.15rem;font-weight:700;color:#fff}.hdr h1 span{color:#00d4aa}
.hdr-sub{font-size:.68rem;color:#7a9bbf;font-family:monospace;margin-top:2px}
.badges{display:flex;gap:6px;flex-wrap:wrap;align-items:center}
.badge{background:rgba(0,212,170,.1);border:1px solid rgba(0,212,170,.3);color:#00d4aa;padding:2px 9px;border-radius:20px;font-size:.66rem;font-family:monospace}
.badge.pu{background:rgba(139,92,246,.1);border-color:rgba(139,92,246,.3);color:#a78bfa}
.back{color:#4a9eff;font-size:.7rem;text-decoration:none;border:1px solid #1a3356;border-radius:5px;padding:4px 10px}
.layout{display:grid;grid-template-columns:250px 1fr;min-height:calc(100vh - 62px)}
.sidebar{background:#0a1628;border-right:1px solid #1a3356;padding:12px;overflow-y:auto;height:calc(100vh - 62px);position:sticky;top:62px}
.ss{font-size:.62rem;font-weight:700;text-transform:uppercase;letter-spacing:.1em;color:#7a9bbf;margin:8px 0 7px}
.tc{background:#0d1f3c;border:1px solid #1a3356;border-radius:7px;padding:9px 11px;margin-bottom:6px;cursor:pointer;transition:all .18s;border-left:3px solid transparent}
.tc:hover,.tc.on{border-left-color:var(--lc,#00d4aa);background:#102040}
.tc-n{font-size:.8rem;font-weight:700;color:#fff;font-family:monospace}
.tc-g{font-size:.62rem;color:#7a9bbf}
.tc-ct{font-size:.62rem;color:#8ab4d4;margin-top:2px}
.ap{background:rgba(34,197,94,.1);border:1px solid rgba(34,197,94,.2);color:#22c55e;font-size:.56rem;padding:1px 5px;border-radius:8px;display:inline-block;margin-top:3px}
.tb{height:2px;background:#1a3356;border-radius:1px;margin-top:5px;overflow:hidden}
.tb-f{height:100%;background:var(--lc,#00d4aa);border-radius:1px}
.cmp-box{padding:10px;background:#060d1a;border:1px solid #1a3356;border-radius:7px;margin-bottom:10px}
.cmp-box label{font-size:.62rem;color:#7a9bbf;text-transform:uppercase;letter-spacing:.08em;display:block;margin-bottom:3px}
.cmp-box select{width:100%;background:#0d1f3c;color:#c8dff5;border:1px solid #1a3356;border-radius:4px;padding:5px 7px;font-size:.7rem;outline:none;margin-bottom:5px}
.cmp-btn{width:100%;background:linear-gradient(135deg,rgba(139,92,246,.2),rgba(0,212,170,.1));border:1px solid rgba(139,92,246,.35);color:#a78bfa;padding:6px;border-radius:5px;font-size:.7rem;cursor:pointer;font-weight:700}
.pdb-box{padding:10px;background:#060d1a;border:1px solid #1a3356;border-radius:7px;margin-top:8px}
.pdb-row{display:flex;gap:5px}
.pi{background:#0d1f3c;color:#c8dff5;border:1px solid #1a3356;border-radius:4px;padding:4px 7px;font-size:.74rem;font-family:monospace;width:75px;outline:none}
.pb{background:rgba(0,212,170,.1);border:1px solid rgba(0,212,170,.3);color:#00d4aa;padding:4px 8px;border-radius:4px;font-size:.68rem;cursor:pointer}
.main{display:flex;flex-direction:column}
.vwraps{display:grid;grid-template-columns:1fr;height:440px;border-bottom:1px solid #1a3356}
.vwraps.split{grid-template-columns:1fr 1fr}
.vw{position:relative;overflow:hidden;background:#060d1a;height:440px}
.vw canvas{position:absolute;top:0;left:0}
.vlbl{position:absolute;top:10px;left:12px;background:rgba(6,13,26,.85);border:1px solid #1a3356;border-radius:5px;padding:3px 9px;font-size:.72rem;font-weight:700;color:#fff;z-index:10;pointer-events:none}
.vbtns{position:absolute;bottom:10px;right:10px;display:flex;gap:4px;z-index:10}
.vb{background:rgba(13,31,60,.92);border:1px solid #1a3356;color:#c8dff5;padding:4px 9px;border-radius:5px;font-size:.66rem;cursor:pointer}
.vb:hover,.vb.on{border-color:#00d4aa;color:#00d4aa;background:rgba(0,212,170,.1)}
.vcnt{position:absolute;top:10px;right:10px;background:rgba(6,13,26,.8);border:1px solid #1a3356;border-radius:4px;padding:3px 8px;font-size:.62rem;color:#7a9bbf;font-family:monospace;z-index:10}
.info{padding:20px 24px;overflow-y:auto}
.ig{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.pc{background:#0d1f3c;border:1px solid #1a3356;border-radius:9px;padding:15px}
.pc h2{font-size:.88rem;font-weight:700;color:#fff;margin-bottom:2px}
.gtag{display:inline-block;background:rgba(74,158,255,.08);border:1px solid rgba(74,158,255,.2);color:#4a9eff;font-size:.62rem;padding:1px 6px;border-radius:8px;font-family:monospace;margin-bottom:8px}
.pc p{font-size:.74rem;color:#c8dff5;line-height:1.6;margin-bottom:10px}
.ir{display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid rgba(26,51,86,.5);font-size:.69rem;gap:10px}
.il{color:#7a9bbf;text-transform:uppercase;letter-spacing:.05em;flex-shrink:0}
.iv{color:#c8dff5;font-family:monospace;font-weight:600;text-align:right}
.dtags{display:flex;flex-wrap:wrap;gap:4px;margin-top:6px}
.dt{background:rgba(34,197,94,.07);border:1px solid rgba(34,197,94,.18);color:#22c55e;font-size:.6rem;padding:2px 6px;border-radius:8px}
.mb{background:rgba(0,212,170,.04);border:1px solid rgba(0,212,170,.12);border-radius:6px;padding:11px 13px;margin-top:9px}
.mb h4{font-size:.62rem;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:#00d4aa;margin-bottom:4px}
.mb p{font-size:.72rem;color:#c8dff5;line-height:1.55}
.mb .bs{font-size:.67rem;color:#7a9bbf;margin-top:5px}
.lgnd{display:flex;flex-wrap:wrap;gap:7px;margin-top:12px}
.li{display:flex;align-items:center;gap:5px;font-size:.66rem;color:#8ab4d4}
.ld{width:8px;height:8px;border-radius:50%}
.shd{font-size:.7rem;font-weight:700;text-transform:uppercase;letter-spacing:.09em;color:#7a9bbf;margin-bottom:10px}
.ph{color:#7a9bbf;font-size:.78rem;padding:14px 0;line-height:1.7}
</style>
</head>
<body>
<div class="hdr">
  <div>
    <h1>3D Protein Target Viewer &mdash; <span>Gene-Editing Cancer Trials</span></h1>
    <div class="hdr-sub">Crystal structures from RCSB PDB &middot; Three.js TubeGeometry &middot; Ca backbone pre-embedded &middot; Offline</div>
  </div>
  <div class="badges">
    <span class="badge">Lakshay Mani</span>
    <span class="badge">MS Analytics &middot; Northeastern</span>
    <span class="badge pu">3D Molecular</span>
    <a href="dashboard.html" class="back">&larr; Dashboard</a>
  </div>
</div>
<div class="layout">
<div class="sidebar">
  <div class="cmp-box">
    <label>Compare Side-by-Side</label>
    <select id="ca"></select>
    <select id="cb"></select>
    <button class="cmp-btn" onclick="startCmp()">Compare Structures</button>
  </div>
  <div class="ss">Protein Targets</div>
  <div id="tlist"></div>
  <div class="pdb-box">
    <div style="font-size:.62rem;color:#7a9bbf;text-transform:uppercase;letter-spacing:.08em;margin-bottom:5px">Load any PDB ID</div>
    <div class="pdb-row">
      <input class="pi" id="cpdb" placeholder="4ABC" maxlength="5">
      <button class="pb" onclick="loadCustom()">Load</button>
    </div>
    <div style="font-size:.6rem;color:#4a9eff;margin-top:4px">Fetches live - needs internet</div>
  </div>
</div>
<div class="main">
  <div class="vwraps" id="vwraps">
    <div class="vw" id="vw-a">
      <div class="vlbl" id="lbl-a">Select a target</div>
      <div class="vbtns" id="vb-a">
        <button class="vb on" onclick="setMode('a','ribbon')">Ribbon</button>
        <button class="vb" onclick="setMode('a','spheres')">Spheres</button>
        <button class="vb" onclick="setMode('a','cpk')">CPK</button>
        <button class="vb" onclick="setMode('a','secstr')">SecStr</button>
        <button class="vb" id="spin-a" onclick="toggleSpin('a')">Spin</button>
      </div>
      <div class="vcnt" id="cnt-a"></div>
    </div>
    <div class="vw" id="vw-b" style="display:none">
      <div class="vlbl" id="lbl-b">-</div>
      <div class="vbtns" id="vb-b">
        <button class="vb on" onclick="setMode('b','ribbon')">Ribbon</button>
        <button class="vb" onclick="setMode('b','spheres')">Spheres</button>
        <button class="vb" onclick="setMode('b','cpk')">CPK</button>
        <button class="vb" onclick="setMode('b','secstr')">SecStr</button>
        <button class="vb" id="spin-b" onclick="toggleSpin('b')">Spin</button>
      </div>
      <div class="vcnt" id="cnt-b"></div>
    </div>
  </div>
  <div class="info" id="info">
    <div class="shd">Protein Details</div>
    <div id="ic"><p class="ph">Click any protein in the sidebar to load its 3D structure.<br><br>Drag to rotate, scroll to zoom.<br>Ribbon / Spheres / CPK / SecStr rendering modes.<br>SecStr: Red=Alpha helix, Yellow=Beta sheet, Gray=Loop</p></div>
    <div style="margin-top:14px"><div class="shd">Color Legend</div>
    <div class="lgnd">
      <div class="li"><div class="ld" style="background:#4a9eff"></div>N-terminus</div>
      <div class="li"><div class="ld" style="background:#00d4aa"></div>Mid-chain</div>
      <div class="li"><div class="ld" style="background:#ef4444"></div>C-terminus / Helix</div>
      <div class="li"><div class="ld" style="background:#f59e0b"></div>Beta Sheet</div>
      <div class="li"><div class="ld" style="background:#888"></div>Loop (SecStr)</div>
    </div></div>
  </div>
</div>
</div>

<script>
var P = __PJSON__;
var MI = __MIJSON__;
var THREE_OK = false;
var S = {
  a:{scene:null,cam:null,rend:null,drag:false,px:0,py:0,rx:0,ry:0,spin:true,mode:'ribbon',key:null},
  b:{scene:null,cam:null,rend:null,drag:false,px:0,py:0,rx:0,ry:0,spin:true,mode:'ribbon',key:null}
};

function buildSidebar(){
  var tlist=document.getElementById('tlist');
  var ca=document.getElementById('ca');
  var cb=document.getElementById('cb');
  tlist.innerHTML='';
  var keys=Object.keys(P);
  keys.forEach(function(key){
    var p=P[key];
    ca.add(new Option(key,key));
    cb.add(new Option(key,key));
    var card=document.createElement('div');
    card.className='tc';
    card.id='tc-'+key;
    card.style.setProperty('--lc',p.color);
    var h='';
    h+='<div class="tc-n">'+key+'</div>';
    h+='<div class="tc-g">Gene: '+p.gene+'</div>';
    h+='<div class="tc-ct">'+p.cancer_types.join(', ')+'</div>';
    if(p.approved) h+='<span class="ap">FDA Approved</span>';
    h+='<div class="tb"><div class="tb-f" style="width:'+Math.round(p.trials_approx/592*100)+'%"></div></div>';
    card.innerHTML=h;
    card.onclick=(function(k){return function(){loadTarget(k,'a',false);};})(key);
    tlist.appendChild(card);
  });
  if(cb.options.length>1) cb.selectedIndex=1;
}

buildSidebar();

function rainbow(t){var r=t<0.5?Math.round(4*t*255):255;var g=t<0.25?Math.round(4*t*255):t<0.75?255:Math.round((1-(t-0.75)*4)*255);var b=t<0.25?255:t<0.5?Math.round((1-(t-0.25)*4)*255):0;return (r<<16)|(g<<8)|b;}
var CPK={C:0x888888,N:0x4a9eff,O:0xef4444,S:0xf59e0b,H:0xcccccc};
function cpkCol(e){return CPK[e.toUpperCase()]||0x888888;}
function ssColor(t){if(t==='H') return new THREE.Color(0xef4444);if(t==='E') return new THREE.Color(0xf59e0b);return new THREE.Color(0x6b7280);}

function initSlot(slot){
  if(S[slot].rend||!THREE_OK) return;
  var wrap=document.getElementById('vw-'+slot);
  var W=wrap.clientWidth||800,H=wrap.clientHeight||440;
  try{
    var rend=new THREE.WebGLRenderer({antialias:true,alpha:false});
    rend.setSize(W,H); rend.setPixelRatio(Math.min(devicePixelRatio,2)); rend.setClearColor(0x060d1a,1);
    wrap.appendChild(rend.domElement);
    var scene=new THREE.Scene();
    var cam=new THREE.PerspectiveCamera(45,W/H,0.1,2000); cam.position.z=100;
    scene.add(new THREE.AmbientLight(0xffffff,0.55));
    var d1=new THREE.DirectionalLight(0xffffff,0.75); d1.position.set(1,2,3); scene.add(d1);
    var d2=new THREE.DirectionalLight(0x4a9eff,0.25); d2.position.set(-2,-1,-2); scene.add(d2);
    S[slot].scene=scene; S[slot].cam=cam; S[slot].rend=rend;
    var el=rend.domElement;
    el.addEventListener('mousedown',function(e){S[slot].drag=true;S[slot].px=e.clientX;S[slot].py=e.clientY;S[slot].spin=false;});
    window.addEventListener('mouseup',function(){S[slot].drag=false;});
    el.addEventListener('mousemove',function(e){
      if(!S[slot].drag) return;
      S[slot].ry+=(e.clientX-S[slot].px)*0.006; S[slot].rx+=(e.clientY-S[slot].py)*0.006;
      S[slot].px=e.clientX; S[slot].py=e.clientY;
      var g=scene.getObjectByName('mol'); if(g){g.rotation.x=S[slot].rx;g.rotation.y=S[slot].ry;}
    });
    el.addEventListener('wheel',function(e){cam.position.z=Math.max(15,Math.min(500,cam.position.z+e.deltaY*.12));},{passive:true});
    (function loop(){requestAnimationFrame(loop);if(S[slot].spin){var g=scene.getObjectByName('mol');if(g){g.rotation.y+=0.006;S[slot].ry=g.rotation.y;}}rend.render(scene,cam);})();
  }catch(e){console.error('WebGL init failed:',e);}
}

function clearSlot(slot){var s=S[slot].scene;if(!s)return;var old=s.getObjectByName('mol');if(old){old.traverse(function(c){if(c.geometry)c.geometry.dispose();if(c.material){if(Array.isArray(c.material))c.material.forEach(function(m){m.dispose();});else c.material.dispose();}});s.remove(old);}}

function buildMol(key,slot){
  if(!THREE_OK||!S[slot].rend) return;
  clearSlot(slot);
  var scene=S[slot].scene;
  var protein=P[key]; if(!protein||!protein.coords) return;
  var chains=protein.coords; var ckeys=Object.keys(chains);
  var cx=0,cy=0,cz=0,n=0;
  ckeys.forEach(function(ch){chains[ch].forEach(function(a){cx+=a[0];cy+=a[1];cz+=a[2];n++;});});
  if(!n) return; cx/=n; cy/=n; cz/=n;
  var group=new THREE.Group(); group.name='mol';
  var mode=S[slot].mode; var totalAtoms=0;
  ckeys.forEach(function(ch){
    var raw=chains[ch]; if(raw.length<3) return;
    totalAtoms+=raw.length;
    var pts=raw.map(function(a){return new THREE.Vector3(a[0]-cx,a[1]-cy,a[2]-cz);});
    if(mode==='ribbon'){
      try{
        var curve=new THREE.CatmullRomCurve3(pts);
        var geo=new THREE.TubeGeometry(curve,Math.min(pts.length*4,800),0.45,7,false);
        var pos=geo.attributes.position; var colors=new Float32Array(pos.count*3);
        for(var i=0;i<pos.count;i++){var t=i/pos.count;var c=new THREE.Color(rainbow(t));colors[i*3]=c.r;colors[i*3+1]=c.g;colors[i*3+2]=c.b;}
        geo.setAttribute('color',new THREE.BufferAttribute(colors,3));
        group.add(new THREE.Mesh(geo,new THREE.MeshPhongMaterial({vertexColors:true,shininess:55})));
        var step=Math.max(1,Math.floor(pts.length/25));
        pts.forEach(function(p,i){if(i%step!==0)return;var t=i/pts.length;var sg=new THREE.SphereGeometry(0.75,8,6);var sm=new THREE.MeshPhongMaterial({color:new THREE.Color(rainbow(t)),shininess:70});var s=new THREE.Mesh(sg,sm);s.position.copy(p);group.add(s);});
      }catch(e){}
    }
    if(mode==='spheres'){pts.forEach(function(p,i){var t=i/pts.length;var sg=new THREE.SphereGeometry(1.2,10,8);var sm=new THREE.MeshPhongMaterial({color:new THREE.Color(rainbow(t)),shininess:60});var s=new THREE.Mesh(sg,sm);s.position.copy(p);group.add(s);});}
    if(mode==='cpk'){var ELEM=['C','N','O','C','C','N','O'];pts.forEach(function(p,i){var el=ELEM[i%ELEM.length];var sg=new THREE.SphereGeometry(0.85,8,6);var sm=new THREE.MeshPhongMaterial({color:new THREE.Color(cpkCol(el)),shininess:80});var s=new THREE.Mesh(sg,sm);s.position.copy(p);group.add(s);});}
    if(mode==='secstr'){
      var sim=MI&&MI.proteins&&MI.proteins[key]; var ssSeq=sim?(sim.secondary_structure.sequence||[]):[];
      pts.forEach(function(p,i){var st=ssSeq[i]||'C';var col=ssColor(st);var r=st==='H'?1.4:st==='E'?1.2:0.9;var sg=new THREE.SphereGeometry(r,8,6);var sm=new THREE.MeshPhongMaterial({color:col,shininess:60});var s=new THREE.Mesh(sg,sm);s.position.copy(p);group.add(s);});
      try{var cur2=new THREE.CatmullRomCurve3(pts);var geo2=new THREE.TubeGeometry(cur2,Math.min(pts.length*3,600),0.18,5,false);group.add(new THREE.Mesh(geo2,new THREE.MeshBasicMaterial({color:0x1a3356})));}catch(e){}
    }
  });
  group.rotation.x=S[slot].rx; group.rotation.y=S[slot].ry; scene.add(group);
  var box=new THREE.Box3().setFromObject(group); var size=box.getSize(new THREE.Vector3()).length();
  S[slot].cam.position.z=size*1.25;
  document.getElementById('cnt-'+slot).textContent=totalAtoms+' Ca';
}

function renderInfo(kA,kB){
  function card(key){
    var p=P[key];
    var h='<div class="pc">';
    h+='<h2>'+p.full_name+'</h2>';
    h+='<span class="gtag">Gene: '+p.gene+'</span>';
    h+='<p>'+p.description+'</p>';
    h+='<div class="ir"><span class="il">Cancer Types</span><span class="iv">'+p.cancer_types.join(', ')+'</span></div>';
    h+='<div class="ir"><span class="il">Therapy</span><span class="iv">'+p.therapy_types.join(', ')+'</span></div>';
    h+='<div class="ir"><span class="il">MW</span><span class="iv">~'+p.mw_kda+' kDa</span></div>';
    h+='<div class="ir"><span class="il">PDB</span><span class="iv">'+p.pdb+'</span></div>';
    h+='<div class="ir"><span class="il">Trials</span><span class="iv">~'+p.trials_approx+'</span></div>';
    h+='<div class="ir"><span class="il">FDA</span><span class="iv" style="color:'+(p.approved?'#22c55e':'#f59e0b')+'">'+(p.approved?'Yes':'In Development')+'</span></div>';
    if(p.fda_drugs&&p.fda_drugs.length){h+='<div class="dtags">';p.fda_drugs.forEach(function(d){h+='<span class="dt">'+d+'</span>';});h+='</div>';}
    var sim=MI&&MI.proteins&&MI.proteins[key];
    if(sim){var dr=sim.druggability,ss=sim.secondary_structure;var col=dr.score>=65?'#22c55e':dr.score>=40?'#f59e0b':'#ef4444';h+='<div style="margin-top:8px;padding:8px;background:#060d1a;border:1px solid #1a3356;border-radius:6px">';h+='<div style="font-size:.62rem;color:#7a9bbf;margin-bottom:4px">DRUGGABILITY: '+dr.score+'/100</div>';h+='<div style="background:#1a3356;border-radius:3px;height:5px"><div style="width:'+dr.score+'%;background:'+col+';height:5px;border-radius:3px"></div></div>';h+='<div style="font-size:.61rem;color:#8ab4d4;margin-top:3px">Helix: '+ss.helix_pct+'%   Sheet: '+ss.strand_pct+'%   Loop: '+ss.coil_pct+'%</div></div>';}
    h+='<div class="mb"><h4>Mechanism</h4><p>'+p.mechanism+'</p><p class="bs">Binding site: '+p.binding_site+'</p></div>';
    h+='</div>';
    return h;
  }
  var hint='<div style="padding:14px;color:#7a9bbf;font-size:.76rem">Use Compare Structures to view two proteins side-by-side.</div>';
  document.getElementById('ic').innerHTML=kB?('<div class="ig">'+card(kA)+card(kB)+'</div>'):('<div class="ig">'+card(kA)+hint+'</div>');
}

function loadTarget(key,slot,isCmp){
  if(!isCmp){
    document.getElementById('vw-b').style.display='none';
    document.getElementById('vwraps').classList.remove('split');
    document.querySelectorAll('.tc').forEach(function(c){c.classList.remove('on');});
    var tc=document.getElementById('tc-'+key);if(tc)tc.classList.add('on');
  }
  S[slot].key=key;
  var p=P[key];
  var lbl=document.getElementById('lbl-'+slot);
  lbl.style.setProperty('--lc',p.color);
  lbl.innerHTML='<span style="color:'+p.color+'">'+key+'</span> &middot; PDB '+p.pdb;
  if(THREE_OK){if(!S[slot].rend)initSlot(slot);buildMol(key,slot);}
  if(!isCmp)renderInfo(key,null);
}

function startCmp(){
  var a=document.getElementById('ca').value,b=document.getElementById('cb').value;
  if(a===b){alert('Select two different proteins.');return;}
  document.getElementById('vw-b').style.display='block';
  document.getElementById('vwraps').classList.add('split');
  setTimeout(function(){
    resizeSlot('a');
    if(!S['b'].rend)initSlot('b');else resizeSlot('b');
    var pa=P[a],pb=P[b];
    document.getElementById('lbl-a').innerHTML='<span style="color:'+pa.color+'">'+a+'</span> &middot; '+pa.pdb;
    document.getElementById('lbl-b').innerHTML='<span style="color:'+pb.color+'">'+b+'</span> &middot; '+pb.pdb;
    S['a'].key=a;S['b'].key=b;buildMol(a,'a');buildMol(b,'b');renderInfo(a,b);
  },80);
}

function setMode(slot,mode){
  S[slot].mode=mode;
  document.querySelectorAll('#vb-'+slot+' .vb').forEach(function(b){b.classList.toggle('on',b.textContent.trim()===mode||(mode==='ribbon'&&b.textContent==='Ribbon'));});
  if(S[slot].key)buildMol(S[slot].key,slot);
}

function toggleSpin(slot){S[slot].spin=!S[slot].spin;document.getElementById('spin-'+slot).classList.toggle('on',S[slot].spin);}
function resizeSlot(slot){var wrap=document.getElementById('vw-'+slot);var r=S[slot].rend,c=S[slot].cam;if(!r||!c)return;var W=wrap.clientWidth,H=wrap.clientHeight;r.setSize(W,H);c.aspect=W/H;c.updateProjectionMatrix();}

function loadCustom(){
  var id=document.getElementById('cpdb').value.trim().toUpperCase();
  if(id.length<3){alert('Enter a valid PDB ID (4-5 chars).');return;}
  fetch('https://files.rcsb.org/download/'+id+'.pdb')
    .then(function(r){if(!r.ok)throw new Error('Not found');return r.text();})
    .then(function(txt){
      var chains={};
      txt.split('\n').forEach(function(line){if(!line.startsWith('ATOM')||line.slice(12,16).trim()!=='CA')return;var ch=line[21]||'A',x=parseFloat(line.slice(30,38)),y=parseFloat(line.slice(38,46)),z=parseFloat(line.slice(46,54));if(isNaN(x))return;if(!chains[ch])chains[ch]=[];chains[ch].push([x,y,z]);});
      P[id]={full_name:'Custom: '+id,gene:id,cancer_types:[],therapy_types:[],color:'#a78bfa',approved:false,fda_drugs:[],description:'Custom PDB '+id+' loaded from RCSB.',mechanism:'-',binding_site:'-',mw_kda:0,trials_approx:0,pdb:id,coords:chains,simulation:null};
      if(!document.getElementById('tc-'+id)){var card=document.createElement('div');card.className='tc';card.id='tc-'+id;card.style.setProperty('--lc','#a78bfa');card.innerHTML='<div class="tc-n">'+id+'</div><div class="tc-g">Custom PDB</div>';card.onclick=(function(k){return function(){loadTarget(k,'a',false);};})(id);document.getElementById('tlist').prepend(card);}
      loadTarget(id,'a',false);
    })
    .catch(function(e){alert('Could not load PDB '+id);});
}

window.addEventListener('resize',function(){resizeSlot('a');resizeSlot('b');});
</script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js" onload="THREE_OK=true;initSlot('a');var fk=Object.keys(P)[0];if(fk)loadTarget(fk,'a',false);" onerror="console.warn('Three.js failed to load from CDN. 3D viewing requires internet.')"></script>
</body>
</html>"""

HTML = HTML_TEMPLATE.replace("__PJSON__", P_JSON).replace("__MIJSON__", MI_JSON)

with open("outputs/molecules.html","w",encoding="utf-8") as f:
    f.write(HTML)

sz = Path("outputs/molecules.html").stat().st_size/1024
has_sim = "with simulation data" if mol_insights else "no simulation data"
print(f"\n  outputs/molecules.html  ({sz:.0f} KB)  {has_sim}")
print(f"  Open:  open outputs/molecules.html")
