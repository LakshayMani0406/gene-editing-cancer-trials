"""
patch_theme.py - Apply unified pharma clinical white theme to all 3 pages
Run from git2/:  python3 patch_theme.py

Also re-applies the nav bar (removes old one first).
"""
import os, re

BASE = os.path.join(os.path.dirname(__file__), "outputs")

PAGES = {
    "dashboard.html":     "Dashboard",
    "molecules.html":     "3D Molecules",
    "image_compare.html": "Image Analyzer",
}
NAV_ICONS = {
    "dashboard.html": "&#9776;",
    "molecules.html": "&#9684;",
    "image_compare.html": "&#128247;",
}

# ── Color map: dark → pharma light ───────────────────────────────────────────
# Applied only inside <style> blocks so chart/accent colors are preserved
CSS_SWAP = {
    # backgrounds
    "#060d1a": "#f1f5f9",   # darkest bg    → light sheet
    "#0a1628": "#f8fafc",   # dark panel    → near white
    "#0d1f3c": "#ffffff",   # dark card     → white
    "#102040": "#eff6ff",   # hover         → blue tint hover
    "#1a3356": "#e2e8f0",   # dark border   → light border
    "#162435": "#cbd5e1",   # darker border → slate border
    # text
    "#c8dff5": "#0f172a",   # primary text  → near black
    "#8ab4d4": "#334155",   # secondary     → slate
    "#7a9bbf": "#64748b",   # muted         → medium slate
}

# ── Colors that must STAY (accents, charts, protein colors) ──────────────────
KEEP = {"#00d4aa","#22c55e","#4a9eff","#8b5cf6","#ec4899","#f59e0b",
        "#ef4444","#f97316","#84cc16","#14b8a6","#a78bfa","#fff","#ffffff",
        "#1d4ed8","#2563eb","#3b82f6","#0ea5e9","#10b981"}

def swap_css_colors(css_block):
    """Replace dark colors with light equivalents in a CSS string."""
    result = css_block
    for dark, light in CSS_SWAP.items():
        # Replace as whole color tokens (not inside longer hex strings)
        result = re.sub(
            r'(?<![0-9a-fA-F])' + re.escape(dark) + r'(?![0-9a-fA-F])',
            light, result, flags=re.IGNORECASE
        )
    return result

# ── Universal pharma theme additions (injected AFTER existing styles) ─────────
PHARMA_ADD = """
<style id="__pharma__">
/* ─── Pharma clinical theme additions ─── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
html,body{font-family:'Inter',system-ui,-apple-system,sans-serif!important;-webkit-font-smoothing:antialiased;}

/* Headers stay dark navy - intentional */
.hero,.hdr,.hdr-row,.main-header{
  background:linear-gradient(135deg,#0f2044 0%,#162a52 60%,#1e3a6e 100%)!important;
  border-bottom:2px solid #2563eb!important;
}
.hero *,.hdr *{color:#e2e8f0!important}
.hero h1,.hdr h1{color:#fff!important}
.hero h1 span,.hdr h1 span,.hero-title span{color:#60a5fa!important}
.hero-sub,.hdr-sub{color:#93c5fd!important}

/* Sidebar stays dark for 3D viewer usability */
.sidebar{background:#0a1628!important;border-right:1px solid #1e3a6e!important}
.tc{background:#0d1f3c!important;border-color:#1a3356!important;color:#c8dff5!important}
.tc:hover,.tc.on{background:#102040!important;border-left-color:var(--lc,#2563eb)!important}
.tc-n{color:#fff!important}.tc-g,.tc-ct{color:#93c5fd!important}
.cmp-box,.pdb-box{background:#060d1a!important;border-color:#1a3356!important}
.cmp-box label,.pdb-box div{color:#93c5fd!important}
.cmp-box select,.pi{background:#0d1f3c!important;color:#e2e8f0!important;border-color:#1a3356!important}
.cmp-btn{background:rgba(37,99,235,.2)!important;border-color:rgba(96,165,250,.4)!important;color:#93c5fd!important}
.pb{background:rgba(37,99,235,.15)!important;border-color:rgba(96,165,250,.3)!important;color:#93c5fd!important}
.ss{color:#93c5fd!important}

/* Main content area - white clinical */
.main,#main,.tab-content,.results-area,.results-wrap,.main-wrap,
.page>.main,.layout>.main{background:#f1f5f9!important}

/* Tab nav - white bar */
.tab-nav,.tabnav,.mode-bar{
  background:#ffffff!important;
  border-bottom:1px solid #e2e8f0!important;
  box-shadow:0 1px 3px rgba(0,0,0,.06)!important;
}
.tab-btn,.mtab,.ntab,.mode-tab{color:#475569!important;font-family:'Inter',sans-serif!important}
.tab-btn.active,.tab-btn:hover,.mtab.on,.ntab.on,.mode-tab.on{
  color:#1d4ed8!important;border-bottom-color:#1d4ed8!important}

/* Cards - white with clean borders */
.card,.insight-card,.insight-box,.pc,.res-panel,.upload-panel,
.detail-card,.dc,.exp-box,.analysis-box,.diff-section,.action-bar,
.action-row,.summary-row,.empty-state,.stat-box{
  background:#ffffff!important;
  border-color:#e2e8f0!important;
  border-radius:8px!important;
  box-shadow:0 1px 3px rgba(0,0,0,.06),0 1px 2px rgba(0,0,0,.04)!important;
  color:#0f172a!important;
}

/* Charts container */
.chart-wrap,.chart-box,.chart-container{
  background:#ffffff!important;border-color:#e2e8f0!important;border-radius:8px!important}

/* Overview stat cards */
.stat-card{background:#ffffff!important;border-color:#e2e8f0!important;border-radius:8px!important}
.stat-label{color:#64748b!important}.stat-value{color:#0f172a!important}

/* Inputs */
.key-input,.key-in,input.pi,input[type="text"],input[type="password"],select{
  background:#f8fafc!important;border-color:#e2e8f0!important;color:#0f172a!important;
  border-radius:6px!important}
.key-input:focus,.key-in:focus{border-color:#2563eb!important;box-shadow:0 0 0 3px rgba(37,99,235,.12)!important}

/* Buttons */
.run-btn,.abtn,.analyze-btn{
  background:#2563eb!important;color:#fff!important;border:none!important;
  border-radius:6px!important;font-family:'Inter',sans-serif!important}
.run-btn:hover,.abtn:hover{background:#1d4ed8!important}

/* Typography */
h1,h2,h3,h4,.col-title,.res-ptitle,.shd{font-family:'Inter',sans-serif!important;color:#0f172a!important}
code,pre,.mono,[class*="mono"],font[face*="mono"]{font-family:'JetBrains Mono','Fira Code',monospace!important}

/* Scrollbars */
::-webkit-scrollbar{width:5px;height:5px}
::-webkit-scrollbar-track{background:#f1f5f9}
::-webkit-scrollbar-thumb{background:#cbd5e1;border-radius:3px}
::-webkit-scrollbar-thumb:hover{background:#94a3b8}

/* 3D viewer dark canvas */
#vw-a,#vw-b,.vw{background:#060d1a!important}
</style>"""

# ── Nav bar ───────────────────────────────────────────────────────────────────
def make_nav(current):
    links = ""
    for fname, label in PAGES.items():
        icon = NAV_ICONS[fname]
        active = fname == current
        if active:
            style = "color:#fff;background:rgba(255,255,255,.18);font-weight:600;border-radius:5px"
        else:
            style = "color:#93c5fd;border-radius:5px"
        links += (
            f'<a href="{fname}" style="{style};font-size:.69rem;font-family:Inter,monospace;'
            f'padding:6px 14px;text-decoration:none;display:inline-flex;align-items:center;'
            f'gap:5px;transition:.15s;" '
            f'onmouseover="this.style.background=\'rgba(147,197,253,.18)\'" '
            f'onmouseout="this.style.background=\'{"rgba(255,255,255,.18)" if active else "transparent"}\'">'
            f'{icon}&nbsp;{label}</a>'
        )
    return (
        '<div id="__nav__" style="background:linear-gradient(135deg,#0f2044,#1e3a6e);'
        'border-bottom:2px solid #1d4ed8;padding:0 20px;height:38px;'
        'display:flex;align-items:center;gap:4px;position:sticky;top:0;z-index:99999;'
        'font-family:Inter,system-ui,sans-serif;box-shadow:0 2px 8px rgba(0,0,0,.2);">'
        + links +
        '<span style="margin-left:auto;font-size:.57rem;font-family:monospace;'
        'color:rgba(147,197,253,.45);white-space:nowrap;">'
        'Gene-Editing Cancer Trials &middot; Lakshay Mani &middot; MS Analytics &middot; Northeastern</span>'
        '</div>'
    )

# ── Process files ─────────────────────────────────────────────────────────────
for fname in PAGES:
    path = os.path.join(BASE, fname)
    if not os.path.exists(path):
        print(f"  SKIP (not found): {fname}")
        continue

    with open(path, encoding="utf-8") as f:
        html = f.read()

    # 1. Remove previous patches
    html = re.sub(r'<div id="__nav__"[^>]*>.*?</div>\s*', '', html, flags=re.DOTALL)
    html = re.sub(r'<style id="__pharma(?:_theme)?__">.*?</style>\s*', '', html, flags=re.DOTALL)

    # 2. Swap dark colors inside <style> blocks only
    def replace_in_styles(m):
        return '<style>' + swap_css_colors(m.group(1)) + '</style>'
    html = re.sub(r'<style>(.*?)</style>', replace_in_styles, html, flags=re.DOTALL)

    # 3. Inject pharma additions before </head>
    html = html.replace('</head>', PHARMA_ADD + '\n</head>', 1)

    # 4. Inject nav after <body>
    nav = make_nav(fname)
    if '<body>' in html:
        html = html.replace('<body>', '<body>\n' + nav, 1)
    else:
        html = re.sub(r'(<body[^>]*>)', r'\1\n' + nav, html, count=1)

    with open(path, 'w', encoding='utf-8') as f:
        f.write(html)

    kb = os.path.getsize(path) // 1024
    print(f"  Patched: {fname}  ({kb} KB)")

print("""
Done. Hard-refresh all tabs (Cmd+Shift+R).

Theme: Clinical white body + dark navy headers + white cards
Nav: sticky bar top of every page - active page highlighted

Now test every feature:
  http://localhost:8080/dashboard.html
  http://localhost:8080/molecules.html
  http://localhost:8080/image_compare.html
""")
