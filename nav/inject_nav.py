#!/usr/bin/env python3
"""Inject the shared site nav (nav/nav.html) into an HTML page, after <body>.
Idempotent: strips any prior shared nav and the legacy __nav__ first, sets
data-nav="<key>" on <body> (active section), and adds the skip-link anchor.

Run from the repo root, e.g. after regenerating the dashboard or molecules page:
  python3 nav/inject_nav.py data outputs/dashboard.html
  python3 nav/inject_nav.py ""   outputs/molecules.html

Keys: findings | data | patients | about | "" (no active item)

The Flask matcher (separate ~/trial-matcher repo) is patched by its own script;
this injector only handles the static HTML pages served by GitHub Pages.
"""
import re, sys, os

NAV = open(os.path.join(os.path.dirname(__file__), "nav.html"), encoding="utf-8").read().strip()
ANCHOR = '<span id="pn-main" tabindex="-1"></span>'

def process(path, key):
    with open(path, encoding="utf-8") as f:
        html = f.read()
    orig = html
    html = re.sub(r'<a class="pn-skip"[^>]*>.*?</a>\s*', '', html, flags=re.DOTALL)
    html = re.sub(r'<!-- shared-project-nav:start.*?shared-project-nav:end -->\s*', '', html, flags=re.DOTALL)
    html = re.sub(r'<span id="pn-main"[^>]*></span>\s*', '', html)
    html = re.sub(r'<div id="__nav__"[^>]*>.*?</div>\s*', '', html, flags=re.DOTALL)
    m = re.search(r'<body[^>]*>', html, flags=re.IGNORECASE)
    if not m:
        raise SystemExit(f"  ERROR: no <body> in {path}")
    new_body = re.sub(r'\s+data-nav="[^"]*"', '', m.group(0))
    if key:
        new_body = new_body[:-1] + f' data-nav="{key}">'
    html = html[:m.start()] + new_body + html[m.end():]
    insert_at = html.index('>', html.lower().index('<body')) + 1
    html = html[:insert_at] + '\n' + NAV + '\n' + ANCHOR + '\n' + html[insert_at:]
    if html == orig:
        print(f"  no change: {path}")
        return
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  injected nav (data-nav={key!r}): {path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    key = sys.argv[1]
    for p in sys.argv[2:]:
        process(p, key)
