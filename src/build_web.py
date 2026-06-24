"""
build_web.py: embed the current artifacts/results.json into web/index.html.

    python src/build_web.py

The explainer prefers to fetch ../artifacts/results.json at runtime (so a deployed copy
stays in sync with the canonical artifact), but falls back to an embedded copy when opened
directly from the filesystem or offline. This script refreshes that embedded copy so the
two never drift. Run it after src/run.py.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "artifacts" / "results.json"
HTML = ROOT / "web" / "index.html"

PAT = re.compile(
    r'(<script id="results-data" type="application/json">)(.*?)(</script>)',
    re.DOTALL,
)


def main():
    data = json.loads(RESULTS.read_text())          # validate it parses
    payload = json.dumps(data, separators=(",", ":"))
    html = HTML.read_text()
    if not PAT.search(html):
        raise SystemExit("marker <script id=\"results-data\"> not found in web/index.html")
    html = PAT.sub(lambda m: m.group(1) + payload + m.group(3), html, count=1)
    HTML.write_text(html)
    print(f"  embedded {RESULTS.relative_to(ROOT)} ({len(payload)//1024} KB) into {HTML.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
