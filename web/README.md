# web/ : static explainer

`index.html` is a single self-contained page. It loads `../artifacts/results.json`
(with an embedded fallback copy, so it also works from the filesystem and offline)
and lets the reader peel back each layer of leakage: leaked, then registration-only,
then strict + temporal, watching the AUC fall at each step. A decision-threshold cost
slider shows what the model would do under asymmetric mistake costs.

Pure client-side. No API keys, no backend, no per-user cost. The only external request
is the Chart.js library from a CDN. Deployable to GitHub Pages or Vercel as-is.

Refresh the embedded data after recomputing results: `python src/build_web.py` (or `make web`).
View locally: `make serve`, then open http://localhost:8000/web/index.html.
