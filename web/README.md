# web/ — static explainer (Phase 3, not yet built)

`index.html` will live here: a single self-contained page that loads
`../artifacts/results.json` and lets the reader peel back each layer of leakage
(leaked → registration-only → strict + temporal), watching the AUC fall at each
step. Pure client-side, no API keys, no backend. Deployable to GitHub Pages or
Vercel as-is.
