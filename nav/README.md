# Shared site navigation

One flat nav bar, identical on every page, linking the project's parts. Built as
a dependency-free HTML/CSS/JS partial so it drops into the static pages (served by
GitHub Pages) and the local Flask matcher alike. No build step.

## Files

- `nav.html` - the canonical partial (skip link, font links, scoped styles, the
  `<nav>`, and the active-state script). This is the single source of truth.
- `inject_nav.py` - injects `nav.html` into a static HTML page after `<body>`,
  idempotently. Re-run it after regenerating a page.

## Links

| Label         | Destination                                   | data-nav key |
|---------------|-----------------------------------------------|--------------|
| (wordmark)    | landing page                                  | -            |
| Findings      | `web/` (leakage explainer)                    | `findings`   |
| Data & Tools  | `outputs/dashboard.html`                       | `data`       |
| For Patients  | `http://127.0.0.1:5050` (matcher, local only) | `patients`   |
| About         | landing page                                  | `about`      |
| Molecules     | `outputs/molecules.html` (3D viewer)           | `molecules`  |

Static-page links use absolute GitHub Pages URLs so they work publicly. The
matcher link is `http://127.0.0.1:5050` and carries a `Local` tag, since it runs
on your machine and is not on GitHub Pages.

The current page is highlighted automatically: each page sets `data-nav` on its
`<body>` (the injector does this); the script falls back to URL matching if the
attribute is absent.

## Applying it

Static pages (run from repo root):

    python3 nav/inject_nav.py findings web/index.html
    python3 nav/inject_nav.py about    index.html
    python3 nav/inject_nav.py data     outputs/dashboard.html
    python3 nav/inject_nav.py molecules outputs/molecules.html

The Flask matcher lives in the separate `~/trial-matcher` repo. Its inline
template gets the same partial, and its sticky safety banner is offset to
`top:60px` so it sits below the nav. Restart the Flask app after editing.

Note: `legacy/patch_theme.py` injected an older 3-link nav and also re-themed the
pages. It is superseded for navigation. Use `inject_nav.py` instead.

## Design tokens

Set on `#proj-nav` (see `nav.html`):

    --pn-ink:    #14181A   text
    --pn-paper:  #F4F2EC   bar background (warm off-white)
    --pn-line:   rgba(20,24,26,.14)   hairline borders
    --pn-muted:  #5A6360   default link color
    --pn-signal: #0E6E64   deep teal - active underline and focus ring ONLY

No gradients, no shadows, border-radius <= 2px, no emoji.

## Fonts

Loaded from Google Fonts (no build): **Fraunces** (serif wordmark) and
**IBM Plex Mono** (uppercase nav labels). To swap to licensed faces (for example
Geist or GT America for the labels, Söhne for the wordmark), change the
`<link>` in `nav.html` and the `font-family` declarations on `.pn-brand` and
`a.pn-link`. The stack falls back to system serif/monospace offline.

## Accessibility

Semantic `<nav aria-label="Primary">`, real links, a skip-to-content link
targeting `#pn-main`, `aria-current="page"` on the active link, a visible
`:focus-visible` ring in `--signal`, and `prefers-reduced-motion` support.
Verified with axe (zero violations) at 360 / 768 / 1280 / 1920.
