# imas-ambix plan site

This directory is **both** a GitHub Pages source folder and the home of the
project's plan HTML documents.

Layout:

| Path | Purpose |
|------|---------|
| `index.html` | Dashboard — top-level entry, served at site root |
| `STRATEGY.html`, `world-model-v0.html`, `v0-runway.html` | Evergreen strategy plans |
| `compute.html`, `data-acquisition.html`, `data-quality.html` | Topic-specific plans |
| `tokenizers.html`, `tokenizer-benchmarks.html`, `tokenizers-12-landed.html` | Tokeniser-track plans |
| `decisions.html` | Decision log |
| `demo.html` | Demo / runbook |
| `assets/` | Shared CSS/JS — `style.css`, `app.js`, `state.js` |
| `state/imas-ambix/*.json` | Repo-tracked state — the canonical record |
| `archive/*.md` | Original markdown sources (preserved as historical record) |

Local view (editable, decisions in browser `localStorage`):

```bash
# Start the docs-server (once per session)
tmux new -d -s docs-server 'python3 ~/docs-server/serve.py'
# Open: http://localhost:8765/imas-ambix/
```

Published view (read-only):

> <https://simon-mcintosh.github.io/imas-ambix/>

State JSON (`plans/state/imas-ambix/*.json`) is the **canonical** record.
HTML decision-capture buttons write to `localStorage` only — promote a
decision to the repo by editing the JSON and committing.

## Shared infrastructure with imas-efit

The plan management system (state.js, mode banner, repo-tracked static state
JSON pattern, .nojekyll, README badge convention) is shared with the
[imas-efit](https://github.com/Simon-McIntosh/efit) project (private repo —
local docs-server access only).  This repo (imas-ambix) is the public
reference site that the team and Claude Design can use to see the patterns
in action.

The canonical home of state.js and the page-template scaffold is the
[`html-docs` skill](~/.claude/skills/html-docs/); the `/html-plan` slash
command bootstraps a new project against that skill.  Per-project
customisation (style.css, app.js, page layouts) is free to differ — only the
mechanics (state file location, mode detection, read-only banner) need to
stay uniform.
