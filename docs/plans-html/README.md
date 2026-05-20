# Ambix HTML Plan Site — Operator Note

## How to view

Open `index.html` in any browser directly from the filesystem:

```bash
xdg-open docs/plans-html/index.html
# or on macOS:
open docs/plans-html/index.html
# or just navigate to the file in your browser (file:// works — no server needed)
```

The site has no external dependencies. All CSS, JS, and HTML are self-contained
under `docs/plans-html/`. It works offline.

## Site structure

```
docs/plans-html/
├── index.html                 # coordination hub: timeline, status table, blocking tasks, open decisions
├── decisions.html             # aggregated open-decisions view (all 9 decisions on one page)
├── STRATEGY.html              # vision, roadmap, success criteria, risk register
├── v0-runway.html             # active operational plan + ROI-ranked next steps
├── compute.html               # SLURM patterns, FSDP, reservation request
├── data-acquisition.html      # FAIR-MAST endpoints, probe findings, bulk-download protocol
├── data-quality.html          # audit framework, FAIR-MAST format reality, training-grade gate
├── tokenizers.html            # multi-modal tokenizer design + expansion roadmap (§12)
├── tokenizer-benchmarks.html  # closed-loop comparison framework, rFID acceptance gates
├── world-model-v0.html        # WHAM-style model spec, training recipe, rollout
├── demo.html                  # wide-angle camera forward-prediction demo
├── _generate.py               # regeneration script (see below)
└── assets/
    ├── style.css              # shared styles + dark-mode CSS variables
    ├── app.js                 # navigation, theme, sidebar, timeline, filter
    └── prompts.js             # decision-capture follow-on prompt builder
```

## Decision capture workflow

Each open decision appears as an interactive form in `index.html` (Section D)
and in `decisions.html`. To use:

1. Open `index.html` or `decisions.html` in a browser.
2. Find the decision you want to resolve.
3. Select an option from the radio buttons.
4. Optionally add clarifying notes in the text area.
5. Click **Generate follow-on prompt**.
6. A copy-pasteable prompt appears below the form.
7. Click **Copy** to copy it to the clipboard.
8. Paste it into a new Claude Code conversation (the AI coordinator).
   The prompt is self-contained: it names the decision, the chosen option,
   your notes, and specifies which plan files to update.

### Example output prompt

```
[decision: drop-charge-exchange-from-training]
Chosen: yes-drop

Drop charge_exchange entirely from the v0 training manifest.
Rationale: audit found ~50 % bit-pattern corruption in t_i and v_i
columns (values 10^26 – 10^38 K / m/s; physical ranges ≤ 30 keV / 10^7 m/s).
These values are 12–28 orders of magnitude beyond physical range and
represent genuine float-encoding defects in the FAIR-MAST CX ingestion.

Please update plans/data-quality.md §5 (training-grade gate) and
plans/world-model-v0.md §3 (training data shape) accordingly, then
re-derive the training-grade-shots.json manifest excluding CX.
```

## How to regenerate

The HTML is generated from the markdown plans under `plans/` by a Python script.

**Prerequisites:** `markdown-it-py` is already in the project venv (installed
as a transitive dependency of `mkdocs-material`).

```bash
# From the repo root:
python docs/plans-html/_generate.py
```

This overwrites all `.html` files under `docs/plans-html/`. The assets
(`style.css`, `app.js`, `prompts.js`) are NOT regenerated — edit them directly
if you need to change styles or decision-prompt logic.

**When to regenerate:**

- After editing any file under `plans/` — the HTML pages are pre-rendered from
  the markdown source and will go stale otherwise.
- After adding a new plan file — add the new stem + label to the `PLAN_ORDER`
  list in `_generate.py` and re-run.

**Editing decision prompts:**

Decision prompt logic lives in `assets/prompts.js`. Each decision has an entry
in the `DECISIONS` map. To add or modify a decision, edit `prompts.js` directly
(no regeneration needed for JS/CSS changes).

## Theme

The site supports three theme modes, toggled via the button in the top-right:

- **Auto** (default) — follows the OS `prefers-color-scheme` setting.
- **Dark** — forced dark theme.
- **Light** — forced light theme.

The selection persists in `localStorage` across browser sessions.
