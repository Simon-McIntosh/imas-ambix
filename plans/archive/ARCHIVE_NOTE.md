# Archive note — 2026-05-21

These markdown files were the original plan source until **2026-05-21**,
when the project transitioned to **hand-authored HTML** under
`docs/plans-html/` as the single source of truth. The original
`README.md` in this directory is preserved verbatim and remains a
useful read-only snapshot of the plan index at the moment of transition.

## Why they're here

- **Git history.** The full edit history of each plan is preserved on
  the active branches; this archive provides a quick read-only view.
- **Authorship trail.** Commit messages from when the plans were first
  written reference the markdown filenames; this directory keeps those
  links resolvable.
- **Migration audit.** Compare each archived `<plan>.md` against the
  hand-authored `docs/plans-html/<plan>.html` to verify the transition
  preserved content.

## Do not edit these

If you need to update a plan, **edit the HTML directly** in
`docs/plans-html/`. The legacy `_generate.py` renderer has been deleted
and the markdown→HTML pipeline is no longer wired. Editing the markdown
here will have no effect on the rendered site.

## What replaced what

| Archived markdown | New HTML source of truth |
|---|---|
| `STRATEGY.md` | `../../docs/plans-html/STRATEGY.html` |
| `v0-runway.md` | `../../docs/plans-html/v0-runway.html` |
| `compute.md` | `../../docs/plans-html/compute.html` |
| `data-acquisition.md` | `../../docs/plans-html/data-acquisition.html` |
| `data-quality.md` | `../../docs/plans-html/data-quality.html` |
| `tokenizers.md` | `../../docs/plans-html/tokenizers.html` (plus the per-stage `tokenizers-12-landed.html`) |
| `tokenizer-benchmarks.md` | `../../docs/plans-html/tokenizer-benchmarks.html` |
| `world-model-v0.md` | `../../docs/plans-html/world-model-v0.html` |
| `demo.md` | `../../docs/plans-html/demo.html` |
| `README.md` | `../../docs/plans-html/index.html` (the old plan index lives on as `README.md` here) |

## The pattern going forward

See `~/.claude/skills/html-docs/SKILL.md`:

- HTML is the source of truth — no markdown round-trip.
- When a plan's work moves into a new phase, **create a new file** with
  a stage suffix (e.g. `tokenizers-12-landed.html`) rather than
  overwriting the evergreen page. Old per-stage HTML is the audit trail.
- Decisions are captured per-page via `assets/state.js`, which writes
  JSON to `~/docs-server/state/imas-ambix/<doc>.json`. Both browsers
  and agents read the same JSON files.
- The cross-plan aggregator at `docs/plans-html/decisions.html` reads
  every owning plan's state file live.

## Migration commit chain (2026-05-21)

- `070a7d2` STRATEGY.html
- `6f7fd8a` v0-runway.html + smoke-training-timing decision
- `81fc92e` compute.html + slurm-training-reservation decision
- `38333b3` data-acquisition.html + camera-selection-v0 decision
- `730fe4b` data-quality.html + drop-charge-exchange decision
- `6979ea5` tokenizer-benchmarks.html
- `5a137fc` tokenizers.html + 4 §12 decisions
- `bbd165d` world-model-v0.html + demo.html (+ demo-shots-selection decision)
- (this commit) — archive markdown, retire generator, decisions aggregator
