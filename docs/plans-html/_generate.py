#!/usr/bin/env python3
"""
Generate the docs/plans-html/ static site from plans/*.md
Run from repo root:  python docs/plans-html/_generate.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from markdown_it import MarkdownIt

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).parent.parent.parent
PLANS_DIR = REPO_ROOT / "plans"
OUT_DIR   = Path(__file__).parent          # docs/plans-html/

md_engine = MarkdownIt("commonmark").enable("table").enable("strikethrough")


def render_md(text: str) -> str:
    """Render markdown to HTML."""
    return md_engine.render(text)


def extract_headings(text: str) -> list[tuple[int, str, str]]:
    """Return list of (level, text, slug) from markdown text."""
    result = []
    for m in re.finditer(r'^(#{1,4})\s+(.*)', text, re.MULTILINE):
        level = len(m.group(1))
        heading_text = m.group(2).strip()
        # Remove inline markdown (bold, code, links)
        clean = re.sub(r'[*_`]', '', heading_text)
        clean = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', clean)
        clean = re.sub(r'~~([^~]+)~~', r'\1', clean)
        slug = re.sub(r'[^\w\s-]', '', clean).strip().lower()
        slug = re.sub(r'[\s_]+', '-', slug)
        slug = re.sub(r'-+', '-', slug).strip('-')
        result.append((level, heading_text, slug))
    return result


def add_heading_ids(html: str, headings: list[tuple[int, str, str]]) -> str:
    """Add id attributes to h1/h2/h3/h4 tags in rendered HTML."""
    # We do a simple sequential pass: each heading tag gets the next id
    heading_iter = iter(headings)
    def replace_heading(m):
        try:
            lvl, _, slug = next(heading_iter)
            return f'<h{lvl} id="{slug}">'
        except StopIteration:
            return m.group(0)
    return re.sub(r'<h([1-4])>', replace_heading, html)


SHELL_TEMPLATE = '''\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} | Ambix Plans</title>
<link rel="stylesheet" href="{assets}style.css">
</head>
<body>
<!-- header -->
<header class="site-header">
  <div class="logo"><a href="{root}index.html">⚗ Ambix Plans</a></div>
  <nav>
    <a href="{root}index.html">Index</a>
    <a href="{root}STRATEGY.html">Strategy</a>
    <a href="{root}v0-runway.html">v0 Runway</a>
    <a href="{root}compute.html">Compute</a>
    <a href="{root}data-acquisition.html">Data Acq</a>
    <a href="{root}data-quality.html">Quality</a>
    <a href="{root}tokenizers.html">Tokenizers</a>
    <a href="{root}tokenizer-benchmarks.html">Benchmarks</a>
    <a href="{root}world-model-v0.html">World Model</a>
    <a href="{root}demo.html">Demo</a>
    <a href="{root}decisions.html">Decisions</a>
  </nav>
  <div class="spacer"></div>
  <button class="sidebar-toggle" id="sidebar-toggle">&#9776; Sections</button>
  <button class="theme-toggle" id="theme-toggle">☽ Dark</button>
</header>

<div class="page-body">
<!-- sidebar -->
<aside class="sidebar">
  <div class="sidebar-title">On this page</div>
  {sidebar}
</aside>

<!-- main -->
<main class="main-content">
  <div class="content-inner">
    {breadcrumb}
    {content}
    {plan_nav}
  </div>
</main>
</div>

<script src="{assets}prompts.js"></script>
<script src="{assets}app.js"></script>
</body>
</html>
'''


def build_sidebar(headings: list[tuple[int, str, str]]) -> str:
    items = []
    for lvl, text, slug in headings:
        if lvl == 1:
            continue  # skip h1, it's the page title
        css = "h3-link" if lvl >= 3 else ""
        clean = re.sub(r'[*_`~]', '', text)
        clean = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', clean)
        if len(clean) > 50:
            clean = clean[:47] + "…"
        items.append(f'<li><a href="#{slug}" class="{css}">{clean}</a></li>')
    return "<ul>" + "\n".join(items) + "</ul>"


def build_breadcrumb(label: str) -> str:
    return (f'<div class="breadcrumb">'
            f'<a href="index.html">Plans</a>'
            f'<span>›</span>'
            f'<span>{label}</span>'
            f'</div>')


# Plan order (dependency graph order)
PLAN_ORDER = [
    ("STRATEGY",             "Strategy"),
    ("v0-runway",            "v0 Runway"),
    ("compute",              "Compute"),
    ("data-acquisition",     "Data Acquisition"),
    ("data-quality",         "Data Quality"),
    ("tokenizers",           "Tokenizers"),
    ("tokenizer-benchmarks", "Tokenizer Benchmarks"),
    ("world-model-v0",       "World Model v0"),
    ("demo",                 "Demo"),
]


def build_plan_nav(current_stem: str) -> str:
    idx = next((i for i, (s, _) in enumerate(PLAN_ORDER) if s == current_stem), -1)
    prev_link = ""
    next_link = ""
    if idx > 0:
        ps, pl = PLAN_ORDER[idx - 1]
        prev_link = (f'<a href="{ps}.html">'
                     f'<span class="nav-label">← Previous</span>'
                     f'<span class="nav-title">{pl}</span>'
                     f'</a>')
    if idx >= 0 and idx < len(PLAN_ORDER) - 1:
        ns, nl = PLAN_ORDER[idx + 1]
        next_link = (f'<a href="{ns}.html" class="nav-next">'
                     f'<span class="nav-label">Next →</span>'
                     f'<span class="nav-title">{nl}</span>'
                     f'</a>')
    if not prev_link and not next_link:
        return ""
    return f'<nav class="plan-nav">{prev_link}{next_link}</nav>'


def generate_plan_page(stem: str, label: str) -> None:
    src = PLANS_DIR / f"{stem}.md"
    if not src.exists():
        print(f"  SKIP (not found): {src}")
        return
    text = src.read_text(encoding="utf-8")
    headings = extract_headings(text)
    body_html = render_md(text)
    body_html = add_heading_ids(body_html, headings)

    sidebar = build_sidebar(headings)
    breadcrumb = build_breadcrumb(label)
    plan_nav = build_plan_nav(stem)

    html = SHELL_TEMPLATE.format(
        title=label,
        assets="assets/",
        root="",
        sidebar=sidebar,
        breadcrumb=breadcrumb,
        content=body_html,
        plan_nav=plan_nav,
    )
    out = OUT_DIR / f"{stem}.html"
    out.write_text(html, encoding="utf-8")
    print(f"  Wrote {out.name}")


# ---------------------------------------------------------------------------
# Decision card HTML builder
# ---------------------------------------------------------------------------
DECISIONS = [
    {
        "id": "drop-charge-exchange",
        "title": "Drop charge_exchange from v0 training corpus?",
        "context": ("Audit on a 25-shot L2 sample found ~50% bit-pattern corruption in "
                    "<code>charge_exchange/t_i</code> and <code>charge_exchange/v_i</code>, "
                    "with values 10²⁶–10³⁸ (physical range: ≤ 30 keV / 10⁷ m/s). "
                    "The diagnostic is only present in 37.9% of the 11,573 shots regardless. "
                    "Source: <code>plans/data-quality.md</code> §10.4."),
        "options": [
            ("yes-drop",   "Yes — drop charge_exchange entirely from v0 training"),
            ("keep-clip",  "Keep but hard-clip abs > 1e25 and zero out corrupted channels"),
            ("defer",      "Defer — run the full 11,573-shot audit first, decide after"),
        ],
    },
    {
        "id": "plasma-decoder-finetune",
        "title": "Open-MAGVIT2 plasma-domain decoder fine-tune — when to trigger?",
        "context": ("The current ImageNet-pretrained decoder produces MAE 324 (uint16) on "
                    "MAST rbb shot 15085. Expected rFID on plasma imagery without fine-tune: "
                    "10–30. Decoder fine-tune cost: ~4–6 GPU-hours exclusive. "
                    "Trigger gate (from plan): rFID > 5 on the 100-shot rbb benchmark. "
                    "Source: <code>plans/tokenizers.md</code> §12.1."),
        "options": [
            ("yes-now",    "Fine-tune now — start before the 125M smoke run"),
            ("yes-if-rfid", "Fine-tune only if measured rFID > 5 (benchmark first)"),
            ("defer-v1",   "Defer to v1 — proceed with ImageNet decoder for v0"),
        ],
    },
    {
        "id": "patchtst-real-embedding",
        "title": "PatchTST real embedding — defer to v1 vs land in v0?",
        "context": ("Current state: identity passthrough (token ID 0, raw floats in metadata). "
                    "The patch-projection matrix trains end-to-end inside the WHAM trunk. "
                    "Real PatchTST would add ~1M trainable params with channel-independent "
                    "self-attention over patches. Estimated effort: 1 Sonnet session. "
                    "Source: <code>plans/tokenizers.md</code> §12.2."),
        "options": [
            ("defer-v1",   "Defer to v1 — keep identity passthrough for v0"),
            ("land-v0",    "Land real PatchTST in v0 (after the 125M smoke run is green)"),
        ],
    },
    {
        "id": "equilibrium-2d-tokenizer",
        "title": "Equilibrium 2-D tokenizer — which architecture?",
        "context": ("Currently: equilibrium 2-D enters as a continuous cross-attention tensor "
                    "(not tokenized). Two options to tokenize: "
                    "Option A — reuse Open-MAGVIT2 at upsampled 256×256 (0.5 Sonnet sessions, no fine-tune). "
                    "Option B — Cosmos-Tokenizer-DV at native 65×65 (NVIDIA OML weights, license check needed). "
                    "Source: <code>plans/tokenizers.md</code> §12.3."),
        "options": [
            ("keep-v0",   "Keep as continuous cross-attention tensor for v0 (current plan)"),
            ("option-a",  "Option A — Open-MAGVIT2 upsampled 256×256 (reuse codebook)"),
            ("option-b",  "Option B — Cosmos-Tokenizer-DV at native 65×65 (separate codebook)"),
        ],
    },
    {
        "id": "ir-camera-codebook",
        "title": "IR camera codebook — share with visible or allocate separate?",
        "context": ("Only 25 rir shots in FAIR-MAST (per data-acquisition.md §10.4). "
                    "Current v0 default: share the Open-MAGVIT2 visible codebook. "
                    "Decision gate: if IR MAE > 2× rbb MAE, allocate separate block. "
                    "rFID unreliable with n=25; use MAE as primary metric. "
                    "Source: <code>plans/tokenizers.md</code> §12.5."),
        "options": [
            ("share",    "Share visible codebook for IR (v0 default)"),
            ("separate", "Allocate separate registry block and fine-tune IR decoder"),
        ],
    },
    {
        "id": "slurm-training-reservation",
        "title": "SLURM dedicated training reservation — file with SDCC or stay with exclusive-pause?",
        "context": ("Current decision (2026-05-20): exclusive-pause mode — stop DeepSeek V4-Flash "
                    "before training, restart after. Alternative: file a dedicated "
                    "<code>gpu_0003_grpA_train</code> reservation (request body in "
                    "<code>plans/compute.md</code> §3.2). "
                    "Source: <code>plans/compute.md</code> §2–3."),
        "options": [
            ("exclusive-pause", "Stay with exclusive-pause (current plan, no SDCC request needed)"),
            ("file-now",        "File the dedicated reservation request with SDCC ops now"),
            ("schedule-windows","Schedule around serving in overnight/weekend windows"),
        ],
    },
    {
        "id": "smoke-training-timing",
        "title": "125M smoke training run — run now or wait for bulk encode?",
        "context": ("Training loop FSDP scaffold is landed. Bulk-encode CLI produced 9,380 tokens "
                    "for shot 15085 (CPU). A smoke run needs ~10 shots. "
                    "The full GPU bulk-encode for ~3,000 rbb shots is pending the SLURM run. "
                    "Source: <code>plans/v0-runway.md</code> §4."),
        "options": [
            ("run-now",   "Run now with the ~10 CPU-encoded shots (validate loop immediately)"),
            ("wait-gpu",  "Wait for GPU bulk-encode to complete (~3,000 shots)"),
        ],
    },
    {
        "id": "demo-shots-selection",
        "title": "Demo shots final selection",
        "context": ("Current candidates: 30420 + 30421 (FAIR-MAST quickstart shots) + one M6-era TBD. "
                    "Audit data now available to pick shots with confirmed clean data. "
                    "Charge-exchange corruption affects ~50% of shots but shots 30420/30421 "
                    "are M9-era — need to verify their CX status. "
                    "Source: <code>plans/demo.md</code> §3."),
        "options": [
            ("30420-30421-m6",    "30420 + 30421 + M6-era TBD (original plan)"),
            ("30420-30421-no-cx", "30420 + 30421 + M8/M9 shot confirmed clean CX"),
            ("defer",             "Defer selection until full corpus audit is complete"),
        ],
    },
    {
        "id": "camera-selection-v0",
        "title": "Camera selection for v0 training",
        "context": ("Level-1 inventory: rbb = 9,527 shots (55.7%), rba = 6,155 shots (36.0%), "
                    "rir = 25 shots (rare). "
                    "rbb is the wide-angle midplane view; rba is the lower/divertor view; "
                    "rir is infrared divertor. Adding more cameras → more training data per shot "
                    "but adds IR codebook complexity. "
                    "Source: <code>plans/data-acquisition.md</code> §11.2."),
        "options": [
            ("rbb-only",  "rbb only — 9,527 shots, widest coverage"),
            ("rbb-rba",   "rbb + rba — both visible cameras, ~6,155 overlap shots"),
            ("rbb-rba-rir", "rbb + rba + rir — all cameras including IR"),
        ],
    },
]


def build_decision_card(d: dict) -> str:
    options_html = ""
    for val, label in d["options"]:
        options_html += (
            f'<label><input type="radio" name="opt" value="{val}"> {label}</label>\n'
        )

    return f'''
<div class="decision-card" id="decision-{d["id"]}">
  <h3>{d["title"]}</h3>
  <div class="decision-context">{d["context"]}</div>
  <form data-decision-id="{d["id"]}">
    <div class="decision-options">
{options_html}    </div>
    <textarea class="decision-notes" placeholder="Optional notes or context…"></textarea>
    <button type="submit" class="btn-generate">Generate follow-on prompt</button>
    <div class="prompt-output">
      <pre class="prompt-pre"></pre>
      <button type="button" class="btn-copy">Copy</button>
    </div>
  </form>
</div>
'''


def generate_index() -> None:
    # Quick-status badges
    status_grid = '''
<div class="status-grid">
  <div class="status-card">
    <div class="card-label">L2 Mirror</div>
    <div class="card-value"><span class="badge badge-done">Complete</span> 11,573 shots · 4.5 TB</div>
  </div>
  <div class="status-card">
    <div class="card-label">L1 Camera Mirror</div>
    <div class="card-value"><span class="badge badge-done">Complete</span> 11,029 shot dirs · 2.7 TB</div>
  </div>
  <div class="status-card">
    <div class="card-label">L1 All-Sources</div>
    <div class="card-value"><span class="badge badge-flight">In flight</span> launched ~07:00 UTC</div>
  </div>
  <div class="status-card">
    <div class="card-label">Tokenizer Scaffold</div>
    <div class="card-value"><span class="badge badge-done">Done</span> registry + Open-MAGVIT2 + Chronos + PatchTST</div>
  </div>
  <div class="status-card">
    <div class="card-label">Data Quality</div>
    <div class="card-value"><span class="badge badge-done">Done</span> 22 tests · 76 % usable rate on 25-shot sample</div>
  </div>
  <div class="status-card">
    <div class="card-label">Bench Framework</div>
    <div class="card-value"><span class="badge badge-done">Done</span> 14 tests · real bench on 8 rbb frames</div>
  </div>
  <div class="status-card">
    <div class="card-label">Training Loop</div>
    <div class="card-value"><span class="badge badge-done">Done</span> FSDP + cosine LR + weighted CE · 25 tests</div>
  </div>
  <div class="status-card">
    <div class="card-label">Demo CLI</div>
    <div class="card-value"><span class="badge badge-done">Done</span> rollout() + 20 tests + 31 eval tests</div>
  </div>
  <div class="status-card">
    <div class="card-label">Bulk GPU Encode</div>
    <div class="card-value"><span class="badge badge-next">Next</span> Open-MAGVIT2 GPU runner ready; needs SLURM run</div>
  </div>
  <div class="status-card">
    <div class="card-label">Smoke Training Run</div>
    <div class="card-value"><span class="badge badge-blocked">Blocked</span> needs bulk encode + reservation</div>
  </div>
  <div class="status-card">
    <div class="card-label">500M Curriculum</div>
    <div class="card-value"><span class="badge badge-blocked">Blocked</span> Phase 2d green + reservation granted</div>
  </div>
  <div class="status-card">
    <div class="card-label">v0 Demo</div>
    <div class="card-value"><span class="badge badge-blocked">Blocked</span> trained checkpoint needed</div>
  </div>
</div>
'''

    # Timeline
    timeline_html = '''
<div class="timeline" role="list">
  <div class="timeline-phase phase-done" role="listitem" tabindex="0">
    <div class="phase-dot"></div>
    <div class="phase-label">Phase 0<br>Probe &amp; Mirror</div>
    <div class="phase-badge"><span class="badge badge-done">Done</span></div>
    <div class="phase-detail">
      L2 mirror: 11,573 shots, 4.5 TB<br>
      L1 cameras: 11,029 shots, 2.7 TB<br>
      L1-all in flight (~07:00 UTC)
    </div>
  </div>
  <div class="timeline-phase phase-done" role="listitem" tabindex="0">
    <div class="phase-dot"></div>
    <div class="phase-label">Phase 1<br>Tokenizer Proto</div>
    <div class="phase-badge"><span class="badge badge-done">Done</span></div>
    <div class="phase-detail">
      Registry + Open-MAGVIT2<br>
      Chronos + PatchTST<br>
      Multimodal aggregator<br>
      51 tokenizer tests green
    </div>
  </div>
  <div class="timeline-phase phase-done" role="listitem" tabindex="0">
    <div class="phase-dot"></div>
    <div class="phase-label">Phase 1.5<br>Quality &amp; Bench</div>
    <div class="phase-badge"><span class="badge badge-done">Done</span></div>
    <div class="phase-detail">
      Audit recalibrated (FAIR-MAST xarray-on-Zarr)<br>
      Bench framework landed (14 tests)<br>
      Calibration library (17 tests)<br>
      76 % usable-for-training on 25-shot sample
    </div>
  </div>
  <div class="timeline-phase phase-done" role="listitem" tabindex="0">
    <div class="phase-dot"></div>
    <div class="phase-label">Phase 2a<br>Model Scaffold</div>
    <div class="phase-badge"><span class="badge badge-done">Done</span></div>
    <div class="phase-detail">
      WhamConfig + WhamModel<br>
      125M (328M params) + 500M (689M params) configs<br>
      8 model tests green
    </div>
  </div>
  <div class="timeline-phase phase-done" role="listitem" tabindex="0">
    <div class="phase-dot"></div>
    <div class="phase-label">Phase 2b<br>Training Loop</div>
    <div class="phase-badge"><span class="badge badge-done">Done</span></div>
    <div class="phase-detail">
      FSDP launcher + cosine LR + weighted CE<br>
      CPU smoke: loss step 0 = 12.72 (log(280k) baseline ✓)<br>
      25 tests green
    </div>
  </div>
  <div class="timeline-phase phase-next" role="listitem" tabindex="0">
    <div class="phase-dot"></div>
    <div class="phase-label">Phase 2c<br>Bulk Encode</div>
    <div class="phase-badge"><span class="badge badge-next">Next</span></div>
    <div class="phase-detail">
      Open-MAGVIT2 GPU runner ready<br>
      Needs SLURM run on betelgeuse<br>
      ~450,000 rbb frames to encode
    </div>
  </div>
  <div class="timeline-phase phase-block" role="listitem" tabindex="0">
    <div class="phase-dot"></div>
    <div class="phase-label">Phase 2d<br>Smoke Run</div>
    <div class="phase-badge"><span class="badge badge-blocked">Blocked</span></div>
    <div class="phase-detail">
      Blocked by: bulk encoding (2c) + dedicated reservation request<br>
      125M model, 1 GPU, ~10 shots, validate loss curve
    </div>
  </div>
  <div class="timeline-phase phase-block" role="listitem" tabindex="0">
    <div class="phase-dot"></div>
    <div class="phase-label">Phase 2e<br>500M Curriculum</div>
    <div class="phase-badge"><span class="badge badge-blocked">Blocked</span></div>
    <div class="phase-detail">
      Blocked by: Phase 2d green + reservation granted<br>
      4×H200 exclusive, 60K steps, bf16 FSDP ZeRO-3
    </div>
  </div>
  <div class="timeline-phase phase-block" role="listitem" tabindex="0">
    <div class="phase-dot"></div>
    <div class="phase-label">Phase 2f<br>Demo</div>
    <div class="phase-badge"><span class="badge badge-blocked">Blocked</span></div>
    <div class="phase-detail">
      Blocked by: trained checkpoint<br>
      Wide-angle camera forward-prediction on shots 30420/30421<br>
      rFID ≤ 8 target, side-by-side MP4
    </div>
  </div>
</div>
'''

    # Status table (from v0-runway)
    status_table_html = '''
<div class="filter-bar">
  <button class="filter-btn active" data-filter="all">All</button>
  <button class="filter-btn" data-filter="done">Done</button>
  <button class="filter-btn" data-filter="flight">In flight</button>
  <button class="filter-btn" data-filter="next">Next</button>
  <button class="filter-btn" data-filter="blocked">Blocked</button>
  <button class="filter-btn" data-filter="pending">Pending</button>
</div>
<table>
<thead>
  <tr>
    <th>Track</th>
    <th>Status</th>
    <th>Notes</th>
  </tr>
</thead>
<tbody>
  <tr class="status-row" data-status="done">
    <td>Level-2 mirror</td>
    <td><span class="badge badge-done">Done</span></td>
    <td>11,573 shots, 4.5 TB; bucket du authoritative</td>
  </tr>
  <tr class="status-row" data-status="done">
    <td>Level-1 camera mirror</td>
    <td><span class="badge badge-done">Done</span></td>
    <td>2.7 TB, 11,029 shot dirs</td>
  </tr>
  <tr class="status-row" data-status="flight">
    <td>Level-1 ALL-sources mirror</td>
    <td><span class="badge badge-flight">In flight</span></td>
    <td>manifest level1-all.json launched ~07:00 UTC</td>
  </tr>
  <tr class="status-row" data-status="done">
    <td>Tokenizer scaffold</td>
    <td><span class="badge badge-done">Done</span></td>
    <td>registry + alignment + multimodal aggregator + CLI</td>
  </tr>
  <tr class="status-row" data-status="done">
    <td>Open-MAGVIT2 frame tokenizer</td>
    <td><span class="badge badge-done">Done</span></td>
    <td>real model end-to-end; MAE 324 on rbb; CUDA wheel installed</td>
  </tr>
  <tr class="status-row" data-status="done">
    <td>Chronos signal tokenizer</td>
    <td><span class="badge badge-done">Done</span></td>
    <td>r ≈ 0.985 round-trip on synthetic sine/cosine</td>
  </tr>
  <tr class="status-row" data-status="done">
    <td>PatchTST identity wrapper</td>
    <td><span class="badge badge-done">Done</span></td>
    <td>bit-exact round-trip</td>
  </tr>
  <tr class="status-row" data-status="done">
    <td>Eval metrics module</td>
    <td><span class="badge badge-done">Done</span></td>
    <td>rFID, PSNR, LPIPS, centroid, chord, edge; 31 tests</td>
  </tr>
  <tr class="status-row" data-status="done">
    <td>WHAM model + 125M/500M configs</td>
    <td><span class="badge badge-done">Done</span></td>
    <td>8 tests; 125M=328M params (embedding-table dominated)</td>
  </tr>
  <tr class="status-row" data-status="done">
    <td>Data loaders + token persistence</td>
    <td><span class="badge badge-done">Done</span></td>
    <td>15 tests; ambix data tokens-status live</td>
  </tr>
  <tr class="status-row" data-status="done">
    <td>block_kind side data</td>
    <td><span class="badge badge-done">Done</span></td>
    <td>51 tokenizer tests; weighted-CE loss mask ready</td>
  </tr>
  <tr class="status-row" data-status="done">
    <td>CLI smoke tests</td>
    <td><span class="badge badge-done">Done</span></td>
    <td>42 tests; data/cli.py 86%, tokenizer/cli.py 97%</td>
  </tr>
  <tr class="status-row" data-status="done">
    <td>Data quality framework</td>
    <td><span class="badge badge-done">Done</span></td>
    <td>22 tests; 25-shot audit: 76% usable_for_training</td>
  </tr>
  <tr class="status-row" data-status="done">
    <td>Tokenizer benchmark framework</td>
    <td><span class="badge badge-done">Done</span></td>
    <td>14 tests; real bench on 8 rbb frames</td>
  </tr>
  <tr class="status-row" data-status="done">
    <td>Calibration library</td>
    <td><span class="badge badge-done">Done</span></td>
    <td>17 tests; real data: Ip mean 294 kA, density 4.6e19 m⁻³</td>
  </tr>
  <tr class="status-row" data-status="done">
    <td>Bulk-encode CLI</td>
    <td><span class="badge badge-done">Done</span></td>
    <td>15 tests; CPU smoke on shot 15085: 9,380 tokens</td>
  </tr>
  <tr class="status-row" data-status="done">
    <td>Training loop FSDP scaffold</td>
    <td><span class="badge badge-done">Done</span></td>
    <td>25 tests; CPU smoke loss=12.72 (log(280k) baseline)</td>
  </tr>
  <tr class="status-row" data-status="done">
    <td>Demo CLI + rollout impl</td>
    <td><span class="badge badge-done">Done</span></td>
    <td>20 tests + 31 eval tests; mock-checkpoint pipeline working</td>
  </tr>
  <tr class="status-row" data-status="next">
    <td>Bulk GPU encode (rbb corpus)</td>
    <td><span class="badge badge-next">Next</span></td>
    <td>Open-MAGVIT2 GPU runner ready; needs SLURM run on betelgeuse</td>
  </tr>
  <tr class="status-row" data-status="blocked">
    <td>Smoke training run (125M)</td>
    <td><span class="badge badge-blocked">Blocked</span></td>
    <td>Waiting on: bulk encode + SLURM reservation</td>
  </tr>
  <tr class="status-row" data-status="blocked">
    <td>500M curriculum run</td>
    <td><span class="badge badge-blocked">Blocked</span></td>
    <td>Waiting on: smoke run green + reservation granted</td>
  </tr>
  <tr class="status-row" data-status="blocked">
    <td>v0 Demo</td>
    <td><span class="badge badge-blocked">Blocked</span></td>
    <td>Waiting on: trained checkpoint</td>
  </tr>
  <tr class="status-row" data-status="pending">
    <td>Tokenizer expansion (§12)</td>
    <td><span class="badge badge-pending">Pending</span></td>
    <td>Plasma decoder fine-tune + PatchTST real + eq. 2-D — after bench baseline</td>
  </tr>
  <tr class="status-row" data-status="pending">
    <td>Mirror integrity verification</td>
    <td><span class="badge badge-pending">Pending</span></td>
    <td>After level-2 done; re-run s5cmd cp, check zero new objects</td>
  </tr>
  <tr class="status-row" data-status="pending">
    <td>SLURM dedicated reservation</td>
    <td><span class="badge badge-pending">Pending</span></td>
    <td>User action: file with SDCC ops (request body in compute.md §3.2)</td>
  </tr>
</tbody>
</table>
'''

    # Blocking tasks
    blocking_html = '''
<ul class="blocking-list">
  <li>
    <strong>Bulk GPU encode of rbb corpus</strong>
    <div class="waits-on">What it unblocks: smoke training run + benchmark baselines<br>
    Action: run <code>ambix data bulk-encode-frames</code> on betelgeuse with Open-MAGVIT2 GPU runner
    </div>
  </li>
  <li>
    <strong>SLURM dedicated reservation request (compute.md §3.2)</strong>
    <div class="waits-on">What it unblocks: 500M curriculum training run<br>
    Action: user files request with SDCC ops — request body ready in <a href="compute.html">compute.html</a> §3.2
    </div>
  </li>
  <li>
    <strong>125M smoke training run</strong>
    <div class="waits-on">Waiting on: bulk GPU encode (above)<br>
    What it unblocks: 500M curriculum + confidence in FSDP pipeline
    </div>
  </li>
  <li>
    <strong>500M curriculum training</strong>
    <div class="waits-on">Waiting on: smoke run green + reservation granted<br>
    What it unblocks: v0 demo
    </div>
  </li>
  <li>
    <strong>Demo shot selection finalisation</strong>
    <div class="waits-on">Waiting on: charge_exchange decision (see open decisions below)<br>
    Current candidates: 30420, 30421, one M6-era TBD
    </div>
  </li>
</ul>
'''

    # Open decisions
    decisions_html = '<div class="decision-section">\n<h2 id="open-decisions">Open Decisions</h2>\n'
    decisions_html += '<p>Each decision below surfaces a choice that can be made now. Fill in the form and click <em>Generate follow-on prompt</em> to produce a copy-pasteable instruction for the AI coordinator.</p>\n'
    for d in DECISIONS:
        decisions_html += build_decision_card(d)
    decisions_html += '</div>'

    # Plan links table
    plan_links_html = '''
<table>
<thead>
  <tr><th>Plan</th><th>Scope</th><th>Status</th></tr>
</thead>
<tbody>
  <tr><td><a href="STRATEGY.html">STRATEGY</a></td><td>Vision, roadmap, success criteria, risk register</td><td><span class="badge badge-flight">Active</span></td></tr>
  <tr><td><a href="v0-runway.html">v0 Runway</a></td><td>Active operational plan — next-step ranking, fleet-dispatch table</td><td><span class="badge badge-flight">Active</span></td></tr>
  <tr><td><a href="compute.html">Compute</a></td><td>SLURM patterns, FSDP, reservation request</td><td><span class="badge badge-flight">Active</span></td></tr>
  <tr><td><a href="data-acquisition.html">Data Acquisition</a></td><td>FAIR-MAST endpoints, probe, bulk-download protocol</td><td><span class="badge badge-done">Done</span></td></tr>
  <tr><td><a href="data-quality.html">Data Quality</a></td><td>Audit framework, FAIR-MAST format reality, training-grade gate</td><td><span class="badge badge-done">Done</span></td></tr>
  <tr><td><a href="tokenizers.html">Tokenizers</a></td><td>Multi-modal tokenizer design, expansion roadmap (§12)</td><td><span class="badge badge-flight">Active</span></td></tr>
  <tr><td><a href="tokenizer-benchmarks.html">Tokenizer Benchmarks</a></td><td>Closed-loop comparison framework, rFID acceptance gates</td><td><span class="badge badge-done">Done</span></td></tr>
  <tr><td><a href="world-model-v0.html">World Model v0</a></td><td>WHAM-style model, training recipe, rollout</td><td><span class="badge badge-flight">In progress</span></td></tr>
  <tr><td><a href="demo.html">Demo</a></td><td>Wide-angle viewing system forward-prediction demo</td><td><span class="badge badge-blocked">Blocked</span></td></tr>
  <tr><td><a href="decisions.html">All Decisions</a></td><td>Aggregated open-decision capture view</td>  <td>—</td></tr>
</tbody>
</table>
'''

    content = f'''
<div class="plan-hero">
  <h1>Ambix Plans — Coordination Hub</h1>
  <div class="page-meta">
    <span>imas-ambix / Fusion World Model</span>
    <span class="sep">·</span>
    <span>Last updated: 2026-05-20</span>
    <span class="sep">·</span>
    <a href="decisions.html">All open decisions →</a>
  </div>
</div>

<h2 id="quick-status">Quick Status</h2>
{status_grid}

<h2 id="timeline">Timeline</h2>
<p>Click any phase for details.</p>
{timeline_html}

<h2 id="plans">Plans</h2>
{plan_links_html}

<h2 id="current-status">Current Status Table</h2>
{status_table_html}

<h2 id="blocking-tasks">Blocking Tasks</h2>
{blocking_html}

{decisions_html}
'''

    sidebar_items = [
        ("#quick-status",  "Quick Status"),
        ("#timeline",      "Timeline"),
        ("#plans",         "Plans"),
        ("#current-status","Status Table"),
        ("#blocking-tasks","Blocking Tasks"),
        ("#open-decisions","Open Decisions"),
    ]
    sidebar = "<ul>" + "".join(
        f'<li><a href="{href}">{label}</a></li>'
        for href, label in sidebar_items
    ) + "</ul>"

    html = SHELL_TEMPLATE.format(
        title="Coordination Hub",
        assets="assets/",
        root="",
        sidebar=sidebar,
        breadcrumb="",
        content=content,
        plan_nav="",
    )
    out = OUT_DIR / "index.html"
    out.write_text(html, encoding="utf-8")
    print(f"  Wrote {out.name}")


def generate_decisions_page() -> None:
    content = '<h1 id="open-decisions">Open Decisions</h1>\n'
    content += '<div class="page-meta"><span>All open decision capture widgets in one place</span></div>\n'
    content += '<p>Each form generates a copy-pasteable follow-on prompt for the AI coordinator. Fill in your choice, optionally add notes, then copy the generated prompt and paste it into a new Claude Code conversation.</p>\n'
    content += '<h2 id="how-it-works">How it works</h2>\n'
    content += ('<ol>'
                '<li>Select an option from the radio group.</li>'
                '<li>Add any clarifying notes in the text area.</li>'
                '<li>Click <strong>Generate follow-on prompt</strong>.</li>'
                '<li>Click <strong>Copy</strong> and paste the result into the AI coordinator.</li>'
                '</ol>\n')

    sidebar_items = []
    for d in DECISIONS:
        content += build_decision_card(d)
        sidebar_items.append((f'#decision-{d["id"]}', d["title"][:55] + ("…" if len(d["title"]) > 55 else "")))

    sidebar = "<ul>" + "".join(
        f'<li><a href="{href}">{label}</a></li>'
        for href, label in sidebar_items
    ) + "</ul>"

    html = SHELL_TEMPLATE.format(
        title="All Decisions",
        assets="assets/",
        root="",
        sidebar=sidebar,
        breadcrumb='<div class="breadcrumb"><a href="index.html">Plans</a><span>›</span><span>All Decisions</span></div>',
        content=content,
        plan_nav="",
    )
    out = OUT_DIR / "decisions.html"
    out.write_text(html, encoding="utf-8")
    print(f"  Wrote {out.name}")


def main() -> None:
    print("Generating Ambix HTML plan site…")

    # Per-plan pages
    for stem, label in PLAN_ORDER:
        generate_plan_page(stem, label)

    # Index + decisions
    generate_index()
    generate_decisions_page()

    print(f"Done. Output: {OUT_DIR}")


if __name__ == "__main__":
    main()
