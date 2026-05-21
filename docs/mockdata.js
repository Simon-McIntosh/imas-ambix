// Mock corpus — real imas-ambix plan inventory + extended state schema.
//
// SCHEMA NOTES — proposed full plan-state shape (one JSON file per plan,
// at docs/state/<project>/<plan>.json):
//
//   {
//     status:   "active" | "pending" | "blocked" | "shipped" | "draft",
//     decisions: { <key>: { choice, rationale, when, by } },
//     notes:     [{ id, who, bot, when, body, quote? }],
//     followups: [
//       { id, written_by, written_at, title, body, recommends_skill,
//         touches, blocked_by?, est_turn, prompt,
//         resolved_at?, resolved_by?, outcome? }
//     ],
//     research:  [{ id, type, title, source, added_by, when, url }],
//     questions: [{ id, section, body, opened_by, opened_at, resolved_at? }],
//   }

window.MOCK = (function () {
  const today = "2026-05-21";

  // ─── Projects mounted on the docs-server (Across-projects view) ───────────
  const projects = [
    {
      project: "imas-ambix",
      path: "~/Code/imas-ambix/docs",
      published: "simon-mcintosh.github.io/imas-ambix/",
      owner: "smc",
      plans_count: 9, active: 3, blocked: 3, pending: 1, shipped: 2,
      last_modified: "2026-05-21",
      milestones: [
        { id: "M0", name: "Probe & mirror",      status: "shipped", pct: 100 },
        { id: "M1", name: "Tokenizer prototype", status: "active",  pct: 64  },
        { id: "M2", name: "World-model v0",      status: "pending", pct: 12  },
        { id: "M3", name: "Public demo",         status: "pending", pct: 0   },
      ],
      top: [
        { slug: "tokenizers",     title: "Multi-modal tokenizers", status: "active",  roi: "high", effort: "L", ms: "M1", impl: 0.78, blockers: 0, dec_open: 4 },
        { slug: "v0-runway",      title: "v0 runway plan",         status: "active",  roi: "high", effort: "M", ms: "M1", impl: 0.62, blockers: 1, dec_open: 1 },
        { slug: "compute",        title: "Compute (SLURM, FSDP)",  status: "blocked", roi: "high", effort: "S", ms: "M1", impl: 0.40, blockers: 1, dec_open: 1 },
      ],
      activity30: [0,0,0,1,2,1,0,2,3,1,1,0,0,1,2,2,3,1,1,0,2,4,3,2,1,1,2,3,4,3],
      tests_30d: { pass: 0.94, runs: 217 },
    },
    {
      project: "efitpp",
      path: "~/Code/efitpp/docs",
      published: "(private)",
      owner: "smc",
      plans_count: 22, active: 6, blocked: 4, pending: 5, shipped: 7,
      last_modified: "2026-05-20",
      milestones: [
        { id: "M0", name: "CMake reset",         status: "shipped", pct: 100 },
        { id: "M1", name: "Test-data overhaul",  status: "active",  pct: 60  },
        { id: "M2", name: "ITER-side runner",    status: "active",  pct: 40  },
        { id: "M3", name: "Release 5.0",         status: "pending", pct: 0   },
      ],
      top: [
        { slug: "test-data-strategy", title: "Test-data strategy", status: "active",  roi: "high", effort: "XL", ms: "M1", impl: 0.61, blockers: 0, dec_open: 2 },
        { slug: "ci-runner-setup",    title: "CI runner setup",    status: "blocked", roi: "high", effort: "M",  ms: "M2", impl: 0.30, blockers: 3, dec_open: 0 },
        { slug: "idamdiff-cmake",     title: "idamdiff CMake",     status: "active",  roi: "mid",  effort: "S",  ms: "M2", impl: 0.85, blockers: 0, dec_open: 0 },
      ],
      activity30: [1,2,1,0,1,1,2,2,1,3,2,1,1,2,3,2,1,0,0,1,2,2,3,2,1,2,3,1,2,2],
      tests_30d: { pass: 0.87, runs: 1142 },
    },
    {
      project: "imas-codex",
      path: "~/Code/imas-codex/docs",
      published: "(private)",
      owner: "smc",
      plans_count: 9, active: 2, blocked: 1, pending: 4, shipped: 2,
      last_modified: "2026-05-19",
      milestones: [
        { id: "M0", name: "Bootstrap",       status: "shipped", pct: 100 },
        { id: "M1", name: "Docs-server",     status: "shipped", pct: 100 },
        { id: "M2", name: "Tunnel CLI",      status: "active",  pct: 66  },
        { id: "M3", name: "Embed pipeline",  status: "pending", pct: 5   },
      ],
      top: [
        { slug: "tunnel-cli",   title: "imas-codex tunnel CLI", status: "active",  roi: "high", effort: "M", ms: "M2", impl: 0.66, blockers: 0, dec_open: 1 },
        { slug: "embed-bench",  title: "Embed benchmark grid",  status: "pending", roi: "mid",  effort: "M", ms: "M3", impl: 0.05, blockers: 0, dec_open: 3 },
        { slug: "neo4j-schema", title: "Neo4j graph schema",    status: "blocked", roi: "high", effort: "L", ms: "M3", impl: 0.12, blockers: 1, dec_open: 0 },
      ],
      activity30: [0,0,1,0,0,1,2,1,0,0,1,2,1,0,0,1,1,0,2,1,0,1,1,1,0,2,1,0,1,1],
      tests_30d: { pass: 0.99, runs: 88 },
    }
  ];

  // ─── imas-ambix plan inventory ────────────────────────────────────────────
  const inventory = [
    { slug: "STRATEGY",             title: "STRATEGY — Fusion World Model",       status: "active",  ms: "M0", roi: "high", effort: "S", tier: "opus",   impl: 0.85, last: "2026-05-19", owner: "smc", blockers: 0, dec_open: 0, comments: 0, sprint: null },
    { slug: "v0-runway",            title: "v0 runway plan",                      status: "active",  ms: "M1", roi: "high", effort: "M", tier: "opus",   impl: 0.62, last: "2026-05-20", owner: "smc", blockers: 1, dec_open: 1, comments: 3, sprint: "S2" },
    { slug: "compute",              title: "Compute (SLURM, FSDP, reservation)",  status: "blocked", ms: "M1", roi: "high", effort: "S", tier: "sonnet", impl: 0.40, last: "2026-05-20", owner: "smc", blockers: 1, dec_open: 1, comments: 2, sprint: "S2" },
    { slug: "data-acquisition",     title: "Data acquisition (FAIR-MAST mirror)", status: "shipped", ms: "M0", roi: "high", effort: "L", tier: "sonnet", impl: 1.00, last: "2026-05-19", owner: "smc", blockers: 0, dec_open: 0, comments: 0, sprint: "S1" },
    { slug: "data-quality",         title: "Data quality (audit + gates)",        status: "active",  ms: "M1", roi: "mid",  effort: "M", tier: "sonnet", impl: 0.90, last: "2026-05-20", owner: "smc", blockers: 0, dec_open: 1, comments: 1, sprint: "S2" },
    { slug: "tokenizers",           title: "Multi-modal tokenizers",              status: "active",  ms: "M1", roi: "high", effort: "L", tier: "opus",   impl: 0.78, last: "2026-05-20", owner: "smc", blockers: 0, dec_open: 4, comments: 6, sprint: "S2" },
    { slug: "tokenizer-benchmarks", title: "Tokenizer benchmark harness",         status: "active",  ms: "M1", roi: "mid",  effort: "M", tier: "sonnet", impl: 0.80, last: "2026-05-20", owner: "smc", blockers: 0, dec_open: 0, comments: 0, sprint: "S2" },
    { slug: "world-model-v0",       title: "World-model v0 (WHAM-style)",         status: "blocked", ms: "M2", roi: "high", effort: "L", tier: "opus",   impl: 0.18, last: "2026-05-21", owner: "smc", blockers: 2, dec_open: 1, comments: 4, sprint: "S2" },
    { slug: "demo",                 title: "Public demo (wide-angle camera)",     status: "blocked", ms: "M3", roi: "low",  effort: "M", tier: "sonnet", impl: 0.05, last: "2026-04-22", owner: "smc", blockers: 1, dec_open: 1, comments: 2, sprint: null },
  ];

  // ─── Multiple sprints ─────────────────────────────────────────────────────
  const sprints = [
    {
      id: "S0", status: "shipped",
      theme: "M0 · probe & mirror MAST corpus",
      starts: "2026-04-29", ends: "2026-05-12",
      items: ["data-acquisition"],
      summary: "Mirrored 11,573 shots (4.5 TB) from FAIR-MAST to /work/projects/imas_gpu/mast/. Phase 0 closed.",
    },
    {
      id: "S1", status: "shipped",
      theme: "Tokenizer scaffold + data-quality gate",
      starts: "2026-05-06", ends: "2026-05-13",
      items: ["data-quality", "tokenizer-benchmarks"],
      summary: "Tokenizer protocol surfaces in tree; data-quality audit landed with 22 tests / 76% usable-rate on 25-shot sample.",
    },
    {
      id: "S2", status: "active",
      theme: "M1 ramp · close tokenizer track, unblock world-model",
      starts: "2026-05-13", ends: "2026-05-27",
      items: [
        "tokenizers", "v0-runway", "compute", "data-quality",
        "tokenizer-benchmarks", "world-model-v0",
      ],
      summary: null,
    },
  ];

  const activeSprint = sprints.find(s => s.status === "active");

  // ─── Top blockers (project-wide) ──────────────────────────────────────────
  const blockers = [
    { summary: "SDCC dedicated training reservation pending file",     origin: "compute",        n: 3, owner: "L. Holzl",  next: "ITER batch-services ticket #4421" },
    { summary: "Bulk-encode SLURM run not started",                     origin: "tokenizers",     n: 2, owner: "smc",       next: "queue once compute reservation lands" },
    { summary: "Trained checkpoint absent",                             origin: "demo",           n: 1, owner: "smc",       next: "after world-model-v0 first eval" },
  ];

  // ─── Project-wide activity ledger ─────────────────────────────────────────
  const timeline = [
    { when: "2026-05-21 09:14", who: "agent/sonnet",  what: "tokenizers §13.3 IR codebook benchmark wired (commit dec0082)" },
    { when: "2026-05-20 22:00", who: "smc",            what: "locked 4 tokenizers §12 decisions on tokenizers-12-landed phase record" },
    { when: "2026-05-20 18:02", who: "smc",            what: "v0-runway: added §3 fleet-dispatch block; 4 Sonnet agents launched" },
    { when: "2026-05-20 17:55", who: "agent/opus",     what: "tokenizers §12 expansion plan drafted (6 sub-items)" },
    { when: "2026-05-20 09:10", who: "smc",            what: "compute §2 lock: exclusive-access training on gpu_0003_grpA" },
    { when: "2026-05-19 10:30", who: "agent/sonnet",   what: "data-acquisition L2 mirror complete (11,573 shots · 4.5 TB)" },
  ];

  // ─── Per-plan content — real ambix plans, abbreviated bodies ──────────────
  // Each entry is what would live at docs/<slug>.html (body) + state/<slug>.json (meta).
  // plan.html?slug=<slug> dispatches into here.
  const plans = {
    "STRATEGY": {
      slug: "STRATEGY",
      title: "STRATEGY — Fusion World Model",
      ms: "M0", sprint: null, status: "active", roi: "high", effort: "S", tier: "opus", owner: "smc",
      impl: 0.85, phase: "evergreen vision doc",
      created: "2026-05-19", last_modified: "2026-05-19 12:00",
      depends_on: [], blocks: ["v0-runway"],
      summary: "Vision, partner-facility roadmap, why MAST first, v0/v1 success criteria, risk register.",
      sections: [
        { id: "s1", sec: "§ 1", h: "Executive vision", body: `Ambix builds the generative engine of the Fusion World Model. Where imas-codex answers "what does this signal mean", ambix answers "what happens next" across the joint distribution over plasma state, controls, and diagnostics. The headline ask: produce a transformer that, given the first N tokens of a discharge, generates the rest with measurable physical plausibility on held-out shots.` },
        { id: "s2", sec: "§ 2", h: "Why MAST first", body: `The 26B state-transition corpus described in the GPU-procurement document is the long-term training target — JET/TCV/JT-60SA/AUG/DIII-D. MAST is the entry point: FAIR-MAST publishes its own IMAS mapping, the dataset is small enough to mirror cheaply (4.5 TB), and the diagnostic coverage is rich enough to exercise every modality the v1 corpus will need.` },
        { id: "s3", sec: "§ 3", h: "Roadmap", body: `Phase 0 — probe & mirror (done). Phase 1 — tokenizer prototype (in flight). Phase 2 — world-model v0 (~500M decoder-only). Phase 3 — partner facilities. Phase 4 — operator-loop integration (deferred).` },
        { id: "s6", sec: "§ 6", h: "Risk register", body: `Top three risks: (1) tokenizer compression collapses high-frequency physics into a single token, (2) compute reservation slips past the M1 close window, (3) world-model overfits the small MAST corpus before partner-facility data lands. Mitigations tracked on the owning plans.` },
      ],
      decisions: [], notes: [], research: [], questions: [],
      followups: [], followups_done: [],
    },
    "v0-runway": {
      slug: "v0-runway",
      title: "v0 runway plan",
      ms: "M1", sprint: "S2", status: "active", roi: "high", effort: "M", tier: "opus", owner: "smc",
      impl: 0.62, phase: "active operational plan",
      created: "2026-05-13", last_modified: "2026-05-20 18:02",
      depends_on: ["STRATEGY", "tokenizers", "compute"], blocks: ["world-model-v0"],
      summary: "Next-step ranking by ROI · fleet-dispatch table · review rubric.",
      sections: [
        { id: "s1", sec: "§ 1", h: "Where we are — 2026-05-20", body: `L2 mirror complete (11,573 shots, 4.5 TB). L1 camera mirror complete (11,029 shot dirs, 2.7 TB). L1 all-sources mirror in flight (launched ~07:00 UTC). Tokenizer scaffold: registry, Open-MAGVIT2, Chronos, PatchTST passthrough — all wired. Bench framework + training loop + demo CLI all green.` },
        { id: "s2", sec: "§ 2", h: "Highest-ROI next steps (ranked)", body: `ROI = value × probability of clean delivery / cost. Top of the queue: bulk GPU encode (Open-MAGVIT2 runner ready; needs SLURM), then 125M smoke training run, then 500M curriculum, then demo. Items marked DONE retained for context.` },
        { id: "s3", sec: "§ 3", h: "Fleet dispatch — parallel batch", body: `Four Sonnet 4.6 agents in parallel with non-overlapping write scopes. Coordinator (Opus) reviews each post-completion. Each agent gets a scoped prompt naming the plan section, the read-state, and the write-target. Every dispatched agent commits its own changes with a feat(<scope>): conventional commit.` },
        { id: "s5", sec: "§ 5", h: "Review rubric", body: `Every dispatched agent's output is reviewed against: (1) does it land what the plan promised, (2) are tests green, (3) is the commit message scoped right, (4) did it update the plan's followup chain. Failures are captured in a new feedback memory.` },
      ],
      decisions: [
        {
          key: "smoke-training-timing",
          title: "125M smoke training run — run now or wait for bulk encode?",
          context: "Current state: training loop tested on tiny shards. Bulk encode is the prerequisite for real-sized training. Option A: kick a small smoke run on tokenized 5-shot subset to validate loss + checkpoint shape. Option B: wait for bulk encode so we don't double-spend GPU time. Cost of A: ~30 min on 1 H200. Cost of B: 0 but delays smoke validation.",
          choices: ["Run smoke now on subset", "Wait for bulk encode"],
          chosen: "", rationale: "", when: "",
        },
      ],
      notes: [
        { id: "n1", who: "smc", bot: false, when: "2026-05-20 18:02", body: "Adding §3 fleet-dispatch as a sub-plan; agent boundary table makes the parallelism real." },
      ],
      research: [
        { id: "r1", type: "plan", title: "tokenizers.html — current state", source: "internal", added_by: "smc", when: "2026-05-20", url: "plan.html?slug=tokenizers" },
        { id: "r2", type: "plan", title: "compute.html — blockers", source: "internal", added_by: "smc", when: "2026-05-20", url: "plan.html?slug=compute" },
      ],
      questions: [],
      followups: [
        {
          id: "f-v0-2026-05-21",
          written_by: "smc", written_at: "2026-05-20 18:02",
          title: "Coordinator review of the 4-agent parallel batch outputs",
          body: "All 4 Sonnet agents have committed their work. Review each output against the §5 rubric and either accept the commits or flag for rework. Update §2 ROI ranking with the new items the batch produced.",
          recommends_skill: "/plan-edit v0-runway --section 5",
          touches: ["docs/v0-runway.html", "imas_ambix/*/*.py (review only)"],
          est_turn: "single Opus turn · ~1h",
          prompt: `Project: imas-ambix
Plan: v0-runway

Coordinator review of the 4-agent parallel batch (see §3 fleet-dispatch).
For each of the 4 commits, score against §5 review rubric:
  (1) lands what plan promised
  (2) tests green
  (3) commit message scoped right
  (4) followup chain updated
Output a §3.1 summary table; flag any rework as new followups on the owning plan.`,
        },
      ],
      followups_done: [
        { id: "f-v0-2026-05-19", written_by: "smc", written_at: "2026-05-19 10:00", resolved_at: "2026-05-20 18:02", resolved_by: "smc",
          title: "Add §3 fleet-dispatch table",
          outcome: "§3 added with 4-agent dispatch table; agent prompts captured under §3.1–§3.4." },
      ],
    },
    "compute": {
      slug: "compute",
      title: "Compute — SLURM, FSDP, reservation",
      ms: "M1", sprint: "S2", status: "blocked", roi: "high", effort: "S", tier: "sonnet", owner: "smc",
      impl: 0.40, phase: "blocked on SDCC reservation file",
      created: "2026-05-13", last_modified: "2026-05-20 09:10",
      depends_on: [], blocks: ["tokenizers", "world-model-v0", "demo"],
      summary: "SLURM patterns for training on the 4 × H200 Group A reservation; dedicated training reservation request; scheduled-around-serving fallback.",
      sections: [
        { id: "s1", sec: "§ 1", h: "Hardware budget recap", body: `The Group A reservation on 98dci4-gpu-0003 (betelgeuse SLURM partition) grants 4 × H200 nodes shared with the V4-Flash production serve. Training has to either preempt the serve (exclusive-access window) or schedule around its idle slots.` },
        { id: "s2", sec: "§ 2", h: "Training takes exclusive access", body: `Decision 2026-05-20 (S. McIntosh): training runs in exclusive-access mode on the existing gpu_0003_grpA reservation. The maintainer of V4-Flash stops the serve before each training run and restarts after. Locks the serve operator into the loop but avoids a second reservation file.` },
        { id: "s3", sec: "§ 3", h: "Dedicated training-reservation request", body: `Proposed name: gpu_0003_grpA_train — same node, same group, separate reservation slot that can be drained without affecting the production serve. Filed as ITER batch-services ticket #4421; pending L. Holzl signoff.` },
        { id: "s8", sec: "§ 8", h: "Risks", body: `Top risk: ticket #4421 slips past the M1 close window (2026-05-27). Mitigation: continue exclusive-access fallback. Second risk: FSDP across nodes hits the betelgeuse interconnect ceiling at 4 nodes; we expect ~70% scaling — adequate for 500M but tight for v1.` },
      ],
      decisions: [
        {
          key: "slurm-training-reservation",
          title: "SLURM dedicated training reservation — file with SDCC or stay with exclusive-pause?",
          context: "Currently exclusive-pause works but couples to V4-Flash operator. Filing a dedicated training reservation (#4421) decouples but adds queue + paperwork.",
          choices: ["File dedicated reservation #4421", "Stay with exclusive-pause", "Both — file but use exclusive in the meantime"],
          chosen: "Both — file but use exclusive in the meantime",
          rationale: "Filed 2026-05-20; using exclusive while ticket is in queue. Will switch when SDCC responds.",
          when: "2026-05-20 09:10",
        },
      ],
      notes: [
        { id: "n1", who: "smc", bot: false, when: "2026-05-20 09:10", body: "Confirmed with L. Holzl that exclusive-pause window is fine for the smoke run; dedicated reservation only matters when curriculum training starts." },
      ],
      research: [
        { id: "r1", type: "doc",  title: "SDCC betelgeuse reservation policy v3.1", source: "ITER SDCC docs", added_by: "smc", when: "2026-05-13", url: "#" },
        { id: "r2", type: "web",  title: "FSDP scaling notes (HF accelerate)",       source: "huggingface.co", added_by: "agent/sonnet", when: "2026-05-15", url: "#" },
      ],
      questions: [],
      followups: [
        {
          id: "f-compute-2026-05-21",
          written_by: "smc", written_at: "2026-05-20 09:10",
          title: "Stand up exclusive-pause SLURM wrapper script",
          body: "Until SDCC ticket #4421 clears, training runs need a wrapper that stops V4-Flash, claims the node exclusively, runs the job, restarts the serve. Should be a single sbatch template + a 30-line bash prologue.",
          recommends_skill: "/plan-implement compute --section 2",
          touches: ["scripts/slurm/train_exclusive.sbatch", "scripts/slurm/preempt_v4flash.sh"],
          est_turn: "single Sonnet turn · ~2h",
          prompt: "Project: imas-ambix\nPlan: compute\n\nWrite the exclusive-pause SLURM wrapper described in §2. Validate by dry-running the prologue (no actual stop of V4-Flash). When done, document the wrapper inline at §2 and write a followup with the dry-run log.",
        },
      ],
      followups_done: [],
    },
    "data-acquisition": {
      slug: "data-acquisition",
      title: "Data acquisition — FAIR-MAST mirror",
      ms: "M0", sprint: "S1", status: "shipped", roi: "high", effort: "L", tier: "sonnet", owner: "smc",
      impl: 1.00, phase: "shipped",
      created: "2026-05-06", last_modified: "2026-05-19 10:30",
      depends_on: [], blocks: ["tokenizers"],
      summary: "FAIR-MAST endpoint inventory, sizing-probe protocol, bulk-download SLURM spec.",
      sections: [
        { id: "s1", sec: "§ 1", h: "FAIR-MAST endpoint inventory", body: `Eight L2 endpoints across magnetics, cameras, equilibrium, summary, pulse_schedule, gas_injection, thomson_scattering, and pf_active. Each has a published IMAS mapping; bulk download via s5cmd cp.` },
        { id: "s10", sec: "§ 10", h: "Mirror status — complete 2026-05-19", body: `Final mirror: 11,573 shots, 4.5 TB landed at /work/projects/imas_gpu/mast/. L1 camera mirror at 11,029 shot dirs / 2.7 TB. L1 all-sources mirror at 83% as of 2026-05-20 (background completion).` },
      ],
      decisions: [],
      notes: [],
      research: [
        { id: "r1", type: "doc", title: "FAIR-MAST API v3.2", source: "fair-mast.org", added_by: "smc", when: "2026-05-06", url: "#" },
      ],
      questions: [],
      followups: [],
      followups_done: [
        { id: "f-da-shipped", written_by: "agent/sonnet", written_at: "2026-05-19 10:30", resolved_at: "2026-05-19 10:30", resolved_by: "agent/sonnet",
          title: "Final L2 mirror sync + checksum verification",
          outcome: "11,573/11,573 shots present; all SHA-256 checksums match published manifest. Plan moved to shipped." },
      ],
    },
    "data-quality": {
      slug: "data-quality",
      title: "Data quality — audit + training-grade gate",
      ms: "M1", sprint: "S2", status: "active", roi: "mid", effort: "M", tier: "sonnet", owner: "smc",
      impl: 0.90, phase: "in flight · gate landing 2026-05-21",
      created: "2026-05-13", last_modified: "2026-05-20 11:00",
      depends_on: ["data-acquisition"], blocks: ["world-model-v0"],
      summary: "Per-shot ShotQualityReport, corpus audit, ambix data audit CLI, acceptance gates for training-grade shots.",
      sections: [
        { id: "s1", sec: "§ 1", h: "Audit framework", body: `Twenty-two pytest tests on a 25-shot sample give 76% usable rate. Failures: charge_exchange ~50% bit-pattern corruption; thomson_scattering ~12% timestamp drift; magnetics ~3% NaN sections. Tracked per-shot in imas_ambix.quality.ShotQualityReport.` },
        { id: "s5", sec: "§ 5", h: "Training-grade gate", body: `A shot is training-grade if (a) magnetics complete, (b) ≥1 camera channel complete, (c) equilibrium time-slice present, (d) no NaN in summary scalars. Drop charge_exchange entirely from the v0 corpus (decision 2026-05-20).` },
      ],
      decisions: [
        {
          key: "drop-charge-exchange",
          title: "Drop charge_exchange from v0 training corpus?",
          context: "Audit found ~50% bit-pattern corruption in t_i and v_i columns (values 10^26 – 10^38 K / m/s; physical ranges ≤ 30 keV / 10^7 m/s). These are float-encoding defects in the FAIR-MAST CX ingestion.",
          choices: ["Yes — drop entirely", "No — keep with corruption filter", "Defer"],
          chosen: "Yes — drop entirely",
          rationale: "12–28 orders of magnitude beyond physical range; filter would mask the underlying ingestion bug. Re-include in v1 after upstream fix.",
          when: "2026-05-20 11:00",
        },
      ],
      notes: [
        { id: "n1", who: "smc", bot: false, when: "2026-05-20 11:00", body: "Re-deriving training-grade-shots.json without CX shrinks the corpus by ~8% — acceptable." },
      ],
      research: [],
      questions: [
        { id: "q1", section: "5 training-grade gate", body: "Should we re-include charge_exchange in v1 once FAIR-MAST fixes the encoding, or treat it as a learned upstream untrustable?", opened_by: "smc", opened_at: "2026-05-20" },
      ],
      followups: [
        {
          id: "f-dq-2026-05-21",
          written_by: "agent/sonnet", written_at: "2026-05-20 14:00",
          title: "Re-derive training-grade-shots.json excluding CX",
          body: "drop-charge-exchange decision is locked. Regenerate the training-grade manifest, update §5 with the new shot count, and confirm world-model-v0 read path picks up the new file.",
          recommends_skill: "/plan-implement data-quality --section 5",
          touches: ["imas_ambix/quality/manifest.py", "training-grade-shots.json"],
          est_turn: "single Sonnet turn · ~30 min",
          prompt: `Project: imas-ambix\nPlan: data-quality\n\nDecision drop-charge-exchange is locked → yes. Re-derive training-grade-shots.json.\n\nWhen done, update §5 with the new shot count and post a followup confirming world-model-v0 reads from the regenerated file.`,
        },
      ],
      followups_done: [],
    },
    "tokenizer-benchmarks": {
      slug: "tokenizer-benchmarks",
      title: "Tokenizer benchmark harness",
      ms: "M1", sprint: "S2", status: "active", roi: "mid", effort: "M", tier: "sonnet", owner: "smc",
      impl: 0.80, phase: "framework landed · awaiting real bench runs",
      created: "2026-05-13", last_modified: "2026-05-20 12:00",
      depends_on: ["tokenizers"], blocks: [],
      summary: "Quantitative tokenizer comparison framework — BenchConfig, BenchResult, frame metrics (rFID/PSNR/LPIPS), signal metrics (Pearson r/NRMSE), throughput, SLURM batch runner.",
      sections: [
        { id: "s1", sec: "§ 1", h: "BenchConfig", body: `One YAML per benchmark run: tokenizer choice, shot ids, metric set, batch size, output dir. The CLI ambix bench run reads it and writes BenchResult JSON. Composable: same config works on visible, IR, magnetics with module swap.` },
        { id: "s2", sec: "§ 2", h: "Frame metrics — rFID / PSNR / LPIPS", body: `rFID computed against the v0 plasma feature extractor (placeholder until §12.1 lands). PSNR + LPIPS as fallback. Acceptance gate: rFID < 5 on 100-shot rbb benchmark for plasma-decoder fine-tune trigger.` },
        { id: "s3", sec: "§ 3", h: "Signal metrics — Pearson r / NRMSE", body: `Per-channel reconstruction quality. Magnetics target Pearson r ≥ 0.95; low-freq signals target NRMSE ≤ 0.02. Below these gates, fall back to identity passthrough.` },
      ],
      decisions: [],
      notes: [],
      research: [
        { id: "r1", type: "paper", title: "rFID — image reconstruction quality metric", source: "arXiv 2104.06692", added_by: "agent/sonnet", when: "2026-05-15", url: "#" },
      ],
      questions: [],
      followups: [
        {
          id: "f-bench-2026-05-21",
          written_by: "agent/sonnet", written_at: "2026-05-20 12:00",
          title: "Run real bench on 100-shot rbb sample",
          body: "Framework is wired. Need a real run to set the rFID baseline that the tokenizers §12.1 plasma-decoder fine-tune decision depends on. Blocked on compute reservation.",
          recommends_skill: "/plan-implement tokenizer-benchmarks --section 2",
          touches: ["scripts/slurm/bench_rbb.sbatch"],
          blocked_by: { slug: "compute", reason: "reservation gating GPU access" },
          est_turn: "single Sonnet turn once compute clears · ~3h SLURM",
          prompt: "Project: imas-ambix\nPlan: tokenizer-benchmarks\n\nRun the 100-shot rbb benchmark. Use the YAML at configs/bench/rbb-100.yaml. When done, write the rFID number into §2 and post a followup linking it to tokenizers §12.1 plasma-decoder fine-tune decision gate.",
        },
      ],
      followups_done: [],
    },
    "world-model-v0": {
      slug: "world-model-v0",
      title: "World-model v0 — WHAM-style decoder",
      ms: "M2", sprint: "S2", status: "blocked", roi: "high", effort: "L", tier: "opus", owner: "smc",
      impl: 0.18, phase: "spec landed · blocked on tokenizer round-trip",
      created: "2026-05-13", last_modified: "2026-05-21 09:00",
      depends_on: ["tokenizers", "compute", "data-quality"], blocks: ["demo"],
      summary: "~500M decoder-only Llama-class AR transformer over interleaved token streams; 125M → 500M curriculum; HF transformers + accelerate FSDP.",
      sections: [
        { id: "s1", sec: "§ 1", h: "Model architecture", body: `Llama-class decoder-only, ~500M parameters at the headline size. 28 layers, d_model=1280, 20 attention heads, RoPE, SwiGLU. We use the HuggingFace LlamaForCausalLM with custom embedding init (block-aware) and the WHAM block-weighted CE loss. Speculative or parallel decoding is a v1 question.` },
        { id: "s2", sec: "§ 2", h: "Token-stream layout", body: `One long interleaved sequence per training sample. Per-timestep schema: <pre>[t] [magnetics block] [low-freq signal block] [visible frame patch* k] [IR frame patch* m] [equil cross-attn key] [scalar actions]</pre> Block boundaries are recorded in BLOCKWEIGHTS so the loader can apply weighted CE.` },
        { id: "s4", sec: "§ 4", h: "Training recipe", body: `Optimizer AdamW(0.95, 0.95, 1e-8), weight decay 0.1, cosine LR 6e-4 → 6e-5 over 30k steps. Warmup 500. FSDP across 4 H200; batch size 32 sequences of 16k tokens. Eval every 5k steps on a held-out 1% slice.` },
        { id: "s5", sec: "§ 5", h: "Evaluation", body: `Headline demo per demo.html. Training-time hooks: token perplexity per block, magnetics Pearson r on next-step prediction, visible frame rFID on next-frame prediction. All three must be improving by step 10k to continue.` },
      ],
      decisions: [
        {
          key: "model-size-curriculum",
          title: "Start at 125M or jump to 500M direct?",
          context: "Curriculum approach: train 125M to ~10k steps to validate loss curve + data pipeline, then scale to 500M. Direct approach: skip 125M, save 4-6 GPU-hours but risk burning a longer 500M run on a data bug.",
          choices: ["Curriculum — 125M smoke → 500M", "Direct — 500M only"],
          chosen: "", rationale: "", when: "",
        },
      ],
      notes: [
        { id: "n1", who: "smc", bot: false, when: "2026-05-20 11:00", body: "Architecturally we're a thin layer over HF LlamaForCausalLM; the interesting bits are the block-aware embedding init and the weighted CE loss." },
      ],
      research: [
        { id: "r1", type: "paper", title: "WHAM — World and Human Action Models", source: "Nature 2025 · doi:10.1038/s41586-025-08600-3", added_by: "smc", when: "2026-05-13", url: "https://doi.org/10.1038/s41586-025-08600-3" },
        { id: "r2", type: "paper", title: "Llama 3 architecture report", source: "arXiv 2407.21783", added_by: "agent/opus", when: "2026-05-15", url: "#" },
      ],
      questions: [
        { id: "q1", section: "4 training recipe", body: "Should we use HuggingFace's accelerate FSDP wrapper or roll our own with torch.distributed.fsdp? The wrapper is simpler but pins us to the HF training loop semantics.", opened_by: "agent/opus", opened_at: "2026-05-20" },
      ],
      followups: [
        {
          id: "f-wm-2026-05-21",
          written_by: "smc", written_at: "2026-05-21 09:00",
          title: "Lock model-size-curriculum decision",
          body: "tokenizers §12 expansion is closing this week. Before the smoke run can start, we need to commit to either 125M-smoke-first or 500M-direct. Read the decision context and lock it (rationale required).",
          recommends_skill: "/plan-edit world-model-v0 --decision model-size-curriculum",
          touches: ["docs/world-model-v0.html"],
          blocked_by: { slug: "tokenizers", reason: "§12 expansion still landing" },
          est_turn: "smc desk-decision · 15min",
          prompt: "Lock decision model-size-curriculum on plan world-model-v0. Read §1, §4, and compute reservation status; choose curriculum vs direct.",
        },
      ],
      followups_done: [],
    },
    "demo": {
      slug: "demo",
      title: "Public demo — wide-angle camera forward-prediction",
      ms: "M3", sprint: null, status: "blocked", roi: "low", effort: "M", tier: "sonnet", owner: "smc",
      impl: 0.05, phase: "blocked · waiting on world-model checkpoint",
      created: "2026-05-13", last_modified: "2026-04-22 14:00",
      depends_on: ["world-model-v0"], blocks: [],
      summary: "Wide-angle viewing system forward-prediction demo on three pinned held-out MAST shots; rFID + centroid MSE acceptance.",
      sections: [
        { id: "s1", sec: "§ 1", h: "Three pinned held-out shots", body: `Three shots withheld from training; demo predicts frames N+1..N+30 given first N. Acceptance: rFID < 8 on each held-out shot and centroid trajectory MSE within 2 pixels.` },
        { id: "s2", sec: "§ 2", h: "Public-facing UI", body: `Single-page HTML rendering the predicted vs ground-truth frames side-by-side as an animation. Hosted at simon-mcintosh.github.io/imas-ambix/demo.html alongside this plan.` },
      ],
      decisions: [
        {
          key: "held-out-shot-selection",
          title: "Pinned held-out shot ids — auto-pick or hand-pick three?",
          context: "Hand-pick: choose three shots covering different plasma regimes (ohmic, NBI-heated, disrupted). Auto-pick: random 3 from training-grade pool. Hand-pick is more demonstrative; auto-pick is more honest.",
          choices: ["Hand-pick three regimes", "Auto-pick random three"],
          chosen: "", rationale: "", when: "",
        },
      ],
      notes: [],
      research: [],
      questions: [],
      followups: [],
      followups_done: [],
    },

    // ─── tokenizers: the canonical full-content plan ──────────────────────
    "tokenizers": {
      slug: "tokenizers",
      title: "Multi-modal tokenizers",
      ms: "M1", sprint: "S2", status: "active", roi: "high", effort: "L", tier: "opus", owner: "smc",
      impl: 0.78, phase: "landed §1–§13 · §12 expansion in flight",
      created: "2026-05-13", last_modified: "2026-05-21 09:14",
      depends_on: ["data-acquisition", "compute"], blocks: ["world-model-v0"],
      summary: "Per-modality tokenizer choice, codebook layout, token-id namespacing, persistence format — the interface world-model-v0 consumes.",
      // The tokenizers plan has so much content the plan.html renders it directly
      // (via the legacy planTokenizers export). The dispatcher here just provides
      // the meta so other pages can resolve the slug.
      isHeroPlan: true,
      sections: [],
      decisions: [
        { key: "plasma-decoder-finetune",  title: "Open-MAGVIT2 plasma-domain decoder fine-tune — when to trigger?", choices: ["Fine-tune now", "Fine-tune only if rFID > 5", "Defer to v1"], chosen: "", rationale: "", when: "" },
        { key: "patchtst-real-embedding",  title: "PatchTST real embedding — defer to v1 vs land in v0?",            choices: ["Defer to v1", "Land real PatchTST in v0"],                                chosen: "", rationale: "", when: "" },
        { key: "equilibrium-2d-tokenizer", title: "Equilibrium 2-D tokenizer — which architecture?",                  choices: ["Keep as cross-attention", "Open-MAGVIT2 upsampled 256×256", "Cosmos-Tokenizer-DV 65×65"], chosen: "", rationale: "", when: "" },
        { key: "ir-camera-codebook",       title: "IR camera codebook — share with visible or allocate separate?",    choices: ["Share visible codebook (v0 default)", "Allocate separate registry block"], chosen: "", rationale: "", when: "" },
      ],
      notes: [], research: [], questions: [],
      followups: [], followups_done: [],
    },
  };

  // ─── Focused tokenizers plan (kept as the dedicated, full-content plan) ───
  const planTokenizers = {
    slug: "tokenizers",
    title: "Multi-modal tokenizers",
    project: "imas-ambix",
    ms: "M1",
    sprint: "S2",
    status: "active",
    roi: "high",
    effort: "L",
    tier: "opus",
    owner: "smc",
    impl: 0.78,
    phase: "landed §1–§13 · §12 expansion in flight",
    last_modified: "2026-05-21 09:14",
    created: "2026-05-13",
    depends_on: ["data-acquisition", "compute"],
    blocks:     ["world-model-v0"],

    followups: [
      {
        id: "f-2026-05-21a",
        written_by: "agent/sonnet",
        written_at: "2026-05-21 09:14",
        title: "Run bulk-encode on rbb + magnetics shards (§12.5 follow-on)",
        body: "§13.3 landed the IR codebook benchmark wiring. To close §12.5 and unblock world-model-v0, we need the real bulk-encode pass across rbb (visible) + magnetics shards.",
        recommends_skill: "/plan-implement tokenizers --section 12.5",
        touches: ["imas_ambix/tokenizer/bulk_encode.py", "scripts/slurm/bench_ir.sbatch"],
        blocked_by: { slug: "compute", reason: "SDCC training reservation pending — ticket #4421" },
        est_turn: "single Opus turn once compute clears · ~4h SLURM",
        prompt: `Project: imas-ambix
Plan: tokenizers (https://simon-mcintosh.github.io/imas-ambix/tokenizers.html)

Run the bulk-encode pass that closes §12.5 (IR camera codebook decision).

State to read:
  docs/state/imas-ambix/tokenizers.json
  docs/state/imas-ambix/compute.json   (confirm reservation cleared)

Locked decisions to honour:
  primary-approach     → A (Open-MAGVIT2 visible codebook)
  vocab-cap            → 32k

Open decisions to surface, not resolve:
  plasma-decoder-finetune, ir-camera-codebook

When done:
  1. Write the bench numbers into tokenizers.html §12.5
  2. POST a new followup to tokenizers.json#followups
  3. Mark this followup resolved`,
      },
    ],

    followups_done: [
      { id: "f-2026-05-20a", written_by: "smc",            written_at: "2026-05-20 18:02", resolved_at: "2026-05-21 09:14", resolved_by: "agent/sonnet",
        title: "Wire IR codebook benchmark for §12.5",
        outcome: "Bench config landed (commit dec0082); ready to run pending GPU + real rir shot ids." },
      { id: "f-2026-05-20b", written_by: "agent/opus",     written_at: "2026-05-20 11:00", resolved_at: "2026-05-20 17:55", resolved_by: "agent/opus",
        title: "Draft §12 expansion plan (6 sub-items)",
        outcome: "All 6 §12 sub-items written and ranked. 4 decisions landed on tokenizers-12-landed.html." },
      { id: "f-2026-05-19a", written_by: "smc",            written_at: "2026-05-19 14:00", resolved_at: "2026-05-20 11:00", resolved_by: "agent/opus",
        title: "Land §9 v0 scaffold for tokenizer package",
        outcome: "imas_ambix.tokenizer/ exists with protocol surfaces + Open-MAGVIT2 / Chronos / PatchTST passthrough impls." },
    ],

    decisions: [
      { key: "plasma-decoder-finetune",
        title: "Open-MAGVIT2 plasma-domain decoder fine-tune — when to trigger?",
        context: "Current ImageNet-pretrained decoder produces MAE 324 (uint16) on MAST rbb shot 15085. Expected rFID without fine-tune: 10–30. Decoder fine-tune cost: ~4–6 GPU-hours exclusive. Trigger gate (from plan): rFID > 5 on the 100-shot rbb benchmark.",
        choices: ["Fine-tune now", "Fine-tune only if rFID > 5", "Defer to v1"],
        chosen: "", rationale: "", when: "" },
      { key: "patchtst-real-embedding",
        title: "PatchTST real embedding — defer to v1 vs land in v0?",
        context: "Current state: identity passthrough (token ID 0, raw floats in metadata). Real PatchTST adds ~1M trainable params with channel-independent self-attention. Estimated effort: 1 Sonnet session.",
        choices: ["Defer to v1", "Land real PatchTST in v0"],
        chosen: "", rationale: "", when: "" },
      { key: "equilibrium-2d-tokenizer",
        title: "Equilibrium 2-D tokenizer — which architecture?",
        context: "Currently: equilibrium 2-D enters as a continuous cross-attention tensor (not tokenized). Option A — Open-MAGVIT2 at upsampled 256×256. Option B — Cosmos-Tokenizer-DV at native 65×65 (license check needed).",
        choices: ["Keep as cross-attention", "Open-MAGVIT2 upsampled 256×256", "Cosmos-Tokenizer-DV 65×65"],
        chosen: "", rationale: "", when: "" },
      { key: "ir-camera-codebook",
        title: "IR camera codebook — share with visible or allocate separate?",
        context: "Only 25 rir shots in FAIR-MAST. Current v0 default: share the Open-MAGVIT2 visible codebook. Decision gate: if IR MAE > 2× rbb MAE, allocate separate block.",
        choices: ["Share visible codebook (v0 default)", "Allocate separate registry block"],
        chosen: "", rationale: "", when: "" },
    ],

    notes: [
      { id: "n1", who: "smc",          bot: false, when: "2026-05-20 22:00", body: "Locked the 4 §12 decisions on tokenizers-12-landed; keeping the evergreen page's decision rows open so anyone reading the live doc sees what's in flight." },
      { id: "n2", who: "agent/sonnet", bot: true,  when: "2026-05-20 17:55", body: "Drafted §12 expansion plan as 6 sub-items. Ranked by ROI; cross-modality alignment quality metric (§12.6) is the lowest-cost and worth landing first.", quote: "re §12 expansion plan" },
      { id: "n3", who: "smc",          bot: false, when: "2026-05-19 09:10", body: "The Apache-2.0 constraint is non-negotiable. If Cosmos-Tokenizer-DV needs an NVIDIA OML license check, we're out — Option B becomes dead even if cheaper to ship." },
    ],

    research: [
      { id: "r1", type: "paper",   title: "Open-MAGVIT2: LFQ codebook for video",     source: "TencentARC · NeurIPS 2024",  added_by: "smc",          when: "2026-05-14", url: "#" },
      { id: "r2", type: "paper",   title: "Chronos: pretrained scaling for time-series", source: "Amazon arXiv 2403.07815", added_by: "agent/opus",   when: "2026-05-15", url: "#" },
      { id: "r3", type: "paper",   title: "PatchTST: long-horizon time-series transformers", source: "arXiv 2211.14730",   added_by: "agent/opus",   when: "2026-05-15", url: "#" },
      { id: "r4", type: "doc",     title: "FAIR-MAST IDS mapping reference",          source: "FAIR-MAST · v3.2 docs",      added_by: "smc",          when: "2026-05-13", url: "#" },
      { id: "r5", type: "plan",    title: "world-model-v0.html §2 (token-stream)",    source: "internal · this project",    added_by: "smc",          when: "2026-05-16", url: "plan.html?slug=world-model-v0" },
      { id: "r6", type: "image",   title: "rbb reconstruction MAE heatmap (shot 15085)", source: "tests/tokenizer_bench/mae-heatmap.png", added_by: "agent/sonnet", when: "2026-05-19", url: "#" },
      { id: "r7", type: "thread",  title: "Slack #ml — IR codebook decision",         source: "#ml · 2026-05-14",           added_by: "smc",          when: "2026-05-14", url: "#" },
      { id: "r8", type: "dataset", title: "rir shard inventory (25 shots, 410 MB)",   source: "scratch/rir-2026-05.parquet", added_by: "agent/sonnet", when: "2026-05-16", url: "#" },
      { id: "r9", type: "web",     title: "Cosmos-Tokenizer license terms (NVIDIA OML)", source: "nvidia.com/cosmos-license", added_by: "smc",         when: "2026-05-15", url: "#" },
    ],

    questions: [
      { id: "q1", section: "12.4 multi-modal alignment",      body: "Should §12.4 use cosine on shared embeddings or per-modality InfoNCE? Need a 2-paragraph trade-off note before locking.", opened_by: "smc",         opened_at: "2026-05-20" },
      { id: "q2", section: "13.1 plasma-decoder fine-tune scaffold", body: "Does the §13.1 scaffold need to wait on §12.4 alignment improvements landing first, or are they independent?", opened_by: "agent/opus",  opened_at: "2026-05-21" },
    ],

    tests: [
      { name: "test_tokenizer_roundtrip",      pass: 12, fail: 0, pulse: "ssssssssssss" },
      { name: "test_blockkind_persistence",    pass: 11, fail: 1, pulse: "ssssfsssssss" },
      { name: "test_chronos_t5_small",         pass: 10, fail: 2, pulse: "ssssfssfssss" },
      { name: "test_patchtst_identity",        pass: 12, fail: 0, pulse: "ssssssssssss" },
      { name: "test_codebook_sharing",         pass:  9, fail: 3, pulse: "fsfssssfssss", fail_now: true },
      { name: "test_token_id_namespacing",     pass: 12, fail: 0, pulse: "ssssssssssss" },
    ],
  };

  return { today, projects, inventory, sprints, sprint: activeSprint, blockers, timeline, planTokenizers, plans };
})();
