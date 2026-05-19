# Ambix: Fusion World Model Strategy

**System:** `imas-ambix`
**Role:** Distillation & Generative Modelling (consumer of imas-codex
mappings, producer of Fusion World Model weights)
**Output:** Trained transformer checkpoints that predict plasma evolution
given control inputs, plus a multi-modal tokenizer toolchain that lets us
do so on heterogeneous tokamak diagnostic streams.

---

## 1. Executive Vision

Ambix builds the generative engine of the Fusion World Model. Where
[imas-codex](https://github.com/iterorganization/imas-codex) answers "what
data exists and how it maps to IMAS", ambix answers "given the data and the
mapping, what does the plasma do next?".

The strategic motivation is captured in the GPU-procurement document
prepared for Science Division (see
`../../imas-codex/plans/gpu-cluster-scoping.md` §3.3). The headline
argument: a Microsoft-style world model trained on tokamak diagnostic
streams gives ITER three capabilities that are unreachable with current
analysis tooling:

1. **Pre-play** of planned pulses against learned plasma physics, before
   machine time is consumed.
2. **Disruption-trajectory exploration** in silico, so the operations team
   can probe scenario boundaries without risking the device.
3. **Validation of physics models** by comparing generated plasma states
   to first-principles predictions on the same control input.

The reference architecture is Microsoft's WHAM (World and Human Action
Model, Kanervisto *et al.*, *Nature* **638** (2025) — DOI
[`10.1038/s41586-025-08600-3`](https://doi.org/10.1038/s41586-025-08600-3),
trained on 7 years of *Bleeding Edge* gameplay). WHAM tokenises game frames
with a ViT-VQGAN, interleaves frame tokens with discretised controller
actions, and trains a decoder-only transformer to autoregress the next
token. The same recipe maps cleanly onto a tokamak: replace game frames
with diagnostic measurements (including camera frames), replace controller
actions with the engineered control vector (coil currents, gas-injection
rates, heating power, pulse-schedule waypoints), and predict the next
multi-modal state token-by-token.

The Fusion World Model is not a substitute for first-principles solvers
(GENE, JOREK, EFIT). It is a generative model of *observed* tokamak
behaviour — extracted directly from experimental data, validated against
held-out shots, and useful precisely where first-principles models are too
expensive or too uncertain to run in the loop.

---

## 2. Why MAST first

The 26 B state-transition corpus across JET / TCV / JT-60SA / AUG / DIII-D
described in the GPU-procurement document is the long-term training target.
It cannot be the v0 dataset. Three reasons make FAIR-MAST the right
starting point:

1. **Permissive open access.** Data is licensed CC BY-SA 4.0, the
   `ukaea/fair-mast` tooling is MIT, and the S3 endpoint at
   `s3.echo.stfc.ac.uk` is **anonymously accessible** — no credentials,
   no data-use agreements, no embargo windows. Every other partner
   facility currently requires bilateral negotiation.
2. **IMAS-mapped at source.** FAIR-MAST level-2 data is already mapped to
   IMAS-aligned diagnostic groups (`magnetics`, `equilibrium`,
   `camera_visible`, `camera_ir`, `thomson_scattering`, …). We do not need
   to wait for imas-codex Phase 5 discovery to complete on MAST(-U) before
   we can train; the mapping is published.
3. **Camera coverage.** MAST(-U) ran a stable wide-angle visible-light
   imaging system (RBA / RBB / RGB cameras → IMAS `camera_visible`) and an
   infrared system (RIR / AIT → `camera_ir`) across enough shots in the M5
   – M9 campaigns to support a frame-prediction prototype. Camera-rich
   tokamaks are rarer than people assume.

The trade-off: MAST is a smaller, lower-current spherical tokamak. The
plasma physics it produces is not directly transferable to ITER scenarios.
We accept this. The v0 model is for *workflow*, not *physics transfer*:
prove the pipeline (download → tokenise → train → roll out) end-to-end on
a permissive corpus, then graft on additional facilities through
imas-codex once their mappings land.

---

## 3. Roadmap

| Phase | Scope | Status | Driver |
|---|---|---|---|
| **0** — Probe & mirror | Probe FAIR-MAST S3 throughput; mirror level-2 to GPFS | Draft | This plan set |
| **1** — Tokenizer prototype | Open-MAGVIT2 frame tokens + Chronos/PatchTST signal tokens, end-to-end round-trip on 10 shots | Draft | `tokenizers.md` |
| **2** — World-model v0 | ~500 M decoder-only AR transformer; WHAM-style interleaved stream; demo on a held-out MAST shot | Draft | `world-model-v0.md`, `demo.md` |
| **3** — Multi-facility expansion | Add TCV (DOI-published level-2 on Zenodo), then JT-60SA (via imas-codex mapping work) | Pending Phase 2 | imas-codex Phase 5 |
| **4** — Scaling | Move to multi-billion parameter models on additional GPU allocation (8-GPU or multi-node) | Pending Phase 3 | Compute procurement |
| **5** — Operational pre-play | Wrap the trained model in an `ambix pre-play` CLI for operations team use | Pending Phase 4 | Operations stakeholder |

Only Phase 0 – 2 are scoped in detail by this plan set. Phase 3+ depends on
data and compute that are not yet committed.

---

## 4. Dependency on imas-codex

Ambix and codex are tightly coupled by design:

| Codex provides | Ambix consumes |
|---|---|
| Federated Fusion Knowledge Graph (Neo4j artifact) | Per-facility IMAS path → native signal name mapping |
| Standard Names ontology (62+ entries, growing) | Stable semantic anchors for cross-facility joins |
| LinkML schemas (`imas_dd.yaml`, `facility.yaml`) | Validated data manifest schema for training corpora |
| Discovery agents (File Explorer, Code Search, Data Inspector) | Out-of-band facility surveys that bootstrap new training data sources |

For the MAST v0 work, ambix does **not** strictly need codex outputs —
FAIR-MAST publishes its own IMAS mapping. But the moment we add a second
facility, codex becomes load-bearing. The `imas_ambix/data/manifest.py`
module is therefore designed to consume codex-style facility manifests
even in v0, even though the MAST manifest is hand-written. This keeps the
data-loader interface stable as we add facilities.

---

## 5. Success criteria

### v0 (this plan set)

- All 11,573 FAIR-MAST level-2 shots mirrored to
  `/work/projects/imas_gpu/mast/` with integrity verified by a re-run
  `s5cmd cp` that reports zero new objects.
- A `MultiModalTokenizer` can encode a single MAST shot's interleaved
  diagnostic streams and round-trip through the decoder with rFID ≤ 5 on
  the wide-angle visible-light frames.
- A WHAM-style decoder-only transformer (≤ 500 M parameters) trained on
  the camera-bearing MAST shots produces a multi-second forward rollout of
  the wide-angle camera that is qualitatively recognisable as plasma when
  compared to the held-out ground truth.
- A self-contained demo notebook (`docs/demo.ipynb`, generated from
  `ambix demo wham-mast`) that runs the full ground-truth-vs-prediction
  comparison and reports physics-derived quantities (chord-integrated
  emission, frame centroid).

### v1 (out of scope of this plan set, but the target Phase 2 closes
toward)

- Quantitative agreement with held-out plasma metrics: chord-integrated
  emission MSE within 10 % of mean over the held-out validation shots;
  rollouts stable to ≥ 1 s of physical time without divergence.
- Sensitivity-correct response to control-input perturbations (vary the
  applied PF current ramp and observe the predicted plasma response shift
  in the correct sign / magnitude).
- Tokenizer codebook adapted to plasma imagery — i.e. fine-tuned Open-MAGVIT2
  decoder weights checked into `/work/projects/imas_gpu/mast-tokens/`.

### Anti-goals (explicitly out of scope for v0/v1)

- Real-time control. The world model is offline-only. Real-time deployment
  requires latency engineering we are not ready to invest in.
- First-principles physics inside the model. The model is purely
  data-driven; physics enters through the diagnostic stream, not through
  added solver terms.
- A new tokenizer architecture. We use Open-MAGVIT2 / Chronos / PatchTST
  off the shelf. If they fail, we swap them; we do not invent.
- A new training framework. HuggingFace `transformers` + `accelerate`
  FSDP only.

---

## 6. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| FAIR-MAST endpoint slow / unstable | Medium | High (blocks Phase 0) | Probe first (`plans/data-acquisition.md`); fall back to fsspec streaming if mirror time > 48 h |
| Camera-bearing shot fraction lower than expected | Medium | Medium | Probe inventories `camera_visible` group presence per shot; reduce scope to magnetics+equilibrium if camera coverage is < 1000 shots |
| Open-MAGVIT2 reconstruction poor on plasma imagery | Medium | Medium | Plan budgets for decoder fine-tune on plasma frames; if rFID > 10, switch to Cosmos-Tokenizer-CV continuous variant |
| 500 M model OOMs on 4×H200 alongside DeepSeek serving | High | Medium | Request dedicated reservation (`plans/compute.md`); scheduled-around-serving as interim |
| WHAM-style AR drift over long rollouts | High | Low (for v0) | v0 demo limits to ≤ 1 s rollout; long-horizon stability is a v1 problem |
| CC BY-SA 4.0 "SA" clause contaminates downstream model weights | Low | Low (legal, not technical) | Legal check before publishing weights; for now keep weights internal |

---

## 7. Plan documents

| Plan | Scope |
|---|---|
| [`STRATEGY.md`](STRATEGY.md) (this file) | Vision, roadmap, success criteria, risk register |
| [`data-acquisition.md`](data-acquisition.md) | FAIR-MAST endpoint inventory, probe protocol, bulk-download protocol |
| [`tokenizers.md`](tokenizers.md) | Per-modality tokenizer design and persistence layout |
| [`world-model-v0.md`](world-model-v0.md) | Decoder-only AR transformer spec and training recipe |
| [`demo.md`](demo.md) | Wide-angle viewing-system forward-prediction demo |
| [`compute.md`](compute.md) | SLURM patterns, reservation request, FSDP sharding |
| [`README.md`](README.md) | Index page (status, summaries) |

---

## 8. References

- Kanervisto *et al.*, "World and Human Action Models towards gameplay
  ideation", *Nature* **638** (2025) — primary reference for WHAM
  architecture. DOI: [`10.1038/s41586-025-08600-3`](https://doi.org/10.1038/s41586-025-08600-3).
- Jackson *et al.*, "FAIR-MAST: A fusion device data management system",
  *SoftwareX* **27** (2024) 101869. DOI:
  [`10.1016/j.softx.2024.101869`](https://doi.org/10.1016/j.softx.2024.101869).
- Jackson *et al.*, "An Open Data Service for Supporting Research in
  Machine Learning on Tokamak Data", *IEEE Trans. Plasma Sci.* (2025).
  DOI: [`10.1109/TPS.2025.3583419`](https://doi.org/10.1109/TPS.2025.3583419).
- ITER Science Division GPU-procurement document, §3.3 "Fusion World Model
  Development": `../../imas-codex/plans/gpu-cluster-scoping.md`.
