# Plans

> Strategic vision and active plans for **imas-ambix** — the Fusion World
> Model framework that consumes [imas-codex](https://github.com/iterorganization/imas-codex)
> mappings and produces trained transformer checkpoints.
>
> **Rule:** Delete plans when implemented. The code is the documentation.
> Once a plan is fully realised in `imas_ambix/`, move it to `completed/`
> for the record and remove the active entry from this index.

## Vision & strategy

| Plan | Scope |
|---|---|
| [STRATEGY.md](STRATEGY.md) | The Fusion World Model strategy — vision, partner-facility roadmap, why MAST first, v0/v1 success criteria, risk register |
| [v0-runway.md](v0-runway.md) | **Active operational plan** — next-step ranking, fleet-dispatch table, review rubric. Edit in place as work lands. |
| [compute.md](compute.md) | SLURM patterns for training on the 4 × H200 Group A reservation; dedicated-training reservation request; scheduled-around-serving fallback |

## Phase 0 — Probe & mirror

| Plan | Scope | Status |
|---|---|---|
| [data-acquisition.md](data-acquisition.md) | FAIR-MAST endpoint inventory, sizing-probe protocol, bulk-download SLURM spec, storage layout under `/work/projects/imas_gpu/mast/` | Draft — probe pending |

## Phase 1 — Tokenizer prototype

| Plan | Scope | Status |
|---|---|---|
| [tokenizers.md](tokenizers.md) | Open-MAGVIT2 frame tokenizer + Chronos / PatchTST signal tokenizers, registry, persistence under `/work/projects/imas_gpu/mast-tokens/`; §12 adds plasma fine-tune, PatchTST-real, equilibrium 2-D, and multi-modal alignment expansion roadmap | Active — tokenizers wired; expansion items in §12 |

## Phase 0/1 supporting plans

| Plan | Scope | Status |
|---|---|---|
| [data-quality.md](data-quality.md) | Pre-training data validation framework — per-shot `ShotQualityReport`, corpus audit, CLI (`ambix data audit`), acceptance gates for training-grade shots | **In flight** — implementation: `imas_ambix/quality/` |
| [tokenizer-benchmarks.md](tokenizer-benchmarks.md) | Quantitative tokenizer comparison framework — `BenchConfig`, `BenchResult`, frame metrics (rFID/PSNR/LPIPS), signal metrics (Pearson r/NRMSE), throughput, SLURM batch runner, results archive | **In flight** — implementation: `imas_ambix/bench/` |

## Phase 2 — World-model v0

| Plan | Scope | Status |
|---|---|---|
| [world-model-v0.md](world-model-v0.md) | ~500 M decoder-only Llama-class AR transformer over interleaved token streams; 125 M → 500 M curriculum; HuggingFace `transformers` + `accelerate` FSDP | Draft — blocked by `tokenizers.md` round-trip |
| [demo.md](demo.md) | Wide-angle viewing system forward-prediction demo on three pinned held-out MAST shots; rFID + centroid MSE acceptance | Draft — blocked by `world-model-v0.md` convergence |

## Dependency graph

```
                         STRATEGY.md
                              │
                  ┌───────────┴───────────┐
                  │                       │
         data-acquisition.md         compute.md
                  │
                  ▼
            tokenizers.md
                  │
                  ▼
          world-model-v0.md
                  │
                  ▼
                demo.md
```

`compute.md` is consumed throughout — the reservation it requests is a
prerequisite for the 500 M step of `world-model-v0.md` and for any
non-trivial fine-tune in `tokenizers.md`.

## How these plans relate to imas-codex

| Plan | Codex dependency |
|---|---|
| STRATEGY.md | Cites `imas-codex/plans/gpu-cluster-scoping.md` §3.3 as the procurement-side rationale; ambix is the "Fusion World Model" arm of that proposal. |
| data-acquisition.md | None for MAST v0 — FAIR-MAST publishes its own IMAS mapping. Phase 3+ facilities depend on codex Phase 5 discovery output. |
| tokenizers.md | None directly; the token registry is the ambix-side interface that future codex-derived facility manifests will plug into. |
| world-model-v0.md | None directly; v1 will start consuming codex standard-names for cross-facility joins. |
| demo.md | None. |
| compute.md | None — ambix and codex share the same GPU cluster but have distinct SLURM patterns. |

## Out of scope of this plan set (deferred)

- Multi-facility expansion (TCV / JT-60SA / AUG / DIII-D). The MAST-only
  v0 is the entry point; partner-facility plans land as separate docs
  under `plans/facilities/` once Phase 2 is complete.
- Real-time inference and operator-loop integration. Phase 5 in
  `STRATEGY.md`, deferred until Phase 2 is closed.
- First-principles physics losses in the model. Captured as an explicit
  anti-goal in `STRATEGY.md` §5.
- Latent-action learning (Genie-style). Captured as an explicit anti-goal
  in `world-model-v0.md` §8.

## Update log

| Date | Change | Author |
|---|---|---|
| 2026-05-19 | Initial plan set created — STRATEGY, data-acquisition, tokenizers, world-model-v0, demo, compute, this README | Simon McIntosh |
| 2026-05-20 | Added v0-runway.md (operational next-step plan); level-2 mirror 83 % done; Open-MAGVIT2 wired and working on real rbb frames; fleet dispatch underway. | Simon McIntosh |
| 2026-05-20 | Added data-quality.md + tokenizer-benchmarks.md; appended tokenizers.md §12 (plasma fine-tune, PatchTST real, equilibrium 2-D, alignment improvements, IR codebook decision, modality coherence metric); updated v0-runway.md ROI ranking with three in-flight concurrent tracks. | Simon McIntosh |
