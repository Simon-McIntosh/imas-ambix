# Plan: v0 Runway — Next Steps

Status: **Active** — updated 2026-05-20.

This is the operational doc that tabulates the next set of work items on
the way to the v0 demo (`plans/demo.md`). It is a *living document* — as
items complete or new ones emerge, edit this file in place. Use
`STRATEGY.md` for vision, `world-model-v0.md` for the model contract,
`tokenizers.md` for tokenizer wiring, `data-acquisition.md` for the
corpus, `compute.md` for SLURM.

---

## 1. Recap — where we are 2026-05-20 (afternoon)

| Track | Status |
|---|---|
| Level-2 mirror | **87 %** (10,099 / 11,573 shots, ETA hours; bucket du = 464 GB authoritative) |
| Level-1 camera mirror | **complete** (2.7 TB, 11,029 shot dirs) |
| **Level-1 ALL-sources mirror** (xdc/amc/amb/aga/efm/etc.) | **in flight** — manifest level1-all.json launched ~07:00 UTC |
| Tokenizer scaffold | **landed** — registry, alignment, multimodal aggregator, CLI |
| Open-MAGVIT2 frame tokenizer | **wired** — real model end-to-end, MAE 324 on rbb; **CUDA wheel installed**, GPU runner code ready, GPU test skips on login |
| Chronos signal tokenizer | **wired** (r ≈ 0.985 round-trip on synthetic sine/cosine) |
| PatchTST identity wrapper | **wired** (bit-exact round-trip) |
| Eval metrics module | **landed** (rFID, PSNR, LPIPS, centroid, chord, edge, rollout stub; 31 tests + 4 skips) |
| WHAM model + 125M/500M Hydra configs | **landed** (8 tests; 125M=328M params, 500M=689M — embedding-table dominated) |
| Data loaders + token persistence | **landed** (15 tests; ambix data tokens-status live) |
| **block_kind side data** | **landed** (51 tokenizer tests; weighted-CE loss mask ready) |
| **CLI smoke tests** (data + tokenize) | **landed** (42 tests; data/cli.py 86 % coverage, tokenizer/cli.py 97 %) |
| Training loop FSDP scaffold | **not started** |
| Demo CLI | **not started** |
| SLURM dedicated reservation | **not requested** (deferred — see below) |
| **Training mode decision** | **EXCLUSIVE** (2026-05-20) — V4-Flash is stopped before training. Maintainer (ambix CLI) is authorised to `scancel` the serve. See `compute.md` §2. |
| **Data quality framework + ambix data audit** | **in flight** (sonnet-data-quality) — `imas_ambix/quality/`, CLI sub-command `ambix data audit`. See `plans/data-quality.md`. |
| **Tokenizer benchmark framework + imas-ambix tokenize bench** | **in flight** (sonnet-bench-framework) — `imas_ambix/bench/`, CLI sub-command `imas-ambix tokenize bench`. See `plans/tokenizer-benchmarks.md`. |
| **Calibration library (signals + frames)** | **landed** (fe18b34) — `imas_ambix/calibration/`, 17 tests; real-data smoke produced Ip mean 294 kA, density 4.6e19 m⁻³ |
| **Data quality framework + ambix data audit** | **landed** (18a3385) — `imas_ambix/quality/`, 22 tests; **audit smoke on 25 L2 shots: 76 % usable_for_training**, dd_version + dynamic_range + time_axis warnings widespread (audit thresholds may need calibration) |
| **Tokenizer benchmark framework** | **landed** (94565f3) — `imas_ambix/bench/`, 14 tests; real bench on 8 rbb frames |
| **Bulk-encode CLI** (frames + signals) | **landed** (9306605) — `ambix data bulk-encode-frames` + `bulk-encode-signals`, 15 tests; CPU smoke on shot 15085 produced 9,380 tokens |
| **Training loop FSDP scaffold** | **landed** (a948b19) — `imas_ambix/train/loop.py`, 25 tests; CPU smoke on tiny synthetic config: loss step 0 = 12.72, step 1 = 12.72 (log(280k) baseline) |
| **Demo CLI + rollout impl** | **landed** (3e3bb2a) — `ambix demo wham-mast` + `rollout()`, 20 tests + 31 eval tests; mock-checkpoint pipeline working end-to-end |

---

## 2. Highest-ROI next steps (ranked)

ROI = value × probability of clean delivery / cost.

Items marked **DONE** are complete as of 2026-05-20; they remain here for
historical reference. In-flight items will close before the training-loop
track begins.

| Rank | Track | Value | Cost | Risk | Sonnet 4.6? | Depends on | Status |
|---|---|---|---|---|---|---|---|
| 1 | ~~Chronos signal tokenizer + PatchTST identity wrapper~~ | High | Low | Low | Yes | none | **DONE** |
| 2 | ~~Eval metrics scaffold (rFID, PSNR, LPIPS, centroid, chord, edge)~~ | High | Low | Low | Yes | none | **DONE** |
| 3 | ~~WHAM model scaffold + Hydra configs~~ | Critical | Medium | Low | Yes | — | **DONE** |
| 4 | ~~Data loaders + token persistence + bulk-encode CLI~~ | Critical | Medium | Low | Yes | — | **DONE** |
| 5 | **Tokenizer expansion (plasma decoder fine-tune, PatchTST real, equilibrium 2-D)** | High (raises rFID floor, PatchTST fidelity) | Medium | Low–Medium | Sonnet (guided) | benchmark baseline measured + rFID > 5 trigger | **Pending — see `tokenizers.md` §12** |
| 6 | Data quality framework + corpus audit (writing training index) | High (gates training set correctness) | Low | Low | Yes | level-2 mirror + level-1 camera mirror | **in flight** |
| 7 | Tokenizer benchmark framework (rFID, Pearson r, throughput) | High (needed to trigger fine-tune decisions) | Low | Low | Yes | GPU runner working | **in flight** |
| 8 | Calibration library (per-channel signal calibration, frame normalisation) | Medium (improves Chronos NRMSE) | Low | Low | Yes | none | **in flight** |
| 9 | Training-loop FSDP scaffold | Critical | Medium | Medium | Sonnet + Opus review | tracks 3 + 4 (DONE) + quality audit | **not started** |
| 10 | Demo CLI + rollout code | Critical | Medium | Low | Yes | training loop | **not started** |
| 11 | Mirror integrity verification | Low | Low | Low | Yes | level-2 done | **not started** |
| 12 | SLURM dedicated-reservation request (`compute.md` §3) | High | Low | Low | No (operational) | none | **deferred (user action)** |

Tracks 6, 7, 8 are in flight in parallel. Track 9 (training loop) unblocks
only after 6 is complete (quality audit produces the training index). Track 5
(tokenizer expansion) runs after 7 provides the rFID baseline.

---

## 3. Fleet dispatch — parallel batch (2026-05-20)

Four Sonnet 4.6 agents in parallel with **non-overlapping write scopes**.
Coordinator (Opus) reviews each output post-completion. See
[[fleet-sonnet-46-offload]] memory.

| Agent | Track | Exclusive write scope | Read-only references |
|---|---|---|---|
| **S1** | Chronos + PatchTST signal tokenizers | `imas_ambix/tokenizer/signals.py`; `tests/test_tokenizer.py` (additions); `plans/tokenizers.md` (§9.2 update) | `tokenizer/base.py`, `tokenizer/registry.py`, `tokenizer/frames.py` (the OpenMagvit2 pattern), `plans/tokenizers.md` §3 |
| **S2** | Evaluation metrics + rollout stub | `imas_ambix/eval/__init__.py`, `imas_ambix/eval/metrics.py`, `imas_ambix/eval/rollout.py`; `tests/test_eval.py` (new) | `plans/demo.md` §4, `plans/world-model-v0.md` §5 |
| **S3** | WHAM model scaffold | `imas_ambix/model/__init__.py`, `imas_ambix/model/config.py`, `imas_ambix/model/wham.py`; `imas_ambix/train/__init__.py`, `imas_ambix/train/configs/v0-125m.yaml`, `imas_ambix/train/configs/v0-500m.yaml`; `tests/test_model.py` (new) | `plans/world-model-v0.md` §1, §2, §4; `tokenizer/registry.py` for vocab size |
| **S4** | Data loaders + token persistence + bulk-encode CLI | `imas_ambix/data/loaders.py`, `imas_ambix/data/persist.py`; `imas_ambix/data/cli.py` (extension, do not touch existing commands); `tests/test_loaders.py`, `tests/test_persist.py` (new) | `plans/world-model-v0.md` §3, §6; `plans/tokenizers.md` §5; `tokenizer/frames.py`, `tokenizer/multimodal.py` |

Mandatory dispatch preamble per AGENTS.md is embedded in every Agent
prompt. Branch: `code/data-skeleton`. Each agent commits + pushes its
own changes with a `feat(<scope>):` conventional commit.

---

## 4. Sequenced follow-up (after the parallel batch)

| Step | Owner | Dependency |
|---|---|---|
| Open-MAGVIT2 GPU runner adaptation | Sonnet (with guidance) | parallel batch complete |
| Bulk-encode the rbb corpus on the GPU node | Opus (operator) | GPU runner + dedicated reservation |
| Training-loop FSDP scaffold | Sonnet, reviewed by Opus | model + loaders + tokens |
| First training smoke run (125 M, 1 GPU, 10 shots) | Opus | training loop |
| 500 M curriculum run on the dedicated reservation | Opus | smoke green + reservation granted |
| Demo CLI + notebook + MP4 | Sonnet | trained checkpoint |
| File the SLURM reservation request (`compute.md` §3.2) | User | none — operational |

---

## 5. Review rubric for Sonnet 4.6 work

Every dispatched agent's output is reviewed by the coordinator against:

- **Protocol conformance**: implements the documented `Tokenizer` /
  `WhamConfig` / etc. interface verbatim.
- **File scope adherence**: touches only its allocated paths.
- **Test coverage**: at least one test per public function and per
  documented behaviour.
- **Lint + format**: `uv run ruff check && uv run ruff format` clean.
- **Live execution**: where applicable, the new code is exercised via
  its CLI on real data (a synthetic numpy/xarray fixture is fine for
  PRs that don't have GPU access).
- **Plan-doc alignment**: cross-references to `plans/*.md` are accurate
  and the plan files are updated in lockstep if the implementation
  diverged from the design.

Below standard ⇒ one round of correction via `SendMessage` to the same
agent; if still below standard, the coordinator fixes in-session and
records the failure mode in a new feedback memory.
