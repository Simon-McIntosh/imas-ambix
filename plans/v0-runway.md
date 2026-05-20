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
| SLURM dedicated reservation | **not requested** |
| Concurrent training+serving analysis | **landed in compute.md §2** — 125M fully concurrent, 500M concurrent with mb=2 |

---

## 2. Highest-ROI next steps (ranked)

ROI = value × probability of clean delivery / cost.

| Rank | Track | Value | Cost | Risk | Sonnet 4.6? | Depends on |
|---|---|---|---|---|---|---|
| 1 | Chronos signal tokenizer + PatchTST identity wrapper | High (closes signal stream) | Low | Low | Yes | none |
| 2 | Eval metrics scaffold (rFID, PSNR, LPIPS, centroid, chord, edge) | High (needed for training-time eval + demo) | Low | Low | Yes | none |
| 3 | WHAM model scaffold + Hydra configs | Critical (no model = no demo) | Medium | Low | Yes | Open-MAGVIT2 + Chronos registries |
| 4 | Data loaders + token persistence + bulk-encode CLI | Critical (no loader = no training) | Medium | Low | Yes | Open-MAGVIT2 |
| 5 | Open-MAGVIT2 GPU runner | High (unlocks production tokenization on betelgeuse) | Low | Low | Yes (with guidance) | none |
| 6 | Training-loop FSDP scaffold | Critical | Medium | Medium | Sonnet + Opus review | tracks 3 + 4 |
| 7 | Demo CLI + rollout code | Critical | Medium | Low | Yes | track 6 |
| 8 | Mirror integrity verification | Low | Low | Low | Yes | level-2 done |
| 9 | SLURM dedicated-reservation request (`compute.md` §3) | High | Low | Low | No (operational, user files) | none |

Tracks 1 – 4 are independent and parallelisable. Tracks 5, 6, 7 are
sequential.

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
