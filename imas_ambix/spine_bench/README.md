# Physics-spine benchmark — a durable performance + quality evolution metric

This is the **equilibrium-engine ("physics spine") benchmark**, distinct from the
world-model camera/rFID benchmark in [`imas_ambix/bench/`](../bench/). It stamps the
reanalysis engine's **performance** and **physics quality** on a **frozen shot set** so
we can measure evolution (the GPU rollout, FSA improvements, closure hardening) against a
fixed reference — the baseline the `greens-filament-solver` §4 (performance) and
`gpu-accelerated-labeler` §6 (throughput) gates, and the `greens-filament-solver` §3 (FSA
integrity) gate, compare against.

## Run

```bash
OMP_NUM_THREADS=1 uv run python -m scripts.spine_benchmark
# → imas_ambix/spine_bench/results/physics-spine-<shotset>-<commit>[-dirty]-<host>.yaml
```

Run on a SLURM compute node (not the login node); timing is at `OMP_NUM_THREADS=1` so the
throughput number is a clean single-core proxy.

## What makes it a *reliable* evolution metric

Three things are versioned and stamped into every record, so two stamps are only compared
when they agree (or the difference is explicit):

- **`schema_version`** (`schema.py`) — the record shape + metric definitions. Bump on any
  change to what a number means.
- **`shotset_version`** (`shots.py`) — the FROZEN named shot set. Bump when shots/roles
  change. Never silently compare across shot sets.
- **`env.engine_config_sha`** — the frozen-spine config SHA (`closure_spine_frozen.json`),
  plus `git_commit` / `git_dirty` and the `machine` block (host, CPU, RAM, OMP threads).

## Metrics (registry in `schema.py::METRICS`)

Each metric is named, has a unit and an improvement direction, and a definition:

Two solvers are benched each run: **`greens-matvec`** — the filament /
single-interaction-matrix **dev spine** (analytic ψ = G·I, the pivot away from the
gridded Δ\*; primary, the GPU target) — and **`grid-delstar`** — the gridded Δ\* solve,
retained as the **baseline check** the dev spine is validated against. Every stamp also
carries a **`device`** field (`cpu`/`gpu`) so CPU and GPU runs sit in one comparison.

| Metric | Unit | Dir | Meaning |
|---|---|---|---|
| `solve_wall_ms_per_slice` | ms/slice | ↓ | median rich ladder solve wall (warmup-excluded) |
| `end_to_end_ms_per_slice` | ms/slice | ↓ | full per-slice wall: disc-seed + K=2 scaffold + rich ladder |
| `throughput_slices_per_core_s` | slice/(core·s) | ↑ | corpus-label-factory throughput proxy at OMP=1 |
| `latency_ms_p50` / `latency_ms_p99` | ms | ↓ | end-to-end per-slice latency (streaming-latency proxy) |
| `axis_reproduce_cm` | cm | ↓ | dev-spine vs grid baseline-check magnetic-axis agreement |
| `lcfs_reproduce_cm` | cm | ↓ | dev-spine vs grid baseline-check LCFS-radii agreement |
| `profile_reproduce_rms` | norm | ↓ | dev-spine vs grid baseline-check jφ(ρ̂) agreement |
| `fsa_d_roughness_nrho32/96` | norm | ↓ | roughness of d = g2·g3/ρ̂ at n_rho 32 / 96 |
| `fsa_d_roughness_resolution_slope` | Δ/Δlog2(nρ) | ↓ | >0 = worsens with resolution (the §3-claimed pathology) |
| `converged_fraction` / `confined_fraction` | frac | ↑ | solve health |

Run-level fields: `complete_run_wall_s`, `peak_rss_gb` (process peak RSS), and a
per-shot per-substrate **`phase_timing_ms`** breakdown (`disc_read` / `scaffold_k2` /
`rich_ladder` / `fsa_readout`) — the component attribution of where solve time goes.

**Scope now = the per-slice STATIC solve** (the GPU inner-loop target). The
dynamics-coupled label rollout (§3 resistive diffusion + passive circuits + temporal
warm-start) is the label-factory throughput — a distinct mode to add before the corpus
GPU run. Per-component / GPU-device memory is added with the GPU rollout; `peak_rss_gb`
here is process-level.

**FSA d-roughness** = `rms(2nd-difference of geo.d_face) / rms(d_face)` over interior
faces — a dimensionless, resolution-comparable measure of the noise the resistive
diffusion integrates against. On the grid-GS substrate the `resolution_slope` is expected
`> 0` (the `greens-filament-solver` §3 motivation); the analytic-ψ substrate should flatten
it. Held-out-MSE pitch is tracked separately (`heldout_mse_gate_eval`), not here.

## Why not asv directly (yet)

We deliberately went **asv-inspired, not asv-driven**, for the first baseline:

- asv's value is cross-commit tracking + a dashboard + timing stats. We keep the good
  parts: commit + machine keying, warmup-excluded timing over a per-slice distribution,
  and `track_*`-style arbitrary quality metrics (FSA roughness, reproduction).
- asv's **isolated-env-per-commit** model does not fit a heavy-data / GPU physics solve
  (real MAST shots on GPFS, minutes per run, fp64, later CUDA). asv is not installed.
- The YAML records are **asv-wrappable later**: a thin `benchmarks/` shim exposing
  `time_*` / `track_*` over the frozen shots + `asv run --environment existing` gives the
  dashboard without changing this schema. Add it when a live dashboard is wanted.

## Storage: repo now, GHCR when it grows

The stamp is a small, human-readable, diffable YAML committed **in the repo** under
`results/`, tied to the commit it measures. This is right while the history is small.

**Scale-out to GHCR** (control the growing time-series separately from the code repo)
when CI drives it or the history bloats — the schema is GHCR-portable (self-describing,
commit+machine keyed). Push each stamp as an OCI artifact:

```bash
oras push ghcr.io/simon-mcintosh/imas-ambix-bench:physics-spine-<commit> \
    <stamp>.yaml:application/yaml
```

(requires a `GITHUB_TOKEN` with `packages:write`; an outward publish — confirm before
first use). Until then, repo storage keeps the baseline self-contained and reviewable.
