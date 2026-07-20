# Agent Guidelines — physics-spine benchmark (`imas_ambix/spine_bench/`)

Guidance scoped to the equilibrium-engine ("physics spine") benchmark. The
repo-wide rules in [`../../AGENTS.md`](../../AGENTS.md) and
[`~/.agents/AGENTS.md`](~/.agents/AGENTS.md) apply in full — this file adds only
what is specific to running and evolving the benchmark, including the **GPU /
H200 policy**.

## What this benchmark is

A durable, versioned **performance + physics-quality** stamp of the reanalysis
engine on a FROZEN shot set (`shots.py`), so evolution is measured against a
fixed reference. Comparability is guarded by three stamped versions — schema
(`schema.py::SCHEMA_VERSION`), shot set (`shots.py::SHOTSET_VERSION`), and the
engine-config SHA — plus the machine/env block. Never compare across a version
bump silently; bump the relevant version when a metric definition, the shot set,
or the record shape changes.

Two solvers are benched each run: **`greens-matvec`** (the filament /
single-interaction-matrix dev spine, the GPU target) and **`grid-delstar`** (the
gridded Δ\* baseline check). Two flux-surface-averaging reads are benched
head-to-head: **`coarea`** (host baseline) and **`connectivity`** (the
contour-free JAX read — see below).

## Run the CPU metric (the frozen comparable stamp)

Always on a SLURM compute node at `OMP_NUM_THREADS=1` (a clean single-core
throughput proxy), never the login node. Debug partitions cap at **1 h** — a
`--time` over `01:00:00` sits in `PartitionTimeLimit` PENDING forever.

```bash
srun --partition=sun_debug --cpus-per-task=8 --mem=16G --time=01:00:00 \
  bash -lc 'export TMPDIR=/tmp OMP_NUM_THREADS=1; cd <repo>; \
    uv run python -m scripts.spine_benchmark'
# → imas_ambix/spine_bench/results/physics-spine-<shotset>-<commit>[-dirty]-<host>.yaml
```

Commit the engine first so the stamp is `git_dirty: false` (the stamp records
the dirty flag from `git status --porcelain --untracked-files=no`).

## GPU / H200 policy (binding for this benchmark)

**Run every GPU-capable path of this benchmark on the H200 as part of
development — a capability demonstration is expected on device, not deferred.**
"Capability" (it runs on the accelerator, batched, in fp64, matching the host
result) comes first; a speedup claim is a separate, later gate.

GPU-capable inventory today:

| Component | Device-ready? | Notes |
|---|---|---|
| Connectivity FSA (`imas_ambix/latent/flux_surface_connectivity.py`) | **Yes** | `jit`/`vmap`/`grad`-safe, fixed-shape, fp64. Demo: `scripts/fsa_gpu_capability.py`. |
| Engine solve (`gs_solve`, scipy sparse-LU + numpy) | No | host-only; GPU port is the `gpu-accelerated-labeler` follow-on. |

### H200 node access

The H200 (`98dci4-gpu-0003`, partition `betelgeuse`) has **`ReqResv=YES`** —
every job MUST name the standing reservation `gpu_0003_grpA` or it is denied.
The reservation holds CPUs only (30-CPU ceiling); all 8 GPUs are requestable
through it.

The CUDA jaxlib is now a **core dependency** (`jax[cuda12]` on Linux, plain
`jax` elsewhere — see the root `pyproject.toml`), so a normal `uv sync` installs
it and no manual step is needed. The one caveat: **compute nodes have no
network** — the FIRST `uv sync` (or any resolve/download) MUST run on a
networked **login** node; once the `.venv` is populated, `uv run` on the H200
reads it offline. On a CPU box the same code just falls back to CPU.

```bash
# once, on the LOGIN node (has network): populate the .venv from the lock
uv sync

# then on the H200 (plain `uv run` uses the .venv — offline is fine):
srun --partition=betelgeuse --reservation=gpu_0003_grpA --gres=gpu:1 \
     --cpus-per-task=4 --mem=32G --time=00:20:00 \
  bash -lc 'export TMPDIR=/tmp; cd <repo>; \
    uv run python -m scripts.fsa_gpu_capability --batches 16,64,256,1024'
# → imas_ambix/spine_bench/results/fsa-gpu-capability-<commit>-<host>.yaml
```

The CUDA jaxlib version must match the installed `jax` (`jax[cuda12]` keeps them
in lockstep); a mismatch silently drops the plugin and JAX falls back to CPU
(`Unable to load CUDA`). The demo prints the resolved `devices` and asserts
GPU/CPU parity, so a CPU fallback is visible in the stamp (`on_gpu: false`),
never a silent pass.

## Extending the benchmark

- New metric → add it to `schema.py::METRICS` (name + unit + direction +
  description) and bump `SCHEMA_VERSION`. A metric's meaning must never drift
  without a version bump.
- New shot / role → edit `shots.py` and bump `SHOTSET_VERSION`.
- Results (`results/*.yaml`) are committed as the durable record; a fresh stamp
  is a new file keyed by commit + host, never an overwrite.
