#!/bin/bash
#SBATCH --job-name=ambix-worldmodel-train
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --account=grpa
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=30
#SBATCH --mem=640G
#SBATCH --time=08:00:00
#SBATCH --output=/work/projects/imas_gpu/logs/ambix-%x-%j.out
# NOTE: --qos intentionally omitted — the gpu_0003_grpa QoS is auto-applied;
#       passing it explicitly errors (AGENTS.md §2).
#
# ============================================================================
#  BLOCKED — DO NOT SUBMIT YET
# ============================================================================
#  This script is the LATER full-GPU training run for the plan-conditioned
#  multi-modal world model.  It is BLOCKED until the betelgeuse GPU node's
#  /work mount is restored — the H200 node currently has a broken /work mount,
#  so the token stores under /work/projects/imas_gpu/mast-tokens are not
#  reachable from the compute node and any submitted job would fail at the
#  first store open.
#
#  The runnable, proven part of this deliverable is the CPU-TINY overfit smoke
#  (tests/worldmodel/test_worldmodel_skeleton.py and
#  imas_ambix/worldmodel/train.py::overfit) — that ran end-to-end on CPU and
#  proved tokens -> model -> recognizable prediction (loss 9.04 -> 0.12).
#  This sbatch scales the SAME code path up to the real corpus once /work is
#  back.  When unblocking:
#    1. Confirm the mount:  ls /work/projects/imas_gpu/mast-tokens/v2/signals_hf | head
#    2. Confirm GPUs clean: nvidia-smi
#    3. Then `sbatch scripts/slurm/worldmodel_train.sbatch`.
# ============================================================================

set -euo pipefail

# TMPDIR on betelgeuse: /run/user/$UID does not exist on the compute node;
# /scratch_local is the node-local NVMe (AGENTS.md §2 SLURM workarounds).
export TMPDIR="/scratch_local/${SLURM_JOB_ID}"
mkdir -p "${TMPDIR}"

# Single-threaded BLAS — torch owns parallelism on the H200 tensor cores.
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

# Offline: betelgeuse has NO outbound network (AGENTS.md §2).  Any HF / torch
# hub fetch must already be cached; fail loud instead of stalling to the limit.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# SLURM copies the batch script to /var/spool/...; use SLURM_SUBMIT_DIR (where
# sbatch was invoked) to find the repo + venv and write results back.
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_ROOT}"
mkdir -p "/work/projects/imas_gpu/logs"

# ── Drain-forensics sidecar (default-on for long GPU jobs; AGENTS.md §2a) ────
# SIGTERM-immune; samples every PID in the job cgroup (state + wchan),
# nvidia-smi, and kernel NVRM/Xid lines once per second to GPFS.  If a drain
# recurs it captures the D-state PID + kernel symbol.
bash "${REPO_ROOT}/scripts/slurm/drain_sidecar.sh" &
SIDECAR_PID=$!

# ── Preflight: refuse to start if the token corpus is unreachable ───────────
# (the /work-mount blocker above manifests here as a missing store root).
TOKEN_ROOT="/work/projects/imas_gpu/mast-tokens"
if [[ ! -d "${TOKEN_ROOT}/v2/signals_hf" ]]; then
    echo "[worldmodel-train] FATAL: ${TOKEN_ROOT}/v2/signals_hf not reachable" >&2
    echo "[worldmodel-train] the /work mount is not restored — aborting (see header)" >&2
    kill "${SIDECAR_PID}" 2>/dev/null || true
    exit 2
fi

# ── Activate the project venv ───────────────────────────────────────────────
VENV="${REPO_ROOT}/.venv"
if [[ -f "${VENV}/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source "${VENV}/bin/activate"
    PY="${VENV}/bin/python"
else
    echo "[worldmodel-train] .venv not found — falling back to uv run" >&2
    PY="uv run python"
fi

# ── Tunable knobs (override via sbatch --export=ALL,VAR=val) ─────────────────
: "${WM_TRAIN_SHOTS:=}"        # comma-separated shots; empty -> discover a pair
: "${WM_STEPS:=2000}"          # optimiser steps
: "${WM_N_STEPS:=128}"         # common-grid steps
: "${WM_CONTEXT_STEPS:=32}"    # context window (rest is the target window)
: "${WM_LR:=1e-3}"

echo "[$(date)] worldmodel-train starting"
echo "  job-id:        ${SLURM_JOB_ID}"
echo "  node:          $(hostname)"
echo "  train_shots:   ${WM_TRAIN_SHOTS:-<discover>}"
echo "  steps:         ${WM_STEPS}  n_steps=${WM_N_STEPS}  context=${WM_CONTEXT_STEPS}  lr=${WM_LR}"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader || true

# ── Clean teardown on SLURM SIGTERM ─────────────────────────────────────────
# The training loop installs its OWN SIGTERM handler (sets a STOP flag, flushes,
# releases the model, empties the CUDA cache within UnkillableStepTimeout — see
# imas_ambix/worldmodel/train.py).  Here we only ensure the sidecar is reaped.
cleanup() {
    echo "[$(date)] worldmodel-train teardown"
    kill "${SIDECAR_PID}" 2>/dev/null || true
}
trap cleanup EXIT

# ── Run the overfit/train driver (in-process; model loaded ONCE) ────────────
# The driver is the same code the CPU-tiny test proves; here it runs the real
# corpus on the H200s.  --token-root left default (the real TOKEN_ROOT).
SHOT_ARG=""
if [[ -n "${WM_TRAIN_SHOTS}" ]]; then
    SHOT_ARG="--shots ${WM_TRAIN_SHOTS}"
fi

# shellcheck disable=SC2086
${PY} -m imas_ambix.worldmodel.train \
    ${SHOT_ARG} \
    --steps "${WM_STEPS}" \
    --n-steps "${WM_N_STEPS}" \
    --context-steps "${WM_CONTEXT_STEPS}" \
    --lr "${WM_LR}"
EXIT_CODE=$?

echo "[$(date)] worldmodel-train finished (exit=${EXIT_CODE})"
exit "${EXIT_CODE}"
