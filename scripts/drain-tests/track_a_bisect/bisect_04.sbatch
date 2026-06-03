#!/bin/bash
# ⚠️  DELIBERATE DRAIN TEST — runs ONLY inside a coordinated admin window
# ⚠️  Admin must be present in case of unexpected drain.
#
# TRACK A BISECT-04: single_rank_only (NEGATIVE CONTROL — expected: NO drain)
# Single process, CUDA context + 256 MB allocation held, no NCCL, infinite loop.
# Tests whether holding a GPU allocation makes a process D-state-capable without
# an active blocking NCCL collective.
# Expected: NO DRAIN — process is SIGKILL-able; SLURM terminates cleanly.
#
# ⚠️  --nproc_per_node=1 is intentional — single-rank, no NCCL init.

#SBATCH --job-name=track-a-bisect04
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --account=grpa
#SBATCH --nodelist=98dci4-gpu-0003
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --output=/work/projects/imas_gpu/logs/track-a-bisect04-%j.out
#SBATCH --error=/work/projects/imas_gpu/logs/track-a-bisect04-%j.err

set -uo pipefail

export TMPDIR=/scratch_local/${SLURM_JOB_ID}
mkdir -p "${TMPDIR}"
mkdir -p /work/projects/imas_gpu/logs

export DRAIN_TEST_LOG_DIR=/work/projects/imas_gpu/logs
# Hang for 600s so SLURM --time=00:15:00 fires and attempts termination
export HANG_SECONDS=600

TORCHRUN="${PYTHON_VENV:-/work/projects/imas_gpu/mast-tokens/v1/open-magvit2/.venv}/bin/torchrun"

echo "=== TRACK A BISECT-04: single_rank_only (negative control) ==="
echo "Job: ${SLURM_JOB_ID}  Node: ${SLURMD_NODENAME}"
echo "Expected: NO DRAIN — CUDA held but no blocking NCCL collective"

${TORCHRUN} \
    --nproc_per_node=1 \
    --master_addr="${SLURMD_NODENAME}" \
    --master_port=29500 \
    "/home/ITER/mcintos/Code/imas-ambix/scripts/drain-tests/track_a_bisect/bisect_04_single_rank_only.py"

echo "=== BISECT-04 ENDED ==="
echo "Job ended. ADMIN: confirm node is NOT drained. If drained unexpectedly: gpu-reset + resume."
