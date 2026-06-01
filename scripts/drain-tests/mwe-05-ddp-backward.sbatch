#!/bin/bash
#SBATCH --job-name=mwe05-drain
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --account=grpa
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=00:30:00
#SBATCH --output=/work/projects/imas_gpu/logs/mwe05-%j.out
#SBATCH --error=/work/projects/imas_gpu/logs/mwe05-%j.err

# MWE-05 v2: DDP loss.backward() + Auto-Scancel — DRAIN REPRODUCER
# Risk: ⚠️⚠️ DESIGNED TO DRAIN THE NODE — GPU reset + admin resume required.
#
# MECHANISM: DDP backward() triggers NCCL gradient all-reduce hooks that call
# cudaStreamSynchronize() inside the NVLink driver. This enters D-state
# (TASK_UNINTERRUPTIBLE). SIGKILL cannot kill D-state processes. SLURM's
# UnkillableStepTimeout (~60 s) fires → "Kill task failed" → node drains.
#
# v2 CHANGES vs v1:
#   - Model: dim=8192, depth=16, batch=256, bfloat16 → ~500ms backward
#   - AUTO-SCANCEL: rank 0 issues scancel 0.5s after READY (no manual timing)
#
# RECOVERY (admin commands after drain confirmed):
#   nvidia-smi -i 0,1,2,3 --gpu-reset
#   scontrol update nodename=98dci4-gpu-0003 state=resume reason=""
#
# NO three-layer defence, NO stop-file, NO signal handler — INTENTIONAL.
# Do not add --signal=B:USR1 here; we want to expose the drain mechanism.

set -uo pipefail

export TMPDIR=/scratch_local/${SLURM_JOB_ID}
mkdir -p "${TMPDIR}" || export TMPDIR=/tmp
mkdir -p /work/projects/imas_gpu/logs

export DRAIN_TEST_LOG_DIR=/work/projects/imas_gpu/logs
export MODEL_DEPTH="${MODEL_DEPTH:-16}"
export HIDDEN_DIM="${HIDDEN_DIM:-8192}"
export BATCH_SIZE="${BATCH_SIZE:-256}"
export AUTO_SCANCEL_DELAY="${AUTO_SCANCEL_DELAY:-0.5}"

PYTHON="${PYTHON_VENV:-/work/projects/imas_gpu/mast-tokens/v1/open-magvit2/.venv}/bin/python"
TORCHRUN="${PYTHON_VENV:-/work/projects/imas_gpu/mast-tokens/v1/open-magvit2/.venv}/bin/torchrun"

echo "=== MWE-05 v2 DDP Backward Drain Reproducer ==="
echo "Job: ${SLURM_JOB_ID}  Node: ${SLURMD_NODENAME}"
echo "*** DESIGNED TO DRAIN THE NODE ***"
echo "Model: depth=${MODEL_DEPTH}, dim=${HIDDEN_DIM}, batch=${BATCH_SIZE}, dtype=bfloat16"
echo "Auto-scancel: ${AUTO_SCANCEL_DELAY}s after READY (no manual action required)"
echo ""
echo "Expected sequence:"
echo "  1. NCCL init (~5s)"
echo "  2. First backward (~500ms) → READY"
echo "  3. Auto-scancel fires ${AUTO_SCANCEL_DELAY}s later (mid step-2 backward)"
echo "  4. D-state processes survive SIGKILL → UnkillableStepTimeout → DRAIN"
echo ""
echo "After drain, notify admin:"
echo "  nvidia-smi -i 0,1,2,3 --gpu-reset"
echo "  scontrol update nodename=${SLURMD_NODENAME} state=resume reason=''"
echo ""

${TORCHRUN} \
    --nproc_per_node=4 \
    --master_addr="${SLURMD_NODENAME}" \
    --master_port=29503 \
    "/home/ITER/mcintos/Code/imas-ambix/scripts/drain-tests/mwe-05-ddp-backward.py"

echo "=== MWE-05 COMPLETE (exited cleanly — no drain triggered) ==="
