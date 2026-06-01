#!/bin/bash
#SBATCH --job-name=mwe05-drain
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --account=grpa
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/work/projects/imas_gpu/logs/mwe05-%j.out
#SBATCH --error=/work/projects/imas_gpu/logs/mwe05-%j.err

# MWE-05: DDP loss.backward() + Scancel — DRAIN REPRODUCER
# Risk: ⚠️⚠️ LIKELY DRAIN — GPU reset + admin resume required.
#
# MECHANISM: DDP backward() triggers NCCL gradient all-reduce hooks that call
# cudaStreamSynchronize() inside the NVLink driver. This enters D-state
# (TASK_UNINTERRUPTIBLE). SIGKILL cannot kill D-state processes. SLURM's
# UnkillableStepTimeout (~60 s) fires → "Kill task failed" → node drains.
#
# RECOVERY (admin commands after drain confirmed):
#   nvidia-smi -i 0,1,2,3 --gpu-reset
#   scontrol update nodename=98dci4-gpu-0003 state=resume reason=""
#
# NO three-layer defence, NO stop-file, NO signal handler — INTENTIONAL.
# Do not add --signal=B:USR1 here; we want to expose the drain mechanism.
#
# PROCEDURE:
#   1. Submit: sbatch mwe-05-ddp-backward.sbatch
#   2. Wait for "READY" in log (~30 s after start)
#   3. Immediately: scancel <jobid>
#   4. Wait 70 s, check: sinfo -N -n 98dci4-gpu-0003 --noheader -o '%T'
#   5. If "drain": mechanism confirmed. Notify admin for recovery.

set -uo pipefail

export TMPDIR=/scratch_local/${SLURM_JOB_ID}
mkdir -p "${TMPDIR}" || export TMPDIR=/tmp
mkdir -p /work/projects/imas_gpu/logs

export DRAIN_TEST_LOG_DIR=/work/projects/imas_gpu/logs
export MODEL_DEPTH="${MODEL_DEPTH:-8}"
export HIDDEN_DIM="${HIDDEN_DIM:-4096}"
export BATCH_SIZE="${BATCH_SIZE:-32}"

PYTHON="${PYTHON_VENV:-/work/projects/imas_gpu/mast-tokens/v1/open-magvit2/.venv}/bin/python"
TORCHRUN="${PYTHON_VENV:-/work/projects/imas_gpu/mast-tokens/v1/open-magvit2/.venv}/bin/torchrun"

echo "=== MWE-05 DDP Backward Drain Reproducer ==="
echo "Job: ${SLURM_JOB_ID}  Node: ${SLURMD_NODENAME}"
echo "WARNING: This test is designed to DRAIN THE NODE"
echo "Model: depth=${MODEL_DEPTH}, dim=${HIDDEN_DIM}, batch=${BATCH_SIZE}"
echo ""
echo "PROCEDURE:"
echo "  1. Wait for 'READY' in this log"
echo "  2. Immediately: scancel ${SLURM_JOB_ID}"
echo "  3. Wait 70s, check: sinfo -N -n ${SLURMD_NODENAME} --noheader -o '%T'"
echo "  4. If 'drain': nvidia-smi -i 0,1,2,3 --gpu-reset"
echo "     Then: scontrol update nodename=${SLURMD_NODENAME} state=resume reason=''"
echo ""

${TORCHRUN} \
    --nproc_per_node=4 \
    --master_addr="${SLURMD_NODENAME}" \
    --master_port=29503 \
    "/home/ITER/mcintos/Code/imas-ambix/scripts/drain-tests/mwe-05-ddp-backward.py"

echo "=== MWE-05 COMPLETE (process exited cleanly — no drain triggered) ==="
