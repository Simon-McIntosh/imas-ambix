#!/bin/bash
# T4: Minimal NCCL collective + scancel drain reproducer.
#
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  WARNING: THIS JOB WILL DRAIN 98dci4-gpu-0003 (betelgeuse)              ║
# ║  Requires admin + nvidia-smi --gpu-reset. Recovery ~20–30 min.          ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# SUBMIT:  sbatch scripts/tests/t4_nccl_scancel.sbatch
# TRIGGER: scancel <JOBID>   (wait until "Running NCCL all_reduce loop" appears)
# PASS:    Log shows "Kill task failed"; sinfo shows node DRAIN state.
#
# RECOVERY:
#   1. nvidia-smi -i 0,1 --gpu-reset
#   2. scontrol update nodename=98dci4-gpu-0003 state=resume reason=""
#
#SBATCH --job-name=ambix-t4-nccldrain
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --account=grpa
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=00:10:00
#SBATCH --output=/work/projects/imas_gpu/logs/t4-nccldrain-%j.log

# Intentionally NO --signal, NO STOP_FILE, NO SOFT_TIME_LIMIT
# This models the vulnerable (un-hardened) pattern.

set -e
export TMPDIR=/scratch_local/${SLURM_JOB_ID}
mkdir -p "${TMPDIR}"

REPO_ROOT="${SLURM_SUBMIT_DIR}"
MAGVIT2_ROOT="/work/projects/imas_gpu/mast-tokens/v1/open-magvit2"
TORCHRUN="${MAGVIT2_ROOT}/.venv/bin/torchrun"
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN

# 256 MB tensor → ~100–500 ms all_reduce → wide SIGTERM blind window
export T4_NUMEL=$((64 * 1024 * 1024))

echo "[T4] Job ${SLURM_JOB_ID} on $(hostname)"
echo "[T4] Running NCCL all_reduce loop — no hardening."
echo "[T4] To trigger drain: scancel ${SLURM_JOB_ID}"
echo "[T4] Monitor: watch -n1 \"sinfo -N -n \$(squeue -j ${SLURM_JOB_ID} -o %N -h 2>/dev/null)\""

PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" \
    "${TORCHRUN}" \
    --nproc_per_node=2 \
    --master_addr=localhost \
    --master_port=29503 \
    "${REPO_ROOT}/scripts/tests/t4_nccl_scancel_test.py"
