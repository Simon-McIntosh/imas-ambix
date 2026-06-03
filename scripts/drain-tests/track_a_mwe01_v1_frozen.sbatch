#!/bin/bash
# ⚠️  DELIBERATE DRAIN TEST — runs ONLY inside a coordinated admin window
# ⚠️  This job WILL drain 98dci4-gpu-0003. Admin must be present to resume.
# ⚠️  window-repeat-count=5: run verbatim up to 5 times; admin gpu-reset between each
#
# TRACK A: FROZEN MWE-01 v1 — verbatim drain artifact reproduction
# Source: job 1209813 (2026-06-01) — collective mismatch → node drain
# Purpose: confirm the frozen pre-fix code reproduces the drain on a clean node
# Scope:   drain-window-campaign §4, Phase 1 (verbatim runs)
#
# DO NOT add STOP-FILE, SIGUSR1 trap, or any protective mechanism.
# The absence of drain protection IS the test condition.

#SBATCH --job-name=track-a-frozen
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --account=grpa
#SBATCH --nodelist=98dci4-gpu-0003
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --output=/work/projects/imas_gpu/logs/track-a-frozen-%j.out
#SBATCH --error=/work/projects/imas_gpu/logs/track-a-frozen-%j.err

set -uo pipefail

export TMPDIR=/scratch_local/${SLURM_JOB_ID}
mkdir -p "${TMPDIR}"
mkdir -p /work/projects/imas_gpu/logs

export DRAIN_TEST_LOG_DIR=/work/projects/imas_gpu/logs
# Disable the NCCL heartbeat watchdog so SLURM --time is the killer (not _exit()).
# Run 1 showed the watchdog fires at 600s and calls _exit() cleanly, preventing drain.
# Disabling it forces SLURM to fire into a live hung process — matches 1209813 condition.
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=99999
export TORCH_NCCL_ASYNC_ERROR_HANDLING=0

TORCHRUN="${PYTHON_VENV:-/work/projects/imas_gpu/mast-tokens/v1/open-magvit2/.venv}/bin/torchrun"

echo "=== TRACK A: FROZEN MWE-01 v1 (pre-928ed0d) ==="
echo "Job: ${SLURM_JOB_ID}  Node: ${SLURMD_NODENAME}"
echo "torchrun: ${TORCHRUN}"
echo "Log dir: ${DRAIN_TEST_LOG_DIR}"
echo "EXPECTED OUTCOME: collective mismatch → NCCL D-state → node drain"

${TORCHRUN} \
    --nproc_per_node=4 \
    --master_addr="${SLURMD_NODENAME}" \
    --master_port=29500 \
    "/home/ITER/mcintos/Code/imas-ambix/scripts/drain-tests/track_a_mwe01_v1_frozen.py"

echo "=== TRACK A FROZEN ENDED (unexpectedly clean — no drain) ==="
echo "Job ended. ADMIN: run nvidia-smi --gpu-reset, then scontrol resume."
