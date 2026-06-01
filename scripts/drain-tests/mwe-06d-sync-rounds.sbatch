#!/bin/bash
#SBATCH --job-name=mwe06d-sync
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --account=grpa
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=/work/projects/imas_gpu/logs/mwe06d-%j.out
#SBATCH --error=/work/projects/imas_gpu/logs/mwe06d-%j.err

# MWE-06d: NCCL Teardown D-State — Synchronous Round Variant
# ===========================================================
# Diagnostic test to determine if torch.cuda.synchronize() after each round
# changes the timing from <1ms (async) to ~1000ms (synchronous NVLink work).
#
# Event #4 had ~156s init + ~1000ms/round timing (actual NVLink transfers).
# MWE-06 had ~0.1ms/round timing (async CUDA queue submission only).
# MWE-06d adds synchronize() to force actual execution and measure timing.
#
# RISK ASSESSMENT:
#   If rounds are FAST (<10ms): teardown hangs 600s in S-state → NO DRAIN
#   If rounds are SLOW (>100ms): drain MAY occur → plan 30 min admin recovery
#
# Recovery if drain occurs:
#   nvidia-smi -i 0,1,2,3 --gpu-reset
#   scontrol update nodename=98dci4-gpu-0003 state=resume reason=""

set -euo pipefail

export TMPDIR=/scratch_local/${SLURM_JOB_ID}
mkdir -p "${TMPDIR}" || export TMPDIR=/tmp

export DRAIN_TEST_LOG_DIR=/work/projects/imas_gpu/logs
export NROUNDS="${NROUNDS:-5}"
export BUFFER_MB="${BUFFER_MB:-256}"

VENV="${PYTHON_VENV:-/work/projects/imas_gpu/mast-tokens/v1/open-magvit2/.venv}"
TORCHRUN="${VENV}/bin/torchrun"
SCRIPT="/home/ITER/mcintos/Code/imas-ambix/scripts/drain-tests/mwe-06d-sync-rounds.py"

echo "=== MWE-06d NCCL Synchronous Round Variant ==="
echo "Job: ${SLURM_JOB_ID}  Node: ${SLURMD_NODENAME}"
echo "Rounds: ${NROUNDS}  Buffer: ${BUFFER_MB}MB"
echo "Purpose: measure per-round timing to determine if NVLink actually executes"
echo "Expected: FAST rounds (<10ms) → NCCL 2.21.5 mitigates, no drain possible"
echo ""

${TORCHRUN} \
    --nproc_per_node=4 \
    --master_port=29511 \
    "${SCRIPT}"

EXIT_CODE=$?
echo "=== MWE-06d EXIT CODE: ${EXIT_CODE} ==="
echo "=== Clean exit — no drain (confirming NCCL 2.21.5 mitigation) ==="
