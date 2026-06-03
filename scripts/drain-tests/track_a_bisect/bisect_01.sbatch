#!/bin/bash
# ⚠️  DELIBERATE DRAIN TEST — runs ONLY inside a coordinated admin window
# ⚠️  This job MAY drain 98dci4-gpu-0003. Admin must be present to resume.
# ⚠️  window-repeat-count=5: run verbatim up to 5 times; admin gpu-reset between each
#
# TRACK A BISECT-01: sync_second_call
# Change from v1: blind-window all_reduce on ALL ranks (second call synchronized).
# Expected: NO DRAIN — this mirrors the 928ed0d fix.
# If no drain: removing the second-call mismatch alone is sufficient.

#SBATCH --job-name=track-a-bisect01
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --account=grpa
#SBATCH --nodelist=98dci4-gpu-0003
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --output=/work/projects/imas_gpu/logs/track-a-bisect01-%j.out
#SBATCH --error=/work/projects/imas_gpu/logs/track-a-bisect01-%j.err

set -uo pipefail

export TMPDIR=/scratch_local/${SLURM_JOB_ID}
mkdir -p "${TMPDIR}"
mkdir -p /work/projects/imas_gpu/logs

export DRAIN_TEST_LOG_DIR=/work/projects/imas_gpu/logs

TORCHRUN="${PYTHON_VENV:-/work/projects/imas_gpu/mast-tokens/v1/open-magvit2/.venv}/bin/torchrun"

echo "=== TRACK A BISECT-01: sync_second_call ==="
echo "Job: ${SLURM_JOB_ID}  Node: ${SLURMD_NODENAME}"
echo "Expected: NO DRAIN (second-call mismatch removed)"

${TORCHRUN} \
    --nproc_per_node=4 \
    --master_addr="${SLURMD_NODENAME}" \
    --master_port=29500 \
    "$(dirname "$0")/bisect_01_sync_second_call.py"

echo "=== BISECT-01 ENDED ==="
echo "Job ended. ADMIN: run nvidia-smi --gpu-reset, then scontrol resume."
