#!/bin/bash
# ⚠️  DELIBERATE DRAIN TEST — runs ONLY inside a coordinated admin window
# ⚠️  This job WILL drain 98dci4-gpu-0003. Admin must be present to resume.
# ⚠️  window-repeat-count=5: run verbatim up to 5 times; admin gpu-reset between each
#
# TRACK A BISECT-02: sync_first_call (NEGATIVE CONTROL — see note)
# ⚠️  NOTE: In v1, the first (timing) all_reduce is ALREADY symmetric across
#     all ranks. This variant makes no structural change — it is logically
#     identical to frozen v1. Expected: DRAINS at the same rate as frozen v1.
#     Purpose: validates that "first call asymmetry" is not a separable axis;
#     confirms the mismatch lives entirely in the second call.
#     Compare drain rate/timing against frozen v1 runs.

#SBATCH --job-name=track-a-bisect02
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --account=grpa
#SBATCH --nodelist=98dci4-gpu-0003
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --output=/work/projects/imas_gpu/logs/track-a-bisect02-%j.out
#SBATCH --error=/work/projects/imas_gpu/logs/track-a-bisect02-%j.err

set -uo pipefail

export TMPDIR=/scratch_local/${SLURM_JOB_ID}
mkdir -p "${TMPDIR}"
mkdir -p /work/projects/imas_gpu/logs

export DRAIN_TEST_LOG_DIR=/work/projects/imas_gpu/logs

TORCHRUN="${PYTHON_VENV:-/work/projects/imas_gpu/mast-tokens/v1/open-magvit2/.venv}/bin/torchrun"

echo "=== TRACK A BISECT-02: sync_first_call (logically identical to frozen v1) ==="
echo "Job: ${SLURM_JOB_ID}  Node: ${SLURMD_NODENAME}"
echo "Expected: DRAINS (mismatch in second call unchanged)"

${TORCHRUN} \
    --nproc_per_node=4 \
    --master_addr="${SLURMD_NODENAME}" \
    --master_port=29500 \
    "$(dirname "$0")/bisect_02_sync_first_call.py"

echo "=== BISECT-02 ENDED (unexpected clean exit) ==="
echo "Job ended. ADMIN: run nvidia-smi --gpu-reset, then scontrol resume."
