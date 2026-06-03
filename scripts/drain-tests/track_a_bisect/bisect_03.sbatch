#!/bin/bash
# ⚠️  DELIBERATE DRAIN TEST — runs ONLY inside a coordinated admin window
# ⚠️  This job WILL drain 98dci4-gpu-0003. Admin must be present to resume.
# ⚠️  window-repeat-count=5: run verbatim up to 5 times; admin gpu-reset between each
#
# TRACK A BISECT-03: reduce_buffer (1 MB vs 256 MB, mismatch preserved)
# Isolates whether large buffer is a necessary condition for D-state wedge.
# Expected: DRAINS (D-state is triggered by collective count mismatch,
#           not buffer size — small buffer completes faster but still mismatches).

#SBATCH --job-name=track-a-bisect03
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --account=grpa
#SBATCH --nodelist=98dci4-gpu-0003
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --output=/work/projects/imas_gpu/logs/track-a-bisect03-%j.out
#SBATCH --error=/work/projects/imas_gpu/logs/track-a-bisect03-%j.err

set -uo pipefail

export TMPDIR=/scratch_local/${SLURM_JOB_ID}
mkdir -p "${TMPDIR}"
mkdir -p /work/projects/imas_gpu/logs

export DRAIN_TEST_LOG_DIR=/work/projects/imas_gpu/logs

TORCHRUN="${PYTHON_VENV:-/work/projects/imas_gpu/mast-tokens/v1/open-magvit2/.venv}/bin/torchrun"

echo "=== TRACK A BISECT-03: reduce_buffer (1 MB, mismatch preserved) ==="
echo "Job: ${SLURM_JOB_ID}  Node: ${SLURMD_NODENAME}"
echo "Expected: DRAINS (buffer size is not the determining factor)"

${TORCHRUN} \
    --nproc_per_node=4 \
    --master_addr="${SLURMD_NODENAME}" \
    --master_port=29500 \
    "/home/ITER/mcintos/Code/imas-ambix/scripts/drain-tests/track_a_bisect/bisect_03_reduce_buffer.py"

echo "=== BISECT-03 ENDED ==="
echo "Job ended. ADMIN: run nvidia-smi --gpu-reset, then scontrol resume."
