#!/bin/bash
#SBATCH --job-name=mwe02-stopfile
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --account=grpa
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:10:00
#SBATCH --signal=B:USR1@60
#SBATCH --output=/work/projects/imas_gpu/logs/mwe02-%j.out
#SBATCH --error=/work/projects/imas_gpu/logs/mwe02-%j.err

# MWE-02: Stop-File Clean Exit Validation (SAFE)
# Risk: NONE — self-terminating via stop-file after 30 s.
# Expected: PASS — all ranks exit 0, node stays idle.

set -uo pipefail

export TMPDIR=/scratch_local/${SLURM_JOB_ID}
mkdir -p "${TMPDIR}" || export TMPDIR=/tmp
mkdir -p /work/projects/imas_gpu/logs
mkdir -p /work/projects/imas_gpu/stops

export DRAIN_TEST_LOG_DIR=/work/projects/imas_gpu/logs
export DRAIN_TEST_STOP_DIR=/work/projects/imas_gpu/stops
export STOP_AFTER_SECONDS=30
export BUFFER_MB=256

PYTHON="${PYTHON_VENV:-/work/projects/imas_gpu/mast-tokens/v1/open-magvit2/.venv}/bin/python"
TORCHRUN="${PYTHON_VENV:-/work/projects/imas_gpu/mast-tokens/v1/open-magvit2/.venv}/bin/torchrun"

echo "=== MWE-02 Stop-File Clean Exit ==="
echo "Job: ${SLURM_JOB_ID}  Node: ${SLURMD_NODENAME}"
echo "Stop-file: ${DRAIN_TEST_STOP_DIR}/${SLURM_JOB_ID}.stop"
echo "Auto-stop in: ${STOP_AFTER_SECONDS} s"

${TORCHRUN} \
    --nproc_per_node=4 \
    --master_addr="${SLURMD_NODENAME}" \
    --master_port=29501 \
    "$(dirname "$0")/mwe-02-stopfile.py"

echo "=== MWE-02 COMPLETE ==="
