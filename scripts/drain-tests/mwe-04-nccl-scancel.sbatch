#!/bin/bash
#SBATCH --job-name=mwe04-nccl
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --account=grpa
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:10:00
#SBATCH --output=/work/projects/imas_gpu/logs/mwe04-%j.out
#SBATCH --error=/work/projects/imas_gpu/logs/mwe04-%j.err

# MWE-04: Simple NCCL All-Reduce + Scancel
# Risk: LOW DRAIN RISK — empirically confirmed R-state (no drain) on this cluster.
# Expected: node stays idle after scancel.
# If drain occurs: unexpected; confirms NCCL changed to D-state.
#
# No stop-file or three-layer defence: intentionally missing to test mechanism.
# Watchdog self-terminates after 120 s if scancel is not issued.
#
# After submitting: wait for "READY", then scancel <jobid>.
# Check node state 70 s later.

set -uo pipefail

export TMPDIR=/scratch_local/${SLURM_JOB_ID}
mkdir -p "${TMPDIR}" || export TMPDIR=/tmp
mkdir -p /work/projects/imas_gpu/logs

export DRAIN_TEST_LOG_DIR=/work/projects/imas_gpu/logs
export BUFFER_MB="${BUFFER_MB:-512}"
export WATCHDOG_SECONDS="${WATCHDOG_SECONDS:-120}"

PYTHON="${PYTHON_VENV:-/work/projects/imas_gpu/mast-tokens/v1/open-magvit2/.venv}/bin/python"
TORCHRUN="${PYTHON_VENV:-/work/projects/imas_gpu/mast-tokens/v1/open-magvit2/.venv}/bin/torchrun"

echo "=== MWE-04 NCCL All-Reduce + Scancel ==="
echo "Job: ${SLURM_JOB_ID}  Node: ${SLURMD_NODENAME}"
echo "Buffer: ${BUFFER_MB} MB, Watchdog: ${WATCHDOG_SECONDS} s"
echo "AFTER 'READY' appears → scancel ${SLURM_JOB_ID}"
echo "Then check: sinfo -N -n ${SLURMD_NODENAME} --noheader -o '%T'"

${TORCHRUN} \
    --nproc_per_node=4 \
    --master_addr="${SLURMD_NODENAME}" \
    --master_port=29502 \
    "$(dirname "$0")/mwe-04-nccl-scancel.py"

echo "=== MWE-04 COMPLETE ==="
