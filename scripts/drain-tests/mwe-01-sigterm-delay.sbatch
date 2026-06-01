#!/bin/bash
#SBATCH --job-name=mwe01-sigterm
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --account=grpa
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --signal=B:USR1@60
#SBATCH --output=/work/projects/imas_gpu/logs/mwe01-%j.out
#SBATCH --error=/work/projects/imas_gpu/logs/mwe01-%j.err

# MWE-01: NCCL SIGTERM Blind-Window Measurement (SAFE)
# Risk: NONE — self-terminating, no scancel required.
# Expected: PASS in ~5 minutes, reports blind window per collective.

set -uo pipefail

export TMPDIR=/scratch_local/${SLURM_JOB_ID}
mkdir -p "${TMPDIR}" || export TMPDIR=/tmp
mkdir -p /work/projects/imas_gpu/logs
mkdir -p /work/projects/imas_gpu/stops

export DRAIN_TEST_LOG_DIR=/work/projects/imas_gpu/logs
export DRAIN_TEST_STOP_DIR=/work/projects/imas_gpu/stops

# Three-layer defence (safe test: we want clean exit)
export AMBIX_STOP_FILE=/work/projects/imas_gpu/stops/${SLURM_JOB_ID}.stop
rm -f "${AMBIX_STOP_FILE}"

_on_sigusr1() {
    echo "[sbatch] SIGUSR1: approaching time limit → touching stop-file" >&2
    touch "${AMBIX_STOP_FILE}"
}
trap _on_sigusr1 USR1

# Python interpreter — override via PYTHON_VENV env var
PYTHON="${PYTHON_VENV:-/work/projects/imas_gpu/mast-tokens/v1/open-magvit2/.venv}/bin/python"
TORCHRUN="${PYTHON_VENV:-/work/projects/imas_gpu/mast-tokens/v1/open-magvit2/.venv}/bin/torchrun"

echo "=== MWE-01 SIGTERM Blind-Window ==="
echo "Job: ${SLURM_JOB_ID}  Node: ${SLURMD_NODENAME}"
echo "Python: ${PYTHON}"
echo "Log dir: ${DRAIN_TEST_LOG_DIR}"

${TORCHRUN} \
    --nproc_per_node=4 \
    --master_addr="${SLURMD_NODENAME}" \
    --master_port=29500 \
    "/home/ITER/mcintos/Code/imas-ambix/scripts/drain-tests/mwe-01-sigterm-delay.py"

echo "=== MWE-01 COMPLETE ==="
