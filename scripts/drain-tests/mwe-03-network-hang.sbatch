#!/bin/bash
#SBATCH --job-name=mwe03-nethang
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --account=grpa
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/work/projects/imas_gpu/logs/mwe03-%j.out
#SBATCH --error=/work/projects/imas_gpu/logs/mwe03-%j.err

# MWE-03: Network Black-Hole D-State Hang
# Risk: UNCERTAIN DRAIN — depends on firewall mode (DROP vs REJECT).
# On betelgeuse (DROP): likely D-state → node drain (scontrol resume only).
# Expected recovery: scontrol update nodename=98dci4-gpu-0003 state=resume reason=""
#
# No three-layer defence: intentionally missing so scancel exposes the mechanism.
# Do NOT add --signal=B:USR1 here.
#
# After submitting: wait for "ATTEMPTING TCP CONNECT", then scancel <jobid>.

set -uo pipefail

export TMPDIR=/scratch_local/${SLURM_JOB_ID}
mkdir -p "${TMPDIR}" || export TMPDIR=/tmp
mkdir -p /work/projects/imas_gpu/logs

export DRAIN_TEST_LOG_DIR=/work/projects/imas_gpu/logs
export TARGET_HOST="${TARGET_HOST:-103.21.244.0}"
export TARGET_PORT="${TARGET_PORT:-443}"
export WAIT_BEFORE="${WAIT_BEFORE:-15}"

PYTHON="${PYTHON_VENV:-/work/projects/imas_gpu/mast-tokens/v1/open-magvit2/.venv}/bin/python"

echo "=== MWE-03 Network Hang Test ==="
echo "Job: ${SLURM_JOB_ID}  Node: ${SLURMD_NODENAME}"
echo "Target: ${TARGET_HOST}:${TARGET_PORT}"
echo "AFTER 'ATTEMPTING URLLIB DOWNLOAD' appears → scancel ${SLURM_JOB_ID}"
echo "Then check: sinfo -N -n ${SLURMD_NODENAME} --noheader -o '%T'"

"${PYTHON}" "/home/ITER/mcintos/Code/imas-ambix/scripts/drain-tests/mwe-03-network-hang.py"

echo "=== MWE-03 COMPLETE (process was killable — no drain) ==="
