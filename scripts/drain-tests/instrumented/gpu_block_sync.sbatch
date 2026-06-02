#!/bin/bash
#SBATCH --job-name=drain-blocksync
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --account=grpa
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:03:00
#SBATCH --output=/work/projects/imas_gpu/logs/drain-blocksync-%j.out
#SBATCH --error=/work/projects/imas_gpu/logs/drain-blocksync-%j.err
#
# SUSTAINED-DRAIN validation MWE. Single GPU, no NCCL. Blocking-event sync on an
# infinite kernel -> permanent D-state in nvidia driver -> --time SIGKILL at
# 03:00 cannot reap -> DRAIN ~04:00. Run ONLY with admin standby.
# Recovery: nvidia-smi -i <id> --gpu-reset (or reboot); scontrol ... state=resume.
set -uo pipefail
export TMPDIR=/scratch_local/${SLURM_JOB_ID}
mkdir -p "${TMPDIR}" || export TMPDIR=/tmp
VENV="${PYTHON_VENV:-/work/projects/imas_gpu/mast-tokens/v1/open-magvit2/.venv}"
echo "=== drain-blocksync === job=${SLURM_JOB_ID} node=${SLURMD_NODENAME} start=$(date -Is)"
"${VENV}/bin/python" /home/ITER/mcintos/Code/imas-ambix/scripts/drain-tests/instrumented/gpu_block_sync.py
echo "=== drain-blocksync exit $? at $(date -Is) (clean exit = no wedge) ==="
