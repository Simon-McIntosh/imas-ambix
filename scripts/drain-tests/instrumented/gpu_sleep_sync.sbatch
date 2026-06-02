#!/bin/bash
#SBATCH --job-name=drain-sleepsync
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --account=grpa
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:04:00
#SBATCH --output=/work/projects/imas_gpu/logs/drain-sleepsync-%j.out
#SBATCH --error=/work/projects/imas_gpu/logs/drain-sleepsync-%j.err
#
# DECISIVE SUSTAINED-DRAIN MWE. Single GPU, single process, NO NCCL.
# torch.cuda._sleep(~20yr) + synchronize -> permanent D-state in nvidia driver
# -> SLURM --time SIGTERM at 04:00 cannot reap -> DRAIN ~05:00.
# Run ONLY with admin standby. Recovery: nvidia-smi -i <id> --gpu-reset; resume.
set -uo pipefail
export TMPDIR=/scratch_local/${SLURM_JOB_ID}
mkdir -p "${TMPDIR}" || export TMPDIR=/tmp
VENV="${PYTHON_VENV:-/work/projects/imas_gpu/mast-tokens/v1/open-magvit2/.venv}"
echo "=== drain-sleepsync === job=${SLURM_JOB_ID} node=${SLURMD_NODENAME} start=$(date -Is)"
echo "=== EXPECT permanent D-state -> --time SIGTERM 04:00 -> unkillable -> DRAIN ~05:00 ==="
"${VENV}/bin/python" /home/ITER/mcintos/Code/imas-ambix/scripts/drain-tests/instrumented/gpu_sleep_sync.py
echo "=== drain-sleepsync exit $? at $(date -Is) (clean exit = no wedge) ==="
