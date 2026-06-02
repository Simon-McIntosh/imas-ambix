#!/bin/bash
#SBATCH --job-name=drain-lockdead
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --account=grpa
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:03:00
#SBATCH --output=/work/projects/imas_gpu/logs/drain-lockdead-%j.out
#SBATCH --error=/work/projects/imas_gpu/logs/drain-lockdead-%j.err
#
# Attempt a permanent NVIDIA-driver rwlock deadlock: 1 holder (infinite kernel +
# cudaDeviceReset, holds write-lock) + N readers (cudaMalloc/free, block behind it),
# all on GPU 0. Targets a sustained D-state -> --time SIGKILL cannot reap -> DRAIN.
# Run ONLY with admin standby; recovery may need a reboot.
set -uo pipefail
export TMPDIR=/scratch_local/${SLURM_JOB_ID}
mkdir -p "${TMPDIR}" || export TMPDIR=/tmp
VENV="${PYTHON_VENV:-/work/projects/imas_gpu/mast-tokens/v1/open-magvit2/.venv}"
PY="${VENV}/bin/python"
SCRIPT=/home/ITER/mcintos/Code/imas-ambix/scripts/drain-tests/instrumented/gpu_lock_deadlock.py
echo "=== drain-lockdead === job=${SLURM_JOB_ID} node=${SLURMD_NODENAME} start=$(date -Is)"
GPU=0 ROLE=holder "$PY" "$SCRIPT" &
sleep 8   # let the holder take the lock first
for i in 1 2 3 4 5 6; do GPU=0 ROLE=reader "$PY" "$SCRIPT" & done
wait
echo "=== drain-lockdead exit at $(date -Is) (clean exit = no wedge) ==="
