#!/bin/bash
#SBATCH --job-name=drain-observer
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --account=grpa
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:40:00
#SBATCH --output=/work/projects/imas_gpu/logs/drain-observer-%j.out
#SBATCH --error=/work/projects/imas_gpu/logs/drain-observer-%j.err
#
# Separate-allocation observer. Requests NO GPU so it co-schedules on
# 98dci4-gpu-0003 alongside a 4-GPU workload. Survives the workload's
# SIGKILL and the node drain (drain only blocks NEW jobs); keeps sampling
# /proc state of the target ranks through the entire unkillable window.
#
# Submit FIRST, wait until RUNNING, then submit the workload.
set -uo pipefail
export OBS_OUT_DIR=/work/projects/imas_gpu/logs
export OBS_INTERVAL="${OBS_INTERVAL:-0.5}"
export OBS_DURATION="${OBS_DURATION:-2100}"
echo "=== drain observer === job=${SLURM_JOB_ID} node=${SLURMD_NODENAME} start=$(date -Is)"
bash /home/ITER/mcintos/Code/imas-ambix/scripts/drain-tests/instrumented/observe_state.sh
echo "=== observer exit $? at $(date -Is) ==="
