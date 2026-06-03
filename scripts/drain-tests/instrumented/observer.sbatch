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
# NEW (§3 dual-half capture): also captures SLURM-side state —
#   SLURM config (UnkillableStepTimeout, KillWait) at startup
#   scontrol node state every 2 s (State + Reason, including drain transition)
#   slurmd log tail (best-effort; empty if not readable by this user)
#   Cgroup PID enumeration every 1 s for TARGET_JOBID (state + wchan per PID)
#
# Submit FIRST, wait until RUNNING, then submit the workload.
# Optionally pass the workload job ID as an argument for cgroup capture:
#   sbatch observer.sbatch <TARGET_JOBID>
# Or set it after submission:
#   scontrol update jobid=<observer_jobid> comment=<target_jobid>
#   (then resubmit once TARGET_JOBID is known)
set -uo pipefail
export OBS_OUT_DIR=/work/projects/imas_gpu/logs
export OBS_INTERVAL="${OBS_INTERVAL:-0.5}"
export OBS_DURATION="${OBS_DURATION:-2100}"
# OBSERVER_JOBID is exported so observe_state.sh can use it for output file naming.
export OBSERVER_JOBID="${SLURM_JOB_ID}"
# TARGET_JOBID: optional first argument — the workload job to track in cgroup capture.
# If not provided, cgroup enumeration is skipped (all other streams still run).
TARGET_JOBID="${1:-}"
echo "=== drain observer === job=${SLURM_JOB_ID} node=${SLURMD_NODENAME} start=$(date -Is)"
echo "=== target_jobid=${TARGET_JOBID:-<none>} ==="
bash /home/ITER/mcintos/Code/imas-ambix/scripts/drain-tests/instrumented/observe_state.sh "${TARGET_JOBID}"
echo "=== observer exit $? at $(date -Is) ==="
