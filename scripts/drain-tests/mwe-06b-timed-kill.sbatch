#!/bin/bash
#SBATCH --job-name=mwe06b-timed
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --account=grpa
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:05:00
#SBATCH --output=/work/projects/imas_gpu/logs/mwe06b-%j.out
#SBATCH --error=/work/projects/imas_gpu/logs/mwe06b-%j.err

# MWE-06b: NCCL Teardown D-State — Timed SLURM Kill During D-State Window
# RISK: CRITICAL — DESIGNED TO DRAIN THE NODE
#
# MWE-06 (job 1209930) with --time=00:20:00 did NOT drain:
#   - ncclCommAbort() hung for 600s
#   - PyTorch watchdog fired SIGABRT at 10:15
#   - Process exited cleanly before SLURM limit
#
# This variant uses --time=00:05:00 to fire SLURM DURING the D-state window:
#   ~0:30  - count-mismatch rounds complete, ncclCommAbort() enters D-state
#   ~5:00  - SLURM --time fires -> SIGTERM -> D-state process can't handle it
#   ~5:30  - SLURM sends SIGKILL -> still fails (D-state) -> "Kill task failed"
#   ~6:30  - UnkillableStepTimeout -> DRAIN
#
# Recovery after drain:
#   nvidia-smi -i 0,1,2,3 --gpu-reset
#   scontrol update nodename=98dci4-gpu-0003 state=resume reason=""
#
# NOTE: No STOP-FILE or safety layers — designed to drain to prove:
#   "SLURM time-limit during ncclCommAbort D-state causes node drain"

set -euo pipefail

export TMPDIR=/scratch_local/${SLURM_JOB_ID}
mkdir -p "${TMPDIR}" || export TMPDIR=/tmp

export DRAIN_TEST_LOG_DIR=/work/projects/imas_gpu/logs
export NROUNDS="${NROUNDS:-10}"
export BUFFER_MB="${BUFFER_MB:-256}"

VENV="${PYTHON_VENV:-/work/projects/imas_gpu/mast-tokens/v1/open-magvit2/.venv}"
TORCHRUN="${VENV}/bin/torchrun"
SCRIPT="/home/ITER/mcintos/Code/imas-ambix/scripts/drain-tests/mwe-06-nccl-teardown.py"

echo "=== MWE-06b NCCL Teardown D-State (Timed SLURM Kill) ==="
echo "Job: ${SLURM_JOB_ID}  Node: ${SLURMD_NODENAME}"
echo "Rounds: ${NROUNDS}  Buffer: ${BUFFER_MB}MB"
echo "SLURM time limit: 5 min (fires ~4m30s into ncclCommAbort D-state)"
echo ""
echo "Mechanism: count-mismatch corrupts ring -> ncclCommAbort() D-state"
echo "SLURM fires at 5min (D-state window 0:30-10:30) -> unkillable -> DRAIN"
echo ""
echo "After drain, notify admin:"
echo "  nvidia-smi -i 0,1,2,3 --gpu-reset"
echo "  scontrol update nodename=98dci4-gpu-0003 state=resume reason=''"

${TORCHRUN} \
    --nproc_per_node=4 \
    --master_port=29509 \
    "${SCRIPT}"

echo "=== MWE-06b COMPLETE (unexpected clean exit) ==="
