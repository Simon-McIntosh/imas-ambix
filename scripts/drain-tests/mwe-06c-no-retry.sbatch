#!/bin/bash
#SBATCH --job-name=mwe06c-drain
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --account=grpa
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:08:00
#SBATCH --output=/work/projects/imas_gpu/logs/mwe06c-%j.out
#SBATCH --error=/work/projects/imas_gpu/logs/mwe06c-%j.err

# MWE-06c: NCCL Teardown D-State — No Elastic Retry + Optimal SLURM Timing
# RISK: CRITICAL — DESIGNED TO DRAIN THE NODE
#
# Root cause of prior non-drains:
#
#   MWE-06 (job 1209930, --time=20min):
#     PyTorch ProcessGroupNCCL watchdog fired SIGABRT at exactly 10:15
#     (600s after teardown). Process exited cleanly before SLURM limit.
#     → NO DRAIN because PyTorch watchdog fires before SLURM does.
#
#   MWE-06b (job 1209935, --time=5min):
#     torchrun elastic retry (default --max-restarts=100) launched 226
#     worker groups after workers failed. Each retry lasted ~1s (NCCL init
#     failure). SLURM fired during retry cycle ~226 — process was in R/S-state
#     (not D-state) → SIGTERM handled cleanly → NO DRAIN.
#
# Fix applied in MWE-06c:
#   --max-restarts=0  : torchrun does ONE attempt and exits on failure.
#                       The teardown D-state persists uninterrupted for 600s.
#   --time=00:08:00   : SLURM fires at 8 min, 7.5 min into the 600s D-state
#                       window. PyTorch watchdog fires at 10:30. SLURM wins.
#
# Expected timeline:
#   ~0:30   10 rounds complete, dist.barrier()/destroy() -> ncclCommAbort D-state
#   ~8:00   SLURM time limit fires -> SIGTERM -> D-state (ignored by kernel)
#   ~8:30   SLURM KillWait (30s) -> SIGKILL -> D-state (still unkillable)
#   ~9:30   UnkillableStepTimeout (60s) -> "Kill task failed" -> DRAIN
#
# Recovery after drain:
#   nvidia-smi -i 0,1,2,3 --gpu-reset
#   scontrol update nodename=98dci4-gpu-0003 state=resume reason=""
#
# NOTE: No STOP-FILE — intentional. This is the drain reproducer.

set -euo pipefail

export TMPDIR=/scratch_local/${SLURM_JOB_ID}
mkdir -p "${TMPDIR}" || export TMPDIR=/tmp

export DRAIN_TEST_LOG_DIR=/work/projects/imas_gpu/logs
export NROUNDS="${NROUNDS:-10}"
export BUFFER_MB="${BUFFER_MB:-256}"

VENV="${PYTHON_VENV:-/work/projects/imas_gpu/mast-tokens/v1/open-magvit2/.venv}"
TORCHRUN="${VENV}/bin/torchrun"
SCRIPT="/home/ITER/mcintos/Code/imas-ambix/scripts/drain-tests/mwe-06-nccl-teardown.py"

echo "=== MWE-06c NCCL Teardown D-State (No Elastic Retry + Timed Kill) ==="
echo "Job: ${SLURM_JOB_ID}  Node: ${SLURMD_NODENAME}"
echo "Rounds: ${NROUNDS}  Buffer: ${BUFFER_MB}MB"
echo "SLURM time limit: 8 min  D-state window: 0:30-10:30  SLURM fires at 8:00"
echo ""
echo "Key changes from MWE-06b:"
echo "  --max-restarts=0  (no elastic retry - one attempt only)"
echo "  --time=00:08:00   (fires 7.5 min into D-state, before PyTorch watchdog at 10:30)"
echo ""
echo "Expected: ncclCommAbort D-state -> SLURM SIGTERM -> SIGKILL -> UnkillableStepTimeout -> DRAIN"
echo ""
echo "Recovery after drain:"
echo "  nvidia-smi -i 0,1,2,3 --gpu-reset"
echo "  scontrol update nodename=98dci4-gpu-0003 state=resume reason=''"

${TORCHRUN} \
    --nproc_per_node=4 \
    --max-restarts=0 \
    --master_port=29510 \
    "${SCRIPT}"

echo "=== MWE-06c COMPLETE (unexpected clean exit — no drain) ==="
