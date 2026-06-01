#!/bin/bash
#SBATCH --job-name=mwe05-drain
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --account=grpa
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/work/projects/imas_gpu/logs/mwe05-%j.out
#SBATCH --error=/work/projects/imas_gpu/logs/mwe05-%j.err

# MWE-05 v6: DDP/NCCL Mutual Communicator Deadlock — Exact Event #4 Replica
# Risk: DESIGNED TO DRAIN THE NODE — GPU reset + admin resume required.
#
# Mechanism: rank 0 calls extra dist.all_reduce (comm_PG) after DDP backward.
#   Ranks 1-3 proceed to step N+1 backward -> DDP gradient all_reduce (comm_DDP)
#   waits for rank 0. TWO-WAY DEADLOCK: all 4 ranks alive but stuck in
#   different NCCL communicators. No disconnect event. All 4 enter D-state
#   via NVLink ioctl. SIGTERM undeliverable. SIGKILL fails. Drain.
#
# RECOVERY (admin commands after drain confirmed):
#   nvidia-smi -i 0,1,2,3 --gpu-reset
#   scontrol update nodename=98dci4-gpu-0003 state=resume reason=""
#
# NO three-layer defence, NO stop-file, NO signal handler — INTENTIONAL.
# Do not add --signal=B:USR1 here; we want to expose the drain mechanism.

set -uo pipefail

export TMPDIR=/scratch_local/${SLURM_JOB_ID}
mkdir -p "${TMPDIR}" || export TMPDIR=/tmp
mkdir -p /work/projects/imas_gpu/logs

export DRAIN_TEST_LOG_DIR=/work/projects/imas_gpu/logs
export AUTO_SCANCEL_DELAY="${AUTO_SCANCEL_DELAY:-2.0}"

TORCHRUN="${PYTHON_VENV:-/work/projects/imas_gpu/mast-tokens/v1/open-magvit2/.venv}/bin/torchrun"

echo "=== MWE-05 v6 DDP/NCCL Mutual Communicator Deadlock — Event #4 Exact Replica ==="
echo "Job: ${SLURM_JOB_ID}  Node: ${SLURMD_NODENAME}"
echo "*** DESIGNED TO DRAIN THE NODE ***"
echo "Auto-scancel: ${AUTO_SCANCEL_DELAY}s after READY"
echo ""
echo "Expected sequence:"
echo "  1. DDP model + NCCL init (warm-up 2 steps) — ~10-15s"
echo "  2. Step N: all 4 ranks do clean DDP backward (comm_DDP, all synced)"
echo "  3. READY: rank 0 enters extra dist.all_reduce (comm_PG)"
echo "  4. Ranks 1-3 start step N+1: backward -> DDP gradient all_reduce (comm_DDP)"
echo "  5. TWO-WAY DEADLOCK: comm_PG waits for ranks 1-3; comm_DDP waits for rank 0"
echo "  6. All 4 D-state via NVLink ioctl"
echo "  7. Auto-scancel fires ${AUTO_SCANCEL_DELAY}s after READY"
echo "  8. SIGTERM -> all 4 D-state -> SIGKILL -> all still D-state"
echo "  9. UnkillableStepTimeout -> Kill task failed -> DRAIN"
echo ""
echo "After drain, notify admin:"
echo "  nvidia-smi -i 0,1,2,3 --gpu-reset"
echo "  scontrol update nodename=${SLURMD_NODENAME} state=resume reason=''"
echo ""

${TORCHRUN} \
    --nproc_per_node=4 \
    --master_addr="${SLURMD_NODENAME}" \
    --master_port=29507 \
    "/home/ITER/mcintos/Code/imas-ambix/scripts/drain-tests/mwe-05-ddp-backward.py"

echo "=== MWE-05 COMPLETE (exited cleanly — no drain triggered) ==="
