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

# MWE-05 v5: NCCL Collective Type Mismatch — All-Rank D-State Drain Reproducer
# Risk: ⚠️⚠️ DESIGNED TO DRAIN THE NODE — GPU reset + admin resume required.
#
# v4 finding: single D-state rank insufficient — ranks 1-3 in sleep() exit
#   cleanly on SIGTERM, NCCL detects peer disconnect, rank 0 exits → no drain.
# v5 fix: ALL ranks enter mismatched NCCL ops simultaneously:
#   rank 0 → AllReduce; ranks 1-3 → AllGather. All 4 stuck in incompatible
#   NCCL ops → all 4 D-state → none can signal peer disconnect → drain.
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

echo "=== MWE-05 v5 NCCL Type Mismatch — All-Rank D-State Drain ==="
echo "Job: ${SLURM_JOB_ID}  Node: ${SLURMD_NODENAME}"
echo "*** DESIGNED TO DRAIN THE NODE ***"
echo "Auto-scancel: ${AUTO_SCANCEL_DELAY}s after READY"
echo ""
echo "Expected sequence:"
echo "  1. NCCL init + barrier (all 4 ranks) → ~5s"
echo "  2. READY (rank 0 entering AllReduce; ranks 1-3 entering AllGather)"
echo "  3. ALL 4 ranks blocked in mismatched NCCL ops → D-state"
echo "  4. Auto-scancel fires ${AUTO_SCANCEL_DELAY}s after READY"
echo "  5. SIGTERM → all 4 D-state → SIGKILL → all still D-state"
echo "  6. UnkillableStepTimeout → Kill task failed → DRAIN"
echo ""
echo "After drain, notify admin:"
echo "  nvidia-smi -i 0,1,2,3 --gpu-reset"
echo "  scontrol update nodename=${SLURMD_NODENAME} state=resume reason=''"
echo ""

${TORCHRUN} \
    --nproc_per_node=4 \
    --master_addr="${SLURMD_NODENAME}" \
    --master_port=29503 \
    "/home/ITER/mcintos/Code/imas-ambix/scripts/drain-tests/mwe-05-ddp-backward.py"

echo "=== MWE-05 COMPLETE (exited cleanly — no drain triggered) ==="
