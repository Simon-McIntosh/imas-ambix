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

# MWE-05 v7: NCCL Cross-Group / Ring Deadlock — D-State Drain Reproducer
# Risk: POSSIBLY DRAIN THE NODE (if NCCL blocking wait is D-state).
#   DRAIN_MODE=cross_group: rank 0 on comm_a, ranks 1-3 on comm_b ->
#     no cross-group matching -> all 4 blocked alive -> D-state?
#   DRAIN_MODE=ring_deadlock: all ranks recv before send -> ring deadlock
#     via P2P NVLink -> all 4 blocked -> D-state?
# If S-state: job exits cleanly and reports no drain.
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
export DRAIN_MODE="${DRAIN_MODE:-cross_group}"
export AUTO_SCANCEL_DELAY="${AUTO_SCANCEL_DELAY:-2.0}"

TORCHRUN="${PYTHON_VENV:-/work/projects/imas_gpu/mast-tokens/v1/open-magvit2/.venv}/bin/torchrun"

echo "=== MWE-05 v7 NCCL Deadlock: mode=${DRAIN_MODE} ==="
echo "Job: ${SLURM_JOB_ID}  Node: ${SLURMD_NODENAME}"
echo "Auto-scancel: ${AUTO_SCANCEL_DELAY}s after READY"
echo ""
echo "cross_group: rank 0 on comm_a, ranks 1-3 on comm_b -> no cross-group match -> deadlock"
echo "ring_deadlock: all ranks recv-before-send -> P2P NVLink ring deadlock"
echo ""
echo "After drain (if triggered), notify admin:"
echo "  nvidia-smi -i 0,1,2,3 --gpu-reset"
echo "  scontrol update nodename=${SLURMD_NODENAME} state=resume reason=''"
echo ""

${TORCHRUN} \
    --nproc_per_node=4 \
    --master_addr="${SLURMD_NODENAME}" \
    --master_port=29507 \
    "/home/ITER/mcintos/Code/imas-ambix/scripts/drain-tests/mwe-05-ddp-backward.py"

echo "=== MWE-05 COMPLETE (exited cleanly — no drain triggered: mode=${DRAIN_MODE}) ==="
