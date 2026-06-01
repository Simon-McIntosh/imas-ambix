#!/bin/bash
#SBATCH --job-name=mwe06-teardown
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --account=grpa
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=/work/projects/imas_gpu/logs/mwe06-%j.out
#SBATCH --error=/work/projects/imas_gpu/logs/mwe06-%j.err

# MWE-06: NCCL Communicator Teardown D-State Reproducer
# RISK: CRITICAL — DESIGNED TO DRAIN THE NODE
#
# This script reproduces Event #4 (job 1209813) exactly.
# The drain occurs in ncclCommAbort() during dist.destroy_process_group().
# Expected timeline:
#   ~0-3 min:  count-mismatch rounds (NCCL ring state corruption)
#   ~3-13 min: ncclCommAbort() D-state (600s PyTorch internal timeout fires)
#   ~20 min:   SLURM --time limit -> SIGTERM -> processes unkillable -> DRAIN
#
# Recovery after drain:
#   nvidia-smi -i 0,1,2,3 --gpu-reset
#   scontrol update nodename=98dci4-gpu-0003 state=resume reason=""
#
# NOTE: No STOP-FILE or safety layers intentionally omitted — this test
# is designed to drain the node to reproduce the confirmed Event #4 mechanism.

set -euo pipefail

export TMPDIR=/scratch_local/${SLURM_JOB_ID}
mkdir -p "${TMPDIR}" || export TMPDIR=/tmp

export DRAIN_TEST_LOG_DIR=/work/projects/imas_gpu/logs
export NROUNDS="${NROUNDS:-10}"
export BUFFER_MB="${BUFFER_MB:-256}"

VENV="${PYTHON_VENV:-/work/projects/imas_gpu/mast-tokens/v1/open-magvit2/.venv}"
TORCHRUN="${VENV}/bin/torchrun"
SCRIPT="$(realpath "$(dirname "${BASH_SOURCE[0]}")/mwe-06-nccl-teardown.py")"

echo "=== MWE-06 NCCL Teardown D-State Reproducer ==="
echo "Job: ${SLURM_JOB_ID}  Node: ${SLURMD_NODENAME}"
echo "Rounds: ${NROUNDS}  Buffer: ${BUFFER_MB}MB"
echo ""
echo "Mechanism: count-mismatch all_reduce corrupts NCCL ring -> ncclCommAbort() D-state"
echo "Expected drain timeline: ~20 min after --time limit fires"
echo ""
echo "After drain, notify admin:"
echo "  nvidia-smi -i 0,1,2,3 --gpu-reset"
echo "  scontrol update nodename=98dci4-gpu-0003 state=resume reason=''"

${TORCHRUN} \
    --nproc_per_node=4 \
    --master_port=29508 \
    "${SCRIPT}"

echo "=== MWE-06 COMPLETE (unexpected clean exit — teardown did not drain) ==="
