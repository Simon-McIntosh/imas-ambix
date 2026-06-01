#!/bin/bash
# T2: SIGTERM delivery-delay diagnostic — SAFE, no drain risk.
#
# Self-terminates: a background thread sends SIGTERM to the process after
# T2_SIGTERM_DELAY_S seconds. We measure how long until the signal handler
# fires, proving that NCCL collectives create a "blind window" where
# SIGTERM cannot be processed.
#
# SUBMIT:  sbatch scripts/tests/t2_sigterm_delay.sbatch
# OUTCOME: Job completes with exit 0; log shows SIGTERM delay in ms.
# EXPECTED RESULT: delay = ~1 full all_reduce duration (10–500 ms).
#
#SBATCH --job-name=ambix-t2-sigtermdelay
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --account=grpa
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=00:10:00
#SBATCH --signal=B:USR1@60
#SBATCH --output=/work/projects/imas_gpu/logs/t2-sigtermdelay-%j.log

set -euo pipefail
export TMPDIR=/scratch_local/${SLURM_JOB_ID}
mkdir -p "${TMPDIR}"

# Three-layer defence on the wrapper (test exits naturally, so just for safety)
mkdir -p /work/projects/imas_gpu/stops
export AMBIX_STOP_FILE=/work/projects/imas_gpu/stops/${SLURM_JOB_ID}.stop
rm -f "${AMBIX_STOP_FILE}"

_on_sigusr1() {
    echo "[T2-sbatch] SIGUSR1 → touching stop-file" >&2
    touch "${AMBIX_STOP_FILE}"
}
trap _on_sigusr1 USR1

export AMBIX_SOFT_TIME_LIMIT=480  # 8 min = 0.85 × 10 min

# ── Environment ─────────────────────────────────────────────────────────────
REPO_ROOT="${SLURM_SUBMIT_DIR}"
MAGVIT2_ROOT="/work/projects/imas_gpu/mast-tokens/v1/open-magvit2"
TORCHRUN="${MAGVIT2_ROOT}/.venv/bin/torchrun"
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN

# Send SIGTERM after 15s warmup; use 256 MB tensor for long collectives
export T2_SIGTERM_DELAY_S=15
export T2_TENSOR_NUMEL=$((64 * 1024 * 1024))  # 64M × float32 = 256 MB

echo "[T2] Job ${SLURM_JOB_ID} on $(hostname)"
echo "[T2] Self-SIGTERM in ${T2_SIGTERM_DELAY_S}s"
echo "[T2] Tensor: $((T2_TENSOR_NUMEL * 4 / 1024 / 1024)) MB"

PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" \
    "${TORCHRUN}" \
    --nproc_per_node=2 \
    --master_addr=localhost \
    --master_port=29502 \
    "${REPO_ROOT}/scripts/tests/t2_sigterm_delay_test.py"

echo "[T2] Done — check log for SIGTERM delay measurement"
