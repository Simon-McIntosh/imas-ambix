#!/bin/bash
# T1: STOP-FILE clean-exit validation — SAFE, no drain risk.
#
# Validates that touching AMBIX_STOP_FILE causes a clean, orderly DDP
# shutdown. This is a positive control proving the primary cancellation
# mechanism works as documented.
#
# SUBMIT:  sbatch scripts/tests/t1_stopfile.sbatch
# TRIGGER: touch /work/projects/imas_gpu/stops/<JOBID>.stop
# PASS:    All ranks exit 0 within ~5s of file creation.
#
#SBATCH --job-name=ambix-t1-stopfile
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --account=grpa
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=00:15:00
#SBATCH --signal=B:USR1@120
#SBATCH --output=/work/projects/imas_gpu/logs/t1-stopfile-%j.log

set -euo pipefail
export TMPDIR=/scratch_local/${SLURM_JOB_ID}
mkdir -p "${TMPDIR}"

# ── Three-layer defence (this script models correct practice) ────────────────
mkdir -p /work/projects/imas_gpu/stops
export AMBIX_STOP_FILE=/work/projects/imas_gpu/stops/${SLURM_JOB_ID}.stop
rm -f "${AMBIX_STOP_FILE}"

_on_sigusr1() {
    echo "[T1-sbatch] SIGUSR1 → touching stop-file" >&2
    touch "${AMBIX_STOP_FILE}"
}
trap _on_sigusr1 USR1

export AMBIX_SOFT_TIME_LIMIT=720   # 12 min = 0.85 × 15 min ceiling

# ── Environment ─────────────────────────────────────────────────────────────
REPO_ROOT="${SLURM_SUBMIT_DIR}"
MAGVIT2_ROOT="/work/projects/imas_gpu/mast-tokens/v1/open-magvit2"
TORCHRUN="${MAGVIT2_ROOT}/.venv/bin/torchrun"
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN

echo "[T1] Job ${SLURM_JOB_ID} on $(hostname)"
echo "[T1] STOP-FILE: ${AMBIX_STOP_FILE}"
echo "[T1] To trigger: touch ${AMBIX_STOP_FILE}"
echo "[T1] To observe: tail -f /work/projects/imas_gpu/logs/t1-stopfile-${SLURM_JOB_ID}.log"

PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" \
    "${TORCHRUN}" \
    --nproc_per_node=2 \
    --master_addr=localhost \
    --master_port=29501 \
    "${REPO_ROOT}/scripts/tests/t1_stopfile_test.py"

echo "[T1] torchrun exited with code $?"
