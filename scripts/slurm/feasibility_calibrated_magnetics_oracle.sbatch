#!/bin/bash
#SBATCH --job-name=ambix-calib-mag-oracle
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --account=grpa
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=/work/projects/imas_gpu/logs/ambix-%x-%j.out
# NOTE: --qos intentionally omitted — under the default `normal` QOS the job
# bursts onto any free card node-wide (the reservation reserves cores only), so
# this single-GPU oracle co-exists with the live DeepSeek serve without taking
# it down (see AGENTS.md §2).

# ── Environment ────────────────────────────────────────────────────────────
set -euo pipefail

export TMPDIR=/scratch_local/$SLURM_JOB_ID
mkdir -p "$TMPDIR"

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

# SLURM copies the batch script elsewhere before running, so use SLURM_SUBMIT_DIR
# (the dir sbatch was invoked from) to find the venv + write results into the repo.
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_ROOT}"
mkdir -p "/work/projects/imas_gpu/logs"

VENV="${REPO_ROOT}/.venv"
if [[ -f "${VENV}/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source "${VENV}/bin/activate"
    PY="${VENV}/bin/python"
else
    echo "[calib-mag-oracle] .venv not found — falling back to uv run" >&2
    PY="uv run --no-sync python"
fi

# ── Tunables (override via sbatch --export=ALL,VAR=val) ─────────────────────
: "${N_TRAIN_SHOTS:=250}"
: "${N_TEST_SHOTS:=60}"
: "${EPOCHS:=80}"
: "${OUT_ROOT:=/work/projects/imas_gpu/worldmodel/calibrated_magnetics_oracle}"
: "${FIG_DIR:=${REPO_ROOT}/docs/figures/joint-multimodal-plasma-wm}"

echo "[$(date)] calib-mag-oracle starting"
echo "  job-id:        ${SLURM_JOB_ID}"
echo "  node:          $(hostname)"
echo "  n_train_shots: ${N_TRAIN_SHOTS}"
echo "  n_test_shots:  ${N_TEST_SHOTS}"
echo "  epochs:        ${EPOCHS}"
echo "  out_root:      ${OUT_ROOT}"

nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

$PY scripts/feasibility_calibrated_magnetics_oracle.py \
    --n-train-shots "${N_TRAIN_SHOTS}" \
    --n-test-shots "${N_TEST_SHOTS}" \
    --epochs "${EPOCHS}" \
    --out-root "${OUT_ROOT}" \
    --fig-dir "${FIG_DIR}"

EXIT_CODE=$?
echo "[$(date)] calib-mag-oracle finished (exit=${EXIT_CODE})"
echo "  report: ${OUT_ROOT}/oracle_report.json"
exit "${EXIT_CODE}"
