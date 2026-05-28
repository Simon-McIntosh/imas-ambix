#!/bin/bash
#SBATCH --job-name=ambix-bench-rbb
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --account=grpa
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=30
#SBATCH --mem=640G
#SBATCH --time=04:00:00
#SBATCH --output=/tmp/ambix-%x-%j.out
# NOTE: --qos intentionally omitted — see train_exclusive.sbatch for rationale.

# ── Environment ────────────────────────────────────────────────────────────
set -euo pipefail

export TMPDIR=/scratch_local/$SLURM_JOB_ID
mkdir -p "$TMPDIR"

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

# ── Activate project venv ──────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="${REPO_ROOT}/.venv"
if [[ -f "${VENV}/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source "${VENV}/bin/activate"
    AMBIX="${VENV}/bin/ambix"
else
    echo "[bench-rbb] .venv not found — falling back to uv run" >&2
    AMBIX="uv run ambix"
fi

# ── Env vars (override via sbatch --export=ALL,VAR=val) ───────────────────
: "${BENCH_CONFIG:=imas_ambix/bench/configs/v0-rbb-100shot.yaml}"
: "${RESULTS_DIR:=${REPO_ROOT}/imas_ambix/bench/results}"

BENCH_NAME="$(basename "${BENCH_CONFIG}" .yaml)"
RESULT_FILE="${RESULTS_DIR}/${BENCH_NAME}-${SLURM_JOB_ID}.json"

echo "[$(date)] bench-rbb starting"
echo "  job-id:      ${SLURM_JOB_ID}"
echo "  node:        $(hostname)"
echo "  bench_config: ${BENCH_CONFIG}"
echo "  results_dir: ${RESULTS_DIR}"
echo "  output:      ${RESULT_FILE}"

mkdir -p "${RESULTS_DIR}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# ── Run the benchmark ──────────────────────────────────────────────────────
$AMBIX tokenize bench \
    --config "${BENCH_CONFIG}" \
    --output "${RESULT_FILE}"

EXIT_CODE=$?
echo "[$(date)] bench-rbb finished (exit=${EXIT_CODE})"
echo "  result: ${RESULT_FILE}"
exit "${EXIT_CODE}"
