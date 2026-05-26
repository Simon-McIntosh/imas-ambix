#!/bin/bash
#SBATCH --job-name=ambix-bench-rbb
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --account=grpa
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=30
#SBATCH --mem=640G
#SBATCH --time=01:00:00
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
: "${BENCH_CONFIG:=imas_ambix/bench/configs/v0-rbb-25shot.yaml}"
: "${RESULTS_DIR:=${REPO_ROOT}/imas_ambix/bench/results}"

# Read shot-ids and config from the YAML bench config
# (ambix tokenize bench accepts --shot-ids and reads tokenizer from config)
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

# ── Parse shot IDs and tokenizer from the bench YAML ──────────────────────
# The bench config YAML carries shot_ids and tokenizer fields.
# We extract them with python -c to avoid a separate yq dependency.
SHOT_IDS=$(python -c "
import yaml, sys
cfg = yaml.safe_load(open('${BENCH_CONFIG}'))
ids = cfg.get('shot_ids', [])
print(','.join(str(i) for i in ids))
" 2>/dev/null) || SHOT_IDS=""

TOKENIZER=$(python -c "
import yaml, sys
cfg = yaml.safe_load(open('${BENCH_CONFIG}'))
print(cfg.get('tokenizer', 'open-magvit2'))
" 2>/dev/null) || TOKENIZER="open-magvit2"

CAMERA=$(python -c "
import yaml, sys
cfg = yaml.safe_load(open('${BENCH_CONFIG}'))
print(cfg.get('camera', 'rbb'))
" 2>/dev/null) || CAMERA="rbb"

if [[ -z "$SHOT_IDS" ]]; then
    echo "[bench-rbb] ERROR: could not parse shot_ids from ${BENCH_CONFIG}" >&2
    exit 1
fi

echo "  shot_ids:    ${SHOT_IDS}"
echo "  tokenizer:   ${TOKENIZER}"
echo "  camera:      ${CAMERA}"

# ── Run the benchmark ──────────────────────────────────────────────────────
$AMBIX tokenize bench \
    --tokenizer "${TOKENIZER}" \
    --kind frame \
    --shot-ids "${SHOT_IDS}" \
    --camera "${CAMERA}" \
    --device cuda \
    --output "${RESULT_FILE}"

EXIT_CODE=$?
echo "[$(date)] bench-rbb finished (exit=${EXIT_CODE})"
echo "  result: ${RESULT_FILE}"
exit "${EXIT_CODE}"
