#!/bin/bash
#SBATCH --job-name=ambix-encode-frames
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
    echo "[bulk-encode-frames] .venv not found — falling back to uv run" >&2
    AMBIX="uv run ambix"
fi

# ── Env vars (override via sbatch --export=ALL,VAR=val) ───────────────────
: "${CAMERA:=rbb}"
: "${QUALITY_INDEX:=/tmp/audit-full.json}"
: "${OUTPUT_DIR:=/work/projects/imas_gpu/mast/tokens}"

# Per-job output report so parallel runs don't clobber each other
REPORT="${OUTPUT_DIR}/${CAMERA}-bulk-${SLURM_JOB_ID}.json"

echo "[$(date)] bulk-encode-frames starting"
echo "  job-id:        ${SLURM_JOB_ID}"
echo "  node:          $(hostname)"
echo "  camera:        ${CAMERA}"
echo "  quality_index: ${QUALITY_INDEX}"
echo "  output_dir:    ${OUTPUT_DIR}"
echo "  report:        ${REPORT}"

mkdir -p "${OUTPUT_DIR}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# ── Run bulk-encode-frames ─────────────────────────────────────────────────
$AMBIX data bulk-encode-frames \
    --from-quality-index "${QUALITY_INDEX}" \
    --tokenizer open-magvit2 \
    --device cuda \
    --workers 4 \
    --camera "${CAMERA}" \
    --output "${REPORT}"

EXIT_CODE=$?
echo "[$(date)] bulk-encode-frames finished (exit=${EXIT_CODE})"
exit "${EXIT_CODE}"
