#!/bin/bash
#SBATCH --job-name=ambix-encode-signals
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
    echo "[bulk-encode-signals] .venv not found — falling back to uv run" >&2
    AMBIX="uv run ambix"
fi

# ── Env vars (override via sbatch --export=ALL,VAR=val) ───────────────────
: "${QUALITY_INDEX:=/tmp/audit-full.json}"
: "${OUTPUT_DIR:=/work/projects/imas_gpu/mast/tokens}"
# Signal groups to encode (space-separated; loop below handles each)
: "${SIGNAL_GROUPS:=magnetics summary}"

echo "[$(date)] bulk-encode-signals starting"
echo "  job-id:        ${SLURM_JOB_ID}"
echo "  node:          $(hostname)"
echo "  quality_index: ${QUALITY_INDEX}"
echo "  signal_groups: ${SIGNAL_GROUPS}"

# ── Run bulk-encode-signals for each group ─────────────────────────────────
OVERALL_EXIT=0
for GROUP in $SIGNAL_GROUPS; do
    REPORT="${OUTPUT_DIR}/signals-${GROUP}-bulk-${SLURM_JOB_ID}.json"
    echo "[$(date)] Encoding group: ${GROUP} → ${REPORT}"
    mkdir -p "${OUTPUT_DIR}"

    $AMBIX data bulk-encode-signals \
        --from-quality-index "${QUALITY_INDEX}" \
        --device cuda \
        --workers 4 \
        --group "${GROUP}" \
        --output "${REPORT}" || {
        echo "[bulk-encode-signals] WARNING: group ${GROUP} failed (exit=$?)" >&2
        OVERALL_EXIT=1
    }
done

echo "[$(date)] bulk-encode-signals finished (exit=${OVERALL_EXIT})"
exit "${OVERALL_EXIT}"
