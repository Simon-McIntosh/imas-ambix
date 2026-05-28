#!/bin/bash
#SBATCH --job-name=ambix-train
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --account=grpa
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=30
#SBATCH --mem=640G
#SBATCH --time=04:00:00
#SBATCH --output=/work/projects/imas_gpu/logs/ambix-%x-%j.out
# NOTE: --qos is intentionally omitted — auto-applied for this account/reservation.
#       Passing --qos explicitly errors with "QOS not permitted".

# ── Environment ────────────────────────────────────────────────────────────
set -euo pipefail

export TMPDIR=/scratch_local/$SLURM_JOB_ID
mkdir -p "$TMPDIR"

# Avoid CPU oversubscription under FSDP/NCCL
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

# TorchInductor / DeepGEMM cache → fast local NVMe, not GPFS
export TORCHINDUCTOR_CACHE_DIR=$TMPDIR/.torch_inductor
mkdir -p "$TORCHINDUCTOR_CACHE_DIR"

# ── Preempt V4-Flash ───────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
bash "${REPO_ROOT}/scripts/slurm/preempt_v4flash.sh"

# ── Activate project venv ──────────────────────────────────────────────────
VENV="${REPO_ROOT}/.venv"
if [[ -f "${VENV}/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source "${VENV}/bin/activate"
    PYTHON="${VENV}/bin/python"
    ACCELERATE="${VENV}/bin/accelerate"
else
    echo "[train] .venv not found at ${VENV} — falling back to uv run" >&2
    PYTHON="uv run python"
    ACCELERATE="uv run accelerate"
fi

# ── Config / output paths (override via sbatch --export) ───────────────────
: "${CONFIG_PATH:=imas_ambix/train/configs/v0-125m.yaml}"
: "${OUTPUT_DIR:=/work/projects/imas_gpu/mast-checkpoints}"

LOG_FILE=/tmp/ambix-train-${SLURM_JOB_ID}.log

echo "[$(date)] ambix-train starting"
echo "  job-id:      ${SLURM_JOB_ID}"
echo "  node:        $(hostname)"
echo "  config:      ${CONFIG_PATH}"
echo "  output_dir:  ${OUTPUT_DIR}"
echo "  log:         ${LOG_FILE}"

nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# ── Launch training ────────────────────────────────────────────────────────
# accelerate launch distributes across the 4 GPUs via FSDP; Hydra picks up
# the config path via --config-path / --config-name resolution.
CONFIG_DIR="$(dirname "${CONFIG_PATH}")"
CONFIG_NAME="$(basename "${CONFIG_PATH}" .yaml)"

$ACCELERATE launch \
    --num_processes 4 \
    --mixed_precision bf16 \
    -m imas_ambix.train.loop \
    --config-path "${CONFIG_DIR}" \
    --config-name "${CONFIG_NAME}" \
    ++checkpoint.root="${OUTPUT_DIR}" \
    2>&1 | tee "${LOG_FILE}"

EXIT_CODE=${PIPESTATUS[0]}
echo "[$(date)] ambix-train finished (exit=${EXIT_CODE})"
exit "${EXIT_CODE}"
