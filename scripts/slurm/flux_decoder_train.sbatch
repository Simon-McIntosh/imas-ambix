#!/bin/bash
#SBATCH --job-name=flux-decoder-train
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --account=grpa
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=160G
#SBATCH --time=10:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=/work/projects/imas_gpu/logs/flux-decoder-%j.out

set -euo pipefail

SUBMIT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
CANONICAL_ROOT=/home/ITER/mcintos/Code/imas-ambix
REPO_ROOT="${IMAS_AMBIX_REPO_ROOT:-${SUBMIT_ROOT}}"
if [[ ! -f "${REPO_ROOT}/imas_ambix/worldmodel/flux_decoder_train.py" ]]; then
    REPO_ROOT="${CANONICAL_ROOT}"
fi
if [[ ! -f "${REPO_ROOT}/imas_ambix/worldmodel/flux_decoder_train.py" ]]; then
    echo "[flux-decoder] no usable imas-ambix checkout at ${REPO_ROOT}" >&2
    exit 2
fi

export TMPDIR=/scratch_local/${SLURM_JOB_ID}
mkdir -p "${TMPDIR}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
export TORCHINDUCTOR_CACHE_DIR="${TMPDIR}/torch-inductor"
mkdir -p "${TORCHINDUCTOR_CACHE_DIR}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

VENV="${CANONICAL_ROOT}/.venv"
if [[ ! -x "${VENV}/bin/python" ]]; then
    echo "[flux-decoder] project Python is missing at ${VENV}/bin/python" >&2
    exit 2
fi

if [[ -x "${REPO_ROOT}/scripts/slurm/drain_sidecar.sh" ]]; then
    bash "${REPO_ROOT}/scripts/slurm/drain_sidecar.sh" &
fi

: "${SESSION_ROOT:=/work/projects/imas_gpu/sophelio/labeller_sessions/76906a29}"
: "${RUN_DIR:=/work/projects/imas_gpu/ambix/flux_decoder/overnight-20260906}"
: "${VQ_CHECKPOINT:=/work/projects/imas_gpu/mast-tokens/v1/open-magvit2/weights/imagenet_256_L.ckpt}"
: "${BATCH_SIZE:=1}"
: "${LEARNING_RATE:=3e-4}"
: "${VALIDATION_BATCHES:=4}"
: "${PIXEL_VALIDATION_SAMPLES:=4}"

mkdir -p "${RUN_DIR}"
echo "[$(date --iso-8601=seconds)] flux-conditioned decoder training starting"
echo "  job: ${SLURM_JOB_ID}"
echo "  node: $(hostname)"
echo "  source: ${REPO_ROOT}"
echo "  sessions: ${SESSION_ROOT}"
echo "  run-directory: ${RUN_DIR}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

cd "${REPO_ROOT}"
set +e
"${VENV}/bin/python" -m imas_ambix.worldmodel.flux_decoder_train \
    --session-root "${SESSION_ROOT}" \
    --run-dir "${RUN_DIR}" \
    --batch-size "${BATCH_SIZE}" \
    --learning-rate "${LEARNING_RATE}" \
    --checkpoint-interval-seconds 1800 \
    --validation-batches "${VALIDATION_BATCHES}" \
    --pixel-validation-samples "${PIXEL_VALIDATION_SAMPLES}" \
    --vq-checkpoint "${VQ_CHECKPOINT}" \
    --device cuda
TRAIN_EXIT=$?
set -e

echo "[$(date --iso-8601=seconds)] flux-conditioned decoder training finished"
echo "TRAIN_EXIT=${TRAIN_EXIT}"
echo "RECEIPT=${RUN_DIR}/receipt.json"
exit "${TRAIN_EXIT}"
