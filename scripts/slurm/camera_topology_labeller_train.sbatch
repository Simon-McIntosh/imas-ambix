#!/bin/bash
#SBATCH --job-name=camera-topology-train
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --account=grpa
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=160G
#SBATCH --time=24:00:00
#SBATCH --output=/work/projects/imas_gpu/logs/camera-topology-%j.out
# The account applies its permitted QOS automatically; an explicit QOS is rejected.

set -euo pipefail

SUBMIT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
CANONICAL_ROOT=/home/ITER/mcintos/Code/imas-ambix
REPO_ROOT="${IMAS_AMBIX_REPO_ROOT:-${SUBMIT_ROOT}}"
if [[ ! -f "${REPO_ROOT}/imas_ambix/worldmodel/camera_topology_labeller.py" ]]; then
    REPO_ROOT="${CANONICAL_ROOT}"
fi
if [[ ! -f "${REPO_ROOT}/imas_ambix/worldmodel/camera_topology_labeller.py" ]]; then
    echo "[camera-topology] no usable imas-ambix checkout at ${REPO_ROOT}" >&2
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
    echo "[camera-topology] project Python is missing at ${VENV}/bin/python" >&2
    exit 2
fi

if [[ -x "${REPO_ROOT}/scripts/slurm/drain_sidecar.sh" ]]; then
    bash "${REPO_ROOT}/scripts/slurm/drain_sidecar.sh" &
fi

: "${COHORT_REPORT:=/home/ITER/mcintos/.config/reckon/crew/reports/physics-carried-playable-plasma/labeller-cohort-census.md}"
: "${FULLSHOT_MANIFEST:=/work/projects/imas_gpu/agents/excitation-corpus/curated_windows_fullshot.json}"
: "${ARTIFACT_ROOT:=/work/projects/imas_gpu/mast-checkpoints/camera-topology-labeller/${SLURM_JOB_ID}}"
: "${EPOCHS:=8}"
: "${BATCH_SIZE:=1024}"
: "${IMAGE_SIZE:=64}"
: "${MODEL_WIDTH:=32}"
: "${MODEL_HIDDEN:=256}"
: "${LEARNING_RATE:=3e-4}"

mkdir -p "${ARTIFACT_ROOT}"
REPORT="${ARTIFACT_ROOT}/training.json"
TRAIN_LOG="${ARTIFACT_ROOT}/training.log"

echo "[$(date --iso-8601=seconds)] camera-topology training starting"
echo "  job: ${SLURM_JOB_ID}"
echo "  node: $(hostname)"
echo "  source: ${REPO_ROOT}"
echo "  cohort: 472 train shots / 1163812 frames"
echo "  held-out: 58 validation shots / 125930 frames"
echo "  artifact-root: ${ARTIFACT_ROOT}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

cd "${REPO_ROOT}"
set +e
"${VENV}/bin/python" -m imas_ambix.worldmodel.camera_topology_labeller \
    --full-corpus \
    --cohort-report "${COHORT_REPORT}" \
    --fullshot-manifest "${FULLSHOT_MANIFEST}" \
    --windows-per-shot 0 \
    --window-frames 3 \
    --image-size "${IMAGE_SIZE}" \
    --width "${MODEL_WIDTH}" \
    --hidden "${MODEL_HIDDEN}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --learning-rate "${LEARNING_RATE}" \
    --device cuda \
    --output "${REPORT}" 2>&1 | tee "${TRAIN_LOG}"
TRAIN_EXIT=${PIPESTATUS[0]}
set -e

echo "[$(date --iso-8601=seconds)] camera-topology training finished"
echo "TRAIN_EXIT=${TRAIN_EXIT}"
echo "REPORT=${REPORT}"
exit "${TRAIN_EXIT}"
