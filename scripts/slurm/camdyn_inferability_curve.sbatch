#!/bin/bash
#SBATCH --job-name=camdyn-inferability
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --account=grpa
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=160G
#SBATCH --time=00:45:00
#SBATCH --output=/work/projects/imas_gpu/logs/camdyn-%x-%j.out

# Eval-only paired clip-area sweep on the trained reconstruction checkpoints.

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
if [[ ! -d "${REPO_ROOT}/imas_ambix/camdyn" ]]; then
    echo "invalid repository root: ${REPO_ROOT}" >&2
    exit 2
fi

export TMPDIR=/scratch_local/${SLURM_JOB_ID}
mkdir -p "${TMPDIR}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
export TORCHINDUCTOR_CACHE_DIR="${TMPDIR}/torch-inductor"
mkdir -p "${TORCHINDUCTOR_CACHE_DIR}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

PYTHON=/home/ITER/mcintos/Code/imas-ambix/.venv/bin/python
if [[ ! -x "${PYTHON}" ]]; then
    echo "project environment is unavailable at ${PYTHON}" >&2
    exit 3
fi

: "${MAX_WINDOWS:=256}"
: "${N_BOOT:=10000}"
: "${ARTIFACT_OUT:=${REPO_ROOT}/imas_ambix/camdyn/artifacts/inferability_curve.json}"
: "${FIGURE_OUT:=${REPO_ROOT}/docs/figures/camera-dynamics-wm/fig-cdw-inferability-curve.png}"

cd "${REPO_ROOT}"
"${PYTHON}" -m imas_ambix.camdyn.inferability_curve \
    --device cuda \
    --max-windows "${MAX_WINDOWS}" \
    --n-boot "${N_BOOT}" \
    --out "${ARTIFACT_OUT}" \
    --figure "${FIGURE_OUT}"
echo "INFERABILITY_EXIT=0"
