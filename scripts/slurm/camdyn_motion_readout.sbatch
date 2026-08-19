#!/bin/bash
#SBATCH --job-name=camdyn-motion-readout
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --account=grpa
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=120G
#SBATCH --time=00:45:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=/work/projects/imas_gpu/logs/camdyn-%x-%j.out

# Eval-only filament-motion dynamic-range validation on the existing dynamics
# checkpoint.  The job performs no optimiser step and writes only the committed
# metric artifact and figure in the submitting worktree.
set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
if [[ ! -d "${REPO_ROOT}/imas_ambix/camdyn" ]]; then
    echo "[camdyn] ERROR: submit this script from the imas-ambix repository root" >&2
    exit 2
fi

export TMPDIR=/scratch_local/${SLURM_JOB_ID}
mkdir -p "${TMPDIR}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
export TORCHINDUCTOR_CACHE_DIR="${TMPDIR}/.torch_inductor"
mkdir -p "${TORCHINDUCTOR_CACHE_DIR}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

bash "${REPO_ROOT}/scripts/slurm/drain_sidecar.sh" &

STOP_FILE="${TMPDIR}/motion-readout.stop"
trap 'touch "${STOP_FILE}"' USR1 TERM INT

PROJECT_VENV="${IMAS_AMBIX_VENV:-/home/ITER/mcintos/Code/imas-ambix/.venv}"
if [[ ! -x "${PROJECT_VENV}/bin/python" ]]; then
    echo "[camdyn] ERROR: project environment missing at ${PROJECT_VENV}" >&2
    exit 2
fi
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

ARTIFACT_OUT="${REPO_ROOT}/imas_ambix/camdyn/artifacts/motion_readout.json"
FIGURE_OUT="${REPO_ROOT}/docs/figures/camera-dynamics-wm/fig-cdw-motion-readout.png"
LOG="/work/projects/imas_gpu/logs/camdyn-motion-readout-${SLURM_JOB_ID}.log"

echo "[$(date)] filament-motion readout starting"
echo "  job-id:   ${SLURM_JOB_ID}"
echo "  node:     $(hostname)"
echo "  artifact: ${ARTIFACT_OUT}"
echo "  figure:   ${FIGURE_OUT}"
echo "  log:      ${LOG}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

cd "${REPO_ROOT}"
set +e
"${PROJECT_VENV}/bin/python" -m imas_ambix.camdyn.motion_readout \
    --device cuda \
    --json "${ARTIFACT_OUT}" \
    --figure "${FIGURE_OUT}" 2>&1 | tee "${LOG}"
MOTION_READOUT_EXIT=${PIPESTATUS[0]}
set -e

echo "[$(date)] filament-motion readout finished" | tee -a "${LOG}"
echo "MOTION_READOUT_EXIT=${MOTION_READOUT_EXIT}" | tee -a "${LOG}"
exit "${MOTION_READOUT_EXIT}"

