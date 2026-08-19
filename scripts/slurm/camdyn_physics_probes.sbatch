#!/bin/bash
#SBATCH --job-name=camdyn-physics-probes
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --account=grpa
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=/work/projects/imas_gpu/logs/camdyn-%x-%j.out
set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
if [[ ! -d "${REPO_ROOT}/imas_ambix/camdyn" ]]; then
    echo "[camdyn-probes] invalid repository root: ${REPO_ROOT}" >&2
    exit 2
fi

export TMPDIR=/scratch_local/${SLURM_JOB_ID}
mkdir -p "$TMPDIR"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
export TORCHINDUCTOR_CACHE_DIR="$TMPDIR/.torch_inductor"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

: "${AMBIX_STOP_FILE:=/work/projects/imas_gpu/logs/camdyn-probes-${SLURM_JOB_ID}.stop}"
export AMBIX_STOP_FILE
trap 'touch "${AMBIX_STOP_FILE}"' USR1 TERM INT
bash "${REPO_ROOT}/scripts/slurm/drain_sidecar.sh" &

: "${AMBIX_PROJECT_ENVIRONMENT:=/home/ITER/mcintos/Code/imas-ambix/.venv}"
if [[ ! -x "${AMBIX_PROJECT_ENVIRONMENT}/bin/python" ]]; then
    echo "[camdyn-probes] project environment unavailable: ${AMBIX_PROJECT_ENVIRONMENT}" >&2
    exit 2
fi
PYTHON="${AMBIX_PROJECT_ENVIRONMENT}/bin/python"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

: "${BASELINE_CHECKPOINT:=/work/projects/imas_gpu/mast-checkpoints/camdyn/cap_v1_baseline/final.pt}"
: "${DYNAMICS_CHECKPOINT:=/work/projects/imas_gpu/mast-checkpoints/camdyn/cap_v1_dynamics/final.pt}"
: "${ARTIFACT_OUT:=${REPO_ROOT}/imas_ambix/camdyn/artifacts/physics_probes.json}"
: "${FIGURE_OUT:=${REPO_ROOT}/docs/figures/camera-dynamics-wm/fig-cdw-physics-probes.png}"
LOG=/work/projects/imas_gpu/logs/camdyn-physics-probes-${SLURM_JOB_ID}.log

cd "${REPO_ROOT}"
set +e
"${PYTHON}" -m imas_ambix.camdyn.physics_probes \
    --baseline "${BASELINE_CHECKPOINT}" \
    --dynamics "${DYNAMICS_CHECKPOINT}" \
    --out "${ARTIFACT_OUT}" \
    --figure "${FIGURE_OUT}" \
    --device cuda >"${LOG}" 2>&1
PROBE_DONE=$?
set -e

echo "PROBE_DONE=${PROBE_DONE}" | tee -a "${LOG}"
exit "${PROBE_DONE}"
