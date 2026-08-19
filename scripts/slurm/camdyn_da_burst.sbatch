#!/bin/bash
#SBATCH --job-name=camdyn-da-burst
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --account=grpa
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=160G
#SBATCH --time=16:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --output=/work/projects/imas_gpu/logs/camdyn-%x-%j.out

# One matched camera-dynamics training arm using native, causally shuffled, or
# slow-only fast-Dalpha conditioning. Every run uses one GPU and writes its
# evaluation beside its checkpoints on GPFS.
set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
if [[ ! -d "${REPO_ROOT}/imas_ambix/camdyn" ]]; then
    echo "[camdyn-da-burst] submit from the imas-ambix repository root" >&2
    exit 2
fi

: "${CONFIG:?set CONFIG to the selected Dalpha conditioning config}"
: "${VARIANT:?set VARIANT to native, shuffled, or slow_only}"
: "${ARM:?set ARM to baseline or dynamics}"

case "${VARIANT}:${ARM}" in
    native:baseline)
        EXPECTED_CONFIG="imas_ambix/camdyn/configs/da_burst_baseline.yaml"
        RUN_NAME="da_burst_baseline"
        ;;
    native:dynamics)
        EXPECTED_CONFIG="imas_ambix/camdyn/configs/da_burst_dynamics.yaml"
        RUN_NAME="da_burst_dynamics"
        ;;
    shuffled:baseline)
        EXPECTED_CONFIG="imas_ambix/camdyn/configs/da_burst_shuffled_baseline.yaml"
        RUN_NAME="da_burst_shuffled_baseline"
        ;;
    shuffled:dynamics)
        EXPECTED_CONFIG="imas_ambix/camdyn/configs/da_burst_shuffled_dynamics.yaml"
        RUN_NAME="da_burst_shuffled_dynamics"
        ;;
    slow_only:baseline)
        EXPECTED_CONFIG="imas_ambix/camdyn/configs/da_burst_slow_baseline.yaml"
        RUN_NAME="da_burst_slow_baseline"
        ;;
    slow_only:dynamics)
        EXPECTED_CONFIG="imas_ambix/camdyn/configs/da_burst_slow_dynamics.yaml"
        RUN_NAME="da_burst_slow_dynamics"
        ;;
    *)
        echo "[camdyn-da-burst] unsupported VARIANT:ARM pair: ${VARIANT}:${ARM}" >&2
        exit 2
        ;;
esac

if [[ "${CONFIG}" != "${EXPECTED_CONFIG}" ]]; then
    echo "[camdyn-da-burst] CONFIG mismatch: expected ${EXPECTED_CONFIG}, got ${CONFIG}" >&2
    exit 2
fi

export TMPDIR="/scratch_local/${SLURM_JOB_ID}"
mkdir -p "${TMPDIR}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
export TORCHINDUCTOR_CACHE_DIR="${TMPDIR}/.torch_inductor"
mkdir -p "${TORCHINDUCTOR_CACHE_DIR}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

: "${AMBIX_PROJECT_ENVIRONMENT:=/home/ITER/mcintos/Code/imas-ambix/.venv}"
if [[ ! -x "${AMBIX_PROJECT_ENVIRONMENT}/bin/python" ]]; then
    echo "[camdyn-da-burst] project environment unavailable: ${AMBIX_PROJECT_ENVIRONMENT}" >&2
    exit 2
fi
PYTHON="${AMBIX_PROJECT_ENVIRONMENT}/bin/python"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

: "${CKPT_ROOT:=/work/projects/imas_gpu/mast-checkpoints/camdyn}"
: "${ARTIFACT_OUT:=${CKPT_ROOT}/${RUN_NAME}/evaluation.json}"
: "${LOG:=/work/projects/imas_gpu/logs/camdyn-da-burst-${VARIANT}-${ARM}-${SLURM_JOB_ID}.log}"
: "${AMBIX_STOP_FILE:=/work/projects/imas_gpu/logs/camdyn-da-burst-${SLURM_JOB_ID}.stop}"
export AMBIX_STOP_FILE

bash "${REPO_ROOT}/scripts/slurm/drain_sidecar.sh" &

EXTRA=()
[[ -n "${MAX_STEPS:-}" ]] && EXTRA+=(--max-steps "${MAX_STEPS}")
[[ -n "${VAL_EVERY:-}" ]] && EXTRA+=(--val-every "${VAL_EVERY}")
[[ -n "${EVAL_WINDOWS:-}" ]] && EXTRA+=(--eval-windows "${EVAL_WINDOWS}")
[[ -n "${VAL_WINDOWS:-}" ]] && EXTRA+=(--val-windows "${VAL_WINDOWS}")

echo "[$(date)] Dalpha conditioning training starting"
echo "  job-id:     ${SLURM_JOB_ID}"
echo "  node:       $(hostname)"
echo "  variant:    ${VARIANT}"
echo "  arm:        ${ARM}"
echo "  config:     ${CONFIG}"
echo "  checkpoint: ${CKPT_ROOT}/${RUN_NAME}/final.pt"
echo "  artifact:   ${ARTIFACT_OUT}"
echo "  log:        ${LOG}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

TRAIN_PID=""
request_stop() {
    touch "${AMBIX_STOP_FILE}"
    if [[ -n "${TRAIN_PID}" ]] && kill -0 "${TRAIN_PID}" 2>/dev/null; then
        kill -TERM "${TRAIN_PID}"
    fi
}
trap request_stop USR1 TERM INT

cd "${REPO_ROOT}"
"${PYTHON}" -m imas_ambix.camdyn.train \
    --config "${CONFIG}" \
    --device cuda \
    --artifact-out "${ARTIFACT_OUT}" \
    --ckpt-root "${CKPT_ROOT}" \
    "${EXTRA[@]}" >"${LOG}" 2>&1 &
TRAIN_PID=$!

set +e
while true; do
    wait "${TRAIN_PID}"
    TRAIN_DONE=$?
    if ! kill -0 "${TRAIN_PID}" 2>/dev/null; then
        break
    fi
done
set -e

echo "[$(date)] Dalpha conditioning training finished" | tee -a "${LOG}"
echo "TRAIN_DONE=${TRAIN_DONE}" | tee -a "${LOG}"
exit "${TRAIN_DONE}"
