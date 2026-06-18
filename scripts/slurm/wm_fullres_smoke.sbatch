#!/bin/bash
#SBATCH --job-name=ambix-wm-fullres-smoke
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --account=grpa
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=160G
#SBATCH --time=01:00:00
#SBATCH --output=/work/projects/imas_gpu/logs/ambix-%x-%j.out
# NOTE: --qos intentionally omitted — gpu_0003_grpa QoS is auto-applied.
#
# ============================================================================
#  Full-resolution rbb predict→decode FEASIBILITY SMOKE (the de-risk proof).
# ============================================================================
#  Overfits a small full-res-rbb world model on a few rbb-bearing shots, rolls
#  it out, validates the model→global id mapping, and decodes ONE predicted
#  full-res rbb frame end-to-end into an image (side-by-side vs ground truth).
#  Runs on a SINGLE GPU and keeps the neighbour's LLM server (DeepSeek-V4-Flash)
#  up — only gpu:1 is requested.
#
#  This must look plasma-like before the full retrain is launched; a garbage
#  decode is most likely an id-mapping bug (STOP, do not retrain).
# ============================================================================

set -uo pipefail

export TMPDIR="/scratch_local/${SLURM_JOB_ID}"
mkdir -p "${TMPDIR}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_ROOT}"
mkdir -p "/work/projects/imas_gpu/logs"

# Drain-forensics sidecar (default-on for GPU jobs; AGENTS.md §2a).
bash "${REPO_ROOT}/scripts/slurm/drain_sidecar.sh" &
SIDECAR_PID=$!
cleanup() { kill "${SIDECAR_PID}" 2>/dev/null || true; }
trap cleanup EXIT

VENV="${REPO_ROOT}/.venv"
# shellcheck source=/dev/null
source "${VENV}/bin/activate"
PY="${VENV}/bin/python"

: "${WM_SMOKE_SHOTS:=15085,15086,15087,15088}"
: "${WM_SMOKE_STEPS:=400}"
: "${WM_SMOKE_LR:=3e-3}"
: "${WM_SMOKE_D_MODEL:=384}"   # 384 enables warm-start from the converged ckpt
: "${WM_SMOKE_N_LAYERS:=8}"
: "${WM_SMOKE_N_HEADS:=8}"
: "${WM_SMOKE_WARM:=1}"        # 1 = warm-start; 0 = fresh init
: "${WM_SMOKE_OUT:=/work/projects/imas_gpu/worldmodel/demo/fullres_smoke}"
mkdir -p "${WM_SMOKE_OUT}"

WARM_ARG=""
if [[ "${WM_SMOKE_WARM}" == "0" ]]; then WARM_ARG="--no-warm-start"; fi

echo "[$(date)] full-res rbb smoke starting (job ${SLURM_JOB_ID} on $(hostname))"
echo "  shots=${WM_SMOKE_SHOTS} steps=${WM_SMOKE_STEPS} d_model=${WM_SMOKE_D_MODEL} warm=${WM_SMOKE_WARM}"
nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader || true

# shellcheck disable=SC2086
${PY} -m imas_ambix.worldmodel.fullres_smoke \
    --shots "${WM_SMOKE_SHOTS}" \
    --steps "${WM_SMOKE_STEPS}" \
    --lr "${WM_SMOKE_LR}" \
    --d-model "${WM_SMOKE_D_MODEL}" \
    --n-layers "${WM_SMOKE_N_LAYERS}" \
    --n-heads "${WM_SMOKE_N_HEADS}" \
    ${WARM_ARG} \
    --out-dir "${WM_SMOKE_OUT}"
EXIT_CODE=$?

echo "[$(date)] full-res rbb smoke finished (exit=${EXIT_CODE})"
echo "  artifacts: ${WM_SMOKE_OUT}"
exit "${EXIT_CODE}"
