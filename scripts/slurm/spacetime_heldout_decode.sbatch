#!/bin/bash
#SBATCH --job-name=ambix-spacetime-decode
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --account=grpa
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=00:30:00
#SBATCH --output=/work/projects/imas_gpu/logs/ambix-%x-%j.out
# NOTE: --qos intentionally omitted — gpu_0003_grpa QoS is auto-applied.
#
# ============================================================================
#  Held-out generalization decode for the spatiotemporal camera transformer.
# ============================================================================
#  Loads a corpus training checkpoint and decodes a HELD-OUT (unseen-in-training)
#  rbb shot to GT-vs-prediction images: teacher-forced reconstruction AND the
#  autoregressive dream, full-resolution (256x256) via the frozen Open-MAGVIT2
#  VQModel (the validated two-venv decode path).  This is the real test — the
#  model never saw these shots, so coherent forecast frames mean it learned
#  plasma dynamics, not memorization.
#
#  Single GPU, one-shot tool (exempt from the in-process loop rule).  Bursts
#  onto a free card on the node; leaves the 4-GPU training job and the
#  neighbour's DeepSeek-V4-Flash untouched.
#
#  Inputs (env):
#    WM_CKPT   — checkpoint .pt (default: latest.pt under WM_OUT_DIR)
#    WM_OUT_DIR — training out-dir holding latest.pt (used if WM_CKPT unset)
#    WM_SHOTS  — comma-separated held-out shot ids (default 18502,18503,18504,18505)
#    WM_FIG_DIR — where GIFs/PNGs land (default <out-dir>/heldout_decode)
#  Window flags MUST match the training window (n_frames/n_plan/context/stride).
# ============================================================================

set -uo pipefail

export TMPDIR="/scratch_local/${SLURM_JOB_ID}"
mkdir -p "${TMPDIR}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_ROOT}"
mkdir -p "/work/projects/imas_gpu/logs"

VENV="${REPO_ROOT}/.venv"
# shellcheck source=/dev/null
source "${VENV}/bin/activate"

: "${WM_OUT_DIR:?set WM_OUT_DIR to the training out-dir (holds latest.pt) or set WM_CKPT}"
: "${WM_CKPT:=${WM_OUT_DIR}/latest.pt}"
: "${WM_SHOTS:=18502,18503,18504,18505}"
: "${WM_FIG_DIR:=${WM_OUT_DIR}/heldout_decode}"
# window — MUST match the training window
: "${WM_N_FRAMES:=24}"
: "${WM_N_PLAN:=8}"
: "${WM_CONTEXT_FRAMES:=8}"
: "${WM_FRAME_STRIDE:=1}"

if [[ ! -f "${WM_CKPT}" ]]; then
    echo "[decode] FATAL: checkpoint not found: ${WM_CKPT}" >&2
    exit 2
fi

echo "[$(date)] spacetime held-out decode (job ${SLURM_JOB_ID} on $(hostname))"
echo "  checkpoint: ${WM_CKPT}"
echo "  shots:      ${WM_SHOTS}"
echo "  fig-dir:    ${WM_FIG_DIR}"
echo "  window:     n_frames=${WM_N_FRAMES} n_plan=${WM_N_PLAN} context=${WM_CONTEXT_FRAMES} stride=${WM_FRAME_STRIDE}"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader || true

EXIT_CODE=0
IFS=',' read -ra SHOT_ARR <<< "${WM_SHOTS}"
for shot in "${SHOT_ARR[@]}"; do
    shot="${shot// /}"
    [[ -z "${shot}" ]] && continue
    echo "[$(date)] === decode held-out shot ${shot} ==="
    python -m imas_ambix.worldmodel.spacetime_dream \
        --checkpoint "${WM_CKPT}" \
        --shot "${shot}" \
        --out-dir "${WM_FIG_DIR}/${shot}" \
        --n-frames "${WM_N_FRAMES}" \
        --n-plan "${WM_N_PLAN}" \
        --context-frames "${WM_CONTEXT_FRAMES}" \
        --frame-stride "${WM_FRAME_STRIDE}" \
        --device cuda
    rc=$?
    if [[ ${rc} -ne 0 ]]; then
        echo "[decode] shot ${shot} FAILED (rc=${rc})" >&2
        EXIT_CODE=${rc}
    fi
done

echo "[$(date)] held-out decode finished (exit=${EXIT_CODE})"
echo "  figures under: ${WM_FIG_DIR}/<shot>/"
exit "${EXIT_CODE}"
