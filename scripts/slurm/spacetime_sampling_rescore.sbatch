#!/bin/bash
#SBATCH --job-name=ambix-sampling-rescore
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --account=grpa
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=/work/projects/imas_gpu/logs/ambix-%x-%j.out
# NOTE: --qos intentionally omitted — gpu_0003_grpa QoS is auto-applied.
#
# ============================================================================
#  Argmax-vs-sampling re-score of the signal-conditioned camera world model.
# ============================================================================
#  Loads the v2 checkpoint + the frozen Open-MAGVIT2 VQModel ONCE (AGENTS.md
#  §2b), and for each held-out shot compares the greedy ARGMAX rollout against a
#  small temperature/top-p sampling grid, scoring each with the distributional /
#  motion / luminance-fair playability harness (NOT the persistence-favouring
#  pixel-MAE).  Answers: does sampling alone reduce the persistence collapse?
#
#  Single GPU — bursts onto a free card; leaves the DeepSeek-V4-Flash neighbour
#  on its cards untouched.  Window flags MUST match the training window.
#
#  Inputs (env):
#    WM_CKPT    — checkpoint .pt to score (default best.pt under WM_CKPT_DIR)
#    WM_SHOTS   — comma-separated held-out shot ids (bright 18502/18505 + dim
#                 18503/18504 by default)
#    WM_OUT_JSON— verdict JSON path
#    WM_ENSEMBLE— sampled rollouts per (T, top_p) grid point (default 4)
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

: "${WM_CKPT_DIR:=/work/projects/imas_gpu/worldmodel/ckpt/spacetime-v2-1219852}"
: "${WM_CKPT:=${WM_CKPT_DIR}/best.pt}"
: "${WM_SHOTS:=18502,18503,18504,18505}"
: "${WM_OUT_JSON:=${WM_CKPT_DIR}/sampling_rescore/verdict.json}"
: "${WM_ENSEMBLE:=4}"
# window — MUST match the training window
: "${WM_N_FRAMES:=24}"
: "${WM_N_PLAN:=8}"
: "${WM_CONTEXT_FRAMES:=8}"
: "${WM_FRAME_STRIDE:=1}"
: "${WM_N_SIGNAL_STEPS:=4}"
: "${WM_CHUNK:=8192}"

if [[ ! -f "${WM_CKPT}" ]]; then
    echo "[rescore] FATAL: checkpoint not found: ${WM_CKPT}" >&2
    exit 2
fi

echo "[$(date)] spacetime sampling re-score (job ${SLURM_JOB_ID} on $(hostname))"
echo "  checkpoint: ${WM_CKPT}"
echo "  shots:      ${WM_SHOTS}"
echo "  out-json:   ${WM_OUT_JSON}"
echo "  ensemble:   ${WM_ENSEMBLE} per (T, top_p) grid point"
echo "  window:     n_frames=${WM_N_FRAMES} n_plan=${WM_N_PLAN} context=${WM_CONTEXT_FRAMES} stride=${WM_FRAME_STRIDE} n_signal_steps=${WM_N_SIGNAL_STEPS}"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader || true

python -m imas_ambix.worldmodel.sampling_rescore \
    --checkpoint "${WM_CKPT}" \
    --shots "${WM_SHOTS}" \
    --out-json "${WM_OUT_JSON}" \
    --ensemble "${WM_ENSEMBLE}" \
    --n-frames "${WM_N_FRAMES}" \
    --n-plan "${WM_N_PLAN}" \
    --context-frames "${WM_CONTEXT_FRAMES}" \
    --frame-stride "${WM_FRAME_STRIDE}" \
    --n-signal-steps "${WM_N_SIGNAL_STEPS}" \
    --chunk "${WM_CHUNK}" \
    --work-dir "${TMPDIR}/rescore" \
    --device cuda
EXIT_CODE=$?

echo "[$(date)] sampling re-score finished (exit=${EXIT_CODE})"
echo "  verdict: ${WM_OUT_JSON}"
exit "${EXIT_CODE}"
