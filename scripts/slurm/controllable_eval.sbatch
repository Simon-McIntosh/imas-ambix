#!/bin/bash
#SBATCH --job-name=ambix-controllable-eval
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --account=grpa
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=/work/projects/imas_gpu/logs/ambix-%x-%j.out
# NOTE: --qos intentionally omitted — defaults to normal (no 4-GPU cap), so this
# single-card eval bursts onto a free card and COEXISTS with the DeepSeek serve.
#
# ============================================================================
#  POST-TRAIN controllability EVAL — the real held-out controllability verdict.
# ============================================================================
#  Runs controllable_eval on a trained controllable checkpoint (the corpus
#  re-train's latest.pt):
#    1. multi_shot_delta_nm — the held-out, DECODED-pixel ΔN-M: the decoded-
#       rollout divergence between the TRUE actuator plan and a REALISTIC coil
#       counterfactual (perturb the commandable coils in-distribution, hold the
#       state channels), scored against the controllable-ONLY floor.  PASS = the
#       true plan and the coil counterfactual produce visibly different decoded
#       rollouts on held-out shots (the model is DRIVEABLE by the coils).
#    2. coil_edit_dream — edit a PF position coil + dream a rollout from a held-out
#       context; emits a dream GIF + a plasma-centroid trace (the interpretable
#       "drive the plasma with a coil" artifact).
#
#  Single GPU — bursts onto a free card; leaves DeepSeek untouched.  --target-
#  horizon-s MUST match the training run (0.25s) so the eval conditions on the
#  same physical horizon the model saw.  --token-root is the curated-token-view
#  so the held-out tokens are the SAME percentile-exposure as the train corpus.
#
#  Env:
#    WM_CHECKPOINT   — the trained controllable checkpoint (default 1220940 latest)
#    WM_OUT_DIR      — eval output dir (verdict JSON + dream GIF)
#    WM_TOKEN_ROOT   — token-root for held-out tokens (curated-token-view)
#    WM_HELD_OUT     — held-out shots (default 18502-5)
#    WM_TARGET_HORIZON_S — physical window span (MUST match training, default 0.25)
#    WM_PERTURB_SCALE / WM_N_RANDOM / WM_FLOOR_RATIO / WM_MARGIN — ΔN-M knobs
#    WM_EDIT_FRAC    — coil-edit magnitude for the dream GIF (default 0.3)
#    WM_NO_DECODE    — set non-empty to score ΔN-M in TOKEN space (no GIF)
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

: "${WM_CHECKPOINT:=/work/projects/imas_gpu/worldmodel/ckpt/controllable-1220940/latest.pt}"
: "${WM_OUT_DIR:=/work/projects/imas_gpu/worldmodel/ckpt/controllable-1220940/eval}"
: "${WM_TOKEN_ROOT:=/work/projects/imas_gpu/worldmodel/curated-token-view}"
: "${WM_HELD_OUT:=18502,18503,18504,18505}"
: "${WM_TARGET_HORIZON_S:=0.25}"
: "${WM_N_FRAMES:=24}"
: "${WM_N_PLAN:=8}"
: "${WM_CONTEXT_FRAMES:=8}"
: "${WM_N_SIGNAL_STEPS:=4}"
: "${WM_N_ACT_STEPS:=8}"
# decode in chunks so the 2**18-vocab head fits the single-card budget.
: "${WM_CHUNK:=2048}"
: "${WM_PERTURB_SCALE:=0.3}"
: "${WM_N_RANDOM:=3}"
: "${WM_FLOOR_RATIO:=1.5}"
: "${WM_MARGIN:=1.0}"
: "${WM_EDIT_FRAC:=0.3}"
: "${WM_NO_DECODE:=}"

mkdir -p "${WM_OUT_DIR}"

echo "[$(date)] controllable-eval (job ${SLURM_JOB_ID} on $(hostname))"
echo "  checkpoint:  ${WM_CHECKPOINT}"
echo "  out-dir:     ${WM_OUT_DIR}"
echo "  token-root:  ${WM_TOKEN_ROOT}"
echo "  held-out:    ${WM_HELD_OUT}"
echo "  horizon:     ${WM_TARGET_HORIZON_S}s (n_frames=${WM_N_FRAMES} context=${WM_CONTEXT_FRAMES})"
echo "  ΔN-M:        perturb_scale=${WM_PERTURB_SCALE} n_random=${WM_N_RANDOM} floor_ratio=${WM_FLOOR_RATIO} margin=${WM_MARGIN}"
nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader || true

if [[ ! -f "${WM_CHECKPOINT}" ]]; then
    echo "[controllable-eval] FATAL: checkpoint not found: ${WM_CHECKPOINT}" >&2
    exit 2
fi

DECODE_FLAG=()
if [[ -n "${WM_NO_DECODE}" ]]; then
    DECODE_FLAG=(--no-decode)
fi

python3 -m imas_ambix.worldmodel.controllable_eval \
    --checkpoint "${WM_CHECKPOINT}" \
    --out-dir "${WM_OUT_DIR}" \
    --token-root "${WM_TOKEN_ROOT}" \
    --held-out "${WM_HELD_OUT}" \
    --target-horizon-s "${WM_TARGET_HORIZON_S}" \
    --n-frames "${WM_N_FRAMES}" \
    --n-plan "${WM_N_PLAN}" \
    --context-frames "${WM_CONTEXT_FRAMES}" \
    --n-signal-steps "${WM_N_SIGNAL_STEPS}" \
    --n-act-steps "${WM_N_ACT_STEPS}" \
    --chunk "${WM_CHUNK}" \
    --perturb-scale "${WM_PERTURB_SCALE}" \
    --n-random "${WM_N_RANDOM}" \
    --floor-ratio "${WM_FLOOR_RATIO}" \
    --margin-threshold "${WM_MARGIN}" \
    --edit-frac "${WM_EDIT_FRAC}" \
    "${DECODE_FLAG[@]}"
EXIT_CODE=$?

echo "[$(date)] controllable-eval finished (exit=${EXIT_CODE})"
echo "  verdict:  ${WM_OUT_DIR}/heldout_delta_nm.json"
echo "  dream:    ${WM_OUT_DIR}/ (coil-edit GIF)"
exit "${EXIT_CODE}"
