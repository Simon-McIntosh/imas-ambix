#!/bin/bash
#SBATCH --job-name=ambix-playable-demo
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --account=grpa
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --output=/work/projects/imas_gpu/logs/ambix-%x-%j.out
# NOTE: --qos intentionally omitted — defaults to normal (no 4-GPU cap), so this
# single-card demo bursts onto a free card and COEXISTS with the DeepSeek serve.
#
# ============================================================================
#  RENDERED "play the plasma with a coil" demos — the sprint headline artifact.
# ============================================================================
#  Runs imas_ambix.worldmodel.playable_demo on a trained controllable checkpoint
#  for ONE driveable held-out exemplar shot.  Renders three demos, each decoding
#  its rollouts in ONE Open-MAGVIT2 VQ pass:
#    1. faithful_replay        — GT decode | TRUE-plan dream (coherence check)
#    2. counterfactual_two_plans — true plan vs a different realistic plan
#    3. coil_knob_sweep        — sweep P6-upper, plot plasma centroid vs the knob
#
#  Single GPU — bursts onto a free card; leaves DeepSeek untouched.  --target-
#  horizon-s MUST match the training run (0.25s) so the demos condition on the
#  same physical horizon the model saw.  --token-root is the curated-token-view
#  so the held-out tokens are the SAME percentile-exposure as the train corpus.
#  Output lands directly in the repo's docs/figures dir so the plan can embed it.
#
#  Env:
#    WM_CHECKPOINT   — the trained controllable checkpoint (default 1220940 latest)
#    WM_OUT_DIR      — figures output dir (GIFs + PNGs + JSON)
#    WM_TOKEN_ROOT   — token-root for held-out tokens (curated-token-view)
#    WM_SHOT         — the driveable held-out exemplar (default 18502)
#    WM_TARGET_HORIZON_S — physical window span (MUST match training, default 0.25)
#    WM_KNOB_KEY     — the swept position coil (default p6u_current)
#    WM_CF_SCALE     — counterfactual re-actuation strength (default 0.3)
#    WM_CHUNK        — argmax-frame chunk for the head (default 2048, single-card)
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
: "${WM_OUT_DIR:=${REPO_ROOT}/docs/figures/playable-plasma-wm-v0}"
: "${WM_TOKEN_ROOT:=/work/projects/imas_gpu/worldmodel/curated-token-view}"
: "${WM_SHOT:=18502}"
: "${WM_TARGET_HORIZON_S:=0.25}"
: "${WM_N_FRAMES:=24}"
: "${WM_N_PLAN:=8}"
: "${WM_CONTEXT_FRAMES:=8}"
: "${WM_N_SIGNAL_STEPS:=4}"
: "${WM_N_ACT_STEPS:=8}"
# decode in chunks so the 2**18-vocab head fits the single-card budget.
: "${WM_CHUNK:=2048}"
: "${WM_KNOB_KEY:=p6u_current}"
: "${WM_CF_SCALE:=0.3}"

mkdir -p "${WM_OUT_DIR}"

echo "[$(date)] playable-demo (job ${SLURM_JOB_ID} on $(hostname))"
echo "  checkpoint:  ${WM_CHECKPOINT}"
echo "  out-dir:     ${WM_OUT_DIR}"
echo "  token-root:  ${WM_TOKEN_ROOT}"
echo "  shot:        ${WM_SHOT}"
echo "  horizon:     ${WM_TARGET_HORIZON_S}s (n_frames=${WM_N_FRAMES} context=${WM_CONTEXT_FRAMES})"
echo "  knob:        ${WM_KNOB_KEY}   counterfactual scale=${WM_CF_SCALE}"
nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader || true

if [[ ! -f "${WM_CHECKPOINT}" ]]; then
    echo "[playable-demo] FATAL: checkpoint not found: ${WM_CHECKPOINT}" >&2
    exit 2
fi

python3 -m imas_ambix.worldmodel.playable_demo \
    --checkpoint "${WM_CHECKPOINT}" \
    --out-dir "${WM_OUT_DIR}" \
    --token-root "${WM_TOKEN_ROOT}" \
    --shot "${WM_SHOT}" \
    --target-horizon-s "${WM_TARGET_HORIZON_S}" \
    --n-frames "${WM_N_FRAMES}" \
    --n-plan "${WM_N_PLAN}" \
    --context-frames "${WM_CONTEXT_FRAMES}" \
    --n-signal-steps "${WM_N_SIGNAL_STEPS}" \
    --n-act-steps "${WM_N_ACT_STEPS}" \
    --chunk "${WM_CHUNK}" \
    --knob-key "${WM_KNOB_KEY}" \
    --counterfactual-scale "${WM_CF_SCALE}"
EXIT_CODE=$?

echo "[$(date)] playable-demo finished (exit=${EXIT_CODE})"
echo "  out-dir:  ${WM_OUT_DIR}"
ls -la "${WM_OUT_DIR}"/*.gif "${WM_OUT_DIR}"/*.png "${WM_OUT_DIR}"/*.json 2>/dev/null || true
exit "${EXIT_CODE}"
