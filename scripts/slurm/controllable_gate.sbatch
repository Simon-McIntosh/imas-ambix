#!/bin/bash
#SBATCH --job-name=ambix-controllable-gate
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
#  M4 controllability GATE — actuator-PLAN conditioning, overfit + test.
# ============================================================================
#  The cheap de-risking gate for the playable-plasma PLAY bridge.  M3 proved the
#  MEASURED-signal conditioning is NOT controllable (redundant observations);
#  this asks the go/no-go question for the 6-GPU re-train: does conditioning the
#  camera model on the demanded actuator PLAN (control-surface=actuator-vector,
#  LOCKED) make the controls causally LOAD-BEARING?
#
#  The re-design (control-conditioning-survey) applies three fixes TOGETHER:
#    (1) camera-history BOTTLENECK — independent per-frame noise/mask of the past
#        FRAME EMBEDDINGS (the corrected lever; the prior gate left the history
#        near-clean and the model ignored the plan);
#    (2) AdaLN-Zero per-layer plan conditioning (the plan modulates every block,
#        gate zero-init) REPLACING the prepended plan tokens;
#    (3) inverse-dynamics auxiliary (predict the plan from consecutive latents).
#
#  Pipeline (1 GPU, minutes — NOT the 6-GPU re-train):
#    1. Overfit a handful of shots with the three fixes + HIGH observation-dropout
#       + control-dropout (CFG).
#    2. PRIMARY gate — ΔN-M action-sensitivity: autoregressive token-rollout
#       divergence between the TRUE plan and a RANDOM plan, scored against a
#       RANDOM-vs-RANDOM noise floor (PASS = true-vs-random clears the floor).
#    3. SECONDARY gate — teacher-forced corrupted-context token mismatch (fast).
#    4. (optional, --ablate) per-leg ablation: which fix moves the needle.
#    5. (optional, --confirm-pixels) DECODE true-plan vs random-plan + M3 metrics.
#
#  Single GPU — bursts onto a free card; leaves the DeepSeek-V4-Flash neighbour
#  untouched.  scancel + normal --time are fine (AGENTS.md §2a).  Window flags
#  MUST match the demo/training window (24f / 8ctx / 8plan / 4 signal steps).
#
#  Inputs (env):
#    WM_SHOTS    — comma-separated shots to overfit + gate on (default 18502-5)
#    WM_OUT_JSON — verdict JSON path
#    WM_STEPS    — overfit steps (default 600)
#    WM_OBS_DROPOUT — measured-observation dropout (default 0.8)
#    WM_HB_NOISE_STD / WM_HB_MASK_PROB / WM_HB_MAX_STRENGTH — history bottleneck
#    WM_INV_DYN_WEIGHT — inverse-dynamics auxiliary weight (default 1.0)
#    WM_N_RANDOM — random plans for the ΔN-M noise floor (default 4)
#    WM_GAS_SCALE / WM_NBI_SCALE — command amplification factor (default 3.0)
#    WM_ABLATE — set non-empty to run the per-leg ablation
#    WM_CONFIRM_PIXELS — set non-empty to also decode + score pixel space
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

: "${WM_SHOTS:=18502,18503,18504,18505}"
: "${WM_OUT_DIR:=/work/projects/imas_gpu/worldmodel/m4_controllability_gate}"
: "${WM_OUT_JSON:=${WM_OUT_DIR}/gate_verdict_${SLURM_JOB_ID}.json}"
: "${WM_STEPS:=600}"
: "${WM_LR:=3e-4}"
: "${WM_OBS_DROPOUT:=0.8}"
: "${WM_CONTROL_DROPOUT:=0.15}"
# camera-history bottleneck (the corrected lever) — set WM_HB_MAX_STRENGTH=0 to
# disable it (reproduces the prior failed near-clean-history recipe).
: "${WM_HB_NOISE_STD:=1.0}"
: "${WM_HB_MASK_PROB:=0.5}"
: "${WM_HB_MAX_STRENGTH:=1.0}"
: "${WM_HB_CLEAN_FRACTION:=0.2}"
: "${WM_INV_DYN_WEIGHT:=1.0}"
: "${WM_N_RANDOM:=4}"
: "${WM_FLOOR_RATIO:=1.5}"
: "${WM_GAS_SCALE:=3.0}"
: "${WM_NBI_SCALE:=3.0}"
: "${WM_MARGIN:=0.02}"
# window — MUST match the demo window
: "${WM_N_FRAMES:=24}"
: "${WM_N_PLAN:=8}"
: "${WM_CONTEXT_FRAMES:=8}"
: "${WM_FRAME_STRIDE:=1}"
: "${WM_N_SIGNAL_STEPS:=4}"
: "${WM_N_ACT_STEPS:=8}"
: "${WM_CHUNK:=4096}"
: "${WM_D_MODEL:=256}"
: "${WM_N_LAYERS:=6}"
: "${WM_N_HEADS:=8}"
: "${WM_D_FF:=1024}"
: "${WM_CONFIRM_PIXELS:=}"
: "${WM_ABLATE:=}"
# warm-start the backbone from the M2 forecaster (strict=False; AdaLN MLP +
# inverse-dynamics head start at init).  Set WM_INIT_CHECKPOINT= (empty) to
# overfit from scratch instead.
: "${WM_INIT_CHECKPOINT:=/work/projects/imas_gpu/worldmodel/ckpt/spacetime-corruption-1220391/best.pt}"
# TRANSIENT-window selection is ON by default (the gate is only fair where the
# actuator plan actually varies).  Set WM_NO_TRANSIENT non-empty to overfit the
# centred flat-top window instead (debug).
: "${WM_NO_TRANSIENT:=}"

mkdir -p "${WM_OUT_DIR}"

echo "[$(date)] M4 controllability gate (job ${SLURM_JOB_ID} on $(hostname))"
echo "  shots:        ${WM_SHOTS}"
echo "  out-json:     ${WM_OUT_JSON}"
echo "  steps:        ${WM_STEPS}  obs_dropout=${WM_OBS_DROPOUT}  control_dropout=${WM_CONTROL_DROPOUT}"
echo "  history-bottleneck: noise_std=${WM_HB_NOISE_STD} mask_prob=${WM_HB_MASK_PROB} max=${WM_HB_MAX_STRENGTH}"
echo "  inv_dyn_weight=${WM_INV_DYN_WEIGHT}  n_random=${WM_N_RANDOM}  margin=${WM_MARGIN}"
echo "  window:       n_frames=${WM_N_FRAMES} n_plan=${WM_N_PLAN} context=${WM_CONTEXT_FRAMES} n_act_steps=${WM_N_ACT_STEPS}"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader || true

EXTRA_FLAGS=()
if [[ -n "${WM_CONFIRM_PIXELS}" ]]; then
    EXTRA_FLAGS+=(--confirm-pixels)
fi
if [[ -n "${WM_ABLATE}" ]]; then
    EXTRA_FLAGS+=(--ablate)
fi
if [[ -n "${WM_NO_TRANSIENT}" ]]; then
    EXTRA_FLAGS+=(--no-transient-windows)
fi
if [[ -n "${WM_INIT_CHECKPOINT}" ]]; then
    EXTRA_FLAGS+=(--init-checkpoint "${WM_INIT_CHECKPOINT}")
fi

python -m imas_ambix.worldmodel.controllable_gate_run \
    --shots "${WM_SHOTS}" \
    --out-json "${WM_OUT_JSON}" \
    --steps "${WM_STEPS}" \
    --lr "${WM_LR}" \
    --observation-dropout "${WM_OBS_DROPOUT}" \
    --control-dropout "${WM_CONTROL_DROPOUT}" \
    --hb-noise-std "${WM_HB_NOISE_STD}" \
    --hb-mask-prob "${WM_HB_MASK_PROB}" \
    --hb-max-strength "${WM_HB_MAX_STRENGTH}" \
    --hb-clean-fraction "${WM_HB_CLEAN_FRACTION}" \
    --inverse-dynamics-weight "${WM_INV_DYN_WEIGHT}" \
    --n-random "${WM_N_RANDOM}" \
    --floor-ratio "${WM_FLOOR_RATIO}" \
    --gas-scale "${WM_GAS_SCALE}" \
    --nbi-scale "${WM_NBI_SCALE}" \
    --margin-threshold "${WM_MARGIN}" \
    --n-frames "${WM_N_FRAMES}" \
    --n-plan "${WM_N_PLAN}" \
    --context-frames "${WM_CONTEXT_FRAMES}" \
    --frame-stride "${WM_FRAME_STRIDE}" \
    --n-signal-steps "${WM_N_SIGNAL_STEPS}" \
    --n-act-steps "${WM_N_ACT_STEPS}" \
    --chunk "${WM_CHUNK}" \
    --d-model "${WM_D_MODEL}" \
    --n-layers "${WM_N_LAYERS}" \
    --n-heads "${WM_N_HEADS}" \
    --d-ff "${WM_D_FF}" \
    --device cuda \
    "${EXTRA_FLAGS[@]}"
EXIT_CODE=$?

echo "[$(date)] M4 controllability gate finished (exit=${EXIT_CODE})"
echo "  verdict: ${WM_OUT_JSON}"
exit "${EXIT_CODE}"
