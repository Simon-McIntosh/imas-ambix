#!/bin/bash
#SBATCH --job-name=ambix-spacetime-corruption-ft
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --account=grpa
#SBATCH --gres=gpu:6
#SBATCH --cpus-per-task=30
#SBATCH --mem=640G
#SBATCH --time=08:00:00
#SBATCH --output=/work/projects/imas_gpu/logs/ambix-%x-%j.out
# NOTE: --qos intentionally omitted — defaults to `normal` (no 4-card group cap),
#       so gpu:6 lands on cards 0,1,2,3,6,7 and COEXISTS with the 2-card DeepSeek
#       serve (6+2 = 8/8).  Do NOT pass --qos and do NOT take DeepSeek down.
#
# ============================================================================
#  Anti-drift context-corruption + control-dropout FINE-TUNE (6-GPU DDP).
# ============================================================================
#  Builds ON the signal-conditioned spatiotemporal checkpoint (the sampling
#  model that already beats persistence on the held-out shots).  The open-loop
#  rollout still DRIFTS because the model was trained teacher-forced and never
#  saw a slightly-wrong history; this fine-tune fixes that with the GameNGen
#  recipe, in discrete-token form:
#
#   * OVERLAPPING-WINDOW enumeration — each long recording is tiled into many
#     sliding windows (50%% overlap) so the fixed corpus yields far more training
#     signal (vs one window per shot).  The window list is built ONLY from the
#     train shots (the held-out shots are subtracted first), so the shot-level
#     held-out guarantee is structural.
#   * CONTEXT-CORRUPTION — a fraction of the CONTEXT (fed-back) frame tokens are
#     replaced with random ids; the model conditions on a per-sample corruption
#     LEVEL embedding and still predicts the TRUE next frame, learning to correct
#     the errors it will later feed itself.
#   * CONTROL-DROPOUT (classifier-free guidance) — a fraction of steps zero the
#     plan + measured-signal conditioning, so at inference the conditioned
#     prediction can be amplified away from the unconditioned one (steerable
#     plasma response to the schedule).
#
#  Fine-tune INIT from the pre-trained checkpoint (--init-checkpoint); the new
#  corruption-level embedding starts at zero (strict=False) and is learned on
#  top of the frozen-start backbone.  Peak LR is kept LOW (fine-tune, not a
#  from-scratch train).  RESUME-SAFE: frequent latest.pt + SIGTERM-clean, so a
#  requeue / --time restart resumes this run's own latest.pt (resume takes
#  precedence over the init checkpoint).
# ============================================================================

set -uo pipefail

export TMPDIR="/scratch_local/${SLURM_JOB_ID}"
mkdir -p "${TMPDIR}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NCCL_P2P_LEVEL=NVL
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_ROOT}"
mkdir -p "/work/projects/imas_gpu/logs"

TOKEN_ROOT="/work/projects/imas_gpu/mast-tokens"
if [[ ! -d "${TOKEN_ROOT}/v1/frames" ]]; then
    echo "[corruption-ft] FATAL: ${TOKEN_ROOT}/v1/frames not reachable" >&2
    exit 2
fi

VENV="${REPO_ROOT}/.venv"
# shellcheck source=/dev/null
source "${VENV}/bin/activate"

# ── corpus / split ──────────────────────────────────────────────────────────
: "${WM_N_SHOTS:=9000}"        # rbb-bearing train shots (eval held out, disjoint)
: "${WM_N_EVAL_SHOTS:=4}"
: "${WM_EVAL_SHOTS:=18502,18503,18504,18505}"  # fixed held-out set (bright: 02,05)

# ── init from the pre-trained sampling checkpoint ───────────────────────────
: "${WM_INIT_CKPT:=/work/projects/imas_gpu/worldmodel/ckpt/spacetime-v2-1219852/best.pt}"

# ── schedule (fine-tune: LOW peak LR, short warmup) ─────────────────────────
: "${WM_STEPS:=12000}"
: "${WM_LR:=1.5e-4}"           # PEAK lr cap for the fine-tune (warmup→peak→cosine)
: "${WM_WARMUP_STEPS:=400}"
: "${WM_MIN_LR_RATIO:=0.02}"
: "${WM_GRAD_CLIP:=1.0}"
: "${WM_WEIGHT_DECAY:=0.1}"

# ── window / sequence (MUST match the init checkpoint's window) ──────────────
: "${WM_BATCH_SIZE:=4}"        # per-card micro-batch (M1 sat ~121/143 GB at this)
: "${WM_GRAD_ACCUM:=2}"        # effective batch = bs * 6 cards * accum = 48
: "${WM_CHUNK:=8192}"
: "${WM_N_FRAMES:=24}"
: "${WM_N_PLAN:=8}"
: "${WM_N_SIGNAL_STEPS:=4}"
: "${WM_CONTEXT_FRAMES:=8}"
: "${WM_FRAME_STRIDE:=1}"
: "${WM_WINDOW_STRIDE:=12}"    # 50%% overlap on the 24-frame span
: "${WM_MAX_WINDOWS_PER_SHOT:=24}"  # bound a very long recording's share

# ── anti-drift corruption + CFG control-dropout ─────────────────────────────
: "${WM_CORRUPTION_LEVELS:=8}"
: "${WM_CORRUPTION_MAX_RATE:=0.30}"
: "${WM_CORRUPTION_CLEAN_FRACTION:=0.25}"
: "${WM_CONTROL_DROPOUT:=0.15}"

# ── model (MUST match the init checkpoint's backbone) ───────────────────────
: "${WM_D_MODEL:=1536}"
: "${WM_N_LAYERS:=16}"
: "${WM_N_HEADS:=16}"
: "${WM_D_FF:=6144}"
: "${WM_DROPOUT:=0.15}"

: "${WM_LOG_EVERY:=25}"
: "${WM_CKPT_EVERY:=250}"      # frequent latest.pt -> resume-safe on requeue
: "${WM_EVAL_EVERY:=1000}"
: "${WM_NUM_WORKERS:=4}"       # per-rank DataLoader workers (30 cores / 6 ranks)
: "${WM_NPROC:=6}"             # processes (== GPUs)

: "${WM_OUT_DIR:=/work/projects/imas_gpu/worldmodel/ckpt/spacetime-corruption-${SLURM_JOB_ID}}"
mkdir -p "${WM_OUT_DIR}"

echo "[$(date)] spacetime-corruption-ft starting (job ${SLURM_JOB_ID} on $(hostname))"
echo "  GPUs:        ${WM_NPROC}"
echo "  init-ckpt:   ${WM_INIT_CKPT}"
echo "  out-dir:     ${WM_OUT_DIR}"
echo "  per-rank bs: ${WM_BATCH_SIZE} x accum ${WM_GRAD_ACCUM} (eff ${WM_BATCH_SIZE}*${WM_NPROC}*${WM_GRAD_ACCUM})  chunk=${WM_CHUNK}"
echo "  window:      n_frames=${WM_N_FRAMES} n_plan=${WM_N_PLAN} n_signal_steps=${WM_N_SIGNAL_STEPS} context=${WM_CONTEXT_FRAMES} stride=${WM_FRAME_STRIDE} window_stride=${WM_WINDOW_STRIDE}"
echo "  schedule:    ${WM_STEPS} steps peak-lr=${WM_LR} warmup=${WM_WARMUP_STEPS} min-lr-ratio=${WM_MIN_LR_RATIO} grad-clip=${WM_GRAD_CLIP}"
echo "  corruption:  levels=${WM_CORRUPTION_LEVELS} max_rate=${WM_CORRUPTION_MAX_RATE} clean_fraction=${WM_CORRUPTION_CLEAN_FRACTION} control_dropout=${WM_CONTROL_DROPOUT}"
echo "  shots:       ${WM_N_SHOTS} train  held-out=${WM_EVAL_SHOTS}"
echo "  model:       d_model=${WM_D_MODEL} n_layers=${WM_N_LAYERS} n_heads=${WM_N_HEADS} d_ff=${WM_D_FF} dropout=${WM_DROPOUT}"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader || true

# Per-card memory + utilisation sampler (proves all 6 cards are filled).  Runs
# for ~the first 12 minutes then exits; SIGTERM-immune is unnecessary (sidecar
# is not used per AGENTS.md §2a for this run).
(
    for i in $(seq 1 24); do
        echo "[gpu-sample $(date +%H:%M:%S)]"
        nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total \
            --format=csv,noheader || true
        sleep 30
    done
) > "${WM_OUT_DIR}/gpu_utilisation.log" 2>&1 &
SAMPLER_PID=$!
cleanup() { kill "${SAMPLER_PID}" 2>/dev/null || true; }
trap cleanup EXIT

torchrun \
    --standalone \
    --nproc_per_node="${WM_NPROC}" \
    -m imas_ambix.worldmodel.spacetime_train_v2 corpus \
    --init-checkpoint "${WM_INIT_CKPT}" \
    --d-model "${WM_D_MODEL}" \
    --n-layers "${WM_N_LAYERS}" \
    --n-heads "${WM_N_HEADS}" \
    --d-ff "${WM_D_FF}" \
    --dropout "${WM_DROPOUT}" \
    --n-shots "${WM_N_SHOTS}" \
    --n-eval-shots "${WM_N_EVAL_SHOTS}" \
    --eval-shots "${WM_EVAL_SHOTS}" \
    --steps "${WM_STEPS}" \
    --batch-size "${WM_BATCH_SIZE}" \
    --grad-accum "${WM_GRAD_ACCUM}" \
    --chunk "${WM_CHUNK}" \
    --n-frames "${WM_N_FRAMES}" \
    --n-plan "${WM_N_PLAN}" \
    --n-signal-steps "${WM_N_SIGNAL_STEPS}" \
    --context-frames "${WM_CONTEXT_FRAMES}" \
    --frame-stride "${WM_FRAME_STRIDE}" \
    --window-stride "${WM_WINDOW_STRIDE}" \
    --max-windows-per-shot "${WM_MAX_WINDOWS_PER_SHOT}" \
    --use-corruption \
    --corruption-levels "${WM_CORRUPTION_LEVELS}" \
    --corruption-max-rate "${WM_CORRUPTION_MAX_RATE}" \
    --corruption-clean-fraction "${WM_CORRUPTION_CLEAN_FRACTION}" \
    --control-dropout "${WM_CONTROL_DROPOUT}" \
    --lr "${WM_LR}" \
    --warmup-steps "${WM_WARMUP_STEPS}" \
    --min-lr-ratio "${WM_MIN_LR_RATIO}" \
    --grad-clip "${WM_GRAD_CLIP}" \
    --weight-decay "${WM_WEIGHT_DECAY}" \
    --log-every "${WM_LOG_EVERY}" \
    --ckpt-every "${WM_CKPT_EVERY}" \
    --eval-every "${WM_EVAL_EVERY}" \
    --num-workers "${WM_NUM_WORKERS}" \
    --out-dir "${WM_OUT_DIR}"
EXIT_CODE=$?

echo "[$(date)] spacetime-corruption-ft finished (exit=${EXIT_CODE})"
echo "  checkpoints: ${WM_OUT_DIR}/latest.pt (+ ckpt-<step>.pt snapshots)"
echo "  gpu sampler: ${WM_OUT_DIR}/gpu_utilisation.log"
exit "${EXIT_CODE}"
