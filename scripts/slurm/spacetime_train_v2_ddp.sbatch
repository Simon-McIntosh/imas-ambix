#!/bin/bash
#SBATCH --job-name=ambix-spacetime-v2-ddp
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --account=grpa
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --mem=600G
#SBATCH --time=08:00:00
#SBATCH --output=/work/projects/imas_gpu/logs/ambix-%x-%j.out
# NOTE: --qos intentionally omitted — gpu_0003_grpa QoS is auto-applied.
#
# ============================================================================
#  Signal-conditioned spatiotemporal camera transformer — 4-GPU DDP train.
# ============================================================================
#  Same camera-video target as the plan-only model (native-cadence rbb frame
#  tokens, 16x16=256 LFQ tokens/frame, factorized space-time attention, chunked
#  cross-entropy head over the 2^18 vocab) — but the model ALSO conditions on
#  the MEASURED diagnostic streams (magnetics xma, density interferometer,
#  soft_x_rays, and the L2 measured groups summary / pf_active / gas_injection),
#  prepended as conditioning frames alongside the plan.  The hypothesis is that
#  the measured plasma state lets the model FORECAST (beat persistence) rather
#  than dream a coherent-but-wrong clip from the demand alone.
#
#  SCALED to use the idle H200 headroom (the plan-only run sat at ~52% / card):
#  a wider backbone (d_model 1536) + the signal prefix lengthens the modelled
#  sequence.  The smoke measures peak memory at this size with batch 4 on one
#  card; the orchestrator confirms it lands in the ~110-130 GB target before the
#  4-GPU launch.
#
#  torchrun launches one process per GPU (nproc_per_node=4); each rank pins its
#  card, trains a DISJOINT shot shard, and the chunked next-frame cross-entropy
#  is computed INSIDE forward so DDP all-reduces grads across the 4 cards.  The
#  plan-channel count AND the signal-stream set + widths are decided on rank 0
#  and broadcast so every rank builds the IDENTICAL model.  Rank 0 ONLY logs /
#  checkpoints / evals.  RESUME-SAFE; drain sidecar on; SIGTERM-clean.
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

# ── Drain-forensics sidecar (default-on for long GPU jobs; AGENTS.md §2a) ────
bash "${REPO_ROOT}/scripts/slurm/drain_sidecar.sh" &
SIDECAR_PID=$!
cleanup() {
    echo "[$(date)] spacetime-v2-ddp teardown"
    kill "${SIDECAR_PID}" 2>/dev/null || true
}
trap cleanup EXIT

TOKEN_ROOT="/work/projects/imas_gpu/mast-tokens"
if [[ ! -d "${TOKEN_ROOT}/v1/frames" ]]; then
    echo "[spacetime-v2-ddp] FATAL: ${TOKEN_ROOT}/v1/frames not reachable" >&2
    kill "${SIDECAR_PID}" 2>/dev/null || true
    exit 2
fi

VENV="${REPO_ROOT}/.venv"
# shellcheck source=/dev/null
source "${VENV}/bin/activate"

: "${WM_N_SHOTS:=9000}"        # rbb-bearing train shots (~11k exist; eval held out)
: "${WM_N_EVAL_SHOTS:=4}"
: "${WM_EVAL_SHOTS:=18502,18503,18504,18505}"  # fixed held-out set (disjoint)
: "${WM_STEPS:=36000}"
: "${WM_BATCH_SIZE:=4}"
: "${WM_CHUNK:=8192}"           # CE token-chunk over the flattened (frame,pos) axis
: "${WM_N_FRAMES:=24}"          # frames per window (native ~600 Hz cadence)
: "${WM_N_PLAN:=8}"             # pulse-schedule conditioning prefix steps
: "${WM_N_SIGNAL_STEPS:=4}"     # conditioning steps per measured stream
: "${WM_CONTEXT_FRAMES:=8}"     # context frames given at eval/dream time
: "${WM_FRAME_STRIDE:=1}"       # 1 = native cadence
: "${WM_LR:=3e-4}"              # PEAK lr (warmup→peak, then cosine decay)
: "${WM_WARMUP_STEPS:=800}"     # linear warmup before cosine decay
: "${WM_MIN_LR_RATIO:=0.01}"    # cosine decays to 1% of peak by the last step
: "${WM_GRAD_CLIP:=1.0}"        # max grad-norm
: "${WM_WEIGHT_DECAY:=0.1}"
: "${WM_LOG_EVERY:=25}"
: "${WM_CKPT_EVERY:=500}"
: "${WM_EVAL_EVERY:=1000}"
: "${WM_NUM_WORKERS:=5}"        # per-rank DataLoader workers (24 cores / 4 ranks)
# Scaled backbone — wider than the plan-only run (1024) to use the idle headroom.
: "${WM_D_MODEL:=1536}"
: "${WM_N_LAYERS:=16}"
: "${WM_N_HEADS:=16}"           # 1536 / 16 = 96 per head
: "${WM_D_FF:=6144}"            # 4 * d_model
: "${WM_DROPOUT:=0.15}"
: "${WM_NPROC:=4}"              # processes (== GPUs) per node

: "${WM_OUT_DIR:=/work/projects/imas_gpu/worldmodel/ckpt/spacetime-v2-${SLURM_JOB_ID}}"
mkdir -p "${WM_OUT_DIR}"

echo "[$(date)] spacetime-v2-ddp starting (job ${SLURM_JOB_ID} on $(hostname))"
echo "  GPUs:        ${WM_NPROC}"
echo "  out-dir:     ${WM_OUT_DIR}"
echo "  per-rank bs: ${WM_BATCH_SIZE} (global ${WM_BATCH_SIZE}*${WM_NPROC})  chunk=${WM_CHUNK}"
echo "  window:      n_frames=${WM_N_FRAMES} n_plan=${WM_N_PLAN} n_signal_steps=${WM_N_SIGNAL_STEPS} context=${WM_CONTEXT_FRAMES} stride=${WM_FRAME_STRIDE}"
echo "  steps:       ${WM_STEPS} peak-lr=${WM_LR} warmup=${WM_WARMUP_STEPS} min-lr-ratio=${WM_MIN_LR_RATIO} grad-clip=${WM_GRAD_CLIP}"
echo "  shots:       ${WM_N_SHOTS} train  held-out=${WM_EVAL_SHOTS}"
echo "  regularize:  weight_decay=${WM_WEIGHT_DECAY} dropout=${WM_DROPOUT}"
echo "  model:       d_model=${WM_D_MODEL} n_layers=${WM_N_LAYERS} n_heads=${WM_N_HEADS} d_ff=${WM_D_FF}"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader || true

torchrun \
    --standalone \
    --nproc_per_node="${WM_NPROC}" \
    -m imas_ambix.worldmodel.spacetime_train_v2 corpus \
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
    --chunk "${WM_CHUNK}" \
    --n-frames "${WM_N_FRAMES}" \
    --n-plan "${WM_N_PLAN}" \
    --n-signal-steps "${WM_N_SIGNAL_STEPS}" \
    --context-frames "${WM_CONTEXT_FRAMES}" \
    --frame-stride "${WM_FRAME_STRIDE}" \
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

echo "[$(date)] spacetime-v2-ddp finished (exit=${EXIT_CODE})"
echo "  checkpoints: ${WM_OUT_DIR}/latest.pt (+ ckpt-<step>.pt snapshots)"
exit "${EXIT_CODE}"
