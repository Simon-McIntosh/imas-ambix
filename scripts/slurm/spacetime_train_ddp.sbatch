#!/bin/bash
#SBATCH --job-name=ambix-spacetime-ddp
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
#  Spatiotemporal camera-video transformer — 4-GPU data-parallel (DDP) train.
# ============================================================================
#  Camera-only model over native-cadence rbb frame tokens (16x16=256 LFQ
#  tokens/frame).  EVERY spatial token keeps its identity (no bag-of-channels
#  sum); factorized space-time attention (full spatial within a frame, causal
#  temporal across frames).  Chunked cross-entropy head over the 2^18 vocab so
#  the full-resolution logits never materialise.
#
#  GPUs: requests 4.  The neighbour's DeepSeek-V4-Flash holds 2 of the node's 8
#  GPUs; this job BURSTS its 4 into the free 4-7 region (4 DSv4 + 4 here = ...
#  no: DSv4 has 2, this has 4 => 6 of 8 used).  We do NOT touch DeepSeek.
#
#  torchrun launches one process per GPU (nproc_per_node=4); each rank pins its
#  card, trains a DISJOINT shot shard (DistributedSampler), and the chunked
#  next-frame cross-entropy is computed INSIDE forward so DDP's reducer
#  all-reduces grads across the 4 cards.  Rank 0 ONLY logs / checkpoints / evals
#  (symmetric barriers keep the ring in sync).  Global batch = per-rank x 4.
#
#  RESUME-SAFE: checkpoints land on GPFS under WM_OUT_DIR; on restart the
#  trainer auto-resumes from latest.pt (all ranks load the same weights before
#  the DDP wrap).  Drain sidecar on; SIGTERM-clean.
# ============================================================================

set -uo pipefail

export TMPDIR="/scratch_local/${SLURM_JOB_ID}"
mkdir -p "${TMPDIR}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# NCCL on a single node over NVLink — keep it on the local fabric.
export NCCL_P2P_LEVEL=NVL
# Reduce allocator fragmentation — the chunk-logit tensor is the large
# transient; expandable segments let it grow/shrink without a fixed slab.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_ROOT}"
mkdir -p "/work/projects/imas_gpu/logs"

# ── Drain-forensics sidecar (default-on for long GPU jobs; AGENTS.md §2a) ────
bash "${REPO_ROOT}/scripts/slurm/drain_sidecar.sh" &
SIDECAR_PID=$!
cleanup() {
    echo "[$(date)] spacetime-ddp teardown"
    kill "${SIDECAR_PID}" 2>/dev/null || true
}
trap cleanup EXIT

TOKEN_ROOT="/work/projects/imas_gpu/mast-tokens"
if [[ ! -d "${TOKEN_ROOT}/v1/frames" ]]; then
    echo "[spacetime-ddp] FATAL: ${TOKEN_ROOT}/v1/frames not reachable" >&2
    kill "${SIDECAR_PID}" 2>/dev/null || true
    exit 2
fi

VENV="${REPO_ROOT}/.venv"
# shellcheck source=/dev/null
source "${VENV}/bin/activate"

: "${WM_N_SHOTS:=9000}"        # rbb-bearing train shots (~11k exist; eval held out)
: "${WM_N_EVAL_SHOTS:=4}"
: "${WM_EVAL_SHOTS:=18502,18503,18504,18505}"  # fixed held-out set (disjoint from train)
: "${WM_STEPS:=36000}"          # ~3.3 h at the measured 3.05 st/s (world=4)
# PER-RANK batch + CE chunk.  The chunk-logit tensor is
# chunk * 2^18 * 4 bytes (plus backward) — modest; the activations of the
# factorized attention dominate.  B=4 per rank fills a 140 GB card well with the
# d_model=1024 / 16-layer backbone; global batch = B x 4 ranks.
: "${WM_BATCH_SIZE:=4}"
: "${WM_CHUNK:=8192}"           # CE token-chunk over the flattened (frame,pos) axis
: "${WM_N_FRAMES:=24}"          # frames per window (native ~600 Hz cadence)
: "${WM_N_PLAN:=8}"             # pulse-schedule conditioning prefix steps
: "${WM_CONTEXT_FRAMES:=8}"     # context frames given at eval/dream time
: "${WM_FRAME_STRIDE:=1}"       # 1 = native cadence
: "${WM_LR:=3e-4}"              # PEAK lr (warmup→peak, then cosine decay)
: "${WM_WARMUP_STEPS:=800}"     # linear warmup before cosine decay (divergence fix)
: "${WM_MIN_LR_RATIO:=0.01}"    # cosine decays to 1% of peak by the last step
: "${WM_GRAD_CLIP:=1.0}"        # max grad-norm (divergence fix)
: "${WM_WEIGHT_DECAY:=0.1}"     # raised for generalization (was 0.05)
: "${WM_LOG_EVERY:=25}"
: "${WM_CKPT_EVERY:=500}"
: "${WM_EVAL_EVERY:=1000}"
: "${WM_NUM_WORKERS:=5}"        # per-rank DataLoader workers (24 cores / 4 ranks)
: "${WM_D_MODEL:=1024}"
: "${WM_N_LAYERS:=16}"
: "${WM_N_HEADS:=16}"
: "${WM_D_FF:=4096}"
: "${WM_DROPOUT:=0.15}"         # raised for generalization (0.1-0.2 band)
: "${WM_NPROC:=4}"              # processes (== GPUs) per node

: "${WM_OUT_DIR:=/work/projects/imas_gpu/worldmodel/ckpt/spacetime-${SLURM_JOB_ID}}"
mkdir -p "${WM_OUT_DIR}"

echo "[$(date)] spacetime-ddp starting (job ${SLURM_JOB_ID} on $(hostname))"
echo "  GPUs:        ${WM_NPROC} (DeepSeek-V4-Flash keeps its 2 — node stays shared)"
echo "  out-dir:     ${WM_OUT_DIR}"
echo "  per-rank bs: ${WM_BATCH_SIZE} (global ${WM_BATCH_SIZE}*${WM_NPROC})  chunk=${WM_CHUNK}"
echo "  window:      n_frames=${WM_N_FRAMES} n_plan=${WM_N_PLAN} context=${WM_CONTEXT_FRAMES} stride=${WM_FRAME_STRIDE}"
echo "  steps:       ${WM_STEPS} peak-lr=${WM_LR} warmup=${WM_WARMUP_STEPS} min-lr-ratio=${WM_MIN_LR_RATIO} grad-clip=${WM_GRAD_CLIP}"
echo "  shots:       ${WM_N_SHOTS} train  held-out=${WM_EVAL_SHOTS}"
echo "  regularize:  weight_decay=${WM_WEIGHT_DECAY} dropout=${WM_DROPOUT}"
echo "  model:       d_model=${WM_D_MODEL} n_layers=${WM_N_LAYERS} n_heads=${WM_N_HEADS} d_ff=${WM_D_FF}"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader || true

# torchrun spawns WM_NPROC ranks; each sets RANK/LOCAL_RANK/WORLD_SIZE which the
# trainer reads (DistEnv.from_environment) — WORLD_SIZE>1 activates DDP.
torchrun \
    --standalone \
    --nproc_per_node="${WM_NPROC}" \
    -m imas_ambix.worldmodel.spacetime_train corpus \
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

echo "[$(date)] spacetime-ddp finished (exit=${EXIT_CODE})"
echo "  checkpoints: ${WM_OUT_DIR}/latest.pt (+ ckpt-<step>.pt snapshots)"
exit "${EXIT_CODE}"
