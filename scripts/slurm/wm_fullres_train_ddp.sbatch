#!/bin/bash
#SBATCH --job-name=ambix-wm-fullres-ddp
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --account=grpa
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=14
#SBATCH --mem=300G
#SBATCH --time=10:00:00
#SBATCH --output=/work/projects/imas_gpu/logs/ambix-%x-%j.out
# NOTE: --qos intentionally omitted — gpu_0003_grpa QoS is auto-applied.
#
# ============================================================================
#  FULL-RESOLUTION rbb world-model retrain on TWO GPUs (data-parallel DDP).
# ============================================================================
#  Group-A reservation is 4 H200; DeepSeek-V4-Flash holds 2, so 2 are FREE.
#  --gres=gpu:2 => 2 (DSv4) + 2 (this) = 4 — DeepSeek STAYS UP.  We do NOT
#  shut down or touch the neighbour's LLM server.
#
#  torchrun launches one process per GPU (nproc_per_node=2); each rank pins its
#  own card, trains a DISJOINT shot shard (DistributedSampler), and the loss is
#  the CHUNKED next-token cross-entropy computed inside forward so DDP's reducer
#  all-reduces grads across both cards.  Rank 0 ONLY logs / checkpoints / evals
#  (barriers keep ranks in sync).  Global batch = per-rank batch × 2.
#
#  MEMORY: at ~50% on one card there is ~70 GB headroom; the per-rank batch +
#  CE chunk size below are tuned to fill most of each 140 GB card with a safe
#  ~15-20 GB margin.  Override via --export=ALL,WM_BATCH_SIZE=.. ,WM_CHUNK=.. .
#
#  RESUME-SAFE: checkpoints land on GPFS under WM_OUT_DIR; on restart the
#  trainer auto-resumes from latest.pt (all ranks load the same weights before
#  the DDP wrap).
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
# Reduce allocator fragmentation — the full-res rbb chunk-logit tensor is the
# large transient; expandable segments let it grow/shrink without reserving a
# fixed slab (avoids the OOM-on-a-fragmented-card failure mode).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_ROOT}"
mkdir -p "/work/projects/imas_gpu/logs"

# ── Drain-forensics sidecar (default-on for long GPU jobs; AGENTS.md §2a) ────
bash "${REPO_ROOT}/scripts/slurm/drain_sidecar.sh" &
SIDECAR_PID=$!
cleanup() {
    echo "[$(date)] wm-fullres-ddp teardown"
    kill "${SIDECAR_PID}" 2>/dev/null || true
}
trap cleanup EXIT

TOKEN_ROOT="/work/projects/imas_gpu/mast-tokens"
if [[ ! -d "${TOKEN_ROOT}/v2/signals_hf" ]]; then
    echo "[wm-fullres-ddp] FATAL: ${TOKEN_ROOT}/v2/signals_hf not reachable" >&2
    kill "${SIDECAR_PID}" 2>/dev/null || true
    exit 2
fi

VENV="${REPO_ROOT}/.venv"
# shellcheck source=/dev/null
source "${VENV}/bin/activate"

: "${WM_N_SHOTS:=3000}"
: "${WM_N_EVAL_SHOTS:=4}"
: "${WM_STEPS:=20000}"
# PER-RANK batch + CE chunk.  The full-res rbb chunk-logit tensor is
# B*(T-1)*chunk*2^18*4 bytes (plus backward) — the memory ceiling.  Empirically
# B=8,chunk=16 is comfortable on a 140 GB H200 alongside the 1.03B-param model
# (chunk tensor ~8 GB; peak ~30-40 GB used/card with margin).  B*chunk>~600 (e.g.
# B=12,chunk=64) OOMs — keep their product modest.  Global batch = B × 2 ranks.
: "${WM_BATCH_SIZE:=8}"         # PER-RANK batch (global = ×2)
: "${WM_CHUNK:=16}"             # CE channel-chunk (B*chunk is the mem driver)
: "${WM_N_STEPS:=64}"
: "${WM_CONTEXT_STEPS:=16}"
: "${WM_LR:=1e-3}"
: "${WM_WEIGHT_DECAY:=0.01}"
: "${WM_LOG_EVERY:=50}"
: "${WM_CKPT_EVERY:=500}"
: "${WM_EVAL_EVERY:=2000}"
: "${WM_NUM_WORKERS:=5}"        # per-rank DataLoader workers (14 cores / 2 ranks)
: "${WM_D_MODEL:=384}"
: "${WM_N_LAYERS:=8}"
: "${WM_N_HEADS:=8}"
: "${WM_DROPOUT:=0.1}"
: "${WM_NPROC:=2}"              # processes (== GPUs) per node

# Node-local NVMe assembled-sample cache (shared across ranks; atomic writes).
: "${WM_CACHE_DIR:=/scratch_local/${SLURM_JOB_ID}/wm_token_cache}"
mkdir -p "${WM_CACHE_DIR}"
export WM_CACHE_DIR

: "${WM_OUT_DIR:=/work/projects/imas_gpu/worldmodel/ckpt/${SLURM_JOB_ID}}"
mkdir -p "${WM_OUT_DIR}"

echo "[$(date)] wm-fullres-ddp starting (job ${SLURM_JOB_ID} on $(hostname))"
echo "  GPUs:        ${WM_NPROC} (DeepSeek-V4-Flash keeps its 2 — node stays shared)"
echo "  out-dir:     ${WM_OUT_DIR}"
echo "  cache-dir:   ${WM_CACHE_DIR}"
echo "  per-rank bs: ${WM_BATCH_SIZE} (global ${WM_BATCH_SIZE}*${WM_NPROC})  chunk=${WM_CHUNK}"
echo "  steps:       ${WM_STEPS} n_steps=${WM_N_STEPS} context=${WM_CONTEXT_STEPS} lr=${WM_LR}"
echo "  model:       d_model=${WM_D_MODEL} n_layers=${WM_N_LAYERS} n_heads=${WM_N_HEADS} dropout=${WM_DROPOUT}"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader || true

# torchrun spawns WM_NPROC ranks; each sets RANK/LOCAL_RANK/WORLD_SIZE which
# train_corpus reads (DistEnv.from_environment) — WORLD_SIZE>1 activates DDP.
# The corpus CLI passes --loss-chunk-channels through to CorpusTrainConfig.
torchrun \
    --standalone \
    --nproc_per_node="${WM_NPROC}" \
    -m imas_ambix.worldmodel.train corpus \
    --d-model "${WM_D_MODEL}" \
    --n-layers "${WM_N_LAYERS}" \
    --n-heads "${WM_N_HEADS}" \
    --dropout "${WM_DROPOUT}" \
    --n-shots "${WM_N_SHOTS}" \
    --n-eval-shots "${WM_N_EVAL_SHOTS}" \
    --steps "${WM_STEPS}" \
    --batch-size "${WM_BATCH_SIZE}" \
    --loss-chunk-channels "${WM_CHUNK}" \
    --n-steps "${WM_N_STEPS}" \
    --context-steps "${WM_CONTEXT_STEPS}" \
    --lr "${WM_LR}" \
    --weight-decay "${WM_WEIGHT_DECAY}" \
    --log-every "${WM_LOG_EVERY}" \
    --ckpt-every "${WM_CKPT_EVERY}" \
    --eval-every "${WM_EVAL_EVERY}" \
    --num-workers "${WM_NUM_WORKERS}" \
    --cache-dir "${WM_CACHE_DIR}" \
    --out-dir "${WM_OUT_DIR}"
EXIT_CODE=$?

echo "[$(date)] wm-fullres-ddp finished (exit=${EXIT_CODE})"
echo "  checkpoints: ${WM_OUT_DIR}/latest.pt (+ ckpt-<step>.pt snapshots)"
exit "${EXIT_CODE}"
