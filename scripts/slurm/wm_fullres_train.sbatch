#!/bin/bash
#SBATCH --job-name=ambix-wm-fullres-train
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --account=grpa
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=320G
#SBATCH --time=10:00:00
#SBATCH --output=/work/projects/imas_gpu/logs/ambix-%x-%j.out
# NOTE: --qos intentionally omitted — gpu_0003_grpa QoS is auto-applied.
#
# ============================================================================
#  FULL-RESOLUTION rbb world-model retrain (the demoable predict→decode model).
# ============================================================================
#  Same plan-conditioned multi-modal trainer as worldmodel_train.sbatch, but:
#    * rbb is FULL resolution — 16×16 = 256 tokens/frame (camera_grid_stride=1
#      in default_modalities), so the model's rbb prediction is a COMPLETE
#      frame that decodes back to a real image.  The other four cameras stay
#      at stride 4 (16-token coarse) auxiliary context.
#    * the loss is the CHUNKED next-token cross-entropy (train.py
#      chunked_next_token_nll) so the 256-channel × 2^18-vocab rbb head never
#      materialises all-channel logits — peak head memory ~chunk×vocab.
#    * SINGLE GPU (--gres=gpu:1) — the neighbour's LLM server
#      (DeepSeek-V4-Flash) STAYS UP; we do NOT shut it down and do NOT take the
#      whole node.  The full-res model fits on one H200 with the chunked head.
#
#  RESUME-SAFE (the H200 node has a drain history): checkpoints land on GPFS
#  under WM_OUT_DIR (default /work/.../worldmodel/ckpt/<jobid>); on restart the
#  trainer auto-resumes from latest.pt.  To resume a DRAINED job into a fresh
#  allocation, point WM_OUT_DIR at the previous job's ckpt dir:
#    sbatch --export=ALL,WM_OUT_DIR=/work/projects/imas_gpu/worldmodel/ckpt/<prev-jobid> ...
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

# ── Drain-forensics sidecar (default-on for long GPU jobs; AGENTS.md §2a) ────
bash "${REPO_ROOT}/scripts/slurm/drain_sidecar.sh" &
SIDECAR_PID=$!
cleanup() {
    echo "[$(date)] wm-fullres-train teardown"
    kill "${SIDECAR_PID}" 2>/dev/null || true
}
trap cleanup EXIT

# ── Preflight: refuse to start if the token corpus is unreachable ───────────
TOKEN_ROOT="/work/projects/imas_gpu/mast-tokens"
if [[ ! -d "${TOKEN_ROOT}/v2/signals_hf" ]]; then
    echo "[wm-fullres-train] FATAL: ${TOKEN_ROOT}/v2/signals_hf not reachable" >&2
    kill "${SIDECAR_PID}" 2>/dev/null || true
    exit 2
fi

VENV="${REPO_ROOT}/.venv"
# shellcheck source=/dev/null
source "${VENV}/bin/activate"
PY="${VENV}/bin/python"

# ── Tunable knobs ────────────────────────────────────────────────────────────
: "${WM_TRAIN_SHOTS:=}"        # comma-separated train shots; empty -> discover
: "${WM_N_SHOTS:=3000}"        # discovered-shot cap (camera_first => rbb-dense)
: "${WM_EVAL_SHOTS:=}"         # comma-separated held-out shots; empty -> auto
: "${WM_N_EVAL_SHOTS:=4}"      # held-out shots scored at each eval
: "${WM_STEPS:=20000}"         # optimiser steps (converges in a few hours)
: "${WM_BATCH_SIZE:=4}"        # shots per batch (full-res rbb head is heavier)
: "${WM_N_STEPS:=64}"          # common-grid steps (matches the demo window)
: "${WM_CONTEXT_STEPS:=16}"    # context window (rest is the target window)
: "${WM_LR:=1e-3}"
: "${WM_WEIGHT_DECAY:=0.01}"
: "${WM_LOG_EVERY:=50}"
: "${WM_CKPT_EVERY:=500}"      # checkpoint cadence (drain-safe, frequent)
: "${WM_EVAL_EVERY:=2000}"     # held-out eval cadence
: "${WM_NUM_WORKERS:=12}"      # DataLoader workers (we hold 14 cores)
: "${WM_MODALITIES:=}"         # empty -> ALL streams (plan + 5 L2 + HF + 5 cams)

# ── Backbone-size knobs (match the converged ckpt so warm-start lines up) ────
: "${WM_D_MODEL:=384}"
: "${WM_N_LAYERS:=8}"
: "${WM_N_HEADS:=8}"
: "${WM_DROPOUT:=0.1}"

# Node-local NVMe assembled-sample cache — the camera frame reads dominate the
# loader, so caching the assembled per-shot sample keeps the single H200 fed
# after the first epoch (AGENTS.md §2b in-process default; train.py cache).
: "${WM_CACHE_DIR:=/scratch_local/${SLURM_JOB_ID}/wm_token_cache}"
mkdir -p "${WM_CACHE_DIR}"

: "${WM_OUT_DIR:=/work/projects/imas_gpu/worldmodel/ckpt/${SLURM_JOB_ID}}"
mkdir -p "${WM_OUT_DIR}"

echo "[$(date)] wm-fullres-train starting (job ${SLURM_JOB_ID} on $(hostname))"
echo "  out-dir:     ${WM_OUT_DIR}"
echo "  cache-dir:   ${WM_CACHE_DIR}"
echo "  shots:       ${WM_TRAIN_SHOTS:-<discover up to ${WM_N_SHOTS}, camera_first>}"
echo "  steps:       ${WM_STEPS} batch=${WM_BATCH_SIZE} n_steps=${WM_N_STEPS} context=${WM_CONTEXT_STEPS} lr=${WM_LR}"
echo "  model:       d_model=${WM_D_MODEL} n_layers=${WM_N_LAYERS} n_heads=${WM_N_HEADS} dropout=${WM_DROPOUT}"
echo "  rbb:         FULL resolution (256 tokens/frame, chunked CE head)"
if [[ -f "${WM_OUT_DIR}/latest.pt" ]]; then
    echo "  RESUME:      found ${WM_OUT_DIR}/latest.pt — trainer will continue"
fi
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader || true

SHOT_ARG=""; [[ -n "${WM_TRAIN_SHOTS}" ]] && SHOT_ARG="--shots ${WM_TRAIN_SHOTS}"
EVAL_ARG=""; [[ -n "${WM_EVAL_SHOTS}" ]] && EVAL_ARG="--eval-shots ${WM_EVAL_SHOTS}"
MOD_ARG="";  [[ -n "${WM_MODALITIES}" ]] && MOD_ARG="--modalities ${WM_MODALITIES}"

# shellcheck disable=SC2086
${PY} -m imas_ambix.worldmodel.train corpus \
    ${SHOT_ARG} \
    ${EVAL_ARG} \
    ${MOD_ARG} \
    --d-model "${WM_D_MODEL}" \
    --n-layers "${WM_N_LAYERS}" \
    --n-heads "${WM_N_HEADS}" \
    --dropout "${WM_DROPOUT}" \
    --n-shots "${WM_N_SHOTS}" \
    --n-eval-shots "${WM_N_EVAL_SHOTS}" \
    --steps "${WM_STEPS}" \
    --batch-size "${WM_BATCH_SIZE}" \
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

echo "[$(date)] wm-fullres-train finished (exit=${EXIT_CODE})"
echo "  checkpoints: ${WM_OUT_DIR}/latest.pt (+ ckpt-<step>.pt snapshots)"
exit "${EXIT_CODE}"
