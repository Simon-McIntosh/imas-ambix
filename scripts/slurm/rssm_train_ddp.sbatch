#!/bin/bash
#SBATCH --job-name=ambix-rssm-train
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --account=grpa
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=12
#SBATCH --mem=640G
#SBATCH --time=12:00:00
#SBATCH --output=/work/projects/imas_gpu/logs/ambix-%x-%j.out
# CPU/MEM sized to coexist with a co-running serve.  The grpA + grpB reservations
# OVERLAP on the same cores, so it is NOT a clean 30-15=15 with DeepSeek up — a
# live `srun --test-only` confirmed only ~12 cores schedule when a serve is up.
# 12 is the RESUME-SAFE default; --mem=640G fits within the 650G QoS limit.  12
# cores is plenty for token DataLoaders (tokens are tiny vs raw frames); keep
# WM_NUM_WORKERS so nproc*workers <= 12.  Override --cpus-per-task=30 to take the
# whole reservation when no serve is up.  Probe occupancy with
# `srun --test-only ... --cpus-per-task=N true` if it changes.
# NOTE: --qos intentionally omitted — defaults to normal (no 4-GPU cap), so an
# 8-card job bursts onto the free cards (repo AGENTS.md §2).  Do NOT pass --qos.
#
# ============================================================================
#  Phase-2 RSSM — command-conditioned recurrent latent world model (DDP).
# ============================================================================
#  The Phase-1 token backbone injected the actuator plan as an AdaLN SIDE-input
#  the model could ignore — the powered ΔN-M gate confirmed it (true-plan
#  rollouts ≈ random-plan rollouts).  The RSSM puts the command INSIDE the
#  recurrent transition, so a different command necessarily yields a different
#  rollout — controllability BY CONSTRUCTION.
#
#  torchrun launches one process per GPU (nproc_per_node=WM_NPROC); each rank
#  pins its card, trains a DISJOINT shot shard, the chunked camera CE +
#  diagnostic CE + KL is computed INSIDE the RSSM forward so DDP all-reduces
#  grads across the cards.  Rank 0 ONLY logs / checkpoints.  RESUME-SAFE
#  (latest.pt every WM_CKPT_EVERY + SIGTERM-clean flush).
#
#  Warm-start: the reusable camera token-embed / head / row-col position /
#  diagnostic heads load from a Phase-1 controllable checkpoint (WM_INIT_CHECKPOINT
#  — the joint-gen controllable-1221834/latest.pt has trained diagnostic heads +
#  the camera head/embed); the recurrent latent core (GRU / prior / posterior /
#  command MLP / decoder) stays fresh.
#
#  Env knobs (WM_*): TOKEN_ROOT, MANIFEST, N_SHOTS, STEPS, BATCH_SIZE, LR,
#  BETA, FREE_BITS, DIAGNOSTIC_WEIGHT, H_DIM, S_DIM, INIT_CHECKPOINT, OUT_DIR.
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
    echo "[$(date)] rssm-train teardown"
    kill "${SIDECAR_PID}" 2>/dev/null || true
}
trap cleanup EXIT

VENV="${REPO_ROOT}/.venv"
# shellcheck source=/dev/null
source "${VENV}/bin/activate"

# curated EXCITED token-root — same one the Phase-1 corpus trainer uses (v1/frames
# curated percentile frames + v2/signals_hf for the measured streams).
: "${WM_TOKEN_ROOT:=/work/projects/imas_gpu/mast-tokens}"
: "${WM_MANIFEST:=/work/projects/imas_gpu/agents/excitation-corpus/curated_windows_unified_6cam.json}"
: "${WM_EVAL_SHOTS:=18502,18503,18504,18505}"
: "${WM_N_SHOTS:=0}"            # 0 = all curated train shots
: "${WM_STEPS:=12000}"
# The RSSM's per-frame latent is tiny vs the token backbone's full-frame
# transformer, so a larger batch fits; chunk caps the camera head's logit tensor.
: "${WM_BATCH_SIZE:=4}"
: "${WM_GRAD_ACCUM:=1}"
: "${WM_CHUNK:=4096}"
: "${WM_N_FRAMES:=24}"
: "${WM_N_PLAN:=8}"
: "${WM_N_SIGNAL_STEPS:=4}"
: "${WM_N_ACT_STEPS:=8}"
: "${WM_CONTEXT_FRAMES:=8}"
: "${WM_FRAME_STRIDE:=1}"
# the n_frames window spans this many PHYSICAL seconds (per-shot stride derived
# from each shot's fps), so 24 frames cover the ~0.25s ramp horizon.
: "${WM_TARGET_HORIZON_S:=0.25}"
: "${WM_LR:=1e-4}"
: "${WM_WARMUP_STEPS:=400}"
: "${WM_MIN_LR_RATIO:=0.02}"
: "${WM_GRAD_CLIP:=1.0}"
: "${WM_WEIGHT_DECAY:=0.1}"
: "${WM_LOG_EVERY:=25}"
: "${WM_CKPT_EVERY:=250}"
: "${WM_NUM_WORKERS:=1}"        # per-rank DataLoader workers (nproc*workers <= cores)
# RSSM objective + latent dims
: "${WM_BETA:=1.0}"                 # KL weight
: "${WM_FREE_BITS:=1.0}"            # per-dim KL floor (nats) — anti-collapse
: "${WM_DIAGNOSTIC_WEIGHT:=0.5}"    # per-stream diagnostic next-step CE weight
# Action-contrastive InfoNCE on the latent — keeps the command load-bearing
# through reconstruction training (the decoder cannot wash it out, unlike the
# token backbone).  OFF by default; WM_ACTION_CONTRASTIVE=1 turns it on.
: "${WM_ACTION_CONTRASTIVE:=0}"
: "${WM_ACTION_CONTRASTIVE_WEIGHT:=1.0}"
AC_FLAGS=()
[[ "${WM_ACTION_CONTRASTIVE}" == "1" ]] && AC_FLAGS+=(--action-contrastive --action-contrastive-weight "${WM_ACTION_CONTRASTIVE_WEIGHT}")
: "${WM_H_DIM:=256}"                # GRU hidden (deterministic) state
: "${WM_S_DIM:=32}"                 # stochastic latent state
# 'extended' = the 13-stream measured-signal set (adds the tokenised HF + boundary
# streams the camera world model does not yet fuse); 'default' is the narrow set.
: "${WM_SIGNAL_MODALITIES:=extended}"
# warm-start the reusable camera/head/diagnostic weights from the joint-gen Phase-1
# checkpoint (the latent core stays fresh).
: "${WM_INIT_CHECKPOINT:=/work/projects/imas_gpu/worldmodel/ckpt/controllable-1221834/latest.pt}"
: "${WM_NPROC:=8}"

: "${WM_OUT_DIR:=/work/projects/imas_gpu/worldmodel/ckpt/rssm-${SLURM_JOB_ID}}"
mkdir -p "${WM_OUT_DIR}"

# The loader resolves camera frames at <token_root>/v1/frames/<shot>/<cam>.zarr
# (camdyn DEFAULT_VOCAB_VERSION=v1).  Accept EITHER the canonical v1/frames layout
# OR a bare frames/ dir, so the guard catches a missing corpus without rejecting
# the view.
if [[ ! -d "${WM_TOKEN_ROOT}/v1/frames" && ! -d "${WM_TOKEN_ROOT}/frames" ]]; then
    echo "[rssm-train] FATAL: no frames under ${WM_TOKEN_ROOT}/v1/frames or ${WM_TOKEN_ROOT}/frames" >&2
    kill "${SIDECAR_PID}" 2>/dev/null || true
    exit 2
fi

echo "[$(date)] rssm-train starting (job ${SLURM_JOB_ID} on $(hostname))"
echo "  GPUs:        ${WM_NPROC}"
echo "  token-root:  ${WM_TOKEN_ROOT}"
echo "  manifest:    ${WM_MANIFEST}"
echo "  out-dir:     ${WM_OUT_DIR}"
echo "  init-ckpt:   ${WM_INIT_CHECKPOINT}"
echo "  window:      n_frames=${WM_N_FRAMES} n_plan=${WM_N_PLAN} n_signal=${WM_N_SIGNAL_STEPS} n_act=${WM_N_ACT_STEPS} context=${WM_CONTEXT_FRAMES}"
echo "  steps:       ${WM_STEPS} peak-lr=${WM_LR} warmup=${WM_WARMUP_STEPS} bs=${WM_BATCH_SIZE}/rank chunk=${WM_CHUNK}"
echo "  rssm:        beta=${WM_BETA} free_bits=${WM_FREE_BITS} diag_weight=${WM_DIAGNOSTIC_WEIGHT} h_dim=${WM_H_DIM} s_dim=${WM_S_DIM} signals=${WM_SIGNAL_MODALITIES}"
echo "  held-out:    ${WM_EVAL_SHOTS}"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader || true

torchrun \
    --standalone \
    --nproc_per_node="${WM_NPROC}" \
    -m imas_ambix.worldmodel.rssm_train corpus \
    --manifest "${WM_MANIFEST}" \
    --token-root "${WM_TOKEN_ROOT}" \
    --eval-shots "${WM_EVAL_SHOTS}" \
    --n-shots "${WM_N_SHOTS}" \
    --init-checkpoint "${WM_INIT_CHECKPOINT}" \
    --signal-modalities "${WM_SIGNAL_MODALITIES}" \
    --n-frames "${WM_N_FRAMES}" \
    --n-plan "${WM_N_PLAN}" \
    --context-frames "${WM_CONTEXT_FRAMES}" \
    --frame-stride "${WM_FRAME_STRIDE}" \
    --target-horizon-s "${WM_TARGET_HORIZON_S}" \
    --n-signal-steps "${WM_N_SIGNAL_STEPS}" \
    --n-act-steps "${WM_N_ACT_STEPS}" \
    --steps "${WM_STEPS}" \
    --batch-size "${WM_BATCH_SIZE}" \
    --grad-accum "${WM_GRAD_ACCUM}" \
    --chunk "${WM_CHUNK}" \
    --lr "${WM_LR}" \
    --warmup-steps "${WM_WARMUP_STEPS}" \
    --min-lr-ratio "${WM_MIN_LR_RATIO}" \
    --weight-decay "${WM_WEIGHT_DECAY}" \
    --grad-clip "${WM_GRAD_CLIP}" \
    --log-every "${WM_LOG_EVERY}" \
    --ckpt-every "${WM_CKPT_EVERY}" \
    --num-workers "${WM_NUM_WORKERS}" \
    --beta "${WM_BETA}" \
    --free-bits "${WM_FREE_BITS}" \
    --diagnostic-weight "${WM_DIAGNOSTIC_WEIGHT}" \
    "${AC_FLAGS[@]}" \
    --h-dim "${WM_H_DIM}" \
    --s-dim "${WM_S_DIM}" \
    --out-dir "${WM_OUT_DIR}"
EXIT_CODE=$?

echo "[$(date)] rssm-train finished (exit=${EXIT_CODE})"
echo "  checkpoints: ${WM_OUT_DIR}/latest.pt (+ ckpt-<step>.pt snapshots)"

# ── Baked-in end-of-run held-out eval (produces the verdict even if the driving
#    session limits out). Runs on ONE of the GPUs the job already holds (the
#    training DDP has exited, so all cards are free), in the SAME process env.
#    rssm_eval.py is built by a concurrent worker; guarded so a missing/failing
#    eval never fails the train.  WM_RUN_EVAL=0 disables.
: "${WM_RUN_EVAL:=1}"
: "${WM_GATE_COHORT:=/work/projects/imas_gpu/worldmodel/gate_cohort.json}"
if [[ "${WM_RUN_EVAL}" == "1" && "${EXIT_CODE}" == "0" && -f "${WM_OUT_DIR}/latest.pt" ]]; then
    echo "[$(date)] baked-in held-out RSSM eval (cohort=${WM_GATE_COHORT}) on ${WM_OUT_DIR}/latest.pt"
    CUDA_VISIBLE_DEVICES=0 python -m imas_ambix.worldmodel.rssm_eval \
        --checkpoint "${WM_OUT_DIR}/latest.pt" \
        --held-out-cohort "${WM_GATE_COHORT}" \
        --n-random 10 \
        --out-dir "${WM_OUT_DIR}/eval" \
        --token-root "${WM_TOKEN_ROOT}" \
        --target-horizon-s "${WM_TARGET_HORIZON_S}" \
        --n-frames "${WM_N_FRAMES}" \
        --n-plan "${WM_N_PLAN}" \
        --context-frames "${WM_CONTEXT_FRAMES}" \
        --n-signal-steps "${WM_N_SIGNAL_STEPS}" \
        --n-act-steps "${WM_N_ACT_STEPS}" \
        --chunk "${WM_CHUNK}" \
        && echo "[$(date)] baked-in RSSM eval done: ${WM_OUT_DIR}/eval" \
        || echo "eval failed (ckpt valid)"
else
    echo "[$(date)] baked-in eval skipped (WM_RUN_EVAL=${WM_RUN_EVAL} exit=${EXIT_CODE} ckpt=$( [[ -f ${WM_OUT_DIR}/latest.pt ]] && echo present || echo missing ))"
fi

exit "${EXIT_CODE}"
