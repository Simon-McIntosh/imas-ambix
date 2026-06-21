#!/bin/bash
#SBATCH --job-name=ambix-controllable-retrain
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --account=grpa
#SBATCH --gres=gpu:6
#SBATCH --cpus-per-task=14
#SBATCH --mem=440G
#SBATCH --time=08:00:00
#SBATCH --output=/work/projects/imas_gpu/logs/ambix-%x-%j.out
# CPU/MEM sized to COEXIST with the DeepSeek serve INSIDE the 30-core / 650G
# Group-A reservation: DeepSeek uses ~15 cores + 200G, so 14 cores (15+14=29<=30)
# + 440G (200+440=640<=650) leaves room and schedules immediately.  Requesting the
# full reservation (--cpus-per-task=30 --mem=640G) pends on Resources for days
# while DeepSeek is up (45 cores / 840G > the reservation).  14 cores is plenty for
# 6-rank token DataLoaders (tokens are tiny vs raw frames); keep WM_NUM_WORKERS so
# 6*workers <= 14.  Override on the CLI only to take the whole reservation when
# DeepSeek is down.
# NOTE: --qos intentionally omitted — defaults to normal (no 4-GPU cap), so a
# 6-card job bursts onto the free cards and COEXISTS with the 2-card DeepSeek
# serve (6 + 2 = 8/8). Do NOT pass --qos. (repo AGENTS.md §2)
#
# ============================================================================
#  M4 re-train — controllable camera model on the EXCITED corpus (6-GPU DDP).
# ============================================================================
#  The overfit gate validated the redesign for 1-step conditioning (AdaLN-Zero +
#  camera-history bottleneck + inverse-dynamics made the plan load-bearing, ~30×
#  over observations) but could not test FREE-RUNNING controllability (the
#  memorization confound on 4 flat-top shots). This is the real test: warm-start
#  the M2 forecaster and fine-tune on B's curated EXCITED corpus (percentile-
#  exposure token-root) with the full controllability machinery + scheduled-
#  sampling so the plan must steer the GENERATED rollout.
#
#  torchrun launches one process per GPU (nproc_per_node=6); each rank pins its
#  card, trains a DISJOINT shot shard, the chunked next-frame CE + inverse-
#  dynamics is computed INSIDE forward so DDP all-reduces grads across 6 cards.
#  Rank 0 ONLY logs / checkpoints. RESUME-SAFE (latest.pt + SIGTERM-clean).
#
#  ONE consistent normalization: the train corpus AND the held-out eval shots are
#  BOTH percentile-[1,99.5]-exposure tokens. Point WM_TOKEN_ROOT at B's curated
#  root; the held-out shots must be encoded into the SAME root (see the held-out
#  encode note in the followup) BEFORE the eval.
#
#  Env knobs (WM_*): TOKEN_ROOT, MANIFEST, N_SHOTS, STEPS, BATCH_SIZE, LR,
#  HB_*, INV_DYN_WEIGHT, SS_MAX, SS_RAMP, INIT_CHECKPOINT, OUT_DIR.
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
    echo "[$(date)] controllable-retrain teardown"
    kill "${SIDECAR_PID}" 2>/dev/null || true
}
trap cleanup EXIT

VENV="${REPO_ROOT}/.venv"
# shellcheck source=/dev/null
source "${VENV}/bin/activate"

# curated EXCITED token-root — the curated-token-view (v1/frames -> curated
# percentile frames + v2/signals_hf -> the old signals_hf; the loader needs both
# version dirs, and B's raw curated dir lacks the v1/ level + signals).
: "${WM_TOKEN_ROOT:=/work/projects/imas_gpu/worldmodel/curated-token-view}"
: "${WM_MANIFEST:=/work/projects/imas_gpu/agents/excitation-corpus/curated_windows_full.json}"
: "${WM_EVAL_SHOTS:=18502,18503,18504,18505}"
: "${WM_N_SHOTS:=0}"            # 0 = all curated train shots
: "${WM_STEPS:=12000}"
# batch=1 + chunk=2048 keep the 1.07B model + AdamW + the 2**18-vocab head under
# the 140G/card budget (chunk=8192 head alone is ~8.6G and batch=4 OOMs).  Raise
# WM_GRAD_ACCUM for a larger effective batch without more peak memory.
: "${WM_BATCH_SIZE:=1}"
: "${WM_GRAD_ACCUM:=1}"
: "${WM_CHUNK:=2048}"
: "${WM_N_FRAMES:=24}"
: "${WM_N_PLAN:=8}"
: "${WM_N_SIGNAL_STEPS:=4}"
: "${WM_N_ACT_STEPS:=8}"
: "${WM_CONTEXT_FRAMES:=8}"
: "${WM_FRAME_STRIDE:=1}"
: "${WM_LR:=1e-4}"
: "${WM_WARMUP_STEPS:=400}"
: "${WM_MIN_LR_RATIO:=0.02}"
: "${WM_GRAD_CLIP:=1.0}"
: "${WM_WEIGHT_DECAY:=0.1}"
: "${WM_LOG_EVERY:=25}"
: "${WM_CKPT_EVERY:=250}"
: "${WM_NUM_WORKERS:=2}"        # per-rank DataLoader workers (6 ranks * 2 = 12 <= 14 cores)
# M4 controllability machinery
: "${WM_OBS_DROPOUT:=0.8}"
: "${WM_CONTROL_DROPOUT:=0.15}"
: "${WM_HB_NOISE_STD:=1.0}"
: "${WM_HB_MASK_PROB:=0.5}"
: "${WM_HB_MAX_STRENGTH:=1.0}"
: "${WM_HB_CLEAN_FRACTION:=0.2}"
: "${WM_INV_DYN_WEIGHT:=1.0}"
: "${WM_SS_MAX:=0.25}"
: "${WM_SS_RAMP:=0.5}"
# model — adopted from the forecaster checkpoint when --init-checkpoint is given,
# so these are usually unset (the trainer reads the ckpt dims). Override only to
# train from scratch at a chosen size.
: "${WM_ADALN_HIDDEN:=256}"
: "${WM_INIT_CHECKPOINT:=/work/projects/imas_gpu/worldmodel/ckpt/spacetime-corruption-1220391/best.pt}"
: "${WM_NPROC:=6}"

: "${WM_OUT_DIR:=/work/projects/imas_gpu/worldmodel/ckpt/controllable-${SLURM_JOB_ID}}"
mkdir -p "${WM_OUT_DIR}"

# The loader resolves camera frames at <token_root>/v1/frames/<shot>/<cam>.zarr
# (camdyn DEFAULT_VOCAB_VERSION=v1).  Accept EITHER the canonical v1/frames layout
# (the curated-token-view) OR a bare frames/ dir (B's raw curated root), so the
# guard catches a genuinely-missing corpus without rejecting the view.
if [[ ! -d "${WM_TOKEN_ROOT}/v1/frames" && ! -d "${WM_TOKEN_ROOT}/frames" ]]; then
    echo "[controllable-retrain] FATAL: no frames under ${WM_TOKEN_ROOT}/v1/frames or ${WM_TOKEN_ROOT}/frames — has the curated encode landed + the v1/frames view been set up?" >&2
    kill "${SIDECAR_PID}" 2>/dev/null || true
    exit 2
fi

echo "[$(date)] controllable-retrain starting (job ${SLURM_JOB_ID} on $(hostname))"
echo "  GPUs:        ${WM_NPROC}"
echo "  token-root:  ${WM_TOKEN_ROOT}"
echo "  manifest:    ${WM_MANIFEST}"
echo "  out-dir:     ${WM_OUT_DIR}"
echo "  init-ckpt:   ${WM_INIT_CHECKPOINT}"
echo "  window:      n_frames=${WM_N_FRAMES} n_plan=${WM_N_PLAN} n_signal=${WM_N_SIGNAL_STEPS} n_act=${WM_N_ACT_STEPS} context=${WM_CONTEXT_FRAMES}"
echo "  steps:       ${WM_STEPS} peak-lr=${WM_LR} warmup=${WM_WARMUP_STEPS} bs=${WM_BATCH_SIZE}/rank chunk=${WM_CHUNK}"
echo "  M4 machinery: obs_dropout=${WM_OBS_DROPOUT} hb(std=${WM_HB_NOISE_STD},mask=${WM_HB_MASK_PROB},max=${WM_HB_MAX_STRENGTH}) inv_dyn=${WM_INV_DYN_WEIGHT} ss(max=${WM_SS_MAX},ramp=${WM_SS_RAMP})"
echo "  held-out:    ${WM_EVAL_SHOTS}"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader || true

torchrun \
    --standalone \
    --nproc_per_node="${WM_NPROC}" \
    -m imas_ambix.worldmodel.controllable_train corpus \
    --manifest "${WM_MANIFEST}" \
    --token-root "${WM_TOKEN_ROOT}" \
    --eval-shots "${WM_EVAL_SHOTS}" \
    --n-shots "${WM_N_SHOTS}" \
    --init-checkpoint "${WM_INIT_CHECKPOINT}" \
    --n-frames "${WM_N_FRAMES}" \
    --n-plan "${WM_N_PLAN}" \
    --context-frames "${WM_CONTEXT_FRAMES}" \
    --frame-stride "${WM_FRAME_STRIDE}" \
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
    --adaln-hidden "${WM_ADALN_HIDDEN}" \
    --observation-dropout "${WM_OBS_DROPOUT}" \
    --control-dropout "${WM_CONTROL_DROPOUT}" \
    --hb-noise-std "${WM_HB_NOISE_STD}" \
    --hb-mask-prob "${WM_HB_MASK_PROB}" \
    --hb-max-strength "${WM_HB_MAX_STRENGTH}" \
    --hb-clean-fraction "${WM_HB_CLEAN_FRACTION}" \
    --inverse-dynamics-weight "${WM_INV_DYN_WEIGHT}" \
    --scheduled-sampling-max "${WM_SS_MAX}" \
    --scheduled-sampling-ramp "${WM_SS_RAMP}" \
    --out-dir "${WM_OUT_DIR}"
EXIT_CODE=$?

echo "[$(date)] controllable-retrain finished (exit=${EXIT_CODE})"
echo "  checkpoints: ${WM_OUT_DIR}/latest.pt (+ ckpt-<step>.pt snapshots)"
exit "${EXIT_CODE}"
