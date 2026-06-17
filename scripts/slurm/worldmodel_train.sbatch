#!/bin/bash
#SBATCH --job-name=ambix-worldmodel-train
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --account=grpa
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=30
#SBATCH --mem=640G
#SBATCH --time=10:00:00
#SBATCH --output=/work/projects/imas_gpu/logs/ambix-%x-%j.out
# NOTE: --qos intentionally omitted — the gpu_0003_grpa QoS is auto-applied;
#       passing it explicitly errors (AGENTS.md §2).
#
# ============================================================================
#  Real corpus training for the plan-conditioned multi-modal world model.
# ============================================================================
#  Runs imas_ambix.worldmodel.train::train_corpus — a DataLoader over MANY
#  distinct shots with the target-window forecasting objective, AdamW,
#  checkpoint+resume, and periodic held-out skill-vs-persistence eval.
#
#  GPUs: --gres=gpu:4 (the FULL Group-A 4-H200 reservation).  This is the
#  SCALED demo run: a sizeable transformer (WM_D_MODEL/WM_N_LAYERS/WM_N_HEADS
#  below) over the FULL tokenised substrate — the conditioning plan, the five
#  measured L2 groups (summary/pf_active/interferometer/gas_injection/
#  soft_x_rays), the three L1 HF streams (xma/xim/xsx), and all five cameras
#  (the richest streams, ~16 tokens/frame each at camera_grid_stride=4) — makes
#  the GPU work, unlike the first tiny data-bound run.  The whole model (incl
#  five separate 2^18 camera tables) fits on ONE H200; the trainer is
#  single-device.  DeepSeek-V4-Flash must be shut down first to free all 4 cards —
#  pre-authorised, see user memory `feedback_authorised_to_stop_dsv4`:
#  `ambix agent shutdown deepseek-v4-flash -y`.  The trainer itself runs on a
#  single device today; the 4-card request reserves the node so the DataLoader
#  gets all 30 cores + 640G and no peer job contends for the GPU.
#
#  RESUME-SAFE (the H200 node has a drain history, AGENTS.md §2a): checkpoints
#  land on GPFS under WM_OUT_DIR (default /work/.../worldmodel/ckpt/<jobid>);
#  on restart the trainer auto-resumes from latest.pt.  To resume a DRAINED
#  job into a fresh allocation, point WM_OUT_DIR at the previous job's ckpt
#  dir:  sbatch --export=ALL,WM_OUT_DIR=/work/projects/imas_gpu/worldmodel/ckpt/<prev-jobid> ...
# ============================================================================

set -euo pipefail

# TMPDIR on betelgeuse: /run/user/$UID does not exist on the compute node;
# /scratch_local is the node-local NVMe (AGENTS.md §2 SLURM workarounds).
export TMPDIR="/scratch_local/${SLURM_JOB_ID}"
mkdir -p "${TMPDIR}"

# torch owns GPU parallelism; keep BLAS single-threaded so the DataLoader
# worker pool (the CPU side) is not oversubscribed.
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

# Offline: betelgeuse has NO outbound network (AGENTS.md §2).  Any HF / torch
# hub fetch must already be cached; fail loud instead of stalling to the limit.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# SLURM copies the batch script to /var/spool/...; use SLURM_SUBMIT_DIR (where
# sbatch was invoked) to find the repo + venv and write results back.
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_ROOT}"
mkdir -p "/work/projects/imas_gpu/logs"

# ── Drain-forensics sidecar (default-on for long GPU jobs; AGENTS.md §2a) ────
# SIGTERM-immune; samples every PID in the job cgroup (state + wchan),
# nvidia-smi, and kernel NVRM/Xid lines once per second to GPFS.  If a drain
# recurs it captures the D-state PID + kernel symbol.
bash "${REPO_ROOT}/scripts/slurm/drain_sidecar.sh" &
SIDECAR_PID=$!

# ── Preflight: refuse to start if the token corpus is unreachable ───────────
TOKEN_ROOT="/work/projects/imas_gpu/mast-tokens"
if [[ ! -d "${TOKEN_ROOT}/v2/signals_hf" ]]; then
    echo "[worldmodel-train] FATAL: ${TOKEN_ROOT}/v2/signals_hf not reachable" >&2
    echo "[worldmodel-train] is the /work mount up on this node? aborting" >&2
    kill "${SIDECAR_PID}" 2>/dev/null || true
    exit 2
fi

# ── Activate the project venv ───────────────────────────────────────────────
VENV="${REPO_ROOT}/.venv"
if [[ -f "${VENV}/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source "${VENV}/bin/activate"
    PY="${VENV}/bin/python"
else
    echo "[worldmodel-train] .venv not found — falling back to uv run" >&2
    PY="uv run python"
fi

# ── Tunable knobs (override via sbatch --export=ALL,VAR=val) ─────────────────
: "${WM_TRAIN_SHOTS:=}"        # comma-separated train shots; empty -> discover
: "${WM_N_SHOTS:=512}"         # discovered-shot cap when WM_TRAIN_SHOTS empty
: "${WM_EVAL_SHOTS:=}"         # comma-separated held-out shots; empty -> tail
: "${WM_N_EVAL_SHOTS:=4}"      # held-out shots scored at each eval
: "${WM_STEPS:=40000}"         # optimiser steps (a step is one batch) — scaled
: "${WM_BATCH_SIZE:=8}"        # shots per batch
: "${WM_N_STEPS:=128}"         # common-grid steps
: "${WM_CONTEXT_STEPS:=32}"    # context window (rest is the target window)
: "${WM_LR:=1e-3}"
: "${WM_WEIGHT_DECAY:=0.01}"
: "${WM_LOG_EVERY:=50}"
: "${WM_CKPT_EVERY:=500}"      # checkpoint cadence (drain-safe)
: "${WM_EVAL_EVERY:=2000}"     # held-out eval cadence
: "${WM_NUM_WORKERS:=24}"      # DataLoader workers (full reservation has 30 cores)
: "${WM_MODALITIES:=}"         # comma-separated modality subset; empty -> ALL streams (plan + 5 L2 + xma/xim/xsx + 5 cameras)

# ── Backbone-size knobs (the SCALED demo) ───────────────────────────────────
# Demo-scaled defaults: a sizeable transformer so the H200 is actually used,
# not data-bound like the first tiny auto-sized run.  d_ff is derived as
# 4*d_model by the trainer when --d-model is set (standard transformer ratio),
# so it is not a separate knob here.  Leave a knob EMPTY to fall back to the
# WorldModelConfig field default (the tiny prototype) — set them to scale up.
: "${WM_D_MODEL:=384}"         # transformer width
: "${WM_N_LAYERS:=8}"          # transformer depth
: "${WM_N_HEADS:=8}"           # attention heads (must divide d_model)
: "${WM_DROPOUT:=0.1}"         # dropout probability
# Checkpoints persist on GPFS keyed by job id; override WM_OUT_DIR to resume a
# drained job's ckpt dir into a fresh allocation.
: "${WM_OUT_DIR:=/work/projects/imas_gpu/worldmodel/ckpt/${SLURM_JOB_ID}}"
mkdir -p "${WM_OUT_DIR}"

echo "[$(date)] worldmodel-train (corpus) starting"
echo "  job-id:        ${SLURM_JOB_ID}"
echo "  node:          $(hostname)"
echo "  out-dir:       ${WM_OUT_DIR}"
echo "  train_shots:   ${WM_TRAIN_SHOTS:-<discover up to ${WM_N_SHOTS}>}"
echo "  steps:         ${WM_STEPS}  batch=${WM_BATCH_SIZE}  n_steps=${WM_N_STEPS}  context=${WM_CONTEXT_STEPS}  lr=${WM_LR}"
echo "  ckpt_every:    ${WM_CKPT_EVERY}  eval_every=${WM_EVAL_EVERY}"
if [[ -f "${WM_OUT_DIR}/latest.pt" ]]; then
    echo "  RESUME:        found ${WM_OUT_DIR}/latest.pt — trainer will continue"
fi
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader || true

# ── Clean teardown on SLURM SIGTERM ─────────────────────────────────────────
# The training loop installs its OWN SIGTERM handler (sets a STOP flag,
# checkpoints latest.pt, releases the model, empties the CUDA cache within
# UnkillableStepTimeout — see imas_ambix/worldmodel/train.py).  Here we only
# ensure the sidecar is reaped.
cleanup() {
    echo "[$(date)] worldmodel-train teardown"
    kill "${SIDECAR_PID}" 2>/dev/null || true
}
trap cleanup EXIT

# ── Run the corpus trainer (in-process; model loaded ONCE — AGENTS.md §2b) ───
SHOT_ARG=""
if [[ -n "${WM_TRAIN_SHOTS}" ]]; then
    SHOT_ARG="--shots ${WM_TRAIN_SHOTS}"
fi
EVAL_ARG=""
if [[ -n "${WM_EVAL_SHOTS}" ]]; then
    EVAL_ARG="--eval-shots ${WM_EVAL_SHOTS}"
fi
MOD_ARG=""
if [[ -n "${WM_MODALITIES}" ]]; then
    MOD_ARG="--modalities ${WM_MODALITIES}"
fi
# Backbone-size args: only pass a knob when it is set, so an empty knob falls
# through to the WorldModelConfig field default (back-compat with the tiny run).
SIZE_ARG=""
if [[ -n "${WM_D_MODEL}" ]];  then SIZE_ARG="${SIZE_ARG} --d-model ${WM_D_MODEL}"; fi
if [[ -n "${WM_N_LAYERS}" ]]; then SIZE_ARG="${SIZE_ARG} --n-layers ${WM_N_LAYERS}"; fi
if [[ -n "${WM_N_HEADS}" ]];  then SIZE_ARG="${SIZE_ARG} --n-heads ${WM_N_HEADS}"; fi
if [[ -n "${WM_DROPOUT}" ]];  then SIZE_ARG="${SIZE_ARG} --dropout ${WM_DROPOUT}"; fi

echo "  model-size:    d_model=${WM_D_MODEL:-<default>} n_layers=${WM_N_LAYERS:-<default>} n_heads=${WM_N_HEADS:-<default>} dropout=${WM_DROPOUT:-<default>}"
echo "  modalities:    ${WM_MODALITIES:-<ALL: plan + 5 L2 + xma/xim/xsx + 5 cameras>}"

# shellcheck disable=SC2086
${PY} -m imas_ambix.worldmodel.train corpus \
    ${SHOT_ARG} \
    ${EVAL_ARG} \
    ${MOD_ARG} \
    ${SIZE_ARG} \
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
    --out-dir "${WM_OUT_DIR}"
EXIT_CODE=$?

echo "[$(date)] worldmodel-train finished (exit=${EXIT_CODE})"
echo "  checkpoints: ${WM_OUT_DIR}/latest.pt (+ ckpt-<step>.pt snapshots)"
exit "${EXIT_CODE}"
