#!/bin/bash
#SBATCH --job-name=ambix-worldmodel-train
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --account=grpa
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=320G
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
#  GPUs: --gres=gpu:2.  DeepSeek-V4-Flash (2x variant) holds the other 2 of
#  the Group-A 4-H200 reservation, so this job takes the 2 free cards.  The
#  prototype model is small (a single H200 is ample); the 2-card request just
#  matches what is FREE.  If DSv4 is shut down (pre-authorised — see the
#  user memory `feedback_authorised_to_stop_dsv4`: `ambix agent shutdown
#  deepseek-v4-flash -y`) and the demo needs more throughput, bump --gres to
#  gpu:4, --cpus-per-task to 30, --mem to 640G — the trainer uses one device
#  but more cores accelerate the DataLoader.
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
: "${WM_STEPS:=20000}"         # optimiser steps (a step is one batch)
: "${WM_BATCH_SIZE:=8}"        # shots per batch
: "${WM_N_STEPS:=128}"         # common-grid steps
: "${WM_CONTEXT_STEPS:=32}"    # context window (rest is the target window)
: "${WM_LR:=1e-3}"
: "${WM_WEIGHT_DECAY:=0.01}"
: "${WM_LOG_EVERY:=50}"
: "${WM_CKPT_EVERY:=500}"      # checkpoint cadence (drain-safe)
: "${WM_EVAL_EVERY:=2000}"     # held-out eval cadence
: "${WM_NUM_WORKERS:=8}"       # DataLoader workers
: "${WM_MODALITIES:=}"         # comma-separated modality subset; empty -> all (incl heavy camera)
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

# shellcheck disable=SC2086
${PY} -m imas_ambix.worldmodel.train corpus \
    ${SHOT_ARG} \
    ${EVAL_ARG} \
    ${MOD_ARG} \
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
