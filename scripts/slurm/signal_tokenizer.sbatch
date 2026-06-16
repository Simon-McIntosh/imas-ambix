#!/bin/bash
#SBATCH --job-name=signal-tokenizer
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --account=grpa
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=160G
#SBATCH --time=06:00:00
#SBATCH --output=/work/projects/imas_gpu/logs/signal-tokenizer-%x-%j.out
# NOTE: --qos is intentionally omitted — auto-applied for this account/
#       reservation. Passing --qos explicitly errors with "QOS not permitted".
#
# Phase-aware patch-transformer 1-D signal tokenizer: codebook decision →
# train winner → encode the high-frequency families (xma fast magnetics,
# xim Dα/CII) into the multi-rate phase-preserving v2 token store.
#
# gpu:1 only — leaves the other Group-A cards free for the concurrent
# camera-encode job and does NOT touch the DeepSeek serve.
#
# In-process (repo §2b): the model is loaded ONCE and many shots stream
# through the same process — NO subprocess-per-shot, NO file-IPC daemon.
#
# Usage:
#   sbatch scripts/slurm/signal_tokenizer.sbatch                  # xma, default ids
#   GROUP=xim sbatch scripts/slurm/signal_tokenizer.sbatch
#   TRAIN_IDS=/path/train.txt HOLDOUT_IDS=/path/hold.txt ENCODE_IDS=/path/all.txt \
#       sbatch scripts/slurm/signal_tokenizer.sbatch
set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
if [[ ! -d "${REPO_ROOT}/imas_ambix/tokenizer" ]]; then
    echo "[signal-tok] ERROR: REPO_ROOT=${REPO_ROOT} is not the imas-ambix repo root." >&2
    echo "[signal-tok] Submit from the repo root: cd <repo> && sbatch scripts/slurm/signal_tokenizer.sbatch" >&2
    exit 2
fi

# ── SLURM-node workarounds (repo §2) ─────────────────────────────────────────
export TMPDIR=/scratch_local/${SLURM_JOB_ID}
mkdir -p "$TMPDIR"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

# ── Drain-forensics sidecar (repo §2a) ───────────────────────────────────────
bash "${REPO_ROOT}/scripts/slurm/drain_sidecar.sh" &

PY="${REPO_ROOT}/.venv/bin/python"
GROUP="${GROUP:-xma}"
EPOCHS="${EPOCHS:-40}"
PATCH_SIZE="${PATCH_SIZE:-64}"
OUT_DIR="${OUT_DIR:-/work/projects/imas_gpu/mast-tokens/v2/_tokenizer}"
mkdir -p "$OUT_DIR"

# Shot id sources.  Defaults: a small interior-info calibration set — override
# with TRAIN_IDS / HOLDOUT_IDS / ENCODE_IDS env vars (path or comma list).
TRAIN_IDS="${TRAIN_IDS:-30460,30461,30462,28000,28001,28002,27000,27001}"
HOLDOUT_IDS="${HOLDOUT_IDS:-30463,28003,27002}"
ENCODE_IDS="${ENCODE_IDS:-$TRAIN_IDS,$HOLDOUT_IDS}"

DECIDE_OUT="${OUT_DIR}/decide_${GROUP}.json"
CKPT="${OUT_DIR}/tokenizer_${GROUP}.pt"
LOG="${TMPDIR}/run.log"

echo "[signal-tok] group=$GROUP epochs=$EPOCHS patch=$PATCH_SIZE" | tee -a "$LOG"

# ── 1. Codebook decision: FSQ vs VQ vs continuous on a phase-fidelity holdout ─
"$PY" -m imas_ambix.tokenizer.signal_hf_encode decide \
    --group "$GROUP" --device cuda --patch-size "$PATCH_SIZE" \
    --train "$TRAIN_IDS" --holdout "$HOLDOUT_IDS" \
    --epochs "$EPOCHS" --out "$DECIDE_OUT" >>"$LOG" 2>&1
echo "DECIDE_EXIT=$?" | tee -a "$LOG"

# Winner is read back from the decision report.
WINNER="$("$PY" -c "import json,sys; print(json.load(open('$DECIDE_OUT'))['decision']['winner'])")"
echo "[signal-tok] winner=$WINNER" | tee -a "$LOG"

# ── 2. Train the winning bottleneck on the train shots ───────────────────────
"$PY" -m imas_ambix.tokenizer.signal_hf_encode train \
    --group "$GROUP" --device cuda --patch-size "$PATCH_SIZE" \
    --bottleneck "$WINNER" --train "$TRAIN_IDS" \
    --epochs "$EPOCHS" --out "$CKPT" >>"$LOG" 2>&1
echo "TRAIN_EXIT=$?" | tee -a "$LOG"

# ── 3. Encode the corpus into the v2 signals_hf store ────────────────────────
"$PY" -m imas_ambix.tokenizer.signal_hf_encode encode \
    --group "$GROUP" --device cuda \
    --shots "$ENCODE_IDS" --ckpt "$CKPT" >>"$LOG" 2>&1
echo "ENCODE_EXIT=$?" | tee -a "$LOG"

cp "$LOG" "/work/projects/imas_gpu/logs/signal-tokenizer-${GROUP}-${SLURM_JOB_ID}.run.log" || true
echo "DONE=1"
