#!/bin/bash
#SBATCH --job-name=signal-tokenizer
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --account=grpa
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=/work/projects/imas_gpu/logs/signal-tokenizer-%x-%j.out
# NOTE: --qos is intentionally omitted — auto-applied for this account/
#       reservation. Passing --qos explicitly errors with "QOS not permitted".
#
# Phase-aware patch-transformer 1-D signal tokenizer: codebook decision →
# train winner → encode the high-frequency families (xma fast magnetics,
# xim Dα/CII, xsx soft X-ray) into the multi-rate phase-preserving v2 token
# store — scaled from the dev set to the FULL level-1 corpus.
#
# gpu:1 only — leaves the other Group-A cards free for the concurrent
# camera-encode chain and the DeepSeek serve.  One card is free; this job
# takes exactly that one.  The encode is IO-bound — measured GPU sm-util ~0.8%
# (the patch-transformer inference is trivial next to the GPFS read of the
# MHz-rate raw signals) — so the card is intentionally under-driven; the speed
# comes from overlapping shot reads via the DataLoader worker pool below, not
# from GPU compute.  cpus-per-task is raised to feed those reader workers.
#
# In-process (repo §2b): the model is loaded ONCE per group and many shots
# stream through the same process — NO subprocess-per-shot, NO file-IPC daemon.
# A torch DataLoader worker pool (NUM_WORKERS) reads shot N+1 off GPFS while
# shot N is encoded + written, the canonical IO-overlap pattern.  The encode is
# checkpoint/resume safe: --shots all enumerates every shot on disk and
# skip-existing means a re-run (after time-limit / cancellation) picks up where
# it left off.
#
# Usage:
#   sbatch scripts/slurm/signal_tokenizer.sbatch                  # full corpus, all groups
#   SIG_GROUPS=xsx sbatch scripts/slurm/signal_tokenizer.sbatch    # one group
#   ENCODE_IDS=/path/all.txt sbatch ...                            # explicit shotlist
#   MODE=decide_only SIG_GROUPS=xsx sbatch ...                     # just the codebook decision
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
# NOTE: do NOT name this 'GROUPS' — bash sets GROUPS to the caller's numeric
# group IDs, which would silently override the default and feed an invalid
# group to the encoder.  Use a namespaced var.
SIG_GROUPS="${SIG_GROUPS:-xma xim xsx}"
EPOCHS="${EPOCHS:-40}"
PATCH_SIZE="${PATCH_SIZE:-64}"
MODE="${MODE:-full}"                 # full | decide_only | encode_only
# DataLoader reader workers for the IO-bound encode.  Leave a couple of the
# allocated cores for the main process (inference + store writes); the rest
# overlap GPFS reads.
NUM_WORKERS="${NUM_WORKERS:-8}"
# Absolute calibration is THE default operative path: the bottleneck is decided
# + trained + the corpus encoded against the persisted corpus calibration so a
# physical value maps to the same token everywhere.  Per-shot (per-window)
# z-scoring is an explicit, clearly-labelled diagnostic only (ABSOLUTE=0).
# Switching normalisation invalidates the per-window codebook, so absolute mode
# forces a re-decide unless the caller has already produced an absolute report.
ABSOLUTE="${ABSOLUTE:-1}"
if [[ "$ABSOLUTE" == "1" ]]; then
    ABS_FLAG="--absolute"
    : "${REDECIDE:=1}"   # codebook decision must be remade under absolute norm
    echo "[signal-tok] calibration_mode=ABSOLUTE (corpus-calibrated tokens)"
else
    ABS_FLAG=""
    echo "[signal-tok] calibration_mode=PER_SHOT (diagnostic only — NOT default)"
fi
OUT_DIR="${OUT_DIR:-/work/projects/imas_gpu/mast-tokens/v2/_tokenizer}"
MANIFEST_DIR="${MANIFEST_DIR:-/work/projects/imas_gpu/mast-tokens/v2/_manifests}"
mkdir -p "$OUT_DIR" "$MANIFEST_DIR"

# Per-group small train/holdout sets for the codebook decision and the
# tokenizer fit.  These are a fixed, deliberately small calibration set — the
# encode runs over the FULL corpus regardless.  Override per group with
# TRAIN_IDS / HOLDOUT_IDS if needed.
TRAIN_IDS="${TRAIN_IDS:-30460,30461,30462,28000,28001,28002,27000,27001}"
HOLDOUT_IDS="${HOLDOUT_IDS:-30463,28003,27002}"
# ENCODE_IDS defaults to the literal 'all' → every shot present on disk.
ENCODE_IDS="${ENCODE_IDS:-all}"

LOG="${TMPDIR}/run.log"
echo "[signal-tok] groups='$SIG_GROUPS' mode=$MODE epochs=$EPOCHS patch=$PATCH_SIZE encode=$ENCODE_IDS" | tee -a "$LOG"

for GROUP in $SIG_GROUPS; do
    DECIDE_OUT="${OUT_DIR}/decide_${GROUP}.json"
    CKPT="${OUT_DIR}/tokenizer_${GROUP}.pt"
    MANIFEST="${MANIFEST_DIR}/encode_${GROUP}.json"
    echo "[signal-tok] === group=$GROUP ===" | tee -a "$LOG"

    if [[ "$MODE" != "encode_only" ]]; then
        # ── 1. Codebook decision: FSQ vs VQ vs continuous on a phase holdout ──
        # Skip if a decision report already exists (the dev decisions are
        # LOCKED: xma→continuous, xim→fsq) unless REDECIDE=1.
        if [[ -f "$DECIDE_OUT" && "${REDECIDE:-0}" != "1" ]]; then
            echo "[signal-tok] reusing existing decision $DECIDE_OUT" | tee -a "$LOG"
        else
            "$PY" -m imas_ambix.tokenizer.signal_hf_encode decide \
                --group "$GROUP" --device cuda --patch-size "$PATCH_SIZE" \
                --train "$TRAIN_IDS" --holdout "$HOLDOUT_IDS" \
                --epochs "$EPOCHS" --out "$DECIDE_OUT" $ABS_FLAG >>"$LOG" 2>&1
            echo "DECIDE_EXIT=$? group=$GROUP" | tee -a "$LOG"
        fi
    fi

    [[ "$MODE" == "decide_only" ]] && continue

    WINNER="$("$PY" -c "import json; print(json.load(open('$DECIDE_OUT'))['decision']['winner'])")"
    echo "[signal-tok] winner($GROUP)=$WINNER" | tee -a "$LOG"

    if [[ "$MODE" != "encode_only" || ! -f "$CKPT" ]]; then
        # ── 2. Train the winning bottleneck on the calibration train shots ────
        "$PY" -m imas_ambix.tokenizer.signal_hf_encode train \
            --group "$GROUP" --device cuda --patch-size "$PATCH_SIZE" \
            --bottleneck "$WINNER" --train "$TRAIN_IDS" \
            --epochs "$EPOCHS" --out "$CKPT" $ABS_FLAG >>"$LOG" 2>&1
        echo "TRAIN_EXIT=$? group=$GROUP" | tee -a "$LOG"
    fi

    # ── 3. Encode the FULL corpus into the v2 signals_hf store ───────────────
    # --shots all enumerates every shot on disk; skip-existing makes it
    # resumable; the per-group manifest records counts / skips / token totals.
    # NO_SKIP=1 forces a full re-encode even of already-absolute stores — needed
    # when the calibration (not the mode) changed, e.g. the dead-detector guard.
    NO_SKIP_FLAG=()
    [[ "${NO_SKIP:-0}" == "1" ]] && NO_SKIP_FLAG=(--no-skip-existing)
    "$PY" -m imas_ambix.tokenizer.signal_hf_encode encode \
        --group "$GROUP" --device cuda --num-workers "$NUM_WORKERS" \
        --shots "$ENCODE_IDS" --ckpt "$CKPT" --manifest "$MANIFEST" \
        $ABS_FLAG "${NO_SKIP_FLAG[@]}" >>"$LOG" 2>&1
    echo "ENCODE_EXIT=$? group=$GROUP" | tee -a "$LOG"

    cp "$LOG" "/work/projects/imas_gpu/logs/signal-tokenizer-${GROUP}-${SLURM_JOB_ID}.run.log" || true
done

echo "DONE=1"
