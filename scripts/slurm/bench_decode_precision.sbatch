#!/bin/bash
#SBATCH --job-name=ambix-bench-precdiag
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --account=grpa
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=30
#SBATCH --mem=640G
#SBATCH --time=04:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=/work/projects/imas_gpu/logs/ambix-%x-%j.out

# ════════════════════════════════════════════════════════════════════════════
# DECODE-PRECISION DIAGNOSTIC  (RCA 2026-05-29)
# ════════════════════════════════════════════════════════════════════════════
# Question: is the fine-tune "regression" (bench rFID 24.5 vs baseline 13.3)
# a real reconstruction failure, or a bf16 weight-PRECISION artifact in the
# bench's decode path?
#
# The trainer's OWN eval reports the fine-tuned decoder at rFID ~14 (fp32 +
# autocast), while the bench reports ~24-26 on the SAME weights.  The bench
# casts the whole model to bf16 (stream_worker load_model) and decodes via
# ema_scope().  Encoder + quantizer are frozen and provably token-identical
# (C6/C7 byte-identical no-ops), so the entire divergence is DECODE-only.
#
# This job runs a 2×2 controlled grid — {baseline imagenet ckpt, optb
# fine-tuned ckpt} × {bf16 decode, fp32 decode} — through the IDENTICAL
# no-ema live-decode path (AMBIX_DECODE_NO_EMA=1).  Only two variables move:
# the checkpoint and the decoder precision.  The two bf16→fp32 deltas are the
# signal:
#   * optb delta >> baseline delta (e.g. optb 24.5→~14, baseline ~stable)
#       → H-bf16: bf16 specifically penalises the fine-tuned weights; the
#         fine-tune is NOT broken and the verdict must be reopened.
#   * both deltas small
#       → H-real: precision is not the cause; the bench was right.
#
# Reference (8-frame pool, established, EMA path):
#   baseline 1209100 = 17.04   optb 1209130 = 24.49
# (the optb live==ema by construction, so optb-bf16 here should re-anchor
#  near 24.49; baseline-bf16 here uses live≠ema so it is a fresh matched
#  control, not expected to equal 17.04 — the DELTA is what matters.)
# ════════════════════════════════════════════════════════════════════════════

set -euo pipefail

export TMPDIR=/scratch_local/$SLURM_JOB_ID
mkdir -p "$TMPDIR"
mkdir -p "/work/projects/imas_gpu/logs"

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

# ── STOP-FILE + SIGUSR1 trap (AGENTS.md §2a-cancel / §2a-cancel-time) ────────
mkdir -p /work/projects/imas_gpu/stops
export AMBIX_STOP_FILE=/work/projects/imas_gpu/stops/${SLURM_JOB_ID}.stop
rm -f "${AMBIX_STOP_FILE}"
_on_sigusr1() {
    echo "[sbatch] SIGUSR1 received (pre-limit) — touching STOP-FILE for clean stop" >&2
    touch "${AMBIX_STOP_FILE}"
}
trap _on_sigusr1 USR1

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_ROOT}"
RESULTS_DIR="${REPO_ROOT}/imas_ambix/bench/results"
mkdir -p "${RESULTS_DIR}"

VENV="${REPO_ROOT}/.venv"
if [[ -f "${VENV}/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source "${VENV}/bin/activate"
    AMBIX="${VENV}/bin/ambix"
else
    echo "[precdiag] .venv not found — falling back to uv run" >&2
    AMBIX="uv run ambix"
fi

echo "[$(date)] bench-precdiag starting"
echo "  job-id:    ${SLURM_JOB_ID}"
echo "  node:      $(hostname)"
echo "  stop-file: ${AMBIX_STOP_FILE}  (touch to cancel cleanly — DO NOT scancel)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# All cells decode via the LIVE decoder (bypass ema_scope) so the fp32 upcast
# is not overwritten by the bf16 EMA shadow.  Encoder stays bf16 → tokens are
# byte-identical to every established bench.
export AMBIX_DECODE_NO_EMA=1

BASELINE_CFG="imas_ambix/bench/configs/v0-rbb-100shot.yaml"        # imagenet decoder
OPTB_CFG="imas_ambix/bench/configs/v0-rbb-100shot-optb.yaml"       # fine-tuned merged ckpt

run_cell () {
    local tag="$1" cfg="$2" fp32="$3"
    local out="${RESULTS_DIR}/v0-rbb-100shot-precdiag-${tag}-${SLURM_JOB_ID}.json"
    if [[ "${fp32}" == "1" ]]; then
        export AMBIX_DECODER_FP32=1
    else
        unset AMBIX_DECODER_FP32 || true
    fi
    echo "──────────────────────────────────────────────────────────────────"
    echo "[$(date)] CELL ${tag}  (AMBIX_DECODER_FP32=${fp32}, AMBIX_DECODE_NO_EMA=1)"
    echo "  config: ${cfg}"
    echo "  output: ${out}"
    if [[ -f "${AMBIX_STOP_FILE}" ]]; then
        echo "[precdiag] STOP-FILE present — skipping remaining cells" >&2
        return 0
    fi
    $AMBIX tokenize bench --config "${cfg}" --output "${out}" --in-process
    echo "[$(date)] CELL ${tag} done"
}

# 2×2 grid — bf16 first (re-anchors to established), then fp32 (the test).
run_cell "baseline-bf16" "${BASELINE_CFG}" 0
run_cell "optb-bf16"     "${OPTB_CFG}"     0
run_cell "baseline-fp32" "${BASELINE_CFG}" 1
run_cell "optb-fp32"     "${OPTB_CFG}"     1

echo "[$(date)] bench-precdiag finished (all cells complete)"
