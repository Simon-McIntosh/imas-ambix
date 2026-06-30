#!/bin/bash
#SBATCH --job-name=l2-input-encode
#SBATCH --partition=sun
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/work/projects/imas_gpu/logs/l2-input-encode-%x-%j.out
#
# The L2 light path is CPU/IO-bound (a uniform quantiser over the control +
# SXR signals — NO model inference), so it runs on the `sun` CPU cluster
# (~19 idle nodes with /work access), NOT the GPU reservation (repo: IO-bound
# builds go to sun).  Absolute corpus calibration is THE default (--per-shot
# is a diagnostic escape).
#
# Leakage-free Level-2 INPUT light path: read → leakage-guard → uniform-
# quantise → write the provenance-verified L2 *inputs* (measured observables
# + authorised planned/demand waveforms) into the native-cadence v2 token
# store as {group}_l2.zarr.  This is the discriminator counterpart to the
# camera/world-model encode — it deliberately does NOT build a second
# high-frequency patch-transformer corpus (that is the in-flight
# signal_tokenizer job's work).
#
# gpu:1 only — the light path is CPU/IO-bound (a uniform quantiser over the
# 4 kHz control + 50 kHz SXR signals; no model inference), so one card is
# requested purely to share the Group-A reservation cleanly alongside the
# concurrent encodes.  The speed comes from streaming shots in a single
# long-lived process (repo §2b): the leakage-classifier + quantiser are
# loaded/constructed once and every shot streams through the same process —
# NO subprocess-per-shot, NO file-IPC daemon.
#
# Checkpoint/resume safe: --shots all enumerates every {id}.zarr on disk and
# the build skips a shot whose {group}_l2.zarr store already exists, so a
# re-run after a time-limit / cancellation picks up where it left off.
#
# Usage:
#   sbatch scripts/slurm/l2_input_encode.sbatch                 # full corpus
#   L2_SHOTS=11766,15199 sbatch scripts/slurm/l2_input_encode.sbatch
#   L2_NO_SKIP=1 sbatch scripts/slurm/l2_input_encode.sbatch    # re-encode all
set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
if [[ ! -d "${REPO_ROOT}/imas_ambix/data" ]]; then
    echo "[l2-input] ERROR: REPO_ROOT=${REPO_ROOT} is not the imas-ambix repo root." >&2
    echo "[l2-input] Submit from the repo root: cd <repo> && sbatch scripts/slurm/l2_input_encode.sbatch" >&2
    exit 2
fi

# ── SLURM-node workarounds (repo §2) ─────────────────────────────────────────
# On the sun CPU cluster /scratch_local does not exist; /tmp is the writable
# scratch (repo §2).
export TMPDIR=/tmp
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

PY="${REPO_ROOT}/.venv/bin/python"
L2_SHOTS="${L2_SHOTS:-all}"            # 'all' (every {id}.zarr on disk) or csv
# Absolute calibration is the default; ABSOLUTE=0 → explicit per-shot diagnostic.
ABSOLUTE="${ABSOLUTE:-1}"
ABS_FLAG=()
[[ "$ABSOLUTE" != "1" ]] && ABS_FLAG=(--per-shot)
LEVEL2_DIR="${LEVEL2_DIR:-/work/projects/imas_gpu/mast/level2/shots}"
MANIFEST_DIR="${MANIFEST_DIR:-/work/projects/imas_gpu/mast-tokens/v2/_manifests}"
MANIFEST="${MANIFEST:-${MANIFEST_DIR}/encode_l2_input.json}"
mkdir -p "$MANIFEST_DIR"

# Default: skip shots already encoded (resume).  L2_NO_SKIP=1 forces re-encode.
SKIP_FLAG=()
if [[ "${L2_NO_SKIP:-0}" == "1" ]]; then
    SKIP_FLAG=(--no-skip-existing)
fi

LOG="${TMPDIR}/run.log"
echo "[l2-input] shots='$L2_SHOTS' level2='$LEVEL2_DIR' manifest='$MANIFEST' no_skip=${L2_NO_SKIP:-0}" | tee -a "$LOG"

# Single long-lived process encodes every shot (repo §2b in-process default).
"$PY" -m imas_ambix.data.l2_input_build \
    --shots "$L2_SHOTS" \
    --level2-dir "$LEVEL2_DIR" \
    --manifest "$MANIFEST" \
    "${ABS_FLAG[@]}" "${SKIP_FLAG[@]}" >>"$LOG" 2>&1
echo "ENCODE_EXIT=$?" | tee -a "$LOG"

cp "$LOG" "/work/projects/imas_gpu/logs/l2-input-encode-${SLURM_JOB_ID}.run.log" || true
echo "DONE=1"
