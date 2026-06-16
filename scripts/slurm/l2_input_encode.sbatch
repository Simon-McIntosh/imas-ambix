#!/bin/bash
#SBATCH --job-name=l2-input-encode
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --account=grpa
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=/work/projects/imas_gpu/logs/l2-input-encode-%x-%j.out
# NOTE: --qos is intentionally omitted — auto-applied for this account/
#       reservation. Passing --qos explicitly errors with "QOS not permitted".
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
export TMPDIR=/scratch_local/${SLURM_JOB_ID}
mkdir -p "$TMPDIR"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

# ── Drain-forensics sidecar (repo §2a) ───────────────────────────────────────
bash "${REPO_ROOT}/scripts/slurm/drain_sidecar.sh" &

PY="${REPO_ROOT}/.venv/bin/python"
L2_SHOTS="${L2_SHOTS:-all}"            # 'all' (every {id}.zarr on disk) or csv
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
    "${SKIP_FLAG[@]}" >>"$LOG" 2>&1
echo "ENCODE_EXIT=$?" | tee -a "$LOG"

cp "$LOG" "/work/projects/imas_gpu/logs/l2-input-encode-${SLURM_JOB_ID}.run.log" || true
echo "DONE=1"
