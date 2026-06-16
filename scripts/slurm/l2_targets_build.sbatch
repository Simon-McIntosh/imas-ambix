#!/bin/bash
#SBATCH --job-name=l2-targets-build
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --account=grpa
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=200G
#SBATCH --time=06:00:00
#SBATCH --output=/work/projects/imas_gpu/logs/l2-targets-build-%x-%j.out
# NOTE: --qos is intentionally omitted — auto-applied for this account/
#       reservation. Passing --qos explicitly errors with "QOS not permitted".
#
# Build the eval-only TARGET store from the MAST Level-2 corpus.
#
# The L2 equilibrium reconstruction (psi / j_phi / q / boundary / X-point /
# magnetic axis / global scalars) plus the reconstruction-derived globals
# (greenwald_density, line_average_n_e=ESM_NE_BAR, loop voltage) and the
# programmed reference waveforms (i_plasma, n_e_line) are the world-model
# PREDICTION targets — never inputs.  They are written to TARGET_ROOT
# (/work/projects/imas_gpu/mast-targets), which is NOT under TOKEN_ROOT, so no
# glob the input loader runs over mast-tokens/v2 can reach them (Wall 1).  Each
# quantity is stored raw, on its native time grid, with a per-quantity finite/
# validity mask (NaN-outside-the-pulse recorded by the mask, never zero-filled).
#
# This is a pure xarray/numpy read+write of L2 Zarr — no model load — so the
# GPU is not actually exercised; gpu:1 is requested only to ride the Group-A
# reservation (the other cards stay free for concurrent encode/serve work).
# The job is IO-bound on the GPFS read of the ~65x65xT equilibrium grids.
#
# Resume-safe: --shots all enumerates the inventory shot list and the driver
# skips any shot whose equilibrium target store already exists, so a re-run
# after a time-limit/cancellation picks up where it left off.
#
# Usage:
#   sbatch scripts/slurm/l2_targets_build.sbatch                       # full corpus
#   TARGET_SHOTS=13277,15325 sbatch scripts/slurm/l2_targets_build.sbatch  # subset
#   TARGET_SHOTS=/path/shots.txt sbatch ...                            # explicit list
#   NO_SKIP=1 sbatch scripts/slurm/l2_targets_build.sbatch            # rebuild all
set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
if [[ ! -d "${REPO_ROOT}/imas_ambix/tokenizer" ]]; then
    echo "[l2-targets] ERROR: REPO_ROOT=${REPO_ROOT} is not the imas-ambix repo root." >&2
    echo "[l2-targets] Submit from the repo root: cd <repo> && sbatch scripts/slurm/l2_targets_build.sbatch" >&2
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
TARGET_SHOTS="${TARGET_SHOTS:-all}"        # 'all' → inventory list; or list / file
MANIFEST_DIR="${MANIFEST_DIR:-/work/projects/imas_gpu/mast-targets/_manifests}"
mkdir -p "$MANIFEST_DIR"
MANIFEST="${MANIFEST_DIR}/build_${SLURM_JOB_ID}.json"

SKIP_FLAG=""
[[ "${NO_SKIP:-0}" == "1" ]] && SKIP_FLAG="--no-skip-existing"

LOG="${TMPDIR}/run.log"
echo "[l2-targets] shots='$TARGET_SHOTS' manifest=$MANIFEST skip=$([[ -n $SKIP_FLAG ]] && echo no || echo yes)" | tee -a "$LOG"

"$PY" -m imas_ambix.data.l2_targets_build \
    --shots "$TARGET_SHOTS" \
    --manifest "$MANIFEST" \
    $SKIP_FLAG >>"$LOG" 2>&1
echo "BUILD_EXIT=$?" | tee -a "$LOG"

cp "$LOG" "/work/projects/imas_gpu/logs/l2-targets-build-${SLURM_JOB_ID}.run.log" || true
echo "DONE=1"
