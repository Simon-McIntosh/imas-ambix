#!/bin/bash
#SBATCH --job-name=camdyn-baseline
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --account=grpa
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=08:00:00
#SBATCH --output=/work/projects/imas_gpu/logs/camdyn-%x-%j.out
# NOTE: --qos is intentionally omitted — auto-applied for this account/reservation.
#       Passing --qos explicitly errors with "QOS not permitted".
#
# D1 — per-frame spatial-inpainting BASELINE (temporal attention OFF) for the
# camera-dynamics world model. Trains the shared factorized ST-transformer +
# LFQ 18-bit head from scratch in token space (no downloads — betelgeuse has no
# outbound network; the token corpus + level-1 raw are already on GPFS) and
# logs the held-out masked-token NLL/top-1 as the LOCKED W1 bar.
#
# In-process (repo §2b): the model is loaded ONCE and many windows stream
# through a bounded torch DataLoader in the same process — NO subprocess-per-
# item, NO file-IPC daemon, NO unbounded prefetch threads.
#
# Usage:
#   sbatch scripts/slurm/camdyn_baseline.sbatch
#   sbatch --export=ALL,CONFIG=imas_ambix/camdyn/configs/baseline_w1_v0.yaml,MAX_STEPS=2000 \
#          scripts/slurm/camdyn_baseline.sbatch
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# ── SLURM-node workarounds (repo §2) ─────────────────────────────────────────
export TMPDIR=/scratch_local/${SLURM_JOB_ID}
mkdir -p "$TMPDIR"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
export TORCHINDUCTOR_CACHE_DIR="$TMPDIR/.torch_inductor"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR"
# Offline: a stray HF/torch-hub fetch fails fast instead of stalling the job
# (from-scratch model → nothing should download anyway).
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

# ── Drain-forensics sidecar (docs/rca-node-drain-final-2026-06-03.html §5) ──
bash "${REPO_ROOT}/scripts/slurm/drain_sidecar.sh" &

# ── Stage token shards GPFS → local NVMe (faster random reads in eval) ──────
# The frame-grid dataset opens one Zarr store per shot on demand; the corpus is
# large, so we DO NOT bulk-copy it. The dataset reads lazily from GPFS — fine
# for this I/O-light workload (256-cell grids). TMPDIR on local NVMe holds the
# torch inductor cache + scratch. (No staging step needed; documented here so a
# future tuner knows the I/O path is intentional.)

# ── Activate project venv (torch 2.11+cu130) ────────────────────────────────
VENV="${REPO_ROOT}/.venv"
if [[ -f "${VENV}/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source "${VENV}/bin/activate"
    PYTHON="${VENV}/bin/python"
else
    echo "[camdyn] .venv not found at ${VENV} — falling back to uv run" >&2
    PYTHON="uv run python"
fi
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# ── Config / overrides ──────────────────────────────────────────────────────
: "${CONFIG:=imas_ambix/camdyn/configs/baseline_w1_v0.yaml}"
: "${ARTIFACT_OUT:=${REPO_ROOT}/imas_ambix/camdyn/artifacts/baseline_w1_v0.json}"
: "${CKPT_ROOT:=/work/projects/imas_gpu/mast-checkpoints/camdyn}"

LOG=/work/projects/imas_gpu/logs/camdyn-train-${SLURM_JOB_ID}.log

echo "[$(date)] camdyn-baseline (D1, temporal OFF) starting"
echo "  job-id:   ${SLURM_JOB_ID}"
echo "  node:     $(hostname)"
echo "  config:   ${CONFIG}"
echo "  artifact: ${ARTIFACT_OUT}"
echo "  ckpt:     ${CKPT_ROOT}"
echo "  log:      ${LOG}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

EXTRA=()
[ -n "${MAX_STEPS:-}" ] && EXTRA+=(--max-steps "$MAX_STEPS")

cd "${REPO_ROOT}"
set +e
"${PYTHON}" -m imas_ambix.camdyn.train \
    --config "${CONFIG}" \
    --device cuda \
    --artifact-out "${ARTIFACT_OUT}" \
    --ckpt-root "${CKPT_ROOT}" \
    "${EXTRA[@]}" 2>&1 | tee "${LOG}"
D1_DONE=${PIPESTATUS[0]}
set -e

echo "[$(date)] camdyn-baseline finished" | tee -a "${LOG}"
echo "D1_DONE=${D1_DONE}" | tee -a "${LOG}"
exit "${D1_DONE}"
