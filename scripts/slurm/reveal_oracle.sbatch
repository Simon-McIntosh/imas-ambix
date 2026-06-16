#!/bin/bash
#SBATCH --job-name=reveal-oracle
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --account=grpa
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/work/projects/imas_gpu/logs/ambix-%x-%j.out
# NOTE: --qos intentionally omitted (auto-applied on this reservation).

# In-view reveal-k oracle + second-crop proxy gate — eval-only, no training,
# no encode.  Runs a smoke (1 window, k=4) first, then the full sweep.

set -euo pipefail

export TMPDIR=/scratch_local/$SLURM_JOB_ID
mkdir -p "$TMPDIR"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_ROOT}"
mkdir -p "/work/projects/imas_gpu/logs"

PY="${REPO_ROOT}/.venv/bin/python"

echo "[reveal] ===== SMOKE (1 window, k=4) ====="
"${PY}" -m imas_ambix.camdyn.reveal_oracle --smoke \
    --json "${TMPDIR}/reveal_smoke.json"
echo "SMOKE_EXIT=$?"

echo "[reveal] ===== FULL SWEEP (both windows, k=2/4/8) ====="
"${PY}" -m imas_ambix.camdyn.reveal_oracle
echo "FULL_EXIT=$?"
echo "DONE=1"
