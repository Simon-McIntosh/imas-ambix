#!/bin/bash
#SBATCH --job-name=ambix-fit-targets
#SBATCH --partition=sun
#SBATCH --array=0-7
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=/work/projects/imas_gpu/logs/ambix-%x-%A_%a.out
# Precompute the fit-amortization targets (per-slice profile fits over the
# TRAINING corpus) on the sun CPU cluster — 8 shards, resume-safe (per-shot
# JSON, existing files skipped), no GPU, no EFIT.

set -uo pipefail
export TMPDIR=/tmp OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/.venv/bin/activate"

python scripts/precompute_profile_fit_targets.py \
    --n-train "${FIT_N_TRAIN:-200}" \
    --max-slices-per-shot "${FIT_MAX_SLICES:-40}" \
    --workers 16 \
    --shard-index "${SLURM_ARRAY_TASK_ID}" \
    --shard-count 8
echo "SHARD_EXIT=$?"
