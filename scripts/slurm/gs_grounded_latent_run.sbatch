#!/bin/bash
# GS-grounded latent engine — real-data gate run (v0 stage 2).
# CPU-light (per-slice GS-inverse + numpy topology + a small MLP smoke fit): runs
# on the debug CPU partition, leaving the GPU node for model serving.  In-process,
# SIGTERM-clean, resume-safe (partial JSON flushed on signal).
#SBATCH --job-name=gs-grounded-latent
#SBATCH --partition=sun_debug
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.log

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
export TMPDIR=/tmp OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

OUT="${OUT:-imas_ambix/latent/artifacts}"
uv run python scripts/gs_grounded_latent_run.py \
    --n-train "${N_TRAIN:-150}" \
    --n-heldout "${N_HELDOUT:-8}" \
    --grid-nr "${GRID_NR:-65}" \
    --grid-nz "${GRID_NZ:-97}" \
    --order "${ORDER:-1}" \
    --ridge "${RIDGE:-0.01}" \
    --max-slices-per-shot "${MAX_SLICES:-40}" \
    --smoke-train-steps "${SMOKE_STEPS:-400}" \
    --out "${OUT}"
echo "RUN_EXIT=$?"
