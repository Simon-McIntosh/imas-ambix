#!/bin/bash
#SBATCH --job-name=ambix-grounded-latent
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --account=grpa
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/work/projects/imas_gpu/logs/ambix-%x-%j.out
# NOTE: --qos intentionally omitted — defaults to normal; a 1-card job bursts
# onto a free card and coexists with any model serve.
#
# ============================================================================
#  GS-grounded hybrid latent, patch-current substrate — corpus trainer.
# ============================================================================
#  Trains ONE shared HybridLatentEncoder against every campaign signature's
#  GSGroundedLatentEngine on consecutive (t, t+1) slice pairs: whitened
#  magnetics misfit + Rogowski Ip anchor + discrepancy-scheduled GS structure
#  residual + flux-diffusion transport guard-rails + the self-supervised
#  closure-coordinate readout.  The referee is never read here.
#
#  Staged retrains (e.g. a 1000-shot run, then a 2000-shot run) are separate
#  launches with RUN and N_TRAIN set explicitly:
#      RUN=n1000 N_TRAIN=1000 sbatch scripts/slurm/train_grounded_latent_engine.sbatch
#      RUN=n2000 N_TRAIN=2000 sbatch scripts/slurm/train_grounded_latent_engine.sbatch
#
#  Corpus assembly is transparently cached (assemble_corpus_cached() in
#  train_grounded_latent_engine.py) under
#  /work/projects/imas_gpu/latent/grounded_latent_engine/corpus_cache/ keyed by
#  a hash of (shots, nr, nz, min_ip_ka, COIL_MODEL_VERSION) — a run reuses a
#  prior run's assembly for the SAME N_TRAIN with no re-read.
#
#  This launcher is single-GPU only (no torchrun/DDP entry point in
#  train_grounded_latent_engine.py — CorpusTrainer supports a distributed
#  all-reduce, but nothing here initialises the process group).  A 2-card
#  budget is spent as TWO independent single-GPU launches (e.g. a scale
#  comparison or a seed ablation), not one multi-GPU job.
# ============================================================================

set -uo pipefail
cd /home/ITER/mcintos/Code/imas-ambix
export TMPDIR=/scratch_local/$SLURM_JOB_ID
mkdir -p "$TMPDIR" || export TMPDIR=/tmp
source .venv/bin/activate
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=16
N_TRAIN="${N_TRAIN:-1000}"
RUN="${RUN:-n${N_TRAIN}}"
STEPS="${STEPS:-20000}"
LOG=/work/projects/imas_gpu/logs/ambix-grounded-latent-${RUN}-${SLURM_JOB_ID}.log
echo "PYTHON=$(which python)  RUN=${RUN}  N_TRAIN=${N_TRAIN}  STEPS=${STEPS}"
nvidia-smi -L | head -2

python scripts/train_grounded_latent_engine.py \
    --run "${RUN}" \
    --n-train "${N_TRAIN}" \
    --steps "${STEPS}" \
    --batch-size 256 \
    --device cuda \
    > "${LOG}" 2>&1
exit_code=$?
echo "TRAIN_EXIT=${exit_code}"
tail -30 "${LOG}"
# propagate the Python exit code as this SCRIPT's exit code -- without this,
# `tail` (unconditionally exit 0) was the last command, so SLURM/sacct
# reported COMPLETED for job 1225258 even though training had crashed.
exit "${exit_code}"
