#!/bin/bash
#SBATCH --job-name=ambix-patch-enc
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
#  Amortised patch-current encoder — corpus trainer on one H200.
# ============================================================================
#  Trains ONE shared transformer that maps a 12-step window of RAW magnetics
#  (+ known PF currents + Rogowski Ip) to the in-limiter patch-current vector
#  in a single forward pass, against the same force-balanced loss the per-slice
#  variational inverse minimises (masked whitened misfit + Ip anchor +
#  discrepancy-scheduled GS structure residual).  The referee is never read.
#
#  The 2-arm ablation (--head direct | lowrank) is two runs of this launcher;
#  set HEAD in the environment (default: direct):
#      HEAD=lowrank sbatch scripts/slurm/train_patch_encoder.sbatch
#  Checkpoints + config: /work/projects/imas_gpu/latent/patch_encoder/<HEAD>/
#
#  Corpus assembly is transparently cached (assemble_corpus_cached() in
#  train_patch_encoder.py) under
#  /work/projects/imas_gpu/latent/patch_encoder/corpus_cache/ keyed by a hash
#  of (N_TRAIN, nr, nz, t_steps, stride, min_ip) — a run reuses a prior run's
#  (or scripts/slurm/patch_encoder_assemble_cpu.sbatch's CPU-sharded) assembly
#  for the SAME N_TRAIN with no re-read.  Staged retrains (e.g. direct-5k,
#  direct-full) are separate launches with RUN set explicitly:
#      RUN=direct-5k N_TRAIN=5000 sbatch scripts/slurm/train_patch_encoder.sbatch
#      RUN=direct-full N_TRAIN=10846 sbatch scripts/slurm/train_patch_encoder.sbatch
# ============================================================================

set -u
cd /home/ITER/mcintos/Code/imas-ambix
export TMPDIR=/scratch_local/$SLURM_JOB_ID
mkdir -p "$TMPDIR" || export TMPDIR=/tmp
source .venv/bin/activate
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
HEAD="${HEAD:-direct}"
N_TRAIN="${N_TRAIN:-200}"
LOG=/work/projects/imas_gpu/logs/ambix-patch-enc-${HEAD}-${SLURM_JOB_ID}.log
echo "PYTHON=$(which python)  HEAD=${HEAD}"
nvidia-smi -L | head -2

python scripts/train_patch_encoder.py \
    --head "${HEAD}" \
    --run "${RUN:-$HEAD}" \
    --n-train "${N_TRAIN}" \
    --epochs 60 \
    --batch-size 64 \
    --device cuda \
    > "${LOG}" 2>&1
echo "TRAIN_EXIT=$?"
tail -20 "${LOG}"
