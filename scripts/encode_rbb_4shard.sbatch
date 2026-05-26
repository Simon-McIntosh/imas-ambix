#!/bin/bash
#SBATCH --job-name=ambix-rbb-shard
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --account=grpa
#SBATCH --array=0-3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=7
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --output=/work/projects/imas_gpu/bulk-rbb-shard-%A_%a.out

set -e
cd /home/ITER/mcintos/Code/imas-ambix
export TMPDIR=/scratch_local/$SLURM_JOB_ID && mkdir -p "$TMPDIR"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1

OUTDIR=/work/projects/imas_gpu/mast/tokens
REPORT=/work/projects/imas_gpu/bulk-rbb-shard-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.json

echo "=== shard ${SLURM_ARRAY_TASK_ID}/4 on $(hostname) ==="
echo "started:    $(date +%H:%M:%S)"
echo "SLURM CVD:  $CUDA_VISIBLE_DEVICES"
echo "report:     $REPORT"

time uv run python scripts/encode_one_shard.py \
    --shard "$SLURM_ARRAY_TASK_ID" \
    --n-shards 4 \
    --camera rbb \
    --output-dir "$OUTDIR" \
    --report "$REPORT"

echo "=== shard ${SLURM_ARRAY_TASK_ID} done $(date +%H:%M:%S) ==="
