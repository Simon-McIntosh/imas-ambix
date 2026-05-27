#!/bin/bash
#SBATCH --job-name=ambix-stream-rbb
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --account=grpa
#SBATCH --array=0-3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=7
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --output=/work/projects/imas_gpu/stream-rbb-shard-%A_%a.out
#
# NEW high-throughput rbb encoder — in-process, continuous-batching, GPU
# saturation path. Runs in the Open-MAGVIT2 venv (torch 2.5.1+cu124, in the
# same process as the driver — NO subprocess/file-IPC). Each array task takes
# one GPU and one stride-shard of the shotlist (same sharding as the live
# encode_rbb_4shard.sbatch), so all 4 cards run a within-task continuous
# batch stream.
#
# Validation output goes to .../v1/frames-stream/ — NOT the live tokens dir.
# This does not touch the live per-shot daemon path.
#
# Usage:
#   # full 4-GPU array:
#   sbatch scripts/slurm/stream_encode_rbb.sbatch
#   # 1-GPU validation of N shots (override via env before sbatch):
#   sbatch --array=0 --export=ALL,MAX_SHOTS=200,BATCH_FRAMES=256 \
#          scripts/slurm/stream_encode_rbb.sbatch

set -e
cd /home/ITER/mcintos/Code/imas-ambix
export TMPDIR=/tmp
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD:$PYTHONPATH"

VENV_PY=/work/projects/imas_gpu/mast-tokens/v1/open-magvit2/.venv/bin/python
N_SHARDS="${N_SHARDS:-4}"
CAMERA="${CAMERA:-rbb}"
BATCH_FRAMES="${BATCH_FRAMES:-256}"
NUM_WORKERS="${NUM_WORKERS:-12}"
STREAM_ROOT="${STREAM_ROOT:-/work/projects/imas_gpu/mast-tokens/v1/frames-stream}"
REPORT=/work/projects/imas_gpu/stream-rbb-shard-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.json

EXTRA=()
[ -n "${MAX_SHOTS:-}" ] && EXTRA+=(--max-shots "$MAX_SHOTS")
[ -n "${MAX_FRAMES:-}" ] && EXTRA+=(--max-frames "$MAX_FRAMES")
[ -n "${EXPLICIT_SHOTS:-}" ] && EXTRA+=(--explicit-shots "$EXPLICIT_SHOTS")

echo "=== stream shard ${SLURM_ARRAY_TASK_ID}/${N_SHARDS} on $(hostname) ==="
echo "started:    $(date +%H:%M:%S)"
echo "SLURM CVD:  $CUDA_VISIBLE_DEVICES"
echo "batch_frames=$BATCH_FRAMES num_workers=$NUM_WORKERS"
echo "stream_root=$STREAM_ROOT"
echo "report:     $REPORT"

time "$VENV_PY" imas_ambix/data/stream_encode.py \
    --shard "$SLURM_ARRAY_TASK_ID" \
    --n-shards "$N_SHARDS" \
    --camera "$CAMERA" \
    --device cuda \
    --batch-frames "$BATCH_FRAMES" \
    --num-workers "$NUM_WORKERS" \
    --stream-root "$STREAM_ROOT" \
    --report "$REPORT" \
    "${EXTRA[@]}"

echo "=== stream shard ${SLURM_ARRAY_TASK_ID} done $(date +%H:%M:%S) ==="
