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
REPO=/home/ITER/mcintos/Code/imas-ambix
MAGVIT2_ROOT=/work/projects/imas_gpu/mast-tokens/v1/open-magvit2

# CWD MUST be the magvit2 root. VQModel.__init__ builds the LPIPS loss, which
# loads vgg.pth via a CWD-RELATIVE path. betelgeuse has NO outbound network, so
# a wrong CWD makes the vendored loader try to DOWNLOAD the weights and hang in
# unkillable I/O — and a scancel of that hang DRAINS the node ("Kill task
# failed", node-drain 2026-05-27 #3, RCA docs/rca-node-drain-2026-05-27b.html).
cd "$MAGVIT2_ROOT"

export TMPDIR=/tmp
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO:$PYTHONPATH"
# Force offline: any stray HF/torch-hub fetch fails fast instead of hanging on
# a doomed (no-network) download that can wedge and drain the node.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

# Preflight: the LPIPS weights MUST be on disk at the CWD-relative path the
# vendored loader uses. Fail FAST (clear error, clean exit) rather than letting
# a download attempt hang — converting a node-drain risk into a 1-line error.
LPIPS_REL="src/Open_MAGVIT2/modules/autoencoder/lpips/vgg.pth"
if [ ! -f "$MAGVIT2_ROOT/$LPIPS_REL" ]; then
    echo "FATAL: LPIPS weights missing at $MAGVIT2_ROOT/$LPIPS_REL" >&2
    echo "       (betelgeuse has no network — cannot download). Aborting." >&2
    exit 1
fi

VENV_PY=/work/projects/imas_gpu/mast-tokens/v1/open-magvit2/.venv/bin/python
N_SHARDS="${N_SHARDS:-4}"
CAMERA="${CAMERA:-rbb}"
BATCH_FRAMES="${BATCH_FRAMES:-256}"
# Workers MUST be <= cpus-per-task (7) — oversubscribing froze a DataLoader
# worker in the 30-shot smoke (1208671), starving the next shot until the
# per-shot watchdog fired and aborted. 6 leaves one core for the main/model.
NUM_WORKERS="${NUM_WORKERS:-6}"
# Per-shot watchdog ceiling. 0=auto (max(60, 8x median)) false-fired on a large
# shot; a fixed 300s never false-fires (no shot is ~37k frames at ~124 fps) yet
# still bounds a genuine wedge so the job exits cleanly (no node drain).
BATCH_TIMEOUT_S="${BATCH_TIMEOUT_S:-300}"
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

time "$VENV_PY" "$REPO/imas_ambix/data/stream_encode.py" \
    --shard "$SLURM_ARRAY_TASK_ID" \
    --n-shards "$N_SHARDS" \
    --camera "$CAMERA" \
    --device cuda \
    --batch-frames "$BATCH_FRAMES" \
    --num-workers "$NUM_WORKERS" \
    --batch-timeout-s "$BATCH_TIMEOUT_S" \
    --stream-root "$STREAM_ROOT" \
    --report "$REPORT" \
    "${EXTRA[@]}"

echo "=== stream shard ${SLURM_ARRAY_TASK_ID} done $(date +%H:%M:%S) ==="
