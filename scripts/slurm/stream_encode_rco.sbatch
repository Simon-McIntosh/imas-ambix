#!/bin/bash
#SBATCH --job-name=ambix-stream-rco
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --account=grpa
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=10:00:00
#SBATCH --output=/work/projects/imas_gpu/stream-rco-%j.out
#
# Single-GPU, single-shard in-process stream encoder for the rco COLOUR
# camera (camera_visible.camera_color). Reuses imas_ambix/data/stream_encode.py
# unchanged.
#
# COLOUR HANDLING (verified, no code change needed): rco arrives as
# (T,H,W,3) uint8 — a REAL 3-channel colour camera, not a single plane.
#   - normalise_frames_to_uint8 short-circuits on uint8 (returns the array
#     untouched), so the true R/G/B values are NOT min/max-collapsed.
#   - frames_to_rgb_uint8 passes (T,H,W,3) uint8 straight through (no
#     grey-replicate branch is taken).
#   - frames_to_input permutes (T,H,W,3)->(T,3,H,W) and bilinear-resizes,
#     preserving colour fully.
# This is byte-identical to the live worker._frames_to_input colour path. The
# encoder does NOT grey-collapse this camera; rco tokens carry real colour.
#
# Output lands in the LIVE frame token store alongside rbb:
#   tokens/v1/frames/<shot>/rco.zarr
#
# One card only (keep one of the reservation's idle pair free for a concurrent
# tokeniser-dev job; DeepSeek stays up). Run cameras sequentially.
#
# Usage:
#   sbatch --export=ALL,MAX_SHOTS=3 scripts/slurm/stream_encode_rco.sbatch  # smoke
#   sbatch scripts/slurm/stream_encode_rco.sbatch                           # full

set -e
REPO=/home/ITER/mcintos/Code/imas-ambix
MAGVIT2_ROOT=/work/projects/imas_gpu/mast-tokens/v1/open-magvit2

cd "$MAGVIT2_ROOT"

bash "${REPO}/scripts/slurm/drain_sidecar.sh" &

export TMPDIR=/scratch_local/$SLURM_JOB_ID
mkdir -p "$TMPDIR"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO:$PYTHONPATH"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

LPIPS_REL="src/Open_MAGVIT2/modules/autoencoder/lpips/vgg.pth"
if [ ! -f "$MAGVIT2_ROOT/$LPIPS_REL" ]; then
    echo "FATAL: LPIPS weights missing at $MAGVIT2_ROOT/$LPIPS_REL" >&2
    echo "       (betelgeuse has no network — cannot download). Aborting." >&2
    exit 1
fi

VENV_PY=/work/projects/imas_gpu/mast-tokens/v1/open-magvit2/.venv/bin/python
CAMERA="${CAMERA:-rco}"
BATCH_FRAMES="${BATCH_FRAMES:-256}"
NUM_WORKERS="${NUM_WORKERS:-7}"
BATCH_TIMEOUT_S="${BATCH_TIMEOUT_S:-300}"
MODEL_FORWARD_BATCH="${MODEL_FORWARD_BATCH:-256}"
STREAM_ROOT="${STREAM_ROOT:-/work/projects/imas_gpu/mast-tokens/v1}"
REPORT=/work/projects/imas_gpu/stream-${CAMERA}-${SLURM_JOB_ID}.json

EXTRA=()
[ -n "${MAX_SHOTS:-}" ] && EXTRA+=(--max-shots "$MAX_SHOTS")
[ -n "${MAX_FRAMES:-}" ] && EXTRA+=(--max-frames "$MAX_FRAMES")
[ -n "${EXPLICIT_SHOTS:-}" ] && EXTRA+=(--explicit-shots "$EXPLICIT_SHOTS")
[ "${SKIP_EXISTING:-1}" = "1" ] && EXTRA+=(--skip-existing)

echo "=== stream ${CAMERA} (1 GPU, single shard) on $(hostname) ==="
echo "started:    $(date +%H:%M:%S)"
echo "SLURM CVD:  $CUDA_VISIBLE_DEVICES"
echo "batch_frames=$BATCH_FRAMES num_workers=$NUM_WORKERS mfb=$MODEL_FORWARD_BATCH"
echo "stream_root=$STREAM_ROOT  (-> frames/<shot>/${CAMERA}.zarr)"
echo "report:     $REPORT"

time "$VENV_PY" "$REPO/imas_ambix/data/stream_encode.py" \
    --shard 0 \
    --n-shards 1 \
    --camera "$CAMERA" \
    --device cuda \
    --batch-frames "$BATCH_FRAMES" \
    --num-workers "$NUM_WORKERS" \
    --batch-timeout-s "$BATCH_TIMEOUT_S" \
    --model-forward-batch "$MODEL_FORWARD_BATCH" \
    --stream-root "$STREAM_ROOT" \
    --report "$REPORT" \
    "${EXTRA[@]}"

echo "=== stream ${CAMERA} done $(date +%H:%M:%S) ==="
