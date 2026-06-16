#!/bin/bash
#SBATCH --job-name=ambix-stream-rgb
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --account=grpa
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=10:00:00
#SBATCH --output=/work/projects/imas_gpu/stream-rgb-%j.out
#
# Single-GPU, single-shard in-process stream encoder for the rgb bremsstrahlung
# camera (camera_visible.bremsstrahlung_a). Reuses
# imas_ambix/data/stream_encode.py unchanged.
#
# NORMALISE (proven path, deliberate): rgb arrives as (T,H,W) uint16 with a
# narrow per-shot range (e.g. [0,4027]). normalise_frames_to_uint8 applies a
# PER-SHOT global min/max -> uint8, then grey-replicates to RGB. This is the
# SAME normalise the rbb corpus uses; keeping it makes the bremsstrahlung
# cameras consistent with the existing token store and preserves byte-identity
# with the live worker path. A bespoke percentile/log normalise was considered
# and rejected: there is no downstream requirement for it and it would fork the
# tokenization away from the rbb corpus. Revisit only if a downstream task
# shows the dynamic range needs it.
#
# Output lands in the LIVE frame token store alongside rbb:
#   tokens/v1/frames/<shot>/rgb.zarr
#
# One card only; run cameras sequentially. DeepSeek stays up.
#
# Usage:
#   sbatch --export=ALL,MAX_SHOTS=3 scripts/slurm/stream_encode_rgb.sbatch  # smoke
#   sbatch scripts/slurm/stream_encode_rgb.sbatch                           # full

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
CAMERA="${CAMERA:-rgb}"
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
