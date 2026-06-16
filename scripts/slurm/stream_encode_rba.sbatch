#!/bin/bash
#SBATCH --job-name=ambix-stream-rba
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --account=grpa
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=10:00:00
#SBATCH --output=/work/projects/imas_gpu/stream-rba-%j.out
#
# Single-GPU, single-shard in-process stream encoder for the rba visible
# camera (camera_visible.camera_lower). Reuses imas_ambix/data/stream_encode.py
# unchanged — the camera is selected via CAMERA below. rba arrives as
# (T,H,W) uint8 single-channel: normalise (no-op for uint8) -> grey-replicate
# to RGB -> resize 256^2, exactly the proven rbb path.
#
# Output lands in the LIVE frame token store alongside rbb:
#   tokens/v1/frames/<shot>/rba.zarr
# (STREAM_ROOT is the v1 root; save path appends frames/<shot>/<cam>.zarr).
#
# One card only (the reservation's idle pair must keep one card free for a
# concurrent tokeniser-dev job; DeepSeek stays up). Run cameras sequentially.
#
# Usage:
#   # decode-fidelity smoke (3 shots):
#   sbatch --export=ALL,MAX_SHOTS=3 scripts/slurm/stream_encode_rba.sbatch
#   # full corpus:
#   sbatch scripts/slurm/stream_encode_rba.sbatch

set -e
REPO=/home/ITER/mcintos/Code/imas-ambix
MAGVIT2_ROOT=/work/projects/imas_gpu/mast-tokens/v1/open-magvit2

# CWD MUST be the magvit2 root. VQModel.__init__ builds the LPIPS loss, which
# loads vgg.pth via a CWD-RELATIVE path. betelgeuse has NO outbound network, so
# a wrong CWD makes the vendored loader try to DOWNLOAD the weights and stall
# the job until its time limit (incident 2026-05-27, job 1208591).
cd "$MAGVIT2_ROOT"

# Drain-forensics sidecar (docs/rca-node-drain-final-2026-06-03.html).
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
CAMERA="${CAMERA:-rba}"
BATCH_FRAMES="${BATCH_FRAMES:-256}"
# Workers <= cpus-per-task; leave a core for the main/model thread.
NUM_WORKERS="${NUM_WORKERS:-7}"
# Fixed 300s ceiling never false-fires (no shot is that long at ~124 fps) yet
# still bounds a genuine wedge so the job exits cleanly instead of stalling.
BATCH_TIMEOUT_S="${BATCH_TIMEOUT_S:-300}"
# Defines the tokenization (the VQ forward is not batch-size invariant) AND
# drives GPU saturation — must match the rbb corpus value.
MODEL_FORWARD_BATCH="${MODEL_FORWARD_BATCH:-256}"
# v1 root: save path appends frames/<shot>/<cam>.zarr -> live corpus.
STREAM_ROOT="${STREAM_ROOT:-/work/projects/imas_gpu/mast-tokens/v1}"
REPORT=/work/projects/imas_gpu/stream-${CAMERA}-${SLURM_JOB_ID}.json

EXTRA=()
[ -n "${MAX_SHOTS:-}" ] && EXTRA+=(--max-shots "$MAX_SHOTS")
[ -n "${MAX_FRAMES:-}" ] && EXTRA+=(--max-frames "$MAX_FRAMES")
[ -n "${EXPLICIT_SHOTS:-}" ] && EXTRA+=(--explicit-shots "$EXPLICIT_SHOTS")
# Resumable by default: a walltime timeout flushes completed shots (SIGTERM
# handler), and a re-run skips them. Harmless no-op on a fresh first run.
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
