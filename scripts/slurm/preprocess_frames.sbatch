#!/bin/bash
# preprocess_frames.sbatch — pre-decode rbb frames into (T, 256, 256, 3) uint8 Zarr stores.
#
# Reads /work/projects/imas_gpu/mast/level1/shots/<shot>.zarr/rbb
# Writes /work/projects/imas_gpu/mast/preprocessed/rbb-256/<shot>.zarr
#
# Usage
# -----
#   sbatch scripts/slurm/preprocess_frames.sbatch
#
# Prerequisites
# -------------
#   - L1 zarr-consolidate (job 1207977) must be complete before running at
#     full corpus scale.  Validates fine beforehand on a small --shot-ids run.
#   - uv (or the ambix venv) must be on PATH.
#
# Partition: sun (CPU, network + GPFS access — no GPU needed)
# Cores:     36 (saturates GPFS bandwidth for the open+normalise+write loop)
# Memory:    128G  [OOM fix 2026-05-27: job 1208031 OOM'd at ~95/9527 shots
#                  with 36 workers × ~2 GB peak >> 64 GB; drop to 12 workers
#                  (12 × ~2 GB = ~24 GB + overhead; smoke shows 10.8 GB for 8
#                  workers → 12 workers ≈ 16 GB peak; 128 GB gives 8× headroom.
#                  sun nodes cap at 192 GB so 256 GB is unavailable.)]
# Workers:   12 (memory-bounded; GPFS throughput still saturated by 12 threads)
# Time:      6h (conservative: ~9,432 remaining shots / 12 workers ≈ 3-4 h)
# Resize:    torch (q1 locked 2026-05-27 — cv2 not installed in ambix venv)
# Skip:      --skip-existing so the ~95 already-done shots are not redone.

#SBATCH --job-name=ambix-preprocess-frames
#SBATCH --partition=sun
#SBATCH --cpus-per-task=36
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=/work/projects/imas_gpu/preprocess-frames-%j.log
#SBATCH --error=/work/projects/imas_gpu/preprocess-frames-%j.log

set -euo pipefail

# /scratch_local only exists on betelgeuse; sun nodes use /tmp (auto-purged).
export TMPDIR=/tmp
mkdir -p "$TMPDIR"

# Activate ambix uv environment (adjust if using a different venv layout)
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
cd /home/ITER/mcintos/Code/imas-ambix

echo "[$(date -Iseconds)] node=$(hostname) job=${SLURM_JOB_ID:-local} cpus=36 workers=12 mem=128G"
echo "[$(date -Iseconds)] TMPDIR=$TMPDIR"

mkdir -p /work/projects/imas_gpu/mast/preprocessed/rbb-256

# --workers 12: memory-bounded (12 × ~2 GB peak ≈ 24 GB + overhead << 256 GB)
# --resize-backend torch: q1 locked — cv2 not installed; torch F.interpolate
#   matches the daemon resize byte-for-byte for token-id parity.
# --skip-existing: ~95 shots from job 1208031 are already done; do not redo.
uv run ambix data preprocess-frames \
    --from-bucket-all \
    --camera rbb \
    --image-size 256 \
    --resize-backend torch \
    --workers 12 \
    --skip-existing \
    --output "/work/projects/imas_gpu/mast/preprocessed/preprocess_report_${SLURM_JOB_ID:-local}.json"

echo "[$(date -Iseconds)] DONE"
