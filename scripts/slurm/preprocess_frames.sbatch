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
# Memory:    64G (36 × ~1.8 GB per worker peak for uint16 + intermediates)
# Time:      3h (conservative: ~9,527 shots at ~0.4 s/shot / 36 workers ≈ 1.7 h)
#
# DO NOT submit until job 1207977 (zarr-consolidate) completes.
# Monitor: squeue -j 1207977

#SBATCH --job-name=ambix-preprocess-frames
#SBATCH --partition=sun
#SBATCH --cpus-per-task=36
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/work/projects/imas_gpu/preprocess-frames-%j.log
#SBATCH --error=/work/projects/imas_gpu/preprocess-frames-%j.log

set -euo pipefail

# /scratch_local only exists on betelgeuse; sun nodes use /tmp (auto-purged).
export TMPDIR=/tmp
mkdir -p "$TMPDIR"

# Activate ambix uv environment (adjust if using a different venv layout)
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
cd /home/ITER/mcintos/Code/imas-ambix

echo "[$(date -Iseconds)] node=$(hostname) job=${SLURM_JOB_ID:-local} cpus=36"
echo "[$(date -Iseconds)] TMPDIR=$TMPDIR"

# resize-backend=torch (q1 locked 2026-05-27): cv2 is not installed in the
# ambix venv, and torch F.interpolate matches the daemon resize byte-for-byte
# so precomputed and live-encoded shots yield identical tokens.
mkdir -p /work/projects/imas_gpu/mast/preprocessed/rbb-256
uv run ambix data preprocess-frames \
    --from-bucket-all \
    --camera rbb \
    --image-size 256 \
    --resize-backend torch \
    --workers 36 \
    --skip-existing \
    --output "/work/projects/imas_gpu/mast/preprocessed/preprocess_report_${SLURM_JOB_ID:-local}.json"

echo "[$(date -Iseconds)] DONE"
