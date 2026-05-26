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
#SBATCH --output=%x-%j.log
#SBATCH --error=%x-%j.err

set -euo pipefail

# Ephemeral scratch (never write model intermediates to login-node /tmp)
export TMPDIR=/scratch_local/${SLURM_JOB_ID:-$$}
mkdir -p "$TMPDIR"

# Activate ambix uv environment (adjust if using a different venv layout)
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
cd /home/ITER/mcintos/Code/imas-ambix

echo "[$(date -Iseconds)] node=$(hostname) job=${SLURM_JOB_ID:-local} cpus=36"
echo "[$(date -Iseconds)] TMPDIR=$TMPDIR"

uv run ambix data preprocess-frames \
    --from-bucket-all \
    --camera rbb \
    --image-size 256 \
    --resize-backend cv2 \
    --workers 36 \
    --skip-existing \
    --output "${TMPDIR}/preprocess_report.json"

# Copy report to a persistent location alongside the job log
cp "${TMPDIR}/preprocess_report.json" \
    "/work/projects/imas_gpu/mast/preprocessed/preprocess_report_${SLURM_JOB_ID:-local}.json" \
    2>/dev/null || true

echo "[$(date -Iseconds)] DONE"
