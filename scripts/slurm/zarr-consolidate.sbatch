#!/bin/bash
#SBATCH --job-name=zarr-consolidate
#SBATCH --partition=sun_debug
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/work/projects/imas_gpu/zarr-consolidate-%j.log
#SBATCH --error=/work/projects/imas_gpu/zarr-consolidate-%j.log

set -eu
export TMPDIR=/tmp
cd /home/ITER/mcintos/Code/imas-ambix

echo "=== node $(hostname)  $(date +%H:%M:%S) ==="
echo "    cores: $SLURM_CPUS_PER_TASK   mem: $SLURM_MEM_PER_NODE MB"
echo ""

uv run python scripts/consolidate_zarr_metadata.py \
    --root /work/projects/imas_gpu/mast/level2/shots \
    --root /work/projects/imas_gpu/mast/level1/shots \
    --workers "$SLURM_CPUS_PER_TASK" --measure \
    --report /work/projects/imas_gpu/zarr-consolidate-report-${SLURM_JOB_ID}.json

echo "=== done  $(date +%H:%M:%S) ==="
