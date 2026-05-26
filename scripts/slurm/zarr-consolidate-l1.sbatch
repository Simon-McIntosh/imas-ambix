#!/bin/bash
#SBATCH --job-name=zarr-consol-l1
#SBATCH --partition=sun
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/work/projects/imas_gpu/zarr-consolidate-l1-%j.log
#SBATCH --error=/work/projects/imas_gpu/zarr-consolidate-l1-%j.log

set -eu
export TMPDIR=/tmp
cd /home/ITER/mcintos/Code/imas-ambix
echo "=== L1 consolidate on $(hostname)  $(date +%H:%M:%S)  ${SLURM_CPUS_PER_TASK} cores ==="
uv run python scripts/consolidate_zarr_metadata.py \
    --root /work/projects/imas_gpu/mast/level1/shots \
    --workers "$SLURM_CPUS_PER_TASK" --measure \
    --report /work/projects/imas_gpu/zarr-consolidate-l1-report-${SLURM_JOB_ID}.json
echo "=== done  $(date +%H:%M:%S) ==="
