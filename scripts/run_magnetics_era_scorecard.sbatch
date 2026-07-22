#!/bin/bash
#SBATCH --job-name=mag_scorecard
#SBATCH --partition=sun_debug
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home/ITER/mcintos/Code/imas-ambix/_mag_scorecard_%j.log

set -euo pipefail
cd /home/ITER/mcintos/Code/imas-ambix
export TMPDIR=/tmp
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

uv run python -m scripts.magnetics_era_scorecard \
    --cap-per-era 50 \
    --n-boot 300 \
    --seed 0 \
    --baseline-era M5 \
    --out-suffix cylv5
echo "SCORECARD_EXIT=$?"
