#!/bin/bash
# Full four-pass engine cross-validation arm on the census harness.
# Usage: sbatch full_engine_crossval.sbatch <arm> <cap> <class...>
#   arm    ∈ plain | nobasin | eddy
#   cap    per-class shot cap (0 = full stratified draw, ~60/class)
#   class  one or more of: limited sn-lower sn-upper connected-dn marginal-dn
#SBATCH --job-name=topo_full
#SBATCH --partition=sun_debug
#SBATCH --cpus-per-task=8
#SBATCH --mem=28G
#SBATCH --time=01:00:00
#SBATCH --output=/home/ITER/mcintos/Code/imas-ambix/output/topology_census/full_%x_%j.log
set -uo pipefail
ARM="${1:?arm}"; shift
CAP="${1:?cap}"; shift
CLASSES="$*"
export TMPDIR=/tmp OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu
cd /home/ITER/mcintos/Code/imas-ambix
CAP_ARG=""
[ "$CAP" != "0" ] && CAP_ARG="--per-class $CAP"
uv run python -m scripts.topology_engine_crossval \
  --chain full --arm "$ARM" --classes $CLASSES --workers 8 $CAP_ARG
echo "FULL_EXIT=$? arm=$ARM cap=$CAP classes=$CLASSES"
