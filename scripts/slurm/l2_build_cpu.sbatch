#!/bin/bash
#SBATCH --job-name=l2-build-cpu
#SBATCH --partition=sun
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/work/projects/imas_gpu/logs/%x-%j.out
#
# Generic CPU-partition launcher for the Level-2 build drivers (the measured/
# planned input light path and the eval-only targets store).  Both are pure
# xarray/numpy GPFS IO with NO model load, so they belong on a CPU compute
# node, not the GPU reservation — running them there leaves all four H200s for
# encode/serve work and avoids queueing behind it.  Standard compute nodes
# (sun/vega/all) have /work/projects/imas_gpu access and the shared venv over
# NFS home.  Both drivers are resume-safe (skip-existing), so a re-run after a
# time-limit picks up where it left off.
#
# Usage (the module + its args are forwarded verbatim):
#   sbatch --job-name=l2-input-cpu scripts/slurm/l2_build_cpu.sbatch \
#       imas_ambix.data.l2_input_build --shots all \
#       --level2-dir /work/projects/imas_gpu/mast/level2/shots \
#       --manifest /work/projects/imas_gpu/mast-tokens/v2/_manifests/encode_l2_input.json
#   sbatch --job-name=l2-targets-cpu scripts/slurm/l2_build_cpu.sbatch \
#       imas_ambix.data.l2_targets_build --shots <shotlist-file> \
#       --manifest /work/projects/imas_gpu/mast-targets/_manifests/build.json
set -uo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_ROOT" || { echo "[l2-build-cpu] cannot cd to REPO_ROOT=$REPO_ROOT" >&2; exit 2; }

# CPU compute nodes: the login-shell TMPDIR (/run/user/$UID) is unwritable here;
# /tmp is correct (repo §2 — /scratch_local only exists on betelgeuse).
export TMPDIR=/tmp
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

MODULE="${1:?usage: l2_build_cpu.sbatch <python-module> [args...]}"
shift
PY="${REPO_ROOT}/.venv/bin/python"

echo "[l2-build-cpu] node=$(hostname) module=${MODULE} args=$*"
"$PY" -m "$MODULE" "$@"
RC=$?
echo "BUILD_EXIT=${RC}"
echo "DONE=1"
exit "$RC"
