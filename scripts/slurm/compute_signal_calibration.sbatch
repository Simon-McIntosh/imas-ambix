#!/bin/bash
#SBATCH --job-name=signal-calib
#SBATCH --partition=sun
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/work/projects/imas_gpu/logs/%x-%j.out
#
# Compute corpus-wide (absolute / SI) per-channel signal calibration and
# persist it to the v2 calibration namespace.  This is the foundational fix
# for the absolute-magnitude blocker: with corpus mean/std, a given physical
# value maps to the SAME token in every shot and on every machine, so the
# signal tokenizers stop destroying absolute magnitude with per-shot z-scoring.
#
# Pure xarray/numpy GPFS IO with NO model load → CPU compute node, not the GPU
# reservation (repo: IO-bound builds go to the `sun` cluster, ~19 idle nodes
# with /work access).  Resume-safe: a group whose JSON already exists is
# skipped unless --overwrite.  Shardable for fan-out across CPU nodes.
#
# Usage (args forwarded verbatim to corpus_compute):
#   # one group, whole corpus, single node:
#   sbatch scripts/slurm/compute_signal_calibration.sbatch --group xma
#   # every known signal group:
#   sbatch scripts/slurm/compute_signal_calibration.sbatch --group all
#   # fan one group across 8 CPU nodes (each writes a shard file):
#   for i in $(seq 0 7); do
#     sbatch scripts/slurm/compute_signal_calibration.sbatch \
#         --group xma --shard ${i}/8
#   done
#   # then reduce the shards into the final {group}.json (single node):
#   sbatch scripts/slurm/compute_signal_calibration.sbatch --group xma --reduce
set -uo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_ROOT" || {
    echo "[signal-calib] cannot cd to REPO_ROOT=$REPO_ROOT" >&2
    exit 2
}

# CPU compute nodes: the login-shell TMPDIR (/run/user/$UID) is unwritable here;
# /tmp is correct (repo: /scratch_local only exists on betelgeuse).
export TMPDIR=/tmp
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

PY="${REPO_ROOT}/.venv/bin/python"

echo "[signal-calib] node=$(hostname) args=$*"
"$PY" -m imas_ambix.calibration.corpus_compute "$@"
RC=$?
echo "CALIB_EXIT=${RC}"
echo "DONE=1"
exit "$RC"
