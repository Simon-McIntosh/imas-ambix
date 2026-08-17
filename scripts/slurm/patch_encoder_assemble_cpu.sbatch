#!/bin/bash
#SBATCH --job-name=ambix-patch-assemble
#SBATCH --partition=sun
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/work/projects/imas_gpu/logs/%x-%j.out
#
# ============================================================================
#  Corpus-cache assembly shard for the amortised patch-current encoder
#  trainer (scripts/train_patch_encoder.py --assemble-only --shot-shard).
# ============================================================================
#  Assembling the raw-magnetics token corpus is IMAS/geometry-IO-bound with NO
#  model on the GPU, so it belongs on the `sun` CPU partition, not the H200
#  reservation (mirrors scripts/slurm/l2_build_cpu.sbatch).  A full-corpus
#  assembly is sharded across N of these jobs; the trainer (or a later
#  --assemble-only call with no --shot-shard) merges every shard into the
#  final per-signature cache automatically the first time it is requested for
#  the same (N_TRAIN, nr, nz, t_steps, stride, min_ip) config — see
#  assemble_corpus_cached() in train_patch_encoder.py.
#
#  Usage (array job — $SLURM_ARRAY_TASK_ID supplies the shard index):
#      N_TRAIN=10846 N_SHARDS=32 \
#          sbatch --array=0-31 scripts/slurm/patch_encoder_assemble_cpu.sbatch
#
#  Or a single explicit shard:
#      N_TRAIN=5000 N_SHARDS=8 SHARD_ID=3 \
#          sbatch scripts/slurm/patch_encoder_assemble_cpu.sbatch
#
#  Set FORCE=1 to bypass an existing shard cache entry and reassemble — the
#  config hash only covers (N_TRAIN, nr, nz, t_steps, stride, min_ip,
#  COIL_MODEL_VERSION, GEOMETRY_TABLE_VERSION), so an internal assembly-logic
#  change with none of those bumped (e.g. widening the declared acquisition
#  scan) would otherwise silently serve a stale shard cache:
#      FORCE=1 N_TRAIN=10846 N_SHARDS=32 \
#          sbatch --array=0-31 scripts/slurm/patch_encoder_assemble_cpu.sbatch
#
#  Cache root: /work/projects/imas_gpu/latent/patch_encoder/corpus_cache/
# ============================================================================
set -uo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_ROOT" || { echo "[patch-assemble] cannot cd to REPO_ROOT=$REPO_ROOT" >&2; exit 2; }

export TMPDIR=/tmp
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
source .venv/bin/activate

N_TRAIN="${N_TRAIN:?set N_TRAIN (the full shot-list length this shard belongs to)}"
N_SHARDS="${N_SHARDS:?set N_SHARDS}"
SHARD_ID="${SLURM_ARRAY_TASK_ID:-${SHARD_ID:?set SHARD_ID or submit with --array}}"
FORCE_FLAG=()
[ "${FORCE:-0}" = "1" ] && FORCE_FLAG=(--force-rebuild-cache)

echo "[patch-assemble] node=$(hostname) n_train=${N_TRAIN} shard=${SHARD_ID}/${N_SHARDS} force=${FORCE:-0}"
python scripts/train_patch_encoder.py \
    --n-train "${N_TRAIN}" \
    --assemble-only \
    --shot-shard "${SHARD_ID}/${N_SHARDS}" \
    "${FORCE_FLAG[@]}"
RC=$?
echo "ASSEMBLE_EXIT=${RC}"
exit "${RC}"
