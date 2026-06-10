#!/bin/bash
#SBATCH --job-name=ambix-bench-cosmos
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --account=grpa
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=30
#SBATCH --mem=640G
#SBATCH --time=01:00:00
#SBATCH --output=/work/projects/imas_gpu/logs/ambix-%x-%j.out

# Zero-shot Cosmos-Tokenizer DI16x16 bench on rbb 100-shot config.
# Apples-to-apples vs 1209100 (baseline imagenet decoder) and 1209101
# (C5+C6+C7 fine-tuned Open-MAGVIT2 decoder).  Same shots, same C8 frame
# selection (uniform stride), same rFID compute path.

set -euo pipefail

export TMPDIR=/scratch_local/$SLURM_JOB_ID
mkdir -p "$TMPDIR"
mkdir -p "/work/projects/imas_gpu/logs"

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_ROOT}"
mkdir -p "${REPO_ROOT}/imas_ambix/bench/results"

# Drain-forensics sidecar (docs/rca-node-drain-final-2026-06-03.html §5).
bash "${REPO_ROOT}/scripts/slurm/drain_sidecar.sh" &

VENV_PY="/work/projects/imas_gpu/mast-tokens/v1/open-magvit2/.venv/bin/python"

LOG_FILE="/work/projects/imas_gpu/logs/ambix-${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out"
RESULT="${REPO_ROOT}/imas_ambix/bench/results/v0-rbb-100shot-cosmos-${SLURM_JOB_ID}.json"

echo "[$(date)] cosmos-bench starting"
echo "  job-id:  ${SLURM_JOB_ID}"
echo "  node:    $(hostname)"
echo "  result:  ${RESULT}"

nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" \
    "${VENV_PY}" "${REPO_ROOT}/scripts/bench_cosmos_zero_shot.py" \
    --config imas_ambix/bench/configs/v0-rbb-100shot.yaml \
    --output "${RESULT}" \
    --device cuda \
    2>&1 | tee "${LOG_FILE}.detail"

EXIT_CODE=${PIPESTATUS[0]}
echo "[$(date)] cosmos-bench finished (exit=${EXIT_CODE})"
exit "${EXIT_CODE}"
