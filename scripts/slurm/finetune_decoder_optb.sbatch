#!/bin/bash
#SBATCH --job-name=ambix-finetune-decoder-optb
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --account=grpa
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=30
#SBATCH --mem=640G
#SBATCH --time=04:00:00
#SBATCH --output=/work/projects/imas_gpu/logs/ambix-%x-%j.out

# Option B (2026-05-28): alternate-loss fine-tune.
# Verdict bench 1209101 confirmed v0 fine-tune (perceptual_weight=0.1) made rFID
# WORSE on every class vs baseline (overall 26.22 vs 17.04).  PSNR went UP, rFID
# went DOWN — suggests pure-L1 + low-LPIPS training optimized pixel reconstruction
# but failed to match the InceptionV3-perceptual distribution the bench measures.
# This run tries perceptual_weight=1.0 (10× higher) — pushes training to favour
# perceptual similarity over pixel accuracy.  Output to a distinct path so it
# doesn't clobber the v0 weights.

set -euo pipefail

export TMPDIR=/scratch_local/$SLURM_JOB_ID
mkdir -p "$TMPDIR"
mkdir -p "/work/projects/imas_gpu/logs"

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export TORCHINDUCTOR_CACHE_DIR=$TMPDIR/.torch_inductor
mkdir -p "$TORCHINDUCTOR_CACHE_DIR"

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_ROOT}"

# Drain-forensics sidecar (docs/rca-node-drain-final-2026-06-03.html §5).
bash "${REPO_ROOT}/scripts/slurm/drain_sidecar.sh" &

export MAGVIT2_ROOT="/work/projects/imas_gpu/mast-tokens/v1/open-magvit2"
export FRAMES_ROOT="/work/projects/imas_gpu/mast-tokens/v1/frames"
MAGVIT2_PYTHON="${MAGVIT2_ROOT}/.venv/bin/python"

LOG_FILE="/work/projects/imas_gpu/logs/ambix-${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out"
echo "[$(date)] ambix-finetune-decoder-optb starting"
echo "  job-id:      ${SLURM_JOB_ID}"
echo "  node:        $(hostname)"
echo "  repo:        ${REPO_ROOT}"
echo "  log:         ${LOG_FILE}"
echo "  sidecar:     /work/projects/imas_gpu/logs/drain-sidecar-${SLURM_JOB_ID}.csv"
echo "  variant:     OPTION B (perceptual_weight=1.0, vs v0 default 0.1)"

nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

bash "${REPO_ROOT}/scripts/slurm/preempt_v4flash.sh"

# Same shot split as production (random-7500-2000, seed=42).
TRAIN_SHOTS="$TMPDIR/train_shots.txt"
VAL_SHOTS="$TMPDIR/val_shots.txt"

"${MAGVIT2_PYTHON}" - <<'PYEOF'
import os, random
from pathlib import Path

frames_dir = Path(os.environ["FRAMES_ROOT"])
tmpdir = os.environ["TMPDIR"]

all_shots = sorted(
    int(p.name) for p in frames_dir.iterdir()
    if p.is_dir() and p.name.isdigit()
)
rng = random.Random(42)
rng.shuffle(all_shots)
train = all_shots[:7500]
val = all_shots[7500:9500]

with open(f"{tmpdir}/train_shots.txt", "w") as f:
    f.write("\n".join(str(s) for s in train) + "\n")
with open(f"{tmpdir}/val_shots.txt", "w") as f:
    f.write("\n".join(str(s) for s in val) + "\n")

print(f"[split] train={len(train)} val={len(val)} total_available={len(all_shots)}")
PYEOF

# Distinct output path so we don't clobber v0 weights.
OUT_PATH="/work/projects/imas_gpu/mast-tokens/v1/open-magvit2/weights/plasma-decoder-v1-optb.safetensors"

export TORCH_HOME="${HOME}/.cache/torch"
TRAIN_LOG="/work/projects/imas_gpu/logs/ambix-finetune-decoder-optb-${SLURM_JOB_ID}.log"

PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" \
    "${MAGVIT2_ROOT}/.venv/bin/torchrun" \
    --nproc_per_node=4 \
    --master_addr=localhost \
    --master_port=29500 \
    "${REPO_ROOT}/imas_ambix/tokenizer/finetune_decoder.py" \
    --train-shots "${TRAIN_SHOTS}" \
    --val-shots   "${VAL_SHOTS}" \
    --max-steps   10000 \
    --batch-size  16 \
    --perceptual-weight 1.0 \
    --output-path "${OUT_PATH}" \
    --device      cuda \
    2>&1 | tee "${TRAIN_LOG}"

EXIT_CODE=${PIPESTATUS[0]}
echo "[$(date)] ambix-finetune-decoder-optb finished (exit=${EXIT_CODE})"

WEIGHTS="${OUT_PATH}"
# Trainer derives merged-ckpt name from OUT_PATH stem (post commit 966ae41 bug
# fix), so Option B's merged is plasma-decoder-v1-optb-merged.ckpt — distinct
# from v0's plasma-decoder-v1-merged.ckpt.  No clobber.
MERGED="$(dirname "${OUT_PATH}")/$(basename "${OUT_PATH}" .safetensors)-merged.ckpt"
if [[ -f "${WEIGHTS}" ]]; then
    echo "[verify] decoder weights: $(du -h "${WEIGHTS}" | cut -f1)  ${WEIGHTS}"
fi
if [[ -f "${MERGED}" ]]; then
    echo "[verify] merged ckpt:     $(du -h "${MERGED}" | cut -f1)  ${MERGED}"
else
    echo "[verify] WARNING: merged ckpt missing at ${MERGED}"
fi

exit "${EXIT_CODE}"
