#!/bin/bash
#SBATCH --job-name=ambix-finetune-decoder
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --account=grpa
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=30
#SBATCH --mem=640G
#SBATCH --time=24:00:00
#SBATCH --output=/tmp/ambix-%x-%j.out
# NOTE: --qos is intentionally omitted — auto-applied for this account/reservation.
#       Training uses a single GPU (cuda:0) from the 4-GPU allocation; the
#       reservation keeps all 4 GPUs exclusive to prevent node drain from
#       concurrent CUDA processes (see AGENTS.md §2a).

# ── Environment ────────────────────────────────────────────────────────────
set -euo pipefail

export TMPDIR=/scratch_local/$SLURM_JOB_ID
mkdir -p "$TMPDIR"

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export TORCHINDUCTOR_CACHE_DIR=$TMPDIR/.torch_inductor
mkdir -p "$TORCHINDUCTOR_CACHE_DIR"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export MAGVIT2_ROOT="/work/projects/imas_gpu/mast-tokens/v1/open-magvit2"
export FRAMES_ROOT="/work/projects/imas_gpu/mast-tokens/v1/frames"
MAGVIT2_PYTHON="${MAGVIT2_ROOT}/.venv/bin/python"

echo "[$(date)] ambix-finetune-decoder starting"
echo "  job-id:      ${SLURM_JOB_ID}"
echo "  node:        $(hostname)"
echo "  repo:        ${REPO_ROOT}"
echo "  log:         /tmp/ambix-${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out"

nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# ── Preempt V4-Flash ───────────────────────────────────────────────────────
bash "${REPO_ROOT}/scripts/slurm/preempt_v4flash.sh"

# ── Generate train / val shot ID files ────────────────────────────────────
# 7,500 train + 2,000 val randomly sampled from the 9,527 tokenised rbb shots
# (locked decision: train-val-split = random-7500-2000, seed=42).
TRAIN_SHOTS="$TMPDIR/train_shots.txt"
VAL_SHOTS="$TMPDIR/val_shots.txt"

python3 - <<'PYEOF'
import os, json, random
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
val   = all_shots[7500:9500]   # 2,000 val; remaining ~27 are discarded

with open(f"{tmpdir}/train_shots.txt", "w") as f:
    f.write("\n".join(str(s) for s in train) + "\n")

with open(f"{tmpdir}/val_shots.txt", "w") as f:
    f.write("\n".join(str(s) for s in val) + "\n")

print(f"[split] train={len(train)} val={len(val)} total_available={len(all_shots)}")
PYEOF

echo "[$(date)] shot split written: train=$(wc -l < "${TRAIN_SHOTS}") val=$(wc -l < "${VAL_SHOTS}")"

# ── Optional: pre-load VGG16 weights for perceptual loss ──────────────────
# The GPU node (betelgeuse) has NO outbound network.  If VGG16 weights are
# already cached under TORCH_HOME, they will load; otherwise compute_loss
# falls back gracefully to L1-only loss with a one-time warning.
# To pre-cache on a network-accessible node before submitting:
#
#   srun --partition=sun_debug --time=00:10:00 \
#     python3 -c "import torchvision.models as tvm; tvm.vgg16(weights=tvm.VGG16_Weights.IMAGENET1K_V1)"
#
export TORCH_HOME="${HOME}/.cache/torch"

LOG_FILE="/tmp/ambix-finetune-decoder-${SLURM_JOB_ID}.log"

# ── Launch fine-tune ───────────────────────────────────────────────────────
# Run inside the Open-MAGVIT2 venv (torch 2.1.1 + lightning 2.2.0).
# PYTHONPATH includes the ambix repo so imas_ambix is importable for
# the rfid metric; Open-MAGVIT2 src is added to sys.path inside build_model.
PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" \
    "${MAGVIT2_PYTHON}" \
    "${REPO_ROOT}/imas_ambix/tokenizer/finetune_decoder.py" \
    --train-shots "${TRAIN_SHOTS}" \
    --val-shots   "${VAL_SHOTS}" \
    --max-steps   10000 \
    --batch-size  64 \
    --device      cuda \
    2>&1 | tee "${LOG_FILE}"

EXIT_CODE=${PIPESTATUS[0]}
echo "[$(date)] ambix-finetune-decoder finished (exit=${EXIT_CODE})"

# ── Verify output ──────────────────────────────────────────────────────────
WEIGHTS="${MAGVIT2_ROOT}/weights/plasma-decoder-v1.safetensors"
MERGED="${MAGVIT2_ROOT}/weights/plasma-decoder-v1-merged.ckpt"
if [[ -f "${WEIGHTS}" ]]; then
    echo "[verify] decoder weights: $(du -h "${WEIGHTS}" | cut -f1)  ${WEIGHTS}"
else
    echo "[verify] WARNING: ${WEIGHTS} not found — training may have failed"
fi
if [[ -f "${MERGED}" ]]; then
    echo "[verify] merged ckpt:     $(du -h "${MERGED}" | cut -f1)  ${MERGED}"
else
    echo "[verify] WARNING: ${MERGED} not found"
fi

exit "${EXIT_CODE}"
