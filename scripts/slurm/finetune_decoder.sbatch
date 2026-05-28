#!/bin/bash
#SBATCH --job-name=ambix-finetune-decoder
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --account=grpa
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=30
#SBATCH --mem=640G
#SBATCH --time=04:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --output=/work/projects/imas_gpu/logs/ambix-%x-%j.out
# NOTE: --qos is intentionally omitted — auto-applied for this account/reservation.
#       --time is the SAFETY CEILING — see AGENTS.md §2a-cancel-time.  Expected
#       runtime is ~1.5 h; the 4 h ceiling is 2-3× margin.  When the SLURM
#       limit nears, the --signal=B:USR1@600 below sends SIGUSR1 to this batch
#       script 600 s before the limit; our USR1 trap touches the STOP-FILE so
#       the workers exit cleanly via the §2a-cancel mechanism — no SIGTERM-mid-
#       NCCL drain risk.  Belt+braces with AMBIX_SOFT_TIME_LIMIT below.
#       Log written to GPFS so it is readable from the login node after the job.

# ── Environment ────────────────────────────────────────────────────────────
set -euo pipefail

export TMPDIR=/scratch_local/$SLURM_JOB_ID
mkdir -p "$TMPDIR"
mkdir -p "/work/projects/imas_gpu/logs"

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export TORCHINDUCTOR_CACHE_DIR=$TMPDIR/.torch_inductor
mkdir -p "$TORCHINDUCTOR_CACHE_DIR"

# STOP-FILE for safe cancellation — see AGENTS.md §2a-cancel.
# To cancel this job cleanly (no drain risk):
#     touch /work/projects/imas_gpu/stops/${SLURM_JOB_ID}.stop
# NEVER `scancel` a CUDA-DDP job — it can leave processes in D-state and
# trigger a node drain.  The 2026-05-28 drain (3rd this week) was caused by
# exactly that mistake.
mkdir -p /work/projects/imas_gpu/stops
export AMBIX_STOP_FILE=/work/projects/imas_gpu/stops/${SLURM_JOB_ID}.stop
# Ensure no stale stop-file lingers from a previous job with the same ID.
rm -f "${AMBIX_STOP_FILE}"

# Layer 1 of the §2a-cancel-time three-layer defence: SIGUSR1 trap that
# touches the STOP-FILE.  Fires from --signal=B:USR1@600 above, 600 s before
# the --time ceiling.  Bash signal handling is reliable (no NCCL involvement)
# so the trap touches the file in milliseconds, the workers see it at their
# next step boundary, and the job exits cleanly with code 0 → no drain risk.
_on_sigusr1() {
    echo "[sbatch] SIGUSR1 received from SLURM (10 min pre-limit) — touching STOP-FILE for graceful shutdown" >&2
    touch "${AMBIX_STOP_FILE}"
}
trap _on_sigusr1 USR1

# Layer 2: internal soft wall-clock limit consumed by the training script.
# Set to ~85% of the --time ceiling so it fires BEFORE the SLURM SIGUSR1.
# 4h ceiling × 0.85 = 3h24m = 12240 s.
export AMBIX_SOFT_TIME_LIMIT=12240

# Required for torch.use_deterministic_algorithms with CuBLAS on CUDA ≥ 10.2
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# NCCL tuning for NVLink topology on 98dci4-gpu-0003
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1   # no InfiniBand on this node; use NVLink/PCIe

# SLURM copies the script to /var/spool before execution — BASH_SOURCE[0]
# points there, not the repo. Use SLURM_SUBMIT_DIR (the directory sbatch was
# invoked from) to locate the repo (same pattern as bench_rbb.sbatch).
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_ROOT}"
export MAGVIT2_ROOT="/work/projects/imas_gpu/mast-tokens/v1/open-magvit2"
export FRAMES_ROOT="/work/projects/imas_gpu/mast-tokens/v1/frames"
MAGVIT2_PYTHON="${MAGVIT2_ROOT}/.venv/bin/python"

LOG_FILE="/work/projects/imas_gpu/logs/ambix-${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out"
echo "[$(date)] ambix-finetune-decoder starting"
echo "  job-id:      ${SLURM_JOB_ID}"
echo "  node:        $(hostname)"
echo "  repo:        ${REPO_ROOT}"
echo "  log:         ${LOG_FILE}"
echo "  stop-file:   ${AMBIX_STOP_FILE}  (touch to cancel cleanly — DO NOT scancel)"

nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# ── Preempt V4-Flash ───────────────────────────────────────────────────────
bash "${REPO_ROOT}/scripts/slurm/preempt_v4flash.sh"

# ── Generate train / val shot ID files ────────────────────────────────────
# 7,500 train + 2,000 val randomly sampled from the 9,527 tokenised rbb shots
# (locked decision: train-val-split = random-7500-2000, seed=42).
TRAIN_SHOTS="$TMPDIR/train_shots.txt"
VAL_SHOTS="$TMPDIR/val_shots.txt"

"${MAGVIT2_PYTHON}" - <<'PYEOF'
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

TRAIN_LOG="/work/projects/imas_gpu/logs/ambix-finetune-decoder-${SLURM_JOB_ID}.log"

# ── Launch fine-tune ───────────────────────────────────────────────────────
# Run inside the Open-MAGVIT2 venv (torch 2.1.1 + lightning 2.2.0).
# PYTHONPATH includes the ambix repo so imas_ambix is importable for
# the rfid metric; Open-MAGVIT2 src is added to sys.path inside build_model.
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
    --device      cuda \
    2>&1 | tee "${TRAIN_LOG}"

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
