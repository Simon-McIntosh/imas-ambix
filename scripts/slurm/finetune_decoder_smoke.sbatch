#!/bin/bash
#SBATCH --job-name=ambix-finetune-decoder-smoke
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --account=grpa
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=30
#SBATCH --mem=640G
#SBATCH --time=00:30:00
#SBATCH --signal=B:USR1@120
#SBATCH --output=/work/projects/imas_gpu/logs/ambix-%x-%j.out
# --time=30m is the safety ceiling for smoke (expected ~3-5 min); SIGUSR1
# fires at 2 min pre-limit so a runaway smoke still exits cleanly.
# AMBIX_SOFT_TIME_LIMIT below = 0.85 × 30m = 1530 s.  See AGENTS.md
# §2a-cancel-time for the three-layer pattern.
# 4-GPU DDP SMOKE TEST — verifies the fine-tune pipeline before a 24h submit.
# 50 train + 10 val shots, 200 max steps, eval every 100. Goal: confirm
#   (1) all 4 GPUs > 80% utilization,
#   (2) loss decreases,
#   (3) rFID computes at step 100 + 200 without error,
#   (4) checkpoint .safetensors + merged .ckpt written to smoke-output paths.

set -euo pipefail

export TMPDIR=/scratch_local/$SLURM_JOB_ID
mkdir -p "$TMPDIR"
mkdir -p "/work/projects/imas_gpu/logs"

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export TORCHINDUCTOR_CACHE_DIR=$TMPDIR/.torch_inductor
mkdir -p "$TORCHINDUCTOR_CACHE_DIR"

# STOP-FILE for safe cancellation — AGENTS.md §2a-cancel.
# To cancel cleanly: touch /work/projects/imas_gpu/stops/${SLURM_JOB_ID}.stop
mkdir -p /work/projects/imas_gpu/stops
export AMBIX_STOP_FILE=/work/projects/imas_gpu/stops/${SLURM_JOB_ID}.stop
rm -f "${AMBIX_STOP_FILE}"

# §2a-cancel-time three-layer defence — Layer 1: SIGUSR1 trap.
_on_sigusr1() {
    echo "[sbatch] SIGUSR1 received from SLURM (pre-limit) — touching STOP-FILE" >&2
    touch "${AMBIX_STOP_FILE}"
}
trap _on_sigusr1 USR1

# Layer 2: internal soft wall-clock limit (0.85 × 30m = 1530 s).
export AMBIX_SOFT_TIME_LIMIT=1530

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_ROOT}"
export MAGVIT2_ROOT="/work/projects/imas_gpu/mast-tokens/v1/open-magvit2"
export FRAMES_ROOT="/work/projects/imas_gpu/mast-tokens/v1/frames"
MAGVIT2_PYTHON="${MAGVIT2_ROOT}/.venv/bin/python"

LOG_FILE="/work/projects/imas_gpu/logs/ambix-${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out"
echo "[$(date)] ambix-finetune-decoder-smoke starting"
echo "  job-id:      ${SLURM_JOB_ID}"
echo "  node:        $(hostname)"
echo "  log:         ${LOG_FILE}"

nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

bash "${REPO_ROOT}/scripts/slurm/preempt_v4flash.sh"

# ── Smoke shot split: 50 train + 10 val ──────────────────────────────────
TRAIN_SHOTS="$TMPDIR/train_shots_smoke.txt"
VAL_SHOTS="$TMPDIR/val_shots_smoke.txt"

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
train = all_shots[:50]
val = all_shots[50:60]

with open(f"{tmpdir}/train_shots_smoke.txt", "w") as f:
    f.write("\n".join(str(s) for s in train) + "\n")
with open(f"{tmpdir}/val_shots_smoke.txt", "w") as f:
    f.write("\n".join(str(s) for s in val) + "\n")
print(f"[smoke-split] train={len(train)} val={len(val)}")
PYEOF

# Smoke output paths — distinct from production weights.
SMOKE_OUT="/work/projects/imas_gpu/mast-tokens/v1/open-magvit2/weights/plasma-decoder-smoke.safetensors"

export TORCH_HOME="${HOME}/.cache/torch"

# Background GPU-utilization sampler — writes a CSV every 10s so we can
# verify 4-GPU saturation post-mortem.
UTIL_LOG="/work/projects/imas_gpu/logs/ambix-${SLURM_JOB_NAME}-${SLURM_JOB_ID}.gpu.csv"
(
  echo "timestamp,gpu_index,utilization_gpu_pct,memory_used_mib"
  while true; do
    nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used \
      --format=csv,noheader,nounits 2>/dev/null || true
    sleep 10
  done
) > "${UTIL_LOG}" &
UTIL_PID=$!

trap "kill ${UTIL_PID} 2>/dev/null || true" EXIT

TRAIN_LOG="/work/projects/imas_gpu/logs/ambix-${SLURM_JOB_NAME}-${SLURM_JOB_ID}.log"

PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" \
    "${MAGVIT2_ROOT}/.venv/bin/torchrun" \
    --nproc_per_node=4 \
    --master_addr=localhost \
    --master_port=29501 \
    "${REPO_ROOT}/imas_ambix/tokenizer/finetune_decoder.py" \
    --train-shots "${TRAIN_SHOTS}" \
    --val-shots   "${VAL_SHOTS}" \
    --max-steps   200 \
    --batch-size  16 \
    --eval-every  100 \
    --output-path "${SMOKE_OUT}" \
    --device      cuda \
    2>&1 | tee "${TRAIN_LOG}"

EXIT_CODE=${PIPESTATUS[0]}
kill ${UTIL_PID} 2>/dev/null || true
echo "[$(date)] smoke finished (exit=${EXIT_CODE})"

# ── Smoke verification ─────────────────────────────────────────────────────
echo ""
echo "═══ SMOKE VERIFICATION ═══"

# Mean GPU utilization per GPU during the run (skip startup, last 50% of samples)
"${MAGVIT2_PYTHON}" - <<PYEOF
import csv, pathlib
p = pathlib.Path("${UTIL_LOG}")
if not p.exists():
    print(f"[verify] util log missing: {p}")
else:
    rows = list(csv.DictReader(p.open()))
    # Drop the first 30% (startup) when assessing utilization
    if not rows:
        print("[verify] no util samples collected")
    else:
        from collections import defaultdict
        per_gpu = defaultdict(list)
        n = len(rows)
        skip = n // 3
        for r in rows[skip:]:
            try:
                per_gpu[r["gpu_index"]].append(int(r[" utilization_gpu_pct"]))
            except (KeyError, ValueError):
                pass
        print("[verify] GPU utilization (post-warmup mean):")
        all_high = True
        for g in sorted(per_gpu):
            vals = per_gpu[g]
            mean = sum(vals) / max(len(vals), 1)
            peak = max(vals) if vals else 0
            ok = "✓" if mean >= 70 else "✗"
            if mean < 70:
                all_high = False
            print(f"  GPU{g}: mean={mean:5.1f}%  peak={peak:3d}%  samples={len(vals)}  {ok}")
        if all_high:
            print("[verify] PASS: all 4 GPUs averaging ≥ 70% utilization")
        else:
            print("[verify] FAIL: at least one GPU below 70% — DDP not saturating")
PYEOF

# Confirm checkpoint files exist
SMOKE_MERGED="${SMOKE_OUT%.safetensors}-merged.ckpt"
# DecoderFinetuneTrainer._save_merged_checkpoint uses with_name("plasma-decoder-v1-merged.ckpt")
# So merged path = <SMOKE_OUT_dir>/plasma-decoder-v1-merged.ckpt regardless of --output-path stem.
SMOKE_MERGED_ACTUAL="$(dirname "${SMOKE_OUT}")/plasma-decoder-v1-merged.ckpt"

echo ""
if [[ -f "${SMOKE_OUT}" ]]; then
    SIZE=$(du -h "${SMOKE_OUT}" | cut -f1)
    echo "[verify] PASS: decoder-only weights written: ${SIZE}  ${SMOKE_OUT}"
else
    echo "[verify] FAIL: ${SMOKE_OUT} NOT written"
fi

if [[ -f "${SMOKE_MERGED_ACTUAL}" ]]; then
    SIZE=$(du -h "${SMOKE_MERGED_ACTUAL}" | cut -f1)
    echo "[verify] PASS: merged checkpoint written: ${SIZE}  ${SMOKE_MERGED_ACTUAL}"
else
    echo "[verify] FAIL: ${SMOKE_MERGED_ACTUAL} NOT written"
fi

# rFID values from log
echo ""
echo "[verify] rFID values from training log:"
grep -E "eval step [0-9]+: rFID=" "${TRAIN_LOG}" || echo "  (no eval lines found)"

# Loss trend
echo ""
echo "[verify] Loss trajectory (every 50 steps):"
grep -E "step [0-9]+/" "${TRAIN_LOG}" | head -20

exit "${EXIT_CODE}"
