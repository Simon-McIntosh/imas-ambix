#!/bin/bash
#SBATCH --job-name=drain-ft
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --account=grpa
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --time=00:03:00
#SBATCH --output=/work/projects/imas_gpu/logs/drain-ft-%j.out
#SBATCH --error=/work/projects/imas_gpu/logs/drain-ft-%j.err
#
# Drain-study variant of the PRODUCTION finetune pipeline. Deliberately UNHARDENED:
# no AMBIX_STOP_FILE, no SIGUSR1 trap, no soft-time-limit. Short --time so the
# SLURM SIGTERM lands mid-real-work (cuDNN/NCCL/eval) and the SIGKILL at +KillWait
# leaves a heavy, real GPU teardown — the residue a rapid relaunch can wedge on
# (faithful to the 1208975 -> 1208980 single-user cascade). Real model
# (VGG/LPIPS/discriminator), real DataLoader/GPFS, 4-GPU DDP.
set -uo pipefail
export TMPDIR=/scratch_local/${SLURM_JOB_ID}; mkdir -p "$TMPDIR" || export TMPDIR=/tmp
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export NCCL_DEBUG=WARN NCCL_IB_DISABLE=1
export TORCH_HOME="${HOME}/.cache/torch"
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"; cd "${REPO_ROOT}"
export MAGVIT2_ROOT="/work/projects/imas_gpu/mast-tokens/v1/open-magvit2"
export FRAMES_ROOT="/work/projects/imas_gpu/mast-tokens/v1/frames"
PY="${MAGVIT2_ROOT}/.venv/bin/python"
echo "=== drain-ft === job=${SLURM_JOB_ID} node=${SLURMD_NODENAME} start=$(date -Is)"

TRAIN_SHOTS="$TMPDIR/train.txt"; VAL_SHOTS="$TMPDIR/val.txt"
"$PY" - <<'PYEOF'
import os, random
from pathlib import Path
fr = Path(os.environ["FRAMES_ROOT"]); td = os.environ["TMPDIR"]
sh = sorted(int(p.name) for p in fr.iterdir() if p.is_dir() and p.name.isdigit())
random.Random(42).shuffle(sh)
open(f"{td}/train.txt","w").write("\n".join(map(str,sh[:50]))+"\n")
open(f"{td}/val.txt","w").write("\n".join(map(str,sh[50:60]))+"\n")
print(f"[split] train=50 val=10")
PYEOF

PORT=$(( 29500 + (SLURM_JOB_ID % 200) ))
PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" \
  "${MAGVIT2_ROOT}/.venv/bin/torchrun" --nproc_per_node=4 \
    --master_addr=localhost --master_port=${PORT} \
    "${REPO_ROOT}/imas_ambix/tokenizer/finetune_decoder.py" \
    --train-shots "${TRAIN_SHOTS}" --val-shots "${VAL_SHOTS}" \
    --max-steps 10000 --batch-size 16 --eval-every 40 \
    --output-path "/scratch_local/${SLURM_JOB_ID}/drain-ft.safetensors" \
    --device cuda
echo "=== drain-ft exit $? at $(date -Is) ==="
