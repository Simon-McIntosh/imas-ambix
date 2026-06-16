#!/bin/bash
#SBATCH --job-name=signal-tok-probe
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --account=grpa
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=160G
#SBATCH --time=00:30:00
#SBATCH --output=/work/projects/imas_gpu/logs/signal-tok-probe-%j.out
# Measure GPU utilisation during a small-batch HF-signal ENCODE to decide
# whether the full-corpus encode belongs on a GPU card (compute-bound) or on
# CPU compute nodes (IO/CPU-bound).  Encodes a few shots per modality on the
# GPU while nvidia-smi dmon samples util once/sec.  No persistent store side
# effects beyond a few extra shots in v2/signals_hf (skip-existing safe).
set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export TMPDIR=/scratch_local/${SLURM_JOB_ID}
mkdir -p "$TMPDIR"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

PY="${REPO_ROOT}/.venv/bin/python"
OUT_DIR="/work/projects/imas_gpu/mast-tokens/v2/_tokenizer"
PROBE_DIR="${TMPDIR}/probe"
mkdir -p "$PROBE_DIR"

# Which physical GPU did SLURM give us? dmon -i needs the index visible here.
GPU_IDX="$(nvidia-smi --query-gpu=index --format=csv,noheader | head -1)"
echo "[probe] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-?} sampled GPU idx=$GPU_IDX"

# A handful of shots that have all three groups.
PROBE_SHOTS="28000,28001,28002,28003,27000,27001"
XSX_TRAIN="30460,30461,30462,28000"
XSX_HOLD="30463,28003"

# xsx is new — decide + train it once so the probe can encode it.
if [[ ! -f "${OUT_DIR}/tokenizer_xsx.pt" ]]; then
    echo "[probe] === xsx decide+train (GPU) ===" | tee -a "${PROBE_DIR}/run.log"
    "$PY" -m imas_ambix.tokenizer.signal_hf_encode decide \
        --group xsx --device cuda --train "$XSX_TRAIN" --holdout "$XSX_HOLD" \
        --epochs 40 --out "${OUT_DIR}/decide_xsx.json" >>"${PROBE_DIR}/run.log" 2>&1
    echo "XSX_DECIDE_EXIT=$?" | tee -a "${PROBE_DIR}/run.log"
    XSX_WIN="$("$PY" -c "import json;print(json.load(open('${OUT_DIR}/decide_xsx.json'))['decision']['winner'])")"
    echo "[probe] xsx winner=$XSX_WIN" | tee -a "${PROBE_DIR}/run.log"
    "$PY" -m imas_ambix.tokenizer.signal_hf_encode train \
        --group xsx --device cuda --bottleneck "$XSX_WIN" --train "$XSX_TRAIN" \
        --epochs 40 --out "${OUT_DIR}/tokenizer_xsx.pt" >>"${PROBE_DIR}/run.log" 2>&1
    echo "XSX_TRAIN_EXIT=$?" | tee -a "${PROBE_DIR}/run.log"
fi

# Sample util once/sec ONLY during the encode phase (training is excluded so
# the measurement isolates encode-inference GPU util, which is the decision).
nvidia-smi dmon -s u -d 1 -o T > "${PROBE_DIR}/dmon.log" 2>&1 &
DMON_PID=$!

for G in xsx xim xma; do
    CKPT="${OUT_DIR}/tokenizer_${G}.pt"
    if [[ ! -f "$CKPT" ]]; then
        echo "[probe] no ckpt for $G — skip" ; continue
    fi
    echo "[probe] === encode $G (GPU) shots=$PROBE_SHOTS ===" | tee -a "${PROBE_DIR}/run.log"
    /usr/bin/time -v "$PY" -m imas_ambix.tokenizer.signal_hf_encode encode \
        --group "$G" --device cuda --shots "$PROBE_SHOTS" --ckpt "$CKPT" \
        --no-skip-existing --manifest "${PROBE_DIR}/manifest_${G}.json" \
        >>"${PROBE_DIR}/run.log" 2>&1
    echo "ENCODE_EXIT=$? group=$G" | tee -a "${PROBE_DIR}/run.log"
done

kill "$DMON_PID" 2>/dev/null || true
sleep 1

# Summarise GPU util for our card during the encode window.
echo "[probe] === GPU util summary (idx $GPU_IDX) ==="
"$PY" - "$GPU_IDX" "${PROBE_DIR}/dmon.log" <<'PY'
import sys
idx, path = sys.argv[1], sys.argv[2]
sm = []
for line in open(path):
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    parts = line.split()
    # dmon -s u -o T columns: time gpu sm mem enc dec ...
    if len(parts) < 4:
        continue
    try:
        g = parts[1]
        smv = float(parts[2])
    except (ValueError, IndexError):
        continue
    if g == idx:
        sm.append(smv)
if sm:
    import statistics
    sm.sort()
    n = len(sm)
    print(f"samples={n} sm_util mean={statistics.mean(sm):.1f}% "
          f"median={statistics.median(sm):.1f}% "
          f"p90={sm[int(0.9*n)-1]:.1f}% max={max(sm):.1f}%")
    nz = [v for v in sm if v > 0]
    print(f"nonzero samples={len(nz)} ({100*len(nz)/n:.0f}% of window) "
          f"mean(nonzero)={statistics.mean(nz):.1f}%" if nz else "no nonzero util")
else:
    print("no samples for our GPU idx — check dmon.log")
PY

cp -r "$PROBE_DIR" "/work/projects/imas_gpu/logs/signal-tok-probe-${SLURM_JOB_ID}" || true
echo "DONE=1"
