#!/bin/bash
#SBATCH --job-name=spine-labels
#SBATCH --partition=sun_debug
#SBATCH --cpus-per-task=9
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=_spine_label_fleet_%j.log
# Spine label factory fleet: shard the train manifest across parallel per-shot
# chains (each chain is sequential by design — warm-started along the time
# axis).  Usage:  sbatch scripts/spine_label_fleet.sh [I0] [I1] [OUT_DIR]
# shards manifest indices [I0, I1) over $CONC concurrent single-shot processes.
set -u
cd /home/ITER/mcintos/Code/imas-ambix
export TMPDIR=/tmp OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
source .venv/bin/activate

I0=${1:-0}
I1=${2:-100}
OUT=${3:-/work/projects/imas_gpu/mast/spine_labels}
CONC=${CONC:-8}
LOG=${LOG:-_spine_label_fleet_logs}
mkdir -p "$LOG" "$OUT"

SHOTS=$(python - "$I0" "$I1" <<'PY'
import sys
from imas_ambix.latent.data import read_split_shot_lists
i0, i1 = int(sys.argv[1]), int(sys.argv[2])
train, _ = read_split_shot_lists(max(i1, 200), 8)
print(" ".join(str(s) for s in train[i0:i1]))
PY
)

echo "labelling shots [$I0:$I1) -> $OUT ($CONC concurrent chains)"
n=0
for s in $SHOTS; do
  if [ -f "$OUT/shot_${s}.npz" ]; then
    echo "skip $s (shard exists)"
    continue
  fi
  python scripts/spine_label_factory.py --shots "$s" \
    --out-dir "$OUT" > "$LOG/shot_${s}.log" 2>&1 &
  n=$((n + 1))
  if [ $((n % CONC)) -eq 0 ]; then wait; fi
done
wait
TOTAL=$(python - "$OUT" <<'PY'
import json
import sys
from pathlib import Path
metas = [json.loads(p.read_text()) for p in Path(sys.argv[1]).glob("shot_*.json")]
n = sum(m["n_scored"] for m in metas)
wall = sum(m["wall_s"] for m in metas)
print(f"{len(metas)} shards, {n} slices, {wall:.0f} s chain time, "
      f"{n / max(wall, 1e-9):.2f} slices/s sequential")
PY
)
echo "FLEET_DONE: $TOTAL"
