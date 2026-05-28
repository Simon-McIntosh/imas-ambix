#!/bin/bash
#SBATCH --job-name=ambix-download-cosmos
#SBATCH --partition=sun_debug
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/work/projects/imas_gpu/logs/ambix-%x-%j.out

# Download NVIDIA Cosmos-Tokenizer DI16x16 from HuggingFace into GPFS.
# Standard compute nodes have outbound network + /work/projects/imas_gpu/
# write access; the betelgeuse GPU node has neither.

set -euo pipefail

export TMPDIR=/tmp
mkdir -p /work/projects/imas_gpu/logs

OUT="/work/projects/imas_gpu/mast-tokens/cosmos/v1/DI16x16"
mkdir -p "${OUT}"

VENV_PY="/work/projects/imas_gpu/mast-tokens/v1/open-magvit2/.venv/bin/python"

echo "[$(date)] downloading Cosmos-Tokenizer-DI16x16 to ${OUT}"

"${VENV_PY}" - <<PYEOF
from huggingface_hub import snapshot_download
import os
path = snapshot_download(
    repo_id="nvidia/Cosmos-Tokenizer-DI16x16",
    local_dir="${OUT}",
    allow_patterns=["*.jit", "*.json", "*.md", "README*"],
)
print(f"snapshot at: {path}")
for f in sorted(os.listdir(path)):
    fp = os.path.join(path, f)
    if os.path.isfile(fp):
        print(f"  {os.path.getsize(fp):>12,} B  {f}")
PYEOF

echo "[$(date)] done"
ls -la "${OUT}"
