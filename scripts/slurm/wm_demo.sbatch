#!/usr/bin/env bash
# Predict-vs-reality demo artifact bundle for the converged world model.
#
# Runs the two-phase builder (imas_ambix.worldmodel.demo_artifacts):
#   PHASE A — WM venv, GPU: load the converged checkpoint ONCE, roll out the
#             held-out shots, score skill vs persistence, dump the eval + GT
#             camera token bundles.
#   PHASE B — Open-MAGVIT2 venv, GPU: decode the GT camera token grids to
#             256x256 images (subprocess into the magvit2 venv).
#
# Requests ONLY gpu:1 so the neighbouring DeepSeek-V4-Flash server keeps its
# cards.  In-process (model loaded once, all shots in a loop), SIGTERM-clean,
# try/finally release + torch.cuda.empty_cache() — repo AGENTS.md §2b.
#
#SBATCH --job-name=wm-demo
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --account=grpa
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --output=/work/projects/imas_gpu/worldmodel/demo/wm_demo-%j.out
#SBATCH --error=/work/projects/imas_gpu/worldmodel/demo/wm_demo-%j.err

set -euo pipefail

REPO=/home/ITER/mcintos/Code/imas-ambix
cd "$REPO"

# Node-local scratch (the default TMPDIR is broken on betelgeuse).
export TMPDIR="/scratch_local/${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"

# Single-threaded BLAS + offline HF (betelgeuse has no outbound network).
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# Ensure the output dir exists before SLURM writes its logs there.
mkdir -p /work/projects/imas_gpu/worldmodel/demo

source "$REPO/.venv/bin/activate"

python -m imas_ambix.worldmodel.demo_artifacts "$@"
