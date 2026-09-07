#!/usr/bin/env bash
#SBATCH --job-name=ambix-full-window
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --account=grpa
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=00:20:00
#SBATCH --output=/home/ITER/mcintos/Code/.reckon-worktrees/imas-ambix-0405601943d2/ambix-s19-01/pcp-full-window-render/docs/figures/physics-carried-playable-plasma/seed-window/full-window-22086.slurm-%j.log

set -euo pipefail

export TMPDIR=/tmp
export PYTHONPATH=/home/ITER/mcintos/Code/.reckon-worktrees/imas-ambix-0405601943d2/ambix-s19-01/pcp-full-window-render
cd /work/projects/imas_gpu/mast-tokens/v1/open-magvit2
/home/ITER/mcintos/Code/imas-ambix/.venv/bin/python \
  /home/ITER/mcintos/Code/.reckon-worktrees/imas-ambix-0405601943d2/ambix-s19-01/pcp-full-window-render/docs/figures/physics-carried-playable-plasma/seed-window/full-window-render.py
