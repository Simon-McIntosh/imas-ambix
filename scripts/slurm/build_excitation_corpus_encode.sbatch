#!/bin/bash
#SBATCH --job-name=ambix-excit-encode
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --account=grpa
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=7
#SBATCH --mem=160G
#SBATCH --time=02:00:00
#SBATCH --output=/work/projects/imas_gpu/excit-encode-%j.out
#
# GATED exposure-balanced re-encode of the curated dynamic-excitation windows.
# This is the multi-hour GPU step — launch ONLY after the lead's go.  For a
# curated subset the manifest's compute estimate is well under 1 GPU-hour, so
# the 2 h wall is generous; the job is in-process + SIGTERM-clean so a scancel
# frees the card in seconds.
#
# Does NOT pass --qos -> defaults to QOS normal -> no 4-GPU cap, 1 card bursts
# onto any free silicon and coexists with the DeepSeek serve (AGENTS.md §2).
#
# Usage (only after the lead's go):
#   # full curated manifest (full-corpus select output):
#   sbatch --export=ALL,MANIFEST=/work/projects/imas_gpu/agents/excitation-corpus/curated_windows_full.json \
#          scripts/slurm/build_excitation_corpus_encode.sbatch
#   # default PoC manifest:
#   sbatch scripts/slurm/build_excitation_corpus_encode.sbatch
#   # PoC dry-run (a handful of shots):
#   sbatch --export=ALL,MAX_SHOTS=8 scripts/slurm/build_excitation_corpus_encode.sbatch
# Env knobs: MANIFEST, CURATED_ROOT, MAX_SHOTS, EXPOSURE (percentile|global).

set -e
REPO=/home/ITER/mcintos/Code/imas-ambix
MAGVIT2_ROOT=/work/projects/imas_gpu/mast-tokens/v1/open-magvit2

# CWD MUST be the magvit2 root: VQModel.__init__ loads vgg.pth via a
# CWD-relative path, and betelgeuse has NO outbound network — a wrong CWD makes
# the loader try to DOWNLOAD weights and stall (incident 2026-05-27).
cd "$MAGVIT2_ROOT"

# Drain-forensics sidecar (settled-RCA instrumentation, AGENTS.md §2a).
bash "${REPO}/scripts/slurm/drain_sidecar.sh" &

export TMPDIR=/scratch_local/$SLURM_JOB_ID
mkdir -p "$TMPDIR"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO:$PYTHONPATH"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

# Run inside the Open-MAGVIT2 venv (torch 2.5.1+cu124) — same as production encode.
VENV=/work/projects/imas_gpu/mast-tokens/v1/open-magvit2/.venv
ARGS=""
if [ -n "${MANIFEST:-}" ]; then ARGS="${ARGS} --manifest ${MANIFEST}"; fi
if [ -n "${CURATED_ROOT:-}" ]; then ARGS="${ARGS} --curated-root ${CURATED_ROOT}"; fi
if [ -n "${MAX_SHOTS:-}" ]; then ARGS="${ARGS} --max-shots ${MAX_SHOTS}"; fi
if [ -n "${EXPOSURE:-}" ]; then ARGS="${ARGS} --exposure ${EXPOSURE}"; fi

"${VENV}/bin/python" "${REPO}/scripts/slurm/build_excitation_corpus_encode.py" ${ARGS}
echo "EXCIT_ENCODE_EXIT=$?"
