#!/bin/bash
#SBATCH --job-name=ambix-score-corruption
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --account=grpa
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --output=/work/projects/imas_gpu/logs/ambix-%x-%j.out
# NOTE: --qos omitted -> normal. Single GPU; coexists with everything.
#
# ============================================================================
#  Score a context-corruption fine-tune checkpoint vs persistence AND the M1
#  sampling baseline, at a FIXED (T, p), on the held-out shots (incl. bright).
# ============================================================================
#  Reuses the M1 playability harness (sampling_rescore -> playability_metrics:
#  ensemble CRPS, motion/non-collapse, SSIM sanity).  The model is loaded ONCE
#  (AGENTS.md §2b).  The fine-tune rolls out at the CLEAN inference default
#  (corruption level 0 — the rate-0 row it was trained on), so no extra flag is
#  needed: the grid already covers T=0.9, p=0.95.
#
#  Two horizons:
#   * MATCHED (n_frames=24, context=8 -> 16 forecast frames) — apples-to-apples
#     with the M1 baseline verdict.json.
#   * LONGER  (n_frames set by WM_LONG_FRAMES <= 26, same context) — tests whether
#     the anti-drift fine-tune holds coherence over a LONGER rollout (the M2
#     claim).  26 is the max the fixed max_frames=58 position table allows after
#     the 32-frame conditioning prefix.
#
#  Compare the emitted verdicts against the M1 baseline:
#    /work/projects/imas_gpu/worldmodel/ckpt/spacetime-v2-1219852/sampling_rescore/verdict.json
# ============================================================================

set -uo pipefail

export TMPDIR="/scratch_local/${SLURM_JOB_ID}"
mkdir -p "${TMPDIR}" 2>/dev/null || export TMPDIR=/tmp
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/.venv/bin/activate"

# REQUIRED: point at the fine-tune checkpoint to score (best.pt or latest.pt).
: "${WM_CKPT:?set WM_CKPT to the checkpoint .pt to score}"
: "${WM_SHOTS:=18502,18503,18504,18505}"   # held-out (bright: 18502, 18505)
: "${WM_N_PLAN:=8}"
: "${WM_N_SIGNAL_STEPS:=4}"
: "${WM_CONTEXT_FRAMES:=8}"
: "${WM_ENSEMBLE:=4}"
: "${WM_CHUNK:=8192}"
# Longer-horizon rollout to expose drift.  The model's frame-position embedding
# is fixed at max_frames=58 (inherited from the pre-train checkpoint), and the
# conditioning prefix is 6 streams x 4 steps + 8 plan = 32 frames, so the camera
# rollout can be at most 58 - 32 = 26 frames.  26 (vs the matched 24) is the
# longest horizon this architecture's position table allows.
: "${WM_LONG_FRAMES:=26}"
: "${WM_OUT_DIR:=$(dirname "${WM_CKPT}")/corruption_rescore}"
: "${TOKEN_ROOT:=/work/projects/imas_gpu/mast-tokens}"
mkdir -p "${WM_OUT_DIR}"

echo "[$(date)] scoring ${WM_CKPT}"
nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader | head -1

# ── MATCHED horizon (24 frames / 8 context = 16 forecast) — vs M1 baseline ──
echo "=== MATCHED horizon (n_frames=24 context=${WM_CONTEXT_FRAMES}) ==="
python -m imas_ambix.worldmodel.sampling_rescore \
    --checkpoint "${WM_CKPT}" \
    --shots "${WM_SHOTS}" \
    --out-json "${WM_OUT_DIR}/verdict_h16.json" \
    --n-frames 24 \
    --n-plan "${WM_N_PLAN}" \
    --context-frames "${WM_CONTEXT_FRAMES}" \
    --n-signal-steps "${WM_N_SIGNAL_STEPS}" \
    --ensemble "${WM_ENSEMBLE}" \
    --chunk "${WM_CHUNK}" \
    --token-root "${TOKEN_ROOT}" \
    --work-dir "${TMPDIR}/h16"
echo "MATCHED_EXIT=$?"

# ── LONGER horizon — does the anti-drift fine-tune hold coherence longer? ──
echo "=== LONGER horizon (n_frames=${WM_LONG_FRAMES} context=${WM_CONTEXT_FRAMES}) ==="
python -m imas_ambix.worldmodel.sampling_rescore \
    --checkpoint "${WM_CKPT}" \
    --shots "${WM_SHOTS}" \
    --out-json "${WM_OUT_DIR}/verdict_h${WM_LONG_FRAMES}.json" \
    --n-frames "${WM_LONG_FRAMES}" \
    --n-plan "${WM_N_PLAN}" \
    --context-frames "${WM_CONTEXT_FRAMES}" \
    --n-signal-steps "${WM_N_SIGNAL_STEPS}" \
    --ensemble "${WM_ENSEMBLE}" \
    --chunk "${WM_CHUNK}" \
    --token-root "${TOKEN_ROOT}" \
    --work-dir "${TMPDIR}/hlong"
echo "LONGER_EXIT=$?"

echo "[$(date)] done -> ${WM_OUT_DIR}/verdict_h16.json + verdict_h${WM_LONG_FRAMES}.json"
echo "  M1 baseline to compare: /work/projects/imas_gpu/worldmodel/ckpt/spacetime-v2-1219852/sampling_rescore/verdict.json"
