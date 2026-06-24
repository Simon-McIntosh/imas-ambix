#!/bin/bash
#SBATCH --job-name=ambix-diag-space-ctrl
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --account=grpa
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --output=/work/projects/imas_gpu/logs/ambix-%x-%j.out
# NOTE: --qos intentionally omitted — defaults to normal (no 4-GPU cap), so this
# single-card job bursts onto a free card and COEXISTS with the DeepSeek serve.
#
# ============================================================================
#  DECODER-FREE dreamed-MAGNETICS command-sensitivity gate (the go/no-go).
# ============================================================================
#  The DECISIVE, cheap go/no-go for the absolute re-tokenise + retrain: do an
#  existing world model's DREAMED MAGNETICS respond to the actuator command?
#  Measured ENTIRELY in diagnostic-token space (NO camera decode / VQ stack) so
#  it is fast.  For each gate-cohort shot it rolls the model under the TRUE plan
#  + N random/wrong plans, extracts the dreamed per-stream diagnostic tokens
#  (argmax of the per-stream diagnostic head) over the forecast window, and
#  scores the magnetics-token-space ΔN-M ratio (true-vs-random / random-vs-random
#  floor).  ratio meaningfully > 1 ⇒ dreamed magnetics command-sensitive ⇒ GO.
#
#  Runs on BOTH checkpoints (RSSM + token joint-gen) so the RSSM-vs-token
#  command-sensitivity is directly comparable; reports the other streams
#  (interferometer / soft_x_rays / Dα) for context.  Single GPU — bursts onto a
#  free card, leaves DeepSeek untouched.
#
#  Env (all optional — defaults are the lead-named checkpoints + cohort):
#    WM_RSSM_CKPT    — RSSM checkpoint (default rssm-1222278 latest)
#    WM_TOKEN_CKPT   — token joint-gen checkpoint (default controllable-1221834 latest)
#    WM_COHORT       — gate cohort JSON (default gate_cohort.json, 25 shots)
#    WM_OUT_DIR      — report.json output dir
#    WM_FIGURE       — bar-figure path (under docs/figures/...)
#    WM_TOKEN_ROOT   — curated-token-view (same exposure as the train corpus)
#    WM_N_RANDOM     — wrong/random plans per shot (default 8)
#    WM_PERTURB_SCALE / WM_CHUNK / WM_TARGET_HORIZON_S — ΔN-M knobs
# ============================================================================

set -uo pipefail

export TMPDIR="/scratch_local/${SLURM_JOB_ID}"
mkdir -p "${TMPDIR}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_ROOT}"
mkdir -p "/work/projects/imas_gpu/logs"

VENV="${REPO_ROOT}/.venv"
# shellcheck source=/dev/null
source "${VENV}/bin/activate"

: "${WM_RSSM_CKPT:=/work/projects/imas_gpu/worldmodel/ckpt/rssm-1222278/latest.pt}"
: "${WM_TOKEN_CKPT:=/work/projects/imas_gpu/worldmodel/ckpt/controllable-1221834/latest.pt}"
: "${WM_COHORT:=/work/projects/imas_gpu/worldmodel/gate_cohort.json}"
: "${WM_OUT_DIR:=/work/projects/imas_gpu/worldmodel/diag_space_controllability}"
: "${WM_FIGURE:=docs/figures/joint-multimodal-plasma-wm/diag_space_controllability.png}"
: "${WM_TOKEN_ROOT:=/work/projects/imas_gpu/worldmodel/curated-token-view}"
: "${WM_N_RANDOM:=8}"
: "${WM_PERTURB_SCALE:=0.3}"
: "${WM_CHUNK:=8192}"
: "${WM_TARGET_HORIZON_S:=0.25}"

mkdir -p "${WM_OUT_DIR}"

echo "[$(date)] diag-space-controllability (job ${SLURM_JOB_ID} on $(hostname))"
echo "  rssm-ckpt:   ${WM_RSSM_CKPT}"
echo "  token-ckpt:  ${WM_TOKEN_CKPT}"
echo "  cohort:      ${WM_COHORT}"
echo "  out-dir:     ${WM_OUT_DIR}"
echo "  token-root:  ${WM_TOKEN_ROOT}"
echo "  n_random=${WM_N_RANDOM} perturb_scale=${WM_PERTURB_SCALE} horizon=${WM_TARGET_HORIZON_S}s"
nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader || true

python3 scripts/diagnostic_space_controllability.py \
    --rssm-checkpoint "${WM_RSSM_CKPT}" \
    --token-checkpoint "${WM_TOKEN_CKPT}" \
    --cohort "${WM_COHORT}" \
    --out-dir "${WM_OUT_DIR}" \
    --figure "${WM_FIGURE}" \
    --token-root "${WM_TOKEN_ROOT}" \
    --n-random "${WM_N_RANDOM}" \
    --perturb-scale "${WM_PERTURB_SCALE}" \
    --chunk "${WM_CHUNK}" \
    --target-horizon-s "${WM_TARGET_HORIZON_S}"
EXIT_CODE=$?

echo "[$(date)] diag-space-controllability finished (exit=${EXIT_CODE})"
echo "  report:  ${WM_OUT_DIR}/report.json"
echo "  figure:  ${WM_FIGURE}"
exit "${EXIT_CODE}"
