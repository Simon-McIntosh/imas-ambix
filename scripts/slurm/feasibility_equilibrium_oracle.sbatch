#!/bin/bash
#SBATCH --job-name=ambix-eq-oracle
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --account=grpa
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=/work/projects/imas_gpu/logs/ambix-%x-%j.out
# NOTE: --qos intentionally omitted — defaults to normal (no 4-GPU cap), so this
# job bursts onto free cards and COEXISTS with the DeepSeek serve.
#
# ============================================================================
#  EQUILIBRIUM FEASIBILITY ORACLE — the make-or-break decisive check.
# ============================================================================
#  Decodes REAL rbb camera tokens (frozen Open-MAGVIT2, loaded once, batched),
#  builds 12-D plasma-geometry labels from the L2 equilibrium store, trains a
#  small heteroscedastic CNN probe a few epochs on TRAIN shots, and evaluates
#  per-component RMSE in METRES on a shot-disjoint TEST set (the controllability
#  gate cohort + standing held-out 18502-05 are FORCED into TEST).
#
#  VERDICT: is the probe RMSE materially below the predict-the-train-mean
#  baseline (the shot-to-shot spread) for axis + X-point?  If not, a physics-
#  grounded controllability gate is INFEASIBLE and the big build stops here.
#
#  EVALUATOR-ONLY: no WM checkpoint is loaded; the probe + labels never enter
#  the world-model training path.
#
#  Uses 2 GPUs of headroom for the decode batch; the probe itself is tiny.
#
#  Env knobs:
#    EQ_N_TRAIN_SHOTS  — TRAIN shots to sample (default 120)
#    EQ_N_TEST_SHOTS   — TEST shots to sample (default 40, plus forced cohort)
#    EQ_N_FRAMES       — rbb window length (default 24, ~0.25 s horizon)
#    EQ_IN_FRAMES      — channel-stacked probe input frames k (default 4)
#    EQ_EPOCHS         — probe training epochs (default 12)
#    EQ_RATIO          — probe must beat baseline by this factor (default 1.3)
#    EQ_OUT_ROOT       — JSON + scatter output root
#    EQ_FIG_DIR        — docs figures dir for the scatter
#    EQ_TOKEN_ROOT     — override token root (default the canonical mast-tokens)
#    EQ_LEVEL2_ROOT    — override L2 equilibrium root
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

: "${EQ_N_TRAIN_SHOTS:=120}"
: "${EQ_N_TEST_SHOTS:=40}"
: "${EQ_N_FRAMES:=24}"
: "${EQ_IN_FRAMES:=4}"
: "${EQ_EPOCHS:=12}"
: "${EQ_RATIO:=1.3}"
: "${EQ_OUT_ROOT:=/work/projects/imas_gpu/worldmodel/equilibrium_oracle}"
: "${EQ_FIG_DIR:=${REPO_ROOT}/docs/figures/joint-multimodal-plasma-wm}"
: "${EQ_TOKEN_ROOT:=}"
: "${EQ_LEVEL2_ROOT:=}"

mkdir -p "${EQ_OUT_ROOT}" "${EQ_FIG_DIR}"

echo "[$(date)] eq-oracle (job ${SLURM_JOB_ID} on $(hostname))"
echo "  out-root:    ${EQ_OUT_ROOT}"
echo "  fig-dir:     ${EQ_FIG_DIR}"
echo "  train/test:  ${EQ_N_TRAIN_SHOTS} / ${EQ_N_TEST_SHOTS} shots"
echo "  window:      ${EQ_N_FRAMES} frames  probe-k=${EQ_IN_FRAMES}  epochs=${EQ_EPOCHS}"
echo "  ratio:       beats baseline by ${EQ_RATIO}x on axis+X-point"
nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader || true

EXTRA=()
if [[ -n "${EQ_TOKEN_ROOT}" ]]; then
    EXTRA+=(--token-root "${EQ_TOKEN_ROOT}")
fi
if [[ -n "${EQ_LEVEL2_ROOT}" ]]; then
    EXTRA+=(--level2-root "${EQ_LEVEL2_ROOT}")
fi

python3 scripts/feasibility_equilibrium_oracle.py \
    --n-train-shots "${EQ_N_TRAIN_SHOTS}" \
    --n-test-shots "${EQ_N_TEST_SHOTS}" \
    --n-frames "${EQ_N_FRAMES}" \
    --in-frames "${EQ_IN_FRAMES}" \
    --epochs "${EQ_EPOCHS}" \
    --ratio-threshold "${EQ_RATIO}" \
    --out-root "${EQ_OUT_ROOT}" \
    --fig-dir "${EQ_FIG_DIR}" \
    "${EXTRA[@]}"
EXIT_CODE=$?

echo "[$(date)] eq-oracle finished (exit=${EXIT_CODE})"
echo "  report:   ${EQ_OUT_ROOT}/oracle_report.json"
echo "  scatter:  ${EQ_OUT_ROOT}/fig-eq-oracle-axis-scatter.png"
exit "${EXIT_CODE}"
