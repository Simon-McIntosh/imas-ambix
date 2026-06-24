#!/bin/bash
#SBATCH --job-name=ambix-diag-eq-oracle
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --account=grpa
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=/work/projects/imas_gpu/logs/ambix-%x-%j.out
# NOTE: --qos intentionally omitted — defaults to normal (no 4-GPU cap), so this
# job bursts onto a free card and COEXISTS with the DeepSeek serve.
#
# ============================================================================
#  DIAGNOSTICS -> EQUILIBRIUM FEASIBILITY ORACLE
# ============================================================================
#  The decisive follow-up to the camera-only oracle.  The camera oracle FAILED
#  on the interior geometry (magnetic-axis / X-point skill ~0 — interior current
#  is not a camera observable).  This oracle swaps the INPUT to the MEASURED
#  DIAGNOSTICS (magnetics flux loops + B-field probes, interferometer, soft-x-ray,
#  Dα-boundary, xsx/xim/ait, summary/pf_active/gas_injection) — exactly the
#  streams the joint world model conditions on AND dreams — and trains a small
#  temporal probe to predict the 12-D equilibrium geometry.
#
#  No camera decode is needed (the signals are already tokenised) so this runs on
#  a single GPU and is much cheaper than the camera oracle.
#
#  VERDICT: do the measured diagnostics resolve axis + X-point (probe RMSE below
#  the predict-the-train-mean baseline by >= the ratio) where the camera could
#  not?  Also reports the magnetics-only vs all-diagnostics ABLATION (does
#  magnetics alone carry the axis/X-point, as EFIT implies?).
#
#  EVALUATOR-ONLY: no WM checkpoint is loaded; the probe + labels never enter
#  the world-model training path.  Equilibrium is an evaluator label only.
#
#  Env knobs:
#    EQ_N_TRAIN_SHOTS  — TRAIN shots to sample (default 250)
#    EQ_N_TEST_SHOTS   — TEST shots to sample (default 60, plus forced cohort)
#    EQ_N_FRAMES       — camera window length (default 24, ~0.25 s horizon)
#    EQ_N_SIGNAL_STEPS — measured-signal temporal positions (default 12)
#    EQ_EPOCHS         — probe training epochs (default 40)
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

: "${EQ_N_TRAIN_SHOTS:=250}"
: "${EQ_N_TEST_SHOTS:=60}"
: "${EQ_N_FRAMES:=24}"
: "${EQ_N_SIGNAL_STEPS:=12}"
: "${EQ_EPOCHS:=40}"
: "${EQ_RATIO:=1.3}"
: "${EQ_OUT_ROOT:=/work/projects/imas_gpu/worldmodel/diagnostics_equilibrium_oracle}"
: "${EQ_FIG_DIR:=${REPO_ROOT}/docs/figures/joint-multimodal-plasma-wm}"
: "${EQ_TOKEN_ROOT:=}"
: "${EQ_LEVEL2_ROOT:=}"

mkdir -p "${EQ_OUT_ROOT}" "${EQ_FIG_DIR}"

echo "[$(date)] diag-eq-oracle (job ${SLURM_JOB_ID} on $(hostname))"
echo "  out-root:    ${EQ_OUT_ROOT}"
echo "  fig-dir:     ${EQ_FIG_DIR}"
echo "  train/test:  ${EQ_N_TRAIN_SHOTS} / ${EQ_N_TEST_SHOTS} shots"
echo "  window:      ${EQ_N_FRAMES} frames  signal-steps=${EQ_N_SIGNAL_STEPS}  epochs=${EQ_EPOCHS}"
echo "  ratio:       beats baseline by ${EQ_RATIO}x on axis+X-point"
nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader || true

EXTRA=()
if [[ -n "${EQ_TOKEN_ROOT}" ]]; then
    EXTRA+=(--token-root "${EQ_TOKEN_ROOT}")
fi
if [[ -n "${EQ_LEVEL2_ROOT}" ]]; then
    EXTRA+=(--level2-root "${EQ_LEVEL2_ROOT}")
fi

python3 scripts/feasibility_diagnostics_equilibrium_oracle.py \
    --n-train-shots "${EQ_N_TRAIN_SHOTS}" \
    --n-test-shots "${EQ_N_TEST_SHOTS}" \
    --n-frames "${EQ_N_FRAMES}" \
    --n-signal-steps "${EQ_N_SIGNAL_STEPS}" \
    --epochs "${EQ_EPOCHS}" \
    --ratio-threshold "${EQ_RATIO}" \
    --out-root "${EQ_OUT_ROOT}" \
    --fig-dir "${EQ_FIG_DIR}" \
    "${EXTRA[@]}"
EXIT_CODE=$?

echo "[$(date)] diag-eq-oracle finished (exit=${EXIT_CODE})"
echo "  report:   ${EQ_OUT_ROOT}/oracle_report.json"
echo "  scatter:  ${EQ_OUT_ROOT}/fig-diag-eq-oracle-geometry-scatter.png"
exit "${EXIT_CODE}"
