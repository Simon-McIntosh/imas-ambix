#!/bin/bash
#SBATCH --job-name=ambix-cohort-expand
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --account=grpa
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --output=/work/projects/imas_gpu/logs/ambix-%x-%j.out
# NOTE: --qos intentionally omitted — defaults to normal (no 4-GPU cap), so this
# 2-card decode bursts onto free cards and COEXISTS with the DeepSeek serve.
#
# ============================================================================
#  EXPANDED-cohort controllability re-eval — wider gate, faster screen.
# ============================================================================
#  The screened controllability cohort was only ~7 shots, so the robust gate's
#  bootstrap CI was very wide.  This job:
#    1. SCREENS a much larger train-disjoint eval-only cohort — scans up to a few
#       hundred candidates and targets ~25 passing shots.  The screen loads the
#       frozen VQ decoder ONCE for the whole scan (no per-chunk reload) and
#       relaxes the brightness/motion gates toward their floors if too few pass
#       (the applied thresholds are recorded in the cohort JSON).
#    2. Re-runs the ROBUST controllability gate (auto modalities, collapse-
#       rejecting normalised ratio + bootstrap-CI majority gate) on the THREE
#       checkpoints (controllable-1221741 baseline / -1221834 joint-gen /
#       -1221883 diag-off) against the expanded cohort, writing a per-checkpoint
#       robust_gate.json + a combined compare JSON under
#       /work/projects/imas_gpu/worldmodel/robust_gate/.
#    3. Overwrites docs/figures/joint-multimodal-plasma-wm/gate_robust_compare.png
#       with the wider-cohort comparison figure.
#
#  Loads each model once and decodes rollouts in one-VQ-pass batches (AGENTS.md
#  §2b).  scancel + ordinary --time are fine (AGENTS.md §2a).  Submit with
#  --dependency=afterany:<training-jobid> so it PENDS behind the live training
#  job and runs automatically when the cards free.
#
#  Env (optional):
#    WM_COHORT_CAP     — max candidates to scan (default 400)
#    WM_COHORT_TARGET  — target passing-cohort size (default 25)
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

: "${WM_COHORT_CAP:=400}"
: "${WM_COHORT_TARGET:=25}"
export WM_COHORT_CAP WM_COHORT_TARGET

echo "[$(date)] cohort-expand re-eval (job ${SLURM_JOB_ID} on $(hostname))"
echo "  cohort-cap:    ${WM_COHORT_CAP}"
echo "  cohort-target: ${WM_COHORT_TARGET}"
echo "  checkpoints:   1221741 (baseline) / 1221834 (joint-gen) / 1221883 (diag-off)"
echo "  cohort json:   /work/projects/imas_gpu/worldmodel/gate_cohort.json"
echo "  out root:      /work/projects/imas_gpu/worldmodel/robust_gate/"
echo "  figure:        docs/figures/joint-multimodal-plasma-wm/gate_robust_compare.png"
nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader || true

python3 "${REPO_ROOT}/scripts/_robust_gate_reeval.py"
EXIT_CODE=$?

echo "[$(date)] cohort-expand re-eval finished (exit=${EXIT_CODE})"
echo "  compare json: /work/projects/imas_gpu/worldmodel/robust_gate/robust_gate_compare.json"
exit "${EXIT_CODE}"
