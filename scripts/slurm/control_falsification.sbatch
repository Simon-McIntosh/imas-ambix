#!/bin/bash
#SBATCH --job-name=ambix-control-falsif
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --account=grpa
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=/work/projects/imas_gpu/logs/ambix-%x-%j.out
# NOTE: --qos intentionally omitted — gpu_0003_grpa QoS is auto-applied.
#
# ============================================================================
#  W1 controllability — CFG sweep + gas-puff falsification.
# ============================================================================
#  Loads the M2 context-corruption checkpoint + the frozen Open-MAGVIT2 VQModel
#  ONCE (AGENTS.md §2b).  For each held-out shot it rolls the dream out at
#  guidance weights w in {1.0, 1.5, 2.0} (classifier-free guidance on the
#  controls), plus a gas_injection-ZEROED counterfactual and same-conditioning
#  spread members, packs every rollout into ONE token bundle, decodes it in a
#  single VQ pass, and scores:
#    * inboard-emission timing correlation vs the gas-puff command,
#    * the counterfactual (true-puff) − (no-puff) inboard-emission delta,
#    * control-divergence (true vs zeroed) vs same-conditioning sample spread.
#  Answers W1: does the predicted emission track the puff command, does CFG
#  strengthen it, and does changing the control move the world more than noise?
#
#  Single GPU — bursts onto a free card; leaves the DeepSeek-V4-Flash neighbour
#  untouched.  Window flags MUST match the training window (24f / 8ctx / 8plan /
#  4 signal steps).  scancel + normal --time are fine (AGENTS.md §2a).
#
#  Inputs (env):
#    WM_CKPT     — checkpoint .pt (default the M2 corruption best.pt)
#    WM_SHOTS    — comma-separated held-out shot ids (default 18502-18505)
#    WM_OUT_JSON — verdict JSON path
#    WM_FIG_DIR  — optional per-shot control-panel figures dir
#    WM_GUIDANCE — comma-separated CFG weights (default 1.0,1.5,2.0)
#    WM_SPREAD   — same-conditioning spread members (default 3)
#    WM_TEMP / WM_TOP_P — sampling settings (default 0.9 / 0.95, the M2 best)
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

: "${WM_CKPT_DIR:=/work/projects/imas_gpu/worldmodel/ckpt/spacetime-corruption-1220391}"
: "${WM_CKPT:=${WM_CKPT_DIR}/best.pt}"
: "${WM_SHOTS:=18502,18503,18504,18505}"
: "${WM_OUT_JSON:=${WM_CKPT_DIR}/control_falsification/verdict.json}"
: "${WM_FIG_DIR:=${WM_CKPT_DIR}/control_falsification/figures}"
: "${WM_GUIDANCE:=1.0,1.5,2.0}"
: "${WM_SPREAD:=3}"
: "${WM_TEMP:=0.9}"
: "${WM_TOP_P:=0.95}"
# window — MUST match the training window
: "${WM_N_FRAMES:=24}"
: "${WM_N_PLAN:=8}"
: "${WM_CONTEXT_FRAMES:=8}"
: "${WM_FRAME_STRIDE:=1}"
: "${WM_N_SIGNAL_STEPS:=4}"
: "${WM_CHUNK:=8192}"

if [[ ! -f "${WM_CKPT}" ]]; then
    echo "[w1] FATAL: checkpoint not found: ${WM_CKPT}" >&2
    exit 2
fi

echo "[$(date)] W1 control falsification (job ${SLURM_JOB_ID} on $(hostname))"
echo "  checkpoint: ${WM_CKPT}"
echo "  shots:      ${WM_SHOTS}"
echo "  out-json:   ${WM_OUT_JSON}"
echo "  guidance:   ${WM_GUIDANCE}  spread=${WM_SPREAD}  T=${WM_TEMP} p=${WM_TOP_P}"
echo "  window:     n_frames=${WM_N_FRAMES} n_plan=${WM_N_PLAN} context=${WM_CONTEXT_FRAMES} stride=${WM_FRAME_STRIDE} n_signal_steps=${WM_N_SIGNAL_STEPS}"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader || true

python -m imas_ambix.worldmodel.control_falsification \
    --checkpoint "${WM_CKPT}" \
    --shots "${WM_SHOTS}" \
    --out-json "${WM_OUT_JSON}" \
    --figure-dir "${WM_FIG_DIR}" \
    --guidance "${WM_GUIDANCE}" \
    --spread-members "${WM_SPREAD}" \
    --temperature "${WM_TEMP}" \
    --top-p "${WM_TOP_P}" \
    --n-frames "${WM_N_FRAMES}" \
    --n-plan "${WM_N_PLAN}" \
    --context-frames "${WM_CONTEXT_FRAMES}" \
    --frame-stride "${WM_FRAME_STRIDE}" \
    --n-signal-steps "${WM_N_SIGNAL_STEPS}" \
    --chunk "${WM_CHUNK}" \
    --work-dir "${TMPDIR}/w1" \
    --device cuda
EXIT_CODE=$?

echo "[$(date)] W1 control falsification finished (exit=${EXIT_CODE})"
echo "  verdict: ${WM_OUT_JSON}"
echo "  figures: ${WM_FIG_DIR}"
exit "${EXIT_CODE}"
