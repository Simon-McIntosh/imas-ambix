#!/bin/bash
#SBATCH --job-name=ambix-cohort-driveable
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --account=grpa
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --output=/work/projects/imas_gpu/logs/ambix-%x-%j.out
# NOTE: --qos intentionally omitted — defaults to normal (no 4-GPU cap), so this
# 2-card decode bursts onto free cards and COEXISTS with any DeepSeek serve.
#
# ============================================================================
#  DRIVEABLE-ENRICHED controllability-gate cohort build.
# ============================================================================
#  The powered gate's variance decomposition showed ACROSS-shot heterogeneity
#  dominates (a/w ~ 4.2): the screened cohort mixes genuinely-driveable shots
#  with shots whose plasma cannot respond to commands in principle, so its
#  cohort-mean UNDERSTATES real controllability.  This job builds a SEPARATE,
#  driveable-enriched cohort so the gate can measure the driveable regime.
#
#  Anti-circularity (binding): "driveable" is a PHYSICAL, MODEL-FREE property of
#  the GROUND-TRUTH data + the DEMANDED commands ONLY — never screened by a
#  world-model's own response.  Each train-disjoint candidate is ranked by a
#  driveability score combining (a) demanded-command variation, (b) GT transient
#  motion, (c) GT emission-centroid path length, and (d) a temporal-association
#  proxy (commands change WHEN the plasma moves).  The top ~15-20 eligible shots
#  are kept.  No model rollout is consulted, so the cohort cannot be biased toward
#  shots a model already drives.
#
#  Loads the frozen VQ decoder ONCE for the whole candidate scan (AGENTS.md §2b);
#  no per-shot/per-chunk reload.  scancel + ordinary --time are fine (§2a).
#
#  Writes a NEW file (does NOT touch gate_cohort.json, which the baked-in eval
#  reads):
#    /work/projects/imas_gpu/worldmodel/gate_cohort_driveable.json
#
#  Submit AFTER the live training job frees the node:
#    sbatch --dependency=afterany:<training-jobid> \
#           scripts/slurm/gate_cohort_driveable.sbatch
#
#  Env (optional):
#    WM_COHORT_CAP     — max candidates to scan (default 400)
#    WM_COHORT_TARGET  — target driveable-cohort size (default 18)
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
: "${WM_COHORT_TARGET:=18}"
export WM_COHORT_CAP WM_COHORT_TARGET

OUT_JSON="/work/projects/imas_gpu/worldmodel/gate_cohort_driveable.json"

echo "[$(date)] driveable-cohort build (job ${SLURM_JOB_ID} on $(hostname))"
echo "  cohort-cap:    ${WM_COHORT_CAP}"
echo "  cohort-target: ${WM_COHORT_TARGET}"
echo "  out json:      ${OUT_JSON}  (NEW file — gate_cohort.json untouched)"
nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader || true

python3 - <<'PYEOF'
import json
import os
from pathlib import Path

import torch

from imas_ambix.worldmodel.controllable_eval import (
    EvalConfig,
    _resolve_eval_modalities,
)
from imas_ambix.worldmodel.controllable_train import (
    load_controllable_model_from_checkpoint,
)
from imas_ambix.worldmodel.gate_cohort import (
    DEFAULT_DRIVEABLE_COHORT_PATH,
    build_driveable_cohort,
    load_cohort,
)
from imas_ambix.worldmodel.spacetime_dataset import SpacetimeWindowConfig

TOKEN_ROOT = Path("/work/projects/imas_gpu/mast-tokens")
MANIFEST = (
    "/work/projects/imas_gpu/agents/excitation-corpus/"
    "curated_windows_unified_6cam.json"
)
CAMERA = "rbb"
DEVICE = "cuda"
HORIZON_S = 0.25
# modality-resolution uses the same baseline checkpoint the screened cohort used;
# the screen needs only the trained stream list (for stream-presence counting),
# NOT a model rollout — so this is anti-circularity-safe.
BASELINE_CKPT = Path(
    "/work/projects/imas_gpu/worldmodel/ckpt/controllable-1221741/latest.pt"
)
OUT_JSON = DEFAULT_DRIVEABLE_COHORT_PATH
WORK_DIR = Path("/work/projects/imas_gpu/worldmodel/robust_gate/_cohort_driveable")

CANDIDATE_CAP = int(os.environ.get("WM_COHORT_CAP", "400"))
COHORT_TARGET = int(os.environ.get("WM_COHORT_TARGET", "18"))

torch.manual_seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.set_float32_matmul_precision("high")


def window_cfg() -> SpacetimeWindowConfig:
    return SpacetimeWindowConfig(
        n_frames=24,
        n_plan=8,
        context_frames=8,
        frame_stride=1,
        target_horizon_s=HORIZON_S,
    )


# resolve the trained stream list from the baseline checkpoint payload, then
# release the model BEFORE the cohort build (no rollout — modality list only).
_model, payload = load_controllable_model_from_checkpoint(
    BASELINE_CKPT, map_location=DEVICE
)
del _model
torch.cuda.empty_cache()
modalities = _resolve_eval_modalities("auto", payload)

screen_cfg = EvalConfig(
    n_signal_steps=4,
    n_act_steps=8,
    modalities=modalities,
    window=window_cfg(),
)

summary = build_driveable_cohort(
    screen_cfg,
    camera=CAMERA,
    token_root=TOKEN_ROOT,
    manifest_path=MANIFEST,
    device=DEVICE,
    out_json=OUT_JSON,
    candidate_cap=CANDIDATE_CAP,
    target_size=COHORT_TARGET,
    work_dir=WORK_DIR,
)
print("DRIVEABLE COHORT SUMMARY:", json.dumps(summary, indent=2), flush=True)
cohort = load_cohort(OUT_JSON)
print(f"DRIVEABLE COHORT ({len(cohort)} shots): {cohort}", flush=True)
print("DONE=1", flush=True)
PYEOF
EXIT_CODE=$?

echo "[$(date)] driveable-cohort build finished (exit=${EXIT_CODE})"
echo "  out json: ${OUT_JSON}"
exit "${EXIT_CODE}"
