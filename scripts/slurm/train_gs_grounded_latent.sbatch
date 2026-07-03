#!/bin/bash
#SBATCH --job-name=ambix-gs-latent-train
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --account=grpa
#SBATCH --gres=gpu:6
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=/work/projects/imas_gpu/logs/ambix-%x-%j.out
# Corpus-train the GS-grounded hybrid latent on the 6 H200s left free by the
# 2-card DeepSeek serve (6 + 2 = 8/8; --qos intentionally omitted — defaults to
# normal, no 4-GPU cap; repo AGENTS.md §2).  CPU/MEM sized to coexist with the
# serve (12 of the ~49 free cores; the corpus windows are small arrays).
#
# torchrun launches one process per GPU; every rank holds a replica of the
# shared encoder + per-campaign transports and samples its own excitation-
# weighted minibatches; the CorpusTrainer all-reduces gradients each step
# (replicated optimiser — effective batch = 6 x batch-size).  Rank 0 only
# checkpoints.  RESUME-SAFE: --resume picks up the atomic checkpoint exactly,
# and a bare `sbatch` requeue continues where SIGTERM left off.

set -uo pipefail

export TMPDIR="/scratch_local/${SLURM_JOB_ID}"
mkdir -p "${TMPDIR}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export NCCL_P2P_LEVEL=NVL

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_ROOT}"
mkdir -p "/work/projects/imas_gpu/logs"

# drain-forensics sidecar (default-on for long GPU jobs; AGENTS.md §2a)
bash "${REPO_ROOT}/scripts/slurm/drain_sidecar.sh" &
SIDECAR_PID=$!
cleanup() {
    echo "[$(date)] gs-latent-train teardown"
    kill "${SIDECAR_PID}" 2>/dev/null || true
}
trap cleanup EXIT

source "${REPO_ROOT}/.venv/bin/activate"

: "${GSL_N_TRAIN:=200}"
: "${GSL_STEPS:=20000}"
: "${GSL_BATCH_SIZE:=256}"
: "${GSL_LR:=3e-4}"
: "${GSL_CKPT_DIR:=imas_ambix/latent/artifacts/checkpoints}"
: "${GSL_NPROC:=6}"

echo "[$(date)] gs-latent-train starting (job ${SLURM_JOB_ID} on $(hostname))"
echo "  ranks=${GSL_NPROC} n_train=${GSL_N_TRAIN} steps=${GSL_STEPS} bs=${GSL_BATCH_SIZE}/rank lr=${GSL_LR}"
echo "  ckpt: ${GSL_CKPT_DIR}/gs_grounded_latent.pt"
nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader || true

torchrun \
    --standalone \
    --nproc_per_node="${GSL_NPROC}" \
    scripts/train_gs_grounded_latent.py \
    --n-train "${GSL_N_TRAIN}" \
    --steps "${GSL_STEPS}" \
    --batch-size "${GSL_BATCH_SIZE}" \
    --lr "${GSL_LR}" \
    --checkpoint-dir "${GSL_CKPT_DIR}" \
    --resume
EXIT_CODE=$?

echo "[$(date)] gs-latent-train finished (exit=${EXIT_CODE})"
echo "  checkpoint: ${GSL_CKPT_DIR}/gs_grounded_latent.pt"
exit "${EXIT_CODE}"
