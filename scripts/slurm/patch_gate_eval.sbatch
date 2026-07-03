#!/bin/bash
#SBATCH --job-name=ambix-patch-gate
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --account=grpa
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/work/projects/imas_gpu/logs/ambix-%x-%j.out
# NOTE: --qos intentionally omitted — defaults to normal; a 1-card job bursts
# onto a free card and coexists with any model serve.
#
# ============================================================================
#  Patch-current gate-2 rematch (variational inverse, no Picard) + the
#  batched-forward H200 throughput measurement.
# ============================================================================
#  1. Measures batched PatchBasis forward throughput (grid ψ + sensors) on one
#     H200 — the substrate gate is ≥1e3 slices/s.
#  2. Runs the 160-slice held-out gate-2 protocol with the variational patch
#     inverse under the three force-balance weight-policy arms
#     (fixed / warm-start / discrepancy), identical shot list, slice
#     selection, train-mean baseline, and skill formulas to the Picard gate —
#     every slice scores (no convergence masking).
#  3. Recovers per-slice closures (p', FF'/mu0 per flux bin with regression
#     uncertainties; integrated p, F^2 with the F^2>=0 check).
#  Artifacts: imas_ambix/latent/artifacts/patch_gate/

set -u
cd /home/ITER/mcintos/Code/imas-ambix
export TMPDIR=/scratch_local/$SLURM_JOB_ID
mkdir -p "$TMPDIR" || export TMPDIR=/tmp
source .venv/bin/activate
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8
echo "PYTHON=$(which python)"
nvidia-smi -L | head -2

python scripts/patch_gate_eval.py \
    --throughput-bench \
    --policies fixed,warm-start,discrepancy \
    --lambda-fb 10 \
    --iters 800 \
    --device cuda
echo "GATE_EXIT=$?"
