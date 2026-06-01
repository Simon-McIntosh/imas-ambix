#!/bin/bash
#SBATCH --job-name=mwe03-nethang
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --account=grpa
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=/work/projects/imas_gpu/logs/mwe03-%j.out
#SBATCH --error=/work/projects/imas_gpu/logs/mwe03-%j.err

# MWE-03 v3: DDP + CUDA + urllib to GPFS — Network Hang Drain Reproducer
# Risk: ⚠️⚠️ LIKELY DRAIN (DDP torchrun orphan workers + D-state).
#
# v1/v2 finding: single-process urllib is S-state (interruptible) — no drain.
# v3: DDP torchrun workers + live CUDA context + urlretrieve to GPFS.
#   SLURM sends SIGTERM to torchrun (parent). Torchrun forwards to workers.
#   Workers in D-state (urllib blocked on DROP-firewalled network) cannot
#   receive SIGTERM. SLURM SIGKILLs torchrun → workers become orphans →
#   D-state orphans survive SIGKILL → UnkillableStepTimeout → node drain.
#
# Expected recovery:
#   nvidia-smi -i 0,1,2,3 --gpu-reset   (in case CUDA ioctl D-state)
#   scontrol update nodename=98dci4-gpu-0003 state=resume reason=""
#
# NO three-layer defence — INTENTIONAL.

set -uo pipefail

export TMPDIR=/scratch_local/${SLURM_JOB_ID}
mkdir -p "${TMPDIR}" || export TMPDIR=/tmp
mkdir -p /work/projects/imas_gpu/logs

export DRAIN_TEST_LOG_DIR=/work/projects/imas_gpu/logs
export DOWNLOAD_URL="${DOWNLOAD_URL:-https://heibox.uni-heidelberg.de/f/607503859d364012b37e/?dl=1}"
export GPFS_DEST="${GPFS_DEST:-/work/projects/imas_gpu/mwe03-download-test.bin}"
export WAIT_BEFORE="${WAIT_BEFORE:-5}"
export AUTO_SCANCEL_DELAY="${AUTO_SCANCEL_DELAY:-2}"

TORCHRUN="${PYTHON_VENV:-/work/projects/imas_gpu/mast-tokens/v1/open-magvit2/.venv}/bin/torchrun"

echo "=== MWE-03 v3 Network Hang + DDP Orphan Drain Reproducer ==="
echo "Job: ${SLURM_JOB_ID}  Node: ${SLURMD_NODENAME}"
echo "*** DESIGNED TO DRAIN THE NODE ***"
echo "URL: ${DOWNLOAD_URL}"
echo "DEST: ${GPFS_DEST}.rankN"
echo "Auto-scancel: ${AUTO_SCANCEL_DELAY}s after READY (all DDP ranks at urllib)"
echo ""
echo "Expected sequence:"
echo "  1. DDP init (4 ranks, NCCL) + CUDA context on each GPU"
echo "  2. All ranks call urlretrieve → hang (DROP firewall, no outbound network)"
echo "  3. Auto-scancel fires → SIGTERM to torchrun → forwarded to workers"
echo "  4. Workers in D-state → torchrun SIGKILLed → orphaned D-state workers"
echo "  5. UnkillableStepTimeout (60s) → 'Kill task failed' → DRAIN"
echo ""

${TORCHRUN} \
    --nproc_per_node=4 \
    --master_addr="${SLURMD_NODENAME}" \
    --master_port=29502 \
    "/home/ITER/mcintos/Code/imas-ambix/scripts/drain-tests/mwe-03-network-hang.py"

echo "=== MWE-03 COMPLETE (exited cleanly — check node state for drain) ==="
