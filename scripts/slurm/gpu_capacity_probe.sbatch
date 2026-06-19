#!/bin/bash
# gpu_capacity_probe.sbatch — short, harmless test that a single Group A submit
# can grab more GPUs than the nominal 4-per-group convention.
#
# It allocates N GPUs (default 6) under the default QOS (normal, no GPU cap),
# enumerates the physical cards SLURM handed out (SLURM_JOB_GPUS), the remapped
# CUDA_VISIBLE_DEVICES the job sees, and the live nvidia-smi view, then exits.
# No model load, no training — pure capacity/affinity probe.
#
# Submit:
#   sbatch scripts/slurm/gpu_capacity_probe.sbatch                 # 6 cards
#   sbatch --gres=gpu:8 scripts/slurm/gpu_capacity_probe.sbatch    # try all 8
#   sbatch --test-only scripts/slurm/gpu_capacity_probe.sbatch     # dry-run

#SBATCH --job-name=gpu-cap-probe
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --account=grpa
#SBATCH --gres=gpu:6
#SBATCH --cpus-per-task=6
#SBATCH --mem=60G
#SBATCH --time=00:03:00
#SBATCH --output=/tmp/gpu-cap-probe-%j.out

set -uo pipefail

echo "================ GPU CAPACITY PROBE ================"
echo "job id            : ${SLURM_JOB_ID:-?}"
echo "node              : $(hostname)"
echo "account           : ${SLURM_JOB_ACCOUNT:-?}"
echo "qos               : ${SLURM_JOB_QOS:-?}"
echo "reservation       : ${SLURM_JOB_RESERVATION:-?}"
echo "requested gpus    : ${SLURM_GPUS_ON_NODE:-?} on node"
echo "SLURM_JOB_GPUS    : ${SLURM_JOB_GPUS:-?}   <- physical card indices SLURM gave us"
echo "CUDA_VISIBLE_DEV  : ${CUDA_VISIBLE_DEVICES:-?}   <- remapped view the job sees"
echo "cpus on node      : ${SLURM_CPUS_ON_NODE:-?}"
echo "cpu bind cores    : $(cat /sys/fs/cgroup/cpuset.cpus.effective 2>/dev/null || echo '?')"
echo "==================================================="
echo
echo "---- nvidia-smi -L (cards visible to this job) ----"
nvidia-smi -L 2>&1 || echo "nvidia-smi -L failed"
echo
echo "---- nvidia-smi (memory/util of visible cards) ----"
nvidia-smi --query-gpu=index,uuid,memory.used,memory.total,utilization.gpu --format=csv 2>&1 || echo "nvidia-smi query failed"
echo
echo "PROBE_DONE=$?"
