#!/bin/bash
#SBATCH --job-name=drain-repro
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --account=grpa
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:05:00
#SBATCH --output=/work/projects/imas_gpu/logs/drain-repro-%j.out
#SBATCH --error=/work/projects/imas_gpu/logs/drain-repro-%j.err
#
# DECISIVE DRAIN REPRODUCER. Faithful to production drains 1209813 / 1208980.
# Intentionally has NO drain-safety (no STOP-FILE, no signal handler): the goal
# is to drive ncclCommAbort into the unkillable state and let the SLURM --time
# limit's SIGTERM fail to reap it -> node drain. Run ONLY with admin standby.
#
# Recovery after drain (admin / operator):
#   ssh 98dci4-gpu-0003 'ps -eo pid,stat,wchan:32,cmd | grep -E "python|torch"'
#   nvidia-smi -i 0,1,2,3 --gpu-reset
#   scontrol update nodename=98dci4-gpu-0003 state=resume reason=""
set -uo pipefail

export TMPDIR=/scratch_local/${SLURM_JOB_ID}
mkdir -p "${TMPDIR}" || export TMPDIR=/tmp
mkdir -p /work/projects/imas_gpu/logs

export REPRO_MODE="${REPRO_MODE:-seq_divergence}"
export ROUNDS="${ROUNDS:-20}"
export NUMEL="${NUMEL:-67108864}"

# stuck_collective mode (NCCL #829): disable the torch/NCCL watchdog so the
# stuck collective is NOT auto-aborted before the SLURM --time kill lands.
if [ "${REPRO_MODE}" = "stuck_collective" ] || [ "${REPRO_MODE}" = "abort_stuck" ]; then
    export TORCH_NCCL_ASYNC_ERROR_HANDLING=0
    export TORCH_NCCL_ENABLE_MONITORING=0
    export NCCL_DEBUG=WARN
fi

VENV="${PYTHON_VENV:-/work/projects/imas_gpu/mast-tokens/v1/open-magvit2/.venv}"
TORCHRUN="${VENV}/bin/torchrun"
SCRIPT="/home/ITER/mcintos/Code/imas-ambix/scripts/drain-tests/instrumented/repro_abort.py"

echo "=== drain-repro === job=${SLURM_JOB_ID} node=${SLURMD_NODENAME} mode=${REPRO_MODE} start=$(date -Is)"
echo "=== EXPECT: ncclCommAbort hang -> --time SIGTERM at 05:00 -> unkillable -> DRAIN ~06:00 ==="

"${TORCHRUN}" --nproc_per_node=4 --master_port=29531 "${SCRIPT}"

echo "=== drain-repro exit $? at $(date -Is) (unexpected clean exit = no wedge) ==="
