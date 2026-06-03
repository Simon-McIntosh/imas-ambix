#!/bin/bash
#SBATCH --job-name=track-c-fuse-wedge
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --account=grpa
#SBATCH --nodelist=98dci4-gpu-0003
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:04:00
#SBATCH --output=/work/projects/imas_gpu/logs/track-c-%j.out
#SBATCH --error=/work/projects/imas_gpu/logs/track-c-%j.err
#
# ⚠️  DELIBERATE DRAIN TEST — runs ONLY inside a coordinated admin window.
# ⚠️  This job WILL drain 98dci4-gpu-0003. Admin must be present to resume.
# ⚠️  DO NOT RUN outside the coordinated window.
#
# Track C: FUSE-freeze D-state wedge (CPU-only, no GPU).
# Locked decision: track-c-wedge → fuse-freeze.
#
# Mechanism:
#   A SquashFS archive is mounted via squashfuse (foreground daemon).  A reader
#   subprocess starts reading a large file from the mount.  The FUSE daemon is
#   then SIGSTOP'd while a read request is in-flight.  The reader's kernel wait
#   becomes UNINTERRUPTIBLE (D-state, wchan=fuse_simple_request).  When SLURM
#   --time fires 4 min later, slurmstepd cannot reap the D-state reader within
#   UnkillableStepTimeout (~60 s) → "Kill task failed" → node DRAIN.
#
# NO STOP-FILE: deliberately omitted.  The STOP-FILE contract (AGENTS.md §2a)
#   exists to prevent drains; this is a test designed to cause one.  Adding a
#   stop-file would defeat Track C's purpose of measuring the actual drain timing.
#
# NO SIGUSR1 trap: deliberately omitted for the same reason.  The three-layer
#   defence (AGENTS.md §2a-cancel-time) protects production jobs; Track C must
#   bypass it to expose the raw UnkillableStepTimeout → drain window.
#
# NO --gres: NO GPU requested.  This is a CPU-only wedge.  Recovery requires
#   only scontrol resume — no nvidia-smi --gpu-reset.
#
# Recovery (admin, after drain):
#   # Confirm no stuck processes (no GPU context):
#   ssh 98dci4-gpu-0003 'ps aux | grep track_c'   # should show only grep
#   # Resume:
#   scontrol update nodename=98dci4-gpu-0003 state=resume reason=""
#   # Confirm:
#   scontrol show node 98dci4-gpu-0003 | grep State
#
# Observer job (start BEFORE this job):
#   sbatch observer.sbatch       # CPU-only; note JOBID
#   # Wait until RUNNING, then submit this job.
set -uo pipefail

export TMPDIR=/scratch_local/${SLURM_JOB_ID}
mkdir -p "${TMPDIR}" || export TMPDIR=/tmp
mkdir -p /work/projects/imas_gpu/logs

echo "=== track-c-fuse-wedge === job=${SLURM_JOB_ID} node=${SLURMD_NODENAME} start=$(date -Is)"
echo "=== ⚠️  DELIBERATE DRAIN TEST — fuse-freeze wedge, no GPU, no STOP-FILE ==="
echo "=== --time=00:04:00 → drain expected at ~t+5 min (2 min wedge + 1 min UnkillableTimeout) ==="

# Activate the project venv for Python 3.12 + standard libs.
# track_c_cgroup_wedge.py uses only stdlib (os, signal, subprocess, etc.) so
# any Python 3 install works.  Try the shared GPFS venv first; fall back to
# the system python3.
VENV_PYTHON=/work/projects/imas_gpu/mast-tokens/v1/open-magvit2/.venv/bin/python3
if [ ! -x "${VENV_PYTHON}" ]; then
    VENV_PYTHON=$(which python3)
fi
echo "=== using python: ${VENV_PYTHON} ($("${VENV_PYTHON}" --version 2>&1)) ==="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="${SCRIPT_DIR}/track_c_cgroup_wedge.py"

"${VENV_PYTHON}" "${SCRIPT}"

# If python exits cleanly it means the wedge setup failed (FUSE not available
# or D-state not achieved).  The node was NOT drained.
EXIT_CODE=$?
echo "=== track-c exit ${EXIT_CODE} at $(date -Is) ==="
echo "=== If exit 0/1 here without a drain, check the script output above for FATAL/ERROR messages ==="
