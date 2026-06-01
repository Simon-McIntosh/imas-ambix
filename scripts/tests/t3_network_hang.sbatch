#!/bin/bash
# T3: Minimal network I/O D-state drain reproducer.
#
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  WARNING: THIS JOB WILL DRAIN 98dci4-gpu-0003 (betelgeuse)              ║
# ║  Run only with an SDCC admin ready to issue:                             ║
# ║    scontrol update nodename=98dci4-gpu-0003 state=resume reason=""       ║
# ║  No nvidia-smi --gpu-reset needed (GPUs are never touched by this job)  ║
# ║  Expected recovery time: ~5 minutes                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# WHAT THIS PROVES:
#   Reproduces the 2026-05-27 16:04 drain (Event 2 / Drain #3).
#   A process enters D-state (uninterruptible kernel sleep) on a blocked
#   network socket BEFORE any CUDA call. scancel → SIGTERM → D-state →
#   UnkillableStepTimeout (60s) → "Kill task failed" → node DRAIN.
#
#   This demonstrates that the drain mechanism does NOT require CUDA/NCCL —
#   ANY uninterruptible kernel sleep on the node can trigger it.
#
# MECHANISM TIMELINE (expected):
#   T+0s  : Job starts, Python opens a TCP socket to 8.8.8.8:81
#   T+5s  : (Operator) scancel <JOBID>   ← SLURM sends SIGTERM
#   T+5s  : Process is in D-state (blocked on connect() syscall)
#            SIGTERM is delivered but process cannot receive it
#   T+65s : SLURM UnkillableStepTimeout expires (60s after SIGTERM)
#            slurmd: "Kill task failed for job <JOBID>"
#            Node state → DRAIN
#
# COUNTER-TEST (without draining):
#   Add --timeout=5 to the curl command. Fast ICMP REJECT (if firewall is
#   configured with REJECT not DROP) would also prevent the D-state.
#
# RECOVERY (after drain):
#   scontrol update nodename=98dci4-gpu-0003 state=resume reason=""
#   (GPU reset NOT required — no CUDA context was created)
#
# SBATCH directives intentionally minimal — no three-layer defence —
# to demonstrate the vulnerable pattern.
#SBATCH --job-name=ambix-t3-nethang
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --account=grpa
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --output=/work/projects/imas_gpu/logs/t3-nethang-%j.log

set -e
export TMPDIR=/scratch_local/${SLURM_JOB_ID}
mkdir -p "${TMPDIR}"

NODE=$(hostname)
echo "[T3] Job ${SLURM_JOB_ID} running on ${NODE}"
echo "[T3] Testing network I/O D-state drain mechanism"
echo "[T3] Attempting TCP connect to 8.8.8.8:81 (BLOCKED — no outbound network on betelgeuse)"
echo "[T3] This will hang until you: scancel ${SLURM_JOB_ID}"
echo "[T3] After scancel: expect 60s SIGTERM→drain sequence"
echo "[T3]"
echo "[T3] To observe in real time:"
echo "[T3]   watch -n2 \"squeue -j ${SLURM_JOB_ID}; sinfo -N -n ${NODE} | tail -1\""
echo "[T3]"
echo "[T3] To cancel and trigger drain:"
echo "[T3]   scancel ${SLURM_JOB_ID}"

# Pure Python: open TCP socket with no timeout.
# This enters uninterruptible D-state on the connect() syscall.
# No CUDA, no GPU, no PyTorch needed.
python3 - <<'PYEOF'
import socket, sys, os, time

print(f"[T3-py] pid={os.getpid()} opening socket...", flush=True)
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setblocking(True)          # no timeout — will block forever
s.settimeout(None)

# 8.8.8.8:81 is Google DNS but port 81 is BLOCKED on betelgeuse.
# The firewall DROPs packets (not REJECT), so connect() hangs
# indefinitely in kernel D-state — exactly as vgg.pth download did.
print(f"[T3-py] connect() to 8.8.8.8:81 (will block until killed)...", flush=True)
s.connect(("8.8.8.8", 81))   # ← hangs here; SIGTERM cannot interrupt
print(f"[T3-py] ERROR: connect() returned — expected to hang!", flush=True)
sys.exit(1)
PYEOF
