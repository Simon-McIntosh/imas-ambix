#!/bin/bash
# preempt_v4flash.sh — idempotent V4-Flash preemption prologue
#
# Stops the ambix-v4flash SLURM job if it is running, then waits until the
# job clears the queue (up to 60 s).  Drops a sentinel file so a companion
# restart function (see bottom of this file) can tell whether *this script*
# cancelled the job.
#
# Usage:
#   source scripts/slurm/preempt_v4flash.sh   # in a job prologue
#   bash   scripts/slurm/preempt_v4flash.sh   # standalone check
#
# Must NOT ssh anywhere — runs on the submitting login node only.
# Idempotent: safe to call multiple times; no-op if V4-Flash is not running.

set -euo pipefail

SENTINEL=/tmp/.v4flash-was-running
JOB_NAME=ambix-v4flash

# ── 1. Locate a running V4-Flash job ───────────────────────────────────────
RUNNING_JOB=$(squeue -u "$USER" --name="$JOB_NAME" --noheader --format="%i" 2>/dev/null | head -1)

if [[ -z "$RUNNING_JOB" ]]; then
    echo "[preempt_v4flash] V4-Flash not running — nothing to stop."
    rm -f "$SENTINEL"
    exit 0
fi

echo "[preempt_v4flash] Found ${JOB_NAME} job ${RUNNING_JOB} — cancelling …"

# ── 2. Cancel the job ──────────────────────────────────────────────────────
scancel "$RUNNING_JOB"
echo "$RUNNING_JOB" > "$SENTINEL"
echo "[preempt_v4flash] scancel issued; sentinel written to $SENTINEL"

# ── 3. Wait for the job to leave the queue (up to 60 s) ────────────────────
WAIT=0
MAX_WAIT=60
INTERVAL=5
while true; do
    STILL_RUNNING=$(squeue -u "$USER" --name="$JOB_NAME" --noheader --format="%i" 2>/dev/null | head -1)
    if [[ -z "$STILL_RUNNING" ]]; then
        echo "[preempt_v4flash] Job ${RUNNING_JOB} has cleared the queue."
        break
    fi
    if (( WAIT >= MAX_WAIT )); then
        echo "[preempt_v4flash] WARNING: job ${RUNNING_JOB} still in queue after ${MAX_WAIT}s — proceeding anyway." >&2
        break
    fi
    echo "[preempt_v4flash] Waiting for job ${RUNNING_JOB} to clear (${WAIT}s / ${MAX_WAIT}s) …"
    sleep "$INTERVAL"
    WAIT=$(( WAIT + INTERVAL ))
done

echo "[preempt_v4flash] V4-Flash preemption complete."

# ── 4. (Commented-out) EpilogScript restart companion ──────────────────────
# Call restart_v4flash from your job's EpilogScript= directive to restart
# V4-Flash after the training job finishes.  V4-Flash restart is currently a
# manual operation — this function is here for future automation only.
#
# restart_v4flash() {
#     if [[ ! -f "$SENTINEL" ]]; then
#         # V4-Flash was not running before this job — do not restart.
#         return 0
#     fi
#     echo "[restart_v4flash] Restarting ${JOB_NAME} …"
#     rm -f "$SENTINEL"
#     # Adjust the profile slug and account as needed.
#     imas-ambix agent serve deepseek-v4-flash
#     echo "[restart_v4flash] Serve job submitted."
# }
