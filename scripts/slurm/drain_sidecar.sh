#!/bin/bash
# drain_sidecar.sh — in-job drain-forensics sampler.
# ===================================================
# Launch as ONE background line near the top of any long-running GPU sbatch:
#
#     bash "${REPO_ROOT}/scripts/slurm/drain_sidecar.sh" &
#
# Purpose (settled findings — docs/rca-node-drain-final-2026-06-03.html):
# the 2026-05/06 node drains were environmental (sustained D-state in the
# NVIDIA/CXI/GPFS kernel paths outliving KillWait + UnkillableStepTimeout),
# not code-triggered, and could not be reproduced on demand. The behavioural
# guards were removed; this sidecar is the replacement — if a drain ever
# recurs organically, its log identifies WHICH PID was in D-state and in
# WHICH kernel symbol (wchan) through the kill window, evidence that took a
# full campaign to assemble after the fact.
#
# Design:
#   - Ignores SIGTERM/SIGINT/SIGHUP/SIGQUIT. When SLURM kills the job
#     (scancel or --time expiry) every process in the cgroup gets SIGTERM,
#     then SIGKILL after KillWait (30 s). Ignoring SIGTERM keeps the sampler
#     alive through exactly the window where an unkillable process shows
#     itself. SIGKILL (unblockable) still reaps it at the end — by then the
#     evidence is on GPFS.
#   - Samples once per second: every PID in this job's cgroup with state +
#     wchan + cmdline. Periodically: nvidia-smi summary and node
#     State/Reason. Streams kernel NVRM/Xid/fabric lines if dmesg is
#     readable.
#   - Cost: one bash loop on a CPU core and a few KB/min of GPFS appends.
#     Because it ignores SIGTERM, a cleanly-finishing job carries a teardown
#     tail of up to KillWait (~30 s) while slurmstepd escalates to SIGKILL —
#     negligible against multi-hour GPU runs, and that window is precisely
#     the evidence we want when a kill does NOT succeed.
#
# Output (survives the job):
#   $OUT_DIR/drain-sidecar-<jobid>.csv     per-second cgroup PID states
#                                          (size-rotated → .csv.1/.csv.2)
#   $OUT_DIR/drain-sidecar-<jobid>.smi     periodic nvidia-smi snapshots
#   $OUT_DIR/drain-sidecar-<jobid>.node    periodic node State/Reason
#   $OUT_DIR/drain-sidecar-<jobid>.dmesg   filtered kernel ring buffer
set -u

# Forensic immunity: survive the job's SIGTERM so the kill window is sampled.
trap '' TERM INT HUP QUIT

OUT_DIR="${SIDECAR_OUT_DIR:-/work/projects/imas_gpu/logs}"
JOB="${SLURM_JOB_ID:-local}"
INTERVAL="${SIDECAR_INTERVAL:-1}"
SMI_EVERY="${SIDECAR_SMI_EVERY:-10}"    # nvidia-smi cadence, in samples
NODE_EVERY="${SIDECAR_NODE_EVERY:-30}"  # scontrol node-state cadence, in samples

# The per-second CSV is append-only and grows ~80 MB/day, so a multi-day
# serving job would fill GPFS without a cap. Rotate by size: when the CSV
# passes MAX_BYTES it is rolled to .1/.2 (BACKUPS kept) and a fresh file
# started. A drain wedges processes in D-state in the FINAL minutes before
# the kill, so the newest file always holds the forensic window — rotation
# discards only stale steady-state history. (Rotation is CSV-only: the .dmesg
# stream is held open by a background pipe and cannot be rolled with mv; it is
# filtered and stays small.)
MAX_BYTES="${SIDECAR_MAX_BYTES:-52428800}"   # 50 MB per CSV before rotation
BACKUPS="${SIDECAR_BACKUPS:-2}"              # rolled CSV copies to retain

mkdir -p "$OUT_DIR"
CSV="${OUT_DIR}/drain-sidecar-${JOB}.csv"
SMI="${OUT_DIR}/drain-sidecar-${JOB}.smi"
NODE="${OUT_DIR}/drain-sidecar-${JOB}.node"
DMESG="${OUT_DIR}/drain-sidecar-${JOB}.dmesg"

echo "ts,pid,state,wchan,cmd" > "$CSV"
echo "[sidecar] job=$JOB node=$(hostname) interval=${INTERVAL}s csv=$CSV" >&2

# ── Locate this job's cgroup (v2) from our own membership ────────────────────
# /proc/self/cgroup → 0::/system.slice/slurmstepd.scope/job_<N>/step_batch/...
# Truncate at job_<N> so all steps (batch, torchrun ranks, srun steps) are
# enumerated. Empty if not under SLURM — then fall back to pgrep.
SELF_CG="$(awk -F:: '/^0::/{print $2}' /proc/self/cgroup 2>/dev/null)"
JOB_CG=""
case "$SELF_CG" in
  */job_*) JOB_CG="/sys/fs/cgroup$(echo "$SELF_CG" | sed -E 's#(/job_[0-9]+).*#\1#')" ;;
esac
echo "[sidecar] job_cgroup=${JOB_CG:-<none — pgrep fallback>}" >&2

list_pids() {
  if [ -n "$JOB_CG" ] && [ -d "$JOB_CG" ]; then
    find "$JOB_CG" -name cgroup.procs -readable 2>/dev/null -exec cat {} + | sort -un
  else
    pgrep -f 'python|torchrun|pt_main_thread' 2>/dev/null
  fi
}

# Roll $CSV when it exceeds MAX_BYTES: shift .1→.2…→.BACKUPS, move the live
# file to .1, and re-seed a fresh header. Safe because each sample writes with
# a fresh `>>` (the file is reopened per append), so a new CSV is created on
# the next iteration. A no-op while under the cap.
maybe_rotate_csv() {
  local sz n
  [ -f "$CSV" ] || return 0
  sz=$(stat -c %s "$CSV" 2>/dev/null || echo 0)
  [ "$sz" -gt "$MAX_BYTES" ] || return 0
  n="$BACKUPS"
  while [ "$n" -gt 1 ]; do
    [ -f "${CSV}.$((n - 1))" ] && mv -f "${CSV}.$((n - 1))" "${CSV}.${n}"
    n=$((n - 1))
  done
  mv -f "$CSV" "${CSV}.1"
  echo "ts,pid,state,wchan,cmd" > "$CSV"
  echo "[sidecar] rotated CSV at ${sz} bytes (keeping ${BACKUPS} backups)" >&2
}

# ── Kernel ring buffer (best-effort; needs dmesg read permission) ────────────
if dmesg -T 2>/dev/null | tail -1 >/dev/null 2>&1; then
  ( dmesg -wT 2>/dev/null | grep --line-buffered -iE 'NVRM|Xid|cxi|mmfs|gpfs|hung|blocked' ) >> "$DMESG" &
  DMESG_PID=$!
else
  echo "[sidecar] dmesg not readable — kernel stream disabled" > "$DMESG"
  DMESG_PID=""
fi

# ── Main loop — runs until SIGKILL at job teardown ───────────────────────────
self=$$
i=0
while :; do
  now=$(date +%s.%N)

  for pid in $(list_pids); do
    [ "$pid" = "$self" ] && continue
    [ -r "/proc/$pid/status" ] || continue
    state=$(awk '/^State:/{print $2; exit}' "/proc/$pid/status" 2>/dev/null)
    wchan=$(tr -d '\0' < "/proc/$pid/wchan" 2>/dev/null)
    cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | cut -c1-60)
    [ -n "$state" ] && echo "${now},${pid},${state},${wchan:-?},${cmd}" >> "$CSV"
  done

  if [ $(( i % SMI_EVERY )) -eq 0 ]; then
    {
      echo "=== ${now} ==="
      nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader 2>&1
    } >> "$SMI"
  fi

  if [ $(( i % NODE_EVERY )) -eq 0 ]; then
    node_out=$(scontrol show node "$(hostname)" 2>/dev/null \
      | grep -iE 'State=|Reason=' | tr '\n' ' ')
    echo "${now} | ${node_out:-(scontrol unavailable)}" >> "$NODE"
    maybe_rotate_csv
  fi

  i=$(( i + 1 ))
  sleep "$INTERVAL"
done
