#!/bin/bash
# observe_state.sh — separate-allocation drain observer.
# ======================================================
# Runs as its OWN SLURM job (separate cgroup) on the same node as the target
# GPU workload. Because SLURM does not isolate the PID namespace, this process
# can read /proc/<pid>/{status,wchan} of the target's ranks. Critically, it is
# NOT in the target's cgroup, so it SURVIVES the SIGTERM/SIGKILL the target
# receives AND survives the node drain — capturing the full unkillable window
# (the 60 s between SIGKILL and "Kill task failed") that an in-cgroup sidecar
# would miss.
#
# Captures, every INTERVAL seconds, for every python/torchrun process on the
# node (excluding itself):
#   - process state (R/S/D/Z/t) from /proc/<pid>/status
#   - wchan symbol (the kernel function it is sleeping in) from /proc/<pid>/wchan
# Separately streams kernel ring buffer (dmesg -w, NVRM/Xid) and periodic
# nvidia-smi (best-effort; may be empty if this job has no GPU device access).
#
# Output (GPFS, survives drain):
#   $OUT_DIR/drain-observer-<jobid>.csv     process states
#   $OUT_DIR/drain-observer-<jobid>.dmesg   kernel ring buffer
#   $OUT_DIR/drain-observer-<jobid>.smi      nvidia-smi snapshots
set -uo pipefail

OUT_DIR="${OBS_OUT_DIR:-/work/projects/imas_gpu/logs}"
JOB="${SLURM_JOB_ID:-local}"
INTERVAL="${OBS_INTERVAL:-0.5}"
DURATION="${OBS_DURATION:-1800}"
SMI_EVERY="${OBS_SMI_EVERY:-6}"   # nvidia-smi/state-summary cadence in samples

CSV="${OUT_DIR}/drain-observer-${JOB}.csv"
DMESG="${OUT_DIR}/drain-observer-${JOB}.dmesg"
SMI="${OUT_DIR}/drain-observer-${JOB}.smi"

echo "ts,pid,state,wchan,cmd" > "$CSV"
echo "[observer] job=$JOB node=$(hostname) interval=${INTERVAL}s duration=${DURATION}s" >&2
echo "[observer] csv=$CSV" >&2

# Stream kernel ring buffer (timestamps) — captures Xid/NVRM the instant they fire.
( dmesg -w 2>/dev/null || while true; do dmesg 2>/dev/null | tail -5; sleep 2; done ) > "$DMESG" 2>&1 &
DMESG_PID=$!

self=$$
end=$(( $(date +%s) + DURATION ))
i=0
while [ "$(date +%s)" -lt "$end" ]; do
    now=$(date +%s.%N)
    # All python/torchrun/pt_main worker processes on the node.
    for pid in $(pgrep -f 'python|torchrun|pt_main_thread|pt_data' 2>/dev/null); do
        [ "$pid" = "$self" ] && continue
        [ -r "/proc/$pid/status" ] || continue
        state=$(awk '/^State:/{print $2; exit}' "/proc/$pid/status" 2>/dev/null)
        wchan=$(cat "/proc/$pid/wchan" 2>/dev/null | tr -d '\0')
        cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | cut -c1-40)
        [ -n "$state" ] && echo "${now},${pid},${state},${wchan:-?},${cmd}" >> "$CSV"
    done
    # Periodic GPU snapshot (best-effort) + a state summary to stderr for live view.
    if [ $(( i % SMI_EVERY )) -eq 0 ]; then
        {
          echo "=== ${now} ==="
          nvidia-smi --query-gpu=index,utilization.gpu,memory.used,clocks_throttle_reasons.active --format=csv,noheader 2>&1
        } >> "$SMI"
        summ=$(awk -F, -v t="$now" '$1==t{c[$3]++} END{for(s in c) printf "%s:%d ", s, c[s]}' "$CSV" 2>/dev/null)
        echo "[observer ${now}] states: ${summ:-none}" >&2
    fi
    i=$(( i + 1 ))
    sleep "$INTERVAL"
done

kill "$DMESG_PID" 2>/dev/null || true
echo "[observer] done; samples in $CSV" >&2
