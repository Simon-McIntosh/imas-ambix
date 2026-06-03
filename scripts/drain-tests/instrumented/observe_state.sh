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
# NEW (§3 dual-half capture):
#   - SLURM config snapshot (UnkillableStepTimeout, KillWait) at startup
#   - scontrol node state every 2 s (State + Reason lines)
#   - slurmd/slurmstepd log tail (best-effort; may be empty if not readable)
#   - Cgroup PID enumeration every 1 s with per-PID state+wchan
#
# Output (GPFS, survives drain):
#   $OUT_DIR/drain-observer-<jobid>.csv          process states (existing)
#   $OUT_DIR/drain-observer-<jobid>.dmesg        kernel ring buffer (existing)
#   $OUT_DIR/drain-observer-<jobid>.smi          nvidia-smi snapshots (existing)
#   $OUT_DIR/drain-observer-<jobid>.slurm_config SLURM timeout config (new)
#   $OUT_DIR/drain-observer-<jobid>.node_state   scontrol node state (new)
#   $OUT_DIR/drain-observer-<jobid>.slurmd       slurmd log tail (new, may be empty)
#   $OUT_DIR/drain-observer-<jobid>.cgroup_pids  cgroup PID enumeration (new)
set -uo pipefail

OUT_DIR="${OBS_OUT_DIR:-/work/projects/imas_gpu/logs}"
JOB="${SLURM_JOB_ID:-local}"
INTERVAL="${OBS_INTERVAL:-0.5}"
DURATION="${OBS_DURATION:-1800}"
SMI_EVERY="${OBS_SMI_EVERY:-6}"   # nvidia-smi/state-summary cadence in samples

# TARGET_JOBID: passed as first positional arg (used for cgroup enumeration).
TARGET_JOBID="${1:-}"

CSV="${OUT_DIR}/drain-observer-${JOB}.csv"
DMESG="${OUT_DIR}/drain-observer-${JOB}.dmesg"
SMI="${OUT_DIR}/drain-observer-${JOB}.smi"
SLURM_CFG="${OUT_DIR}/drain-observer-${JOB}.slurm_config"
NODE_STATE="${OUT_DIR}/drain-observer-${JOB}.node_state"
SLURMD_LOG="${OUT_DIR}/drain-observer-${JOB}.slurmd"
CGROUP_PIDS="${OUT_DIR}/drain-observer-${JOB}.cgroup_pids"

# Cadences derived from INTERVAL so they survive an INTERVAL change.
# Use integer arithmetic (bash): compute how many samples per target period.
# NODE_STATE_EVERY: one sample per 2 s
# CGROUP_EVERY:     one sample per 1 s
# We use awk for the float division since bash only does integer math.
NODE_STATE_EVERY=$(awk "BEGIN{v=int(2/${INTERVAL}); print (v<1)?1:v}")
CGROUP_EVERY=$(awk "BEGIN{v=int(1/${INTERVAL}); print (v<1)?1:v}")

echo "ts,pid,state,wchan,cmd" > "$CSV"
echo "timestamp,pid,state,wchan,cmdline" > "$CGROUP_PIDS"
echo "[observer] job=$JOB node=$(hostname) interval=${INTERVAL}s duration=${DURATION}s" >&2
echo "[observer] csv=$CSV" >&2
echo "[observer] target_jobid=${TARGET_JOBID:-<none>}" >&2
echo "[observer] node_state_every=${NODE_STATE_EVERY} samples, cgroup_every=${CGROUP_EVERY} samples" >&2

# ── Stream 1: SLURM config snapshot (one-time at startup) ────────────────────
{
  echo "=== SLURM config snapshot $(date -Is) ==="
  echo ""
  echo "--- scontrol show config (global) ---"
  scontrol show config 2>/dev/null \
    | grep -iE 'UnkillableStep|KillWait|CgroupPlugin|ProctrackType' \
    || echo "(scontrol show config failed or no matching fields)"
  echo ""
  echo "--- scontrol show partition betelgeuse ---"
  scontrol show partition betelgeuse 2>/dev/null \
    | grep -iE 'Kill|UnkillableStep|MaxTime|State|PreemptMode' \
    || echo "(betelgeuse partition query failed or no overrides)"
  echo ""
  echo "--- scontrol show node 98dci4-gpu-0003 (initial) ---"
  scontrol show node 98dci4-gpu-0003 2>/dev/null \
    | grep -iE 'State|Reason|OS|OS=' \
    || echo "(node query failed)"
} > "$SLURM_CFG" 2>&1
echo "[observer] slurm_config written to $SLURM_CFG" >&2

# ── Stream 2: kernel ring buffer (existing) ──────────────────────────────────
( dmesg -w 2>/dev/null || while true; do dmesg 2>/dev/null | tail -5; sleep 2; done ) > "$DMESG" 2>&1 &
DMESG_PID=$!

# ── Stream 3: slurmd/slurmstepd log tail (best-effort) ───────────────────────
SLURMD_LOG_PATH=""
SLURMD_STREAM_PID=""
for candidate in /var/log/slurm/slurmd.log /var/log/slurmd.log /var/log/slurm/slurmstepd.log; do
  if [ -r "$candidate" ]; then
    SLURMD_LOG_PATH="$candidate"
    break
  fi
done

if [ -n "$SLURMD_LOG_PATH" ]; then
  echo "[observer] tailing slurmd log: $SLURMD_LOG_PATH" >&2
  echo "=== slurmd log tail from $SLURMD_LOG_PATH ($(date -Is)) ===" > "$SLURMD_LOG"
  tail -F "$SLURMD_LOG_PATH" >> "$SLURMD_LOG" 2>&1 &
  SLURMD_STREAM_PID=$!
else
  # Try journald as fallback.
  if journalctl -u slurmd -n 1 >/dev/null 2>&1; then
    echo "[observer] slurmd log not directly readable; using journald" >&2
    echo "=== slurmd via journald -f ($(date -Is)) ===" > "$SLURMD_LOG"
    journalctl -u slurmd -f >> "$SLURMD_LOG" 2>&1 &
    SLURMD_STREAM_PID=$!
  else
    echo "[observer] slurmd log not accessible (permission denied on /var/log/slurm/slurmd.log and journald unavailable); stream empty" >&2
    {
      echo "=== slurmd log not accessible ($(date -Is)) ==="
      echo "Tried: /var/log/slurm/slurmd.log, /var/log/slurmd.log, /var/log/slurm/slurmstepd.log"
      echo "journald (journalctl -u slurmd): not accessible"
      echo "Slurmd log is owned by root (mode 0600) — requires root or 'slurm' group membership."
      echo "Stream is intentionally empty; all other observer streams are unaffected."
    } > "$SLURMD_LOG"
    SLURMD_STREAM_PID=""
  fi
fi

# ── Helper: enumerate all PIDs in the target job's cgroup (cgroup v2) ────────
# cgroup v2: PIDs live only in LEAF cgroups (no internal-process rule).
# The job subtree looks like:
#   /sys/fs/cgroup/system.slice/slurmstepd.scope/job_<N>/step_<M>/user/task_0/
# We discover it at runtime via `find` to handle any layout variations.
# Falls back to cgroup v1 glob if v2 path not found.
enumerate_cgroup_pids() {
  local jobid="${1:-}"
  [ -z "$jobid" ] && return

  local pids=()
  local cg_root=""

  # cgroup v2: find the job_<N> directory anywhere under slurmstepd.scope
  local v2_base="/sys/fs/cgroup/system.slice/slurmstepd.scope"
  if [ -d "$v2_base" ]; then
    # Find the job directory (may be nested: job_N or uid_U/job_N)
    while IFS= read -r jobdir; do
      # Recurse into ALL subdirectories and read cgroup.procs from leaves
      while IFS= read -r cgdir; do
        [ -r "${cgdir}/cgroup.procs" ] || continue
        while IFS= read -r p; do
          [ -n "$p" ] && pids+=("$p")
        done < "${cgdir}/cgroup.procs"
      done < <(find "$jobdir" -type d 2>/dev/null)
      # Also read the job dir itself (may be leaf if no sub-steps yet)
      if [ -r "${jobdir}/cgroup.procs" ]; then
        while IFS= read -r p; do
          [ -n "$p" ] && pids+=("$p")
        done < "${jobdir}/cgroup.procs"
      fi
    done < <(find "$v2_base" -type d -name "job_${jobid}" 2>/dev/null)
  fi

  # cgroup v1 fallback: /sys/fs/cgroup/*/slurm/uid_*/job_<N>/
  if [ ${#pids[@]} -eq 0 ]; then
    for cgfile in /sys/fs/cgroup/*/slurm/uid_*/job_${jobid}/cgroup.procs \
                  /sys/fs/cgroup/*/slurm/uid_*/job_${jobid}/*/cgroup.procs; do
      [ -r "$cgfile" ] || continue
      while IFS= read -r p; do
        [ -n "$p" ] && pids+=("$p")
      done < "$cgfile"
    done
  fi

  if [ ${#pids[@]} -eq 0 ]; then
    echo "$(date +%s.%N),<no_pids>,,,cgroup not found or empty for job_${jobid}" >> "$CGROUP_PIDS"
    return
  fi

  local now
  now=$(date +%s.%N)
  for pid in "${pids[@]}"; do
    [ -r "/proc/$pid/status" ] || continue
    local state wchan cmdline
    state=$(awk '/^State:/{print $2; exit}' "/proc/$pid/status" 2>/dev/null)
    wchan=$(cat "/proc/$pid/wchan" 2>/dev/null | tr -d '\0')
    cmdline=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | cut -c1-80)
    echo "${now},${pid},${state:-?},${wchan:-?},${cmdline}" >> "$CGROUP_PIDS"
  done
}

# ── Main observation loop ─────────────────────────────────────────────────────
self=$$
end=$(( $(date +%s) + DURATION ))
i=0
while [ "$(date +%s)" -lt "$end" ]; do
    now=$(date +%s.%N)

    # ── Existing: process states for python/torchrun ranks ────────────────────
    for pid in $(pgrep -f 'python|torchrun|pt_main_thread|pt_data' 2>/dev/null); do
        [ "$pid" = "$self" ] && continue
        [ -r "/proc/$pid/status" ] || continue
        state=$(awk '/^State:/{print $2; exit}' "/proc/$pid/status" 2>/dev/null)
        wchan=$(cat "/proc/$pid/wchan" 2>/dev/null | tr -d '\0')
        cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | cut -c1-40)
        [ -n "$state" ] && echo "${now},${pid},${state},${wchan:-?},${cmd}" >> "$CSV"
    done

    # ── Existing: periodic GPU snapshot + state summary ───────────────────────
    if [ $(( i % SMI_EVERY )) -eq 0 ]; then
        {
          echo "=== ${now} ==="
          nvidia-smi --query-gpu=index,utilization.gpu,memory.used,clocks_throttle_reasons.active --format=csv,noheader 2>&1
        } >> "$SMI"
        summ=$(awk -F, -v t="$now" '$1==t{c[$3]++} END{for(s in c) printf "%s:%d ", s, c[s]}' "$CSV" 2>/dev/null)
        echo "[observer ${now}] states: ${summ:-none}" >&2
    fi

    # ── New: scontrol node state (every 2 s) ──────────────────────────────────
    if [ $(( i % NODE_STATE_EVERY )) -eq 0 ]; then
        node_out=$(scontrol show node 98dci4-gpu-0003 2>/dev/null \
            | grep -iE 'State|Reason' || echo "(node query failed)")
        echo "${now} | ${node_out}" >> "$NODE_STATE"
    fi

    # ── New: cgroup PID enumeration (every 1 s) ───────────────────────────────
    if [ -n "$TARGET_JOBID" ] && [ $(( i % CGROUP_EVERY )) -eq 0 ]; then
        enumerate_cgroup_pids "$TARGET_JOBID"
    fi

    i=$(( i + 1 ))
    sleep "$INTERVAL"
done

# ── Cleanup background processes ─────────────────────────────────────────────
kill "$DMESG_PID" 2>/dev/null || true
[ -n "$SLURMD_STREAM_PID" ] && kill "$SLURMD_STREAM_PID" 2>/dev/null || true
echo "[observer] done; samples in $CSV" >&2
echo "[observer] node_state=$NODE_STATE" >&2
echo "[observer] cgroup_pids=$CGROUP_PIDS" >&2
