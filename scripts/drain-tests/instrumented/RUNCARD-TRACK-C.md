# RUNCARD — Track C: FUSE-freeze D-state wedge

**Purpose:** Warm-up run for the coordinated drain window.
Uses NO GPU. Validates the §3 observer harness, measures the real
`UnkillableStepTimeout` / "giving up after N sec" value, rehearses the admin
resume procedure — all before spending GPU resources on Track A.

---

## Prerequisites

- [ ] Admin on standby (phone / chat) with resume commands copy-pasted and ready
- [ ] Observer harness (`observer.sbatch`) tested and confirmed working
- [ ] Node confirmed **IDLE+RESERVED** (not DRAIN, not already allocated):
  ```bash
  scontrol show node 98dci4-gpu-0003 | grep -E "State|Reason|CfgTRES"
  ```
- [ ] Log directory exists:
  ```bash
  mkdir -p /work/projects/imas_gpu/logs
  ```
- [ ] Stop-files directory exists (observer uses it):
  ```bash
  mkdir -p /work/projects/imas_gpu/stops
  ```

---

## Step-by-step in-window sequence

### 1. Start the observer job first

The observer is CPU-only and co-schedules alongside Track C. It must be
**RUNNING** before Track C is submitted so it captures the full D-state window.

```bash
cd /home/ITER/mcintos/Code/imas-ambix/scripts/drain-tests/instrumented/
OBS_JOBID=$(sbatch --parsable observer.sbatch)
echo "Observer JOBID: ${OBS_JOBID}"
# Wait until RUNNING (typically 5-10 s):
watch -n2 "squeue -j ${OBS_JOBID} -o '%-10i %-12T %-12R'"
```

### 2. Submit Track C

```bash
TRACK_C_JOBID=$(sbatch --parsable track_c_cgroup_wedge.sbatch)
echo "Track C JOBID: ${TRACK_C_JOBID}"
echo "Log: /work/projects/imas_gpu/logs/track-c-${TRACK_C_JOBID}.out"
```

### 3. Tail the Track C log for early diagnostics

```bash
tail -f /work/projects/imas_gpu/logs/track-c-${TRACK_C_JOBID}.out
```

Key lines to watch for:
- `FUSE daemon PID: <N>` — daemon launched
- `WEDGE ACTIVE — reader <N> in D-state` — wedge confirmed
- `Sleeping forever` — main loop entered; node will drain at `--time` expiry
- `FATAL:` / `ERROR:` — wedge setup failed; see §Failure modes below

### 4. Optionally probe reader D-state from observer output

While Track C is running (before drain):
```bash
# From the login node — observer is on the same node:
squeue -j ${OBS_JOBID} -o '%N'    # get compute node name
# Then check (if you have SSH to compute nodes from login):
ssh 98dci4-gpu-0003 "cat /proc/<READER_PID>/wchan; echo; cat /proc/<READER_PID>/stat" 2>/dev/null
# Expected wchan: fuse_simple_request  OR  fuse_dev_do_read
```

### 5. Wait for the drain

Timeline from job start:
- **t+0 s** — squashfuse mounts, reader starts
- **t+0.5 s** — FUSE daemon SIGSTOP'd (frozen mid-request)
- **t~1 s** — reader enters D-state
- **t+4:00** — SLURM `--time` fires → SIGTERM to job step
- **t+4:00–5:00** — KillWait → SIGKILL to all cgroup processes
- **t+~5:00** — slurmstepd "giving up after N sec" (N = `UnkillableStepTimeout`, ~60 s)
- **t+~5:30–6:00** — node DRAIN confirmed

Watch for drain:
```bash
watch -n5 "scontrol show node 98dci4-gpu-0003 | grep -E 'State|Reason'"
# Expected: State=drain + Reason=Kill task failed...
```

### 6. Confirm drain and collect Reason= string

```bash
scontrol show node 98dci4-gpu-0003 | grep -E "State|Reason"
# Copy the full Reason= string — it contains the timestamp and "giving up after N"
```

### 7. Collect observer logs

```bash
OBS_LOG=/work/projects/imas_gpu/logs/drain-observer-${OBS_JOBID}
ls -lh ${OBS_LOG}*
# Files:
#   drain-observer-<OBS_JOBID>.out       — observer stdout
#   drain-observer-<OBS_JOBID>.err       — observer stderr
#   drain-observer-<OBS_JOBID>.csv       — pid,state,wchan,cmd time-series
#   drain-observer-<OBS_JOBID>.node_state — scontrol node state snapshots
#   drain-observer-<OBS_JOBID>.cgroup_pids — cgroup membership over time
```

### 8. Extract key measurements

```bash
# UnkillableStepTimeout (from .slurm_config captured by observer):
cat ${OBS_LOG}.slurm_config 2>/dev/null || \
    ssh 98dci4-gpu-0003 'scontrol show config | grep -i unkillable' 2>/dev/null

# Reader D-state confirmation:
grep "fuse" ${OBS_LOG}.csv | head -20

# Drain timestamp from node state:
cat ${OBS_LOG}.node_state | grep -i drain | head -5
```

---

## Admin resume steps (NO gpu-reset needed — Track C uses no GPU)

> These steps require admin access. Read them aloud to the admin on standby.

```bash
# 1. Confirm no stuck processes (no GPU context to worry about):
ssh 98dci4-gpu-0003 'ps aux | grep -E "track_c|squashfuse|dd" | grep -v grep'
# Expected: empty (all processes reaped after the drain, just took extra time)

# 2. Resume the node:
scontrol update nodename=98dci4-gpu-0003 state=resume reason=""

# 3. Confirm back to IDLE+RESERVED:
scontrol show node 98dci4-gpu-0003 | grep -E "State|Reason"
# Expected: State=idle+reserved, Reason="" or no entry
```

> **Note:** No `nvidia-smi --gpu-reset` is needed. Track C requested no GPU
> (`--gres` omitted from the sbatch). SLURM has a separate reservation for the
> GPU resources; Track C's cgroup only contained the CPU/memory allocation.

---

## Success criteria

All four must be met for Track C to be declared a clean success:

- [ ] **D-state confirmed in observer CSV**: at least one sample shows the reader
  in state `D` with `wchan` containing `fuse_simple_request` or `fuse_dev_do_read`
- [ ] **Node drained**: `scontrol show node` shows `State=drain` with `Reason=Kill task failed...`
- [ ] **`UnkillableStepTimeout` value recorded**: extracted from `.slurm_config`
  or `scontrol show config` output (expected ~60 s based on SLURM defaults)
- [ ] **Node resumed cleanly**: returned to `idle+reserved` within 2 min of
  admin running `scontrol update ... state=resume`

---

## Failure modes and responses

| Symptom | Cause | Response |
|---|---|---|
| Script outputs `FATAL: no FUSE tool available` | `squashfuse`/`mksquashfs` absent on RHEL 10 compute node | Node NOT drained; no resume needed. File a ticket: install `squashfs-tools` + `squashfuse` on `98dci4-gpu-0003`. |
| Script outputs `D-state may not have been achieved` | Daemon exited on SIGSTOP (unlikely but possible with some FUSE versions) | Node probably NOT drained. Check `squeue` — if Track C job ended cleanly, confirm no drain and re-examine strategy. |
| Track C exits cleanly (no drain) after timeout | squashfuse served reads from cache before SIGSTOP | Examine observer CSV for reader state. If state was always `S`/`R` (not `D`), the freeze did not produce uninterruptible wait. Escalate to Track A. |
| Node already in bad state before test | Prior dirty GPU context | **ABORT.** Do not submit Track C. Report to admin for recovery first. |
| Observer job fails to start | Scheduling / reservation issue | Fix observer first; do not submit Track C without the harness running. |

---

## FUSE availability note (RHEL 10 vs RHEL 9)

> **Important:** `squashfuse` and `mksquashfs` were confirmed present on the
> **RHEL 9 login node** (packages: `squashfuse-0.1.104`, `squashfs-tools`).
> The compute node `98dci4-gpu-0003` runs **RHEL 10.1** — package availability
> is unverified (prepare-only constraint prevented `srun` probing).
>
> The script detects tools at runtime and exits 1 cleanly if none are found.
> If Track C hits `FATAL: no FUSE tool available`, the operator should:
> ```bash
> srun --partition=betelgeuse --reservation=gpu_0003_grpA --account=grpa \
>      --cpus-per-task=1 --mem=1G --time=00:02:00 \
>      bash -c 'which squashfuse; which mksquashfs; which sshfs; which bindfs; \
>               cat /etc/os-release | grep VERSION'
> ```
> and then request the missing packages from SDCC admins.

---

## Key references

- `observer.sbatch` + `observe_state.sh` — §3 capture harness
- `docs/rca-node-drain-mechanism-2026-06-02.html` — root-cause analysis
- AGENTS.md §2a — drain prevention rules and recovery procedures
- AGENTS.md §2a-cancel — STOP-FILE contract (deliberately NOT used in Track C)
