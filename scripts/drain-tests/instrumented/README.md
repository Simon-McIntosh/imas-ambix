# Instrumented drain reproducers + observer

The corrected minimal-working-example set for the node-drain RCA
(`docs/rca-node-drain-mechanism-2026-06-02.html`). Supersedes the earlier
`../mwe-*.py` suite, whose conclusions were drawn **without measuring process
state**. Everything here measures it directly.

## Root cause (one sentence)

A SLURM node drains iff a process is in uninterruptible **D-state**
(`TASK_UNINTERRUPTIBLE`) past `UnkillableStepTimeout` (60 s) — a D-state task
ignores SIGTERM **and** SIGKILL. On `98dci4-gpu-0003` that D-state lives in the
**NVIDIA kernel driver** (`os_acquire_rwlock_read/write`, `os_acquire_semaphore`,
`uvm_gpu_retain_by_uuid`), entered when `ncclCommAbort()` → `cudaStreamSynchronize()`
blocks on a stuck GPU operation (NCCL upstream #829).

## The key instrument

`observe_state.sh` + `observer.sbatch` — a **separate-allocation** (CPU-only,
no GPU) SLURM job on the same node, *outside* the workload's cgroup. SLURM does
not isolate the PID namespace, so it reads `/proc/<pid>/stat` (state) and
`/proc/<pid>/wchan` (the kernel function the task sleeps in) for every rank every
0.5 s. Because it is in its own cgroup it **survives** the workload's SIGKILL and
the node drain, capturing the entire unkillable window. Submit it first, wait
until it is `RUNNING`, then submit a workload.

```bash
sbatch observer.sbatch              # CPU-only; co-schedules with a 4-GPU job
# CSV at /work/projects/imas_gpu/logs/drain-observer-<jobid>.csv  (ts,pid,state,wchan,cmd)
# kernel ring buffer (Xid/NVRM) at  drain-observer-<jobid>.dmesg
```

## Reproducers (`repro_abort.py`, `REPRO_MODE=`)

| Mode | What it does | Measured result on a **healthy** GPU |
|------|--------------|--------------------------------------|
| `abort_stuck` | `ncclCommAbort` on an in-flight stuck collective (the production path, NCCL #829) | **transient** D in NVIDIA driver locks → process stays reapable → clean kill, **no drain** |
| `seq_divergence` | 20 rounds of rank-0-extra `all_reduce`, then teardown (replicates 1209813's pattern) | collectives complete → R/S → clean kill, no drain |
| `ddp_mismatch` | mismatched DDP model shapes → verify failure → teardown (replicates 1208980's error) | `destroy` returns in 0.6 s → no wedge, no drain |
| `stuck_collective` | stuck `all_reduce` + watchdog disabled | transient D → clean kill, no drain |

`gpu_sleep_sync.py` — single-GPU, no NCCL: `torch.cuda._sleep(~20yr)` +
`cudaStreamSynchronize()` stays **R-state** (busy-poll, killable). Isolates that
plain stream-sync is *not* the D-state source.

## The central finding — why a drain can't be triggered on a clean node

All five experiments above ran on the shared node and **none drained it** (the
node stayed healthy; admin recovery was never needed). On a healthy GPU the
NVIDIA-driver D-states are **transient** — the lock is released in ms, the task
oscillates `D → R`, and the queued SIGKILL lands → clean kill. A **drain** needs
the D-state to be **permanent**, which needs the GPU itself wedged (stuck kernel /
GSP-firmware hang / a context left dirty by a *previous* unclean teardown). That
is a **cascade** seeded by a prior wedge — not reachable from a clean start. This
is the actual reason the earlier effort could never "trigger the drain".

To run an end-to-end recovery drill you must first genuinely wedge a GPU (admin-
induced fault, or `abort_stuck` on a device already left dirty). Do **not**
brute-force that on the shared node without admin standby.

## Recovery (admin, if a real drain occurs)

```bash
ssh 98dci4-gpu-0003 'ps -eo pid,stat,wchan:32,cmd | grep -E "python|torch"'   # confirm D-state
nvidia-smi -i 0,1,2,3 --gpu-reset      # hangs if a proc still holds the device -> reboot instead
scontrol update nodename=98dci4-gpu-0003 state=resume reason=""   # only after GPUs confirmed clean
```
