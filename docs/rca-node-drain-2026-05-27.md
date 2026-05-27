# RCA — `98dci4-gpu-0003` node drain, 2026-05-27 12:04:46

**Severity:** node DRAINED (Group A H200 reservation unusable until admin resume)
**Reporter:** Simon McIntosh (simon.mcintosh@iter.org)
**Verdict:** **Self-inflicted.** Our hung GPU job could not be killed; SLURM
drained the node. **No other-user / Group B involvement** (proven below).

## One-line cause

Our rbb bulk-encode job `1208195` deadlocked (prefetch producer/consumer),
leaving GPU processes stuck in uninterruptible state. On `scancel`, they did
not die within SLURM's `UnkillableStepTimeout` (~60 s), so slurmd marked the
step **"Kill task failed"** and auto-drained the node.

## Forensic timeline (all UTC+2, from `sacct`)

| Time | Event | Evidence |
|---|---|---|
| 11:34:29 | `1208195` (4-shard rbb encode, **prefetch on**, batch 32) starts | `sacct` Start |
| 11:38-11:41 | producing tokens ~15/min (prefetch validated alive) | token count 1910→1941 |
| ~11:55-12:01 | run deadlocks — token count frozen at **2119** | 0 delta over 30 s, twice |
| 12:01-12:03 | I `scancel 1208195` (diagnosed hung) | command log |
| **12:03:45** | job marked `CANCELLED` | `sacct` End (`1208195_0`) |
| 12:03:45→12:04:46 | batch step does **not** die on SIGTERM (61 s) | `.batch` ExitCode **0:15** (SIGTERM), End 12:04:46 |
| **12:04:46** | slurmd: **"Kill task failed"** → node `DRAIN` | `scontrol show node` Reason `[root@2026-05-27T12:04:46]` |

The 61 s gap ≈ SLURM's default `UnkillableStepTimeout` (60 s): after the kill
signal, slurmd waited 60 s for the step to exit; it didn't (stuck in
uninterruptible `D` state on a wedged CUDA/IO call), so it drained the node.

## Did we cause it? Yes. Is Group B involved? No.

```
# ALL jobs on 98dci4-gpu-0003, 2026-05-27 00:00–12:10, all users:
#   → every job is user=mcintos, account=grpa (Group A).
# Non-grpa jobs on the node today:           NONE
# Group B (gpu_0003_grpB) jobs today:        NONE
```

Today's drain is **entirely traceable to our `1208195`** — the drain timestamp
(12:04:46) matches our cancelled step's kill-failure to the second, and no
other user had a job on the node at any point today.

## Contrast with yesterday's drain (2026-05-26 18:18:49)

Yesterday's Reason was **"Kill task failed : Not responding"** (note the extra
"Not responding") and it coincided with Group B user `murawan` coming online +
CUDA-init failures (`cuInit rc=3`) across the node. That one had a node-level
health-check ("Not responding") component plausibly linked to Group A/B cgroup
contention wedging the CUDA driver — a *different, partly-external* signature.
**Today's has no "Not responding" and no Group B** — it is purely our
unkillable process.

## Why the process was unkillable (mechanism)

1. The prefetch double-buffer (commit `4f7ab48`) uses a producer thread pool
   feeding a bounded queue consumed by the main thread. If a producer raises on
   a bad shot and dies without signalling, the consumer blocks forever on
   `queue.get()` — **deadlock**.
2. With the daemon mid-CUDA call and threads blocked, the process sits in
   uninterruptible kernel sleep (`D` state). `SIGTERM`/`SIGKILL` cannot reap a
   `D`-state process until the kernel call returns — which never happens for a
   wedged CUDA/GPFS operation.
3. SLURM's `UnkillableStepTimeout` expires → "Kill task failed" → drain.

This is the **same class** as yesterday (un-reapable GPU process → drain),
even though yesterday had an additional external trigger.

## How we stop doing this (prevention — already in motion)

1. **Remove the deadlock sources.** The new `stream_encode.py` (commit
   `908b452`) has **no prefetch producer/consumer threads and no subprocess
   daemon** — it loads the model in-process and streams via a torch
   `DataLoader`. Both of today's and yesterday's failure surfaces are gone.
2. **Graceful SIGTERM handler** in the encode driver: on cancel, cleanly stop
   the DataLoader workers + flush the writer so the step exits in <5 s, well
   under `UnkillableStepTimeout`. (TODO on `stream_encode.py` before production.)
3. **Per-shot watchdog timeout**: abort a single shot that exceeds N× the median
   encode time rather than hanging the run. (TODO.)
4. **Do not `scancel` a CUDA-wedged job and assume clean teardown** — detect the
   hang early (token-rate watchdog) and let the job exit itself; a clean exit
   does not drain the node, an unkillable kill does.
5. **Roll back the prefetch path**: until (2)+(3) land, run the legacy encode
   with `--no-prefetch` (proven stable for the 1,847 + 272 shots it completed).

## Admin instructions (in order)

1. **Check for orphaned/stuck processes first** (the cards still showed 5–22 GB
   held at 0 % util after the drain — un-reaped CUDA contexts):
   ```
   ssh 98dci4-gpu-0003   # or via a maintenance reservation
   nvidia-smi                                   # any processes listed?
   ps -eo pid,stat,user,cmd | grep -E "worker.py|encode_one_shard|magvit|python" | grep -v grep
   #   look for state 'D' (uninterruptible) or 'Z' (zombie) owned by mcintos
   ```
2. **If stuck processes / held GPU memory remain** (likely): a plain resume will
   hand the next job a dirty GPU. Clear it:
   ```
   nvidia-smi --gpu-reset          # requires no processes attached; per-GPU if needed
   # if --gpu-reset fails because a process is wedged → reboot the node
   ```
3. **Resume the node** once GPUs are confirmed clean (0 processes, 0 MiB):
   ```
   scontrol update nodename=98dci4-gpu-0003 state=resume reason=""
   ```
4. **Consider** raising `UnkillableStepTimeout` for this partition (e.g. 120 s)
   if long CUDA teardowns are expected — though the real fix is our side (above).

## Status of our queued work (auto-runs on resume)

- `1208239` (stream byte-diff vs live) + `1208235` (200-shot util/HBM/throughput)
  are `PD`, gated on the node; results auto-write to
  `…/mast-tokens/v1/_stream_validation/RESULTS.txt`.
- 2,119 rbb shots already tokenised are preserved (skip-existing).
- Before re-running production we will (a) add the SIGTERM handler + watchdog,
  and (b) switch to `stream_encode.py`, so a future hang exits cleanly instead
  of draining the node.

— Simon
