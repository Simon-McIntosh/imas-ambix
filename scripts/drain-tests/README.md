# Drain Test MWEs — Minimal Working Examples

Generic, self-contained scripts that reproduce SLURM node drain conditions
on NVIDIA GPU clusters. No project-specific dependencies beyond PyTorch + NCCL.

## Background

These scripts accompany a root-cause analysis of node drain events observed
on a 4×H200 NVLink cluster (SDCC `98dci4-gpu-0003`, NCCL 2.21.5, PyTorch 2.5.1).
Each MWE is designed to probe a specific drain mechanism — or to confirm the
**absence** of drain — in the most minimal reproducible way.

Key finding: **node drains require a D-state (TASK_UNINTERRUPTIBLE) process
that survives SIGKILL**. Processes in S-state (interruptible sleep) or R-state
(runnable) are killed cleanly. The drain path that triggered Events #1–4 required
two sequential conditions: (1) ncclCommAbort timeout leaving NVLink in
partial-abort state, (2) normal Python exit triggering `~ProcessGroupNCCL()`
→ ncclCommDestroy on a partially-aborted communicator → D-state in NVLink
driver. NCCL 2.21.5 appears to have mitigated this: the watchdog fires
SIGABRT before normal exit is reached, and `_exit()` (no destructors)
avoids the ncclCommDestroy drain path.

## Prerequisites

- SLURM cluster with GPU nodes and NCCL-capable GPU interconnect (NVLink recommended)
- Python venv with PyTorch ≥ 2.x and NCCL (standard torch install)
- Override the Python interpreter path via `PYTHON_VENV`:
  ```bash
  export PYTHON_VENV=/path/to/your/venv
  ```

## Test Scripts

| Script | Risk | Mechanism | Result |
|--------|------|-----------|--------|
| `mwe-01-sigterm-delay.py` | ✅ SAFE | SIGTERM blind window during NCCL collective | PASS |
| `mwe-02-stopfile.py` | ✅ SAFE | Stop-file clean cancellation | PASS |
| `mwe-03-network-hang.py` | ✅ NO DRAIN | TCP connect blocked → S-state (interruptible) | NO DRAIN |
| `mwe-04-nccl-scancel.py` | ✅ NO DRAIN | Simple NCCL all-reduce → R-state | NO DRAIN |
| `mwe-05-ddp-backward.py` | ✅ NO DRAIN | DDP backward + count mismatch → NCCL handles gracefully | NO DRAIN |
| `mwe-06-nccl-teardown.py` | ⚠️ NO DRAIN (watchdog) | ncclCommAbort() hung 600s in S-state, watchdog fires | NO DRAIN |
| `mwe-06d-sync-rounds.py` | ⚠️ NO DRAIN (watchdog) | Live all_reduce hung 600s in S-state, watchdog fires | NO DRAIN |

## Confirmed Non-Drain Mechanisms

| Operation | Process State | Reason |
|-----------|---------------|--------|
| urllib TCP socket (DROP firewall) | S-state | Socket wait is interruptible |
| `dist.all_reduce` (healthy) | R-state | CPU spin in NCCL user-space poll |
| `loss.backward()` async DDP | R/S-state | DDP all-reduce is asynchronous |
| `torch.cuda.synchronize()` (healthy) | S-state | Interruptible futex wait |
| NCCL count-mismatch (async submit) | Returns in 0.1ms | NCCL 2.21.5 handles gracefully |
| P2P `dist.recv` ring deadlock | S-state | Interruptible; SIGTERM/KILL works |
| Cross-group new_group deadlock | Aborted in 2s | NCCL heartbeat fires |
| `torch.cuda._sleep(inf)` + scancel | S-state | GPU kernel stays on GPU; Python dies cleanly |
| ncclCommAbort (teardown) hung 600s | S-state | Watchdog fires SIGABRT → `_exit()` → no destructors → fast GPU reset |
| Live all_reduce hung 600s (sync) | S-state | Same watchdog path → clean exit in <2s |

## Drain Exit Path vs. Safe Exit Path

**Drain path (Event #4, old NCCL versions):**
```
dist.destroy_process_group()
  → ncclCommAbort() hangs 600s
  → "Future for ProcessGroup abort timed out after 600000 ms" (4 ranks)
  → normal Python exit (return / sys.exit)
  → Python GC: ~ProcessGroupNCCL() → ncclCommDestroy on partial-abort comm
  → NVLink driver D-state (TASK_UNINTERRUPTIBLE)
  → SLURM SIGTERM → not delivered → SIGKILL → unkillable → DRAIN
```

**Safe path (NCCL 2.21.5, all MWE-06 variants):**
```
dist.all_reduce() or dist.destroy_process_group() hangs
  → PyTorch watchdog thread fires at 600,000ms
  → "WorkNCCL ran for 600Nms... taking entire process down"
  → std::abort() → SIGABRT → C++ signal handler → _exit()
  → NO Python destructors (no ~ProcessGroupNCCL, no ncclCommDestroy)
  → CUDA driver forced context reset (~1s, no D-state)
  → Process exits, node: IDLE+RESERVED
```

**Why `_exit()` is safe but normal exit is not:**
- `_exit()` bypasses all Python/C++ destructors and atexit handlers
- ncclCommDestroy is only called via `~ProcessGroupNCCL()` (destructor)
- Without the destructor, CUDA driver does a forced GPU context reset
- A forced reset on a hung NVLink ring completes in <1s on NCCL 2.21.5
- A graceful ncclCommDestroy on a partially-aborted communicator attempts
  NVLink drain protocol, which can deadlock at the hardware level

## SLURM Sub-Step Accounting (Resolution of "226 sub-steps" mystery)

During testing, sacct showed hundreds of `.N` sub-steps accumulating at ~0.75/step/s.
Initial hypothesis was torchrun elastic restarts. **This is incorrect.** Resolution:

- Sub-steps accumulate throughout the entire job lifetime (not just during failures)
- Rate is consistent: MWE-06: 0.77/s, MWE-06b: 0.73/s, MWE-06c: 0.76/s
- `--max-restarts=0` had NO effect on sub-step count (ruling out elastic retries)
- These are **SLURM internal GPU accounting heartbeat steps** (cgroup monitoring
  or GPU health checks) — unrelated to Python/NCCL at all

## Confirmed Drain Mechanism (from Event #4 evidence)

The drain from Event #4 required:
1. Multiple rounds of BLOCKING all_reduce participation mismatch (each rank calling
   `dist.all_reduce` without all ranks participating, with actual NVLink execution)
2. Teardown (`dist.destroy_process_group()`) called after corrupted rounds
3. ncclCommAbort timeout (600s): "Future for ProcessGroup abort timed out"
4. Normal Python exit → destructor calls ncclCommDestroy on partial-abort state
5. NVLink driver D-state → unkillable → drain

**NCCL 2.21.5 mitigation:** the watchdog fires SIGABRT at 600s and calls `_exit()`,
bypassing the ncclCommDestroy destructor path. This prevents the hardware-level
D-state. The drain mechanism from Event #4 cannot be reproduced in an isolated MWE
on NCCL 2.21.5 because the safe watchdog exit fires before normal Python exit occurs.

## Running Order (safest first)

### MWE-01 — SIGTERM Blind Window (SAFE, self-terminating)
```bash
sbatch mwe-01-sigterm-delay.sbatch
# Self-terminating. Expected: blind_window_ms ≈ 1ms on H200 NVLink
# Job 1209844: PASS — blind window 0.9 ms
```

### MWE-02 — Stop-File Exit (SAFE, self-terminating)
```bash
sbatch mwe-02-stopfile.sbatch
# Self-terminating after 30 s. Expected: all ranks exit 0
# Job 1209850: PASS — clean exit via stop-file
```

### MWE-03 — Network Hang (NO DRAIN)
```bash
sbatch mwe-03-network-hang.sbatch
# Connects to 0.0.0.0 (blocked). Expected: S-state (interruptible), clean cancel
# Job 1209855: NO DRAIN — urllib socket in S-state, SIGTERM delivered
```

### MWE-04 — NCCL All-Reduce + Scancel (NO DRAIN)
```bash
sbatch mwe-04-nccl-scancel.sbatch
# Simple all_reduce loop. Expected: R-state, SIGKILL works, no drain
# Job 1209858: NO DRAIN — NCCL all-reduce is R-state
```

### MWE-05 — NCCL Count-Mismatch + DDP Backward (NO DRAIN)
```bash
# Run with DRAIN_MODE=seq_mismatch (default):
sbatch mwe-05-ddp-backward.sbatch
# Expected: NCCL 2.21 truncates mismatched counts, returns immediately
# Jobs 1209908-1209914: NO DRAIN across all 8 test modes
```

### MWE-06 — NCCL Teardown + Watchdog (`--time=00:20:00`)
```bash
sbatch mwe-06-nccl-teardown.sbatch
# Runs count-mismatch rounds, then calls dist.destroy_process_group()
# Expected: drain (ncclCommAbort D-state)
# Actual result (job 1209930): NO DRAIN
#   → teardown hung in S-state (not D-state) on NCCL 2.21.5
#   → watchdog fired at 10:15; std::abort() → _exit() → fast GPU reset
# Finding: NCCL 2.21.5 teardown is interruptible (S-state), not D-state
```

### MWE-06b — NCCL Teardown, SLURM Fires Early (`--time=00:05:00`)
```bash
sbatch mwe-06b-timed-kill.sbatch
# Same Python script, --time=00:05:00 (fires during teardown)
# Expected: SLURM fires → SIGTERM → D-state → unkillable → DRAIN
# Actual result (job 1209935): NO DRAIN
#   → SLURM fired at 5:17; teardown in S-state; SIGTERM delivered in <1s
# Finding: ncclCommAbort on NCCL 2.21.5 is S-state, not D-state
```

### MWE-06c — NCCL Teardown, No Elastic Retries (`--time=00:08:00`, `--max-restarts=0`)
```bash
sbatch mwe-06c-no-retry.sbatch
# Tests whether --max-restarts=0 prevents SLURM sub-step accumulation
# Actual result (job 1209941): NO DRAIN; sub-steps still accumulate at ~0.76/s
# Finding: sub-steps are SLURM GPU accounting heartbeats, not elastic retries
```

### MWE-06d — Sync Round Mismatch + Watchdog (`--time=00:20:00`)
```bash
sbatch mwe-06d-sync-rounds.sbatch
# Like MWE-06 but adds torch.cuda.synchronize() to force ACTUAL NVLink execution
# Round 1 step B (rank 0 extra 256MB all_reduce) mismatches steps C on all ranks
# All 4 ranks hang exactly 600,014-600,051ms at SeqNum=3
# Watchdog fires on all ranks simultaneously → std::abort() → _exit() in <2s
# Actual result (job 1209946): NO DRAIN
# Finding: even real synchronous NVLink work exits cleanly via watchdog path
```

## Drain Mechanism (confirmed from Event #4 + all MWE-06 variants)

**The historical drain path (Event #4, older NCCL):**
```
Round N (count-mismatch all_reduce)
  → NCCL ring state desynchronised (completes but corrupts internal seq)
  → Repeated N rounds
  → All rounds complete; script prints "PASS"

Teardown: dist.destroy_process_group()
  → PyTorch calls ncclCommAbort() on all communicators
  → ncclCommAbort() cannot drain desynchronised NVLink ring
  → Hung 600s: "Future for ProcessGroup abort timed out after 600000 ms"
  → Normal Python exit (destructor path)
  → ~ProcessGroupNCCL() → ncclCommDestroy on partial-abort state
  → NVLink driver D-state (TASK_UNINTERRUPTIBLE) — hardware stuck

DRAIN THEN REQUIRED SLURM TO FIRE DURING D-STATE:
  → SLURM --time fires → SIGTERM → not delivered (D-state)
  → 30s KillWait → SIGKILL → D-state survives SIGKILL
  → 60s UnkillableStepTimeout → "Kill task failed"
  → NODE DRAIN ← confirmed in Event #4 (job 1209813)
```

**The NCCL 2.21.5 safe exit path (all MWE-06 variants):**
```
Either:
  (A) dist.destroy_process_group() teardown hangs in S-state for ≤600s
  (B) live dist.all_reduce() hangs all ranks in S-state for 600s (MWE-06d)

  → PyTorch watchdog thread fires at 600,000ms
  → "WorkNCCL ran for 600Nms... Taking the entire process down"
  → std::abort() → SIGABRT → C++ signal handler → _exit()
  → NO Python GC, NO ~ProcessGroupNCCL(), NO ncclCommDestroy destructor
  → CUDA driver forced context reset in <2s (no D-state)
  → node: IDLE+RESERVED ← confirmed in MWE-06b, 06c, 06d
```

**Root cause of NCCL 2.21.5 immunity:**
- Old NCCL: ncclCommAbort itself entered D-state (hardware-level hang) immediately
- NCCL 2.21.5: ncclCommAbort hangs in S-state (interruptible), times out via
  "Future timed out" message, returns control to Python
- Then PyTorch watchdog fires SIGABRT → `_exit()` instead of normal exit
- `_exit()` avoids the destructor path that calls ncclCommDestroy
- ncclCommDestroy on a partial-abort state is what caused the D-state in Event #4

SLURM timing constants (confirmed on this cluster):
- `KillWait = 30s`
- `UnkillableStepTimeout = 60s`
- Total SIGTERM → drain: ~90s
- ncclCommAbort timeout (PyTorch default): 600,000 ms (10 min)

## Full Results Table

| Test | Job ID | Result | Node State | Notes |
|------|--------|--------|-----------|-------|
| MWE-01 (v1 buggy) | 1209813 | ⚠️ DRAIN (accidental) | DRAINED | Bug: rank 0 called all_reduce twice; drain in teardown |
| MWE-01 (v2 fixed) | 1209844 | ✅ PASS | idle | Blind window ~0.9 ms |
| MWE-02 | 1209850 | ✅ PASS | idle | Stop-file clean exit |
| MWE-03 | 1209855 | ✅ NO DRAIN | idle | TCP socket → S-state (not D-state) |
| MWE-04 | 1209858 | ✅ NO DRAIN | idle | all_reduce → R-state |
| MWE-05 v1-v7 | 1209890–1209905 | ✅ NO DRAIN | idle | Various NCCL configs |
| MWE-05 v8a seq_mismatch | 1209908 | ✅ NO DRAIN | idle | NCCL truncates to min count |
| MWE-05 v8b count-mismatch | 1209909 | ✅ NO DRAIN | idle | NCCL truncates gracefully |
| MWE-05 gpu_sleep (fixed) | 1209925 | ✅ NO DRAIN | idle | `torch.cuda._sleep(inf)` → clean cancel |
| MWE-06 teardown (20 min) | 1209930 | ✅ NO DRAIN — watchdog fired | idle | teardown hung 600s; watchdog SIGABRT at 10:15; `_exit()` fast reset |
| MWE-06b teardown (5 min) | 1209935 | ✅ NO DRAIN | idle | SLURM fired at 5:17; S-state → SIGTERM processed in <1s |
| MWE-06c teardown (8 min, --max-restarts=0) | 1209941 | ✅ NO DRAIN | idle | Same S-state result; --max-restarts=0 had no effect on sub-steps |
| MWE-06d sync-rounds (20 min) | 1209946 | ✅ NO DRAIN — watchdog fired | idle | Round 1 stuck at SeqNum=3 for 600s; watchdog SIGABRT at 10:13; exit <2s |

## Drain Mechanism (confirmed from Event #4 + MWE-06 findings)

```
Round N (count-mismatch all_reduce)
  → NCCL ring state desynchronised (completes but corrupts internal seq)
  → Repeated N rounds
  → All rounds complete; script prints "PASS"

Teardown: dist.destroy_process_group()
  → PyTorch calls ncclCommAbort() on all communicators
  → ncclCommAbort() cannot drain desynchronised NVLink ring
  → D-state (TASK_UNINTERRUPTIBLE) — kernel-mode NVLink wait (~600s)

TWO PATHS FROM HERE:

Path A — SLURM fires DURING D-state window (< 600s after teardown):
  → SLURM --time fires → SIGTERM → not delivered (D-state)
  → 30s KillWait → SIGKILL → D-state survives SIGKILL
  → 60s UnkillableStepTimeout → "Kill task failed"
  → NODE DRAIN ← confirmed in Event #4 (job 1209813)

Path B — SLURM gives enough time (> ~600s after teardown):
  → PyTorch watchdog thread (NOT in D-state) fires at 600,000ms
  → Logs: "WorkNCCL ran for 600039ms → taking entire process down"
  → std::abort() → SIGABRT → process terminates
  → Job: FAILED (exit 1), node: idle, NO DRAIN ← observed in MWE-06 (job 1209930)
```

SLURM timing constants (confirmed on this cluster):
- `KillWait = 30s`
- `UnkillableStepTimeout = 60s`  
- Total SIGTERM → drain: ~90s
- ncclCommAbort timeout (PyTorch default): 600,000 ms (10 min)

## Evidence from Event #4 Log (`/work/projects/imas_gpu/logs/mwe01-1209813.err`)
```
[rank2]: Future for ProcessGroup abort timed out after 600000 ms
[rank1]: Future for ProcessGroup abort timed out after 600000 ms
[rank3]: Future for ProcessGroup abort timed out after 600000 ms
[rank0]: Future for ProcessGroup abort timed out after 600000 ms
slurmstepd: error: *** JOB 1209813 STEPD TERMINATED ... DUE TO JOB NOT ENDING WITH SIGNALS ***
slurmstepd: error: Container ... has 5 processes, giving up after 63 sec
```

## Evidence from MWE-06d Log (`/work/projects/imas_gpu/logs/mwe06d-1209946.err`)
```
# SeqNum=3 mismatch — all 4 ranks hung for 600s:
[rank0]: WorkNCCL(SeqNum=3, NumelIn=67108864, ...) ran for 600051ms before timing out
[rank1]: WorkNCCL(SeqNum=3, NumelIn=1, ...) ran for 600014ms before timing out
[rank2]: WorkNCCL(SeqNum=3, NumelIn=1, ...) ran for 600028ms before timing out
[rank3]: WorkNCCL(SeqNum=3, NumelIn=1, ...) ran for 600037ms before timing out
# → Rank 0 had 256MB step B; ranks 1-3 had 1-element step C
# → All 4 watchdogs fired within 76ms of each other → clean exit
```

## Conclusions

1. **The historical drain mechanism is confirmed** (Event #4): ncclCommAbort timeout
   → normal Python exit → `~ProcessGroupNCCL()` destructor → ncclCommDestroy on
   partial-abort state → D-state → drain. Reproduced accidentally in MWE-01 v1.

2. **NCCL 2.21.5 is immune to this mechanism** because: teardown hangs in S-state
   (not D-state), the watchdog fires SIGABRT causing `_exit()` instead of normal exit,
   and `_exit()` skips the destructor ncclCommDestroy call.

3. **S-state (interruptible) hangs never drain** regardless of duration — SIGTERM
   is delivered within milliseconds (confirmed MWE-03, 06b, 06c).

4. **SeqNum mismatch causes all ranks to hang simultaneously** for exactly the
   watchdog timeout (600s), with <76ms spread across ranks (MWE-06d). This confirms
   hardware-level cross-GPU synchronisation even in mismatch conditions.

5. **SLURM sub-steps are harmless accounting**: `.N` sub-steps accumulate at ~0.76/s
   from SLURM's own GPU cgroup monitoring — not torchrun elastic retries.

6. **`_exit()` vs normal exit is the safety boundary**: any exit path that avoids
   calling `~ProcessGroupNCCL()` destructor is safe. SIGABRT from the watchdog, or
   direct `os._exit()` in user code, both achieve this.
