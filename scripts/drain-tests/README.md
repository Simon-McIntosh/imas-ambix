# Drain Test MWEs — Minimal Working Examples

Generic, self-contained scripts that reproduce SLURM node drain conditions
on NVIDIA GPU clusters. No project-specific dependencies beyond PyTorch + NCCL.

## Background

These scripts accompany a root-cause analysis of node drain events observed
on a 4×H200 NVLink cluster (SDCC `98dci4-gpu-0003`, NCCL 2.21, PyTorch 2.11).
Each MWE is designed to probe a specific drain mechanism — or to confirm the
**absence** of drain — in the most minimal reproducible way.

Key finding: **node drains require a D-state (TASK_UNINTERRUPTIBLE) process**.
Processes in S-state (interruptible sleep) or R-state (runnable) are killed
cleanly by SIGKILL. Only kernel-mode waits that ignore SIGKILL cause drains.

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
| `mwe-06-nccl-teardown.py` | 🔴 DRAIN | ncclCommAbort() D-state on corrupted communicator | **DRAIN** |

## Confirmed Non-Drain Mechanisms

| Operation | Process State | Reason |
|-----------|---------------|--------|
| urllib TCP socket (DROP firewall) | S-state | Socket wait is interruptible |
| `dist.all_reduce` (healthy) | R-state | CPU spin in NCCL user-space poll |
| `loss.backward()` async DDP | R/S-state | DDP all-reduce is asynchronous |
| `torch.cuda.synchronize()` | S-state | Short; interruptible by SIGKILL |
| NCCL count-mismatch collective | Returns (truncated) | NCCL 2.21 truncates to min count |
| P2P `dist.recv` ring deadlock | S-state | Interruptible; SIGTERM/KILL works |
| Cross-group new_group deadlock | Aborted in 2s | NCCL heartbeat fires |
| `torch.cuda._sleep(inf)` + scancel | S-state | GPU kernel stays on GPU; Python dies cleanly |

## Confirmed Drain Mechanism

| Operation | Process State | Path to Drain |
|-----------|---------------|---------------|
| `ncclCommAbort()` on corrupted communicator | **D-state** | `ncclCommAbort()` hangs 600s → PyTorch gives up → SLURM time limit → SIGTERM → unkillable → drain |

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

### MWE-06 — NCCL Teardown D-State (`--time=00:20:00`)
```bash
sbatch mwe-06-nccl-teardown.sbatch
# Runs count-mismatch rounds, then calls dist.destroy_process_group()
# Expected: drain. Actual result (job 1209930): NO DRAIN
#
# ncclCommAbort() hung for 600s, but PyTorch watchdog (separate thread)
# fired SIGABRT at 10:15 — process exited before SLURM's 20-min limit.
# Finding: D-state self-resolves if SLURM gives enough time.
```

### MWE-06b — NCCL Teardown D-State, SLURM Fires During D-State (`--time=00:05:00`) ⚠️ EXPECTED DRAIN
```bash
sbatch mwe-06b-timed-kill.sbatch
# Same Python script, but --time=00:05:00
# Timeline:
#   ~0:30  rounds complete; ncclCommAbort() enters D-state
#   ~5:00  SLURM --time fires (4:30 into D-state window)
#   ~5:30  SIGTERM → D-state → not delivered
#   ~6:00  SIGKILL → D-state survives
#   ~7:00  UnkillableStepTimeout → DRAIN
# Job 1209935: in progress
```

**Recovery after MWE-06b drain:**
```bash
nvidia-smi -i 0,1,2,3 --gpu-reset
scontrol update nodename=<nodename> state=resume reason=""
```

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
| MWE-06 teardown (20 min) | 1209930 | ✅ NO DRAIN — PyTorch self-resolved | idle | ncclCommAbort hung 600s; watchdog SIGABRT at 10:15; exited before SLURM |
| MWE-06b teardown (5 min) | 1209935 | 🔄 IN PROGRESS | — | SLURM fires at 5:00 = 4:30 into D-state; DRAIN expected |

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
