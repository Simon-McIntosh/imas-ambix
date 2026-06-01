# Drain Test MWEs — Minimal Working Examples

Generic, self-contained scripts that reproduce SLURM node drain conditions
on NVIDIA GPU clusters. No project-specific dependencies required.

## Background

These scripts accompany a root-cause analysis of node drain events observed
on a 4×H200 NVLink cluster (SDCC `98dci4-gpu-0003`). They are designed to:

1. **Characterise** the SIGTERM blind window during NCCL collectives
2. **Validate** the stop-file safe cancellation mechanism
3. **Probe** network D-state (TCP connect to DROP-firewall host)
4. **Confirm** R-state vs D-state for simple NCCL all-reduce
5. **Reproduce** the confirmed drain mechanism (DDP `loss.backward()` + scancel)

## Prerequisites

- SLURM cluster with GPU nodes
- Python venv with PyTorch + NCCL
- Override the Python interpreter path via `PYTHON_VENV`:
  ```bash
  export PYTHON_VENV=/path/to/your/venv
  ```

## Test Scripts

| Script | Risk | Description |
|--------|------|-------------|
| `mwe-01-sigterm-delay.py` | ✅ SAFE | Measures SIGTERM blind window during NCCL collectives |
| `mwe-02-stopfile.py` | ✅ SAFE | Validates stop-file clean cancellation |
| `mwe-03-network-hang.py` | ⚠️ UNCERTAIN | TCP connect to DROP host — may D-state |
| `mwe-04-nccl-scancel.py` | ⚠️ LOW RISK | Simple NCCL + scancel — expected R-state (no drain) |
| `mwe-05-ddp-backward.py` | ⚠️⚠️ LIKELY DRAIN | DDP backward + scancel — confirmed drain mechanism |

## Running Order (safest first)

### MWE-01 — SIGTERM Blind Window (SAFE)
```bash
sbatch mwe-01-sigterm-delay.sbatch
# Self-terminating. Expected: blind_window_ms ≈ 1-3 ms on H200 NVLink
```

### MWE-02 — Stop-File Exit (SAFE)
```bash
sbatch mwe-02-stopfile.sbatch
# Self-terminating after 30 s. Expected: all ranks exit 0, node stays idle
```

### MWE-03 — Network Hang (UNCERTAIN DRAIN)
```bash
sbatch mwe-03-network-hang.sbatch  # note the job ID
# Wait for "ATTEMPTING TCP CONNECT" in the log
tail -f /work/projects/imas_gpu/logs/mwe03-<jobid>.out
# Then cancel:
scancel <jobid>
# Check node state after 70 s:
sinfo -N -n <nodename> --noheader -o "%T"
# "drain" → D-state confirmed; recovery: scontrol update nodename=<node> state=resume reason=""
# "idle"  → S-state (interruptible); no drain; network uses safe sleep
```

### MWE-04 — NCCL All-Reduce + Scancel (LOW RISK)
```bash
sbatch mwe-04-nccl-scancel.sbatch  # note the job ID
# Wait for "READY" in log
tail -f /work/projects/imas_gpu/logs/mwe04-<jobid>.out
# Then cancel:
scancel <jobid>
# Check node state after 70 s:
sinfo -N -n <nodename> --noheader -o "%T"
# Expected: "idle" (R-state, SIGKILL works)
# If "drain": unexpected; NCCL uses D-state on this platform
```

### MWE-05 — DDP Backward + Scancel (LIKELY DRAIN ⚠️)
```bash
sbatch mwe-05-ddp-backward.sbatch  # note the job ID
# Wait for "READY" in log (~30 s)
tail -f /work/projects/imas_gpu/logs/mwe05-<jobid>.out
# Immediately cancel:
scancel <jobid>
# Check node state after 70 s:
sinfo -N -n <nodename> --noheader -o "%T"
# Expected: "drain" (D-state during cudaStreamSynchronize in NCCL grad all-reduce)
```

**Recovery after MWE-05 drain:**
```bash
nvidia-smi -i 0,1,2,3 --gpu-reset
scontrol update nodename=<nodename> state=resume reason=""
```

## Results

| Test | Job ID | Result | Notes |
|------|--------|--------|-------|
| MWE-01 | — | — | Not yet run |
| MWE-02 | — | — | Not yet run |
| MWE-03 | — | — | Not yet run |
| MWE-04 | — | — | Not yet run |
| MWE-05 | — | — | Not yet run |

## Drain Mechanism Summary

```
D-state (TASK_UNINTERRUPTIBLE)
  ↑ caused by:
  ├── cudaStreamSynchronize() inside NCCL grad all-reduce (loss.backward)
  ├── GPFS I/O in DataLoader worker (queue.get() on dead producer)
  └── TCP connect() to DROP-firewalled host
  
  → SLURM SIGTERM → queued, not delivered
    → 60s UnkillableStepTimeout
      → SIGKILL → D-state survives
        → "Kill task failed" → NODE DRAINS
```

Contrast with **R-state** (CPU spin-wait during simple `dist.all_reduce`):
- SIGKILL succeeds immediately
- Node stays idle
- No admin intervention required
