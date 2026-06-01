# GPU Drain Reproducer Tests

Minimal test suite that validates drain mechanisms and confirms fixes for
the `98dci4-gpu-0003` H200 betelgeuse node drain events (2026-05-26 through 2026-05-28).

## Test Summary

| Test | Safe? | What it validates | Recovery needed |
|------|-------|------------------|-----------------|
| T1 — STOP-FILE | ✅ YES | Fix works: clean exit via stop-file | None |
| T2 — SIGTERM delay | ✅ YES | Mechanism: SIGTERM delayed during NCCL | None |
| T3 — Network hang | ⚠️ DRAINS | Mechanism: non-CUDA D-state → drain | `scontrol resume` |
| T4 — NCCL scancel | ⚠️ DRAINS | Mechanism: NCCL D-state → drain | GPU reset + `scontrol resume` |

**Run T1 and T2 first.** T3 and T4 need an SDCC admin standing by.

---

## T1: STOP-FILE Clean-Exit Validation (SAFE)

**What it proves:** Touching `AMBIX_STOP_FILE` causes all DDP ranks to exit
cleanly via the between-collective check, within ~5 s, without triggering drain.

```bash
# Submit
JOB=$(sbatch scripts/tests/t1_stopfile.sbatch | grep -oE '[0-9]+$')
echo "Job: $JOB"

# Wait for job to start and print "Ready"
tail -f /work/projects/imas_gpu/logs/t1-stopfile-${JOB}.log &

# Once you see "Ready. Touch stop-file to test:", trigger it:
touch /work/projects/imas_gpu/stops/${JOB}.stop

# PASS: All ranks print "STOP-FILE detected → clean exit" within 5s
# PASS: Job exits 0
# FAIL: Job hangs > 60s → would indicate STOP-FILE check is broken
```

---

## T2: SIGTERM Delivery-Delay Diagnostic (SAFE)

**What it proves:** Python signal handlers cannot fire during NCCL C++ calls.
The SIGTERM blind window = 1 full all_reduce duration (~10–500 ms on H200).
This is the exact mechanism that makes `scancel` dangerous on DDP jobs.

```bash
# Submit — job self-terminates, no intervention needed
JOB=$(sbatch scripts/tests/t2_sigterm_delay.sbatch | grep -oE '[0-9]+$')

# Wait for completion (~3 min)
squeue -j $JOB

# Read results
grep "SIGTERM delay\|RESULTS\|avg_collective\|MECHANISM" \
  /work/projects/imas_gpu/logs/t2-sigtermdelay-${JOB}.log

# EXPECTED OUTPUT:
#   SIGTERM delay:      XXX ms       ← > 0ms proves the blind window
#   avg collective:     YYY ms       ← delay should be ~= 1 collective
#   MECHANISM CONFIRMED — SIGTERM was delayed Xms
```

---

## T3: Network I/O D-State Drain Reproducer (⚠️ DRAINS)

**Requires admin.** Reproduces 2026-05-27 16:04 drain (Event 2 / Drain #3).
No GPU reset needed — process never touches CUDA.

```bash
# Coordinate with admin first. Then:
JOB=$(sbatch scripts/tests/t3_network_hang.sbatch | grep -oE '[0-9]+$')
NODE=$(squeue -j $JOB -o %N -h)

# Monitor
watch -n2 "squeue -j $JOB; sinfo -N -n $NODE | tail -1"

# Wait for job to print "will hang until you: scancel", then:
scancel $JOB

# EXPECTED: ~61s later, sinfo shows STATE=drain
# Admin recovery:
#   scontrol update nodename=$NODE state=resume reason=""
#   (No GPU reset needed)
```

---

## T4: NCCL + scancel Drain Reproducer (⚠️ DRAINS, needs GPU reset)

**Requires admin + GPU reset.** Reproduces 2026-05-28 13:52 drain (Event 3 / Drain #4).
CUDA contexts orphaned — must run `nvidia-smi --gpu-reset` before resume.

```bash
# Coordinate with admin first. Then:
JOB=$(sbatch scripts/tests/t4_nccl_scancel.sbatch | grep -oE '[0-9]+$')
NODE=$(squeue -j $JOB -o %N -h)

# Wait for "Running NCCL all_reduce loop" in the log, then:
scancel $JOB

# EXPECTED: ~61s later, sinfo shows STATE=drain
# Admin recovery:
#   nvidia-smi -i 0,1 --gpu-reset
#   scontrol update nodename=$NODE state=resume reason=""
```

---

## Interpreting Results for the RCA

After running T1 and T2, key data for the report:

| Measurement | Expected | Significance |
|-------------|----------|--------------|
| T1 stop-file exit latency | < 5 s | Fix confirmed working |
| T2 SIGTERM blind window (ms) | = 1× all_reduce | Proves mechanism |
| T2 all_reduce latency at 256 MB | 50–500 ms | "Unkillable window" duration |
| T3 drain delay after scancel | ~60–62 s | Confirms UnkillableStepTimeout |
| T4 drain delay after scancel | ~60–62 s | Confirms NCCL D-state |
