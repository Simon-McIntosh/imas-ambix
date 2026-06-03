# RUNCARD: Track A — MWE-01 v1 Frozen Backward Bisection

**Part of:** drain-window-campaign §4  
**Runs SECOND** in the admin window (after Track C validates harness and node is resumed).  
**Admin required throughout.** Every run that drains dirties the GPU and requires manual reset.

---

## Prerequisites

1. Track C completed — harness validated, node resumed by admin.
2. Admin on standby (phone/chat) — Track A Phase 1 WILL drain the node.
3. Confirm node is IDLE+RESERVED:
   ```bash
   sinfo -p betelgeuse --reservation=gpu_0003_grpA
   ```
4. Confirm zero GPU processes:
   ```bash
   ssh 98dci4-gpu-0003 nvidia-smi  # expect 0 processes on all 4 GPUs
   ```
5. Confirm observer infrastructure ready (from Track C):
   ```bash
   ls /work/projects/imas_gpu/logs/
   cat scripts/drain-tests/instrumented/observer.sbatch | head -20
   ```

---

## Phase 1: Verbatim Reproduction (up to 5 runs)

Goal: confirm the frozen v1 code drains a clean node. Decision threshold: 3 non-drains → inconclusive.

### Per-run procedure

For each run **i** in {1 … 5}:

**Step a.** Start observer:
```bash
sbatch scripts/drain-tests/instrumented/observer.sbatch
# Note OBSERVER_JOBID from output
```

**Step b.** Submit frozen run:
```bash
sbatch scripts/drain-tests/track_a_mwe01_v1_frozen.sbatch
# Note DRAIN_JOBID from output
```
Update the observer's target if it requires a specific job ID.

**Step c.** Wait for outcome (max 15 min):
```bash
watch -n10 "squeue -j ${DRAIN_JOBID} -o '%T %R'"
```
Expected: job disappears without COMPLETED state (SLURM drains and kills it).

**Step d.** If DRAINED — collect logs, then admin reset:
```bash
# Collect logs before reset
cp /work/projects/imas_gpu/logs/track-a-frozen-${DRAIN_JOBID}.out \
   /work/projects/imas_gpu/logs/track-a-frozen-${DRAIN_JOBID}.err \
   /tmp/  # local backup

# ADMIN: reset the GPU (must be run on the node or as privileged user)
ssh 98dci4-gpu-0003 nvidia-smi --gpu-reset
# Verify GPUs are clean:
ssh 98dci4-gpu-0003 nvidia-smi

# ADMIN: resume the node:
scontrol update nodename=98dci4-gpu-0003 state=resume reason=""
# Confirm:
sinfo -p betelgeuse -n 98dci4-gpu-0003
```

**Step e.** If NOT drained (COMPLETED or TIMEOUT without drain):
```bash
scontrol show job ${DRAIN_JOBID}  # confirm ExitCode
```
Record result. Continue to next run.

**Step f.** After run i, record:
```
Run i: DRAIN_JOBID=XXXXXXX  drained=[yes/no]  observer_log=track-a-frozen-XXXXXXX.out
```

### Stopping criteria

| Outcome | Action |
|---------|--------|
| 1+ drains in 5 runs | Phase 1 success — proceed to Phase 2 bisection |
| 0 drains in 3 runs | Conclude "verbatim no longer drains on clean node". Phase 2 deferred. Record as inconclusive. |
| 0 drains in 5 runs | Conclude "trigger requires pre-existing GPU state or is probabilistic". Record as inconclusive. |

---

## Phase 2: Bisection Ladder (run only if Phase 1 drains)

Goal: find the minimal element whose removal prevents the drain.

**Between every run:** admin gpu-reset + scontrol resume + confirm `nvidia-smi` shows 0 processes.

### Bisect variants (run in order, stop at first non-drain)

| Variant | File | Change from v1 | Expected |
|---------|------|----------------|----------|
| **bisect_01** | `track_a_bisect/bisect_01_sync_second_call.py` | Blind-window all_reduce on ALL ranks (second call synchronized). Mirrors 928ed0d fix. | **NO DRAIN** |
| **bisect_02** | `track_a_bisect/bisect_02_sync_first_call.py` | First call is already symmetric in v1 — **logically identical to frozen v1**. Negative control to confirm mismatch is entirely in second call. | **DRAINS** (same as v1) |
| **bisect_03** | `track_a_bisect/bisect_03_reduce_buffer.py` | Full mismatch preserved, buffer 256 MB → 1 MB. Tests whether buffer size matters. | **DRAINS** (hypothesis: count mismatch is sufficient regardless of size) |
| **bisect_04** | `track_a_bisect/bisect_04_single_rank_only.py` | Single rank (no NCCL), CUDA context + 256 MB held, Python hang loop. Negative control: does any unkillable process drain? | **NO DRAIN** (Python loops are SIGKILL-able, no D-state without blocking NCCL) |

### Per-run procedure (bisect)

**Step a.** Confirm node clean: `ssh 98dci4-gpu-0003 nvidia-smi` (0 processes).

**Step b.** Start observer: `sbatch scripts/drain-tests/instrumented/observer.sbatch` → note OBSERVER_JOBID.

**Step c.** Submit bisect variant:
```bash
# Example for bisect_01:
sbatch scripts/drain-tests/track_a_bisect/bisect_01.sbatch
# Note BISECT_JOBID
```

**Step d.** Wait for outcome (max 15 min).

**Step e.** If DRAINED: admin reset. If NOT drained: note clean exit.

**Step f.** Record:
```
bisect_01: JOBID=XXXXXXX  drained=[yes/no]  observer_log=track-a-bisect01-XXXXXXX.out
```

**Step g.** Proceed to next variant regardless of outcome (ladder must be run fully).

---

## Decision Tree

```
Phase 1: Frozen v1 drains?
  YES → proceed to Phase 2 bisection
  NO (after 5 runs) → INCONCLUSIVE: "trigger requires pre-existing GPU state or probabilistic"
                      Do not proceed to Phase 2.

Phase 2 (if Phase 1 drains):
  bisect_01 does NOT drain → RESULT: removing second-call mismatch alone prevents drain
                              (928ed0d fix was minimal and correct)
  bisect_01 drains, bisect_02 does NOT drain → UNEXPECTED: first-call path matters somehow
                                                (re-examine RCA)
  bisect_01 drains, bisect_02 drains → EXPECTED: first-call fix non-separable
  bisect_03 does NOT drain → buffer size IS a factor (small buffer escapes D-state)
  bisect_03 drains → buffer size is NOT the factor (count mismatch sufficient alone)
  bisect_04 DRAINS → UNEXPECTED: GPU allocation alone can trigger D-state
                     STOP and file RCA before further runs
  bisect_04 does NOT drain → EXPECTED: NCCL blocking collective is required for D-state
```

---

## Log Collection (after each phase)

All observer and job logs are in `/work/projects/imas_gpu/logs/`. After each phase:

```bash
ls -lt /work/projects/imas_gpu/logs/track-a-* | head -20
ls -lt /work/projects/imas_gpu/logs/observer-* | head -20
```

Copy to a local results directory for the campaign record:
```bash
RESULT_DIR=~/Code/imas-ambix/scripts/drain-tests/track_a_bisect/results-$(date +%Y%m%d)
mkdir -p "${RESULT_DIR}"
cp /work/projects/imas_gpu/logs/track-a-*.{out,err} "${RESULT_DIR}/"
cp /work/projects/imas_gpu/logs/observer-*.{out,err} "${RESULT_DIR}/"
```

---

## Critical Check Before Each Resubmit

```bash
ssh 98dci4-gpu-0003 nvidia-smi
# Must show: 0 processes on all 4 GPUs
# If any process remains: DO NOT submit next run. Escalate to admin.
```

---

## Notes

- **bisect_02 is a known-redundant control.** It is logically identical to frozen v1 because v1's
  first (timing) all_reduce is already symmetric. Its purpose is to validate this understanding
  and confirm the mismatch is entirely in the second call. Orchestrator may choose to skip it
  if admin resets are scarce.
- **bisect_04 uses `--gres=gpu:1`** (single GPU) — intentional. No NCCL, no need for 4 GPUs.
- **15 min `--time` ceiling** matches job 1209813's original time limit. Drain typically occurs
  within the first 2-3 rounds (< 2 min). If the job reaches 14 min without drain, it is likely
  not going to drain in that run.
