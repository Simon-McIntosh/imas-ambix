"""
MWE-06d: NCCL Teardown D-State — Synchronous Round Variant
===========================================================
PURPOSE
  Diagnostic variant of MWE-06 that adds torch.cuda.synchronize() after
  each round. If rounds still complete in <1ms, the drain mechanism cannot
  be reproduced on NCCL 2.21.5 (confirming the software-level mitigation).
  If rounds take ~1000ms (as in Event #4), actual NVLink transfers are
  happening and the teardown D-state may be reproducible.

KEY DIFFERENCE FROM MWE-06
  MWE-06: dist.all_reduce() returns in 0.1ms (async queue submission only)
  MWE-06d: torch.cuda.synchronize() after step C forces actual CUDA stream
           completion before the next round begins.

EVENT #4 COMPARISON
  Job 1209813 had 156s per-round timing, indicating actual synchronous
  NVLink work. If MWE-06d also shows long per-round timing (>100ms), the
  NVLink ring is carrying real traffic and teardown D-state is achievable.

EXPECTED OUTCOMES
  Case A (rounds < 10ms): NCCL 2.21.5 handles mismatch synchronously but
    quickly — teardown will still hang 600s in S-state, no drain.
  Case B (rounds ~1000ms): Actual NVLink work is occurring per round — the
    corrupted ring state is real and teardown D-state is reproducible.

RUN WITH
  sbatch scripts/drain-tests/mwe-06d-sync-rounds.sbatch
"""

import os
import sys
import time

import torch
import torch.distributed as dist

NROUNDS = int(os.environ.get("NROUNDS", "5"))
BUFFER_MB = int(os.environ.get("BUFFER_MB", "256"))
LOG_DIR = os.environ.get("DRAIN_TEST_LOG_DIR", "/tmp")
JOB_ID = os.environ.get("SLURM_JOB_ID", "local")


def main() -> None:
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    n_floats = (BUFFER_MB * 1024 * 1024) // 4
    buf_large = torch.randn(n_floats, dtype=torch.float32, device=device)
    buf_small = torch.zeros(1, dtype=torch.float32, device=device)

    if rank == 0:
        print(
            f"[MWE-06d] job={JOB_ID}  ranks={world_size}  rounds={NROUNDS}  buffer={BUFFER_MB}MB",
            flush=True,
        )
        print(
            "[MWE-06d] Variant: torch.cuda.synchronize() after each round", flush=True
        )
        print(
            "[MWE-06d] Timing < 10ms/round → NCCL 2.21.5 mitigates (no drain possible)",
            flush=True,
        )
        print(
            "[MWE-06d] Timing > 100ms/round → actual NVLink work → drain may be reproducible",
            flush=True,
        )

    round_times = []
    for rnd in range(NROUNDS):
        dist.barrier()  # synchronise all ranks at round start
        t_start = time.perf_counter()

        # Step A: ALL ranks — healthy collective
        dist.all_reduce(buf_large, op=dist.ReduceOp.SUM)

        # Step B: rank 0 ONLY — participation mismatch
        if rank == 0:
            dist.all_reduce(buf_large, op=dist.ReduceOp.SUM)

        # Step C: ALL ranks — seq counter mismatch between rank 0 and others
        dist.all_reduce(buf_small, op=dist.ReduceOp.MAX)

        # Force actual CUDA stream completion before timing this round
        # This is the key difference from MWE-06 (which never synchronizes)
        torch.cuda.synchronize()

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        round_times.append(elapsed_ms)
        if rank == 0:
            verdict = (
                "SLOW (actual NVLink work)" if elapsed_ms > 100 else "FAST (async only)"
            )
            print(
                f"[MWE-06d] Round {rnd + 1}/{NROUNDS}  {elapsed_ms:.1f} ms  [{verdict}]",
                flush=True,
            )

    if rank == 0:
        avg_ms = sum(round_times) / len(round_times)
        max_ms = max(round_times)
        print(
            f"[MWE-06d] Round timing: avg={avg_ms:.1f}ms  max={max_ms:.1f}ms",
            flush=True,
        )
        if max_ms < 10:
            print(
                "[MWE-06d] FINDING: Rounds are FAST — NCCL 2.21.5 handles mismatch synchronously",
                flush=True,
            )
            print(
                "[MWE-06d] CONCLUSION: Drain mechanism NOT reproducible on current NCCL version",
                flush=True,
            )
        else:
            print(
                "[MWE-06d] FINDING: Rounds are SLOW — actual NVLink transfers occurring",
                flush=True,
            )
            print(
                "[MWE-06d] CONCLUSION: Drain mechanism MAY be reproducible — see teardown below",
                flush=True,
            )
        print("[MWE-06d] Entering teardown — dist.destroy_process_group()", flush=True)
        sys.stdout.flush()

    # Teardown: will either hang 600s in S-state (no drain) or D-state (drain)
    dist.destroy_process_group()

    if rank == 0:
        print(
            "[MWE-06d] Teardown completed (no drain — confirming NCCL 2.21.5 mitigation)",
            flush=True,
        )


if __name__ == "__main__":
    main()
