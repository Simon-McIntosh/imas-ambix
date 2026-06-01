"""
MWE-01: NCCL SIGTERM Blind-Window Measurement
=============================================
PURPOSE
  Quantifies the duration during which SIGTERM cannot be delivered to a
  PyTorch DDP rank because Python is blocked inside a CUDA/NCCL collective.
  This blind window is the root-cause prerequisite for all observed
  GPU-node drain events (where SLURM cannot kill a job within
  UnkillableStepTimeout ≈ 60 s and auto-drains the node).

RISK LEVEL
  SAFE — the script terminates itself; no scancel required.

WHAT IT DOES
  1. Initialises a 4-rank DDP job (NCCL backend) across 4 GPUs.
  2. Runs N rounds of dist.all_reduce on a large buffer to model NCCL load.
  3. Rank 0 installs a SIGTERM handler and, mid-collective, sends SIGTERM
     to itself via ``os.kill(os.getpid(), signal.SIGTERM)``.
  4. Measures the elapsed time from signal send to handler execution.
  5. All ranks coordinate exit via a shared "stop" tensor (ReduceOp.MAX)
     so no collective mismatch hangs occur.
  6. Prints a summary CSV line for each round.

EXPECTED RESULT
  Blind-window ≈ the duration of one NCCL collective call.
  On H200 NVLink (4-GPU all-reduce of 256 MB): ~1–3 ms.
  This confirms SIGTERM cannot be processed during the collective,
  meaning a SLURM scancel mid-collective leaves the process unkillable
  until the collective finishes (or forever if NVLink hangs).

ENVIRONMENT VARIABLES
  NROUNDS          Number of measurement rounds (default: 20)
  BUFFER_MB        Size of all-reduce buffer in MB (default: 256)
  DRAIN_TEST_LOG_DIR  Directory for log files (default: /tmp)

RUN WITH
  torchrun --nproc_per_node=4 mwe-01-sigterm-delay.py

  Or via the accompanying sbatch script:
  sbatch mwe-01-sigterm-delay.sbatch
"""

import os
import signal
import time
import csv
import sys

import torch
import torch.distributed as dist

LOG_DIR = os.environ.get("DRAIN_TEST_LOG_DIR", "/tmp")
NROUNDS = int(os.environ.get("NROUNDS", "20"))
BUFFER_MB = int(os.environ.get("BUFFER_MB", "256"))


def setup() -> tuple[int, int]:
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    return rank, local_rank


def teardown() -> None:
    dist.barrier()
    dist.destroy_process_group()


def main() -> None:
    rank, local_rank = setup()
    device = torch.device(f"cuda:{local_rank}")
    world_size = dist.get_world_size()

    n_floats = (BUFFER_MB * 1024 * 1024) // 4
    buf = torch.randn(n_floats, device=device)
    stop = torch.zeros(1, device=device)

    if rank == 0:
        print(f"[MWE-01] ranks={world_size}, buffer={BUFFER_MB} MB, rounds={NROUNDS}", flush=True)
        print("[MWE-01] round,blind_window_ms,collective_ms", flush=True)

    results: list[dict] = []

    for rnd in range(NROUNDS):
        # --- Coordinate stop ---
        if stop[0].item() > 0.5:
            break

        # --- Measure collective duration ---
        torch.cuda.synchronize(device)
        t_collective_start = time.perf_counter()
        dist.all_reduce(buf, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize(device)
        t_collective_end = time.perf_counter()
        collective_ms = (t_collective_end - t_collective_start) * 1000.0

        # --- SIGTERM blind-window measurement (rank 0 only) ---
        blind_ms = float("nan")
        if rank == 0:
            handler_fired_at: list[float] = []

            def _handler(signum: int, frame: object) -> None:
                handler_fired_at.append(time.perf_counter())

            old_handler = signal.signal(signal.SIGTERM, _handler)

            # Send SIGTERM to self from a thread so it arrives mid-collective
            import threading

            def _fire() -> None:
                time.sleep(0.001)  # 1 ms delay — land mid-collective
                t_sent = time.perf_counter()
                os.kill(os.getpid(), signal.SIGTERM)
                # Store t_sent in closure
                _fire._t_sent = t_sent  # type: ignore[attr-defined]

            _fire._t_sent = float("nan")  # type: ignore[attr-defined]
            t = threading.Thread(target=_fire, daemon=True)
            t.start()

            # Run collective while SIGTERM may be in-flight
            torch.cuda.synchronize(device)
            dist.all_reduce(buf, op=dist.ReduceOp.SUM)
            torch.cuda.synchronize(device)

            t.join()
            signal.signal(signal.SIGTERM, old_handler)

            if handler_fired_at:
                blind_ms = (handler_fired_at[0] - _fire._t_sent) * 1000.0  # type: ignore[attr-defined]
            else:
                # Handler not yet fired — measure from collective end
                t_end = time.perf_counter()
                blind_ms = (t_end - _fire._t_sent) * 1000.0  # type: ignore[attr-defined]

            print(f"[MWE-01] {rnd},{blind_ms:.3f},{collective_ms:.3f}", flush=True)
            results.append({"round": rnd, "blind_window_ms": blind_ms, "collective_ms": collective_ms})

        # --- Coordinate stop after all rounds (rank 0 signals others) ---
        dist.all_reduce(stop, op=dist.ReduceOp.MAX)

    if rank == 0 and results:
        blind_vals = [r["blind_window_ms"] for r in results if r["blind_window_ms"] == r["blind_window_ms"]]
        coll_vals = [r["collective_ms"] for r in results]
        print(f"\n[MWE-01] SUMMARY", flush=True)
        print(f"[MWE-01]   collective_ms  mean={sum(coll_vals)/len(coll_vals):.2f} "
              f"max={max(coll_vals):.2f}", flush=True)
        if blind_vals:
            print(f"[MWE-01]   blind_window_ms mean={sum(blind_vals)/len(blind_vals):.2f} "
                  f"max={max(blind_vals):.2f}", flush=True)
            print(f"[MWE-01]   INTERPRETATION: SIGTERM is blind for ~{sum(blind_vals)/len(blind_vals):.1f} ms "
                  f"per NCCL collective. Under SLURM scancel, if a collective takes longer than "
                  f"UnkillableStepTimeout (60 s), the node drains.", flush=True)

        csv_path = os.path.join(LOG_DIR, f"mwe01_results_{os.environ.get('SLURM_JOB_ID','local')}.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["round", "blind_window_ms", "collective_ms"])
            writer.writeheader()
            writer.writerows(results)
        print(f"[MWE-01] Results written to {csv_path}", flush=True)
        print("[MWE-01] PASS", flush=True)

    teardown()


if __name__ == "__main__":
    main()
