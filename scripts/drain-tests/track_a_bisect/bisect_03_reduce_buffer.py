# ============================================================
# TRACK A BISECT-03: Full mismatch preserved, buffer reduced 256 MB → 1 MB
# ============================================================
# Base:  track_a_mwe01_v1_frozen.py (verbatim b3a71b6)
# Change: BUFFER_MB default overridden to 1 (from 256). Everything else
#         is verbatim v1, including the rank-0-only second all_reduce.
#
# What this isolates:
#   The v1 collective mismatch is preserved (same rank-0-only second all_reduce).
#   The only change is the buffer size: 256 MB → 1 MB. This tests whether
#   the large buffer is a NECESSARY condition for the drain, or whether any
#   mismatch (even with a tiny buffer) triggers D-state.
#
#   Hypothesis A (buffer matters): a small buffer completes fast enough that
#     the mispaired ranks time out / error cleanly rather than wedging in D-state.
#     If bisect_03 does NOT drain: buffer size is part of the trigger condition.
#
#   Hypothesis B (buffer irrelevant): NCCL D-state wedge is triggered purely
#     by the count mismatch regardless of payload size.
#     If bisect_03 DOES drain: buffer size is not the factor.
#
# Expected: DRAINS (D-state wedge is triggered by collective count mismatch
#           independent of payload size — buffer size buys at most ~µs).
#
# Note: BUFFER_MB can still be overridden via env var, but the default is
# changed to 1 to implement this variant's isolation.
# ============================================================

import csv
import os
import signal
import time

import torch
import torch.distributed as dist

LOG_DIR = os.environ.get("DRAIN_TEST_LOG_DIR", "/tmp")
NROUNDS = int(os.environ.get("NROUNDS", "20"))
# BISECT-03: Override default buffer to 1 MB (v1 default was 256 MB)
BUFFER_MB = int(os.environ.get("BUFFER_MB", "1"))


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
    print(
        f"[bisect_03] reduce_buffer — buffer={BUFFER_MB} MB (v1 used 256 MB), mismatch preserved",
        flush=True,
    )
    rank, local_rank = setup()
    device = torch.device(f"cuda:{local_rank}")
    world_size = dist.get_world_size()

    n_floats = (BUFFER_MB * 1024 * 1024) // 4
    buf = torch.randn(n_floats, device=device)
    stop = torch.zeros(1, device=device)

    if rank == 0:
        print(
            f"[MWE-01] ranks={world_size}, buffer={BUFFER_MB} MB, rounds={NROUNDS}",
            flush=True,
        )
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

        # --- SIGTERM blind-window measurement (rank 0 only — MISMATCH preserved) ---
        blind_ms = float("nan")
        if rank == 0:
            handler_fired_at: list[float] = []

            def _handler(signum: int, frame: object) -> None:
                handler_fired_at.append(time.perf_counter())

            old_handler = signal.signal(signal.SIGTERM, _handler)

            import threading

            def _fire() -> None:
                time.sleep(0.001)
                t_sent = time.perf_counter()
                os.kill(os.getpid(), signal.SIGTERM)
                _fire._t_sent = t_sent  # type: ignore[attr-defined]

            _fire._t_sent = float("nan")  # type: ignore[attr-defined]
            t = threading.Thread(target=_fire, daemon=True)
            t.start()

            # Blind-window collective — rank 0 only (MISMATCH preserved)
            torch.cuda.synchronize(device)
            dist.all_reduce(buf, op=dist.ReduceOp.SUM)
            torch.cuda.synchronize(device)

            t.join()
            signal.signal(signal.SIGTERM, old_handler)

            if handler_fired_at:
                blind_ms = (handler_fired_at[0] - _fire._t_sent) * 1000.0  # type: ignore[attr-defined]
            else:
                t_end = time.perf_counter()
                blind_ms = (t_end - _fire._t_sent) * 1000.0  # type: ignore[attr-defined]

            print(f"[MWE-01] {rnd},{blind_ms:.3f},{collective_ms:.3f}", flush=True)
            results.append(
                {
                    "round": rnd,
                    "blind_window_ms": blind_ms,
                    "collective_ms": collective_ms,
                }
            )

        # --- Coordinate stop after all rounds (rank 0 signals others) ---
        dist.all_reduce(stop, op=dist.ReduceOp.MAX)

    if rank == 0 and results:
        blind_vals = [
            r["blind_window_ms"]
            for r in results
            if r["blind_window_ms"] == r["blind_window_ms"]
        ]
        coll_vals = [r["collective_ms"] for r in results]
        print("\n[MWE-01] SUMMARY", flush=True)
        print(
            f"[MWE-01]   collective_ms  mean={sum(coll_vals) / len(coll_vals):.2f} "
            f"max={max(coll_vals):.2f}",
            flush=True,
        )
        if blind_vals:
            print(
                f"[MWE-01]   blind_window_ms mean={sum(blind_vals) / len(blind_vals):.2f} "
                f"max={max(blind_vals):.2f}",
                flush=True,
            )

        csv_path = os.path.join(
            LOG_DIR, f"bisect03_results_{os.environ.get('SLURM_JOB_ID', 'local')}.csv"
        )
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["round", "blind_window_ms", "collective_ms"]
            )
            writer.writeheader()
            writer.writerows(results)
        print(f"[bisect_03] Results written to {csv_path}", flush=True)

    teardown()


if __name__ == "__main__":
    main()
