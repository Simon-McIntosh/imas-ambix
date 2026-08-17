# ============================================================
# Symmetric second collective: Sync the second (blind-window) all_reduce to ALL ranks
# ============================================================
# Base:  track_a_mwe01_v1_frozen.py (verbatim b3a71b6)
# Change: The blind-window dist.all_reduce (inside `if rank == 0:`) is moved
#         outside that block so ALL 4 ranks execute it — matching 928ed0d's fix.
#
# What this isolates:
#   In v1, rank 0 runs 2 all_reduces per round (timing + blind-window) while
#   ranks 1-3 run only 1 (timing only). The second call's asymmetry is the
#   ENTIRE source of the mismatch — NCCL matches collectives by per-rank
#   issue order and count, so rank 0's extra call pairs with the subsequent
#   stop all_reduce on ranks 1-3, causing a tensor-size mismatch and D-state.
#
#   If bisect_01 does NOT drain: removing the second-call asymmetry alone is
#   sufficient; 928ed0d's fix was exactly right.
#   If bisect_01 DOES drain: some other element is also necessary.
#
# Expected: NO DRAIN (this replicates the fix from 928ed0d).
# ============================================================

import csv
import os
import signal
import time

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
    print(
        "[bisect_01] sync_second_call — all ranks enter blind-window collective",
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

        # --- Measure collective duration (ALL ranks) ---
        torch.cuda.synchronize(device)
        t_collective_start = time.perf_counter()
        dist.all_reduce(buf, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize(device)
        t_collective_end = time.perf_counter()
        collective_ms = (t_collective_end - t_collective_start) * 1000.0

        # --- Rank 0 arms signal handler ---
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

        # --- BISECT-01 CHANGE: blind-window all_reduce on ALL ranks (not rank 0 only) ---
        torch.cuda.synchronize(device)
        dist.all_reduce(buf, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize(device)

        if rank == 0:
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

        # --- Coordinate stop after all rounds ---
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
            LOG_DIR, f"bisect01_results_{os.environ.get('SLURM_JOB_ID', 'local')}.csv"
        )
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["round", "blind_window_ms", "collective_ms"]
            )
            writer.writeheader()
            writer.writerows(results)
        print(f"[bisect_01] Results written to {csv_path}", flush=True)
        print("[bisect_01] PASS (no drain expected)", flush=True)

    teardown()


if __name__ == "__main__":
    main()
