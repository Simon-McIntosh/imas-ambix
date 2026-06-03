# ============================================================
# TRACK A BISECT-02: Sync the FIRST (timing) all_reduce to ALL ranks
# ============================================================
# Base:  track_a_mwe01_v1_frozen.py (verbatim b3a71b6)
# Change: The timing all_reduce is already on all ranks in v1 (it is outside
#         any rank guard). The second (blind-window) all_reduce stays rank-0-only.
#
# ⚠️  STRUCTURAL NOTE — READ BEFORE RUNNING:
#   In the v1 baseline, the FIRST all_reduce (timing) is already symmetric —
#   all 4 ranks execute it. The rank-0-only guard only wraps the SECOND
#   (blind-window) all_reduce. Therefore this variant is LOGICALLY IDENTICAL
#   to the frozen v1 baseline. It makes no structural change.
#
#   This is NOT a design error — it is here to validate this understanding.
#   NCCL matches collectives by per-rank issue order and count, NOT by logical
#   identity. The mismatch lives entirely in the SECOND call. Confirming that
#   "fixing the first call" (which is already fixed) does nothing is a useful
#   control: if bisect_02 drains at the same rate as frozen v1, it confirms
#   the mismatch is entirely in the second call. If it somehow differs, that
#   points to something unexpected in the timing measurement path.
#
#   Expected: DRAINS at the same rate as frozen v1.
#   Interpretation: first-call asymmetry is NOT a separable axis.
#   Admin should compare drain rate / timing against frozen v1 runs.
# ============================================================

import os
import signal
import time
import csv

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
    print("[bisect_02] sync_first_call — NOTE: logically identical to frozen v1 (first call already symmetric)", flush=True)
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

        # --- Timing all_reduce (ALL ranks — identical to v1, already symmetric) ---
        torch.cuda.synchronize(device)
        t_collective_start = time.perf_counter()
        dist.all_reduce(buf, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize(device)
        t_collective_end = time.perf_counter()
        collective_ms = (t_collective_end - t_collective_start) * 1000.0

        # --- SIGTERM blind-window measurement (rank 0 only — UNCHANGED from v1) ---
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

            # Blind-window collective — rank 0 only (MISMATCH preserved, same as v1)
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
            results.append({"round": rnd, "blind_window_ms": blind_ms, "collective_ms": collective_ms})

        # --- Coordinate stop after all rounds ---
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

        csv_path = os.path.join(LOG_DIR, f"bisect02_results_{os.environ.get('SLURM_JOB_ID','local')}.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["round", "blind_window_ms", "collective_ms"])
            writer.writeheader()
            writer.writerows(results)
        print(f"[bisect_02] Results written to {csv_path}", flush=True)

    teardown()


if __name__ == "__main__":
    main()
