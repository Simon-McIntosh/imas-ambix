"""T2: SIGTERM delivery-delay diagnostic during NCCL collectives (SAFE — no drain risk).

WHAT THIS PROVES:
    Python signal handlers can only fire between bytecodes in the main thread.
    During a blocking C-extension call (dist.all_reduce → NCCL), the main
    thread is entirely inside C++ and cannot process Python signals.

    This is the exact mechanism that makes `scancel` dangerous on DDP jobs:
    SIGTERM arrives → Python queues the handler → handler cannot run while
    NCCL holds the GIL equivalent → process appears unkillable → drain.

HOW TO RUN:
    sbatch scripts/tests/t2_sigterm_delay.sbatch

    The job is self-terminating — a background thread sends SIGTERM to the
    process itself after SELF_SIGTERM_DELAY_S seconds. We measure how long
    it takes for the signal handler to fire relative to when the signal was sent.

PASS CRITERIA (mechanism proof):
    Log line: "SIGTERM handler fired Xms after signal was sent"
    X > 0ms (signal was delayed — couldn't fire mid-collective).
    Typical result: X = one full all_reduce duration (10–500 ms on H200).
    This PROVES the delivery delay mechanism documented in the RCA.

SECONDARY OUTPUT:
    Reports per-all_reduce latency so we know the "blind window" duration.
"""
import os
import signal
import sys
import threading
import time

import torch
import torch.distributed as dist

SELF_SIGTERM_DELAY_S = int(os.environ.get("T2_SIGTERM_DELAY_S", "10"))
TENSOR_NUMEL = int(os.environ.get("T2_TENSOR_NUMEL", str(2048 * 2048)))  # ~16 MB

_sigterm_sent_at: float = 0.0
_sigterm_recv_at: float = 0.0
_handler_fired = threading.Event()


def _sigterm_handler(signum: int, _frame) -> None:
    global _sigterm_recv_at
    _sigterm_recv_at = time.monotonic()
    _handler_fired.set()
    print(
        f"[T2] SIGTERM handler fired — "
        f"delay={(_sigterm_recv_at - _sigterm_sent_at)*1000:.1f}ms",
        flush=True,
    )


def _send_sigterm_after(delay_s: float) -> None:
    """Background thread: wait delay_s then SIGTERM the main process."""
    time.sleep(delay_s)
    global _sigterm_sent_at
    _sigterm_sent_at = time.monotonic()
    print(
        f"[T2] Sending SIGTERM to pid={os.getpid()} after {delay_s}s warmup",
        flush=True,
    )
    os.kill(os.getpid(), signal.SIGTERM)


def main() -> None:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")

    signal.signal(signal.SIGTERM, _sigterm_handler)

    buf = torch.ones(TENSOR_NUMEL, device=device, dtype=torch.float32)

    if rank == 0:
        print(f"[T2] world={world}  self-SIGTERM in {SELF_SIGTERM_DELAY_S}s", flush=True)
        print(f"[T2] Measuring NCCL collective latency + SIGTERM delivery delay...", flush=True)
        # Start background timer thread
        t = threading.Thread(
            target=_send_sigterm_after, args=(float(SELF_SIGTERM_DELAY_S),), daemon=True
        )
        t.start()

    # ── Warm up NCCL ────────────────────────────────────────────────────────
    for _ in range(5):
        dist.all_reduce(buf)
    torch.cuda.synchronize()

    # Separate 1-element tensor for stop-flag propagation.
    # Using ReduceOp.MAX means any rank raising the flag stops all ranks via the
    # same collective call — no mismatch risk at teardown.
    stop = torch.zeros(1, device=device)

    # ── Timed loop — measure collective latency ──────────────────────────────
    collective_times = []
    step = 0
    t_start = time.monotonic()

    while True:
        stop[0] = 1.0 if _handler_fired.is_set() else 0.0
        t0 = time.monotonic()
        # This is the "blind window" where SIGTERM cannot be processed.
        dist.all_reduce(buf, op=dist.ReduceOp.SUM)
        dist.all_reduce(stop, op=dist.ReduceOp.MAX)
        torch.cuda.synchronize()
        dt = time.monotonic() - t0
        if stop[0].item() > 0.5:
            break
        collective_times.append(dt)
        step += 1

        if rank == 0 and step % 20 == 0:
            avg_ms = 1000 * sum(collective_times[-20:]) / 20
            print(f"[T2] step={step}  avg_collective={avg_ms:.1f}ms", flush=True)

    # ── Report ───────────────────────────────────────────────────────────────
    if rank == 0 and collective_times:
        avg_ms = 1000 * sum(collective_times) / len(collective_times)
        max_ms = 1000 * max(collective_times)
        delay_ms = (_sigterm_recv_at - _sigterm_sent_at) * 1000
        total_s = time.monotonic() - t_start

        print(f"\n[T2] ══════════════════════════════════════════", flush=True)
        print(f"[T2] RESULTS (rank=0, world={world})", flush=True)
        print(f"[T2]   all_reduce tensor:  {TENSOR_NUMEL * 4 / 1e6:.1f} MB", flush=True)
        print(f"[T2]   steps completed:    {step}", flush=True)
        print(f"[T2]   avg collective:     {avg_ms:.1f} ms", flush=True)
        print(f"[T2]   max collective:     {max_ms:.1f} ms", flush=True)
        print(f"[T2]   SIGTERM delay:      {delay_ms:.1f} ms", flush=True)
        print(f"[T2]   ══════════════════════════════════════════", flush=True)
        print(f"[T2]   INTERPRETATION:", flush=True)
        if delay_ms > 1.0:
            print(f"[T2]   ✓ MECHANISM CONFIRMED — SIGTERM was delayed {delay_ms:.1f}ms", flush=True)
            print(f"[T2]     handler could not fire while main thread was inside NCCL C++", flush=True)
        else:
            print(f"[T2]   ✗ Signal fired immediately — NCCL may have been idle", flush=True)
        print(f"[T2] ══════════════════════════════════════════\n", flush=True)

    # ── Clean teardown ───────────────────────────────────────────────────────
    # Both ranks exited the loop via the shared MAX all_reduce — no mismatch.
    dist.destroy_process_group()
    torch.cuda.empty_cache()
    sys.exit(0)


if __name__ == "__main__":
    main()
