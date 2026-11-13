#!/usr/bin/env python3
"""Windowed GPU+host sampler — runs ON the compute node via srun --overlap.

Samples nvidia-smi once per second. For every GPU the step's cgroup exposes it
reports the PHYSICAL card index — mapped from ``SLURM_STEP_GPUS`` so a step given
physical cards 4,5 reports 4,5, not the step-relative 0,1 — plus the UUID and
util% + mem%. Also node CPU% (from /proc/stat) and host memory.

Emits parseable lines on stdout (all numbers already computed). Environment
fields (die temperature, power draw, board power cap) are APPENDED to the GPU
line so a reader written against the earlier format keeps working -- it reads
the leading fields it knows and ignores the rest:
    stream mode (one block per interval, TICK-delimited):
        GPU<TAB>phys<TAB>uuid<TAB>util_now<TAB>mem_now_pct<TAB>temp_c<TAB>watts<TAB>cap_w
        NODE<TAB>cpu_pct<TAB>mem_used_gb<TAB>mem_total_gb
        TICK
    window mode (one block after <window> s):
        GPU<TAB>phys<TAB>uuid<TAB>util_now<TAB>util_avg<TAB>mem_now_pct<TAB>mem_avg_pct
           <TAB>temp_max_c<TAB>watts_mean<TAB>cap_w
        NODE<TAB>cpu_pct_avg<TAB>mem_used_gb<TAB>mem_total_gb

A card that does not report temperature or power (a MIG instance, a driver that
withholds the board sensor) emits an empty field rather than a zero, so the
reader can leave it out of an average instead of dragging the average down.

CUDA_VISIBLE_DEVICES is cleared so nvidia-smi reports every GPU the cgroup
exposes; the physical index comes from SLURM_STEP_GPUS, not from CVD ordering.
A step only sees the cards in its OWN cgroup (the GPUs SLURM bound to its CPUs);
cross-NUMA cards of the same job are NOT visible here — the parent unions
samplers across all owned jobs and marks any remaining node card unsampled.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time


def _step_phys() -> list[str]:
    """Physical GPU indices for this step (nvidia-smi enumeration order)."""
    raw = os.environ.get("SLURM_STEP_GPUS") or os.environ.get("SLURM_JOB_GPUS") or ""
    return [x for x in raw.split(",") if x != ""]


def _cpu_times() -> tuple[int, int]:
    with open("/proc/stat") as f:
        v = [int(x) for x in f.readline().split()[1:]]
    idle = v[3] + (v[4] if len(v) > 4 else 0)  # idle + iowait
    return idle, sum(v)


def _mem_gb() -> tuple[float, float]:
    info = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, _, rest = line.partition(":")
            info[k] = int(rest.split()[0])  # kB
    total = info.get("MemTotal", 0) / 1e6
    avail = info.get("MemAvailable", info.get("MemFree", 0)) / 1e6
    return total - avail, total


def _fmt(value: float | None, places: int) -> str:
    """A sensor reading for the wire, or an empty field where there is none."""
    return "" if value is None else f"{value:.{places}f}"


def _opt(value: str) -> float | None:
    """A sensor reading, or None where nvidia-smi answered '[N/A]'."""
    try:
        return float(value)
    except ValueError:
        return None


def _query_gpus(env: dict, phys: list[str]) -> list[tuple]:
    """One nvidia-smi snapshot → per visible card:
    (phys_idx, uuid, util%, mem%, temp_c, watts, cap_w).

    The step-relative ``index`` column is mapped to the real node card number via
    ``phys`` (SLURM_STEP_GPUS) so each card is labelled by its physical index.
    Temperature and power come from the same call as utilisation because they
    are the same snapshot: sampling them separately would pair a die
    temperature with a utilisation measured a second earlier.
    """
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,gpu_uuid,utilization.gpu,memory.used,memory.total,"
                "temperature.gpu,power.draw,power.limit",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            env=env,
            timeout=10,
        ).stdout
    except Exception:
        out = ""
    rows = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            step_idx = int(parts[0])
            util = float(parts[2])
            mem_pct = 100.0 * float(parts[3]) / max(float(parts[4]), 1.0)
        except ValueError:
            continue
        pidx = phys[step_idx] if 0 <= step_idx < len(phys) else str(step_idx)
        temp = _opt(parts[5]) if len(parts) > 5 else None
        watts = _opt(parts[6]) if len(parts) > 6 else None
        cap = _opt(parts[7]) if len(parts) > 7 else None
        rows.append((pidx, parts[1], util, mem_pct, temp, watts, cap))
    return rows


def stream(interval: int) -> int:
    """Emit one instantaneous sample block per `interval` seconds, forever.

    A single long-lived step replaces the old one-step-per-tick design (which
    exhausted the job's MaxStepCount lifetime step counter). The parent reads
    this stream and computes its own rolling means.
    """
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    phys = _step_phys()
    prev_idle, prev_tot = _cpu_times()
    while True:
        time.sleep(max(1, interval))
        for pidx, uuid, util, mem_pct, temp, watts, cap in _query_gpus(env, phys):
            print(
                f"GPU\t{pidx}\t{uuid}\t{util:.0f}\t{mem_pct:.0f}"
                f"\t{_fmt(temp, 0)}\t{_fmt(watts, 0)}\t{_fmt(cap, 0)}"
            )
        idle, tot = _cpu_times()
        dt = max(tot - prev_tot, 1)
        cpu_pct = 100.0 * (1.0 - (idle - prev_idle) / dt)
        prev_idle, prev_tot = idle, tot
        used, total = _mem_gb()
        print(f"NODE\t{cpu_pct:.0f}\t{used:.0f}\t{total:.0f}")
        print("TICK", flush=True)


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "stream":
        interval = int(sys.argv[2]) if len(sys.argv) > 2 else 1
        return stream(interval)

    window = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    phys = _step_phys()

    i0, t0 = _cpu_times()
    u_samp: dict[str, list[float]] = {}
    m_samp: dict[str, list[float]] = {}
    t_samp: dict[str, list[float]] = {}
    p_samp: dict[str, list[float]] = {}
    cap_of: dict[str, float] = {}
    last: dict[str, tuple[float, float]] = {}
    uuid_of: dict[str, str] = {}
    deadline = time.time() + window
    while time.time() < deadline:
        for pidx, uuid, util, mem_pct, temp, watts, cap in _query_gpus(env, phys):
            u_samp.setdefault(pidx, []).append(util)
            m_samp.setdefault(pidx, []).append(mem_pct)
            if temp is not None:
                t_samp.setdefault(pidx, []).append(temp)
            if watts is not None:
                p_samp.setdefault(pidx, []).append(watts)
            if cap is not None:
                cap_of[pidx] = cap
            last[pidx] = (util, mem_pct)
            uuid_of[pidx] = uuid
        time.sleep(1)

    i1, t1 = _cpu_times()
    dt = max(t1 - t0, 1)
    cpu_pct = 100.0 * (1.0 - (i1 - i0) / dt)
    mem_used, mem_total = _mem_gb()

    for pidx in u_samp:
        us, ms = u_samp[pidx], m_samp[pidx]
        u_avg = sum(us) / len(us)
        m_avg = sum(ms) / len(ms)
        u_now, m_now = last.get(pidx, (0.0, 0.0))
        # Peak temperature, because a card that touched its throttle point once
        # in the window is the fact worth carrying; mean power, because the
        # question power answers is energy over the window, not its last instant.
        ts, ps = t_samp.get(pidx) or [], p_samp.get(pidx) or []
        temp = max(ts) if ts else None
        watts = (sum(ps) / len(ps)) if ps else None
        print(
            f"GPU\t{pidx}\t{uuid_of.get(pidx, '')}\t{u_now:.0f}\t{u_avg:.0f}"
            f"\t{m_now:.0f}\t{m_avg:.0f}"
            f"\t{_fmt(temp, 0)}\t{_fmt(watts, 0)}\t{_fmt(cap_of.get(pidx), 0)}"
        )
    print(f"NODE\t{cpu_pct:.0f}\t{mem_used:.0f}\t{mem_total:.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
