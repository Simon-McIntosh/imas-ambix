"""Stage an L2 measured group as a conditioning SIGNAL stream (label-agnostic).

The L2 set is calibrated/processed and usually the better training input, but its
IMAS dd-path NAMES are NOT reliably mapped (lead directive: ignore the dd
semantics).  So this stager is label-agnostic: it keeps the numeric value channels
aligned to the group's primary moderate-cadence ``time`` axis, records their
on-disk names only as provenance, and the downstream quantise-on-read loader
quantises the VALUES.

Handles both 1-D traces (time,) and 2-D channel arrays (n_channel, n_time) — the
latter are transposed to (n_time, n_channel) and stored as a profile, so the
calibrated poloidal field probes / flux loops (magnetics) come through as channels.

Multiple time axes (e.g. magnetics ``time`` / ``time_mirnov`` / ``time_saddle``):
only arrays aligned to the chosen ``--time-key`` (default the shortest = the
moderate-cadence base) are kept; the MHz-rate fast arrays are a separate concern.

Leakage: EFIT reconstructions (``equilibrium`` and friends) must never be staged.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import zarr

L2_DIR = Path("/work/projects/imas_gpu/mast/level2/shots")
BANNED_GROUPS: frozenset[str] = frozenset({"equilibrium", "efm", "esm", "esx"})
_SKIP_SUFFIXES: tuple[str, ...] = (
    "_error",
    "_status",
    "_quality",
    "_channel",
    "_geometry_channel",
    "_r",
    "_z",
    "_phi",
    "_phi_1",
    "_phi_2",
    "_length",
    "_name",
)


def _primary_time_key(grp, arrkeys: set[str], override: str | None) -> str | None:
    if override:
        return override if override in arrkeys else None
    times = [a for a in arrkeys if a == "time" or a.endswith("_time") or a.startswith("time")]
    if not times:
        return None
    # the SHORTEST 1-D time axis = the moderate-cadence base (not the MHz fast one).
    best, best_n = None, None
    for a in times:
        try:
            if grp[a].ndim != 1:
                continue
            n = int(grp[a].shape[0])
        except Exception:  # noqa: BLE001
            continue
        if n >= 2 and (best_n is None or n < best_n):
            best, best_n = a, n
    return best


def stage_shot(shot_id: int, group: str, out_root: Path, time_key: str | None) -> tuple[bool, str]:
    if group in BANNED_GROUPS:
        return False, "banned group"
    shot_path = L2_DIR / f"{shot_id}.zarr"
    if not shot_path.exists():
        return False, "no shot"
    try:
        root = zarr.open_group(str(shot_path), mode="r")
    except Exception as exc:  # noqa: BLE001
        return False, f"open: {exc}"
    if group not in set(root.group_keys()):
        return False, "no group"
    grp = root[group]
    ak = set(grp.array_keys())
    tkey = _primary_time_key(grp, ak, time_key)
    if tkey is None:
        return False, "no time axis"
    t = np.asarray(grp[tkey], dtype=np.float64)
    if t.ndim != 1 or t.size < 2:
        return False, "degenerate time"
    nt = int(t.shape[0])

    traces: dict[str, np.ndarray] = {}
    profiles: dict[str, np.ndarray] = {}
    for k in sorted(ak):
        if k == tkey or any(k.endswith(s) for s in _SKIP_SUFFIXES) or k.startswith("time"):
            continue
        try:
            v = np.asarray(grp[k], dtype=np.float32)
        except Exception:  # noqa: BLE001
            continue
        if v.ndim == 1 and v.shape[0] == nt:
            traces[k] = v
        elif v.ndim == 2 and nt in v.shape:
            # orient to (n_time, n_channel)
            prof = v if v.shape[0] == nt else v.T
            profiles[k] = prof
    if not traces and not profiles:
        return False, "no aligned channels"

    out_path = out_root / group / str(shot_id) / f"{group}.zarr"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    store = zarr.open_group(str(out_path), mode="w")
    store.create_array("time", data=t.astype(np.float32))
    for k, v in traces.items():
        store.create_array(k, data=v)
    for k, v in profiles.items():
        store.create_array(k, data=v)
    store.attrs.update(
        {
            "shot_id": int(shot_id),
            "source": group,
            "kind": "measured_signal_l2",
            "n_time": nt,
            "time_key": tkey,
            "traces": sorted(traces),
            "profiles": sorted(profiles),
        }
    )
    return True, ""


def _candidate_shots() -> list[int]:
    return sorted(
        int(p.name[:-5]) for p in L2_DIR.glob("*.zarr") if p.name[:-5].isdigit()
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--group", required=True)
    ap.add_argument("--out-root", default="/work/projects/imas_gpu/mast-tokens/v1")
    ap.add_argument("--time-key", default=None, help="force a specific time axis")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--max-shots", type=int, default=None)
    ap.add_argument("--report", default=None)
    args = ap.parse_args(argv)

    group = str(args.group)
    if group in BANNED_GROUPS:
        print(f"[stage-l2] REFUSING banned group {group!r} (leakage)", flush=True)
        return 2
    out_root = Path(args.out_root) / f"signals-{group}"
    shots = _candidate_shots()
    n_shards = max(1, int(args.n_shards))
    shard = int(args.shard) % n_shards
    if n_shards > 1:
        shots = shots[shard::n_shards]
    if args.max_shots:
        shots = shots[: args.max_shots]
    print(f"[stage-l2 {group} {shard}/{n_shards}] {len(shots)} shots -> {out_root}", flush=True)

    n_ok = 0
    skips: dict[str, int] = {}
    t0 = time.monotonic()
    for i, sid in enumerate(shots):
        ok, reason = stage_shot(sid, group, out_root, args.time_key)
        if ok:
            n_ok += 1
        else:
            skips[reason] = skips.get(reason, 0) + 1
        if (i + 1) % 1000 == 0:
            print(f"[stage-l2 {group}]   {i + 1}/{len(shots)}, {n_ok} staged, {time.monotonic() - t0:.0f}s", flush=True)
    summary = {
        "group": group,
        "shard": shard,
        "n_shards": n_shards,
        "n_candidates": len(shots),
        "staged": n_ok,
        "skips": skips,
        "out_root": str(out_root),
        "elapsed_s": round(time.monotonic() - t0, 1),
    }
    print(json.dumps(summary, indent=2), flush=True)
    if args.report:
        Path(args.report).write_text(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
