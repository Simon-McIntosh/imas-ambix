"""Stage a measured L1 signal group as a conditioning SIGNAL stream.

Generic counterpart to ``encode_ait_signal.py``: for one L1 group (e.g. the
D-alpha boundary diagnostics ``ada`` / ``adg`` / ``aim``, or any measured
moderate-cadence group), extract every 1-D value trace on the group's own
``time`` axis into a per-shot signal Zarr the downstream uniform-quantiser
conditioning loader consumes (read like ``summary_l2`` / ``ait`` and quantised
on read to the L2 257-id vocab).

The dd-path/IMAS NAMES are NOT trusted (the mapping is not
reliable) — this stager is label-agnostic: it keeps the numeric (time,) channels
and records their on-disk array names as provenance only.  Static geometry,
``*_error`` / ``*_status`` / ``passnumber`` and scalar arrays are skipped.

Leakage stance: only MEASURED diagnostics are admissible.  EFIT reconstructions
(``efm`` / ``esm`` / ``esx`` / ``equilibrium``) must never be staged.

CPU work (read + reshape, no model) — runs on ``sun``, sharded like the ait stage.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import zarr

from imas_ambix.camdyn.dataset import level1_shot_path

#: Never stage these — EFIT reconstructions (leakage) or control reconstructions.
BANNED_GROUPS: frozenset[str] = frozenset({"efm", "esm", "esx", "equilibrium", "xdc"})

#: Array-name suffixes/keys that are not value channels.
_SKIP_SUFFIXES: tuple[str, ...] = ("_error", "_status", "_quality")
_SKIP_EXACT: frozenset[str] = frozenset({"passnumber", "status", "time", "quality"})


def _find_time_axis(arrkeys: set[str], g) -> str | None:
    """Pick the group's primary time axis: prefer 'time', else a *_time array."""
    if "time" in arrkeys:
        return "time"
    cands = sorted(a for a in arrkeys if a == "time" or a.endswith("_time"))
    # the longest such axis is the per-sample time base for the value traces.
    best, best_n = None, -1
    for a in cands:
        try:
            n = int(g[a].shape[0]) if g[a].ndim == 1 else -1
        except Exception:  # noqa: BLE001
            n = -1
        if n > best_n:
            best, best_n = a, n
    return best


def stage_shot(shot_id: int, group: str, out_root: Path) -> tuple[bool, str]:
    """Extract one shot's 1-D value traces for ``group`` into a per-shot Zarr."""
    if group in BANNED_GROUPS:
        return False, "banned group"
    try:
        g = zarr.open_group(str(level1_shot_path(int(shot_id))), mode="r")
    except Exception as exc:  # noqa: BLE001
        return False, f"open: {exc}"
    if group not in set(g.group_keys()):
        return False, "no group"
    grp = g[group]
    ak = set(grp.array_keys())
    tkey = _find_time_axis(ak, grp)
    if tkey is None:
        return False, "no time axis"
    t = np.asarray(grp[tkey], dtype=np.float64)
    if t.ndim != 1 or t.size < 2:
        return False, "degenerate time"
    nt = int(t.shape[0])

    kept: list[str] = []
    arrays: dict[str, np.ndarray] = {}
    for k in sorted(ak):
        if k == tkey or k in _SKIP_EXACT:
            continue
        if any(k.endswith(s) for s in _SKIP_SUFFIXES):
            continue
        try:
            v = np.asarray(grp[k], dtype=np.float32)
        except Exception:  # noqa: BLE001
            continue
        # keep only 1-D value traces aligned to this time axis (scalar metadata
        # and multi-dim/other-cadence arrays are skipped — a profile group needs
        # its own dedicated stager, like ait's qprofile path).
        if v.ndim == 1 and v.shape[0] == nt:
            arrays[k] = v
            kept.append(k)
    if not kept:
        return False, "no aligned traces"

    out_path = out_root / group / str(shot_id) / f"{group}.zarr"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    store = zarr.open_group(str(out_path), mode="w")
    store.create_array("time", data=t.astype(np.float32))
    for k, v in arrays.items():
        store.create_array(k, data=v)
    store.attrs.update(
        {
            "shot_id": int(shot_id),
            "source": group,
            "kind": "measured_signal",
            "n_time": nt,
            "traces": kept,
        }
    )
    return True, ""


def _candidate_shots() -> list[int]:
    from imas_ambix.data.paths import LEVEL1_DIR

    return sorted(
        int(p.name[:-5])
        for p in Path(LEVEL1_DIR).glob("*.zarr")
        if p.name[:-5].isdigit()
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--group", required=True, help="L1 group to stage (e.g. ada)")
    ap.add_argument(
        "--out-root",
        default="/work/projects/imas_gpu/mast-tokens/v1",
        help="staged stores land at <out-root>/signals-<group>/<group>/<shot>/<group>.zarr",
    )
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--max-shots", type=int, default=None)
    ap.add_argument("--report", default=None)
    args = ap.parse_args(argv)

    group = str(args.group)
    if group in BANNED_GROUPS:
        print(f"[stage] REFUSING banned group {group!r} (leakage)", flush=True)
        return 2

    out_root = Path(args.out_root) / f"signals-{group}"
    shots = _candidate_shots()
    n_shards = max(1, int(args.n_shards))
    shard = int(args.shard) % n_shards
    if n_shards > 1:
        shots = shots[shard::n_shards]
    if args.max_shots:
        shots = shots[: args.max_shots]
    print(
        f"[stage {group} shard {shard}/{n_shards}] {len(shots)} candidate shots "
        f"-> {out_root}",
        flush=True,
    )

    n_ok = 0
    skips: dict[str, int] = {}
    t0 = time.monotonic()
    for i, sid in enumerate(shots):
        ok, reason = stage_shot(sid, group, out_root)
        if ok:
            n_ok += 1
        else:
            skips[reason] = skips.get(reason, 0) + 1
        if (i + 1) % 1000 == 0:
            print(
                f"[stage {group}]   {i + 1}/{len(shots)} done, {n_ok} staged, "
                f"{time.monotonic() - t0:.0f}s",
                flush=True,
            )
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
