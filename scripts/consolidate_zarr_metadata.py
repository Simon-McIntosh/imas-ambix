"""Consolidate Zarr metadata across the FAIR-MAST L1 and L2 corpora.

Writes a single ``.zmetadata`` file at each shot's Zarr root so subsequent
``xarray.open_zarr(..., consolidated=True)`` reads it in one GPFS round-trip
instead of recursively listing every group / array. On our corpus that's the
difference between an O(10 ms) open and an O(0.5-2 s) open.

Idempotent: safe to re-run; existing ``.zmetadata`` files are overwritten with
the current store contents.

Usage:

    # consolidate just the L2 corpus
    uv run python scripts/consolidate_zarr_metadata.py \\
        --root /work/projects/imas_gpu/mast/level2/shots --workers 16

    # or both at once with a measured before/after open-time check
    uv run python scripts/consolidate_zarr_metadata.py \\
        --root /work/projects/imas_gpu/mast/level1/shots \\
        --root /work/projects/imas_gpu/mast/level2/shots \\
        --workers 16 --measure

Output JSON (--report PATH):

    {
        "roots": [...],
        "n_consolidated": int,
        "n_skipped": int,
        "n_failed": int,
        "elapsed_s": float,
        "open_time_before_s": {shot_id: seconds, ...} | null,
        "open_time_after_s":  {shot_id: seconds, ...} | null,
        "errors": [{"shot_id": int, "error": str}]
    }
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def _consolidate_one(shot_zarr: Path) -> tuple[int, str | None]:
    """Consolidate a single ``<shot>.zarr`` root. Returns (shot_id, error|None)."""
    import zarr

    shot_id = int(shot_zarr.stem)
    try:
        zarr.consolidate_metadata(str(shot_zarr))
        return shot_id, None
    except Exception as exc:  # noqa: BLE001
        return shot_id, f"{type(exc).__name__}: {exc}"


def _measure_open(shot_zarr: Path, group: str, consolidated: bool) -> float:
    """Time a single ``xr.open_zarr`` call (group-level)."""
    import xarray as xr

    t0 = time.monotonic()
    ds = xr.open_zarr(str(shot_zarr / group), consolidated=consolidated)
    # touch the dataset so lazy work actually happens
    _ = list(ds.data_vars)
    return time.monotonic() - t0


def _pick_measure_targets(root: Path, max_n: int = 3) -> list[tuple[Path, str]]:
    """Pick a few (shot_zarr, group) pairs to time before/after."""
    shots = sorted(root.glob("*.zarr"))
    if not shots:
        return []
    # take 3 evenly-spaced shots
    n = len(shots)
    picks = [shots[0], shots[n // 2], shots[-1]] if n >= 3 else shots
    targets: list[tuple[Path, str]] = []
    for s in picks[:max_n]:
        # pick any one group inside the shot
        groups = [d for d in s.iterdir() if d.is_dir() and not d.name.startswith(".")]
        if groups:
            targets.append((s, groups[0].name))
    return targets


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        action="append",
        required=True,
        help=("Zarr corpus root containing <shot>.zarr directories. "
              "Pass multiple times to process several roots."),
    )
    parser.add_argument("--workers", type=int, default=16,
                        help="ThreadPool size (default 16). GPFS metadata is "
                             "I/O bound so high concurrency wins.")
    parser.add_argument("--measure", action="store_true",
                        help="Time xr.open_zarr on 3 representative shots "
                             "before AND after consolidation. Single-threaded.")
    parser.add_argument("--report", type=Path,
                        help="Write JSON report to this path.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process the first N shots (smoke testing).")
    args = parser.parse_args()

    roots = [Path(r) for r in args.root]
    for r in roots:
        if not r.is_dir():
            print(f"FATAL: root {r} is not a directory", file=sys.stderr)
            return 2

    report: dict = {
        "roots": [str(r) for r in roots],
        "n_consolidated": 0,
        "n_skipped": 0,
        "n_failed": 0,
        "errors": [],
        "open_time_before_s": None,
        "open_time_after_s": None,
    }

    # ── optional: measure BEFORE ────────────────────────────────────────
    if args.measure:
        before: dict[str, float] = {}
        for root in roots:
            for shot_zarr, group in _pick_measure_targets(root):
                key = f"{shot_zarr.stem}/{group}"
                before[key] = _measure_open(shot_zarr, group, consolidated=False)
                print(f"[before] {key}: {before[key]*1000:.1f} ms", flush=True)
        report["open_time_before_s"] = before

    # ── consolidate ────────────────────────────────────────────────────
    t0 = time.monotonic()
    for root in roots:
        shots = sorted(root.glob("*.zarr"))
        if args.limit:
            shots = shots[: args.limit]
        print(f"[{root.name}] {len(shots)} shots to consolidate "
              f"with {args.workers} workers", flush=True)

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(_consolidate_one, s): s for s in shots}
            done = 0
            for fut in as_completed(futs):
                shot_id, err = fut.result()
                done += 1
                if err is None:
                    report["n_consolidated"] += 1
                else:
                    report["n_failed"] += 1
                    report["errors"].append({"shot_id": shot_id, "error": err})
                if done % 500 == 0 or done == len(shots):
                    print(f"  [{root.name}] {done}/{len(shots)} done "
                          f"({report['n_failed']} failed)", flush=True)

    report["elapsed_s"] = round(time.monotonic() - t0, 1)
    print(f"\nconsolidation done in {report['elapsed_s']} s "
          f"({report['n_consolidated']} ok, {report['n_failed']} failed)",
          flush=True)

    # ── optional: measure AFTER ────────────────────────────────────────
    if args.measure:
        after: dict[str, float] = {}
        for root in roots:
            for shot_zarr, group in _pick_measure_targets(root):
                key = f"{shot_zarr.stem}/{group}"
                after[key] = _measure_open(shot_zarr, group, consolidated=True)
                print(f"[after]  {key}: {after[key]*1000:.1f} ms", flush=True)
        report["open_time_after_s"] = after

        if before:
            print("\nspeedup summary:")
            for k in before:
                if k in after and after[k] > 0:
                    speedup = before[k] / after[k]
                    print(f"  {k}: {before[k]*1000:6.1f} → "
                          f"{after[k]*1000:6.1f} ms  ({speedup:.1f}×)")

    if args.report:
        args.report.write_text(json.dumps(report, indent=2))
        print(f"\nreport written: {args.report}", flush=True)

    return 0 if report["n_failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
