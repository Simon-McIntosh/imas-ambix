"""Build the MAST Level-2 inventory manifest.

Walks a duration-spread sample of L2 shots and records, per
``group → var``: dims, units, ``uda_name``, native dt / rate, finite
fraction, per-shot coverage, and the reconstruction-vs-plan
classification (:mod:`imas_ambix.data.provenance`). It also flags the
known name-collision pairs (``magnetics.ip`` vs ``summary.ip`` vs
``pulse_schedule.i_plasma`` — measured vs measured-summary vs planned)
and the realised-vs-planned actuator pairs (``pf_active.coil_current``
realised vs ``pf_active.coil_voltage`` planned feed-forward;
``gas_injection`` measured flows vs valve demands).

The output is a machine-readable JSON artifact that downstream code and
review use as the airtight statement of *what is in the L2 input set and
why each field is in / out*.

Run (CPU-only, reads GPFS directly via xarray/zarr — no IMAS AL, no
h5py)::

    uv run python -m imas_ambix.data.l2_inventory --n-shots 24 \
        --out imas_ambix/data/artifacts/l2_inventory.json
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

from imas_ambix.data.paths import LEVEL2_DIR
from imas_ambix.data.provenance import (
    ALL_CLASSIFICATIONS,
    classify_l2_field,
)

logger = logging.getLogger(__name__)


# Name-collision families: the same physical quantity appears under
# distinct (group, var) keys with distinct provenance — KEEP distinct.
NAME_COLLISIONS: dict[str, list[tuple[str, str]]] = {
    "plasma_current": [
        ("magnetics", "ip"),  # measured (AMC)
        ("summary", "ip"),  # measured summary (AMC)
        ("pulse_schedule", "i_plasma"),  # planned demand (XDC)
    ],
    "line_density": [
        ("interferometer", "n_e_line"),  # measured (ANE)
        ("pulse_schedule", "n_e_line"),  # planned demand (XDC)
        ("summary", "line_average_n_e"),  # reconstruction-derived (ESM)
    ],
}

# Realised-vs-planned actuator pairs (the model needs both — the plan to
# predict from, the realised to measure against).
ACTUATOR_PAIRS: list[dict[str, object]] = [
    {
        "actuator": "pf_coils",
        "realised": ("pf_active", "coil_current"),  # measured AMC
        "planned": ("pf_active", "coil_voltage"),  # feed-forward XDC_PF_F
    },
    {
        "actuator": "gas_valves",
        "realised": ("gas_injection", "inboard_total"),  # measured AGA flow
        "planned": ("gas_injection", "valve_voltage"),  # demand XDC_GAS_F
    },
    {
        "actuator": "plasma_current",
        "realised": ("magnetics", "ip"),  # measured AMC
        "planned": ("pulse_schedule", "i_plasma"),  # demand XDC_IP
    },
    {
        "actuator": "line_density",
        "realised": ("interferometer", "n_e_line"),  # measured ANE
        "planned": ("pulse_schedule", "n_e_line"),  # demand XDC_DENSITY
    },
]


def _open_root(shot_path: Path):
    import zarr  # noqa: PLC0415

    return zarr.open_group(str(shot_path), mode="r")


def _list_groups(shot_path: Path) -> list[str]:
    try:
        root = _open_root(shot_path)
        return sorted(root.group_keys())
    except Exception as e:  # pragma: no cover - corpus robustness
        logger.debug("cannot list groups for %s: %s", shot_path, e)
        return []


def _shot_duration(shot_path: Path) -> float | None:
    """Cheap duration probe from the magnetics time base (s)."""
    import xarray as xr  # noqa: PLC0415

    for grp in ("magnetics", "summary", "interferometer"):
        try:
            ds = xr.open_zarr(f"{shot_path}/{grp}", consolidated=False)
        except Exception:
            continue
        if "time" in ds.coords:
            t = np.asarray(ds["time"].values, dtype=np.float64)
            if t.size > 1 and np.isfinite(t).any():
                t = t[np.isfinite(t)]
                return float(t.max() - t.min())
    return None


def select_shots(n_shots: int, level2_dir: Path = LEVEL2_DIR) -> list[int]:
    """Pick ``n_shots`` spread across low / mid / high duration.

    Probes duration on a moderately sized candidate pool, sorts, and takes
    an even spread across the duration range so the manifest's coverage
    statistics are not biased toward one regime.
    """
    paths = sorted(level2_dir.glob("*.zarr"))
    if not paths:
        return []
    # Candidate pool: evenly spaced across the corpus to avoid a local
    # (chronological) bias, capped so the duration probe stays cheap.
    pool_size = min(len(paths), max(n_shots * 6, 120))
    step = max(1, len(paths) // pool_size)
    candidates = paths[::step][:pool_size]

    durations: list[tuple[float, int]] = []
    for p in candidates:
        d = _shot_duration(p)
        if d is not None and d > 0:
            sid = int(p.stem)
            durations.append((d, sid))
    if not durations:
        return [int(p.stem) for p in paths[:n_shots]]

    durations.sort()
    # Even spread across the sorted-by-duration list (low→high).
    idx = np.linspace(0, len(durations) - 1, num=min(n_shots, len(durations)))
    chosen = sorted({durations[int(round(i))][1] for i in idx})
    return chosen


def _time_dim(dims: tuple[str, ...]) -> str | None:
    for d in dims:
        if d == "time" or d.startswith("time"):
            return d
    return None


def _native_dt_rate(ds, time_dim: str) -> tuple[float | None, float | None]:
    src = None
    if time_dim in ds.coords:
        src = ds.coords[time_dim]
    elif time_dim in ds:
        src = ds[time_dim]
    if src is None:
        return None, None
    t = np.asarray(src.values, dtype=np.float64)
    t = t[np.isfinite(t)]
    if t.size < 2:
        return None, None
    dt = float(np.median(np.diff(np.sort(t))))
    if dt <= 0:
        return None, None
    return dt, float(1.0 / dt)


def _finite_fraction(da) -> float:
    vals = np.asarray(da.values)
    if vals.size == 0:
        return 0.0
    if not np.issubdtype(vals.dtype, np.floating):
        return 1.0
    return float(np.isfinite(vals).mean())


def build_manifest(
    n_shots: int = 24,
    level2_dir: Path = LEVEL2_DIR,
) -> dict:
    """Build the L2 inventory manifest over a duration-spread sample."""
    import xarray as xr  # noqa: PLC0415

    shots = select_shots(n_shots, level2_dir=level2_dir)
    logger.info("inventory over %d shots: %s", len(shots), shots)

    # field_key -> aggregate record
    fields: dict[tuple[str, str], dict] = {}
    # per-field per-shot presence + finite fraction
    presence: dict[tuple[str, str], int] = defaultdict(int)
    finite_acc: dict[tuple[str, str], list[float]] = defaultdict(list)
    dt_acc: dict[tuple[str, str], list[float]] = defaultdict(list)
    shot_durations: dict[int, float | None] = {}

    n_seen = 0
    for sid in shots:
        shot_path = level2_dir / f"{sid}.zarr"
        groups = _list_groups(shot_path)
        if not groups:
            continue
        n_seen += 1
        shot_durations[sid] = _shot_duration(shot_path)
        for g in groups:
            try:
                ds = xr.open_zarr(f"{shot_path}/{g}", consolidated=False)
            except Exception as e:  # pragma: no cover - corpus robustness
                logger.debug("cannot open %s/%s: %s", sid, g, e)
                continue
            for v in ds.data_vars:
                da = ds[v]
                uda = da.attrs.get("uda_name")
                key = (g, v)
                presence[key] += 1
                finite_acc[key].append(_finite_fraction(da))
                tdim = _time_dim(tuple(da.dims))
                if tdim is not None:
                    dt, _ = _native_dt_rate(ds, tdim)
                    if dt is not None:
                        dt_acc[key].append(dt)
                if key not in fields:
                    fc = classify_l2_field(g, v, uda)
                    fields[key] = {
                        "group": g,
                        "var": v,
                        "uda_name": (None if uda is None else str(uda)),
                        "source": fc.source,
                        "level1_system": fc.level1_system,
                        "classification": fc.classification,
                        "reason": fc.reason,
                        "review": fc.review,
                        "dims": list(da.dims),
                        "units": (
                            None
                            if da.attrs.get("units") is None
                            else str(da.attrs.get("units"))
                        ),
                        "time_dim": tdim,
                    }

    # Finalise per-field aggregates.
    field_records = []
    for key, rec in sorted(fields.items()):
        cov = presence[key] / n_seen if n_seen else 0.0
        ff = finite_acc[key]
        dts = dt_acc[key]
        rec = dict(rec)
        rec["coverage"] = round(cov, 4)
        rec["n_shots_present"] = presence[key]
        rec["finite_fraction_mean"] = round(float(np.mean(ff)), 4) if ff else None
        rec["native_dt"] = round(float(np.median(dts)), 8) if dts else None
        rec["native_rate_hz"] = round(float(1.0 / np.median(dts)), 2) if dts else None
        field_records.append(rec)

    # Classification tallies.
    tally = {c: 0 for c in ALL_CLASSIFICATIONS}
    for rec in field_records:
        tally[rec["classification"]] += 1
    review_fields = [
        {
            "group": r["group"],
            "var": r["var"],
            "uda_name": r["uda_name"],
            "reason": r["reason"],
        }
        for r in field_records
        if r["review"]
    ]

    # Group summary.
    by_group: dict[str, dict[str, int]] = defaultdict(
        lambda: {c: 0 for c in ALL_CLASSIFICATIONS}
    )
    for rec in field_records:
        by_group[rec["group"]][rec["classification"]] += 1

    manifest = {
        "schema": "imas-ambix.l2-inventory.v1",
        "principle": (
            "reconstruction-vs-plan: banned = code-reconstructed state "
            "(EFM_/ESM_) + reconstruction-derived scalars + XDC "
            "reconstruction residuals; authorised = measured diagnostics "
            "(input) + planned pulse-schedule waveforms (planned-action); "
            "Da is a default-off probe target; geometry is infra."
        ),
        "level2_dir": str(level2_dir),
        "n_shots_requested": n_shots,
        "n_shots_inventoried": n_seen,
        "shots": shots,
        "shot_durations_s": {
            str(k): (round(v, 4) if v is not None else None)
            for k, v in shot_durations.items()
        },
        "n_fields": len(field_records),
        "classification_tally": tally,
        "by_group": {g: dict(t) for g, t in sorted(by_group.items())},
        "name_collisions": _collision_report(fields),
        "actuator_pairs": _actuator_pair_report(fields),
        "review_fields": review_fields,
        "fields": field_records,
    }
    return manifest


def _classification_of(fields: dict[tuple[str, str], dict], key) -> dict:
    rec = fields.get(tuple(key))
    if rec is None:
        return {"present": False}
    return {
        "present": True,
        "uda_name": rec["uda_name"],
        "source": rec["source"],
        "classification": rec["classification"],
    }


def _collision_report(fields) -> dict:
    out = {}
    for family, members in NAME_COLLISIONS.items():
        out[family] = {
            "keep_distinct": True,
            "members": [
                {"group": g, "var": v, **_classification_of(fields, (g, v))}
                for (g, v) in members
            ],
        }
    return out


def _actuator_pair_report(fields) -> list[dict]:
    out = []
    for pair in ACTUATOR_PAIRS:
        out.append(
            {
                "actuator": pair["actuator"],
                "realised": {
                    "group": pair["realised"][0],
                    "var": pair["realised"][1],
                    **_classification_of(fields, pair["realised"]),
                },
                "planned": {
                    "group": pair["planned"][0],
                    "var": pair["planned"][1],
                    **_classification_of(fields, pair["planned"]),
                },
            }
        )
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Build the L2 inventory manifest.")
    p.add_argument("--n-shots", type=int, default=24)
    p.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).parent / "artifacts" / "l2_inventory.json",
    )
    p.add_argument("--level2-dir", type=Path, default=LEVEL2_DIR)
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    manifest = build_manifest(n_shots=args.n_shots, level2_dir=args.level2_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2, sort_keys=False))
    logger.info(
        "wrote %s: %d fields over %d shots — %s",
        args.out,
        manifest["n_fields"],
        manifest["n_shots_inventoried"],
        manifest["classification_tally"],
    )
    if manifest["review_fields"]:
        logger.info(
            "REVIEW: %d field(s) flagged for review",
            len(manifest["review_fields"]),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
