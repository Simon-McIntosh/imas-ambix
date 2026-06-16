"""Build the eval-only TARGET store from MAST Level-2 shots.

Reads each L2 shot's equilibrium reconstruction, the reconstruction-derived
global scalars, and the programmed / demanded reconstruction waveforms, and
writes them — with per-quantity finite/validity masks, on their native time
grids — to :data:`imas_ambix.data.paths.TARGET_ROOT` via
:func:`imas_ambix.tokenizer.store_targets.save_target_group`.

These are the world-model PREDICTION targets, never inputs.  The store lives
under its own root (Wall 1) and carries no token-vocabulary handle (Wall 2);
see :mod:`imas_ambix.tokenizer.store_targets` for the boundary contract.

Three target groups are written per shot:

``equilibrium``
    The EFIT/Solov'ev reconstruction: ψ(R,Z,t), j_φ(R,Z,t), q(profile,t),
    the last-closed-flux-surface boundary, the X-point and magnetic axis,
    and the equilibrium global scalars (β, l_i, W_MHD, q95, …).  ψ and j_φ
    are ~65×65×T; the R/Z grid is captured in the group attrs.
``derived_globals``
    The reconstruction-derived globals that are BANNED as inputs because
    they embed the EFIT boundary: ``greenwald_density`` (ESM_N_GREENWALD)
    and ``line_average_n_e`` (ESM_NE_BAR), plus the equilibrium loop voltage.
``programmed``
    The programmed / demanded waveforms the controller targeted
    (``i_plasma`` = XDC_IP_T_IPREF, ``n_e_line`` = XDC_DENSITY_T_NELREF).
    Stored as targets because they are the *reconstruction reference* the
    forward model's output is graded against.

NaN-outside-the-pulse-window is handled by the finite mask: a sample is
marked valid only where it is finite, and ``original_window`` records the
``[t_start, t_end]`` of the finite span on each group's native time base.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from imas_ambix.data.paths import LEVEL2_DIR, TARGET_ROOT
from imas_ambix.tokenizer.store_targets import (
    TargetV2Attrs,
    save_target_group,
    target_group_path,
)

logger = logging.getLogger(__name__)

# Where the canonical inventory shot list lives.
_INVENTORY = Path(__file__).resolve().parent / "artifacts" / "l2_inventory.json"


# ---------------------------------------------------------------------------
# What to capture, per group.  Each entry: (l2_group, [(var, out_name), ...]).
# The equilibrium ψ / j_φ / q gridded fields plus the EFM/ESM globals are the
# reconstruction; derived_globals are the banned reconstruction-derived
# scalars; programmed are the demanded reference waveforms.
# ---------------------------------------------------------------------------

EQUILIBRIUM_QUANTITIES: tuple[str, ...] = (
    # gridded reconstruction
    "psi",  # ψ(R,Z,t)   EFM_PSI(R,Z)
    "j_phi",  # j_φ(R,Z,t) EFM_PLASMA_CURR(R,Z)
    "q",  # q(profile,t) EFM_Q(R)
    # boundary / X-point / axis
    "lcfs_r",
    "lcfs_z",
    "x_point_r",
    "x_point_z",
    "magnetic_axis_r",
    "magnetic_axis_z",
    # global scalars
    "beta_pol",
    "beta_tor",
    "beta_tor_normal",
    "li",
    "wmhd",
    "elongation",
    "triangularity_lower",
    "triangularity_upper",
    "minor_radius",
    "q_axis",
    "q95",
    "q100",
    # reconstruction-derived loop voltage (ESM_) — banned as input; lives on
    # the equilibrium time grid, so it is captured here, not in derived_globals.
    "vloop_dynamic",  # ESM_V_LOOP_DYNAMIC
    "vloop_static",  # ESM_V_LOOP_STATIC
)

# Reconstruction-derived globals that are BANNED as world-model inputs
# (they embed the EFIT/Solov'ev boundary).  These live in the ``summary`` L2
# group on the summary time base; the equilibrium-group derived scalars
# (vloop_*) are captured in the equilibrium group on their own time grid.
DERIVED_GLOBAL_FIELDS: tuple[tuple[str, str, str], ...] = (
    ("summary", "greenwald_density", "greenwald_density"),  # ESM_N_GREENWALD
    ("summary", "line_average_n_e", "line_average_n_e"),  # ESM_NE_BAR
)

# Programmed / demanded reconstruction-reference waveforms.
PROGRAMMED_FIELDS: tuple[tuple[str, str, str], ...] = (
    ("pulse_schedule", "i_plasma", "i_plasma"),  # XDC_IP_T_IPREF
    ("pulse_schedule", "n_e_line", "n_e_line"),  # XDC_DENSITY_T_NELREF
)


@dataclass
class GroupResult:
    """Outcome of writing one target group for one shot."""

    group: str
    path: Path | None
    n_quantities: int
    quantity_names: tuple[str, ...]
    skipped_reason: str | None = None


def inventory_shot_ids() -> list[int]:
    """The canonical L2 target shot list from the inventory artifact."""
    data = json.loads(_INVENTORY.read_text())
    return [int(s) for s in data["shots"]]


def _open_group(shot_dir: Path, group: str):
    """Open one L2 zarr group with xarray, or return ``None`` if absent.

    Per task contract: ``xr.open_zarr(group_path, consolidated=False)`` — NOT
    imas-python, NOT h5py (this is FAIR-MAST IMAS-mapped Zarr, not AL data).
    """
    import xarray as xr

    gpath = shot_dir / group
    if not gpath.exists():
        return None
    try:
        return xr.open_zarr(str(gpath), consolidated=False)
    except Exception as exc:  # noqa: BLE001 — surface, then skip the group
        logger.warning("could not open %s: %s", gpath, exc)
        return None


def _to_time_last(da, time_dim: str) -> np.ndarray:
    """Return the variable's values with the time axis last (L2 convention)."""
    if time_dim in da.dims and da.dims[-1] != time_dim:
        da = da.transpose(..., time_dim)
    return np.asarray(da.values, dtype=np.float64)


def _finite_window(values: np.ndarray, time_axis: int) -> tuple[float, float]:
    """The ``[t_start, t_end]`` index span (as floats) of any-finite slices.

    Returns ``(0.0, 0.0)`` when nothing is finite.  The caller maps these
    indices onto the native time base to fill ``original_window``.
    """
    if values.size == 0:
        return (0.0, 0.0)
    finite = np.isfinite(values)
    # Collapse every non-time axis → per-time-slice "any finite".
    axes = tuple(a for a in range(values.ndim) if a != time_axis)
    per_t = finite.any(axis=axes) if axes else finite
    idx = np.where(per_t)[0]
    if idx.size == 0:
        return (0.0, 0.0)
    return (float(idx.min()), float(idx.max()))


def _collect_group_arrays(
    ds, specs, time_dim: str
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[str], list[str]]:
    """Pull (var → values, var → finite-mask) for present specs in ``ds``.

    ``specs`` is an iterable of ``(var, out_name)``.  Absent variables are
    skipped (presence-guard); a finite mask is built per quantity so
    NaN-outside-window is never silently zero-filled.
    """
    arrays: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}
    names: list[str] = []
    units: list[str] = []
    for var, out_name in specs:
        if var not in ds.variables:
            continue
        da = ds[var]
        vals = _to_time_last(da, time_dim)
        mask = np.isfinite(vals)
        arrays[out_name] = vals
        masks[out_name] = mask
        names.append(out_name)
        units.append(str(da.attrs.get("units", "")))
    return arrays, masks, names, units


def _native_time(ds, time_dim: str) -> np.ndarray:
    if time_dim in ds.coords:
        return np.asarray(ds[time_dim].values, dtype=np.float64)
    if time_dim in ds.variables:
        return np.asarray(ds[time_dim].values, dtype=np.float64)
    return np.asarray([], dtype=np.float64)


def _window_from_indices(
    time: np.ndarray, arrays: dict[str, np.ndarray], time_axis: int
) -> tuple[float, float]:
    """Map the union of per-quantity finite index spans onto native time."""
    if time.size == 0 or not arrays:
        return (0.0, 0.0)
    lo = None
    hi = None
    for vals in arrays.values():
        ax = time_axis if vals.ndim > time_axis else vals.ndim - 1
        i0, i1 = _finite_window(vals, ax)
        if i0 == 0.0 and i1 == 0.0 and not np.isfinite(vals).any():
            continue
        lo = i0 if lo is None else min(lo, i0)
        hi = i1 if hi is None else max(hi, i1)
    if lo is None or hi is None:
        return (0.0, 0.0)
    lo_i = int(min(max(lo, 0), len(time) - 1))
    hi_i = int(min(max(hi, 0), len(time) - 1))
    return (float(time[lo_i]), float(time[hi_i]))


def build_equilibrium_group(
    shot_id: int, shot_dir: Path, *, target_root: Path | None = None
) -> GroupResult:
    """Write the ``equilibrium`` target group for one shot."""
    ds = _open_group(shot_dir, "equilibrium")
    if ds is None:
        return GroupResult("equilibrium", None, 0, (), "equilibrium group absent")
    try:
        time = _native_time(ds, "time")
        grid_r = (
            np.asarray(ds["major_radius"].values, dtype=np.float64)
            if "major_radius" in ds.coords or "major_radius" in ds.variables
            else np.asarray([], dtype=np.float64)
        )
        grid_z = (
            np.asarray(ds["z"].values, dtype=np.float64)
            if "z" in ds.coords or "z" in ds.variables
            else np.asarray([], dtype=np.float64)
        )
        specs = [(v, v) for v in EQUILIBRIUM_QUANTITIES]
        arrays, masks, names, units = _collect_group_arrays(ds, specs, "time")
        if not names:
            return GroupResult(
                "equilibrium", None, 0, (), "no equilibrium quantities present"
            )
        # time axis is last for every captured quantity (L2 convention).
        window = _window_from_indices(
            time, arrays, time_axis=max(0, max(v.ndim for v in arrays.values()) - 1)
        )
        attrs = TargetV2Attrs(
            quantity_names=tuple(names),
            units=tuple(units),
            grid_r=tuple(float(x) for x in grid_r),
            grid_z=tuple(float(x) for x in grid_z),
            time=tuple(float(x) for x in time),
            original_window=window,
            metadata={
                "source": "EFM/ESM reconstruction (eval-only target)",
                "uda_names": {n: str(ds[n].attrs.get("uda_name", "")) for n in names},
            },
        )
        path = save_target_group(
            shot_id, "equilibrium", arrays, masks, attrs, target_root=target_root
        )
        return GroupResult("equilibrium", path, len(names), tuple(names))
    finally:
        ds.close()


def _build_scalar_group(
    shot_id: int,
    shot_dir: Path,
    group_name: str,
    field_specs: tuple[tuple[str, str, str], ...],
    *,
    target_root: Path | None = None,
) -> GroupResult:
    """Write a per-(L2-group, time-dim) collection of scalar target quantities.

    ``field_specs`` are ``(l2_group, var, out_name)``.  Quantities are pulled
    from possibly several L2 groups but share a single native time base; we
    use the first present L2 group's time as canonical and skip any field
    whose host group's time length disagrees (kept honest, never resampled).
    """
    import xarray as xr  # noqa: F401 — ensures the xarray backend is importable

    by_l2: dict[str, list[tuple[str, str]]] = {}
    for l2_group, var, out_name in field_specs:
        by_l2.setdefault(l2_group, []).append((var, out_name))

    arrays: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}
    names: list[str] = []
    units: list[str] = []
    uda: dict[str, str] = {}
    canon_time: np.ndarray | None = None

    for l2_group, specs in by_l2.items():
        ds = _open_group(shot_dir, l2_group)
        if ds is None:
            continue
        try:
            t = _native_time(ds, "time")
            ga, gm, gn, gu = _collect_group_arrays(ds, specs, "time")
            if not gn:
                continue
            if canon_time is None:
                canon_time = t
            # Honest time alignment: only accept fields whose length matches
            # the canonical time base — never resample a target.
            for n, u in zip(gn, gu, strict=True):
                if ga[n].shape[-1] != len(canon_time):
                    logger.warning(
                        "shot %s %s.%s len %d != canon time %d — skipped",
                        shot_id,
                        l2_group,
                        n,
                        ga[n].shape[-1],
                        len(canon_time),
                    )
                    continue
                arrays[n] = ga[n]
                masks[n] = gm[n]
                names.append(n)
                units.append(u)
                uda[n] = str(ds[n].attrs.get("uda_name", ""))
        finally:
            ds.close()

    if not names or canon_time is None:
        return GroupResult(group_name, None, 0, (), "no quantities present")

    window = _window_from_indices(canon_time, arrays, time_axis=0)
    attrs = TargetV2Attrs(
        quantity_names=tuple(names),
        units=tuple(units),
        grid_r=(),
        grid_z=(),
        time=tuple(float(x) for x in canon_time),
        original_window=window,
        metadata={"uda_names": uda, "eval_only": True},
    )
    path = save_target_group(
        shot_id, group_name, arrays, masks, attrs, target_root=target_root
    )
    return GroupResult(group_name, path, len(names), tuple(names))


def build_derived_globals_group(
    shot_id: int, shot_dir: Path, *, target_root: Path | None = None
) -> GroupResult:
    """Write the banned reconstruction-derived globals target group."""
    return _build_scalar_group(
        shot_id,
        shot_dir,
        "derived_globals",
        DERIVED_GLOBAL_FIELDS,
        target_root=target_root,
    )


def build_programmed_group(
    shot_id: int, shot_dir: Path, *, target_root: Path | None = None
) -> GroupResult:
    """Write the programmed / demanded reference-waveform target group."""
    return _build_scalar_group(
        shot_id,
        shot_dir,
        "programmed",
        PROGRAMMED_FIELDS,
        target_root=target_root,
    )


def build_shot(
    shot_id: int,
    *,
    level2_dir: Path | None = None,
    target_root: Path | None = None,
    skip_existing: bool = True,
) -> list[GroupResult]:
    """Build all target groups for one shot.

    Returns a :class:`GroupResult` per group.  When ``skip_existing`` and the
    equilibrium store already exists, the shot is skipped (resume-safe).
    """
    l2 = level2_dir or LEVEL2_DIR
    shot_dir = l2 / f"{shot_id}.zarr"
    if not shot_dir.exists():
        return [GroupResult("equilibrium", None, 0, (), f"L2 shot {shot_id} absent")]

    if (
        skip_existing
        and target_group_path(shot_id, "equilibrium", target_root=target_root).exists()
    ):
        return [
            GroupResult("equilibrium", None, 0, (), "skip-existing (already built)")
        ]

    results = [
        build_equilibrium_group(shot_id, shot_dir, target_root=target_root),
        build_derived_globals_group(shot_id, shot_dir, target_root=target_root),
        build_programmed_group(shot_id, shot_dir, target_root=target_root),
    ]
    return results


def _parse_shots(text: str | None) -> list[int]:
    """Resolve a shot selector: ``all`` | comma/space list | file path."""
    if text is None or text == "all":
        return inventory_shot_ids()
    p = Path(text)
    if p.exists():
        return [int(x) for x in p.read_text().split()]
    return [int(x) for x in text.replace(",", " ").split()]


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--shots",
        default="all",
        help="'all' (inventory list), a comma/space list, or a file path.",
    )
    p.add_argument(
        "--target-root",
        default=None,
        help="Override TARGET_ROOT (for smoke tests). Default: paths.TARGET_ROOT.",
    )
    p.add_argument(
        "--level2-dir",
        default=None,
        help="Override the L2 shots dir. Default: paths.LEVEL2_DIR.",
    )
    p.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Rebuild shots whose target store already exists.",
    )
    p.add_argument("--manifest", default=None, help="Write a JSON run manifest here.")
    args = p.parse_args(argv)

    target_root = Path(args.target_root) if args.target_root else TARGET_ROOT
    level2_dir = Path(args.level2_dir) if args.level2_dir else LEVEL2_DIR
    shots = _parse_shots(args.shots)
    logger.info(
        "building %d target shots → %s (skip_existing=%s)",
        len(shots),
        target_root,
        not args.no_skip_existing,
    )

    manifest: dict[str, object] = {
        "target_root": str(target_root),
        "level2_dir": str(level2_dir),
        "n_shots": len(shots),
        "shots": {},
    }
    n_built = n_skipped = n_failed = 0
    for shot_id in shots:
        try:
            results = build_shot(
                shot_id,
                level2_dir=level2_dir,
                target_root=target_root,
                skip_existing=not args.no_skip_existing,
            )
        except Exception as exc:  # noqa: BLE001 — record + continue the corpus
            logger.exception("shot %s failed", shot_id)
            manifest["shots"][str(shot_id)] = {"error": str(exc)}
            n_failed += 1
            continue
        rec = {
            r.group: {
                "n_quantities": r.n_quantities,
                "quantity_names": list(r.quantity_names),
                "path": str(r.path) if r.path else None,
                "skipped_reason": r.skipped_reason,
            }
            for r in results
        }
        manifest["shots"][str(shot_id)] = rec
        if any(r.path for r in results):
            n_built += 1
            logger.info(
                "shot %s built: %s",
                shot_id,
                {r.group: r.n_quantities for r in results if r.path},
            )
        else:
            n_skipped += 1

    manifest["n_built"] = n_built
    manifest["n_skipped"] = n_skipped
    manifest["n_failed"] = n_failed
    if args.manifest:
        Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
        Path(args.manifest).write_text(json.dumps(manifest, indent=2))
        logger.info("manifest → %s", args.manifest)
    logger.info("done: built=%d skipped=%d failed=%d", n_built, n_skipped, n_failed)
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
