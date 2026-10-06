"""Render the JT-60SA OP1 and OP2 poloidal cross-sections through imas-ink.

Two figures are written as SVG, one file each, under
``docs/figures/jt60sa-geometry/``:

* ``op1-op2-side-by-side.svg`` — OP1 and OP2 on two axes sharing one viewport.
* ``op1-op2-overlay.svg`` — both phases' first wall and vessel contours on one
  panel, told apart by line style alone.

Every element is drawn by :func:`imas_ink.figures.geometry_figure_mpl` in
imas-ink's own :data:`~imas_ink.DEFAULT_STYLE`; this script lays out the axes and
holds no colour, label or style constant of its own: the reference-outline line
style it borrows for the OP1 overlay comes from the style object itself.  The
phase description store is addressed through the packaged ``jt-60sa`` machine
map, and each phase's geometry is read with ``imas.DBEntry``.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import imas
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from imas_ink import DEFAULT_STYLE
from imas_ink.extract import extract_geometry
from imas_ink.figures import geometry_figure_mpl
from imas_ink.io import render_to_bytes

from imas_alambic.machine_map import (
    MachineMapCatalog,
    load_packaged_machine_map,
)
from imas_alambic.transform_engine import NetCDFTransformEngine

matplotlib.use("Agg")

CATALOGUE_MACHINE = "jt-60sa"
PHASES = ("OP1", "OP2")
GEOMETRY_IDS = ("wall", "pf_active", "magnetics")


def _read_ids(store_path: Path, ids_name: str, dd_version: str):
    """Read one IDS from the phase directory's ``{ids}.nc`` file."""
    with imas.DBEntry(
        str(store_path / f"{ids_name}.nc"), "r", dd_version=dd_version
    ) as entry:
        return entry.get(ids_name, autoconvert=False)


def _open_phase(
    catalogue: MachineMapCatalog, engine: NetCDFTransformEngine, phase: str
):
    """Open one phase map's description store and return it."""
    machine_map = next(item for item in catalogue.maps if item.name == phase)
    return engine.open(
        catalogue.description_store_root_path(),
        machine_map.first_shot,
        catalogue.dd_version,
        machine_map=machine_map,
        store_layout=catalogue.description_store_layout,
    )


def _store_counts(wall, pf, magnetics) -> dict[str, int]:
    """Count the geometry the store holds, before any extraction."""
    desc_2d = wall.description_2d[0]
    first_wall_points = int(
        sum(np.asarray(unit.outline.r).size for unit in desc_2d.limiter.unit)
    )
    vessel_points = 0
    for unit in desc_2d.vessel.unit:
        for skin in ("outline_inner", "outline_outer"):
            outline = getattr(unit.annular, skin, None)
            if outline is not None and np.asarray(outline.r).size:
                vessel_points += int(np.asarray(outline.r).size)
    return {
        "pf_coils": len(pf.coil),
        "filaments": int(sum(len(coil.element) for coil in pf.coil)),
        "first_wall_points": first_wall_points,
        "vessel_points": vessel_points,
        "probes": int(len(getattr(magnetics, "b_field_pol_probe", []))),
        "flux_loops": int(len(magnetics.flux_loop)),
    }


def _drawn_counts(geometry) -> dict[str, int]:
    """Count the geometry handed to imas-ink in one extraction."""
    wall_points = sum(np.asarray(r).size for r, _ in geometry.wall_units)
    vessel_points = sum(
        np.asarray(shell.r).size for shell in geometry.vessel_shells
    )
    return {
        "pf_coils": len(geometry.coil_rects),
        "first_wall_points": int(wall_points),
        "vessel_points": int(vessel_points),
        "probes": int(geometry.probe_r.size),
        "flux_loops": int(geometry.flux_loop_r.size),
    }


def _report(
    phase: str, store: dict[str, int], drawn: dict[str, int]
) -> None:
    """Print the per-phase counts and assert the drawn ones match the store."""
    print(f"[{phase}] passed to imas-ink:")
    print(
        f"  PF coils drawn            : {drawn['pf_coils']} "
        f"(store {store['pf_coils']})"
    )
    print(f"  filaments enclosed        : {store['filaments']}")
    print(
        f"  first-wall contour points : {drawn['first_wall_points']} "
        f"(store {store['first_wall_points']})"
    )
    print(
        f"  vessel-shell points       : {drawn['vessel_points']} "
        f"(store {store['vessel_points']})"
    )
    print(
        f"  flux loops passed         : {drawn['flux_loops']} "
        f"(store {store['flux_loops']})"
    )
    print(
        f"  probes passed             : {drawn['probes']} "
        f"(store {store['probes']})"
    )
    keys = ("pf_coils", "first_wall_points", "vessel_points", "probes", "flux_loops")
    mismatches = {
        key: (drawn[key], store[key]) for key in keys if drawn[key] != store[key]
    }
    if mismatches:
        raise SystemExit(f"[{phase}] drawn counts differ from the store: {mismatches}")


def _save_svg(fig, path: Path) -> None:
    """Write one figure as SVG, with no PNG twin."""
    path.write_bytes(render_to_bytes(fig, format="svg"))
    print(f"  wrote {path}")


def main() -> None:
    catalogue = load_packaged_machine_map(CATALOGUE_MACHINE)
    engine = NetCDFTransformEngine()

    out_dir = Path(__file__).resolve().parents[1] / "docs" / "figures"
    out_dir = out_dir / "jt60sa-geometry"
    out_dir.mkdir(parents=True, exist_ok=True)

    geometries = {}
    for phase in PHASES:
        store = _open_phase(catalogue, engine, phase)
        ids = {
            name: _read_ids(store.path, name, catalogue.dd_version)
            for name in GEOMETRY_IDS
        }
        geometry = extract_geometry(ids["wall"], ids["pf_active"], ids["magnetics"])
        counts = _store_counts(ids["wall"], ids["pf_active"], ids["magnetics"])
        _report(phase, counts, _drawn_counts(geometry))
        geometries[phase] = geometry

    # Figure 1: OP1 and OP2 side by side on one shared viewport.
    fig_side, axes = plt.subplots(
        1,
        len(PHASES),
        figsize=(12, 7),
        facecolor=DEFAULT_STYLE.figure_facecolor,
    )
    for ax, phase in zip(axes, PHASES, strict=True):
        geometry_figure_mpl(geometries[phase], style=DEFAULT_STYLE, ax=ax)
    _save_svg(fig_side, out_dir / "op1-op2-side-by-side.svg")

    # Figure 2: both phases overlaid on one panel, told apart by line style.
    op1_style = replace(
        DEFAULT_STYLE, wall_linestyle=DEFAULT_STYLE.ref_lcfs_linestyle
    )
    fig_overlay, ax = plt.subplots(
        1, 1, figsize=(6, 7), facecolor=DEFAULT_STYLE.figure_facecolor
    )
    geometry_figure_mpl(geometries["OP2"], style=DEFAULT_STYLE, ax=ax)
    geometry_figure_mpl(geometries["OP1"], style=op1_style, ax=ax)
    _save_svg(fig_overlay, out_dir / "op1-op2-overlay.svg")


if __name__ == "__main__":
    main()
