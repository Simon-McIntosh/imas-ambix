"""Draw the JT-60SA OP1/OP2 cross-section from the converted description store.

One reader turns a phase's store into drawable families (PF coil elements,
vessel filaments, the wall outline, flux loops and pickup probes); one
cross-section function draws them; and ``--palette-candidates`` renders the
candidate colour-role palettes onto the real OP1 and OP2 cross-section so the
choice can be made against the machine, not against a swatch.

The store is addressed through the catalogue that owns it: the phase directory
name comes from ``catalog.maps`` rather than a literal, and every IDS is read
with :class:`imas.DBEntry` so the figure shows what the converter wrote.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from imas_ambix.data.machine_map import (  # noqa: E402
    MachineMapCatalog,
    load_packaged_machine_map,
)
from imas_ambix.plot_style import (  # noqa: E402
    SERIES_LINEWIDTH,
    apply_data_ink,
    direct_label,
    palette,
)

IDS_NAMES = ("pf_active", "pf_passive", "wall", "magnetics")
PHASE_ORDER = ("OP1", "OP2")

# Candidate colour-role sets.  Each names a colour for every drawn component;
# the two vessel roles are lightness steps of one hue (the OP1 and OP2 vessels
# are the same structure at the two phase geometries) and the rest are distinct
# hues.  The lead chooses one; nothing downstream assumes which.
CANDIDATES: dict[str, dict[str, str]] = {
    "A-red-magenta": {
        "pf_coils": "#2a78d6",
        "vessel_op1": "#e0483a",
        "vessel_op2": "#9c2a20",
        "wall": "#1baf7a",
        "flux_loops": "#eda100",
        "pickup_probes": "#b83d9a",
    },
    "B-crimson-gold": {
        "pf_coils": "#2a78d6",
        "vessel_op1": "#d64a37",
        "vessel_op2": "#8f2a18",
        "wall": "#1baf7a",
        "flux_loops": "#e8a51c",
        "pickup_probes": "#b83d9a",
    },
    "C-magenta-wall": {
        "pf_coils": "#2a78d6",
        "vessel_op1": "#e0483a",
        "vessel_op2": "#9c2a20",
        "wall": "#b83d9a",
        "flux_loops": "#eda100",
        "pickup_probes": "#1baf7a",
    },
}


@dataclass(frozen=True)
class Rectangle2D:
    """A coil or filament rectangle given by its centre and extents."""

    r: float
    z: float
    width: float
    height: float


@dataclass(frozen=True)
class PhaseGeometry:
    """The drawable families a single phase's store yields."""

    phase: str
    coil_rects: tuple[Rectangle2D, ...]
    vessel_rects: tuple[Rectangle2D, ...]
    wall_r: tuple[float, ...]
    wall_z: tuple[float, ...]
    flux_r: tuple[float, ...]
    flux_z: tuple[float, ...]
    probe_r: tuple[float, ...]
    probe_z: tuple[float, ...]
    probe_angle: tuple[float, ...]


def _read_ids(phase_dir: Path, ids_name: str, dd_version: str):
    import imas  # noqa: PLC0415

    path = phase_dir / f"{ids_name}.nc"
    entry = imas.DBEntry(str(path), "r", dd_version=dd_version)
    try:
        return entry.get(ids_name, autoconvert=False)
    finally:
        entry.close()


def _rect_from_geometry(geometry) -> Rectangle2D:
    rect = geometry.rectangle
    return Rectangle2D(
        r=float(rect.r),
        z=float(rect.z),
        width=float(rect.width),
        height=float(rect.height),
    )


def _iter_rects(container) -> tuple[Rectangle2D, ...]:
    return tuple(_rect_from_geometry(element.geometry) for element in container)


def _wall_outline(wall) -> tuple[tuple[float, ...], tuple[float, ...]]:
    rs: list[float] = []
    zs: list[float] = []
    for description in wall.description_2d:
        for unit in description.limiter.unit:
            outline = unit.outline
            rs.extend(float(value) for value in outline.r)
            zs.extend(float(value) for value in outline.z)
    if not rs:
        for description in wall.description_2d:
            for unit in description.vessel.unit:
                outline = unit.outline
                rs.extend(float(value) for value in outline.r)
                zs.extend(float(value) for value in outline.z)
    return tuple(rs), tuple(zs)


def read_phase(catalog: MachineMapCatalog, phase: str) -> PhaseGeometry:
    """Read one phase's store, addressed through the catalogue, into families."""
    machine_map = next(item for item in catalog.maps if item.name == phase)
    phase_dir = catalog.description_store_root_path() / machine_map.name
    dd_version = catalog.dd_version

    pf_active = _read_ids(phase_dir, "pf_active", dd_version)
    pf_passive = _read_ids(phase_dir, "pf_passive", dd_version)
    wall = _read_ids(phase_dir, "wall", dd_version)
    magnetics = _read_ids(phase_dir, "magnetics", dd_version)

    coil_rects = tuple(
        rect for coil in pf_active.coil for rect in _iter_rects(coil.element)
    )
    vessel_rects = tuple(
        rect for loop in pf_passive.loop for rect in _iter_rects(loop.element)
    )
    wall_r, wall_z = _wall_outline(wall)
    flux = magnetics.flux_loop
    probes = magnetics.b_field_pol_probe
    return PhaseGeometry(
        phase=phase,
        coil_rects=coil_rects,
        vessel_rects=vessel_rects,
        wall_r=wall_r,
        wall_z=wall_z,
        flux_r=tuple(float(loop.position[0].r) for loop in flux),
        flux_z=tuple(float(loop.position[0].z) for loop in flux),
        probe_r=tuple(float(probe.position.r) for probe in probes),
        probe_z=tuple(float(probe.position.z) for probe in probes),
        probe_angle=tuple(float(probe.poloidal_angle) for probe in probes),
    )


def _draw_family_rects(ax, rects: tuple[Rectangle2D, ...], colour: str) -> None:
    for rect in rects:
        ax.add_patch(
            Rectangle(
                (rect.r - rect.width / 2.0, rect.z - rect.height / 2.0),
                rect.width,
                rect.height,
                facecolor=colour,
                edgecolor="none",
            )
        )


def _median_anchor(rs: tuple[float, ...], zs: tuple[float, ...]) -> tuple[float, float]:
    ordered = sorted(zip(zs, rs, strict=True))
    return ordered[len(ordered) // 2][1], ordered[len(ordered) // 2][0]


def cross_section(ax, geometry: PhaseGeometry, colours: dict[str, str]) -> None:
    """Draw one phase's families on ``ax`` using the named colour roles."""
    vessel_role = "vessel_op1" if geometry.phase == "OP1" else "vessel_op2"
    _draw_family_rects(ax, geometry.vessel_rects, colours[vessel_role])
    _draw_family_rects(ax, geometry.coil_rects, colours["pf_coils"])
    ax.plot(
        geometry.wall_r,
        geometry.wall_z,
        color=colours["wall"],
        linewidth=SERIES_LINEWIDTH,
    )
    ax.plot(
        geometry.flux_r,
        geometry.flux_z,
        linestyle="none",
        marker="o",
        markersize=7,
        markerfacecolor="none",
        markeredgecolor=colours["flux_loops"],
        markeredgewidth=2.0,
        color=colours["flux_loops"],
    )
    ax.plot(
        geometry.probe_r,
        geometry.probe_z,
        linestyle="none",
        marker="s",
        markersize=7,
        markerfacecolor=colours["pickup_probes"],
        markeredgecolor="none",
        color=colours["pickup_probes"],
    )

    coil_r, coil_z = _median_anchor(
        tuple(rect.r for rect in geometry.coil_rects),
        tuple(rect.z for rect in geometry.coil_rects),
    )
    direct_label(ax, coil_r, coil_z, "PF coils", colours["pf_coils"], dx=0.12, dy=0.35)
    if geometry.wall_r:
        top = max(range(len(geometry.wall_z)), key=geometry.wall_z.__getitem__)
        direct_label(
            ax,
            geometry.wall_r[top],
            geometry.wall_z[top],
            "wall",
            colours["wall"],
            dx=0.05,
            dy=0.35,
        )
    if geometry.flux_r:
        direct_label(
            ax,
            geometry.flux_r[0],
            geometry.flux_z[0],
            "flux loops",
            colours["flux_loops"],
            dx=0.08,
            dy=-0.35,
        )
    if geometry.probe_r:
        direct_label(
            ax,
            geometry.probe_r[0],
            geometry.probe_z[0],
            "probes",
            colours["pickup_probes"],
            dx=0.08,
            dy=0.35,
        )
    ven = _median_anchor(
        tuple(rect.r for rect in geometry.vessel_rects),
        tuple(rect.z for rect in geometry.vessel_rects),
    )
    direct_label(
        ax,
        ven[0],
        ven[1],
        f"{geometry.phase} vessel",
        colours[vessel_role],
        dx=0.10,
        dy=-0.5,
    )

    ax.set_aspect("equal", adjustable="box")
    rs = list(geometry.wall_r) + [rect.r for rect in geometry.coil_rects]
    zs = list(geometry.wall_z) + [rect.z for rect in geometry.coil_rects]
    if rs and zs:
        pad = 1.2
        ax.set_xlim(min(rs) - pad, max(rs) + pad)
        ax.set_ylim(min(zs) - pad, max(zs) + pad)
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.set_title(geometry.phase, color="#0b0b0b")


def render_candidate(
    name: str, overrides: dict[str, str], geometries, out_dir: Path
) -> tuple[Path, Path]:
    """Render one candidate palette on the OP1/OP2 cross-section pair."""
    colours = palette(overrides)
    fig, axes = plt.subplots(1, len(geometries), figsize=(14.0, 8.0))
    for ax, geometry in zip(list(axes), geometries, strict=True):
        cross_section(ax, geometry, colours)
    fig.tight_layout()
    png = out_dir / f"cross_section_{name}.png"
    svg = out_dir / f"cross_section_{name}.svg"
    fig.savefig(png)
    fig.savefig(svg)
    plt.close(fig)
    return png, svg


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--palette-candidates",
        action="store_true",
        help="render every candidate palette onto the OP1/OP2 cross-section",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("docs/figures/jt60sa-palette"),
        help="directory the figures are written to",
    )
    args = parser.parse_args(argv)

    apply_data_ink()
    catalog = load_packaged_machine_map("jt-60sa")
    geometries = tuple(read_phase(catalog, phase) for phase in PHASE_ORDER)

    if args.palette_candidates:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        for name, overrides in CANDIDATES.items():
            png, svg = render_candidate(name, overrides, geometries, args.out_dir)
            print(f"{name}: {png} {svg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
