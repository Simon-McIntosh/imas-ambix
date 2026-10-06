"""Draw the JT-60SA OP1/OP2 cross-section from the converted description store.

One reader turns a phase's store into drawable families (PF coil elements,
vessel elements, the wall outline, flux loops and pickup probes); one
cross-section function draws them; and ``--palette-candidates`` renders the
candidate colour-role palettes onto the real OP1 and OP2 cross-section so the
choice can be made against the machine, not against a swatch.

The store is addressed through the catalogue that owns it: the phase directory
name comes from ``catalog.maps`` rather than a literal, and every IDS is read
with :class:`imas.DBEntry` so the figure shows what the converter wrote.  Each
coil and vessel element is drawn by its declared ``geometry_type`` rather than
assumed to be a rectangle, and every wall description unit is drawn as its own
polyline so no stroke joins two separate contours.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Annulus, Rectangle  # noqa: E402

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

PHASE_ORDER = ("OP1", "OP2")
IDS_NAMES = ("pf_active", "pf_passive", "wall", "magnetics")

# Declared pf element geometry_type values (Data Dictionary 4.1.1 ordering).
_GEOMETRY_TYPES = {
    1: "outline",
    2: "rectangle",
    3: "oblique",
    4: "arcs_of_circle",
    5: "annulus",
    6: "thick_line",
}

# Candidate colour-role sets.  Each names a colour for every drawn component.
# The two vessel roles are lightness steps of one hue (the OP1 and OP2 vessels
# are the same structure at the two phase geometries); the remaining components
# take distinct, non-red hues, so red stays reserved for a limit being broken.
# Every colour holds >= 3:1 contrast against the white surface.  The lead
# chooses one; nothing downstream assumes which.
CANDIDATES: dict[str, dict[str, str]] = {
    "A-blue-vessel": {
        "pf_coils": "#872b6d",
        "vessel_op1": "#588cfe",
        "vessel_op2": "#345dba",
        "wall": "#016900",
        "flux_loops": "#a08d00",
        "pickup_probes": "#1e9ea7",
    },
    "B-green-vessel": {
        "pf_coils": "#1657af",
        "vessel_op1": "#52a338",
        "vessel_op2": "#006d00",
        "wall": "#9c6fe6",
        "flux_loops": "#802777",
        "pickup_probes": "#009fb0",
    },
    "C-gold-vessel": {
        "pf_coils": "#418af4",
        "vessel_op1": "#a89207",
        "vessel_op2": "#7e5200",
        "wall": "#2d854b",
        "flux_loops": "#b16698",
        "pickup_probes": "#7033a4",
    },
}


@dataclass(frozen=True)
class Shape:
    """One drawable element, tagged by its declared geometry type."""

    kind: str
    r: tuple[float, ...]
    z: tuple[float, ...]
    width: float = 0.0
    height: float = 0.0
    thickness: float = 0.0
    alpha: float = 0.0
    length: float = 0.0
    radius_inner: float = 0.0
    radius_outer: float = 0.0

    def points(self) -> tuple[list[float], list[float]]:
        """Every vertex of this shape, for the axis window."""
        return list(self.r), list(self.z)


@dataclass(frozen=True)
class PhaseGeometry:
    """The drawable families a single phase's store yields."""

    phase: str
    coil_shapes: tuple[Shape, ...]
    vessel_shapes: tuple[Shape, ...]
    wall_polylines: tuple[tuple[tuple[float, ...], tuple[float, ...]], ...]
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


def _shape_from_geometry(geometry, context: str) -> Shape:
    """Build one :class:`Shape` from an element's declared geometry."""
    geometry_type = int(geometry.geometry_type)
    kind = _GEOMETRY_TYPES.get(geometry_type)
    if kind is None:
        raise ValueError(f"{context}: unknown geometry_type {geometry_type}")
    if kind == "rectangle":
        rect = geometry.rectangle
        return Shape(
            "rectangle",
            (float(rect.r),),
            (float(rect.z),),
            width=float(rect.width),
            height=float(rect.height),
        )
    if kind == "outline":
        outline = geometry.outline
        return Shape(
            "outline",
            tuple(float(v) for v in outline.r),
            tuple(float(v) for v in outline.z),
        )
    if kind == "oblique":
        oblique = geometry.oblique
        return Shape(
            "oblique",
            (float(oblique.r),),
            (float(oblique.z),),
            thickness=float(oblique.thickness),
            alpha=float(oblique.alpha),
            length=float(oblique.length_alpha),
        )
    if kind == "arcs_of_circle":
        arcs = geometry.arcs_of_circle
        return Shape(
            "arcs_of_circle",
            tuple(float(v) for v in arcs.r),
            tuple(float(v) for v in arcs.z),
        )
    if kind == "annulus":
        annulus = geometry.annulus
        return Shape(
            "annulus",
            (float(annulus.r),),
            (float(annulus.z),),
            radius_inner=float(annulus.radius_inner),
            radius_outer=float(annulus.radius_outer),
        )
    thick = geometry.thick_line
    return Shape(
        "thick_line",
        (float(thick.first_point.r), float(thick.second_point.r)),
        (float(thick.first_point.z), float(thick.second_point.z)),
        thickness=float(thick.thickness),
    )


def _shapes_from_elements(elements, context: str) -> tuple[Shape, ...]:
    shapes: list[Shape] = []
    for index, element in enumerate(elements):
        shapes.append(_shape_from_geometry(element.geometry, f"{context}[{index}]"))
    return tuple(shapes)


def _wall_polylines(wall) -> tuple[tuple[tuple[float, ...], tuple[float, ...]], ...]:
    """Return each wall description unit as its own polyline.

    The lines are kept per unit so no stroke is drawn between two separate
    contours; joining them would invent geometry the store does not carry.
    """
    polylines: list[tuple[tuple[float, ...], tuple[float, ...]]] = []
    for description in wall.description_2d:
        for unit in list(description.limiter.unit) + list(description.vessel.unit):
            outline = unit.outline
            rs = tuple(float(v) for v in outline.r)
            zs = tuple(float(v) for v in outline.z)
            if len(rs) >= 2:
                polylines.append((rs, zs))
    return tuple(polylines)


def read_phase(catalog: MachineMapCatalog, phase: str) -> PhaseGeometry:
    """Read one phase's store, addressed through the catalogue, into families."""
    machine_map = next(item for item in catalog.maps if item.name == phase)
    phase_dir = catalog.description_store_root_path() / machine_map.name
    dd_version = catalog.dd_version

    pf_active = _read_ids(phase_dir, "pf_active", dd_version)
    pf_passive = _read_ids(phase_dir, "pf_passive", dd_version)
    wall = _read_ids(phase_dir, "wall", dd_version)
    magnetics = _read_ids(phase_dir, "magnetics", dd_version)

    coil_shapes = tuple(
        shape
        for coil_index, coil in enumerate(pf_active.coil)
        for shape in _shapes_from_elements(coil.element, f"coil[{coil_index}]")
    )
    vessel_shapes = tuple(
        shape
        for loop_index, loop in enumerate(pf_passive.loop)
        for shape in _shapes_from_elements(loop.element, f"loop[{loop_index}]")
    )
    flux = magnetics.flux_loop
    probes = magnetics.b_field_pol_probe
    return PhaseGeometry(
        phase=phase,
        coil_shapes=coil_shapes,
        vessel_shapes=vessel_shapes,
        wall_polylines=_wall_polylines(wall),
        flux_r=tuple(float(loop.position[0].r) for loop in flux),
        flux_z=tuple(float(loop.position[0].z) for loop in flux),
        probe_r=tuple(float(probe.position.r) for probe in probes),
        probe_z=tuple(float(probe.position.z) for probe in probes),
        probe_angle=tuple(float(probe.poloidal_angle) for probe in probes),
    )


def _draw_shape(ax, shape: Shape, colour: str) -> None:
    if shape.kind == "rectangle":
        ax.add_patch(
            Rectangle(
                (shape.r[0] - shape.width / 2.0, shape.z[0] - shape.height / 2.0),
                shape.width,
                shape.height,
                facecolor=colour,
                edgecolor="none",
            )
        )
    elif shape.kind in ("outline", "arcs_of_circle"):
        ax.plot(shape.r, shape.z, color=colour, linewidth=1.6)
    elif shape.kind == "annulus":
        ax.add_patch(
            Annulus(
                (shape.r[0], shape.z[0]),
                shape.radius_outer,
                shape.radius_outer - shape.radius_inner,
                facecolor=colour,
                edgecolor="none",
            )
        )
    elif shape.kind == "oblique":
        end_r = shape.r[0] + shape.length * np.cos(shape.alpha)
        end_z = shape.z[0] + shape.length * np.sin(shape.alpha)
        ax.plot([shape.r[0], end_r], [shape.z[0], end_z], color=colour, linewidth=2.0)
    else:  # thick_line
        ax.plot(shape.r, shape.z, color=colour, linewidth=2.0)


def _family_centroid(shapes: tuple[Shape, ...]) -> tuple[float, float]:
    rs = [value for shape in shapes for value in shape.r]
    zs = [value for shape in shapes for value in shape.z]
    return float(np.median(rs)), float(np.median(zs))


def _gather_extent(geometry: PhaseGeometry) -> tuple[float, float, float, float]:
    rs: list[float] = []
    zs: list[float] = []
    for shape in geometry.coil_shapes + geometry.vessel_shapes:
        rs.extend(shape.points()[0])
        zs.extend(shape.points()[1])
    for poly_r, poly_z in geometry.wall_polylines:
        rs.extend(poly_r)
        zs.extend(poly_z)
    rs.extend(geometry.flux_r)
    zs.extend(geometry.flux_z)
    rs.extend(geometry.probe_r)
    zs.extend(geometry.probe_z)
    return min(rs), max(rs), min(zs), max(zs)


def cross_section(ax, geometry: PhaseGeometry, colours: dict[str, str]) -> None:
    """Draw one phase's families on ``ax`` using the named colour roles."""
    vessel_role = "vessel_op1" if geometry.phase == "OP1" else "vessel_op2"

    for shape in geometry.vessel_shapes:
        _draw_shape(ax, shape, colours[vessel_role])
    for shape in geometry.coil_shapes:
        _draw_shape(ax, shape, colours["pf_coils"])
    for poly_r, poly_z in geometry.wall_polylines:
        ax.plot(poly_r, poly_z, color=colours["wall"], linewidth=SERIES_LINEWIDTH)

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

    # Axis window contains every drawn element, padded so nothing is clipped.
    # The horizontal padding is generous: it leaves empty side bands for the
    # family labels, so no label lands on the machine.
    rmin, rmax, zmin, zmax = _gather_extent(geometry)
    pad_r = 0.45 * (rmax - rmin)
    pad_z = 0.30 * (zmax - zmin)
    ax.set_xlim(rmin - pad_r, rmax + pad_r)
    ax.set_ylim(zmin - pad_z, zmax + pad_z)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.set_title(geometry.phase, color="#0b0b0b")

    # Direct labels sit in the empty margin beside each family, each tied to its
    # family by a short leader in the series colour, so no label lands on data,
    # on another label, or outside the axes.  Anchors are in axes fractions,
    # placed in the padding bands the window above leaves empty.
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    xw = x1 - x0
    yh = y1 - y0

    def anchor(fx: float, fy: float) -> tuple[float, float]:
        return x0 + fx * xw, y0 + fy * yh

    coil_r, coil_z = _family_centroid(geometry.coil_shapes)
    _label_family(
        ax, coil_r, coil_z, anchor(0.20, 0.7), "PF coils", colours["pf_coils"], "right"
    )
    if geometry.vessel_shapes:
        vessel_r, vessel_z = _family_centroid(geometry.vessel_shapes)
        _label_family(
            ax,
            vessel_r,
            vessel_z,
            anchor(0.5, 0.02),
            f"{geometry.phase} vessel",
            colours[vessel_role],
            "center",
        )
    if geometry.wall_polylines:
        wall_pts = [
            (r, z)
            for poly_r, poly_z in geometry.wall_polylines
            for r, z in zip(poly_r, poly_z, strict=True)
        ]
        wall_r, wall_z = min(wall_pts, key=lambda point: point[0])
        _label_family(
            ax, wall_r, wall_z, anchor(0.20, 0.3), "wall", colours["wall"], "right"
        )
    if geometry.flux_r:
        flux_index = int(np.argmax(geometry.flux_r))
        _label_family(
            ax,
            geometry.flux_r[flux_index],
            geometry.flux_z[flux_index],
            anchor(0.80, 0.7),
            "flux loops",
            colours["flux_loops"],
            "left",
        )
    if geometry.probe_r:
        probe_index = int(np.argmax(geometry.probe_r))
        _label_family(
            ax,
            geometry.probe_r[probe_index],
            geometry.probe_z[probe_index],
            anchor(0.80, 0.3),
            "probes",
            colours["pickup_probes"],
            "left",
        )


def _label_family(ax, x, y, anchor, text, colour, ha) -> None:
    """Label a family from an empty-margin anchor with a thin leader line."""
    ax.annotate(
        "",
        xy=(x, y),
        xytext=anchor,
        arrowprops={
            "arrowstyle": "-",
            "color": colour,
            "linewidth": 1.4,
            "shrinkA": 2.0,
            "shrinkB": 2.0,
        },
        annotation_clip=False,
    )
    direct_label(ax, anchor[0], anchor[1], text, colour, ha=ha, va="center")


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
