"""Independently validate the MAST machine-geometry table.

``imas_ambix.data.description_reader.read_geometry_table`` emits the declared
machine description and adapts its limiter contour, PF-coil filaments,
B-probes, and flux loops for the GS-grounded latent engine's Green's-function
observation operator.  This script cross-checks that adapted table against the
underlying level-2 structures and independent geometric invariants:

* the level-2 Zarr mirror's ``wall`` / ``pf_active`` / ``pf_passive`` /
  ``magnetics`` groups, which carry independently-curated MAST geometry
  (named PF coils, named passive structures, named sensor families) rather
  than the raw efm setup arrays;
* the level-2 ``equilibrium`` group's reconstructed last-closed-flux-surface
  (LCFS) for a flat-top time slice, which must lie entirely inside the
  limiter if the limiter is the genuine plasma-facing boundary;
* elementary polygon geometry (closure, self-intersection, signed area).

Run with ``uv run python scripts/validate_mast_geometry.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import zarr

from imas_ambix.data.description_reader import read_geometry_table
from imas_ambix.data.paths import local_shot_path

if TYPE_CHECKING:
    from imas_ambix.gs.geometry import GeometryTable

SHOT_ID = 18502

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = (
    REPO_ROOT / "imas_ambix" / "latent" / "artifacts" / "geometry_validation.json"
)
OUT_FIG = (
    REPO_ROOT
    / "docs"
    / "figures"
    / "gs-grounded-latent-engine"
    / "fig-geometry-validation.png"
)

# Named PF-active coil groups in the level-2 ``pf_active`` Zarr group — an
# independently-curated reference for the efm ``fcoil`` circuits.
PF_ACTIVE_NAMES = (
    "p2_inner_lower",
    "p2_inner_upper",
    "p2_outer_lower",
    "p2_outer_upper",
    "p3_lower",
    "p3_upper",
    "p4_lower",
    "p4_upper",
    "p5_lower",
    "p5_upper",
    "p6_lower",
    "p6_upper",
    "sol",
)

# Named passive-structure groups in the level-2 ``pf_passive`` Zarr group.
PF_PASSIVE_NAMES = (
    "botcol",
    "coil_cases",
    "endcrown_l",
    "endcrown_u",
    "incon",
    "lhorw",
    "mid",
    "p2larm",
    "p2ldivpl",
    "p2uarm",
    "p2udivpl",
    "ring",
    "rodgr",
    "topcol",
    "uhorw",
    "vertw",
)

# efm ``magpr`` (B-probe) entries are restricted to these amb-mapped
# families (see geometry.py ``_VERTICAL_PREFIXES`` / ``_RADIAL_PREFIXES``);
# the level-2 ``magnetics`` group additionally carries ``cc`` and ``omv``
# families that efm's static setup does not include.
B_PROBE_FAMILIES = ("obr", "obv", "ccbv")

# Sanity envelope: no sensor / filament / limiter point should fall outside
# this box for a machine with vessel outer wall R~2.0 m, divertor Z~-1.83..1.83.
R_BOUNDS = (0.05, 2.2)
Z_BOUNDS = (-2.6, 2.6)


def open_l2_group(shot_id: int, group: str) -> Any:
    root = local_shot_path(shot_id, tier="level2")
    store: Any = zarr.open(str(root), mode="r")
    return store[group]


# --- Polygon geometry ---------------------------------------------------


def shoelace_area(r: np.ndarray, z: np.ndarray) -> float:
    """Signed polygon area (shoelace formula); positive = counter-clockwise."""
    return 0.5 * float(np.sum(r * np.roll(z, -1) - np.roll(r, -1) * z))


def _segments_intersect(
    p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray
) -> bool:
    """Proper intersection test for segments p1-p2 and p3-p4 (excludes shared endpoints)."""

    def cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))

    d1 = cross(p3, p4, p1)
    d2 = cross(p3, p4, p2)
    d3 = cross(p1, p2, p3)
    d4 = cross(p1, p2, p4)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))


def is_simple_polygon(r: np.ndarray, z: np.ndarray) -> bool:
    """True iff no two non-adjacent edges of the closed polygon cross."""
    n = len(r)
    pts = np.stack([r, z], axis=1)
    for i in range(n - 1):
        for j in range(i + 1, n - 1):
            if j in (i, i + 1) or (i == 0 and j == n - 2):
                continue
            if _segments_intersect(pts[i], pts[i + 1], pts[j], pts[j + 1]):
                return False
    return True


def point_in_polygon(
    px: np.ndarray, py: np.ndarray, poly_r: np.ndarray, poly_z: np.ndarray
) -> np.ndarray:
    """Ray-casting point-in-polygon test, vectorised over query points."""
    n = len(poly_r)
    inside = np.zeros(px.shape, dtype=bool)
    xj, yj = poly_r[-1], poly_z[-1]
    for i in range(n):
        xi, yi = poly_r[i], poly_z[i]
        intersects = ((yi > py) != (yj > py)) & (
            px < (xj - xi) * (py - yi) / (yj - yi + 1e-300) + xi
        )
        inside ^= intersects
        xj, yj = xi, yi
    return inside


def validate_limiter(table: GeometryTable, l2_wall: Any) -> dict[str, Any]:
    lr = np.array(table.limiter_r)
    lz = np.array(table.limiter_z)
    closed = bool(np.isclose(lr[0], lr[-1]) and np.isclose(lz[0], lz[-1]))
    # drop the duplicated closing point before area/simplicity checks
    lr_open, lz_open = (lr[:-1], lz[:-1]) if closed else (lr, lz)
    area = shoelace_area(lr_open, lz_open)
    simple = (
        is_simple_polygon(lr, lz)
        if closed
        else is_simple_polygon(np.append(lr, lr[0]), np.append(lz, lz[0]))
    )

    wr = np.asarray(l2_wall["limiter_r"][:])
    wz = np.asarray(l2_wall["limiter_z"][:])
    same_length = wr.size == lr.size
    max_diff_r = float(np.max(np.abs(wr - lr))) if same_length else float("nan")
    max_diff_z = float(np.max(np.abs(wz - lz))) if same_length else float("nan")
    identical_to_l2_wall = same_length and max_diff_r < 1e-6 and max_diff_z < 1e-6

    verdict = "PASS" if closed and simple and identical_to_l2_wall else "FAIL"
    return {
        "n_points": int(lr.size),
        "closed": closed,
        "simple_polygon": simple,
        "signed_area_m2": area,
        "orientation": "counter-clockwise" if area > 0 else "clockwise",
        "identical_to_l2_wall_group": identical_to_l2_wall,
        "l2_wall_max_diff_r_m": max_diff_r,
        "l2_wall_max_diff_z_m": max_diff_z,
        "l2_wall_source_description": dict(l2_wall["limiter_r"].attrs).get(
            "source", ""
        ),
        "verdict": verdict,
    }


# --- PF coils -------------------------------------------------------------


def reference_pf_groups(shot_id: int) -> dict[str, tuple[float, float, int]]:
    """Return {name: (mean_r, mean_z, n_points)} for every named L2 coil/structure group."""
    pf_active = open_l2_group(shot_id, "pf_active")
    pf_passive = open_l2_group(shot_id, "pf_passive")
    refs: dict[str, tuple[float, float, int]] = {}
    for name in PF_ACTIVE_NAMES:
        r = np.asarray(pf_active[f"{name}_r"][:]).ravel()
        z = np.asarray(pf_active[f"{name}_z"][:]).ravel()
        refs[f"active/{name}"] = (float(r.mean()), float(z.mean()), int(r.size))
    for name in PF_PASSIVE_NAMES:
        r = np.asarray(pf_passive[f"{name}_r"][:]).ravel()
        z = np.asarray(pf_passive[f"{name}_z"][:]).ravel()
        refs[f"passive/{name}"] = (float(r.mean()), float(z.mean()), int(r.size))
    return refs


def validate_pf_coils(
    table: GeometryTable,
    refs: dict[str, tuple[float, float, int]],
    match_tol_m: float = 0.02,
) -> dict[str, Any]:
    circuits = sorted({f.circuit for f in table.pf_filaments})
    ref_names = list(refs.keys())
    ref_rz = np.array([[refs[n][0], refs[n][1]] for n in ref_names])

    entries = []
    n_matched = 0
    for c in circuits:
        rs = np.array([f.r for f in table.pf_filaments if f.circuit == c])
        zs = np.array([f.z for f in table.pf_filaments if f.circuit == c])
        cr, cz = float(rs.mean()), float(zs.mean())
        d = np.hypot(ref_rz[:, 0] - cr, ref_rz[:, 1] - cz)
        j = int(np.argmin(d))
        residual = float(d[j])
        matched = residual < match_tol_m
        n_matched += matched
        entries.append(
            {
                "circuit": int(c),
                "n_filament": int(rs.size),
                "centroid_r": cr,
                "centroid_z": cz,
                "matched_name": ref_names[j] if matched else None,
                "nearest_name": ref_names[j],
                "residual_m": residual,
            }
        )
    return {
        "n_circuits": len(circuits),
        "n_reference_groups": len(ref_names),
        "n_matched_within_tol": n_matched,
        "match_tol_m": match_tol_m,
        "circuits": entries,
    }


# --- Sensors ---------------------------------------------------------------


def validate_sensors(
    table: GeometryTable, shot_id: int, limiter_r: np.ndarray, limiter_z: np.ndarray
) -> dict[str, Any]:
    mag = open_l2_group(shot_id, "magnetics")

    ref_r_parts, ref_z_parts = [], []
    for fam in B_PROBE_FAMILIES:
        ref_r_parts.append(np.asarray(mag[f"b_field_pol_probe_{fam}_r"][:]).ravel())
        ref_z_parts.append(np.asarray(mag[f"b_field_pol_probe_{fam}_z"][:]).ravel())
    ref_bp_r = np.concatenate(ref_r_parts)
    ref_bp_z = np.concatenate(ref_z_parts)

    br = np.array([p.r for p in table.b_probes])
    bz = np.array([p.z for p in table.b_probes])
    bp_resid = np.array(
        [
            np.hypot(ref_bp_r - r, ref_bp_z - z).min()
            for r, z in zip(br, bz, strict=True)
        ]
    )

    ref_fl_r = np.asarray(mag["flux_loop_r"][:]).ravel()
    ref_fl_z = np.asarray(mag["flux_loop_z"][:]).ravel()
    fr = np.array([p.r for p in table.flux_loops])
    fz = np.array([p.z for p in table.flux_loops])
    fl_resid = np.array(
        [
            np.hypot(ref_fl_r - r, ref_fl_z - z).min()
            for r, z in zip(fr, fz, strict=True)
        ]
    )

    n_bp_outside_box = int(
        np.sum(
            (br < R_BOUNDS[0])
            | (br > R_BOUNDS[1])
            | (bz < Z_BOUNDS[0])
            | (bz > Z_BOUNDS[1])
        )
    )
    n_fl_outside_box = int(
        np.sum(
            (fr < R_BOUNDS[0])
            | (fr > R_BOUNDS[1])
            | (fz < Z_BOUNDS[0])
            | (fz > Z_BOUNDS[1])
        )
    )
    bp_inside_limiter = point_in_polygon(br, bz, limiter_r, limiter_z)
    fl_inside_limiter = point_in_polygon(fr, fz, limiter_r, limiter_z)

    return {
        "n_bprobe": int(br.size),
        "n_bprobe_reference": int(ref_bp_r.size),
        "n_bprobe_l2_families_used": list(B_PROBE_FAMILIES),
        "bprobe_l2_residual_max_m": float(bp_resid.max()),
        "bprobe_l2_residual_mean_m": float(bp_resid.mean()),
        "n_fluxloop": int(fr.size),
        "n_fluxloop_reference": int(ref_fl_r.size),
        "fluxloop_l2_residual_max_m": float(fl_resid.max()),
        "fluxloop_l2_n_within_1cm": int(np.sum(fl_resid < 1e-2)),
        "n_bprobe_outside_vessel_box": n_bp_outside_box,
        "n_fluxloop_outside_vessel_box": n_fl_outside_box,
        "n_bprobe_inside_limiter": int(bp_inside_limiter.sum()),
        "n_fluxloop_inside_limiter": int(fl_inside_limiter.sum()),
        "vessel_box_r_bounds": list(R_BOUNDS),
        "vessel_box_z_bounds": list(Z_BOUNDS),
    }


# --- LCFS-inside-limiter consistency check ---------------------------------


def flattop_lcfs(shot_id: int) -> tuple[float, np.ndarray, np.ndarray]:
    """Return (time, lcfs_r, lcfs_z) for a flat-top-Ip slice with a valid boundary.

    ``equilibrium/lcfs_r`` is shaped ``(n_boundary_max, n_time)``; the valid
    boundary points for a given time column are the finite entries (variable
    count per reconstruction). We pick the finite-boundary slice nearest the
    Ip flat-top (found from ``magnetics/ip``).
    """
    mag = open_l2_group(shot_id, "magnetics")
    ip = np.asarray(mag["ip"][:])
    tmag = np.asarray(mag["time"][:])
    absip = np.abs(ip)
    win = 50
    best_i, best_std = None, np.inf
    for i in range(win, len(ip) - win):
        seg = ip[i - win : i + win]
        if np.abs(seg).mean() < 0.5 * absip.max():
            continue
        std = float(np.std(seg))
        if std < best_std:
            best_std, best_i = std, i
    flat_t = float(tmag[best_i])

    eq = open_l2_group(shot_id, "equilibrium")
    teq = np.asarray(eq["time"][:])
    lcr_all = np.asarray(eq["lcfs_r"][:])
    lcz_all = np.asarray(eq["lcfs_z"][:])
    order = np.argsort(np.abs(teq - flat_t))
    for i in order:
        finite = np.isfinite(lcr_all[:, i]) & np.isfinite(lcz_all[:, i])
        if finite.sum() > 20:
            return float(teq[i]), lcr_all[finite, i], lcz_all[finite, i]
    raise RuntimeError("no equilibrium time slice with a valid LCFS found")


def validate_lcfs_inside_limiter(
    shot_id: int, limiter_r: np.ndarray, limiter_z: np.ndarray
) -> dict[str, Any]:
    t, lcr, lcz = flattop_lcfs(shot_id)
    inside = point_in_polygon(lcr, lcz, limiter_r, limiter_z)
    return {
        "flattop_time_s": t,
        "n_lcfs_points": int(lcr.size),
        "n_inside_limiter": int(inside.sum()),
        "inside_fraction": float(inside.mean()),
        "lcfs_r_range": [float(lcr.min()), float(lcr.max())],
        "lcfs_z_range": [float(lcz.min()), float(lcz.max())],
        "lcfs_r": lcr.tolist(),
        "lcfs_z": lcz.tolist(),
    }


# --- Figure -----------------------------------------------------------------


def make_figure(
    table: GeometryTable,
    pf_result: dict[str, Any],
    lcfs_r: np.ndarray,
    lcfs_z: np.ndarray,
    flattop_t: float,
) -> None:
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 9))

    lr = np.array(table.limiter_r)
    lz = np.array(table.limiter_z)
    ax.plot(lr, lz, "-", color="black", lw=2.2, zorder=5, label="limiter (efm, closed)")

    circuits = sorted({f.circuit for f in table.pf_filaments})
    cmap = plt.get_cmap("tab20")
    for k, c in enumerate(circuits):
        rs = np.array([f.r for f in table.pf_filaments if f.circuit == c])
        zs = np.array([f.z for f in table.pf_filaments if f.circuit == c])
        color = cmap(k % 20)
        ax.scatter(rs, zs, s=6, color=color, alpha=0.7, zorder=3)
    for entry in pf_result["circuits"]:
        if entry["matched_name"] is not None and entry["n_filament"] >= 4:
            ax.annotate(
                entry["matched_name"].split("/")[-1],
                (entry["centroid_r"], entry["centroid_z"]),
                fontsize=6,
                color="darkred",
                ha="center",
                va="center",
                zorder=6,
            )

    br = np.array([p.r for p in table.b_probes])
    bz = np.array([p.z for p in table.b_probes])
    bang = np.array([p.angle_deg for p in table.b_probes])
    vertical = np.isclose(bang, 90.0)
    ax.scatter(
        br[vertical],
        bz[vertical],
        marker="^",
        s=22,
        color="tab:blue",
        label="B-probe (vertical)",
        zorder=4,
    )
    ax.scatter(
        br[~vertical],
        bz[~vertical],
        marker=">",
        s=22,
        color="tab:cyan",
        label="B-probe (radial)",
        zorder=4,
    )

    fr = np.array([p.r for p in table.flux_loops])
    fz = np.array([p.z for p in table.flux_loops])
    ax.scatter(
        fr,
        fz,
        marker="o",
        s=16,
        facecolors="none",
        edgecolors="tab:orange",
        label="flux loop",
        zorder=4,
    )

    ax.plot(
        lcfs_r,
        lcfs_z,
        ".",
        color="tab:red",
        ms=3,
        label=f"LCFS shot {SHOT_ID} @ t={flattop_t:.3f}s (flat-top)",
        zorder=7,
    )

    ax.set_aspect("equal")
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.set_title(f"MAST geometry validation — shot {SHOT_ID}")
    ax.legend(loc="upper right", fontsize=7, framealpha=0.9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=160)
    plt.close(fig)


# --- Main --------------------------------------------------------------


def main() -> None:
    table = read_geometry_table(SHOT_ID)
    l2_wall = open_l2_group(SHOT_ID, "wall")

    limiter_result = validate_limiter(table, l2_wall)
    lr = np.array(table.limiter_r)
    lz = np.array(table.limiter_z)

    refs = reference_pf_groups(SHOT_ID)
    pf_result = validate_pf_coils(table, refs)

    sensor_result = validate_sensors(table, SHOT_ID, lr, lz)
    lcfs_result = validate_lcfs_inside_limiter(SHOT_ID, lr, lz)

    make_figure(
        table,
        pf_result,
        np.array(lcfs_result["lcfs_r"]),
        np.array(lcfs_result["lcfs_z"]),
        lcfs_result["flattop_time_s"],
    )

    overall_pass = (
        limiter_result["verdict"] == "PASS"
        and lcfs_result["inside_fraction"] > 0.98
        and pf_result["n_matched_within_tol"] >= 12  # 12 real PF-active coils + sol
        and sensor_result["n_bprobe_outside_vessel_box"] == 0
        and sensor_result["n_fluxloop_outside_vessel_box"] == 0
        and sensor_result["bprobe_l2_residual_max_m"] < 1e-6
    )

    report = {
        "schema": "mast-geometry-validation-v0",
        "shot_id": SHOT_ID,
        "limiter": limiter_result,
        "coils": pf_result,
        "sensors": sensor_result,
        "lcfs_check": {
            k: v for k, v in lcfs_result.items() if k not in ("lcfs_r", "lcfs_z")
        },
        "overall_verdict": "PASS" if overall_pass else "FAIL",
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2))

    print(json.dumps(report, indent=2))
    print(f"\nfigure written to {OUT_FIG}")
    print(f"report written to {OUT_JSON}")


if __name__ == "__main__":
    main()
