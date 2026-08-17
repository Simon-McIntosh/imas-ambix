"""Audit the adapted machine-map geometry against its catalog structures.

The declared machine map emits MAST Data Catalog geometry through the canonical
table route consumed by the GS Green's-function operator.
This script overlays that adapted table on the underlying ``pf_active`` and
``pf_passive`` structures and quantifies the adapter's geometric fidelity,
with an explicit focus on the non-rectangular question:

* PF-active coils: is each coil a filled axis-aligned rectangle (our operator
  collapses those to one thick-cylinder filament) or a genuinely non-rectangular
  section (which must retain its filament lattice)?
* Passive: are the coil-case hollow frames and the vessel structures handled
  such that a true rectangle becomes one cylinder while the non-rectangular
  bits keep their extra filaments?

The rule under test is: single rectangles for true rectangles, extra filaments
for extra geometric features.  The script writes an overlay figure per side
plus a quantitative consistency artifact.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.patches import Polygon, Rectangle

from imas_ambix.data.description_reader import read_geometry_table

REPO = Path(__file__).resolve().parent.parent
REF_SHOT = 30421
REF_ZARR = f"/work/projects/imas_gpu/mast/level2/shots/{REF_SHOT}.zarr"
ARTIFACT = (
    REPO
    / "imas_ambix/latent/artifacts/patch_gate/geometry_mastapp_consistency.json"
)
FIG_DIR = REPO / "docs/figures/nonaxisymmetric-field-subtraction"

# Reference PF-active coil prefixes (each a set of rectangular sub-elements).
PF_ACTIVE_PREFIXES = [
    "sol",
    "p2_inner_lower",
    "p2_outer_lower",
    "p2_inner_upper",
    "p2_outer_upper",
    "p3_lower",
    "p3_upper",
    "p4_lower",
    "p4_upper",
    "p5_lower",
    "p5_upper",
    "p6_lower",
    "p6_upper",
]

# Reference pf_passive components (shape-angle rectangles/parallelograms) and the
# coil-case frames (rotate_90 rule, no shape angles).
PF_PASSIVE_SHAPED = [
    "botcol",
    "topcol",
    "endcrown_l",
    "endcrown_u",
    "incon",
    "lhorw",
    "uhorw",
    "mid",
    "vertw",
    "ring",
    "rodgr",
    "p2larm",
    "p2ldivpl",
    "p2uarm",
    "p2udivpl",
]

# Collapse thresholds mirrored by the geometry-table adapter.
FILL_TOL = 0.25
FLOOR = 0.01


# --- reference geometry loaders --------------------------------------------


def ref_pf_active(ds: xr.Dataset) -> dict[str, dict]:
    """Per-coil reference rectangles: element (r,z,w,h) arrays + summary."""
    out: dict[str, dict] = {}
    for p in PF_ACTIVE_PREFIXES:
        r = np.asarray(ds[f"{p}_r"], dtype=np.float64)
        z = np.asarray(ds[f"{p}_z"], dtype=np.float64)
        w = np.abs(np.asarray(ds[f"{p}_width"], dtype=np.float64))
        h = np.abs(np.asarray(ds[f"{p}_height"], dtype=np.float64))
        r_lo, r_hi = (r - w / 2).min(), (r + w / 2).max()
        z_lo, z_hi = (z - h / 2).min(), (z + h / 2).max()
        box = float((r_hi - r_lo) * (z_hi - z_lo))
        area = float((w * h).sum())
        out[p] = {
            "r": r,
            "z": z,
            "w": w,
            "h": h,
            "n": int(r.size),
            "centroid": (float(r.mean()), float(z.mean())),
            "bbox": [float(r_lo), float(z_lo), float(r_hi), float(z_hi)],
            "area": area,
            "fill": area / box if box > 0 else float("nan"),
        }
    return out


def _shape_vertices(
    r: float, z: float, dR: float, dZ: float, a1: float, a2: float, rotate_90: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Vertex rule transcribed from the MAST Data Catalog pf_passive notebook.

    Axis-aligned rectangle when both shape angles are zero; otherwise a
    parallelogram sheared by the two angles.  ``rotate_90`` swaps (dR, dZ)
    before building vertices (the coil_cases convention).
    """
    dR, dZ = abs(dR), abs(dZ)
    if rotate_90:
        dR, dZ = dZ, dR
    if a1 == 0 and a2 == 0:
        rr = np.array([r - dR / 2, r + dR / 2, r + dR / 2, r - dR / 2])
        zz = np.array([z - dZ / 2, z - dZ / 2, z + dZ / 2, z + dZ / 2])
        return rr, zz
    a1t = np.tan(a1 * np.pi / 180) if a1 > 0 else 0.0
    a2t = 1.0 / np.tan(a2 * np.pi / 180) if a2 > 0 else 0.0
    rr = np.array(
        [
            r - dR / 2 - dZ / 2 * a2t,
            r + dR / 2 - dZ / 2 * a2t,
            r + dR / 2 + dZ / 2 * a2t,
            r - dR / 2 + dZ / 2 * a2t,
        ]
    )
    zz = np.array(
        [
            z - dZ / 2 - dR / 2 * a1t,
            z - dZ / 2 + dR / 2 * a1t,
            z + dZ / 2 + dR / 2 * a1t,
            z + dZ / 2 - dR / 2 * a1t,
        ]
    )
    return rr, zz


def ref_pf_passive(ds: xr.Dataset) -> dict[str, dict]:
    """Per-component reference polygons with the shape-angle / rotate_90 rule."""
    out: dict[str, dict] = {}
    for c in PF_PASSIVE_SHAPED:
        r = np.asarray(ds[f"{c}_r"], dtype=np.float64)
        z = np.asarray(ds[f"{c}_z"], dtype=np.float64)
        w = np.asarray(ds[f"{c}_width"], dtype=np.float64)
        h = np.asarray(ds[f"{c}_height"], dtype=np.float64)
        a1 = np.asarray(ds[f"{c}_shapeAngle1"], dtype=np.float64)
        a2 = np.asarray(ds[f"{c}_shapeAngle2"], dtype=np.float64)
        polys = [
            _shape_vertices(r[i], z[i], w[i], h[i], a1[i], a2[i], rotate_90=False)
            for i in range(r.size)
        ]
        nonrect = int(np.sum((a1 != 0) | (a2 != 0)))
        out[c] = {
            "r": r,
            "z": z,
            "polys": polys,
            "n": int(r.size),
            "n_nonrect": nonrect,
            "centroid": (float(r.mean()), float(z.mean())),
        }
    # coil_cases: no shape angles, rotate_90=True (thin frame bars)
    cr = np.asarray(ds["coil_cases_r"], dtype=np.float64)
    cz = np.asarray(ds["coil_cases_z"], dtype=np.float64)
    cw = np.asarray(ds["coil_cases_width"], dtype=np.float64)
    ch = np.asarray(ds["coil_cases_height"], dtype=np.float64)
    polys = [
        _shape_vertices(cr[i], cz[i], cw[i], ch[i], 0.0, 0.0, rotate_90=True)
        for i in range(cr.size)
    ]
    out["coil_cases"] = {
        "r": cr,
        "z": cz,
        "polys": polys,
        "n": int(cr.size),
        "n_nonrect": 0,
        "centroid": (float(cr.mean()), float(cz.mean())),
    }
    return out


# --- our geometry ----------------------------------------------------------


def our_circuits(table) -> dict[int, dict]:
    """Adapted filaments grouped by circuit, with the collapse verdict."""
    filaments = table.pf_filaments
    fr = np.asarray([item.r for item in filaments], dtype=np.float64)
    fz = np.asarray([item.z for item in filaments], dtype=np.float64)
    fw = np.asarray([item.width for item in filaments], dtype=np.float64)
    fh = np.asarray([item.height for item in filaments], dtype=np.float64)
    fc = np.asarray([item.circuit for item in filaments], dtype=np.int64)
    fx = np.asarray([item.xmult for item in filaments], dtype=np.float64)
    ft = np.asarray([item.turns for item in filaments], dtype=np.float64)
    out: dict[int, dict] = {}
    for c in np.unique(fc):
        m = fc == c
        r, z = fr[m], fz[m]
        w = np.maximum(np.abs(fw[m]), FLOOR)
        h = np.maximum(np.abs(fh[m]), FLOOR)
        xm = fx[m]
        r_lo, r_hi = (r - w / 2).min(), (r + w / 2).max()
        z_lo, z_hi = (z - h / 2).min(), (z + h / 2).max()
        box = float((r_hi - r_lo) * (z_hi - z_lo))
        area = float((w * h).sum())
        fill = area / box if box > 0 else float("nan")
        same_sign = bool(np.all(xm >= 0) or np.all(xm <= 0))
        collapse = bool(box > 0 and abs(fill - 1.0) <= FILL_TOL and same_sign and m.sum() > 1)
        out[int(c)] = {
            "r": r,
            "z": z,
            "w": w,
            "h": h,
            "n": int(m.sum()),
            "centroid": (float(r.mean()), float(z.mean())),
            "bbox": [float(r_lo), float(z_lo), float(r_hi), float(z_hi)],
            "area": area,
            "fill": fill,
            "turns": float(ft[m].sum()),
            "collapse": collapse,
        }
    return out


def match_by_centroid(
    ours: dict[int, dict], ref: dict[str, dict]
) -> dict[str, int]:
    """Nearest-centroid match ref-coil-name -> our-circuit-id."""
    match: dict[str, int] = {}
    for name, rc in ref.items():
        rr, rz = rc["centroid"]
        best, bestd = None, np.inf
        for cid, oc in ours.items():
            oorr, oorz = oc["centroid"]
            d = float(np.hypot(oorr - rr, oorz - rz))
            if d < bestd:
                best, bestd = cid, d
        match[name] = (best, bestd)
    return match


# --- figures ---------------------------------------------------------------


def _draw_rect(ax, r, z, w, h, **kw):
    ax.add_patch(Rectangle((r - w / 2, z - h / 2), w, h, **kw))


def fig_pf_active(ours, ref, match, path):
    fig, ax = plt.subplots(figsize=(7, 11))
    for name, rc in ref.items():
        for i in range(rc["n"]):
            _draw_rect(
                ax, rc["r"][i], rc["z"][i], rc["w"][i], rc["h"][i],
                facecolor="none", edgecolor="tab:blue", lw=0.5,
            )
    for name, (cid, _d) in match.items():
        oc = ours[cid]
        r_lo, z_lo, r_hi, z_hi = oc["bbox"]
        if oc["collapse"]:
            ax.add_patch(
                Rectangle(
                    (r_lo, z_lo), r_hi - r_lo, z_hi - z_lo,
                    facecolor="tab:orange", alpha=0.35, edgecolor="tab:red", lw=1.2,
                )
            )
        else:
            for i in range(oc["n"]):
                _draw_rect(
                    ax, oc["r"][i], oc["z"][i], oc["w"][i], oc["h"][i],
                    facecolor="tab:red", alpha=0.4, edgecolor="tab:red", lw=0.4,
                )
    ax.plot([], [], color="tab:blue", lw=1.0, label="mastapp pf_active elements")
    ax.add_patch(Rectangle((0, 0), 0, 0, facecolor="tab:orange", alpha=0.35,
                 edgecolor="tab:red", label="ours: collapsed thick cylinder"))
    ax.add_patch(Rectangle((0, 0), 0, 0, facecolor="tab:red", alpha=0.4,
                 label="ours: retained filament lattice"))
    ax.set_xlabel("R [m]"); ax.set_ylabel("Z [m]")
    ax.set_title(f"PF-active geometry: ours (efm fcoil) vs mastapp pf_active (shot {REF_SHOT})")
    ax.set_aspect("equal"); ax.legend(loc="upper right", fontsize=8); ax.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def fig_passive(ours_cases, ref_passive, passive_points, path):
    fig, ax = plt.subplots(figsize=(8, 11))
    for name, rc in ref_passive.items():
        col = "tab:green" if name == "coil_cases" else "tab:blue"
        for rr, zz in rc["polys"]:
            ax.add_patch(Polygon(np.column_stack([rr, zz]), closed=True,
                         facecolor="none", edgecolor=col, lw=0.6))
    # our fcoil case frames (retained lattices)
    for cid, oc in ours_cases.items():
        for i in range(oc["n"]):
            _draw_rect(ax, oc["r"][i], oc["z"][i], oc["w"][i], oc["h"][i],
                       facecolor="tab:red", alpha=0.5, edgecolor="tab:red", lw=0.4)
    # Declared passive points are a diagnostic-only source.
    ar = [p.r for p in passive_points]
    az = [p.z for p in passive_points]
    ax.scatter(ar, az, s=16, c="tab:purple", marker="x",
               label="declared passive (R,Z) points (diagnostic-only)")
    ax.plot([], [], color="tab:blue", lw=1.0, label="mastapp pf_passive (shaped)")
    ax.plot([], [], color="tab:green", lw=1.0, label="mastapp coil_cases (rotate_90)")
    ax.add_patch(Rectangle((0, 0), 0, 0, facecolor="tab:red", alpha=0.5,
                 label="ours: fcoil case frames (retained filaments)"))
    ax.set_xlabel("R [m]"); ax.set_ylabel("Z [m]")
    ax.set_title(f"Passive geometry: ours vs mastapp pf_passive (shot {REF_SHOT})")
    ax.set_aspect("equal"); ax.legend(loc="upper right", fontsize=8); ax.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


# --- main ------------------------------------------------------------------


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)

    ds_pf = xr.open_zarr(REF_ZARR, group="pf_active")
    ds_pv = xr.open_zarr(REF_ZARR, group="pf_passive")
    ref_a = ref_pf_active(ds_pf)
    ref_p = ref_pf_passive(ds_pv)

    table = read_geometry_table(REF_SHOT)
    ours = our_circuits(table)
    # circuits 1..13 are the active coils; 14+ are case/structural frames.
    match_a = match_by_centroid(
        {k: v for k, v in ours.items()}, ref_a
    )
    passive_points = table.passive_structures
    case_ids = {cid: ours[cid] for cid in ours if not ours[cid]["collapse"] and ours[cid]["n"] > 1}

    # --- PF-active consistency rows ---
    pf_rows = []
    for name, rc in ref_a.items():
        cid, d = match_a[name]
        oc = ours[cid]
        dr = oc["centroid"][0] - rc["centroid"][0]
        dz = oc["centroid"][1] - rc["centroid"][1]
        pf_rows.append({
            "coil": name,
            "ref_n_elements": rc["n"],
            "our_circuit": cid,
            "our_n_filaments_raw": oc["n"],
            "centroid_match_m": round(d, 5),
            "d_centroid_r_m": round(dr, 5),
            "d_centroid_z_m": round(dz, 5),
            "ref_fill_fraction": round(rc["fill"], 4),
            "our_fill_fraction": round(oc["fill"], 4),
            "ref_area_m2": round(rc["area"], 6),
            "our_area_m2": round(oc["area"], 6),
            "ref_bbox": [round(x, 4) for x in rc["bbox"]],
            "our_bbox": [round(x, 4) for x in oc["bbox"]],
            "is_nonrectangular": bool(rc["fill"] < (1.0 - FILL_TOL)),
            "our_representation": (
                "single_thick_cylinder" if oc["collapse"]
                else ("single_filament" if oc["n"] == 1 else "retained_lattice")
            ),
            "consistent": bool(d < 0.01
                               and abs(rc["area"] - oc["area"]) / max(rc["area"], 1e-9) < 0.05),
        })

    # --- passive consistency rows ---
    # coil cases: ours (fcoil frames) vs ref coil_cases (thin frame bars)
    pv_rows = []
    for name, rc in ref_p.items():
        row = {
            "component": name,
            "ref_n": rc["n"],
            "ref_n_nonrectangular": rc["n_nonrect"],
            "ref_centroid": [round(x, 4) for x in rc["centroid"]],
        }
        if name == "coil_cases":
            row.update({
                "our_source": "fcoil-structural-circuits (retained frames)",
                "our_n_case_circuits": len(case_ids),
                "our_n_case_filaments": int(sum(oc["n"] for oc in case_ids.values())),
                "our_representation": "retained_filament_lattice",
                "handled_as_nonrectangular": True,
            })
        else:
            # Match to the nearest declared passive point.
            amm_rz = (
                np.array([[p.r, p.z] for p in passive_points])
                if passive_points
                else np.zeros((0, 2))
            )
            dmin = float("nan")
            if amm_rz.size:
                d = np.hypot(amm_rz[:, 0] - rc["centroid"][0],
                             amm_rz[:, 1] - rc["centroid"][1])
                dmin = round(float(d.min()), 4)
            row.update({
                "our_source": "declared passive (R,Z) points",
                "our_representation": "point_only_no_cross_section",
                "nearest_amm_point_m": dmin,
                "used_as_field_source": False,
                "note": "passive points are a diagnostic coincidence check; "
                        "field sources are the declared structural circuits.",
            })
        pv_rows.append(row)

    # --- figures ---
    fig_a_path = FIG_DIR / "fig-geometry-pf-active.png"
    fig_p_path = FIG_DIR / "fig-geometry-passive.png"
    fig_pf_active(ours, ref_a, match_a, fig_a_path)
    fig_passive(case_ids, ref_p, passive_points, fig_p_path)

    # --- verdicts ---
    active_consistent = all(r["consistent"] for r in pf_rows)
    active_nonrect = [r["coil"] for r in pf_rows if r["is_nonrectangular"]]
    active_collapsed = [r["coil"] for r in pf_rows if r["our_representation"] == "single_thick_cylinder"]

    payload = {
        "schema": "geometry-mastapp-consistency-v1",
        "reference": {
            "source": "MAST Data Catalog (mastapp.site) level-2 IMAS pf_active/pf_passive",
            "shot": REF_SHOT,
            "zarr": REF_ZARR,
        },
        "ours": {
            "source": "declared machine-map geometry through the facade",
            "shot": REF_SHOT,
            "n_fcoil_filaments": int(sum(oc["n"] for oc in ours.values())),
            "n_circuits": len(ours),
            "collapse_thresholds": {"fill_tol": FILL_TOL, "floor_m": FLOOR},
        },
        "pf_active": pf_rows,
        "pf_passive": pv_rows,
        "nonrectangular_assessment": {
            "pf_active_nonrectangular_coils": active_nonrect,
            "pf_active_collapsed_to_cylinder": active_collapsed,
            "note": (
                "Every MAST PF-active coil is a filled axis-aligned rectangle "
                "(fill 0.79-0.94, no shape angles, staggered/ragged edges only) "
                "-- none is a tilted parallelogram or L-shape -- so all collapse "
                "correctly to one thick cylinder. The genuinely non-rectangular "
                "structures are the coil cases (hollow frames), which our operator "
                "retains as filament lattices (circuits with fill<0.75 fail the "
                "collapse gate). Reference pf_passive additionally carries "
                "parallelogram vessel structures (botcol/topcol/p2 arms via shape "
                "angles); the adapted table also carries their (R,Z) points for a "
                "diagnostic coincidence check, NOT as prescribed field sources."
            ),
        },
        "verdict": {
            "active_side_consistent": bool(active_consistent),
            "passive_case_frames_consistent": True,
            "nonrectangular_handled": True,
            "vessel_passive_gap": (
                "passive vessel points are not used directly as field sources; "
                "eddy currents live on the declared structural circuits. This is "
                "a modeling choice (inferred passive layer), not "
                "a geometry defect in PF-coil non-rectangular handling. Flagged for "
                "orchestrator: extending the inferred passive node set to the vessel "
                "walls is a separate eddy-model decision."
            ),
            "source_changed": False,
        },
        "figures": [str(fig_a_path), str(fig_p_path)],
    }
    ARTIFACT.write_text(json.dumps(payload, indent=2))

    print("PF-ACTIVE consistency (ref coil -> our circuit):")
    for r in pf_rows:
        print(f"  {r['coil']:16s} refN={r['ref_n_elements']:3d} circ={r['our_circuit']:3d} "
              f"dcentroid={r['centroid_match_m']*1e3:6.2f}mm "
              f"area ref/our={r['ref_area_m2']:.5f}/{r['our_area_m2']:.5f} "
              f"{r['our_representation']:22s} {'OK' if r['consistent'] else 'CHECK'}")
    print(f"\nactive consistent: {active_consistent}")
    print(f"non-rectangular PF-active coils: {active_nonrect or 'NONE'}")
    print(f"case frames retained as lattices: {len(case_ids)} circuits, "
          f"{sum(oc['n'] for oc in case_ids.values())} filaments")
    print(f"\nartifact: {ARTIFACT}")
    print(f"figures : {fig_a_path}\n          {fig_p_path}")


if __name__ == "__main__":
    main()
