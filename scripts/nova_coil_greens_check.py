"""Cross-check the coil sensor forward map: point filament vs finite-area cylinder.

The equilibrium forward operator (:mod:`imas_ambix.gs.operator`) builds its
sensor columns ``g_pf`` from POINT-FILAMENT Green's functions
(:func:`greens_psi` / :func:`greens_bz_br`, weighted by ``xmult``), while the
GS solve-domain coil ψ columns use the FINITE-AREA cylinder kernel
(:func:`imas_ambix.gs.cylinder.hybrid_greens`, extracted from the nova EM
package and golden-pinned).  A point filament is log-singular at the source, so
sensors that sit CLOSE to a winding pack (the P4/P5 flux loops, which carried
the largest spurious plasma-fitted offsets) are exactly where a point-vs-area
difference would appear in the sensor map.

This script quantifies that difference three ways per (coil, sensor):

  (i)   in-tree POINT filament  — reproduces ``fwd.g_pf`` exactly (harness pin);
  (ii)  in-tree finite-area CYLINDER (same xmult weights, |width|/|height|
        floors 0.01 as in gs_solve), merged-circuit averaged per coil column;
  (iii) NOVA's own Cylinder Biot-Savart, driven via CoilSet at the ACTUAL
        sensor points on a representative filament set (largest solenoid pack,
        P4/P5 packs, and a spread over coils/sizes) — validates that (ii) IS
        nova at real MAST sensor-to-coil separations.

Outputs the per-cell fractional delta and the delta in units of the per-channel
noise σ at typical coil currents, plus a geometry/turns audit of the
efm-derived filament table, plus the adequacy verdict.

Heavy sweep — run on a SLURM debug node, never the login node.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
NOVA = Path.home() / "Code" / "nova"
THIS_FILE = Path(__file__).resolve()

ARTIFACT = (
    REPO / "imas_ambix" / "latent" / "artifacts" / "patch_gate"
    / "nova_coil_greens_check.json"
)
FIGDIR = REPO / "docs" / "figures" / "force-balance-spine"

SHOT = 18502
SIGMA_SHOTS = (18502, 18503)
CYL_FLOOR = 0.01  # gs_solve winding-pack extent floor [m]


def _import_tree():
    from imas_ambix.gs import operator as op
    from imas_ambix.gs.cylinder import cylinder_greens
    from imas_ambix.gs.geometry import build_table_for_shot

    return op, cylinder_greens, build_table_for_shot


# --------------------------------------------------------------------------
# forward-map column builders (mirror operator.build_operator exactly)
# --------------------------------------------------------------------------


def _sensor_rows(op, table):
    """(sensor_r, sensor_z, sensor_ang, is_flux, channels) in operator row order."""
    rs, zs, angs, kinds, chans = [], [], [], [], []
    for m in table.sensor_map:
        chans.append(m.amb_channel)
        kinds.append(m.kind)
        rs.append(m.r)
        zs.append(m.z)
        angs.append(0.0 if m.angle_deg is None else float(m.angle_deg))
    is_flux = np.array([k == "flux_loop" for k in kinds], dtype=bool)
    return (
        np.array(rs, float),
        np.array(zs, float),
        np.array(angs, float),
        is_flux,
        chans,
    )


def _point_col(op, fr, fz, fw, sr, sz, sang, is_flux):
    """One point-filament column at each sensor (Σ w·ψ or Σ w·B_proj)."""
    col = np.zeros(sr.shape)
    for ar, az, w in zip(fr, fz, fw, strict=True):
        if w == 0.0:
            continue
        psi = op.greens_psi(sr, sz, float(ar), float(az))
        bz, br = op.greens_bz_br(sr, sz, float(ar), float(az))
        bproj = op._project_bprobe(bz, br, sang)
        col = col + w * np.where(is_flux, psi, bproj)
    return col


def _cyl_col(cylinder_greens, fr, fz, fw, fdr, fdz, sr, sz, sang, is_flux):
    """One finite-area cylinder column at each sensor (floored extents)."""
    th = np.deg2rad(sang)
    col = np.zeros(sr.shape)
    for ar, az, w, dr, dz in zip(fr, fz, fw, fdr, fdz, strict=True):
        if w == 0.0:
            continue
        psi, br, bz = cylinder_greens(
            sr, sz, float(ar), float(az),
            max(abs(float(dr)), CYL_FLOOR), max(abs(float(dz)), CYL_FLOOR),
        )
        bproj = br * np.cos(th) + bz * np.sin(th)
        col = col + w * np.where(is_flux, psi, bproj)
    return col


def build_columns(op, cylinder_greens, table, fwd):
    """Return (g_point, g_cyl) each (n_sensor, n_coil) in fwd column order."""
    sr, sz, sang, is_flux, _ = _sensor_rows(op, table)
    by_circ: dict[int, list] = {}
    for f in table.pf_filaments:
        by_circ.setdefault(f.circuit, []).append(f)

    def circ_arrays(circ):
        fs = by_circ[circ]
        return (
            np.array([f.r for f in fs], float),
            np.array([f.z for f in fs], float),
            np.array([f.xmult for f in fs], float),  # turns=1 → weight=xmult
            np.array([f.width for f in fs], float),
            np.array([f.height for f in fs], float),
        )

    g_point_cols, g_cyl_cols = [], []
    for circs in fwd.pf_merged_circuits:
        pc, cc = [], []
        for circ in circs:
            fr, fz, fw, fdr, fdz = circ_arrays(circ)
            pc.append(_point_col(op, fr, fz, fw, sr, sz, sang, is_flux))
            cc.append(
                _cyl_col(cylinder_greens, fr, fz, fw, fdr, fdz, sr, sz, sang, is_flux)
            )
        g_point_cols.append(np.mean(pc, axis=0))
        g_cyl_cols.append(np.mean(cc, axis=0))
    return np.column_stack(g_point_cols), np.column_stack(g_cyl_cols)


# --------------------------------------------------------------------------
# nova cross-check on a representative filament set
# --------------------------------------------------------------------------


def representative_filaments(table, fwd):
    """Pick representative filaments spanning coils + section sizes.

    Per coil column: the widest and the tallest filament (the extremes that
    stress the finite-area kernel most), plus the centroid-nearest one.  The
    solenoid pack and the P4/P5 packs are guaranteed included because they are
    their own coil columns.
    """
    by_circ: dict[int, list] = {}
    for f in table.pf_filaments:
        by_circ.setdefault(f.circuit, []).append(f)
    picks = []
    for circs, chan in zip(
        fwd.pf_merged_circuits, fwd.pf_amc_channels, strict=True
    ):
        fs = [f for c in circs for f in by_circ[c]]
        if not fs:
            continue
        widest = max(fs, key=lambda f: abs(f.width))
        tallest = max(fs, key=lambda f: abs(f.height))
        for f, why in ((widest, "widest"), (tallest, "tallest")):
            picks.append(
                {
                    "coil": chan,
                    "why": why,
                    "a": float(f.r),
                    "z0": float(f.z),
                    "da": max(abs(float(f.width)), CYL_FLOOR),
                    "dz": max(abs(float(f.height)), CYL_FLOOR),
                }
            )
    # de-duplicate identical boxes
    seen, uniq = set(), []
    for p in picks:
        key = (round(p["a"], 6), round(p["z0"], 6), round(p["da"], 6), round(p["dz"], 6))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(p)
    return uniq


def nova_cross_check(cylinder_greens, table, fwd, sr, sz):
    """Compare in-tree cylinder vs nova Cylinder at actual sensors, per filament."""
    fils = representative_filaments(table, fwd)
    job = {
        "sources": [
            {"a": f["a"], "z0": f["z0"], "da": f["da"], "dz": f["dz"]} for f in fils
        ],
        "targets": [[float(r), float(z)] for r, z in zip(sr, sz, strict=True)],
    }
    proc = subprocess.run(
        ["uv", "run", "--project", str(NOVA), "python", str(THIS_FILE),
         "--nova-driver"],
        input=json.dumps(job),
        capture_output=True,
        text=True,
        cwd=str(NOVA),
        timeout=1800,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"nova driver failed:\n{proc.stderr[-2000:]}")
    res = json.loads(proc.stdout)
    nova_psi = np.array(res["psi"])  # (n_fil, n_sensor)
    nova_br = np.array(res["br"])
    nova_bz = np.array(res["bz"])

    worst_frac = 0.0
    per_fil = []
    for i, f in enumerate(fils):
        psi, br, bz = cylinder_greens(sr, sz, f["a"], f["z0"], f["da"], f["dz"])
        # compare psi (flux loops) and both field comps on all sensor points
        for label, tree_v, nova_v in (
            ("psi", psi, nova_psi[i]),
            ("br", br, nova_br[i]),
            ("bz", bz, nova_bz[i]),
        ):
            scale = np.maximum(np.abs(nova_v), 1e-30)
            frac = np.abs(tree_v - nova_v) / scale
            # only judge where the value is non-negligible for that quantity
            big = np.abs(nova_v) > 1e-3 * np.nanmax(np.abs(nova_v))
            mx = float(np.nanmax(frac[big])) if big.any() else 0.0
            worst_frac = max(worst_frac, mx)
        per_fil.append(
            {
                "coil": f["coil"],
                "why": f["why"],
                "box": [f["a"], f["z0"], f["da"], f["dz"]],
            }
        )
    return {
        "mu0_nova": res["mu0"],
        "n_filament": len(fils),
        "worst_fractional_delta_tree_vs_nova": worst_frac,
        "filaments": per_fil,
    }


# --------------------------------------------------------------------------
# noise σ and typical currents
# --------------------------------------------------------------------------


def channel_sigma_and_currents(op, fwd):
    """(sigma[S], typ_current[C]) — per-channel noise σ and p90 |I| per coil."""
    from imas_ambix.latent.data import (
        feature_schema,
        load_shot_windows,
        robust_channel_scale,
    )

    schema = feature_schema()
    stds, ipfs = [], []
    for s in SIGMA_SHOTS:
        w = load_shot_windows(int(s), fwd, "audit", schema, with_referee=False)
        if w is None:
            continue
        stds.append(np.nanstd(w.raw_mag, axis=0))
        ipfs.append(np.abs(w.i_pf))
    std = np.nanmean(np.vstack(stds), axis=0)
    sigma = robust_channel_scale(std, fwd.sensor_channels)
    typ_current = np.nanpercentile(np.vstack(ipfs), 90, axis=0)  # (C,)
    return sigma, typ_current


# --------------------------------------------------------------------------
# geometry / turns audit (efm table internal consistency)
# --------------------------------------------------------------------------


def geometry_audit(op, table, fwd):
    """Audit the efm filament table per coil column.

    nova carries NO static MAST PF-coil description (its MAST projects pull
    geometry from UDA/fair-mast at runtime), so there is no nova geometry to
    diff against; the efm ``fcoil`` table IS the machine description EFIT sees.
    This audits its internal consistency: turns==1 everywhere (the SI-conversion
    premise), per-coil Σxmult (≈1 per redundant circuit representation), the
    weighted centroid vs the operator's reference centroid, and section extents.
    """
    by_circ: dict[int, list] = {}
    for f in table.pf_filaments:
        by_circ.setdefault(f.circuit, []).append(f)
    all_turns = np.array([f.turns for f in table.pf_filaments], float)
    rows = []
    for circs, chan, cls_circ in zip(
        fwd.pf_merged_circuits, fwd.pf_amc_channels, fwd.pf_circuits, strict=True
    ):
        cc = next(c for c in fwd.circuit_classes if c.circuit == cls_circ)
        fs = [f for c in circs for f in by_circ[c]]
        w = np.array([f.xmult for f in fs], float)
        rr = np.array([f.r for f in fs], float)
        zz = np.array([f.z for f in fs], float)
        cr = float((w * rr).sum() / w.sum())
        cz = float((w * zz).sum() / w.sum())
        label = cc.coil_label.replace("_case", "")
        ref = op._PF_COIL_CENTROID.get(label)
        cen_delta_mm = (
            float(np.hypot(cr - ref[0], cz - ref[1]) * 1e3) if ref else None
        )
        rows.append(
            {
                "coil": chan,
                "coil_label": cc.coil_label,
                "merged_circuits": circs,
                "n_filament": len(fs),
                "sum_xmult_per_circuit": [
                    round(float(sum(f.xmult for f in by_circ[c])), 4) for c in circs
                ],
                "weighted_centroid": [round(cr, 4), round(cz, 4)],
                "ref_centroid": list(ref) if ref else None,
                "centroid_delta_mm": (
                    round(cen_delta_mm, 1) if cen_delta_mm is not None else None
                ),
                "width_range_m": [
                    round(float(min(abs(f.width) for f in fs)), 4),
                    round(float(max(abs(f.width) for f in fs)), 4),
                ],
                "height_range_m": [
                    round(float(min(abs(f.height) for f in fs)), 4),
                    round(float(max(abs(f.height) for f in fs)), 4),
                ],
            }
        )
    return {
        "nova_mast_geometry_available": False,
        "nova_note": (
            "nova has no static MAST PF-coil description; its MAST projects read"
            " geometry from UDA/fair-mast at runtime. The efm fcoil table is the"
            " authoritative machine geometry (EFIT's own). Audit below is the"
            " efm table's internal consistency, not a nova diff."
        ),
        "turns_all_unity": bool(np.allclose(all_turns, 1.0)),
        "turns_min_max": [float(all_turns.min()), float(all_turns.max())],
        "per_coil": rows,
    }


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def main():
    op, cylinder_greens, build_table_for_shot = _import_tree()

    print(f"[1/6] building operator for shot {SHOT} ...", flush=True)
    table = build_table_for_shot(SHOT)
    fwd = op.build_operator(table)
    sr, sz, sang, is_flux, chans = _sensor_rows(op, table)
    n_sensor = len(chans)
    n_coil = len(fwd.pf_amc_channels)
    print(f"      {n_coil} coil columns, {n_sensor} sensors", flush=True)

    print("[2/6] point-filament reproduction (harness pin) ...", flush=True)
    g_point, g_cyl = build_columns(op, cylinder_greens, table, fwd)
    max_abs_reproduce = float(np.max(np.abs(g_point - fwd.g_pf)))
    reproduces = bool(np.allclose(g_point, fwd.g_pf, rtol=1e-10, atol=1e-18))
    print(f"      max|g_point - fwd.g_pf| = {max_abs_reproduce:.3e}  ok={reproduces}",
          flush=True)
    if not reproduces:
        raise RuntimeError("point-filament reproduction does not match fwd.g_pf")

    print("[3/6] nova cross-check on representative filaments ...", flush=True)
    nova = nova_cross_check(cylinder_greens, table, fwd, sr, sz)
    print(f"      worst frac(tree cyl vs nova) = "
          f"{nova['worst_fractional_delta_tree_vs_nova']:.3e} over "
          f"{nova['n_filament']} filaments", flush=True)

    print("[4/6] channel σ and typical currents ...", flush=True)
    sigma, typ_current = channel_sigma_and_currents(op, fwd)

    print("[5/6] delta maps ...", flush=True)
    delta = g_cyl - g_point  # (S, C) per amp
    denom = np.where(np.abs(g_point) > 0, np.abs(g_point), np.nan)
    frac = np.abs(delta) / denom  # fractional (cyl-point)/|point|
    # delta at typical currents, in units of channel σ
    delta_at_I = np.abs(delta) * typ_current[None, :]  # (S, C) in sensor units
    delta_sigma = delta_at_I / np.where(sigma > 0, sigma, np.nan)[:, None]

    # worst cells
    def top_cells(mat, kthresh, n=25):
        flat = []
        for si in range(n_sensor):
            for ci in range(n_coil):
                v = mat[si, ci]
                if np.isfinite(v) and v >= kthresh:
                    flat.append((float(v), chans[si], is_flux[si], fwd.pf_amc_channels[ci],
                                 float(frac[si, ci]) if np.isfinite(frac[si, ci]) else None,
                                 float(g_point[si, ci]), float(g_cyl[si, ci])))
        flat.sort(reverse=True)
        return [
            {
                "delta_sigma": round(v, 4),
                "sensor": ch,
                "kind": "flux_loop" if fl else "b_probe",
                "coil": coil,
                "frac_delta": round(fr, 5) if fr is not None else None,
                "g_point": gp,
                "g_cyl": gc,
            }
            for (v, ch, fl, coil, fr, gp, gc) in flat[:n]
        ]

    above_1sigma = top_cells(delta_sigma, 1.0, n=100)
    above_0p1sigma = top_cells(delta_sigma, 0.1, n=100)
    worst_sigma = float(np.nanmax(delta_sigma))
    worst_frac = float(np.nanmax(frac))

    print(f"      worst Δ/σ = {worst_sigma:.3f}  worst frac = {worst_frac:.3e}",
          flush=True)
    print(f"      #cells > 0.1σ = {len(above_0p1sigma)}  > 1σ = {len(above_1sigma)}",
          flush=True)

    print("[6/6] geometry audit ...", flush=True)
    geom = geometry_audit(op, table, fwd)

    adequate = worst_sigma < 0.1
    verdict = (
        "point-filament sensor map ADEQUATE (< 0.1σ everywhere)"
        if adequate
        else (
            f"point-filament sensor map INADEQUATE — {len(above_0p1sigma)} "
            f"(coil,sensor) cells exceed 0.1σ, {len(above_1sigma)} exceed 1σ; "
            "operator.py should switch g_pf to cylinder columns"
        )
    )
    print("VERDICT:", verdict, flush=True)

    payload = {
        "schema": "nova-coil-greens-check-v0",
        "shot": SHOT,
        "signature_key": fwd.signature_key,
        "n_coil": n_coil,
        "n_sensor": n_sensor,
        "coil_columns": fwd.pf_amc_channels,
        "sigma_shots": list(SIGMA_SHOTS),
        "cyl_extent_floor_m": CYL_FLOOR,
        "harness_pin": {
            "point_reproduces_g_pf": reproduces,
            "max_abs_diff": max_abs_reproduce,
        },
        "nova_cross_check": nova,
        "mu0_tree": float(op.MU0),
        "delta_summary": {
            "worst_delta_over_sigma_at_typical_current": worst_sigma,
            "worst_fractional_delta": worst_frac,
            "n_cells_gt_0p1_sigma": len(above_0p1sigma),
            "n_cells_gt_1_sigma": len(above_1sigma),
            "cells_gt_1_sigma": above_1sigma,
            "cells_gt_0p1_sigma": above_0p1sigma,
        },
        "adequate_point_map": adequate,
        "verdict": verdict,
        "geometry_audit": geom,
    }
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT.write_text(json.dumps(payload, indent=2, default=float))
    print("wrote", ARTIFACT, flush=True)

    # stash arrays for the figure step
    np.savez(
        ARTIFACT.with_suffix(".npz"),
        g_point=g_point,
        g_cyl=g_cyl,
        delta_sigma=np.nan_to_num(delta_sigma),
        frac=np.nan_to_num(frac),
        sigma=sigma,
        typ_current=typ_current,
        is_flux=is_flux,
        sensor_r=sr,
        sensor_z=sz,
        coils=np.array(fwd.pf_amc_channels, dtype=object),
        sensors=np.array(chans, dtype=object),
    )
    _make_figures(op, table, fwd, payload, ARTIFACT.with_suffix(".npz"))


def _make_figures(op, table, fwd, payload, npz_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = np.load(npz_path, allow_pickle=True)
    delta_sigma = d["delta_sigma"]
    coils = list(d["coils"])
    sensors = list(d["sensors"])
    is_flux = d["is_flux"]
    FIGDIR.mkdir(parents=True, exist_ok=True)

    # --- heatmap coil × sensor of Δ/σ ---
    fig, ax = plt.subplots(figsize=(14, 8))
    im = ax.imshow(delta_sigma.T, aspect="auto", cmap="magma",
                   vmin=0, vmax=max(1.0, np.percentile(delta_sigma, 99.5)))
    ax.set_yticks(range(len(coils)))
    ax.set_yticklabels(coils, fontsize=7)
    ax.set_xlabel("sensor row (flux loops + B-probes, operator order)")
    ax.set_ylabel("PF coil column")
    ax.set_title(
        f"|Δ(cylinder − point)| / σ at typical coil current  (shot {payload['shot']})\n"
        f"worst = {payload['delta_summary']['worst_delta_over_sigma_at_typical_current']:.3f}σ  "
        f"| cells >0.1σ: {payload['delta_summary']['n_cells_gt_0p1_sigma']}  "
        f">1σ: {payload['delta_summary']['n_cells_gt_1_sigma']}"
    )
    fig.colorbar(im, ax=ax, label="Δ / σ")
    # annotate worst cells
    for cell in payload["delta_summary"]["cells_gt_1_sigma"][:12]:
        try:
            si = sensors.index(cell["sensor"])
            ci = coils.index(cell["coil"])
        except ValueError:
            continue
        ax.text(si, ci, f"{cell['delta_sigma']:.1f}", color="cyan",
                fontsize=6, ha="center", va="center")
    fig.tight_layout()
    f1 = FIGDIR / "fig-nova-greens-deltas.png"
    fig.savefig(f1, dpi=110)
    plt.close(fig)

    # --- geometry audit: R-Z filament overlay + centroid deltas ---
    fig, (axg, axb) = plt.subplots(1, 2, figsize=(14, 8))
    by_circ: dict[int, list] = {}
    for f in table.pf_filaments:
        by_circ.setdefault(f.circuit, []).append(f)
    # all filaments coloured known (coil) vs inferred passive
    known_circs = {c for circs in fwd.pf_merged_circuits for c in circs}
    for circ, fs in by_circ.items():
        rr = [f.r for f in fs]
        zz = [f.z for f in fs]
        known = circ in known_circs
        axg.scatter(rr, zz, s=6,
                    c=("#1f77b4" if known else "#cccccc"),
                    label=None, zorder=3 if known else 1)
    # reference centroids
    for label, (lr, lz) in op._PF_COIL_CENTROID.items():
        axg.scatter([lr], [lz], marker="x", c="red", s=40, zorder=5)
        axg.text(lr + 0.02, lz, label, fontsize=6, color="red")
    # sensors
    axg.scatter(d["sensor_r"][is_flux], d["sensor_z"][is_flux], marker="o",
                facecolors="none", edgecolors="green", s=18, label="flux loop")
    axg.scatter(d["sensor_r"][~is_flux], d["sensor_z"][~is_flux], marker="^",
                c="orange", s=12, label="B-probe")
    axg.set_aspect("equal")
    axg.set_xlabel("R [m]")
    axg.set_ylabel("Z [m]")
    axg.set_title("efm filaments (blue=known coil, grey=inferred) +\n"
                  "reference centroids (red x) + sensors")
    axg.legend(fontsize=7, loc="upper right")

    labels = [r["coil"] for r in payload["geometry_audit"]["per_coil"]]
    cdelta = [r["centroid_delta_mm"] or 0.0
              for r in payload["geometry_audit"]["per_coil"]]
    axb.barh(range(len(labels)), cdelta, color="#1f77b4")
    axb.set_yticks(range(len(labels)))
    axb.set_yticklabels(labels, fontsize=7)
    axb.set_xlabel("weighted centroid − reference centroid [mm]")
    axb.set_title("per-coil centroid delta vs operator reference\n"
                  f"turns all unity: {payload['geometry_audit']['turns_all_unity']}")
    axb.axvline(_match_mm(op), color="red", ls="--", lw=1,
                label=f"match tol {_match_mm(op):.0f} mm")
    axb.legend(fontsize=7)
    fig.tight_layout()
    f2 = FIGDIR / "fig-nova-geometry-audit.png"
    fig.savefig(f2, dpi=110)
    plt.close(fig)
    print("wrote", f1, f2, flush=True)


def _match_mm(op):
    return op._COIL_MATCH_M * 1e3


def _run_nova_driver():
    """Nova side: read a {sources, targets} job on stdin, write ψ/Br/Bz on stdout.

    Runs under nova's OWN venv (``uv run --project ~/Code/nova``); imports nova,
    not imas_ambix.  Drives nova's authoritative Cylinder Biot-Savart via a
    single-coil CoilSet at unit total current — ψ/Br/Bz per ampere at targets.
    """
    from scipy.constants import mu_0  # noqa: PLC0415

    from nova.frame.coilset import CoilSet  # noqa: PLC0415

    job = json.load(sys.stdin)
    targets = np.asarray(job["targets"], dtype=float)  # (N, 2) -> xz
    out = {"mu0": float(mu_0), "psi": [], "br": [], "bz": []}
    for s in job["sources"]:
        cs = CoilSet(dcoil=-1, dplasma=-1, field_attrs=["Br", "Bz", "Psi"])
        cs.coil.insert(
            s["a"], s["z0"], s["da"], s["dz"],
            segment="cylinder", turn="r", nturn=1,
        )
        cs.saloc["Ic"] = 1.0
        cs.point.solve(targets)
        ic = np.asarray(cs.sloc["Ic"], dtype=float)
        out["psi"].append((np.asarray(cs.point.data.Psi.values) @ ic).ravel().tolist())
        out["br"].append((np.asarray(cs.point.data.Br.values) @ ic).ravel().tolist())
        out["bz"].append((np.asarray(cs.point.data.Bz.values) @ ic).ravel().tolist())
    json.dump(out, sys.stdout)


if __name__ == "__main__":
    if "--nova-driver" in sys.argv:
        _run_nova_driver()
    else:
        sys.exit(main())
