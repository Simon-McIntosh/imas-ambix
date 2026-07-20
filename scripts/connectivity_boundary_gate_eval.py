"""Go/no-go gate for the GPU-native connectivity boundary read.

The connectivity LCFS read (:func:`imas_ambix.latent.topology.lcfs_contour`) is
the one GPU-hostile step in the batched engine — host contourpy + a bisection
over traced contours.  :mod:`imas_ambix.latent.connectivity_boundary` reproduces
the SAME monotone connectivity algorithm with fixed-shape device primitives (a
flood-fill from the axis + a candidate-level sweep + a fixed-count bisection),
so the read runs on-device, batches over slices, and is differentiable.  This
gate records the pre-declared verdicts:

  T-B1 (reproduction) — on held-out slices the device flood-fill ψ_bnd and LCFS
        radii match the host ``lcfs_contour`` (the disc-pushout reference,
        ``clip_legs=True``) to tolerance, across LIMITED and DIVERTED slices.
  T-B2 (on-device)    — the read imports no contourpy / ndimage / argwhere, and a
        batch of slices sharing one grid is a single ``jax.vmap`` (fixed shapes),
        byte-parity with the per-slice read; batched throughput reported.
  T-B3 (continuity)   — on a synthetic limited→diverted sweep the connectivity
        ψ_bnd is continuous through the marginal transition, where the classify-
        first boundary (innermost in-vessel X-point flux, else limiter) steps.

Pre-declared tolerances: LCFS-radius agreement ≤ 3.0 cm median (the host label
tolerance; one grid cell is ~2-3 cm), ψ_bnd agreement ≤ 0.03 of the axis→boundary
flux span (the grid discretisation floor — the connectivity level snaps to the
binding grid cell).  Firewall unchanged: the read consumes only the solved ψ +
the wall mask; no EFIT.

Usage:
    uv run python -m scripts.connectivity_boundary_gate_eval --n-shots 5 --max-slices 6
"""

from __future__ import annotations

import argparse
import ast
import inspect
import json
import logging
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("connectivity_boundary_gate")

FIG_DIR = Path("docs/figures/connectivity-topology-reader")

LCFS_TOL_CM = 3.0
PSI_BND_FRAC_TOL = 0.03
CONFINED_AXIS_R_MAX = 1.4

# §3 classify-after tolerances
AXIS_TOL_CM = 3.0  # sub-grid axis vs CPU magnetic_axis (one grid cell)
XSET_TOL_CM = 3.0  # X-point position vs CPU emergent_xpoints
CLASS_ACC_TOL = 0.80  # device is_diverted vs CPU class agreement (soft near transition)
DIVERTED_PSI_BND_TOL = 0.005  # unified diverted ψ_bnd residual (fraction of span)


# ---------------------------------------------------------------------------
# real-slice reproduction (T-B1)
# ---------------------------------------------------------------------------


def _classify_diverted(psi, grid, ring) -> bool:
    """Is the slice diverted? — an in-vessel saddle sits ON the LCFS ring."""
    from imas_ambix.latent.topology import emergent_xpoints, find_critical_points

    cp = find_critical_points(psi, grid.rg, grid.zg)
    _xset, is_div = emergent_xpoints(cp.x_points, ring, tol=1.5 * float(grid.dr))
    return bool(is_div)


def _xset_match_cm(dev_xset, cpu_xset) -> float:
    """Match distance [cm] between two NaN-padded (S, 2) X-point sets.

    Order-invariant: for each finite CPU X-point, the nearest finite device
    X-point; the reported value is the worst (max) such pairing, so a missed or
    displaced X-point shows up.  NaN when either set has no finite point.
    """
    dev = np.asarray(dev_xset, dtype=np.float64).reshape(-1, 2)
    cpu = np.asarray(cpu_xset, dtype=np.float64).reshape(-1, 2)
    dev = dev[np.isfinite(dev).all(axis=1)]
    cpu = cpu[np.isfinite(cpu).all(axis=1)]
    if dev.shape[0] == 0 or cpu.shape[0] == 0:
        return float("nan")
    d = np.hypot(
        cpu[:, None, 0] - dev[None, :, 0], cpu[:, None, 1] - dev[None, :, 1]
    ).min(axis=1)
    return 100.0 * float(np.max(d))


def _slice_rows(shot: int, *, nr: int, nz: int, max_slices: int, min_ip_ka: float):
    """Per-slice host-vs-device boundary reads on one held-out shot's disc fields."""
    from imas_ambix.latent.boundary_disc import disc_read
    from imas_ambix.latent.connectivity_boundary import boundary_read
    from imas_ambix.latent.topology import (
        emergent_xpoints,
        find_critical_points,
        lcfs_contour,
        magnetic_axis,
    )
    from scripts.heldout_mse_gate_eval import _campaign_table
    from scripts.spine_label_factory import factory_shot_payloads

    table = _campaign_table(shot)
    if table is None:
        return [], []
    payload = factory_shot_payloads(
        shot, nr=nr, nz=nz, max_slices=max_slices, min_ip_ka=min_ip_ka, table=table
    )
    if payload is None:
        return [], []
    grid, tbl, basis = payload["grid"], payload["table"], payload["basis"]
    rows = []
    overlays = []  # (class, psi, cpu, gpu) kept for figure overlays
    for p in payload["payloads"]:
        try:
            inv = disc_read(p, grid, tbl, basis)
        except Exception:  # noqa: BLE001 — record nothing, keep sweeping
            inv = None
        if inv is None or inv.ring is None:
            continue
        psi = np.asarray(inv.psi_tot, dtype=np.float64)
        centroid = (float(inv.centroid_r), float(inv.centroid_z))
        if not (np.isfinite(centroid[0]) and centroid[0] <= CONFINED_AXIS_R_MAX):
            continue
        cpu = lcfs_contour(
            psi,
            grid.rg,
            grid.zg,
            centroid,
            clip_legs=True,
            limiter_r=grid.limiter_r,
            limiter_z=grid.limiter_z,
        )
        if not cpu.found:
            continue
        gpu = boundary_read(psi, grid, centroid, lcfs_norm=1.0)
        span = cpu.psi_bnd - gpu.psi_axis
        if abs(span) < 1e-12:
            continue
        dpsi_frac = abs(gpu.psi_bnd - cpu.psi_bnd) / abs(span)
        ok = np.isfinite(cpu.radii) & np.isfinite(gpu.radii)
        if not ok.any():
            continue
        dr = 100.0 * np.abs(gpu.radii[ok] - cpu.radii[ok])
        # CPU reference nulls (§3): magnetic axis (O-point) + emergent X-set/class
        cp = find_critical_points(psi, grid.rg, grid.zg)
        cpu_axis = magnetic_axis(
            psi,
            grid.rg,
            grid.zg,
            limiter_r=grid.limiter_r,
            limiter_z=grid.limiter_z,
            cp=cp,
        )
        cpu_xset, is_div = emergent_xpoints(
            cp.x_points, cpu.ring, tol=1.5 * float(grid.dr)
        )
        axis_d_cm = float("nan")
        if (
            cpu_axis is not None
            and np.isfinite(gpu.axis[0])
            and np.isfinite(gpu.axis[1])
        ):
            axis_d_cm = 100.0 * float(
                np.hypot(gpu.axis[0] - cpu_axis[0], gpu.axis[1] - cpu_axis[1])
            )
        xset_d_cm = _xset_match_cm(gpu.xset, cpu_xset)
        # --- T-D3: exact g_wall node tangency vs the bilerp read (no-regression) ---
        # Same field (inv.psi_tot), same read — only the wall-tangency SOURCE swaps
        # from bilinear off the grid to the exact campaign g_wall GEMM.  The grid
        # kernel reconstruction is checked against the basis psi_tot so the swap is
        # a clean A/B (the consistency residual quantifies any kernel offset).
        gwall = _gwall_leg(grid, p, inv, psi, centroid, cpu, float(span))
        rows.append(
            {
                "shot": shot,
                "time_s": float(p.time_s),
                "ip_a": float(abs(p.ip_amperes)),
                "diverted": is_div,
                "psi_axis": float(gpu.psi_axis),
                "cpu_psi_bnd": float(cpu.psi_bnd),
                "gpu_psi_bnd": float(gpu.psi_bnd),
                "dpsi_frac": float(dpsi_frac),
                "radii_dmed_cm": float(np.median(dr)),
                "radii_dmax_cm": float(np.max(dr)),
                "n_core_cells": int(gpu.n_core_cells),
                # T-D3 g_wall swap
                "gwall_dpsi_frac": gwall["dpsi_frac"],
                "gwall_radii_dmed_cm": gwall["radii_dmed_cm"],
                "gwall_vs_bilerp_dpsi_frac": gwall["vs_bilerp_dpsi_frac"],
                "gwall_consistency_frac": gwall["consistency_frac"],
                # §3 classify-after diagnostics
                "cpu_axis_r": None if cpu_axis is None else float(cpu_axis[0]),
                "cpu_axis_z": None if cpu_axis is None else float(cpu_axis[1]),
                "dev_axis_r": float(gpu.axis[0]),
                "dev_axis_z": float(gpu.axis[1]),
                "axis_d_cm": axis_d_cm,
                "dev_is_diverted": bool(gpu.is_diverted),
                "class_margin": float(np.clip(gpu.class_margin, -1.0, 1.0)),
                "xset_d_cm": xset_d_cm,
                "found_both": True,
            }
        )
        if len(overlays) < 6:
            overlays.append(
                ("diverted" if is_div else "limited", psi, grid, cpu, gpu, centroid)
            )
    return rows, overlays


def _tb1(shots, *, nr, nz, max_slices, min_ip_ka):
    rows_all = []
    overlays_all = []
    for s in shots:
        logger.info("shot %d ...", s)
        try:
            rows, ov = _slice_rows(
                int(s), nr=nr, nz=nz, max_slices=max_slices, min_ip_ka=min_ip_ka
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("  shot %d failed: %s", s, exc)
            rows, ov = [], []
        n_div = sum(1 for r in rows if r["diverted"])
        logger.info(
            "  %d slices (%d diverted, %d limited)", len(rows), n_div, len(rows) - n_div
        )
        rows_all += rows
        overlays_all += ov

    def _med(key, subset=None):
        vals = [r[key] for r in rows_all if (subset is None or r["diverted"] == subset)]
        return float(np.median(vals)) if vals else float("nan")

    def _p90(key):
        vals = [r[key] for r in rows_all]
        return float(np.percentile(vals, 90)) if vals else float("nan")

    n = len(rows_all)
    n_div = sum(1 for r in rows_all if r["diverted"])
    radii_med = _med("radii_dmed_cm")
    dpsi_med = _med("dpsi_frac")
    tb1 = (
        np.isfinite(radii_med)
        and radii_med <= LCFS_TOL_CM
        and np.isfinite(dpsi_med)
        and dpsi_med <= PSI_BND_FRAC_TOL
    )
    return {
        "verdict": "PASS" if tb1 else "FAIL",
        "n_slices": n,
        "n_diverted": n_div,
        "n_limited": n - n_div,
        "radii_dmed_cm_overall": radii_med,
        "radii_dmed_cm_diverted": _med("radii_dmed_cm", True),
        "radii_dmed_cm_limited": _med("radii_dmed_cm", False),
        "radii_dmax_cm_p90": _p90("radii_dmax_cm"),
        "dpsi_frac_median": dpsi_med,
        "dpsi_frac_p90": _p90("dpsi_frac"),
        "tolerances": {"lcfs_cm": LCFS_TOL_CM, "psi_bnd_frac": PSI_BND_FRAC_TOL},
        "rows": rows_all,
    }, overlays_all


# ---------------------------------------------------------------------------
# on-device / batchable (T-B2)
# ---------------------------------------------------------------------------


def _tb2(overlays):
    """Batched vmap parity + throughput, and a static import audit of the module."""
    import jax
    import jax.numpy as jnp

    from imas_ambix.latent import connectivity_boundary as cb

    # (a) static audit: no host contour / label / argwhere machinery imported
    tree = ast.parse(inspect.getsource(cb))
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            imported += [base] + [f"{base}.{a.name}" for a in node.names]
    banned = ("contourpy", "matplotlib", "skimage", "scipy", "ndimage")
    bad_import = [i for i in imported if any(b in i for b in banned)]
    calls = {
        n.func.attr
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
    }
    bad_calls = [c for c in ("argwhere", "label", "contour") if c in calls]

    # (b) vmap parity + throughput on one shot's slices sharing a grid
    same = [ov for ov in overlays if ov[2] is overlays[0][2]] if overlays else []
    parity_ok = None
    batched_ms = per_slice_ms = None
    n_batch = 0
    if len(same) >= 2:
        grid = same[0][2]
        psis = jnp.asarray(np.stack([ov[1] for ov in same]))
        axes = jnp.asarray(np.array([ov[5] for ov in same]))
        n_batch = int(psis.shape[0])
        out = cb.boundary_read_batch(psis, grid, axes)
        out["psi_bnd"].block_until_ready()
        # per-slice reference
        ref = [cb.boundary_read(ov[1], grid, ov[5], lcfs_norm=0.999) for ov in same]
        gpu_bnd = np.asarray(out["psi_bnd"])
        ref_bnd = np.array([r.psi_bnd for r in ref])
        parity_ok = bool(
            np.allclose(gpu_bnd, ref_bnd, atol=1e-9, rtol=0, equal_nan=True)
        )
        # throughput (compiled)
        t0 = time.perf_counter()
        for _ in range(3):
            cb.boundary_read_batch(psis, grid, axes)["psi_bnd"].block_until_ready()
        batched_ms = 1e3 * (time.perf_counter() - t0) / (3 * n_batch)
        cb.boundary_read(same[0][1], grid, same[0][5])  # warm compile
        t0 = time.perf_counter()
        for ov in same:
            cb.boundary_read(ov[1], grid, ov[5])
        per_slice_ms = 1e3 * (time.perf_counter() - t0) / len(same)

    tb2 = (not bad_import) and (not bad_calls) and (parity_ok is not False)
    return {
        "verdict": "PASS" if tb2 else "FAIL",
        "banned_imports": bad_import,
        "banned_calls": bad_calls,
        "device": str(jax.devices()[0]),
        "vmap_parity": parity_ok,
        "n_batch": n_batch,
        "batched_ms_per_slice": batched_ms,
        "loop_ms_per_slice": per_slice_ms,
    }


# ---------------------------------------------------------------------------
# continuity through the marginal transition (T-B3)
# ---------------------------------------------------------------------------


def _sweep_field(amp, *, nr=121, nz=161):
    """Plasma O-point + a growing lower blob (amp) that pulls an X-point in.

    amp small → single O-point, wall-limited (no in-vessel binding X); amp large →
    a saddle appears in-vessel and the plasma diverts.  The classify-first read
    (innermost in-vessel X flux, else limiter) steps when the X is first detected;
    the connectivity read never decides which surface bounds, so it is smooth.
    """
    from imas_ambix.latent.topology import _inside_polygon

    rg = np.linspace(0.25, 1.75, nr)
    zg = np.linspace(-1.15, 1.15, nz)
    rr, zz = np.meshgrid(rg, zg)
    s0, s1 = 0.34, 0.26
    plasma = np.exp(-(((rr - 1.0) ** 2 + (zz - 0.1) ** 2) / s0**2))
    lower = np.exp(-(((rr - 1.0) ** 2 + (zz + 0.7) ** 2) / s1**2))
    psi = plasma + amp * lower
    lr = np.array([0.3, 1.7, 1.7, 0.3, 0.3])
    lz = np.array([-1.05, -1.05, 1.05, 1.05, -1.05])
    inside = _inside_polygon(rr.ravel(), zz.ravel(), lr, lz).reshape(nz, nr)

    class _G:
        pass

    g = _G()
    g.rg, g.zg = rg, zg
    g.nr, g.nz = nr, nz
    g.dr = float(rg[1] - rg[0])
    g.dz = float(zg[1] - zg[0])
    g.inside_limiter = inside
    g.limiter_r, g.limiter_z = lr, lz
    return psi, g, (1.0, 0.1), lr, lz


def _classify_first_psi_bnd(psi, rg, zg, axis, lr, lz):
    """The classify-first boundary flux: innermost in-vessel X flux, else limiter."""
    from imas_ambix.latent.topology import (
        _bilerp,
        boundary_flux,
        find_critical_points,
        magnetic_axis,
    )

    cp = find_critical_points(psi, rg, zg)
    ax = magnetic_axis(psi, rg, zg, limiter_r=lr, limiter_z=lz, cp=cp)
    apsi = _bilerp(psi, rg, zg, axis[0], axis[1])
    bf = boundary_flux(cp, ax, apsi, limiter_r=lr, limiter_z=lz)
    if bf is None:
        lp = np.array(
            [
                _bilerp(psi, rg, zg, float(a), float(b))
                for a, b in zip(lr, lz, strict=True)
            ]
        )
        bf = float(lp[int(np.argmin(np.abs(lp - apsi)))])
    n_x_invessel = int(
        _inside_count(cp.x_points, lr, lz) if cp.x_points.shape[0] else 0
    )
    return bf - apsi, n_x_invessel  # relative to axis


def _inside_count(pts, lr, lz):
    from imas_ambix.latent.topology import _inside_polygon

    return int(_inside_polygon(pts[:, 0], pts[:, 1], lr, lz).sum())


def _tb3():
    from imas_ambix.latent.connectivity_boundary import boundary_read

    amps = np.linspace(0.0, 0.9, 31)
    conn, clf, nx = [], [], []
    for a in amps:
        psi, g, axis, lr, lz = _sweep_field(a)
        b = boundary_read(psi, g, axis, lcfs_norm=0.999)
        conn.append(b.psi_bnd - b.psi_axis)
        c, n = _classify_first_psi_bnd(psi, g.rg, g.zg, axis, lr, lz)
        clf.append(c)
        nx.append(n)
    conn = np.array(conn)
    clf = np.array(clf)
    nx = np.array(nx)
    span = float(np.nanmax(np.abs(conn)))
    conn_step = float(np.max(np.abs(np.diff(conn)))) / span
    clf_step = float(np.max(np.abs(np.diff(clf)))) / span
    # continuity PASS: connectivity max relative step is small AND markedly below
    # classify-first's (which jumps where the in-vessel X count changes)
    x_changes = int(np.sum(np.diff(nx) != 0))
    tb3 = conn_step < 0.05 and (clf_step > 3.0 * conn_step) and x_changes >= 1
    return {
        "verdict": "PASS" if tb3 else "FAIL",
        "amps": amps.tolist(),
        "conn_psi_bnd_rel": conn.tolist(),
        "classify_psi_bnd_rel": clf.tolist(),
        "n_xpoint_invessel": nx.tolist(),
        "conn_max_rel_step": conn_step,
        "classify_max_rel_step": clf_step,
        "n_xpoint_transitions": x_changes,
    }


# ---------------------------------------------------------------------------
# classify-after nulls: axis (T-C1), X-point + class (T-C2), unified binding (T-C3)
# ---------------------------------------------------------------------------


def _tc1(rows):
    """Sub-grid axis position vs CPU ``magnetic_axis``."""
    vals = [r["axis_d_cm"] for r in rows if np.isfinite(r["axis_d_cm"])]
    n = len(vals)
    med = float(np.median(vals)) if vals else float("nan")
    p90 = float(np.percentile(vals, 90)) if vals else float("nan")
    ok = n > 0 and np.isfinite(med) and med <= AXIS_TOL_CM
    return {
        "verdict": "PASS" if ok else "FAIL",
        "n_axis": n,
        "axis_dmed_cm": med,
        "axis_dp90_cm": p90,
        "tol_cm": AXIS_TOL_CM,
    }


def _tc1_grad(overlays):
    """A finite ``jax.grad`` of the sub-grid axis R w.r.t. ψ on one real slice."""
    import jax
    import jax.numpy as jnp

    from imas_ambix.latent.connectivity_boundary import _densify_wall, boundary_read_jax
    from imas_ambix.worldmodel.equilibrium_labels import LCFS_ANGLES

    picks = [ov for ov in overlays if np.isfinite(ov[4].axis[0])]
    if not picks:
        return {"grad_finite": None, "grad_nonzero": None}
    _cls, psi, grid, _cpu, _gpu, centroid = picks[0]
    wr, wz = _densify_wall(grid)
    rg = jnp.asarray(np.asarray(grid.rg, dtype=np.float64))
    zg = jnp.asarray(np.asarray(grid.zg, dtype=np.float64))
    inside = jnp.asarray(np.asarray(grid.inside_limiter, dtype=bool))
    ang = jnp.asarray(np.asarray(LCFS_ANGLES, dtype=np.float64))

    def axis_r(p):
        out = boundary_read_jax(
            p,
            rg,
            zg,
            inside,
            jnp.asarray(float(centroid[0])),
            jnp.asarray(float(centroid[1])),
            96,
            18,
            512,
            ang,
            jnp.asarray(1.0),
            jnp.asarray(wr),
            jnp.asarray(wz),
        )
        return out["axis_r"]

    g = np.asarray(jax.grad(axis_r)(jnp.asarray(np.asarray(psi, dtype=np.float64))))
    return {
        "grad_finite": bool(np.all(np.isfinite(g))),
        "grad_nonzero": bool(np.any(g != 0.0)),
    }


def _tc2(rows):
    """X-point positions + diverted class vs the CPU read (on the diverted subset)."""
    n = len(rows)
    div = [r for r in rows if r["diverted"]]
    agree = sum(1 for r in rows if bool(r["dev_is_diverted"]) == bool(r["diverted"]))
    class_acc = agree / n if n else float("nan")
    xd = [r["xset_d_cm"] for r in div if np.isfinite(r["xset_d_cm"])]
    x_med = float(np.median(xd)) if xd else float("nan")
    x_p90 = float(np.percentile(xd, 90)) if xd else float("nan")
    # softness: the disagreeing slices should carry a small |class_margin|
    disagree = [
        abs(r["class_margin"])
        for r in rows
        if bool(r["dev_is_diverted"]) != bool(r["diverted"])
    ]
    soft = float(np.median(disagree)) if disagree else 0.0
    xset_ok = (not xd) or x_med <= XSET_TOL_CM
    ok = np.isfinite(class_acc) and class_acc >= CLASS_ACC_TOL and xset_ok
    return {
        "verdict": "PASS" if ok else "FAIL",
        "n_slices": n,
        "n_diverted": len(div),
        "class_accuracy": class_acc,
        "n_class_disagree": len(disagree),
        "disagree_abs_margin_median": soft,
        "xset_dmed_cm": x_med,
        "xset_dp90_cm": x_p90,
        "n_xset_matched": len(xd),
        "tol_cm": XSET_TOL_CM,
        "class_acc_tol": CLASS_ACC_TOL,
    }


def _tc3(rows):
    """Unified binding: diverted ψ_bnd residual closed, limited not regressed."""
    div = [r for r in rows if r["diverted"]]
    lim = [r for r in rows if not r["diverted"]]

    def _med(subset, key):
        v = [r[key] for r in subset]
        return float(np.median(v)) if v else float("nan")

    def _signed(subset):
        v = [
            (r["gpu_psi_bnd"] - r["cpu_psi_bnd"])
            / abs(r["cpu_psi_bnd"] - r["psi_axis"])
            for r in subset
            if abs(r["cpu_psi_bnd"] - r["psi_axis"]) > 1e-12
        ]
        return float(np.median(v)) if v else float("nan")

    div_dpsi = _med(div, "dpsi_frac")
    lim_dpsi = _med(lim, "dpsi_frac")
    div_radii = _med(div, "radii_dmed_cm")
    lim_radii = _med(lim, "radii_dmed_cm")
    diverted_ok = np.isfinite(div_dpsi) and div_dpsi <= DIVERTED_PSI_BND_TOL
    limited_ok = np.isfinite(lim_radii) and lim_radii <= LCFS_TOL_CM
    ok = diverted_ok and limited_ok
    return {
        "verdict": "PASS" if ok else "FAIL",
        "diverted_dpsi_frac_median": div_dpsi,
        "diverted_dpsi_tol": DIVERTED_PSI_BND_TOL,
        "limited_dpsi_frac_median": lim_dpsi,
        "diverted_radii_dmed_cm": div_radii,
        "limited_radii_dmed_cm": lim_radii,
        "diverted_signed_bias_frac": _signed(div),
        "limited_signed_bias_frac": _signed(lim),
        "n_diverted": len(div),
        "n_limited": len(lim),
    }


# ---------------------------------------------------------------------------
# imas-ink poloidal cross-section (device read vs host read)
# ---------------------------------------------------------------------------


def _machine_geometry_from_grid(grid):
    """MachineGeometry (wall + coil boxes) from an EquilibriumGrid alone."""
    from imas_ink._types import CoilRect, MachineGeometry

    lr = np.asarray(grid.limiter_r, dtype=np.float64)
    lz = np.asarray(grid.limiter_z, dtype=np.float64)
    clip = np.column_stack([np.append(lr, lr[0]), np.append(lz, lz[0])])
    rects = [
        CoilRect(r=r0, z=z0, width=r1 - r0, height=z1 - z0, name=str(i))
        for i, (r0, r1, z0, z1) in enumerate(np.asarray(grid.conductor_rects))
    ]
    return MachineGeometry(
        wall_r=lr,
        wall_z=lz,
        coil_rects=rects,
        wall_clip_vertices=clip,
        wall_units=[(lr, lz)],
    )


def _dense_ring(psi, grid, centroid, *, n_ang=180, lcfs_norm=0.999):
    """Device LCFS as a dense (R, Z) polygon: ray-cast at ``n_ang`` fine angles."""
    import jax.numpy as jnp

    from imas_ambix.latent.connectivity_boundary import boundary_read_jax

    ang = np.linspace(0.0, 2.0 * np.pi, n_ang, endpoint=False)
    out = boundary_read_jax(
        jnp.asarray(np.asarray(psi, dtype=np.float64)),
        jnp.asarray(np.asarray(grid.rg, dtype=np.float64)),
        jnp.asarray(np.asarray(grid.zg, dtype=np.float64)),
        jnp.asarray(np.asarray(grid.inside_limiter, dtype=bool)),
        jnp.asarray(float(centroid[0])),
        jnp.asarray(float(centroid[1])),
        96,
        18,
        512,
        jnp.asarray(ang),
        jnp.asarray(float(lcfs_norm)),
    )
    rad = np.asarray(out["radii"], dtype=np.float64)
    ok = np.isfinite(rad)
    rr = centroid[0] + rad[ok] * np.cos(ang[ok])
    zz = centroid[1] + rad[ok] * np.sin(ang[ok])
    return np.column_stack([rr, zz]) if ok.any() else None


def _magnetic_axis(psi, grid, ring, psi_bnd):
    """The magnetic axis = the flux extremum INSIDE the boundary ring (the O-point).

    NOT the current centroid — on a Shafranov-shifted equilibrium the axis sits
    outboard of the centroid, so plotting the centroid on the flux surfaces would
    be misleading.  The confined side is the ψ direction away from ψ_bnd; the axis
    is the ring-interior cell furthest along it.
    """
    from imas_ambix.latent.topology import _inside_polygon

    rr, zz = np.meshgrid(grid.rg, grid.zg)
    inside = _inside_polygon(rr.ravel(), zz.ravel(), ring[:, 0], ring[:, 1]).reshape(
        psi.shape
    )
    sign = np.sign(np.nanmax(psi[inside]) - psi_bnd) if inside.any() else 1.0
    score = np.where(inside, psi * sign, -np.inf)
    iz, ir = np.unravel_index(int(np.argmax(score)), psi.shape)
    return float(grid.rg[ir]), float(grid.zg[iz])


def _ink_slice(psi, grid, axis_rz, psi_axis, psi_bnd, ring, ip, time_s):
    """Build an imas-ink EquilibriumSlice for one boundary read on ``psi``.

    ``axis_rz`` is the MAGNETIC AXIS (O-point, flux extremum inside the ring), used
    both for imas-ink's confined-contour classification and the plotted axis marker.
    """
    from imas_ink._types import EquilibriumSlice

    return EquilibriumSlice(
        psi_2d=np.ascontiguousarray(np.asarray(psi, dtype=np.float64).T),  # (nR, nZ)
        r_grid=np.asarray(grid.rg, dtype=np.float64),
        z_grid=np.asarray(grid.zg, dtype=np.float64),
        psi_axis=float(psi_axis),
        psi_boundary=float(psi_bnd),
        r_axis=float(axis_rz[0]),
        z_axis=float(axis_rz[1]),
        ip=float(ip),
        time=float(time_s),
        converged=True,
        x_points=[],
        boundary_r=None if ring is None else np.asarray(ring)[:, 0],
        boundary_z=None if ring is None else np.asarray(ring)[:, 1],
    )


def _ink_cross_section(overlays):
    """imas-ink poloidal cross-section: device flood-fill LCFS (primary) with the
    host ``lcfs_contour`` ring as the faint reference underlay, on a limited AND a
    diverted held-out slice."""
    from imas_ink.figures import equilibrium_figure_mpl

    want = {}
    for ov in overlays:
        want.setdefault(ov[0], ov)
    panels = [want[k] for k in ("limited", "diverted") if k in want]
    if not panels:
        return None
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    tiles = []
    for cls, psi, grid, cpu, gpu, centroid in panels:
        geom = _machine_geometry_from_grid(grid)
        ring = _dense_ring(psi, grid, centroid, lcfs_norm=1.0)
        # magnetic axis (O-point), NOT the current centroid, for the marker + the
        # confined-contour classification
        o_dev = (
            _magnetic_axis(psi, grid, ring, gpu.psi_bnd)
            if ring is not None
            else centroid
        )
        o_host = _magnetic_axis(psi, grid, cpu.ring, cpu.psi_bnd)
        dev = _ink_slice(psi, grid, o_dev, gpu.psi_axis, gpu.psi_bnd, ring, 0.0, 0.0)
        host = _ink_slice(
            psi, grid, o_host, gpu.psi_axis, cpu.psi_bnd, cpu.ring, 0.0, 0.0
        )
        fig, ax = equilibrium_figure_mpl(
            dev,
            geom,
            reference_slice=host,
            reference_name="host lcfs_contour",
            figsize=(4.4, 6.4),
            show_probes=False,
            show_flux_loops=False,
        )
        ax.set_title(
            f"{cls}\ndevice ψ_bnd {gpu.psi_bnd:.4f} · host {cpu.psi_bnd:.4f}",
            fontsize=9,
        )
        fig.canvas.draw()
        tiles.append(np.asarray(fig.canvas.buffer_rgba()))
        plt.close(fig)
    fig, axes = plt.subplots(1, len(tiles), figsize=(4.6 * len(tiles), 6.6))
    axes = np.atleast_1d(axes)
    for ax, img in zip(axes, tiles, strict=True):
        ax.imshow(img)
        ax.axis("off")
    fig.suptitle(
        "imas-ink cross-section — device connectivity flood-fill LCFS (blue) vs "
        "host lcfs_contour ring (faint sienna)\none code path, limited & diverted",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = FIG_DIR / "overlay_ink.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return str(out)


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------


def _figures(tb1, tb3, overlays):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rows = tb1["rows"]

    # (1) reproduction scatter + radii-Δ histogram, coloured by class
    if rows:
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))
        for cls, col in [(True, "#c73"), (False, "#268")]:
            sub = [r for r in rows if r["diverted"] == cls]
            if not sub:
                continue
            x = [r["cpu_psi_bnd"] - r["psi_axis"] for r in sub]
            y = [r["gpu_psi_bnd"] - r["psi_axis"] for r in sub]
            a1.scatter(
                x,
                y,
                s=18,
                alpha=0.7,
                color=col,
                label=f"{'diverted' if cls else 'limited'} (n={len(sub)})",
            )
        lims = [
            min(a1.get_xlim()[0], a1.get_ylim()[0]),
            max(a1.get_xlim()[1], a1.get_ylim()[1]),
        ]
        a1.plot(lims, lims, "k--", lw=0.8, label="y = x")
        a1.set_xlabel("host lcfs_contour  ψ_bnd − ψ_axis [Wb]")
        a1.set_ylabel("device flood-fill  ψ_bnd − ψ_axis [Wb]")
        _dpm = tb1["dpsi_frac_median"] * 100
        a1.set_title(f"T-B1 — ψ_bnd reproduction (median Δ {_dpm:.2f}% of span)")
        a1.legend(fontsize=8)

        dr = [r["radii_dmed_cm"] for r in rows]
        a2.hist(dr, bins=18, color="#484", alpha=0.8)
        a2.axvline(
            LCFS_TOL_CM, color="k", ls="--", lw=1, label=f"tol {LCFS_TOL_CM:.0f} cm"
        )
        a2.axvline(
            tb1["radii_dmed_cm_overall"],
            color="#484",
            ls=":",
            lw=1.4,
            label=f"median {tb1['radii_dmed_cm_overall']:.2f} cm",
        )
        a2.set_xlabel("LCFS-radius agreement, host vs device [cm]")
        a2.set_ylabel("slices")
        a2.set_title(f"T-B1 — LCFS radii ({tb1['verdict']})")
        a2.legend(fontsize=8)
        fig.suptitle(
            f"Device connectivity boundary reproduces host lcfs_contour — "
            f"{tb1['verdict']} ({tb1['n_slices']} slices: {tb1['n_diverted']} "
            f"diverted, {tb1['n_limited']} limited)"
        )
        fig.tight_layout()
        fig.savefig(FIG_DIR / "reproduction.png", dpi=130)
        plt.close(fig)

    # (2) boundary overlay on a limited AND a diverted slice
    want = {}
    for ov in overlays:
        want.setdefault(ov[0], ov)
    panels = [want[k] for k in ("limited", "diverted") if k in want]
    if panels:
        fig, axs = plt.subplots(
            1, len(panels), figsize=(5.6 * len(panels), 5.4), squeeze=False
        )
        ang = np.asarray(
            __import__(
                "imas_ambix.worldmodel.equilibrium_labels", fromlist=["LCFS_ANGLES"]
            ).LCFS_ANGLES
        )
        for ax, (cls, psi, grid, cpu, gpu, centroid) in zip(
            axs[0], panels, strict=False
        ):
            lv = np.linspace(np.min(psi), np.max(psi), 22)
            ax.contour(
                grid.mesh_r, grid.mesh_z, psi, levels=lv, colors="#bbb", linewidths=0.5
            )
            ax.plot(grid.limiter_r, grid.limiter_z, "k-", lw=1.2, label="wall")
            ax.plot(
                cpu.ring[:, 0],
                cpu.ring[:, 1],
                "-",
                color="#268",
                lw=1.8,
                label="host contourpy ring",
            )
            rr = np.asarray(gpu.radii)
            ok = np.isfinite(rr)
            gx = centroid[0] + rr[ok] * np.cos(ang[ok])
            gy = centroid[1] + rr[ok] * np.sin(ang[ok])
            ax.plot(
                gx,
                gy,
                "o",
                color="#c73",
                ms=8,
                mew=1.5,
                mfc="none",
                label="device ray-cast radii",
            )
            ax.plot(centroid[0], centroid[1], "k+", ms=10)
            ax.set_aspect("equal")
            ax.set_xlabel("R [m]")
            ax.set_ylabel("Z [m]")
            ax.set_title(
                f"{cls}  —  ψ_bnd host {cpu.psi_bnd:.4f} / device {gpu.psi_bnd:.4f}"
            )
            ax.legend(fontsize=8, loc="upper right")
        fig.suptitle(
            "Connectivity boundary: host contourpy ring vs "
            "device flood-fill ray-cast radii"
        )
        fig.tight_layout()
        fig.savefig(FIG_DIR / "overlay.png", dpi=130)
        plt.close(fig)

    # (3) continuity through the marginal transition
    amps = np.array(tb3["amps"])
    conn = np.array(tb3["conn_psi_bnd_rel"])
    clf = np.array(tb3["classify_psi_bnd_rel"])
    nx = np.array(tb3["n_xpoint_invessel"])
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(
        amps,
        conn,
        "-o",
        color="#268",
        ms=3,
        label="connectivity ψ_bnd − ψ_axis (this read)",
    )
    ax.plot(
        amps,
        clf,
        "-s",
        color="#c73",
        ms=3,
        label="classify-first (innermost X, else limiter)",
    )
    trans = np.where(np.diff(nx) != 0)[0]
    for t in trans:
        ax.axvline(0.5 * (amps[t] + amps[t + 1]), color="k", ls=":", lw=1)
    if len(trans):
        ax.axvline(
            0.5 * (amps[trans[0]] + amps[trans[0] + 1]),
            color="k",
            ls=":",
            lw=1,
            label="in-vessel X-point appears",
        )
    ax.set_xlabel("lower-blob amplitude  (limited → diverted)")
    ax.set_ylabel("ψ_bnd − ψ_axis [Wb]")
    _cs = tb3["conn_max_rel_step"] * 100
    _fs = tb3["classify_max_rel_step"] * 100
    ax.set_title(
        f"T-B3 — continuity through the marginal transition ({tb3['verdict']})\n"
        f"connectivity max step {_cs:.1f}% vs classify-first {_fs:.1f}% of span"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "continuity.png", dpi=130)
    plt.close(fig)

    # (4) imas-ink poloidal cross-section (device vs host boundary)
    try:
        _ink_cross_section(overlays)
    except Exception as exc:  # noqa: BLE001 — the ink stack is an optional sibling
        logger.warning("imas-ink cross-section skipped: %s", exc)

    # (5) §3 classify-after: axis agreement + axis/X overlay
    try:
        _figures_classify_after(rows, overlays)
    except Exception as exc:  # noqa: BLE001
        logger.warning("classify-after figures skipped: %s", exc)


def _figures_classify_after(rows, overlays):
    """Axis sub-grid agreement (device vs CPU) and an axis/X-point overlay."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # (a) axis agreement: device vs CPU magnetic_axis (R and Z) + distance hist
    ax_rows = [
        r for r in rows if r["cpu_axis_r"] is not None and np.isfinite(r["axis_d_cm"])
    ]
    if ax_rows:
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))
        for comp, col, lab in [("r", "#268", "R"), ("z", "#c73", "Z")]:
            cx = [r[f"cpu_axis_{comp}"] for r in ax_rows]
            dy = [r[f"dev_axis_{comp}"] for r in ax_rows]
            a1.scatter(cx, dy, s=20, alpha=0.7, color=col, label=f"{lab} [m]")
        lims = [
            min(a1.get_xlim()[0], a1.get_ylim()[0]),
            max(a1.get_xlim()[1], a1.get_ylim()[1]),
        ]
        a1.plot(lims, lims, "k--", lw=0.8, label="y = x")
        a1.set_xlabel("host magnetic_axis  [m]")
        a1.set_ylabel("device sub-grid axis  [m]")
        a1.set_title("T-C1 — sub-grid axis (O-point) vs host")
        a1.legend(fontsize=8)
        d = [r["axis_d_cm"] for r in ax_rows]
        a2.hist(d, bins=16, color="#484", alpha=0.8)
        a2.axvline(
            AXIS_TOL_CM, color="k", ls="--", lw=1, label=f"tol {AXIS_TOL_CM:.0f} cm"
        )
        a2.axvline(
            float(np.median(d)),
            color="#484",
            ls=":",
            lw=1.4,
            label=f"median {np.median(d):.2f} cm",
        )
        a2.set_xlabel("axis position agreement, host vs device [cm]")
        a2.set_ylabel("slices")
        a2.set_title("T-C1 — axis agreement")
        a2.legend(fontsize=8)
        fig.suptitle(
            "Sub-grid magnetic axis (nova stencil + biquadratic subnull) vs host"
        )
        fig.tight_layout()
        fig.savefig(FIG_DIR / "axis_agreement.png", dpi=130)
        plt.close(fig)

    # (b) axis + X-point overlay on a diverted slice (O-point axis, NOT centroid)
    want = {}
    for ov in overlays:
        want.setdefault(ov[0], ov)
    panel = want.get("diverted") or want.get("limited")
    if panel is not None:
        from imas_ambix.latent.topology import find_critical_points, magnetic_axis

        cls, psi, grid, cpu, gpu, centroid = panel
        ring = _dense_ring(psi, grid, centroid, lcfs_norm=1.0)
        cp = find_critical_points(psi, grid.rg, grid.zg)
        cpu_ax = magnetic_axis(
            psi,
            grid.rg,
            grid.zg,
            limiter_r=grid.limiter_r,
            limiter_z=grid.limiter_z,
            cp=cp,
        )
        fig, axp = plt.subplots(figsize=(5.4, 6.6))
        lv = np.linspace(np.min(psi), np.max(psi), 26)
        axp.contour(
            grid.mesh_r, grid.mesh_z, psi, levels=lv, colors="#ccc", linewidths=0.4
        )
        axp.plot(grid.limiter_r, grid.limiter_z, "k-", lw=1.2, label="wall")
        if ring is not None:
            axp.plot(
                ring[:, 0], ring[:, 1], "-", color="#268", lw=1.6, label="device LCFS"
            )
        axp.plot(
            gpu.axis[0],
            gpu.axis[1],
            "o",
            color="#268",
            ms=11,
            mfc="none",
            mew=2,
            label="device axis (O)",
        )
        if cpu_ax is not None:
            axp.plot(
                cpu_ax[0],
                cpu_ax[1],
                "+",
                color="#111",
                ms=12,
                mew=1.6,
                label="host axis",
            )
        xs = np.asarray(gpu.xset)
        okx = np.isfinite(xs).all(axis=1)
        if okx.any():
            axp.plot(
                xs[okx, 0],
                xs[okx, 1],
                "X",
                color="#c73",
                ms=13,
                mew=2,
                label="device X-point",
            )
        if cp.x_points.shape[0]:
            axp.plot(
                cp.x_points[:, 0],
                cp.x_points[:, 1],
                "1",
                color="#711",
                ms=13,
                mew=1.6,
                label="host X-points",
            )
        axp.set_aspect("equal")
        axp.set_xlabel("R [m]")
        axp.set_ylabel("Z [m]")
        axp.set_title(f"{cls} — sub-grid axis + X-point (classify-after)")
        axp.legend(fontsize=8, loc="upper right")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "axis_xpoint_overlay.png", dpi=140)
        plt.close(fig)


# ---------------------------------------------------------------------------
# §4 machine-agnostic wall gates: T-D1 (multi-wall), T-D2 (thin tile),
# T-D3 (g_wall exactness + no regression), T-D4 (AUG-class raster)
# ---------------------------------------------------------------------------

# tolerances
WALL_PSI_BND_TOL = 0.03  # device ψ_bnd vs the true (fine-sampled) tangency, / span
GWALL_SWAP_TOL = 0.02  # g_wall vs bilerp ψ_bnd on MAST-36 (no regression), / span
GWALL_CONSISTENCY_TOL = 0.01  # grid-kernel reconstruction vs basis psi_tot, / span


class _WallGrid:
    """Duck-type grid for a synthetic multi-unit wall.

    Carries exactly the surface the connectivity read consumes — the raster
    ``inside_limiter`` mask and the per-unit ``wall_r``/``wall_z`` node string —
    both built from an arbitrary ``wall_mask.WallUnit`` list through the SAME code
    path an :class:`EquilibriumGrid` uses.  No solver machinery: the field is
    supplied analytically and the exact node flux is evaluated in closed form.
    """

    def __init__(self, rg, zg, units):
        from imas_ambix.latent.wall_mask import build_wall_mask, densify_units

        self.rg = np.asarray(rg, dtype=np.float64)
        self.zg = np.asarray(zg, dtype=np.float64)
        self.nr, self.nz = self.rg.size, self.zg.size
        self.dr = float(self.rg[1] - self.rg[0])
        self.dz = float(self.zg[1] - self.zg[0])
        self.inside_limiter, self.wall_diagnostics = build_wall_mask(
            self.rg, self.zg, units
        )
        self.wall_r, self.wall_z, self.wall_unit_id = densify_units(
            units, 0.5 * min(self.dr, self.dz)
        )
        vessels = [u for u in units if u.kind == "vessel"]
        self.limiter_r = vessels[0].r if vessels else np.asarray([1.0e30])
        self.limiter_z = vessels[0].z if vessels else np.asarray([1.0e30])
        self.units = units


def _gauss_field(centers, amps, sig):
    """Closed-form ψ = Σ aₖ·exp(−((r−rₖ)²+(z−zₖ)²)/σ²) — evaluable exactly anywhere."""

    def fn(r, z):
        r = np.asarray(r, dtype=np.float64)
        z = np.asarray(z, dtype=np.float64)
        out = np.zeros(np.broadcast_shapes(r.shape, z.shape), dtype=np.float64)
        for (r0, z0), a in zip(centers, amps, strict=True):
            out = out + a * np.exp(-(((r - r0) ** 2 + (z - z0) ** 2) / sig**2))
        return out

    return fn


def _psi_out(psi2d, psi_axis):
    """Domain-edge flux extreme (the read's ψ_out) for a gridded field."""
    edge = np.concatenate([psi2d[0, :], psi2d[-1, :], psi2d[:, 0], psi2d[:, -1]])
    return float(edge[np.argmax(np.abs(edge - psi_axis))])


def _tangency_ref(fn, units, psi_axis, psi_out):
    """True limited-tangency flux + binding unit from a fine surface sampling.

    Densifies every unit's surface an order finer than the grid, reads the exact
    field, and takes the confined-most node (nearest ψ_N to the axis) — the true
    wall tangency, independent of the grid/flood.  Returns ``(psi_bnd, unit_id)``.
    """
    from imas_ambix.latent.wall_mask import densify_units

    span = psi_out - psi_axis
    wr, wz, uid = densify_units(units, spacing=0.002)  # 2 mm — well sub-cell
    u = (fn(wr, wz) - psi_axis) / span
    k = int(np.argmin(u))
    return float(fn(np.array([wr[k]]), np.array([wz[k]]))[0]), int(uid[k])


def _saddle_ref(fn, rg, zg, limiter_r, limiter_z):
    """True diverted separatrix flux — confined-most in-vessel saddle (fine grid)."""
    from imas_ambix.latent.topology import _inside_polygon, find_critical_points

    rf = np.linspace(rg[0], rg[-1], int(2.5 * rg.size))
    zf = np.linspace(zg[0], zg[-1], int(2.5 * zg.size))
    psif = _gridfield(fn, rf, zf)
    cp = find_critical_points(psif, rf, zf)
    if cp.x_points.shape[0] == 0:
        return None
    ins = _inside_polygon(cp.x_points[:, 0], cp.x_points[:, 1], limiter_r, limiter_z)
    if not ins.any():
        return None
    xpsi = np.asarray(cp.x_psi)[ins]  # in-vessel saddle fluxes
    # confined-most (binding) separatrix = the in-vessel saddle flux nearest the
    # axis O-point flux (the innermost X); no tile dependence — a field feature.
    o_ins = _inside_polygon(cp.o_points[:, 0], cp.o_points[:, 1], limiter_r, limiter_z)
    if o_ins.any():
        psi_ax = float(
            np.asarray(cp.o_psi)[o_ins][np.argmax(np.abs(np.asarray(cp.o_psi)[o_ins]))]
        )
        k = int(np.argmin(np.abs(xpsi - psi_ax)))
    else:
        k = int(np.argmax(np.abs(xpsi)))
    return float(xpsi[k])


def _gridfield(fn, rg, zg):
    rr, zz = np.meshgrid(rg, zg)
    return fn(rr, zz)


def _read_wall(fn, grid, axis, lcfs_norm=1.0):
    """Run the device connectivity read on a synthetic field with EXACT node flux."""
    from imas_ambix.latent.connectivity_boundary import boundary_read

    psi2d = _gridfield(fn, grid.rg, grid.zg)
    wall_psi = fn(grid.wall_r, grid.wall_z)
    return boundary_read(psi2d, grid, axis, lcfs_norm=lcfs_norm, wall_psi=wall_psi)


def _gwall_leg(grid, payload_item, inv, psi, centroid, cpu, span):
    """T-D3 held-out leg: swap the bilerp tangency for the exact g_wall GEMM.

    Reads the same field (``inv.psi_tot``) with the exact campaign ``g_wall`` node
    flux instead of bilinear, and reports the ψ_bnd/radii shift plus the
    grid-kernel-vs-basis consistency residual (so the swap is an honest A/B).
    """
    from imas_ambix.latent.connectivity_boundary import boundary_read

    out = {
        "dpsi_frac": float("nan"),
        "radii_dmed_cm": float("nan"),
        "vs_bilerp_dpsi_frac": float("nan"),
        "consistency_frac": float("nan"),
    }
    if getattr(grid, "_coil_packs", None) is None or abs(span) < 1e-12:
        return out
    try:
        i_cell = np.asarray(inv.i_cell, dtype=np.float64)
        i_pf = np.asarray(payload_item.i_pf, dtype=np.float64)
        wall_psi = grid.wall_flux(i_pf, i_cell)
        gw = boundary_read(psi, grid, centroid, lcfs_norm=1.0, wall_psi=wall_psi)
        bl = boundary_read(psi, grid, centroid, lcfs_norm=1.0)  # bilerp baseline
        out["dpsi_frac"] = abs(gw.psi_bnd - cpu.psi_bnd) / abs(span)
        out["vs_bilerp_dpsi_frac"] = abs(gw.psi_bnd - bl.psi_bnd) / abs(span)
        okg = np.isfinite(cpu.radii) & np.isfinite(gw.radii)
        if okg.any():
            out["radii_dmed_cm"] = float(
                np.median(100.0 * np.abs(gw.radii[okg] - cpu.radii[okg]))
            )
        recon = (grid.coil_psi(i_pf) + grid.plasma_grid_psi(i_cell)).reshape(
            grid.nz, grid.nr
        )
        out["consistency_frac"] = float(np.max(np.abs(recon - psi))) / abs(span)
    except Exception as exc:  # noqa: BLE001 — record nothing, keep sweeping
        logger.warning("g_wall leg skipped: %s", exc)
    return out


# --- synthetic wall fixtures ------------------------------------------------


def _vessel_box(rlo, rhi, zlo, zhi):
    from imas_ambix.latent.wall_mask import vessel_unit

    return vessel_unit(
        np.array([rlo, rhi, rhi, rlo, rlo]),
        np.array([zlo, zlo, zhi, zhi, zlo]),
        name="vessel",
    )


def _tile(rc, zc, hw, hh, *, name="tile", closed=True):
    from imas_ambix.latent.wall_mask import material_unit

    return material_unit(
        np.array([rc - hw, rc + hw, rc + hw, rc - hw, rc - hw]),
        np.array([zc - hh, zc - hh, zc + hh, zc + hh, zc - hh]),
        closed=closed,
        name=name,
    )


def _td1():
    """Multi-wall correctness: single-loop, multi-polygon, movable — lim + div."""
    cases = []

    def limited_case(name, units, centre, axis):
        grid = _WallGrid(
            np.linspace(0.25, 1.75, 121), np.linspace(-1.15, 1.15, 161), units
        )
        fn = _gauss_field([centre], [1.0], 0.30)
        b = _read_wall(fn, grid, axis)
        psi_ax = float(b.psi_axis)
        psi2d = _gridfield(fn, grid.rg, grid.zg)
        p_out = _psi_out(psi2d, psi_ax)
        ref_bnd, ref_unit = _tangency_ref(fn, units, psi_ax, p_out)
        span = abs(p_out - psi_ax)
        err = abs(b.psi_bnd - ref_bnd) / span
        return {
            "case": name,
            "class": "limited",
            "found": bool(b.found),
            "dpsi_frac": float(err),
            "ref_unit": int(ref_unit),
            "pass": bool(b.found and err < WALL_PSI_BND_TOL),
        }

    def diverted_case(name, units, axis):
        grid = _WallGrid(
            np.linspace(0.25, 1.75, 121), np.linspace(-1.2, 1.2, 161), units
        )
        fn = _gauss_field([(1.0, 0.25), (1.0, -0.7)], [1.0, 0.9], 0.28)
        b = _read_wall(fn, grid, axis)
        ref = _saddle_ref(fn, grid.rg, grid.zg, grid.limiter_r, grid.limiter_z)
        if ref is None:
            return {
                "case": name,
                "class": "diverted",
                "pass": False,
                "found": bool(b.found),
            }
        psi2d = _gridfield(fn, grid.rg, grid.zg)
        span = abs(_psi_out(psi2d, float(b.psi_axis)) - float(b.psi_axis))
        err = abs(b.psi_bnd - ref) / span
        return {
            "case": name,
            "class": "diverted",
            "found": bool(b.found),
            "dpsi_frac": float(err),
            "dev_is_diverted": bool(b.is_diverted),
            "pass": bool(b.found and err < WALL_PSI_BND_TOL),
        }

    # (1) single closed vessel loop (MAST-like)
    box = [_vessel_box(0.3, 1.7, -1.05, 1.05)]
    cases.append(limited_case("single_loop", box, (1.0, 0.1), (1.0, 0.1)))
    cases.append(diverted_case("single_loop", box, (1.0, 0.25)))
    # (2) multi-polygon: vessel + two discrete limiter tiles (AUG-like)
    multi = [
        _vessel_box(0.3, 1.7, -1.05, 1.05),
        _tile(1.42, 0.0, 0.05, 0.35, name="inner_limiter"),
        _tile(0.55, -0.6, 0.10, 0.06, name="lower_tile"),
    ]
    # plasma leans on the inner limiter (nearest surface at R≈1.37)
    cases.append(limited_case("multi_polygon", multi, (1.0, 0.0), (1.0, 0.0)))
    # (3) time-varying wall: a movable limiter steps inward between two pulses
    for pos, tag in ((1.55, "wall_far"), (1.30, "wall_near")):
        mv = [
            _vessel_box(0.3, 1.7, -1.05, 1.05),
            _tile(pos, 0.0, 0.04, 0.4, name="movable_limiter"),
        ]
        cases.append(limited_case(f"movable_{tag}", mv, (1.0, 0.0), (1.0, 0.0)))

    n_pass = sum(c.get("pass", False) for c in cases)
    verdict = "PASS" if n_pass == len(cases) else "FAIL"
    return {"verdict": verdict, "n_pass": n_pass, "n_cases": len(cases), "cases": cases}


def _td2():
    """Thin tile (t < Δ): the warning FIRES and ψ_bnd still lands on the true flux."""
    rg = np.linspace(0.25, 1.75, 121)
    zg = np.linspace(-1.15, 1.15, 161)
    delta = min(float(rg[1] - rg[0]), float(zg[1] - zg[0]))
    thin = delta / 3.0  # blade thinner than the grid
    units = [
        _vessel_box(0.3, 1.7, -1.05, 1.05),
        _tile(1.30, 0.0, thin, 0.4, name="thin_blade"),
    ]
    grid = _WallGrid(rg, zg, units)
    warned = [d for d in grid.wall_diagnostics if d.kind == "thin_unit"]
    fn = _gauss_field([(1.0, 0.0)], [1.0], 0.30)
    b = _read_wall(fn, grid, (1.0, 0.0))
    psi_ax = float(b.psi_axis)
    psi2d = _gridfield(fn, grid.rg, grid.zg)
    p_out = _psi_out(psi2d, psi_ax)
    ref_bnd, ref_unit = _tangency_ref(fn, units, psi_ax, p_out)
    span = abs(p_out - psi_ax)
    err = abs(b.psi_bnd - ref_bnd) / span
    ok = bool(len(warned) >= 1 and b.found and ref_unit == 1 and err < WALL_PSI_BND_TOL)
    return {
        "verdict": "PASS" if ok else "FAIL",
        "warning_fired": bool(len(warned) >= 1),
        "thickness_proxy_cm": (
            100.0 * warned[0].detail["thickness_proxy_m"] if warned else None
        ),
        "delta_cm": 100.0 * delta,
        "binds_on_blade": bool(ref_unit == 1),
        "dpsi_frac": float(err),
        "found": bool(b.found),
    }


def _td3_exactness():
    """g_wall node flux is exact (machine-eps) where bilerp carries an O(Δ²) floor."""
    import jax.numpy as jnp

    from imas_ambix.latent.connectivity_boundary import _bilerp
    from imas_ambix.latent.gs_solve import EquilibriumGrid

    rg = np.linspace(0.25, 1.75, 65)
    zg = np.linspace(-1.05, 1.05, 97)
    lr = np.array([0.4, 1.6, 1.6, 0.4, 0.4])
    lz = np.array([-0.9, -0.9, 0.9, 0.9, -0.9])
    grid = EquilibriumGrid(
        rg=rg,
        zg=zg,
        limiter_r=lr,
        limiter_z=lz,
        coil_psi_columns=np.zeros((rg.size * zg.size, 0)),
        r0=1.0,
    )
    cr = grid.flat_r[grid.cells]
    cz = grid.flat_z[grid.cells]
    i_cell = np.exp(-(((cr - 1.0) ** 2 + cz**2) / 0.3**2)) * 1.0e3
    exact = grid.wall_flux(np.zeros(0), i_cell)
    direct = grid.wall_greens()["g_cells"] @ i_cell
    psi2d = grid.plasma_grid_psi(i_cell).reshape(grid.nz, grid.nr)
    rgj, zgj, psij = jnp.asarray(rg), jnp.asarray(zg), jnp.asarray(psi2d)
    bilerp = np.array(
        [
            float(_bilerp(psij, rgj, zgj, float(r), float(z)))
            for r, z in zip(grid.wall_r, grid.wall_z, strict=True)
        ]
    )
    span = float(psi2d.max() - psi2d.min())
    exact_err = float(np.max(np.abs(exact - direct))) / span
    bilerp_err = float(np.max(np.abs(bilerp - exact))) / span
    ok = bool(exact_err < 1e-12 and bilerp_err > 1e-5)
    return {
        "verdict": "PASS" if ok else "FAIL",
        "exact_vs_direct_frac": exact_err,
        "bilerp_floor_frac": bilerp_err,
        "ratio": (bilerp_err / exact_err) if exact_err > 0 else float("inf"),
    }


def _td3_mast(rows):
    """No-regression: swapping bilerp→g_wall on MAST-36 does not move ψ_bnd/radii."""

    def med(key):
        v = [r[key] for r in rows if np.isfinite(r.get(key, np.nan))]
        return float(np.median(v)) if v else float("nan")

    def p90(key):
        v = [r[key] for r in rows if np.isfinite(r.get(key, np.nan))]
        return float(np.percentile(v, 90)) if v else float("nan")

    swap = med("gwall_vs_bilerp_dpsi_frac")
    swap90 = p90("gwall_vs_bilerp_dpsi_frac")
    cons = med("gwall_consistency_frac")
    cons90 = p90("gwall_consistency_frac")
    gw_dpsi = med("gwall_dpsi_frac")
    n = sum(1 for r in rows if np.isfinite(r.get("gwall_vs_bilerp_dpsi_frac", np.nan)))
    ok = bool(
        n > 0
        and np.isfinite(swap90)
        and swap90 < GWALL_SWAP_TOL
        and cons90 < GWALL_CONSISTENCY_TOL
    )
    return {
        "verdict": "PASS" if ok else "FAIL",
        "n_slices": n,
        "swap_dpsi_frac_med": swap,
        "swap_dpsi_frac_p90": swap90,
        "consistency_frac_med": cons,
        "consistency_frac_p90": cons90,
        "gwall_vs_host_dpsi_frac_med": gw_dpsi,
    }


def _aug_class_wall():
    """An AUG-class multi-unit wall: a vessel + ~29 discrete tiles (thin inner-
    column heat shields, outer limiter tiles, up/down divertor tiles) + one open
    limiter blade.  "AUG-class" = the many-discrete-unit STRUCTURE (a real AUG
    wall is ~29 units); clean per-unit R,Z for the real AUG geometry was not in
    the imas-efit tree, so this fixture exercises the SAME build_wall_mask path a
    real 29-unit wall would.  Dimensions are chosen divertor-capable (a lower
    divertor the single-null separatrix escapes to) so BOTH a limited and a
    diverted field bind through the multi-unit raster."""
    from imas_ambix.latent.wall_mask import material_unit

    units = [
        _vessel_box(0.35, 1.75, -1.15, 1.15),  # vessel (occupiable)
    ]
    # inner-column heat-shield tiles (sub-grid thin, stacked) — fire the thin warn
    for k, zc in enumerate(np.linspace(-0.75, 0.75, 8)):
        units.append(_tile(0.40, zc, 0.004, 0.09, name=f"innercol_{k}"))
    # outer discrete limiter tiles
    for k, zc in enumerate(np.linspace(-0.6, 0.6, 7)):
        units.append(_tile(1.70, zc, 0.03, 0.09, name=f"outer_{k}"))
    # upper divertor tiles
    for k, rc in enumerate(np.linspace(0.7, 1.3, 6)):
        units.append(_tile(rc, 1.08, 0.05, 0.04, name=f"updiv_{k}"))
    # lower divertor tiles (the single-null separatrix escapes here)
    for k, rc in enumerate(np.linspace(0.7, 1.3, 6)):
        units.append(_tile(rc, -1.08, 0.05, 0.04, name=f"lowdiv_{k}"))
    # one OPEN line primitive (a limiter blade)
    units.append(
        material_unit(
            np.array([1.55, 1.55]), np.array([-0.2, 0.2]), closed=False, name="blade"
        )
    )
    return units


def _td4():
    """AUG-class 29-unit raster: no special-casing, thin warnings, correct binding."""
    units = _aug_class_wall()
    rg = np.linspace(0.35, 1.75, 141)
    zg = np.linspace(-1.15, 1.15, 181)
    grid = _WallGrid(rg, zg, units)
    thin = [d for d in grid.wall_diagnostics if d.kind == "thin_unit"]
    n_units = len(units)

    # limited: a centred plasma leans on the nearest discrete unit
    fn_lim = _gauss_field([(1.0, 0.0)], [1.0], 0.32)
    b_lim = _read_wall(fn_lim, grid, (1.0, 0.0))
    psi_ax = float(b_lim.psi_axis)
    p_out = _psi_out(_gridfield(fn_lim, rg, zg), psi_ax)
    ref_bnd, ref_unit = _tangency_ref(fn_lim, units, psi_ax, p_out)
    span_l = abs(p_out - psi_ax)
    err_lim = abs(b_lim.psi_bnd - ref_bnd) / span_l

    # diverted: main blob + a lower-divertor blob (single-null) — the separatrix
    # escapes to the lower divertor tiles, so the X-saddle is the binding surface
    fn_div = _gauss_field([(1.0, 0.25), (1.0, -0.7)], [1.0, 0.9], 0.28)
    b_div = _read_wall(fn_div, grid, (1.0, 0.25))
    ref_saddle = _saddle_ref(fn_div, rg, zg, grid.limiter_r, grid.limiter_z)
    div_ok = False
    err_div = float("nan")
    if ref_saddle is not None and b_div.found:
        span_d = abs(
            _psi_out(_gridfield(fn_div, rg, zg), float(b_div.psi_axis))
            - float(b_div.psi_axis)
        )
        err_div = abs(b_div.psi_bnd - ref_saddle) / span_d
        div_ok = err_div < WALL_PSI_BND_TOL

    ok = bool(
        len(thin) >= 1
        and b_lim.found
        and err_lim < WALL_PSI_BND_TOL
        and b_div.is_diverted
        and div_ok
    )
    return {
        "verdict": "PASS" if ok else "FAIL",
        "n_units": n_units,
        "n_thin_warnings": len(thin),
        "limited_found": bool(b_lim.found),
        "limited_dpsi_frac": float(err_lim),
        "limited_binds_unit": int(ref_unit),
        "diverted_found": bool(b_div.found),
        "diverted_is_diverted": bool(b_div.is_diverted),
        "diverted_dpsi_frac": float(err_div),
        "fixture": "AUG-class synthetic multi-unit wall (29 units)",
    }


def _ring_from_radii(read, angles):
    """(ring_r, ring_z) from a read's LCFS radii about its axis (NaN-safe)."""
    ar, az = read.axis
    if not (np.isfinite(ar) and np.isfinite(az)):
        return None, None
    rad = np.asarray(read.radii, dtype=np.float64)
    rr = ar + rad * np.cos(angles)
    zz = az + rad * np.sin(angles)
    return rr, zz


def _panel(ax, grid, units, read, title):
    """One boundary-read panel: occupiable mask + unit polygons + LCFS ring."""
    from imas_ambix.worldmodel.equilibrium_labels import LCFS_ANGLES

    ang = np.asarray(LCFS_ANGLES)
    ax.imshow(
        grid.inside_limiter,
        origin="lower",
        extent=[grid.rg[0], grid.rg[-1], grid.zg[0], grid.zg[-1]],
        cmap="Greys",
        alpha=0.25,
        aspect="auto",
    )
    for u in units:
        r = np.append(u.r, u.r[0]) if u.closed else u.r
        z = np.append(u.z, u.z[0]) if u.closed else u.z
        ax.plot(r, z, "-", lw=1.0, color=("k" if u.kind == "vessel" else "C3"))
    rr, zz = _ring_from_radii(read, ang)
    if rr is not None:
        ax.plot(np.append(rr, rr[0]), np.append(zz, zz[0]), "-", color="C0", lw=1.6)
    if np.isfinite(read.axis[0]):
        ax.plot(read.axis[0], read.axis[1], "x", color="C0", ms=7)
    cls = "diverted" if read.is_diverted else "limited"
    ax.set_title(f"{title}\n({cls})", fontsize=8)
    ax.set_xlabel("R [m]", fontsize=7)
    ax.set_ylabel("Z [m]", fontsize=7)
    ax.tick_params(labelsize=6)


def _figures_wall():
    """Boundary overlays: single-loop, multi-polygon, thin-tile, movable, AUG-class."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    delta_grid = (np.linspace(0.25, 1.75, 121), np.linspace(-1.15, 1.15, 161))
    dr = float(delta_grid[0][1] - delta_grid[0][0])
    thin = (min(dr, float(delta_grid[1][1] - delta_grid[1][0]))) / 3.0
    panels = []
    # single loop (limited)
    u = [_vessel_box(0.3, 1.7, -1.05, 1.05)]
    g = _WallGrid(*delta_grid, u)
    panels.append(
        (
            g,
            u,
            _read_wall(_gauss_field([(1.0, 0.1)], [1.0], 0.30), g, (1.0, 0.1)),
            "single loop",
        )
    )
    # multi-polygon (two discrete limiters)
    u = [
        _vessel_box(0.3, 1.7, -1.05, 1.05),
        _tile(1.42, 0.0, 0.05, 0.35, name="inner_limiter"),
        _tile(0.55, -0.6, 0.10, 0.06, name="lower_tile"),
    ]
    g = _WallGrid(*delta_grid, u)
    panels.append(
        (
            g,
            u,
            _read_wall(_gauss_field([(1.0, 0.0)], [1.0], 0.30), g, (1.0, 0.0)),
            "multi-polygon",
        )
    )
    # thin tile (t < Δ)
    u = [
        _vessel_box(0.3, 1.7, -1.05, 1.05),
        _tile(1.30, 0.0, thin, 0.4, name="thin_blade"),
    ]
    g = _WallGrid(*delta_grid, u)
    panels.append(
        (
            g,
            u,
            _read_wall(_gauss_field([(1.0, 0.0)], [1.0], 0.30), g, (1.0, 0.0)),
            "thin tile (t<Δ)",
        )
    )
    # movable limiter (near)
    u = [
        _vessel_box(0.3, 1.7, -1.05, 1.05),
        _tile(1.30, 0.0, 0.04, 0.4, name="movable"),
    ]
    g = _WallGrid(*delta_grid, u)
    panels.append(
        (
            g,
            u,
            _read_wall(_gauss_field([(1.0, 0.0)], [1.0], 0.30), g, (1.0, 0.0)),
            "movable limiter",
        )
    )
    # AUG-class limited + diverted
    au = _aug_class_wall()
    ag = _WallGrid(np.linspace(0.35, 1.75, 141), np.linspace(-1.15, 1.15, 181), au)
    panels.append(
        (
            ag,
            au,
            _read_wall(_gauss_field([(1.0, 0.0)], [1.0], 0.32), ag, (1.0, 0.0)),
            "AUG-class (29 units)",
        )
    )
    panels.append(
        (
            ag,
            au,
            _read_wall(
                _gauss_field([(1.0, 0.25), (1.0, -0.7)], [1.0, 0.9], 0.28),
                ag,
                (1.0, 0.25),
            ),
            "AUG-class single-null",
        )
    )

    n = len(panels)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(3.2 * ((n + 1) // 2), 6.4))
    for axp, (g, u, b, title) in zip(np.ravel(axes), panels, strict=False):
        _panel(axp, g, u, b, title)
    for axp in np.ravel(axes)[n:]:
        axp.axis("off")
    fig.suptitle(
        "§4 machine-agnostic wall — one read, arbitrary raster mask (tiles as holes)",
        fontsize=10,
    )
    fig.tight_layout()
    out = FIG_DIR / "wall_multi.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return str(out)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shots", type=str, default="")
    ap.add_argument("--n-shots", type=int, default=5)
    ap.add_argument("--max-slices", type=int, default=6)
    ap.add_argument("--min-ip-ka", type=float, default=200.0)
    ap.add_argument("--nr", type=int, default=65)
    ap.add_argument("--nz", type=int, default=97)
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument(
        "--td-only",
        action="store_true",
        help="run only the §4 wall gates (synthetic; no held-out data)",
    )
    ap.add_argument(
        "--ink-shot",
        type=int,
        default=0,
        help="regenerate ONLY the imas-ink cross-section from this one shot, then exit",
    )
    ap.add_argument(
        "--out",
        type=str,
        default="imas_ambix/latent/artifacts/patch_gate/connectivity_boundary_gate-v0.json",
    )
    return ap


def main() -> int:
    args = build_parser().parse_args()
    from imas_ambix.eval import prediction_bar as pbar

    if args.ink_shot:
        _rows, overlays = _slice_rows(
            int(args.ink_shot),
            nr=args.nr,
            nz=args.nz,
            max_slices=args.max_slices,
            min_ip_ka=args.min_ip_ka,
        )
        out = _ink_cross_section(overlays)
        logger.info("wrote imas-ink cross-section: %s", out)
        return 0

    if args.td_only:
        td1 = _td1()
        td2 = _td2()
        td3x = _td3_exactness()
        td4 = _td4()
        for tag, r in [
            ("T-D1", td1),
            ("T-D2", td2),
            ("T-D3-exact", td3x),
            ("T-D4", td4),
        ]:
            logger.info("%s: %s", tag, json.dumps(r, indent=2))
        if not args.no_figures:
            logger.info("wall figures: %s", _figures_wall())
        logger.info(
            "VERDICTS (§4 synthetic): %s",
            {
                "T-D1": td1["verdict"],
                "T-D2": td2["verdict"],
                "T-D3-exact": td3x["verdict"],
                "T-D4": td4["verdict"],
            },
        )
        return 0

    if args.shots:
        shots = [int(s) for s in args.shots.split(",") if s.strip()]
    else:
        manifest = pbar.load_locked_manifest()
        shots = list(pbar.held_out_shot_ids(manifest))
        if args.n_shots > 0:
            shots = shots[: args.n_shots]
    logger.info(
        "connectivity-boundary gate over %d held-out shots: %s", len(shots), shots
    )

    tb1, overlays = _tb1(
        shots,
        nr=args.nr,
        nz=args.nz,
        max_slices=args.max_slices,
        min_ip_ka=args.min_ip_ka,
    )
    logger.info(
        "T-B1: %s", json.dumps({k: v for k, v in tb1.items() if k != "rows"}, indent=2)
    )
    tb2 = _tb2(overlays)
    logger.info("T-B2: %s", json.dumps(tb2, indent=2))
    tb3 = _tb3()
    logger.info(
        "T-B3: %s",
        json.dumps(
            {
                k: v
                for k, v in tb3.items()
                if k
                not in (
                    "amps",
                    "conn_psi_bnd_rel",
                    "classify_psi_bnd_rel",
                    "n_xpoint_invessel",
                )
            },
            indent=2,
        ),
    )

    # --- §3 classify-after gates -------------------------------------------
    rows = tb1["rows"]
    tc1 = _tc1(rows)
    tc1["grad"] = _tc1_grad(overlays)
    logger.info("T-C1: %s", json.dumps(tc1, indent=2))
    tc2 = _tc2(rows)
    logger.info("T-C2: %s", json.dumps(tc2, indent=2))
    tc3 = _tc3(rows)
    logger.info("T-C3: %s", json.dumps(tc3, indent=2))

    # --- §4 machine-agnostic wall gates ------------------------------------
    td1 = _td1()
    logger.info("T-D1: %s", json.dumps(td1, indent=2))
    td2 = _td2()
    logger.info("T-D2: %s", json.dumps(td2, indent=2))
    td3 = {"exactness": _td3_exactness(), "no_regression_mast": _td3_mast(rows)}
    td3["verdict"] = (
        "PASS"
        if td3["exactness"]["verdict"] == "PASS"
        and td3["no_regression_mast"]["verdict"] == "PASS"
        else "FAIL"
    )
    logger.info("T-D3: %s", json.dumps(td3, indent=2))
    td4 = _td4()
    logger.info("T-D4: %s", json.dumps(td4, indent=2))

    if not args.no_figures:
        _figures(tb1, tb3, overlays)
        try:
            _figures_wall()
        except Exception as exc:  # noqa: BLE001
            logger.warning("wall figures skipped: %s", exc)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "schema": "connectivity-boundary-gate-v0",
                "shots": shots,
                "tb1_reproduction": {k: v for k, v in tb1.items() if k != "rows"},
                "tb1_rows": tb1["rows"],
                "tb2_on_device": tb2,
                "tb3_continuity": tb3,
                "tc1_axis": tc1,
                "tc2_xpoint_class": tc2,
                "tc3_unified_binding": tc3,
                "td1_multi_wall": td1,
                "td2_thin_tile": td2,
                "td3_wall_flux": td3,
                "td4_aug_raster": td4,
            },
            indent=2,
        )
    )
    logger.info("wrote %s", out)
    verdicts = {
        "T-B1": tb1["verdict"],
        "T-B2": tb2["verdict"],
        "T-B3": tb3["verdict"],
        "T-C1": tc1["verdict"],
        "T-C2": tc2["verdict"],
        "T-C3": tc3["verdict"],
        "T-D1": td1["verdict"],
        "T-D2": td2["verdict"],
        "T-D3": td3["verdict"],
        "T-D4": td4["verdict"],
    }
    logger.info("VERDICTS: %s", verdicts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
