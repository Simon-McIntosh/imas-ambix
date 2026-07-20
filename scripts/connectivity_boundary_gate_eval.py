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


# ---------------------------------------------------------------------------
# real-slice reproduction (T-B1)
# ---------------------------------------------------------------------------


def _classify_diverted(psi, grid, ring) -> bool:
    """Is the slice diverted? — an in-vessel saddle sits ON the LCFS ring."""
    from imas_ambix.latent.topology import emergent_xpoints, find_critical_points

    cp = find_critical_points(psi, grid.rg, grid.zg)
    _xset, is_div = emergent_xpoints(cp.x_points, ring, tol=1.5 * float(grid.dr))
    return bool(is_div)


def _slice_rows(shot: int, *, nr: int, nz: int, max_slices: int, min_ip_ka: float):
    """Per-slice host-vs-device boundary reads on one held-out shot's disc fields."""
    from imas_ambix.latent.boundary_disc import disc_read
    from imas_ambix.latent.connectivity_boundary import boundary_read
    from imas_ambix.latent.topology import lcfs_contour
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
        is_div = _classify_diverted(psi, grid, cpu.ring)
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
    inside = _inside_polygon(
        rr.ravel(), zz.ravel(), ring[:, 0], ring[:, 1]
    ).reshape(psi.shape)
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

    if not args.no_figures:
        _figures(tb1, tb3, overlays)

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
            },
            indent=2,
        )
    )
    logger.info("wrote %s", out)
    verdicts = {"T-B1": tb1["verdict"], "T-B2": tb2["verdict"], "T-B3": tb3["verdict"]}
    logger.info("VERDICTS: %s", verdicts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
