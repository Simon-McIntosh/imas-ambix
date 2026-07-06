#!/usr/bin/env python
"""Illustrations for the constrained current-moment boundary read.

Compares a low-order current-moment boundary read against the free-current
patch inverse, both scored against the firewalled EFIT referee on the
held-out set.  Regenerates four PNGs from the precomputed gate artifacts:

1. ``thread1-skills-comparison.png`` — grouped bars: X-point-set skill, LCFS
   skill, and LCFS median offset (cm, all + flattop), free-current vs moment.
2. ``thread1-lcfs-per-angle.png`` — per-poloidal-angle median |model - ref|
   LCFS radial offset (cm) for both reads (the honest edge finding).
3. ``thread1-flux-comparison.png`` — imas-ink ψ(R,Z) overlay for one flat-top
   held-out slice: our current-moment ψ (contours + read boundary / X-point)
   with the firewalled EFIT reference underlaid at matched absolute levels.
4. ``thread1-current-distribution.png`` — free-current (lumpy) vs moment
   (smooth) per-cell current on the grid for the same slice.

Conventions (verified this session — see scripts/patch_flux_map_report.py):
internal flux is TOTAL poloidal Φ = 2πR·A_φ [Wb]; MAST sign psi_axis >
psi_boundary; EFIT ``efm`` ψ is Wb/rad and is multiplied by 2π to compare.
The firewalled EFIT map is read ONLY inside ``evaluator_context()``.

Artifacts read (patch_gate/):
  boundary_read_grid_baseline_arrays.npz + boundary_read_grid_eval.json
      — free-current baseline (points[0]).
  boundary_read_moment-o3_arrays.npz + boundary_read_moment-o3.json
      — current-moment (polynomial order 3, centroid axis).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("plot_boundary_moment_figures")

ARTIFACTS = Path("imas_ambix/latent/artifacts/patch_gate")
FIGURES = Path("docs/figures/equilibrium-topology-fidelity")

# free-current baseline (grid sweep, points[0]) + its 14-D arrays
BASELINE_JSON = ARTIFACTS / "boundary_read_grid_eval.json"
BASELINE_NPZ = ARTIFACTS / "boundary_read_grid_baseline_arrays.npz"
# current-moment read (polynomial order 3, centroid axis)
MOMENT_JSON = ARTIFACTS / "boundary_read_moment-o3.json"
MOMENT_NPZ = ARTIFACTS / "boundary_read_moment-o3_arrays.npz"

# 8 fixed LCFS poloidal angles (equilibrium_labels.LCFS_ANGLES = 2*pi*k/8)
ANGLE_DEG = np.array([0, 45, 90, 135, 180, 225, 270, 315])
ANGLE_TAG = [
    "0°\nout",
    "45°",
    "90°\ntop",
    "135°",
    "180°\nin",
    "225°",
    "270°\nbot",
    "315°",
]

# consistent colours for the two reads
C_FREE = "#c04a2e"  # free-current baseline (warm)
C_MOM = "#2a6f97"  # current-moment (cool)


# --------------------------------------------------------------------------
def _load_baseline_point() -> dict:
    data = json.loads(BASELINE_JSON.read_text())
    return data["points"][0]  # the free-current baseline


def _load_moment() -> dict:
    return json.loads(MOMENT_JSON.read_text())


# --------------------------------------------------------------------------
# Figure 1 — headline skills comparison
# --------------------------------------------------------------------------
def _skill_panel(ax, key, title, base, mom, *, fmt="{:.2f}", unit=""):
    """One grouped free-vs-moment bar pair on its own y-scale."""
    bv, mv = base[key], mom[key]
    b = ax.bar([0], [bv], 0.62, color=C_FREE)
    m = ax.bar([1], [mv], 0.62, color=C_MOM)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["free-\ncurrent", "current-\nmoment"], fontsize=9)
    ax.set_title(title, fontsize=11)
    for rect in list(b) + list(m):
        h = rect.get_height()
        ax.annotate(
            (fmt + unit).format(h),
            (rect.get_x() + rect.get_width() / 2, h),
            textcoords="offset points",
            xytext=(0, -13 if h < 0 else 4),
            va="top" if h < 0 else "bottom",
            ha="center",
            fontsize=10,
            fontweight="bold",
            color="0.1",
        )
    return bv, mv


def fig_skills_comparison(out: Path) -> None:
    base = _load_baseline_point()
    mom = _load_moment()

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 5.0))

    specs = [
        (
            "xpoint_set_skill",
            "X-point-set skill\n(>0 beats train-mean)",
            "{:.2f}",
            "",
            True,
        ),
        ("lcfs_skill", "LCFS skill\n(>0 beats train-mean)", "{:.2f}", "", True),
    ]
    for ax, (key, title, fmt, unit, _is_skill) in zip(axes[:2], specs, strict=True):
        bv, mv = _skill_panel(ax, key, title, base, mom, fmt=fmt, unit=unit)
        ax.axhline(0.0, color="0.4", lw=1.0, ls="--")
        d = mv - bv
        ax.annotate(
            f"Δ = +{d:.2f}",
            (0.5, 0.0),
            xycoords=("data", "data"),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=10.5,
            color=C_MOM,
            fontweight="bold",
        )
        # headroom so labels/deltas clear the frame
        lo = min(bv, mv, 0.0)
        ax.set_ylim(lo * 1.32, max(0.0, bv, mv) + abs(lo) * 0.28 + 0.05)
    axes[0].set_ylabel("skill  (higher is better)")

    # ---- panel 3: LCFS median radial offset [cm] (grouped, lower better) ---
    ax_off = axes[2]
    keys = ["lcfs_offset_median_cm_all", "lcfs_offset_median_cm_flattop"]
    olabels = ["all\nslices", "flat-top\nonly"]
    bo = [base[k] for k in keys]
    mo = [mom[k] for k in keys]
    xo = np.arange(len(keys))
    w = 0.36
    b3 = ax_off.bar(xo - w / 2, bo, w, label="free-current", color=C_FREE)
    b4 = ax_off.bar(xo + w / 2, mo, w, label="current-moment", color=C_MOM)
    ax_off.set_xticks(xo)
    ax_off.set_xticklabels(olabels, fontsize=9)
    ax_off.set_ylabel("median LCFS radial offset [cm]")
    ax_off.set_title("LCFS boundary offset vs EFIT\n(lower is better)", fontsize=11)
    ax_off.legend(loc="upper right", framealpha=0.9, fontsize=8.5)
    ax_off.set_ylim(0, max(bo + mo) * 1.24)
    for rect in list(b3) + list(b4):
        h = rect.get_height()
        ax_off.annotate(
            f"{h:.1f}",
            (rect.get_x() + rect.get_width() / 2, h),
            textcoords="offset points",
            xytext=(0, 3),
            ha="center",
            fontsize=9,
            fontweight="bold",
        )
    for xi, bv, mv in zip(xo, bo, mo, strict=True):
        ax_off.annotate(
            f"Δ = {mv - bv:.1f} cm",
            (xi, max(bv, mv)),
            textcoords="offset points",
            xytext=(0, 18),
            ha="center",
            fontsize=9.5,
            color=C_MOM,
            fontweight="bold",
        )

    fig.suptitle(
        "Constrained current-moment vs free-current boundary read "
        "(held-out, EFIT-scored)\n"
        "improved on every metric, but still a partial recovery — all skills "
        "stay negative",
        fontsize=12.5,
        y=1.03,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", out)


# --------------------------------------------------------------------------
# Figure 2 — per-angle LCFS offset (the honest edge)
# --------------------------------------------------------------------------
def fig_lcfs_per_angle(out: Path) -> None:
    zb = np.load(BASELINE_NPZ)
    zm = np.load(MOMENT_NPZ)
    # columns 6:14 = lcfs_r_0..7 (radii at the 8 fixed poloidal angles)
    eb = np.abs(zb["model"][:, 6:14] - zb["ref"][:, 6:14]) * 100.0
    em = np.abs(zm["model"][:, 6:14] - zm["ref"][:, 6:14]) * 100.0
    med_b = np.nanmedian(eb, axis=0)
    med_m = np.nanmedian(em, axis=0)

    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    x = np.arange(8)
    w = 0.38
    ax.bar(x - w / 2, med_b, w, label="free-current", color=C_FREE)
    ax.bar(x + w / 2, med_m, w, label="current-moment", color=C_MOM)
    for xi, v in zip(x - w / 2, med_b, strict=True):
        ax.annotate(
            f"{v:.0f}",
            (xi, v),
            textcoords="offset points",
            xytext=(0, 2),
            ha="center",
            fontsize=7.5,
        )
    for xi, v in zip(x + w / 2, med_m, strict=True):
        ax.annotate(
            f"{v:.0f}",
            (xi, v),
            textcoords="offset points",
            xytext=(0, 2),
            ha="center",
            fontsize=7.5,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(ANGLE_TAG, fontsize=9)
    ax.set_xlabel("LCFS poloidal angle about the magnetic axis")
    ax.set_ylabel("median |model − ref| radial offset  [cm]")
    ax.set_title(
        "Per-angle LCFS radial error — the moment read helps the "
        "midplane, not the upper/lower legs"
    )
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    cap = (
        "The current-moment read shrinks the boundary offset most strongly at "
        "the top/bottom (90°, 270°) and near the outboard/inboard midplane, but "
        "the largest residuals persist at the upper/lower off-axis angles "
        "(135°, 225° ≈ 39–40 cm, essentially unchanged; inboard 180° even "
        "worsens slightly) — the divertor-leg / X-point region a smooth "
        "low-order current model cannot resolve."
    )
    fig.text(0.5, -0.06, cap, ha="center", va="top", fontsize=8.5, wrap=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", out)


# --------------------------------------------------------------------------
# Figures 3 & 4 — flux overlay + current distribution for one flat-top slice
# --------------------------------------------------------------------------
def _pick_flattop_slice(preferred=(18502,)):
    """Return (shot, grid, basis, payloads, table, refs, flattop_pick).

    flattop_pick = (k, efit_dict) for the chosen flat-top held-out slice.
    Prefers the ``preferred`` shots, then the first held-out shot that snaps.
    """
    from patch_flux_map_report import select_slices
    from patch_gate_eval import shot_payloads

    from imas_ambix.latent.data import read_split_shot_lists

    _, held = read_split_shot_lists(40, 8)
    order = [s for s in preferred if s in held] + [
        s for s in held if s not in preferred
    ]
    for shot in order:
        try:
            payload = shot_payloads(
                shot,
                nr=65,
                nz=97,
                max_slices=20,
                min_ip_ka=300.0,
                split="eval",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("shot %s load failed: %s", shot, exc)
            continue
        if payload is None:
            continue
        payloads = payload["payloads"]
        picks = select_slices(payloads, shot)
        flat = [p for p in picks if p[0] == "flattop"]
        if not flat:
            continue
        _, k, efit = flat[0]
        logger.info("flux slice: shot %s k=%s t=%.3fs", shot, k, payloads[k].time_s)
        return (
            shot,
            payload["grid"],
            payload["basis"],
            payloads,
            payload["table"],
            payload["refs"],
            (k, efit),
        )
    return None


def fig_flux_and_current(out_flux: Path, out_curr: Path) -> None:
    import torch
    from patch_flux_map_report import (
        WINNER,
        _closed_contour_about,
        _machine_geometry,
    )
    from patch_gate_eval import geometry_target

    from imas_ambix.latent.boundary_moment import MomentFitConfig, fit_moment_currents
    from imas_ambix.latent.patch_inverse import InverseConfig, invert_slices

    picked = _pick_flattop_slice()
    if picked is None:
        logger.warning("no flat-top slice snapped; skipping flux + current figs")
        return
    shot, grid, basis, payloads, table, refs, (k, efit) = picked
    payload = payloads[k]

    # ---- current-moment ψ (polynomial order 3) ----------------------------
    cfg = MomentFitConfig(model="polynomial", order=3)
    mom = fit_moment_currents(basis, payload, cfg)
    psi_mom = basis.psi_grid_2d_np(mom.i_cell, payload.i_pf)  # (nz, nr) [Wb]
    target, psi_ax, psi_b = geometry_target(psi_mom, grid)
    our_axis = (float(target[0]), float(target[1]))

    geom = _machine_geometry(grid, table)
    our_lcfs = _closed_contour_about(grid.rg, grid.zg, psi_mom, psi_b, *our_axis)

    # -------- Figure 3: imas-ink flux overlay (fallback: matplotlib) -------
    # Written FIRST — it needs only the (fast) moment fit, so the required
    # flux figure lands before the slow free-current inverse below.
    _flux_overlay(
        out_flux,
        shot,
        payload,
        grid,
        geom,
        psi_mom,
        target,
        psi_ax,
        psi_b,
        our_axis,
        our_lcfs,
        efit,
    )

    # -------- Figure 4: current distribution (free vs moment) -------------
    # The free-current winner-config inverse (800 iters) is the slow step; run
    # it only after fig 3 is on disk.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _ = InverseConfig  # (imported for provenance; WINNER already an InverseConfig)
    inv_free = invert_slices(basis, [payload], WINNER, device=device)[0]
    _current_distribution(
        out_curr,
        shot,
        payload,
        basis,
        grid,
        inv_free.i_cell,
        mom.i_cell,
    )


def _flux_overlay(
    out,
    shot,
    payload,
    grid,
    geom,
    psi_mom,
    target,
    psi_ax,
    psi_b,
    our_axis,
    our_lcfs,
    efit,
):
    """imas-ink ψ(R,Z) overlay; matplotlib contour fallback if it fails."""
    try:
        from imas_ink.figures import equilibrium_figure_mpl
        from patch_flux_map_report import _efit_slice, _our_slice

        our_sl = _our_slice(
            psi_mom,
            grid,
            target,
            psi_ax,
            psi_b,
            payload.ip_amperes,
            payload.time_s,
            our_lcfs,
        )
        fig, _ax = equilibrium_figure_mpl(
            our_sl,
            geom,
            reference_slice=_efit_slice(efit),
            reference_name="EFIT",
            figsize=(5.4, 6.8),
            show_probes=False,
            show_flux_loops=False,
        )
        fig.suptitle(
            f"Current-moment ψ(R,Z) vs EFIT — shot {shot}  "
            f"t={payload.time_s:.3f}s (flat-top)\n"
            "solid: current-moment read  ·  faint: firewalled EFIT reference",
            fontsize=10.5,
        )
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("wrote %s (imas-ink)", out)
        return
    except Exception as exc:  # noqa: BLE001
        logger.warning("imas-ink overlay failed (%s); using matplotlib fallback", exc)

    # --- matplotlib fallback: matched absolute ψ levels, both x2pi-consistent

    fig, ax = plt.subplots(figsize=(5.6, 7.0))
    ax.plot(
        np.append(grid.limiter_r, grid.limiter_r[0]),
        np.append(grid.limiter_z, grid.limiter_z[0]),
        color="0.3",
        lw=1.3,
        label="limiter",
    )
    # matched normalised levels psi_N in {0.1..0.9} in each frame's own scale
    frac = np.linspace(0.1, 0.9, 5)
    our_levels = psi_ax + frac * (psi_b - psi_ax)
    ax.contour(
        grid.rg,
        grid.zg,
        psi_mom,
        levels=np.sort(our_levels),
        colors=C_MOM,
        linewidths=1.1,
    )
    ef_levels = efit["psi_axis"] + frac * (efit["psi_boundary"] - efit["psi_axis"])
    ax.contour(
        efit["rg"],
        efit["zg"],
        efit["psi_zr"],
        levels=np.sort(ef_levels),
        colors=C_FREE,
        linewidths=1.0,
        linestyles="--",
    )
    # LCFS (ψ_boundary) contours
    if our_lcfs is not None:
        ax.plot(
            our_lcfs[:, 0],
            our_lcfs[:, 1],
            color=C_MOM,
            lw=2.2,
            label="current-moment LCFS",
        )
    ax.plot(
        efit["lcfs_r"], efit["lcfs_z"], color=C_FREE, lw=1.8, ls="--", label="EFIT LCFS"
    )
    ax.plot(*our_axis, "*", color=C_MOM, ms=13, label="current-moment axis")
    ax.plot(efit["axis_r"], efit["axis_z"], "P", color=C_FREE, ms=10, label="EFIT axis")
    for slot in range(2):
        rr, zz = target[2 + 2 * slot], target[3 + 2 * slot]
        if np.isfinite(rr) and np.isfinite(zz):
            ax.plot(rr, zz, "x", color=C_MOM, ms=10, mew=2.2)
    ax.set_aspect("equal")
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.set_title(
        f"Current-moment ψ(R,Z) vs EFIT — shot {shot}  "
        f"t={payload.time_s:.3f}s (flat-top)\n"
        "blue: current-moment read  ·  dashed sienna: firewalled EFIT",
        fontsize=10,
    )
    ax.legend(loc="upper right", fontsize=7.5, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s (matplotlib fallback)", out)


def _current_distribution(out, shot, payload, basis, grid, i_free, i_mom):
    """Free-current (lumpy) vs moment (smooth) per-cell current on the grid."""
    r_cells = np.asarray(basis.r_cells.detach().cpu().numpy(), dtype=np.float64)
    z_cells = np.asarray(basis.z_cells.detach().cpu().numpy(), dtype=np.float64)
    i_free = np.asarray(i_free, dtype=np.float64)
    i_mom = np.asarray(i_mom, dtype=np.float64)

    # symmetric colour scale in kA/cell across both panels
    both = np.concatenate([i_free, i_mom]) / 1e3  # kA
    vmax = np.percentile(np.abs(both), 99.5)
    vmax = max(vmax, 1e-6)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 6.4), sharex=True, sharey=True)
    for ax, ic, title in (
        (axes[0], i_free / 1e3, "free-current inverse (lumpy)"),
        (axes[1], i_mom / 1e3, "current-moment fit (smooth, order 3)"),
    ):
        sc = ax.scatter(
            r_cells,
            z_cells,
            c=ic,
            s=14,
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            marker="s",
        )
        ax.plot(
            np.append(grid.limiter_r, grid.limiter_r[0]),
            np.append(grid.limiter_z, grid.limiter_z[0]),
            color="0.3",
            lw=1.2,
        )
        ax.set_aspect("equal")
        ax.set_xlabel("R [m]")
        ax.set_title(title, fontsize=10.5)
    axes[0].set_ylabel("Z [m]")
    cb = fig.colorbar(sc, ax=axes, fraction=0.046, pad=0.03)
    cb.set_label("per-cell plasma current  [kA]")
    fig.suptitle(
        f"Why the moment read is smoother — per-cell current, shot {shot} "
        f"t={payload.time_s:.3f}s\n"
        "same external magnetics, same total Ip; the moment basis suppresses "
        "the free inverse's cell-to-cell lumpiness",
        fontsize=11.5,
    )
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", out)


# --------------------------------------------------------------------------
def main() -> int:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig_skills_comparison(FIGURES / "thread1-skills-comparison.png")
    fig_lcfs_per_angle(FIGURES / "thread1-lcfs-per-angle.png")
    fig_flux_and_current(
        FIGURES / "thread1-flux-comparison.png",
        FIGURES / "thread1-current-distribution.png",
    )
    for p in sorted(FIGURES.glob("thread1-*.png")):
        kb = p.stat().st_size / 1024.0
        flag = "OK" if kb > 5 else "TOO-SMALL"
        logger.info("  %s  %.1f KB  %s", p.name, kb, flag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
