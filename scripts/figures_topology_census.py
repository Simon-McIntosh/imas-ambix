"""Figures for the EFIT-scored real-topology validation campaign.

Renders three figure sets into ``docs/figures/connectivity-topology-reader/``:

* ``census_by_class_campaign.png`` — corpus census: slice counts per shot-number
  bin stacked by topology class, plus per-class totals.
* ``topology_class_overlays.png`` — one representative slice per class: EFIT ψ
  contours + EFIT LCFS/X-points against the device connectivity read.
* ``topology_residuals_by_class.png`` — per-class residual distributions for the
  EFIT-ψ read-reproduction leg and the disc-engine cross-validation leg.

Colors: fixed CVD-validated class palette (identity, never cycled).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("figures_topology_census")

from scripts.topology_census import CLASSES  # noqa: E402

FIG_DIR = Path("docs/figures/connectivity-topology-reader")
ART_DIR = Path("imas_ambix/latent/artifacts/patch_gate")

CLASS_COLOR = {
    "limited": "#009E73",
    "sn-lower": "#0072B2",
    "sn-upper": "#56B4E9",
    "connected-dn": "#D55E00",
    "marginal-dn": "#E69F00",
}
SCORED = list(CLASS_COLOR)


def census_figure(rows: np.ndarray) -> None:
    fig, (ax0, ax1) = plt.subplots(
        1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [2.2, 1.0]}
    )
    bins = np.arange(11500, 31001, 500)
    centers = 0.5 * (bins[:-1] + bins[1:])
    bottom = np.zeros(centers.size)
    for cname in SCORED:
        ci = CLASSES.index(cname)
        h, _ = np.histogram(rows["shot"][rows["cls"] == ci], bins=bins)
        ax0.bar(
            centers,
            h,
            width=440,
            bottom=bottom,
            color=CLASS_COLOR[cname],
            label=cname,
            edgecolor="white",
            linewidth=0.4,
        )
        bottom += h
    ax0.set_xlabel("shot number")
    ax0.set_ylabel("classified slices / 500-shot bin")
    ax0.set_title("MAST corpus topology census (EFIT-derived, valid slices)")
    ax0.legend(frameon=False, fontsize=8, ncol=2)
    ax0.spines[["top", "right"]].set_visible(False)

    counts = [int((rows["cls"] == CLASSES.index(c)).sum()) for c in SCORED]
    ypos = np.arange(len(SCORED))
    ax1.barh(ypos, counts, color=[CLASS_COLOR[c] for c in SCORED], height=0.62)
    ax1.set_yticks(ypos, SCORED, fontsize=9)
    ax1.invert_yaxis()
    for y, v in zip(ypos, counts, strict=True):
        ax1.text(v, y, f" {v:,}", va="center", fontsize=8)
    ax1.set_xlabel("slices")
    ax1.set_title("class totals")
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.set_xlim(0, max(counts) * 1.22)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "census_by_class_campaign.png", dpi=150)
    plt.close(fig)
    logger.info("wrote census_by_class_campaign.png")


def overlay_figure(selection: dict) -> None:
    """One representative slice per class: EFIT reference vs device read."""
    from imas_ambix.latent.connectivity_boundary import boundary_read  # noqa: PLC0415
    from imas_ambix.worldmodel.equilibrium_labels import LCFS_ANGLES  # noqa: PLC0415
    from scripts.topology_efit_read_eval import load_slice  # noqa: PLC0415

    fig, axes = plt.subplots(1, len(SCORED), figsize=(3.0 * len(SCORED), 5.4))
    for ax, cname in zip(np.atleast_1d(axes), SCORED, strict=False):
        pairs = selection.get(cname, [])
        if not pairs:
            ax.set_axis_off()
            continue
        shot, k = pairs[len(pairs) // 2]
        psi, grid, axis, lcfs, xpts = load_slice(int(shot), int(k))
        dev = boundary_read(psi, grid, axis, lcfs_norm=1.0)
        rr, zz = np.meshgrid(grid.rg, grid.zg)
        ax.contour(rr, zz, psi, levels=18, colors="0.82", linewidths=0.5)
        ax.plot(
            np.append(grid.limiter_r, grid.limiter_r[0]),
            np.append(grid.limiter_z, grid.limiter_z[0]),
            color="0.35",
            lw=1.0,
        )
        ax.plot(lcfs[:, 0], lcfs[:, 1], color="0.15", lw=1.6, label="EFIT LCFS")
        ang = np.asarray(LCFS_ANGLES)
        r = np.asarray(dev.radii)
        okm = np.isfinite(r)
        ax.plot(
            axis[0] + r[okm] * np.cos(ang[okm]),
            axis[1] + r[okm] * np.sin(ang[okm]),
            "o",
            ms=5,
            mfc="none",
            color=CLASS_COLOR[cname],
            label="device radii",
        )
        fx = xpts[np.isfinite(xpts).all(axis=1)]
        if fx.size:
            ax.plot(fx[:, 0], fx[:, 1], "x", ms=9, color="0.15", mew=1.8)
        dx = np.asarray(dev.xset)
        dx = dx[np.isfinite(dx).all(axis=1)]
        if dx.size:
            ax.plot(dx[:, 0], dx[:, 1], "+", ms=11, color=CLASS_COLOR[cname], mew=1.8)
        ax.plot(*axis, ".", ms=6, color="0.15")
        ax.set_title(f"{cname}\n{shot} @ k={k}", fontsize=9)
        ax.set_aspect("equal")
        ax.set_xlim(0.0, 2.0)
        ax.set_ylim(-2.0, 2.0)
        ax.set_xticks([0.5, 1.0, 1.5])
        if cname != SCORED[0]:
            ax.set_yticks([])
    axes[0].set_ylabel("Z [m]")
    axes[0].legend(frameon=False, fontsize=7, loc="upper right")
    fig.suptitle(
        "device connectivity read (marks) vs EFIT reconstruction (black) — "
        "EFIT ψ fed to both",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "topology_class_overlays.png", dpi=150)
    plt.close(fig)
    logger.info("wrote topology_class_overlays.png")


def residuals_figure(efit_art: dict, engine_rows: dict) -> None:
    metrics = [
        ("radii_dmed_cm", "LCFS radii residual [cm]"),
        ("axis_d_cm", "axis distance [cm]"),
        ("xset_d_cm", "X-set match [cm]"),
    ]
    fig, axes = plt.subplots(2, len(metrics), figsize=(12, 6.4), sharey="col")
    for row, (rows_by_class, leg_title) in enumerate(
        [
            (efit_art["rows"], "device read on EFIT ψ (read isolation)"),
            (engine_rows, "disc-engine solve vs EFIT (characterisation)"),
        ]
    ):
        for col, (key, label) in enumerate(metrics):
            ax = axes[row, col]
            data, names = [], []
            for cname in SCORED:
                vals = [
                    r[key]
                    for r in rows_by_class.get(cname, [])
                    if r.get(key) is not None and np.isfinite(r[key])
                ]
                if vals:
                    data.append(vals)
                    names.append(cname)
            if not data:
                ax.set_axis_off()
                continue
            bp = ax.boxplot(
                data, tick_labels=names, showfliers=False, patch_artist=True
            )
            for patch, cname in zip(bp["boxes"], names, strict=True):
                patch.set_facecolor(CLASS_COLOR[cname])
                patch.set_alpha(0.55)
            for med in bp["medians"]:
                med.set_color("0.15")
            ax.set_yscale("log")
            ax.tick_params(axis="x", labelsize=7, rotation=20)
            ax.spines[["top", "right"]].set_visible(False)
            if row == 0:
                ax.set_title(label, fontsize=10)
            if col == 0:
                ax.set_ylabel(leg_title, fontsize=8)
    fig.suptitle(
        "per-class residuals vs EFIT — log scale; boxes = IQR, whiskers 1.5×IQR",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "topology_residuals_by_class.png", dpi=150)
    plt.close(fig)
    logger.info("wrote topology_residuals_by_class.png")


def main() -> None:
    rows = np.load(ART_DIR / "topology_census-v0.npz")["rows"]
    efit_art = json.loads((ART_DIR / "topology_efit_read_eval-v0.json").read_text())
    engine_rows: dict = {}
    for cname in SCORED:
        p = ART_DIR / f"topology_engine_crossval_{cname}.json"
        if p.exists():
            engine_rows[cname] = json.loads(p.read_text())["rows"].get(cname, [])
    census_figure(rows)
    overlay_figure(efit_art["selection"])
    residuals_figure(efit_art, engine_rows)


if __name__ == "__main__":
    main()
