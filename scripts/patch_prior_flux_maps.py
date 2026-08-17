#!/usr/bin/env python
"""Prior-arm comparison exhibits: current distributions + flux surfaces.

For each held-out shot this renders one rampup and one flattop slice under
TWO inverse arms — the free (no-prior) P3-winner config and the physics-prior
arm (unidirectional softplus + free-boundary support consistency at the
frozen tune winner) — as paired exhibits:

* ``fig-prior-flux-maps-<regime>.png`` — imas-ink ψ(R,Z) flux surfaces vs the
  firewalled EFIT reference, arms side by side per shot.
* ``fig-prior-current-maps-<regime>.png`` — per-cell current distributions
  (diverging map, LCFS + limiter overlays), arms side by side per shot.

The rampup/flattop split is the diagnostic axis: if the anti-parallel halo
were vessel eddy current it would concentrate in rampup (large dIp/dt) and
vanish at flat top; a phase-independent halo instead points at a static
coil-model / calibration error.  Firewall: EFIT read for plotting/scoring
only, inside evaluator_context (same contract as patch_flux_map_report).

Artifacts:  imas_ambix/latent/artifacts/patch_gate/prior_flux_maps.json
Figures:    docs/figures/plasma-current-priors-hardening/fig-prior-*-maps-*.png
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from imas_ambix.latent.data import read_split_shot_lists
from imas_ambix.latent.patch_inverse import (
    InverseConfig,
    invert_slices,
    negative_current_fraction,
    outside_current_fraction,
    support_outside_mask,
)
from scripts.patch_flux_map_report import (
    WINNER,
    _closed_contour_about,
    _efit_slice,
    _fig_to_rgba,
    _machine_geometry,
    _our_slice,
    select_slices,
)
from scripts.patch_gate_eval import geometry_target, shot_payloads

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("patch_prior_flux_maps")

ARTIFACTS = Path("imas_ambix/latent/artifacts/patch_gate")
FIGURES = Path("docs/figures/plasma-current-priors-hardening")

ARMS = ("free", "priors")


def _prior_config(grid) -> InverseConfig:
    """The frozen tune-winner prior arm on top of the P3-winner base."""
    return InverseConfig(
        policy=WINNER.policy,
        lambda_fb=WINNER.lambda_fb,
        misfit_ratio=WINNER.misfit_ratio,
        lambda_max=WINNER.lambda_max,
        iters=WINNER.iters,
        connectivity=WINNER.connectivity,
        sign_prior="softplus",
        support_prior=True,
        support_weight=2000.0,
        halo_budget=0.03,
        limiter_r=np.asarray(grid.limiter_r, dtype=np.float64),
        limiter_z=np.asarray(grid.limiter_z, dtype=np.float64),
    )


def _current_map(ax, grid, basis, i_cell, lcfs, axis_rz, title):
    """Per-cell current distribution [kA/cell] on the machine cross-section."""
    r_c = np.asarray(basis.r_cells.detach().cpu(), dtype=np.float64)
    z_c = np.asarray(basis.z_cells.detach().cpu(), dtype=np.float64)
    rg = np.asarray(grid.rg, dtype=np.float64)
    zg = np.asarray(grid.zg, dtype=np.float64)
    ir = np.abs(rg[None, :] - r_c[:, None]).argmin(axis=1)
    iz = np.abs(zg[None, :] - z_c[:, None]).argmin(axis=1)
    cur = np.full((zg.size, rg.size), np.nan)
    cur[iz, ir] = i_cell / 1e3  # kA per cell
    vmax = np.nanmax(np.abs(cur))
    pm = ax.pcolormesh(rg, zg, cur, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.plot(
        np.append(grid.limiter_r, grid.limiter_r[0]),
        np.append(grid.limiter_z, grid.limiter_z[0]),
        color="#333",
        lw=1.0,
    )
    if lcfs is not None:
        ax.plot(lcfs[:, 0], lcfs[:, 1], color="#b2182b", lw=1.2)
    if np.all(np.isfinite(axis_rz)):
        ax.plot(*axis_rz, "+", color="k", ms=8)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("R [m]", fontsize=8)
    ax.tick_params(labelsize=7)
    return pm


def main() -> int:
    import argparse

    from imas_ink.figures import equilibrium_figure_mpl

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--calibration",
        type=str,
        default="",
        help="frozen static calibration JSON applied to raw payloads (name-mapped)",
    )
    args = ap.parse_args()
    suffix = "-calibrated" if args.calibration else ""
    cal_by_name: dict[str, tuple[float, float]] = {}
    if args.calibration:
        cal = json.loads(Path(args.calibration).read_text())
        cal_by_name = {
            c: (g, o)
            for c, g, o in zip(cal["channels"], cal["gain"], cal["offset"], strict=True)
        }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    FIGURES.mkdir(parents=True, exist_ok=True)
    _, held_shots = read_split_shot_lists(40, 8)
    logger.info("device=%s held-out shots: %s", device, held_shots)

    flux_panels = {"rampup": [], "flattop": []}
    cur_panels = {"rampup": [], "flattop": []}
    metrics = []

    for shot in held_shots:
        try:
            payload = shot_payloads(
                shot, nr=65, nz=97, max_slices=20, min_ip_ka=300.0, split="eval"
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("shot %s failed to load: %s", shot, exc)
            continue
        if payload is None:
            continue
        grid, basis = payload["grid"], payload["basis"]
        payloads, table = payload["payloads"], payload["table"]
        if cal_by_name:
            names = list(basis.sensor_channels)
            gain = np.array([cal_by_name.get(c, (1.0, 0.0))[0] for c in names])
            offs = np.array([cal_by_name.get(c, (1.0, 0.0))[1] for c in names])
            for p in payloads:
                p.measured[:] = (p.measured - offs) / gain
                p.scale[:] = p.scale / np.abs(gain)
        picks = select_slices(payloads, shot)
        if not picks:
            continue
        sel = [payloads[k] for _, k, _ in picks]
        geom = _machine_geometry(grid, table)
        lim_r = np.asarray(grid.limiter_r, dtype=np.float64)
        lim_z = np.asarray(grid.limiter_z, dtype=np.float64)

        inversions = {
            "free": invert_slices(basis, sel, WINNER, device=device),
            "priors": invert_slices(basis, sel, _prior_config(grid), device=device),
        }

        for j, (kind, k, efit) in enumerate(picks):
            row_flux, row_cur = [], []
            for arm in ARMS:
                r = inversions[arm][j]
                psi2d = basis.psi_grid_2d_np(r.i_cell, payloads[k].i_pf)
                target, psi_ax, psi_b = geometry_target(psi2d, grid)
                axis_rz = (float(target[0]), float(target[1]))
                lcfs = _closed_contour_about(grid.rg, grid.zg, psi2d, psi_b, *axis_rz)

                out_mask = support_outside_mask(
                    basis, r.i_cell, payloads[k].i_pf, limiter_r=lim_r, limiter_z=lim_z
                )
                neg = negative_current_fraction(r.i_cell, payloads[k].ip_amperes)
                out = outside_current_fraction(
                    r.i_cell, payloads[k].ip_amperes, out_mask
                )
                metrics.append(
                    {
                        "shot": int(shot),
                        "regime": kind,
                        "arm": arm,
                        "time_s": round(float(payloads[k].time_s), 4),
                        "negative_fraction": round(neg, 4),
                        "outside_fraction": round(out, 4),
                        "misfit": round(float(r.misfit), 4),
                    }
                )

                sl = _our_slice(
                    psi2d,
                    grid,
                    target,
                    psi_ax,
                    psi_b,
                    payloads[k].ip_amperes,
                    payloads[k].time_s,
                    lcfs,
                )
                fig, _ax = equilibrium_figure_mpl(
                    sl,
                    geom,
                    reference_slice=_efit_slice(efit),
                    reference_name="EFIT",
                    figsize=(4.4, 5.8),
                    show_probes=False,
                    show_flux_loops=False,
                )
                fig.suptitle(f"{shot} t={payloads[k].time_s:.3f}s {arm}", fontsize=9)
                row_flux.append(_fig_to_rgba(fig))
                plt.close(fig)

                fig, ax = plt.subplots(figsize=(4.0, 5.4))
                pm = _current_map(
                    ax,
                    grid,
                    basis,
                    r.i_cell,
                    lcfs,
                    axis_rz,
                    f"{shot} t={payloads[k].time_s:.3f}s {arm}\n"
                    f"anti-∥ {neg:.2f}·Ip, outside {out:.2f}·Ip",
                )
                fig.colorbar(pm, ax=ax, shrink=0.8, label="I cell [kA]")
                fig.tight_layout()
                row_cur.append(_fig_to_rgba(fig))
                plt.close(fig)

            flux_panels[kind].append(row_flux)
            cur_panels[kind].append(row_cur)
            logger.info("%d %-7s rendered both arms", shot, kind)

    fig_paths = []
    for regime in ("rampup", "flattop"):
        for label, panels in (("flux", flux_panels), ("current", cur_panels)):
            rows = panels[regime]
            if not rows:
                continue
            ncol = len(rows)
            fig, axes = plt.subplots(2, ncol, figsize=(3.4 * ncol, 9.2))
            axes = np.atleast_2d(axes)
            for c, row in enumerate(rows):
                for a in range(2):
                    axes[a, c].imshow(row[a])
                    axes[a, c].axis("off")
            fig.suptitle(
                f"{label} maps — {regime}: free inverse (top) vs "
                f"unidirectional+support priors (bottom)",
                fontsize=12,
            )
            fig.tight_layout(rect=(0, 0, 1, 0.97))
            out = FIGURES / f"fig-prior-{label}-maps-{regime}{suffix}.png"
            fig.savefig(out, dpi=110)
            plt.close(fig)
            fig_paths.append(str(out))
            logger.info("wrote %s (%d shots)", out, ncol)

    (ARTIFACTS / f"prior_flux_maps{suffix.replace('-', '_')}.json").write_text(
        json.dumps(
            {
                "calibration": args.calibration or None,
                "metrics": metrics,
                "figures": fig_paths,
            },
            indent=2,
        )
    )
    logger.info("wrote prior_flux_maps.json (%d slice-arm rows)", len(metrics))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
