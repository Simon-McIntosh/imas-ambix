#!/usr/bin/env python
"""Evidence figures: soft-prior-anchored poloidal flux maps vs firewalled EFIT.

For each held-out shot (a ramp-up and a flat-top slice), solve the interior
free-boundary GS fixed point TWICE — the free-boundary baseline (no prior)
and the annulus-anchored soft-prior solve (annulus-anchored) — and render ψ(R, Z) with
imas-ink's ``equilibrium_figure_mpl``, the firewalled EFIT boundary overlaid as
the faint reference.  EFIT is SCORING/PLOT ONLY (never an input to either solve).

Writes docs/figures/equilibrium-boundary-closure/fig-fluxmap-<regime>.png.
Comparison helper; reuses the closure gate + flux-map
report machinery verbatim so the panels match the scored geometry read.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from imas_ambix.latent.data import read_split_shot_lists

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("flux-evidence")

FIGDIR = Path("docs/figures/equilibrium-boundary-closure")
FROZEN = "imas_ambix/latent/artifacts/patch_gate/harmonic_prior_frozen.json"


def _th_lcfs_ring(p, grid, table, basis, meta):
    """The two-pass source-free TH read's own LCFS ring for this slice
    (moderate-order + graded ridge, origin re-sited on the LCFS centroid,
    ψ_N=1 leg-clip) — the boundary the anchor targets, to overlay in green."""
    from imas_ambix.latent.boundary_disc import disc_read

    inv = disc_read(p, grid, table, basis)
    return None if inv is None else inv.ring


def _pushout_lcfs_ring(psi, grid, axis):
    """Read OUR interior ψ with the push-out algorithm (outermost closed
    axis-enclosing ring) instead of the innermost-X-point read — the diagnostic
    for whether the interior LCFS is under-sized by the READ vs by the current."""
    from imas_ambix.latent.topology import lcfs_contour

    lc = lcfs_contour(
        np.asarray(psi),
        grid.rg,
        grid.zg,
        tuple(axis),
        limiter_r=grid.limiter_r,
        limiter_z=grid.limiter_z,
    )
    return lc.ring if lc.found else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor-weight", type=float, default=10.0)
    ap.add_argument("--n-train", type=int, default=40)
    ap.add_argument("--n-heldout", type=int, default=8)
    ap.add_argument("--nr", type=int, default=65)
    ap.add_argument("--nz", type=int, default=97)
    ap.add_argument("--max-slices-per-shot", type=int, default=20)
    ap.add_argument("--min-ip-ka", type=float, default=300.0)
    ap.add_argument("--shots", type=str, default="")
    args = ap.parse_args()

    from imas_ink.figures import equilibrium_figure_mpl

    from scripts.closure_gate_eval import (
        fit_and_read_slice,
        geometry_target_pushout,
        load_frozen_lookup,
        shot_payloads,
    )
    from scripts.patch_flux_map_report import (
        _closed_contour_about,
        _efit_slice,
        _fig_to_rgba,
        _machine_geometry,
        _our_slice,
        select_slices,
    )

    _lookup, meta = load_frozen_lookup(FROZEN)
    spc = {
        "anchor_weight": args.anchor_weight,
        "anchor_form": "abs-psi",
        "boundary_prior": "disc",
        "anchor_robust_clip": 3.0,
        "sol_cap": 1.0,
        "q_bound": False,
        "q_weight": 1.0,
        "b_phi0": 0.55,
        "ip_soft_sigma": 0.0,
    }
    _, held = read_split_shot_lists(args.n_train, args.n_heldout)
    if args.shots:
        want = {int(s) for s in args.shots.split(",") if s.strip()}
        held = [s for s in held if int(s) in want]

    FIGDIR.mkdir(parents=True, exist_ok=True)
    panels = {"rampup": [], "flattop": []}
    for shot in held:
        try:
            payload = shot_payloads(
                shot,
                nr=args.nr,
                nz=args.nz,
                max_slices=args.max_slices_per_shot,
                min_ip_ka=args.min_ip_ka,
                split="eval",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("shot %s load failed: %s", shot, exc)
            continue
        if payload is None:
            continue
        grid, table, basis = payload["grid"], payload["table"], payload["basis"]
        payloads = payload["payloads"]
        geom = _machine_geometry(grid, table)
        picks = select_slices(payloads, shot)
        for kind, k, efit in picks:
            p = payloads[k]
            common = dict(
                beta0_grid=(0.5,),
                alpha_grid=(1.0,),
                cost_limit=float("inf"),
                convergence_limit=5e-3,
                retry_max_iterations=160,
                fit_mode="ladder",
                n_p=3,
                n_f=3,
                nonneg=True,
                smoothness=1e-2,
                passive=None,
                reseed_axis_r_max=1.4,
                keep_psi=True,
            )
            # Two-pass source-free TH read boundary (green): moderate order +
            # graded ridge, origin re-sited on the LCFS centroid, ψ_N=1 leg-clip
            th_ring = _th_lcfs_ring(p, grid, table, basis, meta)
            base = fit_and_read_slice(grid, table, p, **common)
            anc = fit_and_read_slice(
                grid,
                table,
                p,
                **common,
                basis=basis,
                meta=meta,
                soft_prior_cfg=spc,
            )
            for tag, fit in (("free-boundary", base), ("annulus-anchored", anc)):
                if not fit.scored or fit.psi is None:
                    logger.info("%d %s %s not scored", shot, kind, tag)
                    continue
                # score the LCFS with the push-out reader (the fixed, canonical
                # read) — this is the scored boundary, and it matches the EFIT reference
                target, psi_ax, psi_b = geometry_target_pushout(fit.psi, grid)
                axis_rz = (float(target[0]), float(target[1]))
                lcfs = _closed_contour_about(grid.rg, grid.zg, fit.psi, psi_b, *axis_rz)
                sl = _our_slice(
                    fit.psi, grid, target, psi_ax, psi_b, p.ip_amperes, p.time_s, lcfs
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
                ax0 = np.atleast_1d(_ax).ravel()[0]
                if th_ring is not None:
                    ax0.plot(
                        th_ring[:, 0],
                        th_ring[:, 1],
                        "-",
                        color="#1b9e2f",
                        lw=2.0,
                        label="disc read",
                        zorder=6,
                    )
                ax0.legend(fontsize=6, loc="upper right")
                fig.suptitle(f"{shot} t={p.time_s:.3f}s ({kind}) — {tag}", fontsize=9)
                panels[kind].append((f"{shot} {tag}", _fig_to_rgba(fig)))
                plt.close(fig)
                logger.info("%d %-7s %s rendered", shot, kind, tag)

    written = []
    for regime, items in panels.items():
        if not items:
            continue
        ncol = 4
        nrow = int(np.ceil(len(items) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.8 * ncol, 5.0 * nrow))
        axes = np.atleast_1d(axes).ravel()
        for ax in axes:
            ax.axis("off")
        for ax, (title, img) in zip(axes, items, strict=False):
            ax.imshow(img)
            ax.set_title(title, fontsize=8)
        fig.suptitle(
            f"Interior ψ(R,Z) vs firewalled EFIT — {regime} "
            "(free-boundary vs annulus-anchored; faint = EFIT)",
            fontsize=13,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        out = FIGDIR / f"fig-fluxmap-{regime}.png"
        fig.savefig(out, dpi=110)
        plt.close(fig)
        written.append(str(out))
        logger.info("wrote %s (%d panels)", out, len(items))
    print("FIGURES:", written)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
