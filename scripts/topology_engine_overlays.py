"""Engine ψ(R,Z) cross-section overlays for the census cross-validation.

Reads the full four-pass engine cross-validation rows (per class, plain arm),
selects the worst and representative slices per class and phase, re-solves the
owning shots through the SAME validated chain
(:func:`scripts.heldout_mse_gate_eval.coupled_solve_chain`), and renders the
engine's own force-balanced poloidal flux against the EFIT reconstruction with
the imas-ink equilibrium figure.  The overlays make the boundary-residual
tables legible: where the coupled solve's axis / LCFS sit relative to EFIT, and
how that changes between the eddy-dominated ramp and the flat-top.

The EFIT field is read ONLY as a visualization reference (the referee context,
same path the flux-map report uses); the engine solve itself consumes measured
magnetics alone (firewall intact — this script does not feed EFIT into any fit).

Outputs (docs/figures/connectivity-topology-reader/):
  fig-engine-overlay-<class>.png   ramp + flat-top ψ panels per class
  fig-engine-worst-sn-lower.png    the sn-lower coupled-drift deep dive
  fig-engine-overlay-grid.png      one worst-flat-top panel per class
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("topology_engine_overlays")

ARTIFACT_DIR = Path("imas_ambix/latent/artifacts/patch_gate")
FIGURE_DIR = Path("docs/figures/connectivity-topology-reader")
CHAIN_MAX_SLICES = 24
CHAIN_MIN_IP_KA = 60.0
RAMP_END_S = 0.2

SCORED_CLASSES = ("limited", "sn-lower", "sn-upper", "connected-dn", "marginal-dn")


def _plain_rows(cname: str) -> list[dict]:
    p = ARTIFACT_DIR / f"topology_full_engine_crossval-plain-{cname}.json"
    if not p.exists():
        return []
    return json.loads(p.read_text()).get("rows", [])


def _pick_slices(cname: str) -> list[tuple[str, dict]]:
    """(role, row) picks for one class: worst/median flat-top + worst ramp."""
    rows = [r for r in _plain_rows(cname) if r["cls"] == cname]
    picks: list[tuple[str, dict]] = []
    ft = sorted((r for r in rows if r["phase"] == "flattop"),
                key=lambda r: r["radii_dmed_cm"])
    ramp = sorted((r for r in rows if r["phase"] == "ramp"),
                  key=lambda r: r["radii_dmed_cm"])
    if ft:
        picks.append(("flat-top worst", ft[-1]))
        picks.append(("flat-top median", ft[len(ft) // 2]))
        picks.append(("flat-top best", ft[0]))
    if ramp:
        picks.append(("ramp worst", ramp[-1]))
    return picks


def _render_slice(shot: int, time_s: float, *, nr: int, nz: int):
    """Re-solve ``shot`` and return (our_slice, efit_slice, geom, fit_info) for
    the coupled fit nearest ``time_s`` — or None when it cannot be rendered."""
    from imas_ambix.eval.efit_referee import evaluator_context  # noqa: PLC0415
    from scripts.closure_gate_eval import geometry_target  # noqa: PLC0415
    from scripts.heldout_mse_gate_eval import (  # noqa: PLC0415
        coupled_solve_chain,
        frozen_eta_params,
    )
    from scripts.patch_flux_map_report import (  # noqa: PLC0415
        _closed_contour_about,
        _efit_slice,
        _machine_geometry,
        _our_slice,
        read_efit_slice,
    )

    chain = coupled_solve_chain(
        int(shot), nr=nr, nz=nz, sigma=0.02, eta_params=list(frozen_eta_params()),
        prior_weight=0.3, n_sub=24, par_weight=1.0, n_rho=24,
        max_slices=CHAIN_MAX_SLICES, min_ip_ka=CHAIN_MIN_IP_KA)
    if not chain["slices"]:
        return None
    grid = chain["grid"]
    times = np.array([float(s["p"].time_s) for s in chain["slices"]])
    j = int(np.argmin(np.abs(times - time_s)))
    if abs(times[j] - time_s) > 0.02:
        return None
    f = chain["fits"][j]
    if not (f.scored and f.psi is not None):
        return None
    psi2d = np.asarray(f.psi, dtype=np.float64)
    target, psi_ax, psi_b = geometry_target(psi2d, grid)
    axis_rz = (float(target[0]), float(target[1]))
    our_lcfs = _closed_contour_about(grid.rg, grid.zg, psi2d, psi_b, *axis_rz)
    p = chain["slices"][j]["p"]
    our = _our_slice(psi2d, grid, target, psi_ax, psi_b, p.ip_amperes, p.time_s,
                     our_lcfs)
    geom = _machine_geometry(grid, chain["table"])
    with evaluator_context():
        efit = read_efit_slice(int(shot), float(p.time_s))
    efit_slice = _efit_slice(efit) if efit is not None else None
    return our, efit_slice, geom, {"axis_r": axis_rz[0], "time_s": float(p.time_s)}


def _panel(shot: int, time_s: float, role: str, dmed: float, *, nr: int, nz: int):
    """Render one slice to an RGBA panel; returns (rgba, subtitle) or None."""
    from imas_ink.figures import equilibrium_figure_mpl  # noqa: PLC0415

    from scripts.patch_flux_map_report import _fig_to_rgba  # noqa: PLC0415

    got = _render_slice(shot, time_s, nr=nr, nz=nz)
    if got is None:
        return None
    our, efit_slice, geom, info = got
    fig, _ax = equilibrium_figure_mpl(
        our, geom, reference_slice=efit_slice, reference_name="EFIT",
        figsize=(4.4, 6.0), show_probes=False, show_flux_loops=False)
    sub = (f"{shot} t={info['time_s']:.3f}s\n{role}: LCFS Δ={dmed:.1f} cm, "
           f"axis R={info['axis_r']:.2f} m")
    fig.suptitle(sub, fontsize=8)
    rgba = _fig_to_rgba(fig)
    plt.close(fig)
    return rgba, sub


def class_overlay(cname: str, *, nr: int, nz: int) -> None:
    picks = _pick_slices(cname)
    if not picks:
        logger.warning("%s: no rows to overlay", cname)
        return
    panels = []
    for role, r in picks:
        got = _panel(int(r["shot"]), float(r["time_s"]), role,
                     float(r["radii_dmed_cm"]), nr=nr, nz=nz)
        if got is not None:
            panels.append(got)
            logger.info("  %s %s: %s", cname, role, got[1].replace("\n", " "))
    if not panels:
        return
    ncol = len(panels)
    fig, axes = plt.subplots(1, ncol, figsize=(3.4 * ncol, 6.2))
    for ax, (rgba, _sub) in zip(np.atleast_1d(axes), panels, strict=False):
        ax.imshow(rgba)
        ax.axis("off")
    fig.suptitle(f"§6b engine ψ(R,Z) vs EFIT — {cname} "
                 f"(engine solid, EFIT faint; measured magnetics only)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = FIGURE_DIR / f"fig-engine-overlay-{cname}.png"
    fig.savefig(out, dpi=115)
    plt.close(fig)
    logger.info("wrote %s (%d panels)", out, len(panels))


def sn_lower_deepdive(*, nr: int, nz: int) -> None:
    """The sn-lower coupled-drift deep dive: worst flat-top slices, showing the
    outboard axis drift that inverts sn-lower below the bare seed."""
    rows = [r for r in _plain_rows("sn-lower") if r["cls"] == "sn-lower"
            and r["phase"] == "flattop"]
    rows = sorted(rows, key=lambda r: -r["radii_dmed_cm"])[:4]
    panels = []
    for r in rows:
        got = _panel(int(r["shot"]), float(r["time_s"]), "flat-top",
                     float(r["radii_dmed_cm"]), nr=nr, nz=nz)
        if got is not None:
            panels.append(got)
    if not panels:
        logger.warning("sn-lower deep dive: no panels")
        return
    ncol = len(panels)
    fig, axes = plt.subplots(1, ncol, figsize=(3.4 * ncol, 6.2))
    for ax, (rgba, _sub) in zip(np.atleast_1d(axes), panels, strict=False):
        ax.imshow(rgba)
        ax.axis("off")
    fig.suptitle("§6b sn-lower deep dive — worst flat-top coupled fits: the "
                 "outboard axis drift EFIT (faint) does not share",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = FIGURE_DIR / "fig-engine-worst-sn-lower.png"
    fig.savefig(out, dpi=115)
    plt.close(fig)
    logger.info("wrote %s (%d panels)", out, len(panels))


def worst_grid(*, nr: int, nz: int) -> None:
    """One worst-flat-top panel per class — the census-wide failure gallery."""
    panels = []
    for cname in SCORED_CLASSES:
        ft = sorted((r for r in _plain_rows(cname) if r["cls"] == cname
                     and r["phase"] == "flattop"),
                    key=lambda r: -r["radii_dmed_cm"])
        if not ft:
            continue
        r = ft[0]
        got = _panel(int(r["shot"]), float(r["time_s"]), f"{cname} worst",
                     float(r["radii_dmed_cm"]), nr=nr, nz=nz)
        if got is not None:
            panels.append((cname, got))
    if not panels:
        return
    ncol = len(panels)
    fig, axes = plt.subplots(1, ncol, figsize=(3.2 * ncol, 6.2))
    for ax, (_c, (rgba, _sub)) in zip(np.atleast_1d(axes), panels, strict=False):
        ax.imshow(rgba)
        ax.axis("off")
    fig.suptitle("§6b worst flat-top per class — engine ψ vs EFIT (faint)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = FIGURE_DIR / "fig-engine-overlay-grid.png"
    fig.savefig(out, dpi=115)
    plt.close(fig)
    logger.info("wrote %s (%d panels)", out, len(panels))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nr", type=int, default=65)
    ap.add_argument("--nz", type=int, default=97)
    ap.add_argument("--classes", nargs="*", default=list(SCORED_CLASSES))
    args = ap.parse_args()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for cname in args.classes:
        class_overlay(cname, nr=args.nr, nz=args.nz)
    sn_lower_deepdive(nr=args.nr, nz=args.nz)
    worst_grid(nr=args.nr, nz=args.nz)


if __name__ == "__main__":
    main()
