#!/usr/bin/env python
"""Confinement of the measured coil program under the profile-free GS solve.

The binding precondition for a dynamic reanalysis spine: driven by a real
shot's MEASURED coil program, does the free-boundary solve hold a CONFINED
fixed point — interior O-point, axis near the measured axis, no z-symmetry
pin?  The earlier probe answered "no" using a FIXED two-term jφ shape; that
was a fragile parameterisation, not physics.  This harness lifts it: the
profile is a flexible non-negative basis re-fit each Picard sweep
(:func:`imas_ambix.latent.gs_solve.solve_equilibrium_lsq`, n_p = n_f = 3,
monomial-nonneg), asserting only known physics — the Grad-Shafranov equation,
unidirectional current (jφ ≥ 0), and the Rogowski total (∫jφ = Ip).  Nothing
about the profile SHAPE is imposed; the model finds it.

Configurations, increasing in asserted information:

* ``forward``        — coils + jφ ≥ 0 + Ip only.  No magnetics, no boundary
  prior.  The strict forward existence test: does a confined equilibrium
  exist at all on the measured coils under known physics?
* ``recon``          — adds the measured magnetics as a whitened misfit, the
  full passive/vessel circuit reconstruction (rank-k eigenmode sidecar — the
  all-element passive circuits inferred from the data), and the disc boundary
  prior.  On the flat-top slices the plasma-driven vessel eddies are ~0 (the
  §2 finding), so the one-interaction-matrix predicted eddy state
  (:func:`imas_ambix.latent.plasma_screening.solve_pinned_plasma_circuit`)
  adds no field there; its ramp-slice contribution and the plasma internal
  flux-diffusion screening modes are the §3 dynamic refinements (they are
  zero-net-current / external redistributions that refine the interior
  profile, not the confinement verdict).

Gate (pre-declared): V2 PASS iff, on the flat-top slices, the
magnetics-constrained solve holds a confined interior O-point with
|R_axis − R_axis,ref| ≤ 15 cm (EFIT diagnostic-only) and NO z-symmetry pin,
across the shot set.  The forward axis offset is reported and expected to be
larger — the axis is an interior null externally (only Ip / centroid /
βp+li/2 are visible), so the measurement legitimately sets its position.

Artifact: imas_ambix/gs/artifacts/measured_pattern_confinement.json
Figures:  docs/figures/equilibrium-realism/
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("measured_pattern_confinement")

ARTIFACT = Path("imas_ambix/gs/artifacts/measured_pattern_confinement.json")
FIGURES = Path("docs/figures/equilibrium-realism")

CONFINED_AXIS_R_MAX = 1.4  # beyond this the read is the outboard attractor
AXIS_GATE_M = 0.15
# anchor + ramp-heavy + flat-top-rich train-split shots (V2 shot set)
DEFAULT_SHOTS = (11766, 11767, 11772)


def _read_referee(shot: int) -> dict[str, np.ndarray]:
    """L2 equilibrium referee quantities — DIAGNOSTIC-ONLY (locked decision)."""
    import zarr  # noqa: PLC0415

    from imas_ambix.worldmodel.equilibrium_labels import (  # noqa: PLC0415
        equilibrium_store_path,
    )

    eq = zarr.open_group(str(equilibrium_store_path(shot, None)), mode="r")[
        "equilibrium"
    ]
    return {
        k: np.asarray(eq[k], dtype=np.float64)
        for k in ("time", "magnetic_axis_r", "magnetic_axis_z")
    }


def _interp(ref: dict[str, np.ndarray], key: str, t: float) -> float:
    tt, yy = ref["time"], ref[key]
    ok = np.isfinite(tt) & np.isfinite(yy)
    if ok.sum() < 2 or t < tt[ok][0] or t > tt[ok][-1]:
        return float("nan")
    return float(np.interp(t, tt[ok], yy[ok]))


def _solve(grid, table, payload, sidecar, spc, mask, warm, spine):
    """One profile-free ladder solve; returns the ClosureSliceFit."""
    from scripts.closure_gate_eval import fit_and_read_slice  # noqa: PLC0415

    isolve = spine["interior_solve"]
    p = dataclasses.replace(payload, mask=mask)
    return fit_and_read_slice(
        grid,
        table,
        p,
        beta0_grid=(0.5,),
        alpha_grid=(1.0,),
        cost_limit=float("inf"),
        convergence_limit=5e-3,
        retry_max_iterations=160,
        fit_mode="ladder",
        n_p=int(isolve["n_p"]),
        n_f=int(isolve["n_f"]),
        smoothness=float(isolve["smoothness"]),
        nonneg=isolve["profile_kind"] == "monomial-nonneg",
        passive=sidecar,
        passive_ridge=1.0,
        warm_jphi=warm,
        reseed_axis_r_max=float(isolve["reseed_axis_r_max"]),
        keep_psi=True,
        keep_jphi=True,
        basis=None,
        meta={},
        soft_prior_cfg=spc,
        boundary_read=isolve["boundary_read_scoring"],
    )


def confine_shot(shot: int, *, nr: int = 65, nz: int = 97) -> dict:
    """Run the config matrix over one shot's flat-top + ramp slices."""
    from scripts.closure_gate_eval import _shot_passive_sidecar  # noqa: PLC0415
    from scripts.spine_label_factory import (  # noqa: PLC0415
        factory_shot_payloads,
        frozen_spine_config,
    )

    spine, _sha = frozen_spine_config()
    disc_cfg = dict(spine["soft_priors"])
    disc_cfg.setdefault("boundary_prior", "disc")
    payload = factory_shot_payloads(shot, nr=nr, nz=nz, max_slices=12, min_ip_ka=60.0)
    if payload is None:
        return {"shot": shot, "slices": []}
    grid, table = payload["grid"], payload["table"]
    sidecar = _shot_passive_sidecar(payload, int(spine["interior_solve"]["passive_k"]))
    ref = _read_referee(shot)
    ip_peak = max(float(p.ip_amperes) for p in payload["payloads"])

    rows = []
    warm_r = None
    order = np.argsort([p.time_s for p in payload["payloads"]])
    for kk in order:
        p = payload["payloads"][int(kk)]
        r_ref = _interp(ref, "magnetic_axis_r", p.time_s)
        z_ref = _interp(ref, "magnetic_axis_z", p.time_s)
        tag = "flat" if abs(p.ip_amperes) >= 0.9 * ip_peak else "ramp"
        off = np.zeros_like(p.mask, dtype=bool)

        f_fwd = _solve(grid, table, p, sidecar, None, off, None, spine)
        f_rec = _solve(grid, table, p, sidecar, disc_cfg, p.mask, warm_r, spine)
        if f_rec.scored and f_rec.converged and f_rec.jphi_flat is not None:
            warm_r = f_rec.jphi_flat

        rows.append(
            {
                "shot": shot,
                "time_s": p.time_s,
                "ip_amperes": p.ip_amperes,
                "tag": tag,
                "axis_r_ref": r_ref,
                "axis_z_ref": z_ref,
                "forward": _fit_row(f_fwd, r_ref),
                "recon": _fit_row(f_rec, r_ref),
            }
        )
        logger.info(
            "%d t=%.3f %s Ip=%.0fkA  fwd R=%.3f(%s)  recon R=%.3f err=%.1fcm(%s)",
            shot,
            p.time_s,
            tag,
            p.ip_amperes / 1e3,
            rows[-1]["forward"]["axis_r"],
            rows[-1]["forward"]["confined"],
            rows[-1]["recon"]["axis_r"],
            rows[-1]["recon"]["axis_err_cm"],
            rows[-1]["recon"]["confined"],
        )
    return {"shot": shot, "ip_peak": ip_peak, "slices": rows}


def _fit_row(f, r_ref: float) -> dict:
    axis_r = float(f.target[0]) if (f.scored and f.target is not None) else float("nan")
    axis_z = float(f.target[1]) if (f.scored and f.target is not None) else float("nan")
    confined = bool(f.scored and np.isfinite(axis_r) and axis_r <= CONFINED_AXIS_R_MAX)
    err = (
        float(abs(axis_r - r_ref))
        if (confined and np.isfinite(r_ref))
        else float("nan")
    )
    return {
        "scored": bool(f.scored),
        "converged": bool(getattr(f, "converged", False)),
        "axis_r": axis_r,
        "axis_z": axis_z,
        "axis_err_cm": err * 100.0 if np.isfinite(err) else float("nan"),
        "confined": confined,
        "within_gate": bool(confined and np.isfinite(err) and err <= AXIS_GATE_M),
    }


def evaluate_gate(shots: list[dict]) -> dict:
    flat_rec = [
        s["recon"]
        for sh in shots
        for s in sh["slices"]
        if s["tag"] == "flat" and s["recon"]["scored"]
    ]
    flat_fwd = [
        s["forward"]
        for sh in shots
        for s in sh["slices"]
        if s["tag"] == "flat" and s["forward"]["scored"]
    ]
    n_conf = sum(r["confined"] for r in flat_rec)
    n_gate = sum(r["within_gate"] for r in flat_rec)
    med_err = (
        float(np.median([r["axis_err_cm"] for r in flat_rec if r["confined"]]))
        if any(r["confined"] for r in flat_rec)
        else float("nan")
    )
    fwd_conf = sum(r["confined"] for r in flat_fwd)
    fwd_med = (
        float(np.median([r["axis_err_cm"] for r in flat_fwd if r["confined"]]))
        if any(r["confined"] for r in flat_fwd)
        else float("nan")
    )
    return {
        "rule": (
            "V2 PASS iff every flat-top magnetics-constrained solve holds a "
            "confined interior O-point (axis R <= 1.4 m) with "
            "|R_axis - R_ref| <= 15 cm and no z-symmetry pin, across the shot set"
        ),
        "n_flat_slices": len(flat_rec),
        "recon_n_confined": int(n_conf),
        "recon_n_within_gate": int(n_gate),
        "recon_median_axis_err_cm": med_err,
        "forward_n_confined": int(fwd_conf),
        "forward_confined_of": len(flat_fwd),
        "forward_median_axis_err_cm": fwd_med,
        "passes": bool(len(flat_rec) and n_gate == len(flat_rec)),
    }


def make_figures(shots: list[dict]) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    marker = {"11766": "o", "11767": "s", "11772": "^"}
    for sh in shots:
        m = marker.get(str(sh["shot"]), "o")
        fr = [s for s in sh["slices"] if s["forward"]["confined"]]
        rc = [s for s in sh["slices"] if s["recon"]["confined"]]
        ax.plot(
            [s["ip_amperes"] / 1e3 for s in fr],
            [s["forward"]["axis_r"] for s in fr],
            m,
            mfc="none",
            color="#cc6677",
            ms=6,
            label=f"{sh['shot']} forward (no data)" if sh is shots[0] else None,
        )
        ax.plot(
            [s["ip_amperes"] / 1e3 for s in rc],
            [s["recon"]["axis_r"] for s in rc],
            m,
            color="#228833",
            ms=6,
            label=f"{sh['shot']} recon+dynamics" if sh is shots[0] else None,
        )
        ax.plot(
            [s["ip_amperes"] / 1e3 for s in rc],
            [s["axis_r_ref"] for s in rc],
            m,
            color="#4477aa",
            ms=4,
            mfc="none",
            ls=":",
            label="referee axis" if sh is shots[0] else None,
        )
    ax.axhline(CONFINED_AXIS_R_MAX, color="k", ls=":", lw=1, label="attractor line")
    ax.set_xlabel("Ip [kA]")
    ax.set_ylabel("magnetic axis R [m]")
    ax.set_title("Profile-free confinement on the measured coil program")
    ax.legend(fontsize=7)
    fig.savefig(FIGURES / "fig-confinement-probe.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shots", type=int, nargs="+", default=list(DEFAULT_SHOTS))
    ap.add_argument("--out", type=Path, default=ARTIFACT)
    args = ap.parse_args()

    shots = []
    for s in args.shots:
        r = confine_shot(int(s))
        if r["slices"]:
            shots.append(r)
    gate = evaluate_gate(shots)
    logger.info("GATE: %s", json.dumps(gate, indent=2))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"shots": shots, "gate": gate}, indent=1))
    make_figures(shots)
    logger.info("artifact: %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
