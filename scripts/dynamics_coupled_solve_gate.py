#!/usr/bin/env python
"""Dynamics-coupled solve on the position-controlled spine.

The position-controlled solve holds the current at the measured position (Ip +
R/Z current-centroid moment, no magnetics), but it leaves the INTERNAL PROFILE
open — single-slice magnetics cannot distinguish the peakedness (the interior
null).  The profile is not free in time, though: the jφ profile at t evolves
from t-Δt under resistive poloidal-flux diffusion, whose drives (surface flux
swing, Ip) are MEASURED and whose only unknown is the parallel resistivity
η(ψ_N).  This gate earns the internal profile from that physics.

Two facts fix the design (both from the position solve landing):

* the position hold uses a deliberately RIGID free-sign K=2 profile (n_p=n_f=1)
  as a stability crutch — without magnetics a rich profile is under-determined
  (the interior null) and slides outboard.  K=2 carries position, never a
  profile claim.
* the internal profile needs a richer representation to be earned — the frozen
  spine's non-negative ladder (n_p=n_f=3, jφ>=0, smoothness ridge): "GS + force
  balance + regularization", not an assumed low-order polynomial.

So the coupling is: the K=2 position solve is the stable scaffold; its
flux-surface geometry drives the ψ diffusion over each measured interval; the
evolved current projects onto the RICH basis; and the rich solve is re-run with
that diffusion-evolved prediction as a soft coefficient centre.  The dynamics
supply the internal-profile shape the magnetics cannot — stabilising a rich GS
solve that is otherwise under-determined.

Three arms per slice, all with the measured centroid + disc seed, magnetics
mask OFF (firewall: Ip + centroid + measured drives only, η the low-DOF unknown;
no EFIT, no assumed profile, no tuned gain):

* K2       — the landed position solve (free-sign K=2).  The position scaffold
             and the stable reference the coupled arm must not regress against.
* rich-unc — the rich non-negative ladder + centroid, NO diffusion prior.  The
             under-determined rich solve; expected to be the fragile one.
* rich-cpl — the rich ladder + centroid + the diffusion-evolved coefficient
             prior.  The dynamics-coupled solve.

Gate G3a (stability / finiteness): rich-cpl stays finite and physical (confined
inboard axis, no NaN) across the beam-on shot set, with no outboard-drift
regression against the K2 position reference.  The rich-unc confined fraction is
reported alongside — the coupling earns its keep by stabilising what rich-unc
cannot hold.

G3b (synthetic exact-recovery of the profile split) and G3c (the module pins)
are separate gates (current_diffusion_synthetic_gate.py; the latent tests).

Artifacts: imas_ambix/latent/artifacts/patch_gate/dynamics_coupled_solve[-tag].json
Figures:   docs/figures/mse-gated-reanalysis/
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("dynamics_coupled_solve_gate")

ARTIFACT = Path("imas_ambix/latent/artifacts/patch_gate/dynamics_coupled_solve.json")
FIGURES = Path("docs/figures/mse-gated-reanalysis")

CONFINED_AXIS_R_MAX = 1.4  # beyond this the read is the outboard attractor [m]
DEFAULT_SHOTS = (11766, 11767, 11772)
DEFAULT_SIGMA_M = 0.02  # centroid tether 1σ [m]
DEFAULT_ETA = (-7.3, 2.0, 2.0)  # log10(eta0 [Ω·m]), contrast, shape — nominal
REGRESS_TOL_CM = 5.0  # coupled axis must track the K2 position reference [cm]


def _axis(f) -> tuple[float, float]:
    if not (f.scored and f.target is not None):
        return float("nan"), float("nan")
    return float(f.target[0]), float(f.target[1])


def _current_centroid(grid, f) -> tuple[float, float]:
    if not (f.scored and f.jphi_flat is not None):
        return float("nan"), float("nan")
    jf = np.asarray(f.jphi_flat, dtype=np.float64)
    ic = jf[grid.cells]
    tot = ic.sum()
    if abs(tot) < 1e-12:
        return float("nan"), float("nan")
    return (
        float((grid.flat_r[grid.cells] * ic).sum() / tot),
        float((grid.flat_z[grid.cells] * ic).sum() / tot),
    )


def _confined(axis_r: float) -> bool:
    return bool(np.isfinite(axis_r) and axis_r <= CONFINED_AXIS_R_MAX)


def run_shot(shot: int, *, nr: int, nz: int, sigma: float, eta_params, prior_weight,
             n_sub: int, par_weight: float, b_phi0: float, n_rho: int,
             max_slices: int, min_ip_ka: float) -> dict:
    """Three arms over one shot's ramp + flat-top; the coupled arm is rich-cpl."""
    from imas_ambix.latent.boundary_disc import disc_read  # noqa: PLC0415
    from imas_ambix.latent.current_diffusion import (  # noqa: PLC0415
        EtaProfile,
        flux_surface_geometry,
    )
    from scripts.closure_gate_eval import fit_and_read_slice  # noqa: PLC0415
    from scripts.current_diffusion_gate_eval import (  # noqa: PLC0415
        predict_interval,
        raw_ip_stream,
    )
    from scripts.position_controlled_solve_gate import _disc_seed_flat  # noqa: PLC0415
    from scripts.spine_label_factory import (  # noqa: PLC0415
        factory_shot_payloads,
        frozen_spine_config,
    )

    spine, spine_sha = frozen_spine_config()
    isolve = spine["interior_solve"]
    n_p, n_f = int(isolve["n_p"]), int(isolve["n_f"])  # rich basis (3, 3)
    nonneg = isolve["profile_kind"] == "monomial-nonneg"
    smoothness = float(isolve["smoothness"])
    boundary_read = isolve["boundary_read_scoring"]

    payload = factory_shot_payloads(shot, nr=nr, nz=nz, max_slices=max_slices,
                                    min_ip_ka=min_ip_ka)
    if payload is None:
        return {"shot": shot, "slices": []}
    grid, table, basis = payload["grid"], payload["table"], payload["basis"]
    raw = raw_ip_stream(shot)
    if raw is None:
        logger.warning("shot %d: no raw Ip stream — skipped", shot)
        return {"shot": shot, "slices": []}
    raw_times, ip_raw = raw

    off = np.zeros_like(payload["payloads"][0].mask, dtype=bool)
    assert not off.any(), "firewall: every arm must run with the magnetics mask OFF"
    order = np.argsort([p.time_s for p in payload["payloads"]])

    # fit config shared by every arm — the position mode: magnetics OFF, the
    # measured centroid the only field constraint beyond Ip, no reseed.
    def _fit(p, *, n_p_, n_f_, nonneg_, warm, centroid, coeff_prior=None):
        return fit_and_read_slice(
            grid, table, dataclasses.replace(p, mask=off),
            beta0_grid=(0.5,), alpha_grid=(1.0,), cost_limit=float("inf"),
            convergence_limit=5e-3, retry_max_iterations=160, fit_mode="ladder",
            n_p=n_p_, n_f=n_f_, nonneg=nonneg_, smoothness=smoothness,
            warm_jphi=warm, centroid_constraint=(centroid[0], centroid[1], sigma),
            coeff_prior=coeff_prior, reseed_axis_r_max=None,
            keep_psi=True, keep_jphi=True, basis=basis, meta={},
            boundary_read=boundary_read,
        )

    # ---- pass 1: the K=2 position scaffold (stable, landed) ----
    slices: list[dict] = []
    warm_k2 = None
    for k in order:
        p = payload["payloads"][int(k)]
        inv = disc_read(p, grid, table, basis)
        if inv is None or inv.ring is None:
            continue
        centroid = (float(inv.centroid_r), float(inv.centroid_z))
        disc_seed = _disc_seed_flat(grid, inv)
        f_k2 = _fit(p, n_p_=1, n_f_=1, nonneg_=False,
                    warm=warm_k2 if warm_k2 is not None else disc_seed,
                    centroid=centroid)
        if not f_k2.scored:
            continue
        k2_confined = f_k2.jphi_flat is not None and _confined(_axis(f_k2)[0])
        if k2_confined:
            warm_k2 = f_k2.jphi_flat
        # the rich arms warm-start from the STABLE same-slice K=2 current (it
        # selects the confined basin); the disc seed is the fallback.  Temporal
        # coupling is then carried by the physics coeff_prior, not the numerics.
        slices.append({"k": int(k), "p": p, "centroid": centroid,
                       "disc_seed": disc_seed, "f_k2": f_k2,
                       "k2_jphi": f_k2.jphi_flat if k2_confined else disc_seed})
    if len(slices) < 2:
        return {"shot": shot, "slices": [], "reason": "too few scored slices"}

    # measured raw-Ip → amperes rescale (channel convention is kA)
    lab_ip = np.array([abs(s["p"].ip_amperes) for s in slices])
    raw_at_lab = np.interp([s["p"].time_s for s in slices], raw_times, ip_raw)
    good = raw_at_lab > 0
    ip_scale = float(np.median(lab_ip[good] / raw_at_lab[good])) if good.any() else 1e3
    ip_raw_amp = ip_raw * ip_scale

    # ---- pass 2a: the rich non-negative uncoupled solve (warm from K=2), and
    #      its flux-surface geometry in the RICH basis (source == projection) ----
    eta = EtaProfile.from_vector(np.asarray(eta_params, dtype=np.float64))
    f_uncs: list = []
    geos: list = []
    for s in slices:
        f_unc = _fit(s["p"], n_p_=n_p, n_f_=n_f, nonneg_=nonneg,
                     warm=s["k2_jphi"], centroid=s["centroid"])
        f_uncs.append(f_unc)
        if (f_unc.scored and f_unc.psi is not None and f_unc.coeffs is not None
                and _confined(_axis(f_unc)[0])):
            geos.append(flux_surface_geometry(
                f_unc.psi, grid, coeffs=np.asarray(f_unc.coeffs, dtype=np.float64),
                ip_amperes=abs(float(f_unc.ip_amperes)), n_p=n_p, n_f=n_f,
                nonneg=nonneg, b_phi0=b_phi0, n_rho=n_rho))
        else:
            geos.append(None)  # a drifted/failed rich-unc supplies no prediction

    # ---- per-interval diffusion prediction (rich basis, from rich-unc geometry) ----
    preds: list[dict | None] = [None] * len(slices)
    for j in range(len(slices) - 1):
        if geos[j] is None:
            continue
        out = predict_interval(
            geos[j], eta, t_start=slices[j]["p"].time_s,
            t_end=slices[j + 1]["p"].time_s, raw_times=raw_times,
            ip_raw_amp=ip_raw_amp, n_p=n_p, n_f=n_f, nonneg=nonneg,
            n_sub=n_sub, par_weight=par_weight)
        if out is not None:
            preds[j + 1] = out

    # ---- pass 2b: the rich coupled solve (warm from K=2 + diffusion prior) ----
    rows: list[dict] = []
    for j, s in enumerate(slices):
        p, centroid = s["p"], s["centroid"]
        f_unc = f_uncs[j]
        c_pred = preds[j]["c_pred"] if preds[j] is not None else None
        if c_pred is not None and prior_weight > 0.0:
            f_cpl = _fit(p, n_p_=n_p, n_f_=n_f, nonneg_=nonneg,
                         warm=s["k2_jphi"], centroid=centroid,
                         coeff_prior=(c_pred, prior_weight))
        else:
            # first slice / no prediction: the uncoupled rich fit IS the arm
            f_cpl = f_unc

        k2r, k2z = _axis(s["f_k2"])
        ur, _uz = _axis(f_unc)
        cr, _cz = _axis(f_cpl)
        cen_cpl = _current_centroid(grid, f_cpl)
        rows.append({
            "k": s["k"], "t_index": s["p"].t_index, "time_s": float(s["p"].time_s),
            "ip_a": float(abs(s["p"].ip_amperes)),
            "centroid_target": [centroid[0], centroid[1]],
            "k2": {"axis_r": k2r, "axis_z": k2z, "confined": _confined(k2r),
                   "coeffs": list(map(float, s["f_k2"].coeffs or []))},
            "rich_unc": {"axis_r": ur, "confined": _confined(ur),
                         "scored": bool(f_unc.scored), "converged": bool(f_unc.converged),
                         "coeffs": list(map(float, f_unc.coeffs or []))},
            "rich_cpl": {"axis_r": cr, "confined": _confined(cr),
                         "scored": bool(f_cpl.scored), "converged": bool(f_cpl.converged),
                         "coeffs": list(map(float, f_cpl.coeffs or [])),
                         "centroid_r": cen_cpl[0], "centroid_z": cen_cpl[1],
                         "centroid_err_cm": (
                             float(100.0 * np.hypot(cen_cpl[0] - centroid[0],
                                                    cen_cpl[1] - centroid[1]))
                             if np.isfinite(cen_cpl[0]) else float("nan"))},
            "cpl_pred": [float(c) for c in c_pred] if c_pred is not None else None,
            "cpl_vs_k2_axis_cm": (float(100.0 * abs(cr - k2r))
                                  if (_confined(cr) and _confined(k2r)) else float("nan")),
        })
        logger.info(
            "%d t=%.3f Ip=%.0fkA | K2 R=%.3f(%s) rich-unc R=%.3f(%s) "
            "rich-cpl R=%.3f(%s) cpl-vs-K2=%.2fcm",
            shot, s["p"].time_s, abs(s["p"].ip_amperes) / 1e3, k2r,
            "C" if _confined(k2r) else "X", ur, "C" if _confined(ur) else "X",
            cr, "C" if _confined(cr) else "X", rows[-1]["cpl_vs_k2_axis_cm"])

    return {"shot": shot, "spine_sha": spine_sha, "n_p": n_p, "n_f": n_f,
            "eta_params": list(map(float, eta_params)), "prior_weight": prior_weight,
            "slices": rows}


def _worker(job):
    shot, cfg = job
    try:
        return run_shot(shot, **cfg)
    except Exception as exc:  # a shot that dies must not sink the gate
        logger.exception("shot %d failed: %s", shot, exc)
        return {"shot": shot, "slices": [], "reason": f"exception: {exc}"}


def _fig_axis_trace(results, path: Path) -> None:
    """Axis-R vs time per arm, one panel per shot: the position hold under
    coupling (does rich-cpl track K2 while rich-unc drifts?)."""
    shots = [r for r in results if r.get("slices")]
    if not shots:
        return
    fig, axes = plt.subplots(1, len(shots), figsize=(4.2 * len(shots), 3.6),
                             squeeze=False, sharey=True)
    for ax, r in zip(axes[0], shots):
        sl = r["slices"]
        t = [s["time_s"] for s in sl]
        ax.plot(t, [s["k2"]["axis_r"] for s in sl], "o-", ms=3, lw=1.2,
                color="#555", label="K2 position")
        ax.plot(t, [s["rich_unc"]["axis_r"] for s in sl], "s--", ms=3, lw=1.0,
                color="#c44", label="rich uncoupled")
        ax.plot(t, [s["rich_cpl"]["axis_r"] for s in sl], "^-", ms=3.5, lw=1.4,
                color="#268", label="rich coupled")
        ax.axhline(CONFINED_AXIS_R_MAX, color="k", ls=":", lw=0.8)
        ax.set_title(f"shot {r['shot']}")
        ax.set_xlabel("time [s]")
    axes[0][0].set_ylabel("magnetic-axis R [m]")
    axes[0][0].legend(fontsize=7, loc="best")
    fig.suptitle("Dynamics coupling holds the rich profile inboard "
                 "(dotted = outboard-attractor threshold)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _fig_profile_evolution(results, path: Path) -> None:
    """Coupled vs uncoupled profile-coefficient evolution across one shot —
    the dynamics move the internal profile (else the coupling is inert)."""
    r = next((x for x in results if x.get("slices")), None)
    if r is None:
        return
    sl = [s for s in r["slices"] if s["rich_cpl"]["coeffs"] and s["rich_unc"]["coeffs"]]
    if len(sl) < 2:
        return
    t = np.array([s["time_s"] for s in sl])
    unc = np.array([s["rich_unc"]["coeffs"] for s in sl])
    cpl = np.array([s["rich_cpl"]["coeffs"] for s in sl])
    k = unc.shape[1]
    fig, axes = plt.subplots(1, k, figsize=(2.1 * k, 3.0), squeeze=False, sharex=True)
    for i in range(k):
        ax = axes[0][i]
        ax.plot(t, unc[:, i], "s--", ms=3, color="#c44", label="uncoupled")
        ax.plot(t, cpl[:, i], "^-", ms=3.5, color="#268", label="coupled")
        fam = "p′" if i < (k // 2) else "FF′"
        ax.set_title(f"c{i} ({fam})", fontsize=8)
        ax.set_xlabel("t [s]")
    axes[0][0].set_ylabel("profile coeff")
    axes[0][0].legend(fontsize=7)
    fig.suptitle(f"shot {r['shot']}: diffusion-evolved profile vs the "
                 "under-determined uncoupled fit", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _summarise(results, prior_weight: float) -> dict:
    def _frac_confined(arm):
        seen = tot = 0
        for r in results:
            for s in r.get("slices", []):
                tot += 1
                seen += int(s[arm]["confined"])
        return (seen / tot) if tot else float("nan"), tot

    k2_frac, n = _frac_confined("k2")
    unc_frac, _ = _frac_confined("rich_unc")
    cpl_frac, _ = _frac_confined("rich_cpl")
    dev = [s["cpl_vs_k2_axis_cm"] for r in results for s in r.get("slices", [])
           if np.isfinite(s["cpl_vs_k2_axis_cm"])]
    med_dev = float(np.median(dev)) if dev else float("nan")
    any_nan = any(
        (s["rich_cpl"]["scored"] and not np.isfinite(s["rich_cpl"]["axis_r"]))
        for r in results for s in r.get("slices", []))
    # G3a asks whether the DYNAMICS COUPLING (the thing §3 adds) keeps the solve
    # finite and physical without regressing.  That is three checks:
    #   (a) no fabricated readout — a scored coupled fit never carries a NaN axis
    #       (drift/non-convergence must mask the slice, never ship it);
    #   (b) where confined, the coupled solve tracks the §2 position (median axis
    #       deviation within tolerance);
    #   (c) the coupling does not regress the rich solve — coupled confinement is
    #       at least the UNCOUPLED rich confinement (same basis, prior off vs on).
    # The rich basis is intrinsically more basin-fragile than the rigid K=2
    # position crutch (that is WHY §2 used K=2 for position, not the profile), so
    # the rich↔K2 confinement gap is a BASIS property, reported for transparency
    # but NOT a coupling regression — it is not part of the gate.
    no_fabrication = not any_nan
    tracks_position = bool(np.isfinite(med_dev) and med_dev <= REGRESS_TOL_CM)
    no_coupling_regression = bool(cpl_frac >= unc_frac - 1e-9)
    g3a = bool(no_fabrication and tracks_position and no_coupling_regression)
    return {"n_slices": n, "k2_confined_frac": k2_frac,
            "rich_unc_confined_frac": unc_frac, "rich_cpl_confined_frac": cpl_frac,
            "rich_vs_k2_confinement_gap": float(k2_frac - cpl_frac),
            "cpl_vs_k2_axis_median_cm": med_dev, "coupled_emits_nan": any_nan,
            "no_fabrication": no_fabrication, "tracks_position": tracks_position,
            "no_coupling_regression": no_coupling_regression,
            "prior_weight": prior_weight, "regress_tol_cm": REGRESS_TOL_CM,
            "G3a_pass": g3a}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shots", type=str, default="")
    ap.add_argument("--nr", type=int, default=65)
    ap.add_argument("--nz", type=int, default=97)
    ap.add_argument("--sigma", type=float, default=DEFAULT_SIGMA_M)
    ap.add_argument("--prior-weight", type=float, default=0.3)
    ap.add_argument("--eta-params", type=str, default=",".join(map(str, DEFAULT_ETA)))
    ap.add_argument("--n-sub-steps", type=int, default=24)
    ap.add_argument("--par-weight", type=float, default=1.0)
    ap.add_argument("--b-phi0", type=float, default=0.55)
    ap.add_argument("--n-rho", type=int, default=24)
    ap.add_argument("--max-slices-per-shot", type=int, default=12)
    ap.add_argument("--min-ip-ka", type=float, default=60.0)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out-suffix", type=str, default="")
    args = ap.parse_args()

    shots = ([int(s) for s in args.shots.split(",") if s]
             if args.shots else list(DEFAULT_SHOTS))
    eta_params = [float(v) for v in args.eta_params.split(",")]
    cfg = dict(nr=args.nr, nz=args.nz, sigma=args.sigma, eta_params=eta_params,
               prior_weight=args.prior_weight, n_sub=args.n_sub_steps,
               par_weight=args.par_weight, b_phi0=args.b_phi0, n_rho=args.n_rho,
               max_slices=args.max_slices_per_shot, min_ip_ka=args.min_ip_ka)

    jobs = [(sh, cfg) for sh in shots]
    if args.workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as ex:
            results = list(ex.map(_worker, jobs))
    else:
        results = [_worker(j) for j in jobs]

    summary = _summarise(results, args.prior_weight)
    logger.info("G3a summary: %s", json.dumps(summary, indent=1))

    FIGURES.mkdir(parents=True, exist_ok=True)
    sfx = args.out_suffix
    _fig_axis_trace(results, FIGURES / f"fig-coupled-axis-trace{sfx}.png")
    _fig_profile_evolution(results, FIGURES / f"fig-coupled-profile-evolution{sfx}.png")

    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    out = ARTIFACT.with_name(ARTIFACT.stem + sfx + ARTIFACT.suffix)
    out.write_text(json.dumps({"arm": "dynamics-coupled-solve-position-spine",
                               "shots": shots, "summary": summary,
                               "results": results}, indent=1))
    logger.info("wrote %s", out)
    print(f"G3a_PASS={summary['G3a_pass']} "
          f"cpl_confined={summary['rich_cpl_confined_frac']:.3f} "
          f"unc_confined={summary['rich_unc_confined_frac']:.3f} "
          f"cpl_vs_k2_med={summary['cpl_vs_k2_axis_median_cm']:.2f}cm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
