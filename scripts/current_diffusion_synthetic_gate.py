#!/usr/bin/env python
"""Synthetic exact-recovery gate for the current-diffusion temporal prior.

Sequences of manufactured equilibria whose profile coefficients EVOLVE UNDER
THE RESISTIVE DIFFUSION PHYSICS with a KNOWN η(ψ_N) and a ramping Ip: at
each step the truth equilibrium's own flux-surface geometry drives the ψ
diffusion over the interval (fine sub-steps, finer ρ̂ grid than the recovery
arm uses — no inverse crime at the discretisation level), the evolved
(j_tot, ⟨J·B⟩) profiles project onto the ladder basis, and the projected
coefficients manufacture the next truth.  The generator also integrates the
exact flux-consumption ledger, so the TRUE windowed Ejima coefficient of
every sequence is known.

Arms, each fitted against the same corrupted synthetic payloads:

* SPINE — the frozen classical ladder, slices independent (warm chain only);
* DIFFUSION — the same ladder with each slice's coefficients soft-centred on
  the diffusion prediction evolved from the PREVIOUS SPINE fit (η exact —
  the oracle ceiling: does temporal linking through known physics recover
  the degenerate split?);
* DIFFUSION-FITTED-η — as above with η re-fitted from the sequences by the
  real-data machinery (the honest transfer arm).

Score: the p′-group current-fraction error |split − truth| per slice
(median), the standing exact-truth bar of the plan: the diffusion arm must
beat the recorded independent-slice spine at 0.120 median |err| (and the
spine arm re-scored on these same sequences, as the paired comparison).
The recovered windowed Ejima coefficient is reported against the known one.

Artifacts: imas_ambix/latent/artifacts/patch_gate/
           current_diffusion_synth_gate[-tag].json
Figures:   docs/figures/temporal-physics-spine/
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np

from imas_ambix.latent.current_diffusion import (
    EtaProfile,
    basis_projection_images,
    diffuse_psi,
    ejima_coefficient,
    flux_budget,
    flux_surface_geometry,
    predicted_current,
    project_coefficients,
)
from imas_ambix.latent.gs_solve import build_passive_sidecar
from imas_ambix.latent.synthetic_truth import build_campaign, manufacture
from scripts.closure_gate_eval import fit_and_read_slice
from scripts.spine_label_factory import frozen_spine_config
from scripts.synthetic_eddy_pretrain import split_fraction

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("current_diffusion_synth_gate")

ARTIFACTS = Path("imas_ambix/latent/artifacts/patch_gate")
FIGURES = Path("docs/figures/temporal-physics-spine")
CAMPAIGN_SHOT = 11766  # train-split shot: geometry + noise floor only

_CAMPAIGN: dict[str, object] = {}


def _campaign():
    if "c" not in _CAMPAIGN:
        _CAMPAIGN["c"] = build_campaign(CAMPAIGN_SHOT, nr=65, nz=97)
    return _CAMPAIGN["c"]


def generate_sequence(job: tuple) -> dict | None:
    """One diffusion-consistent synthetic shot with known η and ledger."""
    seed, n_steps, eta_vec, n_rho_truth, n_sub_truth = job
    rng = np.random.default_rng(seed)
    campaign = _campaign()
    grid = campaign.grid
    eta_true = EtaProfile.from_vector(np.asarray(eta_vec, dtype=np.float64))

    from imas_ambix.latent.synthetic_truth import build_confining_i_pf

    i_pf = build_confining_i_pf(campaign.fwd, 6.0e4)
    times = 0.05 + np.cumsum(rng.uniform(0.015, 0.03, size=n_steps))
    # ramping Ip exercises both consumption channels (skin drive + resistive)
    ip0 = 4.5e5 * rng.uniform(0.9, 1.1)
    ip_seq = ip0 * np.linspace(rng.uniform(0.65, 0.8), 1.0, n_steps)

    c0 = np.clip(rng.uniform(0.2, 1.0, size=6), 0.05, None)
    rows = []
    coeffs = c0
    warm = None
    ledger = {"d_psi_res": 0.0, "d_psi_int": 0.0, "d_psi_bdry": 0.0}
    for t in range(n_steps):
        truth = manufacture(
            campaign,
            coeffs=coeffs,
            n_p=3,
            n_f=3,
            nonneg_basis=True,
            i_pf=i_pf * (0.85 + 0.15 * (t + 1) / n_steps),
            ip_amperes=float(ip_seq[t]),
            seed=int(seed * 1000 + t),
            warm_jphi=warm,
            continuation=warm is None,
        )
        if not truth.confined:
            logger.warning("seq %d step %d not confined — dropped", seed, t)
            return None
        warm = np.zeros(grid.flat_r.size)
        warm[grid.cells] = truth.cell_currents / (grid.dr * grid.dz)
        rows.append(
            {"truth": truth, "time_s": float(times[t]), "coeffs_true": coeffs.copy()}
        )
        if t == n_steps - 1:
            break
        # evolve the truth to the next step under the KNOWN diffusion physics
        geo = flux_surface_geometry(
            truth.psi,
            grid,
            coeffs=coeffs,
            ip_amperes=float(ip_seq[t]),
            n_p=3,
            n_f=3,
            nonneg=True,
            n_rho=n_rho_truth,
        )
        if geo is None:
            logger.warning("seq %d step %d: geometry failed — dropped", seed, t)
            return None
        t_sub = np.linspace(times[t], times[t + 1], n_sub_truth)
        ip_sub = np.interp(t_sub, times, ip_seq)
        step = diffuse_psi(geo, eta_true, t_grid=t_sub, ip_of_t=ip_sub)
        pred = predicted_current(
            geo, step["psi_face"][-1], step["psidot_face"], eta_true
        )
        images = basis_projection_images(geo, geo.s_k, n_p=3, n_f=3, nonneg=True)
        c_next = project_coefficients(
            geo, images, pred["j_tor"], pred["j_par_b"], nonneg=True
        )
        if c_next is None:
            return None
        coeffs = np.clip(c_next, 1e-3, None)
        b = flux_budget(step, geo)
        ledger["d_psi_res"] += b["d_psi_axis"]
        ledger["d_psi_int"] += b["d_psi_internal"]
        ledger["d_psi_bdry"] += b["d_psi_bdry"]
    d_ip = float(ip_seq[-1] - ip_seq[0])
    ledger["ejima_true"] = (
        ejima_coefficient(ledger["d_psi_res"], d_ip, grid.r0) if d_ip > 0 else None
    )
    ledger["d_ip"] = d_ip
    return {
        "seed": int(seed),
        "rows": rows,
        "times": [float(v) for v in times],
        "ip_seq": [float(v) for v in ip_seq],
        "ledger": ledger,
    }


def fit_sequence(job: tuple) -> dict:
    """Both fit arms for one sequence: independent spine + diffusion-chained."""
    seq, args_d = job
    campaign = _campaign()
    grid = campaign.grid
    spine, _sha = frozen_spine_config()
    isolve = spine["interior_solve"]
    sidecar = build_passive_sidecar(
        campaign.table,
        campaign.grid,
        g_passive=campaign.passive_g_sens,
        sensor_scale=campaign.scale,
        k=int(spine["interior_solve"]["passive_k"]),
    )
    fit_kw = dict(
        beta0_grid=(0.5,),
        alpha_grid=(1.0,),
        cost_limit=float("inf"),
        convergence_limit=5e-3,
        retry_max_iterations=160,
        fit_mode="ladder",
        n_p=3,
        n_f=3,
        smoothness=float(isolve["smoothness"]),
        nonneg=True,
        passive=sidecar,
        passive_ridge=1.0,
        reseed_axis_r_max=float(isolve["reseed_axis_r_max"]),
        keep_psi=True,
        keep_jphi=True,
    )
    eta = EtaProfile.from_vector(np.asarray(args_d["eta_vector"], dtype=np.float64))
    w = float(args_d["prior_weight"])
    times = np.asarray(seq["times"], dtype=np.float64)
    ip_seq = np.asarray(seq["ip_seq"], dtype=np.float64)

    # arm 1: the frozen spine, slices independent (warm chain only)
    spine_fits = []
    warm = None
    t0 = time.perf_counter()
    for r in seq["rows"]:
        f = fit_and_read_slice(
            grid, campaign.table, r["truth"].to_payload(), warm_jphi=warm, **fit_kw
        )
        if f.scored and f.converged and f.jphi_flat is not None:
            warm = f.jphi_flat
        spine_fits.append(f)

    # arm 2: diffusion-chained — the prior centre evolves the PREVIOUS spine
    # fit (pass-1-driven, mirroring the real-data harness; no feedback)
    dyn_fits = [spine_fits[0]]
    for j in range(1, len(seq["rows"])):
        f_prev = spine_fits[j - 1]
        c_pred = None
        if f_prev.scored and f_prev.psi is not None and f_prev.coeffs:
            geo = flux_surface_geometry(
                f_prev.psi,
                grid,
                coeffs=np.asarray(f_prev.coeffs, dtype=np.float64),
                ip_amperes=abs(float(f_prev.ip_amperes)),
                n_p=3,
                n_f=3,
                nonneg=True,
                n_rho=int(args_d["n_rho"]),
            )
            if geo is not None:
                t_sub = np.linspace(times[j - 1], times[j], int(args_d["n_sub_steps"]))
                ip_sub = np.interp(t_sub, times, ip_seq)
                step = diffuse_psi(geo, eta, t_grid=t_sub, ip_of_t=ip_sub)
                pred = predicted_current(
                    geo, step["psi_face"][-1], step["psidot_face"], eta
                )
                images = basis_projection_images(
                    geo, geo.s_k, n_p=3, n_f=3, nonneg=True
                )
                c_pred = project_coefficients(
                    geo,
                    images,
                    pred["j_tor"],
                    pred["j_par_b"],
                    nonneg=True,
                    par_weight=float(args_d["par_weight"]),
                )
        if c_pred is None or w <= 0:
            dyn_fits.append(spine_fits[j])
            continue
        f2 = fit_and_read_slice(
            grid,
            campaign.table,
            seq["rows"][j]["truth"].to_payload(),
            warm_jphi=spine_fits[j].jphi_flat if spine_fits[j].scored else None,
            coeff_prior=(c_pred, w),
            **fit_kw,
        )
        dyn_fits.append(f2 if f2.scored else spine_fits[j])

    # score the p′-group current fraction per slice against the exact truth
    r_cells = grid.flat_r[grid.cells]
    out_rows = []
    for r, fs, fd in zip(seq["rows"], spine_fits, dyn_fits, strict=True):
        truth = r["truth"]
        true_psin = (
            (truth.psi.ravel()[grid.cells] - truth.axis_psi)
            / (truth.boundary_psi - truth.axis_psi)
        ).clip(0.0, 1.5)
        s_true = split_fraction(r["coeffs_true"], true_psin, r_cells, grid.r0)

        def _split(fit):
            if not fit.scored or not fit.coeffs or fit.psi is None:
                return float("nan")
            from imas_ambix.latent.current_diffusion import (
                reconstruct_profile_scales,
            )

            rec = reconstruct_profile_scales(
                fit.psi, grid, abs(float(fit.ip_amperes)), n_p=3, n_f=3, nonneg=True
            )
            return split_fraction(
                np.asarray(fit.coeffs), rec["psi_n"][grid.cells], r_cells, grid.r0
            )

        out_rows.append(
            {
                "s_true": float(s_true),
                "s_spine": float(_split(fs)),
                "s_dyn": float(_split(fd)),
                "cost_spine": float(fs.cost) if fs.scored else None,
                "cost_dyn": float(fd.cost) if fd.scored else None,
            }
        )
    return {
        "seed": seq["seed"],
        "rows": out_rows,
        "ledger": seq["ledger"],
        "wall_s": time.perf_counter() - t0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-sequences", type=int, default=8)
    ap.add_argument("--n-steps", type=int, default=12)
    ap.add_argument(
        "--eta-true",
        type=str,
        default="-7.15,1.5,1.5",
        help="known truth eta: log10(eta0), contrast, shape",
    )
    ap.add_argument(
        "--eta-arm",
        type=str,
        default="",
        help="recovery-arm eta (empty = exact truth — the oracle arm)",
    )
    ap.add_argument("--prior-weight", type=float, default=0.3)
    ap.add_argument("--n-rho", type=int, default=24)
    ap.add_argument("--n-sub-steps", type=int, default=24)
    ap.add_argument("--n-rho-truth", type=int, default=40)
    ap.add_argument("--n-sub-truth", type=int, default=80)
    ap.add_argument("--par-weight", type=float, default=1.0)
    ap.add_argument("--split-bar", type=float, default=0.120)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed0", type=int, default=2000)
    ap.add_argument("--out-suffix", type=str, default="")
    args = ap.parse_args()

    eta_true = [float(v) for v in args.eta_true.split(",")]
    eta_arm = (
        [float(v) for v in args.eta_arm.split(",")] if args.eta_arm else list(eta_true)
    )
    jobs = [
        (args.seed0 + k, args.n_steps, eta_true, args.n_rho_truth, args.n_sub_truth)
        for k in range(args.n_sequences)
    ]
    ctx = multiprocessing.get_context("fork")
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
        seqs = [s for s in pool.map(generate_sequence, jobs) if s is not None]
    logger.info(
        "generated %d/%d diffusion-consistent sequences in %.0f s",
        len(seqs),
        len(jobs),
        time.perf_counter() - t0,
    )
    if not seqs:
        raise SystemExit("no sequences generated")

    args_d = {
        "eta_vector": eta_arm,
        "prior_weight": args.prior_weight,
        "n_rho": args.n_rho,
        "n_sub_steps": args.n_sub_steps,
        "par_weight": args.par_weight,
    }
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
        fitted = list(pool.map(fit_sequence, [(s, args_d) for s in seqs]))

    err_spine, err_dyn = [], []
    for f in fitted:
        for r in f["rows"]:
            if np.isfinite(r["s_spine"]):
                err_spine.append(abs(r["s_spine"] - r["s_true"]))
            if np.isfinite(r["s_dyn"]):
                err_dyn.append(abs(r["s_dyn"] - r["s_true"]))
    med_spine = float(np.median(err_spine)) if err_spine else float("nan")
    med_dyn = float(np.median(err_dyn)) if err_dyn else float("nan")
    gate_bar = bool(med_dyn < args.split_bar)
    gate_paired = bool(med_dyn < med_spine)

    result = {
        "arm": "current-diffusion-synthetic-split-recovery",
        "eta_true": eta_true,
        "eta_arm": eta_arm,
        "eta_arm_is_oracle": eta_arm == eta_true,
        "prior_weight": args.prior_weight,
        "par_weight": args.par_weight,
        "n_sequences": len(fitted),
        "n_slices": len(err_spine),
        "split_bar": args.split_bar,
        "split_abs_err_spine_median": med_spine,
        "split_abs_err_dyn_median": med_dyn,
        "split_abs_err_spine_mean": float(np.mean(err_spine)) if err_spine else None,
        "split_abs_err_dyn_mean": float(np.mean(err_dyn)) if err_dyn else None,
        "gate_beats_bar": gate_bar,
        "gate_beats_spine_paired": gate_paired,
        "ejima_true": [
            f["ledger"]["ejima_true"]
            for f in fitted
            if f["ledger"]["ejima_true"] is not None
        ],
        "wall_s_total": float(sum(f["wall_s"] for f in fitted)),
    }
    tag = f"-{args.out_suffix}" if args.out_suffix else ""
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    out_json = ARTIFACTS / f"current_diffusion_synth_gate{tag}.json"
    out_json.write_text(json.dumps(result, indent=2))

    import matplotlib.pyplot as plt

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    axes[0].hist(err_spine, bins=24, alpha=0.6, color="#4477aa", label="spine")
    axes[0].hist(err_dyn, bins=24, alpha=0.6, color="#228833", label="diffusion")
    axes[0].axvline(
        args.split_bar, color="k", ls="--", lw=1, label=f"bar {args.split_bar}"
    )
    axes[0].set_xlabel("|split − truth|")
    axes[0].set_title(
        f"split recovery: spine {med_spine:.3f} vs diffusion {med_dyn:.3f} "
        f"→ {'PASS' if (gate_bar and gate_paired) else 'FAIL'}"
    )
    axes[0].legend(fontsize=8)
    s_t = [r["s_true"] for f in fitted for r in f["rows"]]
    s_s = [r["s_spine"] for f in fitted for r in f["rows"]]
    s_d = [r["s_dyn"] for f in fitted for r in f["rows"]]
    axes[1].scatter(s_t, s_s, s=14, alpha=0.6, color="#4477aa", label="spine")
    axes[1].scatter(s_t, s_d, s=14, alpha=0.6, color="#228833", label="diffusion")
    axes[1].plot([0, 1], [0, 1], "k--", lw=1)
    axes[1].set_xlabel("true p′ fraction")
    axes[1].set_ylabel("recovered")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    out_fig = FIGURES / f"fig-current-diffusion-synth-gate{tag}.png"
    fig.savefig(out_fig, dpi=120)
    plt.close(fig)

    logger.info(
        "synthetic split gate: spine %.3f vs diffusion %.3f (bar %.3f) — "
        "beats bar %s, beats spine %s | %s %s",
        med_spine,
        med_dyn,
        args.split_bar,
        gate_bar,
        gate_paired,
        out_json,
        out_fig,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
