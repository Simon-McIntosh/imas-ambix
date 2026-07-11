#!/usr/bin/env python
"""Synthetic-truth identifiability harness for the force-balance spine.

The standing instrument that would have caught the null-space fill and the
outboard-attractor drift automatically. Manufactured equilibria with KNOWN
Tier-2 coefficients (:mod:`imas_ambix.latent.synthetic_truth`) are pushed
through the inverse arms and four questions are answered, all firewall-clean
(no EFIT anywhere — truth is the injected coefficient):

RECOVERY
    Push synthetic payloads through the arms (free patch inverse, physics-prior
    arm, closure grid, closure continuous, ladder K=2, ladder + passive) and
    report recovered-vs-injected coefficient error with bootstrap CIs: fitted
    (β0, α) for the closure arms, axis for every arm, and the passive-mode
    amplitudes for the sidecar arm.

ALIASING
    Inject one perturbation class alone (rotation with a static arm; calibration
    corruption with a physics arm; passive currents with a no-sidecar arm) and
    measure what each mis-modelled term leaks INTO (Δβ0, Δα, Δaxis, ΔLCFS). The
    rotation → p₀′ contamination magnitude at MAST geometry is the headline.

BASIN
    For a grid of (β0, α) truths × seed strategies (midplane Gaussian, seed_z0
    offsets, boundary-continuation bootstrap, free-sign scout, warm-start from a
    neighbour), map which cold solves reach the confined branch vs the outboard
    attractor; plus a damped-relaxation arm as a candidate principled
    replacement for the scout stage. Verdict: does anything retire scouting?

INFORMATION
    Per sensor set (magnetics now; the code is structured so interferometer /
    SXR / CXRS rows slot in), the whitened-Jacobian effective rank and the
    per-coefficient Fisher sensitivity — the quantitative input for
    equilibrium-topology-fidelity thread 2.

Artifacts:  imas_ambix/latent/artifacts/patch_gate/identifiability_*.json|npz
Figures:    docs/figures/force-balance-spine/fig-identifiability-*.png,
            fig-basin-*.png
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from imas_ambix.gs.operator import COIL_MODEL_VERSION
from imas_ambix.latent import synthetic_truth as st
from imas_ambix.latent.gs_solve import (
    build_passive_sidecar,
    fit_profile,
    fit_profile_continuous,
    fit_profile_ladder,
    solve_equilibrium,
    solve_equilibrium_bootstrapped,
    solve_equilibrium_lsq,
)
from imas_ambix.latent.patch_inverse import InverseConfig, invert_slices
from imas_ambix.latent.structure_residual import fit_flux_functions

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("identifiability_harness")

ARTIFACTS = Path("imas_ambix/latent/artifacts/patch_gate")
FIGURES = Path("docs/figures/force-balance-spine")

# a realistic (β0, α) truth range: pressure-dominated → FF′-dominated, peaked
# → broad (the physics-spine-findings measured span).
BETA0_RANGE = (0.25, 0.75)
ALPHA_RANGE = (0.8, 2.2)


# ---------------------------------------------------------------------------
# arm recovery — each arm reads a synthetic payload and reports what it can
# ---------------------------------------------------------------------------


def _read_axis_from_psi(campaign: st.Campaign, i_cell, i_pf):
    """Sign-aware conductor-clear axis (R, Z) of the ψ these currents make."""
    from imas_ambix.latent.gs_solve import _read_axis

    psi2d = campaign.basis.psi_grid_2d_np(np.asarray(i_cell), np.asarray(i_pf))
    med = float(np.median(psi2d))
    ax_p, psi_p = _read_axis(psi2d, campaign.grid, +1.0)
    ax_n, psi_n = _read_axis(psi2d, campaign.grid, -1.0)
    return ax_p if abs(psi_p - med) >= abs(psi_n - med) else ax_n


def _closures_from_currents(campaign: st.Campaign, i_cell, i_pf, *, form="affine-r2"):
    """Recover p′ (a) / FF′/μ0 (b) [and c, the R⁴ rotation coefficient] on the
    core cells of the ψ these currents make, via the structure-residual fit."""
    grid = campaign.grid
    psi_cells = campaign.basis.psi_cells_np(np.asarray(i_cell), np.asarray(i_pf))
    r_c = grid.flat_r[grid.cells]
    z_c = grid.flat_z[grid.cells]
    jphi_c = np.asarray(i_cell) / (grid.dr * grid.dz)
    fit = fit_flux_functions(
        torch.as_tensor(psi_cells),
        torch.as_tensor(r_c),
        torch.as_tensor(jphi_c),
        form=form,
        z_c=torch.as_tensor(z_c),
        connectivity="locality",
    )
    mass = np.asarray(fit.weight_mass)
    keep = mass > 1e-3 * (mass.max() if mass.size else 0.0)

    def _wmean(v):
        v = np.asarray(v)
        return float(np.sum(v[keep] * mass[keep]) / max(mass[keep].sum(), 1e-30))

    out = {"p_prime_wmean": _wmean(fit.a_k), "ffprime_wmean": _wmean(fit.b_k)}
    if fit.c_k is not None:
        out["c_r4_wmean"] = _wmean(np.asarray(fit.c_k))
    return out


def _invert_batch(campaign, payloads, cfg, device):
    return invert_slices(campaign.basis, payloads, cfg, device=device)


#: frozen P3-winner inverse config for the free / priors arms (patch gate).
_P3 = dict(policy="discrepancy", lambda_fb=3.0, misfit_ratio=1.5, lambda_max=100.0)


def recover_free(campaign, truths, *, iters, device, priors=False):
    """Free (or physics-prior) patch inverse over a batch of synthetic slices."""
    payloads = [t.to_payload() for t in truths]
    prior_kw = {}
    if priors:
        prior_kw = dict(
            sign_prior="softplus",
            support_prior=True,
            limiter_r=campaign.grid.limiter_r,
            limiter_z=campaign.grid.limiter_z,
        )
    cfg = InverseConfig(iters=iters, connectivity="locality", **_P3, **prior_kw)
    inv = _invert_batch(campaign, payloads, cfg, device)
    rows = []
    for t, r in zip(truths, inv, strict=True):
        axis = _read_axis_from_psi(campaign, r.i_cell, t.i_pf)
        clo = _closures_from_currents(campaign, r.i_cell, t.i_pf)
        rows.append(
            {
                "axis_r": float(axis[0]),
                "axis_z": float(axis[1]),
                "misfit": float(r.misfit),
                "negative_fraction": float(r.negative_fraction),
                **clo,
            }
        )
    return rows


def recover_closure(campaign, truths, *, mode, iters, device, passive_k=0):
    """Closure-family arm (grid / continuous / ladder [+passive]) per slice."""
    grid, table = campaign.grid, campaign.table
    rows = []
    sidecar = None
    if passive_k:
        sidecar = build_passive_sidecar(
            table,
            grid,
            g_passive=campaign.passive_g_sens,
            sensor_scale=campaign.scale,
            k=passive_k,
        )
    for t in truths:
        p = t.to_payload()
        kw = dict(
            i_pf=p.i_pf,
            ip_amperes=p.ip_amperes,
            measured=p.measured,
            vacuum_prediction=p.vacuum,
            sensor_scale=p.scale,
            sensor_mask=p.mask,
        )
        rec = {"beta0": None, "alpha": None, "z0": None, "coeffs": None}
        if mode == "grid":
            fit = fit_profile(grid, table, **kw)
            if fit is not None:
                rec.update(
                    beta0=fit.beta0,
                    alpha=fit.alpha,
                    axis=fit.result.axis,
                    cost=fit.cost,
                )
        elif mode == "continuous":
            fit = fit_profile_continuous(grid, table, fit_z0=True, maxfev=40, **kw)
            if fit is not None:
                rec.update(
                    beta0=fit.beta0,
                    alpha=fit.alpha,
                    z0=fit.z0,
                    axis=fit.result.axis,
                    cost=fit.cost,
                )
        elif mode == "ladder":
            fit = fit_profile_ladder(grid, table, n_p=1, n_f=1, **kw)
            rec.update(
                coeffs=np.asarray(fit.coeffs).tolist(),
                axis=fit.result.axis,
                cost=fit.cost,
            )
        elif mode == "ladder_passive":
            fit = fit_profile_ladder(
                grid, table, n_p=1, n_f=1, passive=sidecar, passive_ridge=1.0, **kw
            )
            rec.update(
                coeffs=np.asarray(fit.coeffs).tolist(),
                axis=fit.result.axis,
                cost=fit.cost,
                passive_amplitudes=np.asarray(fit.passive_amplitudes).tolist()
                if fit.passive_amplitudes is not None
                else None,
            )
        else:
            raise ValueError(f"unknown closure mode {mode!r}")
        ax = rec.get("axis")
        rec["axis_r"] = float(ax[0]) if ax is not None else float("nan")
        rec["axis_z"] = float(ax[1]) if ax is not None else float("nan")
        rec.pop("axis", None)
        rows.append(rec)
    return rows


# ---------------------------------------------------------------------------
# bootstrap CI over the sampled truths
# ---------------------------------------------------------------------------


def _boot_ci(values, *, n_boot=2000, seed=0, reducer=np.nanmean):
    v = np.asarray(values, dtype=np.float64)
    finite = v[np.isfinite(v)]
    if finite.size < 2:
        return {"point": float(reducer(v)) if finite.size else None, "ci": [None, None]}
    rng = np.random.default_rng(seed)
    draws = np.array(
        [
            reducer(rng.choice(finite, size=finite.size, replace=True))
            for _ in range(n_boot)
        ]
    )
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    return {
        "point": float(reducer(finite)),
        "ci": [float(lo), float(hi)],
        "n": int(finite.size),
    }


# ---------------------------------------------------------------------------
# mode: RECOVERY
# ---------------------------------------------------------------------------


def _sample_truths(campaign, warm, *, n, seed, noise, ip):
    rng = np.random.default_rng(seed)
    truths, betas, alphas = [], [], []
    for i in range(n):
        b = float(rng.uniform(*BETA0_RANGE))
        a = float(rng.uniform(*ALPHA_RANGE))
        t = st.manufacture(
            campaign,
            beta0=b,
            alpha=a,
            ip_amperes=ip,
            warm_jphi=warm,
            noise=noise,
            seed=seed * 1000 + i,
        )
        if not t.confined:
            continue
        truths.append(t)
        betas.append(b)
        alphas.append(a)
    return truths, np.array(betas), np.array(alphas)


def run_recovery(args, campaign, warm):
    arms = (
        args.arms.split(",")
        if args.arms
        else [
            "free",
            "priors",
            "closure_grid",
            "closure_cont",
            "ladder",
            "ladder_passive",
        ]
    )
    truths, betas, alphas = _sample_truths(
        campaign,
        warm,
        n=args.n_samples,
        seed=args.seed,
        noise=not args.no_noise,
        ip=args.ip,
    )
    logger.info("recovery: %d/%d confined truths", len(truths), args.n_samples)
    report = {
        "coil_model_version": COIL_MODEL_VERSION,
        "shot": int(args.shot),
        "n_confined": len(truths),
        "arms": {},
    }
    per_arm_axis_err = {}
    for arm in arms:
        t0 = time.perf_counter()
        if arm == "free":
            rows = recover_free(campaign, truths, iters=args.iters, device=args.device)
        elif arm == "priors":
            rows = recover_free(
                campaign, truths, iters=args.iters, device=args.device, priors=True
            )
        elif arm == "closure_grid":
            rows = recover_closure(
                campaign, truths, mode="grid", iters=args.iters, device=args.device
            )
        elif arm == "closure_cont":
            rows = recover_closure(
                campaign,
                truths,
                mode="continuous",
                iters=args.iters,
                device=args.device,
            )
        elif arm == "ladder":
            rows = recover_closure(
                campaign, truths, mode="ladder", iters=args.iters, device=args.device
            )
        elif arm == "ladder_passive":
            rows = recover_closure(
                campaign,
                truths,
                mode="ladder_passive",
                iters=args.iters,
                device=args.device,
                passive_k=args.passive_k,
            )
        else:
            raise ValueError(f"unknown arm {arm!r}")
        axis_err = np.array(
            [
                np.hypot(r["axis_r"] - t.axis[0], r["axis_z"] - t.axis[1])
                for r, t in zip(rows, truths, strict=True)
            ]
        )
        per_arm_axis_err[arm] = axis_err
        entry = {"axis_error_m": _boot_ci(axis_err, seed=args.seed)}
        if rows and rows[0].get("beta0") is not None:
            db = np.array(
                [
                    r["beta0"] - t.beta0_true
                    for r, t in zip(rows, truths, strict=True)
                    if r["beta0"] is not None
                ]
            )
            da = np.array(
                [
                    r["alpha"] - t.alpha_true
                    for r, t in zip(rows, truths, strict=True)
                    if r["alpha"] is not None
                ]
            )
            entry["beta0_bias"] = _boot_ci(db, seed=args.seed)
            entry["alpha_bias"] = _boot_ci(da, seed=args.seed)
            entry["beta0_abs_err"] = _boot_ci(np.abs(db), seed=args.seed)
        report["arms"][arm] = entry
        logger.info(
            "[%s] axis_err=%.3f m (%d slices) %.0fs",
            arm,
            entry["axis_error_m"]["point"] or float("nan"),
            len(rows),
            time.perf_counter() - t0,
        )

    (ARTIFACTS / f"identifiability_recovery{args.out_suffix}.json").write_text(
        json.dumps(report, indent=2)
    )
    _plot_recovery(
        per_arm_axis_err, betas, alphas, per_arm=report["arms"], suffix=args.out_suffix
    )
    return report


def _plot_recovery(per_arm_axis_err, betas, alphas, *, per_arm, suffix):
    FIGURES.mkdir(parents=True, exist_ok=True)
    arms = list(per_arm_axis_err)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    pts = [per_arm_axis_err[a][np.isfinite(per_arm_axis_err[a])] * 100.0 for a in arms]
    ax.boxplot(pts, tick_labels=arms, showmeans=True)
    ax.set_ylabel("axis recovery error [cm]")
    ax.set_title("Recovery — axis error per arm (synthetic truth, MAST geometry)")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / f"fig-identifiability-recovery{suffix}.png", dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# mode: ALIASING — inject one perturbation class alone, measure the leakage
# ---------------------------------------------------------------------------


def run_aliasing(args, campaign, warm):
    n = args.n_samples
    seed = args.seed
    n_ch = len(campaign.channels)
    beta0, alpha = 0.5, 1.2  # a representative confined operating point

    def cont_fit(truth):
        p = truth.to_payload()
        return fit_profile_continuous(
            campaign.grid,
            campaign.table,
            i_pf=p.i_pf,
            ip_amperes=p.ip_amperes,
            measured=p.measured,
            vacuum_prediction=p.vacuum,
            sensor_scale=p.scale,
            sensor_mask=p.mask,
            fit_z0=True,
            maxfev=40,
        )

    classes = {
        "rotation": {"gamma0": [0.0, 0.35, 0.7]},
        "calibration": {"offset_sigma": [0.0, 1.0, 2.0]},
        "passive": {"passive_ka": [0.0, 25.0, 50.0]},
    }
    # one continuous-arm fit costs minutes; the full 9-cell matrix does not
    # fit the debug-partition hour — shard by class and merge offline
    if args.alias_classes:
        wanted = {c.strip() for c in args.alias_classes.split(",") if c.strip()}
        classes = {k: v for k, v in classes.items() if k in wanted}
    matrix = {}  # perturbation -> level -> {dbeta0, dalpha, daxis, dc_r4}
    for cls, spec in classes.items():
        key, levels = next(iter(spec.items()))
        rows = []
        for lvl in levels:
            dbeta, dalpha, daxis, dc = [], [], [], []
            for i in range(n):
                kw = dict(
                    beta0=beta0,
                    alpha=alpha,
                    ip_amperes=args.ip,
                    warm_jphi=warm,
                    noise=not args.no_noise,
                    seed=seed * 100 + i,
                )
                if cls == "rotation":
                    kw["gamma0"] = lvl
                elif cls == "calibration":
                    rng = np.random.default_rng(seed * 100 + i)
                    off = rng.normal(0.0, 1.0, n_ch) * campaign.scale * lvl
                    kw["offsets"] = off
                elif cls == "passive":
                    pa = np.zeros(campaign.n_passive)
                    pa[: min(3, campaign.n_passive)] = lvl * 1e3
                    kw["passive_amplitudes"] = pa
                t = st.manufacture(campaign, **kw)
                if not t.confined:
                    continue
                fit = cont_fit(t)
                if fit is None:
                    continue
                dbeta.append(fit.beta0 - beta0)
                dalpha.append(fit.alpha - alpha)
                daxis.append(
                    np.hypot(
                        fit.result.axis[0] - t.axis[0], fit.result.axis[1] - t.axis[1]
                    )
                )
                clo = _closures_from_currents(
                    campaign,
                    fit.result.cell_currents,
                    t.i_pf,
                    form="affine-r2-rotation",
                )
                dc.append(clo.get("c_r4_wmean", np.nan))
            rows.append(
                {
                    "level": lvl,
                    "level_key": key,
                    "dbeta0": _boot_ci(dbeta, seed=seed),
                    "dalpha": _boot_ci(dalpha, seed=seed),
                    "daxis_m": _boot_ci(daxis, seed=seed),
                    "recovered_c_r4": _boot_ci(dc, seed=seed),
                    "n": len(dbeta),
                }
            )
            logger.info(
                "[aliasing %s=%.2f] dbeta0=%s daxis=%.3f n=%d",
                key,
                lvl,
                None
                if rows[-1]["dbeta0"]["point"] is None
                else round(rows[-1]["dbeta0"]["point"], 4),
                rows[-1]["daxis_m"]["point"] or float("nan"),
                len(dbeta),
            )
        matrix[cls] = rows

    # headline: rotation → p0' (β0) contamination at the largest rotation level
    rot = matrix["rotation"][-1] if "rotation" in matrix else None
    report = {
        "coil_model_version": COIL_MODEL_VERSION,
        "shot": int(args.shot),
        "recovery_arm": "closure_continuous (static, no rotation term)",
        "headline_rotation_to_p0prime": None
        if rot is None
        else {
            "gamma0": rot["level"],
            "delta_beta0": rot["dbeta0"],
            "delta_axis_m": rot["daxis_m"],
        },
        "matrix": matrix,
    }
    (ARTIFACTS / f"identifiability_aliasing{args.out_suffix}.json").write_text(
        json.dumps(report, indent=2)
    )
    # npz matrix: rows = (perturbation, level), cols = affected quantities
    labels, mat = [], []
    for cls, rows in matrix.items():
        for r in rows:
            labels.append(f"{cls}={r['level']:g}")
            mat.append(
                [
                    r["dbeta0"]["point"] or np.nan,
                    r["dalpha"]["point"] or np.nan,
                    r["daxis_m"]["point"] or np.nan,
                    r["recovered_c_r4"]["point"] or np.nan,
                ]
            )
    np.savez(
        ARTIFACTS / f"identifiability_aliasing{args.out_suffix}_matrix.npz",
        matrix=np.array(mat, dtype=np.float64),
        row_labels=np.array(labels),
        col_labels=np.array(["dbeta0", "dalpha", "daxis_m", "recovered_c_r4"]),
    )
    _plot_aliasing(labels, np.array(mat, dtype=np.float64), suffix=args.out_suffix)
    return report


def _plot_aliasing(labels, mat, *, suffix):
    FIGURES.mkdir(parents=True, exist_ok=True)
    cols = ["Δβ0", "Δα", "Δaxis [m]", "recovered c(R⁴)"]
    # normalise each column to its max abs for a readable heatmap
    norm = mat / (np.nanmax(np.abs(mat), axis=0, keepdims=True) + 1e-30)
    fig, ax = plt.subplots(figsize=(7, 0.6 * len(labels) + 2))
    im = ax.imshow(norm, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=20)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            ax.text(
                j,
                i,
                "" if not np.isfinite(v) else f"{v:.3g}",
                ha="center",
                va="center",
                fontsize=8,
            )
    ax.set_title("Aliasing matrix — mis-modelled term → recovered-coefficient leak")
    fig.colorbar(im, ax=ax, label="column-normalised leak")
    fig.tight_layout()
    fig.savefig(FIGURES / f"fig-identifiability-aliasing{suffix}.png", dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# mode: BASIN — which cold seeds / relaxations reach the confined branch
# ---------------------------------------------------------------------------


def run_basin(args, campaign, warm):
    grid, table = campaign.grid, campaign.table
    ip = args.ip
    i_pf = st.build_confining_i_pf(campaign.fwd, st.DEFAULT_VF_STRENGTH)
    betas = np.linspace(*BETA0_RANGE, 3)
    alphas = np.linspace(*ALPHA_RANGE, 3)

    def is_confined(res_or_axis):
        ax = res_or_axis.axis if hasattr(res_or_axis, "axis") else res_or_axis
        return float(ax[0]) <= st._CONFINED_AXIS_R_MAX

    def payload_for(b, a):
        t = st.manufacture(
            campaign, beta0=b, alpha=a, ip_amperes=ip, warm_jphi=warm, noise=False
        )
        return t, t.to_payload()

    strategies = [
        "midplane",
        "seed_z0+",
        "seed_z0-",
        "boundary_continuation",
        "free_sign_scout",
        "warm_neighbour",
        "damped_relax",
    ]
    grid_map = {s: np.zeros((len(betas), len(alphas)), dtype=bool) for s in strategies}
    axis_map = {s: np.full((len(betas), len(alphas)), np.nan) for s in strategies}

    for ib, b in enumerate(betas):
        for ia, a in enumerate(alphas):
            t, p = payload_for(b, a)
            # 1. midplane cold Gaussian, direct
            r = solve_equilibrium(
                grid, i_pf, ip, beta0=b, alpha=a, seed_width=(0.2, 0.35)
            )
            grid_map["midplane"][ib, ia] = is_confined(r)
            axis_map["midplane"][ib, ia] = r.axis[0]
            # 2/3. seed_z0 offsets
            for tag, z0 in (("seed_z0+", 0.2), ("seed_z0-", -0.2)):
                r = solve_equilibrium(
                    grid, i_pf, ip, beta0=b, alpha=a, seed_z0=z0, seed_width=(0.2, 0.35)
                )
                grid_map[tag][ib, ia] = is_confined(r)
                axis_map[tag][ib, ia] = r.axis[0]
            # 4. boundary-continuation bootstrap
            r = solve_equilibrium_bootstrapped(
                grid, i_pf, ip, beta0=b, alpha=a, seed_width=(0.2, 0.35)
            )
            grid_map["boundary_continuation"][ib, ia] = is_confined(r)
            axis_map["boundary_continuation"][ib, ia] = r.axis[0]
            # 5. free-sign LSQ scout (cold), then read axis
            lad = solve_equilibrium_lsq(
                grid,
                table,
                i_pf,
                ip,
                measured=p.measured,
                vacuum_prediction=p.vacuum,
                sensor_scale=p.scale,
                sensor_mask=p.mask,
                n_p=1,
                n_f=1,
                seed_width=(0.2, 0.35),
            )
            grid_map["free_sign_scout"][ib, ia] = is_confined(lad.result)
            axis_map["free_sign_scout"][ib, ia] = lad.result.axis[0]
            # 6. warm-start from the confined neighbour seed
            r = solve_equilibrium(
                grid, i_pf, ip, beta0=b, alpha=a, initial_jphi=warm, relax=0.3
            )
            grid_map["warm_neighbour"][ib, ia] = is_confined(r)
            axis_map["warm_neighbour"][ib, ia] = r.axis[0]
            # 7. damped relaxation (low relax) from cold seed — scout replacement?
            r = solve_equilibrium(
                grid,
                i_pf,
                ip,
                beta0=b,
                alpha=a,
                seed_width=(0.2, 0.35),
                relax=0.15,
                max_iterations=300,
            )
            grid_map["damped_relax"][ib, ia] = is_confined(r)
            axis_map["damped_relax"][ib, ia] = r.axis[0]
        logger.info("basin: beta0=%.2f row done", b)

    frac = {s: float(grid_map[s].mean()) for s in strategies}
    report = {
        "coil_model_version": COIL_MODEL_VERSION,
        "shot": int(args.shot),
        "betas": betas.tolist(),
        "alphas": alphas.tolist(),
        "confined_fraction": frac,
        "verdict": _basin_verdict(frac),
        "axis_r_map": {s: axis_map[s].tolist() for s in strategies},
        "confined_map": {s: grid_map[s].astype(int).tolist() for s in strategies},
    }
    (ARTIFACTS / f"identifiability_basin{args.out_suffix}.json").write_text(
        json.dumps(report, indent=2)
    )
    _plot_basin(betas, alphas, grid_map, axis_map, suffix=args.out_suffix)
    logger.info(
        "basin confined fraction: %s", {k: round(v, 2) for k, v in frac.items()}
    )
    return report


def _basin_verdict(frac):
    scout = frac.get("free_sign_scout", 0.0)
    damped = frac.get("damped_relax", 0.0)
    warm = frac.get("warm_neighbour", 0.0)
    cold = frac.get("midplane", 0.0)
    if damped >= max(scout, warm) - 1e-9 and damped > cold + 0.1:
        return (
            "damped relaxation matches/exceeds scouting — candidate scout replacement"
        )
    if warm > max(scout, damped, cold):
        return (
            "warm-start from a confined neighbour is the most reliable branch selector"
        )
    return "scouting/continuation still needed — no cold-seed relaxation retires it"


def _plot_basin(betas, alphas, grid_map, axis_map, *, suffix):
    FIGURES.mkdir(parents=True, exist_ok=True)
    strategies = list(grid_map)
    ncol = 4
    nrow = int(np.ceil(len(strategies) / ncol))
    fig, axes = plt.subplots(
        nrow, ncol, figsize=(3.0 * ncol, 2.8 * nrow), squeeze=False
    )
    for k, s in enumerate(strategies):
        ax = axes[k // ncol][k % ncol]
        im = ax.imshow(
            axis_map[s],
            origin="lower",
            aspect="auto",
            cmap="viridis",
            extent=[alphas[0], alphas[-1], betas[0], betas[-1]],
            vmin=0.4,
            vmax=1.9,
        )
        # overlay confined markers
        for ib, b in enumerate(betas):
            for ia, a in enumerate(alphas):
                ax.text(
                    a,
                    b,
                    "C" if grid_map[s][ib, ia] else "A",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=9,
                    fontweight="bold",
                )
        ax.set_title(f"{s}\n(confined {grid_map[s].mean():.0%})", fontsize=9)
        ax.set_xlabel("α")
        ax.set_ylabel("β0")
    for k in range(len(strategies), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.colorbar(
        im, ax=axes.ravel().tolist(), label="axis R [m]  (C=confined, A=attractor)"
    )
    fig.suptitle(
        "Fixed-point basin structure — seed / relaxation strategy vs (β0, α) truth"
    )
    fig.savefig(FIGURES / f"fig-basin-map{suffix}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# mode: INFORMATION — whitened-Jacobian rank + per-coefficient Fisher
# ---------------------------------------------------------------------------


def run_information(args, campaign, warm):
    ip = args.ip
    b0, a0, g0 = 0.5, 1.2, 0.3
    base = st.manufacture(
        campaign,
        beta0=b0,
        alpha=a0,
        gamma0=g0,
        ip_amperes=ip,
        warm_jphi=warm,
        noise=False,
    )
    scale = campaign.scale

    def whitened(truth):
        return (truth.measured_clean) / scale

    y0 = whitened(base)
    # finite-difference Jacobian columns wrt each Tier-2 coefficient
    coeffs = []
    jac = []

    def col(name, perturbed, h):
        coeffs.append(name)
        jac.append((whitened(perturbed) - y0) / h)

    hb = 0.05
    col(
        "beta0",
        st.manufacture(
            campaign,
            beta0=b0 + hb,
            alpha=a0,
            gamma0=g0,
            ip_amperes=ip,
            warm_jphi=warm,
            noise=False,
        ),
        hb,
    )
    ha = 0.1
    col(
        "alpha",
        st.manufacture(
            campaign,
            beta0=b0,
            alpha=a0 + ha,
            gamma0=g0,
            ip_amperes=ip,
            warm_jphi=warm,
            noise=False,
        ),
        ha,
    )
    hg = 0.1
    col(
        "gamma0",
        st.manufacture(
            campaign,
            beta0=b0,
            alpha=a0,
            gamma0=g0 + hg,
            ip_amperes=ip,
            warm_jphi=warm,
            noise=False,
        ),
        hg,
    )
    # passive modes are LINEAR in the sensors — one column per top mode, no re-solve
    sidecar = build_passive_sidecar(
        campaign.table,
        campaign.grid,
        g_passive=campaign.passive_g_sens,
        sensor_scale=scale,
        k=args.passive_k,
    )
    g_mode = np.asarray(sidecar["g_cols"])  # (S, k), already unit-whitened-norm
    for m in range(g_mode.shape[1]):
        coeffs.append(f"passive_mode_{m}")
        jac.append(g_mode[:, m])  # d(whitened sensor)/d(mode amplitude)

    jac_mat = np.column_stack(jac)  # (S, K)

    # sensor sets: magnetics now; structure for interferometer/SXR/CXRS later
    sensor_sets = {
        "magnetics": np.ones(len(campaign.channels), dtype=bool),
        "flux_loops": np.array([c.lower().startswith("fl") for c in campaign.channels]),
        "b_probes": np.array(
            [not c.lower().startswith("fl") for c in campaign.channels]
        ),
    }
    out = {
        "coil_model_version": COIL_MODEL_VERSION,
        "shot": int(args.shot),
        "coefficients": coeffs,
        "sensor_sets": {},
    }
    for sname, sel in sensor_sets.items():
        jac_set = jac_mat[sel, :]
        sv = np.linalg.svd(jac_set, compute_uv=False)
        eff_rank = float((sv.sum() ** 2) / (np.sum(sv**2) + 1e-30)) if sv.size else 0.0
        fisher = np.sum(jac_set**2, axis=0)  # per-coefficient diagonal Fisher
        out["sensor_sets"][sname] = {
            "n_sensor": int(sel.sum()),
            "singular_values": sv.tolist(),
            "effective_rank": eff_rank,
            "hard_rank_1pct": int(np.sum(sv > 0.01 * (sv[0] if sv.size else 1.0))),
            "fisher_per_coefficient": {
                c: float(f) for c, f in zip(coeffs, fisher, strict=True)
            },
        }
        logger.info(
            "[information %s] eff_rank=%.2f n=%d", sname, eff_rank, int(sel.sum())
        )
    (ARTIFACTS / f"identifiability_information{args.out_suffix}.json").write_text(
        json.dumps(out, indent=2)
    )
    _plot_information(out, suffix=args.out_suffix)
    return out


def _plot_information(out, *, suffix):
    FIGURES.mkdir(parents=True, exist_ok=True)
    coeffs = out["coefficients"]
    mag = out["sensor_sets"]["magnetics"]
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4))
    sv = np.array(mag["singular_values"])
    ax0.semilogy(range(1, sv.size + 1), sv / sv[0], "o-")
    ax0.set_xlabel("singular index")
    ax0.set_ylabel("σ / σ₀")
    ax0.set_title(f"magnetics Jacobian spectrum (eff rank {mag['effective_rank']:.1f})")
    ax0.grid(True, alpha=0.3)
    fish = np.array([mag["fisher_per_coefficient"][c] for c in coeffs])
    ax1.barh(range(len(coeffs)), fish)
    ax1.set_yticks(range(len(coeffs)))
    ax1.set_yticklabels(coeffs, fontsize=8)
    ax1.set_xscale("log")
    ax1.set_xlabel("per-coefficient Fisher (whitened)")
    ax1.set_title("magnetics identifiability of each Tier-2 coefficient")
    ax1.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / f"fig-identifiability-information{suffix}.png", dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=("all", "recovery", "aliasing", "basin", "information"),
    )
    ap.add_argument("--shot", type=int, default=18502)
    ap.add_argument("--nr", type=int, default=65)
    ap.add_argument("--nz", type=int, default=97)
    ap.add_argument("--n-samples", type=int, default=24)
    ap.add_argument("--arms", type=str, default="")
    ap.add_argument(
        "--alias-classes",
        type=str,
        default="",
        help="aliasing mode: restrict to these perturbation classes "
        "(comma list of rotation/calibration/passive; '' = all)",
    )
    ap.add_argument("--ip", type=float, default=st.DEFAULT_IP_AMPERES)
    ap.add_argument("--iters", type=int, default=600)
    ap.add_argument("--passive-k", type=int, default=8)
    ap.add_argument("--no-noise", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-suffix", type=str, default="")
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    campaign = st.build_campaign(args.shot, nr=args.nr, nz=args.nz)
    warm, warm_axr = st.confined_seed(campaign, ip_amperes=args.ip)
    logger.info(
        "campaign built in %.0fs; confined seed axis R=%.3f",
        time.perf_counter() - t0,
        warm_axr,
    )
    if warm_axr > st._CONFINED_AXIS_R_MAX:
        logger.warning(
            "confined seed FAILED (axis R=%.3f) — increase vf_strength", warm_axr
        )

    modes = (
        ["recovery", "aliasing", "basin", "information"]
        if args.mode == "all"
        else [args.mode]
    )
    for m in modes:
        logger.info("=== mode: %s ===", m)
        {
            "recovery": run_recovery,
            "aliasing": run_aliasing,
            "basin": run_basin,
            "information": run_information,
        }[m](args, campaign, warm)
    logger.info("done in %.0fs", time.perf_counter() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
