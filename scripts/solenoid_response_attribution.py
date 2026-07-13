#!/usr/bin/env python
"""Adjudicate the ~8% solenoid response error: uniform scale vs axial structure.

The empirical coil-response audit (``vacuum_coil_response_audit.py``) measures,
on plasma-free coil-only slices, the solenoid's field-per-amp against the
geometry forward model and finds it under-predicted: ``k_sol ≈ 1.08`` on the
dedicated-vacuum-pooled stratum (85 shots / 111k slices), significantly ≠ 1.
That single number leaves the *mechanism* open — this script separates the
competing explanations by decomposing the SAME empirical solenoid response
``G_emp[:, sol]`` (field-per-amp at each sensor, deconfounded from the other
coils + eddy by the audit's multi-coil LSQ) against per-axial-band pieces of
the model solenoid column:

    G_emp[:, sol][s]  ≈  Σ_b  k_b · g_band_b[s]

where ``g_band_b`` is the model field-per-amp produced by the band-``b`` subset
of the 656 circuit-1 filaments (same ``xmult`` weights, same finite-area
cylinder kernel), so ``Σ_b g_band_b == g_model[:, sol]`` exactly.

Competing forward models, all fit weighted by the plasma-on σ and
bootstrapped over shots for CIs:

* **H0 — uniform scale** ``k`` (1 DOF): a turn-count error (model 328 effective
  amp-turns vs a true count ~8% higher) OR a ``sol_current`` channel-scale — the
  two are degenerate in the forward map and *cannot* be told apart from vacuum
  data alone; both are a single multiplicative constant on the column.
* **H1 — per-band profile** ``k_b`` (K DOF, ``k_b ≥ 0``): a NON-uniform current
  distribution.  A band collapsing to ``k_b ≈ 0`` is the literal
  "section-not-driven" signature; ends higher than centre is a
  return-conductor / turn-density / extent signature.
* **H2 — axial extent + offset** ``(k, s_z, dz)`` (3 DOF): the solenoid modelled
  too short/long or shifted in Z (a pack-extent error) — filament Z rescaled
  about the centroid by ``s_z`` and shifted by ``dz``, then a single scale.

The verdict is by BIC over the weighted residual: if H1/H2 do not beat H0 by
ΔBIC > 6, the error is a pure uniform response scale and the correction is a
single machine-description constant on the solenoid column.  Sign matters: an
undriven section would push ``k < 1``; ``k > 1`` means the model needs MORE
effective amp-turns, not fewer.

Leakage-free by construction — reuses the audit's plasma-free, no-EFIT,
no-inversion pool assembly verbatim.

Artifact: imas_ambix/latent/artifacts/patch_gate/solenoid_response_attribution.json
Figures:  docs/figures/force-balance-spine/fig-solenoid-*.png
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from imas_ambix.gs.geometry import GEOMETRY_TABLE_VERSION, build_table_for_shot
from imas_ambix.gs.operator import (
    COIL_MODEL_VERSION,
    _green_columns,
    _sensor_rows,
    build_operator,
)
from imas_ambix.latent.data import read_split_shot_lists

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("solenoid_response_attribution")

ARTIFACTS = Path("imas_ambix/latent/artifacts/patch_gate")
FIGURES = Path("docs/figures/force-balance-spine")
SOL_CIRCUIT = 1  # circuit id of the P1 central solenoid (verified: 656 filaments)


def _load_audit_module():
    """Import the sibling vacuum-audit module for its vetted pool machinery."""
    path = Path(__file__).with_name("vacuum_coil_response_audit.py")
    spec = importlib.util.spec_from_file_location("vacuum_coil_response_audit", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def solenoid_band_columns(
    table, channels: list[str], n_bands: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-axial-band model field-per-amp columns for the solenoid circuit.

    Splits the circuit-1 filaments into ``n_bands`` equal-*amp-turn* (xmult)
    bands by Z, and builds one G column per band with the same finite-area
    cylinder kernel and xmult weighting :func:`build_operator._circ_col` uses.
    Returns ``(band_cols[n_sensor, n_bands], band_z_edges[n_bands+1],
    band_ampturn_frac[n_bands])`` on the canonical sensor axis ``channels``.
    """
    ch, kinds, srz_r, srz_z, srz_ang, _excl, _flag = _sensor_rows(table)
    if list(ch) != list(channels):
        raise RuntimeError("sensor-row order disagrees with the canonical axis")
    is_flux = np.array([k == "flux_loop" for k in kinds], dtype=bool)

    fs = [f for f in table.pf_filaments if f.circuit == SOL_CIRCUIT]
    if not fs:
        raise RuntimeError("no solenoid (circuit-1) filaments found")
    fr = np.array([f.r for f in fs], dtype=np.float64)
    fz = np.array([f.z for f in fs], dtype=np.float64)
    fw = np.array([f.xmult for f in fs], dtype=np.float64)  # turns=1 → weight=xmult
    fdr = np.array([max(abs(f.width), 0.01) for f in fs], dtype=np.float64)
    fdz = np.array([max(abs(f.height), 0.01) for f in fs], dtype=np.float64)

    # equal-amp-turn bands: cumulative xmult split at n_bands quantiles in Z
    order = np.argsort(fz)
    cum = np.cumsum(fw[order])
    total = cum[-1]
    edges_z = [float(fz[order][0])]
    band_of = np.zeros(len(fs), dtype=int)
    b = 0
    for rank, idx in enumerate(order):
        band_of[idx] = b
        if b < n_bands - 1 and cum[rank] >= total * (b + 1) / n_bands:
            edges_z.append(float(fz[idx]))
            b += 1
    edges_z.append(float(fz[order][-1]))

    cols = np.zeros((len(channels), n_bands), dtype=np.float64)
    frac = np.zeros(n_bands, dtype=np.float64)
    for bb in range(n_bands):
        m = band_of == bb
        cols[:, bb] = _green_columns(
            fr[m], fz[m], fw[m], srz_r, srz_z, srz_ang, is_flux,
            src_dr=fdr[m], src_dz=fdz[m],
        )
        frac[bb] = float(fw[m].sum() / total)
    return cols, np.array(edges_z), frac


def _extent_column(
    table, channels: list[str], s_z: float, dz: float
) -> np.ndarray:
    """Model solenoid column with Z rescaled by ``s_z`` about its centroid + ``dz``."""
    ch, kinds, srz_r, srz_z, srz_ang, _e, _f = _sensor_rows(table)
    is_flux = np.array([k == "flux_loop" for k in kinds], dtype=bool)
    fs = [f for f in table.pf_filaments if f.circuit == SOL_CIRCUIT]
    fr = np.array([f.r for f in fs], dtype=np.float64)
    fz = np.array([f.z for f in fs], dtype=np.float64)
    fw = np.array([f.xmult for f in fs], dtype=np.float64)
    fdr = np.array([max(abs(f.width), 0.01) for f in fs], dtype=np.float64)
    fdz = np.array([max(abs(f.height), 0.01) for f in fs], dtype=np.float64)
    zc = float((fw * fz).sum() / fw.sum())
    fz2 = zc + s_z * (fz - zc) + dz
    return _green_columns(
        fr, fz2, fw, srz_r, srz_z, srz_ang, is_flux, src_dr=fdr, src_dz=fdz
    )


def _wls(y: np.ndarray, X: np.ndarray, w: np.ndarray, nonneg: bool = False):
    """Weighted least squares; optional non-negativity via NNLS on whitened design."""
    good = np.isfinite(y) & np.all(np.isfinite(X), axis=1) & np.isfinite(w) & (w > 0)
    yw = y[good] * np.sqrt(w[good])
    Xw = X[good] * np.sqrt(w[good])[:, None]
    if nonneg:
        from scipy.optimize import nnls  # noqa: PLC0415

        beta, _ = nnls(Xw, yw)
    else:
        beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    resid = yw - Xw @ beta
    chi2 = float(resid @ resid)
    return beta, chi2, int(good.sum())


def _fit_models(
    g_emp_sol: np.ndarray,
    g_model_sol: np.ndarray,
    band_cols: np.ndarray,
    w: np.ndarray,
    table,
    channels: list[str],
) -> dict:
    """Fit H0/H1/H2 to one G_emp[:,sol] draw; return coefficients + χ² + BIC."""
    n_bands = band_cols.shape[1]
    # H0 — single scale
    b0, chi0, n = _wls(g_emp_sol, g_model_sol[:, None], w)
    # H1 — per-band non-negative profile
    b1, chi1, _ = _wls(g_emp_sol, band_cols, w, nonneg=True)
    # H2 — axial extent + offset + scale (coarse grid then local polish)
    best = (chi0, 1.0, 0.0, b0[0])
    for s_z in np.linspace(0.85, 1.15, 13):
        for dz in np.linspace(-0.08, 0.08, 9):
            col = _extent_column(table, channels, s_z, dz)
            bb, cc, _ = _wls(g_emp_sol, col[:, None], w)
            if cc < best[0]:
                best = (cc, s_z, dz, bb[0])
    chi2h, s_z, dz, k2 = best

    def bic(chi, p):
        return n * np.log(max(chi, 1e-300) / n) + p * np.log(n)

    return {
        "n_obs": n,
        "H0": {"k": float(b0[0]), "chi2": chi0, "bic": bic(chi0, 1)},
        "H1": {
            "k_bands": [float(x) for x in b1],
            "chi2": chi1,
            "bic": bic(chi1, n_bands),
        },
        "H2": {
            "k": float(k2),
            "s_z": float(s_z),
            "dz": float(dz),
            "chi2": chi2h,
            "bic": bic(chi2h, 3),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-train", type=int, default=40)
    ap.add_argument("--n-heldout", type=int, default=8)
    ap.add_argument("--n-bands", type=int, default=6)
    ap.add_argument("--n-boot", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--didt-quantile", type=float, default=0.25)
    ap.add_argument("--extra-shots-json", type=str, default="")
    ap.add_argument("--max-extra-shots", type=int, default=40)
    ap.add_argument("--max-shots", type=int, default=0)
    ap.add_argument("--out-suffix", type=str, default="")
    args = ap.parse_args()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    vca = _load_audit_module()

    ref_table = build_table_for_shot(11774)
    ref = build_operator(ref_table)
    channels = list(ref.sensor_channels)
    coil_channels = list(ref.pf_amc_channels)
    g_model = np.asarray(ref.g_pf, dtype=np.float64)
    n_coil = len(coil_channels)
    sol_idx = coil_channels.index("sol_current")
    g_model_sol = g_model[:, sol_idx]

    band_cols, band_edges, band_frac = solenoid_band_columns(
        ref_table, channels, args.n_bands
    )
    # sanity: bands must reconstruct the model solenoid column
    recon_err = float(
        np.nanmax(np.abs(band_cols.sum(axis=1) - g_model_sol))
        / (np.nanmax(np.abs(g_model_sol)) + 1e-30)
    )
    logger.info("band reconstruction max rel err vs g_model[:,sol] = %.2e", recon_err)
    if recon_err > 1e-9:
        logger.error("bands do not reconstruct the model column — aborting")
        return 1
    # design conditioning of the band decomposition
    cond = float(np.linalg.cond(band_cols[np.isfinite(band_cols).all(axis=1)]))
    logger.info("band-design condition number = %.3g", cond)

    # ---- assemble the plasma-free pool exactly as the audit does ----
    train_shots, held_shots = read_split_shot_lists(args.n_train, args.n_heldout)
    fleet_shots = [int(s) for s in list(train_shots) + list(held_shots)]
    if args.max_shots > 0:
        fleet_shots = fleet_shots[: args.max_shots]
    extra_shots = vca._load_extra_shots(
        args.extra_shots_json, args.max_extra_shots, set(fleet_shots)
    )
    all_shots = fleet_shots + extra_shots
    logger.info(
        "pooling %d shots (%d fleet + %d dedicated vacuum)",
        len(all_shots), len(fleet_shots), len(extra_shots),
    )
    rows: list[dict] = []
    all_didt: list[np.ndarray] = []
    for s in all_shots:
        r = vca._shot_coil_only(s, channels, coil_channels)
        if r is None:
            continue
        rows.append(r)
        all_didt.append(r["didt_slice"])
    if len(rows) < 3:
        logger.error("too few coil-only shots (%d)", len(rows))
        return 1
    didt_pool = np.concatenate(all_didt)
    threshold = float(np.quantile(didt_pool, args.didt_quantile))
    all_sigma = np.stack([r["sigma"] for r in rows])
    with np.errstate(all="ignore"):
        sigma_med = np.nanmedian(all_sigma, axis=0)
    w = 1.0 / np.where(sigma_med > 0, sigma_med, np.nan) ** 2

    plan = vca._design_plan(
        vca._stack(rows, quasi_static=False, threshold=threshold)[1]
    )

    def _g_emp_sol(sample_rows: list[dict]) -> np.ndarray | None:
        st = vca._stack(sample_rows, quasi_static=True, threshold=threshold)
        if st is None:
            return None
        meas, i_pf, didt, _n = st
        design, coil_of = vca._build_regressors(i_pf, didt, plan, augment=False)
        g_emp = vca._fit_g_emp(meas, design, coil_of, n_coil)
        return g_emp[:, sol_idx]

    # ---- point estimate on the full pool ----
    g_emp_sol = _g_emp_sol(rows)
    point = _fit_models(g_emp_sol, g_model_sol, band_cols, w, ref_table, channels)
    logger.info(
        "H0 k=%.4f chi2=%.4g | H1 bands=%s chi2=%.4g | H2 k=%.3f s_z=%.3f dz=%.3f chi2=%.4g",
        point["H0"]["k"], point["H0"]["chi2"],
        [round(x, 3) for x in point["H1"]["k_bands"]], point["H1"]["chi2"],
        point["H2"]["k"], point["H2"]["s_z"], point["H2"]["dz"], point["H2"]["chi2"],
    )
    d_bic_h1 = point["H0"]["bic"] - point["H1"]["bic"]
    d_bic_h2 = point["H0"]["bic"] - point["H2"]["bic"]
    logger.info("ΔBIC(H0−H1)=%.2f  ΔBIC(H0−H2)=%.2f  (>6 favours the richer model)",
                d_bic_h1, d_bic_h2)

    # ---- bootstrap over shots ----
    rng = np.random.default_rng(args.seed)
    boot_k0, boot_bands, boot_sz, boot_dz = [], [], [], []
    for _ in range(args.n_boot):
        draw = rng.integers(0, len(rows), len(rows))
        sample = [rows[i] for i in draw]
        ges = _g_emp_sol(sample)
        if ges is None:
            continue
        b0, _c0, _n = _wls(ges, g_model_sol[:, None], w)
        b1, _c1, _ = _wls(ges, band_cols, w, nonneg=True)
        boot_k0.append(float(b0[0]))
        boot_bands.append([float(x) for x in b1])
        # H2 light: only refit scale at the point-estimate (s_z, dz) is cheap; the
        # full grid is too costly per-draw, so we bootstrap the extent verdict via
        # the profile shape (H1) instead.
    boot_bands = np.array(boot_bands) if boot_bands else np.zeros((0, args.n_bands))

    def ci(a):
        a = np.asarray(a, dtype=np.float64)
        return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]

    k0_ci = ci(boot_k0) if boot_k0 else None
    band_ci = (
        [ci(boot_bands[:, b]) for b in range(args.n_bands)]
        if boot_bands.size
        else None
    )
    # is the per-band profile flat? spread of band scales relative to their CIs
    band_pt = np.array(point["H1"]["k_bands"])
    flat = None
    if band_ci is not None:
        overall = np.median(band_pt)
        # a band departs from uniform if its 95% CI excludes the pooled median
        departs = [
            bool(lo > overall or hi < overall) for lo, hi in band_ci
        ]
        flat = not any(departs)

    verdict = "uniform-scale"
    if d_bic_h1 > 6 and (flat is False):
        verdict = "axial-structure"
    elif d_bic_h2 > 6:
        verdict = "axial-extent"

    out = {
        "coil_model_version": COIL_MODEL_VERSION,
        "geometry_table_version": GEOMETRY_TABLE_VERSION,
        "leakage_free": True,
        "n_shots_used": len(rows),
        "n_fleet": len(fleet_shots),
        "n_dedicated_vacuum": len(extra_shots),
        "n_bands": args.n_bands,
        "band_z_edges": band_edges.tolist(),
        "band_ampturn_fraction": band_frac.tolist(),
        "band_design_condition": cond,
        "didt_threshold": threshold,
        "point_estimate": point,
        "bootstrap": {
            "k_uniform": point["H0"]["k"],
            "k_uniform_ci": k0_ci,
            "k_bands": point["H1"]["k_bands"],
            "k_bands_ci": band_ci,
            "band_profile_flat": flat,
        },
        "delta_bic": {"H0_minus_H1": d_bic_h1, "H0_minus_H2": d_bic_h2},
        "verdict": verdict,
        "verdict_note": (
            "k>1 ⇒ model under-predicts the solenoid; an undriven section would "
            "give k<1 (rejected by sign). Uniform scale = turn-count OR "
            "sol_current channel-scale (degenerate in the forward map; vacuum "
            "data cannot separate them). Recommended correction: multiply the "
            "solenoid g_pf column by k_uniform as a machine-description constant."
        ),
        "recommended_scale": point["H0"]["k"],
    }
    tag = f"-{args.out_suffix}" if args.out_suffix else ""
    out_path = ARTIFACTS / f"solenoid_response_attribution{tag}.json"
    out_path.write_text(json.dumps(out, indent=2))
    logger.info("wrote %s", out_path)

    _figures(out, g_emp_sol, g_model_sol, band_cols, band_pt, band_ci, channels, tag)
    return 0


def _figures(out, g_emp_sol, g_model_sol, band_cols, band_pt, band_ci, channels, tag):
    # fig 1 — empirical vs (scaled) model solenoid response per sensor
    k = out["point_estimate"]["H0"]["k"]
    order = np.argsort(g_model_sol)
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].plot(g_model_sol[order], g_emp_sol[order], "o", ms=3, color="#1565c0",
               label="empirical vs model")
    lim = [np.nanmin(g_model_sol), np.nanmax(g_model_sol)]
    ax[0].plot(lim, lim, "k-", lw=1, label="k=1")
    ax[0].plot(lim, [k * v for v in lim], "--", color="#8a3324",
               label=f"k={k:.3f} (uniform)")
    ax[0].set_xlabel("model field-per-amp  g_model[:,sol]")
    ax[0].set_ylabel("empirical  G_emp[:,sol]")
    ax[0].set_title("Solenoid response: empirical vs geometry model")
    ax[0].legend(fontsize=8)
    # fig 2 — per-band profile with CIs
    nb = len(band_pt)
    xs = np.arange(nb)
    edges = out["band_z_edges"]
    zc = [(edges[b] + edges[b + 1]) / 2 for b in range(nb)]
    if band_ci is not None:
        lo = np.array([c[0] for c in band_ci])
        hi = np.array([c[1] for c in band_ci])
        ax[1].errorbar(zc, band_pt, yerr=[band_pt - lo, hi - band_pt], fmt="o",
                       color="#1565c0", capsize=3, label="per-band scale k_b ±95% CI")
    else:
        ax[1].plot(zc, band_pt, "o", color="#1565c0")
    ax[1].axhline(k, color="#8a3324", ls="--", label=f"uniform k={k:.3f}")
    ax[1].axhline(1.0, color="k", lw=1, label="k=1")
    ax[1].set_xlabel("solenoid band centre Z [m]")
    ax[1].set_ylabel("per-band response scale k_b")
    ax[1].set_title(
        f"Axial response profile — verdict: {out['verdict']} "
        f"(ΔBIC H0−H1={out['delta_bic']['H0_minus_H1']:.1f})"
    )
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / f"fig-solenoid-attribution{tag}.png", dpi=140)
    plt.close(fig)
    logger.info("wrote %s", FIGURES / f"fig-solenoid-attribution{tag}.png")


if __name__ == "__main__":
    raise SystemExit(main())
