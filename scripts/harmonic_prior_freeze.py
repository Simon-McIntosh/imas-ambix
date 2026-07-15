"""Freeze the source-free harmonic boundary read + measure its gauge.

Two deliverables over the cohort loaded EXACTLY as
``scripts/boundary_harmonic_gate_eval.py`` loads it (same split, sensor order,
per-slice current-centroid origin + tracking pole, frozen best config):

(a) GAUGE MEASUREMENT.  The gate's annulus consistency metric blindly removes a
    per-slice MEAN offset before its RMS
    (``consistency_rms``, boundary_harmonic_gate_eval.py) — that discards the
    DC (absolute-gauge) disagreement between the two independent external-
    magnetics reads.  The interior soft-prior penalty must instead KEEP the
    flux-loop gauge (plan §3.4).  For every slice this recomputes both TOTAL-flux
    fields on the shared vacuum annulus (the current-moment carrier and the
    harmonic total) and records the offset the blind metric WOULD remove,
    ``mean(psi_carrier - psi_harmonic)``, in absolute [Wb] and as a fraction of
    the annulus dynamic range.  It then decides the PENALTY FORM:
      * a SYSTEMATIC, consistently-signed level bias  => the interior penalty
        needs a fitted rank-1 offset DOF ("rank1-offset");
      * zero-mean scatter (fit noise)                 => matching grad-psi in the
        annulus is safe and simplest ("grad-psi").
    Also records the whitened flux-loop residual of the fit — evidence the gauge
    IS pinned by the 12 flux loops (§3.4).

(b) FROZEN PER-SLICE ARTIFACT (§3.3f).  Persists an NPZ+JSON per split under
    ``imas_ambix/latent/artifacts/patch_gate/`` carrying, PER SLICE: shot,
    t_index, time_s, ip_amperes, harmonic coeffs, coeff_cov, misfit, the origin
    (centroid) and pole used, the annulus dynamic range, and the frozen config.
    The interior solve LOADS this (``load_frozen_harmonic_prior``) instead of
    refitting — pinning the prior against editable-install drift while in-flight
    jobs run.

Firewall: no EFIT in any fit path; the referee only ever scores.  Run on BOTH
splits (``--split train`` and ``--split eval``); the artifacts are small and
committed.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import numpy as np

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from boundary_harmonic_gate_eval import (  # noqa: E402
    _adaptive_radii,
    _origin_and_pole,
    build_parser,
    hybrid_target_harmonic,
    sensor_arrays,
)
from boundary_moment_gate_eval import load_cohort  # noqa: E402
from patch_gate_eval import ARTIFACTS  # noqa: E402

from imas_ambix.latent.boundary_harmonic import (  # noqa: E402
    HarmonicFitConfig,
    _fit_one,
    harmonic_columns,
    harmonic_labels,
    harmonic_sensor_matrix,
    save_frozen_harmonic_prior,
)
from imas_ambix.latent.boundary_moment import (  # noqa: E402
    MomentFitConfig,
    fit_moment_currents,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("harmonic-prior-freeze")

# The §2 winner (recovered from imas_ambix/latent/artifacts/patch_gate/
# boundary_read_harmonic-o3-centroidorigin-frac0.41.json): order 3, ridge 1e-8,
# centroid origin, tracking pole 0.41 inboard, mask_frac 0.5, exclude_frac 1.1,
# ip_anchor OFF.  Frozen here so the interior solve reads ONE fixed prior.
FROZEN_ORDER = 3
FROZEN_RIDGE = 1e-8
FROZEN_FRACTION = 0.41


def _annulus_mask(psi_carrier, grid, axis_psi, boundary_psi):
    """The gate's shared vacuum annulus: in-limiter but outside the confined
    region (matches ``annulus_consistency_rms``)."""
    sign = np.sign(axis_psi - boundary_psi)
    confined = (psi_carrier - boundary_psi) * sign > 0.0
    inside = np.asarray(grid.inside_limiter, dtype=bool)
    return inside & ~confined, inside


def measure_and_freeze(shots, split, args) -> dict:
    """One pass over the cohort: gauge measurement + frozen per-slice prior."""
    slices: list[dict] = []
    offset_wb, offset_frac, flux_dc, misfits, shot_ids = [], [], [], [], []
    for payload in shots:
        grid, basis, table = payload["grid"], payload["basis"], payload["table"]
        n_cells = int(basis.r_cells.shape[0])
        sr, sz, sang, is_flux = sensor_arrays(table)
        rr, zz = np.meshgrid(grid.rg, grid.zg)
        gr, gz = rr.ravel(), zz.ravel()
        mom_cfg = MomentFitConfig(order=3)
        for k, p in enumerate(payload["payloads"]):
            mom = fit_moment_currents(basis, p, mom_cfg)
            psi_mom = basis.psi_grid_2d_np(mom.i_cell, p.i_pf)  # carrier (total)
            origin, pole = _origin_and_pole(
                (mom.centroid_r, mom.centroid_z), grid, args, FROZEN_FRACTION
            )
            mask_r, excl_r = _adaptive_radii(origin, pole, args)
            cfg = HarmonicFitConfig(
                pole_r=pole[0],
                pole_z=pole[1],
                order=FROZEN_ORDER,
                ridge=FROZEN_RIDGE,
                ip_anchor=False,
            )
            a_sens = harmonic_sensor_matrix(sr, sz, sang, is_flux, cfg)
            coeffs, misfit, cov = _fit_one(
                a_sens, p.measured, p.vacuum, p.mask, p.scale, cfg.ridge
            )
            grid_cols, _ = harmonic_columns(gr, gz, cfg)
            psi_plasma = (grid_cols @ coeffs).reshape(grid.nz, grid.nr)
            psi_coil = basis.psi_grid_2d_np(np.zeros(n_cells), p.i_pf)
            psi_tot = psi_plasma + psi_coil  # like-for-like total flux
            _t, axis_psi, boundary_psi, _f, _d = hybrid_target_harmonic(
                psi_tot, grid, origin, pole, mask_r, excl_r, xpoint_tol=args.xpoint_tol
            )

            annulus, inside = _annulus_mask(psi_mom, grid, axis_psi, boundary_psi)
            diff = (psi_mom - psi_tot)[annulus]  # the DC the blind metric removes
            diff = diff[np.isfinite(diff)]
            in_vals = psi_mom[inside]
            dyn = (
                float(np.nanmax(in_vals) - np.nanmin(in_vals)) if in_vals.size else 0.0
            )
            off = float(diff.mean()) if diff.size else float("nan")
            frac = off / dyn if dyn > 0.0 else float("nan")

            # whitened flux-loop residual of the fit (gauge pinned by flux loops)
            keep = np.asarray(p.mask, dtype=bool) & np.asarray(is_flux, dtype=bool)
            w = 1.0 / np.maximum(np.asarray(p.scale, dtype=np.float64), 1e-12)
            pred = (
                np.nan_to_num(np.asarray(p.vacuum, dtype=np.float64)) + a_sens @ coeffs
            )
            resid = (pred - np.nan_to_num(np.asarray(p.measured, dtype=np.float64))) * w
            fl_dc = (
                float(np.sqrt(np.mean(resid[keep] ** 2)))
                if keep.any()
                else float("nan")
            )

            offset_wb.append(off)
            offset_frac.append(frac)
            flux_dc.append(fl_dc)
            misfits.append(float(misfit))
            shot_ids.append(int(p.shot))
            slices.append(
                {
                    "shot": int(p.shot),
                    "t_index": int(getattr(p, "t_index", k)),
                    "time_s": float(getattr(p, "time_s", float("nan"))),
                    "ip_amperes": float(getattr(p, "ip_amperes", float("nan"))),
                    "coeffs": np.asarray(coeffs, dtype=np.float64),
                    "coeff_cov": np.asarray(cov, dtype=np.float64),
                    "misfit": float(misfit),
                    "origin": np.asarray(origin, dtype=np.float64),
                    "pole": np.asarray(pole, dtype=np.float64),
                    "dyn_range": dyn,
                }
            )

    meta = {
        "order": FROZEN_ORDER,
        "ridge": FROZEN_RIDGE,
        "kind": "P",
        "ip_anchor": False,
        "origin_source": "centroid",
        "pole_source": "track",
        "pole_inboard_fraction": FROZEN_FRACTION,
        "mask_frac": args.mask_frac,
        "exclude_frac": args.exclude_frac,
        "labels": harmonic_labels(FROZEN_ORDER),
        "split": split,
    }
    tag = "harmonic_prior_frozen" if split == "eval" else "harmonic_prior_frozen-tune"
    npz_path = save_frozen_harmonic_prior(ARTIFACTS / f"{tag}.npz", slices, meta)
    logger.info("froze %d slices -> %s", len(slices), npz_path)

    report = _gauge_report(offset_wb, offset_frac, flux_dc, misfits, shot_ids, meta)
    report_path = ARTIFACTS / (
        "harmonic_prior_gauge" + ("" if split == "eval" else "-tune") + ".json"
    )
    report_path.write_text(json.dumps(report, indent=2))
    logger.info(
        "gauge: median|offset|/range=%.4f frac>0.05=%.2f frac_positive=%.2f "
        "flux_dc_median=%.3e -> %s (%s)",
        report["abs_offset_frac_median"],
        report["frac_abs_offset_above_0.05"],
        report["frac_positive_signed"],
        report["flux_loop_dc_residual_median"],
        report["penalty_form_recommendation"],
        report_path,
    )
    return report


def _gauge_report(offset_wb, offset_frac, flux_dc, misfits, shot_ids, meta) -> dict:
    of = np.asarray(offset_frac, dtype=np.float64)
    ow = np.asarray(offset_wb, dtype=np.float64)
    of = of[np.isfinite(of)]
    ow = ow[np.isfinite(ow)]
    absf = np.abs(of)
    q25, q75 = np.percentile(absf, [25, 75]) if absf.size else (np.nan, np.nan)
    median_signed = float(np.median(of)) if of.size else float("nan")
    frac_pos = float(np.mean(of > 0.0)) if of.size else float("nan")
    frac_above = float(np.mean(absf > 0.05)) if absf.size else float("nan")
    median_abs = float(np.median(absf)) if absf.size else float("nan")

    # per-shot mean signed offset fraction: is the bias systematic across shots?
    shots = np.asarray(shot_ids)[: of.size]
    per_shot = {}
    for s in np.unique(shots):
        per_shot[int(s)] = float(np.mean(of[shots == s]))
    shot_means = np.asarray(list(per_shot.values()))
    shot_sign_consistency = (
        float(max(np.mean(shot_means > 0), np.mean(shot_means < 0)))
        if shot_means.size
        else float("nan")
    )

    # VERDICT.  A consistently-signed, non-negligible level bias (systematic
    # across slices AND shots) means the interior penalty must carry a fitted
    # rank-1 offset DOF; zero-mean scatter (sign flips, small median) means the
    # gauge-free grad-psi field-match is safe and simplest.
    systematic = (
        abs(median_signed) >= 0.03
        and (frac_pos >= 0.80 or frac_pos <= 0.20)
        and shot_sign_consistency >= 0.75
    )
    recommendation = "rank1-offset" if systematic else "grad-psi"

    return {
        "split": meta["split"],
        "n_slices": int(of.size),
        "frozen_config": {
            k: meta[k]
            for k in ("order", "ridge", "kind", "pole_inboard_fraction", "ip_anchor")
        },
        "abs_offset_wb_median": float(np.median(np.abs(ow))) if ow.size else None,
        "abs_offset_frac_median": median_abs,
        "abs_offset_frac_iqr": [float(q25), float(q75)],
        "signed_offset_frac_median": median_signed,
        "frac_positive_signed": frac_pos,
        "frac_abs_offset_above_0.05": frac_above,
        "per_shot_signed_offset_frac_mean": per_shot,
        "shot_sign_consistency": shot_sign_consistency,
        "flux_loop_dc_residual_median": float(np.nanmedian(flux_dc))
        if len(flux_dc)
        else None,
        "flux_loop_dc_residual_mean": float(np.nanmean(flux_dc))
        if len(flux_dc)
        else None,
        "misfit_median": float(np.median(misfits)) if misfits else None,
        "systematic_level_bias": bool(systematic),
        "penalty_form_recommendation": recommendation,
        "verdict_basis": (
            "systematic, consistently-signed level bias (|median signed "
            "offset/range| >= 0.03, sign consistent across slices and shots) "
            "=> interior penalty needs a fitted rank-1 offset DOF"
            if systematic
            else "zero-mean scatter (sign flips / small median offset) => "
            "gauge-free grad-psi field-match in the annulus is safe and simplest"
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=["train", "eval"], default="eval")
    known, _ = ap.parse_known_args()

    # reuse the gate's loader + geometry defaults (they match the frozen config)
    args = build_parser().parse_args([])
    args.split = known.split
    args.origin_source = "centroid"
    args.pole_source = "track"
    args.ip_anchor = False

    shots, _ = load_cohort(args.split, args)
    logger.info("loaded %d shots (split=%s)", len(shots), args.split)
    if not shots:
        logger.error("no shots loaded — nothing to freeze")
        return 1
    measure_and_freeze(shots, args.split, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
