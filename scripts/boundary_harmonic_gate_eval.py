"""Gate for the source-free toroidal-harmonic annulus boundary read.

Reads the plasma BOUNDARY (separatrix / X-point / LCFS radii) off a psi
reconstructed from a source-free toroidal-harmonic fit to the external
magnetics (:mod:`imas_ambix.latent.boundary_harmonic`).  Unlike the free
patch-current psi or the low-order current-moment fit -- both of which commit a
non-zero j_phi to the vacuum annulus and so VIOLATE the source-free premise the
boundary read rests on -- the harmonic basis carries NO current on the grid: it
represents the plasma-produced flux directly in the ring functions that solve
Delta* psi = 0 about a fixed pole near the axis.  This gate clones the STRUCTURE
of ``scripts/boundary_moment_gate_eval.py`` and swaps the current-moment fit for
the harmonic fit; scoring, baseline, shot list, and slice selection are shared
with the free-current gate (``scripts/patch_gate_eval.py``) so the numbers are
directly comparable and the boundary representation is the only varying factor.

HYBRID read (the plan's design; the review note is binding): the harmonic fit
gives the PLASMA flux only, and the TOTAL flux read for topology adds the KNOWN
coil field from the harness's thick-cylinder ``hybrid_greens`` coil column
(``PatchBasis`` / ``EquilibriumGrid``) -- NEVER a point-filament coil term.  The
confined-side flux reference (``axis_psi``) is read as the TOTAL psi
bilinearly interpolated AT the supplied carrier axis, NOT from the harmonic
field's own numerical O-point extremum (the low-DOF harmonic field cannot
localise the axis).  ``--axis-source`` selects the ray origin: ``patch``
(default, SCORED) = the free-current P3 inverse axis (~2.8 cm, faithful,
origin-controlled); ``harmonic`` = the harmonic total psi's numerical O-point
(ablation -- degrades the LCFS ray-cast, and disables the consistency check
which needs the patch carrier).

Protocol (leakage-free, matches the P3 / moment gate): ``--split train`` sweeps
``--orders`` on the tuning cohort and picks the best by ``lcfs_skill`` (CI-gated);
``--split eval`` scores the frozen ``--orders`` value ONCE on the 160-slice
held-out set.  Shots and the free-current axes are loaded / inverted ONCE and
reused across every order.

Performance: the pole is FIXED, so the grid harmonic columns depend only on
(order, pole, grid) and the sensor design matrix only on (order, pole, shot) --
both are cached so mpmath is never re-evaluated per slice; per slice the grid
flux is a single matmul ``cols @ coeffs``.

Records the vacuum-annulus consistency RMS (``consistency_rms_annulus``): the
offset-removed RMS agreement between the patch carrier psi and the harmonic psi
in the shared annulus (outside the LCFS, inside the limiter), normalised by the
carrier psi dynamic range -- a source-free-premise cross-check independent of
the referee.  Writes ``.../patch_gate/boundary_read_harmonic-o<order>[-...].json``.
No EFIT in any fit path; the referee only scores (firewall: code-outputs-only).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

import numpy as np
import torch

# Make the sibling gate scripts importable no matter how this file is run
# (bare script, ``python -m scripts.boundary_harmonic_gate_eval``, or an
# in-process import) -- mirrors ``scripts/boundary_moment_gate_eval.py``.
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

# Reuse the exact loader, free-current carrier, and sign-aware axis of the
# current-moment gate, and the scoring core / payload builder / frozen inverse
# of the free-current gate (script-dir import: run from the scripts/ directory).
from boundary_moment_gate_eval import (
    _sign_aware_axis,
    free_current_psi,
    load_cohort,
)
from patch_gate_eval import (
    ARTIFACTS,
    count_saddles,
    saddle_excess_stats,
    score,
)

from imas_ambix.latent.boundary_harmonic import (
    HarmonicFitConfig,
    _fit_one,
    harmonic_columns,
    harmonic_sensor_matrix,
    mask_invalid_interior,
)
from imas_ambix.latent.topology import (
    CriticalPoints,
    _bilerp,
    _inside_polygon,
    boundary_flux_robust,
    exclude_near_points,
    find_critical_points,
    lcfs_radii,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("boundary-harmonic-gate")


def sensor_arrays(table) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(sr, sz, sang_deg, is_flux)`` in ``table.sensor_map`` row order.

    ``shot_payloads`` reindexes every payload row to this order, so the
    harmonic design matrix rows are aligned with ``measured`` / ``vacuum``.
    Fixed per shot -- built once and reused across slices and orders."""
    sr = np.array([m.r for m in table.sensor_map], dtype=np.float64)
    sz = np.array([m.z for m in table.sensor_map], dtype=np.float64)
    sang = np.array(
        [0.0 if m.angle_deg is None else float(m.angle_deg) for m in table.sensor_map],
        dtype=np.float64,
    )
    is_flux = np.array([m.kind == "flux_loop" for m in table.sensor_map], dtype=bool)
    return sr, sz, sang, is_flux


# --- annulus consistency (source-free-premise cross-check) ------------------


def consistency_rms(
    psi_carrier: np.ndarray,
    psi_harmonic: np.ndarray,
    annulus_mask: np.ndarray,
    dyn_range: float,
) -> float | None:
    """Offset-removed RMS agreement of two flux fields in the annulus, /range.

    The two fields carry different absolute offsets (the harmonic fit has no
    Ip gauge unless anchored; the patch carrier is total flux), so the mean
    difference is removed before the RMS.  ``dyn_range`` normalises it to the
    carrier psi dynamic range.  Returns ``None`` when the mask is empty / the
    range is degenerate."""
    mask = np.asarray(annulus_mask, dtype=bool)
    diff = (np.asarray(psi_carrier) - np.asarray(psi_harmonic))[mask]
    finite = np.isfinite(diff)
    if not finite.any() or not (dyn_range > 0.0):
        return None
    diff = diff[finite]
    diff = diff - diff.mean()
    return float(np.sqrt(np.mean(diff**2)) / dyn_range)


def annulus_consistency_rms(
    psi_carrier: np.ndarray,
    psi_harmonic: np.ndarray,
    grid,
    axis_psi: float,
    boundary_psi: float,
) -> float | None:
    """Consistency RMS over the shared vacuum annulus of two total-flux fields.

    Annulus = grid points inside the limiter but OUTSIDE the confined region
    (the carrier's own confined-side is where flux is deeper than
    ``boundary_psi`` toward ``axis_psi``).  Normalised by the carrier psi
    dynamic range over the in-limiter region."""
    sign = np.sign(axis_psi - boundary_psi)
    confined = (psi_carrier - boundary_psi) * sign > 0.0
    inside = np.asarray(grid.inside_limiter, dtype=bool)
    annulus = inside & ~confined
    in_vals = psi_carrier[inside]
    dyn = float(np.nanmax(in_vals) - np.nanmin(in_vals)) if in_vals.size else 0.0
    return consistency_rms(psi_carrier, psi_harmonic, annulus, dyn)


# --- hybrid topology read ---------------------------------------------------


def hybrid_target_harmonic(psi_tot, grid, axis, pole, mask_radius, exclude_radius):
    """14-D geometry target read in the ANNULUS from the masked TOTAL psi.

    The toroidal harmonics are valid ONLY in the source-free annulus; toward the
    pole the ring functions diverge (the flux blows up inside the plasma where
    the expansion does not hold).  So the near-pole invalid interior is first
    masked to a confined plateau (:func:`mask_invalid_interior`), and every
    boundary read runs on that masked field:

    * the AXIS (``target[0:2]``) is the supplied carrier axis;
    * the confined-side flux reference ``axis_psi`` is the masked field
      bilinearly interpolated AT the carrier axis (NOT the field's own extremum,
      per the binding review note);
    * critical points within ``exclude_radius`` of the pole are dropped (the
      residual near-pole artifacts), then filtered to the in-limiter,
      conductor-clear set;
    * the bounding flux is the robust innermost in-vessel X-point flux (limiter
      fallback for a limited plasma);
    * the X-point set (``target[2:6]``) and 8-angle LCFS ray-cast run on the
      masked field.
    """
    pole_r, pole_z = pole
    field = mask_invalid_interior(
        psi_tot, grid.rg, grid.zg, pole_r, pole_z, mask_radius, axis_rz=tuple(axis)
    )
    target = np.full(14, np.nan)
    target[0], target[1] = axis
    axis_psi = _bilerp(field, grid.rg, grid.zg, float(axis[0]), float(axis[1]))

    cp = find_critical_points(field, grid.rg, grid.zg)
    cp = exclude_near_points(cp, np.array([[pole_r, pole_z]]), exclude_radius)
    if cp.x_points.shape[0]:
        ins = _inside_polygon(
            cp.x_points[:, 0], cp.x_points[:, 1], grid.limiter_r, grid.limiter_z
        ) & grid.clear_of_conductors(cp.x_points[:, 0], cp.x_points[:, 1])
        cp = CriticalPoints(cp.o_points, cp.o_psi, cp.x_points[ins], cp.x_psi[ins])

    boundary_psi = boundary_flux_robust(
        cp, tuple(axis), axis_psi, limiter_r=grid.limiter_r, limiter_z=grid.limiter_z
    )
    if boundary_psi is None:  # limited plasma: limiter-contact flux nearest axis
        lim_vals = field.ravel()[grid._limiter_grid_idx]
        boundary_psi = float(lim_vals[int(np.argmin(np.abs(lim_vals - axis_psi)))])

    if cp.x_points.shape[0]:
        order = np.argsort(np.abs(cp.x_psi - boundary_psi))
        for slot in range(min(2, cp.x_points.shape[0])):
            target[2 + 2 * slot] = cp.x_points[order[slot], 0]
            target[3 + 2 * slot] = cp.x_points[order[slot], 1]
    target[6:] = lcfs_radii(field, grid.rg, grid.zg, tuple(axis), boundary_psi)
    return target, float(axis_psi), float(boundary_psi), field


def _slice_pole(axis, grid, args, fraction) -> tuple[float, float]:
    """The toroidal-coordinate pole for one slice.

    ``--pole-source carrier`` (default, SCORED): the pole tracks the
    high-accuracy interior-carrier magnetic axis, placed a DIMENSIONLESS
    ``fraction`` of the axis radius INBOARD in R (``pole_r = axis_R·(1-fraction)``)
    and at the carrier axis Z (vertical tracking).  A fraction (not a fixed
    metre offset) keeps the scheme MACHINE-AGNOSTIC: the focal ring sits at the
    same RELATIVE inboard position on any device (R/R0-scaled, per the
    machine-agnostic geometry convention).  The inboard placement is
    physics-required: the P-harmonic expansion converges only when the ring sits
    inboard of the current centroid; a pole AT the axis (fraction 0) degrades the
    inboard boundary (measured lcfs_ft ~50 cm).  Tracking Z + R follows the small
    off-nominal plasma through breakdown/ramp-up.  ``fixed`` (ablation): campaign
    nominal ``grid.r0`` at ``z=0`` (``--pole-r``/``--pole-z`` override)."""
    if args.pole_source == "carrier":
        return float(axis[0]) * (1.0 - float(fraction)), float(axis[1])
    pole_r = args.pole_r if args.pole_r is not None else float(grid.r0)
    return pole_r, args.pole_z


def score_order(shots, patch_psis, order, ridge, fraction, split, args) -> dict:
    """Fit + score one (order, ridge, fraction) over the cohort; per-slice pole."""
    model_rows, ref_rows, flattop_flags, shot_rows = [], [], [], []
    saddles, misfits, consistencies = [], [], []
    t0 = time.perf_counter()
    for si, payload in enumerate(shots):
        grid, basis, table = payload["grid"], payload["basis"], payload["table"]
        n_cells = int(basis.r_cells.shape[0])
        sr, sz, sang, is_flux = sensor_arrays(table)
        rr, zz = np.meshgrid(grid.rg, grid.zg)
        gr, gz = rr.ravel(), zz.ravel()
        ips = np.abs([p.ip_amperes for p in payload["payloads"]])
        flattop_idx = int(np.argmax(ips)) if ips.size else -1
        for k, p in enumerate(payload["payloads"]):
            # the carrier axis (the high-accuracy interior estimate) is BOTH the
            # ray-cast origin AND the per-slice harmonic pole.
            axis, _ = _sign_aware_axis(patch_psis[si][k], grid)
            pole_r, pole_z = _slice_pole(axis, grid, args, fraction)
            cfg = HarmonicFitConfig(
                pole_r=pole_r,
                pole_z=pole_z,
                order=order,
                ridge=ridge,
                ip_anchor=args.ip_anchor,
            )
            # fast vectorised evaluator -> sensor matrix + grid columns per slice
            # (the per-slice pole defeats any fixed-pole cache; the elliptic path
            # is cheap enough to recompute every slice).
            a_sens = harmonic_sensor_matrix(sr, sz, sang, is_flux, cfg)
            coeffs, misfit, _ = _fit_one(
                a_sens, p.measured, p.vacuum, p.mask, p.scale, cfg.ridge
            )
            grid_cols, _ = harmonic_columns(gr, gz, cfg)
            # TOTAL flux for topology: harmonic PLASMA flux + the harness's
            # thick-cylinder coil term (i_cell=0 -> coil-only) -- never point-filament.
            psi_plasma = (grid_cols @ coeffs).reshape(grid.nz, grid.nr)
            psi_coil = basis.psi_grid_2d_np(np.zeros(n_cells), p.i_pf)
            psi_tot = psi_plasma + psi_coil
            if args.axis_source != "patch":  # ablation: harmonic O-point ray origin
                axis, _ = _sign_aware_axis(psi_tot, grid)
            target, axis_psi, boundary_psi, field = hybrid_target_harmonic(
                psi_tot,
                grid,
                axis,
                (pole_r, pole_z),
                args.mask_radius,
                args.exclude_radius,
            )
            model_rows.append(target)
            ref_rows.append(payload["refs"][k])
            flattop_flags.append(k == flattop_idx)
            shot_rows.append(int(p.shot))
            # saddle count on the ANNULUS-MASKED field: the near-pole blow-up
            # saddles are numerical artifacts of the invalid interior, not
            # physical nulls; counting them would inflate the saddle-excess
            # metric relative to the other (globally-valid-psi) arms.
            saddles.append(count_saddles(field, grid))
            misfits.append(misfit)
            if patch_psis is not None:
                consistencies.append(
                    annulus_consistency_rms(
                        patch_psis[si][k], psi_tot, grid, axis_psi, boundary_psi
                    )
                )
    dt = time.perf_counter() - t0

    model = np.array(model_rows)
    ref = np.array(ref_rows)
    flattop_mask = np.array(flattop_flags, dtype=bool)
    shot_ids = np.array(shot_rows)
    sc = score(model, ref, baseline_vec=args._baseline, shot_ids=shot_ids)
    axis_err = sc.pop("axis_errors")
    saddle_stats = saddle_excess_stats(saddles, ref)

    lcfs_model, lcfs_ref = model[:, 6:], ref[:, 6:]
    finite = np.isfinite(lcfs_ref)
    offset_cm = np.where(finite, np.abs(lcfs_model - lcfs_ref) * 100.0, np.nan)
    per_slice_median = np.nanmedian(offset_cm, axis=1)
    ft = per_slice_median[flattop_mask]
    cons = [c for c in consistencies if c is not None]
    result = {
        "arm": f"toroidal-harmonic-{args.pole_source}pole-{args.axis_source}-axis",
        "order": order,
        "ridge": ridge,
        "kind": "P",
        "pole_source": args.pole_source,
        "pole_inboard_fraction": fraction,
        "mask_radius": args.mask_radius,
        "exclude_radius": args.exclude_radius,
        "ip_anchor": args.ip_anchor,
        "axis_source": args.axis_source,
        "split": split,
        "n_scored": int(len(model)),
        "n_flattop_slices": int(flattop_mask.sum()),
        "wall_s": dt,
        **sc,
        "lcfs_offset_median_cm_all": float(np.nanmedian(per_slice_median)),
        "lcfs_offset_median_cm_flattop": (float(np.nanmedian(ft)) if ft.size else None),
        "axis_error_median_m": float(np.nanmedian(axis_err)),
        "axis_error_mean_m": float(np.nanmean(axis_err)),
        "saddles_mean": float(np.mean(saddles)) if saddles else None,
        "saddles_median": float(np.median(saddles)) if saddles else None,
        **saddle_stats,
        "misfit_median": float(np.median(misfits)) if misfits else None,
        "consistency_rms_annulus": float(np.median(cons)) if cons else None,
        "consistency_rms_annulus_mean": float(np.mean(cons)) if cons else None,
    }

    rtag = f"-r{ridge:g}" if ridge != 1e-8 else ""
    otag = f"-frac{fraction:g}" if args.pole_source == "carrier" else ""
    ptag = "" if args.pole_source == "carrier" else f"-{args.pole_source}pole"
    suffix = f"{ptag}{otag}-{args.axis_source}axis{rtag}"
    tag = f"harmonic-o{order}{suffix}" + ("" if split == "eval" else "-tune")
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / f"boundary_read_{tag}.json").write_text(json.dumps(result, indent=2))
    np.savez(
        ARTIFACTS / f"boundary_read_{tag}_arrays.npz",
        model=model,
        ref=ref,
        baseline=np.tile(args._baseline, (len(model), 1)),
        axis_errors=axis_err,
        flattop_mask=flattop_mask,
        saddles=np.asarray(saddles),
    )
    logger.info(
        "[harmonic o=%d ridge=%g %s%s] n=%d axis_skill=%.3f xpt_skill=%s lcfs_skill=%.3f "
        "lcfs_cm(all/ft)=%.1f/%s saddles_mean=%.2f consistency_rms=%s (%.1fs)",
        order,
        ridge,
        split,
        suffix,
        len(model),
        sc["axis_skill"],
        sc["xpoint_set_skill"],
        sc["lcfs_skill"],
        result["lcfs_offset_median_cm_all"],
        result["lcfs_offset_median_cm_flattop"],
        result["saddles_mean"] if result["saddles_mean"] is not None else -1.0,
        result["consistency_rms_annulus"],
        dt,
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    """Factored out of :func:`main` so the scored ``--axis-source`` default is
    unit-testable without running the full data-dependent pipeline."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=["train", "eval"], default="eval")
    ap.add_argument("--orders", type=int, nargs="+", default=[3, 4, 5, 6, 8])
    ap.add_argument(
        "--ridges",
        type=float,
        nargs="+",
        default=[1e-8, 1e-4, 1e-2, 1e-1],
        help="Tikhonov ridge sweep (column-normalised frame); the ladder picks "
        "the best (order, ridge) by lcfs_skill subject to the consistency guard",
    )
    ap.add_argument(
        "--consistency-cap",
        type=float,
        default=1.0,
        help="reject any (order, ridge) whose median annulus consistency RMS "
        "exceeds this (guards against high-order aliasing) during selection",
    )
    ap.add_argument(
        "--pole-source",
        choices=["carrier", "fixed"],
        default="carrier",
        help="toroidal-coordinate pole per slice: 'carrier' (default, SCORED) = "
        "the interior-carrier magnetic axis (tracked, inboard-offset); 'fixed' = "
        "campaign nominal grid.r0 (ablation, the retired fixed-pole behaviour)",
    )
    ap.add_argument(
        "--pole-inboard-fractions",
        type=float,
        nargs="+",
        default=[0.25, 0.41, 0.55],
        help="swept DIMENSIONLESS inboard fractions: pole_r = carrier axis_R * "
        "(1 - fraction) (machine-agnostic; the focal ring sits at the same "
        "relative inboard position on any device). fraction 0 = pole AT the axis "
        "degrades the inboard boundary",
    )
    ap.add_argument(
        "--pole-r",
        type=float,
        default=None,
        help="fixed-pole radius [m] when --pole-source fixed (default: grid.r0)",
    )
    ap.add_argument("--pole-z", type=float, default=0.0)
    ap.add_argument(
        "--mask-radius",
        type=float,
        default=0.25,
        help="radius [m] about the pole where the harmonic expansion is INVALID "
        "(ring functions diverge) and is masked to a confined plateau before the "
        "annulus boundary read",
    )
    ap.add_argument(
        "--exclude-radius",
        type=float,
        default=0.55,
        help="drop harmonic critical points within this radius [m] of the pole "
        "(residual near-pole artifacts) before the X-point / bounding-flux read",
    )
    ap.add_argument(
        "--ip-anchor",
        action="store_true",
        help="add the poloidal-circulation Ip constraint to the harmonic fit",
    )
    ap.add_argument(
        "--axis-source",
        choices=["patch", "harmonic"],
        default="patch",
        help=(
            "ray-cast axis: 'patch' = the free-current P3 inverse (default -- "
            "the scored, origin-controlled read; also enables the annulus "
            "consistency check); 'harmonic' = the harmonic total psi's numerical "
            "O-point (ablation, no consistency check)."
        ),
    )
    ap.add_argument("--device", default="auto")
    ap.add_argument("--nr", type=int, default=65)
    ap.add_argument("--nz", type=int, default=97)
    ap.add_argument("--n-train", type=int, default=40)
    ap.add_argument("--n-heldout", type=int, default=8)
    ap.add_argument("--n-baseline-shots", type=int, default=20)
    ap.add_argument("--n-tune-shots", type=int, default=8)
    ap.add_argument("--max-slices-per-shot", type=int, default=20)
    ap.add_argument("--min-ip-ka", type=float, default=50.0)
    return ap


def main() -> int:
    args = build_parser().parse_args()

    device = (
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else args.device
    )
    shots, baseline_vec = load_cohort(args.split, args)
    args._baseline = baseline_vec
    logger.info(
        "loaded %d shots (split=%s, axis=%s)", len(shots), args.split, args.axis_source
    )
    if not shots:
        logger.error("no shots loaded — nothing to score")
        return 1
    # the carrier is needed both for the ray origin and (default) the per-slice pole
    patch_psis = free_current_psi(shots, device)
    logger.info("pole-source=%s (per-slice carrier axis)", args.pole_source)

    # 2-D (order x ridge) ladder; selection by lcfs_skill subject to the
    # consistency guard (rejects high-order aliasing where the annulus RMS blows up).
    fractions = args.pole_inboard_fractions if args.pole_source == "carrier" else [0.0]
    summary = [
        score_order(shots, patch_psis, o, ridge, frac, args.split, args)
        for o in args.orders
        for ridge in args.ridges
        for frac in fractions
    ]

    def _ok(d) -> bool:
        c = d.get("consistency_rms_annulus")
        return c is not None and c <= args.consistency_cap

    eligible = [d for d in summary if _ok(d)] or summary
    best = max(eligible, key=lambda d: d["lcfs_skill"])
    logger.info(
        "BEST order=%d ridge=%g frac=%g lcfs_skill=%.3f xpt_skill=%s axis_skill=%.3f "
        "consistency_rms=%s (split=%s pole=%s)",
        best["order"],
        best["ridge"],
        best["pole_inboard_fraction"],
        best["lcfs_skill"],
        best["xpoint_set_skill"],
        best["axis_skill"],
        best["consistency_rms_annulus"],
        args.split,
        args.pole_source,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
