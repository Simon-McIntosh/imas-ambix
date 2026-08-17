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
confined-side flux reference (``axis_psi``) is read as the TOTAL psi bilinearly
interpolated AT the supplied ORIGIN, NOT from the harmonic field's own O-point
extremum (the low-DOF harmonic field cannot localise the axis).

Origin + pole (``--origin-source``, default ``centroid``, SCORED): the ray-cast
origin and the per-slice toroidal-coordinate pole reference both come from the
magnetically-constrained, Ip-anchored CURRENT CENTROID (the moment fit's
centroid).  It is robust across plasma phases -- unlike the flat-top-tuned
free-current patch O-point, which mislocates the axis ~30 cm outboard at ramp-up
and collapses the boundary there -- machine-agnostic, and needs NO interior
carrier (the expensive patch inverse is skipped entirely).  ``patch`` / ``harmonic``
are ablation origins.  The pole tracks the origin a dimensionless fraction of the
origin radius inboard (``pole_r = origin_R·(1-fraction)``), and the near-pole
invalid-disk mask / critical-point exclusion are SIZE-ADAPTIVE (fractions of the
pole-to-origin distance) so they never reach the axis as the plasma shrinks.

Protocol (leakage-free): ``--split train`` sweeps the order × ridge × fraction
ladder, selecting by ``lcfs_skill`` under the consistency-RMS guard; ``--split
eval`` scores the frozen config ONCE on the 160-slice held-out set.

Performance: the fast vectorised P^1_{n-1/2} evaluator makes the per-slice pole
+ moment-centroid origin cheap (no mpmath per slice, no carrier inverse).

Records the vacuum-annulus consistency RMS (``consistency_rms_annulus``): the
offset-removed RMS agreement between the current-moment psi and the harmonic psi
in the shared annulus -- two independent external-magnetics reads agreeing where
both are valid, a source-free-premise cross-check independent of the referee.
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
from imas_ambix.latent.boundary_moment import MomentFitConfig, fit_moment_currents
from imas_ambix.latent.topology import (
    _bilerp,
    _inside_polygon,
    emergent_xpoints,
    exclude_near_points,
    find_critical_points,
    lcfs_contour,
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


def hybrid_target_harmonic(
    psi_tot,
    grid,
    axis,
    pole,
    mask_radius,
    exclude_radius,
    *,
    xpoint_tol=0.05,
    clip_legs=False,
):
    """14-D geometry target: LCFS SHAPE by the outermost closed axis-enclosing
    flux contour; X-points + limited/diverted class emergent.

    The toroidal harmonics are valid ONLY in the source-free annulus; toward the
    pole the ring functions diverge, so the near-pole interior is masked to a
    confined plateau (:func:`mask_invalid_interior`) and the read runs on that
    masked field:

    * the AXIS (``target[0:2]``) is the supplied origin (the interior estimate);
    * the confined-side flux reference ``axis_psi`` is the masked field bilinearly
      interpolated AT the origin (NOT the field's own extremum);
    * **PRIMARY — the LCFS shape** (:func:`imas_ambix.latent.topology.lcfs_contour`):
      one monotone flux-offset push outward from the axis keeps the OUTERMOST
      closed axis-enclosing ring that lies inside the limiter (contourpy).  The 8
      radii (``target[6:]``) are read straight off that ring polygon with the
      evaluator's own angle-interp — no outward ray-march.  ONE code path handles
      both topologies (a diverted ring pinches at the X-point; a limited ring
      rides the wall), with NO X-point gate, NO limited/diverted branch, and NO
      distance cap (far-field garbage is a separate ring, excluded by the
      enclose-the-axis test; the masked interior is a deep plateau that never
      produces a contour in the outward sweep).
    * **EMERGENT (diagnostic) — X-points + class**
      (:func:`imas_ambix.latent.topology.emergent_xpoints`): AFTER the boundary is
      known, the ∇ψ=0 saddles are found, near-pole artifacts dropped, filtered to
      the in-limiter conductor-clear set, and the class + X-slots (``target[2:6]``)
      read off by proximity to the LCFS ring.  An X-point within ``xpoint_tol`` of
      the ring ⇒ diverted (report it); else limited (X-slots NaN).  Soft margin —
      the class is genuinely undefined near the limited↔diverted transition, never
      a code-path switch.  The LCFS shape is the scored deliverable; the X-point
      set is emergent/secondary (reported, not optimised against EFIT's flag).

    Returns ``(target, axis_psi, boundary_psi, field, diverted)`` — ``diverted``
    is the emergent per-slice class diagnostic (``None`` if no boundary found).
    """
    pole_r, pole_z = pole
    field = mask_invalid_interior(
        psi_tot, grid.rg, grid.zg, pole_r, pole_z, mask_radius, axis_rz=tuple(axis)
    )
    target = np.full(14, np.nan)
    target[0], target[1] = axis
    axis_psi = _bilerp(field, grid.rg, grid.zg, float(axis[0]), float(axis[1]))

    # PRIMARY: the continuous LCFS shape (one push, no branch, no cap).
    lcfs = lcfs_contour(
        field,
        grid.rg,
        grid.zg,
        tuple(axis),
        clip_legs=clip_legs,
        limiter_r=grid.limiter_r,
        limiter_z=grid.limiter_z,
    )
    boundary_psi = lcfs.psi_bnd
    target[6:] = lcfs.radii

    # EMERGENT: X-points + limited/diverted class, read AFTER the boundary.
    diverted: bool | None = None
    if lcfs.found:
        cp = find_critical_points(field, grid.rg, grid.zg)
        cp = exclude_near_points(cp, np.array([[pole_r, pole_z]]), exclude_radius)
        xpts = cp.x_points
        if xpts.shape[0]:
            ins = _inside_polygon(
                xpts[:, 0], xpts[:, 1], grid.limiter_r, grid.limiter_z
            ) & grid.clear_of_conductors(xpts[:, 0], xpts[:, 1])
            xpts = xpts[ins]
        xset, diverted = emergent_xpoints(xpts, lcfs.ring, tol=xpoint_tol)
        target[2:6] = xset.reshape(-1)
    return target, float(axis_psi), float(boundary_psi), field, diverted


def _origin_and_pole(axis, grid, args, fraction) -> tuple[tuple, tuple]:
    """The ray-cast ORIGIN and the toroidal-coordinate POLE for one slice.

    ``axis`` is the chosen origin (see ``--origin-source``).  The pole tracks it,
    placed a DIMENSIONLESS ``fraction`` of the origin radius INBOARD in R
    (``pole_r = origin_R·(1-fraction)``) at the origin Z.  A fraction (not a fixed
    metre offset) keeps the scheme MACHINE-AGNOSTIC (the focal ring sits at the
    same RELATIVE inboard position on any device); the inboard placement is
    physics-required (the P-expansion converges only with the ring inboard of the
    current centroid).  ``--pole-source fixed`` (ablation) pins the campaign
    nominal ``grid.r0`` at ``z=0``."""
    if args.pole_source == "fixed":
        pole_r = args.pole_r if args.pole_r is not None else float(grid.r0)
        return (float(axis[0]), float(axis[1])), (pole_r, args.pole_z)
    pole = (float(axis[0]) * (1.0 - float(fraction)), float(axis[1]))
    return (float(axis[0]), float(axis[1])), pole


def _adaptive_radii(origin, pole, args) -> tuple[float, float]:
    """Size-adaptive (mask, exclude) radii [m] from the pole-to-origin distance.

    A FIXED metre mask is both machine-specific and wrong at ramp-up: when the
    plasma is small the pole-to-axis distance shrinks and a fixed 0.25 m mask
    reaches the axis, collapsing the ray-cast.  Scaling the mask/exclude with the
    pole-to-origin distance ``d = |origin - pole|`` keeps the near-pole invalid
    disk strictly inside the plasma at every plasma size and on any machine."""
    d = float(np.hypot(origin[0] - pole[0], origin[1] - pole[1]))
    if d <= 0.0:  # fixed-pole ablation: fall back to the absolute radii
        return args.mask_radius, args.exclude_radius
    return args.mask_frac * d, args.exclude_frac * d


def score_order(shots, patch_psis, order, ridge, fraction, split, args) -> dict:
    """Fit + score one (order, ridge, fraction) over the cohort; per-slice pole."""
    model_rows, ref_rows, flattop_flags, shot_rows = [], [], [], []
    saddles, misfits, consistencies, diverted_flags = [], [], [], []
    t0 = time.perf_counter()
    for si, payload in enumerate(shots):
        grid, basis, table = payload["grid"], payload["basis"], payload["table"]
        n_cells = int(basis.r_cells.shape[0])
        sr, sz, sang, is_flux = sensor_arrays(table)
        rr, zz = np.meshgrid(grid.rg, grid.zg)
        gr, gz = rr.ravel(), zz.ravel()
        ips = np.abs([p.ip_amperes for p in payload["payloads"]])
        flattop_idx = int(np.argmax(ips)) if ips.size else -1
        mom_cfg = MomentFitConfig(order=3)
        for k, p in enumerate(payload["payloads"]):
            # magnetically-constrained current centroid (Ip-anchored moment fit):
            # the DEFAULT ray-cast origin + pole reference, and the source-free
            # consistency reference (its annulus psi vs the harmonic annulus psi).
            # Robust across plasma phases where the flat-top-tuned free-current
            # carrier O-point mislocates the axis (ramp-up: ~30 cm outboard).
            mom = fit_moment_currents(basis, p, mom_cfg)
            psi_mom = basis.psi_grid_2d_np(mom.i_cell, p.i_pf)
            if args.origin_source == "centroid":
                origin = (mom.centroid_r, mom.centroid_z)
            elif args.origin_source == "patch":
                origin, _ = _sign_aware_axis(patch_psis[si][k], grid)
            else:  # "harmonic" — the field's own O-point (set below, ablation)
                origin = (mom.centroid_r, mom.centroid_z)
            origin, pole = _origin_and_pole(origin, grid, args, fraction)
            mask_r, excl_r = _adaptive_radii(origin, pole, args)
            cfg = HarmonicFitConfig(
                pole_r=pole[0],
                pole_z=pole[1],
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
            if args.origin_source == "harmonic":  # ablation: harmonic O-point
                origin, _ = _sign_aware_axis(psi_tot, grid)
                origin, pole = _origin_and_pole(origin, grid, args, fraction)
                mask_r, excl_r = _adaptive_radii(origin, pole, args)
            target, axis_psi, boundary_psi, field, diverted = hybrid_target_harmonic(
                psi_tot, grid, origin, pole, mask_r, excl_r, xpoint_tol=args.xpoint_tol
            )
            diverted_flags.append(diverted)
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
            # source-free consistency: harmonic vs the current-moment psi in the
            # shared annulus (both external-magnetics reads; no interior carrier).
            consistencies.append(
                annulus_consistency_rms(psi_mom, psi_tot, grid, axis_psi, boundary_psi)
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
    # emergent limited/diverted class diagnostic (reported, NOT a tuned objective):
    # the fraction of slices the boundary read finds an X-point ON the LCFS ring.
    div = np.array([bool(d) for d in diverted_flags if d is not None], dtype=bool)
    div_ft = np.array(
        [
            bool(d)
            for d, f in zip(diverted_flags, flattop_flags, strict=True)
            if d is not None and f
        ],
        dtype=bool,
    )
    result = {
        "arm": f"toroidal-harmonic-{args.origin_source}origin",
        "order": order,
        "ridge": ridge,
        "kind": "P",
        "origin_source": args.origin_source,
        "pole_source": args.pole_source,
        "pole_inboard_fraction": fraction,
        "mask_frac": args.mask_frac,
        "exclude_frac": args.exclude_frac,
        "ip_anchor": args.ip_anchor,
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
        "xpoint_tol": args.xpoint_tol,
        "diverted_fraction_all": float(div.mean()) if div.size else None,
        "diverted_fraction_flattop": float(div_ft.mean()) if div_ft.size else None,
        "n_boundary_found": int(div.size),
    }

    rtag = f"-r{ridge:g}" if ridge != 1e-8 else ""
    ftag = f"-frac{fraction:g}" if args.pole_source != "fixed" else "-fixedpole"
    suffix = f"-{args.origin_source}origin{ftag}{rtag}"
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
        "[harmonic o=%d ridge=%g %s%s] n=%d axis_skill=%.3f xpt_skill=%s "
        "lcfs_skill=%.3f lcfs_cm(all/ft)=%.1f/%s diverted_frac(all/ft)=%s/%s "
        "saddles_mean=%.2f consistency_rms=%s (%.1fs)",
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
        result["diverted_fraction_all"],
        result["diverted_fraction_flattop"],
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
        "--origin-source",
        choices=["centroid", "patch", "harmonic"],
        default="centroid",
        help="ray-cast origin + pole reference per slice: 'centroid' (default, "
        "SCORED) = the magnetically-constrained Ip-anchored current centroid "
        "(robust across ramp-up/flat-top, machine-agnostic, no interior carrier); "
        "'patch' = the free-current P3 inverse O-point (ablation; mislocates the "
        "axis ~30 cm at ramp-up); 'harmonic' = the harmonic field O-point (ablation)",
    )
    ap.add_argument(
        "--pole-source",
        choices=["track", "fixed"],
        default="track",
        help="'track' (default) = pole tracks the origin, inboard by the fraction; "
        "'fixed' = campaign nominal grid.r0 (ablation)",
    )
    ap.add_argument(
        "--pole-inboard-fractions",
        type=float,
        nargs="+",
        default=[0.25, 0.41, 0.55],
        help="swept DIMENSIONLESS inboard fractions: pole_r = origin_R*(1-fraction) "
        "(machine-agnostic; ring at the same relative inboard position on any "
        "device). fraction 0 = pole AT the origin degrades the inboard boundary",
    )
    ap.add_argument(
        "--pole-r",
        type=float,
        default=None,
        help="fixed-pole radius [m] when --pole-source fixed (default: grid.r0)",
    )
    ap.add_argument("--pole-z", type=float, default=0.0)
    ap.add_argument(
        "--mask-frac",
        type=float,
        default=0.5,
        help="near-pole invalid-disk mask radius as a FRACTION of the pole-to-origin "
        "distance (size- and machine-adaptive; a fixed metre mask reaches the axis "
        "at ramp-up and collapses the ray-cast)",
    )
    ap.add_argument(
        "--exclude-frac",
        type=float,
        default=1.1,
        help="drop harmonic critical points within this fraction of the pole-to-"
        "origin distance from the pole (residual near-pole artifacts)",
    )
    ap.add_argument(
        "--mask-radius", type=float, default=0.25, help="absolute mask [m] (fixed-pole)"
    )
    ap.add_argument(
        "--exclude-radius",
        type=float,
        default=0.55,
        help="absolute exclude [m] (fixed-pole)",
    )
    ap.add_argument(
        "--ip-anchor",
        action="store_true",
        help="add the poloidal-circulation Ip constraint to the harmonic fit",
    )
    ap.add_argument(
        "--xpoint-tol",
        type=float,
        default=0.05,
        help="EMERGENT-ONLY soft margin [m]: an X-point within this distance of "
        "the LCFS ring is ON the boundary -> diverted (report it); else limited "
        "(X-slots NaN). A soft read-off by proximity to the already-found "
        "boundary; the class is undefined near the transition. NOT a scored "
        "objective (LCFS shape is primary); tune on train, report on held-out.",
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
        "loaded %d shots (split=%s, origin=%s)",
        len(shots),
        args.split,
        args.origin_source,
    )
    if not shots:
        logger.error("no shots loaded — nothing to score")
        return 1
    # the free-current patch carrier is ONLY needed for the 'patch' origin ablation;
    # the default 'centroid' origin comes from the (cheap) moment fit per slice, so
    # the expensive ~13-min carrier inverse is skipped entirely.
    patch_psis = (
        free_current_psi(shots, device) if args.origin_source == "patch" else None
    )

    # order x ridge x fraction ladder; selection by lcfs_skill subject to the
    # consistency guard (rejects high-order aliasing where the annulus RMS blows up).
    fractions = args.pole_inboard_fractions if args.pole_source != "fixed" else [0.0]
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
        "consistency_rms=%s (split=%s origin=%s)",
        best["order"],
        best["ridge"],
        best["pole_inboard_fraction"],
        best["lcfs_skill"],
        best["xpoint_set_skill"],
        best["axis_skill"],
        best["consistency_rms_annulus"],
        args.split,
        args.origin_source,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
