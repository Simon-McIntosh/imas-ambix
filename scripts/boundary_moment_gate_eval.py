"""Gate 1 for the constrained external-field (current-moment) boundary read.

Reads the plasma BOUNDARY (separatrix / X-point / LCFS radii) off a smooth psi
reconstructed from a low-order CURRENT-MOMENT fit to the external magnetics
(:mod:`imas_ambix.latent.boundary_moment`), instead of off the free ~5000-DOF
patch-current psi that scored LCFS skill -6.26.

HYBRID read (the plan's design, §5): the boundary (bounding flux, X-point set,
and the 8-angle LCFS ray-cast) is read from the constrained MOMENT psi, while
the LCFS ray-cast ORIGIN (the magnetic axis) comes from a stable interior
estimate rather than the low-DOF moment field's numerical O-point.  The topology
read and the firewalled-EFIT scoring are otherwise identical to the free-current
gate (``scripts/patch_gate_eval.py``), so the boundary representation is the only
varying factor.  ``--axis-source`` selects the ray origin: ``centroid`` (default)
= the moment fit's current centroid (free, analytic); ``patch`` = the
free-current P3 inverse axis (~2.8 cm, faithful but slow); ``moment`` = the
moment psi's numerical O-point (ablation -- the low-DOF field cannot localise the
axis, so the LCFS ray-cast degrades).

Protocol (leakage-free, matches the P3 / A4 gate): ``--split train`` sweeps the
moment order on the tuning cohort; ``--split eval`` scores the frozen order ONCE
on the 160-slice held-out set (STANDING_HELD_OUT + test_ood_regime).  Shots and
the free-current axes are loaded/inverted ONCE and reused across every order.

Per slice it records the in-vessel saddle count (the free-current read's
spurious off-axis nulls are what under-sized the LCFS, so a drop is direct
evidence the deficit was a representation artifact -- Gate 1a).  Writes
``.../patch_gate/boundary_read_moment-o<order>[-tune].json``.  No EFIT in any fit
path; the referee only scores (firewall: code-outputs-only).
"""

from __future__ import annotations

import argparse
import json
import logging
import time

import numpy as np
import torch

# Reuse the exact scoring core + payload builder + frozen inverse of the
# free-current gate (script-dir import: run from the scripts/ directory).
from patch_gate_eval import (
    ARTIFACTS,
    P3_WINNER_KW,
    score,
    shot_payloads,
    train_mean_baseline,
)

from imas_ambix.latent.boundary_moment import MomentFitConfig, invert_slices_moment
from imas_ambix.latent.data import read_split_shot_lists
from imas_ambix.latent.gs_solve import _read_axis, _read_boundary_psi_robust
from imas_ambix.latent.patch_inverse import InverseConfig, invert_slices
from imas_ambix.latent.topology import (
    _inside_polygon,
    find_critical_points,
    lcfs_radii,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("boundary-moment-gate")


def _sign_aware_axis(psi2d, grid):
    """The sign-aware magnetic-axis read of ``geometry_target`` (MAST: axis is
    the flux extremum furthest from the field median; sign chosen from the field)."""
    ax_pos, psi_pos = _read_axis(psi2d, grid, +1.0)
    ax_neg, psi_neg = _read_axis(psi2d, grid, -1.0)
    med = float(np.median(psi2d))
    if abs(psi_pos - med) >= abs(psi_neg - med):
        return ax_pos, psi_pos
    return ax_neg, psi_neg


def hybrid_target(psi_mom, grid, axis):
    """14-D geometry target with the AXIS supplied (from the patch psi) and the
    boundary read from ``psi_mom``.  Mirrors ``geometry_target`` exactly except
    the axis location is an input rather than read from the same field."""
    target = np.full(14, np.nan)
    target[0], target[1] = axis
    # confined-side flux reference on the moment field (its own extremum)
    _, axis_psi = _sign_aware_axis(psi_mom, grid)
    boundary_psi = _read_boundary_psi_robust(psi_mom, grid, tuple(axis), axis_psi)
    cp = find_critical_points(psi_mom, grid.rg, grid.zg)
    if cp.x_points.shape[0]:
        ins = _inside_polygon(
            cp.x_points[:, 0], cp.x_points[:, 1], grid.limiter_r, grid.limiter_z
        ) & grid.clear_of_conductors(cp.x_points[:, 0], cp.x_points[:, 1])
        pts = cp.x_points[ins]
        xpsi = cp.x_psi[ins]
        if pts.shape[0]:
            order = np.argsort(np.abs(xpsi - boundary_psi))
            for slot in range(min(2, pts.shape[0])):
                target[2 + 2 * slot] = pts[order[slot], 0]
                target[3 + 2 * slot] = pts[order[slot], 1]
    target[6:] = lcfs_radii(psi_mom, grid.rg, grid.zg, tuple(axis), boundary_psi)
    return target, float(axis_psi), float(boundary_psi)


def _count_saddles(psi2d, grid, limiter_r, limiter_z) -> int:
    cp = find_critical_points(psi2d, grid.rg, grid.zg)
    if not cp.x_points.shape[0]:
        return 0
    ins = _inside_polygon(cp.x_points[:, 0], cp.x_points[:, 1], limiter_r, limiter_z)
    return int(np.count_nonzero(ins))


def load_cohort(split, args):
    train_shots, held_shots = read_split_shot_lists(args.n_train, args.n_heldout)
    baseline_vec = train_mean_baseline(
        args.n_train, args.n_baseline_shots, args.min_ip_ka
    )
    eval_shots = (
        train_shots[args.n_baseline_shots : args.n_baseline_shots + args.n_tune_shots]
        if split == "train"
        else held_shots
    )
    shots = []
    for s in eval_shots:
        try:
            payload = shot_payloads(
                s,
                nr=args.nr,
                nz=args.nz,
                max_slices=args.max_slices_per_shot,
                min_ip_ka=args.min_ip_ka,
                split=split,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("shot %s failed to load: %s", s, exc)
            continue
        if payload is not None:
            shots.append(payload)
    return shots, baseline_vec


def free_current_psi(shots, device):
    """Invert every slice ONCE with the frozen P3-winner inverse and cache the
    patch psi field per slice (the interior carrier the hybrid axis reads)."""
    cfg = InverseConfig(iters=800, **P3_WINNER_KW)
    cache = []
    for payload in shots:
        basis = payload["basis"]
        inv = invert_slices(basis, payload["payloads"], cfg, device=device)
        psis = [
            basis.psi_grid_2d_np(r.i_cell, payload["payloads"][k].i_pf)
            for k, r in enumerate(inv)
        ]
        cache.append(psis)
    return cache


def score_order(shots, patch_psis, order, split, args) -> dict:
    cfg = MomentFitConfig(
        model=args.model, order=order, ip_anchor=not args.no_ip_anchor
    )
    model_rows, ref_rows, flattop_flags = [], [], []
    saddles, ip_rel_errs, misfits = [], [], []
    t0 = time.perf_counter()
    for si, payload in enumerate(shots):
        grid, basis, table = payload["grid"], payload["basis"], payload["table"]
        lim_r = np.asarray(table.limiter_r, dtype=float)
        lim_z = np.asarray(table.limiter_z, dtype=float)
        ips = np.abs([p.ip_amperes for p in payload["payloads"]])
        flattop_idx = int(np.argmax(ips)) if ips.size else -1
        inv = invert_slices_moment(basis, payload["payloads"], cfg)
        for k, r in enumerate(inv):
            psi_mom = basis.psi_grid_2d_np(r.i_cell, payload["payloads"][k].i_pf)
            if args.axis_source == "patch":
                axis, _ = _sign_aware_axis(patch_psis[si][k], grid)
            elif args.axis_source == "moment":
                axis, _ = _sign_aware_axis(psi_mom, grid)
            else:  # "centroid" (default) — the moment fit's current centroid
                axis = np.array([r.centroid_r, r.centroid_z])
            target, _, _ = hybrid_target(psi_mom, grid, axis)
            model_rows.append(target)
            ref_rows.append(payload["refs"][k])
            flattop_flags.append(k == flattop_idx)
            saddles.append(_count_saddles(psi_mom, grid, lim_r, lim_z))
            ip_rel_errs.append(r.ip_rel_err)
            misfits.append(r.misfit)
    dt = time.perf_counter() - t0

    model = np.array(model_rows)
    ref = np.array(ref_rows)
    flattop_mask = np.array(flattop_flags, dtype=bool)
    sc = score(model, ref, baseline_vec=args._baseline)
    axis_err = sc.pop("axis_errors")

    lcfs_model, lcfs_ref = model[:, 6:], ref[:, 6:]
    finite = np.isfinite(lcfs_ref)
    offset_cm = np.where(finite, np.abs(lcfs_model - lcfs_ref) * 100.0, np.nan)
    per_slice_median = np.nanmedian(offset_cm, axis=1)
    ft = per_slice_median[flattop_mask]
    result = {
        "arm": f"current-moment-{args.model}-{args.axis_source}-axis",
        "model": args.model,
        "order": order,
        "ip_anchor": not args.no_ip_anchor,
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
        "saddle_free_fraction": (
            float(np.mean(np.asarray(saddles) <= 1)) if saddles else None
        ),
        "ip_rel_err_median": float(np.median(ip_rel_errs)) if ip_rel_errs else None,
        "misfit_median": float(np.median(misfits)) if misfits else None,
    }
    parts = [] if args.model == "polynomial" else [args.model]
    if args.axis_source != "centroid":
        parts.append(f"{args.axis_source}axis")
    suffix = ("-" + "-".join(parts)) if parts else ""
    tag = f"moment-o{order}{suffix}" + ("" if split == "eval" else "-tune")
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
        "[moment o=%d %s%s] n=%d axis_skill=%.3f xpt_skill=%s lcfs_skill=%.3f "
        "lcfs_cm(all/ft)=%.1f/%s saddles_mean=%.2f ip_rel_err_med=%.2g (%.1fs)",
        order,
        split,
        suffix,
        len(model),
        sc["axis_skill"],
        sc["xpoint_set_skill"],
        sc["lcfs_skill"],
        result["lcfs_offset_median_cm_all"],
        result["lcfs_offset_median_cm_flattop"],
        result["saddles_mean"] if result["saddles_mean"] is not None else -1.0,
        result["ip_rel_err_median"]
        if result["ip_rel_err_median"] is not None
        else -1.0,
        dt,
    )
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=["train", "eval"], default="eval")
    ap.add_argument(
        "--model",
        choices=["polynomial", "gaussian"],
        default="polynomial",
        help="current model: 'polynomial' low-order moment basis (default) or "
        "'gaussian' compact elliptical blob",
    )
    ap.add_argument("--orders", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--no-ip-anchor", action="store_true")
    ap.add_argument(
        "--axis-source",
        choices=["centroid", "moment", "patch"],
        default="centroid",
        help=(
            "ray-cast axis: 'centroid' = the moment fit's current centroid "
            "(default, free); 'patch' = the free-current P3 inverse (faithful, "
            "slow); 'moment' = the moment psi's numerical O-point (ablation)."
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
    args = ap.parse_args()

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
    patch_psis = (
        free_current_psi(shots, device) if args.axis_source == "patch" else None
    )

    summary = [score_order(shots, patch_psis, o, args.split, args) for o in args.orders]
    best = max(summary, key=lambda d: d["lcfs_skill"])
    logger.info(
        "BEST order=%d lcfs_skill=%.3f axis_skill=%.3f (split=%s axis=%s)",
        best["order"],
        best["lcfs_skill"],
        best["axis_skill"],
        args.split,
        args.axis_source,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
