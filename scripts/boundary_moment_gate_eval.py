"""Gate 1 for the constrained external-field (current-moment) boundary read.

Reads the plasma boundary (separatrix / X-point / LCFS radii) off a smooth psi
reconstructed from a low-order CURRENT-MOMENT fit to the external magnetics
(:mod:`imas_ambix.latent.boundary_moment`), instead of off the free ~5000-DOF
patch-current psi that scored LCFS skill -6.26.  Everything downstream of the
psi field -- the sign-aware axis read, the in-polygon X-point set, the fixed
8-angle LCFS ray-cast, and the per-quantity RMSE skill against the firewalled
EFIT referee -- is the SAME machinery the free-current gate uses
(``scripts/patch_gate_eval.py``), so the two reads are directly comparable.

Protocol (leakage-free, matches the P3 / A4 gate):

* ``--split train`` sweeps the moment order on the tuning cohort;
* ``--split eval`` scores the frozen order ONCE on the 160-slice held-out set
  (STANDING_HELD_OUT + test_ood_regime).

Per slice it also records the number of in-vessel saddles the topology read
finds -- the free-current read's spurious off-axis nulls are the mechanism that
under-sized the LCFS, so a drop in the saddle count is the direct evidence that
the boundary deficit was a representation artifact (Gate 1a), independent of the
skill recovery (Gate 1b/1c).

Writes ``.../patch_gate/boundary_read_moment-o<order>[-tune].json`` under the
latent artifacts.  No EFIT in any fit path; the referee only scores
(firewall: code-outputs-only).
"""

from __future__ import annotations

import argparse
import json
import logging
import time

import numpy as np

from imas_ambix.latent.boundary_moment import MomentFitConfig, invert_slices_moment
from imas_ambix.latent.data import read_split_shot_lists
from imas_ambix.latent.topology import find_critical_points

# Reuse the exact scoring core + payload builder of the free-current gate
# (script-dir import: run as `python scripts/boundary_moment_gate_eval.py`).
from patch_gate_eval import (
    ARTIFACTS,
    geometry_target,
    score,
    shot_payloads,
    train_mean_baseline,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("boundary-moment-gate")


def _count_saddles(psi2d, grid, axis_rz, limiter_r, limiter_z) -> int:
    """In-vessel saddle (X-point) count for one psi field -- the spurious-null
    proxy.  Uses the same critical-point finder the boundary read uses."""
    cp = find_critical_points(psi2d, grid.rg, grid.zg)
    if not len(cp.x_points):
        return 0
    from imas_ambix.latent.topology import _inside_polygon

    n = 0
    for xr, xz in cp.x_points:
        if _inside_polygon(float(xr), float(xz), limiter_r, limiter_z):
            n += 1
    return int(n)


def run(order: int, split: str, args) -> dict:
    train_shots, held_shots = read_split_shot_lists(args.n_train, args.n_heldout)
    baseline_vec = train_mean_baseline(args.n_train, args.n_baseline_shots, args.min_ip_ka)
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

    cfg = MomentFitConfig(order=order, ip_anchor=not args.no_ip_anchor)
    model_rows, ref_rows, flattop_flags = [], [], []
    saddles, ip_rel_errs, misfits = [], [], []
    t0 = time.perf_counter()
    for payload in shots:
        grid, basis = payload["grid"], payload["basis"]
        table = payload["table"]
        lim_r = np.asarray(table.limiter_r, dtype=float)
        lim_z = np.asarray(table.limiter_z, dtype=float)
        ips = np.abs([p.ip_amperes for p in payload["payloads"]])
        flattop_idx = int(np.argmax(ips)) if ips.size else -1
        inv = invert_slices_moment(basis, payload["payloads"], cfg)
        for k, r in enumerate(inv):
            psi2d = basis.psi_grid_2d_np(r.i_cell, payload["payloads"][k].i_pf)
            target, _, _ = geometry_target(psi2d, grid)
            model_rows.append(target)
            ref_rows.append(payload["refs"][k])
            flattop_flags.append(k == flattop_idx)
            saddles.append(_count_saddles(psi2d, grid, target[:2], lim_r, lim_z))
            ip_rel_errs.append(r.ip_rel_err)
            misfits.append(r.misfit)
    dt = time.perf_counter() - t0

    model = np.array(model_rows)
    ref = np.array(ref_rows)
    flattop_mask = np.array(flattop_flags, dtype=bool)
    sc = score(model, ref, baseline_vec)
    axis_err = sc.pop("axis_errors")

    # LCFS radial offset (cm) vs referee, per the free-current gate's stat.
    lcfs_model = model[:, 6:]
    lcfs_ref = ref[:, 6:]
    finite = np.isfinite(lcfs_ref)
    offset_cm = np.where(finite, np.abs(lcfs_model - lcfs_ref) * 100.0, np.nan)
    per_slice_median = np.nanmedian(offset_cm, axis=1)
    ft = per_slice_median[flattop_mask]
    result = {
        "arm": "current-moment",
        "order": order,
        "ip_anchor": not args.no_ip_anchor,
        "split": split,
        "n_scored": int(len(model)),
        "n_flattop_slices": int(flattop_mask.sum()),
        "wall_s": dt,
        **sc,
        "lcfs_offset_median_cm_all": float(np.nanmedian(per_slice_median)),
        "lcfs_offset_median_cm_flattop": (
            float(np.nanmedian(ft)) if ft.size else None
        ),
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
    tag = f"moment-o{order}" + ("" if split == "eval" else "-tune")
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / f"boundary_read_{tag}.json").write_text(json.dumps(result, indent=2))
    np.savez(
        ARTIFACTS / f"boundary_read_{tag}_arrays.npz",
        model=model,
        ref=ref,
        baseline=np.tile(baseline_vec, (len(model), 1)),
        axis_errors=axis_err,
        flattop_mask=flattop_mask,
        saddles=np.asarray(saddles),
    )
    logger.info(
        "[moment o=%d %s] n=%d axis_skill=%.3f xpt_skill=%s lcfs_skill=%.3f "
        "lcfs_cm(all/ft)=%.1f/%s saddles_mean=%.2f ip_rel_err_med=%.3g (%.0fs)",
        order,
        split,
        len(model),
        sc["axis_skill"],
        sc["xpoint_set_skill"],
        sc["lcfs_skill"],
        result["lcfs_offset_median_cm_all"],
        result["lcfs_offset_median_cm_flattop"],
        result["saddles_mean"] or -1.0,
        result["ip_rel_err_median"] or -1.0,
        dt,
    )
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=["train", "eval"], default="eval")
    ap.add_argument("--orders", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--no-ip-anchor", action="store_true")
    ap.add_argument("--nr", type=int, default=65)
    ap.add_argument("--nz", type=int, default=97)
    ap.add_argument("--n-train", type=int, default=40)
    ap.add_argument("--n-heldout", type=int, default=8)
    ap.add_argument("--n-baseline-shots", type=int, default=20)
    ap.add_argument("--n-tune-shots", type=int, default=8)
    ap.add_argument("--max-slices-per-shot", type=int, default=20)
    ap.add_argument("--min-ip-ka", type=float, default=50.0)
    args = ap.parse_args()

    summary = [run(o, args.split, args) for o in args.orders]
    best = max(summary, key=lambda d: d["lcfs_skill"])
    logger.info(
        "BEST order=%d lcfs_skill=%.3f (split=%s)",
        best["order"],
        best["lcfs_skill"],
        args.split,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
