#!/usr/bin/env python
"""Gate-2 evaluation with the force-balanced ψ decoder (training-free baseline).

Per held-out slice: fit the two-parameter jφ(ψ_N; β0, α) profile against the
MEASURED magnetics through the free-boundary GS solve (measured Ip constraint,
KNOWN coil currents), read axis / X-point set / LCFS radii from the solved,
force-balanced ψ, and score against the firewalled EFIT referee with the
oracle's RMSE-skill formula.  Slices where no equilibrium converges, or the
fit cost exceeds the honesty threshold, are reported as masked — never scored
with fabricated geometry.

This replaces the linear GS-inverse gate run (whose ψ had no interior O-point
— see the psi-decoder-for-topology decision); the same fit is the map the
learned encoder amortises.
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from imas_ambix.gs.geometry import build_table_for_shot
from imas_ambix.gs.operator import build_operator
from imas_ambix.latent.data import (
    feature_schema,
    load_shot_windows,
    read_split_shot_lists,
)
from imas_ambix.latent.evaluate import (
    headline_skill,
    matched_xpoint_error,
    per_quantity_skill,
)
from imas_ambix.latent.gs_solve import EquilibriumGrid, fit_profile
from imas_ambix.latent.topology import (
    _inside_polygon,
    find_critical_points,
    lcfs_radii,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("gs_solve_gate_eval")

TARGET_NAMES = [
    "axis_R",
    "axis_Z",
    "xpt0_R",
    "xpt0_Z",
    "xpt1_R",
    "xpt1_Z",
    *[f"lcfs_r_{k}" for k in range(8)],
]

_WORKER: dict = {}


def equilibrium_target(grid: EquilibriumGrid, res) -> np.ndarray:
    """The oracle-shaped 14-D geometry read from one solved equilibrium."""
    target = np.full(14, np.nan)
    ar, az = res.axis
    target[0], target[1] = ar, az
    psi2d = res.psi
    cp = find_critical_points(psi2d, grid.rg, grid.zg)
    if cp.x_points.shape[0]:
        ins = _inside_polygon(
            cp.x_points[:, 0], cp.x_points[:, 1], grid.limiter_r, grid.limiter_z
        )
        pts = cp.x_points[ins]
        xpsi = cp.x_psi[ins]
        if pts.shape[0]:
            order = np.argsort(np.abs(xpsi - res.boundary_psi))
            for slot in range(min(2, pts.shape[0])):
                target[2 + 2 * slot] = pts[order[slot], 0]
                target[3 + 2 * slot] = pts[order[slot], 1]
    target[6:] = lcfs_radii(psi2d, grid.rg, grid.zg, (ar, az), res.boundary_psi)
    return target


def _init_worker(shot_payload):
    """Fork-shared per-shot state for the slice workers."""
    _WORKER.update(shot_payload)


def _fit_slice(args):
    t, cost_limit = args
    grid = _WORKER["grid"]
    table = _WORKER["table"]
    w = _WORKER["windows"]
    vac_ch = _WORKER["vac_by_slice"][t]
    meas_ch = _WORKER["meas_by_slice"][t]
    mask_ch = _WORKER["mask_by_slice"][t]
    scale_ch = _WORKER["scale_ch"]
    fit = fit_profile(
        grid,
        table,
        i_pf=w.i_pf[t],
        ip_amperes=float(abs(w.anchored[t, 0])) * 1e3,
        measured=meas_ch,
        vacuum_prediction=vac_ch,
        sensor_scale=scale_ch,
        sensor_mask=mask_ch,
        beta0_grid=(0.1, 0.3, 0.5, 0.7, 0.9),
        alpha_grid=(1.0, 2.0),
    )
    if fit is None or fit.cost > cost_limit or not fit.result.converged:
        return t, None, (fit.cost if fit else None)
    return t, equilibrium_target(grid, fit.result), fit.cost


_GRID_CACHE: dict = {}


def evaluate_shot(shot_id, *, nr, nz, max_slices, min_ip_ka, cost_limit, workers):
    table = build_table_for_shot(int(shot_id))
    fwd = build_operator(table)
    cache_key = (table.signature.key, nr, nz)
    grid = _GRID_CACHE.get(cache_key)
    if grid is None:
        grid = EquilibriumGrid.from_table(table, nr=nr, nz=nz)
        _GRID_CACHE[cache_key] = grid
    g_sens, channels = grid.sensor_greens(table)

    w = load_shot_windows(
        int(shot_id), fwd, "eval", feature_schema(), with_referee=True
    )
    if w is None or w.ref_target is None:
        return None

    # align raw_mag (operator row order) onto the sensor_greens channel order
    row_of = {ch: i for i, ch in enumerate(fwd.sensor_channels)}
    ch_rows = np.array([row_of.get(ch, -1) for ch in channels])
    present = ch_rows >= 0

    valid = [
        t
        for t in range(w.times.size)
        if np.isfinite(w.ref_target[t, :2]).all() and abs(w.anchored[t, 0]) > min_ip_ka
    ]
    if len(valid) > max_slices:
        valid = valid[:: max(1, len(valid) // max_slices)][:max_slices]
    if not valid:
        return None

    scale = np.nanstd(w.raw_mag, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)
    scale_ch = np.where(present, scale[np.clip(ch_rows, 0, None)], 1.0)

    vac_by_slice = {}
    meas_by_slice = {}
    mask_by_slice = {}
    for t in valid:
        vac = fwd.vacuum_prediction(w.i_pf[t])
        vac_by_slice[t] = np.where(present, vac[np.clip(ch_rows, 0, None)], 0.0)
        meas = w.raw_mag[t]
        meas_by_slice[t] = np.where(present, meas[np.clip(ch_rows, 0, None)], np.nan)
        mask = w.mag_mask[t]
        mask_by_slice[t] = present & mask[np.clip(ch_rows, 0, None)]

    payload = {
        "grid": grid,
        "table": table,
        "windows": w,
        "vac_by_slice": vac_by_slice,
        "meas_by_slice": meas_by_slice,
        "mask_by_slice": mask_by_slice,
        "scale_ch": scale_ch,
    }
    # populate the worker state IN THE PARENT and fork: children inherit the
    # (unpicklable SuperLU-holding) grid by copy-on-write; Python 3.14's default
    # forkserver start method would try to pickle it
    _init_worker(payload)
    results = []
    if workers > 1:
        ctx = multiprocessing.get_context("fork")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
            results = list(pool.map(_fit_slice, [(t, cost_limit) for t in valid]))
    else:
        results = [_fit_slice((t, cost_limit)) for t in valid]

    model, ref, costs, masked = [], [], [], 0
    for t, target, cost in results:
        if target is None:
            masked += 1
            continue
        model.append(target)
        ref.append(w.ref_target[t])
        costs.append(cost)
    logger.info(
        "shot %d: %d scored, %d masked (no converged/low-cost equilibrium)",
        shot_id,
        len(model),
        masked,
    )
    return {
        "shot": int(shot_id),
        "model": np.array(model) if model else np.zeros((0, 14)),
        "ref": np.array(ref) if ref else np.zeros((0, 14)),
        "costs": costs,
        "masked": masked,
        "n_candidate": len(valid),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=40)
    ap.add_argument("--n-heldout", type=int, default=8)
    ap.add_argument("--nr", type=int, default=65)
    ap.add_argument("--nz", type=int, default=97)
    ap.add_argument("--max-slices-per-shot", type=int, default=20)
    ap.add_argument("--min-ip-ka", type=float, default=300.0)
    ap.add_argument("--cost-limit", type=float, default=3.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--n-baseline-shots", type=int, default=10)
    ap.add_argument("--out", type=str, default="imas_ambix/latent/artifacts")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_shots, held_shots = read_split_shot_lists(args.n_train, args.n_heldout)

    # baseline = train-mean of the referee geometry over a few TRAIN shots
    schema = feature_schema()
    base_rows = []
    for s in train_shots[: args.n_baseline_shots]:
        try:
            fwd = build_operator(build_table_for_shot(int(s)))
        except Exception:  # noqa: BLE001
            continue
        wtr = load_shot_windows(int(s), fwd, "train", schema, with_referee=True)
        if wtr is not None and wtr.ref_target is not None:
            on = np.abs(wtr.anchored[:, 0]) > args.min_ip_ka
            base_rows.append(wtr.ref_target[on])
    baseline_vec = (
        np.nanmean(np.concatenate(base_rows, axis=0), axis=0)
        if base_rows
        else np.full(14, np.nan)
    )
    logger.info(
        "baseline (train-mean) axis: (%.3f, %.3f)", baseline_vec[0], baseline_vec[1]
    )

    per_shot = []
    for s in held_shots:
        try:
            r = evaluate_shot(
                s,
                nr=args.nr,
                nz=args.nz,
                max_slices=args.max_slices_per_shot,
                min_ip_ka=args.min_ip_ka,
                cost_limit=args.cost_limit,
                workers=args.workers,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("shot %s failed: %s", s, exc)
            continue
        if r is not None:
            per_shot.append(r)

    model = np.concatenate([r["model"] for r in per_shot], axis=0)
    ref = np.concatenate([r["ref"] for r in per_shot], axis=0)
    n_masked = sum(r["masked"] for r in per_shot)
    n_candidate = sum(r["n_candidate"] for r in per_shot)
    baseline = np.tile(baseline_vec, (len(model), 1))

    skill = per_quantity_skill(model, ref, baseline, TARGET_NAMES)
    xm = np.array(
        [
            matched_xpoint_error(model[i, 2:6].reshape(2, 2), ref[i, 2:6].reshape(2, 2))
            for i in range(len(model))
        ]
    )
    xb = np.array(
        [
            matched_xpoint_error(
                baseline[i, 2:6].reshape(2, 2), ref[i, 2:6].reshape(2, 2)
            )
            for i in range(len(model))
        ]
    )
    finite = np.isfinite(xm) & np.isfinite(xb)
    xpt_skill = (
        1.0
        - np.sqrt(np.nanmean(xm[finite] ** 2)) / np.sqrt(np.nanmean(xb[finite] ** 2))
        if finite.any()
        else np.nan
    )
    axis_err = np.hypot(model[:, 0] - ref[:, 0], model[:, 1] - ref[:, 1])

    result = {
        "n_scored": int(len(model)),
        "n_masked": int(n_masked),
        "n_candidate": int(n_candidate),
        "scored_fraction": float(len(model) / max(n_candidate, 1)),
        "axis_error_mean_m": float(np.nanmean(axis_err)),
        "axis_error_median_m": float(np.nanmedian(axis_err)),
        "per_quantity_skill": {
            k: (None if not np.isfinite(v) else float(v)) for k, v in skill.items()
        },
        "axis_skill": headline_skill(skill, ["axis_R", "axis_Z"]),
        "xpoint_set_skill": None if not np.isfinite(xpt_skill) else float(xpt_skill),
        "lcfs_skill": headline_skill(skill, [f"lcfs_r_{k}" for k in range(8)]),
        "per_shot": [
            {"shot": r["shot"], "scored": int(len(r["model"])), "masked": r["masked"]}
            for r in per_shot
        ],
        "config": vars(args),
    }
    (out_dir / "gs_solve_gate_eval.json").write_text(json.dumps(result, indent=2))
    np.savez(
        out_dir / "gs_solve_gate_arrays.npz", model=model, ref=ref, baseline=baseline
    )
    logger.info(
        "GATE2 axis_skill=%s xpt_skill=%s lcfs_skill=%s axis_err median %.3f m (scored %d/%d)",
        result["axis_skill"],
        result["xpoint_set_skill"],
        result["lcfs_skill"],
        result["axis_error_median_m"],
        result["n_scored"],
        result["n_candidate"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
