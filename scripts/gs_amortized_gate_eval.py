#!/usr/bin/env python
"""Gate-2 re-measure: the TRAINED encoder amortizes the per-slice profile fit.

Per held-out slice: the checkpoint's profile head predicts (β0, α) from that
slice's features, ONE bootstrapped free-boundary solve reconstructs the
force-balanced ψ, the whitened magnetics cost of the reconstruction is
computed exactly as in the training-free fit, and the same honesty gates apply
(converged + cost ≤ limit, else masked).  Topology targets are read from the
solved ψ and scored against the firewalled EFIT referee — identical metrics to
``gs_solve_gate_eval.py`` so the two are directly comparable: the amortized
path replaces a 20-candidate solve grid with a single solve per slice.
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from gs_solve_gate_eval import TARGET_NAMES, equilibrium_target  # noqa: E402

from imas_ambix.gs.geometry import build_table_for_shot
from imas_ambix.gs.operator import build_operator
from imas_ambix.latent.data import (
    ANCHORED_NAMES,
    feature_schema,
    load_shot_windows,
    read_split_shot_lists,
)
from imas_ambix.latent.encoder import HybridLatentEncoder, LatentConfig
from imas_ambix.latent.evaluate import (
    headline_skill,
    matched_xpoint_error,
    per_quantity_skill,
)
from imas_ambix.latent.gs_solve import EquilibriumGrid, solve_equilibrium_bootstrapped

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("gs_amortized_gate_eval")

_WORKER: dict = {}


def _solve_slice(args):
    t, beta0, alpha, cost_limit = args
    grid = _WORKER["grid"]
    w = _WORKER["windows"]
    res = solve_equilibrium_bootstrapped(
        grid,
        w.i_pf[t],
        float(abs(w.anchored[t, 0])) * 1e3,
        beta0=float(beta0),
        alpha=float(alpha),
    )
    g_sens = _WORKER["g_sens"]
    pred = _WORKER["vac_by_slice"][t] + g_sens @ res.cell_currents
    mask = _WORKER["mask_by_slice"][t]
    resid = (pred[mask] - _WORKER["meas_by_slice"][t][mask]) / _WORKER["scale_ch"][mask]
    cost = float(np.sqrt(np.mean(resid * resid))) if mask.any() else np.inf
    if not res.converged or cost > cost_limit:
        return t, None, cost
    return t, equilibrium_target(grid, res), cost


_GRID_CACHE: dict = {}


def evaluate_shot(
    shot_id, *, encoder, stats, nr, nz, max_slices, min_ip_ka, cost_limit, workers
):
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

    # ONE encoder pass in the parent — workers only solve
    x = np.nan_to_num(stats.normalise(w.features_raw[valid]), nan=0.0)
    with torch.no_grad():
        lat = encoder(torch.tensor(x, dtype=torch.float64))
    profile = np.asarray(lat.profile)

    payload = {
        "grid": grid,
        "windows": w,
        "g_sens": g_sens,
        "scale_ch": scale_ch,
        "vac_by_slice": {},
        "meas_by_slice": {},
        "mask_by_slice": {},
    }
    for t in valid:
        vac = fwd.vacuum_prediction(w.i_pf[t])
        payload["vac_by_slice"][t] = np.where(
            present, vac[np.clip(ch_rows, 0, None)], 0.0
        )
        meas = w.raw_mag[t]
        payload["meas_by_slice"][t] = np.where(
            present, meas[np.clip(ch_rows, 0, None)], np.nan
        )
        mask = w.mag_mask[t]
        payload["mask_by_slice"][t] = (
            present & mask[np.clip(ch_rows, 0, None)]
        ) & np.isfinite(payload["meas_by_slice"][t])

    _WORKER.update(payload)
    jobs = [(t, profile[k, 0], profile[k, 1], cost_limit) for k, t in enumerate(valid)]
    if workers > 1:
        ctx = multiprocessing.get_context("fork")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
            results = list(pool.map(_solve_slice, jobs))
    else:
        results = [_solve_slice(j) for j in jobs]

    model, ref, costs, masked = [], [], [], 0
    for t, target, cost in results:
        if target is None:
            masked += 1
            continue
        model.append(target)
        ref.append(w.ref_target[t])
        costs.append(cost)
    logger.info(
        "shot %d: %d scored, %d masked (amortized single-solve)",
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
    ap.add_argument(
        "--checkpoint",
        type=str,
        default="imas_ambix/latent/artifacts/checkpoints/gs_grounded_latent.pt",
    )
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

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    extra = payload.get("extra", {})
    stats = extra["feature_stats"]
    cfg = extra.get("config", {})
    if not any(k.startswith("profile_head") for k in payload["encoder"]):
        logger.error("checkpoint has no profile head — train with fit targets first")
        return 1

    train_shots, held_shots = read_split_shot_lists(args.n_train, args.n_heldout)

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

    # encoder rebuilt at checkpoint dims; features counted from a held-out shot
    fwd0 = build_operator(build_table_for_shot(int(held_shots[0])))
    w0 = load_shot_windows(int(held_shots[0]), fwd0, "eval", schema)
    from imas_ambix.latent.data import build_campaign_operators

    gs_by_campaign, _lim, _cof = build_campaign_operators(
        held_shots[:1],
        grid_nr=int(cfg.get("grid_nr", 65)),
        grid_nz=int(cfg.get("grid_nz", 97)),
        profile_order=int(cfg.get("order", 1)),
    )
    encoder = HybridLatentEncoder(
        LatentConfig(
            n_features=w0.features_raw.shape[1],
            n_theta=next(iter(gs_by_campaign.values())).n_dof,
            n_anchored=len(ANCHORED_NAMES),
            n_free=int(cfg.get("n_free", 16)),
            hidden=int(cfg.get("hidden", 256)),
            depth=int(cfg.get("depth", 4)),
            profile_head=True,
        )
    ).double()
    encoder.load_state_dict(payload["encoder"])
    encoder.eval()

    per_shot = []
    for s in held_shots:
        try:
            r = evaluate_shot(
                s,
                encoder=encoder,
                stats=stats,
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
        "checkpoint": args.checkpoint,
        "step": int(payload.get("step", -1)),
        "n_scored": int(len(model)),
        "n_masked": int(n_masked),
        "n_candidate": int(n_candidate),
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
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "gs_amortized_gate_eval.json").write_text(json.dumps(result, indent=2))
    logger.info(
        "AMORTIZED GATE2 axis_skill=%s xpt_skill=%s "
        "axis_err median %.3f m (scored %d/%d)",
        result["axis_skill"],
        result["xpoint_set_skill"],
        result["axis_error_median_m"],
        result["n_scored"],
        result["n_candidate"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
