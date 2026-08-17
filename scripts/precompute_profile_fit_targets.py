#!/usr/bin/env python
"""Precompute per-slice profile-fit targets over the TRAINING corpus.

The locked ``greens-kernel-and-training-path`` decision trains the encoder via
FIT AMORTIZATION: for every training slice, the (β0, α) minimising the whitened
magnetics residual through the free-boundary GS solve — derived purely from
that slice's RAW measured magnetics, the KNOWN coil currents and the measured
Ip — becomes a label-free regression target for the encoder's profile heads
(alongside the soft GS residual).  This driver precomputes those fits once per
corpus so training never pays the ~30 s/slice solve cost.

Firewall: TRAIN split only, ``with_referee=False`` — no EFIT anywhere.
Resume-safe: one JSON per shot, existing files skipped; shardable with
``--shard-index/--shard-count`` for multi-node runs.
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from imas_ambix.data.description_reader import read_geometry_table
from imas_ambix.gs.operator import build_operator
from imas_ambix.latent.data import (
    feature_schema,
    load_shot_windows,
    read_split_shot_lists,
)
from imas_ambix.latent.gs_solve import EquilibriumGrid, fit_profile

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("precompute_profile_fit_targets")

_WORKER: dict = {}


def _fit_slice(args):
    t, cost_limit = args
    w = _WORKER["windows"]
    fit = fit_profile(
        _WORKER["grid"],
        _WORKER["table"],
        i_pf=w.i_pf[t],
        ip_amperes=float(abs(w.anchored[t, 0])) * 1e3,
        measured=_WORKER["meas_by_slice"][t],
        vacuum_prediction=_WORKER["vac_by_slice"][t],
        sensor_scale=_WORKER["scale_ch"],
        sensor_mask=_WORKER["mask_by_slice"][t],
        beta0_grid=(0.1, 0.3, 0.5, 0.7, 0.9),
        alpha_grid=(1.0, 2.0),
    )
    if fit is None:
        return t, None
    ar, az = fit.result.axis
    return t, {
        "time": float(w.times[t]),
        "beta0": fit.beta0,
        "alpha": fit.alpha,
        "cost": fit.cost,
        "converged": bool(fit.result.converged),
        "residual": float(fit.result.residual),
        "axis_r": float(ar),
        "axis_z": float(az),
        "ip_ka": float(w.anchored[t, 0]),
    }


_GRID_CACHE: dict = {}


def fit_shot(shot_id, *, nr, nz, max_slices, min_ip_ka, cost_limit, workers):
    table = read_geometry_table(int(shot_id))
    fwd = build_operator(table)
    cache_key = (table.signature.key, nr, nz)
    grid = _GRID_CACHE.get(cache_key)
    if grid is None:
        grid = EquilibriumGrid.from_table(table, nr=nr, nz=nz)
        _GRID_CACHE[cache_key] = grid
    grid.sensor_greens(table)  # build the cache before fork

    w = load_shot_windows(int(shot_id), fwd, "train", feature_schema())
    if w is None:
        return None

    row_of = {ch: i for i, ch in enumerate(fwd.sensor_channels)}
    _g, channels = grid.sensor_greens(table)
    ch_rows = np.array([row_of.get(ch, -1) for ch in channels])
    present = ch_rows >= 0

    valid = [t for t in range(w.times.size) if abs(w.anchored[t, 0]) > min_ip_ka]
    if len(valid) > max_slices:
        valid = valid[:: max(1, len(valid) // max_slices)][:max_slices]
    if not valid:
        return None

    scale = np.nanstd(w.raw_mag, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)
    scale_ch = np.where(present, scale[np.clip(ch_rows, 0, None)], 1.0)

    payload = {
        "grid": grid,
        "table": table,
        "windows": w,
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
        payload["mask_by_slice"][t] = present & mask[np.clip(ch_rows, 0, None)]

    _WORKER.update(payload)
    jobs = [(t, cost_limit) for t in valid]
    if workers > 1:
        ctx = multiprocessing.get_context("fork")  # SuperLU is not picklable
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
            results = list(pool.map(_fit_slice, jobs))
    else:
        results = [_fit_slice(j) for j in jobs]

    fits = [rec for _t, rec in results if rec is not None]
    return {
        "shot": int(shot_id),
        "campaign": w.campaign,
        "n_candidate": len(valid),
        "n_fit": len(fits),
        "cost_limit_hint": cost_limit,
        "fits": fits,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=200)
    ap.add_argument("--n-heldout", type=int, default=8)
    ap.add_argument("--nr", type=int, default=65)
    ap.add_argument("--nz", type=int, default=97)
    ap.add_argument("--max-slices-per-shot", type=int, default=40)
    ap.add_argument("--min-ip-ka", type=float, default=300.0)
    ap.add_argument("--cost-limit", type=float, default=3.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1)
    ap.add_argument(
        "--out", type=str, default="imas_ambix/latent/artifacts/fit_targets"
    )
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_shots, _held = read_split_shot_lists(args.n_train, args.n_heldout)
    shard = [
        s
        for i, s in enumerate(train_shots)
        if i % args.shard_count == args.shard_index
    ]
    logger.info(
        "shard %d/%d: %d of %d training shots",
        args.shard_index,
        args.shard_count,
        len(shard),
        len(train_shots),
    )

    done = skipped = failed = 0
    for s in shard:
        out_path = out_dir / f"shot{int(s)}.json"
        if out_path.exists():
            skipped += 1
            continue
        try:
            rec = fit_shot(
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
            failed += 1
            continue
        if rec is None:
            logger.info("shot %s: no usable slices", s)
            failed += 1
            continue
        tmp = out_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(rec, indent=1))
        tmp.replace(out_path)
        done += 1
        logger.info(
            "shot %s: %d/%d slices fitted -> %s",
            s,
            rec["n_fit"],
            rec["n_candidate"],
            out_path,
        )
    logger.info(
        "shard complete: %d fitted, %d skipped, %d failed", done, skipped, failed
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
