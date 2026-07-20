#!/usr/bin/env python
"""Spine label factory: the frozen classical solve manufactures training labels.

Runs the frozen classical spine — staged-disc boundary read as a soft prior +
profile-ladder free-boundary GS solve — over corpus shots and writes per-shot
label shards: converged cell currents, ψ(R,Z), the normalised-flux map on the
plasma cells, profile coefficients, the 14-D geometry read, and a raw payload
snapshot (measured / vacuum / scale / mask / coil currents / Ip / n_e) so the
residual-operator trainer needs no further data assembly.

Firewall: EFIT is NEVER read here — slice selection is by plasma current only
and the labels descend from raw magnetics through our own physics.  The
standing held-out shots and the gate-eval cohort are refused outright.

Config is pinned to the frozen spine JSON
(``imas_ambix/latent/artifacts/patch_gate/closure_spine_frozen.json``); its
sha256 is recorded in every shard's provenance sidecar.  Throughput
(slices/s, single sequential warm-started chain per shot) is recorded per
shard — the cohort-sizing measurement the compute plan requires.

Shards: ``<out-dir>/shot_<id>.npz`` + ``shot_<id>.json`` (provenance).
Parallelism: one process per shot (the chain is sequential by design — the
previous slice's converged current warm-starts the next); shard across a CPU
fleet with ``--shot-index-range`` over the train manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from pathlib import Path

import numpy as np

from imas_ambix.gs.geometry import GEOMETRY_TABLE_VERSION, build_table_for_shot
from imas_ambix.gs.operator import COIL_MODEL_VERSION, build_operator
from imas_ambix.latent.boundary_disc import sensor_signature_arrays
from imas_ambix.latent.data import (
    STANDING_HELD_OUT,
    feature_schema,
    load_shot_windows,
    read_split_shot_lists,
    robust_channel_scale,
)
from imas_ambix.latent.gs_solve import EquilibriumGrid
from imas_ambix.latent.patch_basis import PatchBasis
from imas_ambix.latent.patch_inverse import SlicePayload
from scripts.closure_gate_eval import (
    _shot_passive_sidecar,
    fit_and_read_slice,
    geometry_target_pushout,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("spine_label_factory")

FROZEN_SPINE = Path("imas_ambix/latent/artifacts/patch_gate/closure_spine_frozen.json")
DEFAULT_OUT = Path("/work/projects/imas_gpu/mast/spine_labels")


def frozen_spine_config() -> tuple[dict, str]:
    """The pinned spine configuration + the sha256 of the frozen file."""
    raw = FROZEN_SPINE.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def factory_shot_payloads(
    shot: int,
    *,
    nr: int,
    nz: int,
    max_slices: int,
    min_ip_ka: float,
    table=None,
    cache_grid: bool = False,
) -> dict | None:
    """Referee-free per-shot payload assembly (mirror of the gate harness's
    ``shot_payloads`` with the EFIT referee load removed — labels must not
    depend on referee availability, and the firewall forbids reading it here).
    Slice selection is by plasma current alone.

    ``table`` (optional): a pre-built geometry table for this shot's campaign.
    The passive-structure geometry (amm) is a CAMPAIGN property, so a table
    built from any shot in the same campaign is valid; passing one lets callers
    score shots whose own zarr lacks the amm group (the sensor windows read
    below need only ama/amb/amc/ane).  When None the table is built from this
    shot as before."""
    if table is None:
        table = build_table_for_shot(int(shot))
    fwd = build_operator(table)
    # campaign-scope grid cache: the corpus factory processes a contiguous
    # range of shots per process, so same-campaign shots reuse the built grid,
    # Δ* factorisation, and Green's / interaction matrices instead of rebuilding
    # them per shot (greens-filament-solver §4).
    grid = EquilibriumGrid.from_table(table, nr=nr, nz=nz, cache=cache_grid)
    basis = PatchBasis.from_table(table, nr=nr, nz=nz)
    _g_sens, channels = grid.sensor_greens(table)

    w = load_shot_windows(
        int(shot), fwd, table.signature.key, feature_schema(), with_referee=False
    )
    if w is None:
        return None
    row_of = {ch: i for i, ch in enumerate(fwd.sensor_channels)}
    ch_rows = np.array([row_of.get(ch, -1) for ch in channels])
    present = ch_rows >= 0
    valid = [t for t in range(w.times.size) if abs(w.anchored[t, 0]) > min_ip_ka]
    if len(valid) > max_slices:
        valid = valid[:: max(1, len(valid) // max_slices)][:max_slices]
    if not valid:
        return None
    scale = robust_channel_scale(np.nanstd(w.raw_mag, axis=0), fwd.sensor_channels)
    scale_ch = np.where(present, scale[np.clip(ch_rows, 0, None)], 1.0)

    payloads, n_e = [], []
    for t in valid:
        vac = fwd.vacuum_prediction(w.i_pf[t])
        payloads.append(
            SlicePayload(
                measured=np.where(
                    present, w.raw_mag[t][np.clip(ch_rows, 0, None)], np.nan
                ),
                vacuum=np.where(present, vac[np.clip(ch_rows, 0, None)], 0.0),
                mask=present & w.mag_mask[t][np.clip(ch_rows, 0, None)],
                scale=scale_ch,
                i_pf=w.i_pf[t],
                ip_amperes=float(abs(w.anchored[t, 0])) * 1e3,
                shot=int(shot),
                t_index=int(t),
                time_s=float(w.times[t]),
            )
        )
        n_e.append(float(w.anchored[t, 1]))
    sr, sz, sang, is_flux = sensor_signature_arrays(table)
    return {
        "table": table,
        "grid": grid,
        "basis": basis,
        "payloads": payloads,
        "n_e": np.asarray(n_e, dtype=np.float64),
        "channels": channels,
        "sensor_geometry": (sr, sz, sang, is_flux),
        "campaign": table.signature.key,
    }


def run_shot(shot: int, args, spine: dict, config_sha: str) -> dict | None:
    """One shot's sequential warm-started label chain → shard on disk."""
    isolve = spine["interior_solve"]
    spc = dict(spine["soft_priors"])
    spc["boundary_prior"] = spc.pop("boundary_prior", "disc")

    payload = factory_shot_payloads(
        shot,
        nr=args.nr,
        nz=args.nz,
        max_slices=args.max_slices_per_shot,
        min_ip_ka=args.min_ip_ka,
        cache_grid=True,
    )
    if payload is None:
        logger.warning("shot %d: no usable payloads", shot)
        return None
    grid, table = payload["grid"], payload["table"]
    sidecar = _shot_passive_sidecar(payload, int(isolve["passive_k"]))
    cell_area = grid.dr * grid.dz
    modes = np.asarray(sidecar["modes"], dtype=np.float64)

    t0 = time.perf_counter()
    rows: list[dict] = []
    warm_jphi = None
    order = np.argsort([p.time_s for p in payload["payloads"]])
    n_candidate = len(order)
    for k in order:
        p = payload["payloads"][int(k)]
        f = fit_and_read_slice(
            grid,
            table,
            p,
            beta0_grid=(0.5,),
            alpha_grid=(1.0,),
            cost_limit=float("inf"),
            convergence_limit=args.convergence_limit,
            retry_max_iterations=args.retry_max_iterations,
            fit_mode="ladder",
            n_p=int(isolve["n_p"]),
            n_f=int(isolve["n_f"]),
            smoothness=float(isolve["smoothness"]),
            nonneg=isolve["profile_kind"] == "monomial-nonneg",
            passive=sidecar,
            passive_ridge=1.0,
            warm_jphi=warm_jphi,
            reseed_axis_r_max=float(isolve["reseed_axis_r_max"]),
            keep_psi=True,
            keep_jphi=True,
            basis=payload["basis"],
            meta={},
            soft_prior_cfg=spc,
            boundary_read=isolve["boundary_read_scoring"],
        )
        if not f.scored:
            logger.info("shot %d t=%.3fs masked (%s)", shot, p.time_s, f.reason)
            continue
        if f.converged:  # match the frozen harness chain: strict-converged only
            warm_jphi = f.jphi_flat
        i_cell = f.jphi_flat[grid.cells] * cell_area
        _t, psi_ax, psi_b = geometry_target_pushout(f.psi, grid)
        psi_n = (f.psi.ravel()[grid.cells] - psi_ax) / (psi_b - psi_ax)
        if f.passive_amp is not None:
            a_pass, *_ = np.linalg.lstsq(modes, np.asarray(f.passive_amp), rcond=None)
            sens_passive = np.asarray(sidecar["g_cols"], dtype=np.float64) @ a_pass
        else:
            sens_passive = np.zeros(p.measured.size)
        rows.append(
            {
                "t_index": p.t_index,
                "time_s": p.time_s,
                "ip_amperes": p.ip_amperes,
                "n_e": float(payload["n_e"][int(k)]),
                "i_pf": p.i_pf,
                "measured": p.measured,
                "vacuum": p.vacuum,
                "mask": p.mask,
                "sens_passive": sens_passive,
                "i_cell": i_cell,
                "psi": f.psi,
                "psi_n_cells": np.clip(psi_n, 0.0, 1.5),
                "coeffs": np.asarray(f.coeffs, dtype=np.float64),
                "target": f.target,
                "cost": f.cost,
                "converged": bool(f.converged),
                "reseeded": f.reason == "scored-reseeded",
                "ip_closure_rel": float(i_cell.sum() / p.ip_amperes - 1.0),
            }
        )
    wall_s = time.perf_counter() - t0
    if not rows:
        logger.warning("shot %d: 0/%d slices scored", shot, n_candidate)
        return None

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stack = {
        key: np.stack([np.asarray(r[key]) for r in rows]).astype(dt)
        for key, dt in (
            ("i_pf", np.float64),
            ("measured", np.float64),
            ("vacuum", np.float64),
            ("mask", bool),
            ("sens_passive", np.float64),
            ("i_cell", np.float32),
            ("psi", np.float32),
            ("psi_n_cells", np.float32),
            ("coeffs", np.float64),
            ("target", np.float64),
        )
    }
    sr, sz, sang, is_flux = payload["sensor_geometry"]
    np.savez_compressed(
        out_dir / f"shot_{shot}.npz",
        **stack,
        t_index=np.array([r["t_index"] for r in rows], dtype=np.int64),
        time_s=np.array([r["time_s"] for r in rows]),
        ip_amperes=np.array([r["ip_amperes"] for r in rows]),
        n_e=np.array([r["n_e"] for r in rows]),
        cost=np.array([r["cost"] for r in rows]),
        converged=np.array([r["converged"] for r in rows], dtype=bool),
        reseeded=np.array([r["reseeded"] for r in rows], dtype=bool),
        scale=np.asarray(payload["payloads"][0].scale, dtype=np.float64),
        sensor_r=sr,
        sensor_z=sz,
        sensor_angle_deg=sang,
        is_flux=is_flux.astype(bool),
        cells=payload["grid"].cells,
        limiter_r=np.asarray(grid.limiter_r, dtype=np.float64),
        limiter_z=np.asarray(grid.limiter_z, dtype=np.float64),
    )
    meta = {
        "shot": int(shot),
        "campaign": payload["campaign"],
        "split": "train",
        "spine_config": spine,
        "spine_config_sha256": config_sha,
        "coil_model_version": COIL_MODEL_VERSION,
        "geometry_table_version": GEOMETRY_TABLE_VERSION,
        "channels": payload["channels"],
        "n_candidate": n_candidate,
        "n_scored": len(rows),
        "wall_s": wall_s,
        "slices_per_s": len(rows) / max(wall_s, 1e-9),
        "reseed_fraction": float(np.mean([r["reseeded"] for r in rows])),
        "ip_closure_rel_max": float(
            np.max(np.abs([r["ip_closure_rel"] for r in rows]))
        ),
        "convergence_limit": args.convergence_limit,
        "max_slices_per_shot": args.max_slices_per_shot,
        "min_ip_ka": args.min_ip_ka,
        "nr": args.nr,
        "nz": args.nz,
    }
    (out_dir / f"shot_{shot}.json").write_text(json.dumps(meta, indent=2))
    logger.info(
        "shot %d: %d/%d scored in %.0f s (%.2f slices/s) -> %s",
        shot,
        len(rows),
        n_candidate,
        wall_s,
        meta["slices_per_s"],
        out_dir / f"shot_{shot}.npz",
    )
    return meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shots", type=str, default="", help="explicit comma list")
    ap.add_argument(
        "--shot-index-range",
        type=str,
        default="",
        help="'i0:i1' slice of the train-split manifest (fleet sharding)",
    )
    ap.add_argument("--n-train", type=int, default=200, help="manifest read depth")
    ap.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT))
    ap.add_argument("--max-slices-per-shot", type=int, default=30)
    ap.add_argument("--min-ip-ka", type=float, default=300.0)
    ap.add_argument("--nr", type=int, default=65)
    ap.add_argument("--nz", type=int, default=97)
    ap.add_argument("--convergence-limit", type=float, default=5e-3)
    ap.add_argument("--retry-max-iterations", type=int, default=160)
    args = ap.parse_args()

    spine, config_sha = frozen_spine_config()
    train_shots, held_shots = read_split_shot_lists(args.n_train, 8)
    forbidden = set(held_shots) | set(STANDING_HELD_OUT)

    if args.shots:
        shots = [int(s) for s in args.shots.split(",") if s.strip()]
    elif args.shot_index_range:
        i0, i1 = (int(v) for v in args.shot_index_range.split(":"))
        shots = train_shots[i0:i1]
    else:
        raise SystemExit("pass --shots or --shot-index-range")
    blocked = [s for s in shots if s in forbidden]
    if blocked:
        raise SystemExit(
            f"refusing held-out / eval-cohort shots {blocked} — labels are train-only"
        )

    logger.info("label factory: %d shots, config sha %s", len(shots), config_sha[:12])
    metas = []
    for s in shots:
        try:
            m = run_shot(int(s), args, spine, config_sha)
        except Exception as exc:  # noqa: BLE001 — one bad shot never kills a shard
            logger.warning("shot %s failed: %s", s, exc)
            continue
        if m is not None:
            metas.append(m)
    total = sum(m["n_scored"] for m in metas)
    wall = sum(m["wall_s"] for m in metas)
    logger.info(
        "DONE: %d shots, %d labelled slices, %.0f s chain time "
        "(%.2f slices/s sequential; per-node-hour at 8 concurrent chains ~%.0f)",
        len(metas),
        total,
        wall,
        total / max(wall, 1e-9),
        8 * 3600.0 * total / max(wall, 1e-9),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
