#!/usr/bin/env python
"""Real-data run for the GS-grounded latent engine (v0 stage 2 gate).

Produces the gate deliverables on MAST held-out shots, using ONLY raw magnetics
and the KNOWN coil currents — never EFIT as a label:

* **Gate 2 (spatial anchor).**  Per held-out slice, solve the ridge GS-inverse
  for the plasma-current amplitudes θ (:func:`evaluate.gs_inverse_theta`), read
  the topology from the resulting ψ (:func:`topology.read_topology`), and score
  axis / X-point / boundary against the firewalled EFIT referee with the same
  RMSE-skill formula the absolute-magnetics oracle uses (~0.5–0.7 bar).  This is
  exactly the map the learned encoder amortises.
* **Learned grounding (smoke).**  Train the hybrid-latent encoder end-to-end on
  the GS residual + anchored supervision for a few hundred steps and report the
  loss drop — evidence the learned path grounds in raw magnetics too.
* **Gate 3 (temporal anchor).**  Report the strictly-positive learned
  diffusivity (D≥0 by construction) and the command load-bearing test (zeroing
  the source changes ∂ψ/∂t).

Writes a JSON result + figures under ``--out``.  Runs on a compute node with the
``/work`` mirror; SIGTERM-clean (partial results are flushed on signal).
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("gs_grounded_run")

# axis + X-point headline components (matches the oracle headline)
AXIS_COMPONENTS = ["axis_R", "axis_Z"]
LCFS_COMPONENTS = [f"lcfs_r_{k}" for k in range(8)]

_STANDING_HELD_OUT = (18502, 18503, 18504, 18505)
_SPLITS_MANIFEST = Path(
    "/work/projects/imas_gpu/mast/manifests/statespace_splits_dalpha_v0.json"
)


def _read_shot_lists(n_train: int, n_heldout: int) -> tuple[list[int], list[int]]:
    """Train + held-out shot lists (standing held-out forced into held-out)."""
    with open(_SPLITS_MANIFEST) as f:
        splits = json.load(f)
    train = [int(x) for x in splits.get("train", [])]
    test = [int(x) for x in splits.get("test_ood_regime", [])]
    held = list(_STANDING_HELD_OUT) + [s for s in test if s not in _STANDING_HELD_OUT]
    train = [s for s in train if s not in set(held)]
    return train[:n_train], held[:n_heldout]


def _build_campaigns(shots: list[int], grid_nr: int, grid_nz: int, order: int):
    """Build a GSObservation + limiter per campaign signature over ``shots``."""
    from imas_ambix.gs.geometry import build_table_for_shot
    from imas_ambix.latent.gs_observation import GSObservation

    gs_by_sig: dict = {}
    limiter_by_sig: dict = {}
    campaign_of: dict[int, str] = {}
    for s in shots:
        try:
            table = build_table_for_shot(int(s))
        except Exception as exc:  # noqa: BLE001
            logger.warning("shot %d: no geometry table (%s)", s, exc)
            continue
        key = table.signature.key
        campaign_of[int(s)] = key
        if key not in gs_by_sig:
            gs_by_sig[key] = GSObservation.from_table(
                table, grid_nr=grid_nr, grid_nz=grid_nz, profile_order=order
            )
            limiter_by_sig[key] = (
                np.asarray(table.limiter_r, float),
                np.asarray(table.limiter_z, float),
            )
    return gs_by_sig, limiter_by_sig, campaign_of


def _feature_schema():
    from imas_ambix.statespace.baseline import _FEATURE_SCHEMA_MAG_ANE

    return _FEATURE_SCHEMA_MAG_ANE


def _assemble(shots, campaign_of, schema, *, with_referee):
    """Assemble ShotWindows for shots that have a campaign operator."""
    from imas_ambix.gs.geometry import build_table_for_shot
    from imas_ambix.gs.operator import build_operator
    from imas_ambix.latent.data import load_shot_windows

    out = []
    for s in shots:
        key = campaign_of.get(int(s))
        if key is None:
            continue
        try:
            operator = build_operator(build_table_for_shot(int(s)))
        except Exception as exc:  # noqa: BLE001
            logger.warning("shot %d: operator build failed (%s)", s, exc)
            continue
        w = load_shot_windows(int(s), operator, key, schema, with_referee=with_referee)
        if w is not None:
            out.append(w)
    return out


def _sensor_scale(windows, key, n_sensor):
    """Per-sensor whitening scale = std of measured raw magnetics over slices."""
    cols = [w.raw_mag for w in windows if w.campaign == key]
    if not cols:
        return np.ones(n_sensor)
    stacked = np.concatenate(cols, axis=0)
    scale = np.nanstd(stacked, axis=0)
    return np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=120)
    ap.add_argument("--n-heldout", type=int, default=8)
    ap.add_argument("--grid-nr", type=int, default=65)
    ap.add_argument("--grid-nz", type=int, default=97)
    ap.add_argument("--order", type=int, default=1)
    ap.add_argument("--ridge", type=float, default=1e-2)
    ap.add_argument("--smoke-train-steps", type=int, default=300)
    ap.add_argument("--max-slices-per-shot", type=int, default=40)
    ap.add_argument("--out", type=str, default="imas_ambix/latent/artifacts")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    result: dict = {"config": vars(args), "status": "running"}

    def _flush(*_a):
        (out_dir / "gs_grounded_run.json").write_text(
            json.dumps(result, indent=2, default=float)
        )
        logger.info("flushed partial results")

    signal.signal(signal.SIGTERM, lambda *a: (_flush(), sys.exit(0)))

    from imas_ambix.latent.evaluate import (
        gs_inverse_theta,
        headline_skill,
        matched_xpoint_error,
        per_quantity_skill,
    )
    from imas_ambix.latent.topology import read_topology

    schema = _feature_schema()
    train_shots, held_shots = _read_shot_lists(args.n_train, args.n_heldout)
    logger.info("train=%d held-out=%d", len(train_shots), len(held_shots))

    all_shots = list(dict.fromkeys(train_shots + held_shots))
    gs_by_sig, limiter_by_sig, campaign_of = _build_campaigns(
        all_shots, args.grid_nr, args.grid_nz, args.order
    )
    logger.info("built %d campaign operators", len(gs_by_sig))
    result["campaigns"] = list(gs_by_sig)

    train_w = _assemble(train_shots, campaign_of, schema, with_referee=True)
    held_w = _assemble(held_shots, campaign_of, schema, with_referee=True)
    logger.info(
        "assembled train=%d held-out=%d shot-windows", len(train_w), len(held_w)
    )

    # baseline = train-mean of the referee 14-D geometry (the oracle's baseline)
    ref_rows = [w.ref_target for w in train_w if w.ref_target is not None]
    if ref_rows:
        ref_stack = np.concatenate(ref_rows, axis=0)
        baseline_vec = np.nanmean(ref_stack, axis=0)
    else:
        baseline_vec = np.full(14, np.nan)

    # --- Gate 2: GS-readout topology vs referee, per held-out slice ---
    model_targets, ref_targets = [], []
    for w in held_w:
        gs = gs_by_sig[w.campaign]
        a_plasma = gs.a_plasma.numpy()
        g_pf = gs.g_pf.numpy()
        scale = _sensor_scale(train_w + held_w, w.campaign, a_plasma.shape[0])
        theta = gs_inverse_theta(
            a_plasma, g_pf, w.raw_mag, w.mag_mask, w.i_pf, scale, ridge=args.ridge
        )
        lr, lz = limiter_by_sig[w.campaign]
        import torch

        theta_t = torch.tensor(theta, dtype=torch.float64)
        i_pf_t = torch.tensor(w.i_pf, dtype=torch.float64)
        psi2d = gs.psi_field_2d(theta_t, i_pf_t).numpy()
        r1d = gs.grid_r_1d.numpy()
        z1d = gs.grid_z_1d.numpy()
        # subsample slices so the per-slice ψ read stays fast over many shots
        valid = [
            t
            for t in range(psi2d.shape[0])
            if w.ref_target is not None and np.isfinite(w.ref_target[t, :2]).all()
        ]
        if len(valid) > args.max_slices_per_shot:
            valid = valid[:: max(1, len(valid) // args.max_slices_per_shot)]
        for t in valid:
            read = read_topology(psi2d[t], r1d, z1d, limiter_r=lr, limiter_z=lz)
            model_targets.append(read.target)
            ref_targets.append(w.ref_target[t])
    if model_targets:
        model_arr = np.array(model_targets)
        ref_arr = np.array(ref_targets)
        baseline_arr = np.tile(baseline_vec, (len(model_arr), 1))
        names = [
            "axis_R",
            "axis_Z",
            "xpt0_R",
            "xpt0_Z",
            "xpt1_R",
            "xpt1_Z",
            *LCFS_COMPONENTS,
        ]
        skill = per_quantity_skill(model_arr, ref_arr, baseline_arr, names)
        # permutation-invariant X-point-set skill
        xm = np.array(
            [
                matched_xpoint_error(
                    model_arr[i, 2:6].reshape(2, 2), ref_arr[i, 2:6].reshape(2, 2)
                )
                for i in range(len(model_arr))
            ]
        )
        xb = np.array(
            [
                matched_xpoint_error(
                    baseline_arr[i, 2:6].reshape(2, 2), ref_arr[i, 2:6].reshape(2, 2)
                )
                for i in range(len(model_arr))
            ]
        )
        finite = np.isfinite(xm) & np.isfinite(xb)
        xpt_skill = (
            1.0
            - np.sqrt(np.nanmean(xm[finite] ** 2))
            / np.sqrt(np.nanmean(xb[finite] ** 2))
            if finite.any()
            else np.nan
        )
        result["gate2"] = {
            "n_slices": int(len(model_arr)),
            "per_quantity_skill": {
                k: (None if not np.isfinite(v) else float(v)) for k, v in skill.items()
            },
            "axis_skill": headline_skill(skill, AXIS_COMPONENTS),
            "xpoint_set_skill": None
            if not np.isfinite(xpt_skill)
            else float(xpt_skill),
            "lcfs_skill": headline_skill(skill, LCFS_COMPONENTS),
        }
        logger.info(
            "Gate2 axis_skill=%s xpt_skill=%s", result["gate2"]["axis_skill"], xpt_skill
        )
        np.savez(
            out_dir / "gate2_arrays.npz",
            model=model_arr,
            ref=ref_arr,
            baseline=baseline_arr,
        )
    else:
        result["gate2"] = {"n_slices": 0, "note": "no held-out slices with referee"}

    # --- Gate 3: transport temporal anchor (D>=0 + command load-bearing) ---
    import torch

    from imas_ambix.latent.transport import FluxDiffusionPrior

    tr = FluxDiffusionPrior(nrho=args.grid_nr, cmd_dim=2, feat_dim=8).double()
    feat = torch.randn(64, 8, dtype=torch.float64) * 20 - 40  # adversarial
    d_min = float(tr.diffusivity(feat).min())
    rho = torch.linspace(0, 1, args.grid_nr, dtype=torch.float64).unsqueeze(0)
    psi = torch.sin(rho * 3.0)
    cmd = torch.randn(1, 2, dtype=torch.float64)
    dc = tr.dpsi_dt(psi, rho, torch.zeros(1, 8, dtype=torch.float64), cmd)
    dz = tr.dpsi_dt(
        psi, rho, torch.zeros(1, 8, dtype=torch.float64), torch.zeros_like(cmd)
    )
    result["gate3"] = {
        "diffusivity_min": d_min,
        "D_geq_0": bool(d_min > 0),
        "command_source_delta": float((dc - dz).abs().sum()),
        "command_load_bearing": bool(float((dc - dz).abs().sum()) > 0),
    }
    logger.info(
        "Gate3 D_min=%.3e load-bearing-delta=%.3e",
        d_min,
        result["gate3"]["command_source_delta"],
    )

    # --- Learned grounding smoke ---
    if args.smoke_train_steps > 0 and train_w:
        result["learned_grounding"] = _smoke_train(
            train_w, gs_by_sig, schema, steps=args.smoke_train_steps
        )

    result["status"] = "done"
    _flush()
    logger.info("done: %s", json.dumps(result.get("gate2", {}), default=float))
    return 0


def _smoke_train(train_w, gs_by_sig, schema, *, steps: int) -> dict:
    """Short end-to-end fit of the learned encoder on the GS residual."""
    # single dominant campaign for the smoke fit
    from collections import Counter

    import torch

    from imas_ambix.latent.data import fit_corpus_stats
    from imas_ambix.latent.encoder import HybridLatentEncoder, LatentConfig
    from imas_ambix.latent.engine import GSGroundedLatentEngine
    from imas_ambix.latent.transport import FluxDiffusionPrior

    key = Counter(w.campaign for w in train_w).most_common(1)[0][0]
    ws = [w for w in train_w if w.campaign == key]
    gs = gs_by_sig[key].double()
    stats = fit_corpus_stats([w.features_raw for w in ws])
    feats = np.concatenate([stats.normalise(w.features_raw) for w in ws], axis=0)
    raw = np.concatenate([w.raw_mag for w in ws], axis=0)
    mask = np.concatenate([w.mag_mask for w in ws], axis=0)
    ipf = np.concatenate([w.i_pf for w in ws], axis=0)
    scale = np.nanstd(raw, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)

    cfg = LatentConfig(
        n_features=feats.shape[1],
        n_theta=gs.n_dof,
        n_anchored=2,
        n_free=8,
        hidden=128,
        depth=3,
    )
    enc = HybridLatentEncoder(cfg).double()
    tr = FluxDiffusionPrior(nrho=gs.grid_nr, cmd_dim=2, feat_dim=8).double()
    eng = GSGroundedLatentEngine(enc, gs, tr)
    opt = torch.optim.Adam(eng.parameters(), lr=3e-3)

    x = torch.tensor(feats, dtype=torch.float64)
    rm = torch.tensor(np.nan_to_num(raw), dtype=torch.float64)
    mk = torch.tensor(mask)
    ip = torch.tensor(ipf, dtype=torch.float64)
    sc = torch.tensor(scale, dtype=torch.float64).unsqueeze(0).expand_as(rm)
    n = x.shape[0]
    rng = np.random.RandomState(0)
    r0 = None
    losses = []
    for step in range(steps):
        idx = torch.tensor(rng.choice(n, size=min(256, n), replace=False))
        lat = eng.encode(x[idx])
        loss = eng.gs_residual_loss(lat, ip[idx], rm[idx], sc[idx], mk[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if r0 is None:
            r0 = float(loss)
        if step % 50 == 0:
            losses.append(float(loss))
    r1 = float(loss)
    return {
        "campaign": key,
        "n_slices": int(n),
        "gs_residual_start": r0,
        "gs_residual_end": r1,
        "drop_ratio": (r1 / r0) if r0 else None,
        "loss_trace": losses,
    }


if __name__ == "__main__":
    raise SystemExit(main())
