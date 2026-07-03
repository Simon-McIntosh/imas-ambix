#!/usr/bin/env python
"""Held-out evaluation of a trained GS-grounded latent checkpoint.

Measures, on the HELD-OUT shots (never trained on):

1. **Grounding (gate-1 at corpus scale)** — the whitened GS residual of the
   TRAINED encoder vs a random-init encoder of identical architecture, per
   held-out slice.  The trained/untrained ratio quantifies how much raw-
   magnetics structure the learned latent actually carries out of the corpus.
2. **Anchored-head skill** — normalised RMSE of the anchored (Ip, n_e) head
   against the corpus-mean baseline (RMSE 1 in normalised units by
   construction on the training distribution).
3. **Gate-3 re-verify** — minimum learned diffusivity (must stay > 0) and the
   command-load-bearing delta (volt-second prior with the true dI_pf/dt
   command vs a zeroed command) on held-out consecutive pairs.

The checkpoint's own normalisation stats (saved in ``extra``) are used —
never refit on eval data.  EFIT is not read anywhere here.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from imas_ambix.latent.data import (
    ANCHORED_NAMES,
    assemble_shot_windows,
    build_campaign_operators,
    feature_schema,
    read_split_shot_lists,
    sensor_scale_for_campaign,
)
from imas_ambix.latent.encoder import HybridLatentEncoder, LatentConfig
from imas_ambix.latent.engine import GSGroundedLatentEngine, LossWeights
from imas_ambix.latent.training import consecutive_pairs
from imas_ambix.latent.transport import FluxDiffusionPrior

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("eval_gs_grounded_checkpoint")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--checkpoint",
        type=str,
        default="imas_ambix/latent/artifacts/checkpoints/gs_grounded_latent.pt",
    )
    ap.add_argument("--n-train", type=int, default=200)
    ap.add_argument("--n-heldout", type=int, default=8)
    ap.add_argument(
        "--out", type=str, default="imas_ambix/latent/artifacts/checkpoint_eval.json"
    )
    args = ap.parse_args()

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    extra = payload.get("extra", {})
    stats = extra["feature_stats"]
    anchored_stats = extra["anchored_stats"]
    cmd_stats = extra.get("cmd_stats", {})
    cfg = extra.get("config", {})
    logger.info("checkpoint step %s (config %s)", payload.get("step"), cfg)

    _train, held = read_split_shot_lists(args.n_train, args.n_heldout)
    schema = feature_schema()
    gs_by_campaign, _lim, campaign_of = build_campaign_operators(
        held,
        grid_nr=int(cfg.get("grid_nr", 65)),
        grid_nz=int(cfg.get("grid_nz", 97)),
        profile_order=int(cfg.get("order", 1)),
    )
    windows = assemble_shot_windows(held, campaign_of, schema, with_referee=False)
    logger.info("%d held-out shot-windows", len(windows))
    if not windows:
        logger.error("no held-out windows")
        return 1

    n_features = windows[0].features_raw.shape[1]
    any_gs = next(iter(gs_by_campaign.values()))
    lat_cfg = LatentConfig(
        n_features=n_features,
        n_theta=any_gs.n_dof,
        n_anchored=len(ANCHORED_NAMES),
        n_free=int(cfg.get("n_free", 16)),
        hidden=int(cfg.get("hidden", 256)),
        depth=int(cfg.get("depth", 4)),
        profile_head=any(k.startswith("profile_head") for k in payload["encoder"]),
    )
    trained = HybridLatentEncoder(lat_cfg).double()
    trained.load_state_dict(payload["encoder"])
    trained.eval()
    torch.manual_seed(0)
    untrained = HybridLatentEncoder(lat_cfg).double()
    untrained.eval()

    per_shot = []
    cmd_deltas: list[float] = []
    d_mins: list[float] = []
    for w in windows:
        gs = gs_by_campaign[w.campaign].double()
        n_coil = gs.g_pf.shape[1]
        transport = FluxDiffusionPrior(
            nrho=gs.grid_nr, cmd_dim=max(n_coil, 1), feat_dim=lat_cfg.n_free
        ).double()
        t_state = payload.get("transport", {}).get(w.campaign)
        if t_state is not None:
            transport.load_state_dict(t_state)
        engines = {
            "trained": GSGroundedLatentEngine(
                trained, gs, transport, weights=LossWeights()
            ),
            "untrained": GSGroundedLatentEngine(
                untrained, gs, transport, weights=LossWeights()
            ),
        }

        x = np.nan_to_num(stats.normalise(w.features_raw), nan=0.0)
        xt = torch.tensor(x, dtype=torch.float64)
        ipf = torch.tensor(w.i_pf, dtype=torch.float64)
        raw = torch.tensor(np.nan_to_num(w.raw_mag, nan=0.0), dtype=torch.float64)
        mask = torch.tensor(np.isfinite(w.raw_mag))
        scale_np = sensor_scale_for_campaign([w], w.campaign, raw.shape[1])
        scale = torch.tensor(scale_np, dtype=torch.float64).expand_as(raw)

        rec = {"shot": w.shot_id, "n_slices": int(xt.shape[0])}
        with torch.no_grad():
            for name, eng in engines.items():
                lat = eng.encode(xt)
                rec[f"gs_residual_{name}"] = float(
                    eng.gs_residual_loss(lat, ipf, raw, scale, mask)
                )
            lat = engines["trained"].encode(xt)
            anch_pred = lat.anchored
            anch_tgt = anchored_stats.normalise(w.anchored)
            finite = np.isfinite(anch_tgt)
            err = np.asarray(anch_pred) - np.nan_to_num(anch_tgt, nan=0.0)
            for j, nm in enumerate(ANCHORED_NAMES):
                m = finite[:, j]
                if m.any():
                    rec[f"anchored_rmse_{nm}"] = float(np.sqrt(np.mean(err[m, j] ** 2)))
                    rec[f"baseline_rmse_{nm}"] = float(
                        np.sqrt(np.mean(np.asarray(anch_tgt)[m, j] ** 2))
                    )

            # gate-3 re-verify on consecutive pairs — only meaningful where the
            # checkpoint actually trained this campaign's transport (and its
            # command normalisation): a fresh transport + unnormalised dI/dt
            # would report an astronomically large, meaningless delta
            pairs = consecutive_pairs(w.times)
            if t_state is None or cmd_stats.get(w.campaign) is None:
                pairs = []
            if pairs:
                a_idx = [a for a, _b, _dt in pairs]
                b_idx = [b for _a, b, _dt in pairs]
                dt = float(np.mean([d for _a, _b, d in pairs]))
                eng = engines["trained"]
                lat_a = eng.encode(xt[a_idx])
                lat_b = eng.encode(xt[b_idx])
                prof_a, rho = eng.psi_profile(lat_a, ipf[a_idx])
                prof_b, _ = eng.psi_profile(lat_b, ipf[b_idx])
                cmd = (w.i_pf[b_idx] - w.i_pf[a_idx]) / max(dt, 1e-9)
                cst = cmd_stats.get(w.campaign)
                if cst is not None:
                    cmd = cst.normalise(cmd)
                cmd_t = torch.tensor(cmd, dtype=torch.float64)
                pri = transport.priors(
                    prof_a, prof_b, dt=dt, rho=rho, feat=lat_a.free, cmd=cmd_t
                )
                pri0 = transport.priors(
                    prof_a,
                    prof_b,
                    dt=dt,
                    rho=rho,
                    feat=lat_a.free,
                    cmd=torch.zeros_like(cmd_t),
                )
                rec["diffusivity_min"] = float(pri["diffusivity_min"])
                rec["command_load_bearing_delta"] = float(
                    abs(pri["volt_second"] - pri0["volt_second"])
                )
                d_mins.append(rec["diffusivity_min"])
                cmd_deltas.append(rec["command_load_bearing_delta"])
        per_shot.append(rec)
        logger.info("%s", rec)

    tr = [r["gs_residual_trained"] for r in per_shot]
    un = [r["gs_residual_untrained"] for r in per_shot]
    result = {
        "checkpoint": args.checkpoint,
        "step": int(payload.get("step", -1)),
        "heldout_gs_residual_trained_mean": float(np.mean(tr)),
        "heldout_gs_residual_untrained_mean": float(np.mean(un)),
        "grounding_ratio": float(np.mean(un) / max(np.mean(tr), 1e-12)),
        "diffusivity_min": float(np.min(d_mins)) if d_mins else None,
        "command_load_bearing_delta_mean": float(np.mean(cmd_deltas))
        if cmd_deltas
        else None,
        "per_shot": per_shot,
    }
    Path(args.out).write_text(json.dumps(result, indent=2))
    logger.info(
        "HELD-OUT grounding: trained %.3f vs untrained %.3f (ratio %.2fx); "
        "D_min=%s cmd-delta=%s -> %s",
        result["heldout_gs_residual_trained_mean"],
        result["heldout_gs_residual_untrained_mean"],
        result["grounding_ratio"],
        result["diffusivity_min"],
        result["command_load_bearing_delta_mean"],
        args.out,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
