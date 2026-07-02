#!/usr/bin/env python
"""Corpus-train the GS-grounded hybrid latent (f-ggl-02 — closing gate 2 for real).

The gate-2 measurement in ``scripts/gs_grounded_latent_run.py`` used a
TRAINING-FREE per-slice ridge GS-inverse as a diagnostic lower bound — not the
plan's actual design.  This script trains the intended object: ONE shared
:class:`~imas_ambix.latent.encoder.HybridLatentEncoder` across every campaign's
:class:`~imas_ambix.latent.engine.GSGroundedLatentEngine`
(:class:`~imas_ambix.latent.training.CorpusTrainer`), on the composite raw-signal
objective — GS residual vs RAW magnetics + anchored (Ip, n_e) supervision +
flux-diffusion transport guard-rails + dimensionless/KL regularisers — using
ONLY training shots (the firewalled EFIT referee is never read here).

The command channel driving the transport prior's inductive source is the
per-coil current derivative ``dI_pf/dt`` — a direct proxy for loop voltage
(V ∝ -dΦ/dt, and Φ is sourced inductively by the coil currents); this is a
documented modelling choice, not a hidden default.

In-process, SIGTERM-clean (atomic checkpoint on signal), resume-safe (--resume
picks up the latest checkpoint's optimiser + step count exactly).
"""

from __future__ import annotations

import argparse
import logging
import signal
import time
from pathlib import Path

import numpy as np
import torch

from imas_ambix.eval.excitation_selector import coil_ramp_profile
from imas_ambix.latent.data import (
    ANCHORED_NAMES,
    CorpusStats,
    build_campaign_operators,
    feature_schema,
    fit_corpus_stats,
    read_split_shot_lists,
    sensor_scale_for_campaign,
)
from imas_ambix.latent.encoder import HybridLatentEncoder, LatentConfig
from imas_ambix.latent.engine import GSGroundedLatentEngine, LossWeights
from imas_ambix.latent.training import CorpusTrainer, consecutive_pairs
from imas_ambix.latent.transport import FluxDiffusionPrior

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train_gs_grounded_latent")


class _ShotCache:
    """One campaign's shot windows + precomputed excitation-weighted pairs."""

    def __init__(self, campaign: str, gs, transport) -> None:
        self.campaign = campaign
        self.gs = gs
        self.transport = transport
        self.shots: list = []  # ShotWindows
        self.pairs: list[list[tuple[int, int, float]]] = []  # per-shot pairs
        self.pair_weights: list[np.ndarray] = []  # per-shot per-pair sample weight
        self.cmd_stats: CorpusStats | None = None  # dI_pf/dt normalisation


def _load_training_data(
    shots: list[int],
    *,
    grid_nr: int,
    grid_nz: int,
    profile_order: int,
    max_shots_per_campaign: int | None,
):
    """Build per-campaign GS operators + transport priors + shot windows.

    Never reads the EFIT referee (``with_referee=False`` throughout) — training
    stays behind the firewall by construction, not by discipline alone.
    """
    from imas_ambix.latent.data import assemble_shot_windows

    schema = feature_schema()
    gs_by_campaign, _limiter_by_campaign, campaign_of = build_campaign_operators(
        shots, grid_nr=grid_nr, grid_nz=grid_nz, profile_order=profile_order
    )
    logger.info("built %d campaign operators", len(gs_by_campaign))

    caches: dict[str, _ShotCache] = {}
    for key, gs in gs_by_campaign.items():
        n_coil = gs.g_pf.shape[1]
        n_free = 16
        transport = FluxDiffusionPrior(
            nrho=gs.grid_nr, cmd_dim=max(n_coil, 1), feat_dim=n_free
        )
        caches[key] = _ShotCache(key, gs, transport)

    windows = assemble_shot_windows(shots, campaign_of, schema, with_referee=False)
    logger.info("assembled %d shot-windows (no referee read)", len(windows))

    per_campaign_shots: dict[str, list] = {k: [] for k in caches}
    for w in windows:
        per_campaign_shots.setdefault(w.campaign, []).append(w)

    for key, cache in caches.items():
        shot_list = per_campaign_shots.get(key, [])
        if max_shots_per_campaign is not None:
            shot_list = shot_list[:max_shots_per_campaign]
        for w in shot_list:
            pairs = consecutive_pairs(w.times)
            if not pairs:
                continue
            # excitation weighting: |dI/dt| at the pair's start slice (the
            # transport prior's command is most informative on ramps, per §9's
            # excitation-curated-training requirement)
            ramp = coil_ramp_profile(w.i_pf, w.times)
            weights = np.array([ramp[a] for a, _b, _dt in pairs], dtype=np.float64)
            weights = weights + 1e-6 * weights.max() + 1e-9  # never fully zero
            cache.shots.append(w)
            cache.pairs.append(pairs)
            cache.pair_weights.append(weights / weights.sum())

    # The transport prior's command channel is dI_pf/dt (a loop-voltage proxy,
    # see the module docstring) — raw values reach ~1e7-1e8 A/s on a fast coil
    # ramp, which would blow up the untrained inductive_source Linear layer's
    # output (and the volt-second penalty that squares it) by many orders of
    # magnitude if fed in unnormalised.  Fit corpus-level (SI) stats per
    # campaign, exactly as for the input features and anchored scalars.
    for key, cache in caches.items():
        cmd_samples = []
        for w, pairs in zip(cache.shots, cache.pairs, strict=True):
            for a, b, dt in pairs:
                cmd_samples.append((w.i_pf[b] - w.i_pf[a]) / max(dt, 1e-9))
        if cmd_samples:
            cache.cmd_stats = fit_corpus_stats([np.array(cmd_samples)])

    return caches


def _build_sample(cache: _ShotCache, stats, anchored_stats, batch_size: int, rng):
    """One (t, t+1) minibatch for one campaign, excitation-weighted sampling."""
    n_shots = len(cache.shots)
    if n_shots == 0:
        return None
    x_t, x_tp1, i_pf_t, i_pf_tp1, raw_mag_t, cmd_t = [], [], [], [], [], []
    anchored_t, dts = [], []
    for _ in range(batch_size):
        si = rng.integers(0, n_shots)
        w = cache.shots[si]
        pairs = cache.pairs[si]
        weights = cache.pair_weights[si]
        pi = rng.choice(len(pairs), p=weights)
        a, b, dt = pairs[pi]
        x_t.append(stats.normalise(w.features_raw[a]))
        x_tp1.append(stats.normalise(w.features_raw[b]))
        i_pf_t.append(w.i_pf[a])
        i_pf_tp1.append(w.i_pf[b])
        raw_mag_t.append(w.raw_mag[a])
        cmd_t.append((w.i_pf[b] - w.i_pf[a]) / max(dt, 1e-9))  # dI/dt proxy [A/s]
        anchored_t.append(w.anchored[a])
        dts.append(dt)
    dt_mean = float(np.mean(dts))
    anchored_arr = anchored_stats.normalise(np.array(anchored_t))
    anchored_mask = np.isfinite(anchored_arr)
    anchored_arr = np.nan_to_num(anchored_arr, nan=0.0)
    cmd_arr = np.array(cmd_t)
    if cache.cmd_stats is not None:
        cmd_arr = cache.cmd_stats.normalise(cmd_arr)

    def _t(a):
        return torch.tensor(np.array(a), dtype=torch.float64)

    scale = sensor_scale_for_campaign(
        cache.shots, cache.campaign, len(cache.gs.sensor_channels)
    )
    n_b = len(x_t)
    raw_mag_arr = np.array(raw_mag_t)
    mask = np.isfinite(raw_mag_arr)
    return {
        "x_t": _t(x_t),
        "x_tp1": _t(x_tp1),
        "i_pf_t": _t(i_pf_t),
        "i_pf_tp1": _t(i_pf_tp1),
        "raw_mag_t": _t(np.nan_to_num(raw_mag_arr, nan=0.0)),
        "sensor_scale": _t(np.tile(scale, (n_b, 1))),
        "mag_mask": torch.tensor(mask),
        "cmd_t": _t(cmd_arr),
        "anchored_target": _t(anchored_arr),
        "anchored_mask": torch.tensor(anchored_mask),
        "dt": dt_mean,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=200)
    ap.add_argument("--n-heldout", type=int, default=8)
    ap.add_argument("--max-shots-per-campaign", type=int, default=None)
    ap.add_argument("--grid-nr", type=int, default=65)
    ap.add_argument("--grid-nz", type=int, default=97)
    ap.add_argument("--order", type=int, default=1)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n-free", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument(
        "--checkpoint-dir", type=str, default="imas_ambix/latent/artifacts/checkpoints"
    )
    ap.add_argument("--resume", action="store_true")
    ap.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "gs_grounded_latent.pt"

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    train_shots, _held = read_split_shot_lists(args.n_train, args.n_heldout)
    logger.info(
        "training on %d shots (held-out excluded, referee never read)", len(train_shots)
    )

    caches = _load_training_data(
        train_shots,
        grid_nr=args.grid_nr,
        grid_nz=args.grid_nz,
        profile_order=args.order,
        max_shots_per_campaign=args.max_shots_per_campaign,
    )
    caches = {k: c for k, c in caches.items() if c.shots}
    if not caches:
        logger.error("no campaign has usable training shots — aborting")
        return 1
    for k, c in caches.items():
        n_pairs = sum(len(p) for p in c.pairs)
        logger.info(
            "campaign %s: %d shots, %d training pairs", k, len(c.shots), n_pairs
        )

    n_features = next(iter(caches.values())).shots[0].features_raw.shape[1]
    stats = fit_corpus_stats([w.features_raw for c in caches.values() for w in c.shots])
    anchored_stats = fit_corpus_stats(
        [w.anchored for c in caches.values() for w in c.shots]
    )

    device = torch.device(args.device)
    n_dof = next(iter(caches.values())).gs.n_dof
    encoder = HybridLatentEncoder(
        LatentConfig(
            n_features=n_features,
            n_theta=n_dof,
            n_anchored=len(ANCHORED_NAMES),
            n_free=args.n_free,
            hidden=args.hidden,
            depth=args.depth,
        )
    ).double()

    engines: dict[str, GSGroundedLatentEngine] = {}
    for key, c in caches.items():
        c.gs = c.gs.double()
        c.transport = c.transport.double()
        engines[key] = GSGroundedLatentEngine(
            encoder, c.gs, c.transport, weights=LossWeights()
        )

    trainer = CorpusTrainer(encoder, engines, lr=args.lr)
    trainer.to(device)

    start_step = 0
    if args.resume and ckpt_path.exists():
        start_step = trainer.load(ckpt_path, map_location=str(device))
        logger.info("resumed from step %d", start_step)

    stop = {"flag": False}

    def _on_sigterm(*_a):
        logger.warning("SIGTERM received — checkpointing and exiting")
        stop["flag"] = True

    signal.signal(signal.SIGTERM, _on_sigterm)

    t0 = time.time()
    for step in range(start_step, args.steps):
        if stop["flag"]:
            break
        batch_fns = {
            key: (
                lambda c=c: (
                    _build_sample(c, stats, anchored_stats, args.batch_size, rng) or {}
                )
            )
            for key, c in caches.items()
        }
        totals = trainer.step(batch_fns)
        if step % 50 == 0:
            elapsed = time.time() - t0
            logger.info(
                "step %d/%d  totals=%s  (%.1fs elapsed)",
                step,
                args.steps,
                {k: round(v, 4) for k, v in totals.items()},
                elapsed,
            )
        if step > 0 and step % args.ckpt_every == 0:
            trainer.save(ckpt_path, step=step)
            logger.info("checkpoint saved at step %d -> %s", step, ckpt_path)

    trainer.save(ckpt_path, step=trainer.step_count)
    logger.info(
        "final checkpoint saved at step %d -> %s", trainer.step_count, ckpt_path
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
