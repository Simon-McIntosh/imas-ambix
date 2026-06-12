"""Conditioning ablation — what does the actuator vector buy the arm?

The matched-arm verdict isolated the value of *temporal attention*; this
isolates the value of *conditioning*.  Three dynamics arms (temporal
attention ON, everything else matched) differ ONLY in which conditioning
channels they may see:

``full``
    The complete physical conditioning vector (the existing
    ``cap_v1_dynamics`` arm).  Not retrained here — the matched-budget
    comparison uses its ``step8000.pt`` checkpoint.

``ip_ne``
    Only ``plasma_current`` + ``ne_line_integrated`` (the minimal scalar
    pair).  Every other channel is suppressed.

``none``
    No conditioning at all (camera-only).  Every channel suppressed.

How a channel is "suppressed" without touching the locked model
---------------------------------------------------------------
The model's ``cond_proj`` consumes ``[values | missing | dt]``; the
parameter count is fixed by ``cond_channels`` (the full set), so to keep
all three arms byte-identical in architecture we do NOT shrink
``cond_channels``.  Instead a channel is suppressed by zeroing its
(z-scored) value AND raising its missing-flag to 1.0 — exactly the
"actuator absent for this shot" signal the model already learns to ignore
(``conditioning.py`` missingness contract).  ``none`` suppresses every
channel; ``ip_ne`` keeps only the two scalar columns.  Δt is never
suppressed (it is cadence, not an actuator).

This is applied by overriding the trainer's single conditioning
chokepoint (``_batch_to_tensors``) in a thin subclass, so train, periodic
VAL, and the final held-out eval all see the same suppression — no edit to
the locked ``train.py`` / ``loader.py``.

Reduced budget (honest)
-----------------------
The two new arms train for 8000 steps (turnaround), versus the full arm's
20000.  The matched-budget comparison therefore uses the full arm's
``step8000.pt`` checkpoint, NOT ``final.pt`` — so all three arms are
compared at an identical 8000-step token budget.  :func:`compare_conditioning`
scores the three at 8000 steps on the held-out suite (mixture + named
geometries + the frontier forecasting mode, where conditioning should
matter most).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from imas_ambix.camdyn.conditioning import CONDITIONING_CHANNELS
from imas_ambix.camdyn.train import TrainConfig, Trainer, _normalise_conditioning

logger = logging.getLogger(__name__)

# Channel-key sets per conditioning regime.
KEEP_FULL: tuple[str, ...] = tuple(c.key for c in CONDITIONING_CHANNELS)
KEEP_IP_NE: tuple[str, ...] = ("plasma_current", "ne_line_integrated")
KEEP_NONE: tuple[str, ...] = ()


def channel_keep_mask(keep_keys) -> np.ndarray:
    """Boolean ``(C,)`` over CONDITIONING_CHANNELS — True = channel kept."""
    keep = set(keep_keys)
    return np.array([c.key in keep for c in CONDITIONING_CHANNELS], dtype=bool)


def suppress_conditioning(cond_values, cond_missing, keep_mask):
    """Zero values + flag-missing every channel NOT in ``keep_mask``.

    Operates on RAW (pre-z-score) values: a suppressed channel becomes the
    "absent actuator" signal (value 0, missing 1).  Returns new arrays;
    inputs are not mutated.  Shapes ``(B,F,C)``.
    """
    cv = np.array(cond_values, dtype=np.float32, copy=True)
    cm = np.array(cond_missing, dtype=np.float32, copy=True)
    drop = ~np.asarray(keep_mask, dtype=bool)
    cv[..., drop] = 0.0
    cm[..., drop] = 1.0
    return cv, cm


class CondMaskedTrainer(Trainer):
    """Trainer that suppresses all conditioning channels outside a keep-set.

    Overrides the single conditioning chokepoint so training, VAL, and the
    held-out eval all see the masked conditioning, without editing the
    locked trainer.  ``keep_keys`` is the conditioning regime; an empty set
    is the camera-only (``none``) arm.
    """

    def __init__(self, cfg: TrainConfig, keep_keys) -> None:
        super().__init__(cfg)
        self._keep_mask = channel_keep_mask(keep_keys)
        self._keep_keys = tuple(keep_keys)

    def _batch_to_tensors(self, arr, torch, device):
        # Suppress RAW conditioning columns BEFORE the parent z-scores them.
        cv, cm = suppress_conditioning(
            arr["cond_values"], arr["cond_missing"], self._keep_mask
        )
        masked = dict(arr)
        masked["cond_values"] = cv
        masked["cond_missing"] = cm
        return super()._batch_to_tensors(masked, torch, device)


# ---------------------------------------------------------------------------
# Matched-budget cross-arm comparison (eval-only)
# ---------------------------------------------------------------------------


def _suppressed_score(model, batches, torch, dev, keep_mask, cond_stats, *, named=None):
    """Score cached batches with a channel-keep mask applied to conditioning.

    Mirrors ``Trainer._score_cached`` but z-scores the suppressed
    conditioning (so a ``none``/``ip_ne`` arm is evaluated under the SAME
    suppression it trained with).  Returns flattened per-token nll/acc.
    """
    from imas_ambix.camdyn.masking import named_geometry_mask  # noqa: PLC0415
    from imas_ambix.camdyn.model import score_window_bits  # noqa: PLC0415

    nll_all, acc_all = [], []
    with torch.no_grad():
        for arr in batches:
            cv, cm = suppress_conditioning(
                arr["cond_values"], arr["cond_missing"], keep_mask
            )
            cvz = _normalise_conditioning(cv, cond_stats)
            t_tokens = torch.from_numpy(np.ascontiguousarray(arr["tokens"])).to(dev)
            if named is not None:
                nf = arr["tokens"].shape[1]
                gmask = named_geometry_mask(named, nf)
                vis = np.broadcast_to(gmask[None], arr["visible"].shape).copy()
            else:
                vis = arr["visible"]
            loss_mask_np = ~vis
            t_vis = torch.from_numpy(np.ascontiguousarray(vis)).to(dev)
            t_cv = torch.from_numpy(cvz.astype(np.float32)).to(dev)
            t_cm = torch.from_numpy(cm.astype(np.float32)).to(dev)
            t_dt = torch.from_numpy(arr["dt"].astype(np.float32)).to(dev)
            with torch.autocast(
                device_type=dev.type,
                dtype=torch.bfloat16,
                enabled=(dev.type == "cuda"),
            ):
                logits = model.module(t_tokens, t_vis, t_cv, t_cm, t_dt)
            bl = logits.float().cpu().numpy()
            for b in range(bl.shape[0]):
                vf = arr["valid"][b]
                lm_b = loss_mask_np[b] & vf[:, None, None]
                sc = score_window_bits(bl[b], arr["tokens"][b], lm_b)
                if not sc.n:
                    continue
                nll_all.append(sc.nll_per_token)
                acc_all.append(sc.acc_per_token)

    def _cat(xs):
        return np.concatenate(xs) if xs else np.array([])

    return _cat(nll_all), _cat(acc_all)


def compare_conditioning(
    arm_ckpts: dict,
    *,
    split_path=None,
    device="cuda",
    eval_seed=999,
    frontier=8,
):
    """Score the three conditioning arms on the held-out suite at matched budget.

    ``arm_ckpts`` maps ``{"full":path, "ip_ne":path, "none":path}``; each is
    a step-8000 checkpoint.  Every arm is scored with ITS OWN trained
    conditioning regime (full sees all channels; ip_ne/none have the
    matching channels suppressed) on the SAME materialised held-out windows
    (mixture mask + named geometries + the frontier forecasting mode).
    """
    import torch  # noqa: PLC0415

    from imas_ambix.camdyn.arm_compare import _load_arm  # noqa: PLC0415
    from imas_ambix.camdyn.dataset import FrameWindowConfig  # noqa: PLC0415
    from imas_ambix.camdyn.masking import NAMED_GEOMETRIES  # noqa: PLC0415
    from imas_ambix.camdyn.metrics import bootstrap_ci  # noqa: PLC0415
    from imas_ambix.camdyn.train import _agg, _specs_for_shots  # noqa: PLC0415

    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    regimes = {
        "full": KEEP_FULL,
        "ip_ne": KEEP_IP_NE,
        "none": KEEP_NONE,
    }
    models = {}
    stats = {}
    full_cfg = None
    for name, path in arm_ckpts.items():
        m, cfg, st = _load_arm(path, torch, dev)
        models[name] = m
        stats[name] = st
        if name == "full":
            full_cfg = cfg

    tcfg = TrainConfig.from_dict(full_cfg)
    if split_path is not None:
        tcfg.split_path = str(split_path)
    tcfg.device = str(dev)
    tcfg.num_workers = 0
    tr = Trainer(tcfg)
    tr._cond_stats = stats["full"]
    split = tr._load_split()
    ho_specs = _specs_for_shots(split.held_out, max_shots=tcfg.max_heldout_shots)
    frame_cfg = FrameWindowConfig(
        n_frames=tcfg.n_frames, stride=tcfg.stride, seed=tcfg.seed
    )
    logger.info(
        "[cond-ablation] materialising <=%d held-out windows (%d shots)",
        tcfg.eval_windows,
        len(ho_specs),
    )
    batches = tr._materialize_eval(
        ho_specs, frame_cfg, max_windows=tcfg.eval_windows, seed=eval_seed
    )

    out: dict = {
        "task": "conditioning ablation (matched 8000-step budget)",
        "metrics_provenance": "camdyn.metrics.bootstrap_ci (pre-registered)",
        "budget_note": (
            "full arm uses cap_v1_dynamics/step8000.pt (NOT final.pt) so all "
            "three arms compare at an identical 8000-step token budget; the "
            "full arm's final.pt (20000 steps) is the headline verdict, not this."
        ),
        "regimes": {k: list(v) for k, v in regimes.items()},
        "n_heldout_shots": len(ho_specs),
        "n_batches": len(batches),
        "arm_ckpts": {k: str(v) for k, v in arm_ckpts.items()},
        "frontier_frame": int(frontier),
    }

    def _score_all(named):
        sec = {}
        per_arm_acc = {}
        per_arm_nll = {}
        for name, mask in regimes.items():
            nll, acc = _suppressed_score(
                models[name],
                batches,
                torch,
                dev,
                channel_keep_mask(mask),
                stats[name],
                named=named,
            )
            sec[name] = {"masked_nll": _agg(nll), "masked_top1": _agg(acc)}
            per_arm_acc[name] = acc
            per_arm_nll[name] = nll
        # paired diffs vs the no-conditioning (none) arm, oriented positive =
        # conditioning helps (the arm with channels beats camera-only).
        for name in ("full", "ip_ne"):
            a = per_arm_acc[name]
            n0 = per_arm_acc["none"]
            k = min(a.size, n0.size)
            if k:
                sec[f"{name}_vs_none_top1"] = bootstrap_ci(a[:k] - n0[:k])
                sec[f"{name}_vs_none_top1"]["n_pairs"] = int(k)
            an = per_arm_nll[name]
            n0n = per_arm_nll["none"]
            kn = min(an.size, n0n.size)
            if kn:
                sec[f"{name}_vs_none_nll"] = bootstrap_ci(n0n[:kn] - an[:kn])
                sec[f"{name}_vs_none_nll"]["n_pairs"] = int(kn)
        return sec

    out["held_out"] = _score_all(None)
    out["named_geometry"] = {name: _score_all(name) for name in NAMED_GEOMETRIES}

    # one-line story flags
    ho = out["held_out"]
    out["verdict"] = {
        "full_beats_none_nll": bool(
            ho.get("full_vs_none_nll", {}).get("favours_dynamics", False)
        ),
        "ip_ne_beats_none_nll": bool(
            ho.get("ip_ne_vs_none_nll", {}).get("favours_dynamics", False)
        ),
        "frontier_full_beats_none_nll": bool(
            out["named_geometry"]["frontier_half"]
            .get("full_vs_none_nll", {})
            .get("favours_dynamics", False)
        ),
    }
    return out


# ---------------------------------------------------------------------------
# Training entry (one ablation arm, conditioning suppressed)
# ---------------------------------------------------------------------------

REGIME_KEEP = {"full": KEEP_FULL, "ip_ne": KEEP_IP_NE, "none": KEEP_NONE}


def train_arm(
    config_path,
    regime,
    *,
    device=None,
    max_steps=None,
    val_every=None,
    eval_windows=None,
    val_windows=None,
    artifact_out=None,
    ckpt_root=None,
):
    """Train one conditioning-ablation arm with the given keep-regime.

    The config is the matched dynamics architecture (temporal ON); the only
    difference between arms is which conditioning channels survive
    :class:`CondMaskedTrainer`'s suppression.
    """
    if regime not in REGIME_KEEP:
        raise ValueError(f"unknown regime {regime!r}; choose {sorted(REGIME_KEEP)}")
    cfg = TrainConfig.load(config_path)
    if device is not None:
        cfg.device = device
    if max_steps is not None:
        cfg.max_steps = max_steps
    if val_every is not None:
        cfg.val_every = val_every
    if eval_windows is not None:
        cfg.eval_windows = eval_windows
    if val_windows is not None:
        cfg.val_windows = val_windows
    if artifact_out is not None:
        cfg.artifact_out = artifact_out
    if ckpt_root is not None:
        cfg.ckpt_root = ckpt_root
    trainer = CondMaskedTrainer(cfg, REGIME_KEEP[regime])
    return trainer.train()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="camdyn conditioning ablation")
    sub = p.add_subparsers(dest="cmd", required=True)

    pt = sub.add_parser("train", help="train one ablation arm")
    pt.add_argument("--config", required=True, help="dynamics-arch YAML config")
    pt.add_argument(
        "--regime",
        required=True,
        choices=sorted(REGIME_KEEP),
        help="conditioning keep-set (none = camera-only, ip_ne = scalars)",
    )
    pt.add_argument("--device", default=None)
    pt.add_argument("--max-steps", type=int, default=None)
    pt.add_argument("--val-every", type=int, default=None)
    pt.add_argument("--eval-windows", type=int, default=None)
    pt.add_argument("--val-windows", type=int, default=None)
    pt.add_argument("--artifact-out", default=None)
    pt.add_argument("--ckpt-root", default=None)

    pc = sub.add_parser("compare", help="compare three arms at matched budget")
    pc.add_argument("--full", required=True, help="full-conditioning step8000 ckpt")
    pc.add_argument("--ip-ne", required=True, help="ip+ne step8000 ckpt")
    pc.add_argument("--none", required=True, help="camera-only step8000 ckpt")
    pc.add_argument("--out", required=True)
    pc.add_argument("--device", default="cuda")
    pc.add_argument("--split-path", default=None)

    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.cmd == "train":
        w1 = train_arm(
            args.config,
            args.regime,
            device=args.device,
            max_steps=args.max_steps,
            val_every=args.val_every,
            eval_windows=args.eval_windows,
            val_windows=args.val_windows,
            artifact_out=args.artifact_out,
            ckpt_root=args.ckpt_root,
        )
        ho = w1.get("held_out", {})
        logger.info(
            "[cond-ablation] DONE regime=%s held_out nll=%.4f top1=%.4f",
            args.regime,
            ho.get("masked_nll", {}).get("mean", float("nan")),
            ho.get("masked_top1", {}).get("mean", float("nan")),
        )
        return 0

    res = compare_conditioning(
        {"full": args.full, "ip_ne": args.ip_ne, "none": args.none},
        split_path=args.split_path,
        device=args.device,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=2), encoding="utf-8")
    logger.info("[cond-ablation] written to %s | verdict=%s", out_path, res["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
