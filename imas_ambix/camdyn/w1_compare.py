"""W1 verdict — the cross-arm PAIRED comparison (dynamics vs baseline).

Each arm's training run writes its own artifact with only AGGREGATE held-out
NLL/top-1 (mean/std/n), which is not enough for the pre-registered W1 test:
W1 is a *paired* bootstrap of ``baseline - dynamics`` over the SAME masked
tokens.  This module loads BOTH ``final.pt`` checkpoints, scores the identical
frozen held-out windows + named-geometry suite with each model, and runs
:func:`imas_ambix.camdyn.metrics.bootstrap_ci` on the per-token paired diff.

Pairing is exact: both arms score the SAME materialised windows in the SAME
order, and (held-out) the SAME mixture masks / (geometry) the SAME deterministic
mask — so ``base_nll[i]`` and ``dyn_nll[i]`` are the same token, and their diff
is a valid paired sample.

W1 wins iff ``favours_dynamics`` (the ``(1-alpha)`` bootstrap CI lower bound on
the paired diff is > 0): the dynamics arm is significantly better.  For NLL the
diff is ``baseline_nll - dynamics_nll`` (positive = dynamics better); for
accuracy it is ``dynamics_acc - baseline_acc``.

Usage::

    python -m imas_ambix.camdyn.w1_compare \\
        --baseline /work/.../camdyn/cap_v1_baseline/final.pt \\
        --dynamics /work/.../camdyn/cap_v1_dynamics/final.pt \\
        --out imas_ambix/camdyn/artifacts/w1_verdict_v1.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from imas_ambix.camdyn.dataset import FrameWindowConfig
from imas_ambix.camdyn.masking import NAMED_GEOMETRIES
from imas_ambix.camdyn.metrics import bootstrap_ci
from imas_ambix.camdyn.model import CamdynConfig, CamdynModel
from imas_ambix.camdyn.train import TrainConfig, Trainer, _agg, _specs_for_shots

logger = logging.getLogger(__name__)


def _load_arm(ckpt_path, torch, device):
    """Load a trained arm's model + config + conditioning stats from a ckpt."""
    ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    model_cfg = CamdynConfig.from_dict(ck["config"]["model"])
    model = CamdynModel.from_config(model_cfg)
    model.module.load_state_dict(ck["model_state"])
    model.module.to(device).eval()
    cond_stats = (
        np.asarray(ck["cond_stats"][0], dtype=np.float32),
        np.asarray(ck["cond_stats"][1], dtype=np.float32),
    )
    return model, ck["config"], cond_stats


def _paired(base, dyn, *, orient):
    """Bootstrap CI on a paired diff oriented so positive favours dynamics.

    ``orient="nll"`` → diff = base - dyn (lower NLL is better for dynamics).
    ``orient="acc"`` → diff = dyn - base (higher accuracy is better).
    """
    base = np.asarray(base, dtype=np.float64).reshape(-1)
    dyn = np.asarray(dyn, dtype=np.float64).reshape(-1)
    n = min(base.size, dyn.size)
    if n == 0:
        return {"n_pairs": 0, "favours_dynamics": False, "clear_of_zero": False}
    diff = (base[:n] - dyn[:n]) if orient == "nll" else (dyn[:n] - base[:n])
    ci = bootstrap_ci(diff)
    ci["n_pairs"] = int(n)
    ci["base_mean"] = float(base[:n].mean())
    ci["dyn_mean"] = float(dyn[:n].mean())
    return ci


def compare_w1(
    baseline_ckpt,
    dynamics_ckpt,
    *,
    split_path=None,
    device="cuda",
    eval_seed=999,
):
    """Score both arms on identical held-out windows → the paired W1 verdict.

    Returns the verdict dict (held-out paired NLL/acc CI + favours_dynamics,
    per-named-geometry paired CIs, motion-weighted paired CI, and each arm's
    aggregate means).
    """
    # The baseline ckpt's full TrainConfig drives the eval substrate (window
    # cfg, eval_windows, held-out shot cap, batch size, num_workers).  Both
    # arms are matched on these, so either is fine; conditioning stats are
    # shared (same seed) — use the baseline's.
    import torch  # noqa: PLC0415

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    base_model, base_full_cfg, base_stats = _load_arm(baseline_ckpt, torch, dev)
    dyn_model, _dyn_full_cfg, _dyn_stats = _load_arm(dynamics_ckpt, torch, dev)

    tcfg = TrainConfig.from_dict(base_full_cfg)
    if split_path is not None:
        tcfg.split_path = str(split_path)
    tcfg.device = str(dev)
    tr = Trainer(tcfg)
    tr._cond_stats = base_stats  # shared normalisation (same seed across arms)

    split = tr._load_split()
    ho_specs = _specs_for_shots(split.held_out, max_shots=tcfg.max_heldout_shots)
    frame_cfg = FrameWindowConfig(
        n_frames=tcfg.n_frames, stride=tcfg.stride, seed=tcfg.seed
    )

    logger.info(
        "[w1-compare] materialising <=%d held-out windows (%d shots)",
        tcfg.eval_windows,
        len(ho_specs),
    )
    batches = tr._materialize_eval(
        ho_specs, frame_cfg, max_windows=tcfg.eval_windows, seed=eval_seed
    )

    out: dict = {
        "comparison": "W1 paired bootstrap (baseline vs dynamics)",
        "metrics_provenance": "camdyn.metrics.bootstrap_ci (pre-registered D0)",
        "n_heldout_shots": len(ho_specs),
        "n_batches": len(batches),
        "baseline_ckpt": str(baseline_ckpt),
        "dynamics_ckpt": str(dynamics_ckpt),
        "baseline_params": int(base_model.num_parameters()),
        "dynamics_params": int(dyn_model.num_parameters()),
    }

    # --- held-out (mixture mask): the headline W1 task ---
    b_nll, b_acc, b_mnll, b_macc = tr._score_cached(base_model, batches, torch, dev)
    d_nll, d_acc, d_mnll, d_macc = tr._score_cached(dyn_model, batches, torch, dev)
    out["held_out"] = {
        "baseline": {"masked_nll": _agg(b_nll), "masked_top1": _agg(b_acc)},
        "dynamics": {"masked_nll": _agg(d_nll), "masked_top1": _agg(d_acc)},
        "paired_nll_ci": _paired(b_nll, d_nll, orient="nll"),
        "paired_top1_ci": _paired(b_acc, d_acc, orient="acc"),
        "motion_weighted": {
            "paired_nll_ci": _paired(b_mnll, d_mnll, orient="nll"),
            "paired_top1_ci": _paired(b_macc, d_macc, orient="acc"),
        },
    }
    # W1 verdict: dynamics beats statics on the held-out masked-token NLL with
    # the paired bootstrap CI clear of zero in the dynamics-favouring direction.
    out["W1_verdict"] = {
        "favours_dynamics_nll": bool(
            out["held_out"]["paired_nll_ci"]["favours_dynamics"]
        ),
        "favours_dynamics_top1": bool(
            out["held_out"]["paired_top1_ci"]["favours_dynamics"]
        ),
        "nll_ci": [
            out["held_out"]["paired_nll_ci"]["lo"],
            out["held_out"]["paired_nll_ci"]["hi"],
        ],
    }

    # --- per named-geometry (frozen eval suite) ---
    geo = {}
    for name in NAMED_GEOMETRIES:
        gb_nll, gb_acc, _, _ = tr._score_cached(
            base_model, batches, torch, dev, named=name
        )
        gd_nll, gd_acc, _, _ = tr._score_cached(
            dyn_model, batches, torch, dev, named=name
        )
        geo[name] = {
            "baseline": {"masked_nll": _agg(gb_nll), "masked_top1": _agg(gb_acc)},
            "dynamics": {"masked_nll": _agg(gd_nll), "masked_top1": _agg(gd_acc)},
            "paired_nll_ci": _paired(gb_nll, gd_nll, orient="nll"),
            "paired_top1_ci": _paired(gb_acc, gd_acc, orient="acc"),
        }
    out["named_geometry"] = geo
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="W1 paired comparison (dynamics vs baseline)"
    )
    p.add_argument("--baseline", required=True, help="baseline arm final.pt")
    p.add_argument("--dynamics", required=True, help="dynamics arm final.pt")
    p.add_argument("--out", required=True, help="output verdict JSON path")
    p.add_argument("--device", default="cuda")
    p.add_argument("--split-path", default=None)
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    verdict = compare_w1(
        args.baseline, args.dynamics, split_path=args.split_path, device=args.device
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    v = verdict["W1_verdict"]
    logger.info(
        "[w1-compare] W1 favours_dynamics(nll)=%s top1=%s "
        "| held-out nll base=%.4f dyn=%.4f | -> %s",
        v["favours_dynamics_nll"],
        v["favours_dynamics_top1"],
        verdict["held_out"]["baseline"]["masked_nll"]["mean"],
        verdict["held_out"]["dynamics"]["masked_nll"]["mean"],
        out_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
