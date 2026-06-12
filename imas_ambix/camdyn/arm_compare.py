"""Paired comparison of the trained reconstruction arms (the verdict step).

Each arm's training run writes only AGGREGATE held-out NLL/top-1 (mean/std/n),
which is not enough for the headline test: whether the temporal-**dynamics**
arm beats the per-frame **baseline** arm is a *paired* bootstrap over the SAME
masked tokens.  This module loads BOTH trained checkpoints, scores the identical
frozen held-out windows + named-geometry suite with each model, and runs
:func:`imas_ambix.camdyn.metrics.bootstrap_ci` on the per-token paired diff.

It also scores a **zero-order-hold (ZOH)** reference: for each masked cell,
carry forward the most recent OBSERVED token at that grid location.  ZOH is the
trivial "the scene does not change" predictor (for the temporal-frontier mode
it reduces to persistence of the last visible frame).  The dynamics arm must
beat BOTH the per-frame baseline AND this carry-forward reference — otherwise it
is only exploiting temporal redundancy a trivial predictor captures for free.

Pairing is exact: every arm (and ZOH) scores the SAME materialised windows in
the SAME order, with the SAME masks, so ``base[i]`` / ``dyn[i]`` / ``zoh[i]``
are the same masked token and their diffs are valid paired samples.

The verdict is ``dynamics_wins`` = the dynamics arm's held-out masked-token NLL
is significantly below the baseline's (paired bootstrap CI clear of zero) AND
its top-1 accuracy is significantly above the carry-forward reference's.

Usage::

    python -m imas_ambix.camdyn.arm_compare \\
        --baseline /work/.../camdyn/cap_v1_baseline/final.pt \\
        --dynamics /work/.../camdyn/cap_v1_dynamics/final.pt \\
        --out imas_ambix/camdyn/artifacts/arm_compare_v1.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from imas_ambix.camdyn.dataset import FrameWindowConfig
from imas_ambix.camdyn.masking import NAMED_GEOMETRIES, named_geometry_mask
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


def _paired(base, other, *, orient):
    """Bootstrap CI on a paired diff oriented so positive favours dynamics.

    ``orient="nll"`` → diff = base - other (lower NLL is better for dynamics).
    ``orient="acc"`` → diff = other - base (higher accuracy is better).
    Here ``base`` is the reference arm (baseline or carry-forward) and
    ``other`` is the dynamics arm.
    """
    base = np.asarray(base, dtype=np.float64).reshape(-1)
    other = np.asarray(other, dtype=np.float64).reshape(-1)
    n = min(base.size, other.size)
    if n == 0:
        return {"n_pairs": 0, "favours_dynamics": False, "clear_of_zero": False}
    diff = (base[:n] - other[:n]) if orient == "nll" else (other[:n] - base[:n])
    ci = bootstrap_ci(diff)
    ci["n_pairs"] = int(n)
    ci["ref_mean"] = float(base[:n].mean())
    ci["dynamics_mean"] = float(other[:n].mean())
    return ci


def _carry_forward_pred(tokens: np.ndarray, visible: np.ndarray) -> np.ndarray:
    """Zero-order hold: per cell, carry forward the last OBSERVED token.

    ``tokens`` (F,H,W) int, ``visible`` (F,H,W) bool (True = the model saw it).
    The prediction at frame ``f`` uses only frames ``< f`` (causal), so a
    masked cell is predicted by the most recent time its grid location was
    visible; cells never observed before ``f`` get ``-1`` (never matches a real
    token id → scored as a miss).  For the temporal-frontier mask this reduces
    to persistence of the last visible frame.
    """
    tokens = np.asarray(tokens)
    visible = np.asarray(visible, dtype=bool)
    n_frames = tokens.shape[0]
    pred = np.full_like(tokens, -1)
    last = np.full(tokens.shape[1:], -1, dtype=tokens.dtype)
    seen = np.zeros(tokens.shape[1:], dtype=bool)
    for f in range(n_frames):
        pred[f] = np.where(seen, last, -1)  # only frames < f
        vis_f = visible[f]
        last = np.where(vis_f, tokens[f], last)
        seen = seen | vis_f
    return pred


def _masks_for(arr, named):
    """Return ``(visible, loss_mask)`` for a batch: mixture or frozen geometry."""
    if named is None:
        return arr["visible"], arr["loss_mask"]
    nf = arr["tokens"].shape[1]
    gmask = named_geometry_mask(named, nf)  # (F,H,W) True = visible
    vis = np.broadcast_to(gmask[None], arr["visible"].shape).copy()
    return vis, ~vis


def _score_carry_forward(batches, *, named=None):
    """Per-token carry-forward (ZOH) top-1 accuracy over the masked cells.

    Mirrors the model scoring loop EXACTLY (same batch order, same masked-cell
    boolean-index order, same empty-mask skip) so the returned per-token array
    is element-wise paired with the models' ``acc_per_token`` arrays.
    """
    acc_all = []
    for arr in batches:
        visible_np, loss_mask_np = _masks_for(arr, named)
        tokens = arr["tokens"]
        for b in range(tokens.shape[0]):
            vf = arr["valid"][b]
            lm_b = loss_mask_np[b] & vf[:, None, None]
            if not lm_b.any():
                continue
            pred = _carry_forward_pred(tokens[b], visible_np[b])
            acc_all.append((pred[lm_b] == tokens[b][lm_b]).astype(np.float64))
    return np.concatenate(acc_all) if acc_all else np.array([])


def compare_arms(
    baseline_ckpt,
    dynamics_ckpt,
    *,
    split_path=None,
    device="cuda",
    eval_seed=999,
):
    """Score both arms + the carry-forward reference on identical held-out
    windows → the paired verdict (dynamics vs baseline, dynamics vs ZOH).
    """
    import torch  # noqa: PLC0415

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    base_model, base_full_cfg, base_stats = _load_arm(baseline_ckpt, torch, dev)
    dyn_model, _dyn_full_cfg, _dyn_stats = _load_arm(dynamics_ckpt, torch, dev)

    # The baseline ckpt's full config drives the eval substrate (window cfg,
    # held-out window/shot caps, batch, workers); both arms are matched on
    # these, and conditioning stats are shared (same seed) — use the baseline's.
    tcfg = TrainConfig.from_dict(base_full_cfg)
    if split_path is not None:
        tcfg.split_path = str(split_path)
    tcfg.device = str(dev)
    # Single-process materialisation: the training config's worker count is
    # tuned for a multi-hour run, but here it spawns N processes each holding
    # a torch runtime + a per-shot LRU of full raw conditioning traces, on top
    # of TWO loaded models — OOM-killing the compare (job 1216726 died here at
    # 120G).  Reading the few hundred eval windows once in the main process is
    # minutes of I/O and keeps a single trace cache.
    tcfg.num_workers = 0
    tr = Trainer(tcfg)
    tr._cond_stats = base_stats

    split = tr._load_split()
    ho_specs = _specs_for_shots(split.held_out, max_shots=tcfg.max_heldout_shots)
    frame_cfg = FrameWindowConfig(
        n_frames=tcfg.n_frames, stride=tcfg.stride, seed=tcfg.seed
    )

    logger.info(
        "[arm-compare] materialising <=%d held-out windows (%d shots)",
        tcfg.eval_windows,
        len(ho_specs),
    )
    batches = tr._materialize_eval(
        ho_specs, frame_cfg, max_windows=tcfg.eval_windows, seed=eval_seed
    )

    out: dict = {
        "comparison": "paired bootstrap: dynamics vs baseline + carry-forward(ZOH)",
        "metrics_provenance": "camdyn.metrics.bootstrap_ci (pre-registered)",
        "n_heldout_shots": len(ho_specs),
        "n_batches": len(batches),
        "baseline_ckpt": str(baseline_ckpt),
        "dynamics_ckpt": str(dynamics_ckpt),
        "baseline_params": int(base_model.num_parameters()),
        "dynamics_params": int(dyn_model.num_parameters()),
    }

    def _section(named):
        b_nll, b_acc, b_mnll, b_macc = tr._score_cached(
            base_model, batches, torch, dev, named=named
        )
        d_nll, d_acc, d_mnll, d_macc = tr._score_cached(
            dyn_model, batches, torch, dev, named=named
        )
        z_acc = _score_carry_forward(batches, named=named)
        sec = {
            "baseline": {"masked_nll": _agg(b_nll), "masked_top1": _agg(b_acc)},
            "dynamics": {"masked_nll": _agg(d_nll), "masked_top1": _agg(d_acc)},
            "carry_forward": {"masked_top1": _agg(z_acc)},
            "dynamics_vs_baseline_nll": _paired(b_nll, d_nll, orient="nll"),
            "dynamics_vs_baseline_top1": _paired(b_acc, d_acc, orient="acc"),
            "dynamics_vs_carry_forward_top1": _paired(z_acc, d_acc, orient="acc"),
        }
        if named is None:
            sec["motion_weighted"] = {
                "dynamics_vs_baseline_nll": _paired(b_mnll, d_mnll, orient="nll"),
                "dynamics_vs_baseline_top1": _paired(b_macc, d_macc, orient="acc"),
            }
        return sec

    # --- held-out (mixture mask): the headline reconstruction task ---
    ho = _section(None)
    out["held_out"] = ho

    beats_baseline = bool(ho["dynamics_vs_baseline_nll"]["favours_dynamics"])
    beats_carry_forward = bool(ho["dynamics_vs_carry_forward_top1"]["favours_dynamics"])
    out["verdict"] = {
        "dynamics_beats_baseline_nll": beats_baseline,
        "dynamics_beats_baseline_top1": bool(
            ho["dynamics_vs_baseline_top1"]["favours_dynamics"]
        ),
        "dynamics_beats_carry_forward_top1": beats_carry_forward,
        # the dynamics arm must beat BOTH the matched per-frame baseline and the
        # trivial carry-forward predictor to count as a genuine win.
        "dynamics_wins": beats_baseline and beats_carry_forward,
        "baseline_nll_ci": [
            ho["dynamics_vs_baseline_nll"]["lo"],
            ho["dynamics_vs_baseline_nll"]["hi"],
        ],
        "carry_forward_top1_ci": [
            ho["dynamics_vs_carry_forward_top1"]["lo"],
            ho["dynamics_vs_carry_forward_top1"]["hi"],
        ],
    }

    # --- per named-geometry (frozen eval suite) ---
    out["named_geometry"] = {name: _section(name) for name in NAMED_GEOMETRIES}
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Paired comparison: dynamics vs baseline + carry-forward"
    )
    p.add_argument("--baseline", required=True, help="baseline arm checkpoint")
    p.add_argument("--dynamics", required=True, help="dynamics arm checkpoint")
    p.add_argument("--out", required=True, help="output verdict JSON path")
    p.add_argument("--device", default="cuda")
    p.add_argument("--split-path", default=None)
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    verdict = compare_arms(
        args.baseline, args.dynamics, split_path=args.split_path, device=args.device
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    v = verdict["verdict"]
    ho = verdict["held_out"]
    logger.info(
        "[arm-compare] dynamics_wins=%s (beats_baseline_nll=%s beats_carry_forward=%s) "
        "| held-out nll base=%.4f dyn=%.4f | top1 dyn=%.4f zoh=%.4f | -> %s",
        v["dynamics_wins"],
        v["dynamics_beats_baseline_nll"],
        v["dynamics_beats_carry_forward_top1"],
        ho["baseline"]["masked_nll"]["mean"],
        ho["dynamics"]["masked_nll"]["mean"],
        ho["dynamics"]["masked_top1"]["mean"],
        ho["carry_forward"]["masked_top1"]["mean"],
        out_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
