"""Gas-puff attribution — the falsification probe (token-space, decoder-free).

Does the dynamics arm actually USE the inboard gas-puff command, or did it
just learn camera-only spatial texture?  This probes the causal link in
token space (no decoder needed):

On held-out shots with non-trivial inboard gas-puff activity (the ``aga``
``gas_inboard_*`` channels), mask the **inboard-region token columns** and
have the dynamics arm reconstruct them.  Then two tests:

(a) **Timing correlation** — across each shot's windows, does a
    *predicted-activity* score in the masked inboard region track the
    measured ``gas_inboard_total`` command?  The activity score is
    decoder-free: the fraction of masked inboard cells whose predicted
    (per-bit-MAP) token DIFFERS from the last-visible-frame token at that
    cell — i.e. how much *change* the model predicts in the inboard region.
    A bright in-rushing puff is a region of changing tokens; if the arm
    learned the puff→bright-spot dynamics, predicted change should rise
    with the puff command.

(b) **Counterfactual** — re-run the SAME windows with the ``aga`` channels
    ZEROED (value 0, missing-flag 1 — the "actuator absent" signal) in the
    conditioning vector.  If the prediction is causally driven by the puff
    command, the predicted inboard activity should change when the command
    is removed.  The per-window delta (with-aga minus aga-zeroed) is the
    counterfactual effect size.

Inboard-region token columns (documented choice)
------------------------------------------------
The 16-wide token grid's centre-stack sightline is the centre-column
strip (named geometry ``centre_column_strip`` = cols 6..10).  The inboard
gas-puff bright spot sits to the LEFT of the centre column in image space
(``conditioning.py``: "inboard: the bright-spot cause").  We take the
inboard region as the left-of-centre band cols ``[2, 8)`` (a 6-column
strip, full height) — left of the centre column, excluding the extreme
edge.  Exposed as ``inboard_cols`` so the choice is auditable, not buried.

An honest null (no timing correlation AND no counterfactual effect) is a
reportable result — this probe is designed to be able to FALSIFY puff
attribution, not to manufacture it.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# aga gas-puff channel indices in CONDITIONING_CHANNELS (verified).
GAS_INBOARD_TOTAL_KEY = "gas_inboard_total"
AGA_SOURCE = "aga"

# Inboard-region token columns (left-of-centre band, full height) — see
# module docstring for the convention.
DEFAULT_INBOARD_COLS = (2, 8)


def aga_channel_mask():
    """Boolean ``(C,)`` — True for every ``aga`` gas-puff channel."""
    from imas_ambix.camdyn.conditioning import CONDITIONING_CHANNELS  # noqa: PLC0415

    return np.array([c.source == AGA_SOURCE for c in CONDITIONING_CHANNELS], dtype=bool)


def gas_inboard_total_index():
    """Column index of ``gas_inboard_total`` in the conditioning vector."""
    from imas_ambix.camdyn.conditioning import CONDITIONING_CHANNELS  # noqa: PLC0415

    for i, c in enumerate(CONDITIONING_CHANNELS):
        if c.key == GAS_INBOARD_TOTAL_KEY:
            return i
    raise KeyError(GAS_INBOARD_TOTAL_KEY)


def inboard_visibility_mask(n_frames, inboard_cols, grid=(16, 16)):
    """Visibility mask ``(F,H,W)`` — everything visible EXCEPT inboard cols.

    The inboard column band is masked (must be reconstructed); the rest of
    the grid is visible, so the arm reconstructs the inboard region from the
    surrounding scene + conditioning.
    """
    h, w = grid
    c0, c1 = inboard_cols
    vis = np.ones((n_frames, h, w), dtype=bool)
    vis[:, :, c0:c1] = False
    return vis


def _bit_map_pred(bit_logits):
    bl = np.asarray(bit_logits)
    nbits = bl.shape[-1]
    shifts = np.arange(nbits, dtype=np.int64)
    return ((bl > 0.0).astype(np.int64) << shifts).sum(axis=-1)


def inboard_activity_score(bit_logits, tokens, inboard_cols, valid):
    """Per-frame predicted-change fraction in the masked inboard region.

    For each frame ``f >= 1`` (a previous frame exists), the score is the
    fraction of inboard cells whose predicted token differs from the SAME
    cell's token one frame earlier (the last-observed reference for a
    persistence comparison).  Returns ``(scores (F,), valid_f (F,))``.
    """
    c0, c1 = inboard_cols
    pred = _bit_map_pred(bit_logits)  # (F,H,W)
    tokens = np.asarray(tokens)
    nf = tokens.shape[0]
    scores = np.zeros(nf, dtype=np.float64)
    valid_f = np.zeros(nf, dtype=bool)
    for f in range(1, nf):
        if not (valid[f] and valid[f - 1]):
            continue
        pf = pred[f, :, c0:c1]
        ref = tokens[f - 1, :, c0:c1]
        scores[f] = float((pf != ref).mean())
        valid_f[f] = True
    return scores, valid_f


def _pearson(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _forward_inboard(
    model, arr, torch, dev, inboard_cols, cond_stats, *, zero_aga=False
):
    """Forward one batch with the inboard cols masked → bit-logits (numpy).

    ``zero_aga=True`` zeroes every aga channel (value 0, missing 1) before
    z-scoring — the counterfactual.
    """
    from imas_ambix.camdyn.train import _normalise_conditioning  # noqa: PLC0415

    nf = arr["tokens"].shape[1]
    vis = np.broadcast_to(
        inboard_visibility_mask(nf, inboard_cols)[None], arr["visible"].shape
    ).copy()
    cv = np.array(arr["cond_values"], dtype=np.float32, copy=True)
    cm = np.array(arr["cond_missing"], dtype=np.float32, copy=True)
    if zero_aga:
        amask = aga_channel_mask()
        cv[..., amask] = 0.0
        cm[..., amask] = 1.0
    cvz = _normalise_conditioning(cv, cond_stats)
    t_tokens = torch.from_numpy(np.ascontiguousarray(arr["tokens"])).to(dev)
    t_vis = torch.from_numpy(vis).to(dev)
    t_cv = torch.from_numpy(cvz.astype(np.float32)).to(dev)
    t_cm = torch.from_numpy(cm.astype(np.float32)).to(dev)
    t_dt = torch.from_numpy(arr["dt"].astype(np.float32)).to(dev)
    with (
        torch.no_grad(),
        torch.autocast(
            device_type=dev.type,
            dtype=torch.bfloat16,
            enabled=(dev.type == "cuda"),
        ),
    ):
        logits = model.module(t_tokens, t_vis, t_cv, t_cm, t_dt)
    return logits.float().cpu().numpy()


def _shot_puff_activity(arr, gas_idx, valid_b):
    """Per-frame held puff command for window ``b`` (already on frame grid)."""
    return np.asarray(arr["cond_values"])[:, :, gas_idx]


def select_puff_shots(batches, gas_idx, *, min_std=0.0):
    """Shot ids whose windows carry non-trivial inboard puff variation.

    A shot qualifies if its pooled ``gas_inboard_total`` command (across its
    windows, present frames only) has std > ``min_std`` and is not all-missing.
    Returns the set of qualifying shot ids.
    """
    by_shot: dict = {}
    for arr in batches:
        gas = np.asarray(arr["cond_values"])[:, :, gas_idx]  # (B,F)
        miss = np.asarray(arr["cond_missing"])[:, :, gas_idx]  # (B,F)
        for b in range(gas.shape[0]):
            sid = int(arr["shot_id"][b]) if "shot_id" in arr else b
            present = miss[b] < 0.5
            if present.any():
                by_shot.setdefault(sid, []).append(gas[b][present])
    qualifying = set()
    for sid, chunks in by_shot.items():
        vals = np.concatenate(chunks)
        if vals.size >= 3 and float(np.std(vals)) > min_std:
            qualifying.add(sid)
    return qualifying


def probe_puff_attribution(
    dynamics_ckpt,
    *,
    split_path=None,
    device="cuda",
    inboard_cols=DEFAULT_INBOARD_COLS,
    eval_seed=999,
):
    """Run the gas-puff attribution probe on the held-out suite.

    Returns the artifact dict: per-shot timing correlations (predicted
    inboard activity vs the puff command), pooled correlation, the
    counterfactual deltas (with-aga minus aga-zeroed activity), and the
    honest verdict (positive attribution / null).
    """
    import torch  # noqa: PLC0415

    from imas_ambix.camdyn.arm_compare import _load_arm  # noqa: PLC0415
    from imas_ambix.camdyn.dataset import FrameWindowConfig  # noqa: PLC0415
    from imas_ambix.camdyn.metrics import bootstrap_ci  # noqa: PLC0415
    from imas_ambix.camdyn.train import (  # noqa: PLC0415
        TrainConfig,
        Trainer,
        _specs_for_shots,
    )

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    dyn_model, full_cfg, dyn_stats = _load_arm(dynamics_ckpt, torch, dev)

    tcfg = TrainConfig.from_dict(full_cfg)
    if split_path is not None:
        tcfg.split_path = str(split_path)
    tcfg.device = str(dev)
    tcfg.num_workers = 0
    tr = Trainer(tcfg)
    tr._cond_stats = dyn_stats
    split = tr._load_split()
    ho_specs = _specs_for_shots(split.held_out, max_shots=tcfg.max_heldout_shots)
    frame_cfg = FrameWindowConfig(
        n_frames=tcfg.n_frames, stride=tcfg.stride, seed=tcfg.seed
    )
    logger.info(
        "[puff] materialising <=%d held-out windows (%d shots)",
        tcfg.eval_windows,
        len(ho_specs),
    )
    batches = tr._materialize_eval(
        ho_specs, frame_cfg, max_windows=tcfg.eval_windows, seed=eval_seed
    )

    gas_idx = gas_inboard_total_index()
    puff_shots = select_puff_shots(batches, gas_idx)

    # accumulate per-(shot,window,frame) paired samples
    pooled_cmd, pooled_act = [], []
    cf_deltas = []  # per-frame (with_aga - zeroed_aga) activity
    per_shot: dict = {}
    n_windows_used = 0

    for arr in batches:
        bl = _forward_inboard(dyn_model, arr, torch, dev, inboard_cols, dyn_stats)
        bl_cf = _forward_inboard(
            dyn_model, arr, torch, dev, inboard_cols, dyn_stats, zero_aga=True
        )
        gas = np.asarray(arr["cond_values"])[:, :, gas_idx]  # (B,F)
        miss = np.asarray(arr["cond_missing"])[:, :, gas_idx]
        for b in range(arr["tokens"].shape[0]):
            sid = int(arr["shot_id"][b]) if "shot_id" in arr else b
            if sid not in puff_shots:
                continue
            valid_b = np.asarray(arr["valid"][b], dtype=bool)
            act, vf = inboard_activity_score(
                bl[b], arr["tokens"][b], inboard_cols, valid_b
            )
            act_cf, vf_cf = inboard_activity_score(
                bl_cf[b], arr["tokens"][b], inboard_cols, valid_b
            )
            present = (miss[b] < 0.5) & vf
            if present.sum() < 3:
                continue
            n_windows_used += 1
            cmd_p = gas[b][present]
            act_p = act[present]
            pooled_cmd.append(cmd_p)
            pooled_act.append(act_p)
            cf_deltas.append(act[present] - act_cf[present])
            ps = per_shot.setdefault(sid, {"cmd": [], "act": [], "cf": []})
            ps["cmd"].append(cmd_p)
            ps["act"].append(act_p)
            ps["cf"].append(act[present] - act_cf[present])

    # per-shot timing correlations
    shot_corrs = {}
    for sid, d in per_shot.items():
        cmd = np.concatenate(d["cmd"])
        act = np.concatenate(d["act"])
        shot_corrs[str(sid)] = {
            "n_frames": int(cmd.size),
            "pearson_cmd_vs_activity": _pearson(cmd, act),
            "mean_counterfactual_delta": float(np.concatenate(d["cf"]).mean()),
        }

    out: dict = {
        "task": "gas-puff attribution probe (token-space, decoder-free)",
        "dynamics_ckpt": str(dynamics_ckpt),
        "inboard_cols": list(inboard_cols),
        "inboard_cols_note": (
            "left-of-centre token column band [2,8); centre-stack strip is "
            "cols 6..10 (named geometry centre_column_strip), inboard puff "
            "spot sits left of centre per conditioning.py."
        ),
        "activity_score": (
            "fraction of masked inboard cells whose predicted (bit-MAP) token "
            "differs from the same cell one frame earlier (predicted change)."
        ),
        "n_heldout_shots": len(ho_specs),
        "n_puff_shots": len(puff_shots),
        "n_windows_used": int(n_windows_used),
    }

    if not pooled_cmd:
        out["verdict"] = {
            "attribution": "untested",
            "note": "no held-out windows with non-trivial inboard puff variation",
        }
        return out

    cmd_all = np.concatenate(pooled_cmd)
    act_all = np.concatenate(pooled_act)
    cf_all = np.concatenate(cf_deltas)

    pooled_corr = _pearson(cmd_all, act_all)
    cf_ci = bootstrap_ci(cf_all)  # positive = aga raises predicted activity
    # per-shot correlation summary
    finite_corrs = [
        v["pearson_cmd_vs_activity"]
        for v in shot_corrs.values()
        if np.isfinite(v["pearson_cmd_vs_activity"])
    ]
    out["pooled"] = {
        "n_frames": int(cmd_all.size),
        "pearson_cmd_vs_activity": pooled_corr,
        "counterfactual_delta_mean": float(cf_all.mean()),
        "counterfactual_delta_ci": [cf_ci["lo"], cf_ci["hi"]],
        "counterfactual_clear_of_zero": bool(cf_ci["clear_of_zero"]),
    }
    out["per_shot_correlation"] = {
        "n_shots": len(finite_corrs),
        "median_pearson": (
            float(np.median(finite_corrs)) if finite_corrs else float("nan")
        ),
        "mean_pearson": (
            float(np.mean(finite_corrs)) if finite_corrs else float("nan")
        ),
        "frac_positive": (
            float(np.mean(np.asarray(finite_corrs) > 0))
            if finite_corrs
            else float("nan")
        ),
    }
    out["per_shot"] = shot_corrs

    timing_positive = np.isfinite(pooled_corr) and pooled_corr > 0.1
    cf_effect = bool(cf_ci["clear_of_zero"])
    out["verdict"] = {
        "timing_correlation_positive": bool(timing_positive),
        "counterfactual_effect_detected": cf_effect,
        "attribution": (
            "positive"
            if (timing_positive and cf_effect)
            else ("partial" if (timing_positive or cf_effect) else "null")
        ),
        "framing": (
            "honest null reported — the probe can falsify attribution; a "
            "non-result is a result, not a failure."
        ),
    }
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="camdyn gas-puff attribution probe")
    p.add_argument("--dynamics", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--split-path", default=None)
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    res = probe_puff_attribution(
        args.dynamics, split_path=args.split_path, device=args.device
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=2), encoding="utf-8")
    logger.info("[puff] written to %s | verdict=%s", out_path, res.get("verdict"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
