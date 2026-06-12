"""Forward-horizon forecasting table — the W2 win condition.

Conditioned on the visible camera stream up to a temporal frontier ``t``
(the FRONTIER mask: frames ``< t`` visible, frames ``>= t`` masked), this
scores full-frame reconstruction at the locked PHYSICAL horizons
``h = 10, 50, 200 ms`` for THREE predictors, PAIRED on the same windows:

1. the **dynamics** arm (temporal attention ON) — given the pre-frontier
   stream + conditioning, it predicts the masked future frames;
2. **persistence** — copy the last observed frame to every horizon (the
   pre-registered :func:`persistence_baseline_accuracy`);
3. the **per-frame baseline** arm (temporal attention OFF) — given the
   identical frontier mask; it has no cross-frame path, so its
   post-frontier prediction is conditioning + prior only.  That is the
   point: it shows what a static inpainter recovers from the actuator
   vector alone.

Physical, not index-based
-------------------------
The horizon→frame-offset map is the pre-registered
:func:`horizon_frame_offsets` (median Δt of the window), so a horizon is
the same physical lead-time regardless of cadence.  The rbb cadence is
wildly heterogeneous (verified: 13 µs … 1 ms per frame across held-out
shots), so at NATIVE cadence a 16-frame window physically spans
~0.2 … 15 ms and most horizons fall OUTSIDE the window — the
pre-registered functions return ``valid=0`` for those, and we aggregate
only valid horizons and report the per-horizon valid counts honestly.

Two cadence regimes are scored and both reported:

``native``
    Contiguous native frames exactly as the trained arms saw them.  The
    honest answer to "how far ahead can the in-window future reach?" — for
    most shots only the shortest horizon (or none) is reachable.

``matched``
    Per-shot frame decimation: every window samples 1-of-``k`` frames with
    ``k`` chosen so the 16-frame window spans the full horizon range.  The
    model is Δt-conditioned (the conditioning vector carries the per-frame
    Δt), so feeding the decimated cadence is the intended use of the Δt
    hook — it lets a single model absorb cadence heterogeneity (package
    docstring, D0 time-grid recommendation).  This regime makes the
    physical horizons reachable so the forecasting table is populated.

Scoring uses the same bit-head adapter as the rest of camdyn
(:func:`score_window_bits` machinery): exact full-vocab bitwise NLL +
per-bit-MAP top-1.  Paired CIs (dynamics vs persistence, dynamics vs
baseline) come from the pre-registered :func:`bootstrap_ci` on per-frame
paired diffs at each horizon.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from imas_ambix.camdyn.arm_compare import _load_arm
from imas_ambix.camdyn.dataset import FrameWindowConfig
from imas_ambix.camdyn.metrics import (
    HORIZON_MS,
    bootstrap_ci,
    horizon_frame_offsets,
)
from imas_ambix.camdyn.model import bitwise_nll
from imas_ambix.camdyn.train import TrainConfig, Trainer, _specs_for_shots

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Frame decimation (cadence-matched regime)
# ---------------------------------------------------------------------------


def decimate_window(arr: dict, stride: int) -> dict:
    """Sub-sample every ``stride``-th frame of a materialised batch dict.

    The trained window is contiguous native frames; decimating spreads the
    16-frame window across a longer physical span so the physical horizons
    become reachable.  ``dt`` is recomputed as the forward difference of the
    decimated ``frame_time`` so the model's Δt conditioning correctly
    reflects the wider spacing (the model is Δt-conditioned — package
    docstring).  Returns a NEW dict (the native batch is untouched so the
    same materialised windows can be scored in both regimes).
    """
    if stride <= 1:
        return arr
    out: dict = {}
    for k, v in arr.items():
        v = np.asarray(v)
        if v.ndim >= 2 and v.shape[1] == arr["tokens"].shape[1]:
            out[k] = v[:, ::stride].copy()
        else:
            out[k] = v
    # recompute per-frame forward dt on the decimated time base
    ft = out["frame_time"]
    dt = np.stack([_forward_dt_1d(ft[b]) for b in range(ft.shape[0])])
    out["dt"] = dt.astype(np.float32)
    return out


def _forward_dt_1d(frame_time: np.ndarray) -> np.ndarray:
    ft = np.asarray(frame_time, dtype=np.float64)
    if ft.size < 2:
        return np.zeros_like(ft)
    d = np.diff(ft)
    return np.concatenate([d, d[-1:]])


def matched_stride_for(frame_time: np.ndarray, n_frames: int, max_horizon_ms: float):
    """Decimation stride so a ``n_frames``-window spans ``max_horizon_ms``.

    Returns ``(stride, ok)`` — ``ok`` False when the native window already
    has too few frames to reach the horizon even fully decimated (then the
    horizon stays out of window and is reported invalid, not faked).
    """
    ft = np.asarray(frame_time, dtype=np.float64).reshape(-1)
    if ft.size < 2:
        return 1, False
    dt = float(np.median(np.diff(ft)))
    if not np.isfinite(dt) or dt <= 0:
        return 1, False
    # frames needed (native) to span the horizon, then spread over n_frames
    frames_needed = (max_horizon_ms / 1000.0) / dt
    stride = max(1, int(np.ceil(frames_needed / max(1, n_frames - 1))))
    # the decimated window only covers (n_frames-1)*stride native frames;
    # ok if that is enough to reach the horizon
    reach_ms = (n_frames - 1) * stride * dt * 1000.0
    return stride, bool(reach_ms >= max_horizon_ms)


def decimate_to_n(arr: dict, n_target: int, max_horizon_ms: float) -> dict:
    """Down-sample a WIDE native window to ``n_target`` frames spanning a horizon.

    The matched regime needs the ``n_target``-frame window the model expects
    (e.g. 16) to physically span the longest horizon.  A *contiguous* native
    16-frame window only spans ~15 ms at MAST cadence, so instead we read a
    WIDE native window (many more native frames) and pick ``n_target`` of them
    with a per-shot stride so the kept frames span ``max_horizon_ms``.  ``dt``
    is recomputed on the kept frames so the model's Δt conditioning reflects
    the wider spacing (the model is Δt-conditioned — package docstring).

    Returns a NEW batch dict with exactly ``min(n_target, available)`` frames
    on the frame axis.  When the wide window's own span is shorter than the
    horizon (very high cadence, or padded short shots) the kept window simply
    spans as far as the available real frames reach — :func:`score_window_horizons`
    then reports the unreachable horizons as ``valid=0``.
    """
    ft0 = np.asarray(arr["frame_time"][0], dtype=np.float64).reshape(-1)
    nwide = ft0.size
    if nwide <= n_target:
        return arr
    dt = float(np.median(np.diff(ft0)))
    if not np.isfinite(dt) or dt <= 0:
        idx = np.linspace(0, nwide - 1, n_target).round().astype(int)
    else:
        # stride so n_target frames span the horizon; clip so the picked
        # indices stay inside the wide window (don't run past real frames)
        frames_needed = (max_horizon_ms / 1000.0) / dt
        stride = max(1, int(np.ceil(frames_needed / max(1, n_target - 1))))
        stride = min(stride, max(1, (nwide - 1) // max(1, n_target - 1)))
        idx = np.arange(n_target, dtype=int) * stride
        idx = idx[idx < nwide]
    out: dict = {}
    tok_axis_n = arr["tokens"].shape[1]
    for k, v in arr.items():
        v = np.asarray(v)
        if v.ndim >= 2 and v.shape[1] == tok_axis_n:
            out[k] = v[:, idx].copy()
        else:
            out[k] = v
    ft = out["frame_time"]
    dt_arr = np.stack([_forward_dt_1d(ft[b]) for b in range(ft.shape[0])])
    out["dt"] = dt_arr.astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# Per-window horizon scoring (bit-head, matches the rest of camdyn)
# ---------------------------------------------------------------------------


def _bit_map_pred(bit_logits: np.ndarray) -> np.ndarray:
    """Per-bit MAP token id (same rule as model.score_window_bits)."""
    bl = np.asarray(bit_logits)
    nbits = bl.shape[-1]
    shifts = np.arange(nbits, dtype=np.int64)
    return ((bl > 0.0).astype(np.int64) << shifts).sum(axis=-1)


def score_window_horizons(
    bit_logits: np.ndarray,
    tokens: np.ndarray,
    frame_time: np.ndarray,
    valid: np.ndarray,
    frontier_frame: int,
    *,
    horizons_ms=HORIZON_MS,
):
    """Score one window's three predictors at each physical horizon.

    ``bit_logits`` ``(F,H,W,bits)`` are the dynamics arm's per-frame
    reconstruction.  The horizon→offset map is the pre-registered
    :func:`horizon_frame_offsets` (verbatim).  For each horizon the target
    frame is ``frontier + offset``; a horizon whose target frame falls
    outside the window (or is padded/invalid) is marked ``valid=0`` and
    contributes nothing (the caller aggregates only valid horizons).

    Returns ``{h_ms: {valid, target_frame, dyn_top1[], dyn_nll[],
    persist_top1[], n_cells}}`` where the ``*_top1`` arrays are per-cell
    correctness (paired element-wise across predictors for the same frame).
    """
    tokens = np.asarray(tokens)
    n_frames = tokens.shape[0]
    offsets = horizon_frame_offsets(frame_time, horizons_ms=horizons_ms)
    last_obs = max(0, frontier_frame - 1)
    dyn_pred = _bit_map_pred(bit_logits)  # (F,H,W)
    # In the matched (decimated) regime a window may end up with fewer frames
    # than the frontier itself — then no horizon is in-window and the last
    # observed frame does not exist.  Guard the reference frame before any
    # indexing so a short decimated window is honestly reported invalid, not
    # an IndexError.
    persist = tokens[last_obs] if last_obs < n_frames else None  # (H,W)

    out: dict = {}
    for h, off in offsets.items():
        tgt_f = frontier_frame + off
        ok = (
            frontier_frame >= 1
            and last_obs < n_frames
            and tgt_f < n_frames
            and bool(valid[tgt_f])
            and bool(valid[last_obs])
        )
        if not ok:
            out[h] = {"valid": 0, "target_frame": int(tgt_f)}
            continue
        tok_f = tokens[tgt_f]  # (H,W)
        dyn_correct = (dyn_pred[tgt_f] == tok_f).astype(np.float64).reshape(-1)
        dyn_nll = bitwise_nll(bit_logits[tgt_f], tok_f).reshape(-1)
        persist_correct = (persist == tok_f).astype(np.float64).reshape(-1)
        out[h] = {
            "valid": 1,
            "target_frame": int(tgt_f),
            "dyn_top1": dyn_correct,
            "dyn_nll": dyn_nll,
            "persist_top1": persist_correct,
            "n_cells": int(tok_f.size),
        }
    return out


# ---------------------------------------------------------------------------
# Frontier-mask forward pass for one cached batch
# ---------------------------------------------------------------------------


def _frontier_mask(n_frames: int, frontier: int):
    """Visibility mask ``(F,H,W)`` — frames < frontier visible, rest masked."""
    from imas_ambix.camdyn.model import GRID_H, GRID_W  # noqa: PLC0415

    m = np.zeros((n_frames, GRID_H, GRID_W), dtype=bool)
    m[:frontier] = True
    return m


def _forward_batch(model, arr, torch, device, frontier: int, cond_stats):
    """Forward one batch under a FRONTIER mask → per-window bit-logits (numpy).

    Conditioning is z-scored here with the provided stats (mirrors
    ``Trainer._batch_to_tensors``).  Returns the float bit-logits
    ``(B,F,H,W,bits)`` on CPU.
    """
    from imas_ambix.camdyn.train import _normalise_conditioning  # noqa: PLC0415

    nf = arr["tokens"].shape[1]
    vis = np.broadcast_to(
        _frontier_mask(nf, frontier)[None], arr["visible"].shape
    ).copy()
    cv = _normalise_conditioning(arr["cond_values"], cond_stats)
    t_tokens = torch.from_numpy(np.ascontiguousarray(arr["tokens"])).to(device)
    t_vis = torch.from_numpy(vis).to(device)
    t_cv = torch.from_numpy(cv.astype(np.float32)).to(device)
    t_cm = torch.from_numpy(arr["cond_missing"].astype(np.float32)).to(device)
    t_dt = torch.from_numpy(arr["dt"].astype(np.float32)).to(device)
    with (
        torch.no_grad(),
        torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=(device.type == "cuda"),
        ),
    ):
        logits = model.module(t_tokens, t_vis, t_cv, t_cm, t_dt)
    return logits.float().cpu().numpy()


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _agg_horizon(records, horizons_ms):
    """Collapse per-window horizon records into per-horizon paired arrays.

    Returns ``{h_ms: {dyn_top1, dyn_nll, persist_top1, base_top1, base_nll,
    n_windows, n_cells}}`` with the per-cell arrays concatenated across all
    valid windows (paired element-wise across predictors).
    """
    agg: dict = {
        h: {
            "dyn_top1": [],
            "dyn_nll": [],
            "persist_top1": [],
            "base_top1": [],
            "base_nll": [],
            "n_windows": 0,
        }
        for h in horizons_ms
    }
    for rec in records:
        for h in horizons_ms:
            d = rec["dyn"][h]
            if not d.get("valid"):
                continue
            b = rec["base"][h]
            agg[h]["dyn_top1"].append(d["dyn_top1"])
            agg[h]["dyn_nll"].append(d["dyn_nll"])
            agg[h]["persist_top1"].append(d["persist_top1"])
            agg[h]["base_top1"].append(b["dyn_top1"])  # base arm scored same fn
            agg[h]["base_nll"].append(b["dyn_nll"])
            agg[h]["n_windows"] += 1
    out: dict = {}
    for h in horizons_ms:
        a = agg[h]
        if a["n_windows"] == 0:
            out[h] = {"valid_windows": 0, "note": "no in-window valid windows"}
            continue
        dyn_top1 = np.concatenate(a["dyn_top1"])
        dyn_nll = np.concatenate(a["dyn_nll"])
        persist_top1 = np.concatenate(a["persist_top1"])
        base_top1 = np.concatenate(a["base_top1"])
        base_nll = np.concatenate(a["base_nll"])
        # paired diffs oriented positive = dynamics better
        dyn_vs_persist = bootstrap_ci(dyn_top1 - persist_top1)
        dyn_vs_base = bootstrap_ci(dyn_top1 - base_top1)
        dyn_vs_base_nll = bootstrap_ci(base_nll - dyn_nll)
        out[h] = {
            "valid_windows": int(a["n_windows"]),
            "n_cells": int(dyn_top1.size),
            "dynamics": {
                "top1": float(dyn_top1.mean()),
                "nll": float(dyn_nll.mean()),
            },
            "persistence": {"top1": float(persist_top1.mean())},
            "baseline": {
                "top1": float(base_top1.mean()),
                "nll": float(base_nll.mean()),
            },
            "dynamics_vs_persistence_top1": dyn_vs_persist,
            "dynamics_vs_baseline_top1": dyn_vs_base,
            "dynamics_vs_baseline_nll": dyn_vs_base_nll,
        }
    return out


# ---------------------------------------------------------------------------
# Main eval
# ---------------------------------------------------------------------------


def horizon_table(
    baseline_ckpt,
    dynamics_ckpt,
    *,
    split_path=None,
    device="cuda",
    frontier=8,
    eval_seed=999,
    horizons_ms=HORIZON_MS,
):
    """Build the three-predictor forward-horizon table (native + matched).

    ``frontier`` is the conditioning frontier frame (frames ``< frontier``
    visible).  Default 8 (half a 16-frame window) leaves room for the
    in-window future and matches the ``frontier_half`` named geometry.
    """
    import torch  # noqa: PLC0415

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    base_model, base_full_cfg, base_stats = _load_arm(baseline_ckpt, torch, dev)
    dyn_model, _dyn_cfg, _dyn_stats = _load_arm(dynamics_ckpt, torch, dev)

    tcfg = TrainConfig.from_dict(base_full_cfg)
    if split_path is not None:
        tcfg.split_path = str(split_path)
    tcfg.device = str(dev)
    tcfg.num_workers = 0  # single-process materialise (compare OOM lesson)
    tr = Trainer(tcfg)
    tr._cond_stats = base_stats

    split = tr._load_split()
    ho_specs = _specs_for_shots(split.held_out, max_shots=tcfg.max_heldout_shots)
    nf = tcfg.n_frames
    max_h = float(max(horizons_ms))

    # NATIVE regime: contiguous nf-frame windows exactly as the arms trained.
    frame_cfg = FrameWindowConfig(n_frames=nf, stride=tcfg.stride, seed=tcfg.seed)
    logger.info(
        "[horizon] materialising <=%d native held-out windows (%d shots)",
        tcfg.eval_windows,
        len(ho_specs),
    )
    batches = tr._materialize_eval(
        ho_specs, frame_cfg, max_windows=tcfg.eval_windows, seed=eval_seed
    )

    # MATCHED regime: a contiguous nf-frame window only spans ~15 ms at MAST
    # cadence, so it can never reach the 50/200 ms horizons.  Read WIDE native
    # windows (nf * WIDE_FACTOR frames) and decimate each down to nf frames
    # spanning the horizon range — the Δt-conditioned model then sees the wider
    # spacing through its dt input.  WIDE_FACTOR=16 → a 256-frame native window
    # spans the full 200 ms horizon at >=0.78 ms/frame; faster shots reach only
    # the shorter horizons (reported valid=0 for the rest).
    wide_factor = 16
    wide_n = nf * wide_factor
    wide_cfg = FrameWindowConfig(n_frames=wide_n, stride=wide_n, seed=tcfg.seed)
    logger.info(
        "[horizon] materialising <=%d wide (%d-frame) held-out windows for the "
        "matched regime (%d shots)",
        tcfg.eval_windows,
        wide_n,
        len(ho_specs),
    )
    wide_batches = tr._materialize_eval(
        ho_specs, wide_cfg, max_windows=tcfg.eval_windows, seed=eval_seed
    )

    out: dict = {
        "task": (
            "forward-horizon reconstruction (W2): dynamics vs persistence vs baseline"
        ),
        "metrics_provenance": (
            "camdyn.metrics.horizon_frame_offsets + bootstrap_ci (pre-registered)"
        ),
        "horizons_ms": list(horizons_ms),
        "frontier_frame": int(frontier),
        "n_frames": int(nf),
        "n_heldout_shots": len(ho_specs),
        "n_batches": len(batches),
        "n_wide_batches": len(wide_batches),
        "wide_window_frames": int(wide_n),
        "baseline_ckpt": str(baseline_ckpt),
        "dynamics_ckpt": str(dynamics_ckpt),
        "cadence_note": (
            "rbb cadence is heterogeneous (~13us..1ms/frame); a native 16-frame "
            "window spans ~0.2..15 ms so most physical horizons fall OUTSIDE the "
            "window at native cadence (valid=0). The matched regime reads WIDE "
            f"{wide_n}-frame native windows and decimates each to {nf} frames "
            "spanning the horizon range; the model is Dt-conditioned so the wider "
            "spacing is reflected in the cond vector. Shots faster than "
            "~0.78 ms/frame still cannot reach 200 ms within the wide window and "
            "report valid=0 for the unreachable horizons (honest)."
        ),
    }

    for regime in ("native", "matched"):
        records = []
        source_batches = batches if regime == "native" else wide_batches
        for arr in source_batches:
            # matched: decimate the wide native window down to nf frames spanning
            # the horizon range (per-shot stride from the window's dt)
            barr = decimate_to_n(arr, nf, max_h) if regime == "matched" else arr
            dyn_bl = _forward_batch(dyn_model, barr, torch, dev, frontier, _dyn_stats)
            base_bl = _forward_batch(base_model, barr, torch, dev, frontier, base_stats)
            for b in range(barr["tokens"].shape[0]):
                dyn_rec = score_window_horizons(
                    dyn_bl[b],
                    barr["tokens"][b],
                    barr["frame_time"][b],
                    barr["valid"][b],
                    frontier,
                    horizons_ms=horizons_ms,
                )
                base_rec = score_window_horizons(
                    base_bl[b],
                    barr["tokens"][b],
                    barr["frame_time"][b],
                    barr["valid"][b],
                    frontier,
                    horizons_ms=horizons_ms,
                )
                records.append({"dyn": dyn_rec, "base": base_rec})
        table = _agg_horizon(records, horizons_ms)
        out[regime] = {
            "table": {str(h): table[h] for h in horizons_ms},
            "n_windows_scored": len(records),
        }
        if regime == "matched":
            out[regime]["n_reachable_horizons"] = int(
                sum(1 for h in horizons_ms if table[h].get("valid_windows"))
            )

    # verdict: in the matched regime (the populated table), does dynamics beat
    # persistence AND the baseline at every reachable horizon on top-1?
    verdict: dict = {}
    for h in horizons_ms:
        cell = out["matched"]["table"][str(h)]
        if not cell.get("valid_windows"):
            verdict[str(h)] = {"reachable": False}
            continue
        verdict[str(h)] = {
            "reachable": True,
            "beats_persistence": bool(
                cell["dynamics_vs_persistence_top1"]["favours_dynamics"]
            ),
            "beats_baseline": bool(
                cell["dynamics_vs_baseline_top1"]["favours_dynamics"]
            ),
        }
    out["verdict_matched"] = verdict
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="camdyn forward-horizon forecasting table")
    p.add_argument("--baseline", required=True)
    p.add_argument("--dynamics", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--split-path", default=None)
    p.add_argument("--frontier", type=int, default=8)
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    table = horizon_table(
        args.baseline,
        args.dynamics,
        split_path=args.split_path,
        device=args.device,
        frontier=args.frontier,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(table, indent=2), encoding="utf-8")
    logger.info("[horizon] table written to %s", out_path)
    for h in table["horizons_ms"]:
        cell = table["matched"]["table"][str(h)]
        if cell.get("valid_windows"):
            logger.info(
                "[horizon] matched h=%.0fms: dyn=%.4f persist=%.4f base=%.4f "
                "(beats_persist=%s beats_base=%s, n=%d windows)",
                h,
                cell["dynamics"]["top1"],
                cell["persistence"]["top1"],
                cell["baseline"]["top1"],
                cell["dynamics_vs_persistence_top1"]["favours_dynamics"],
                cell["dynamics_vs_baseline_top1"]["favours_dynamics"],
                cell["valid_windows"],
            )
        else:
            logger.info("[horizon] matched h=%.0fms: %s", h, cell.get("note"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
