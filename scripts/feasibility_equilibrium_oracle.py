#!/usr/bin/env python
"""Feasibility oracle: can decoded camera frames predict plasma geometry?

The make-or-break check for the physics-grounded controllability gate.  If a
small CNN over DECODED rbb camera frames cannot predict the L2 equilibrium
geometry (magnetic axis / X-point / LCFS shape) with error materially below the
shot-to-shot spread, then a geometry-based controllability gate is INFEASIBLE
and we stop before the big build.

EVALUATOR-ONLY (binding firewall)
---------------------------------
This is a third-party EVALUATOR.  It uses real decoded camera pixels as the
probe input and the L2 equilibrium as the LABEL.  Nothing here is, or is
importable by, the world-model training path — the probe + labels only consume
data and produce evaluator metrics.  No WM checkpoint is loaded.

What it does
------------
1.  Build a shot-disjoint split (``imas_ambix.camdyn.splits``) over rbb-token
    shots, FORCING the controllability gate cohort + the standing held-out
    shots into the oracle TEST set (so the oracle is evaluated on the SAME
    held-out plasma the gate will be scored on, and never trains on it).
2.  Sample ~80-150 TRAIN shots; assemble one rbb window (24 frames over the
    ~0.25 s horizon) per shot for TRAIN and TEST.
3.  DECODE the real camera tokens to 256x256 pixels through the frozen
    Open-MAGVIT2 decoder (loaded ONCE, batched — AGENTS.md §2b).
4.  Build the 12-D equilibrium labels per frame
    (:mod:`imas_ambix.worldmodel.equilibrium_labels`).
5.  Train :class:`~imas_ambix.worldmodel.equilibrium_probe.EquilibriumProbe`
    a few epochs on TRAIN; evaluate on TEST.
6.  Report PER-COMPONENT RMSE in METRES and the predict-the-TRAIN-mean baseline
    RMSE (the shot-to-shot spread).  VERDICT: is the probe RMSE materially below
    the mean-predictor baseline for axis + X-point?

Outputs (JSON + a pred-vs-true axis scatter) under
``/work/projects/imas_gpu/worldmodel/equilibrium_oracle/`` and
``docs/figures/joint-multimodal-plasma-wm/``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
from pathlib import Path

import numpy as np

logger = logging.getLogger("equilibrium_oracle")

# --- Output locations -------------------------------------------------------

DEFAULT_OUT_ROOT = Path("/work/projects/imas_gpu/worldmodel/equilibrium_oracle")
DEFAULT_FIG_DIR = Path("docs/figures/joint-multimodal-plasma-wm")

# --- Cohorts that MUST be in the oracle TEST set (never trained on) ----------
#: The controllability gate cohort + the standing held-out shots.  Forced into
#: the oracle TEST partition so the feasibility verdict is read on the SAME
#: plasma the downstream gate scores.
GATE_COHORT = (15089, 15223, 15517, 15963, 15972, 16024, 16223)
STANDING_HELD_OUT = (18502, 18503, 18504, 18505)
FORCED_TEST_SHOTS = tuple(sorted(set(GATE_COHORT) | set(STANDING_HELD_OUT)))

#: Token-grid side (Open-MAGVIT2 LFQ) and decoded image side.
GRID_H, GRID_W = 16, 16
IMAGE_SIZE = 256


# ---------------------------------------------------------------------------
# Window assembly (rbb tokens -> one 24-frame window per shot, + labels)
# ---------------------------------------------------------------------------


def _select_brightest_window(token_path: Path, level1_path, camera, n_frames, stride):
    """Pick the brightest n_frames window for a shot (most plasma-active).

    Returns (start, frame_times, tokens) or None.  Brightness ranks windows by
    mean raw-frame intensity (the honest activity proxy) so the oracle sees an
    established plasma, not a dark ramp-up.  Falls back to the first full window
    if the raw frames are unavailable.
    """
    from imas_ambix.camdyn.dataset import (
        FrameTokenDataset,
        FrameTokenShotSpec,
        FrameWindowConfig,
        _token_n_frames,
    )
    from imas_ambix.camdyn.reconstruction_demo import _window_brightness

    n = _token_n_frames(token_path)
    if n is None or n < n_frames:
        return None
    spec = FrameTokenShotSpec(
        shot_id=0,
        n_frames=n,
        token_path=token_path,
        level1_path=level1_path,
        camera=camera,
    )
    cfg = FrameWindowConfig(n_frames=n_frames, stride=stride, seed=0)
    ds = FrameTokenDataset([spec], cfg)
    if len(ds) == 0:
        return None
    starts = [ds._windows[i][1] for i in range(len(ds))]
    # _window_brightness reads the level-1 raw frames by SHOT id; recover it.
    shot_id = int(Path(token_path).parent.name)
    bright = _window_brightness(shot_id, starts, n_frames)
    pick = int(np.argmax(bright)) if bright is not None else 0
    win = ds[pick]
    return (
        int(win.start),
        np.asarray(win.frame_time, np.float64),
        np.asarray(win.tokens, np.int64),
    )


def assemble_windows(shot_ids, *, camera, n_frames, stride, level2_root, token_root):
    """Build one labelled rbb window per shot.

    Returns a list of dicts: ``{shot_id, start, frame_times, tokens (F,16,16),
    target (F,12), mask (F,12)}`` for every shot that yields a full window with
    at least one finite-label frame.
    """
    from imas_ambix.camdyn.dataset import discover_token_shots
    from imas_ambix.worldmodel.equilibrium_labels import load_equilibrium_geometry

    specs = discover_token_shots(
        camera=camera,
        token_root=token_root,
        shot_ids=list(shot_ids),
        read_n_frames=False,
    )
    spec_by_shot = {s.shot_id: s for s in specs}

    out = []
    for sid in shot_ids:
        spec = spec_by_shot.get(int(sid))
        if spec is None:
            logger.info("shot %d: no rbb tokens — skip", sid)
            continue
        sel = _select_brightest_window(
            spec.token_path, spec.level1_path, camera, n_frames, stride
        )
        if sel is None:
            logger.info("shot %d: no full %d-frame window — skip", sid, n_frames)
            continue
        start, frame_times, tokens = sel
        try:
            geo = load_equilibrium_geometry(
                int(sid), frame_times, level2_root=level2_root
            )
        except (KeyError, FileNotFoundError) as exc:
            logger.info("shot %d: no equilibrium (%s) — skip", sid, exc)
            continue
        if not geo.finite_mask.any():
            logger.info("shot %d: all-masked equilibrium window — skip", sid)
            continue
        out.append(
            {
                "shot_id": int(sid),
                "start": int(start),
                "frame_times": frame_times,
                "tokens": tokens,  # (F,16,16) global ids
                "target": geo.target,  # (F,12) metres, NaN masked
                "mask": geo.finite_mask,  # (F,12) bool
            }
        )
        logger.info(
            "shot %d: window start %d, finite-frame frac %.2f",
            sid,
            start,
            float(geo.finite_mask.any(axis=1).mean()),
        )
    return out


# ---------------------------------------------------------------------------
# Decode all windows' tokens to grayscale frames in ONE VQ pass
# ---------------------------------------------------------------------------


def decode_all(windows, *, device, work_dir):
    """Decode every window's tokens to grayscale ``(F,256,256)`` in one VQ pass.

    Stacks all (N,F,16,16) token grids into a single bundle and runs the frozen
    Open-MAGVIT2 decoder once (loaded once, batched — AGENTS.md §2b).  Returns a
    list of ``(F,256,256)`` float32 grayscale arrays in [0,1], aligned to
    ``windows``.
    """
    from imas_ambix.camdyn.reconstruction_demo import run_decode_subprocess

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    token_bundle = work_dir / "tokens.npz"
    image_bundle = work_dir / "images.npz"

    grids = np.stack([w["tokens"] for w in windows]).astype(np.int64)  # (N,F,16,16)
    index = [{"role": str(i), "slot": i} for i in range(len(windows))]
    np.savez_compressed(
        token_bundle,
        grids=grids,
        index=json.dumps(index),
        meta=json.dumps({"format": "reconstruction_demo", "grid_hw": [GRID_H, GRID_W]}),
    )
    run_decode_subprocess(token_bundle, image_bundle, device)
    data = np.load(str(image_bundle), allow_pickle=True)
    images = np.asarray(data["images"], dtype=np.uint8)  # (N,F,256,256,3)

    gray = []
    for i in range(len(windows)):
        rgb = images[i].astype(np.float32)  # (F,256,256,3)
        g = rgb.mean(axis=-1) / 255.0  # grayscale in [0,1]
        gray.append(g.astype(np.float32))
    return gray


# ---------------------------------------------------------------------------
# Build (X, y, mask) probe tensors from decoded frames + labels
# ---------------------------------------------------------------------------


def build_examples(windows, gray, *, in_frames):
    """Channel-stack ``in_frames`` decoded frames per FRAME-CENTRED example.

    For each frame f in a window we form a k-frame grayscale stack
    ``[f-k//2 .. f+k//2]`` (clamped at the window edges) as the probe input and
    the 12-D label at frame f as the target.  Only frames with at least one
    finite label are kept.  Returns (X, y, mask, shot_of_example).
    """
    half = in_frames // 2
    x_stacks, y, m, shots = [], [], [], []
    for w, g in zip(windows, gray, strict=True):
        n_f = g.shape[0]
        for f in range(n_f):
            if not w["mask"][f].any():
                continue  # no finite label at this frame
            idx = np.clip(np.arange(f - half, f - half + in_frames), 0, n_f - 1)
            stack = g[idx]  # (k,256,256)
            x_stacks.append(stack)
            y.append(w["target"][f])
            m.append(w["mask"][f])
            shots.append(w["shot_id"])
    if not x_stacks:
        raise RuntimeError("no labelled examples assembled")
    return (
        np.stack(x_stacks).astype(np.float32),
        np.stack(y).astype(np.float32),
        np.stack(m).astype(bool),
        np.asarray(shots, dtype=np.int64),
    )


# ---------------------------------------------------------------------------
# Train + evaluate the probe
# ---------------------------------------------------------------------------


def standardise_stats(y, mask):
    """Per-component mean / std over the finite TRAIN labels (NaN-safe)."""
    dim = y.shape[1]
    mean = np.zeros(dim)
    std = np.ones(dim)
    for d in range(dim):
        vals = y[mask[:, d], d]
        if vals.size > 1:
            mean[d] = float(np.mean(vals))
            std[d] = float(np.std(vals)) or 1.0
    return mean, std


def train_probe(
    x_tr, ytr, mtr, *, in_frames, target_dim, epochs, batch_size, lr, device, seed
):
    """Train the heteroscedastic probe; returns (model, target_mean, target_std)."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from imas_ambix.worldmodel.equilibrium_probe import (
        EquilibriumProbe,
        ProbeConfig,
        gaussian_nll,
    )

    torch.manual_seed(seed)
    mean, std = standardise_stats(ytr, mtr)
    # Standardise labels; NaN (masked) -> 0 (zeroed by the mask in the loss).
    ystd = (np.nan_to_num(ytr, nan=0.0) - mean) / std
    ystd = np.where(mtr, ystd, 0.0).astype(np.float32)

    dev = torch.device(device)
    ds = TensorDataset(
        torch.from_numpy(x_tr),
        torch.from_numpy(ystd),
        torch.from_numpy(mtr.astype(np.float32)),
    )
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    model = EquilibriumProbe(
        ProbeConfig(in_frames=in_frames, image_size=IMAGE_SIZE, target_dim=target_dim)
    ).to(dev)
    logger.info("probe params: %.2fM", model.n_parameters() / 1e6)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    model.train()
    for ep in range(epochs):
        tot, nb = 0.0, 0
        for xb, yb, mb in dl:
            xb, yb, mb = xb.to(dev), yb.to(dev), mb.to(dev)
            opt.zero_grad()
            with torch.autocast(
                device_type=dev.type, dtype=torch.bfloat16, enabled=(dev.type == "cuda")
            ):
                pmean, plog = model(xb)
            loss = gaussian_nll(pmean.float(), plog.float(), yb, mb)
            loss.backward()
            opt.step()
            tot += float(loss.detach())
            nb += 1
        logger.info("epoch %d/%d  NLL=%.4f", ep + 1, epochs, tot / max(nb, 1))
    return model, mean, std


def evaluate(model, x_te, yte, mte, mean, std, *, device, batch_size):
    """Predict on TEST -> per-component predictions in metres (NaN where masked)."""
    import torch

    dev = torch.device(device)
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, x_te.shape[0], batch_size):
            xb = torch.from_numpy(x_te[i : i + batch_size]).to(dev)
            pmean_m, _ = model.predict_metres(xb, mean, std)
            preds.append(pmean_m)
    pred = np.concatenate(preds, axis=0)  # (n_te, dim) metres
    return pred


def per_component_rmse(pred, y, mask):
    """Per-component RMSE in metres over finite-label TEST elements."""
    dim = y.shape[1]
    out = np.full(dim, np.nan)
    for d in range(dim):
        sel = mask[:, d]
        if sel.sum() == 0:
            continue
        err = pred[sel, d] - y[sel, d]
        out[d] = float(np.sqrt(np.mean(err**2)))
    return out


def mean_predictor_rmse(ytr, mtr, yte, mte):
    """Baseline RMSE: predict the TRAIN mean for every TEST example.

    This is the shot-to-shot spread of each component on the TEST set — the bar
    the probe must beat to be feasible.
    """
    dim = yte.shape[1]
    out = np.full(dim, np.nan)
    for d in range(dim):
        tr = ytr[mtr[:, d], d]
        te = yte[mte[:, d], d]
        if tr.size == 0 or te.size == 0:
            continue
        out[d] = float(np.sqrt(np.mean((te - float(np.mean(tr))) ** 2)))
    return out


# ---------------------------------------------------------------------------
# Scatter figure
# ---------------------------------------------------------------------------


def axis_scatter(pred, y, mask, names, out_path):
    """Pred-vs-true scatter for axis_R, axis_Z (the headline components)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), constrained_layout=True)
    for ax, d, label in zip(axes, (0, 1), ("axis_R", "axis_Z"), strict=True):
        sel = mask[:, d]
        if sel.sum() == 0:
            ax.set_title(f"{label}: no finite test labels")
            continue
        yt = y[sel, d]
        yp = pred[sel, d]
        lo = float(min(yt.min(), yp.min()))
        hi = float(max(yt.max(), yp.max()))
        ax.scatter(yt, yp, s=10, alpha=0.5, edgecolors="none")
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="ideal")
        rmse = float(np.sqrt(np.mean((yp - yt) ** 2)))
        ax.set_xlabel(f"true {label} (m)")
        ax.set_ylabel(f"predicted {label} (m)")
        ax.set_title(f"{label}  RMSE={rmse * 100:.1f} cm  (n={sel.sum()})")
        ax.legend(fontsize=8)
    fig.suptitle(
        "equilibrium feasibility oracle — decoded rbb frames -> plasma geometry",
        fontsize=11,
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("wrote %s", out_path)


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------


def verdict(rmse_probe, rmse_baseline, names, *, ratio_threshold):
    """Feasibility verdict: probe RMSE materially below the mean-predictor.

    PASS criterion: for axis + X-point components, the probe RMSE is below the
    mean-predictor baseline by at least ``ratio_threshold`` (probe/baseline <
    1/ratio_threshold), i.e. the probe captures real geometry signal beyond the
    shot-to-shot spread.  Reports per-component skill = 1 - probe/baseline.
    """
    rows = []
    key = {"axis_R", "axis_Z", "xpt_R", "xpt_Z"}
    axis_xpt_pass = []
    for d, nm in enumerate(names):
        rp = rmse_probe[d]
        rb = rmse_baseline[d]
        if rb is None or not np.isfinite(rb) or rb == 0 or not np.isfinite(rp):
            skill = None
            beats = None
        else:
            skill = 1.0 - rp / rb
            beats = bool(rp < rb / ratio_threshold)
        rows.append(
            {
                "component": nm,
                "rmse_probe_m": None if not np.isfinite(rp) else float(rp),
                "rmse_baseline_m": None
                if (rb is None or not np.isfinite(rb))
                else float(rb),
                "skill": None if skill is None else float(skill),
                "beats_baseline": beats,
            }
        )
        if nm in key and beats is not None:
            axis_xpt_pass.append(beats)
    overall = bool(axis_xpt_pass) and all(axis_xpt_pass)
    return {
        "feasible": overall,
        "criterion": (
            f"probe RMSE < baseline / {ratio_threshold:g} for ALL of "
            "axis_R, axis_Z, xpt_R, xpt_Z (probe captures geometry beyond "
            "shot-to-shot spread)"
        ),
        "ratio_threshold": ratio_threshold,
        "components": rows,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(args) -> int:
    from imas_ambix.camdyn.dataset import list_token_shot_ids
    from imas_ambix.camdyn.splits import build_camdyn_split
    from imas_ambix.worldmodel.equilibrium_labels import TARGET_DIM, TARGET_NAMES

    rng = np.random.default_rng(args.seed)
    token_root = Path(args.token_root) if args.token_root else None
    level2_root = Path(args.level2_root) if args.level2_root else None

    # 1) split — force the gate cohort + held-out into the oracle TEST set.
    all_shots = list_token_shot_ids(camera=args.camera, token_root=token_root)
    logger.info("rbb token shots on disk: %d", len(all_shots))
    split = build_camdyn_split(
        all_shots,
        mse_heldout=list(FORCED_TEST_SHOTS),  # forced into held_out (TEST)
        val_fraction=0.0,
        held_out_fraction=args.held_out_fraction,
        seed=args.seed,
    )
    train_pool = list(split.train)
    test_pool = list(split.held_out)
    rng.shuffle(train_pool)
    rng.shuffle(test_pool)
    train_shots = train_pool[: args.n_train_shots]
    test_shots = test_pool[: args.n_test_shots]
    # always include the forced cohort that has tokens
    forced_present = [s for s in FORCED_TEST_SHOTS if s in set(all_shots)]
    test_shots = sorted(set(test_shots) | set(forced_present))
    logger.info(
        "TRAIN shots=%d  TEST shots=%d (forced present: %s)",
        len(train_shots),
        len(test_shots),
        forced_present,
    )

    # 2-3) assemble + decode
    work_dir = Path(
        tempfile.mkdtemp(prefix="eq-oracle-", dir=os.environ.get("TMPDIR", "/tmp"))
    )
    tr_windows = assemble_windows(
        train_shots,
        camera=args.camera,
        n_frames=args.n_frames,
        stride=args.stride,
        level2_root=level2_root,
        token_root=token_root,
    )
    te_windows = assemble_windows(
        test_shots,
        camera=args.camera,
        n_frames=args.n_frames,
        stride=args.stride,
        level2_root=level2_root,
        token_root=token_root,
    )
    if not tr_windows or not te_windows:
        logger.error("empty TRAIN or TEST window set — cannot run oracle")
        return 2

    device = "cuda" if _cuda_available() else "cpu"
    logger.info(
        "decoding %d TRAIN + %d TEST windows on %s",
        len(tr_windows),
        len(te_windows),
        device,
    )
    tr_gray = decode_all(tr_windows, device=device, work_dir=work_dir / "train")
    te_gray = decode_all(te_windows, device=device, work_dir=work_dir / "test")

    # 4) probe tensors
    x_tr, ytr, mtr, _ = build_examples(tr_windows, tr_gray, in_frames=args.in_frames)
    x_te, yte, mte, shots_te = build_examples(
        te_windows, te_gray, in_frames=args.in_frames
    )
    logger.info("TRAIN examples=%d  TEST examples=%d", x_tr.shape[0], x_te.shape[0])

    # 5) train + eval
    model, tmean, tstd = train_probe(
        x_tr,
        ytr,
        mtr,
        in_frames=args.in_frames,
        target_dim=TARGET_DIM,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        seed=args.seed,
    )
    pred = evaluate(
        model, x_te, yte, mte, tmean, tstd, device=device, batch_size=args.batch_size
    )

    # 6) metrics + verdict
    rmse_probe = per_component_rmse(pred, yte, mte)
    rmse_base = mean_predictor_rmse(ytr, mtr, yte, mte)
    verd = verdict(
        rmse_probe, rmse_base, TARGET_NAMES, ratio_threshold=args.ratio_threshold
    )

    coverage = {
        "train_label_finite_frac": float(mtr.mean()),
        "test_label_finite_frac": float(mte.mean()),
        "train_examples": int(x_tr.shape[0]),
        "test_examples": int(x_te.shape[0]),
        "train_shots": [int(s) for s in train_shots],
        "test_shots": [int(s) for s in test_shots],
        "forced_test_present": [int(s) for s in forced_present],
    }
    report = {
        "task": "equilibrium feasibility oracle (decoded rbb -> geometry)",
        "evaluator_only": True,
        "camera": args.camera,
        "n_frames": args.n_frames,
        "in_frames": args.in_frames,
        "epochs": args.epochs,
        "target_names": list(TARGET_NAMES),
        "target_units": "m",
        "coverage": coverage,
        "verdict": verd,
    }

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    json_path = out_root / "oracle_report.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("wrote %s", json_path)

    fig_local = out_root / "fig-eq-oracle-axis-scatter.png"
    axis_scatter(pred, yte, mte, TARGET_NAMES, fig_local)
    fig_docs = Path(args.fig_dir) / "fig-eq-oracle-axis-scatter.png"
    try:
        axis_scatter(pred, yte, mte, TARGET_NAMES, fig_docs)
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not write docs figure %s: %s", fig_docs, exc)

    # console summary
    logger.info(
        "=== FEASIBILITY VERDICT: %s ===",
        "FEASIBLE" if verd["feasible"] else "INFEASIBLE",
    )
    for row in verd["components"]:
        logger.info(
            "  %-10s probe=%s  baseline=%s  skill=%s  beats=%s",
            row["component"],
            "n/a"
            if row["rmse_probe_m"] is None
            else f"{row['rmse_probe_m'] * 100:.1f}cm",
            "n/a"
            if row["rmse_baseline_m"] is None
            else f"{row['rmse_baseline_m'] * 100:.1f}cm",
            "n/a" if row["skill"] is None else f"{row['skill']:+.2f}",
            row["beats_baseline"],
        )
    return 0


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--camera", default="rbb")
    p.add_argument("--n-frames", type=int, default=24, help="frames per rbb window")
    p.add_argument("--stride", type=int, default=12)
    p.add_argument(
        "--in-frames",
        type=int,
        default=4,
        help="channel-stacked probe input frames (k)",
    )
    p.add_argument("--n-train-shots", type=int, default=120)
    p.add_argument("--n-test-shots", type=int, default=40)
    p.add_argument("--held-out-fraction", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument(
        "--ratio-threshold",
        type=float,
        default=1.3,
        help="probe must beat baseline by this factor on axis+X-point",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--token-root", default=None, help="override token root")
    p.add_argument("--level2-root", default=None, help="override L2 equilibrium root")
    p.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    p.add_argument("--fig-dir", default=str(DEFAULT_FIG_DIR))
    return p


def main(argv=None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    args = build_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
