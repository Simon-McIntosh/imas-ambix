#!/usr/bin/env python
"""Feasibility oracle: can MEASURED DIAGNOSTICS predict plasma geometry?

The decisive follow-up to the camera-only oracle.  The camera oracle FAILED on
the interior geometry (magnetic-axis / X-point skill ~0 — the interior current
distribution is not a camera observable), but the MEASURED magnetics (flux loops
+ B-field probes) are precisely the inputs an EFIT-class reconstruction uses to
*determine* the boundary / X-point.  So a diagnostics→equilibrium referee should
be feasible where the camera one was not — and because the joint world model
already DREAMS these diagnostics, a feasible referee here lets the
controllability gate score the DREAMED diagnostics through this same map.

This oracle mirrors :mod:`scripts.feasibility_equilibrium_oracle` but swaps the
INPUT: instead of decoding camera tokens to pixels, it reads the per-window
MEASURED-SIGNAL tokens (the same magnetics / interferometer / soft-x-ray /
Dα-boundary / xsx / xim / ait / summary / pf_active / gas_injection streams the
WM conditions on) and trains a small temporal probe to predict the 12-D
equilibrium geometry.

EVALUATOR-ONLY (binding firewall)
---------------------------------
A third-party EVALUATOR.  The probe input is measured diagnostics; the LABEL is
the L2 equilibrium.  Nothing here is, or is importable by, the world-model
training path — the probe + labels only consume data and produce evaluator
metrics.  No WM checkpoint is loaded.  Equilibrium is an evaluator label only.

What it does
------------
1.  Build a shot-disjoint split (:mod:`imas_ambix.camdyn.splits`) over rbb-token
    shots, FORCING the controllability gate cohort + the standing held-out shots
    into the oracle TEST set (so the oracle is read on the SAME held-out plasma
    the gate is scored on, and never trains on it).
2.  Sample ~150-300 TRAIN shots; assemble one camera window (~0.25 s horizon) per
    shot for TRAIN and TEST to define the window's time span.
3.  Read the per-window MEASURED-SIGNAL tokens at ``n_signal_steps`` temporal
    positions across that span (NO camera decode — much cheaper).
4.  Build the 12-D equilibrium labels at the signal-grid times
    (:mod:`imas_ambix.worldmodel.equilibrium_labels`); the probe predicts the
    window-CENTRE geometry.
5.  Train the diagnostics probe
    (:mod:`imas_ambix.worldmodel.diagnostics_equilibrium_probe`) a few epochs on
    TRAIN; evaluate on TEST.
6.  Report PER-COMPONENT RMSE in METRES + the predict-the-TRAIN-mean baseline
    (the shot-to-shot spread) + skill = 1 - rmse/baseline, especially for
    axis_R, axis_Z, xpt_R, xpt_Z.  ABLATION: magnetics-only vs all-diagnostics.

Outputs (JSON + a pred-vs-true axis/X-point scatter) under
``/work/projects/imas_gpu/worldmodel/diagnostics_equilibrium_oracle/`` and
``docs/figures/joint-multimodal-plasma-wm/``.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger("diag_eq_oracle")

# --- Output locations -------------------------------------------------------

DEFAULT_OUT_ROOT = Path(
    "/work/projects/imas_gpu/worldmodel/diagnostics_equilibrium_oracle"
)
DEFAULT_FIG_DIR = Path("docs/figures/joint-multimodal-plasma-wm")

# --- Cohorts that MUST be in the oracle TEST set (never trained on) ----------
#: The controllability gate cohort + the standing held-out shots.  Forced into
#: the oracle TEST partition so the feasibility verdict is read on the SAME
#: plasma the downstream gate scores.
GATE_COHORT = (15089, 15223, 15517, 15963, 15972, 16024, 16223)
STANDING_HELD_OUT = (18502, 18503, 18504, 18505)
FORCED_TEST_SHOTS = tuple(sorted(set(GATE_COHORT) | set(STANDING_HELD_OUT)))

#: Streams treated as "magnetics" for the magnetics-only ablation arm.  ``xma``
#: is the HF magnetics codebook; ``magnetics`` is the calibrated L2 flux-loop +
#: B-field probe array — together they are the EFIT-class position/shape sensor.
MAGNETICS_STREAMS = ("xma", "magnetics")


# ---------------------------------------------------------------------------
# Window + signal + label assembly (one labelled example per shot window)
# ---------------------------------------------------------------------------


def _select_brightest_start(token_path, level1_path, camera, config):
    """Pick the brightest valid window start for a shot (most plasma-active).

    Returns an int start frame, or None to fall back to the centred window.
    Brightness ranks candidate starts by mean raw-frame intensity (the honest
    activity proxy) so the probe sees an established plasma, not a dark ramp.
    """
    try:
        from imas_ambix.camdyn.reconstruction_demo import _window_brightness
        from imas_ambix.worldmodel.spacetime_dataset import (
            _fps_from_times,
            _frame_times,
            camera_frame_count,
            effective_frame_stride,
        )
    except Exception:  # noqa: BLE001
        return None
    try:
        shot_id = int(Path(token_path).parent.name)
        times = _frame_times(shot_id, camera, token_root=None)
        n_total = camera_frame_count(shot_id, camera, token_root=None)
    except Exception:  # noqa: BLE001
        return None
    fps = _fps_from_times(times)
    stride = effective_frame_stride(config, fps)
    span = (config.n_frames - 1) * stride + 1
    if n_total < span:
        return None
    # candidate starts every span//2 frames
    step = max(1, span // 2)
    starts = list(range(0, n_total - span + 1, step))
    if not starts:
        return None
    bright = _window_brightness(shot_id, starts, span)
    if bright is None:
        return None
    return int(starts[int(np.argmax(bright))])


def assemble_examples(
    shot_ids,
    *,
    camera,
    modalities,
    n_signal_steps,
    config,
    level2_root,
    token_root,
):
    """Build one labelled diagnostics example per shot.

    For each shot: assemble a camera window (brightest start, fall back centred)
    to fix the ~0.25 s time span; read the measured-signal tokens at
    ``n_signal_steps`` positions across that span; build the 12-D equilibrium
    labels at those signal-grid times; keep the WINDOW-CENTRE label as the target.

    Returns a list of dicts: ``{shot_id, signals {name:(S,C)}, target (12,),
    mask (12,)}`` for every shot that yields a window with >=1 present stream and
    >=1 finite label component at the centre.
    """
    from imas_ambix.camdyn.dataset import discover_token_shots
    from imas_ambix.worldmodel.equilibrium_labels import load_equilibrium_geometry
    from imas_ambix.worldmodel.spacetime_dataset import assemble_window
    from imas_ambix.worldmodel.spacetime_dataset_v2 import read_window_signals

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
        start = _select_brightest_start(
            spec.token_path, spec.level1_path, camera, config
        )
        try:
            sample = assemble_window(
                int(sid),
                config,
                camera=camera,
                token_root=token_root,
                start_frame=start,
            )
        except (ValueError, FileNotFoundError, KeyError) as exc:
            logger.info("shot %d: no window (%s) — skip", sid, exc)
            continue
        signals = read_window_signals(
            int(sid), sample, modalities, n_signal_steps, token_root=token_root
        )
        if not signals:
            logger.info("shot %d: no readable measured streams — skip", sid)
            continue
        # equilibrium labels on the SIGNAL grid (same span as the window).
        ftime = np.asarray(sample.frame_time, dtype=np.float64)
        t0, t1 = float(ftime.min()), float(ftime.max())
        grid = np.linspace(t0, t1, int(n_signal_steps), dtype=np.float64)
        try:
            geo = load_equilibrium_geometry(int(sid), grid, level2_root=level2_root)
        except (KeyError, FileNotFoundError) as exc:
            logger.info("shot %d: no equilibrium (%s) — skip", sid, exc)
            continue
        # window-CENTRE label (the probe predicts the geometry at mid-window).
        cidx = int(n_signal_steps // 2)
        tgt = geo.target[cidx]  # (12,)
        msk = geo.finite_mask[cidx]  # (12,)
        if not msk.any():
            # centre is masked (off-plasma) — try the nearest finite step.
            any_finite = geo.finite_mask.any(axis=1)
            if not any_finite.any():
                logger.info("shot %d: all-masked equilibrium window — skip", sid)
                continue
            order = np.argsort(np.abs(np.arange(n_signal_steps) - cidx))
            for j in order:
                if any_finite[j]:
                    tgt = geo.target[j]
                    msk = geo.finite_mask[j]
                    break
        out.append(
            {
                "shot_id": int(sid),
                "signals": {k: np.asarray(v, np.int64) for k, v in signals.items()},
                "target": np.asarray(tgt, np.float32),
                "mask": np.asarray(msk, bool),
            }
        )
        logger.info(
            "shot %d: streams=%s  finite-comp=%d/12",
            sid,
            ",".join(sorted(signals)),
            int(msk.sum()),
        )
    return out


# ---------------------------------------------------------------------------
# Stream sizing + tensor batching
# ---------------------------------------------------------------------------


def probe_channels(examples, modalities):
    """Max channel count seen per stream across the assembled examples.

    Caps at each modality's ``max_channels``.  A stream never present keeps 0 and
    is dropped from the model's stream list.
    """
    cap = {m.name: int(m.max_channels) for m in modalities}
    seen = {m.name: 0 for m in modalities}
    for ex in examples:
        for name, arr in ex["signals"].items():
            if name in seen:
                seen[name] = max(seen[name], int(arr.shape[1]))
    return {k: min(v, cap.get(k, v)) for k, v in seen.items()}


def build_stream_specs(channels, modalities, *, restrict=None):
    """Build probe :class:`StreamSpec` list from probed channels.

    ``restrict`` (optional set of stream names) keeps only those streams — the
    ablation lever.  A stream with 0 probed channels is dropped.
    """
    from imas_ambix.worldmodel.diagnostics_equilibrium_probe import StreamSpec

    specs = []
    for m in modalities:
        if restrict is not None and m.name not in restrict:
            continue
        c = int(channels.get(m.name, 0))
        if c <= 0:
            continue
        specs.append(StreamSpec(name=m.name, vocab=int(m.vocab), channels=c))
    return specs


def batch_signals(examples, specs, n_steps, *, device):
    """Stack examples into ``{stream: (N, n_steps, channels) int64 tensors}``.

    Each stream is padded / truncated to its spec channel count (PAD id 0); an
    example missing a stream gets an all-PAD block (so the probe's zero-fill path
    is exercised consistently and shapes are uniform).
    """
    import torch

    from imas_ambix.worldmodel.dataset import PAD_LOCAL_ID

    n = len(examples)
    out = {}
    for sp in specs:
        block = np.full((n, n_steps, sp.channels), PAD_LOCAL_ID, dtype=np.int64)
        for i, ex in enumerate(examples):
            arr = ex["signals"].get(sp.name)
            if arr is None:
                continue
            s = min(n_steps, arr.shape[0])
            c = min(sp.channels, arr.shape[1])
            block[i, :s, :c] = np.clip(arr[:s, :c], 0, sp.vocab - 1)
        out[sp.name] = torch.from_numpy(block).to(device)
    return out


# ---------------------------------------------------------------------------
# Standardisation + train + eval
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
    tr_examples, specs, *, n_steps, target_dim, epochs, batch_size, lr, device, seed
):
    """Train the heteroscedastic probe; returns (model, target_mean, target_std)."""
    import torch

    from imas_ambix.worldmodel.diagnostics_equilibrium_probe import (
        DiagnosticsEquilibriumProbe,
        DiagnosticsProbeConfig,
        gaussian_nll,
    )

    torch.manual_seed(seed)
    ytr = np.stack([ex["target"] for ex in tr_examples]).astype(np.float32)
    mtr = np.stack([ex["mask"] for ex in tr_examples]).astype(bool)
    mean, std = standardise_stats(ytr, mtr)
    ystd = (np.nan_to_num(ytr, nan=0.0) - mean) / std
    ystd = np.where(mtr, ystd, 0.0).astype(np.float32)

    dev = torch.device(device)
    sig = batch_signals(tr_examples, specs, n_steps, device=dev)
    y_t = torch.from_numpy(ystd).to(dev)
    m_t = torch.from_numpy(mtr.astype(np.float32)).to(dev)
    n = y_t.shape[0]

    cfg = DiagnosticsProbeConfig(
        streams=list(specs), n_steps=n_steps, target_dim=target_dim
    )
    model = DiagnosticsEquilibriumProbe(cfg).to(dev)
    logger.info(
        "probe params: %.2fM  streams=%s",
        model.n_parameters() / 1e6,
        [s.name for s in specs],
    )
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    model.train()
    g = torch.Generator(device="cpu").manual_seed(seed)
    for ep in range(epochs):
        perm = torch.randperm(n, generator=g)
        tot, nb = 0.0, 0
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size].to(dev)
            sb = {k: v[idx] for k, v in sig.items()}
            yb = y_t[idx]
            mb = m_t[idx]
            opt.zero_grad()
            with torch.autocast(
                device_type=dev.type, dtype=torch.bfloat16, enabled=(dev.type == "cuda")
            ):
                pmean, plog = model(sb)
            loss = gaussian_nll(pmean.float(), plog.float(), yb, mb)
            loss.backward()
            opt.step()
            tot += float(loss.detach())
            nb += 1
        logger.info("epoch %d/%d  NLL=%.4f", ep + 1, epochs, tot / max(nb, 1))
    return model, mean, std


def evaluate(model, te_examples, specs, mean, std, *, n_steps, device, batch_size):
    """Predict on TEST -> (pred (n,12) metres, y (n,12), mask (n,12))."""
    import torch

    dev = torch.device(device)
    sig = batch_signals(te_examples, specs, n_steps, device=dev)
    yte = np.stack([ex["target"] for ex in te_examples]).astype(np.float32)
    mte = np.stack([ex["mask"] for ex in te_examples]).astype(bool)
    n = yte.shape[0]
    model.eval()
    preds = []
    for i in range(0, n, batch_size):
        sb = {k: v[i : i + batch_size] for k, v in sig.items()}
        pmean_m, _ = model.predict_metres(sb, mean, std)
        preds.append(pmean_m)
    pred = np.concatenate(preds, axis=0)
    return pred, yte, mte


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
    """Baseline RMSE: predict the TRAIN mean for every TEST example."""
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
# Verdict
# ---------------------------------------------------------------------------


def verdict(rmse_probe, rmse_baseline, names, *, ratio_threshold):
    """Feasibility verdict: probe RMSE materially below the mean-predictor.

    PASS: for axis + X-point components, probe RMSE < baseline / ratio_threshold.
    Reports per-component skill = 1 - probe/baseline.
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
# Scatter figure (axis + X-point)
# ---------------------------------------------------------------------------


def geometry_scatter(pred, y, mask, names, out_path, *, title):
    """Pred-vs-true scatter for axis_R, axis_Z, xpt_R, xpt_Z."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    comps = [(0, "axis_R"), (1, "axis_Z"), (2, "xpt_R"), (3, "xpt_Z")]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), constrained_layout=True)
    for ax, (d, label) in zip(axes, comps, strict=True):
        sel = mask[:, d]
        if sel.sum() == 0:
            ax.set_title(f"{label}: no finite test labels")
            continue
        yt = y[sel, d]
        yp = pred[sel, d]
        lo = float(min(yt.min(), yp.min()))
        hi = float(max(yt.max(), yp.max()))
        ax.scatter(yt, yp, s=14, alpha=0.55, edgecolors="none")
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="ideal")
        rmse = float(np.sqrt(np.mean((yp - yt) ** 2)))
        ax.set_xlabel(f"true {label} (m)")
        ax.set_ylabel(f"predicted {label} (m)")
        ax.set_title(f"{label}  RMSE={rmse * 100:.1f} cm  (n={int(sel.sum())})")
        ax.legend(fontsize=8)
    fig.suptitle(title, fontsize=12)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("wrote %s", out_path)


# ---------------------------------------------------------------------------
# One arm = (train probe on a stream subset, eval, metrics)
# ---------------------------------------------------------------------------


def run_arm(tr_examples, te_examples, channels, modalities, args, *, restrict, label):
    """Train + evaluate one ablation arm; returns (report_dict, pred, yte, mte)."""
    from imas_ambix.worldmodel.equilibrium_labels import TARGET_DIM, TARGET_NAMES

    specs = build_stream_specs(channels, modalities, restrict=restrict)
    if not specs:
        logger.warning("arm '%s': no streams present — skipping", label)
        return None
    device = "cuda" if _cuda_available() else "cpu"
    model, tmean, tstd = train_probe(
        tr_examples,
        specs,
        n_steps=args.n_signal_steps,
        target_dim=TARGET_DIM,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        seed=args.seed,
    )
    pred, yte, mte = evaluate(
        model,
        te_examples,
        specs,
        tmean,
        tstd,
        n_steps=args.n_signal_steps,
        device=device,
        batch_size=args.batch_size,
    )
    ytr = np.stack([ex["target"] for ex in tr_examples]).astype(np.float32)
    mtr = np.stack([ex["mask"] for ex in tr_examples]).astype(bool)
    rmse_probe = per_component_rmse(pred, yte, mte)
    rmse_base = mean_predictor_rmse(ytr, mtr, yte, mte)
    verd = verdict(
        rmse_probe, rmse_base, TARGET_NAMES, ratio_threshold=args.ratio_threshold
    )
    report = {
        "arm": label,
        "streams": [s.name for s in specs],
        "stream_channels": {s.name: s.channels for s in specs},
        "probe_params_M": None,
        "verdict": verd,
    }
    # console summary
    logger.info(
        "=== ARM '%s' VERDICT: %s ===",
        label,
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
    return report, pred, yte, mte


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(args) -> int:
    from imas_ambix.camdyn.dataset import list_token_shot_ids
    from imas_ambix.camdyn.splits import build_camdyn_split
    from imas_ambix.worldmodel.equilibrium_labels import TARGET_NAMES
    from imas_ambix.worldmodel.spacetime_dataset import SpacetimeWindowConfig
    from imas_ambix.worldmodel.spacetime_dataset_v2 import extended_signal_modalities

    rng = np.random.default_rng(args.seed)
    token_root = Path(args.token_root) if args.token_root else None
    level2_root = Path(args.level2_root) if args.level2_root else None
    modalities = extended_signal_modalities()

    config = SpacetimeWindowConfig(
        n_frames=args.n_frames,
        n_plan=8,
        context_frames=max(1, args.n_frames // 3),
        target_horizon_s=args.target_horizon_s,
    )

    # 1) split — force the gate cohort + held-out into the oracle TEST set.
    all_shots = list_token_shot_ids(camera=args.camera, token_root=token_root)
    logger.info("rbb token shots on disk: %d", len(all_shots))
    split = build_camdyn_split(
        all_shots,
        mse_heldout=list(FORCED_TEST_SHOTS),
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
    forced_present = [s for s in FORCED_TEST_SHOTS if s in set(all_shots)]
    test_shots = sorted(set(test_shots) | set(forced_present))
    logger.info(
        "TRAIN shots=%d  TEST shots=%d (forced present: %s)",
        len(train_shots),
        len(test_shots),
        forced_present,
    )

    # 2-4) assemble examples (window span -> measured signals -> 12-D labels)
    tr_examples = assemble_examples(
        train_shots,
        camera=args.camera,
        modalities=modalities,
        n_signal_steps=args.n_signal_steps,
        config=config,
        level2_root=level2_root,
        token_root=token_root,
    )
    te_examples = assemble_examples(
        test_shots,
        camera=args.camera,
        modalities=modalities,
        n_signal_steps=args.n_signal_steps,
        config=config,
        level2_root=level2_root,
        token_root=token_root,
    )
    if not tr_examples or not te_examples:
        logger.error("empty TRAIN or TEST example set — cannot run oracle")
        return 2
    logger.info(
        "TRAIN examples=%d  TEST examples=%d", len(tr_examples), len(te_examples)
    )

    channels = probe_channels(tr_examples + te_examples, modalities)
    logger.info("probed channels: %s", {k: v for k, v in channels.items() if v > 0})

    # 5-6) two arms: all-diagnostics, then magnetics-only ablation.
    all_arm = run_arm(
        tr_examples,
        te_examples,
        channels,
        modalities,
        args,
        restrict=None,
        label="all_diagnostics",
    )
    mag_arm = run_arm(
        tr_examples,
        te_examples,
        channels,
        modalities,
        args,
        restrict=set(MAGNETICS_STREAMS),
        label="magnetics_only",
    )
    if all_arm is None:
        logger.error("all-diagnostics arm produced no streams — abort")
        return 3

    all_report, all_pred, all_yte, all_mte = all_arm

    coverage = {
        "train_examples": len(tr_examples),
        "test_examples": len(te_examples),
        "train_shots": [int(s) for s in train_shots],
        "test_shots": [int(s) for s in test_shots],
        "forced_test_present": [int(s) for s in forced_present],
        "probed_channels": {k: int(v) for k, v in channels.items() if v > 0},
        "n_signal_steps": args.n_signal_steps,
        "target_horizon_s": args.target_horizon_s,
    }
    report = {
        "task": "diagnostics feasibility oracle (measured signals -> geometry)",
        "evaluator_only": True,
        "camera": args.camera,
        "n_frames": args.n_frames,
        "n_signal_steps": args.n_signal_steps,
        "epochs": args.epochs,
        "target_names": list(TARGET_NAMES),
        "target_units": "m",
        "coverage": coverage,
        "arms": {
            "all_diagnostics": all_report[0],
            "magnetics_only": (mag_arm[0] if mag_arm is not None else None),
        },
        # top-level verdict mirrors the all-diagnostics arm (the headline check)
        "verdict": all_report[0]["verdict"],
    }

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    json_path = out_root / "oracle_report.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("wrote %s", json_path)

    # scatter for the headline all-diagnostics arm.
    title = (
        "diagnostics feasibility oracle — measured signals -> plasma geometry "
        "(all-diagnostics arm)"
    )
    fig_local = out_root / "fig-diag-eq-oracle-geometry-scatter.png"
    geometry_scatter(all_pred, all_yte, all_mte, TARGET_NAMES, fig_local, title=title)
    fig_docs = Path(args.fig_dir) / "fig-diag-eq-oracle-geometry-scatter.png"
    try:
        geometry_scatter(
            all_pred, all_yte, all_mte, TARGET_NAMES, fig_docs, title=title
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not write docs figure %s: %s", fig_docs, exc)

    logger.info(
        "=== TOP-LEVEL (all-diagnostics) FEASIBILITY: %s ===",
        "FEASIBLE" if report["verdict"]["feasible"] else "INFEASIBLE",
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
    p.add_argument("--n-frames", type=int, default=24, help="camera frames per window")
    p.add_argument(
        "--target-horizon-s",
        type=float,
        default=0.25,
        help="physical time span a window covers (s)",
    )
    p.add_argument(
        "--n-signal-steps",
        type=int,
        default=12,
        help="measured-signal temporal positions across the window span",
    )
    p.add_argument("--n-train-shots", type=int, default=250)
    p.add_argument("--n-test-shots", type=int, default=60)
    p.add_argument("--held-out-fraction", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=40)
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
