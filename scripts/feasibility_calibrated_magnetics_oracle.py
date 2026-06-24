#!/usr/bin/env python
"""Confirmation oracle: does RAW / CALIBRATED magnetics resolve plasma geometry?

The decisive root-cause isolation for the diagnostics->equilibrium gate.  The
tokenised-diagnostics oracle
(:mod:`scripts.feasibility_diagnostics_equilibrium_oracle`) found that the
magnetics-conditioned probe barely beat the shot-to-shot baseline on the
INTERIOR geometry (axis_R skill ~+0.22, X-point ~0).  But the magnetics tokens it
read were produced by the world-model's per-channel z-score quantiser
(:func:`imas_ambix.worldmodel.spacetime_dataset_v2._quantise_l2`), which
standardises EACH channel against its OWN per-shot finite mean/std before binning
to 256 levels.  That step throws away the ABSOLUTE inter-shot field magnitude — and
the absolute flux-loop / B-probe / plasma-current scale is precisely what an
EFIT-class reconstruction integrates to place the magnetic axis and the X-point.

This oracle MIRRORS the tokenised one (same forced split, same per-component
RMSE-vs-mean-predictor skill, same JSON / verdict format) but changes the INPUT
to RAW, CALIBRATED magnetics:

  * read the PRE-quantised FLOAT channels straight from the staged magnetics zarr
    store (``flux_loop_flux``, the ccbv / obr / obv poloidal B-field probe arrays,
    and the plasma current ``ip``) — physical units, no per-shot quantisation;
  * standardise with a SINGLE CORPUS-LEVEL mean / std PER CHANNEL, fit on the
    TRAIN shots and applied to every shot, so the ABSOLUTE inter-shot field
    magnitude is PRESERVED (no per-shot / per-window re-normalisation — the whole
    point of the experiment);
  * resample each channel onto the camera-frame grid
    (:func:`imas_ambix.statespace.align.align_chord2d_to_grid`);
  * train a thin temporal probe (continuous-float input -> Gaussian head on the
    12-D geometry target, reusing
    :func:`imas_ambix.worldmodel.diagnostics_equilibrium_probe.gaussian_nll`).

VERDICT
-------
If the calibrated-magnetics probe resolves axis + X-point materially BETTER than
the tokenised version (skill clearly > 0, beating the mean-predictor), the
physics IS recoverable from absolute signals and the WM's ``_quantise_l2``
tokenisation is the specific blocker -> a re-tokenise + retrain is warranted.  If
raw calibrated magnetics ALSO fail, the physics gate is broadly closed for this
corpus / window and tokenisation is exonerated.

EVALUATOR-ONLY (binding firewall)
---------------------------------
A third-party EVALUATOR.  The probe input is raw measured magnetics; the LABEL
is the L2 equilibrium.  Nothing here is, or is importable by, the world-model
training path — it only consumes data and produces evaluator metrics.  No WM
checkpoint is loaded.  Equilibrium is an evaluator label only.

Outputs (JSON + a pred-vs-true axis / X-point scatter) under
``/work/projects/imas_gpu/worldmodel/calibrated_magnetics_oracle/`` and
``docs/figures/joint-multimodal-plasma-wm/``.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger("calib_mag_oracle")

# --- Output locations -------------------------------------------------------

DEFAULT_OUT_ROOT = Path(
    "/work/projects/imas_gpu/worldmodel/calibrated_magnetics_oracle"
)
DEFAULT_FIG_DIR = Path("docs/figures/joint-multimodal-plasma-wm")

# --- Cohorts that MUST be in the oracle TEST set (never trained on) ----------
#: The controllability gate cohort + the standing held-out shots — forced into
#: the oracle TEST partition so the verdict is read on the SAME held-out plasma
#: the downstream gate is scored on (identical to the tokenised oracle).
GATE_COHORT = (15089, 15223, 15517, 15963, 15972, 16024, 16223)
STANDING_HELD_OUT = (18502, 18503, 18504, 18505)
FORCED_TEST_SHOTS = tuple(sorted(set(GATE_COHORT) | set(STANDING_HELD_OUT)))

#: Staged-magnetics raw-float store group + channel keys, in a DETERMINISTIC
#: column order (flux loops, then the three poloidal B-field probe arrays, then
#: the plasma current).  Each is a physical-unit float trace on the store's
#: ``time`` axis: ``flux_loop_flux`` (Wb), the ``b_field_pol_probe_*`` arrays (T),
#: and ``ip`` (A).  Together they are the EFIT-class position / shape sensor at
#: their ABSOLUTE calibrated scale.
MAGNETICS_GROUP = "magnetics"
RAW_CHANNEL_KEYS = (
    "flux_loop_flux",
    "b_field_pol_probe_ccbv_field",
    "b_field_pol_probe_obr_field",
    "b_field_pol_probe_obv_field",
    "ip",
)


# ---------------------------------------------------------------------------
# Raw calibrated-magnetics read (PRE-quantisation floats)
# ---------------------------------------------------------------------------


def _magnetics_store_path(shot_id: int, *, token_root: Path | None = None) -> Path:
    """Resolve the staged raw-float magnetics zarr store for one shot.

    Reuses the staged-store path convention (and the eval-only boundary guard)
    from :func:`imas_ambix.worldmodel.spacetime_dataset_v2._staged_store_path` so
    the layout and the target-root refusal are identical to the WM read path —
    but the caller pulls the PRE-quantised float arrays from it, not the tokens.
    """
    from imas_ambix.worldmodel.spacetime_dataset_v2 import _staged_store_path

    return _staged_store_path(MAGNETICS_GROUP, shot_id, token_root=token_root)


def read_raw_magnetics(
    shot_id: int, *, token_root: Path | None = None
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Read the RAW CALIBRATED magnetics floats for a shot (NO quantisation).

    Returns ``(values (T, C) float64 physical units, time (T,) float64 s,
    channel_names)``.  Channels are concatenated in :data:`RAW_CHANNEL_KEYS`
    order, expanding each 2-D ``(T, n)`` array to ``n`` named columns and each
    1-D trace to a single column.  A key absent from the store is silently
    skipped (the column layout stays deterministic over the present keys).

    Raises ``FileNotFoundError`` / ``KeyError`` when the store or its ``time``
    axis is missing, so the caller drops the shot exactly like the other reads.
    """
    import zarr  # noqa: PLC0415

    path = _magnetics_store_path(shot_id, token_root=token_root)
    store = zarr.open_group(str(path), mode="r")  # raises if absent
    if "time" not in store:
        raise KeyError(f"magnetics store {path} has no 'time' axis")
    time = np.asarray(store["time"], dtype=np.float64).reshape(-1)
    n_t = time.shape[0]
    present = set(store.array_keys())
    cols: list[np.ndarray] = []
    names: list[str] = []
    for key in RAW_CHANNEL_KEYS:
        if key not in present:
            continue
        arr = np.asarray(store[key], dtype=np.float64)
        if arr.ndim == 1:
            if arr.shape[0] != n_t:
                continue
            arr = arr[:, None]
            cols.append(arr)
            names.append(key)
        elif arr.ndim == 2 and arr.shape[0] == n_t:
            cols.append(arr)
            names.extend(f"{key}[{i}]" for i in range(arr.shape[1]))
    if not cols:
        raise KeyError(f"magnetics store {path} has no time-resolved raw channels")
    values = np.concatenate(cols, axis=1)  # (T, C) physical units
    return values, time, names


# ---------------------------------------------------------------------------
# Window + raw-signal + label assembly (one labelled example per shot window)
# ---------------------------------------------------------------------------


def _select_brightest_start(token_path, camera, config):
    """Pick the brightest valid window start for a shot (most plasma-active).

    Mirrors the tokenised oracle: ranks candidate starts by mean raw-frame
    intensity so the probe sees an established plasma, not a dark ramp.  Returns
    an int start frame, or None to fall back to the centred window.
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
    n_signal_steps,
    config,
    level2_root,
    token_root,
):
    """Build one labelled raw-calibrated-magnetics example per shot.

    For each shot: assemble a camera window (brightest start, fall back centred)
    to fix the ~0.25 s time span; read the RAW magnetics floats and resample them
    to ``n_signal_steps`` positions across the span; build the 12-D equilibrium
    labels at those same grid times; keep the WINDOW-CENTRE label as the target.

    Returns a list of dicts ``{shot_id, raw (S, C) float64, target (12,),
    mask (12,)}`` for every shot that yields a window with finite raw magnetics
    and >=1 finite label component at the centre.  The raw blocks are NOT
    standardised here — the corpus-level per-channel stats are fit on TRAIN and
    applied at batching time so the absolute scale is preserved coherently.
    """
    from imas_ambix.camdyn.dataset import discover_token_shots
    from imas_ambix.statespace.align import align_chord2d_to_grid
    from imas_ambix.worldmodel.equilibrium_labels import load_equilibrium_geometry
    from imas_ambix.worldmodel.spacetime_dataset import assemble_window

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
        start = _select_brightest_start(spec.token_path, camera, config)
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

        ftime = np.asarray(sample.frame_time, dtype=np.float64)
        if ftime.size < 2:
            logger.info("shot %d: degenerate window time — skip", sid)
            continue
        t0, t1 = float(ftime.min()), float(ftime.max())
        if not (t1 > t0):
            logger.info("shot %d: zero-span window — skip", sid)
            continue
        grid = np.linspace(t0, t1, int(n_signal_steps), dtype=np.float64)

        # RAW calibrated magnetics floats resampled to the window grid.
        try:
            values, vtime, _names = read_raw_magnetics(int(sid), token_root=token_root)
        except (FileNotFoundError, KeyError) as exc:
            logger.info("shot %d: no raw magnetics (%s) — skip", sid, exc)
            continue
        raw = align_chord2d_to_grid(values, vtime, grid).astype(np.float64)  # (S, C)
        if not np.isfinite(raw).any():
            logger.info("shot %d: raw magnetics all-NaN on grid — skip", sid)
            continue

        # equilibrium labels on the SAME grid (same span as the window).
        try:
            geo = load_equilibrium_geometry(int(sid), grid, level2_root=level2_root)
        except (KeyError, FileNotFoundError) as exc:
            logger.info("shot %d: no equilibrium (%s) — skip", sid, exc)
            continue
        cidx = int(n_signal_steps // 2)
        tgt = geo.target[cidx]  # (12,)
        msk = geo.finite_mask[cidx]  # (12,)
        if not msk.any():
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
                "raw": np.asarray(raw, np.float64),
                "target": np.asarray(tgt, np.float32),
                "mask": np.asarray(msk, bool),
            }
        )
        logger.info(
            "shot %d: raw_mag=(%d,%d)  finite-comp=%d/12",
            sid,
            raw.shape[0],
            raw.shape[1],
            int(msk.sum()),
        )
    return out


# ---------------------------------------------------------------------------
# Corpus-level per-channel standardisation (PRESERVES absolute inter-shot scale)
# ---------------------------------------------------------------------------


def raw_channel_count(examples) -> int:
    """Max raw channel count across the assembled examples (uniform on disk)."""
    return max((int(ex["raw"].shape[1]) for ex in examples), default=0)


def fit_channel_stats(tr_examples, n_channels):
    """Fit ONE corpus-level mean / std per raw channel over ALL TRAIN steps.

    This is the heart of the experiment: a SINGLE per-channel statistic shared by
    every shot — fit on the pooled TRAIN (shot, step) samples — so the ABSOLUTE
    inter-shot field magnitude survives standardisation (a high-current shot stays
    higher than a low-current one).  No per-shot / per-window re-normalisation.
    NaN-safe; a channel with no finite TRAIN data gets mean 0 / std 1.
    """
    sums = np.zeros(n_channels, dtype=np.float64)
    sumsq = np.zeros(n_channels, dtype=np.float64)
    cnt = np.zeros(n_channels, dtype=np.float64)
    for ex in tr_examples:
        r = ex["raw"]
        c = min(n_channels, r.shape[1])
        block = r[:, :c]
        fin = np.isfinite(block)
        b0 = np.where(fin, block, 0.0)
        sums[:c] += b0.sum(axis=0)
        sumsq[:c] += (b0 * b0).sum(axis=0)
        cnt[:c] += fin.sum(axis=0)
    mean = np.zeros(n_channels, dtype=np.float64)
    std = np.ones(n_channels, dtype=np.float64)
    ok = cnt > 1
    mean[ok] = sums[ok] / cnt[ok]
    var = np.zeros(n_channels, dtype=np.float64)
    var[ok] = np.maximum(sumsq[ok] / cnt[ok] - mean[ok] ** 2, 0.0)
    s = np.sqrt(var)
    std[ok] = np.where(s[ok] > 1e-12, s[ok], 1.0)
    return mean, std


def batch_raw(examples, n_channels, ch_mean, ch_std, n_steps, *, device):
    """Stack examples into a standardised ``(N, n_steps, C)`` float tensor + mask.

    Each example's raw block is z-scored by the SHARED corpus-level per-channel
    mean / std (absolute scale preserved), NaNs filled with 0 (the standardised
    channel centre) and tracked by a finite mask so a missing step contributes no
    spurious value.  Returns ``(x (N, S, C) float32, valid (N, S, C) float32)``.
    """
    import torch  # noqa: PLC0415

    n = len(examples)
    x = np.zeros((n, n_steps, n_channels), dtype=np.float32)
    valid = np.zeros((n, n_steps, n_channels), dtype=np.float32)
    inv = 1.0 / ch_std
    for i, ex in enumerate(examples):
        r = ex["raw"]
        s = min(n_steps, r.shape[0])
        c = min(n_channels, r.shape[1])
        block = r[:s, :c]
        fin = np.isfinite(block)
        z = (np.where(fin, block, ch_mean[:c]) - ch_mean[:c]) * inv[:c]
        x[i, :s, :c] = z.astype(np.float32)
        valid[i, :s, :c] = fin.astype(np.float32)
    return (
        torch.from_numpy(x).to(device),
        torch.from_numpy(valid).to(device),
    )


# ---------------------------------------------------------------------------
# Thin continuous-float temporal probe -> Gaussian head on the 12-D target
# ---------------------------------------------------------------------------


def build_probe(
    n_channels, n_steps, target_dim, *, d_model, n_layers, n_heads, dropout
):
    """A small temporal Transformer over the standardised raw-magnetics channels.

    Mirrors the token probe's shape (per-step linear-in -> learned positional ->
    Transformer encoder -> time mean-pool -> Gaussian head over the standardised
    target), but the INPUT is the continuous z-scored magnetics block (with its
    finite mask appended per channel) rather than embedded token ids.  Returns an
    ``nn.Module`` whose ``forward(x, valid)`` gives ``(mean, log_sigma)`` in
    standardised target space.
    """
    import torch  # noqa: PLC0415
    from torch import nn  # noqa: PLC0415

    from imas_ambix.worldmodel.diagnostics_equilibrium_probe import (
        LOG_SIGMA_MAX,
        LOG_SIGMA_MIN,
    )

    class RawMagneticsProbe(nn.Module):
        def __init__(self):
            super().__init__()
            # value channels + their finite mask = 2*C per step.
            self.in_proj = nn.Linear(2 * n_channels, d_model)
            self.pos = nn.Parameter(torch.zeros(1, n_steps, d_model))
            nn.init.trunc_normal_(self.pos, std=0.02)
            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
            self.pool_norm = nn.LayerNorm(d_model)
            self.head = nn.Sequential(
                nn.Linear(d_model, 256),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(256, 2 * target_dim),
            )

        def forward(self, x, valid):
            feat = torch.cat([x, valid], dim=-1)  # (B, S, 2C)
            h = self.in_proj(feat) + self.pos
            h = self.encoder(h)
            h = self.pool_norm(h.mean(dim=1))
            out = self.head(h)
            mean, log_sigma = out.chunk(2, dim=-1)
            log_sigma = torch.clamp(log_sigma, LOG_SIGMA_MIN, LOG_SIGMA_MAX)
            return mean, log_sigma

        def n_parameters(self):
            return int(sum(p.numel() for p in self.parameters()))

    return RawMagneticsProbe()


# ---------------------------------------------------------------------------
# Standardisation + train + eval (target side mirrors the tokenised oracle)
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
    tr_examples,
    n_channels,
    ch_mean,
    ch_std,
    *,
    n_steps,
    target_dim,
    epochs,
    batch_size,
    lr,
    device,
    seed,
    d_model,
    n_layers,
    n_heads,
    dropout,
):
    """Train the heteroscedastic raw-magnetics probe.

    Returns ``(model, target_mean, target_std)`` — the model plus the TRAIN-split
    per-component target standardisation used to map predictions back to metres.
    """
    import torch  # noqa: PLC0415

    from imas_ambix.worldmodel.diagnostics_equilibrium_probe import gaussian_nll

    torch.manual_seed(seed)
    ytr = np.stack([ex["target"] for ex in tr_examples]).astype(np.float32)
    mtr = np.stack([ex["mask"] for ex in tr_examples]).astype(bool)
    tmean, tstd = standardise_stats(ytr, mtr)
    ystd = (np.nan_to_num(ytr, nan=0.0) - tmean) / tstd
    ystd = np.where(mtr, ystd, 0.0).astype(np.float32)

    dev = torch.device(device)
    x_t, v_t = batch_raw(tr_examples, n_channels, ch_mean, ch_std, n_steps, device=dev)
    y_t = torch.from_numpy(ystd).to(dev)
    m_t = torch.from_numpy(mtr.astype(np.float32)).to(dev)
    n = y_t.shape[0]

    model = build_probe(
        n_channels,
        n_steps,
        target_dim,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        dropout=dropout,
    ).to(dev)
    logger.info(
        "probe params: %.2fM  raw_channels=%d  n_steps=%d",
        model.n_parameters() / 1e6,
        n_channels,
        n_steps,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    model.train()
    g = torch.Generator(device="cpu").manual_seed(seed)
    for ep in range(epochs):
        perm = torch.randperm(n, generator=g)
        tot, nb = 0.0, 0
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size].to(dev)
            xb = x_t[idx]
            vb = v_t[idx]
            yb = y_t[idx]
            mb = m_t[idx]
            opt.zero_grad()
            with torch.autocast(
                device_type=dev.type,
                dtype=torch.bfloat16,
                enabled=(dev.type == "cuda"),
            ):
                pmean, plog = model(xb, vb)
            loss = gaussian_nll(pmean.float(), plog.float(), yb, mb)
            loss.backward()
            opt.step()
            tot += float(loss.detach())
            nb += 1
        logger.info("epoch %d/%d  NLL=%.4f", ep + 1, epochs, tot / max(nb, 1))
    return model, tmean, tstd


def evaluate(
    model,
    te_examples,
    n_channels,
    ch_mean,
    ch_std,
    tmean,
    tstd,
    *,
    n_steps,
    device,
    batch_size,
):
    """Predict on TEST -> ``(pred (n,12) metres, y (n,12), mask (n,12))``."""
    import torch  # noqa: PLC0415

    dev = torch.device(device)
    x_t, v_t = batch_raw(te_examples, n_channels, ch_mean, ch_std, n_steps, device=dev)
    yte = np.stack([ex["target"] for ex in te_examples]).astype(np.float32)
    mte = np.stack([ex["mask"] for ex in te_examples]).astype(bool)
    n = yte.shape[0]
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, n, batch_size):
            xb = x_t[i : i + batch_size]
            vb = v_t[i : i + batch_size]
            pmean, _ = model(xb, vb)
            mu = pmean.detach().cpu().float().numpy()
            preds.append(mu * tstd + tmean)
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
# Verdict (mirrors the tokenised oracle exactly)
# ---------------------------------------------------------------------------


def verdict(rmse_probe, rmse_baseline, names, *, ratio_threshold):
    """Feasibility verdict: probe RMSE materially below the mean-predictor.

    PASS: for axis + X-point components, probe RMSE < baseline / ratio_threshold.
    Reports per-component skill = 1 - probe / baseline.
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
            "axis_R, axis_Z, xpt_R, xpt_Z (calibrated magnetics resolve "
            "geometry beyond shot-to-shot spread)"
        ),
        "ratio_threshold": ratio_threshold,
        "components": rows,
    }


# ---------------------------------------------------------------------------
# Scatter figure (axis + X-point) — mirrors the tokenised oracle
# ---------------------------------------------------------------------------


def geometry_scatter(pred, y, mask, names, out_path, *, title):
    """Pred-vs-true scatter for axis_R, axis_Z, xpt_R, xpt_Z."""
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

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
# Main
# ---------------------------------------------------------------------------


def run(args) -> int:
    from imas_ambix.camdyn.dataset import list_token_shot_ids
    from imas_ambix.camdyn.splits import build_camdyn_split
    from imas_ambix.worldmodel.equilibrium_labels import TARGET_DIM, TARGET_NAMES
    from imas_ambix.worldmodel.spacetime_dataset import SpacetimeWindowConfig

    rng = np.random.default_rng(args.seed)
    token_root = Path(args.token_root) if args.token_root else None
    level2_root = Path(args.level2_root) if args.level2_root else None

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

    # 2-4) assemble examples (window span -> raw magnetics floats -> 12-D labels)
    tr_examples = assemble_examples(
        train_shots,
        camera=args.camera,
        n_signal_steps=args.n_signal_steps,
        config=config,
        level2_root=level2_root,
        token_root=token_root,
    )
    te_examples = assemble_examples(
        test_shots,
        camera=args.camera,
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

    # 5) corpus-level per-channel standardisation fit on TRAIN ONLY.
    n_channels = max(raw_channel_count(tr_examples), raw_channel_count(te_examples))
    ch_mean, ch_std = fit_channel_stats(tr_examples, n_channels)
    logger.info(
        "raw magnetics channels=%d  (corpus-level per-channel standardisation, "
        "absolute scale preserved)",
        n_channels,
    )

    device = "cuda" if _cuda_available() else "cpu"
    logger.info("device=%s", device)

    model, tmean, tstd = train_probe(
        tr_examples,
        n_channels,
        ch_mean,
        ch_std,
        n_steps=args.n_signal_steps,
        target_dim=TARGET_DIM,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        seed=args.seed,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
    )
    pred, yte, mte = evaluate(
        model,
        te_examples,
        n_channels,
        ch_mean,
        ch_std,
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

    # console summary
    logger.info(
        "=== CALIBRATED-MAGNETICS VERDICT: %s ===",
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

    coverage = {
        "train_examples": len(tr_examples),
        "test_examples": len(te_examples),
        "train_shots": [int(s) for s in train_shots],
        "test_shots": [int(s) for s in test_shots],
        "forced_test_present": [int(s) for s in forced_present],
        "raw_channels": int(n_channels),
        "raw_channel_keys": list(RAW_CHANNEL_KEYS),
        "n_signal_steps": args.n_signal_steps,
        "target_horizon_s": args.target_horizon_s,
    }
    report = {
        "task": (
            "calibrated-magnetics feasibility oracle (RAW physical-unit "
            "magnetics -> plasma geometry; corpus-level standardisation, "
            "absolute scale preserved)"
        ),
        "evaluator_only": True,
        "input": "raw_calibrated_magnetics_floats",
        "standardisation": "single corpus-level mean/std per channel (TRAIN-fit)",
        "camera": args.camera,
        "n_frames": args.n_frames,
        "n_signal_steps": args.n_signal_steps,
        "epochs": args.epochs,
        "target_names": list(TARGET_NAMES),
        "target_units": "m",
        "coverage": coverage,
        "probe_params_M": model.n_parameters() / 1e6,
        "verdict": verd,
    }

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    json_path = out_root / "oracle_report.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("wrote %s", json_path)

    title = (
        "calibrated-magnetics feasibility oracle — RAW physical-unit magnetics "
        "-> plasma geometry"
    )
    fig_local = out_root / "fig-calib-mag-oracle-geometry-scatter.png"
    geometry_scatter(pred, yte, mte, TARGET_NAMES, fig_local, title=title)
    fig_docs = Path(args.fig_dir) / "fig-calib-mag-oracle-geometry-scatter.png"
    try:
        geometry_scatter(pred, yte, mte, TARGET_NAMES, fig_docs, title=title)
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not write docs figure %s: %s", fig_docs, exc)

    logger.info(
        "=== TOP-LEVEL (calibrated-magnetics) FEASIBILITY: %s ===",
        "FEASIBLE" if report["verdict"]["feasible"] else "INFEASIBLE",
    )
    return 0


def _cuda_available() -> bool:
    try:
        import torch  # noqa: PLC0415

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
        help="raw-magnetics temporal positions across the window span",
    )
    p.add_argument("--n-train-shots", type=int, default=250)
    p.add_argument("--n-test-shots", type=int, default=60)
    p.add_argument("--held-out-fraction", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--d-model", type=int, default=192)
    p.add_argument("--n-layers", type=int, default=3)
    p.add_argument("--n-heads", type=int, default=6)
    p.add_argument("--dropout", type=float, default=0.1)
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
