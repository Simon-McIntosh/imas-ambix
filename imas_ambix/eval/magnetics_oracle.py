"""Absolute-magnetics equilibrium oracle — the MEASURED BAR for the GS readout.

This module is the model-independent EVALUATOR that maps ABSOLUTE (corpus-level,
SI, pre-quantisation) magnetics straight to equilibrium geometry — magnetic axis
(R, Z), X-point (R, Z) and the LCFS boundary descriptors.  It establishes the
*bar* a learned world model's stage-2 Grad-Shafranov readout must match: a thin
temporal probe over raw calibrated flux-loop / B-probe / plasma-current traces,
scored against a mean-predictor baseline.

Why ABSOLUTE / corpus-level standardisation matters
----------------------------------------------------
A per-shot z-score throws away the inter-shot field magnitude — and the absolute
flux-loop / B-probe / plasma-current scale is exactly what an EFIT-class
reconstruction integrates to place the magnetic axis and the X-point.  So every
raw channel is standardised by a SINGLE corpus-level mean / std fit on the TRAIN
shots and applied to every shot (:func:`fit_channel_stats`, :func:`batch_raw`).
A high-current shot stays higher than a low-current one; there is no per-shot or
per-window re-normalisation.

Evaluator firewall (binding)
----------------------------
A third-party evaluator.  The probe INPUT is raw measured magnetics; the LABEL is
the L2 equilibrium geometry.  Nothing here is, or is importable by, the
world-model training path — it only consumes data and produces evaluator metrics.
No world-model checkpoint is loaded.  Equilibrium is an evaluator label only.

Layering
--------
The standardisation, baseline and verdict math (:func:`fit_channel_stats`,
:func:`batch_raw_arrays`, :func:`standardise_target_stats`, :func:`per_component_rmse`,
:func:`mean_predictor_rmse`, :func:`verdict`, :func:`oracle_skill`) are pure NumPy
and import without torch — they are what the unit tests exercise on synthetic
arrays.  The probe construction and training (:func:`build_probe`,
:func:`batch_raw`, :func:`train_probe`, :func:`evaluate`) import torch lazily so
the module loads cheaply in a torch-free environment.  The data-read helpers
(:func:`read_raw_magnetics`, :func:`assemble_examples`) lazily import the corpus
data stack and only run where the staged stores are reachable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger("magnetics_oracle")

# --- Cohorts that MUST be in the oracle TEST set (never trained on) ----------
#: The controllability-gate cohort + the standing held-out shots — forced into
#: the oracle TEST partition so the bar is read on the SAME held-out plasma the
#: downstream gate is scored on.  These shots must NEVER enter the TRAIN split.
GATE_COHORT = (15089, 15223, 15517, 15963, 15972, 16024, 16223)
STANDING_HELD_OUT = (18502, 18503, 18504, 18505)
FORCED_TEST_SHOTS = tuple(sorted(set(GATE_COHORT) | set(STANDING_HELD_OUT)))

#: Geometry components scored for the headline bar (axis + X-point).
AXIS_XPT_COMPONENTS = ("axis_R", "axis_Z", "xpt_R", "xpt_Z")

#: Staged-magnetics raw-float store group + channel keys, in a DETERMINISTIC
#: column order (flux loops, then the three poloidal B-field probe arrays, then
#: the plasma current).  Each is a physical-unit float trace on the store's
#: ``time`` axis: ``flux_loop_flux`` (Wb), the ``b_field_pol_probe_*`` arrays (T),
#: and ``ip`` (A) — the EFIT-class position / shape sensor at absolute scale.
MAGNETICS_GROUP = "magnetics"
RAW_CHANNEL_KEYS = (
    "flux_loop_flux",
    "b_field_pol_probe_ccbv_field",
    "b_field_pol_probe_obr_field",
    "b_field_pol_probe_obv_field",
    "ip",
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OracleConfig:
    """Knobs for the absolute-magnetics oracle (data window + probe shape)."""

    camera: str = "rbb"
    n_frames: int = 24
    target_horizon_s: float = 0.25
    n_signal_steps: int = 12
    n_train_shots: int = 250
    n_test_shots: int = 60
    held_out_fraction: float = 0.1
    epochs: int = 80
    batch_size: int = 64
    lr: float = 3e-4
    d_model: int = 192
    n_layers: int = 3
    n_heads: int = 6
    dropout: float = 0.1
    ratio_threshold: float = 1.3
    seed: int = 42


@dataclass
class ChannelStats:
    """Corpus-level per-channel standardisation (absolute scale preserved)."""

    mean: np.ndarray
    std: np.ndarray
    n_channels: int


@dataclass
class TargetStats:
    """Per-component target standardisation fit on the finite TRAIN labels."""

    mean: np.ndarray
    std: np.ndarray


@dataclass
class Verdict:
    """Per-component skill plus the headline axis+X-point bar."""

    feasible: bool
    ratio_threshold: float
    components: list[dict]
    headline_skill: float | None
    axis_skill: float | None
    xpt_skill: float | None
    criterion: str = field(default="")

    def to_dict(self) -> dict:
        return {
            "feasible": self.feasible,
            "criterion": self.criterion,
            "ratio_threshold": self.ratio_threshold,
            "headline_skill": self.headline_skill,
            "axis_skill": self.axis_skill,
            "xpt_skill": self.xpt_skill,
            "components": self.components,
        }


# ---------------------------------------------------------------------------
# Raw calibrated-magnetics read (PRE-quantisation floats)
# ---------------------------------------------------------------------------


def magnetics_store_path(shot_id: int, *, token_root: Path | None = None) -> Path:
    """Resolve the staged raw-float magnetics zarr store for one shot.

    Reuses the staged-store path convention (and the eval-only boundary guard)
    from the world-model read path so the layout and the target-root refusal are
    identical — but the caller pulls the PRE-quantised float arrays, not tokens.
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
    import zarr

    path = magnetics_store_path(shot_id, token_root=token_root)
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
            cols.append(arr[:, None])
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

    Ranks candidate starts by mean raw-frame intensity so the probe sees an
    established plasma, not a dark ramp.  Returns an int start frame, or None to
    fall back to the centred window.
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
    to ``n_signal_steps`` positions across the span; build the geometry labels at
    those same grid times; keep the WINDOW-CENTRE label as the target.

    Returns a list of dicts ``{shot_id, raw (S, C) float64, target (D,),
    mask (D,)}`` for every shot that yields a window with finite raw magnetics
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
            logger.info("shot %d: no %s tokens — skip", sid, camera)
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
        tgt = geo.target[cidx]
        msk = geo.finite_mask[cidx]
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
            "shot %d: raw_mag=(%d,%d)  finite-comp=%d/%d",
            sid,
            raw.shape[0],
            raw.shape[1],
            int(msk.sum()),
            int(msk.size),
        )
    return out


# ---------------------------------------------------------------------------
# Corpus-level per-channel standardisation (PRESERVES absolute inter-shot scale)
# ---------------------------------------------------------------------------


def raw_channel_count(examples) -> int:
    """Max raw channel count across the assembled examples (uniform on disk)."""
    return max((int(ex["raw"].shape[1]) for ex in examples), default=0)


def fit_channel_stats(tr_examples, n_channels) -> ChannelStats:
    """Fit ONE corpus-level mean / std per raw channel over ALL TRAIN steps.

    The heart of the experiment: a SINGLE per-channel statistic shared by every
    shot — fit on the pooled TRAIN (shot, step) samples — so the ABSOLUTE
    inter-shot field magnitude survives standardisation (a high-current shot
    stays higher than a low-current one).  No per-shot re-normalisation.
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
    return ChannelStats(mean=mean, std=std, n_channels=int(n_channels))


def batch_raw_arrays(examples, stats: ChannelStats, n_steps):
    """Stack examples into a standardised ``(N, n_steps, C)`` array + finite mask.

    Each example's raw block is z-scored by the SHARED corpus-level per-channel
    mean / std (absolute scale preserved), NaNs filled with 0 (the standardised
    channel centre) and tracked by a finite mask so a missing step contributes no
    spurious value.  Returns ``(x (N, S, C) float32, valid (N, S, C) float32)`` —
    pure NumPy, no torch (the torch wrapper is :func:`batch_raw`).
    """
    n = len(examples)
    c_max = stats.n_channels
    x = np.zeros((n, n_steps, c_max), dtype=np.float32)
    valid = np.zeros((n, n_steps, c_max), dtype=np.float32)
    inv = 1.0 / stats.std
    for i, ex in enumerate(examples):
        r = ex["raw"]
        s = min(n_steps, r.shape[0])
        c = min(c_max, r.shape[1])
        block = r[:s, :c]
        fin = np.isfinite(block)
        z = (np.where(fin, block, stats.mean[:c]) - stats.mean[:c]) * inv[:c]
        x[i, :s, :c] = z.astype(np.float32)
        valid[i, :s, :c] = fin.astype(np.float32)
    return x, valid


def batch_raw(examples, stats: ChannelStats, n_steps, *, device):
    """Torch wrapper over :func:`batch_raw_arrays` placing tensors on ``device``."""
    import torch

    x, valid = batch_raw_arrays(examples, stats, n_steps)
    return torch.from_numpy(x).to(device), torch.from_numpy(valid).to(device)


# ---------------------------------------------------------------------------
# Target standardisation (per-component, finite-label TRAIN stats)
# ---------------------------------------------------------------------------


def standardise_target_stats(y, mask) -> TargetStats:
    """Per-component mean / std over the finite TRAIN labels (NaN-safe)."""
    dim = y.shape[1]
    mean = np.zeros(dim)
    std = np.ones(dim)
    for d in range(dim):
        vals = y[mask[:, d], d]
        if vals.size > 1:
            mean[d] = float(np.mean(vals))
            std[d] = float(np.std(vals)) or 1.0
    return TargetStats(mean=mean, std=std)


# ---------------------------------------------------------------------------
# Thin continuous-float temporal probe -> Gaussian head on the target
# ---------------------------------------------------------------------------


def build_probe(
    n_channels, n_steps, target_dim, *, d_model, n_layers, n_heads, dropout
):
    """A small temporal Transformer over the standardised raw-magnetics channels.

    Per-step linear-in (value channels + their finite mask) -> learned positional
    -> Transformer encoder -> time mean-pool -> Gaussian head over the standardised
    target.  Returns an ``nn.Module`` whose ``forward(x, valid)`` gives
    ``(mean, log_sigma)`` in standardised target space.
    """
    import torch
    from torch import nn

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
# Train + eval (target side mirrors the tokenised oracle)
# ---------------------------------------------------------------------------


def train_probe(
    tr_examples,
    stats: ChannelStats,
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

    Returns ``(model, TargetStats)`` — the model plus the TRAIN-split
    per-component target standardisation used to map predictions back to metres.
    """
    import torch

    from imas_ambix.worldmodel.diagnostics_equilibrium_probe import gaussian_nll

    torch.manual_seed(seed)
    ytr = np.stack([ex["target"] for ex in tr_examples]).astype(np.float32)
    mtr = np.stack([ex["mask"] for ex in tr_examples]).astype(bool)
    tstats = standardise_target_stats(ytr, mtr)
    ystd = (np.nan_to_num(ytr, nan=0.0) - tstats.mean) / tstats.std
    ystd = np.where(mtr, ystd, 0.0).astype(np.float32)

    dev = torch.device(device)
    x_t, v_t = batch_raw(tr_examples, stats, n_steps, device=dev)
    y_t = torch.from_numpy(ystd).to(dev)
    m_t = torch.from_numpy(mtr.astype(np.float32)).to(dev)
    n = y_t.shape[0]

    model = build_probe(
        stats.n_channels,
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
        stats.n_channels,
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
    return model, tstats


def evaluate(
    model,
    te_examples,
    stats: ChannelStats,
    tstats: TargetStats,
    *,
    n_steps,
    device,
    batch_size,
):
    """Predict on TEST -> ``(pred (n,D) metres, y (n,D), mask (n,D))``."""
    import torch

    dev = torch.device(device)
    x_t, v_t = batch_raw(te_examples, stats, n_steps, device=dev)
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
            preds.append(mu * tstats.std + tstats.mean)
    pred = np.concatenate(preds, axis=0)
    return pred, yte, mte


# ---------------------------------------------------------------------------
# Skill (probe RMSE vs mean-predictor baseline) + verdict
# ---------------------------------------------------------------------------


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


def oracle_skill(rmse_probe, rmse_baseline, names, components):
    """Mean skill (1 - probe/baseline) over the named components (NaN-safe).

    Returns ``None`` when no named component has a finite skill.
    """
    idx = {nm: d for d, nm in enumerate(names)}
    vals = []
    for nm in components:
        d = idx.get(nm)
        if d is None:
            continue
        rp = rmse_probe[d]
        rb = rmse_baseline[d]
        if rb is None or not np.isfinite(rb) or rb == 0 or not np.isfinite(rp):
            continue
        vals.append(1.0 - rp / rb)
    return float(np.mean(vals)) if vals else None


def verdict(rmse_probe, rmse_baseline, names, *, ratio_threshold) -> Verdict:
    """Feasibility verdict: probe RMSE materially below the mean-predictor.

    PASS: for axis + X-point components, probe RMSE < baseline / ratio_threshold.
    Reports per-component skill = 1 - probe / baseline, plus headline (axis +
    X-point mean), axis-only and X-point-only skills.
    """
    rows = []
    key = set(AXIS_XPT_COMPONENTS)
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
    return Verdict(
        feasible=overall,
        ratio_threshold=ratio_threshold,
        components=rows,
        headline_skill=oracle_skill(
            rmse_probe, rmse_baseline, names, AXIS_XPT_COMPONENTS
        ),
        axis_skill=oracle_skill(rmse_probe, rmse_baseline, names, ("axis_R", "axis_Z")),
        xpt_skill=oracle_skill(rmse_probe, rmse_baseline, names, ("xpt_R", "xpt_Z")),
        criterion=(
            f"probe RMSE < baseline / {ratio_threshold:g} for ALL of "
            f"{', '.join(AXIS_XPT_COMPONENTS)} (calibrated magnetics resolve "
            "geometry beyond shot-to-shot spread)"
        ),
    )


# ---------------------------------------------------------------------------
# Scatter figure (axis + X-point)
# ---------------------------------------------------------------------------


def geometry_scatter(pred, y, mask, names, out_path, *, title):
    """Pred-vs-true scatter for axis_R, axis_Z, xpt_R, xpt_Z."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    idx = {nm: d for d, nm in enumerate(names)}
    comps = [(idx[nm], nm) for nm in AXIS_XPT_COMPONENTS if nm in idx]
    fig, axes = plt.subplots(
        1, len(comps), figsize=(4 * len(comps), 4.2), constrained_layout=True
    )
    if len(comps) == 1:
        axes = [axes]
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
# End-to-end driver (read corpus -> train -> evaluate -> verdict)
# ---------------------------------------------------------------------------


def cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


@dataclass
class OracleResult:
    """Everything a caller needs to report or persist the bar."""

    verdict: Verdict
    rmse_probe: np.ndarray
    rmse_baseline: np.ndarray
    pred: np.ndarray
    y_test: np.ndarray
    mask_test: np.ndarray
    target_names: tuple
    n_channels: int
    train_examples: int
    test_examples: int
    train_shots: list
    test_shots: list
    forced_test_present: list
    probe_params_millions: float


def run_oracle(
    config: OracleConfig, *, token_root=None, level2_root=None
) -> OracleResult:
    """Read the corpus, fit the corpus-level standardisation, train the probe and
    score the held-out bar.  Forces the gate / standing-held-out cohorts into TEST.

    Imports the corpus data stack lazily and only runs where the staged stores are
    reachable; raises if the TRAIN or TEST example set comes back empty.
    """
    from imas_ambix.camdyn.dataset import list_token_shot_ids
    from imas_ambix.camdyn.splits import build_camdyn_split
    from imas_ambix.worldmodel.equilibrium_labels import TARGET_DIM, TARGET_NAMES
    from imas_ambix.worldmodel.spacetime_dataset import SpacetimeWindowConfig

    rng = np.random.default_rng(config.seed)
    win = SpacetimeWindowConfig(
        n_frames=config.n_frames,
        n_plan=8,
        context_frames=max(1, config.n_frames // 3),
        target_horizon_s=config.target_horizon_s,
    )

    all_shots = list_token_shot_ids(camera=config.camera, token_root=token_root)
    logger.info("%s token shots on disk: %d", config.camera, len(all_shots))
    split = build_camdyn_split(
        all_shots,
        mse_heldout=list(FORCED_TEST_SHOTS),
        val_fraction=0.0,
        held_out_fraction=config.held_out_fraction,
        seed=config.seed,
    )
    train_pool = list(split.train)
    test_pool = list(split.held_out)
    rng.shuffle(train_pool)
    rng.shuffle(test_pool)
    train_shots = train_pool[: config.n_train_shots]
    test_shots = test_pool[: config.n_test_shots]
    forced_present = [s for s in FORCED_TEST_SHOTS if s in set(all_shots)]
    test_shots = sorted(set(test_shots) | set(forced_present))
    # belt-and-braces: a forced-test shot must NEVER be in TRAIN.
    train_shots = [s for s in train_shots if s not in set(FORCED_TEST_SHOTS)]
    logger.info(
        "TRAIN shots=%d  TEST shots=%d (forced present: %s)",
        len(train_shots),
        len(test_shots),
        forced_present,
    )

    tr_examples = assemble_examples(
        train_shots,
        camera=config.camera,
        n_signal_steps=config.n_signal_steps,
        config=win,
        level2_root=level2_root,
        token_root=token_root,
    )
    te_examples = assemble_examples(
        test_shots,
        camera=config.camera,
        n_signal_steps=config.n_signal_steps,
        config=win,
        level2_root=level2_root,
        token_root=token_root,
    )
    if not tr_examples or not te_examples:
        raise RuntimeError("empty TRAIN or TEST example set — cannot run oracle")
    logger.info(
        "TRAIN examples=%d  TEST examples=%d", len(tr_examples), len(te_examples)
    )

    n_channels = max(raw_channel_count(tr_examples), raw_channel_count(te_examples))
    stats = fit_channel_stats(tr_examples, n_channels)
    logger.info(
        "raw magnetics channels=%d  (corpus-level per-channel standardisation, "
        "absolute scale preserved)",
        n_channels,
    )

    device = "cuda" if cuda_available() else "cpu"
    logger.info("device=%s", device)

    model, tstats = train_probe(
        tr_examples,
        stats,
        n_steps=config.n_signal_steps,
        target_dim=TARGET_DIM,
        epochs=config.epochs,
        batch_size=config.batch_size,
        lr=config.lr,
        device=device,
        seed=config.seed,
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        dropout=config.dropout,
    )
    pred, yte, mte = evaluate(
        model,
        te_examples,
        stats,
        tstats,
        n_steps=config.n_signal_steps,
        device=device,
        batch_size=config.batch_size,
    )

    ytr = np.stack([ex["target"] for ex in tr_examples]).astype(np.float32)
    mtr = np.stack([ex["mask"] for ex in tr_examples]).astype(bool)
    rmse_probe = per_component_rmse(pred, yte, mte)
    rmse_base = mean_predictor_rmse(ytr, mtr, yte, mte)
    verd = verdict(
        rmse_probe, rmse_base, TARGET_NAMES, ratio_threshold=config.ratio_threshold
    )

    return OracleResult(
        verdict=verd,
        rmse_probe=rmse_probe,
        rmse_baseline=rmse_base,
        pred=pred,
        y_test=yte,
        mask_test=mte,
        target_names=tuple(TARGET_NAMES),
        n_channels=int(n_channels),
        train_examples=len(tr_examples),
        test_examples=len(te_examples),
        train_shots=[int(s) for s in train_shots],
        test_shots=[int(s) for s in test_shots],
        forced_test_present=[int(s) for s in forced_present],
        probe_params_millions=model.n_parameters() / 1e6,
    )
