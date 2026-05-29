"""Static cross-family baseline: deep ensemble + split-conformal calibration.

Implements the S7.2 static comparator for the plasma-state-space-v0 pipeline.

Architecture
------------
- Deep ensemble of M=5 independent MLPs (different random seeds), each with
  a Gaussian predictive head (μ and log σ per target channel), trained with NLL.
- Epistemic uncertainty = inter-member variance; aleatoric = per-member σ.
- Split-conformal wrapper for marginal 90% coverage guarantee.

Inputs (v0)
-----------
  - magnetics: ama + amb + amc channels  (NO efm/esm; NO xdc)
  - interferometer: ane (density only)
  EXCLUDED: all dalpha families (xim, ada, aim) — leakage-audited hold-out.
  EXCLUDED: efm, esm, xdc per task spec.
  Incremental lift: report mag-only → mag+ane.

Target
------
  Primary:  xim/da_hm10_t  (midplane Dα filterscope, raw)
  Extended: multi-chord xim Dα array (da_bo10, da_hl11_*, da_hm10_*, da_hu10_*, da_to10)

Temporal model
--------------
  STRICTLY instantaneous (per-slice).  Features at time t → target at time t.
  No context window.  model_hz=1000 preserves ELM structure (avoids aliasing).

4-way split discipline (all splits are on SHOT IDs — never on slices)
----------------------------------------------------------------------
  (1) TRAIN       - 10,846 shots from statespace_splits_dalpha_v0.json
  (2) CONFORMAL-CAL  - first ~740 calibration shots (seed=42 sub-split)
  (3) IN-DIST-TEST   - remaining ~739 calibration shots (disjoint from above)
  (4) OOD-REGIME-TEST - 265 shots from ood_regime split

Conformal mechanics
-------------------
  Normalized residual score: s = |y − μ| / σ_total
  where σ_total = sqrt(σ_aleatoric² + σ_epistemic²)
  Quantile with finite-sample correction:
    level = ceil((n+1) * 0.90) / n
    q̂ = np.quantile(scores, level, method="higher")
  Coverage-achieving interval: μ ± q̂ · σ_total
  For NLL/CRPS, use RAW ensemble σ (not conformal-rescaled).
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — fixed feature schema
# ---------------------------------------------------------------------------

# Channels present in ≥90% of shots (empirically verified on 300-shot sample).
# Order is fixed — must be deterministic for feature vector consistency.
_AMA_CHANNELS: list[str] = [
    "n=2_amplitude",
    "n=2_frequency",
    "n=2_signal",
    "n=odd_amplitude",
    "n=odd_frequency",
    "n=odd_signal",
]

# All 73 amb channels present in >90% of shots (empirically verified on 300-shot sample).
_AMB_CHANNELS: list[str] = [
    "ccbv03", "ccbv04", "ccbv05", "ccbv07", "ccbv08", "ccbv09",
    "ccbv11", "ccbv12", "ccbv14", "ccbv15", "ccbv16", "ccbv17",
    "ccbv18", "ccbv19", "ccbv20", "ccbv21", "ccbv24", "ccbv25",
    "ccbv26", "ccbv27", "ccbv28", "ccbv29", "ccbv30", "ccbv31",
    "ccbv32", "ccbv33", "ccbv34", "ccbv36", "ccbv37", "ccbv38",
    "ccbv39", "ccbv40",
    "fl_cc03", "fl_cc04", "fl_cc05", "fl_cc07", "fl_cc09",
    "fl_p2l_1", "fl_p3u_1", "fl_p3u_4",
    "fl_p4l_1", "fl_p4l_4", "fl_p4u_4",
    "fl_p5l_1", "fl_p5l_4", "fl_p5u_1",
    "obr01", "obr04", "obr05", "obr06", "obr08",
    "obr09", "obr10", "obr12", "obr13", "obr14",
    "obr15", "obr16", "obr17", "obr18",
    "obv03", "obv04", "obv05", "obv06", "obv08",
    "obv09", "obv11", "obv13", "obv14", "obv16",
    "obv17", "obv18", "obv19",
]

# amc channels (excludes: time, passnumber, status, timesec, error_field_{a,b})
_AMC_CHANNELS: list[str] = [
    "efps_current",
    "p2il_coil_current", "p2il_feed_current",
    "p2iu_coil_current", "p2iu_feed_current",
    "p2l_case_current", "p2l_current",
    "p2ol_coil_current", "p2ol_feed_current",
    "p2ou_coil_current", "p2ou_feed_current",
    "p2u_case_current", "p2u_current",
    "p3l_case_current", "p3l_coil_current", "p3l_current", "p3l_feed_current",
    "p3u_case_current", "p3u_coil_current", "p3u_current", "p3u_feed_current",
    "p4l_case_current", "p4l_coil_current", "p4l_current", "p4l_feed_current",
    "p4u_case_current", "p4u_coil_current", "p4u_current", "p4u_feed_current",
    "p5l_case_current", "p5l_coil_current", "p5l_current", "p5l_feed_current",
    "p5u_case_current", "p5u_coil_current", "p5u_current", "p5u_feed_current",
    "p6l_current", "p6u_current",
    "plasma_current", "sol_current", "tf_current",
]

# ane: single density channel (line-integrated, m^-2)
_ANE_CHANNELS: list[str] = ["density"]

# Stable multi-chord target channels (≥95% coverage across shots)
_XIM_CHANNELS_PRIMARY: list[str] = ["da_hm10_t"]  # headline target
_XIM_CHANNELS_MULTI: list[str] = [
    "da_bo10",
    "da_hm10_r",
    "da_hm10_t",
    "da_hu10_t",
    "da_to10",
]  # subset with ≥99% coverage; excludes suffix-varying channels

# Input feature group schemas
_FEATURE_SCHEMA_MAG: dict[str, list[str]] = {
    "ama": _AMA_CHANNELS,
    "amb": _AMB_CHANNELS,
    "amc": _AMC_CHANNELS,
}
_FEATURE_SCHEMA_MAG_ANE: dict[str, list[str]] = {
    **_FEATURE_SCHEMA_MAG,
    "ane": _ANE_CHANNELS,
}

MODEL_HZ: float = 1000.0  # 1 kHz — preserves ELM spike structure

# Dα ELM activity threshold: |dDα/dt| above this fraction of per-shot std
# is classified as TRANSIENT (ELM active).  Tuned for 1 kHz grid.
_TRANSIENT_DDALPHA_THRESHOLD: float = 2.0  # units: × per-shot std(|d/dt|)

# Conformal nominal level
_CONFORMAL_ALPHA: float = 0.10  # → 90% nominal coverage

# Splits manifest path
_SPLITS_MANIFEST: Path = Path(
    "/work/projects/imas_gpu/mast/manifests/statespace_splits_dalpha_v0.json"
)
_LEVEL1_DIR: Path = Path("/work/projects/imas_gpu/mast/level1/shots")

# ---------------------------------------------------------------------------
# Data loading — per-shot slice extraction
# ---------------------------------------------------------------------------


def _read_group_channels(
    store: Any,
    group: str,
    channels: list[str],
    target_time: np.ndarray,
) -> np.ndarray | None:
    """Read specified channels from a Zarr group and interpolate to target_time.

    Returns shape (T, C) or None if the group is absent / has no valid data.
    All NaN columns cause the group to be marked unavailable.

    Parameters
    ----------
    store : zarr.Group
        Open Zarr store for the shot.
    group : str
        Group name (e.g. "amc").
    channels : list[str]
        Channel names to read.
    target_time : np.ndarray
        Common time grid (shape (T,)) to interpolate onto.
    """
    if group not in store:
        return None
    grp = store[group]
    if "time" not in grp:
        return None
    t = np.asarray(grp["time"], dtype=np.float64)
    if t.size < 2:
        return None

    cols: list[np.ndarray] = []
    for ch in channels:
        if ch not in grp:
            cols.append(np.full(len(target_time), np.nan))
            continue
        try:
            arr = np.asarray(grp[ch], dtype=np.float64)
        except Exception:
            cols.append(np.full(len(target_time), np.nan))
            continue
        if arr.ndim != 1 or arr.shape[0] != t.shape[0]:
            cols.append(np.full(len(target_time), np.nan))
            continue
        # Interpolate onto common time grid (clamp to data range)
        interp = np.interp(target_time, t, arr, left=arr[0], right=arr[-1])
        cols.append(interp)

    return np.stack(cols, axis=1)  # (T, C)


def _build_common_time_grid(
    store: Any,
    anchor_group: str = "amc",
    model_hz: float = MODEL_HZ,
) -> np.ndarray | None:
    """Build a uniform time grid for one shot from the anchor group's extent."""
    if anchor_group not in store:
        return None
    t = np.asarray(store[anchor_group]["time"] if "time" in store[anchor_group] else [])
    if t.size < 2:
        return None
    t_start = float(t.min())
    t_end = float(t.max())
    n = max(1, int(round((t_end - t_start) * model_hz)))
    return np.linspace(t_start, t_end, n)


def load_shot_slices(
    shot_id: int,
    feature_schema: dict[str, list[str]],
    target_channels: list[str],
    level1_dir: Path = _LEVEL1_DIR,
    model_hz: float = MODEL_HZ,
    max_slices: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Load one shot and return aligned (X, y, times, plasma_on_mask).

    Returns
    -------
    (X, y, times, plasma_on_mask) where
      X : (T, F)  feature matrix (all input channels)
      y : (T, D)  target matrix (D=1 for primary, D>1 for multi-chord)
      times : (T,) float  common time axis
      plasma_on_mask : (T,) bool  plasma-on window mask

    Returns None if the shot lacks either the target or any required group.

    NaN slices (in X or y) are dropped before return.
    """
    import zarr  # noqa: PLC0415

    shot_path = level1_dir / f"{shot_id}.zarr"
    if not shot_path.exists():
        return None

    try:
        store = zarr.open_group(str(shot_path), mode="r")
    except Exception:
        return None

    # Build common time grid from amc anchor
    times = _build_common_time_grid(store, model_hz=model_hz)
    if times is None:
        return None

    # Load target (xim group) — required
    if "xim" not in store:
        return None
    xim_grp = store["xim"]
    if "time" not in xim_grp:
        return None
    xim_t = np.asarray(xim_grp["time"], dtype=np.float64)
    target_cols: list[np.ndarray] = []
    for ch in target_channels:
        if ch not in xim_grp:
            return None  # required channel missing → skip shot
        arr = np.asarray(xim_grp[ch], dtype=np.float64)
        if arr.ndim != 1 or arr.shape[0] != xim_t.shape[0]:
            return None
        interp = np.interp(times, xim_t, arr, left=arr[0], right=arr[-1])
        target_cols.append(interp)
    y = np.stack(target_cols, axis=1)  # (T, D)

    # Load input groups
    feature_parts: list[np.ndarray] = []
    for group, channels in feature_schema.items():
        mat = _read_group_channels(store, group, channels, times)
        if mat is None:
            return None  # required group missing → skip shot
        feature_parts.append(mat)

    X = np.concatenate(feature_parts, axis=1)  # (T, F)

    # Plasma-on mask from amc/plasma_current
    plasma_on = np.zeros(len(times), dtype=bool)
    if "amc" in store and "plasma_current" in store["amc"]:
        ip = np.asarray(store["amc"]["plasma_current"], dtype=np.float64)
        ip_t = np.asarray(store["amc"]["time"], dtype=np.float64) if "time" in store["amc"] else None
        if ip_t is not None and ip.shape == ip_t.shape:
            ip_on_grid = np.interp(times, ip_t, np.abs(ip))
            peak = float(np.nanmax(ip_on_grid))
            if peak > 50.0:  # kA threshold
                thresh = max(50.0, 0.2 * peak)
                plasma_on = ip_on_grid > thresh

    # Impute NaN features with column mean (channels absent in this shot or
    # with missing values get imputed to 0 after normalization, i.e. the
    # training-set channel mean).  Target NaN slices are dropped.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        col_means = np.nanmean(X, axis=0)
    col_means = np.where(np.isfinite(col_means), col_means, 0.0)
    for j in range(X.shape[1]):
        nan_mask = ~np.isfinite(X[:, j])
        if nan_mask.any():
            X[nan_mask, j] = col_means[j]

    # Drop slices where target is NaN (cannot train/evaluate on these)
    valid = np.isfinite(y).all(axis=1)
    if not valid.any():
        return None
    X = X[valid]
    y = y[valid]
    times = times[valid]
    plasma_on = plasma_on[valid]

    # After imputation X should be all finite; guard anyway
    valid = np.isfinite(X).all(axis=1)
    if not valid.any():
        return None

    X = X[valid]
    y = y[valid]
    times = times[valid]
    plasma_on = plasma_on[valid]

    # Restrict to plasma-on window only (avoids fitting on off-plasma noise)
    if plasma_on.any():
        X = X[plasma_on]
        y = y[plasma_on]
        times = times[plasma_on]
        plasma_on = plasma_on[plasma_on]

    if len(X) == 0:
        return None

    # Subsample slices if requested
    if max_slices is not None and len(X) > max_slices:
        idx = np.linspace(0, len(X) - 1, max_slices, dtype=int)
        X = X[idx]
        y = y[idx]
        times = times[idx]
        plasma_on = plasma_on[idx]

    return X, y, times, plasma_on


def compute_transient_mask(y: np.ndarray, threshold_sigma: float = _TRANSIENT_DDALPHA_THRESHOLD) -> np.ndarray:
    """Classify slices as transient (ELM) vs quiescent from Dα activity.

    A slice is transient if |dDα/dt| > threshold_sigma × std(|dDα/dt|),
    evaluated per-shot.  At 1 kHz, dt = 0.001 s.

    Parameters
    ----------
    y : (T, D) or (T,) array of Dα values.
    threshold_sigma : multiplier on std.

    Returns
    -------
    (T,) bool array, True = transient.
    """
    y1d = y[:, 0] if y.ndim == 2 else y
    if len(y1d) < 3:
        return np.zeros(len(y1d), dtype=bool)
    dy = np.abs(np.diff(y1d, prepend=y1d[0]))
    std_dy = float(np.std(dy))
    if std_dy < 1e-15:
        return np.zeros(len(y1d), dtype=bool)
    return dy > threshold_sigma * std_dy


# ---------------------------------------------------------------------------
# Normalisation helpers (fit on train, apply everywhere)
# ---------------------------------------------------------------------------


@dataclass
class ChannelStats:
    """Per-channel mean and std for normalisation.

    Fit on training slices only; applied to all splits.
    """
    feature_mean: np.ndarray  # (F,)
    feature_std: np.ndarray   # (F,) — clamped to ≥ 1e-8
    target_mean: np.ndarray   # (D,)
    target_std: np.ndarray    # (D,) — clamped to ≥ 1e-8

    def normalise_X(self, X: np.ndarray) -> np.ndarray:
        return (X - self.feature_mean) / self.feature_std

    def normalise_y(self, y: np.ndarray) -> np.ndarray:
        return (y - self.target_mean) / self.target_std

    def denormalise_y_mean(self, y_hat: np.ndarray) -> np.ndarray:
        return y_hat * self.target_std + self.target_mean

    def denormalise_y_std(self, y_std: np.ndarray) -> np.ndarray:
        return y_std * self.target_std

    def to_dict(self) -> dict:
        return {
            "feature_mean": self.feature_mean.tolist(),
            "feature_std": self.feature_std.tolist(),
            "target_mean": self.target_mean.tolist(),
            "target_std": self.target_std.tolist(),
        }

    @classmethod
    def fit(cls, X_list: list[np.ndarray], y_list: list[np.ndarray]) -> "ChannelStats":
        """Fit on a list of per-shot arrays (pools all slices)."""
        X_all = np.concatenate(X_list, axis=0)
        y_all = np.concatenate(y_list, axis=0)
        fm = np.nanmean(X_all, axis=0)
        fs = np.nanstd(X_all, axis=0)
        fs = np.where(fs < 1e-8, 1e-8, fs)
        tm = np.nanmean(y_all, axis=0)
        ts = np.nanstd(y_all, axis=0)
        ts = np.where(ts < 1e-8, 1e-8, ts)
        return cls(feature_mean=fm, feature_std=fs, target_mean=tm, target_std=ts)


# ---------------------------------------------------------------------------
# Deep ensemble MLP (numpy + manual SGD — no heavy ML dependency)
# ---------------------------------------------------------------------------


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _softplus(x: np.ndarray) -> np.ndarray:
    # Numerically stable: log(1 + exp(x))
    return np.where(x > 20.0, x, np.log1p(np.exp(np.clip(x, -500, 20))))


class MLPGaussian:
    """Small MLP with Gaussian predictive head, trained with NLL.

    Architecture: Linear(F,H) → ReLU → Linear(H,H) → ReLU → Linear(H, 2D)
    where H = hidden_size, D = output_dim.

    Output: [μ₁,...,μD, log_σ₁,...,log_σD].  σ = softplus(log_σ) + 1e-4.
    Loss: mean NLL = 0.5 * [log(2π σ²) + ((y−μ)/σ)²].
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_size: int = 128,
        seed: int = 0,
    ) -> None:
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_size = hidden_size
        self.seed = seed
        rng = np.random.default_rng(seed)

        # He-initialise weights, zero-init biases
        def _he(fan_in: int, fan_out: int) -> np.ndarray:
            return rng.normal(0, math.sqrt(2.0 / fan_in), (fan_in, fan_out)).astype(np.float64)

        self.W1 = _he(input_dim, hidden_size)
        self.b1 = np.zeros(hidden_size)
        self.W2 = _he(hidden_size, hidden_size)
        self.b2 = np.zeros(hidden_size)
        self.W3 = _he(hidden_size, 2 * output_dim)
        self.b3 = np.zeros(2 * output_dim)
        # log_σ head initialised slightly negative → σ ≈ 0.6
        self.b3[output_dim:] = -0.5

        # Adam state
        self._adam_m: list[np.ndarray] = [np.zeros_like(p) for p in self._params()]
        self._adam_v: list[np.ndarray] = [np.zeros_like(p) for p in self._params()]
        self._adam_t: int = 0

    def _params(self) -> list[np.ndarray]:
        return [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]

    def forward(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Forward pass. Returns (mu, sigma) each shape (N, D)."""
        h1 = _relu(X @ self.W1 + self.b1)
        h2 = _relu(h1 @ self.W2 + self.b2)
        out = h2 @ self.W3 + self.b3
        mu = out[:, : self.output_dim]
        log_sigma = out[:, self.output_dim :]
        sigma = _softplus(log_sigma) + 1e-4
        return mu, sigma

    def nll_and_grads(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[float, list[np.ndarray]]:
        """Compute NLL loss and gradients via backprop.

        Returns (loss, [dW1, db1, dW2, db2, dW3, db3]).
        """
        N = X.shape[0]
        # Forward
        h1 = _relu(X @ self.W1 + self.b1)
        h2 = _relu(h1 @ self.W2 + self.b2)
        out = h2 @ self.W3 + self.b3
        mu = out[:, : self.output_dim]
        log_sigma = out[:, self.output_dim :]
        sigma = _softplus(log_sigma) + 1e-4

        # NLL
        err = y - mu
        nll_per = 0.5 * (np.log(2 * np.pi * sigma**2) + (err / sigma) ** 2)
        loss = float(np.mean(nll_per))

        # Gradients through output layer
        d_nll_mu = -err / (sigma**2)  # (N, D)
        d_nll_sigma = 1.0 / sigma - err**2 / (sigma**3)  # (N, D)
        # sigma = softplus(log_sigma) + eps → d_sigma/d_log_sigma = sigmoid(log_sigma)
        sig_log = 1.0 / (1.0 + np.exp(-log_sigma))
        d_nll_log_sigma = d_nll_sigma * sig_log

        d_out = np.concatenate([d_nll_mu, d_nll_log_sigma], axis=1) / N  # (N, 2D)

        dW3 = h2.T @ d_out
        db3 = d_out.sum(axis=0)

        dh2 = d_out @ self.W3.T
        dh2 *= (h2 > 0).astype(float)  # ReLU gate

        dW2 = h1.T @ dh2
        db2 = dh2.sum(axis=0)

        dh1 = dh2 @ self.W2.T
        dh1 *= (h1 > 0).astype(float)

        dW1 = X.T @ dh1
        db1 = dh1.sum(axis=0)

        return loss, [dW1, db1, dW2, db2, dW3, db3]

    def adam_step(
        self,
        grads: list[np.ndarray],
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        """Adam optimiser update."""
        self._adam_t += 1
        t = self._adam_t
        params = self._params()
        for i, (p, g) in enumerate(zip(params, grads, strict=True)):
            self._adam_m[i] = beta1 * self._adam_m[i] + (1 - beta1) * g
            self._adam_v[i] = beta2 * self._adam_v[i] + (1 - beta2) * g**2
            m_hat = self._adam_m[i] / (1 - beta1**t)
            v_hat = self._adam_v[i] / (1 - beta2**t)
            p -= lr * m_hat / (np.sqrt(v_hat) + eps)

    def fit_sgd(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_epochs: int = 50,
        batch_size: int = 512,
        lr: float = 1e-3,
        rng: np.random.Generator | None = None,
        grad_clip: float | None = None,
    ) -> list[float]:
        """Mini-batch Adam training. Returns list of per-epoch mean NLL.

        ``grad_clip`` (opt-in; default ``None`` = off, preserving the original
        decimated-data behaviour) applies global-norm gradient clipping. Set it
        (e.g. ``grad_clip=10.0``) when training on sharp targets — dense,
        un-decimated Dα ELM spikes drove unclipped members to NLL≈+7 / σ≈87 and
        a meaningless predictor (found during S7.3, 2026-05-29). Off by default
        so existing S7.2 (decimated-slice) runs are bit-for-bit unchanged.
        """
        if rng is None:
            rng = np.random.default_rng(self.seed + 1000)
        N = X_train.shape[0]
        epoch_losses: list[float] = []
        for epoch in range(n_epochs):
            perm = rng.permutation(N)
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, N, batch_size):
                idx = perm[start : start + batch_size]
                loss, grads = self.nll_and_grads(X_train[idx], y_train[idx])
                if grad_clip is not None and grad_clip > 0:
                    gnorm = math.sqrt(sum(float(np.sum(g * g)) for g in grads))
                    if gnorm > grad_clip:
                        scale = grad_clip / (gnorm + 1e-12)
                        grads = [g * scale for g in grads]
                self.adam_step(grads, lr=lr)
                epoch_loss += loss
                n_batches += 1
            epoch_losses.append(epoch_loss / max(n_batches, 1))
        return epoch_losses


# ---------------------------------------------------------------------------
# Deep Ensemble
# ---------------------------------------------------------------------------


@dataclass
class EnsembleConfig:
    """Configuration for the deep ensemble."""

    n_members: int = 5
    hidden_size: int = 128
    n_epochs: int = 60
    batch_size: int = 512
    lr: float = 1e-3
    seed_base: int = 0  # member i uses seed = seed_base + i


class DeepEnsemble:
    """Ensemble of MLPGaussian members, one per seed.

    Predictive distribution: mixture of Gaussians (analytic moment-matched
    to a single Gaussian for downstream metrics).
    """

    def __init__(self, members: list[MLPGaussian]) -> None:
        self.members = members

    @classmethod
    def build(
        cls,
        input_dim: int,
        output_dim: int,
        cfg: EnsembleConfig,
    ) -> "DeepEnsemble":
        members = [
            MLPGaussian(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_size=cfg.hidden_size,
                seed=cfg.seed_base + i,
            )
            for i in range(cfg.n_members)
        ]
        return cls(members)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        cfg: EnsembleConfig,
    ) -> None:
        """Fit all ensemble members independently."""
        for i, m in enumerate(self.members):
            t0 = time.time()
            rng = np.random.default_rng(cfg.seed_base + i + 999)
            losses = m.fit_sgd(
                X_train,
                y_train,
                n_epochs=cfg.n_epochs,
                batch_size=cfg.batch_size,
                lr=cfg.lr,
                rng=rng,
            )
            logger.info(
                "  Member %d/%d: final NLL=%.4f  (%.1fs)",
                i + 1,
                len(self.members),
                losses[-1],
                time.time() - t0,
            )

    def predict(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict predictive mean, total std, and per-member arrays.

        Returns
        -------
        mu : (N, D)  — mixture-of-Gaussians mean
        sigma_total : (N, D)  — sqrt(mean aleatoric² + epistemic)
        ensemble_preds : (N, M, D)  — per-member μ (for OOD + energy score)
        """
        M = len(self.members)
        N, _ = X.shape

        mu_members = np.zeros((N, M, self.members[0].output_dim))
        sigma_members = np.zeros_like(mu_members)

        for i, m in enumerate(self.members):
            mu_m, sigma_m = m.forward(X)
            mu_members[:, i, :] = mu_m
            sigma_members[:, i, :] = sigma_m

        mu = mu_members.mean(axis=1)  # (N, D)
        # Law of total variance: Var_total = E[Var_aleatoric] + Var[E[member_mean]]
        mean_aleatoric_var = (sigma_members**2).mean(axis=1)
        epistemic_var = mu_members.var(axis=1)
        sigma_total = np.sqrt(mean_aleatoric_var + epistemic_var)

        return mu, sigma_total, mu_members

    @property
    def n_members(self) -> int:
        return len(self.members)


# ---------------------------------------------------------------------------
# Split-conformal calibration wrapper
# ---------------------------------------------------------------------------


class ConformalWrapper:
    """Split-conformal calibration on top of a DeepEnsemble.

    Fits a normalised-residual quantile on CONFORMAL-CAL slices.
    Coverage-achieving scale factor q̂ such that:
      P(|y − μ| ≤ q̂ · σ_total) ≥ 1 - α

    At test time, feeds scaled σ (q̂ · σ / z_α) to calibration.py so that
    the harness's μ ± z_α · σ_eff reproduces the conformal interval.
    """

    def __init__(
        self,
        ensemble: DeepEnsemble,
        stats: ChannelStats,
        alpha: float = _CONFORMAL_ALPHA,
    ) -> None:
        self.ensemble = ensemble
        self.stats = stats
        self.alpha = alpha
        self.q_hat: float = 1.0  # set by .fit_conformal()
        self._n_cal: int = 0

    def fit_conformal(
        self,
        X_conf: np.ndarray,
        y_conf: np.ndarray,
    ) -> None:
        """Fit conformal quantile on normalised-input arrays."""
        mu, sigma, _ = self.ensemble.predict(X_conf)
        scores = np.abs(y_conf - mu) / np.maximum(sigma, 1e-12)  # (N, D)
        # Flatten to per-output scores, take max over D for marginal coverage
        scores_flat = scores.max(axis=1)  # conservative marginal
        n = len(scores_flat)
        self._n_cal = n
        level = math.ceil((n + 1) * (1.0 - self.alpha)) / n
        level = min(level, 1.0)
        self.q_hat = float(np.quantile(scores_flat, level, method="higher"))
        logger.info(
            "Conformal quantile q̂=%.4f  (n=%d, level=%.4f, alpha=%.2f)",
            self.q_hat,
            n,
            level,
            self.alpha,
        )

    def predict_calibrated(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Predict with conformal-scaled uncertainty.

        Returns
        -------
        mu : (N, D)  predictive mean (in normalised space)
        sigma_raw : (N, D)  raw ensemble total σ
        sigma_conf : (N, D)  conformal-scaled σ  (q̂ · sigma_raw / z_α)
        ensemble_preds : (N, M, D)  per-member means (for disagreement)
        """
        from scipy.stats import norm  # noqa: PLC0415

        z_alpha = float(norm.ppf(1.0 - self.alpha / 2.0))
        mu, sigma_raw, ens = self.ensemble.predict(X)
        sigma_conf = self.q_hat * sigma_raw / z_alpha
        return mu, sigma_raw, sigma_conf, ens


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


@dataclass
class BaselineConfig:
    """Full pipeline configuration."""

    # Input schema: "mag" or "mag_ane"
    input_modality: str = "mag_ane"
    # Target: "primary" (da_hm10_t) or "multi" (5-chord array)
    target_mode: str = "primary"
    # Max slices per shot for tractable v0 training
    max_slices_per_shot: int = 200
    # Max shots to subsample for training (None = use all)
    max_train_shots: int | None = 2000
    # Conformal-cal / in-dist-test sub-split fraction (from calibration shots)
    conformal_cal_fraction: float = 0.50
    sub_split_seed: int = 42
    # Ensemble config
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    # Paths
    splits_manifest: Path = _SPLITS_MANIFEST
    level1_dir: Path = _LEVEL1_DIR


@dataclass
class BaselineResult:
    """All evaluation outputs from one baseline run."""

    config: BaselineConfig
    # Split sizes
    n_train_shots: int = 0
    n_train_slices: int = 0
    n_conf_cal_shots: int = 0
    n_conf_cal_slices: int = 0
    n_in_dist_test_shots: int = 0
    n_in_dist_test_slices: int = 0
    n_ood_shots: int = 0
    n_ood_slices: int = 0
    # In-dist metrics
    in_dist: dict = field(default_factory=dict)
    # OOD metrics
    ood: dict = field(default_factory=dict)
    # Transient stratification
    transient: dict = field(default_factory=dict)
    quiescent: dict = field(default_factory=dict)
    # ane lift (mag-only vs mag+ane on same shots)
    ane_lift: dict = field(default_factory=dict)
    # Coverage gate
    coverage_gate_passed: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "config": {
                "input_modality": self.config.input_modality,
                "target_mode": self.config.target_mode,
                "max_slices_per_shot": self.config.max_slices_per_shot,
                "max_train_shots": self.config.max_train_shots,
                "ensemble_n_members": self.config.ensemble.n_members,
                "ensemble_n_epochs": self.config.ensemble.n_epochs,
                "ensemble_hidden_size": self.config.ensemble.hidden_size,
                "conformal_alpha": _CONFORMAL_ALPHA,
                "model_hz": MODEL_HZ,
            },
            "split_sizes": {
                "n_train_shots": self.n_train_shots,
                "n_train_slices": self.n_train_slices,
                "n_conformal_cal_shots": self.n_conf_cal_shots,
                "n_conformal_cal_slices": self.n_conf_cal_slices,
                "n_in_dist_test_shots": self.n_in_dist_test_shots,
                "n_in_dist_test_slices": self.n_in_dist_test_slices,
                "n_ood_shots": self.n_ood_shots,
                "n_ood_slices": self.n_ood_slices,
            },
            "in_dist": self.in_dist,
            "ood": self.ood,
            "transient": self.transient,
            "quiescent": self.quiescent,
            "ane_lift": self.ane_lift,
            "coverage_gate_passed": self.coverage_gate_passed,
            "notes": self.notes,
        }


def _load_split_slices(
    shot_ids: list[int],
    feature_schema: dict[str, list[str]],
    target_channels: list[str],
    level1_dir: Path,
    max_slices_per_shot: int,
    max_shots: int | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[int]]:
    """Load slices for a list of shots.

    Returns
    -------
    (Xs, ys, transient_masks, surviving_shot_ids)
        surviving_shot_ids — shot IDs that were successfully loaded (same
        length as Xs/ys/tmasks).  Used to align per-shot distances to slices.
    """
    if max_shots is not None and max_shots < len(shot_ids):
        if rng is None:
            rng = np.random.default_rng(42)
        idxs = rng.choice(len(shot_ids), size=max_shots, replace=False)
        shot_ids = [shot_ids[i] for i in sorted(idxs)]

    Xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    tmasks: list[np.ndarray] = []
    ok_ids: list[int] = []

    n_ok = 0
    n_skip = 0
    for sid in shot_ids:
        result = load_shot_slices(
            sid,
            feature_schema,
            target_channels,
            level1_dir=level1_dir,
            max_slices=max_slices_per_shot,
        )
        if result is None:
            n_skip += 1
            continue
        X_shot, y_shot, _times, _pon = result
        tmask = compute_transient_mask(y_shot)
        Xs.append(X_shot)
        ys.append(y_shot)
        tmasks.append(tmask)
        ok_ids.append(int(sid))
        n_ok += 1

    logger.info(
        "Loaded %d / %d shots (%d skipped)",
        n_ok,
        n_ok + n_skip,
        n_skip,
    )
    return Xs, ys, tmasks, ok_ids


def _compute_regime_distances(
    shot_ids: list[int],
    regime_scalars: dict[str | int, dict],
    train_ip: np.ndarray,
    train_ne: np.ndarray,
) -> np.ndarray:
    """Standardised Euclidean distance from training centroid per shot.

    Parameters
    ----------
    train_ip, train_ne :
        Arrays of ip_mean and ne_mean for training shots (used to fit normaliser).

    Returns
    -------
    (len(shot_ids),) float array — one distance per shot.
    NaN for shots without regime scalars.
    """
    mu_ip = float(np.mean(train_ip))
    std_ip = float(np.std(train_ip)) or 1.0
    mu_ne = float(np.mean(train_ne))
    std_ne = float(np.std(train_ne)) or 1.0

    dists = np.full(len(shot_ids), np.nan)
    for i, sid in enumerate(shot_ids):
        sc = regime_scalars.get(int(sid)) or regime_scalars.get(str(sid))
        if sc is None:
            continue
        ip = sc.get("ip_mean", np.nan)
        ne = sc.get("ne_mean", np.nan)
        if not (np.isfinite(ip) and np.isfinite(ne)):
            continue
        dip = (ip - mu_ip) / std_ip
        dne = (ne / 1e19 - mu_ne / 1e19) / std_ne
        dists[i] = math.sqrt(dip**2 + dne**2)
    return dists


def _eval_split(
    label: str,
    Xs: list[np.ndarray],
    ys: list[np.ndarray],
    tmasks: list[np.ndarray],
    conformal: ConformalWrapper,
    stats: ChannelStats,
    ood_in_dist_scores: np.ndarray | None = None,
    ood_ood_scores: np.ndarray | None = None,
    distances: np.ndarray | None = None,
) -> dict:
    """Evaluate all §4 metrics on a split.

    Returns a dict of metrics (JSON-serialisable).
    """
    from imas_ambix.statespace.calibration import (  # noqa: PLC0415
        compute_calibration_report,
        coverage_vs_distance,
        crps_ensemble as _crps_ens,
        ensemble_disagreement,
    )

    if not Xs:
        return {"error": "no data"}

    X_all = np.concatenate(Xs, axis=0)
    y_all = np.concatenate(ys, axis=0)
    tmask_all = np.concatenate(tmasks, axis=0)

    X_norm = stats.normalise_X(X_all)
    y_norm = stats.normalise_y(y_all)

    mu_norm, sigma_raw, sigma_conf, ens_preds = conformal.predict_calibrated(X_norm)

    # Denormalise for physical-space metrics
    mu_phys = stats.denormalise_y_mean(mu_norm)
    sigma_phys_raw = stats.denormalise_y_std(sigma_raw)
    sigma_phys_conf = stats.denormalise_y_std(sigma_conf)

    # Squeeze to 1D for primary target
    D = y_all.shape[1] if y_all.ndim == 2 else 1
    if D == 1:
        y_1d = y_all[:, 0] if y_all.ndim == 2 else y_all
        mu_1d = mu_phys[:, 0] if mu_phys.ndim == 2 else mu_phys
        sr_1d = sigma_phys_raw[:, 0] if sigma_phys_raw.ndim == 2 else sigma_phys_raw
        sc_1d = sigma_phys_conf[:, 0] if sigma_phys_conf.ndim == 2 else sigma_phys_conf
    else:
        # Headline on first channel (da_hm10_t is index 2 in multi — handled below)
        y_1d = y_all[:, 0]
        mu_1d = mu_phys[:, 0]
        sr_1d = sigma_phys_raw[:, 0]
        sc_1d = sigma_phys_conf[:, 0]

    # Ensemble disagreement per sample (for OOD scoring)
    ens_disag = ensemble_disagreement(ens_preds)  # (N,)

    # Full calibration report (using conformal-scaled σ for coverage gate;
    # raw σ for NLL + CRPS to avoid distorting likelihood metrics)
    report_conf = compute_calibration_report(
        y_1d, mu_1d, sc_1d,
        ensemble=ens_preds[:, :, 0:1] if D >= 1 else ens_preds,
        ood_in_dist_scores=ood_in_dist_scores,
        ood_ood_scores=ood_ood_scores,
    )
    # Raw report for NLL / CRPS
    report_raw = compute_calibration_report(
        y_1d, mu_1d, sr_1d,
    )

    # Energy score (true multivariate, using ensemble samples)
    # ES = E||X_m - y||_2 - 0.5 * E||X_m - X_m'||_2
    # Sample from each member's Gaussian: X_m_sample ~ N(mu_m, sigma_m)
    rng_es = np.random.default_rng(1234)
    n_es = min(len(X_all), 10000)  # cap for speed
    idx_es = np.arange(len(X_all)) if len(X_all) <= n_es else rng_es.choice(len(X_all), size=n_es, replace=False)
    ens_sample = ens_preds[idx_es]  # (n_es, M, D) — member means (use as samples)
    y_es = y_all[idx_es] if y_all.ndim == 2 else y_all[idx_es, np.newaxis]
    if D == 1:
        # Squeeze D=1 for crps_ensemble (handles 1D)
        energy_score = _crps_ens(y_es[:, 0], ens_sample[:, :, 0])
    else:
        # True vector energy score
        N_es = ens_sample.shape[0]
        M_es = ens_sample.shape[1]
        # E||X_m - y||_2
        diff_xy = np.linalg.norm(ens_sample - y_es[:, np.newaxis, :], axis=-1)  # (N, M)
        term1 = diff_xy.mean()
        # E||X_m - X_m'||_2 via pairwise
        pair_sum = 0.0
        for mi in range(M_es):
            for mj in range(mi + 1, M_es):
                pair_sum += np.linalg.norm(ens_sample[:, mi, :] - ens_sample[:, mj, :], axis=-1).mean()
        n_pairs = M_es * (M_es - 1) / 2
        term2 = pair_sum / max(n_pairs, 1)
        energy_score = float(term1 - 0.5 * term2)

    # Coverage-vs-distance (if distances provided)
    covdist: dict = {}
    if distances is not None:
        valid_d = np.isfinite(distances)
        if valid_d.any():
            # Broadcast per-shot distances to slices — use mean distance if
            # distances array length matches shots, not slices
            # Here distances should already be per-slice if passed correctly
            if len(distances) == len(y_1d):
                covdist = coverage_vs_distance(y_1d, mu_1d, sc_1d, distances)

    result = {
        "n_slices": int(len(y_1d)),
        # Coverage gate metric: conformal-scaled σ
        "coverage_90_conf": float(report_conf.coverage_90),
        # Raw predictive metrics
        "nll_raw": float(report_raw.nll),
        "crps_raw": float(report_raw.crps),
        "pi_width_90_conf": float(report_conf.pi_width_90),
        "ece_conf": float(report_conf.ece),
        "coverage_by_level_conf": {
            str(k): float(v) for k, v in report_conf.coverage_by_level.items()
        },
        "ensemble_disagreement_mean": float(np.mean(ens_disag)),
        "ood_auroc": report_conf.ood_auroc,
        "energy_score": float(energy_score),
        "coverage_vs_distance": covdist,
        "reliability_diagram": report_conf.reliability_diagram,
    }

    # Transient / quiescent stratification
    for stratum_label, mask in [("transient", tmask_all), ("quiescent", ~tmask_all)]:
        if mask.sum() < 10:
            result[f"{stratum_label}_n"] = int(mask.sum())
            continue
        y_s = y_1d[mask]
        mu_s = mu_1d[mask]
        sc_s = sc_1d[mask]
        sr_s = sr_1d[mask]
        from imas_ambix.statespace.calibration import (  # noqa: PLC0415
            crps_gaussian,
            interval_coverage,
            nll_gaussian,
            prediction_interval_width,
        )
        result[f"{stratum_label}_n"] = int(mask.sum())
        result[f"{stratum_label}_coverage_90"] = float(interval_coverage(y_s, mu_s, sc_s, alpha=0.10))
        result[f"{stratum_label}_crps"] = float(crps_gaussian(y_s, mu_s, sr_s))
        result[f"{stratum_label}_nll"] = float(nll_gaussian(y_s, mu_s, sr_s))
        result[f"{stratum_label}_pi_width_90"] = float(prediction_interval_width(sc_s, alpha=0.10))

    logger.info(
        "[%s] N=%d  cov@90=%.3f  CRPS=%.4f  NLL=%.4f  transient=%d  quiescent=%d",
        label,
        len(y_1d),
        result["coverage_90_conf"],
        result["crps_raw"],
        result["nll_raw"],
        result.get("transient_n", 0),
        result.get("quiescent_n", 0),
    )
    return result


# ---------------------------------------------------------------------------
# Top-level entrypoint
# ---------------------------------------------------------------------------


def run_baseline(cfg: BaselineConfig) -> BaselineResult:
    """Run the full S7.2 static baseline pipeline.

    Reads pre-computed splits, loads data, trains ensemble, calibrates
    with split-conformal, and evaluates all §4 metrics.

    Parameters
    ----------
    cfg : BaselineConfig
        Full pipeline configuration.

    Returns
    -------
    BaselineResult with all metrics populated.
    """
    t_total = time.time()
    result = BaselineResult(config=cfg)

    # -----------------------------------------------------------------------
    # 1. Load split manifest
    # -----------------------------------------------------------------------
    logger.info("Loading splits from %s", cfg.splits_manifest)
    with open(cfg.splits_manifest) as f:
        splits_data = json.load(f)

    train_shots = [int(x) for x in splits_data["train"]]
    cal_shots = [int(x) for x in splits_data["calibration"]]
    ood_shots = [int(x) for x in splits_data["test_ood_regime"]]
    regime_scalars = splits_data.get("regime_scalars", {})

    # -----------------------------------------------------------------------
    # 2. Sub-split calibration shots into CONFORMAL-CAL + IN-DIST-TEST
    #    (deterministic, fixed seed, shot-level)
    # -----------------------------------------------------------------------
    rng_sub = np.random.default_rng(cfg.sub_split_seed)
    cal_arr = np.array(cal_shots)
    perm = rng_sub.permutation(len(cal_arr))
    n_conf = int(round(len(cal_arr) * cfg.conformal_cal_fraction))
    conf_cal_shots = sorted(cal_arr[perm[:n_conf]].tolist())
    in_dist_test_shots = sorted(cal_arr[perm[n_conf:]].tolist())

    # Verify disjointness (defensive check)
    assert frozenset(train_shots).isdisjoint(frozenset(conf_cal_shots)), "TRAIN ∩ CONF-CAL not empty!"
    assert frozenset(train_shots).isdisjoint(frozenset(in_dist_test_shots)), "TRAIN ∩ IN-DIST-TEST not empty!"
    assert frozenset(conf_cal_shots).isdisjoint(frozenset(in_dist_test_shots)), "CONF-CAL ∩ IN-DIST-TEST not empty!"
    assert frozenset(ood_shots).isdisjoint(frozenset(conf_cal_shots + in_dist_test_shots + train_shots)), "OOD overlaps!"

    logger.info(
        "4-way split: train=%d  conf_cal=%d  in_dist_test=%d  ood=%d",
        len(train_shots),
        len(conf_cal_shots),
        len(in_dist_test_shots),
        len(ood_shots),
    )

    # -----------------------------------------------------------------------
    # 3. Feature schema selection
    # -----------------------------------------------------------------------
    feature_schema = (
        _FEATURE_SCHEMA_MAG_ANE
        if cfg.input_modality == "mag_ane"
        else _FEATURE_SCHEMA_MAG
    )
    target_channels = (
        _XIM_CHANNELS_MULTI if cfg.target_mode == "multi" else _XIM_CHANNELS_PRIMARY
    )

    # -----------------------------------------------------------------------
    # 4. Load training slices (may be subsampled)
    # -----------------------------------------------------------------------
    logger.info("Loading TRAIN slices (max_shots=%s)...", cfg.max_train_shots)
    rng_load = np.random.default_rng(cfg.sub_split_seed + 1)
    Xs_train, ys_train, tmasks_train, train_ok_ids = _load_split_slices(
        train_shots,
        feature_schema,
        target_channels,
        cfg.level1_dir,
        cfg.max_slices_per_shot,
        max_shots=cfg.max_train_shots,
        rng=rng_load,
    )
    if not Xs_train:
        raise RuntimeError("No training data loaded — check LEVEL1_DIR and shot IDs")

    result.n_train_shots = len(Xs_train)
    result.n_train_slices = sum(len(x) for x in Xs_train)
    logger.info("TRAIN: %d shots, %d slices", result.n_train_shots, result.n_train_slices)

    # -----------------------------------------------------------------------
    # 5. Fit normaliser on TRAIN slices only
    # -----------------------------------------------------------------------
    logger.info("Fitting channel stats on TRAIN slices...")
    stats = ChannelStats.fit(Xs_train, ys_train)

    X_train_all = np.concatenate(Xs_train, axis=0)
    y_train_all = np.concatenate(ys_train, axis=0)
    X_train_norm = stats.normalise_X(X_train_all)
    y_train_norm = stats.normalise_y(y_train_all)

    input_dim = X_train_norm.shape[1]
    output_dim = y_train_norm.shape[1] if y_train_norm.ndim == 2 else 1

    # -----------------------------------------------------------------------
    # 6. Train deep ensemble
    # -----------------------------------------------------------------------
    logger.info(
        "Training ensemble (%d members, %d epochs, in_dim=%d, out_dim=%d)...",
        cfg.ensemble.n_members,
        cfg.ensemble.n_epochs,
        input_dim,
        output_dim,
    )
    ensemble = DeepEnsemble.build(input_dim, output_dim, cfg.ensemble)
    ensemble.fit(X_train_norm, y_train_norm, cfg.ensemble)

    # -----------------------------------------------------------------------
    # 7. Load CONFORMAL-CAL slices + fit conformal quantile
    # -----------------------------------------------------------------------
    logger.info("Loading CONFORMAL-CAL slices...")
    Xs_conf, ys_conf, tmasks_conf, conf_ok_ids = _load_split_slices(
        conf_cal_shots, feature_schema, target_channels, cfg.level1_dir, cfg.max_slices_per_shot
    )
    result.n_conf_cal_shots = len(Xs_conf)
    result.n_conf_cal_slices = sum(len(x) for x in Xs_conf)
    logger.info("CONF-CAL: %d shots, %d slices", result.n_conf_cal_shots, result.n_conf_cal_slices)

    X_conf_all = np.concatenate(Xs_conf, axis=0)
    y_conf_all = np.concatenate(ys_conf, axis=0)
    X_conf_norm = stats.normalise_X(X_conf_all)
    y_conf_norm = stats.normalise_y(y_conf_all)

    conformal = ConformalWrapper(ensemble, stats)
    conformal.fit_conformal(X_conf_norm, y_conf_norm)

    # -----------------------------------------------------------------------
    # 8. Load IN-DIST-TEST slices + evaluate (coverage gate)
    # -----------------------------------------------------------------------
    logger.info("Loading IN-DIST-TEST slices...")
    Xs_idt, ys_idt, tmasks_idt, _idt_ok_ids = _load_split_slices(
        in_dist_test_shots, feature_schema, target_channels, cfg.level1_dir, cfg.max_slices_per_shot
    )
    result.n_in_dist_test_shots = len(Xs_idt)
    result.n_in_dist_test_slices = sum(len(x) for x in Xs_idt)

    in_dist_metrics = _eval_split("IN-DIST-TEST", Xs_idt, ys_idt, tmasks_idt, conformal, stats)
    result.in_dist = in_dist_metrics

    cov_90 = float(in_dist_metrics.get("coverage_90_conf", 0.0))
    result.coverage_gate_passed = 0.88 <= cov_90 <= 0.92
    result.notes.append(
        f"Coverage gate: {'PASSED' if result.coverage_gate_passed else 'FAILED'}  "
        f"(coverage@90% = {cov_90:.3f}, gate [0.88, 0.92])"
    )

    # -----------------------------------------------------------------------
    # 9. Load OOD slices + evaluate
    # -----------------------------------------------------------------------
    logger.info("Loading OOD-REGIME-TEST slices...")
    Xs_ood, ys_ood, tmasks_ood, ood_ok_ids = _load_split_slices(
        ood_shots, feature_schema, target_channels, cfg.level1_dir, cfg.max_slices_per_shot
    )
    result.n_ood_shots = len(Xs_ood)
    result.n_ood_slices = sum(len(x) for x in Xs_ood)

    # Compute ensemble disagreement for OOD-AUROC
    X_idt_all_norm = stats.normalise_X(np.concatenate(Xs_idt, axis=0)) if Xs_idt else None
    X_ood_all_norm = stats.normalise_X(np.concatenate(Xs_ood, axis=0)) if Xs_ood else None

    from imas_ambix.statespace.calibration import ensemble_disagreement  # noqa: PLC0415
    idt_disag = None
    ood_disag = None
    if X_idt_all_norm is not None:
        _, _, ens_idt = ensemble.predict(X_idt_all_norm)
        idt_disag = ensemble_disagreement(ens_idt)
    if X_ood_all_norm is not None:
        _, _, ens_ood = ensemble.predict(X_ood_all_norm)
        ood_disag = ensemble_disagreement(ens_ood)

    # Compute coverage-vs-distance for OOD shots
    # Get training regime scalars for normalisation
    train_ip_vals = []
    train_ne_vals = []
    for sid in train_shots:
        sc = regime_scalars.get(str(sid)) or regime_scalars.get(int(sid))
        if sc and "ip_mean" in sc and "ne_mean" in sc:
            train_ip_vals.append(sc["ip_mean"])
            train_ne_vals.append(sc["ne_mean"])
    train_ip_arr = np.array(train_ip_vals) if train_ip_vals else np.array([500.0])
    train_ne_arr = np.array(train_ne_vals) if train_ne_vals else np.array([1e19])

    # OOD shot distances — use ood_ok_ids (surviving shots) not ood_shots (full list)
    # This avoids indexing Xs_ood by position in the full list (which includes skipped shots)
    ood_ok_dists = _compute_regime_distances(ood_ok_ids, regime_scalars, train_ip_arr, train_ne_arr)
    # Broadcast per-shot distances to per-slice
    ood_slice_dists: list[float] = []
    for i, (sid, xs) in enumerate(zip(ood_ok_ids, Xs_ood, strict=True)):
        ood_slice_dists.extend([float(ood_ok_dists[i])] * len(xs))
    ood_slice_dist_arr = np.array(ood_slice_dists) if ood_slice_dists else None

    ood_metrics = _eval_split(
        "OOD-REGIME-TEST",
        Xs_ood, ys_ood, tmasks_ood,
        conformal, stats,
        ood_in_dist_scores=idt_disag,
        ood_ood_scores=ood_disag,
        distances=ood_slice_dist_arr,
    )
    result.ood = ood_metrics

    # -----------------------------------------------------------------------
    # 10. ane lift: mag-only via column-slicing of already-loaded arrays
    #    (no new Zarr reads — avoids data confound and I/O bottleneck)
    # -----------------------------------------------------------------------
    logger.info("Computing ane lift (mag-only vs mag+ane)...")
    result.ane_lift = _compute_ane_lift(
        cfg,
        Xs_train,
        ys_train,
        Xs_conf,
        ys_conf,
        Xs_idt,
        ys_idt,
        conformal,
        stats,
    )

    elapsed = time.time() - t_total
    result.notes.append(f"Total pipeline time: {elapsed:.1f}s")
    logger.info("Pipeline complete in %.1fs", elapsed)
    return result


def _compute_ane_lift(
    cfg: BaselineConfig,
    Xs_train: list[np.ndarray],
    ys_train: list[np.ndarray],
    Xs_conf: list[np.ndarray],
    ys_conf: list[np.ndarray],
    Xs_idt: list[np.ndarray],
    ys_idt: list[np.ndarray],
    conformal_mag_ane: ConformalWrapper,
    stats_mag_ane: ChannelStats,
) -> dict:
    """Compute incremental lift from adding ane to magnetics.

    Uses column-slicing of already-loaded mag+ane arrays — NO new Zarr reads.
    The last column group is ane (1 column), so ``X[:, :-1]`` = mag-only.
    This ensures the same shots/slices/ordering for both models, making the
    comparison element-wise and confound-free.
    """
    from imas_ambix.statespace.calibration import (  # noqa: PLC0415
        crps_gaussian,
        interval_coverage,
        nll_gaussian,
    )

    # --- 1. Derive mag-only arrays by column-slicing (drop last ane column) ---
    # Feature layout: [ama (6 cols), amb (73 cols), amc (42 cols), ane (1 col)]
    # Total mag+ane = 122; mag-only = 121 = X[:, :-1]
    n_ane_cols = len(_ANE_CHANNELS)  # = 1

    def _slice_mag(Xs: list[np.ndarray]) -> list[np.ndarray]:
        return [X[:, :-n_ane_cols] for X in Xs]

    Xs_tr_mag = _slice_mag(Xs_train)
    Xs_cf_mag = _slice_mag(Xs_conf)
    Xs_idt_mag = _slice_mag(Xs_idt)

    if not Xs_tr_mag:
        return {"error": "no mag-only train data (Xs_train empty)"}

    # --- 2. Fit normaliser on mag-only training slices ---
    stats_mag = ChannelStats.fit(Xs_tr_mag, ys_train)
    X_tr_mag_n = stats_mag.normalise_X(np.concatenate(Xs_tr_mag, axis=0))
    y_tr_mag_n = stats_mag.normalise_y(np.concatenate(ys_train, axis=0))

    # --- 3. Train mag-only ensemble ---
    logger.info("  ane lift: training mag-only ensemble (col-sliced, no I/O)...")
    mag_cfg = EnsembleConfig(
        n_members=cfg.ensemble.n_members,
        hidden_size=cfg.ensemble.hidden_size,
        n_epochs=cfg.ensemble.n_epochs,
        batch_size=cfg.ensemble.batch_size,
        lr=cfg.ensemble.lr,
        seed_base=cfg.ensemble.seed_base + 100,  # different seed from main model
    )
    ens_mag = DeepEnsemble.build(
        X_tr_mag_n.shape[1],
        y_tr_mag_n.shape[1] if y_tr_mag_n.ndim == 2 else 1,
        mag_cfg,
    )
    ens_mag.fit(X_tr_mag_n, y_tr_mag_n, mag_cfg)

    # --- 4. Fit conformal on same conf-cal slices (mag-only features) ---
    logger.info("  ane lift: fitting mag-only conformal (same conf-cal shots)...")
    X_cf_mag_n = stats_mag.normalise_X(np.concatenate(Xs_cf_mag, axis=0))
    y_cf_mag_n = stats_mag.normalise_y(np.concatenate(ys_conf, axis=0))
    conf_mag = ConformalWrapper(ens_mag, stats_mag)
    conf_mag.fit_conformal(X_cf_mag_n, y_cf_mag_n)

    # --- 5. Evaluate on the SAME in-dist-test slices ---
    y_all = np.concatenate(ys_idt, axis=0)
    y_1d = y_all[:, 0] if y_all.ndim == 2 else y_all  # (N,)

    # Mag-only predictions
    X_idt_mag_n = stats_mag.normalise_X(np.concatenate(Xs_idt_mag, axis=0))
    mu_mag_n, sr_mag_n, sc_mag_n, _ = conf_mag.predict_calibrated(X_idt_mag_n)
    mu_mag_p = stats_mag.denormalise_y_mean(mu_mag_n)[:, 0]
    sr_mag_p = stats_mag.denormalise_y_std(sr_mag_n)[:, 0]
    sc_mag_p = stats_mag.denormalise_y_std(sc_mag_n)[:, 0]

    # Mag+ane predictions on the SAME slices
    X_idt_mane_n = stats_mag_ane.normalise_X(np.concatenate(Xs_idt, axis=0))
    mu_mane_n, sr_mane_n, sc_mane_n, _ = conformal_mag_ane.predict_calibrated(X_idt_mane_n)
    mu_mane_p = stats_mag_ane.denormalise_y_mean(mu_mane_n)[:, 0]
    sr_mane_p = stats_mag_ane.denormalise_y_std(sr_mane_n)[:, 0]
    sc_mane_p = stats_mag_ane.denormalise_y_std(sc_mane_n)[:, 0]

    # All arrays have identical (N,) shape — no truncation needed
    assert len(y_1d) == len(mu_mag_p) == len(mu_mane_p), (
        f"Length mismatch: y={len(y_1d)}, mag={len(mu_mag_p)}, mane={len(mu_mane_p)}"
    )

    return {
        "n_test_slices": int(len(y_1d)),
        "n_train_slices_mag_only": int(sum(len(x) for x in Xs_tr_mag)),
        "mag_only_crps": float(crps_gaussian(y_1d, mu_mag_p, sr_mag_p)),
        "mag_ane_crps": float(crps_gaussian(y_1d, mu_mane_p, sr_mane_p)),
        "crps_lift": float(
            crps_gaussian(y_1d, mu_mag_p, sr_mag_p)
            - crps_gaussian(y_1d, mu_mane_p, sr_mane_p)
        ),
        "mag_only_nll": float(nll_gaussian(y_1d, mu_mag_p, sr_mag_p)),
        "mag_ane_nll": float(nll_gaussian(y_1d, mu_mane_p, sr_mane_p)),
        "nll_lift": float(
            nll_gaussian(y_1d, mu_mag_p, sr_mag_p)
            - nll_gaussian(y_1d, mu_mane_p, sr_mane_p)
        ),
        "mag_only_coverage90": float(interval_coverage(y_1d, mu_mag_p, sc_mag_p, alpha=0.10)),
        "mag_ane_coverage90": float(interval_coverage(y_1d, mu_mane_p, sc_mane_p, alpha=0.10)),
        "method": "column-slice: mag-only = mag+ane X[:, :-1] (same shots/slices)",
        "note": "positive CRPS/NLL lift = mag+ane is better (lower score); negative = mag-only is better",
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """Run baseline pipeline and save results artifact."""
    import argparse  # noqa: PLC0415

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="S7.2 static baseline: deep ensemble + conformal")
    parser.add_argument("--target", choices=["primary", "multi"], default="primary")
    parser.add_argument("--modality", choices=["mag", "mag_ane"], default="mag_ane")
    parser.add_argument("--max-train-shots", type=int, default=2000)
    parser.add_argument("--max-slices", type=int, default=200)
    parser.add_argument("--n-members", type=int, default=5)
    parser.add_argument("--n-epochs", type=int, default=60)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    cfg = BaselineConfig(
        input_modality=args.modality,
        target_mode=args.target,
        max_train_shots=args.max_train_shots,
        max_slices_per_shot=args.max_slices,
        ensemble=EnsembleConfig(
            n_members=args.n_members,
            n_epochs=args.n_epochs,
            hidden_size=args.hidden,
        ),
    )

    result = run_baseline(cfg)

    # Default output path
    if args.output is None:
        artifact_dir = Path(__file__).parent / "artifacts"
        artifact_dir.mkdir(exist_ok=True)
        args.output = artifact_dir / "baseline_metrics_v0.json"

    with open(args.output, "w") as f:
        json.dump(result.to_dict(), f, indent=2, default=float)

    logger.info("Results saved to %s", args.output)

    # Print summary
    print("\n" + "=" * 60)
    print("S7.2 STATIC BASELINE RESULTS")
    print("=" * 60)
    splits = result.to_dict()["split_sizes"]
    for k, v in splits.items():
        print(f"  {k}: {v}")
    print()
    print("IN-DIST-TEST:")
    for k in ["coverage_90_conf", "nll_raw", "crps_raw", "pi_width_90_conf", "ece_conf", "ood_auroc", "energy_score"]:
        v = result.in_dist.get(k)
        if v is not None:
            print(f"  {k}: {v}")
    print()
    print("TRANSIENT vs QUIESCENT (in-dist):")
    for s in ("transient", "quiescent"):
        n = result.in_dist.get(f"{s}_n", "?")
        cov = result.in_dist.get(f"{s}_coverage_90", "?")
        crps = result.in_dist.get(f"{s}_crps", "?")
        print(f"  {s}: n={n}  cov@90={cov}  crps={crps}")
    print()
    print("OOD-REGIME:")
    for k in ["coverage_90_conf", "ood_auroc", "crps_raw", "nll_raw"]:
        v = result.ood.get(k)
        if v is not None:
            print(f"  {k}: {v}")
    print()
    print("ane LIFT:")
    for k in ["mag_only_crps", "mag_ane_crps", "crps_lift", "mag_only_nll", "mag_ane_nll", "nll_lift"]:
        v = result.ane_lift.get(k)
        if v is not None:
            print(f"  {k}: {v:.4f}")
    print()
    for note in result.notes:
        print(f"  NOTE: {note}")


if __name__ == "__main__":
    main()
