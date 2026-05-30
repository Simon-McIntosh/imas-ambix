"""Filtering / forecasting / smoothing inference for the RKN engine (S7.3).

Three inference modes over the latent state-space engine in ``engine.py``:

  1. FILTERING   — causal belief update; estimate Dα_t from inputs_{1:t}.
  2. FORECASTING — PURE AUTONOMOUS ROLLOUT: encode inputs up to t, then roll the
                   transition kernel forward h steps WITHOUT seeing inputs_{t+1..}
                   and WITHOUT observing Dα; push the propagated belief through the
                   observation head → Dα_{t+h} with propagated uncertainty.
  3. SMOOTHING   — OPTIONAL: backward RTS-style pass for the best trajectory
                   ("for later discovery" per plan §3).

These wrap the module methods (``filter_sequence``, ``rollout``) and return
numpy arrays so the calibration harness (``calibration.py``) can score them.

Conventions (match baseline.py so the comparison is clean)
----------------------------------------------------------
- Inputs / targets are normalised with the SAME ``ChannelStats`` as the static
  comparator; predictions are denormalised back to physical Dα units before
  scoring, so CRPS/NLL are in the same units as S7.2's 0.334 / 0.634.
- CRPS / NLL are reported on RAW predictive σ (matching baseline ``_eval_split``);
  coverage / PI-width use per-horizon split-conformal σ.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import torch

from imas_ambix.statespace.engine import RKNEngine

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


@torch.no_grad()
def filter_shot(
    model: RKNEngine,
    x_norm: np.ndarray,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Causal filtering over one shot run.

    Parameters
    ----------
    x_norm : (T, F) normalised inputs for a single contiguous run.

    Returns
    -------
    mu : (T, D)  filtered Dα mean (normalised space).
    var : (T, D) filtered Dα variance (normalised space).
    """
    model.eval()
    xb = torch.from_numpy(x_norm[np.newaxis]).float().to(device)  # (1, T, F)
    _z, _v, obs_mu, obs_var = model.filter_sequence(xb)
    return obs_mu[0].cpu().numpy(), obs_var[0].cpu().numpy()


@torch.no_grad()
def filter_innovation_shot(
    model: RKNEngine,
    x_norm: np.ndarray,
    device: str = "cpu",
) -> np.ndarray:
    """Per-slice engine-native OOD score (normalised filter-innovation magnitude).

    Wraps ``RKNEngine.filter_innovation`` for one contiguous run.  Returns (T,):
    the Mahalanobis innovation Σ_d (w−z_prior)²/(var_prior+r) at each timestep.
    Larger = the operating point surprises the learned dynamics → more OOD.
    """
    model.eval()
    xb = torch.from_numpy(x_norm[np.newaxis]).float().to(device)  # (1, T, F)
    s = model.filter_innovation(xb)  # (1, T)
    return s[0].cpu().numpy()


# ---------------------------------------------------------------------------
# Forecasting — pure autonomous rollout
# ---------------------------------------------------------------------------


@torch.no_grad()
def forecast_pairs(
    model: RKNEngine,
    x_norm: np.ndarray,
    anchors: np.ndarray,
    horizons: list[int],
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Autonomous h-step forecasts from a set of anchor times.

    For each anchor t the model is filtered over inputs_{1:t} (inclusive), and
    then the predict step is rolled forward h steps WITHOUT any further inputs
    and WITHOUT observing Dα.  This is the v0 pure-autonomous-dynamics forecast:
    no future inputs, no xdc conditioning, no Dα peeking.

    Parameters
    ----------
    x_norm : (T, F) normalised inputs for one contiguous run.
    anchors : (A,) int array of anchor indices t (0-based).  For each, every
        ``t + h`` must be < T.
    horizons : list of positive step counts.

    Returns
    -------
    mu : (A, H, D)  forecast mean (normalised space).
    var : (A, H, D) forecast variance (normalised space).
    """
    model.eval()
    T = x_norm.shape[0]
    h_max = max(horizons)
    xb = torch.from_numpy(x_norm[np.newaxis]).float().to(device)  # (1, T, F)

    # One causal filter pass gives the posterior belief at every t.  The belief
    # at t uses inputs_{1:t} only, so anchoring there is leak-free.
    z_post, var_post, _mu, _ov = model.filter_sequence(xb)  # (1, T, L)

    valid = [int(a) for a in anchors if int(a) + h_max < T]
    if not valid:
        D = model.cfg.output_dim
        return np.empty((0, len(horizons), D)), np.empty((0, len(horizons), D))

    z_a = z_post[0, valid, :]  # (A, L)
    var_a = var_post[0, valid, :]  # (A, L)
    mu, var = model.rollout(z_a, var_a, tuple(horizons))  # (A, H, D)
    return mu.cpu().numpy(), var.cpu().numpy()


# ---------------------------------------------------------------------------
# Smoothing (optional — RTS-style backward pass on the diagonal belief)
# ---------------------------------------------------------------------------


@torch.no_grad()
def smooth_shot(
    model: RKNEngine,
    x_norm: np.ndarray,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Diagonal RTS smoother over one shot run (optional mode).

    A factorised (per-dimension) Rauch-Tung-Striebel smoother: a forward filter
    pass is followed by a backward pass that blends each filtered belief with the
    one-step-ahead prediction using a per-dimension smoother gain derived from the
    learned variance-transition factor a².  Returns smoothed Dα (mean, var).

    This is the "best trajectory, for later discovery" mode (plan §3); it does not
    enter the acceptance gate.
    """
    model.eval()
    xb = torch.from_numpy(x_norm[np.newaxis]).float().to(device)
    z_f, var_f, _mu, _ov = model.filter_sequence(xb)  # filtered (1, T, L)
    z_f = z_f[0]
    var_f = var_f[0]
    T, L = z_f.shape

    a2 = model.trans_log_a.exp().pow(2.0).to(z_f.device)  # (L,)
    q = model.log_q.exp().to(z_f.device)  # (L,)

    z_s = z_f.clone()
    var_s = var_f.clone()
    for t in range(T - 2, -1, -1):
        # one-step prediction from filtered belief at t
        z_pred = z_f[t] + model.trans_mean(z_f[t : t + 1])[0]
        var_pred = (a2 * var_f[t] + q).clamp(1e-6, 1e6)
        # diagonal smoother gain: C = a² var_f / var_pred  (linearised transition)
        gain = a2 * var_f[t] / var_pred
        z_s[t] = z_f[t] + gain * (z_s[t + 1] - z_pred)
        var_s[t] = var_f[t] + gain * gain * (var_s[t + 1] - var_pred)
    var_s = var_s.clamp(1e-6, 1e6)

    # push smoothed beliefs through the observation head
    mu, ov = model.observe(z_s, var_s)
    return mu.cpu().numpy(), ov.cpu().numpy()


# ---------------------------------------------------------------------------
# Latent-surfacing variants (T7 discovery track — additive; no change to above)
# ---------------------------------------------------------------------------


@torch.no_grad()
def filter_shot_latents(
    model: RKNEngine,
    x_norm: np.ndarray,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Causal filtering — same as filter_shot but returns the LATENT trajectories.

    A strict superset of :func:`filter_shot`: the existing public function is
    unchanged and its return contract is unchanged.  This companion surfaces the
    posterior latent means / variances that ``filter_shot`` previously discarded
    (the ``_z, _v`` stubs in the original).

    Parameters
    ----------
    x_norm : (T, F) normalised inputs for a single contiguous run.

    Returns
    -------
    z_post : (T, L)    filtered posterior latent mean at each timestep.
    var_post : (T, L)  filtered posterior latent variance at each timestep.
    """
    model.eval()
    xb = torch.from_numpy(x_norm[np.newaxis]).float().to(device)  # (1, T, F)
    z_post, var_post, _obs_mu, _obs_var = model.filter_sequence(xb)
    return z_post[0].cpu().numpy(), var_post[0].cpu().numpy()


@torch.no_grad()
def smooth_shot_latents(
    model: RKNEngine,
    x_norm: np.ndarray,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """RTS smoother — same backward pass as smooth_shot but returns the LATENT trajectories.

    A strict superset of :func:`smooth_shot`: the existing public function is
    unchanged.  This companion exposes the per-timestep smoothed (and filtered)
    latent means / variances that the discovery track (T7+) needs, without
    duplicating the backward pass.

    Parameters
    ----------
    x_norm : (T, F) normalised inputs for a single contiguous run.

    Returns
    -------
    z_f   : (T, L)  filtered posterior latent mean  (from the forward pass).
    var_f : (T, L)  filtered posterior latent variance (from the forward pass).
    z_s   : (T, L)  RTS-smoothed latent mean  (backward pass output).
    var_s : (T, L)  RTS-smoothed latent variance (backward pass output).
    """
    model.eval()
    xb = torch.from_numpy(x_norm[np.newaxis]).float().to(device)
    z_filt, var_filt, _mu, _ov = model.filter_sequence(xb)  # filtered (1, T, L)
    z_f = z_filt[0]
    var_f = var_filt[0]
    T, L = z_f.shape

    a2 = model.trans_log_a.exp().pow(2.0).to(z_f.device)  # (L,)
    q = model.log_q.exp().to(z_f.device)  # (L,)

    z_s = z_f.clone()
    var_s = var_f.clone()
    for t in range(T - 2, -1, -1):
        z_pred = z_f[t] + model.trans_mean(z_f[t : t + 1])[0]
        var_pred = (a2 * var_f[t] + q).clamp(1e-6, 1e6)
        gain = a2 * var_f[t] / var_pred
        z_s[t] = z_f[t] + gain * (z_s[t + 1] - z_pred)
        var_s[t] = var_f[t] + gain * gain * (var_s[t + 1] - var_pred)
    var_s = var_s.clamp(1e-6, 1e6)

    return (
        z_f.cpu().numpy(),
        var_f.cpu().numpy(),
        z_s.cpu().numpy(),
        var_s.cpu().numpy(),
    )


# ---------------------------------------------------------------------------
# Per-horizon split-conformal calibration
# ---------------------------------------------------------------------------


def fit_horizon_conformal(
    y_cal: np.ndarray,
    mu_cal: np.ndarray,
    sigma_cal: np.ndarray,
    alpha: float = 0.10,
) -> float:
    """Split-conformal scale q̂ for one horizon (normalised-residual score).

    Returns q̂ such that ``μ ± q̂ σ`` achieves marginal (1-α) coverage on the
    calibration residuals.  Mirrors baseline.ConformalWrapper.fit_conformal.
    """
    sigma = np.maximum(np.asarray(sigma_cal), 1e-12)
    scores = np.abs(np.asarray(y_cal) - np.asarray(mu_cal)) / sigma
    scores = scores.reshape(-1)
    n = len(scores)
    if n == 0:
        return 1.0
    level = math.ceil((n + 1) * (1.0 - alpha)) / n
    level = min(level, 1.0)
    return float(np.quantile(scores, level, method="higher"))
