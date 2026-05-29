"""Dynamic latent state-space engine (RKN-style) for plasma-state-space-v0.

S7.3 — the novel architectural core.  Proves that modelling DYNAMICS produces
calibrated forecasting THROUGH TRANSIENTS that a per-slice static model cannot.

Architecture (Recurrent Kalman Network, Becker et al. 2019, arXiv:1905.07357)
----------------------------------------------------------------------------
A latent Gaussian belief with a FACTORISED (diagonal) covariance — no matrix
inversions anywhere — propagated by a learned stochastic transition kernel and
updated by a learned encoder, exactly the RKN recipe recommended in
docs/probabilistic-state-space-methods.html §2 (RKN row) and §7.2.

  Encoder       : inputs_t (mag+ane, 122-d) → latent observation w_t (L-d)
                  and its diagonal observation variance r_t (L-d, softplus).
  Belief        : factorised Gaussian N(z_t, diag(σ²_t)) over the L-d latent.
  Update step   : Kalman update with a per-dimension scalar gain
                    k = σ²_prior / (σ²_prior + r)
                    z_post = z_prior + k (w - z_prior)
                    σ²_post = (1 - k) σ²_prior
                  (the RKN diagonal special case — no matrix inverse).
  Predict step  : learned transition f_θ on the mean; variance propagated by a
                  learned linear-in-variance map plus a strictly-positive learned
                  process noise Q.  INPUT-FREE BY CONSTRUCTION — only latent→latent.
                  This is what makes autonomous rollout possible: forecasting is
                  just "filter to t, then run predict-only h times (no updates)".
  Observation   : a head latent z → Dα (μ, σ).  σ propagated from the latent
                  belief variance (linearised) plus a learned emission noise.

The win mechanism for forecasting through transients is the LEARNED PROCESS
NOISE Q: each predict step adds Q, so the predictive variance grows with the
horizon → calibrated WIDENING through transients.  A per-slice static map has
no such mechanism (its variance at horizon h is its h=0 variance), so it is
confidently-wrong-narrow through ELMs.  The legitimate CRPS win is calibration,
not spike-timing accuracy.

Training objective — MULTI-STEP ROLLOUT NLL (critical)
------------------------------------------------------
We do NOT train one-step filtering NLL alone: that does not shape the
propagated rollout variance and forecast CRPS would be poor even with a correct
architecture.  Instead, for each minibatch sequence we filter to an anchor t,
then roll the predict step forward for each h in a horizon set and accumulate
the Gaussian NLL of Dα_{t+h} from the propagated belief.  This is what makes Q
match the rollout error and the forecast intervals calibrated.

Design-for-invariance (light)
-----------------------------
The encoder consumes per-channel z-scored inputs (ChannelStats from baseline.py)
rather than raw machine-absolute scales, and the latent is dimensionless — so a
later move to dimensionless / invariant coordinates (Stage-2) is not precluded.
v0 does not test cross-machine.

CPU-first: the model is tiny (latent ≤ 32) and I/O dominates; CUDA buys nothing
and adds node-drain risk (AGENTS.md §2a).  Runs on CPU on the data-access host.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

logger = logging.getLogger(__name__)

# Reproducibility / determinism (AGENTS.md §2b)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.set_float32_matmul_precision("high")

_VAR_FLOOR = 1e-6  # floor on every variance for numerical stability
_VAR_CEIL = 1e6


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class EngineConfig:
    """RKN engine hyper-parameters."""

    input_dim: int = 122          # mag (ama+amb+amc) + ane
    latent_dim: int = 16          # RKN latent size (sweep [8,16,32])
    output_dim: int = 1           # Dα channels (1 = primary, 5 = multi)
    enc_hidden: int = 128         # encoder MLP hidden size
    trans_hidden: int = 64        # transition MLP hidden size
    obs_hidden: int = 64          # observation head hidden size
    # Training
    n_epochs: int = 30
    batch_size: int = 64          # sequences per batch
    seq_len: int = 64             # timesteps per training sequence (1 kHz → 64 ms)
    lr: float = 1e-3
    weight_decay: float = 1e-5
    # Multi-step rollout horizons used in the training NLL (steps at 1 kHz)
    train_horizons: tuple[int, ...] = (1, 2, 5, 10, 20)
    # Weight of the one-step filtering NLL term (keeps the encoder/update honest)
    filter_loss_weight: float = 1.0
    # Weight of the rollout NLL term (the forecasting objective)
    rollout_loss_weight: float = 1.0
    grad_clip: float = 5.0
    seed: int = 0


# ---------------------------------------------------------------------------
# RKN engine module
# ---------------------------------------------------------------------------


class RKNEngine(nn.Module):
    """RKN-style latent state-space model with a factorised Gaussian belief.

    All beliefs are diagonal: a mean ``z`` (B, L) and a variance ``var`` (B, L).
    No matrix inversions are performed anywhere.
    """

    def __init__(self, cfg: EngineConfig) -> None:
        super().__init__()
        self.cfg = cfg
        L = cfg.latent_dim  # noqa: N806

        # Encoder: inputs_t -> (latent observation w, log obs-variance)
        self.encoder = nn.Sequential(
            nn.Linear(cfg.input_dim, cfg.enc_hidden),
            nn.ReLU(),
            nn.Linear(cfg.enc_hidden, cfg.enc_hidden),
            nn.ReLU(),
        )
        self.enc_w = nn.Linear(cfg.enc_hidden, L)        # latent observation mean
        self.enc_logr = nn.Linear(cfg.enc_hidden, L)     # log obs variance

        # Transition: mean update f_theta(z) -> delta z (residual / locally-linear)
        self.trans_mean = nn.Sequential(
            nn.Linear(L, cfg.trans_hidden),
            nn.Tanh(),
            nn.Linear(cfg.trans_hidden, L),
        )
        # Variance transition: per-dim multiplicative factor on prior variance
        # (a in [0, ~]) — learned, input-free.  log-parameterised for positivity.
        self.trans_log_a = nn.Parameter(torch.zeros(L))   # var <- a^2 * var + Q
        # Learned strictly-positive process noise Q (the WIN mechanism).
        # Init small-but-nonzero so it can grow; never regularised toward 0.
        self.log_q = nn.Parameter(torch.full((L,), math.log(0.05)))

        # Observation head: latent mean -> Dα mean; latent var -> Dα var (linearised)
        self.obs_mean = nn.Sequential(
            nn.Linear(L, cfg.obs_hidden),
            nn.ReLU(),
            nn.Linear(cfg.obs_hidden, cfg.output_dim),
        )
        # Jacobian-free variance map: a learned non-negative linear map from latent
        # variance to output variance, plus a learned emission noise floor.
        self.obs_var_w = nn.Parameter(torch.zeros(L, cfg.output_dim))
        self.log_obs_noise = nn.Parameter(
            torch.full((cfg.output_dim,), math.log(0.1))
        )

        # Initial belief (prior at t=0): learned mean ~0, broad variance.
        self.z0 = nn.Parameter(torch.zeros(L))
        self.log_var0 = nn.Parameter(torch.full((L,), math.log(1.0)))

    # -- belief ops ---------------------------------------------------------

    def initial_belief(self, batch: int, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.z0.to(device=device, dtype=dtype).unsqueeze(0).expand(batch, -1)
        var = (
            self.log_var0.exp()
            .to(device=device, dtype=dtype)
            .unsqueeze(0)
            .expand(batch, -1)
        )
        return z.contiguous(), var.contiguous()

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """inputs_t (B, F) -> latent observation w (B, L), obs variance r (B, L)."""
        h = self.encoder(x)
        w = self.enc_w(h)
        r = F.softplus(self.enc_logr(h)) + _VAR_FLOOR
        return w, r

    def update_step(
        self,
        z_prior: torch.Tensor,
        var_prior: torch.Tensor,
        w: torch.Tensor,
        r: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Diagonal Kalman update (RKN special case — no matrix inverse).

        k = var_prior / (var_prior + r)         (per-dim scalar gain)
        z_post = z_prior + k (w - z_prior)
        var_post = (1 - k) var_prior
        """
        k = var_prior / (var_prior + r)
        z_post = z_prior + k * (w - z_prior)
        var_post = (1.0 - k) * var_prior
        var_post = var_post.clamp(_VAR_FLOOR, _VAR_CEIL)
        return z_post, var_post

    def predict_step(
        self, z: torch.Tensor, var: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Learned transition (predict).  INPUT-FREE: latent -> latent only.

        z_{t+1}   = z_t + f_theta(z_t)
        var_{t+1} = a^2 var_t + Q          (a, Q learned, Q strictly positive)

        Each call adds Q → variance grows with the number of predict steps →
        calibrated widening through transients during autonomous rollout.
        """
        z_next = z + self.trans_mean(z)
        a2 = self.trans_log_a.exp().pow(2.0)
        q = self.log_q.exp() + _VAR_FLOOR
        var_next = a2.unsqueeze(0) * var + q.unsqueeze(0)
        var_next = var_next.clamp(_VAR_FLOOR, _VAR_CEIL)
        return z_next, var_next

    def observe(
        self, z: torch.Tensor, var: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Latent belief -> Dα predictive (mean, variance).

        mean = g_theta(z)
        var  = softplus(W)^T var_latent + emission_noise^2     (>= 0)
        """
        mu = self.obs_mean(z)
        w_pos = F.softplus(self.obs_var_w)               # (L, D) non-negative
        out_var = var @ w_pos                            # (B, D)
        out_var = out_var + self.log_obs_noise.exp().pow(2.0).unsqueeze(0)
        out_var = out_var.clamp(_VAR_FLOOR, _VAR_CEIL)
        return mu, out_var

    # -- sequence ops -------------------------------------------------------

    def filter_sequence(
        self,
        x_seq: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Causal filtering over a sequence.

        Parameters
        ----------
        x_seq : (B, T, F) inputs.

        Returns
        -------
        z_post : (B, T, L)  posterior latent mean after the update at each t.
        var_post : (B, T, L)  posterior latent variance after the update at each t.
        obs_mu : (B, T, D)  observation-head mean from the posterior belief.
        obs_var : (B, T, D)  observation-head variance from the posterior belief.

        At each t: predict from the previous posterior, encode inputs_t, update.
        The belief at t therefore uses inputs_{1:t} only (causal).
        """
        B, T, _ = x_seq.shape  # noqa: N806
        device, dtype = x_seq.device, x_seq.dtype
        z, var = self.initial_belief(B, device, dtype)

        z_list, var_list, mu_list, ov_list = [], [], [], []
        for t in range(T):
            if t > 0:
                z, var = self.predict_step(z, var)
            w, r = self.encode(x_seq[:, t, :])
            z, var = self.update_step(z, var, w, r)
            mu, ov = self.observe(z, var)
            z_list.append(z)
            var_list.append(var)
            mu_list.append(mu)
            ov_list.append(ov)

        z_post = torch.stack(z_list, dim=1)
        var_post = torch.stack(var_list, dim=1)
        obs_mu = torch.stack(mu_list, dim=1)
        obs_var = torch.stack(ov_list, dim=1)
        return z_post, var_post, obs_mu, obs_var

    def rollout(
        self,
        z_anchor: torch.Tensor,
        var_anchor: torch.Tensor,
        horizons: tuple[int, ...] | list[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """PURE AUTONOMOUS ROLLOUT from an anchor belief.

        Runs the predict step (input-free, no updates, no Dα observation) for
        max(horizons) steps and reads out the observation head at each requested
        horizon.  NO future inputs, NO Dα peeking — by construction the only
        thing fed forward is the latent belief.

        Parameters
        ----------
        z_anchor, var_anchor : (B, L) belief at the anchor time t.
        horizons : iterable of positive step counts.

        Returns
        -------
        mu : (B, H, D)   forecast mean at each horizon.
        var : (B, H, D)  forecast variance at each horizon (grows with h via Q).
        """
        hs = list(horizons)
        h_max = max(hs)
        want = {h: i for i, h in enumerate(sorted(hs))}
        z, var = z_anchor, var_anchor
        out_mu: dict[int, torch.Tensor] = {}
        out_var: dict[int, torch.Tensor] = {}
        for step in range(1, h_max + 1):
            z, var = self.predict_step(z, var)
            if step in want:
                mu, ov = self.observe(z, var)
                out_mu[step] = mu
                out_var[step] = ov
        order = sorted(hs)
        mu = torch.stack([out_mu[h] for h in order], dim=1)   # (B, H, D)
        ov = torch.stack([out_var[h] for h in order], dim=1)  # (B, H, D)
        return mu, ov


# ---------------------------------------------------------------------------
# Gaussian NLL (matches calibration.nll_gaussian convention)
# ---------------------------------------------------------------------------


def gaussian_nll(y: torch.Tensor, mu: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
    """Mean Gaussian NLL = 0.5 [log(2π var) + (y-μ)²/var]."""
    var = var.clamp(_VAR_FLOOR, _VAR_CEIL)
    return 0.5 * (torch.log(2.0 * math.pi * var) + (y - mu) ** 2 / var).mean()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


@dataclass
class TrainState:
    """Lightweight training-progress record."""

    epoch_losses: list[float] = field(default_factory=list)
    epoch_filter_nll: list[float] = field(default_factory=list)
    epoch_rollout_nll: list[float] = field(default_factory=list)
    seconds: float = 0.0


def _sample_anchor_rollout_loss(
    model: RKNEngine,
    z_post: torch.Tensor,
    var_post: torch.Tensor,
    y_seq: torch.Tensor,
    horizons: tuple[int, ...],
    rng: torch.Generator,
) -> torch.Tensor:
    """Multi-step rollout NLL.

    For each sequence in the batch, pick the SAME anchor index t such that
    t + max(h) < T, take the FILTERED posterior belief at t (uses inputs_{1:t}
    only), roll the predict step forward, and accumulate Gaussian NLL of the
    true Dα_{t+h} against the propagated forecast.
    """
    B, T, _ = z_post.shape  # noqa: N806
    h_max = max(horizons)
    if T - h_max - 1 < 1:
        return z_post.new_zeros(())
    # Sample one anchor per batch step (shared across batch for vectorisation).
    t_anchor = int(torch.randint(0, T - h_max, (1,), generator=rng).item())
    z_a = z_post[:, t_anchor, :]
    var_a = var_post[:, t_anchor, :]
    mu, var = model.rollout(z_a, var_a, horizons)  # (B, H, D)
    losses = []
    for i, h in enumerate(sorted(horizons)):
        y_h = y_seq[:, t_anchor + h, :]            # (B, D)
        losses.append(gaussian_nll(y_h, mu[:, i, :], var[:, i, :]))
    return torch.stack(losses).mean()


def train_engine(
    model: RKNEngine,
    x_train: list[np.ndarray],
    y_train: list[np.ndarray],
    cfg: EngineConfig,
    device: str = "cpu",
    stop_flag=None,
) -> TrainState:
    """Train the RKN engine with combined filtering + multi-step rollout NLL.

    Parameters
    ----------
    x_train, y_train : lists of per-shot arrays, ALREADY NORMALISED.
        x : (T_shot, F), y : (T_shot, D).  Each array must be a single
        contiguous 1 kHz run (the sequence builder guarantees this).
    cfg : EngineConfig.
    stop_flag : optional callable returning True to request a clean early exit
        (STOP-FILE / soft-time-limit contract, AGENTS.md §2a-cancel).

    Returns
    -------
    TrainState with per-epoch losses.
    """
    model = model.to(device)
    model.train()
    opt = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    torch.manual_seed(cfg.seed)
    rng = torch.Generator()
    rng.manual_seed(cfg.seed + 7)

    # Build fixed-length training windows (contiguous slices of each shot run).
    windows_x, windows_y = _build_training_windows(
        x_train, y_train, cfg.seq_len, seed=cfg.seed
    )
    if not windows_x:
        raise RuntimeError(
            f"No training windows of length {cfg.seq_len} — shots too short."
        )
    X = torch.from_numpy(np.stack(windows_x)).float()  # (N, T, F)
    Y = torch.from_numpy(np.stack(windows_y)).float()  # (N, T, D)
    n = X.shape[0]
    logger.info(
        "Engine training: %d windows of length %d (latent=%d, F=%d, D=%d)",
        n, cfg.seq_len, cfg.latent_dim, cfg.input_dim, cfg.output_dim,
    )

    state = TrainState()
    t0 = time.time()
    np_rng = np.random.default_rng(cfg.seed + 11)
    for epoch in range(cfg.n_epochs):
        if stop_flag is not None and stop_flag():
            logger.info("[engine] STOP-FILE / soft-limit → clean exit at epoch %d", epoch)
            break
        perm = np_rng.permutation(n)
        ep_loss = ep_filt = ep_roll = 0.0
        nb = 0
        for start in range(0, n, cfg.batch_size):
            idx = perm[start : start + cfg.batch_size]
            xb = X[idx].to(device)
            yb = Y[idx].to(device)

            z_post, var_post, obs_mu, obs_var = model.filter_sequence(xb)
            filt = gaussian_nll(yb, obs_mu, obs_var)
            roll = _sample_anchor_rollout_loss(
                model, z_post, var_post, yb, cfg.train_horizons, rng
            )
            loss = cfg.filter_loss_weight * filt + cfg.rollout_loss_weight * roll

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()

            ep_loss += float(loss.item())
            ep_filt += float(filt.item())
            ep_roll += float(roll.item())
            nb += 1
        state.epoch_losses.append(ep_loss / max(nb, 1))
        state.epoch_filter_nll.append(ep_filt / max(nb, 1))
        state.epoch_rollout_nll.append(ep_roll / max(nb, 1))
        if epoch % 5 == 0 or epoch == cfg.n_epochs - 1:
            logger.info(
                "  epoch %3d/%d  loss=%.4f  filt_nll=%.4f  roll_nll=%.4f  (%.1fs)",
                epoch, cfg.n_epochs, state.epoch_losses[-1],
                state.epoch_filter_nll[-1], state.epoch_rollout_nll[-1],
                time.time() - t0,
            )
    state.seconds = time.time() - t0
    model.eval()
    return state


def _build_training_windows(
    x_list: list[np.ndarray],
    y_list: list[np.ndarray],
    seq_len: int,
    max_windows_per_shot: int = 8,
    seed: int = 0,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Slice each contiguous shot run into fixed-length training windows.

    Non-overlapping windows of length ``seq_len`` are taken from each run; if a
    run yields more than ``max_windows_per_shot``, a random subset is kept (so a
    handful of very long shots do not dominate the batch distribution).
    """
    rng = np.random.default_rng(seed)
    wx: list[np.ndarray] = []
    wy: list[np.ndarray] = []
    for x, y in zip(x_list, y_list, strict=True):
        T = x.shape[0]
        if T < seq_len:
            continue
        starts = list(range(0, T - seq_len + 1, seq_len))
        if len(starts) > max_windows_per_shot:
            sel = rng.choice(len(starts), size=max_windows_per_shot, replace=False)
            starts = [starts[i] for i in sorted(sel)]
        for s in starts:
            wx.append(x[s : s + seq_len])
            wy.append(y[s : s + seq_len])
    return wx, wy
