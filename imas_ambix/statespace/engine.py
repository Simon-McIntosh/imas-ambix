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

import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

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

        Performance: the encoder is input-only and time-independent, so it is
        applied to ALL timesteps in ONE batched matmul OUTSIDE the sequential
        loop (the loop keeps only the cheap elementwise predict/update on the
        diagonal belief).  This is mathematically identical to encoding inside
        the loop — the belief at t still accumulates only encodings 1..t — but
        collapses the encoder sub-graph from T node-sets to 1, giving a large
        forward+backward speed-up on the overhead-bound CPU path.  The
        observation head is likewise batched after the scan.  Causality and the
        no-future-input invariant are unchanged (verified by the engine tests).
        """
        B, T, _ = x_seq.shape  # noqa: N806
        device, dtype = x_seq.device, x_seq.dtype
        z, var = self.initial_belief(B, device, dtype)

        # Batched encode over all timesteps (one matmul, no leak — future
        # encodings are computed then only consumed at their own / later t).
        w_all, r_all = self.encode(x_seq.reshape(B * T, x_seq.shape[-1]))
        w_all = w_all.reshape(B, T, -1)
        r_all = r_all.reshape(B, T, -1)

        z_list, var_list = [], []
        for t in range(T):
            if t > 0:
                z, var = self.predict_step(z, var)
            z, var = self.update_step(z, var, w_all[:, t], r_all[:, t])
            z_list.append(z)
            var_list.append(var)

        z_post = torch.stack(z_list, dim=1)    # (B, T, L)
        var_post = torch.stack(var_list, dim=1)
        # Batched observation head over the full posterior trajectory.
        obs_mu, obs_var = self.observe(
            z_post.reshape(B * T, -1), var_post.reshape(B * T, -1)
        )
        obs_mu = obs_mu.reshape(B, T, -1)
        obs_var = obs_var.reshape(B, T, -1)
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
    transient_w: torch.Tensor | None = None,
    n_anchors: int = 2,
) -> torch.Tensor:
    """Multi-step rollout NLL, anchor sampling BIASED toward transients.

    For ``n_anchors`` sampled anchor indices t (each with t + max(h) < T), take
    the FILTERED posterior belief at t (uses inputs_{1:t} only), roll the predict
    step forward, and accumulate Gaussian NLL of the true Dα_{t+h}.

    CRITICAL (acceptance): the engine is *scored* on dense transient windows, so
    the propagated process noise Q must be calibrated to TRANSIENT rollout error,
    not to the ~97%-quiescent average.  We therefore bias the anchor distribution
    toward timesteps whose horizon window straddles ELM activity (``transient_w``
    is a per-window, per-timestep weight ∝ transient mass in [t, t+h_max]).  This
    fixes the otherwise-overconfident-through-transients failure mode WITHOUT
    making the predict step input-aware (the rollout stays purely autonomous).
    """
    B, T, _ = z_post.shape  # noqa: N806
    h_max = max(horizons)
    n_valid = T - h_max
    if n_valid < 1:
        return z_post.new_zeros(())

    # Build the anchor sampling distribution over valid indices [0, n_valid).
    if transient_w is not None:
        w = transient_w[:n_valid].clamp_min(0.0)
        if float(w.sum()) <= 0:
            probs = None
        else:
            # Mixture: 70% transient-weighted + 30% uniform (keeps quiescent
            # dynamics represented so the filter stays honest off-ELM).
            p = w / w.sum()
            u = torch.full_like(p, 1.0 / n_valid)
            probs = 0.7 * p + 0.3 * u
    else:
        probs = None

    if probs is None:
        anchors = torch.randint(0, n_valid, (n_anchors,), generator=rng)
    else:
        anchors = torch.multinomial(probs, n_anchors, replacement=True, generator=rng)

    h_sorted = sorted(horizons)
    losses = []
    for t_anchor in anchors.tolist():
        t_anchor = int(t_anchor)
        z_a = z_post[:, t_anchor, :]
        var_a = var_post[:, t_anchor, :]
        mu, var = model.rollout(z_a, var_a, horizons)  # (B, H, D)
        for i, h in enumerate(h_sorted):
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
    # Per-window, per-timestep transient weight ∝ ELM mass in [t, t+h_max].
    # Used to bias rollout-anchor sampling toward transients (acceptance fix).
    W = _build_transient_weights(windows_y, max(cfg.train_horizons))  # (N, T)
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

            wb = W[idx].to(device)            # (B, T) transient weights
            z_post, var_post, obs_mu, obs_var = model.filter_sequence(xb)
            filt = gaussian_nll(yb, obs_mu, obs_var)
            # Batch-mean transient weight per timestep drives anchor sampling.
            roll = _sample_anchor_rollout_loss(
                model, z_post, var_post, yb, cfg.train_horizons, rng,
                transient_w=wb.mean(dim=0),
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
        if seq_len > T:
            continue
        starts = list(range(0, T - seq_len + 1, seq_len))
        if len(starts) > max_windows_per_shot:
            sel = rng.choice(len(starts), size=max_windows_per_shot, replace=False)
            starts = [starts[i] for i in sorted(sel)]
        for s in starts:
            wx.append(x[s : s + seq_len])
            wy.append(y[s : s + seq_len])
    return wx, wy


def _build_transient_weights(
    windows_y: list[np.ndarray], h_max: int
) -> torch.Tensor:
    """Per-window, per-timestep weight ∝ transient (ELM) mass in [t, t+h_max].

    Uses baseline.compute_transient_mask on each window's Dα to flag ELM-active
    timesteps, then for each anchor t sums the flag over its forecast horizon
    window [t, t+h_max].  An anchor whose forecast window contains an ELM gets a
    high weight; a purely quiescent anchor gets ~0 (plus the uniform floor added
    in the sampler).  Returns (N, T) float weights.
    """
    from imas_ambix.statespace.baseline import compute_transient_mask  # noqa: PLC0415

    n = len(windows_y)
    t_len = windows_y[0].shape[0]
    w = np.zeros((n, t_len), dtype=np.float32)
    for i, y in enumerate(windows_y):
        tm = compute_transient_mask(y).astype(np.float32)  # (T,)
        # cumulative transient mass over the forecast window [t, t+h_max]
        for t in range(t_len):
            hi = min(t_len, t + h_max + 1)
            w[i, t] = tm[t:hi].sum()
    return torch.from_numpy(w)


# ===========================================================================
# Data pipeline — contiguous 1 kHz runs (shared selector for BOTH models)
# ===========================================================================
#
# The acceptance comparison requires that the engine and the static comparator
# see the EXACT SAME slices, (t, t+h) pairs, normalisation, and information set.
# This pipeline is the single source of truth for that: it loads each shot as
# one or more CONTIGUOUS 1 kHz runs (splitting at any time gap the plasma-on
# mask introduces — otherwise the transition kernel would silently mismodel a
# non-uniform dt), drops dead/disconnected filterscopes (all-zero Dα), and
# caches the result to /work scratch so the corpus is loaded once and reused by
# the engine training, the static retrain, filtering eval, forecasting eval, and
# the dense transient comparison.

_SCRATCH = Path("/work/projects/imas_gpu/mast/scratch/statespace_v0")
_SPLITS_MANIFEST = Path(
    "/work/projects/imas_gpu/mast/manifests/statespace_splits_dalpha_v0.json"
)

_DT_NOMINAL = 1.0e-3          # 1 kHz model grid
_DT_TOL_FRAC = 0.2           # split a run where |dt - dt_med| > 20% dt_med
_MIN_RUN_LEN = 32            # discard contiguous runs shorter than this
_DEAD_DALPHA_STD = 1e-6      # Dα std below this → dead/disconnected filterscope


@dataclass
class ShotRun:
    """One contiguous 1 kHz run from a shot (the unit the engine consumes)."""

    shot_id: int
    X: np.ndarray  # (T, F) raw (un-normalised) inputs
    y: np.ndarray  # (T, D) raw Dα target
    times: np.ndarray  # (T,) seconds


def _split_into_runs(
    X: np.ndarray, y: np.ndarray, times: np.ndarray, shot_id: int
) -> list[ShotRun]:
    """Split a shot's slices into maximal contiguous 1 kHz runs.

    ``load_shot_slices`` applies a threshold-only plasma-on mask that can glue
    across time holes, so consecutive samples are NOT guaranteed to be 1 ms
    apart.  We cut wherever dt deviates from the nominal 1 ms by more than the
    tolerance, and keep only runs of at least ``_MIN_RUN_LEN`` samples.
    """
    if len(times) < _MIN_RUN_LEN:
        return []
    dt = np.diff(times)
    med = float(np.median(dt)) if len(dt) else _DT_NOMINAL
    if not np.isfinite(med) or med <= 0:
        med = _DT_NOMINAL
    cuts = np.where(np.abs(dt - med) > _DT_TOL_FRAC * med)[0]
    idx_runs = np.split(np.arange(len(times)), cuts + 1)
    runs: list[ShotRun] = []
    for idx in idx_runs:
        if len(idx) < _MIN_RUN_LEN:
            continue
        runs.append(
            ShotRun(
                shot_id=shot_id,
                X=X[idx].astype(np.float32),
                y=y[idx].astype(np.float32),
                times=times[idx].astype(np.float64),
            )
        )
    return runs


def _load_split_runs(
    shot_ids: list[int],
    feature_schema: dict,
    target_channels: list[str],
    level1_dir: Path,
    max_shots: int | None,
    seed: int,
    cache_tag: str,
) -> list[ShotRun]:
    """Load a list of shots as contiguous runs, with /work scratch caching.

    Dead filterscopes (Dα std < floor) are dropped — this auto-applies to BOTH
    models because the runs are the shared substrate.
    """
    from imas_ambix.statespace.baseline import load_shot_slices  # noqa: PLC0415

    cache = _SCRATCH / f"runs_{cache_tag}_n{max_shots}_s{seed}.npz"
    if cache.exists():
        logger.info("Loading cached runs from %s", cache)
        data = np.load(cache, allow_pickle=True)
        runs = [
            ShotRun(int(s), x, y, t)
            for s, x, y, t in zip(
                data["shot_ids"], data["Xs"], data["ys"], data["times"], strict=True
            )
        ]
        logger.info("  %d cached runs", len(runs))
        return runs

    sids = list(shot_ids)
    if max_shots is not None and max_shots < len(sids):
        rng = np.random.default_rng(seed)
        sel = rng.choice(len(sids), size=max_shots, replace=False)
        sids = [sids[i] for i in sorted(sel)]

    runs: list[ShotRun] = []
    n_none = n_dead = n_ok = 0
    t0 = time.time()
    for k, sid in enumerate(sids):
        r = load_shot_slices(
            int(sid), feature_schema, target_channels,
            level1_dir=level1_dir, max_slices=None,
        )
        if r is None:
            n_none += 1
            continue
        X, y, times, _pon = r
        if float(np.std(y[:, 0])) < _DEAD_DALPHA_STD:
            n_dead += 1
            continue
        shot_runs = _split_into_runs(X, y, times, int(sid))
        if shot_runs:
            runs.extend(shot_runs)
            n_ok += 1
        if (k + 1) % 100 == 0:
            logger.info("  loaded %d/%d shots (%.0fs)", k + 1, len(sids), time.time() - t0)
    logger.info(
        "[%s] %d shots → %d runs (%d ok, %d none, %d dead) in %.0fs",
        cache_tag, len(sids), len(runs), n_ok, n_none, n_dead, time.time() - t0,
    )

    # Cache as object arrays (variable-length runs)
    _SCRATCH.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache,
        shot_ids=np.array([r.shot_id for r in runs]),
        Xs=np.array([r.X for r in runs], dtype=object),
        ys=np.array([r.y for r in runs], dtype=object),
        times=np.array([r.times for r in runs], dtype=object),
    )
    return runs


# ---------------------------------------------------------------------------
# Static comparator at horizon h>0 (frozen-input / persistence-of-nowcast)
# ---------------------------------------------------------------------------
#
# The S7.2 baseline is an instantaneous map inputs_t → Dα_t.  The static
# comparator at horizon h predicts Dα_{t+h} = static_map(inputs_t): the same
# deep-ensemble + conformal architecture (baseline.py classes) applied to inputs
# at time t.  This gives the static model the SAME information set as the engine
# (inputs up to t only).  NOTE: the static comparator here is RETRAINED on the
# SAME dense contiguous runs as the engine (NOT on S7.2's linspace-decimated
# slices) — so it does not literally reproduce the committed 0.334/0.634 numbers.
# That is deliberate: an identical dense train set + identical eval windows is the
# only clean head-to-head; the committed S7.2 numbers are sanity references, per
# the task spec, not the bar.  Only the eval *targets* shift by h (frozen nowcast).


# ---------------------------------------------------------------------------
# Dense transient eval windows (full 1 kHz, around ELM events — no decimation)
# ---------------------------------------------------------------------------


def _dense_transient_anchors(
    run: ShotRun, h_max: int, pad: int = 5
) -> np.ndarray:
    """Anchor indices inside a run whose horizon window straddles ELM activity.

    Uses the Dα activity flag (baseline.compute_transient_mask) on the FULL 1 kHz
    run (NOT linspace-decimated → no ELM aliasing).  An anchor t is selected when
    any of Dα_{t..t+h_max} is flagged transient, so the (t, t+h) forecast pairs
    actually probe transients.  Returns sorted unique anchor indices.
    """
    from imas_ambix.statespace.baseline import compute_transient_mask  # noqa: PLC0415

    T = run.X.shape[0]
    if h_max + pad >= T:
        return np.empty(0, dtype=int)
    tmask = compute_transient_mask(run.y)  # (T,)
    anchors = []
    for t in range(pad, T - h_max):
        if tmask[t : t + h_max + 1].any():
            anchors.append(t)
    return np.array(sorted(set(anchors)), dtype=int)


# ===========================================================================
# Experiment runner — the STAGE-1 ACCEPTANCE GATE
# ===========================================================================


@dataclass
class ExperimentConfig:
    """Top-level S7.3 experiment configuration."""

    latent_dim: int = 16
    max_train_shots: int = 500     # bound corpus for tractable v0 (reported, not silent)
    max_cal_shots: int = 300
    max_ood_shots: int = 200
    horizons: tuple[int, ...] = (1, 2, 5, 10, 20)  # steps @ 1 kHz
    n_epochs: int = 30
    seq_len: int = 64
    seed: int = 0
    do_smoothing: bool = False     # optional mode, only if budget remains
    device: str = "cpu"


def _stack_runs_for_static(runs: list[ShotRun]) -> tuple[np.ndarray, np.ndarray]:
    """Pool all run slices into (X, y) for the static deep-ensemble (no time)."""
    X = np.concatenate([r.X for r in runs], axis=0)
    y = np.concatenate([r.y for r in runs], axis=0)
    return X, y


def _static_horizon_pairs(
    runs: list[ShotRun], anchors_per_run: list[np.ndarray], horizons: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray]:
    """Build (X_t, Y_{t+h}) arrays for the static comparator at every horizon.

    Returns
    -------
    X_anchor : (P, F)         inputs at the anchor t for each (run, anchor) pair.
    Y_h      : (P, H, D)      true Dα at t+h for each horizon (NaN where OOB).
    The static map predicts Dα_{t+h} = static_map(inputs_t) — SAME inputs_t for
    every horizon, mirroring the engine's information set (inputs up to t only).
    """
    Xs, Ys = [], []
    H = len(horizons)
    for run, anchors in zip(runs, anchors_per_run, strict=True):
        T, D = run.y.shape
        for t in anchors:
            t = int(t)
            Xs.append(run.X[t])
            yh = np.full((H, D), np.nan, dtype=np.float64)
            for i, h in enumerate(sorted(horizons)):
                if t + h < T:
                    yh[i] = run.y[t + h]
            Ys.append(yh)
    if not Xs:
        return np.empty((0, runs[0].X.shape[1])), np.empty((0, H, runs[0].y.shape[1]))
    return np.stack(Xs), np.stack(Ys)


def _engine_horizon_pairs(
    model: RKNEngine,
    runs: list[ShotRun],
    anchors_per_run: list[np.ndarray],
    horizons: tuple[int, ...],
    stats,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Engine forecasts + truths at the SAME (run, anchor, horizon) triples.

    Returns
    -------
    mu_phys : (P, H, D)   forecast mean in physical Dα units.
    var_phys : (P, H, D)  forecast variance in physical Dα units.
    y_phys : (P, H, D)    true Dα at t+h (NaN where out-of-bounds).
    """
    from imas_ambix.statespace.filter import forecast_pairs  # noqa: PLC0415

    H = len(horizons)
    h_sorted = sorted(horizons)
    mus, vars_, ys = [], [], []
    for run, anchors in zip(runs, anchors_per_run, strict=True):
        if len(anchors) == 0:
            continue
        x_norm = stats.normalise_X(run.X.astype(np.float64))
        mu_n, var_n = forecast_pairs(model, x_norm, anchors, list(horizons), device=device)
        if mu_n.shape[0] == 0:
            continue
        # forecast_pairs filters internally; it returns only valid anchors
        # (t + h_max < T).  Re-derive which anchors were valid to align truths.
        T = run.X.shape[0]
        h_max = max(horizons)
        valid = [int(a) for a in anchors if int(a) + h_max < T]
        # Denormalise mean (target std scaling) and variance (std² scaling).
        mu_phys = stats.denormalise_y_mean(
            mu_n.reshape(-1, mu_n.shape[-1])
        ).reshape(mu_n.shape)
        var_phys = var_n * (stats.target_std ** 2)[np.newaxis, np.newaxis, :]
        # truths
        D = run.y.shape[1]
        y_phys = np.full((len(valid), H, D), np.nan, dtype=np.float64)
        for j, t in enumerate(valid):
            for i, h in enumerate(h_sorted):
                if t + h < T:
                    y_phys[j, i] = run.y[t + h]
        mus.append(mu_phys)
        vars_.append(var_phys)
        ys.append(y_phys)
    if not mus:
        D = runs[0].y.shape[1]
        return (np.empty((0, H, D)), np.empty((0, H, D)), np.empty((0, H, D)))
    return np.concatenate(mus), np.concatenate(vars_), np.concatenate(ys)


def _score_horizons(
    label: str,
    mu: np.ndarray,
    var: np.ndarray,
    y: np.ndarray,
    horizons: tuple[int, ...],
    conformal_q: dict[int, float] | None = None,
) -> dict:
    """Per-horizon CRPS / NLL (raw σ) + coverage / PI-width (conformal σ).

    mu, var, y : (P, H, D).  CRPS/NLL use raw σ (baseline convention); coverage
    uses the per-horizon conformal scale ``conformal_q[h]`` if supplied.
    """
    from imas_ambix.statespace.calibration import (  # noqa: PLC0415
        crps_gaussian,
        interval_coverage,
        nll_gaussian,
        prediction_interval_width,
    )

    out: dict[str, dict] = {}
    h_sorted = sorted(horizons)
    for i, h in enumerate(h_sorted):
        m = mu[:, i, 0]
        v = var[:, i, 0]
        yt = y[:, i, 0]
        ok = np.isfinite(yt) & np.isfinite(m) & np.isfinite(v)
        if ok.sum() < 10:
            out[str(h)] = {"n": int(ok.sum())}
            continue
        m, v, yt = m[ok], v[ok], yt[ok]
        sigma_raw = np.sqrt(np.maximum(v, 1e-12))
        rec = {
            "n": int(ok.sum()),
            "crps_raw": float(crps_gaussian(yt, m, sigma_raw)),
            "nll_raw": float(nll_gaussian(yt, m, sigma_raw)),
            "rmse": float(np.sqrt(np.mean((yt - m) ** 2))),
            "mean_sigma_raw": float(np.mean(sigma_raw)),
        }
        if conformal_q is not None and h in conformal_q:
            from scipy.stats import norm  # noqa: PLC0415

            z = float(norm.ppf(0.95))
            sigma_conf = conformal_q[h] * sigma_raw / z
            rec["coverage_90_conf"] = float(
                interval_coverage(yt, m, sigma_conf, alpha=0.10)
            )
            rec["pi_width_90_conf"] = float(
                prediction_interval_width(sigma_conf, alpha=0.10)
            )
        out[str(h)] = rec
    logger.info(
        "[%s] per-horizon CRPS(raw): %s",
        label,
        {h: round(out[str(h)].get("crps_raw", float("nan")), 4) for h in h_sorted},
    )
    return out


def run_experiment(cfg: ExperimentConfig, output: Path | None = None) -> dict:
    """Run the full S7.3 acceptance experiment and return the metrics dict.

    Stages: load runs → fit shared normaliser → train engine →
    retrain static comparator → FILTERING eval → build dense transient windows →
    FORECASTING comparison (engine vs static, same windows/horizons) →
    OOD honesty → optional smoothing.
    """
    from imas_ambix.statespace.baseline import (  # noqa: PLC0415
        _FEATURE_SCHEMA_MAG_ANE,
        _LEVEL1_DIR,
        _XIM_CHANNELS_PRIMARY,
        ChannelStats,
        ConformalWrapper,
        DeepEnsemble,
        EnsembleConfig,
    )
    from imas_ambix.statespace.filter import (  # noqa: PLC0415
        fit_horizon_conformal,
    )

    t_total = time.time()
    metrics: dict = {"config": {
        "latent_dim": cfg.latent_dim,
        "max_train_shots": cfg.max_train_shots,
        "max_cal_shots": cfg.max_cal_shots,
        "max_ood_shots": cfg.max_ood_shots,
        "horizons_steps_at_1kHz": list(cfg.horizons),
        "n_epochs": cfg.n_epochs,
        "seq_len": cfg.seq_len,
        "seed": cfg.seed,
        "device": cfg.device,
        "model_hz": 1000.0,
        "target": "xim/da_hm10_t (raw, primary)",
        "inputs": "mag (ama+amb+amc) + ane",
        "held_out_family": "dalpha",
    }}

    # --- 1. splits + sub-split (same conf-cal / in-dist-test as S7.2) -------
    with open(_SPLITS_MANIFEST) as f:
        splits = json.load(f)
    train_shots = [int(x) for x in splits["train"]]
    cal_shots = [int(x) for x in splits["calibration"]]
    ood_shots = [int(x) for x in splits["test_ood_regime"]]
    regime_scalars = splits.get("regime_scalars", {})

    rng_sub = np.random.default_rng(42)  # same seed as baseline.py
    cal_arr = np.array(cal_shots)
    perm = rng_sub.permutation(len(cal_arr))
    n_conf = int(round(len(cal_arr) * 0.50))
    conf_cal_shots = sorted(cal_arr[perm[:n_conf]].tolist())
    in_dist_test_shots = sorted(cal_arr[perm[n_conf:]].tolist())

    fs = _FEATURE_SCHEMA_MAG_ANE
    tc = _XIM_CHANNELS_PRIMARY

    # --- 2. load runs (cached) ----------------------------------------------
    train_runs = _load_split_runs(train_shots, fs, tc, _LEVEL1_DIR, cfg.max_train_shots, cfg.seed + 1, "train")
    conf_runs = _load_split_runs(conf_cal_shots, fs, tc, _LEVEL1_DIR, cfg.max_cal_shots, cfg.seed + 2, "conf")
    idt_runs = _load_split_runs(in_dist_test_shots, fs, tc, _LEVEL1_DIR, cfg.max_cal_shots, cfg.seed + 3, "idt")
    ood_runs = _load_split_runs(ood_shots, fs, tc, _LEVEL1_DIR, cfg.max_ood_shots, cfg.seed + 4, "ood")
    metrics["split_sizes"] = {
        "n_train_runs": len(train_runs), "n_train_slices": int(sum(len(r.X) for r in train_runs)),
        "n_conf_runs": len(conf_runs), "n_idt_runs": len(idt_runs), "n_ood_runs": len(ood_runs),
    }
    if not train_runs:
        raise RuntimeError("No training runs loaded")

    # --- 3. shared normaliser (train only) ----------------------------------
    stats = ChannelStats.fit([r.X.astype(np.float64) for r in train_runs],
                             [r.y.astype(np.float64) for r in train_runs])
    input_dim = train_runs[0].X.shape[1]
    output_dim = train_runs[0].y.shape[1]

    # --- 4. train RKN engine -------------------------------------------------
    eng_cfg = EngineConfig(
        input_dim=input_dim, latent_dim=cfg.latent_dim, output_dim=output_dim,
        n_epochs=cfg.n_epochs, seq_len=cfg.seq_len, train_horizons=cfg.horizons,
        seed=cfg.seed,
    )
    x_train_n = [stats.normalise_X(r.X.astype(np.float64)) for r in train_runs]
    y_train_n = [stats.normalise_y(r.y.astype(np.float64)) for r in train_runs]
    model = RKNEngine(eng_cfg)
    stop = _make_stop_flag()
    tstate = train_engine(model, x_train_n, y_train_n, eng_cfg, device=cfg.device, stop_flag=stop)
    metrics["engine_train"] = {
        "epochs_run": len(tstate.epoch_losses),
        "final_loss": tstate.epoch_losses[-1] if tstate.epoch_losses else None,
        "final_filter_nll": tstate.epoch_filter_nll[-1] if tstate.epoch_filter_nll else None,
        "final_rollout_nll": tstate.epoch_rollout_nll[-1] if tstate.epoch_rollout_nll else None,
        "seconds": round(tstate.seconds, 1),
    }

    # --- 5. retrain static comparator (baseline classes, same train slices) --
    logger.info("Retraining static comparator (deep ensemble) on same train runs...")
    Xtr, ytr = _stack_runs_for_static(train_runs)
    Xtr_n = stats.normalise_X(Xtr.astype(np.float64))
    ytr_n = stats.normalise_y(ytr.astype(np.float64))
    ens_cfg = EnsembleConfig(n_members=5, n_epochs=60, hidden_size=128, seed_base=0)
    ensemble = DeepEnsemble.build(input_dim, output_dim, ens_cfg)
    ensemble.fit(Xtr_n, ytr_n, ens_cfg)
    # static conformal at h=0 (S7.2-style split-conformal on the dense runs)
    Xcf, ycf = _stack_runs_for_static(conf_runs)
    static_conf = ConformalWrapper(ensemble, stats)
    static_conf.fit_conformal(stats.normalise_X(Xcf.astype(np.float64)),
                              stats.normalise_y(ycf.astype(np.float64)))

    # --- 6. FILTERING eval (engine, causal, in-dist-test) -------------------
    metrics["filtering"] = _eval_filtering(model, idt_runs, conf_runs, stats, cfg.device)

    # --- 7. dense transient anchors (shared selector) -----------------------
    h_max = max(cfg.horizons)
    idt_anchors = [_dense_transient_anchors(r, h_max) for r in idt_runs]
    conf_anchors = [_dense_transient_anchors(r, h_max) for r in conf_runs]
    ood_anchors = [_dense_transient_anchors(r, h_max) for r in ood_runs]
    n_idt_pairs = int(sum(len(a) for a in idt_anchors))
    logger.info("Dense transient anchors: idt=%d conf=%d ood=%d", n_idt_pairs,
                int(sum(len(a) for a in conf_anchors)), int(sum(len(a) for a in ood_anchors)))

    # --- 8. per-horizon conformal calibration on CONF runs (dense windows) --
    # Engine conformal: from engine forecasts on conf anchors.
    eng_mu_cf, eng_var_cf, eng_y_cf = _engine_horizon_pairs(model, conf_runs, conf_anchors, cfg.horizons, stats, cfg.device)
    eng_q = _per_horizon_q(eng_mu_cf, eng_var_cf, eng_y_cf, cfg.horizons, fit_horizon_conformal)
    # Static conformal: static map at each horizon on conf anchors.
    stat_mu_cf, stat_var_cf, stat_y_cf = _static_horizon_predict(ensemble, stats, conf_runs, conf_anchors, cfg.horizons)
    stat_q = _per_horizon_q(stat_mu_cf, stat_var_cf, stat_y_cf, cfg.horizons, fit_horizon_conformal)

    # --- 9. FORECASTING comparison on the SAME dense transient windows ------
    eng_mu, eng_var, eng_y = _engine_horizon_pairs(model, idt_runs, idt_anchors, cfg.horizons, stats, cfg.device)
    stat_mu, stat_var, stat_y = _static_horizon_predict(ensemble, stats, idt_runs, idt_anchors, cfg.horizons)
    # Verify the two models scored the SAME (t, t+h) truths (identical windows).
    same_truth = _verify_same_truths(eng_y, stat_y)
    metrics["forecasting_indist_dense_transient"] = {
        "n_anchor_pairs": int(eng_mu.shape[0]),
        "same_truths_engine_vs_static": same_truth,
        "engine": _score_horizons("ENGINE/idt", eng_mu, eng_var, eng_y, cfg.horizons, eng_q),
        "static": _score_horizons("STATIC/idt", stat_mu, stat_var, stat_y, cfg.horizons, stat_q),
    }

    # --- 10. OOD honesty (forecasting on OOD dense transient windows) -------
    eng_mu_o, eng_var_o, eng_y_o = _engine_horizon_pairs(model, ood_runs, ood_anchors, cfg.horizons, stats, cfg.device)
    stat_mu_o, stat_var_o, stat_y_o = _static_horizon_predict(ensemble, stats, ood_runs, ood_anchors, cfg.horizons)
    metrics["forecasting_ood_dense_transient"] = {
        "n_anchor_pairs": int(eng_mu_o.shape[0]),
        "engine": _score_horizons("ENGINE/ood", eng_mu_o, eng_var_o, eng_y_o, cfg.horizons, eng_q),
        "static": _score_horizons("STATIC/ood", stat_mu_o, stat_var_o, stat_y_o, cfg.horizons, stat_q),
    }
    metrics["ood_honesty"] = _ood_honesty(
        model, ensemble, idt_runs, ood_runs, stats, regime_scalars, train_runs, cfg.device
    )

    # --- 11. optional smoothing --------------------------------------------
    if cfg.do_smoothing:
        metrics["smoothing"] = _eval_smoothing(model, idt_runs, stats, cfg.device)

    # --- 12. verdict --------------------------------------------------------
    metrics["acceptance"] = _verdict(metrics)
    metrics["total_seconds"] = round(time.time() - t_total, 1)

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            json.dump(metrics, f, indent=2, default=float)
        logger.info("Metrics written to %s", output)
    return metrics


def _make_stop_flag():
    """STOP-FILE + soft-time-limit poll (AGENTS.md §2a-cancel / §2a-cancel-time)."""
    stop_path = os.environ.get("AMBIX_STOP_FILE")
    soft_limit = os.environ.get("AMBIX_SOFT_TIME_LIMIT")
    t0 = time.monotonic()
    soft = float(soft_limit) if soft_limit else None
    if stop_path:
        logger.info("[engine] STOP-FILE = %s", stop_path)
    if soft:
        logger.info("[engine] AMBIX_SOFT_TIME_LIMIT = %.0fs", soft)

    def _flag() -> bool:
        if stop_path and Path(stop_path).exists():
            return True
        if soft is not None and (time.monotonic() - t0) > soft:
            return True
        return False

    return _flag


def _per_horizon_q(mu, var, y, horizons, fit_fn) -> dict[int, float]:
    """Fit a split-conformal scale q̂ per horizon from calibration forecasts."""
    q: dict[int, float] = {}
    for i, h in enumerate(sorted(horizons)):
        m, v, yt = mu[:, i, 0], var[:, i, 0], y[:, i, 0]
        ok = np.isfinite(yt) & np.isfinite(m) & np.isfinite(v)
        if ok.sum() < 10:
            q[h] = 1.0
            continue
        q[h] = fit_fn(yt[ok], m[ok], np.sqrt(np.maximum(v[ok], 1e-12)), alpha=0.10)
    return q


def _static_horizon_predict(ensemble, stats, runs, anchors_per_run, horizons):
    """Static comparator forecasts: Dα_{t+h} = static_map(inputs_t) for all h.

    Returns (mu, var, y) each (P, H, D) in physical units, aligned exactly to the
    engine's (run, anchor, horizon) triples (same valid-anchor rule: t+h_max<T).
    """
    H = len(horizons)
    h_sorted = sorted(horizons)
    h_max = max(horizons)
    mus, vars_, ys = [], [], []
    for run, anchors in zip(runs, anchors_per_run, strict=True):
        valid = [int(a) for a in anchors if int(a) + h_max < run.X.shape[0]]
        if not valid:
            continue
        X_t = np.stack([run.X[t] for t in valid]).astype(np.float64)
        mu_n, sigma_n, _ens = ensemble.predict(stats.normalise_X(X_t))  # (P, D)
        mu_p = stats.denormalise_y_mean(mu_n)            # (P, D)
        var_p = (stats.denormalise_y_std(sigma_n)) ** 2  # (P, D)
        T, D = run.y.shape
        P = len(valid)
        # Broadcast the SAME static prediction across all horizons (frozen nowcast)
        mu_h = np.repeat(mu_p[:, np.newaxis, :], H, axis=1)   # (P, H, D)
        var_h = np.repeat(var_p[:, np.newaxis, :], H, axis=1)
        y_h = np.full((P, H, D), np.nan, dtype=np.float64)
        for j, t in enumerate(valid):
            for i, h in enumerate(h_sorted):
                if t + h < T:
                    y_h[j, i] = run.y[t + h]
        mus.append(mu_h)
        vars_.append(var_h)
        ys.append(y_h)
    if not mus:
        D = runs[0].y.shape[1]
        return np.empty((0, H, D)), np.empty((0, H, D)), np.empty((0, H, D))
    return np.concatenate(mus), np.concatenate(vars_), np.concatenate(ys)


def _verify_same_truths(y_eng: np.ndarray, y_stat: np.ndarray) -> bool:
    """Assert both models were scored on identical (t, t+h) truth values."""
    if y_eng.shape != y_stat.shape:
        return False
    a, b = y_eng, y_stat
    both_nan = np.isnan(a) & np.isnan(b)
    return bool(np.all(both_nan | (a == b)))


def _eval_filtering(model, idt_runs, conf_runs, stats, device, burn_in: int = 20) -> dict:
    """Causal filtering coverage / CRPS / NLL on in-dist-test runs.

    Coverage uses split-conformal fit on conf runs' filtered residuals.

    BURN-IN (critical for honest coverage): the belief starts at the broad prior
    (var0 = data-std); the first few updates are prior-dominated, so the filtered
    σ is non-stationary across a run's leading edge.  Split-conformal cannot
    transfer a single quantile across that non-stationarity (it shrinks/inflates
    intervals → coverage collapses).  We therefore drop the first ``burn_in``
    slices (~20 ms at 1 kHz) of EACH run from both the conformal fit and the eval
    — standard filter practice: you do not estimate during the prior-dominated
    warmup.  Both ``coverage_90_raw`` (pre-conformal) and the conformal coverage
    are reported so the mechanism is visible.
    """
    from scipy.stats import norm  # noqa: PLC0415

    from imas_ambix.statespace.calibration import (  # noqa: PLC0415
        crps_gaussian,
        interval_coverage,
        nll_gaussian,
        prediction_interval_width,
    )
    from imas_ambix.statespace.filter import (  # noqa: PLC0415
        filter_shot,
        fit_horizon_conformal,
    )

    def _collect(runs):
        ys, mus, sigs = [], [], []
        for r in runs:
            if r.X.shape[0] <= burn_in + 1:
                continue
            mu_n, var_n = filter_shot(model, stats.normalise_X(r.X.astype(np.float64)), device)
            mu_p = stats.denormalise_y_mean(mu_n)[:, 0]
            sig_p = np.sqrt(np.maximum(var_n[:, 0] * stats.target_std[0] ** 2, 1e-12))
            ys.append(r.y[burn_in:, 0])
            mus.append(mu_p[burn_in:])
            sigs.append(sig_p[burn_in:])
        return (np.concatenate(ys), np.concatenate(mus), np.concatenate(sigs))

    cf_y, cf_mu, cf_sig = _collect(conf_runs)
    q = fit_horizon_conformal(cf_y, cf_mu, cf_sig, alpha=0.10)
    z = float(norm.ppf(0.95))

    y, mu, sig = _collect(idt_runs)
    sig_conf = q * sig / z
    return {
        "n_slices": int(len(y)),
        "burn_in_dropped": int(burn_in),
        "conformal_q": float(q),
        "coverage_90_raw": float(interval_coverage(y, mu, sig, alpha=0.10)),
        "coverage_90_conf": float(interval_coverage(y, mu, sig_conf, alpha=0.10)),
        "pi_width_90_conf": float(prediction_interval_width(sig_conf, alpha=0.10)),
        "mean_sigma_raw": float(np.mean(sig)),
        "mean_sigma_conf": float(np.mean(sig_conf)),
        "crps_raw": float(crps_gaussian(y, mu, sig)),
        "nll_raw": float(nll_gaussian(y, mu, sig)),
        "rmse": float(np.sqrt(np.mean((y - mu) ** 2))),
    }


def _eval_smoothing(model, idt_runs, stats, device) -> dict:
    from imas_ambix.statespace.calibration import (  # noqa: PLC0415
        crps_gaussian,
        nll_gaussian,
    )
    from imas_ambix.statespace.filter import smooth_shot  # noqa: PLC0415

    ys, mus, sigs = [], [], []
    for r in idt_runs:
        mu_n, var_n = smooth_shot(model, stats.normalise_X(r.X.astype(np.float64)), device)
        mu_p = stats.denormalise_y_mean(mu_n)[:, 0]
        sig_p = np.sqrt(np.maximum(var_n[:, 0] * stats.target_std[0] ** 2, 1e-12))
        ys.append(r.y[:, 0])
        mus.append(mu_p)
        sigs.append(sig_p)
    y = np.concatenate(ys)
    mu = np.concatenate(mus)
    sig = np.concatenate(sigs)
    return {
        "n_slices": int(len(y)),
        "crps_raw": float(crps_gaussian(y, mu, sig)),
        "nll_raw": float(nll_gaussian(y, mu, sig)),
        "rmse": float(np.sqrt(np.mean((y - mu) ** 2))),
    }


def _ood_honesty(model, ensemble, idt_runs, ood_runs, stats, regime_scalars, train_runs, device) -> dict:
    """OOD honesty: coverage non-collapse + a quantified novelty signal.

    Reports (a) engine filtering coverage on OOD vs in-dist (non-collapse),
    (b) OOD-AUROC from ensemble disagreement (input-novelty), and
    (c) coverage-vs-distance from the regime axis.  Reported honestly — not
    hard-thresholded (static OOD-AUROC≈0.568 ~ random; the high-Iₚ×ne axis may
    not surface as encoder-visible input novelty).
    """
    from imas_ambix.statespace.calibration import (  # noqa: PLC0415
        coverage_vs_distance,
        ensemble_disagreement,
        interval_coverage,
        ood_auroc,
    )
    from imas_ambix.statespace.filter import filter_shot  # noqa: PLC0415

    def _cov(runs):
        ys, mus, sigs = [], [], []
        for r in runs:
            mu_n, var_n = filter_shot(model, stats.normalise_X(r.X.astype(np.float64)), device)
            mu_p = stats.denormalise_y_mean(mu_n)[:, 0]
            sig_p = np.sqrt(np.maximum(var_n[:, 0] * stats.target_std[0] ** 2, 1e-12))
            ys.append(r.y[:, 0])
            mus.append(mu_p)
            sigs.append(sig_p)
        if not ys:
            return None, None, None
        return np.concatenate(ys), np.concatenate(mus), np.concatenate(sigs)

    y_i, mu_i, sig_i = _cov(idt_runs)
    y_o, mu_o, sig_o = _cov(ood_runs)
    out: dict = {}
    if y_i is not None and y_o is not None:
        # raw (unconformalised) coverage so we see the model's native widening
        out["filter_coverage90_raw_indist"] = float(interval_coverage(y_i, mu_i, sig_i, alpha=0.10))
        out["filter_coverage90_raw_ood"] = float(interval_coverage(y_o, mu_o, sig_o, alpha=0.10))
        out["mean_sigma_indist"] = float(np.mean(sig_i))
        out["mean_sigma_ood"] = float(np.mean(sig_o))

    # OOD-AUROC from ensemble disagreement (static-model novelty score)
    Xi = np.concatenate([r.X for r in idt_runs]).astype(np.float64)
    Xo = np.concatenate([r.X for r in ood_runs]).astype(np.float64)
    _, _, ens_i = ensemble.predict(stats.normalise_X(Xi))
    _, _, ens_o = ensemble.predict(stats.normalise_X(Xo))
    try:
        out["ood_auroc_ensemble_disagreement"] = float(
            ood_auroc(ensemble_disagreement(ens_i), ensemble_disagreement(ens_o))
        )
    except Exception as e:  # noqa: BLE001
        out["ood_auroc_error"] = str(e)

    # coverage-vs-distance on OOD runs (regime axis)
    train_ip = np.array([regime_scalars.get(str(r.shot_id), {}).get("ip_mean", np.nan) for r in train_runs])
    train_ne = np.array([regime_scalars.get(str(r.shot_id), {}).get("ne_mean", np.nan) for r in train_runs])
    train_ip = train_ip[np.isfinite(train_ip)]
    train_ne = train_ne[np.isfinite(train_ne)]
    if train_ip.size and train_ne.size and y_o is not None:
        mu_ip, std_ip = float(np.mean(train_ip)), float(np.std(train_ip)) or 1.0
        mu_ne, std_ne = float(np.mean(train_ne)), float(np.std(train_ne)) or 1.0
        dists = []
        for r in ood_runs:
            sc = regime_scalars.get(str(r.shot_id), {})
            ip, ne = sc.get("ip_mean", np.nan), sc.get("ne_mean", np.nan)
            d = (
                math.sqrt(((ip - mu_ip) / std_ip) ** 2 + ((ne - mu_ne) / std_ne) ** 2)
                if np.isfinite(ip) and np.isfinite(ne) else np.nan
            )
            dists.extend([d] * len(r.y))
        dists = np.array(dists)
        valid = np.isfinite(dists)
        if valid.sum() > 50:
            out["coverage_vs_distance_ood"] = coverage_vs_distance(
                y_o[valid], mu_o[valid], sig_o[valid], dists[valid]
            )
    return out


def _verdict(metrics: dict) -> dict:
    """Decide the three acceptance criteria from the computed metrics."""
    v: dict = {}
    # (1) filtering coverage 88-92% (burn-in-excluded, conformal)
    filt = metrics.get("filtering", {})
    fcov = filt.get("coverage_90_conf")
    v["filtering_coverage_in_band"] = (
        bool(fcov is not None and 0.88 <= fcov <= 0.92)
    )
    v["filtering_coverage_value"] = fcov
    v["filtering_coverage_raw"] = filt.get("coverage_90_raw")

    # (2) forecast CRPS beats static at H>0 on the same dense windows
    fc = metrics.get("forecasting_indist_dense_transient", {})
    eng = fc.get("engine", {})
    stat = fc.get("static", {})
    wins = {}
    for h in eng:
        e = eng[h].get("crps_raw")
        s = stat.get(h, {}).get("crps_raw")
        if e is not None and s is not None:
            wins[h] = {"engine_crps": e, "static_crps": s, "engine_wins": bool(e < s)}
    v["forecast_crps_by_horizon"] = wins
    h_pos = [h for h in wins if int(h) > 0]
    v["forecast_beats_static_at_Hgt0"] = bool(
        h_pos and all(wins[h]["engine_wins"] for h in h_pos)
    )
    v["forecast_beats_static_any_Hgt0"] = bool(
        any(wins[h]["engine_wins"] for h in h_pos)
    )
    v["same_windows_verified"] = fc.get("same_truths_engine_vs_static", False)

    # Calibration-quality diagnostic: is the engine's predictive σ ≈ its rmse at
    # each horizon?  σ≈rmse → honest widening (the win mechanism is working);
    # σ<<rmse → overconfident (CRPS loses); σ>>rmse → overwide (CRPS loses).
    # This separates "training didn't converge" from "architecture wrong".
    sigma_vs_rmse = {}
    for h in sorted(eng, key=lambda x: int(x)):
        rec = eng[h]
        sig = rec.get("mean_sigma_raw")
        rmse = rec.get("rmse")
        if sig is not None and rmse is not None and rmse > 0:
            sigma_vs_rmse[h] = {
                "engine_mean_sigma": sig,
                "engine_rmse": rmse,
                "sigma_over_rmse": sig / rmse,
            }
    v["engine_sigma_vs_rmse_by_horizon"] = sigma_vs_rmse

    # Sharpness-at-matched-coverage: at large h the static needs a ballooning
    # conformal q to cover grown residuals → wider intervals.  Report engine vs
    # static conformal PI-width per horizon (a second honest win, if present).
    pi_width = {}
    for h in sorted(eng, key=lambda x: int(x)):
        ew = eng[h].get("pi_width_90_conf")
        sw = stat.get(h, {}).get("pi_width_90_conf")
        ec = eng[h].get("coverage_90_conf")
        sc = stat.get(h, {}).get("coverage_90_conf")
        if ew is not None and sw is not None:
            pi_width[h] = {
                "engine_pi_width": ew, "static_pi_width": sw,
                "engine_coverage": ec, "static_coverage": sc,
                "engine_sharper": bool(ew < sw),
            }
    v["forecast_pi_width_by_horizon"] = pi_width

    # (3) OOD honesty quantified + coverage non-collapse
    ood = metrics.get("ood_honesty", {})
    v["ood_auroc"] = ood.get("ood_auroc_ensemble_disagreement")
    ci = ood.get("filter_coverage90_raw_indist")
    co = ood.get("filter_coverage90_raw_ood")
    v["ood_coverage_noncollapse"] = bool(co is not None and co > 0.5)
    v["ood_coverage_indist"] = ci
    v["ood_coverage_ood"] = co
    return v


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    import argparse  # noqa: PLC0415

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    p = argparse.ArgumentParser(description="S7.3 RKN engine acceptance experiment")
    p.add_argument("--latent-dim", type=int, default=16)
    p.add_argument("--max-train-shots", type=int, default=500)
    p.add_argument("--max-cal-shots", type=int, default=300)
    p.add_argument("--max-ood-shots", type=int, default=200)
    p.add_argument("--n-epochs", type=int, default=30)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoothing", action="store_true")
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", type=Path, default=None)
    a = p.parse_args(argv)

    cfg = ExperimentConfig(
        latent_dim=a.latent_dim,
        max_train_shots=a.max_train_shots,
        max_cal_shots=a.max_cal_shots,
        max_ood_shots=a.max_ood_shots,
        n_epochs=a.n_epochs,
        seq_len=a.seq_len,
        seed=a.seed,
        do_smoothing=a.smoothing,
        device=a.device,
    )
    out = a.output or (Path(__file__).parent / "artifacts" / "engine_metrics_v0.json")
    metrics = run_experiment(cfg, output=out)

    print("\n" + "=" * 64)
    print("S7.3 RKN ENGINE — STAGE-1 ACCEPTANCE")
    print("=" * 64)
    acc = metrics["acceptance"]
    print(f"(1) filtering coverage@90 = {acc.get('filtering_coverage_value')}  in-band={acc['filtering_coverage_in_band']}")
    print(f"(2) forecast beats static at any H>0 = {acc['forecast_beats_static_any_Hgt0']}  all H>0 = {acc['forecast_beats_static_at_Hgt0']}")
    print("    per-horizon CRPS (engine vs static):")
    for h, w in acc.get("forecast_crps_by_horizon", {}).items():
        flag = "WIN" if w["engine_wins"] else "loss"
        print(f"      h={h:>3}: engine={w['engine_crps']:.4f}  static={w['static_crps']:.4f}  [{flag}]")
    print(f"(3) OOD: AUROC={acc.get('ood_auroc')}  cov(indist)={acc.get('ood_coverage_indist')}  cov(ood)={acc.get('ood_coverage_ood')}  noncollapse={acc['ood_coverage_noncollapse']}")
    print(f"total: {metrics.get('total_seconds')}s")


if __name__ == "__main__":
    main()
