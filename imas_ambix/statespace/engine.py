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

    input_dim: int = 122  # mag (ama+amb+amc) + ane
    latent_dim: int = 16  # RKN latent size (sweep [8,16,32])
    output_dim: int = 1  # Dα channels (1 = primary, 5 = multi)
    enc_hidden: int = 128  # encoder MLP hidden size
    trans_hidden: int = 64  # transition MLP hidden size
    obs_hidden: int = 64  # observation head hidden size
    # Training
    n_epochs: int = 30
    batch_size: int = 64  # sequences per batch
    seq_len: int = 64  # timesteps per training sequence (1 kHz → 64 ms)
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
    # --- S7.4 rollout-drift reduction (bounded iteration lever) -------------
    # Penalise the transition-mean increment f_θ(z) on QUIESCENT dynamics so the
    # autonomous latent rollout tracks the (96.8%-stationary) Dα baseline instead
    # of drifting off it.  The penalty is weighted by (1 - transient_weight): a
    # quiescent anchor pulls the predicted latent toward PERSISTENCE (Δz→0, i.e.
    # the transition toward identity); an ELM-active anchor is left free to move.
    # This attacks the actual cause of the v0 bulk-CRPS loss (autonomous drift
    # inflating the residual tail) and — unlike σ-rescaling — shrinks RESIDUALS,
    # so it helps CRPS and NLL together (no σ zero-sum).  0.0 = off (v0 behaviour).
    drift_reg_weight: float = 0.0
    # --- S7.5 heavy-tailed emission head ------------------------------------
    # "gaussian" (v0/v1) or "student_t".  A single-Gaussian observation head
    # cannot be CRPS-sharp on the ~97%-quiescent Dα bulk AND NLL-tail-safe on the
    # heavy-tailed ELM spikes at once — the NLL-optimal σ ≈ rmse ≫ the
    # CRPS-optimal σ ≈ typical |residual| on a heavy-tailed residual.  A Student-t
    # predictive resolves the tension: a SHARP scale tracks the bulk (low CRPS)
    # while the heavy tail (finite ν) keeps the spike NLL bounded.  This is the
    # principled fix for the v1 transient/bulk-CRPS loss.
    emission: str = "gaussian"
    # Student-t degrees of freedom.  If student_t_learn_nu, ν is a learned
    # per-output parameter (softplus, floored at student_t_nu_floor so Var is
    # finite, ν>2); else ν is FIXED at student_t_nu.  A moderate fixed ν≈4-6 is a
    # robust default for ELM-spiked residuals (clear excess kurtosis but not
    # Cauchy-like).
    student_t_learn_nu: bool = True
    student_t_nu: float = 5.0  # used when student_t_learn_nu is False
    student_t_nu_floor: float = 2.1  # keep ν>2 so the t variance ν/(ν-2) is finite
    # --- S7.5 perf: cap intra-op threads in training (the real ~100x lever) --
    # The model is tiny (latent≤32) so training is OVERHEAD-bound: torch's default
    # intra-op thread count (48 on the data host) thrashes and inflates the
    # per-batch time ~100x (3 s vs 30 ms), which is what made the S7.4 drift-reg
    # run look "85x slower" — it was actually thread oversubscription, present
    # with drift-reg OFF too.  4-8 threads is the measured sweet spot.  None =
    # leave torch's setting untouched.
    num_threads: int | None = 4
    # --- S8-T6 GS GROUNDING (additive; 0/off = the ungrounded v2 path) -------
    # When ``grounding`` is True, a GroundingHead (gs/grounding.py) maps the
    # filtered latent z_t to the LOCKED restricted GS currents (order-1 plasma
    # poly DOF + rank-4 passive SVD) pushed through the T2 forward operator to
    # predict RAW magnetics.  Two terms add to the joint loss:
    #   gs_data_weight · L_data  (raw-magnetics reconstruction NLL through G)
    #   L_GS  (the GS force-balance soft prior, current-space L2, weight gs_lambda)
    # The collinearity fix is STRUCTURAL (low DOF + low-rank passive — the head
    # emits exactly the DOF the standalone frontier found near-vacuum-sound at
    # λ=0); gs_lambda only biases toward small currents (soft so data overrides).
    grounding: bool = False
    gs_profile_order: int = 1  # locked order-1 plasma poly (3 DOF)
    gs_passive_rank: int = 4  # locked passive SVD rank
    gs_lambda: float = 1e-2  # L_GS soft-prior weight (current-space L2)
    gs_data_weight: float = 0.1  # weight on L_data in the joint objective


def _plasma_poly_dof(order: int) -> int:
    """Number of 2-D polynomial profile-DOF for a given order (matches
    residual.plasma_poly_basis): order-0→1, order-1→3, order-2→6, order-4→15."""
    if order >= 4:
        return 15
    if order >= 2:
        return 6
    if order >= 1:
        return 3
    return 1


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
        self.enc_w = nn.Linear(cfg.enc_hidden, L)  # latent observation mean
        self.enc_logr = nn.Linear(cfg.enc_hidden, L)  # log obs variance

        # Transition: mean update f_theta(z) -> delta z (residual / locally-linear)
        self.trans_mean = nn.Sequential(
            nn.Linear(L, cfg.trans_hidden),
            nn.Tanh(),
            nn.Linear(cfg.trans_hidden, L),
        )
        # Variance transition: per-dim multiplicative factor on prior variance
        # (a in [0, ~]) — learned, input-free.  log-parameterised for positivity.
        self.trans_log_a = nn.Parameter(torch.zeros(L))  # var <- a^2 * var + Q
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
        self.log_obs_noise = nn.Parameter(torch.full((cfg.output_dim,), math.log(0.1)))

        # S7.5: learned Student-t degrees of freedom (per output dim).  Only used
        # when cfg.emission == "student_t".  Parameterised as ν = floor +
        # softplus(raw) so ν > floor > 2 (finite variance).  Init near ν≈5.
        nu_init = max(cfg.student_t_nu, cfg.student_t_nu_floor + 0.5)
        raw_init = math.log(math.expm1(nu_init - cfg.student_t_nu_floor))
        self.t_log_nu = nn.Parameter(torch.full((cfg.output_dim,), raw_init))

        # Initial belief (prior at t=0): learned mean ~0, broad variance.
        self.z0 = nn.Parameter(torch.zeros(L))
        self.log_var0 = nn.Parameter(torch.full((L,), math.log(1.0)))

        # S8-T6: GS grounding head (latent → restricted GS currents).  Only
        # constructed when grounding is on; the plasma DOF count is derived from
        # the locked profile order (order-1 → {1, ρ_R, ρ_Z} = 3 DOF) so the head
        # is built without any campaign data.  The forward operator + per-campaign
        # tensors are supplied at train time (gs/grounding.CampaignGrounding).
        self.grounding_head: nn.Module | None = None
        if cfg.grounding:
            from imas_ambix.gs.grounding import GroundingHead  # noqa: PLC0415

            n_dof = _plasma_poly_dof(cfg.gs_profile_order)
            self.grounding_head = GroundingHead(L, n_dof, cfg.gs_passive_rank)

    # -- belief ops ---------------------------------------------------------

    def initial_belief(
        self, batch: int, device, dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        self, z: torch.Tensor, var: torch.Tensor, return_incr: bool = False
    ):
        """Learned transition (predict).  INPUT-FREE: latent -> latent only.

        z_{t+1}   = z_t + f_theta(z_t)
        var_{t+1} = a^2 var_t + Q          (a, Q learned, Q strictly positive)

        Each call adds Q → variance grows with the number of predict steps →
        calibrated widening through transients during autonomous rollout.

        ``return_incr`` (S7.5 perf): also return the transition-mean increment
        f_θ(z) so the quiescent-drift penalty can REUSE the increment computed in
        the forward scan instead of running a second full ``trans_mean`` forward
        over all (B·T) latents (which built a redundant autograd subgraph).
        """
        incr = self.trans_mean(z)
        z_next = z + incr
        a2 = self.trans_log_a.exp().pow(2.0)
        q = self.log_q.exp() + _VAR_FLOOR
        var_next = a2.unsqueeze(0) * var + q.unsqueeze(0)
        var_next = var_next.clamp(_VAR_FLOOR, _VAR_CEIL)
        if return_incr:
            return z_next, var_next, incr
        return z_next, var_next

    def observe(
        self, z: torch.Tensor, var: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Latent belief -> Dα predictive (mean, variance).

        mean = g_theta(z)
        var  = softplus(W)^T var_latent + emission_noise^2     (>= 0)
        """
        mu = self.obs_mean(z)
        w_pos = F.softplus(self.obs_var_w)  # (L, D) non-negative
        out_var = var @ w_pos  # (B, D)
        out_var = out_var + self.log_obs_noise.exp().pow(2.0).unsqueeze(0)
        out_var = out_var.clamp(_VAR_FLOOR, _VAR_CEIL)
        return mu, out_var

    def nu(self) -> torch.Tensor:
        """Student-t degrees of freedom ν (D,) = floor + softplus(raw) (> floor > 2).

        Learned when cfg.student_t_learn_nu; otherwise frozen at cfg.student_t_nu
        (the parameter is initialised there and detached from the optimiser by
        ``configure`` — see train_engine).  ν > 2 guarantees a finite t variance.
        """
        return self.cfg.student_t_nu_floor + F.softplus(self.t_log_nu)

    def observe_student_t(
        self, z: torch.Tensor, var: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Latent belief -> Student-t predictive (location μ, SCALE², ν).

        Reuses the exact mean head and variance-propagation machinery of
        ``observe`` but REINTERPRETS the propagated quantity as the t SCALE²
        (the location-scale parameter), not the variance.  The predictive
        *variance* is then scale² · ν/(ν-2) — wider than the Gaussian for the
        same scale, which is the heavy tail.  Keeping the scale tied to the same
        well-behaved belief-variable map means the only added expressiveness is
        the tail shape ν, exactly the degree of freedom we want.

        Returns
        -------
        mu    : (B, D) location.
        scale2: (B, D) squared scale (the t's σ² location-scale parameter).
        nu    : (D,)   degrees of freedom (broadcast over the batch).
        """
        mu, scale2 = self.observe(z, var)  # scale2 == the propagated belief var
        return mu, scale2, self.nu()

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

        z_post = torch.stack(z_list, dim=1)  # (B, T, L)
        var_post = torch.stack(var_list, dim=1)
        # Batched observation head over the full posterior trajectory.
        obs_mu, obs_var = self.observe(
            z_post.reshape(B * T, -1), var_post.reshape(B * T, -1)
        )
        obs_mu = obs_mu.reshape(B, T, -1)
        obs_var = obs_var.reshape(B, T, -1)
        return z_post, var_post, obs_mu, obs_var

    def filter_innovation(self, x_seq: torch.Tensor) -> torch.Tensor:
        """Per-step normalised filter-innovation magnitude (S7.5 ENGINE-NATIVE OOD).

        The innovation at t is the latent observation w_t (from the encoder)
        minus the one-step PRIOR mean z_prior_t (predicted from the previous
        posterior).  Normalised by the innovation variance var_prior + r — a
        Mahalanobis-style "surprise" — and summed over latent dims:

            s_t = Σ_d (w_{t,d} − z_prior_{t,d})² / (var_prior_{t,d} + r_{t,d})

        A LARGE innovation means the encoder's read of inputs_t disagrees with
        what the learned dynamics predicted — exactly the model-internal signal
        that the current operating point is novel.  Unlike the static ensemble's
        disagreement (an input-space novelty external to the dynamics), this is a
        TRUE engine signal: it uses the transition kernel + belief.  Returns
        (B, T); t=0 (no prior) is set to its t=1 value to avoid a warmup spike.
        """
        B, T, _ = x_seq.shape  # noqa: N806
        device, dtype = x_seq.device, x_seq.dtype
        z, var = self.initial_belief(B, device, dtype)
        w_all, r_all = self.encode(x_seq.reshape(B * T, x_seq.shape[-1]))
        w_all = w_all.reshape(B, T, -1)
        r_all = r_all.reshape(B, T, -1)
        scores = []
        for t in range(T):
            if t > 0:
                z, var = self.predict_step(z, var)
            innov = w_all[:, t] - z  # (B, L) prior-mean innovation
            denom = (var + r_all[:, t]).clamp_min(_VAR_FLOOR)
            s = (innov * innov / denom).sum(dim=-1)  # (B,)
            scores.append(s)
            z, var = self.update_step(z, var, w_all[:, t], r_all[:, t])
        out = torch.stack(scores, dim=1)  # (B, T)
        if T > 1:
            out[:, 0] = out[:, 1]
        return out

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
        mu = torch.stack([out_mu[h] for h in order], dim=1)  # (B, H, D)
        ov = torch.stack([out_var[h] for h in order], dim=1)  # (B, H, D)
        return mu, ov


# ---------------------------------------------------------------------------
# Gaussian NLL (matches calibration.nll_gaussian convention)
# ---------------------------------------------------------------------------


def gaussian_nll(y: torch.Tensor, mu: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
    """Mean Gaussian NLL = 0.5 [log(2π var) + (y-μ)²/var]."""
    var = var.clamp(_VAR_FLOOR, _VAR_CEIL)
    return 0.5 * (torch.log(2.0 * math.pi * var) + (y - mu) ** 2 / var).mean()


def student_t_nll(
    y: torch.Tensor, mu: torch.Tensor, scale2: torch.Tensor, nu: torch.Tensor
) -> torch.Tensor:
    """Mean Student-t NLL for a location-scale t with squared scale ``scale2``.

    For X ~ t_ν(μ, s) with s² = scale2:
        log p(y) = log Γ((ν+1)/2) − log Γ(ν/2) − 0.5 log(ν π s²)
                   − (ν+1)/2 · log(1 + (y−μ)² / (ν s²))
    The first two (constant-in-data) terms still depend on ν (a free parameter),
    so learning ν is well-posed.  This is the training counterpart of
    ``gaussian_nll`` for the heavy-tailed emission head.
    """
    scale2 = scale2.clamp(_VAR_FLOOR, _VAR_CEIL)
    nu = nu.clamp_min(2.0 + 1e-3)
    z2 = (y - mu) ** 2 / scale2
    log_norm = (
        torch.lgamma((nu + 1.0) / 2.0)
        - torch.lgamma(nu / 2.0)
        - 0.5 * torch.log(nu * math.pi * scale2)
    )
    log_kernel = -((nu + 1.0) / 2.0) * torch.log1p(z2 / nu)
    return (-(log_norm + log_kernel)).mean()


# ---------------------------------------------------------------------------
# Student-t scoring (numpy) — CRPS + NLL for the harness
# ---------------------------------------------------------------------------
#
# calibration.py (import-only for S7.5) provides only Gaussian closed forms, so
# the Student-t emission head's CRPS/NLL live here.  Both are standard
# closed-form expressions validated against Monte-Carlo in the engine tests.


def student_t_nll_np(
    y: np.ndarray, mu: np.ndarray, scale2: np.ndarray, nu: np.ndarray | float
) -> float:
    """Mean Student-t NLL (numpy), location-scale t with squared scale scale2."""
    from scipy.special import gammaln  # noqa: PLC0415

    scale2 = np.maximum(np.asarray(scale2, dtype=np.float64), 1e-12)
    nu = np.asarray(nu, dtype=np.float64)
    nu = np.maximum(nu, 2.0 + 1e-3)
    z2 = (np.asarray(y) - np.asarray(mu)) ** 2 / scale2
    log_norm = (
        gammaln((nu + 1.0) / 2.0)
        - gammaln(nu / 2.0)
        - 0.5 * np.log(nu * np.pi * scale2)
    )
    log_kernel = -((nu + 1.0) / 2.0) * np.log1p(z2 / nu)
    return float(np.mean(-(log_norm + log_kernel)))


def crps_student_t(
    y: np.ndarray, mu: np.ndarray, scale: np.ndarray, nu: np.ndarray | float
) -> float:
    """CRPS for a location-scale Student-t predictive (analytic closed form).

    For Y ~ t_ν(μ, σ) with σ = ``scale`` and ω = (y-μ)/σ (Jordan, Krüger &
    Lerch 2019, "Evaluating probabilistic forecasts with scoringRules", eq. for
    crps_t; equivalently scoringRules::crps_t):

      CRPS/σ = ω (2 T_ν(ω) − 1)
               + 2 f_ν(ω) (ν + ω²)/(ν − 1)
               − (2 √ν)/(ν − 1) · B(½, ν − ½) / B(½, ν/2)²

    where f_ν, T_ν are the standard-t pdf/cdf and B is the beta function.  Valid
    for ν > 1.  Units match ``y``.  Returns the mean over all samples.
    """
    from scipy.special import beta as betafn  # noqa: PLC0415
    from scipy.stats import t as student_t  # noqa: PLC0415

    scale = np.maximum(np.abs(np.asarray(scale, dtype=np.float64)), 1e-12)
    nu = np.asarray(nu, dtype=np.float64)
    nu = np.maximum(nu, 1.0 + 1e-3)
    omega = (np.asarray(y) - np.asarray(mu)) / scale
    f = student_t.pdf(omega, df=nu)
    Tc = student_t.cdf(omega, df=nu)  # noqa: N806
    term1 = omega * (2.0 * Tc - 1.0)
    term2 = 2.0 * f * (nu + omega**2) / (nu - 1.0)
    term3 = (2.0 * np.sqrt(nu) / (nu - 1.0)) * (
        betafn(0.5, nu - 0.5) / (betafn(0.5, nu / 2.0) ** 2)
    )
    crps = scale * (term1 + term2 - term3)
    return float(np.mean(crps))


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
    # S8-T6 GS grounding (per-epoch mean L_data / L_GS; empty when ungrounded)
    epoch_gs_data: list[float] = field(default_factory=list)
    epoch_gs_prior: list[float] = field(default_factory=list)


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
    step forward, and accumulate the emission NLL (Gaussian or Student-t per
    ``model.cfg.emission``) of the true Dα_{t+h}.

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
    is_t = model.cfg.emission == "student_t"
    nu = model.nu() if is_t else None
    losses = []
    for t_anchor in anchors.tolist():
        t_anchor = int(t_anchor)
        z_a = z_post[:, t_anchor, :]
        var_a = var_post[:, t_anchor, :]
        mu, var = model.rollout(z_a, var_a, horizons)  # (B, H, D); var == scale² for t
        for i, h in enumerate(h_sorted):
            y_h = y_seq[:, t_anchor + h, :]  # (B, D)
            if is_t:
                losses.append(student_t_nll(y_h, mu[:, i, :], var[:, i, :], nu))
            else:
                losses.append(gaussian_nll(y_h, mu[:, i, :], var[:, i, :]))
    return torch.stack(losses).mean()


def _quiescent_drift_penalty(
    model: RKNEngine,
    z_post: torch.Tensor,
    transient_w: torch.Tensor,
) -> torch.Tensor:
    """Persistence regulariser on the transition increment over QUIESCENT steps.

    The autonomous rollout drift that costs the v0 engine its bulk CRPS is the
    transition-mean increment f_θ(z) accumulating off the stationary Dα baseline.
    Here we penalise ||f_θ(z_post)||² at every (b, t), weighting each step by its
    QUIESCENCE = 1 - (transient mass in its forecast window, normalised to [0,1]).
    A purely quiescent step (no ELM in its horizon window) is pulled toward
    persistence (Δz → 0, transition → identity); an ELM-active step keeps (close
    to) full freedom to move the latent.  This shrinks the rollout residual on the
    96.8% quiescent bulk WITHOUT touching the predictive σ (no CRPS/NLL zero-sum)
    and WITHOUT making the predict step input-aware (the rollout stays autonomous).

    Parameters
    ----------
    z_post : (B, T, L) filtered posterior latent means.
    transient_w : (B, T) per-step transient mass (>= 0; not yet normalised).
    """
    B, T, L = z_post.shape  # noqa: N806
    delta = model.trans_mean(z_post.reshape(B * T, L))  # (B*T, L)
    incr_sq = (delta * delta).sum(dim=-1).reshape(B, T)  # (B, T) ||f_θ(z)||²
    # Per-batch normalise the transient mass to [0,1]; quiescence = 1 - that.
    tw = transient_w.clamp_min(0.0)
    tw_max = tw.amax(dim=1, keepdim=True).clamp_min(1.0)
    quiescence = 1.0 - (tw / tw_max)  # (B, T) in [0,1]
    denom = quiescence.sum().clamp_min(1.0)
    return (quiescence * incr_sq).sum() / denom


def train_engine(
    model: RKNEngine,
    x_train: list[np.ndarray],
    y_train: list[np.ndarray],
    cfg: EngineConfig,
    device: str = "cpu",
    stop_flag=None,
    grounding_ctx=None,
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
    grounding_ctx : optional ``gs.grounding.GroundingContext`` (S8-T6).  When
        supplied AND ``cfg.grounding``, the joint loss adds the per-campaign GS
        grounding terms (L_data raw-magnetics reconstruction NLL through the T2
        operator + the L_GS force-balance soft prior) on the windows whose shot
        has a campaign operator.  The window→run order MUST match the order
        produced by ``_build_training_windows(..., return_run_index=True)`` with
        the SAME seq_len/seed (the experiment builds the context that way).
    """
    # S7.5 PERF FIX: cap intra-op threads.  The model is tiny (latent≤32) so the
    # forward+backward is overhead-bound; torch's default 48-thread pool thrashes
    # and inflates the per-batch time ~100x (3 s vs ~30 ms at 4-8 threads).  This —
    # NOT the drift regulariser — was the S7.4 "85x slower per batch" finding (the
    # 48-thread cost is present with drift-reg OFF too; the penalty itself adds
    # only ~7%).  We restore the original thread count after training.
    _saved_threads = torch.get_num_threads()
    if cfg.num_threads is not None:
        torch.set_num_threads(int(cfg.num_threads))
        logger.info(
            "[engine] torch intra-op threads capped %d -> %d (perf; model is tiny)",
            _saved_threads,
            cfg.num_threads,
        )

    model = model.to(device)
    model.train()
    # Freeze ν when fixed: keep it out of the optimiser so it stays at the init.
    if cfg.emission == "student_t" and not cfg.student_t_learn_nu:
        model.t_log_nu.requires_grad_(False)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    torch.manual_seed(cfg.seed)
    rng = torch.Generator()
    rng.manual_seed(cfg.seed + 7)

    # Build fixed-length training windows (contiguous slices of each shot run).
    # With grounding, also return the per-window source-run index so the window
    # order matches the GroundingContext's per-window signature list (built with
    # the SAME seq_len/seed) — the selection itself is unchanged either way.
    use_grounding = bool(
        cfg.grounding and grounding_ctx is not None and model.grounding_head is not None
    )
    if use_grounding:
        windows_x, windows_y, _window_runidx = _build_training_windows(
            x_train, y_train, cfg.seq_len, seed=cfg.seed, return_run_index=True
        )
    else:
        windows_x, windows_y = _build_training_windows(
            x_train, y_train, cfg.seq_len, seed=cfg.seed
        )
    if not windows_x:
        raise RuntimeError(
            f"No training windows of length {cfg.seq_len} — shots too short."
        )
    if use_grounding:
        win_sig = grounding_ctx.window_signature
        if len(win_sig) != len(windows_x):
            raise RuntimeError(
                f"GroundingContext window count {len(win_sig)} != training "
                f"window count {len(windows_x)} — seq_len/seed mismatch."
            )
        from imas_ambix.gs.grounding import grounding_losses  # noqa: PLC0415

        ep_gs_data = 0.0
        ep_gs_prior = 0.0
    X = torch.from_numpy(np.stack(windows_x)).float()  # (N, T, F)
    Y = torch.from_numpy(np.stack(windows_y)).float()  # (N, T, D)
    # Per-window, per-timestep transient weight ∝ ELM mass in [t, t+h_max].
    # Used to bias rollout-anchor sampling toward transients (acceptance fix).
    W = _build_transient_weights(windows_y, max(cfg.train_horizons))  # (N, T)
    n = X.shape[0]
    logger.info(
        "Engine training: %d windows of length %d (latent=%d, F=%d, D=%d)",
        n,
        cfg.seq_len,
        cfg.latent_dim,
        cfg.input_dim,
        cfg.output_dim,
    )

    state = TrainState()
    t0 = time.time()
    np_rng = np.random.default_rng(cfg.seed + 11)
    epoch_loop_complete = False
    for epoch in range(cfg.n_epochs):
        if stop_flag is not None and stop_flag():
            logger.info(
                "[engine] STOP-FILE / soft-limit → clean exit at epoch %d", epoch
            )
            break
        perm = np_rng.permutation(n)
        ep_loss = ep_filt = ep_roll = 0.0
        nb = 0
        for start in range(0, n, cfg.batch_size):
            idx = perm[start : start + cfg.batch_size]
            xb = X[idx].to(device)
            yb = Y[idx].to(device)

            wb = W[idx].to(device)  # (B, T) transient weights
            z_post, var_post, obs_mu, obs_var = model.filter_sequence(xb)
            # obs_var == the t SCALE² when emission == "student_t".
            if cfg.emission == "student_t":
                filt = student_t_nll(yb, obs_mu, obs_var, model.nu())
            else:
                filt = gaussian_nll(yb, obs_mu, obs_var)
            # Batch-mean transient weight per timestep drives anchor sampling.
            roll = _sample_anchor_rollout_loss(
                model,
                z_post,
                var_post,
                yb,
                cfg.train_horizons,
                rng,
                transient_w=wb.mean(dim=0),
            )
            loss = cfg.filter_loss_weight * filt + cfg.rollout_loss_weight * roll
            if cfg.drift_reg_weight > 0.0:
                drift = _quiescent_drift_penalty(model, z_post, wb)
                loss = loss + cfg.drift_reg_weight * drift

            # --- S8-T6 GS grounding terms (per-campaign subset of the batch) --
            # For each campaign signature present in this batch, gather the
            # subset's filtered z_post + the NORMALISED inputs xb at ALL
            # timesteps (the operator de-normalises internally), flatten over
            # (window, time), and add gs_data_weight·L_data + L_GS.  Windows
            # whose shot has no operator (signature None) contribute Dα loss
            # only — the ungrounded path for them is untouched.
            if use_grounding:
                sigs_batch = [win_sig[int(i)] for i in idx]
                # group window-positions in the batch by signature
                by_sig: dict[str, list[int]] = {}
                for bpos, sg in enumerate(sigs_batch):
                    if sg is not None and sg in grounding_ctx.by_signature:
                        by_sig.setdefault(sg, []).append(bpos)
                gs_terms = []
                for sg, positions in by_sig.items():
                    cg = grounding_ctx.by_signature[sg]
                    sel = torch.tensor(positions, dtype=torch.long, device=device)
                    z_sub = z_post.index_select(0, sel)  # (b, T, L)
                    x_sub = xb.index_select(0, sel)  # (b, T, F)
                    z_fl = z_sub.reshape(-1, z_sub.shape[-1])  # (b*T, L)
                    x_fl = x_sub.reshape(-1, x_sub.shape[-1])  # (b*T, F)
                    l_data, l_gs, _ = grounding_losses(
                        model.grounding_head, z_fl, x_fl, cg, grounding_ctx.gs_lambda
                    )
                    gs_terms.append(
                        (grounding_ctx.gs_data_weight * l_data + l_gs, l_data, l_gs)
                    )
                if gs_terms:
                    gs_total = torch.stack([t[0] for t in gs_terms]).mean()
                    loss = loss + gs_total
                    ep_gs_data += float(
                        torch.stack([t[1] for t in gs_terms]).mean().item()
                    )
                    ep_gs_prior += float(
                        torch.stack([t[2] for t in gs_terms]).mean().item()
                    )

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
        if use_grounding:
            state.epoch_gs_data.append(ep_gs_data / max(nb, 1))
            state.epoch_gs_prior.append(ep_gs_prior / max(nb, 1))
            ep_gs_data = 0.0
            ep_gs_prior = 0.0
        if epoch % 5 == 0 or epoch == cfg.n_epochs - 1:
            gs_msg = (
                f"  gs_data={state.epoch_gs_data[-1]:.4f}  gs_prior={state.epoch_gs_prior[-1]:.4f}"
                if use_grounding and state.epoch_gs_data
                else ""
            )
            logger.info(
                "  epoch %3d/%d  loss=%.4f  filt_nll=%.4f  roll_nll=%.4f%s  (%.1fs)",
                epoch,
                cfg.n_epochs,
                state.epoch_losses[-1],
                state.epoch_filter_nll[-1],
                state.epoch_rollout_nll[-1],
                gs_msg,
                time.time() - t0,
            )
    epoch_loop_complete = True
    state.seconds = time.time() - t0
    model.eval()
    # Restore the process-wide thread count we changed for the training run.
    if cfg.num_threads is not None:
        torch.set_num_threads(_saved_threads)
    _ = epoch_loop_complete  # documents normal completion (no early raise)
    return state


def _build_training_windows(
    x_list: list[np.ndarray],
    y_list: list[np.ndarray],
    seq_len: int,
    max_windows_per_shot: int = 8,
    seed: int = 0,
    return_run_index: bool = False,
):
    """Slice each contiguous shot run into fixed-length training windows.

    Non-overlapping windows of length ``seq_len`` are taken from each run; if a
    run yields more than ``max_windows_per_shot``, a random subset is kept (so a
    handful of very long shots do not dominate the batch distribution).

    ``return_run_index`` (S8-T6, additive): also return a per-window int index
    into ``x_list`` (the source run), so the GS grounding can map each window to
    its shot's campaign operator.  The window selection is UNCHANGED (same rng,
    same order) whether or not the index is returned — the ungrounded path is
    bit-identical.
    """
    rng = np.random.default_rng(seed)
    wx: list[np.ndarray] = []
    wy: list[np.ndarray] = []
    wr: list[int] = []
    for ri, (x, y) in enumerate(zip(x_list, y_list, strict=True)):
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
            wr.append(ri)
    if return_run_index:
        return wx, wy, wr
    return wx, wy


def _build_transient_weights(windows_y: list[np.ndarray], h_max: int) -> torch.Tensor:
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

_DT_NOMINAL = 1.0e-3  # 1 kHz model grid
_DT_TOL_FRAC = 0.2  # split a run where |dt - dt_med| > 20% dt_med
_MIN_RUN_LEN = 32  # discard contiguous runs shorter than this
_DEAD_DALPHA_STD = 1e-6  # Dα std below this → dead/disconnected filterscope


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
            int(sid),
            feature_schema,
            target_channels,
            level1_dir=level1_dir,
            max_slices=None,
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
            logger.info(
                "  loaded %d/%d shots (%.0fs)", k + 1, len(sids), time.time() - t0
            )
    logger.info(
        "[%s] %d shots → %d runs (%d ok, %d none, %d dead) in %.0fs",
        cache_tag,
        len(sids),
        len(runs),
        n_ok,
        n_none,
        n_dead,
        time.time() - t0,
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


def _dense_transient_anchors(run: ShotRun, h_max: int, pad: int = 5) -> np.ndarray:
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
    max_train_shots: int = 500  # bound corpus for tractable v0 (reported, not silent)
    max_cal_shots: int = 300
    max_ood_shots: int = 200
    horizons: tuple[int, ...] = (1, 2, 5, 10, 20)  # steps @ 1 kHz
    n_epochs: int = 30
    seq_len: int = 64
    seed: int = 0
    do_smoothing: bool = False  # optional mode, only if budget remains
    device: str = "cpu"
    # S7.4 levers (bounded iteration; reported, not silent)
    drift_reg_weight: float = 0.0  # quiescent persistence regulariser (0 = v0)
    train_horizons: tuple[int, ...] | None = None  # override training rollout horizons
    # S7.5 levers
    emission: str = "gaussian"  # "gaussian" (v0/v1) or "student_t" (heavy-tail head)
    student_t_learn_nu: bool = True
    student_t_nu: float = 5.0
    num_threads: int | None = 4  # cap intra-op threads in training (perf)
    # S8-T6 GS grounding (0/off = the ungrounded v2 path; both stay runnable)
    grounding: bool = False
    gs_profile_order: int = 1
    gs_passive_rank: int = 4
    gs_lambda: float = 1e-2
    gs_data_weight: float = 0.1


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
        mu_n, var_n = forecast_pairs(
            model, x_norm, anchors, list(horizons), device=device
        )
        if mu_n.shape[0] == 0:
            continue
        # forecast_pairs filters internally; it returns only valid anchors
        # (t + h_max < T).  Re-derive which anchors were valid to align truths.
        T = run.X.shape[0]
        h_max = max(horizons)
        valid = [int(a) for a in anchors if int(a) + h_max < T]
        # Denormalise mean (target std scaling) and variance (std² scaling).
        mu_phys = stats.denormalise_y_mean(mu_n.reshape(-1, mu_n.shape[-1])).reshape(
            mu_n.shape
        )
        var_phys = var_n * (stats.target_std**2)[np.newaxis, np.newaxis, :]
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
    transient_flag: np.ndarray | None = None,
    nu: float | None = None,
) -> dict:
    """Per-horizon CRPS / NLL (raw σ) + coverage / PI-width (conformal σ).

    mu, var, y : (P, H, D).  CRPS/NLL use raw σ (baseline convention); coverage
    uses the per-horizon conformal scale ``conformal_q[h]`` if supplied.

    nu : if not None, ``var`` is interpreted as the Student-t SCALE² (location-
    scale parameter) and CRPS/NLL use the closed-form Student-t scores (S7.5
    heavy-tail head).  The reported ``mean_sigma_raw`` is then the predictive
    STD = scale·√(ν/(ν-2)) (so the σ-vs-rmse diagnostic stays comparable), while
    the conformal interval rescales the predictive std — conformal coverage is
    distribution-corrected either way.  ``nu`` None → Gaussian (v0/v1 behaviour).

    transient_flag : (P, H) bool.  If supplied, CRPS is ALSO reported on the
    subset where the FORECAST TARGET Dα_{t+h} is itself ELM-active — the
    transient-target stratum where "calibrated widening beats confidently-narrow"
    is supposed to live (the task's "explicit reporting on transient windows").
    The overall (window-straddles-ELM) aggregate is ~97% quiescent and hides
    this; the transient-target subset is the acceptance-relevant slice.
    """
    from imas_ambix.statespace.calibration import (  # noqa: PLC0415
        crps_gaussian,
        interval_coverage,
        nll_gaussian,
        prediction_interval_width,
    )

    is_t = nu is not None
    # For a t_ν the predictive std relates to the scale by √(ν/(ν-2)).
    std_factor = math.sqrt(nu / (nu - 2.0)) if (is_t and nu > 2.0) else 1.0

    def _crps(yv, mv, scale_or_sigma):
        if is_t:
            return crps_student_t(yv, mv, scale_or_sigma, nu)
        return crps_gaussian(yv, mv, scale_or_sigma)

    def _nll(yv, mv, scale_or_sigma):
        if is_t:
            return student_t_nll_np(yv, mv, scale_or_sigma**2, nu)
        return nll_gaussian(yv, mv, scale_or_sigma)

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
        mo, vo, yto = m[ok], v[ok], yt[ok]
        # ``scale_raw`` is the t scale (or the Gaussian σ); ``std`` is the
        # predictive std reported for the σ-vs-rmse diagnostic.
        scale_raw = np.sqrt(np.maximum(vo, 1e-12))
        std = scale_raw * std_factor
        rec = {
            "n": int(ok.sum()),
            "crps_raw": float(_crps(yto, mo, scale_raw)),
            "nll_raw": float(_nll(yto, mo, scale_raw)),
            "rmse": float(np.sqrt(np.mean((yto - mo) ** 2))),
            "mean_sigma_raw": float(np.mean(std)),
        }
        if conformal_q is not None and h in conformal_q:
            from scipy.stats import norm  # noqa: PLC0415

            z = float(norm.ppf(0.95))
            # Conformal rescales the PREDICTIVE STD; the residual-quantile fit
            # (fit_horizon_conformal) used the same std, so coverage is correct
            # for either emission.
            sigma_conf = conformal_q[h] * std / z
            rec["coverage_90_conf"] = float(
                interval_coverage(yto, mo, sigma_conf, alpha=0.10)
            )
            rec["pi_width_90_conf"] = float(
                prediction_interval_width(sigma_conf, alpha=0.10)
            )
        # Transient-target stratum: forecast TARGET Dα_{t+h} is ELM-active.
        if transient_flag is not None:
            tf = transient_flag[:, i] & ok
            if tf.sum() >= 10:
                mt = mu[tf, i, 0]
                vt = var[tf, i, 0]
                ytt = y[tf, i, 0]
                st = np.sqrt(np.maximum(vt, 1e-12))  # t scale or Gaussian σ
                rec["n_transient"] = int(tf.sum())
                rec["crps_raw_transient"] = float(_crps(ytt, mt, st))
                rec["nll_raw_transient"] = float(_nll(ytt, mt, st))
                rec["rmse_transient"] = float(np.sqrt(np.mean((ytt - mt) ** 2)))
                rec["mean_sigma_raw_transient"] = float(np.mean(st * std_factor))
            else:
                rec["n_transient"] = int(tf.sum())
        out[str(h)] = rec
    logger.info(
        "[%s] per-horizon CRPS(raw) all=%s transient=%s",
        label,
        {h: round(out[str(h)].get("crps_raw", float("nan")), 4) for h in h_sorted},
        {
            h: round(out[str(h)].get("crps_raw_transient", float("nan")), 4)
            for h in h_sorted
        },
    )
    return out


def _fit_ensemble_clipped(
    ensemble,
    x_train: np.ndarray,
    y_train: np.ndarray,
    cfg,
    grad_clip: float = 5.0,
) -> None:
    """Train a baseline DeepEnsemble with global-norm gradient clipping.

    In-scope replacement for ``DeepEnsemble.fit`` that reuses MLPGaussian's
    PUBLIC ``nll_and_grads`` / ``adam_step`` API (no edit to baseline.py) and
    inserts a global L2-norm gradient clip between them — mirroring the engine's
    ``clip_grad_norm_(…, 5.0)``.  This bounds the runaway step that otherwise
    diverges the unclipped MLP on dense ELM-spike targets.  Same minibatch
    schedule, seeds and Adam as ``MLPGaussian.fit_sgd`` otherwise.
    """
    n = x_train.shape[0]
    for i, m in enumerate(ensemble.members):
        t0 = time.time()
        rng = np.random.default_rng(cfg.seed_base + i + 999)
        last = float("nan")
        for _epoch in range(cfg.n_epochs):
            perm = rng.permutation(n)
            ep = 0.0
            nb = 0
            for start in range(0, n, cfg.batch_size):
                idx = perm[start : start + cfg.batch_size]
                loss, grads = m.nll_and_grads(x_train[idx], y_train[idx])
                # global L2 grad-norm clip
                total = math.sqrt(sum(float(np.sum(g * g)) for g in grads))
                if total > grad_clip and total > 0:
                    scale = grad_clip / total
                    grads = [g * scale for g in grads]
                m.adam_step(grads, lr=cfg.lr)
                ep += loss
                nb += 1
            last = ep / max(nb, 1)
        logger.info(
            "  [static-clip] member %d/%d final NLL=%.4f (%.0fs)",
            i + 1,
            len(ensemble.members),
            last,
            time.time() - t0,
        )


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
    metrics: dict = {
        "config": {
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
            "drift_reg_weight": cfg.drift_reg_weight,
            "train_horizons": list(cfg.train_horizons)
            if cfg.train_horizons is not None
            else list(cfg.horizons),
            "emission": cfg.emission,
            "student_t_learn_nu": cfg.student_t_learn_nu,
            "student_t_nu_init": cfg.student_t_nu,
            "num_threads": cfg.num_threads,
        }
    }

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
    train_runs = _load_split_runs(
        train_shots, fs, tc, _LEVEL1_DIR, cfg.max_train_shots, cfg.seed + 1, "train"
    )
    conf_runs = _load_split_runs(
        conf_cal_shots, fs, tc, _LEVEL1_DIR, cfg.max_cal_shots, cfg.seed + 2, "conf"
    )
    idt_runs = _load_split_runs(
        in_dist_test_shots, fs, tc, _LEVEL1_DIR, cfg.max_cal_shots, cfg.seed + 3, "idt"
    )
    ood_runs = _load_split_runs(
        ood_shots, fs, tc, _LEVEL1_DIR, cfg.max_ood_shots, cfg.seed + 4, "ood"
    )
    metrics["split_sizes"] = {
        "n_train_runs": len(train_runs),
        "n_train_slices": int(sum(len(r.X) for r in train_runs)),
        "n_conf_runs": len(conf_runs),
        "n_idt_runs": len(idt_runs),
        "n_ood_runs": len(ood_runs),
    }
    if not train_runs:
        raise RuntimeError("No training runs loaded")

    # --- 3. shared normaliser (train only) ----------------------------------
    stats = ChannelStats.fit(
        [r.X.astype(np.float64) for r in train_runs],
        [r.y.astype(np.float64) for r in train_runs],
    )
    input_dim = train_runs[0].X.shape[1]
    output_dim = train_runs[0].y.shape[1]

    # --- 4. train RKN engine -------------------------------------------------
    train_h = cfg.train_horizons if cfg.train_horizons is not None else cfg.horizons
    eng_cfg = EngineConfig(
        input_dim=input_dim,
        latent_dim=cfg.latent_dim,
        output_dim=output_dim,
        n_epochs=cfg.n_epochs,
        seq_len=cfg.seq_len,
        train_horizons=tuple(train_h),
        seed=cfg.seed,
        drift_reg_weight=cfg.drift_reg_weight,
        emission=cfg.emission,
        student_t_learn_nu=cfg.student_t_learn_nu,
        student_t_nu=cfg.student_t_nu,
        num_threads=cfg.num_threads,
        grounding=cfg.grounding,
        gs_profile_order=cfg.gs_profile_order,
        gs_passive_rank=cfg.gs_passive_rank,
        gs_lambda=cfg.gs_lambda,
        gs_data_weight=cfg.gs_data_weight,
    )
    x_train_n = [stats.normalise_X(r.X.astype(np.float64)) for r in train_runs]
    y_train_n = [stats.normalise_y(r.y.astype(np.float64)) for r in train_runs]
    model = RKNEngine(eng_cfg)
    stop = _make_stop_flag()
    # --- S8-T6: build the GS grounding context (per-campaign operators +
    # whitening + per-window signature) when grounding is on. ---------------
    grounding_ctx = None
    if cfg.grounding:
        from imas_ambix.gs.grounding import build_grounding_context  # noqa: PLC0415

        grounding_ctx = build_grounding_context(
            train_runs,
            stats,
            fs,
            profile_order=cfg.gs_profile_order,
            passive_rank=cfg.gs_passive_rank,
            lam=cfg.gs_lambda,
            gs_data_weight=cfg.gs_data_weight,
            seq_len=cfg.seq_len,
            seed=cfg.seed,
        )
        metrics["grounding_coverage"] = {
            "n_campaign_operators": len(grounding_ctx.by_signature),
            "n_grounded_windows": grounding_ctx.n_grounded_windows,
            "n_total_windows": grounding_ctx.n_total_windows,
            "grounded_timestep_fraction": grounding_ctx.grounded_timestep_fraction,
            "campaign_window_counts": grounding_ctx.campaign_window_counts,
            "gs_profile_order": cfg.gs_profile_order,
            "gs_passive_rank": cfg.gs_passive_rank,
            "gs_lambda": cfg.gs_lambda,
            "gs_data_weight": cfg.gs_data_weight,
            "collinearity_fix": (
                "STRUCTURAL — head emits exactly the locked order-1 plasma poly "
                "(3 DOF) + rank-4 passive SVD the standalone frontier found "
                "near-vacuum-sound at lambda=0 (q1 net-current ratio 0.030). "
                "gs_lambda only biases toward small currents (current-space L2 "
                "Tikhonov M=blkdiag(B^TB, V^TV)); data overrides (soft)."
            ),
        }
    tstate = train_engine(
        model,
        x_train_n,
        y_train_n,
        eng_cfg,
        device=cfg.device,
        stop_flag=stop,
        grounding_ctx=grounding_ctx,
    )
    # S8-T6: capture the trained model + grounding context for the grounding
    # evaluators (run_grounding_experiment reuses these — no second train).
    if cfg.grounding:
        global _LAST_MODEL, _LAST_GROUNDING_CTX, _LAST_TRAIN_RUNS, _LAST_STATS
        _LAST_MODEL = model
        _LAST_GROUNDING_CTX = grounding_ctx
        _LAST_TRAIN_RUNS = train_runs
        _LAST_STATS = stats
    # Student-t dof to pass into the scoring path (None for Gaussian).
    eng_nu = float(model.nu()[0].item()) if cfg.emission == "student_t" else None
    metrics["engine_train"] = {
        "epochs_run": len(tstate.epoch_losses),
        "final_loss": tstate.epoch_losses[-1] if tstate.epoch_losses else None,
        "final_filter_nll": tstate.epoch_filter_nll[-1]
        if tstate.epoch_filter_nll
        else None,
        "final_rollout_nll": tstate.epoch_rollout_nll[-1]
        if tstate.epoch_rollout_nll
        else None,
        "seconds": round(tstate.seconds, 1),
        "emission": cfg.emission,
        "student_t_nu_learned": eng_nu,
        "grounding": cfg.grounding,
        "final_gs_data_nll": tstate.epoch_gs_data[-1] if tstate.epoch_gs_data else None,
        "final_gs_prior": tstate.epoch_gs_prior[-1] if tstate.epoch_gs_prior else None,
    }

    # --- 5. retrain static comparator (baseline classes, same train slices) --
    logger.info("Retraining static comparator (deep ensemble) on same train runs...")
    Xtr, ytr = _stack_runs_for_static(train_runs)
    Xtr_n = stats.normalise_X(Xtr.astype(np.float64))
    ytr_n = stats.normalise_y(ytr.astype(np.float64))
    ens_cfg = EnsembleConfig(n_members=5, n_epochs=60, hidden_size=128, seed_base=0)
    ensemble = DeepEnsemble.build(input_dim, output_dim, ens_cfg)
    # NOTE: train with a GRADIENT-CLIPPED loop (in-scope; uses MLPGaussian's
    # public nll_and_grads/adam_step API).  baseline.MLPGaussian.fit_sgd has NO
    # grad clipping, which DIVERGES on dense un-decimated ELM-spike targets
    # (seed-dependent: members get NLL≈+7/+9, μ/σ explode → static rmse≈25,
    # σ≈87 → a meaningless comparator).  S7.2 never hit this because its
    # max_slices_per_shot=200 linspace-decimation aliased the ms-scale spikes
    # away — exactly what S7.3 must NOT do.  Clipping the global grad norm at 5.0
    # (mirrors the engine's clip_grad_norm_) makes all seeds converge.  The
    # missing clip in baseline.py is a latent bug → recommended followup for the
    # orchestrator to fix at source (also hardens S7.2 on un-decimated data).
    # Static comparator MUST train clipped (task spec): an unclipped MLP diverges
    # on dense ELM-spike targets → meaningless strawman.  We REUSE the existing
    # _fit_ensemble_clipped with grad_clip=5.0 — bit-identical to the S7.3 (v0)
    # static fit — so the v0→v1 comparison is purely engine-driven (the static
    # comparator does not move between versions; clean head-to-head).  This
    # mirrors the engine's own clip_grad_norm_(…, 5.0).
    _fit_ensemble_clipped(ensemble, Xtr_n, ytr_n, ens_cfg, grad_clip=5.0)
    # static conformal at h=0 (S7.2-style split-conformal on the dense runs)
    Xcf, ycf = _stack_runs_for_static(conf_runs)
    static_conf = ConformalWrapper(ensemble, stats)
    static_conf.fit_conformal(
        stats.normalise_X(Xcf.astype(np.float64)),
        stats.normalise_y(ycf.astype(np.float64)),
    )

    # --- 6. FILTERING eval (engine, causal, in-dist-test) -------------------
    metrics["filtering"] = _eval_filtering(
        model, idt_runs, conf_runs, stats, cfg.device, nu=eng_nu
    )

    # --- 7. dense transient anchors (shared selector) -----------------------
    h_max = max(cfg.horizons)
    idt_anchors = [_dense_transient_anchors(r, h_max) for r in idt_runs]
    conf_anchors = [_dense_transient_anchors(r, h_max) for r in conf_runs]
    ood_anchors = [_dense_transient_anchors(r, h_max) for r in ood_runs]
    n_idt_pairs = int(sum(len(a) for a in idt_anchors))
    logger.info(
        "Dense transient anchors: idt=%d conf=%d ood=%d",
        n_idt_pairs,
        int(sum(len(a) for a in conf_anchors)),
        int(sum(len(a) for a in ood_anchors)),
    )

    # --- 8. per-horizon conformal calibration on CONF runs (dense windows) --
    # Engine conformal: from engine forecasts on conf anchors.
    eng_std_factor = (
        math.sqrt(eng_nu / (eng_nu - 2.0))
        if (eng_nu is not None and eng_nu > 2.0)
        else 1.0
    )
    eng_mu_cf, eng_var_cf, eng_y_cf = _engine_horizon_pairs(
        model, conf_runs, conf_anchors, cfg.horizons, stats, cfg.device
    )
    eng_q = _per_horizon_q(
        eng_mu_cf,
        eng_var_cf,
        eng_y_cf,
        cfg.horizons,
        fit_horizon_conformal,
        std_factor=eng_std_factor,
    )
    # Static conformal: static map at each horizon on conf anchors.
    stat_mu_cf, stat_var_cf, stat_y_cf = _static_horizon_predict(
        ensemble, stats, conf_runs, conf_anchors, cfg.horizons
    )
    stat_q = _per_horizon_q(
        stat_mu_cf, stat_var_cf, stat_y_cf, cfg.horizons, fit_horizon_conformal
    )

    # --- 9. FORECASTING comparison on the SAME dense transient windows ------
    eng_mu, eng_var, eng_y = _engine_horizon_pairs(
        model, idt_runs, idt_anchors, cfg.horizons, stats, cfg.device
    )
    stat_mu, stat_var, stat_y = _static_horizon_predict(
        ensemble, stats, idt_runs, idt_anchors, cfg.horizons
    )
    # Per-(pair,horizon) flag: is the forecast TARGET Dα_{t+h} itself ELM-active?
    idt_tflag = _target_transient_flag(idt_runs, idt_anchors, cfg.horizons)
    # Verify the two models scored the SAME (t, t+h) truths (identical windows).
    same_truth = _verify_same_truths(eng_y, stat_y)
    # Dump per-pair arrays to scratch for offline stratification (last run).
    _dump_pairs(
        "idt", eng_mu, eng_var, stat_mu, stat_var, eng_y, idt_tflag, cfg.horizons
    )
    metrics["forecasting_indist_dense_transient"] = {
        "n_anchor_pairs": int(eng_mu.shape[0]),
        "same_truths_engine_vs_static": same_truth,
        "engine": _score_horizons(
            "ENGINE/idt",
            eng_mu,
            eng_var,
            eng_y,
            cfg.horizons,
            eng_q,
            idt_tflag,
            nu=eng_nu,
        ),
        "static": _score_horizons(
            "STATIC/idt", stat_mu, stat_var, stat_y, cfg.horizons, stat_q, idt_tflag
        ),
    }

    # --- 10. OOD honesty (forecasting on OOD dense transient windows) -------
    eng_mu_o, eng_var_o, eng_y_o = _engine_horizon_pairs(
        model, ood_runs, ood_anchors, cfg.horizons, stats, cfg.device
    )
    stat_mu_o, stat_var_o, stat_y_o = _static_horizon_predict(
        ensemble, stats, ood_runs, ood_anchors, cfg.horizons
    )
    ood_tflag = _target_transient_flag(ood_runs, ood_anchors, cfg.horizons)
    _dump_pairs(
        "ood",
        eng_mu_o,
        eng_var_o,
        stat_mu_o,
        stat_var_o,
        eng_y_o,
        ood_tflag,
        cfg.horizons,
    )
    metrics["forecasting_ood_dense_transient"] = {
        "n_anchor_pairs": int(eng_mu_o.shape[0]),
        "engine": _score_horizons(
            "ENGINE/ood",
            eng_mu_o,
            eng_var_o,
            eng_y_o,
            cfg.horizons,
            eng_q,
            ood_tflag,
            nu=eng_nu,
        ),
        "static": _score_horizons(
            "STATIC/ood",
            stat_mu_o,
            stat_var_o,
            stat_y_o,
            cfg.horizons,
            stat_q,
            ood_tflag,
        ),
    }
    metrics["ood_honesty"] = _ood_honesty(
        model,
        ensemble,
        idt_runs,
        ood_runs,
        stats,
        regime_scalars,
        train_runs,
        cfg.device,
        nu=eng_nu,
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


def _per_horizon_q(
    mu, var, y, horizons, fit_fn, std_factor: float = 1.0
) -> dict[int, float]:
    """Fit a split-conformal scale q̂ per horizon from calibration forecasts.

    ``std_factor`` (S7.5 Student-t): the conformal residual-quantile is fit on
    the PREDICTIVE STD = scale·std_factor (std_factor = √(ν/(ν-2)) for a t_ν),
    matching what ``_score_horizons`` rescales for coverage.  1.0 = Gaussian.
    """
    q: dict[int, float] = {}
    for i, h in enumerate(sorted(horizons)):
        m, v, yt = mu[:, i, 0], var[:, i, 0], y[:, i, 0]
        ok = np.isfinite(yt) & np.isfinite(m) & np.isfinite(v)
        if ok.sum() < 10:
            q[h] = 1.0
            continue
        std = np.sqrt(np.maximum(v[ok], 1e-12)) * std_factor
        q[h] = fit_fn(yt[ok], m[ok], std, alpha=0.10)
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
        mu_p = stats.denormalise_y_mean(mu_n)  # (P, D)
        var_p = (stats.denormalise_y_std(sigma_n)) ** 2  # (P, D)
        T, D = run.y.shape
        P = len(valid)
        # Broadcast the SAME static prediction across all horizons (frozen nowcast)
        mu_h = np.repeat(mu_p[:, np.newaxis, :], H, axis=1)  # (P, H, D)
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


def _target_transient_flag(runs, anchors_per_run, horizons) -> np.ndarray:
    """(P, H) bool: is the forecast TARGET Dα_{t+h} itself ELM-active?

    Aligned EXACTLY to the engine/static (run, valid-anchor, horizon) triples
    (same valid rule: t + h_max < T).  Uses baseline.compute_transient_mask on
    each run's full-1 kHz Dα and indexes the mask at t+h.
    """
    from imas_ambix.statespace.baseline import compute_transient_mask  # noqa: PLC0415

    H = len(horizons)
    h_sorted = sorted(horizons)
    h_max = max(horizons)
    flags = []
    for run, anchors in zip(runs, anchors_per_run, strict=True):
        T = run.y.shape[0]
        valid = [int(a) for a in anchors if int(a) + h_max < T]
        if not valid:
            continue
        tmask = compute_transient_mask(run.y)  # (T,)
        for t in valid:
            row = np.zeros(H, dtype=bool)
            for i, h in enumerate(h_sorted):
                if t + h < T:
                    row[i] = bool(tmask[t + h])
            flags.append(row)
    if not flags:
        return np.zeros((0, H), dtype=bool)
    return np.stack(flags)


def _dump_pairs(tag, eng_mu, eng_var, stat_mu, stat_var, y, tflag, horizons) -> None:
    """Dump per-(pair,horizon) arrays to /work scratch for offline stratification.

    Lets any further analysis (e.g. spike-amplitude bins) run in seconds without
    re-running the full pipeline.  Both models share the same y / tflag.
    """
    try:
        path = _SCRATCH / f"pairs_{tag}.npz"
        np.savez_compressed(
            path,
            horizons=np.array(sorted(horizons)),
            eng_mu=eng_mu[:, :, 0],
            eng_var=eng_var[:, :, 0],
            stat_mu=stat_mu[:, :, 0],
            stat_var=stat_var[:, :, 0],
            y=y[:, :, 0],
            transient_flag=tflag,
        )
        logger.info(
            "[%s] per-pair arrays dumped to %s (%d pairs)", tag, path, eng_mu.shape[0]
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("Failed to dump pairs for %s: %s", tag, e)


def _eval_filtering(
    model,
    idt_runs,
    conf_runs,
    stats,
    device,
    burn_in: int = 20,
    nu: float | None = None,
) -> dict:
    """Causal filtering coverage / CRPS / NLL on in-dist-test runs.

    Coverage uses split-conformal fit on conf runs' filtered residuals.

    ``nu`` (S7.5): if set, the filtered ``var`` is the Student-t SCALE² and the
    CRPS/NLL use the closed-form t scores; the conformal fit + coverage use the
    predictive std = scale·√(ν/(ν-2)).  ``nu`` None → Gaussian (v0/v1 behaviour).

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

    is_t = nu is not None
    std_factor = math.sqrt(nu / (nu - 2.0)) if (is_t and nu > 2.0) else 1.0

    def _collect(runs):
        # Returns y, mu, SCALE (t scale, or Gaussian σ).  Predictive std =
        # scale·std_factor is derived where coverage is computed.
        ys, mus, scales = [], [], []
        for r in runs:
            if r.X.shape[0] <= burn_in + 1:
                continue
            mu_n, var_n = filter_shot(
                model, stats.normalise_X(r.X.astype(np.float64)), device
            )
            mu_p = stats.denormalise_y_mean(mu_n)[:, 0]
            scale_p = np.sqrt(np.maximum(var_n[:, 0] * stats.target_std[0] ** 2, 1e-12))
            ys.append(r.y[burn_in:, 0])
            mus.append(mu_p[burn_in:])
            scales.append(scale_p[burn_in:])
        return (np.concatenate(ys), np.concatenate(mus), np.concatenate(scales))

    z = float(norm.ppf(0.95))
    cf_y, cf_mu, cf_scale = _collect(conf_runs)
    # conformal fits on the PREDICTIVE STD (matches coverage rescale)
    q = fit_horizon_conformal(cf_y, cf_mu, cf_scale * std_factor, alpha=0.10)

    y, mu, scale = _collect(idt_runs)
    std = scale * std_factor  # predictive std (for coverage / PI width)
    sig_conf = q * std / z
    if is_t:
        crps = crps_student_t(y, mu, scale, nu)
        nll = student_t_nll_np(y, mu, scale**2, nu)
    else:
        crps = crps_gaussian(y, mu, scale)
        nll = nll_gaussian(y, mu, scale)
    return {
        "n_slices": int(len(y)),
        "burn_in_dropped": int(burn_in),
        "conformal_q": float(q),
        "coverage_90_raw": float(interval_coverage(y, mu, std, alpha=0.10)),
        "coverage_90_conf": float(interval_coverage(y, mu, sig_conf, alpha=0.10)),
        "pi_width_90_conf": float(prediction_interval_width(sig_conf, alpha=0.10)),
        "mean_sigma_raw": float(np.mean(std)),
        "mean_sigma_conf": float(np.mean(sig_conf)),
        "crps_raw": float(crps),
        "nll_raw": float(nll),
        "rmse": float(np.sqrt(np.mean((y - mu) ** 2))),
        "emission": "student_t" if is_t else "gaussian",
    }


def _eval_smoothing(model, idt_runs, stats, device) -> dict:
    from imas_ambix.statespace.calibration import (  # noqa: PLC0415
        crps_gaussian,
        nll_gaussian,
    )
    from imas_ambix.statespace.filter import smooth_shot  # noqa: PLC0415

    ys, mus, sigs = [], [], []
    for r in idt_runs:
        mu_n, var_n = smooth_shot(
            model, stats.normalise_X(r.X.astype(np.float64)), device
        )
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


def _ood_honesty(
    model,
    ensemble,
    idt_runs,
    ood_runs,
    stats,
    regime_scalars,
    train_runs,
    device,
    nu: float | None = None,
) -> dict:
    """OOD honesty: coverage non-collapse + quantified novelty signals.

    Reports (a) engine filtering coverage on OOD vs in-dist (non-collapse),
    (b) ENGINE-NATIVE OOD-AUROC from the engine's OWN signals — the filtered
    predictive σ and the normalised filter-innovation magnitude — so the OOD
    comparison is a TRUE engine-vs-static one (S7.5 refinement #2), (c) the
    static deep-ensemble disagreement AUROC kept for reference (CLEARLY labelled
    as the static model's, not the engine's — the S7.4 artifact mislabelled it),
    and (d) coverage-vs-distance from the regime axis using the engine-native
    score.  Reported honestly — not hard-thresholded.

    ``nu`` (S7.5): when set, the filtered ``var`` is the Student-t SCALE² and the
    reported predictive σ = scale·√(ν/(ν-2)).
    """
    from imas_ambix.statespace.calibration import (  # noqa: PLC0415
        coverage_vs_distance,
        ensemble_disagreement,
        interval_coverage,
        ood_auroc,
    )
    from imas_ambix.statespace.filter import (  # noqa: PLC0415
        filter_innovation_shot,
        filter_shot,
    )

    is_t = nu is not None
    std_factor = math.sqrt(nu / (nu - 2.0)) if (is_t and nu > 2.0) else 1.0

    def _cov(runs):
        # Per-slice: truth, mean, predictive std, AND the two engine-native OOD
        # scores (predictive σ itself, and the filter-innovation magnitude).
        ys, mus, sigs, innos = [], [], [], []
        for r in runs:
            x_norm = stats.normalise_X(r.X.astype(np.float64))
            mu_n, var_n = filter_shot(model, x_norm, device)
            mu_p = stats.denormalise_y_mean(mu_n)[:, 0]
            scale_p = np.sqrt(np.maximum(var_n[:, 0] * stats.target_std[0] ** 2, 1e-12))
            sig_p = scale_p * std_factor  # predictive std (t or Gaussian)
            inno_p = filter_innovation_shot(model, x_norm, device)  # (T,)
            ys.append(r.y[:, 0])
            mus.append(mu_p)
            sigs.append(sig_p)
            innos.append(inno_p)
        if not ys:
            return None, None, None, None
        return (
            np.concatenate(ys),
            np.concatenate(mus),
            np.concatenate(sigs),
            np.concatenate(innos),
        )

    y_i, mu_i, sig_i, inno_i = _cov(idt_runs)
    y_o, mu_o, sig_o, inno_o = _cov(ood_runs)
    out: dict = {}
    if y_i is not None and y_o is not None:
        # raw (unconformalised) coverage so we see the model's native widening
        out["filter_coverage90_raw_indist"] = float(
            interval_coverage(y_i, mu_i, sig_i, alpha=0.10)
        )
        out["filter_coverage90_raw_ood"] = float(
            interval_coverage(y_o, mu_o, sig_o, alpha=0.10)
        )
        out["mean_sigma_indist"] = float(np.mean(sig_i))
        out["mean_sigma_ood"] = float(np.mean(sig_o))

    # ENGINE-NATIVE OOD-AUROC (S7.5 refinement #2): the engine's own signals.
    #   (1) predictive σ magnitude — does the model widen its belief on OOD?
    #   (2) filter-innovation magnitude — does inputs_t surprise the dynamics?
    if y_i is not None and y_o is not None:
        try:
            out["ood_auroc_engine_predictive_sigma"] = float(ood_auroc(sig_i, sig_o))
        except Exception as e:  # noqa: BLE001
            out["ood_auroc_engine_predictive_sigma_error"] = str(e)
        try:
            out["ood_auroc_engine_innovation"] = float(ood_auroc(inno_i, inno_o))
        except Exception as e:  # noqa: BLE001
            out["ood_auroc_engine_innovation_error"] = str(e)
        # Headline engine-native score = the better-motivated innovation signal.
        out["ood_auroc_engine"] = out.get("ood_auroc_engine_innovation")
        out["mean_innovation_indist"] = float(np.mean(inno_i))
        out["mean_innovation_ood"] = float(np.mean(inno_o))

    # STATIC deep-ensemble disagreement AUROC — kept for reference, CLEARLY
    # labelled as the STATIC model's input-novelty score (NOT an engine signal;
    # the S7.4 artifact's "ood_auroc 0.81" was this number, mislabelled).
    Xi = np.concatenate([r.X for r in idt_runs]).astype(np.float64)
    Xo = np.concatenate([r.X for r in ood_runs]).astype(np.float64)
    _, _, ens_i = ensemble.predict(stats.normalise_X(Xi))
    _, _, ens_o = ensemble.predict(stats.normalise_X(Xo))
    try:
        out["ood_auroc_static_ensemble_disagreement"] = float(
            ood_auroc(ensemble_disagreement(ens_i), ensemble_disagreement(ens_o))
        )
        # back-compat alias (the v0/v1 key) — same static number, now labelled
        out["ood_auroc_ensemble_disagreement"] = out[
            "ood_auroc_static_ensemble_disagreement"
        ]
    except Exception as e:  # noqa: BLE001
        out["ood_auroc_error"] = str(e)

    # coverage-vs-distance on OOD runs (regime axis)
    train_ip = np.array(
        [
            regime_scalars.get(str(r.shot_id), {}).get("ip_mean", np.nan)
            for r in train_runs
        ]
    )
    train_ne = np.array(
        [
            regime_scalars.get(str(r.shot_id), {}).get("ne_mean", np.nan)
            for r in train_runs
        ]
    )
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
                if np.isfinite(ip) and np.isfinite(ne)
                else np.nan
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
    v["filtering_coverage_in_band"] = bool(fcov is not None and 0.88 <= fcov <= 0.92)
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

    # (2b) ACCEPTANCE-RELEVANT slice: CRPS on the TRANSIENT TARGET subset
    # (Dα_{t+h} itself ELM-active).  The overall aggregate above is ~97%
    # quiescent and hides the calibrated-widening mechanism; this stratum is
    # where "dynamics earn their keep" must show if it shows anywhere.
    wins_t = {}
    for h in eng:
        e = eng[h].get("crps_raw_transient")
        s = stat.get(h, {}).get("crps_raw_transient")
        if e is not None and s is not None:
            wins_t[h] = {
                "engine_crps_transient": e,
                "static_crps_transient": s,
                "n_transient": eng[h].get("n_transient"),
                "engine_wins": bool(e < s),
            }
    v["forecast_crps_transient_by_horizon"] = wins_t
    h_pos_t = [h for h in wins_t if int(h) > 0]
    v["forecast_beats_static_transient_any_Hgt0"] = bool(
        any(wins_t[h]["engine_wins"] for h in h_pos_t)
    )
    v["forecast_beats_static_transient_all_Hgt0"] = bool(
        h_pos_t and all(wins_t[h]["engine_wins"] for h in h_pos_t)
    )

    # (2c) ACCEPTANCE-RELEVANT slice — NLL on the TRANSIENT TARGET subset.
    # The re-scoped Stage-1 bar (f-s7-stage1-decision, option B) is met when the
    # engine beats the static comparator on transient-subset CRPS *OR* transient-
    # subset NLL at H>0.  NLL is the well-motivated criterion for a DYNAMICS
    # claim: a confidently-narrow static is caught CATASTROPHICALLY when an ELM
    # lands at t+h (its σ is its h=0 σ, with no widening mechanism), so its
    # transient NLL explodes with horizon; the engine's learned process noise Q
    # widens the belief through the rollout and absorbs the spike → bounded NLL.
    # CRPS, by contrast, rewards tracking the ~97%-quiescent baseline and is the
    # wrong yardstick for the dynamics claim (the re-scoping rationale).  A single
    # Gaussian σ cannot win BOTH on the violently heavy-tailed transient residual
    # (NLL-optimal σ ≈ rmse ≫ CRPS-optimal σ ≈ typical |residual|), so we take the
    # NLL win where the catastrophic-miss-avoidance story lives.
    wins_tn = {}
    for h in eng:
        e = eng[h].get("nll_raw_transient")
        s = stat.get(h, {}).get("nll_raw_transient")
        if e is not None and s is not None:
            wins_tn[h] = {
                "engine_nll_transient": e,
                "static_nll_transient": s,
                "n_transient": eng[h].get("n_transient"),
                "engine_wins": bool(e < s),
            }
    v["forecast_nll_transient_by_horizon"] = wins_tn
    h_pos_tn = [h for h in wins_tn if int(h) > 0]
    v["forecast_beats_static_transient_nll_any_Hgt0"] = bool(
        any(wins_tn[h]["engine_wins"] for h in h_pos_tn)
    )
    v["forecast_beats_static_transient_nll_all_Hgt0"] = bool(
        h_pos_tn and all(wins_tn[h]["engine_wins"] for h in h_pos_tn)
    )

    # Re-scoped criterion (2): engine beats static on the dense TRANSIENT windows
    # at H>0 on AT LEAST ONE well-motivated criterion — transient CRPS OR NLL.
    v["criterion2_transient_win_any_Hgt0"] = bool(
        v["forecast_beats_static_transient_any_Hgt0"]
        or v["forecast_beats_static_transient_nll_any_Hgt0"]
    )

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
                "engine_pi_width": ew,
                "static_pi_width": sw,
                "engine_coverage": ec,
                "static_coverage": sc,
                "engine_sharper": bool(ew < sw),
            }
    v["forecast_pi_width_by_horizon"] = pi_width

    # (3) OOD honesty quantified + coverage non-collapse
    ood = metrics.get("ood_honesty", {})
    # S7.5: criterion 3 is judged on ENGINE-NATIVE OOD signals (the engine's own
    # innovation + predictive-σ AUROC), NOT the static-ensemble disagreement.
    # We report BOTH engine signals; the headline engine-native AUROC is the
    # innovation (the most-principled "inputs surprise the dynamics" signal), with
    # the predictive-σ AUROC reported alongside.  The static-ensemble disagreement
    # is kept ONLY as a clearly-labelled SAME-DATA reference (v1 mislabelled this
    # 0.81 as the engine's — it is the STATIC model's input-novelty score).
    engine_innov = ood.get("ood_auroc_engine_innovation")
    engine_sigma = ood.get("ood_auroc_engine_predictive_sigma")
    engine_auroc = ood.get("ood_auroc_engine")  # headline = innovation
    static_disagreement = ood.get(
        "ood_auroc_static_ensemble_disagreement",
        ood.get("ood_auroc_ensemble_disagreement"),
    )
    v["ood_auroc"] = engine_auroc if engine_auroc is not None else static_disagreement
    v["ood_auroc_engine_native"] = engine_auroc
    v["ood_auroc_engine_predictive_sigma"] = engine_sigma
    v["ood_auroc_engine_innovation"] = engine_innov
    v["ood_auroc_static_ensemble_disagreement"] = static_disagreement
    ci = ood.get("filter_coverage90_raw_indist")
    co = ood.get("filter_coverage90_raw_ood")
    v["ood_coverage_noncollapse"] = bool(co is not None and co > 0.5)
    v["ood_coverage_indist"] = ci
    v["ood_coverage_ood"] = co

    # ---- RE-SCOPED STAGE-1 ACCEPTANCE (f-s7-stage1-decision, A+B) -----------
    # (1) filtering calibrated 88-92%; (2) engine beats static on transient
    # CRPS OR NLL at H>0 on identical dense windows; (3) OOD honesty.
    #
    # S7.5 criterion-3 honesty (the v1 number was the mislabelled static score).
    # The plan wording is "a quantified OOD signal CLEARLY EXCEEDING the static",
    # so criterion 3 is decomposed into two EXPLICIT sub-parts, both required:
    #   (3a) coverage non-collapse (OOD filter coverage stays > 0.5);
    #   (3b) the DYNAMICS-NATIVE engine OOD-AUROC (the filter-innovation — a signal
    #        the static CANNOT produce) is (i) usable (> 0.65 absolute) AND
    #        (ii) CLEARLY exceeds the SAME-DATA static-ensemble disagreement (a
    #        REAL margin ≥ 0.05 — not a within-noise tie).  The apples-to-apples
    #        reference is the SAME-DATA static disagreement, NOT the 0.568
    #        decimated number (a cross-dataset mismatch would relaunder v1's bug).
    # The predictive-σ AUROC is REPORTED but NOT used for (3b): predictive-σ
    # inflation and the static ensemble-disagreement measure the SAME construct
    # (predictive-uncertainty growth), so "engine pred-σ ≈ static disagreement" is
    # the engine MATCHING the static at the same detector, not its DYNAMICS adding
    # an OOD signal the static lacks — privileging pred-σ would launder a tie into
    # a win.  If (3a) holds but (3b) does not, criterion 3 is an honest PARTIAL —
    # recorded as such, NOT forced to pass.  Reference choice is the ORCHESTRATOR's;
    # this verdict surfaces all four numbers and does not retune the gate to taste.
    _MARGIN = 0.05  # "clearly exceeds" demands a real margin, not a tie
    crit1 = bool(v["filtering_coverage_in_band"])
    crit2 = bool(v["criterion2_transient_win_any_Hgt0"] and v["same_windows_verified"])
    best_engine_auroc = max(
        [a for a in (engine_innov, engine_sigma) if a is not None], default=None
    )
    crit3a_noncollapse = bool(v["ood_coverage_noncollapse"])
    # (3b) uses the DYNAMICS-NATIVE innovation signal (NOT pred-σ; see note above).
    crit3b_auroc = bool(
        engine_innov is not None
        and engine_innov > 0.65
        and static_disagreement is not None
        and engine_innov >= static_disagreement + _MARGIN
    )
    crit3 = bool(crit3a_noncollapse and crit3b_auroc)
    v["ood_best_engine_auroc"] = best_engine_auroc
    # criterion 3 is a PARTIAL when coverage non-collapse holds but the
    # engine-native AUROC does not clearly match/exceed the same-data static.
    crit3_partial = bool(crit3a_noncollapse and not crit3)
    v["rescoped_acceptance"] = {
        "criterion1_filtering_calibrated": crit1,
        "criterion2_transient_dynamics_win": crit2,
        "criterion2_win_basis": (
            "transient_NLL"
            if v["forecast_beats_static_transient_nll_any_Hgt0"]
            else (
                "transient_CRPS"
                if v["forecast_beats_static_transient_any_Hgt0"]
                else "none"
            )
        ),
        "criterion3_ood_honesty": crit3,
        "criterion3_partial": crit3_partial,
        "criterion3a_coverage_noncollapse": crit3a_noncollapse,
        "criterion3b_innovation_auroc_clearly_exceeds_same_data_static": crit3b_auroc,
        "criterion3b_margin_required": _MARGIN,
        "ood_auroc_engine_innovation": engine_innov,
        "ood_auroc_engine_predictive_sigma": engine_sigma,
        "ood_best_engine_auroc": best_engine_auroc,
        "ood_auroc_static_ensemble_disagreement_same_data": static_disagreement,
        "ood_auroc_static_decimated_ref": 0.568,
        "all_met": bool(crit1 and crit2 and crit3),
        "all_met_with_partial_ood": bool(crit1 and crit2 and crit3a_noncollapse),
        "stretch_bulk_crps_beats_static": bool(v["forecast_beats_static_at_Hgt0"]),
    }
    return v


# ===========================================================================
# S8-T6 GROUNDING-VALUE ACCEPTANCE EVALUATORS
# ===========================================================================
#
# These judge the GROUNDED latent on GROUNDING VALUE (NOT |dB/dt| detection):
#   (1) help/hurt Dα — comes from run_experiment's filtering + forecasting
#       numbers on the SAME S7 harness (compared to v2 in the driver below);
#   (2) DISCOVERY-DISCRIMINATE — the T8 identity-null vs true-fθ Dα-skill gap on
#       the grounded latent (does grounding enrich fθ beyond T8's 1.3%?);
#   (3) PHYSICAL INTERPRETABILITY — near-vacuum c_plasma collapse + the grounded
#       latent carrying force-balance structure.


def _discovery_discriminate_grounded(
    model: RKNEngine,
    train_runs: list[ShotRun],
    stats,
    horizons: tuple[int, ...],
    device: str,
    max_runs: int = 40,
    max_basis_samples: int = 40000,
) -> dict:
    """T8 identity-null vs true-fθ Dα-skill gap, re-run on the GROUNDED latent.

    The KEY grounding test (acceptance #2).  Reuses the T8 skill-validator
    machinery (discovery_sindy.build_reduced_basis + ReducedTransition +
    _score_with_transition).  The identity-null transition (Δz≡0) is
    BASIS-INDEPENDENT (zero coefficients → Δz=0 regardless of V_r), so we do NOT
    need to regenerate the discovery trajectory cache — we build a fresh basis
    from a sample of THIS model's filtered latents purely to instantiate the
    zero-coefficient ReducedTransition, then score:

        true-fθ    : the grounded ``model.trans_mean`` (transition unchanged)
        identity   : a ReducedTransition with ZERO coefficients (Δz≡0 rollout)

    on the SAME train runs / horizons / max_runs as T8.  A WIDENED gap (beyond
    T8's 1.3%) ⇒ grounding gave the latent real dynamics (fθ now rich enough
    that the discovery metric DISCRIMINATES).  ~1.3% still ⇒ grounding did not
    enrich fθ.
    """
    from imas_ambix.statespace.discovery_sindy import (  # noqa: PLC0415
        ReducedTransition,
        _mean_crps,
        _score_with_transition,
        build_library,
        build_reduced_basis,
    )

    # Collect a sample of filtered latents to build the reduced basis.
    z_samples = []
    for r in train_runs[:max_runs]:
        x_norm = stats.normalise_X(r.X.astype(np.float64))
        z_post, _var = _filter_latent(model, x_norm, device)
        if z_post.shape[0]:
            z_samples.append(z_post)
        if sum(z.shape[0] for z in z_samples) > max_basis_samples:
            break
    if not z_samples:
        return {"error": "no latent samples for basis"}
    z_all = np.concatenate(z_samples, axis=0)[:max_basis_samples]
    r_dim = min(3, z_all.shape[1])
    basis = build_reduced_basis(z_all, r=r_dim)
    # zero-coefficient library → Δz≡0 identity transition (basis-independent)
    _theta, powers = build_library(basis.project(z_all[:2]), degree=2)
    null_coeffs = np.zeros((len(powers), r_dim), dtype=np.float64)
    identity_mod = ReducedTransition(basis, null_coeffs, powers)

    true_scores = _score_with_transition(
        model,
        train_runs[:max_runs],
        stats,
        horizons,
        None,
        "true_ftheta_grounded",
        device=device,
    )
    identity_scores = _score_with_transition(
        model,
        train_runs[:max_runs],
        stats,
        horizons,
        identity_mod,
        "identity_null_grounded",
        device=device,
    )
    crps_true = _mean_crps(true_scores)
    crps_identity = _mean_crps(identity_scores)
    rel_gap = (
        (crps_identity - crps_true) / crps_true
        if np.isfinite(crps_true) and crps_true > 1e-12
        else float("nan")
    )
    discriminates = bool(np.isfinite(rel_gap) and rel_gap > 0.02)
    return {
        "mean_crps_true": crps_true,
        "mean_crps_identity": crps_identity,
        "identity_minus_true_rel": rel_gap,
        "metric_discriminates": discriminates,
        "t8_baseline_gap": 0.012776796636206629,
        "widened_beyond_t8": bool(
            np.isfinite(rel_gap) and rel_gap > 0.012776796636206629 + 1e-9
        ),
        "n_runs_scored": min(max_runs, len(train_runs)),
        "interpretation": (
            f"Grounded-latent discovery gap = {rel_gap:.2%} "
            f"(T8 ungrounded baseline = 1.28%). "
            + (
                "WIDENED — grounding enriched fθ; the discovery metric now "
                "discriminates."
                if discriminates
                else "NOT materially widened — grounding did not enrich fθ "
                "(the discovery metric remains under-powered)."
            )
        ),
    }


def _filter_latent(model: RKNEngine, x_norm: np.ndarray, device: str):
    """Return (z_post (T,L), var_post (T,L)) filtered latent for one run."""
    with torch.no_grad():
        xb = torch.from_numpy(np.ascontiguousarray(x_norm)).float().unsqueeze(0)
        z_post, var_post, _mu, _v = model.filter_sequence(xb.to(device))
    return z_post[0].cpu().numpy(), var_post[0].cpu().numpy()


def _grounded_physical_interpretability(
    model: RKNEngine,
    grounding_ctx,
    train_runs: list[ShotRun],
    stats,
    device: str,
    max_shots: int = 60,
) -> dict:
    """Near-vacuum c_plasma collapse + force-balance structure on the GROUNDED head.

    Acceptance #3.  Unlike the standalone monitor (which solves c_plasma per slice
    by lstsq), here the currents are PREDICTED from z by the grounding head — so
    near-vacuum soundness does NOT transfer for free; it must be re-checked.  For
    each shot with a near-vacuum slice (|Ip|≈0, solenoid sizeable) we read the
    head's predicted net toroidal plasma current at near-vacuum vs flat-top and
    report the ratio (PHYSICAL, references only |Ip|, never labels) — the same
    metric the standalone near_vacuum_sanity uses.
    """
    by_sig = grounding_ctx.by_signature
    head = model.grounding_head
    head.eval()

    ratios: list[float] = []
    per_shot: list[dict] = []
    n_checked = 0
    for r in train_runs:
        if n_checked >= max_shots:
            break
        # _grounded_near_vacuum_for_run derives this shot's campaign itself (via
        # its geometry signature) and returns None if it has no operator / no
        # near-vacuum slice.
        info = _grounded_near_vacuum_for_run(model, r, by_sig, stats, device)
        if info is None:
            continue
        n_checked += 1
        per_shot.append(info)
        if info.get("net_ratio") is not None and np.isfinite(info["net_ratio"]):
            ratios.append(info["net_ratio"])

    med_ratio = float(np.median(ratios)) if ratios else float("nan")
    return {
        "n_near_vacuum_shots_checked": len(ratios),
        "median_net_nearvac_over_flattop_ratio_GROUNDED": med_ratio,
        "tol_frac": 0.25,
        "near_vacuum_ok_GROUNDED": bool(np.isfinite(med_ratio) and med_ratio <= 0.25),
        "standalone_monitor_ratio_ref": 0.030235146241937558,
        "per_shot_sample": per_shot[:10],
        "note": (
            "GROUNDED near-vacuum check: net toroidal plasma current predicted by "
            "the head from z at the near-vacuum slice vs flat-top (|sum c_plasma|). "
            "References only raw |Ip| (EFIT-free), never labels. The standalone "
            "monitor (per-slice lstsq) achieved 0.030; this re-checks the head."
        ),
    }


def _grounded_near_vacuum_for_run(model, run, by_sig, stats, device):
    """Predict the head's net plasma current at near-vacuum vs flat-top for a run.

    Returns None if the run has no campaign operator or no near-vacuum slice.
    Uses the raw amc plasma_current proxy already in ShotRun.X to find the
    near-vacuum / flat-top slices, and the head's predicted θ→c_plasma.
    """
    from imas_ambix.gs.geometry import build_table_for_shot  # noqa: PLC0415
    from imas_ambix.gs.grounding import _feature_offsets  # noqa: PLC0415
    from imas_ambix.statespace.baseline import _FEATURE_SCHEMA_MAG_ANE  # noqa: PLC0415

    try:
        sig = build_table_for_shot(int(run.shot_id)).signature.key
    except Exception:  # noqa: BLE001
        return None
    cg = by_sig.get(sig)
    if cg is None:
        return None
    # find Ip proxy: amc plasma_current column in X
    fs = _FEATURE_SCHEMA_MAG_ANE
    offsets = _feature_offsets(fs)
    amc_off = offsets["amc"]
    amc_feat = fs["amc"]
    if "plasma_current" not in amc_feat or "sol_current" not in amc_feat:
        return None
    ip_col = amc_off + amc_feat.index("plasma_current")
    sol_col = amc_off + amc_feat.index("sol_current")
    x = np.asarray(run.X, dtype=np.float64)
    ip = x[:, ip_col]
    sol = x[:, sol_col]
    nv_mask = (np.abs(ip) < 3.0) & (np.abs(sol) > 5.0)
    if not nv_mask.any():
        return None
    nv_idx = int(np.where(nv_mask)[0][np.argmax(np.abs(sol[np.where(nv_mask)[0]]))])
    ft_idx = int(np.argmax(np.abs(ip)))
    # filter the run to get the latent at those slices
    x_norm = stats.normalise_X(x)
    z_post, _v = _filter_latent(model, x_norm, device)
    if z_post.shape[0] <= max(nv_idx, ft_idx):
        return None
    head = model.grounding_head
    with torch.no_grad():
        zt = torch.from_numpy(z_post[[nv_idx, ft_idx]]).float().to(device)
        theta, _psi = head(zt)
        theta = theta.cpu().numpy()
    # c_plasma = B_poly @ theta; net toroidal current = sum(c_plasma)
    b_poly = _campaign_b_poly(cg)
    c_nv = b_poly @ theta[0]
    c_ft = b_poly @ theta[1]
    net_nv = float(abs(np.sum(c_nv)))
    net_ft = float(abs(np.sum(c_ft)))
    net_ratio = net_nv / net_ft if net_ft > 0 else float("nan")
    return {
        "shot_id": int(run.shot_id),
        "ip_near_vacuum": float(ip[nv_idx]),
        "ip_flattop": float(ip[ft_idx]),
        "net_plasma_current_near_vacuum_A": net_nv,
        "net_plasma_current_flattop_A": net_ft,
        "net_ratio": float(net_ratio) if np.isfinite(net_ratio) else None,
    }


def _campaign_b_poly(cg) -> np.ndarray:
    """Recover the plasma poly basis B for a campaign (for c_plasma = B@theta)."""
    from imas_ambix.gs.residual import plasma_poly_basis  # noqa: PLC0415

    op = cg.operator
    return plasma_poly_basis(op.plasma_rz, cg.profile_order, op.r0, op.minor_radius)


def run_grounding_experiment(
    cfg: ExperimentConfig,
    output: Path | None = None,
    v2_metrics_path: Path | None = None,
) -> dict:
    """S8-T6 grounding-value experiment: grounded retrain + re-scoped acceptance.

    Runs the GROUNDED engine through the SAME run_experiment harness (so the
    filtering + forecasting numbers are directly comparable to v2), then adds the
    two grounding-specific evaluators (discovery-discriminate gap; near-vacuum /
    physical interpretability).  The help/hurt-Dα delta vs v2 is computed from
    the v2 metrics artifact.
    """
    if not cfg.grounding:
        raise ValueError("run_grounding_experiment requires cfg.grounding=True")

    # Run the grounded engine through the shared harness.  We need the trained
    # model + grounding_ctx + train_runs for the grounding evals, so we run the
    # harness and then re-derive them — but to avoid a second train we capture
    # them by running the harness body's pieces here is heavy; instead we attach
    # the grounding evals by re-loading the model state from run_experiment.
    # run_experiment returns metrics; for the grounding evals we re-train would be
    # wasteful, so we instead call an instrumented harness that also returns the
    # model.  We do that by setting a module-level capture.
    global _LAST_MODEL, _LAST_GROUNDING_CTX, _LAST_TRAIN_RUNS, _LAST_STATS
    _LAST_MODEL = None
    _LAST_GROUNDING_CTX = None
    _LAST_TRAIN_RUNS = None
    _LAST_STATS = None
    metrics = run_experiment(cfg, output=None)

    model = _LAST_MODEL
    grounding_ctx = _LAST_GROUNDING_CTX
    train_runs = _LAST_TRAIN_RUNS
    stats = _LAST_STATS

    # --- (2) discovery-discriminate gap on the grounded latent --------------
    if model is not None and train_runs is not None:
        logger.info("[grounding] discovery-discriminate gap on grounded latent...")
        metrics["discovery_discriminate"] = _discovery_discriminate_grounded(
            model, train_runs, stats, cfg.horizons, cfg.device
        )

    # --- (3) near-vacuum / physical interpretability ------------------------
    if model is not None and grounding_ctx is not None and train_runs is not None:
        logger.info("[grounding] near-vacuum / physical interpretability...")
        metrics["physical_interpretability"] = _grounded_physical_interpretability(
            model, grounding_ctx, train_runs, stats, cfg.device
        )

    # --- (1) help/hurt Dα vs v2 (same harness numbers) ----------------------
    metrics["help_hurt_dalpha"] = _grounding_help_hurt(metrics, v2_metrics_path)

    metrics["grounding_verdict"] = _grounding_verdict(metrics)

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            json.dump(metrics, f, indent=2, default=float)
        logger.info("Grounding metrics written to %s", output)
    return metrics


def _grounding_help_hurt(metrics: dict, v2_path: Path | None) -> dict:
    """Compare grounded filtering + forecasting Dα skill vs the v2 (ungrounded)."""
    out: dict = {"v2_metrics_path": str(v2_path) if v2_path else None}
    if v2_path is None or not v2_path.exists():
        out["error"] = "v2 metrics not found — cannot compute help/hurt delta"
        return out
    with open(v2_path) as f:
        v2 = json.load(f)
    gf = metrics.get("filtering", {})
    vf = v2.get("filtering", {})
    out["filtering"] = {
        k: {
            "grounded": gf.get(k),
            "v2": vf.get(k),
            "delta": (gf.get(k) - vf.get(k))
            if (gf.get(k) is not None and vf.get(k) is not None)
            else None,
        }
        for k in ("crps_raw", "nll_raw", "rmse", "coverage_90_conf")
    }
    # forecasting CRPS/NLL per horizon (overall)
    g_eng = metrics.get("forecasting_indist_dense_transient", {}).get("engine", {})
    v_eng = v2.get("forecasting_indist_dense_transient", {}).get("engine", {})
    fc = {}
    for h in sorted(set(g_eng) | set(v_eng), key=lambda x: int(x)):
        gh = g_eng.get(h, {})
        vh = v_eng.get(h, {})
        fc[h] = {
            "crps_grounded": gh.get("crps_raw"),
            "crps_v2": vh.get("crps_raw"),
            "nll_grounded": gh.get("nll_raw"),
            "nll_v2": vh.get("nll_raw"),
        }
    out["forecasting_by_horizon"] = fc
    # honest help/hurt summary on filtering CRPS (the headline Dα skill)
    fc_crps_g = gf.get("crps_raw")
    fc_crps_v = vf.get("crps_raw")
    if fc_crps_g is not None and fc_crps_v is not None:
        out["filtering_crps_verdict"] = (
            "HELP (lower CRPS)"
            if fc_crps_g < fc_crps_v
            else "HURT (higher CRPS)"
            if fc_crps_g > fc_crps_v
            else "NEUTRAL"
        )
        out["filtering_crps_rel_change"] = (fc_crps_g - fc_crps_v) / fc_crps_v
    return out


def _grounding_verdict(metrics: dict) -> dict:
    """Honest grounding-value verdict across the three re-scoped criteria.

    CRITICAL COUPLING (criteria 1 ↔ 2): a WIDENED discovery gap is evidence of
    ENRICHED f_θ *only if Dα is preserved*.  If grounding wrecks Dα, a bigger
    identity-vs-true gap can simply mean "a WORSE model's f_θ thrashes more than
    a frozen belief" — NOT enrichment.  The verdict states this explicitly so a
    confounded win is not over-claimed: ``grounding_enriched_latent`` requires
    BOTH (gap widened + discriminates) AND (Dα preserved).  The clean-comparison
    flag also asserts the run matched v2 (student_t + drift_reg=0.3) so the
    gap-widening is attributable to grounding, not to dropping drift_reg (which
    alone unfreezes f_θ — discovery_sindy crux_1).
    """
    hh = metrics.get("help_hurt_dalpha", {})
    dd = metrics.get("discovery_discriminate", {})
    pi = metrics.get("physical_interpretability", {})
    cfgd = metrics.get("config", {})
    rel = hh.get("filtering_crps_rel_change")
    dalpha_preserved = bool(rel is not None and rel <= 0.10)
    gap_widened = bool(dd.get("widened_beyond_t8")) and bool(
        dd.get("metric_discriminates")
    )
    # clean comparison = grounded run matched v2's emission + drift_reg
    clean_vs_v2 = bool(
        cfgd.get("emission") == "student_t"
        and abs(float(cfgd.get("drift_reg_weight", -1)) - 0.3) < 1e-9
    )
    return {
        "criterion1_dalpha_help_hurt": hh.get("filtering_crps_verdict"),
        "criterion1_filtering_crps_rel_change": rel,
        # "did not silently destroy Dα" = within ~10% of v2 (or better)
        "criterion1_dalpha_preserved": dalpha_preserved,
        "criterion2_discovery_gap_grounded": dd.get("identity_minus_true_rel"),
        "criterion2_t8_baseline_gap": dd.get("t8_baseline_gap"),
        "criterion2_widened_beyond_t8": dd.get("widened_beyond_t8"),
        "criterion2_metric_discriminates": dd.get("metric_discriminates"),
        "criterion3_near_vacuum_ratio_grounded": pi.get(
            "median_net_nearvac_over_flattop_ratio_GROUNDED"
        ),
        "criterion3_near_vacuum_ok": pi.get("near_vacuum_ok_GROUNDED"),
        # --- coupled grounding-value claim (the honest headline) -------------
        "clean_comparison_vs_v2": clean_vs_v2,
        "grounding_enriched_latent": bool(gap_widened and dalpha_preserved),
        "enrichment_claim_note": (
            "ENRICHED — gap widened beyond T8 AND Dα preserved AND clean v2 "
            "comparison."
            if (gap_widened and dalpha_preserved and clean_vs_v2)
            else (
                "CONFOUNDED — gap widened but Dα NOT preserved: the bigger gap "
                "may be a worse model's f_θ thrashing, not enrichment."
                if (gap_widened and not dalpha_preserved)
                else (
                    "gap did NOT widen — grounding did not enrich f_θ."
                    if not gap_widened
                    else "gap widened but the v2 comparison is NOT clean "
                    "(emission/drift_reg differ) — attribution to grounding "
                    "(vs dropping drift_reg) is ambiguous."
                )
            )
        ),
    }


# module-level capture so run_grounding_experiment can reuse the trained model
# without a second train (set inside run_experiment when grounding is on).
_LAST_MODEL = None
_LAST_GROUNDING_CTX = None
_LAST_TRAIN_RUNS = None
_LAST_STATS = None


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
    p.add_argument(
        "--drift-reg-weight",
        type=float,
        default=0.0,
        help="quiescent persistence regulariser weight (S7.4; 0=v0)",
    )
    p.add_argument(
        "--train-horizons",
        type=str,
        default=None,
        help="comma-separated training rollout horizons (e.g. 1,2,5,10,20,40)",
    )
    p.add_argument(
        "--emission",
        choices=["gaussian", "student_t"],
        default="gaussian",
        help="emission head (S7.5: student_t = heavy-tailed predictive)",
    )
    p.add_argument(
        "--student-t-fixed-nu",
        type=float,
        default=None,
        help="fix Student-t dof to this value (else ν is learned, init=5)",
    )
    p.add_argument(
        "--num-threads",
        type=int,
        default=4,
        help="cap torch intra-op threads in training (S7.5 perf; 0=untouched)",
    )
    p.add_argument("--output", type=Path, default=None)
    p.add_argument(
        "--grounding",
        action="store_true",
        help="S8-T6: enable the GS grounding head + run the grounding-value "
        "acceptance (discovery-discriminate gap + near-vacuum/interpretability)",
    )
    p.add_argument(
        "--gs-lambda", type=float, default=1e-2, help="L_GS soft-prior weight"
    )
    p.add_argument("--gs-data-weight", type=float, default=0.1, help="weight on L_data")
    p.add_argument("--gs-profile-order", type=int, default=1)
    p.add_argument("--gs-passive-rank", type=int, default=4)
    p.add_argument(
        "--v2-metrics",
        type=Path,
        default=Path(__file__).parent / "artifacts" / "engine_metrics_v2.json",
        help="ungrounded v2 metrics for the help/hurt-Dα comparison",
    )
    a = p.parse_args(argv)

    train_h = (
        tuple(int(x) for x in a.train_horizons.split(",")) if a.train_horizons else None
    )
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
        drift_reg_weight=a.drift_reg_weight,
        train_horizons=train_h,
        emission=a.emission,
        student_t_learn_nu=(a.student_t_fixed_nu is None),
        student_t_nu=(
            a.student_t_fixed_nu if a.student_t_fixed_nu is not None else 5.0
        ),
        num_threads=(a.num_threads if a.num_threads and a.num_threads > 0 else None),
        grounding=a.grounding,
        gs_lambda=a.gs_lambda,
        gs_data_weight=a.gs_data_weight,
        gs_profile_order=a.gs_profile_order,
        gs_passive_rank=a.gs_passive_rank,
    )
    if a.grounding:
        out = a.output or (
            Path(__file__).parent / "artifacts" / "grounding_metrics_v0.json"
        )
        metrics = run_grounding_experiment(
            cfg, output=out, v2_metrics_path=a.v2_metrics
        )
        gv = metrics.get("grounding_verdict", {})
        cov = metrics.get("grounding_coverage", {})
        print("\n" + "=" * 64)
        print("S8-T6 GS GROUNDING — GROUNDING-VALUE ACCEPTANCE")
        print("=" * 64)
        print(
            f"coverage: {cov.get('n_grounded_windows')}/{cov.get('n_total_windows')} "
            f"windows grounded ({100.0 * (cov.get('grounded_timestep_fraction') or 0):.1f}% "
            f"of timesteps); campaigns={cov.get('campaign_window_counts')}"
        )
        print(
            f"(1) Dα help/hurt (filtering CRPS): {gv.get('criterion1_dalpha_help_hurt')} "
            f"(rel change {gv.get('criterion1_filtering_crps_rel_change')}); "
            f"preserved={gv.get('criterion1_dalpha_preserved')}"
        )
        print(
            f"(2) DISCOVERY-DISCRIMINATE gap (grounded) = "
            f"{gv.get('criterion2_discovery_gap_grounded')}  "
            f"(T8 baseline = {gv.get('criterion2_t8_baseline_gap')}; "
            f"widened={gv.get('criterion2_widened_beyond_t8')}, "
            f"discriminates={gv.get('criterion2_metric_discriminates')})"
        )
        print(
            f"(3) near-vacuum c_plasma ratio (grounded) = "
            f"{gv.get('criterion3_near_vacuum_ratio_grounded')}  "
            f"ok={gv.get('criterion3_near_vacuum_ok')}"
        )
        print(f"total: {metrics.get('total_seconds')}s")
        return
    out = a.output or (Path(__file__).parent / "artifacts" / "engine_metrics_v0.json")
    metrics = run_experiment(cfg, output=out)

    print("\n" + "=" * 64)
    print("S7.3 RKN ENGINE — STAGE-1 ACCEPTANCE")
    print("=" * 64)
    acc = metrics["acceptance"]
    print(
        f"(1) filtering coverage@90 = {acc.get('filtering_coverage_value')}  in-band={acc['filtering_coverage_in_band']}"
    )
    print("(2) TRANSIENT-subset forecast (the re-scoped criterion, engine vs static):")
    print("    per-horizon transient CRPS (lower=better):")
    for h, w in acc.get("forecast_crps_transient_by_horizon", {}).items():
        flag = "WIN" if w["engine_wins"] else "loss"
        print(
            f"      h={h:>3}: engine={w['engine_crps_transient']:.4f}  static={w['static_crps_transient']:.4f}  [{flag}]"
        )
    print(
        "    per-horizon transient NLL (lower=better; static explodes on caught ELMs):"
    )
    for h, w in acc.get("forecast_nll_transient_by_horizon", {}).items():
        flag = "WIN" if w["engine_wins"] else "loss"
        print(
            f"      h={h:>3}: engine={w['engine_nll_transient']:.3f}  static={w['static_nll_transient']:.3f}  [{flag}]"
        )
    print(
        f"    bulk CRPS beats static at all H>0 (STRETCH) = {acc['forecast_beats_static_at_Hgt0']}"
    )
    print(
        f"(3) OOD: engine-native AUROC={acc.get('ood_auroc_engine_native')}"
        f"  (innovation={acc.get('ood_auroc_engine_innovation')}, "
        f"pred-σ={acc.get('ood_auroc_engine_predictive_sigma')})"
        f"  static-ref={acc.get('ood_auroc_static_ensemble_disagreement')}"
        f"  cov(indist)={acc.get('ood_coverage_indist')}  cov(ood)={acc.get('ood_coverage_ood')}"
        f"  noncollapse={acc['ood_coverage_noncollapse']}"
    )
    rs = acc.get("rescoped_acceptance", {})
    print("-" * 64)
    print(f"RE-SCOPED STAGE-1 ACCEPTANCE: ALL MET = {rs.get('all_met')}")
    print(
        f"  (1) filtering calibrated      = {rs.get('criterion1_filtering_calibrated')}"
    )
    print(
        f"  (2) transient dynamics win    = {rs.get('criterion2_transient_dynamics_win')}  (basis: {rs.get('criterion2_win_basis')})"
    )
    print(
        f"  (3) OOD honesty               = {rs.get('criterion3_ood_honesty')}"
        f"  (partial={rs.get('criterion3_partial')}; "
        f"3a-noncollapse={rs.get('criterion3a_coverage_noncollapse')}, "
        f"3b-innovation-clearly-exceeds-same-data-static={rs.get('criterion3b_innovation_auroc_clearly_exceeds_same_data_static')})"
    )
    print(
        f"      engine-native AUROC: innovation={rs.get('ood_auroc_engine_innovation')}, "
        f"pred-σ={rs.get('ood_auroc_engine_predictive_sigma')}  "
        f"vs same-data static={rs.get('ood_auroc_static_ensemble_disagreement_same_data')} "
        f"(decimated-ref={rs.get('ood_auroc_static_decimated_ref')})"
    )
    print(f"  ALL MET (with PARTIAL OOD)    = {rs.get('all_met_with_partial_ood')}")
    print(
        f"  STRETCH bulk-CRPS-beats-static = {rs.get('stretch_bulk_crps_beats_static')}"
    )
    print(f"total: {metrics.get('total_seconds')}s")


if __name__ == "__main__":
    main()
