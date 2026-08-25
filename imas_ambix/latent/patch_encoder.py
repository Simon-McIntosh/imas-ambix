"""Amortised patch-current encoder — one forward pass from magnetics to currents.

This is the amortised counterpart of the training-free variational inverse
(:mod:`imas_ambix.latent.patch_inverse`): instead of running an Adam loop per
slice, a single :class:`PatchCurrentEncoder` forward maps a temporal window of
raw magnetics + known coil currents to a per-cell patch-current vector.  It is
trained with the SAME self-supervised objective the inverse minimises —
whitened masked sensor misfit + a Rogowski Ip anchor + the profile-free
Grad-Shafranov structure residual under the bounded-discrepancy weight policy
(:func:`amortised_losses`, :class:`DiscrepancyLambda`).  No reconstructed
equilibrium ever enters the training path; the referee scores only at eval.

Tokenisation recipe (validated against the diagnostics-to-equilibrium oracle)
-----------------------------------------------------------------------------
* one token per ``(sensor, step)`` — the channel and time axes are never pooled
  before the relational attention;
* per-sensor geometry ``(R, Z, sin θ, cos θ)`` enters as an ADDITIVE positional
  encoding (angles pre-resolved to a seam-continuous sin/cos pair by the
  caller, so a probe at θ = 2π−ε sits next to one at θ = 0+ε);
* the value embedding is CONTINUOUS (a linear lift of the standardised scalar) —
  measured 2-3x better than a quantised code on this oracle;
* non-finite entries are handled by a has-value FLAG embedding with the value
  zeroed, never by mean-imputation;
* a per-sensor-kind embedding; KNOWN coil currents enter as extra tokens with a
  coil kind and their centroid geometry;
* aggregation is a learned query-token attention pool (never a mean over
  sensors), then a per-cell current head.

Firewall discipline: this module imports nothing from the evaluator / EFIT side
and reimplements the small geometry encoding locally so it pulls in no
world-model code.  The only cross-module dependency is the physics forward
substrate (:class:`~imas_ambix.latent.patch_basis.PatchBasis`) and the profile-
free structure residual — both firewall-clean by construction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn

from imas_ambix.latent.structure_residual import structure_residual

if TYPE_CHECKING:
    from imas_ambix.latent.patch_basis import PatchBasis

#: log-σ clamp (x-space, dimensionless) for the "gaussian-direct" head — keeps
#: the predicted variance in a numerically sane range regardless of what the
#: head learns (roughly ``σ ∈ [3e-4, 20]`` of the per-cell dimensionless shape).
GAUSSIAN_LOG_SIGMA_MIN = -8.0
GAUSSIAN_LOG_SIGMA_MAX = 3.0

#: variance floor [whitened units²] the Gaussian NLL clamps to before the
#: ``log`` / division — guards the loss when a sensor's propagated variance
#: underflows (e.g. a cell with a near-zero learned σ dominating a row).
_NLL_VAR_FLOOR = 1e-12

#: Sensor-kind vocabulary the per-kind embedding switches on (index 0 = the
#: catch-all so an unmapped channel still embeds).  ``coil`` is the known-current
#: token kind.
PATCH_SENSOR_KINDS: tuple[str, ...] = ("unknown", "b_probe", "flux_loop", "coil")
_KIND_TO_IDX = {k: i for i, k in enumerate(PATCH_SENSOR_KINDS)}

#: Geometry feature width the encoder consumes per token: (R, Z, sin θ, cos θ).
#: Orientation is already resolved to a seam-continuous (sin, cos) pair; a
#: geometry-free coil token pads the angle pair with zeros.
N_GEOM_FEATURES = 4


def kind_index(kind: str) -> int:
    """Map a sensor-kind string to its embedding row (0 = unknown)."""
    return _KIND_TO_IDX.get(str(kind), 0)


def sensor_geometry_from_records(r, z, angle_deg, kind) -> np.ndarray:
    """Build the ``(S, 5)`` construction geometry from per-sensor records.

    Columns are ``[R, Z, sin θ, cos θ, kind-index]``.  A non-finite orientation
    (flux loops have none) encodes to ``(0, 0)`` — distinct from any real
    unit-circle point.  ``kind`` is a sequence of kind strings
    (see :data:`PATCH_SENSOR_KINDS`).
    """
    r = np.asarray(r, dtype=np.float64).reshape(-1)
    z = np.asarray(z, dtype=np.float64).reshape(-1)
    rad = np.deg2rad(np.asarray(angle_deg, dtype=np.float64).reshape(-1))
    finite = np.isfinite(rad)
    sin = np.where(finite, np.sin(rad), 0.0)
    cos = np.where(finite, np.cos(rad), 0.0)
    kinds = np.array([kind_index(k) for k in kind], dtype=np.float64)
    return np.column_stack([r, z, sin, cos, kinds])


@dataclass
class PatchEncoderConfig:
    """Architecture + head configuration (defaults = the oracle's config)."""

    d_model: int = 160
    n_heads: int = 4
    n_layers: int = 4
    dim_feedforward: int = 640
    dropout: float = 0.15
    n_time: int = 12  # temporal window length T (fixed per instance)
    head: str = "direct"  # "direct" | "lowrank" | "gaussian-direct"
    rank: int = 64  # low-rank latent width (head == "lowrank")
    pool: str = "query"  # "query" (learned attention pool) | "mean"
    activation: str = "gelu"


class _GeometryMLP(nn.Module):
    """Project a per-token geometry row ``(R, Z, sin θ, cos θ)`` to ``d_model``.

    A learned "has-geometry" flag is concatenated so a geometry-free token is
    distinguishable from a sensor at the coordinate origin.
    """

    def __init__(self, n_feat: int, d_model: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(n_feat + 1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, geom: torch.Tensor) -> torch.Tensor:
        finite_any = torch.isfinite(geom).any(dim=-1, keepdim=True).to(geom.dtype)
        filled = torch.where(torch.isfinite(geom), geom, torch.zeros_like(geom))
        return self.proj(torch.cat([filled, finite_any], dim=-1))


class PatchCurrentEncoder(nn.Module):
    """Amortised map: magnetics window + known coil currents -> patch currents.

    Geometry is fixed per campaign and supplied at construction (registered as
    buffers); ``T`` (window length) and ``S`` (sensor count) are fixed per
    instance.  A forward pass emits per-cell currents in AMPERES via the
    dimensionless convention ``I = x · Ip / n_cells · candidate_mask`` — the
    same convention the variational inverse optimises, so one head output serves
    any plasma current and the conductor-clear candidate mask (factual geometry)
    zeroes forbidden cells.
    """

    def __init__(
        self,
        config: PatchEncoderConfig,
        *,
        sensor_geometry: np.ndarray,  # (S, >=4) [R, Z, sin, cos, (kind-index)]
        coil_centroids: np.ndarray | None,  # (C, 2) coil centroid [R, Z] or None
        candidate_mask: np.ndarray,  # (n_cells,) conductor-clear in-limiter cells
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.config = config

        sg = np.asarray(sensor_geometry, dtype=np.float64)
        if sg.ndim != 2 or sg.shape[1] < N_GEOM_FEATURES:
            raise ValueError(
                f"sensor_geometry must be (S, >={N_GEOM_FEATURES}); got {sg.shape}"
            )
        self.n_sensor = int(sg.shape[0])
        self.n_time = int(config.n_time)
        geom = sg[:, :N_GEOM_FEATURES]
        if sg.shape[1] >= N_GEOM_FEATURES + 1:
            sensor_kind = sg[:, N_GEOM_FEATURES].astype(np.int64)
        else:
            sensor_kind = np.zeros(self.n_sensor, dtype=np.int64)

        cm = np.asarray(candidate_mask, dtype=np.float64).reshape(-1)
        self.n_cells = int(cm.shape[0])

        if coil_centroids is None:
            cc = np.zeros((0, 2), dtype=np.float64)
        else:
            cc = np.asarray(coil_centroids, dtype=np.float64).reshape(-1, 2)
        self.n_coil = int(cc.shape[0])
        coil_geom = np.zeros((self.n_coil, N_GEOM_FEATURES), dtype=np.float64)
        coil_geom[:, :2] = cc  # [R, Z, 0, 0] — no orientation for a coil
        coil_kind = np.full(self.n_coil, kind_index("coil"), dtype=np.int64)

        def buf(x: np.ndarray, is_long: bool = False) -> torch.Tensor:
            if is_long:
                return torch.tensor(np.asarray(x, dtype=np.int64), dtype=torch.long)
            return torch.tensor(np.asarray(x, dtype=np.float64), dtype=dtype)

        self.register_buffer("sensor_geom", buf(geom))
        self.register_buffer("sensor_kind", buf(sensor_kind, is_long=True))
        self.register_buffer("coil_geom", buf(coil_geom))
        self.register_buffer("coil_kind", buf(coil_kind, is_long=True))
        self.register_buffer("candidate_mask", buf(cm))

        d = config.d_model
        self.value_proj = nn.Linear(1, d)
        self.geom_encoder = _GeometryMLP(N_GEOM_FEATURES, d)
        self.kind_emb = nn.Embedding(len(PATCH_SENSOR_KINDS), d)
        self.flag_emb = nn.Embedding(2, d)  # 0 = non-finite, 1 = has-value
        self.temporal_emb = nn.Embedding(self.n_time, d)

        layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=config.n_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation=config.activation,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=config.n_layers, enable_nested_tensor=False
        )

        if config.pool not in ("query", "mean"):
            raise ValueError(f"unknown pool: {config.pool!r} (use 'query'|'mean')")
        if config.pool == "query":
            self.query_token = nn.Parameter(torch.zeros(1, 1, d))
            nn.init.normal_(self.query_token, std=0.02)
            self.pool_attn = nn.MultiheadAttention(
                d, config.n_heads, dropout=config.dropout, batch_first=True
            )
        self.final_norm = nn.LayerNorm(d)

        if config.head == "direct":
            self.head = nn.Linear(d, self.n_cells)
        elif config.head == "gaussian-direct":
            self.head = nn.Linear(d, self.n_cells)  # mean arm — identical to "direct"
            self.log_sigma_head = nn.Linear(d, self.n_cells)
            # a modest initial σ (exp(-2) ≈ 0.14 of the dimensionless shape) so
            # the NLL's log-variance term doesn't dominate the misfit term at
            # the start of training
            nn.init.constant_(self.log_sigma_head.bias, -2.0)
        elif config.head == "lowrank":
            self.z_proj = nn.Linear(d, config.rank)
            self.r_proj = nn.Linear(d, self.n_cells)
            u = torch.empty(self.n_cells, config.rank)
            nn.init.orthogonal_(u)  # orthonormal columns
            self.basis_u = nn.Parameter(u)
            self.residual_alpha = nn.Parameter(torch.tensor(0.1))
        else:
            raise ValueError(
                f"unknown head: {config.head!r} "
                "(use 'direct'|'lowrank'|'gaussian-direct')"
            )

    # ---- head arms ----

    def _decode(
        self, pooled: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Pooled representation -> dimensionless per-cell shape.

        ``"direct"``/``"lowrank"`` return the shape ``x`` ``(B, n)``.
        ``"gaussian-direct"`` returns ``(mu_x, log_sigma_x)``, each ``(B, n)``
        — the mean arm is architecturally identical to ``"direct"``; log-σ is
        clamped to :data:`GAUSSIAN_LOG_SIGMA_MIN`/:data:`GAUSSIAN_LOG_SIGMA_MAX`.
        """
        if self.config.head == "direct":
            return self.head(pooled)
        if self.config.head == "gaussian-direct":
            mu = self.head(pooled)
            log_sigma = self.log_sigma_head(pooled).clamp(
                GAUSSIAN_LOG_SIGMA_MIN, GAUSSIAN_LOG_SIGMA_MAX
            )
            return mu, log_sigma
        z = self.z_proj(pooled)  # (B, rank)
        r = self.r_proj(pooled)  # (B, n_cells)
        return z @ self.basis_u.T + self.residual_alpha * r

    # ---- forward ----

    def forward(
        self,
        values: torch.Tensor,  # (B, T, S) standardised raw magnetics window
        finite: torch.Tensor,  # (B, T, S) bool — measured this (sensor, step)
        i_pf: torch.Tensor,  # (B, C) standardised known coil currents
        ip: torch.Tensor,  # (B,) raw plasma current [A]
        *,
        return_variance: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Return per-cell currents ``(B, n_cells)`` in AMPERES (the mean, for
        every head).  ``return_variance=True`` additionally returns the
        per-cell current VARIANCE ``(B, n_cells)`` [A²] — only meaningful for
        ``head="gaussian-direct"``; other heads ignore the flag (point
        estimate only, unaffected by it) so passing it is a no-op for them.
        """
        b, t, s = values.shape
        if t != self.n_time or s != self.n_sensor:
            raise ValueError(
                f"expected (B, {self.n_time}, {self.n_sensor}); got {values.shape}"
            )
        dtype = self.sensor_geom.dtype
        dev = self.sensor_geom.device
        d = self.config.d_model
        finite_b = finite.to(torch.bool)

        v = torch.where(
            finite_b,
            values.to(dtype=dtype, device=dev),
            torch.zeros((), dtype=dtype, device=dev),
        )
        val_tok = self.value_proj(v.unsqueeze(-1))  # (B, T, S, d)
        geom_tok = self.geom_encoder(self.sensor_geom)  # (S, d)
        kind_tok = self.kind_emb(self.sensor_kind)  # (S, d)
        flag_tok = self.flag_emb(finite_b.long())  # (B, T, S, d)
        temp_tok = self.temporal_emb(torch.arange(t, device=dev))  # (T, d)
        tok = (
            val_tok
            + geom_tok[None, None]
            + kind_tok[None, None]
            + temp_tok[None, :, None, :]
            + flag_tok
        )
        sensor_tokens = tok.reshape(b, t * s, d)

        if self.n_coil > 0:
            v_pf = i_pf.to(dtype=dtype, device=dev)
            valc = self.value_proj(v_pf.unsqueeze(-1))  # (B, C, d)
            geomc = self.geom_encoder(self.coil_geom)  # (C, d)
            kindc = self.kind_emb(self.coil_kind)  # (C, d)
            flagc = self.flag_emb(
                torch.ones(self.n_coil, dtype=torch.long, device=dev)
            )  # coil currents are known → finite
            coil_tokens = valc + (geomc + kindc + flagc)[None]
            tokens = torch.cat([sensor_tokens, coil_tokens], dim=1)
        else:
            tokens = sensor_tokens

        enc = self.transformer(tokens)  # (B, N, d)
        if self.config.pool == "query":
            q = self.query_token.expand(b, -1, -1)
            pooled, _ = self.pool_attn(q, enc, enc, need_weights=False)
            pooled = pooled.squeeze(1)
        else:
            pooled = enc.mean(dim=1)
        pooled = self.final_norm(pooled)

        # (B, n_cells) dimensionless shape, or (mu, log_sigma).
        x = self._decode(pooled)
        if self.config.head == "gaussian-direct":
            mu_x, log_sigma_x = x
            i_mean = (
                mu_x
                * (ip.to(dtype=mu_x.dtype, device=mu_x.device)[:, None] / self.n_cells)
                * self.candidate_mask[None, :].to(mu_x.dtype)
            )
            if not return_variance:
                return i_mean
            # exact linear rescale of x's diagonal Gaussian: the current-space
            # std is |scale| · σ_x (never form a covariance — this IS diagonal)
            sigma_x = torch.exp(log_sigma_x)
            cell_scale = (
                ip.to(dtype=mu_x.dtype, device=mu_x.device)[:, None] / self.n_cells
            ) * self.candidate_mask[None, :].to(mu_x.dtype)
            i_std = sigma_x * cell_scale.abs()
            return i_mean, i_std * i_std

        i_cell = (
            x
            * (ip.to(dtype=x.dtype, device=x.device)[:, None] / self.n_cells)
            * self.candidate_mask[None, :].to(x.dtype)
        )
        return i_cell


# --------------------------------------------------------------------------
# self-supervised losses (batched, the amortised mirror of patch_inverse)
# --------------------------------------------------------------------------


def _coil_psi_cells(
    basis: PatchBasis,
    i_pf_amperes,
    b: int,
    n: int,
    dev: torch.device,
    dt: torch.dtype,
) -> torch.Tensor:
    """KNOWN-coil ψ at the cell centroids ``(B, n)`` [Wb] (zero if no coils)."""
    zero = torch.zeros(b, n, device=dev, dtype=dt)
    if i_pf_amperes is None or int(basis.psi_coil_cells.shape[1]) == 0:
        return zero
    ipf = torch.as_tensor(i_pf_amperes, device=dev, dtype=dt)
    if ipf.dim() == 1:
        ipf = ipf[None]
    if int(ipf.shape[-1]) == 0:
        return zero
    return ipf @ basis.psi_coil_cells.to(device=dev, dtype=dt).T  # (B, n)


def amortised_losses(
    basis: PatchBasis,
    i_cell: torch.Tensor,  # (B, n) currents [A] (from the encoder forward)
    *,
    measured: torch.Tensor,  # (B, S) raw magnetics [Wb / T]; NaN where absent
    vacuum: torch.Tensor,  # (B, S) KNOWN-coil sensor prediction [Wb / T]
    mask: torch.Tensor,  # (B, S) bool — measured AND mapped
    scale: torch.Tensor,  # (B, S) per-sensor whitening scale
    i_pf_amperes,  # (B, C) KNOWN coil currents [A] (raw) or None
    ip: torch.Tensor,  # (B,) Rogowski plasma current [A]
    lam: torch.Tensor,  # (B,) per-example force-balance weight λ
    ip_weight: float = 10.0,
    n_bins: int = 24,
    form: str = "affine-r2",
    connectivity: str | None = None,
    locality_scale: float | None = None,
    i_var: torch.Tensor | None = None,  # (B, n) cell current VARIANCE [A²]
) -> dict[str, torch.Tensor]:
    """Batched self-supervised objective — the amortised mirror of the inverse.

    ``L = misfit + w_ip · ((ΣI − Ip)/Ip)² + λ · R_structure`` with the physics
    matmuls in fp64 (the sensor + structure residual need the precision).
    Returns the three per-example terms (each ``(B,)``) plus the scalar
    ``total`` (their λ-weighted sum reduced over the batch).

    ``i_var`` is the per-cell current VARIANCE from a Gaussian head's diagonal
    covariance (mean = ``i_cell``).  When given, sensor variance propagates
    EXACTLY through the linear forward — ``pred_var = i_var @ (m_sens²)ᵀ``, a
    matvec over the diagonal, never a full covariance — and a whitened
    Gaussian NLL (``nll``) is returned and REPLACES the plain misfit as the
    data term inside ``total``; ``misfit`` is still reported (computed on the
    mean) for comparability.  ``ip_pen`` and ``fb`` are always computed on the
    mean currents, gaussian head or not.  ``i_var=None`` (the default) is
    byte-identical to the pre-Gaussian-head behaviour.
    """
    dt = torch.float64
    ic = i_cell.to(dt)
    b, n = ic.shape
    dev = ic.device

    m_sens = basis.m_sens.to(device=dev, dtype=dt)  # (S, n)
    g_cc = basis.g_cc.to(device=dev, dtype=dt)  # (n, n)
    r_c = basis.r_cells.to(device=dev, dtype=dt)  # (n,)
    z_c = basis.z_cells.to(device=dev, dtype=dt)  # (n,)
    cell_area = float(basis.cell_area)

    measured = torch.nan_to_num(torch.as_tensor(measured, device=dev, dtype=dt))
    vacuum = torch.as_tensor(vacuum, device=dev, dtype=dt)
    mask = torch.as_tensor(mask, device=dev, dtype=dt)
    scale = torch.as_tensor(scale, device=dev, dtype=dt)
    ip = torch.as_tensor(ip, device=dev, dtype=dt).reshape(-1)
    lam = torch.as_tensor(lam, device=dev, dtype=dt).reshape(-1)

    pred = vacuum + ic @ m_sens.T  # (B, S)
    misfit = (mask * ((pred - measured) / scale) ** 2).sum(-1) / mask.sum(-1).clamp_min(
        1.0
    )
    ip_pen = ((ic.sum(-1) - ip) / ip) ** 2

    psi_c = ic @ g_cc.T + _coil_psi_cells(basis, i_pf_amperes, b, n, dev, dt)
    jphi_c = ic / cell_area
    fb_rows = [
        structure_residual(
            psi_c[k],
            r_c,
            jphi_c[k],
            n_bins=n_bins,
            form=form,
            z_c=z_c,
            connectivity=connectivity,
            locality_scale=locality_scale,
        )
        for k in range(b)
    ]
    fb = torch.stack(fb_rows)

    data_term = misfit
    out: dict[str, torch.Tensor] = {"misfit": misfit, "ip_pen": ip_pen, "fb": fb}
    if i_var is not None:
        iv = torch.as_tensor(i_var, device=dev, dtype=dt)
        pred_var = iv @ (m_sens**2).T  # (B, S) — exact diagonal propagation
        pred_var_wh = (pred_var / scale**2).clamp_min(_NLL_VAR_FLOOR)
        resid_wh_sq = ((pred - measured) / scale) ** 2
        nll_terms = 0.5 * (
            torch.log(2.0 * math.pi * pred_var_wh) + resid_wh_sq / pred_var_wh
        )
        nll = (mask * nll_terms).sum(-1) / mask.sum(-1).clamp_min(1.0)
        out["nll"] = nll
        data_term = nll

    out["total"] = (data_term + ip_weight * ip_pen + lam * fb).sum()
    return out


# --------------------------------------------------------------------------
# discrepancy-principle λ schedule (the locked residual-weight policy, amortised)
# --------------------------------------------------------------------------


class DiscrepancyLambda:
    """Per-example force-balance weight λ under the bounded-discrepancy policy.

    Translates the variational inverse's ``discrepancy`` arm to amortised
    training: λ = 0 through ``warmup_epochs`` (the data term settles the
    sensor-visible modes first); at the warm-up boundary each example's target
    is frozen to ``ratio × misfit``; thereafter λ is nudged multiplicatively
    (×/÷ ``adapt_factor``) so the misfit tracks its target — up when the misfit
    is below target, down when it exceeds ``1.2 × target`` — clamped to
    ``[lam0 / lam_max, lam_max]``.

    The current epoch is set by :meth:`update`; :meth:`get` reads the frozen λ
    (zero during warm-up) for the example ids of a batch.
    """

    def __init__(
        self,
        n_examples: int,
        *,
        warmup_epochs: int = 3,
        ratio: float = 1.5,
        lam0: float = 3.0,
        lam_max: float = 100.0,
        adapt_factor: float = 1.5,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        self.n_examples = int(n_examples)
        self.warmup_epochs = int(warmup_epochs)
        self.ratio = float(ratio)
        self.lam0 = float(lam0)
        self.lam_max = float(lam_max)
        self.adapt_factor = float(adapt_factor)
        self.device = torch.device(device)
        self.dtype = dtype
        self.lam = torch.full((self.n_examples,), lam0, dtype=dtype, device=self.device)
        self.target = torch.full(
            (self.n_examples,), float("inf"), dtype=dtype, device=self.device
        )
        self._warm_misfit = torch.full(
            (self.n_examples,), float("nan"), dtype=dtype, device=self.device
        )
        self._epoch = 0

    def _ids(self, example_ids) -> torch.Tensor:
        return torch.as_tensor(
            example_ids, dtype=torch.long, device=self.device
        ).reshape(-1)

    def get(self, example_ids) -> torch.Tensor:
        """Per-example λ for ``example_ids`` — zero during warm-up."""
        ids = self._ids(example_ids)
        if self._epoch < self.warmup_epochs:
            return torch.zeros(ids.numel(), dtype=self.dtype, device=self.device)
        return self.lam[ids].clone()

    def update(self, example_ids, misfit_detached, epoch: int) -> None:
        """Record the batch's misfit and advance the schedule to ``epoch``."""
        ids = self._ids(example_ids)
        m = (
            torch.as_tensor(misfit_detached, dtype=self.dtype, device=self.device)
            .reshape(-1)
            .detach()
        )
        self._epoch = int(epoch)
        if epoch < self.warmup_epochs:
            self._warm_misfit[ids] = m
            return
        if epoch == self.warmup_epochs:
            # freeze the per-example target from the warm-up-end misfit (fall
            # back to the current misfit if warm-up never recorded these ids)
            base = torch.where(
                torch.isfinite(self._warm_misfit[ids]), self._warm_misfit[ids], m
            )
            self.target[ids] = self.ratio * base
            return
        tgt = self.target[ids]
        up = m < tgt
        down = m > 1.2 * tgt
        lam = self.lam[ids]
        lam = torch.where(up, lam * self.adapt_factor, lam)
        lam = torch.where(down, lam / self.adapt_factor, lam)
        self.lam[ids] = lam.clamp(self.lam0 / self.lam_max, self.lam_max)


__all__ = [
    "GAUSSIAN_LOG_SIGMA_MAX",
    "GAUSSIAN_LOG_SIGMA_MIN",
    "N_GEOM_FEATURES",
    "PATCH_SENSOR_KINDS",
    "DiscrepancyLambda",
    "PatchCurrentEncoder",
    "PatchEncoderConfig",
    "amortised_losses",
    "kind_index",
    "sensor_geometry_from_records",
]
