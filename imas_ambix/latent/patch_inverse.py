"""Training-free variational patch-current inverse + force-balance weight policies.

Per slice, minimise over the patch-current vector ``I`` (in-limiter cells,
conductor interiors excluded as factual geometry):

    L(I; λ) = misfit(I) + w_ip · ((Σ I − Ip)/Ip)² + λ · R_structure(ψ(I), jφ(I))

``misfit`` is the masked whitened mean-square sensor residual (RAW measured
magnetics against the patch + KNOWN-coil prediction); the Rogowski Ip anchor is
a raw measurement (it also kills the trivial zero-current minimum); and
``R_structure`` is the profile-free Grad-Shafranov structure residual
(:func:`imas_ambix.latent.structure_residual.structure_residual`).  There is no
Picard iteration and no topology extraction anywhere in the loop — the axis /
X-points / LCFS are read from the assembled ψ at *evaluation* time, so every
slice yields a scored readout (there is no convergence to fail).

Weight-policy arms (the ``residual-weight-policy`` decision of the
patch-current force-balance plan):

``fixed``
    constant λ for all iterations.
``warm-start``
    λ = 0 for the first ``warmup_fraction`` of iterations, then the fixed λ —
    the data term settles the sensor-visible modes first; the physics term
    then owns the null space.
``discrepancy``
    λ = 0 warm-up as above, recording the achievable data-only misfit m₀ per
    slice; then λ is adapted multiplicatively on a cadence so the misfit
    tracks ``misfit_ratio × m₀`` — the discrepancy principle with a
    data-derived target (per-shot sensor scales overestimate the noise floor
    on these shots, so an absolute χ² = 1 target would over-regularise).

All slices of a batch are optimised jointly (independent rows, one Adam), so
GPU throughput is a batched matmul; λ is a per-row vector, letting the
discrepancy arm adapt each slice independently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch

from imas_ambix.latent.structure_residual import structure_residual

if TYPE_CHECKING:
    from imas_ambix.latent.patch_basis import PatchBasis

POLICIES = ("fixed", "warm-start", "discrepancy")


@dataclass
class SlicePayload:
    """Raw per-slice inputs to the inverse (all aligned to the sensor rows)."""

    measured: np.ndarray  # (S,) raw magnetics [Wb / T]; NaN where absent
    vacuum: np.ndarray  # (S,) KNOWN-coil prediction [Wb / T]
    mask: np.ndarray  # (S,) bool — measured AND mapped
    scale: np.ndarray  # (S,) whitening scale [same units]
    i_pf: np.ndarray  # (C,) KNOWN coil currents [A]
    ip_amperes: float  # Rogowski plasma current [A]
    shot: int = 0
    t_index: int = 0
    time_s: float = float("nan")


@dataclass
class InverseConfig:
    """Optimiser + loss configuration for the variational inverse."""

    iters: int = 800
    lr: float = 0.05
    lambda_fb: float = 10.0
    policy: str = "warm-start"
    warmup_fraction: float = 0.25
    misfit_ratio: float = 2.0  # discrepancy target = ratio × warm-up misfit
    adapt_every: int = 25
    adapt_factor: float = 1.5
    lambda_max: float = 1e4
    ip_weight: float = 10.0
    n_bins: int = 24
    form: str = "affine-r2"
    connectivity: str | None = None
    locality_scale: float | None = None
    seed_width_r: float = 0.35
    seed_width_z: float = 0.5
    dtype: torch.dtype = torch.float64


@dataclass
class SliceInversion:
    """One slice's inverse: final currents + loss-term diagnostics."""

    i_cell: np.ndarray  # (n_cells,) final patch currents [A]
    misfit: float  # whitened mean-square sensor residual
    structure: float  # structure residual at the optimum
    lambda_final: float
    ip_rel_err: float
    shot: int = 0
    t_index: int = 0
    time_s: float = float("nan")
    misfit_trace: np.ndarray | None = field(default=None, repr=False)


def _lambda_schedule(
    cfg: InverseConfig,
    step: int,
    lam: torch.Tensor,
    misfit: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Per-row λ update for one step (no-grad); returns the new λ vector."""
    warmup_end = int(cfg.warmup_fraction * cfg.iters)
    if cfg.policy == "fixed":
        return torch.full_like(lam, cfg.lambda_fb)
    if cfg.policy == "warm-start":
        if step >= warmup_end:
            return torch.full_like(lam, cfg.lambda_fb)
        return torch.zeros_like(lam)
    if cfg.policy == "discrepancy":
        if step < warmup_end:
            return torch.zeros_like(lam)
        if step == warmup_end:
            return torch.full_like(lam, cfg.lambda_fb)
        if (step - warmup_end) % cfg.adapt_every == 0:
            up = misfit < target
            down = misfit > 1.2 * target
            lam = torch.where(up, lam * cfg.adapt_factor, lam)
            lam = torch.where(down, lam / cfg.adapt_factor, lam)
            return lam.clamp(cfg.lambda_fb / cfg.lambda_max, cfg.lambda_max)
        return lam
    raise ValueError(f"unknown weight policy: {cfg.policy!r} (use one of {POLICIES})")


def invert_slices(
    basis: PatchBasis,
    payloads: list[SlicePayload],
    cfg: InverseConfig | None = None,
    *,
    device: str | torch.device = "cpu",
) -> list[SliceInversion]:
    """Jointly invert a batch of slices (independent rows, one Adam loop).

    The optimisation variable is the dimensionless per-cell shape
    ``x = I · n_cells / Ip`` so a single learning rate serves every slice
    regardless of its plasma current; the conductor-interior candidate mask
    multiplies the effective current everywhere (factual geometry, not a
    prior).  Returns one :class:`SliceInversion` per payload, in order.
    """
    cfg = cfg or InverseConfig()
    if cfg.policy not in POLICIES:
        raise ValueError(
            f"unknown weight policy: {cfg.policy!r} (use one of {POLICIES})"
        )
    dev = torch.device(device)
    dt = cfg.dtype
    n = int(basis.r_cells.shape[0])
    b = len(payloads)

    m_sens = basis.m_sens.to(device=dev, dtype=dt)  # (S, n)
    g_cc = basis.g_cc.to(device=dev, dtype=dt)  # (n, n)
    r_c = basis.r_cells.to(device=dev, dtype=dt)  # (n,)
    z_c = basis.z_cells.to(device=dev, dtype=dt)  # (n,)
    candidate = basis.candidate_mask.to(device=dev, dtype=dt)  # (n,)
    cell_area = float(basis.cell_area)

    meas = torch.stack(
        [torch.as_tensor(np.nan_to_num(p.measured), dtype=dt) for p in payloads]
    ).to(dev)
    vac = torch.stack([torch.as_tensor(p.vacuum, dtype=dt) for p in payloads]).to(dev)
    mask = torch.stack(
        [torch.as_tensor(p.mask.astype(np.float64), dtype=dt) for p in payloads]
    ).to(dev)
    scale = torch.stack([torch.as_tensor(p.scale, dtype=dt) for p in payloads]).to(dev)
    ip = torch.tensor([p.ip_amperes for p in payloads], dtype=dt, device=dev)  # (B,)
    psi_coil = torch.stack(
        [
            basis.psi_coil_cells_for(np.asarray(p.i_pf, dtype=np.float64))
            for p in payloads
        ]
    ).to(device=dev, dtype=dt)  # (B, n)

    # dimensionless seed: Gaussian blob normalised so I0 sums to Ip
    seed = torch.exp(
        -(((r_c - basis.r0) / cfg.seed_width_r) ** 2 + (z_c / cfg.seed_width_z) ** 2)
    )
    seed = seed / seed.sum() * n  # x-space: I = x · Ip / n
    x = seed.expand(b, n).clone().requires_grad_(True)
    opt = torch.optim.Adam([x], lr=cfg.lr)

    lam = torch.zeros(b, dtype=dt, device=dev)
    target = torch.full((b,), float("inf"), dtype=dt, device=dev)
    warmup_end = int(cfg.warmup_fraction * cfg.iters)
    misfit_trace = np.empty((cfg.iters, b))

    misfit = torch.zeros(b, dtype=dt, device=dev)
    fb = torch.zeros(b, dtype=dt, device=dev)
    for step in range(cfg.iters):
        with torch.no_grad():
            lam = _lambda_schedule(cfg, step, lam, misfit.detach(), target)
        opt.zero_grad()
        i_eff = x * candidate * (ip[:, None] / n)  # (B, n) [A]
        pred = vac + i_eff @ m_sens.T
        misfit = (mask * ((pred - meas) / scale) ** 2).sum(-1) / mask.sum(-1).clamp_min(
            1.0
        )
        ip_pen = ((i_eff.sum(-1) - ip) / ip) ** 2
        psi_c = i_eff @ g_cc.T + psi_coil  # (B, n) total flux at cells
        fb_rows = []
        for k in range(b):
            fb_rows.append(
                structure_residual(
                    psi_c[k],
                    r_c,
                    i_eff[k] / cell_area,
                    n_bins=cfg.n_bins,
                    form=cfg.form,
                    z_c=z_c,
                    connectivity=cfg.connectivity,
                    locality_scale=cfg.locality_scale,
                )
            )
        fb = torch.stack(fb_rows)
        loss = (misfit + cfg.ip_weight * ip_pen + lam * fb).sum()
        loss.backward()
        opt.step()
        misfit_trace[step] = misfit.detach().cpu().numpy()
        if cfg.policy == "discrepancy" and step == max(warmup_end - 1, 0):
            target = cfg.misfit_ratio * misfit.detach().clone()

    out: list[SliceInversion] = []
    with torch.no_grad():
        i_fin = (x * candidate * (ip[:, None] / n)).cpu().numpy()
        for k, p in enumerate(payloads):
            out.append(
                SliceInversion(
                    i_cell=i_fin[k],
                    misfit=float(misfit[k]),
                    structure=float(fb[k]),
                    lambda_final=float(lam[k]),
                    ip_rel_err=float(abs(i_fin[k].sum() - p.ip_amperes) / p.ip_amperes),
                    shot=p.shot,
                    t_index=p.t_index,
                    time_s=p.time_s,
                    misfit_trace=misfit_trace[:, k],
                )
            )
    return out


__all__ = [
    "POLICIES",
    "InverseConfig",
    "SliceInversion",
    "SlicePayload",
    "invert_slices",
]
