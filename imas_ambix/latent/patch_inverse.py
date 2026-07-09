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

Physics priors on the null space (both OFF by default):

``sign_prior``
    unidirectional toroidal current, jφ·sign(Ip) ≥ 0.  ``"softplus"`` imposes
    it hard by reparametrisation (the optimisation variable is pre-softplus,
    so the null space cannot express sign-indefinite fills at all — the
    Rogowski sign is a measured fact, like the conductor mask);
    ``"penalty"`` keeps the free parametrisation and adds
    ``sign_weight · mean(relu(−x)²)`` so the constraint is measured, not
    assumed.  Doublets (two same-sign lobes) and current holes (j → 0) are
    unaffected; only reversed-current transients are excluded.

``support_prior``
    free-boundary support consistency, jφ = 0 outside the LCFS of the very ψ
    the current generates.  Imposed softly: the outside-cell mask is read from
    the assembled ψ on the λ-adaptation cadence (detached — no topology
    extraction in the gradient path) and the penalty charges only the current
    beyond ``halo_budget`` (a fraction of Ip exempted so real SOL/halo current
    surfaces as a measured excess over budget instead of being fitted away).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch
from scipy import ndimage

from imas_ambix.latent.structure_residual import structure_residual
from imas_ambix.latent.topology import _inside_polygon, read_topology

if TYPE_CHECKING:
    from imas_ambix.latent.patch_basis import PatchBasis

POLICIES = ("fixed", "warm-start", "discrepancy")
SIGN_PRIORS = (None, "softplus", "penalty")


def negative_current_fraction(i_cell: np.ndarray, ip_amperes: float) -> float:
    """|Σ anti-parallel current| / |Ip| — the sign-indefinite null-space fill."""
    par = np.asarray(i_cell, dtype=np.float64) * np.sign(ip_amperes)
    return float(-par[par < 0.0].sum() / abs(ip_amperes))


def outside_current_fraction(
    i_cell: np.ndarray, ip_amperes: float, outside: np.ndarray
) -> float:
    """|current on outside-LCFS cells| / |Ip| for a boolean cell mask."""
    i = np.asarray(i_cell, dtype=np.float64)
    return float(np.abs(i[np.asarray(outside, dtype=bool)]).sum() / abs(ip_amperes))


def _closed_axis_component(
    d: np.ndarray, dthr: float, ia: int, ja: int, open_mask: np.ndarray
) -> np.ndarray | None:
    """The axis-containing component of ``d <= dthr``, or None if it is open.

    ``d`` is the outward flux coordinate (0 at the axis); a component is open
    when it reaches any ``open_mask`` point (outside the limiter, or the grid
    edge) — the connectivity definition of "the surface is no longer closed".
    """
    labels, _ = ndimage.label(d <= dthr)
    comp = labels[ia, ja]
    if comp == 0:
        return None
    comp_mask = labels == comp
    if (comp_mask & open_mask).any():
        return None
    return comp_mask


def support_outside_mask(
    basis: PatchBasis,
    i_cell: np.ndarray,
    i_pf: np.ndarray,
    *,
    limiter_r: np.ndarray | None = None,
    limiter_z: np.ndarray | None = None,
) -> np.ndarray:
    """Cells outside the last CLOSED flux surface of the ψ these currents make.

    Free-boundary consistency is a fixed-point property: the support estimate
    follows ψ(j), not an external label.  The LCFS is found by CONNECTIVITY,
    not by the innermost-saddle flux read: the bounding level is pushed
    outward (bisection) for as long as the axis-containing region of the flux
    window stays closed — i.e. reaches no point outside the limiter (or the
    grid edge).  A doublet's internal saddle therefore does not bound the
    support: the closed envelope around both same-sign lobes does, and both
    lobes stay inside.  Returns all-False when no axis can be placed — a soft
    prior must not fire on an unreadable field.
    """
    r_1d = np.asarray(basis.grid_r.detach().cpu(), dtype=np.float64)
    z_1d = np.asarray(basis.grid_z.detach().cpu(), dtype=np.float64)
    n = int(basis.r_cells.shape[0])
    out = np.zeros(n, dtype=bool)
    psi2d = basis.psi_grid_2d_np(
        np.asarray(i_cell, dtype=np.float64), np.asarray(i_pf, dtype=np.float64)
    )
    read = read_topology(psi2d, r_1d, z_1d, limiter_r=limiter_r, limiter_z=limiter_z)
    if read.axis is None or not np.isfinite(read.axis_psi):
        return out
    ar, az = read.axis
    ia = int(np.argmin(np.abs(z_1d - az)))
    ja = int(np.argmin(np.abs(r_1d - ar)))

    rr, zz = np.meshgrid(r_1d, z_1d)
    open_mask = np.zeros(psi2d.shape, dtype=bool)
    open_mask[0, :] = open_mask[-1, :] = open_mask[:, 0] = open_mask[:, -1] = True
    if limiter_r is not None and limiter_z is not None:
        open_mask |= ~_inside_polygon(
            rr.ravel(),
            zz.ravel(),
            np.asarray(limiter_r, dtype=np.float64),
            np.asarray(limiter_z, dtype=np.float64),
        ).reshape(psi2d.shape)

    # outward flux coordinate: 0 at the axis, growing toward the boundary
    sign = (
        np.sign(read.boundary_psi - read.axis_psi)
        if np.isfinite(read.boundary_psi)
        else -np.sign(read.axis_psi - float(np.median(psi2d[open_mask])))
    )
    if sign == 0:
        sign = 1.0
    d = (psi2d - read.axis_psi) * sign

    lo = float(d[ia, ja])
    hi = float(d.max())
    inside_grid = _closed_axis_component(d, lo, ia, ja, open_mask)
    if inside_grid is None:
        return out  # axis point itself is open — unreadable field
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        comp = _closed_axis_component(d, mid, ia, ja, open_mask)
        if comp is None:
            hi = mid
        else:
            lo, inside_grid = mid, comp

    r_c = np.asarray(basis.r_cells.detach().cpu(), dtype=np.float64)
    z_c = np.asarray(basis.z_cells.detach().cpu(), dtype=np.float64)
    ir = np.abs(r_1d[None, :] - r_c[:, None]).argmin(axis=1)
    iz = np.abs(z_1d[None, :] - z_c[:, None]).argmin(axis=1)
    return ~inside_grid[iz, ir]


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
    sign_prior: str | None = None  # None | "softplus" (hard) | "penalty" (soft)
    sign_weight: float = 10.0
    support_prior: bool = False
    support_weight: float = 20.0
    halo_budget: float = 0.03  # fraction of |Ip| exempt outside the LCFS
    limiter_r: np.ndarray | None = None  # polygon for the in-loop boundary read
    limiter_z: np.ndarray | None = None
    dtype: torch.dtype = torch.float64


@dataclass
class SliceInversion:
    """One slice's inverse: final currents + loss-term diagnostics."""

    i_cell: np.ndarray  # (n_cells,) final patch currents [A]
    misfit: float  # whitened mean-square sensor residual
    structure: float  # structure residual at the optimum
    lambda_final: float
    ip_rel_err: float
    negative_fraction: float = float("nan")  # |anti-parallel current| / |Ip|
    outside_fraction: float = float("nan")  # |outside-LCFS current| / |Ip|
    support_excess: float = float("nan")  # max(0, outside_fraction − halo_budget)
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
    if cfg.sign_prior not in SIGN_PRIORS:
        raise ValueError(
            f"unknown sign_prior: {cfg.sign_prior!r} (use one of {SIGN_PRIORS})"
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
    hard_sign = cfg.sign_prior == "softplus"
    if hard_sign:
        # optimise pre-softplus so x = softplus(y) ≥ 0 by construction;
        # init at softplus⁻¹(seed) (clamped away from 0 for a finite inverse)
        s = seed.clamp_min(1e-6)
        raw = s + torch.log(-torch.expm1(-s))
    else:
        raw = seed
    x_raw = raw.expand(b, n).clone().requires_grad_(True)
    opt = torch.optim.Adam([x_raw], lr=cfg.lr)

    lam = torch.zeros(b, dtype=dt, device=dev)
    target = torch.full((b,), float("inf"), dtype=dt, device=dev)
    warmup_end = int(cfg.warmup_fraction * cfg.iters)
    misfit_trace = np.empty((cfg.iters, b))

    report_outside = cfg.limiter_r is not None and cfg.limiter_z is not None
    outside = torch.zeros(b, n, dtype=dt, device=dev)  # detached support mask

    misfit = torch.zeros(b, dtype=dt, device=dev)
    fb = torch.zeros(b, dtype=dt, device=dev)
    for step in range(cfg.iters):
        with torch.no_grad():
            lam = _lambda_schedule(cfg, step, lam, misfit.detach(), target)
        opt.zero_grad()
        x = torch.nn.functional.softplus(x_raw) if hard_sign else x_raw
        i_eff = x * candidate * (ip[:, None] / n)  # (B, n) [A]
        if (
            cfg.support_prior
            and step >= warmup_end
            and (step - warmup_end) % cfg.adapt_every == 0
        ):
            # boundary re-read on the λ-adaptation cadence, fully detached —
            # no topology extraction in the gradient path
            with torch.no_grad():
                i_np = i_eff.detach().cpu().numpy()
                mask_np = np.stack(
                    [
                        support_outside_mask(
                            basis,
                            i_np[k],
                            payloads[k].i_pf,
                            limiter_r=cfg.limiter_r,
                            limiter_z=cfg.limiter_z,
                        )
                        for k in range(b)
                    ]
                )
                outside = torch.as_tensor(
                    mask_np.astype(np.float64), dtype=dt, device=dev
                )
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
        loss_rows = misfit + cfg.ip_weight * ip_pen + lam * fb
        if cfg.sign_prior == "penalty":
            sign_pen = (torch.relu(-x) ** 2 * candidate).sum(-1) / candidate.sum()
            loss_rows = loss_rows + cfg.sign_weight * sign_pen
        if cfg.support_prior and step >= warmup_end:
            out_frac = (i_eff.abs() * outside).sum(-1) / ip.abs()
            sup_pen = torch.relu(out_frac - cfg.halo_budget) ** 2
            loss_rows = loss_rows + cfg.support_weight * sup_pen
        loss = loss_rows.sum()
        loss.backward()
        opt.step()
        misfit_trace[step] = misfit.detach().cpu().numpy()
        if cfg.policy == "discrepancy" and step == max(warmup_end - 1, 0):
            target = cfg.misfit_ratio * misfit.detach().clone()

    out: list[SliceInversion] = []
    with torch.no_grad():
        x_fin = torch.nn.functional.softplus(x_raw) if hard_sign else x_raw
        i_fin = (x_fin * candidate * (ip[:, None] / n)).cpu().numpy()
        for k, p in enumerate(payloads):
            out_frac = float("nan")
            excess = float("nan")
            if report_outside:
                mask_fin = support_outside_mask(
                    basis,
                    i_fin[k],
                    p.i_pf,
                    limiter_r=cfg.limiter_r,
                    limiter_z=cfg.limiter_z,
                )
                out_frac = outside_current_fraction(i_fin[k], p.ip_amperes, mask_fin)
                excess = max(0.0, out_frac - cfg.halo_budget)
            out.append(
                SliceInversion(
                    i_cell=i_fin[k],
                    misfit=float(misfit[k]),
                    structure=float(fb[k]),
                    lambda_final=float(lam[k]),
                    ip_rel_err=float(abs(i_fin[k].sum() - p.ip_amperes) / p.ip_amperes),
                    negative_fraction=negative_current_fraction(i_fin[k], p.ip_amperes),
                    outside_fraction=out_frac,
                    support_excess=excess,
                    shot=p.shot,
                    t_index=p.t_index,
                    time_s=p.time_s,
                    misfit_trace=misfit_trace[:, k],
                )
            )
    return out


__all__ = [
    "POLICIES",
    "SIGN_PRIORS",
    "InverseConfig",
    "SliceInversion",
    "SlicePayload",
    "invert_slices",
    "negative_current_fraction",
    "outside_current_fraction",
    "support_outside_mask",
]
