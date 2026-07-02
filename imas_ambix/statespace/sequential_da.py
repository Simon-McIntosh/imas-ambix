"""S10 v1 sequential current-profile DA substrate.

This module implements the causal current-profile baseline requested by
``docs/sequential-current-da-v1.html``:

* the filtered state lives in ``psi(rho)``, not directly in ``j``;
* TORAX stays out of the inner loop by providing a once-per-shot nominal
  trajectory;
* the magnetics analysis update is localized onto the leading observable modes of
  the row-scaled ``H_psi`` operator; and
* the scored output reuses the locked ``mse_eval`` contract.

The implementation deliberately keeps the full ``psi(rho)`` state while applying
only a *reduced observable correction* on top of the nominal physics trajectory:

    psi_t = psi_nominal_t + V_obs @ c_t

where ``psi_nominal_t`` comes from a once-per-shot TORAX run and ``V_obs`` are the
leading right-singular vectors of the whitened ``H_psi`` design.  Modes below the
cutoff stay on the physics prior, matching the plan's "localize the analysis
update, leave the hidden directions on the prior" requirement.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from imas_ambix.data.paths import LEVEL1_DIR, MANIFEST_DIR
from imas_ambix.gs.residual import robust_sensor_scale
from imas_ambix.statespace.enkf_baseline import (
    EnKFConfig,
    MagneticsObs,
    MAST_A,
    MAST_B0,
    MAST_R0,
    ShotInputs,
    ToraxTrajectory,
    _campaign_representatives,
    _operator_for_shot,
    _torax_config,
    load_shot_inputs,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

MU0 = 4.0e-7 * np.pi
_TORAX_Q_CONVENTION = 1.25


@dataclass
class SequentialDAConfig(EnKFConfig):
    """Configuration for the S10 sequential ψ-state baseline."""

    localization_rank: int = 6
    correction_decay: float = 0.94
    correction_process_var: float = 2.5e-5
    correction_inflation: float = 1.02
    obs_inflation: float = 1.0
    n_samples: int = 32
    nominal_zeff: float = 2.0
    nominal_resist_mult: float = 1.0
    nominal_current_peaking: float = 2.0
    nominal_ip_frac: float = 1.0
    validation_slices: int = 3
    q_floor: float = 1.0e-9
    innovation_clip_sigma: float = 12.0
    cov_eigen_cap: float = 1.0e6
    cov_ridge: float = 1.0e-9

    def nominal_theta(self) -> dict[str, float]:
        """The once-per-shot nominal TORAX parameter vector."""

        return {
            "zeff": float(self.nominal_zeff),
            "resist_mult": float(self.nominal_resist_mult),
            "current_peaking": float(self.nominal_current_peaking),
            "ip_frac": float(self.nominal_ip_frac),
        }


@dataclass
class SequentialShotResult:
    """Two-arm result for one shot, aligned with the locked mse_eval contract."""

    shot_id: int
    slice_t: np.ndarray
    pitch_samples: np.ndarray  # (K, C, M) analysis arm
    q0_analysis: np.ndarray  # (K,)
    pitch_samples_forecast: np.ndarray  # (K, C, M) forecast arm
    q0_forecast: np.ndarray  # (K,)
    innovation_forecast: float
    innovation_analysis: float
    localization_rank: int
    singular_values: np.ndarray
    pitch_error: np.ndarray | None = None


@dataclass
class ConformalScale:
    """Frozen split-conformal scale factors for the sequential pitch predictions."""

    alpha: float
    z_alpha: float
    global_q: float
    channel_q: dict[int, float]
    band_q: dict[int, float]
    band_edges: tuple[float, float]
    min_points: int

    def scale_for(self, active_channel_rpos: np.ndarray) -> np.ndarray:
        """Per-channel σ scale = max(channel-q̂, band-q̂) / z_α."""

        rminor = np.abs(np.asarray(active_channel_rpos, dtype=np.float64) - MAST_R0)
        bands = np.digitize(rminor, self.band_edges, right=False)
        q_channel = np.array(
            [self.channel_q.get(int(i), self.global_q) for i in range(rminor.size)],
            dtype=np.float64,
        )
        q_band = np.array(
            [self.band_q.get(int(b), self.global_q) for b in bands],
            dtype=np.float64,
        )
        return np.maximum(q_channel, q_band) / max(self.z_alpha, 1.0e-12)

    def to_dict(self) -> dict[str, Any]:
        return {
            "alpha": self.alpha,
            "z_alpha": self.z_alpha,
            "global_q": self.global_q,
            "channel_q": {str(k): float(v) for k, v in self.channel_q.items()},
            "band_q": {str(k): float(v) for k, v in self.band_q.items()},
            "band_edges": [float(x) for x in self.band_edges],
            "min_points": self.min_points,
        }


@dataclass
class NominalTrajectory:
    """Once-per-shot TORAX trajectory with the ψ/Φ fields the S10 filter needs."""

    time: np.ndarray
    rho_norm: np.ndarray
    psi: np.ndarray
    phi: np.ndarray
    j_total: np.ndarray
    q: np.ndarray
    ok: bool


def _as_2d(arr: np.ndarray) -> tuple[np.ndarray, bool]:
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim == 1:
        return a[np.newaxis, :], True
    if a.ndim != 2:
        raise ValueError(f"expected 1D or 2D array, got shape {a.shape}")
    return a, False


def _cumtrapz(values: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Vectorised cumulative trapezoid along the last axis."""

    out = np.zeros_like(values, dtype=np.float64)
    if x.size < 2:
        return out
    dx = np.diff(np.asarray(x, dtype=np.float64))
    trap = 0.5 * (values[..., 1:] + values[..., :-1]) * dx
    out[..., 1:] = np.cumsum(trap, axis=-1)
    return out


def psi_from_current_profile(
    profile: np.ndarray,
    rho_grid: np.ndarray,
    *,
    r_major: float = MAST_R0,
    a_minor: float = MAST_A,
) -> np.ndarray:
    """Cylindrical poloidal-flux coordinate from ``j_phi(rho)``.

    We fix the gauge with ``psi(0) = 0``; only spatial derivatives enter the
    measurement operator, so the absolute offset is irrelevant.
    """

    j_rho, squeezed = _as_2d(profile)
    rho = np.asarray(rho_grid, dtype=np.float64)
    if rho.ndim != 1 or rho.size != j_rho.shape[1]:
        raise ValueError("rho_grid must be 1D and match the profile axis")
    r = rho * float(a_minor)
    ring_current = 2.0 * np.pi * j_rho * r[np.newaxis, :]
    i_enclosed = _cumtrapz(ring_current, r)
    grad = np.zeros_like(j_rho)
    if r.size > 1:
        grad[..., 1:] = (
            MU0
            * float(r_major)
            * i_enclosed[..., 1:]
            / np.maximum(2.0 * np.pi * r[np.newaxis, 1:], 1.0e-12)
        )
    psi = _cumtrapz(grad, r)
    return psi[0] if squeezed else psi


def current_from_psi_profile(
    profile: np.ndarray,
    rho_grid: np.ndarray,
    *,
    r_major: float = MAST_R0,
    a_minor: float = MAST_A,
) -> np.ndarray:
    """Cylindrical ``psi(rho) -> j_phi(rho)`` map used by the S10 EKF."""

    psi_rho, squeezed = _as_2d(profile)
    rho = np.asarray(rho_grid, dtype=np.float64)
    if rho.ndim != 1 or rho.size != psi_rho.shape[1]:
        raise ValueError("rho_grid must be 1D and match the profile axis")
    if rho.size < 3:
        raise ValueError("need at least 3 rho nodes for the ψ→j stencil")
    r = rho * float(a_minor)
    dpsi_dr = np.gradient(psi_rho, r, axis=-1, edge_order=2)
    flux = r[np.newaxis, :] * dpsi_dr
    dflux_dr = np.gradient(flux, r, axis=-1, edge_order=2)
    j = dflux_dr / (MU0 * float(r_major) * np.maximum(r[np.newaxis, :], 1.0e-12))
    dr0 = float(r[1] - r[0])
    j[..., 0] = 4.0 * (psi_rho[..., 1] - psi_rho[..., 0]) / (
        MU0 * float(r_major) * max(dr0 * dr0, 1.0e-12)
    )
    return j[0] if squeezed else j


def q_from_current_profile(
    profile: np.ndarray,
    rho_grid: np.ndarray,
    *,
    r_major: float = MAST_R0,
    a_minor: float = MAST_A,
    bt0: float = MAST_B0,
    floor: float = 1.0e-9,
) -> np.ndarray:
    """Circular-geometry safety factor from a toroidal current-density profile."""

    j_rho, squeezed = _as_2d(profile)
    rho = np.asarray(rho_grid, dtype=np.float64)
    if rho.ndim != 1 or rho.size != j_rho.shape[1]:
        raise ValueError("rho_grid must be 1D and match the profile axis")
    r = rho * float(a_minor)
    ring_current = 2.0 * np.pi * j_rho * r[np.newaxis, :]
    i_enclosed = _cumtrapz(ring_current, r)
    q = np.full_like(j_rho, np.nan)
    if r.size > 1:
        denom = MU0 * float(r_major) * np.maximum(i_enclosed[..., 1:], floor)
        q[..., 1:] = 2.0 * np.pi * (r[np.newaxis, 1:] ** 2) * abs(float(bt0)) / denom
    j0 = np.maximum(np.abs(j_rho[..., 0]), floor)
    q[..., 0] = 2.0 * abs(float(bt0)) / (MU0 * float(r_major) * j0)
    return q[0] if squeezed else q


def q_from_psi_phi(
    psi_profile: np.ndarray,
    phi_profile: np.ndarray,
    rho_grid: np.ndarray,
    *,
    floor: float = 1.0e-12,
) -> np.ndarray:
    """Safety factor on the TORAX face grid from ``Phi(ρ)`` and ``psi(ρ)``.

    TORAX reports ``Phi`` and ``psi`` on the ``rho_norm`` grid and ``q`` on the
    face grid.  The natural discrete analogue is the ratio of adjacent finite
    differences:

        q_{i+1/2} = (5/4) · ΔPhi / Δpsi

        The extra ``5/4`` factor is TORAX's flux convention for the exported ``psi``
        field: the raw finite-difference ratio reproduces the reported ``q`` shape
        exactly but sits at a fixed 0.8 scale across representative shots.  Applying
        the constant convention factor recovers TORAX's face-grid ``q``.
        """

    psi_rho, squeezed = _as_2d(psi_profile)
    phi_rho, squeezed_phi = _as_2d(phi_profile)
    if squeezed != squeezed_phi:
        raise ValueError("psi_profile and phi_profile must have matching rank")
    rho = np.asarray(rho_grid, dtype=np.float64)
    if rho.ndim != 1 or rho.size != psi_rho.shape[1] or rho.size != phi_rho.shape[1]:
        raise ValueError("rho_grid must match the last axis of psi_profile and phi_profile")
    dpsi = np.diff(psi_rho, axis=-1)
    dphi = np.diff(phi_rho, axis=-1)
    q = _TORAX_Q_CONVENTION * dphi / np.where(np.abs(dpsi) > floor, dpsi, np.nan)
    return q[0] if squeezed else q


def _higher_quantile(scores: Sequence[float], alpha: float) -> float:
    vals = np.asarray(list(scores), dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 1.0
    n = vals.size
    level = math.ceil((n + 1) * (1.0 - alpha)) / n
    level = min(level, 1.0)
    return float(np.quantile(vals, level, method="higher"))


def build_h_psi(
    obs: MagneticsObs,
    rho_grid: np.ndarray,
    *,
    r_major: float,
    a_minor: float,
) -> np.ndarray:
    """Explicit ``H_psi`` matrix by basis application.

    ``gs/operator.py`` is linear in plasma current but the S10 filter state lives
    in ``psi(rho)``.  ``H_psi`` therefore composes

        psi(rho) -> j_phi(rho) -> c_plasma -> trustworthy amb
    """

    rho = np.asarray(rho_grid, dtype=np.float64)
    zero_pf = np.zeros(len(obs.operator.pf_amc_channels), dtype=np.float64)
    h = np.zeros((obs.trust_rows.size, rho.size), dtype=np.float64)
    basis = np.zeros(rho.size, dtype=np.float64)
    for idx in range(rho.size):
        basis.fill(0.0)
        basis[idx] = 1.0
        j_basis = current_from_psi_profile(
            basis, rho, r_major=float(r_major), a_minor=float(a_minor)
        )
        h[:, idx] = obs.predict_amb(j_basis, rho, zero_pf)
    return h


def leading_observable_modes(
    h_psi: np.ndarray,
    sensor_scale: np.ndarray,
    rank: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Leading right-singular vectors of the row-scaled ``H_psi`` design."""

    scale = np.maximum(np.asarray(sensor_scale, dtype=np.float64), 1.0e-12)
    hw = np.asarray(h_psi, dtype=np.float64) / scale[:, np.newaxis]
    _u, s, vt = np.linalg.svd(hw, full_matrices=False)
    r = max(1, min(int(rank), vt.shape[0]))
    return vt[:r].T, s[:r]


def kalman_update(
    mean: np.ndarray,
    cov: np.ndarray,
    h_mat: np.ndarray,
    observation_residual: np.ndarray,
    sensor_std: np.ndarray,
    *,
    innovation_clip_sigma: float = 12.0,
    cov_eigen_cap: float = 1.0e6,
    cov_ridge: float = 1.0e-9,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """One linear-Gaussian update in the reduced observable subspace."""

    mean = np.asarray(mean, dtype=np.float64)
    cov = np.asarray(cov, dtype=np.float64)
    h = np.asarray(h_mat, dtype=np.float64)
    resid = np.asarray(observation_residual, dtype=np.float64)
    std = np.maximum(np.asarray(sensor_std, dtype=np.float64), 1.0e-12)
    if h.size == 0 or resid.size == 0:
        return mean, cov, float("nan"), float("nan")
    cov = _stabilize_cov(cov, eigen_cap=cov_eigen_cap, ridge=cov_ridge)
    pred = h @ mean
    innov_prior = resid - pred
    innov_prior = np.clip(
        innov_prior,
        -float(innovation_clip_sigma) * std,
        float(innovation_clip_sigma) * std,
    )
    r_cov = np.diag(std**2)
    s_mat = h @ cov @ h.T + r_cov + cov_ridge * np.eye(resid.size, dtype=np.float64)
    try:
        gain = np.linalg.solve(s_mat, h @ cov).T
    except np.linalg.LinAlgError:
        gain = np.linalg.pinv(s_mat) @ (h @ cov)
        gain = gain.T
    mean_post = mean + gain @ innov_prior
    ident = np.eye(mean.size, dtype=np.float64)
    cov_post = (ident - gain @ h) @ cov @ (ident - gain @ h).T + gain @ r_cov @ gain.T
    cov_post = _stabilize_cov(cov_post, eigen_cap=cov_eigen_cap, ridge=cov_ridge)
    innov_post = resid - h @ mean_post
    prior_norm = float(np.linalg.norm(innov_prior / std) / np.sqrt(resid.size))
    post_norm = float(np.linalg.norm(innov_post / std) / np.sqrt(resid.size))
    return mean_post, cov_post, prior_norm, post_norm


def _stabilize_cov(
    cov: np.ndarray,
    *,
    eigen_cap: float,
    ridge: float,
) -> np.ndarray:
    """Project a small covariance back onto a finite PSD cone."""

    cov_arr = np.asarray(cov, dtype=np.float64)
    cov_arr = 0.5 * (cov_arr + cov_arr.T)
    cov_arr = np.nan_to_num(cov_arr, nan=0.0, posinf=eigen_cap, neginf=-eigen_cap)
    try:
        vals, vecs = np.linalg.eigh(cov_arr)
    except np.linalg.LinAlgError:
        diag = np.clip(np.diag(cov_arr), 0.0, eigen_cap)
        return np.diag(diag + ridge)
    vals = np.clip(vals, 0.0, eigen_cap)
    return vecs @ np.diag(vals + ridge) @ vecs.T


def _sample_gaussian(
    mean: np.ndarray,
    cov: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Stable multivariate normal sampler with eigenvalue clipping."""

    mean = np.asarray(mean, dtype=np.float64)
    cov = _stabilize_cov(np.asarray(cov, dtype=np.float64), eigen_cap=1.0e6, ridge=1.0e-9)
    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, 0.0, 1.0e6)
    root = vecs @ np.diag(np.sqrt(vals))
    z = rng.standard_normal((int(n_samples), mean.size))
    return mean[np.newaxis, :] + z @ root.T


def _q_at_slices(traj: ToraxTrajectory, slice_t: np.ndarray) -> np.ndarray:
    """Interpolate TORAX ``q(rho,t)`` onto the eval slice times."""

    if traj.q.shape[0] == 1 or not np.isfinite(traj.q).any():
        return np.repeat(traj.q[:1], slice_t.size, axis=0)
    out = np.empty((slice_t.size, traj.q.shape[1]), dtype=np.float64)
    for g in range(traj.q.shape[1]):
        out[:, g] = np.interp(slice_t, traj.time, traj.q[:, g])
    return out


def _interp_profile(values: np.ndarray, time: np.ndarray, slice_t: np.ndarray) -> np.ndarray:
    """Interpolate a ``(T, G)`` profile bank onto the eval slice times."""

    vals = np.asarray(values, dtype=np.float64)
    t = np.asarray(time, dtype=np.float64)
    if vals.shape[0] == 1:
        return np.repeat(vals, slice_t.size, axis=0)
    out = np.empty((slice_t.size, vals.shape[1]), dtype=np.float64)
    for g in range(vals.shape[1]):
        out[:, g] = np.interp(slice_t, t, vals[:, g])
    return out


def run_nominal_trajectory(inp: ShotInputs, cfg: SequentialDAConfig) -> NominalTrajectory:
    """Run the once-per-shot TORAX nominal trajectory, exposing ψ and Φ."""

    import torax  # noqa: PLC0415

    config = _torax_config(inp, cfg, cfg.nominal_theta())
    try:
        tcfg = torax.ToraxConfig.from_dict(config)
        dt_out, _hist = torax.run_simulation(tcfg, progress_bar=False)
    except Exception as e:  # noqa: BLE001
        logger.debug("TORAX nominal failed for shot %d: %s", inp.shot_id, e)
        return NominalTrajectory(
            time=np.array([inp.t_final]),
            rho_norm=np.linspace(0.0, 1.0, 27),
            psi=np.zeros((1, 27)),
            phi=np.zeros((1, 27)),
            j_total=np.zeros((1, 27)),
            q=np.full((1, 26), np.nan),
            ok=False,
        )
    prof = dt_out["profiles"]
    jt = prof["j_total"]
    rho_norm = np.asarray(jt.coords["rho_norm"], dtype=np.float64)
    time = np.asarray(jt.coords["time"], dtype=np.float64)
    return NominalTrajectory(
        time=time,
        rho_norm=rho_norm,
        psi=np.asarray(prof["psi"], dtype=np.float64),
        phi=np.asarray(prof["Phi"], dtype=np.float64),
        j_total=np.asarray(jt, dtype=np.float64),
        q=np.asarray(prof["q"], dtype=np.float64)
        if "q" in prof.data_vars
        else np.full((time.size, rho_norm.size - 1), np.nan),
        ok=True,
    )


def run_shot(
    inp: ShotInputs,
    obs: MagneticsObs,
    cfg: SequentialDAConfig,
) -> SequentialShotResult:
    """Run the S10 sequential ψ-state baseline on one shot."""

    from imas_ambix.statespace.mse_eval import pitch_from_current_profile

    nominal = run_nominal_trajectory(inp, cfg)
    if not nominal.ok:
        c = inp.n_active
        empty = np.zeros((inp.slice_t.size, c, cfg.n_samples), dtype=np.float64)
        return SequentialShotResult(
            shot_id=inp.shot_id,
            slice_t=inp.slice_t,
            pitch_samples=empty.copy(),
            q0_analysis=np.full(inp.slice_t.size, np.nan),
            pitch_samples_forecast=empty,
            q0_forecast=np.full(inp.slice_t.size, np.nan),
            innovation_forecast=float("nan"),
            innovation_analysis=float("nan"),
            localization_rank=0,
            singular_values=np.zeros(0),
            pitch_error=inp.pitch_error,
        )

    rho = np.asarray(nominal.rho_norm, dtype=np.float64)
    rho_minor = rho * float(cfg.a_minor)
    j_nom = _interp_profile(nominal.j_total, nominal.time, inp.slice_t)  # (K, G)
    psi_nom = _interp_profile(nominal.psi, nominal.time, inp.slice_t)
    phi_nom = _interp_profile(nominal.phi, nominal.time, inp.slice_t)
    sensor_scale = robust_sensor_scale(inp.amb_trust)
    h_psi = build_h_psi(obs, rho, r_major=inp.r0, a_minor=cfg.a_minor)
    modes, singular_values = leading_observable_modes(
        h_psi, sensor_scale, cfg.localization_rank
    )
    h_red = h_psi @ modes
    j_modes = current_from_psi_profile(
        modes.T, rho, r_major=inp.r0, a_minor=cfg.a_minor
    )  # (R, G)

    rank = modes.shape[1]
    identity = np.eye(rank, dtype=np.float64)
    corr = np.zeros(rank, dtype=np.float64)
    cov = cfg.correction_process_var * identity
    rng = np.random.default_rng(cfg.seed + int(inp.shot_id))

    k_slices = inp.slice_t.size
    n_ch = inp.n_active
    ps_forecast = np.full((k_slices, n_ch, cfg.n_samples), np.nan, dtype=np.float64)
    ps_analysis = np.full_like(ps_forecast, np.nan)
    q0_forecast = np.full(k_slices, np.nan, dtype=np.float64)
    q0_analysis = np.full(k_slices, np.nan, dtype=np.float64)
    innov_f: list[float] = []
    innov_a: list[float] = []

    for k in range(k_slices):
        corr_prior = cfg.correction_decay * corr
        cov_prior = cfg.correction_inflation * (
            (cfg.correction_decay**2) * cov + cfg.correction_process_var * identity
        )
        cov_prior = _stabilize_cov(
            cov_prior, eigen_cap=cfg.cov_eigen_cap, ridge=cfg.cov_ridge
        )
        y_obs = inp.amb_trust[k]
        y_nom = obs.predict_amb(j_nom[k], rho, inp.i_pf[k])
        valid = np.isfinite(y_obs) & np.isfinite(y_nom) & np.isfinite(sensor_scale)
        if valid.any():
            corr_post, cov_post, norm_f, norm_a = kalman_update(
                corr_prior,
                cov_prior,
                h_red[valid],
                y_obs[valid] - y_nom[valid],
                cfg.obs_inflation * sensor_scale[valid],
                innovation_clip_sigma=cfg.innovation_clip_sigma,
                cov_eigen_cap=cfg.cov_eigen_cap,
                cov_ridge=cfg.cov_ridge,
            )
        else:
            corr_post, cov_post = corr_prior, cov_prior
            norm_f = norm_a = float("nan")
        corr, cov = corr_post, cov_post
        innov_f.append(norm_f)
        innov_a.append(norm_a)

        samp_f = _sample_gaussian(corr_prior, cov_prior, cfg.n_samples, rng)
        samp_a = _sample_gaussian(corr_post, cov_post, cfg.n_samples, rng)
        j_fc = j_nom[k][np.newaxis, :] + samp_f @ j_modes
        j_an = j_nom[k][np.newaxis, :] + samp_a @ j_modes
        ps_forecast[k] = pitch_from_current_profile(
            j_fc,
            rho_minor,
            inp.active_channel_rpos,
            inp.r0,
            inp.bt0,
            kind="j",
        ).T
        ps_analysis[k] = pitch_from_current_profile(
            j_an,
            rho_minor,
            inp.active_channel_rpos,
            inp.r0,
            inp.bt0,
            kind="j",
        ).T
        psi_fc_mean = psi_nom[k] + corr_prior @ modes.T
        psi_an_mean = psi_nom[k] + corr_post @ modes.T
        q0_forecast[k] = q_from_psi_phi(
            psi_fc_mean, phi_nom[k], rho, floor=cfg.q_floor
        )[0]
        q0_analysis[k] = q_from_psi_phi(
            psi_an_mean, phi_nom[k], rho, floor=cfg.q_floor
        )[0]

    return SequentialShotResult(
        shot_id=inp.shot_id,
        slice_t=inp.slice_t,
        pitch_samples=ps_analysis,
        q0_analysis=q0_analysis,
        pitch_samples_forecast=ps_forecast,
        q0_forecast=q0_forecast,
        innovation_forecast=float(np.nanmean(innov_f)) if innov_f else float("nan"),
        innovation_analysis=float(np.nanmean(innov_a)) if innov_a else float("nan"),
        localization_rank=rank,
        singular_values=singular_values,
        pitch_error=inp.pitch_error,
    )


def shot_result_to_prediction(result: SequentialShotResult):
    """Canonical ``mse_eval.ShotPrediction`` from the sequential DA result."""

    from imas_ambix.statespace.mse_eval import ShotPrediction

    samples = np.asarray(result.pitch_samples, dtype=np.float64)
    mean = np.nanmean(samples, axis=2)
    epi = np.nanstd(samples, axis=2)
    epi = np.where(np.isfinite(epi) & (epi > 1.0e-6), epi, 0.0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    ale = (
        np.asarray(result.pitch_error, dtype=np.float64)
        if result.pitch_error is not None
        else np.zeros_like(mean)
    )
    ale = np.where(np.isfinite(ale) & (ale > 0), ale, 0.1)
    std = np.sqrt(epi**2 + ale**2)
    std = np.where(np.isfinite(std) & (std > 1.0e-4), std, 0.1)
    rng = np.random.default_rng(0)
    samples_full = samples + ale[:, :, None] * rng.standard_normal(samples.shape)
    return ShotPrediction(
        t=result.slice_t,
        pitch_mean=mean,
        pitch_std=std,
        pitch_samples=samples_full,
    )


def fit_conformal_scales(
    predictions: dict[int, Any],
    manifest: dict,
    *,
    truth=None,
    alpha: float = 0.10,
    min_points: int = 64,
) -> ConformalScale:
    """Fit per-channel and per-radial-band split-conformal scales."""

    from scipy.stats import norm  # noqa: PLC0415

    from imas_ambix.statespace import mse_eval as eval_mod  # noqa: PLC0415
    from imas_ambix.statespace import mse_split as M  # noqa: PLC0415

    if truth is None:
        truth = eval_mod.MseTruth(level1_dir=LEVEL1_DIR)

    shots_meta = manifest["shots"]
    rminor_all: list[float] = []
    for sid, pred in predictions.items():
        entry = shots_meta.get(str(sid))
        if entry is None or entry.get("partition") != "calibration":
            continue
        try:
            pred.validate(len(entry["active_channel_ids"]))
        except ValueError:
            continue
        rminor_all.extend(
            np.abs(np.asarray(entry["active_channel_rpos"], dtype=np.float64) - eval_mod.DEFAULT_R0)
        )
    if rminor_all:
        q33, q66 = np.quantile(np.asarray(rminor_all), [1.0 / 3.0, 2.0 / 3.0])
        band_edges = (float(q33), float(q66))
    else:
        band_edges = (0.12, 0.24)

    global_scores: list[float] = []
    channel_scores: dict[int, list[float]] = {}
    band_scores: dict[int, list[float]] = {0: [], 1: [], 2: []}

    for sid, pred in predictions.items():
        entry = shots_meta.get(str(sid))
        if entry is None or entry.get("partition") != "calibration":
            continue
        tr = truth.get(int(sid))
        if tr is None:
            continue
        c = len(entry["active_channel_ids"])
        try:
            pred.validate(c)
        except ValueError:
            continue

        pt_truth = np.asarray(tr.pitch, dtype=np.float64)
        pm = np.asarray(pred.pitch_mean, dtype=np.float64)
        ps = np.asarray(pred.pitch_std, dtype=np.float64)
        gate = M.pitch_point_gate(tr.pitch, tr.pitch_error)
        valid = gate & np.isfinite(pt_truth) & np.isfinite(pm) & np.isfinite(ps) & (ps > 1.0e-12)
        if not valid.any():
            continue
        scores = np.abs(pt_truth - pm) / np.maximum(ps, 1.0e-12)
        global_scores.extend(scores[valid].tolist())
        rminor = np.abs(np.asarray(entry["active_channel_rpos"], dtype=np.float64) - eval_mod.DEFAULT_R0)
        bands = np.digitize(rminor, band_edges, right=False)
        for idx in range(c):
            chan_vals = scores[:, idx][valid[:, idx]]
            if chan_vals.size:
                channel_scores.setdefault(int(idx), []).extend(chan_vals.tolist())
                band_scores[int(bands[idx])].extend(chan_vals.tolist())

    global_q = _higher_quantile(global_scores, alpha)
    per_channel = {
        idx: _higher_quantile(vals, alpha) if len(vals) >= min_points else global_q
        for idx, vals in channel_scores.items()
    }
    per_band = {
        idx: _higher_quantile(vals, alpha) if len(vals) >= min_points else global_q
        for idx, vals in band_scores.items()
    }
    return ConformalScale(
        alpha=float(alpha),
        z_alpha=float(norm.ppf(1.0 - alpha / 2.0)),
        global_q=float(global_q),
        channel_q=per_channel,
        band_q=per_band,
        band_edges=band_edges,
        min_points=int(min_points),
    )


def apply_conformal_scales(
    predictions: dict[int, Any],
    manifest: dict,
    scales: ConformalScale,
) -> dict[int, Any]:
    """Apply frozen conformal scales to ``pitch_std`` and centered samples."""

    from imas_ambix.statespace.mse_eval import ShotPrediction  # noqa: PLC0415

    shots_meta = manifest["shots"]
    calibrated: dict[int, Any] = {}
    for sid, pred in predictions.items():
        entry = shots_meta.get(str(sid))
        if entry is None:
            calibrated[sid] = pred
            continue
        scale = scales.scale_for(np.asarray(entry["active_channel_rpos"], dtype=np.float64))
        pm = np.asarray(pred.pitch_mean, dtype=np.float64)
        ps = np.asarray(pred.pitch_std, dtype=np.float64) * scale[np.newaxis, :]
        samples = None
        if pred.pitch_samples is not None:
            raw = np.asarray(pred.pitch_samples, dtype=np.float64)
            samples = pm[:, :, None] + (raw - pm[:, :, None]) * scale[np.newaxis, :, None]
        calibrated[sid] = ShotPrediction(
            t=np.asarray(pred.t, dtype=np.float64),
            pitch_mean=pm,
            pitch_std=ps,
            pitch_samples=samples,
            q0_mean=pred.q0_mean,
            q0_std=pred.q0_std,
            rax_mean=pred.rax_mean,
            rax_std=pred.rax_std,
        )
    return calibrated


def score_calibrated_holdout(
    predictions: dict[int, Any],
    manifest: dict,
    *,
    truth=None,
    alpha: float = 0.10,
    min_points: int = 64,
) -> dict[str, Any]:
    """Fit on CALIBRATION, apply to HELD-OUT, and score the recalibrated result."""

    from imas_ambix.statespace import mse_eval as eval_mod  # noqa: PLC0415

    scales = fit_conformal_scales(
        predictions, manifest, truth=truth, alpha=alpha, min_points=min_points
    )
    calibrated = apply_conformal_scales(predictions, manifest, scales)
    return {
        "conformal": scales.to_dict(),
        "metrics": eval_mod.score(calibrated, manifest, truth),
    }


def score_manifest_artifact(
    cfg: SequentialDAConfig | None = None,
    *,
    manifest_path: Path | None = None,
    out_path: Path | None = None,
    cal_limit: int | None = None,
    held_limit: int | None = None,
    alpha: float = 0.10,
    min_points: int = 64,
) -> dict[str, Any]:
    """Run the manifest-backed calibration + held-out scoring pass and persist it."""

    import time

    cfg = cfg or SequentialDAConfig()
    manifest_path = manifest_path or (MANIFEST_DIR / "mse_heldout_split_v0.json")
    manifest = json.loads(manifest_path.read_text())
    cal = [int(k) for k, v in manifest["shots"].items() if v.get("partition") == "calibration"]
    held = [int(k) for k, v in manifest["shots"].items() if v.get("partition") == "held_out"]
    if cal_limit is not None:
        cal = cal[: int(cal_limit)]
    if held_limit is not None:
        held = held[: int(held_limit)]
    shot_ids = cal + held
    t0 = time.time()
    preds = predict_manifest_shots(manifest, shot_ids, cfg)
    elapsed = time.time() - t0
    scored = score_calibrated_holdout(
        preds, manifest, alpha=alpha, min_points=min_points
    )
    q_validation_path = Path(__file__).parent / "artifacts" / "sequential_da_q_validation_v1.json"
    payload: dict[str, Any] = {
        "schema": "sequential-da-metrics-v1",
        "method": "psi-state sequential baseline with manifest-calibrated split conformal",
        "config": {
            **cfg.to_dict(),
            "localization_rank": cfg.localization_rank,
            "correction_decay": cfg.correction_decay,
            "correction_process_var": cfg.correction_process_var,
            "correction_inflation": cfg.correction_inflation,
            "obs_inflation": cfg.obs_inflation,
            "n_samples": cfg.n_samples,
        },
        "manifest_path": str(manifest_path),
        "n_calibration_shots": len(cal),
        "n_heldout_shots": len(held),
        "n_predictions": len(preds),
        "prediction_runtime_sec": float(elapsed),
        "conformal": scored["conformal"],
        "metrics": scored["metrics"],
    }
    v0_path = Path(__file__).parent / "artifacts" / "enkf_baseline_metrics_v0.json"
    if v0_path.exists():
        payload["v0_reference"] = json.loads(v0_path.read_text())
    if q_validation_path.exists():
        payload["q_validation"] = json.loads(q_validation_path.read_text())
    out_path = out_path or (Path(__file__).parent / "artifacts" / "sequential_da_metrics_v1.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, default=float))
    return payload


def _main() -> None:
    import argparse  # noqa: PLC0415

    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser(description="Run the S10 sequential current-DA baseline")
    ap.add_argument(
        "--manifest",
        type=Path,
        default=MANIFEST_DIR / "mse_heldout_split_v0.json",
        help="locked MSE manifest",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).parent / "artifacts" / "sequential_da_metrics_v1.json",
        help="output artifact path",
    )
    ap.add_argument("--cal-limit", type=int, default=None)
    ap.add_argument("--held-limit", type=int, default=None)
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--min-points", type=int, default=64)
    a = ap.parse_args()
    cfg = SequentialDAConfig(n_samples=a.n_samples)
    payload = score_manifest_artifact(
        cfg,
        manifest_path=a.manifest,
        out_path=a.out,
        cal_limit=a.cal_limit,
        held_limit=a.held_limit,
        alpha=a.alpha,
        min_points=a.min_points,
    )
    print(
        f"SEQUENTIAL_DA_DONE held={payload['n_heldout_shots']} cal={payload['n_calibration_shots']} out={a.out}",
        flush=True,
    )


def predict_shots(
    shot_ids: Sequence[int],
    cfg: SequentialDAConfig | None = None,
    *,
    return_results: bool = False,
    manifest_grid: dict[int, dict[str, np.ndarray]] | None = None,
):
    """Produce canonical predictions for a set of shots."""

    cfg = cfg or SequentialDAConfig()
    op_cache: dict[str, Any] = {}
    reps = _campaign_representatives()
    preds: dict[int, Any] = {}
    results: dict[int, SequentialShotResult] = {}
    for sid in shot_ids:
        op = _operator_for_shot(int(sid), op_cache, reps)
        if op is None:
            continue
        a_minor = float(getattr(op, "minor_radius", cfg.a_minor) or cfg.a_minor)
        shot_cfg = replace(cfg, a_minor=a_minor)
        obs = MagneticsObs.build(op, shot_cfg)
        grid = (manifest_grid or {}).get(int(sid))
        inp = load_shot_inputs(
            int(sid),
            op,
            shot_cfg,
            slice_times_override=(grid.get("t") if grid else None),
            channel_rpos_override=(grid.get("rpos") if grid else None),
        )
        if inp is None:
            logger.warning("no usable MSE/magnetics for shot %d — skipped", sid)
            continue
        res = run_shot(inp, obs, shot_cfg)
        results[int(sid)] = res
        preds[int(sid)] = shot_result_to_prediction(res)
    if return_results:
        return preds, results
    return preds


def manifest_grid_from_manifest(
    manifest: dict,
    shot_ids: Sequence[int],
) -> dict[int, dict[str, np.ndarray]]:
    """Exact prediction grid + channel geometry from the locked manifest."""

    grid: dict[int, dict[str, np.ndarray]] = {}
    shots_meta = manifest["shots"]
    for sid in shot_ids:
        entry = shots_meta.get(str(int(sid)))
        if entry is None:
            continue
        grid[int(sid)] = {
            "t": np.asarray(entry["beam_on_slice_times"], dtype=np.float64),
            "rpos": np.asarray(entry["active_channel_rpos"], dtype=np.float64),
        }
    return grid


def predict_manifest_shots(
    manifest: dict,
    shot_ids: Sequence[int],
    cfg: SequentialDAConfig | None = None,
    *,
    return_results: bool = False,
):
    """Predict a shot set on the manifest's exact beam-on slice grid."""

    return predict_shots(
        shot_ids,
        cfg,
        return_results=return_results,
        manifest_grid=manifest_grid_from_manifest(manifest, shot_ids),
    )


def validate_q_representation(
    shot_ids: Sequence[int],
    cfg: SequentialDAConfig | None = None,
    *,
    out_path: Path | None = None,
) -> dict[str, Any]:
    """Validate the ψ-state representation against the once-per-shot TORAX q."""

    cfg = cfg or SequentialDAConfig()
    reps = _campaign_representatives()
    op_cache: dict[str, Any] = {}
    per_shot: dict[str, dict[str, Any]] = {}

    for sid in shot_ids:
        op = _operator_for_shot(int(sid), op_cache, reps)
        if op is None:
            continue
        a_minor = float(getattr(op, "minor_radius", cfg.a_minor) or cfg.a_minor)
        shot_cfg = replace(cfg, a_minor=a_minor)
        inp = load_shot_inputs(int(sid), op, shot_cfg)
        if inp is None:
            continue
        nominal = run_nominal_trajectory(inp, shot_cfg)
        if not nominal.ok:
            continue
        rho = np.asarray(nominal.rho_norm, dtype=np.float64)
        q_ref = _q_at_slices(nominal, inp.slice_t)
        psi_nom = _interp_profile(nominal.psi, nominal.time, inp.slice_t)
        phi_nom = _interp_profile(nominal.phi, nominal.time, inp.slice_t)
        q_round_cell = q_from_psi_phi(psi_nom, phi_nom, rho, floor=shot_cfg.q_floor)
        sample_idx = np.linspace(
            0, inp.slice_t.size - 1, min(inp.slice_t.size, shot_cfg.validation_slices)
        ).astype(int)
        rho_face = 0.5 * (rho[:-1] + rho[1:])
        core = rho_face <= 0.7
        q_ref_sel = q_ref[sample_idx][:, core]
        q_round_sel = q_round_cell[sample_idx][:, core]
        valid = np.isfinite(q_ref_sel) & np.isfinite(q_round_sel)
        if not valid.any():
            continue
        diff = q_round_sel[valid] - q_ref_sel[valid]
        per_shot[str(int(sid))] = {
            "n_slices": int(inp.slice_t.size),
            "sample_slice_indices": [int(x) for x in sample_idx],
            "rho_face_max": 0.7,
            "q_rmse": float(np.sqrt(np.mean(diff**2))),
            "q_max_abs": float(np.max(np.abs(diff))),
        }

    payload = {
        "schema": "sequential-da-q-validation-v1",
        "method": "psi-state roundtrip vs once-per-shot TORAX q(rho,t)",
        "shots": per_shot,
    }
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, default=float))
    return payload


__all__ = [
    "ConformalScale",
    "apply_conformal_scales",
    "SequentialDAConfig",
    "SequentialShotResult",
    "build_h_psi",
    "fit_conformal_scales",
    "current_from_psi_profile",
    "kalman_update",
    "leading_observable_modes",
    "manifest_grid_from_manifest",
    "predict_manifest_shots",
    "predict_shots",
    "psi_from_current_profile",
    "q_from_psi_phi",
    "q_from_current_profile",
    "run_shot",
    "score_manifest_artifact",
    "score_calibrated_holdout",
    "shot_result_to_prediction",
    "validate_q_representation",
]


if __name__ == "__main__":
    _main()
