"""Patch-current layer as an ensemble-filter observation operator.

The patch forward substrate (:class:`~imas_ambix.latent.patch_basis.PatchBasis`)
is a fixed, batched linear map from in-limiter patch currents to raw sensor
magnetics: ``y = vacuum + M @ I`` where ``M = PatchBasis.m_sens`` is a constant
``(S, n)`` matrix (``S`` sensors, ``n`` cells).  That is exactly the shape an
ensemble Kalman filter wants for its observation operator ``H`` — and because it
is a *fixed* matrix (no re-linearisation, no grid solve), the analysis step is a
matmul, not an optimisation.

This module wires the patch layer into that role as a prototype for the
stage-3 closed-loop filter (``docs/closed-loop-latent-filter.html``):

* :func:`build_observation_matrix` restricts ``M`` to a caller-chosen set of
  trusted/finite sensor rows — the same "trust mask" concept used throughout
  the patch-inverse / sequential-DA code (``SlicePayload.mask``,
  ``MagneticsObs.trust_rows``);
* :func:`restrict_observation` applies the identical row mask to the raw
  measurement / vacuum / whitening-scale triple, so caller and operator never
  drift out of alignment;
* :func:`ensemble_correct` runs a perturbed-observation EnKF analysis
  (Burgers et al. 1998) on a patch-current ensemble, reusing
  :func:`imas_ambix.statespace.sequential_da.kalman_update` verbatim for the
  per-member linear-Gaussian update — this module does not reimplement the
  Kalman gain;
* the optional ``rank`` argument routes the correction through
  :func:`imas_ambix.statespace.sequential_da.leading_observable_modes` first,
  localising the update onto the leading observable directions of ``H`` before
  the ensemble covariance (an ``n x n`` matrix, ``n`` in the thousands for a
  full patch grid) is ever formed — the reduced path is the one the
  closed-loop filter should actually run.

Every correction is expressed relative to the ensemble mean (a state
deviation, not the absolute patch currents): ``resid_target = (y_obs -
vacuum) - H @ mean_prior`` is what the mean fails to explain, and each member
perturbs that same target with independent per-sensor noise before calling
``kalman_update`` — the standard stochastic-EnKF construction, so the
posterior ensemble spread reflects the analysis covariance without ever
sampling a Gaussian from scratch.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from imas_ambix.statespace.sequential_da import kalman_update, leading_observable_modes

__all__ = [
    "EnsembleAnalysis",
    "build_observation_matrix",
    "restrict_observation",
    "ensemble_correct",
]


def build_observation_matrix(
    m_sens: np.ndarray,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Restrict the patch-current -> sensor matrix to trusted/finite rows.

    Parameters
    ----------
    m_sens : ``(S, n)`` patch-current -> sensor Green's matrix
        (``PatchBasis.m_sens`` as a numpy array; ``[Wb/A]`` for flux loops,
        ``[T/A]`` for B-probes).
    mask : ``(S,)`` bool, optional
        ``True`` = sensor row is trusted and finite this slice; keep it.
        ``None`` keeps every row.

    Returns
    -------
    h : ``(n_obs, n)`` restricted observation matrix.
    keep : ``(S,)`` bool — the mask actually applied (all-``True`` if
        ``mask`` was ``None``), so the caller can apply the identical
        restriction to the measurement side via :func:`restrict_observation`.
    """
    m = np.asarray(m_sens, dtype=np.float64)
    if m.ndim != 2:
        raise ValueError(f"m_sens must be 2D (S, n), got shape {m.shape}")
    keep = np.ones(m.shape[0], dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if keep.shape != (m.shape[0],):
        raise ValueError(f"mask shape {keep.shape} != sensor rows {(m.shape[0],)}")
    return m[keep], keep


def restrict_observation(
    y_obs: np.ndarray,
    vacuum: np.ndarray,
    sensor_scale: np.ndarray,
    keep: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the row mask from :func:`build_observation_matrix` to raw obs.

    Keeps the measurement / vacuum-prediction / whitening-scale triple in
    lock-step with ``h`` so a caller can never accidentally correct against a
    row that was dropped from the operator (or vice versa).
    """
    y = np.asarray(y_obs, dtype=np.float64)
    v = np.asarray(vacuum, dtype=np.float64)
    s = np.asarray(sensor_scale, dtype=np.float64)
    keep = np.asarray(keep, dtype=bool)
    if not (y.shape == v.shape == s.shape == keep.shape):
        raise ValueError(
            "y_obs, vacuum, sensor_scale, keep must share shape; got "
            f"{y.shape}, {v.shape}, {s.shape}, {keep.shape}"
        )
    return y[keep], v[keep], s[keep]


@dataclass
class EnsembleAnalysis:
    """Result of one perturbed-observation EnKF correction step."""

    ensemble_post: np.ndarray  # (K, n) corrected patch-current ensemble [A]
    mean_prior: np.ndarray  # (n,)
    mean_post: np.ndarray  # (n,)
    cov_prior: np.ndarray  # state-space (or reduced-coefficient) prior covariance
    cov_post: np.ndarray  # posterior covariance, same space as cov_prior
    innovation_prior_norm: float  # mean whitened |innovation| pre-update
    innovation_post_norm: float  # mean whitened |innovation| post-update
    modes: np.ndarray | None = field(default=None, repr=False)  # (n, r) if rank given
    singular_values: np.ndarray = field(default_factory=lambda: np.zeros(0), repr=False)


def _stochastic_correct(
    coeff_prior: np.ndarray,  # (K, r) ensemble deviations from the mean
    cov_prior: np.ndarray,  # (r, r)
    h_red: np.ndarray,  # (n_obs, r)
    resid_target: np.ndarray,  # (n_obs,) mean-relative residual
    sensor_scale: np.ndarray,  # (n_obs,)
    rng: np.random.Generator,
    *,
    innovation_clip_sigma: float,
    cov_eigen_cap: float,
    cov_ridge: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Perturbed-observation EnKF: one ``kalman_update`` call per member.

    All members share the same prior covariance (hence the same Kalman gain
    up to floating-point determinism) and differ only in their own state
    deviation and an independent observation perturbation — the standard
    Burgers et al. (1998) construction.  ``kalman_update`` is imported from
    :mod:`imas_ambix.statespace.sequential_da`, not reimplemented here.
    """
    k = coeff_prior.shape[0]
    coeff_post = np.empty_like(coeff_prior)
    cov_post = cov_prior
    innov_prior = np.empty(k)
    innov_post = np.empty(k)
    for i in range(k):
        perturb = rng.normal(0.0, sensor_scale)
        m_post, cov_post, n_prior, n_post = kalman_update(
            coeff_prior[i],
            cov_prior,
            h_red,
            resid_target + perturb,
            sensor_scale,
            innovation_clip_sigma=innovation_clip_sigma,
            cov_eigen_cap=cov_eigen_cap,
            cov_ridge=cov_ridge,
        )
        coeff_post[i] = m_post
        innov_prior[i] = n_prior
        innov_post[i] = n_post
    return coeff_post, cov_post, innov_prior, innov_post


def ensemble_correct(
    ensemble: np.ndarray,
    h_mat: np.ndarray,
    y_obs: np.ndarray,
    vacuum: np.ndarray,
    sensor_scale: np.ndarray,
    *,
    rank: int | None = None,
    rng: np.random.Generator | None = None,
    cov_inflation: float = 1.0,
    cov_ridge: float = 1.0e-9,
    innovation_clip_sigma: float = 12.0,
    cov_eigen_cap: float = 1.0e6,
) -> EnsembleAnalysis:
    """Correct a patch-current ensemble against raw magnetics through ``H``.

    Parameters
    ----------
    ensemble : ``(K, n)`` prior patch-current ensemble ``[A]`` (``K`` >= 2
        members).
    h_mat : ``(n_obs, n)`` observation matrix, e.g. from
        :func:`build_observation_matrix`.
    y_obs, vacuum, sensor_scale : ``(n_obs,)`` raw measured magnetics, the
        KNOWN-coil (vacuum) prediction, and the per-sensor whitening scale —
        already row-restricted to match ``h_mat`` (see
        :func:`restrict_observation`).
    rank : int, optional
        If given, localise the correction onto the leading ``rank``
        observable directions of ``h_mat`` (via
        :func:`~imas_ambix.statespace.sequential_da.leading_observable_modes`)
        *before* forming the ensemble covariance — the reduced-rank path
        needed when ``n`` (patch cells) greatly exceeds ``n_obs`` (sensors),
        so the analysis never touches an ``n x n`` covariance.  ``None`` runs
        the direct full-rank correction in cell space.
    rng : numpy Generator, optional
        Source of the per-member observation perturbations (Burgers et al.
        1998 stochastic EnKF).  Defaults to an unseeded generator.
    cov_inflation : float
        Multiplicative prior-covariance inflation (matches
        ``SequentialDAConfig.correction_inflation`` in spirit).

    Returns
    -------
    :class:`EnsembleAnalysis`
    """
    ens = np.asarray(ensemble, dtype=np.float64)
    if ens.ndim != 2:
        raise ValueError(f"ensemble must be (K, n), got shape {ens.shape}")
    k, n = ens.shape
    if k < 2:
        raise ValueError(f"ensemble needs >= 2 members to estimate a covariance, got {k}")

    h = np.asarray(h_mat, dtype=np.float64)
    y = np.asarray(y_obs, dtype=np.float64)
    vac = np.asarray(vacuum, dtype=np.float64)
    scale = np.maximum(np.asarray(sensor_scale, dtype=np.float64), 1.0e-12)
    if h.ndim != 2 or h.shape[1] != n:
        raise ValueError(f"h_mat must be (n_obs, {n}), got shape {h.shape}")
    if not (y.shape == vac.shape == scale.shape == (h.shape[0],)):
        raise ValueError(
            "y_obs, vacuum, sensor_scale must all be (n_obs,) matching h_mat rows; "
            f"got {y.shape}, {vac.shape}, {scale.shape}, h_mat rows {h.shape[0]}"
        )
    rng = rng if rng is not None else np.random.default_rng()

    mean_prior = ens.mean(axis=0)
    resid_target = (y - vac) - h @ mean_prior  # what the ensemble mean fails to explain

    modes: np.ndarray | None
    singular_values: np.ndarray
    if rank is not None:
        modes, singular_values = leading_observable_modes(h, scale, rank)  # (n, r)
        h_red = h @ modes  # (n_obs, r)
        coeff_prior = (ens - mean_prior) @ modes  # (K, r)
    else:
        modes, singular_values = None, np.zeros(0)
        h_red = h
        coeff_prior = ens - mean_prior  # (K, n)

    # np.cov collapses to a 0-d scalar when there is only one coefficient (r=1);
    # atleast_2d restores the (r, r) shape kalman_update expects.
    cov_prior = cov_inflation * np.atleast_2d(np.cov(coeff_prior, rowvar=False))

    coeff_post, cov_post, innov_prior, innov_post = _stochastic_correct(
        coeff_prior,
        cov_prior,
        h_red,
        resid_target,
        scale,
        rng,
        innovation_clip_sigma=innovation_clip_sigma,
        cov_eigen_cap=cov_eigen_cap,
        cov_ridge=cov_ridge,
    )

    ensemble_post = mean_prior[np.newaxis, :] + (
        coeff_post @ modes.T if modes is not None else coeff_post
    )

    return EnsembleAnalysis(
        ensemble_post=ensemble_post,
        mean_prior=mean_prior,
        mean_post=ensemble_post.mean(axis=0),
        cov_prior=cov_prior,
        cov_post=cov_post,
        innovation_prior_norm=float(np.mean(innov_prior)),
        innovation_post_norm=float(np.mean(innov_post)),
        modes=modes,
        singular_values=singular_values,
    )
