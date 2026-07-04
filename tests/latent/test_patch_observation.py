"""Tests for the patch-current ensemble observation operator.

Pins :mod:`imas_ambix.latent.patch_observation` on the same synthetic
``_confining_table`` fixture used by :mod:`tests.latent.test_patch_basis`: a
truth patch-current blob defines the "measured" sensor signal, a biased +
noisy ensemble stands in for the stage-3 filter's prior, and the analysis
step must move that ensemble toward the truth in SENSOR space (the only
space an EnKF can actually see), contract its spread, and agree between the
full-rank and reduced-rank (leading-observable-modes) code paths on the one
direction both can resolve exactly.
"""

from __future__ import annotations

import numpy as np
import pytest

from imas_ambix.latent.patch_basis import PatchBasis
from imas_ambix.latent.patch_observation import (
    build_observation_matrix,
    ensemble_correct,
    restrict_observation,
)
from tests.latent.test_patch_basis import _confining_table


def _blob(r_c, z_c, cr, cz, wr, wz):
    b = np.exp(-(((r_c - cr) / wr) ** 2 + ((z_c - cz) / wz) ** 2))
    return b / b.sum()


@pytest.fixture(scope="module")
def basis():
    table = _confining_table()
    return PatchBasis.from_table(table, nr=9, nz=13, cache_dir=None)


@pytest.fixture(scope="module")
def problem(basis):
    """Truth blob + a biased, noisy prior ensemble around it.

    The whitening scale is set to half the ensemble's OWN sensor-space
    spread (``sqrt(diag(H cov H^T))``) rather than an arbitrary fraction of
    the signal magnitude: this is the well-posed EnKF regime (observation
    noise comparable to, or a bit tighter than, the prior forecast spread).
    Picking the scale independent of the ensemble spread routinely either
    saturates the ``innovation_clip_sigma`` clip (scale far too small) or
    yields a negligible correction (scale far too large) — both were
    measured while tuning this fixture and are the reason the scale is
    derived from the ensemble itself.
    """
    m_sens = basis.m_sens.numpy().astype(np.float64)
    r_c = basis.r_cells.numpy()
    z_c = basis.z_cells.numpy()
    r0 = basis.r0
    n = r_c.size
    s = m_sens.shape[0]

    truth = _blob(r_c, z_c, r0, 0.0, 0.35, 0.5) * 4.0e5
    vacuum = np.zeros(s)
    y_true = vacuum + m_sens @ truth

    rng = np.random.default_rng(0)
    bias = _blob(r_c, z_c, r0 + 0.08, 0.08, 0.35, 0.5) * 4.0e5
    sigma = 0.5 * 4.0e5 / n
    k = 300
    ensemble = bias[np.newaxis, :] + sigma * rng.standard_normal((k, n))

    cov_full = np.cov(ensemble, rowvar=False)
    h_cov_h = np.einsum("ij,jk,ik->i", m_sens, cov_full, m_sens)
    scale = np.sqrt(h_cov_h) * 0.5

    return {
        "m_sens": m_sens,
        "truth": truth,
        "vacuum": vacuum,
        "y_true": y_true,
        "scale": scale,
        "ensemble": ensemble,
    }


def test_correction_moves_ensemble_toward_truth_in_sensor_space(problem):
    """The posterior mean predicts the observed sensors better than the prior."""
    h, keep = build_observation_matrix(problem["m_sens"])
    y_obs, vac, sc = restrict_observation(
        problem["y_true"], problem["vacuum"], problem["scale"], keep
    )
    result = ensemble_correct(
        problem["ensemble"], h, y_obs, vac, sc, rng=np.random.default_rng(1)
    )

    target = y_obs - vac
    err_prior = np.linalg.norm((h @ result.mean_prior - target) / sc)
    err_post = np.linalg.norm((h @ result.mean_post - target) / sc)
    assert err_post < 0.6 * err_prior, (err_prior, err_post)

    # the innovation norm kalman_update reports must also fall
    assert result.innovation_post_norm < 0.6 * result.innovation_prior_norm


def test_covariance_contracts(problem):
    """Posterior covariance trace is smaller than the prior's."""
    h, keep = build_observation_matrix(problem["m_sens"])
    y_obs, vac, sc = restrict_observation(
        problem["y_true"], problem["vacuum"], problem["scale"], keep
    )
    result = ensemble_correct(
        problem["ensemble"], h, y_obs, vac, sc, rng=np.random.default_rng(1)
    )
    assert np.trace(result.cov_post) < np.trace(result.cov_prior)


def test_masked_sensor_path(problem):
    """Dropping an untrusted sensor row still runs and keeps shapes aligned."""
    mask = np.ones(problem["m_sens"].shape[0], dtype=bool)
    mask[2] = False
    h, keep = build_observation_matrix(problem["m_sens"], mask=mask)
    assert h.shape[0] == mask.sum()
    y_obs, vac, sc = restrict_observation(
        problem["y_true"], problem["vacuum"], problem["scale"], keep
    )
    result = ensemble_correct(
        problem["ensemble"], h, y_obs, vac, sc, rng=np.random.default_rng(2)
    )
    assert result.ensemble_post.shape == problem["ensemble"].shape
    assert np.isfinite(result.ensemble_post).all()
    assert result.cov_post.shape == (problem["ensemble"].shape[1],) * 2


def test_reduced_rank_agrees_with_full_rank_on_leading_mode(problem):
    """rank=1 localisation recovers the same move as full-rank along that mode.

    With a near-isotropic prior ensemble covariance, the full-rank Kalman
    gain's dominant direction of movement coincides with the leading
    observable mode of ``H`` (the direction the reduced-rank path resolves
    exactly by construction) — the one case the two code paths can be
    checked against each other without re-deriving a Kalman gain formula.
    """
    h, keep = build_observation_matrix(problem["m_sens"])
    y_obs, vac, sc = restrict_observation(
        problem["y_true"], problem["vacuum"], problem["scale"], keep
    )
    res_full = ensemble_correct(
        problem["ensemble"], h, y_obs, vac, sc, rng=np.random.default_rng(1)
    )
    res_r1 = ensemble_correct(
        problem["ensemble"], h, y_obs, vac, sc, rng=np.random.default_rng(1), rank=1
    )
    assert res_r1.modes is not None
    mode0 = res_r1.modes[:, 0]

    delta_full = res_full.mean_post - res_full.mean_prior
    proj_full = float(delta_full @ mode0)
    delta_r1 = res_r1.mean_post - res_r1.mean_prior
    coeff_r1 = float(delta_r1 @ mode0)

    assert coeff_r1 != pytest.approx(0.0, abs=1e-6)
    rel_err = abs(proj_full - coeff_r1) / abs(coeff_r1)
    assert rel_err < 0.05, (proj_full, coeff_r1, rel_err)

    # the reduced-rank correction lies (by construction) entirely along mode0
    resid = delta_r1 - coeff_r1 * mode0
    assert np.linalg.norm(resid) < 1e-8 * max(np.linalg.norm(delta_r1), 1.0)


def test_ensemble_needs_at_least_two_members(problem):
    h, keep = build_observation_matrix(problem["m_sens"])
    y_obs, vac, sc = restrict_observation(
        problem["y_true"], problem["vacuum"], problem["scale"], keep
    )
    with pytest.raises(ValueError):
        ensemble_correct(problem["ensemble"][:1], h, y_obs, vac, sc)
