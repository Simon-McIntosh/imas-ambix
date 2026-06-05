"""Unit tests for the S10 sequential ψ-state baseline substrate."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from imas_ambix.statespace.sequential_da import (
    MAST_A,
    MAST_B0,
    MAST_R0,
    MU0,
    build_h_psi,
    current_from_psi_profile,
    kalman_update,
    psi_from_current_profile,
    q_from_current_profile,
)


def test_psi_current_roundtrip_on_smooth_profile():
    rho = np.linspace(0.0, 1.0, 33)
    j_true = 1.8e6 * (1.0 - rho**2)
    psi = psi_from_current_profile(j_true, rho, r_major=MAST_R0, a_minor=MAST_A)
    j_back = current_from_psi_profile(psi, rho, r_major=MAST_R0, a_minor=MAST_A)
    # Interior agreement matters; the edge stencil is intentionally approximate.
    rel = np.abs(j_back[1:-1] - j_true[1:-1]) / np.maximum(np.abs(j_true[1:-1]), 1.0)
    assert rel.mean() < 0.10
    assert rel.max() < 0.35


def test_q_profile_matches_axis_limit_for_uniform_current():
    rho = np.linspace(0.0, 1.0, 33)
    j0 = 2.4e6
    q = q_from_current_profile(
        np.full_like(rho, j0),
        rho,
        r_major=MAST_R0,
        a_minor=MAST_A,
        bt0=MAST_B0,
    )
    expected_q0 = 2.0 * MAST_B0 / (MU0 * MAST_R0 * j0)
    assert np.isfinite(q).all()
    assert abs(q[0] - expected_q0) / expected_q0 < 0.05
    assert np.all(q > 0)


@dataclass
class _FakeOperator:
    pf_amc_channels: tuple[str, ...] = ("pf",)


class _FakeObs:
    def __init__(self, matrix):
        self.matrix = np.asarray(matrix, dtype=np.float64)
        self.operator = _FakeOperator()
        self.trust_rows = np.arange(self.matrix.shape[0], dtype=int)

    def predict_amb(self, j_rho, rho_grid, i_pf):
        return self.matrix @ np.asarray(j_rho, dtype=np.float64)


def test_build_h_psi_matches_direct_linearisation():
    rho = np.linspace(0.0, 1.0, 9)
    obs = _FakeObs([[1.0, 0.5, -0.2, 0.0, 0.0, 0.1, 0.2, -0.1, 0.0]])
    h_psi = build_h_psi(obs, rho, r_major=MAST_R0, a_minor=MAST_A)
    psi = np.sin(np.pi * rho) * 0.1
    direct = obs.predict_amb(
        current_from_psi_profile(psi, rho, r_major=MAST_R0, a_minor=MAST_A),
        rho,
        np.zeros(1),
    )
    via_matrix = h_psi @ psi
    np.testing.assert_allclose(via_matrix, direct, rtol=1e-6, atol=1e-8)


def test_kalman_update_reduces_whitened_residual():
    mean = np.zeros(2)
    cov = 0.5 * np.eye(2)
    h = np.array([[1.0, 0.0], [0.5, 1.0]])
    residual = np.array([1.0, -0.25])
    sensor_std = np.array([0.2, 0.2])
    mean_post, cov_post, prior_norm, post_norm = kalman_update(
        mean, cov, h, residual, sensor_std
    )
    assert post_norm < prior_norm
    assert np.all(np.linalg.eigvalsh(cov_post) >= -1e-10)
    assert mean_post.shape == mean.shape
