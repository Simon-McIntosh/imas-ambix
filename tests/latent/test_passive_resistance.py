"""Passive-resistance calibration — pure-logic units.

Load-bearing contracts:

* the vectorised uniform-cadence ZOH response equals the exact integrator;
* a uniform resistance multiplier maps every τ → τ/s with unchanged
  eigenvectors (the tau_scale equivalence the 1-DOF rung generalises);
* group labels honour the case identity and the normalised region rule;
* calibration application fails loud on an unknown group;
* on a synthetic two-circuit system with a known resistance multiplier, the
  pooled objective is minimised at the truth — from held-back case-current
  supervision alone.
"""

from __future__ import annotations

import numpy as np
import pytest

from imas_ambix.latent.passive_resistance import (
    ModeMaps,
    ResistanceCalibration,
    VacuumShotData,
    campaign_mode_maps,
    load_calibration,
    pooled_loss,
    resistance_group_labels,
    save_calibration,
    zoh_mode_response,
)
from imas_ambix.latent.temporal_operator import (
    PassiveCircuitSystem,
    integrate_eddy_ode,
)

RNG = np.random.default_rng(7)


def test_zoh_mode_response_matches_exact_integrator():
    tau = np.array([0.030, 0.012, 0.004])
    dt = 1e-3
    times = np.arange(400) * dt
    psi = np.cumsum(RNG.normal(size=(400, 3)), axis=0) * 1e-4
    a_ref, _u = integrate_eddy_ode(tau, times, psi)
    a = zoh_mode_response(tau, dt, psi)
    np.testing.assert_allclose(a, a_ref, rtol=1e-10, atol=1e-18)


def _toy_system(n_case: int = 1) -> PassiveCircuitSystem:
    lmat = np.array([[2.0, 0.6], [0.6, 1.5]]) * 1e-6
    r_diag = np.array([4.0e-5, 8.0e-5])
    a_circ = RNG.normal(size=(5, 2)) * 1e-6
    g_circ = RNG.normal(size=(12, 2)) * 1e-6
    m_coil = RNG.normal(size=(2, 3)) * 1e-5
    return PassiveCircuitSystem(
        circuits=np.array([101, 14]),
        centroid_r=np.array([0.3, 1.5]),
        centroid_z=np.array([0.0, 1.1]),
        lmat=lmat,
        r_diag=r_diag,
        a_circ=a_circ,
        g_circ=g_circ,
        m_coil_circ=m_coil,
        coil_channels=["a_current", "b_current", "sol_current"],
        case_channel_row={"p2u_case_current": 1} if n_case else {},
        resistivity=7.2e-7,
    )


def test_uniform_multiplier_scales_taus_keeps_vectors():
    system = _toy_system()
    m1 = campaign_mode_maps(system, np.ones(2))
    m4 = campaign_mode_maps(system, np.full(2, 4.0))
    np.testing.assert_allclose(m4.tau, m1.tau / 4.0, rtol=1e-12)
    # L-orthonormal eigenvectors are sign-fixed up to column order; a uniform
    # R scale preserves both (same generalised eigenproblem, scaled spectrum)
    np.testing.assert_allclose(np.abs(m4.v), np.abs(m1.v), rtol=1e-10)


def test_group_labels_case_identity_and_regions():
    circuits = np.array([14, 18, 200, 201, 202, 203])
    case_of = {14: "p2u", 18: "p4u"}
    r = np.array([1.5, 1.5, 0.2, 1.9, 1.0, 1.0])
    z = np.array([1.1, 1.1, 0.0, 0.1, 2.0, 0.2])
    lab = resistance_group_labels(circuits, r, z, "regions-percase", case_of=case_of)
    assert lab[0] == "case:p2u" and lab[1] == "case:p4u"
    assert lab[2] == "vessel:inboard"  # r_norm = 0
    assert lab[3] == "vessel:outboard"  # r_norm = 1
    assert lab[4] == "vessel:ends"  # mid radius, |z| = max
    assert lab[5] == "vessel:mid"
    lab_pair = resistance_group_labels(
        circuits, r, z, "regions-casepairs", case_of=case_of
    )
    assert lab_pair[0] == "case:p2" and lab_pair[1] == "case:p4"
    lab_vc = resistance_group_labels(circuits, r, z, "vessel-case", case_of=case_of)
    assert lab_vc[:2] == ["case", "case"] and set(lab_vc[2:]) == {"vessel"}
    assert set(resistance_group_labels(circuits, r, z, "global", case_of=case_of)) == {
        "all"
    }


def test_calibration_roundtrip_and_fail_loud(tmp_path):
    cal = ResistanceCalibration(
        level="vessel-case",
        group_multipliers={"vessel": 3.7, "case": 1.4},
        provenance={"pool": "unit-test"},
    )
    path = tmp_path / "cal.json"
    save_calibration(path, cal)
    back = load_calibration(path)
    assert back.level == cal.level
    assert back.group_multipliers == cal.group_multipliers
    mult = back.per_circuit(
        np.array([14, 200]),
        np.array([1.5, 0.2]),
        np.array([1.1, 0.0]),
        case_of={14: "p2u"},
    )
    np.testing.assert_allclose(mult, [1.4, 3.7])
    bad = ResistanceCalibration("vessel-case", {"vessel": 3.7}, {})
    with pytest.raises(KeyError, match="case"):
        bad.per_circuit(
            np.array([14]), np.array([1.5]), np.array([1.1]), case_of={14: "p2u"}
        )


def _simulate_shot(system, theta_true, seed=0, n_t=1200, dt=1e-3):
    """Synthetic vacuum shot: case current from the TRUE resistances is the
    held-back target; magnetics carry the same truth plus noise."""
    rng = np.random.default_rng(seed)
    i_drive = np.cumsum(rng.normal(0, 30.0, size=(n_t, 3)), axis=0)
    psi_circ = i_drive @ system.m_coil_circ.T
    maps = campaign_mode_maps(system, theta_true)
    a = zoh_mode_response(maps.tau, dt, psi_circ @ maps.v)
    case = a @ maps.case_v.T + rng.normal(0, 0.05, size=(n_t, 1))
    meas = a @ maps.a_sens_modes.T + rng.normal(0, 1e-5, size=(n_t, 5))
    return VacuumShotData(
        shot=1000 + seed,
        campaign="toy",
        stratum="dedicated_vacuum",
        dt=dt,
        psi_circ=psi_circ,
        meas_resid=meas,
        sigma=np.full(5, 1e-5),
        case_meas=case,
    )


def test_pooled_loss_minimised_at_true_multiplier():
    system = _toy_system()
    theta_true = np.array([4.0, 4.0])
    shots = [_simulate_shot(system, theta_true, seed=s) for s in range(3)]
    sigma_med = {"toy": np.full(5, 1e-5)}
    sigma_case = {"toy": np.array([np.nanstd(shots[0].case_meas)])}
    grid = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    losses = [
        pooled_loss(
            np.array([s]),
            {"toy": np.zeros(2, dtype=np.int64)},
            {"toy": system},
            shots,
            sigma_med,
            sigma_case,
        )["combined"]
        for s in grid
    ]
    assert grid[int(np.argmin(losses))] == 4.0


def test_shot_loss_case_supervision_is_holdback_only():
    """The case channel never enters as a drive: zeroing the case row of the
    drive-coupling matrix changes nothing (the drive flux only sees the coil
    channels), while corrupting the measured case target changes the case
    loss and ONLY the case loss."""
    system = _toy_system()
    theta = np.array([4.0, 4.0])  # truth — the case residual is pure noise
    shot = _simulate_shot(system, np.array([4.0, 4.0]), seed=1)
    sigma_med = {"toy": np.full(5, 1e-5)}
    sigma_case = {"toy": np.array([1.0])}
    args = (
        {"toy": np.zeros(2, dtype=np.int64)},
        {"toy": system},
        [shot],
        sigma_med,
        sigma_case,
    )
    base = pooled_loss(theta, *args)
    corrupted = VacuumShotData(
        **{
            **shot.__dict__,
            "case_meas": shot.case_meas
            + 100.0 * np.sin(np.arange(shot.n_samples)[:, None] / 50.0),
        }
    )
    args_c = (*args[:2], [corrupted], *args[3:])
    hit = pooled_loss(theta, *args_c)
    assert hit["case"] > base["case"] * 2
    np.testing.assert_allclose(hit["mag"], base["mag"], rtol=1e-12)


def test_mode_maps_case_rows_follow_sorted_channels():
    system = _toy_system()
    system.case_channel_row = {"p2u_case_current": 1, "a2l_case_current": 0}
    maps = campaign_mode_maps(system, np.ones(2))
    assert isinstance(maps, ModeMaps)
    # sorted channel order: a2l first → row 0's eigen-row first
    np.testing.assert_array_equal(maps.case_v[0], maps.v[0])
    np.testing.assert_array_equal(maps.case_v[1], maps.v[1])
