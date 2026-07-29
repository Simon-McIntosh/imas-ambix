"""Passive-resistance calibration — pure-logic units.

Load-bearing contracts:

* the vectorised uniform-cadence ZOH response equals the exact integrator;
* a uniform resistance multiplier maps every τ → τ/s with unchanged
  eigenvectors;
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
    a_circuit = RNG.normal(size=(5, 2)) * 1e-6
    g_grid = RNG.normal(size=(12, 2)) * 1e-6
    m_channel = RNG.normal(size=(2, 3)) * 1e-5
    return PassiveCircuitSystem(
        circuits=np.array([101, 14]),
        centroid_r=np.array([0.3, 1.5]),
        centroid_z=np.array([0.0, 1.1]),
        lmat=lmat,
        r_diag=r_diag,
        a_circuit=a_circuit,
        g_grid=g_grid,
        m_channel=m_channel,
        channels=["a_current", "b_current", "sol_current"],
        measured_channel_row={"p2u_case_current": 1} if n_case else {},
        resistivity=7.2e-7,
        section_scale=np.array([0.1, 0.1]),
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
    psi_circ = i_drive @ system.m_channel.T
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
    system.measured_channel_row = {
        "p2u_case_current": 1,
        "a2l_case_current": 0,
    }
    maps = campaign_mode_maps(system, np.ones(2))
    assert isinstance(maps, ModeMaps)
    # sorted channel order: a2l first → row 0's eigen-row first
    np.testing.assert_array_equal(maps.case_v[0], maps.v[0])
    np.testing.assert_array_equal(maps.case_v[1], maps.v[1])


# ---------------------------------------------------------------------------
# Structure discovery
# ---------------------------------------------------------------------------
from imas_ambix.latent.passive_resistance import (  # noqa: E402
    PassiveStructure,
    build_structure_hypothesis,
    case_parent_coil_channels,
    coil_pair_channels,
    load_structure,
    neighbour_edges,
    save_structure,
    series_reduction,
    structured_mode_maps,
    structured_shot_loss,
)


def test_voltage_drive_zoh_matches_exact_integrator_and_steady_state():
    tau = np.array([0.030, 0.008])
    dt = 1e-3
    times = np.arange(600) * dt
    psi = np.zeros((600, 2))
    volt = np.zeros((600, 2))
    volt[100:] = np.array([2.0, -1.0])  # step voltage
    a_ref, _ = integrate_eddy_ode(tau, times, psi, volt_m=volt)
    a = zoh_mode_response(tau, dt, psi, volt_m=volt)
    np.testing.assert_allclose(a, a_ref, rtol=1e-10, atol=1e-16)
    # constant-voltage steady state of da/dt + a/τ = v is a = τ·v
    np.testing.assert_allclose(a[-1], tau * volt[-1], rtol=1e-3)


def test_voltage_drive_matches_dense_substepping():
    tau = np.array([0.012])
    dt = 2e-3
    n = 200
    rng = np.random.default_rng(3)
    volt_coarse = np.cumsum(rng.normal(size=(n, 1)), axis=0) * 0.1
    times = np.arange(n) * dt
    a = zoh_mode_response(tau, dt, np.zeros((n, 1)), volt_m=volt_coarse)
    # dense grid: linearly interpolated voltage, 50× sub-stepping
    fine = 50
    tf = np.arange((n - 1) * fine + 1) * (dt / fine)
    vf = np.interp(tf, times, volt_coarse[:, 0])[:, np.newaxis]
    a_dense = zoh_mode_response(tau, dt / fine, np.zeros_like(vf), volt_m=vf)
    np.testing.assert_allclose(a[-1], a_dense[-1], rtol=2e-3)


def test_series_reduction_classical_inductance_algebra():
    lmat = np.array([[2.0, 0.6], [0.6, 1.5]])
    c_ser = series_reduction(2, [(0, 1, +1)])
    c_ant = series_reduction(2, [(0, 1, -1)])
    # series: L_eff = L11 + L22 + 2M; anti-series: − 2M
    np.testing.assert_allclose(c_ser.T @ lmat @ c_ser, [[2.0 + 1.5 + 1.2]])
    np.testing.assert_allclose(c_ant.T @ lmat @ c_ant, [[2.0 + 1.5 - 1.2]])
    r = np.diag([4.0, 8.0])
    np.testing.assert_allclose(c_ser.T @ r @ c_ser, [[12.0]])
    np.testing.assert_allclose(c_ant.T @ r @ c_ant, [[12.0]])
    # drives sum with the wiring sign
    u = np.array([1.0, 0.25])
    np.testing.assert_allclose(c_ser.T @ u, [1.25])
    np.testing.assert_allclose(c_ant.T @ u, [0.75])
    with pytest.raises(ValueError, match="disjoint"):
        series_reduction(3, [(0, 1, 1), (1, 2, 1)])


def test_neighbour_edges_size_normalised_rule():
    r = np.array([1.0, 1.0, 1.0, 2.0])
    z = np.array([0.0, 0.1, 0.5, 0.0])
    s = np.array([0.1, 0.1, 0.1, 0.1])
    edges = neighbour_edges(r, z, s, factor=1.5)
    assert (0, 1) in edges  # 0.1 apart, threshold 0.15
    assert (0, 2) not in edges and (1, 2) not in edges
    assert all(3 not in e for e in edges)
    # excluding a row removes its edges
    assert neighbour_edges(r, z, s, factor=1.5, exclude_rows={1}) == []


def test_case_parent_and_pair_channel_rules():
    coils = [
        "p2il_coil_current",
        "p2iu_coil_current",
        "p2ol_coil_current",
        "p2ou_coil_current",
        "p4u_coil_current",
        "p4l_coil_current",
        "p6u_current",
        "p6l_current",
        "sol_current",
    ]
    assert case_parent_coil_channels("p2l_case_current", coils) == [
        "p2il_coil_current",
        "p2ol_coil_current",
    ]
    assert case_parent_coil_channels("p4u_case_current", coils) == ["p4u_coil_current"]
    pairs = coil_pair_channels(coils)
    assert ("p2iu_coil_current", "p2il_coil_current") in pairs
    assert ("p4u_coil_current", "p4l_coil_current") in pairs
    assert ("p6u_current", "p6l_current") in pairs
    assert all("sol" not in p for pair in pairs for p in pair)


def _toy_system3() -> PassiveCircuitSystem:
    """Three passive circuits (row 2 is a measured case), three drives."""
    lmat = np.array([[2.0, 0.5, 0.2], [0.5, 1.8, 0.3], [0.2, 0.3, 1.2]]) * 1e-6
    r_diag = np.array([4.0e-5, 6.0e-5, 9.0e-5])
    rng = np.random.default_rng(11)
    return PassiveCircuitSystem(
        circuits=np.array([201, 202, 14]),
        centroid_r=np.array([1.0, 1.0, 1.4]),
        centroid_z=np.array([0.0, 0.08, 1.0]),
        lmat=lmat,
        r_diag=r_diag,
        a_circuit=rng.normal(size=(5, 3)) * 1e-6,
        g_grid=rng.normal(size=(12, 3)) * 1e-6,
        m_channel=rng.normal(size=(3, 3)) * 1e-5,
        channels=["p4l_coil_current", "p4u_coil_current", "sol_current"],
        measured_channel_row={"p4u_case_current": 2},
        resistivity=7.2e-7,
        section_scale=np.array([0.1, 0.1, 0.05]),
    )


def test_structured_maps_reduce_to_diagonal_model_when_structure_empty():
    system = _toy_system3()
    mult = np.array([1.5, 2.0, 0.8])
    base = campaign_mode_maps(system, mult)
    hyp = build_structure_hypothesis(system, np.arange(3))
    smaps = structured_mode_maps(hyp, mult)
    np.testing.assert_allclose(np.sort(smaps.tau), np.sort(base.tau), rtol=1e-12)
    # identical drive response: psi_m @ decay == via base maps on a test drive
    rng = np.random.default_rng(0)
    i_drive = np.cumsum(rng.normal(size=(400, 3)), axis=0) * 10.0
    psi_circ = i_drive @ system.m_channel.T
    a_base = zoh_mode_response(base.tau, 1e-3, psi_circ @ base.v)
    pred_base = a_base @ base.a_sens_modes.T
    a_s = zoh_mode_response(smaps.tau, 1e-3, i_drive @ smaps.drive_flux.T)
    pred_s = a_s @ smaps.a_sens_modes.T
    np.testing.assert_allclose(pred_s, pred_base, rtol=1e-8, atol=1e-24)


def test_adjacency_stamp_couples_and_preserves_spd():
    system = _toy_system3()
    hyp = build_structure_hypothesis(system, np.arange(3), edges=[(0, 1)])
    # strong coupling: the differential mode of the pair decays ~instantly
    smaps = structured_mode_maps(hyp, np.ones(3), edge_r=np.array([1.0]))
    base = campaign_mode_maps(system, np.ones(3))
    assert np.min(smaps.tau) < np.min(base.tau) * 1e-3
    assert np.all(smaps.tau > 0)  # SPD preserved
    # zero coupling: byte-identical spectrum
    smaps0 = structured_mode_maps(hyp, np.ones(3), edge_r=np.array([0.0]))
    np.testing.assert_allclose(np.sort(smaps0.tau), np.sort(base.tau), rtol=1e-12)


def test_wiring_flux_edit_and_voltage_term():
    system = _toy_system3()
    lam_channels = list(system.channels)
    lam = np.array([[5.0, 1.0, 0.3], [1.0, 4.0, 0.2], [0.3, 0.2, 8.0]]) * 1e-5
    hyp = build_structure_hypothesis(
        system,
        np.arange(3),
        wiring_cases=["p4u_case_current"],
        drive_linkage=(lam_channels, lam),
    )
    # parent of p4u is the p4u winding — column 1
    np.testing.assert_allclose(hyp.wiring_sel, [[0.0, 1.0, 0.0]])
    np.testing.assert_allclose(hyp.wiring_lam, [lam[1]])
    g_v, r_w = np.array([2.0]), np.array([3e-3])
    smaps = structured_mode_maps(hyp, np.ones(3), g_v=g_v, r_w=r_w)
    # flux edit lands only on the case row: m_eff = m − g_v·lam_parent
    m_eff_expected = system.m_channel.copy()
    m_eff_expected[2] -= 2.0 * lam[1]
    np.testing.assert_allclose(
        smaps.drive_flux, smaps.v_phys.T @ m_eff_expected, rtol=1e-12
    )
    volt_cols = np.zeros((3, 3))
    volt_cols[2, 1] = 3e-3
    np.testing.assert_allclose(smaps.drive_volt, smaps.v_phys.T @ volt_cols, rtol=1e-12)


def test_structured_loss_recovers_true_wiring_gain():
    """Synthetic truth with a galvanic case wiring: the structured loss is
    minimised at the true g_v — from held-back case supervision alone."""
    system = _toy_system3()
    lam_channels = list(system.channels)
    lam = np.array([[5.0, 1.0, 0.3], [1.0, 4.0, 0.2], [0.3, 0.2, 8.0]]) * 1e-5
    hyp = build_structure_hypothesis(
        system,
        np.arange(3),
        wiring_cases=["p4u_case_current"],
        drive_linkage=(lam_channels, lam),
    )
    g_true = 3.0
    rng = np.random.default_rng(5)
    i_drive = np.cumsum(rng.normal(0, 30.0, size=(1500, 3)), axis=0)
    truth = structured_mode_maps(hyp, np.ones(3), g_v=np.array([g_true]))
    a = zoh_mode_response(truth.tau, 1e-3, i_drive @ truth.drive_flux.T)
    case = a @ truth.case_map.T + rng.normal(0, 0.02, size=(1500, 1))
    meas = a @ truth.a_sens_modes.T + rng.normal(0, 1e-5, size=(1500, 5))
    data = VacuumShotData(
        shot=1,
        campaign="toy",
        stratum="dedicated_vacuum",
        dt=1e-3,
        psi_circ=i_drive @ system.m_channel.T,
        meas_resid=meas,
        sigma=np.full(5, 1e-5),
        case_meas=case,
        i_drive=i_drive,
    )
    sig_case = np.array([max(float(np.nanstd(case)), 1.0)])
    losses = []
    grid = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 6.0])
    for g in grid:
        m = structured_mode_maps(hyp, np.ones(3), g_v=np.array([g]))
        sm, nm, sc, nc = structured_shot_loss(data, m, np.full(5, 1e-5), sig_case)
        losses.append(sm / max(nm, 1) + sc / max(nc, 1))
    assert grid[int(np.argmin(losses))] == g_true


def test_structured_loss_requires_i_drive_and_preserves_holdback():
    system = _toy_system3()
    hyp = build_structure_hypothesis(system, np.arange(3))
    smaps = structured_mode_maps(hyp, np.ones(3))
    shot = _simulate_shot(_toy_system(), np.array([4.0, 4.0]), seed=1)
    with pytest.raises(ValueError, match="i_drive"):
        structured_shot_loss(shot, smaps, np.full(5, 1e-5), np.array([1.0]))
    # corrupting the held-back case target changes the case loss ONLY —
    # the drive assembly never reads case_meas
    rng = np.random.default_rng(2)
    i_drive = np.cumsum(rng.normal(size=(800, 3)), axis=0) * 20.0
    a = zoh_mode_response(smaps.tau, 1e-3, i_drive @ smaps.drive_flux.T)
    data = VacuumShotData(
        shot=1,
        campaign="toy",
        stratum="fleet",
        dt=1e-3,
        psi_circ=i_drive @ system.m_channel.T,
        meas_resid=a @ smaps.a_sens_modes.T,
        sigma=np.full(5, 1e-5),
        case_meas=a @ smaps.case_map.T,
        i_drive=i_drive,
    )
    base = structured_shot_loss(data, smaps, np.full(5, 1e-5), np.array([1.0]))
    corrupted = VacuumShotData(
        **{
            **data.__dict__,
            "case_meas": data.case_meas + 50.0 * np.sin(np.arange(800)[:, None] / 30.0),
        }
    )
    hit = structured_shot_loss(corrupted, smaps, np.full(5, 1e-5), np.array([1.0]))
    assert hit[2] > base[2] + 1.0
    np.testing.assert_allclose(hit[0], base[0], rtol=1e-12)


def test_series_constrained_case_pair_predicts_both_channels_equal():
    system = _toy_system3()
    system.measured_channel_row = {
        "p4u_case_current": 2,
        "p4l_case_current": 1,
    }
    hyp = build_structure_hypothesis(
        system,
        np.arange(3),
        case_series=[("p4l_case_current", "p4u_case_current", +1)],
    )
    assert hyp.c_reduce.shape == (3, 2)
    smaps = structured_mode_maps(hyp, np.ones(3))
    rng = np.random.default_rng(9)
    i_drive = np.cumsum(rng.normal(size=(300, 3)), axis=0) * 15.0
    a = zoh_mode_response(smaps.tau, 1e-3, i_drive @ smaps.drive_flux.T)
    pred = a @ smaps.case_map.T  # sorted channels: p4l first, p4u second
    np.testing.assert_allclose(pred[:, 0], pred[:, 1], rtol=1e-12)


def test_structured_reduced_basis_matches_diagonal_reduction():
    """Empty structure → byte-comparable to reduce_passive_system; wiring →
    voltage-channel terms present and the flux columns carry the wiring edit."""
    from imas_ambix.latent.passive_resistance import (
        structure_hypothesis_parts,
        structured_reduced_basis,
    )
    from imas_ambix.latent.temporal_operator import reduce_passive_system

    class _Grid:
        cells = np.array([3, 7, 11])

    system = _toy_system3()
    empty = PassiveStructure(
        case_series_pairs=[],
        case_wiring={},
        pair_drive_gains=[],
        adjacency={},
        neighbour_rule={},
        r_level="regions-percase",
        r_group_multipliers={
            # circuit id 14 hits the real machine's case metadata (p2u), so
            # both case groups appear in the toy's label set
            "case:p2u": 1.3,
            "case:p4u": 1.3,
            "vessel:inboard": 2.0,
            "vessel:outboard": 4.0,
            "vessel:mid": 8.0,
            "vessel:ends": 16.0,
        },
        provenance={},
    )
    scale = np.full(5, 1e-5)
    hyp, parts = structure_hypothesis_parts(system, empty)
    ref = reduce_passive_system(
        system, _Grid, sensor_scale=scale, k=2, r_multipliers=parts["multipliers"]
    )
    got = structured_reduced_basis(
        system, empty, sensor_scale=scale, k=2, cells=_Grid.cells
    )
    np.testing.assert_allclose(got.tau, ref.tau, rtol=1e-12)
    np.testing.assert_allclose(np.abs(got.v), np.abs(ref.v), rtol=1e-9)
    np.testing.assert_allclose(np.abs(got.m_channel), np.abs(ref.m_channel), rtol=1e-9)
    np.testing.assert_allclose(np.abs(got.m_cell), np.abs(ref.m_cell), rtol=1e-9)
    assert got.volt_channel is None

    wired = PassiveStructure(
        case_series_pairs=[],
        case_wiring={
            "p4u_case_current": {
                "parents": ["p4u_coil_current"],
                "g_v": 5.0,
                "r_w": 2e-3,
            }
        },
        pair_drive_gains=[],
        adjacency={},
        neighbour_rule={},
        r_level=empty.r_level,
        r_group_multipliers=empty.r_group_multipliers,
        provenance={},
    )
    lam = np.array([[5.0, 1.0, 0.3], [1.0, 4.0, 0.2], [0.3, 0.2, 8.0]]) * 1e-5
    got_w = structured_reduced_basis(
        system,
        wired,
        sensor_scale=scale,
        k=3,
        cells=_Grid.cells,
        drive_linkage=(list(system.channels), lam),
    )
    assert got_w.volt_channel is not None and got_w.volt_channel.shape == (3, 3)
    with pytest.raises(ValueError, match="drive_linkage"):
        structured_reduced_basis(
            system, wired, sensor_scale=scale, k=2, cells=_Grid.cells
        )


def test_structure_roundtrip(tmp_path):
    s = PassiveStructure(
        case_series_pairs=[
            {"channels": ["p3l_case_current", "p3u_case_current"], "sign": 1}
        ],
        case_wiring={
            "p2l_case_current": {
                "parents": ["p2il_coil_current", "p2ol_coil_current"],
                "g_v": 11.2,
                "r_w": 2.4e-3,
            }
        },
        pair_drive_gains=[
            {
                "channels": ["p4u_coil_current", "p4l_coil_current"],
                "common": 0.02,
                "differential": -0.01,
            }
        ],
        adjacency={"fc1004": [{"i": 201, "j": 202, "r_couple": 3.0e-4}]},
        neighbour_rule={"factor": 1.5, "metric": "pair-mean section scale"},
        r_level="regions-percase",
        r_group_multipliers={"vessel:mid": 12.7},
        provenance={"pool": "unit-test"},
    )
    path = tmp_path / "structure.json"
    save_structure(path, s)
    back = load_structure(path)
    assert back.case_wiring["p2l_case_current"]["g_v"] == 11.2
    assert back.adjacency["fc1004"][0]["r_couple"] == 3.0e-4
    assert back.case_series_pairs == s.case_series_pairs
    with pytest.raises(ValueError, match="not a passive-structure"):
        (tmp_path / "junk.json").write_text('{"kind": "other"}')
        load_structure(tmp_path / "junk.json")
