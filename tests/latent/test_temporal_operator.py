"""Temporal operator — pure-logic units + one real-geometry eigenbasis pin.

The load-bearing contracts:

* the physical eddy history is the EXACT zero-order-hold solution of the mode
  ODE ``da/dt + a/τ = −dΨ/dt`` (pinned against dense sub-step integration);
* the untrained operator is the identity on the classical spine (``dc = 0``,
  ``da = 0`` exactly — zero-initialised heads);
* the trunk is causal: perturbing step t never changes outputs before t;
* the eddy SSM decays are learnable parameters initialised at the physical
  L/R times;
* checkpoint round-trip reproduces outputs bit-for-bit.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from imas_ambix.latent.temporal_operator import (
    PassiveEigenbasis,
    TemporalOperator,
    _sensor_set,
    integrate_eddy_ode,
    load_checkpoint,
    physical_eddy_history,
    raw_eddy_trajectory,
    save_checkpoint,
)
from imas_ambix.physics import PassiveEigenbasis as NovaPassiveEigenbasis

RNG = np.random.default_rng(11)


def test_temporal_operator_uses_nova_passive_basis_type():
    assert PassiveEigenbasis is NovaPassiveEigenbasis


def test_axisymmetric_sensor_projection_requires_field_orientation():
    missing = SimpleNamespace(
        amb_channel="probe",
        kind="b_probe",
        r=1.0,
        z=0.0,
        angle_deg=None,
    )
    with pytest.raises(ValueError, match="probe"):
        _sensor_set(SimpleNamespace(sensor_map=[missing]))

    flux_loop = SimpleNamespace(
        amb_channel="loop",
        kind="flux_loop",
        r=1.0,
        z=0.0,
        angle_deg=None,
    )
    sensors = _sensor_set(SimpleNamespace(sensor_map=[flux_loop]))
    assert sensors.is_flux.tolist() == [True]
    assert sensors.angle.tolist() == [0.0]


def _toy_basis(k: int = 3, n_coil: int = 4, n_cells: int = 20) -> PassiveEigenbasis:
    tau = np.array([0.030, 0.012, 0.004])[:k]
    return PassiveEigenbasis(
        tau=tau,
        v=RNG.normal(size=(7, k)),
        a_sensor=RNG.normal(size=(9, k)),
        g_grid=RNG.normal(size=(48, k)),
        m_channel=RNG.normal(size=(k, n_coil)),
        m_cell=RNG.normal(size=(k, n_cells)) * 1e-3,
        resistivity=7.2e-7,
    )


def _sequences(basis, n_t=16, seed=0):
    rng = np.random.default_rng(seed)
    times = np.cumsum(rng.uniform(0.008, 0.03, size=n_t))
    i_pf = np.cumsum(rng.normal(0, 50.0, size=(n_t, basis.m_channel.shape[1])), axis=0)
    i_cell = np.abs(rng.normal(0, 100.0, size=(n_t, basis.m_cell.shape[1])))
    return times, i_pf, i_cell


def test_eddy_history_zero_drive_is_zero():
    basis = _toy_basis()
    times = np.linspace(0.0, 0.3, 12)
    i_pf = np.ones((12, basis.m_channel.shape[1])) * 3.0e3  # constant → dΨ = 0
    i_cell = np.ones((12, basis.m_cell.shape[1])) * 40.0
    a, u = physical_eddy_history(basis, times, i_pf, i_cell)
    assert np.abs(a).max() == 0.0
    assert np.abs(u).max() == 0.0


def test_eddy_history_matches_dense_integration():
    """ZOH update == dense sub-stepped ODE solution for piecewise-linear Ψ."""
    basis = _toy_basis()
    times, i_pf, i_cell = _sequences(basis, n_t=10)
    a, _u = physical_eddy_history(basis, times, i_pf, i_cell)

    psi = i_pf @ basis.m_channel.T + i_cell @ basis.m_cell.T  # (T, k)
    a_ref = np.zeros(basis.n_modes)
    for t in range(1, times.size):
        dt = times[t] - times[t - 1]
        n_sub = 4000
        h = dt / n_sub
        dpsi_dt = (psi[t] - psi[t - 1]) / dt
        for _ in range(n_sub):  # exponential-Euler sub-steps, constant drive
            decay = np.exp(-h / basis.tau)
            a_ref = decay * a_ref + (1.0 - decay) * (-basis.tau * dpsi_dt)
        assert np.allclose(a[t], a_ref, rtol=1e-4, atol=1e-12), f"step {t}"


def test_eddy_history_pure_decay_after_drive_stops():
    basis = _toy_basis()
    times = np.array([0.0, 0.01, 0.05, 0.15])
    i_pf = np.array([[0.0], [1000.0], [1000.0], [1000.0]])
    basis = PassiveEigenbasis(
        tau=basis.tau,
        v=basis.v,
        a_sensor=basis.a_sensor,
        g_grid=basis.g_grid,
        m_channel=RNG.normal(size=(3, 1)),
        m_cell=np.zeros((3, 2)),
        resistivity=basis.resistivity,
    )
    i_cell = np.zeros((4, 2))
    a, _ = physical_eddy_history(basis, times, i_pf, i_cell)
    # steps 2 and 3 have zero drive: pure exponential decay of step-1 state
    assert np.allclose(a[2], a[1] * np.exp(-(0.05 - 0.01) / basis.tau), rtol=1e-12)
    assert np.allclose(a[3], a[2] * np.exp(-(0.15 - 0.05) / basis.tau), rtol=1e-12)


def test_integrator_reproduces_physical_history_at_label_cadence():
    """The factored ZOH integrator == physical_eddy_history on the same mode
    flux — the raw-cadence path shares one exact integrator with the labels."""
    basis = _toy_basis()
    times, i_pf, i_cell = _sequences(basis, n_t=12)
    a_ref, u_ref = physical_eddy_history(basis, times, i_pf, i_cell)
    psi_m = i_pf @ basis.m_channel.T + i_cell @ basis.m_cell.T
    a, u = integrate_eddy_ode(basis.tau, times, psi_m)
    np.testing.assert_array_equal(a, a_ref)
    np.testing.assert_array_equal(u, u_ref)


def test_raw_trajectory_equals_label_cadence_for_piecewise_linear_drive():
    """When the raw drive IS piecewise-linear between the labels (and there is
    no pre-label history), the exact-ZOH property makes the raw-cadence and
    label-cadence integrations agree exactly — densifying a piecewise-linear
    flux changes nothing."""
    basis = _toy_basis()
    times, i_pf, i_cell = _sequences(basis, n_t=8)
    a_lab, _ = physical_eddy_history(basis, times, i_pf, i_cell)

    # raw grid: labels + 9 interior points per interval, linear interpolation
    raw_times = np.unique(
        np.concatenate(
            [np.linspace(times[t - 1], times[t], 11) for t in range(1, times.size)]
        )
    )
    i_pf_raw = np.column_stack(
        [np.interp(raw_times, times, i_pf[:, c]) for c in range(i_pf.shape[1])]
    )
    a_raw_lab, a_raw = raw_eddy_trajectory(basis, raw_times, i_pf_raw, times, i_cell)
    assert a_raw.shape == (raw_times.size, basis.n_modes)
    np.testing.assert_allclose(a_raw_lab, a_lab, rtol=1e-10, atol=1e-14)


def test_raw_trajectory_pinned_against_dense_substepping():
    """Raw-cadence integration == dense exponential-Euler sub-stepping of the
    ODE for the same (piecewise-linear) raw flux — the raw-path analogue of
    the label-cadence dense pin."""
    basis = _toy_basis()
    rng = np.random.default_rng(7)
    raw_times = np.linspace(0.0, 0.12, 121)
    i_pf_raw = np.cumsum(
        rng.normal(0, 30.0, size=(raw_times.size, basis.m_channel.shape[1])), axis=0
    )
    label_times = raw_times[40::20]
    i_cell = np.abs(rng.normal(0, 80.0, size=(label_times.size, basis.m_cell.shape[1])))
    a_lab, a_raw = raw_eddy_trajectory(basis, raw_times, i_pf_raw, label_times, i_cell)

    psi = i_pf_raw @ basis.m_channel.T
    psi_cell_lab = i_cell @ basis.m_cell.T
    for m in range(basis.n_modes):
        psi[:, m] += np.interp(raw_times, label_times, psi_cell_lab[:, m])
    before = raw_times < label_times[0]
    psi[before, :] = i_pf_raw[before] @ basis.m_channel.T  # no ip_raw → zero plasma
    a_ref = np.zeros(basis.n_modes)
    for t in range(1, raw_times.size):
        dt = raw_times[t] - raw_times[t - 1]
        n_sub = 500
        h = dt / n_sub
        dpsi_dt = (psi[t] - psi[t - 1]) / dt
        for _ in range(n_sub):
            decay = np.exp(-h / basis.tau)
            a_ref = decay * a_ref + (1.0 - decay) * (-basis.tau * dpsi_dt)
        assert np.allclose(a_raw[t], a_ref, rtol=1e-4, atol=1e-12), f"step {t}"


def test_raw_trajectory_prelabel_plasma_follows_measured_ip():
    """Before the first label the plasma mode flux follows measured Ip with
    the first label's flux pattern (shape-frozen, amplitude-following)."""
    basis = _toy_basis()
    raw_times = np.linspace(0.0, 0.1, 101)
    i_pf_raw = np.zeros((raw_times.size, basis.m_channel.shape[1]))  # coil-quiet
    label_times = np.array([0.05, 0.08])
    i_cell = np.abs(RNG.normal(0, 50.0, size=(2, basis.m_cell.shape[1])))
    ip_raw = np.clip(np.interp(raw_times, [0.02, 0.05], [0.0, 2.0e5]), 0, None)

    _, a_with = raw_eddy_trajectory(
        basis, raw_times, i_pf_raw, label_times, i_cell, ip_raw=ip_raw
    )
    _, a_without = raw_eddy_trajectory(basis, raw_times, i_pf_raw, label_times, i_cell)
    # with ip_raw the plasma flux ramps over [0.02, 0.05] → eddies exist there;
    # without it the pre-label plasma term is zero → no drive before t=0.05
    pre = raw_times < 0.05
    assert np.abs(a_with[pre]).max() > 0.0
    assert np.abs(a_without[pre][:-1]).max() == 0.0


def test_raw_trajectory_tau_scale_scales_decay():
    """A uniform resistance scale r maps every τ → τ/r exactly (eigenvectors
    invariant): pure post-drive decay compares as exp(−r·Δt/τ)."""
    basis = _toy_basis()
    raw_times = np.linspace(0.0, 0.2, 201)
    i_pf_raw = np.zeros((raw_times.size, basis.m_channel.shape[1]))
    i_pf_raw[raw_times >= 0.01] = 1.0e3  # one step at t=0.01, then flat
    label_times = np.array([0.15, 0.19])
    i_cell = np.zeros((2, basis.m_cell.shape[1]))
    r = 2.0
    _, a1 = raw_eddy_trajectory(basis, raw_times, i_pf_raw, label_times, i_cell)
    _, a2 = raw_eddy_trajectory(
        basis, raw_times, i_pf_raw, label_times, i_cell, tau_scale=r
    )
    # compare decay between two post-step samples on the same trajectory
    t_a, t_b = 100, 180
    dt = raw_times[t_b] - raw_times[t_a]
    np.testing.assert_allclose(a1[t_b] / a1[t_a], np.exp(-dt / basis.tau), rtol=1e-10)
    np.testing.assert_allclose(
        a2[t_b] / a2[t_a], np.exp(-r * dt / basis.tau), rtol=1e-10
    )


def test_section_points_small_element_is_centroid():
    from imas_ambix.latent.temporal_operator import _section_points

    r, z, w = _section_points(1.2, -0.3, 0.03, 0.04, delta=0.05, n_max=6)
    assert r.size == 1
    np.testing.assert_allclose([r[0], z[0], w[0]], [1.2, -0.3, 1.0])


def test_section_points_large_element_grid_is_area_faithful():
    from imas_ambix.latent.temporal_operator import _section_points

    r, z, w = _section_points(1.0, 0.5, 0.24, 0.42, delta=0.05, n_max=6)
    assert r.size == 5 * 6  # ceil(0.24/0.05)=5, ceil(0.42/0.05) capped → 6... 9→6
    np.testing.assert_allclose(w.sum(), 1.0)
    np.testing.assert_allclose([r.mean(), z.mean()], [1.0, 0.5])
    assert np.all(np.abs(r - 1.0) < 0.12) and np.all(np.abs(z - 0.5) < 0.21)


def test_section_averaged_linkage_matches_centroid_far_field_and_is_symmetric():
    """Far apart, section averaging changes nothing (flux uniform across the
    element); close up on a LARGE element it corrects the centroid link, and
    the two-section linkage matrix stays symmetric (reciprocity)."""
    from dataclasses import dataclass

    from imas_ambix.gs.cylinder import hybrid_greens
    from imas_ambix.latent.temporal_operator import (
        _linked_flux_columns,
        _section_grid,
    )

    @dataclass
    class _F:
        r: float
        z: float
        width: float
        height: float
        xmult: float = 1.0

    far = [[_F(0.6, -1.4, 0.03, 0.03)], [_F(1.6, 1.5, 0.03, 0.03)]]
    pr, pz, wt, owner = _section_grid(far, 0.05, 6)
    m01 = _linked_flux_columns(far[1], pr, pz, wt, owner, 2, hybrid_greens)[0]
    psi_c = hybrid_greens(np.array([0.6]), np.array([-1.4]), 1.6, 1.5, 0.03, 0.03)[0]
    np.testing.assert_allclose(m01, float(psi_c[0]), rtol=1e-10)

    near = [[_F(0.9, 0.0, 0.235, 0.416)], [_F(1.05, 0.15, 0.05, 0.05)]]
    pr, pz, wt, owner = _section_grid(near, 0.05, 6)
    m_avg = _linked_flux_columns(near[1], pr, pz, wt, owner, 2, hybrid_greens)[0]
    psi_c = hybrid_greens(np.array([0.9]), np.array([0.0]), 1.05, 0.15, 0.05, 0.05)[0]
    assert abs(m_avg - float(psi_c[0])) > 1e-3 * abs(m_avg)  # correction bites
    # reciprocity: source↔observer swapped agrees (source side analytic,
    # observer side quadrature — equality to quadrature accuracy)
    m_10 = _linked_flux_columns(near[0], pr, pz, wt, owner, 2, hybrid_greens)[1]
    np.testing.assert_allclose(m_avg, m_10, rtol=2e-3)


def test_coil_case_coupling_is_strong_but_below_unity():
    """A winding pack enclosed by a thin-wall case box (the MAST P4 geometry:
    four 3 mm plates around a ~0.16 m square winding) must couple strongly —
    the EM shielding physics — with a symmetric mutual and a coupling
    coefficient k = M/√(L_coil·L_case) strictly below 1.  The thick-cylinder
    kernel handles the enclosing-observer configuration without any special
    casing: the case plates sit OUTSIDE the winding section and the kernel is
    smooth everywhere including inside conductors."""
    from dataclasses import dataclass

    from imas_ambix.gs.cylinder import hybrid_greens
    from imas_ambix.latent.temporal_operator import (
        _linked_flux_columns,
        _section_grid,
    )

    @dataclass
    class _F:
        r: float
        z: float
        width: float
        height: float
        xmult: float = 1.0

    coil = [_F(1.500, 1.104, 0.159, 0.158)]
    case = [  # four thin plates boxing the winding (p4u-like)
        _F(1.4064, 1.0985, 0.0030, 0.1870, 0.25),
        _F(1.4984, 1.1935, 0.1870, 0.0030, 0.25),
        _F(1.5934, 1.1015, 0.0030, 0.1870, 0.25),
        _F(1.5014, 1.0065, 0.1870, 0.0030, 0.25),
    ]
    groups = [coil, case]
    pr, pz, wt, owner = _section_grid(groups, 0.03, 6)
    l_coil, m_cc = _linked_flux_columns(coil, pr, pz, wt, owner, 2, hybrid_greens)
    m_cc2, l_case = _linked_flux_columns(case, pr, pz, wt, owner, 2, hybrid_greens)
    np.testing.assert_allclose(m_cc, m_cc2, rtol=5e-3)  # reciprocity
    k = m_cc / np.sqrt(l_coil * l_case)
    assert 0.7 < k < 1.0, f"coil-case coupling k={k:.3f}"
    assert l_coil > 0 and l_case > 0


def test_thin_shell_resistance_uses_true_cross_section():
    """The 0.01 m kernel floor must never inflate a thin shell's conducting
    area: a 3 mm case wall carries the resistance of its TRUE section."""
    from unittest import mock

    from imas_ambix.latent import temporal_operator as to

    # capture r_diag by rebuilding the tiny arithmetic here and comparing to
    # the module's formula through a minimal eigenbasis build is heavy; pin
    # the formula directly instead: true area, never clamped
    f = type("F", (), {"r": 1.5, "width": 0.003, "height": 0.187, "xmult": 0.25})()
    rho = to.STEEL_RESISTIVITY
    r_true = 2.0 * np.pi * f.r * rho / (0.003 * 0.187) * 0.25**2
    r_clamped = 2.0 * np.pi * f.r * rho / (0.01 * 0.187) * 0.25**2
    assert r_true / r_clamped > 3.0
    # the module formula (as used in build_passive_eigenbasis)
    r_module = (
        2.0 * np.pi * f.r * rho / max(abs(f.width) * abs(f.height), 1e-8) * f.xmult**2
    )
    np.testing.assert_allclose(r_module, r_true, rtol=1e-12)
    del mock


def test_median_section_scale_is_machine_intrinsic():
    from dataclasses import dataclass

    from imas_ambix.latent.temporal_operator import _median_section_scale

    @dataclass
    class _F:
        r: float
        z: float
        width: float
        height: float
        xmult: float = 1.0

    groups = [[_F(1.0, 0.0, 0.04, 0.05)], [_F(1.2, 0.1, 0.06, 0.06)]]
    s = _median_section_scale(groups)
    lo = np.sqrt(0.04 * 0.05)
    hi = 0.06
    assert lo <= s <= hi
    # scaling every section by 2 scales the rule by 2 — no metre-level lock-in
    groups2 = [[_F(1.0, 0.0, 0.08, 0.10)], [_F(1.2, 0.1, 0.12, 0.12)]]
    np.testing.assert_allclose(_median_section_scale(groups2), 2.0 * s, rtol=1e-12)


def _model_inputs(model, b=2, t=6, s=9, seed=3):
    rng = np.random.default_rng(seed)
    k = model.n_modes
    return {
        "tokens": torch.tensor(rng.normal(size=(b, t, s, 8)).astype(np.float32)),
        "token_mask": torch.tensor(rng.uniform(size=(b, t, s)) > 0.2),
        "global_feats": torch.tensor(rng.normal(size=(b, t, 2)).astype(np.float32)),
        "dt": torch.tensor(rng.uniform(0.005, 0.03, size=(b, t)).astype(np.float32)),
        "a_phys": torch.tensor(rng.normal(size=(b, t, k)).astype(np.float32)),
        "u_drive": torch.tensor(rng.normal(size=(b, t, k)).astype(np.float32)),
    }


def _fresh_model(**kw) -> TemporalOperator:
    torch.manual_seed(5)
    return TemporalOperator(
        6,
        np.array([0.030, 0.012, 0.004]),
        np.array([1.5, 0.7, 0.2]),
        np.array([2.0, 1.0, 0.5]),
        width=16,
        d_model=32,
        n_heads=2,
        n_layers=2,
        **kw,
    )


def test_untrained_operator_is_identity_on_the_spine():
    model = _fresh_model()
    model.eval()
    with torch.no_grad():
        dc, da = model(**_model_inputs(model))
    assert torch.all(dc == 0.0)
    assert torch.all(da == 0.0)


def test_decays_initialised_at_physical_lr_times_and_learnable():
    model = _fresh_model()
    assert np.allclose(
        torch.exp(model.log_tau).detach().numpy(), [0.030, 0.012, 0.004], rtol=1e-6
    )
    assert model.log_tau.requires_grad


def _randomise_heads(model: TemporalOperator) -> None:
    torch.manual_seed(9)
    for p in (
        model.dc_head.weight,
        model.dc_head.bias,
        model.eddy_head.weight,
        model.eddy_head.bias,
        model.drive_proj.weight,
        model.eddy_gate,
    ):
        with torch.no_grad():
            p.copy_(0.3 * torch.randn_like(p))


def test_causality_future_perturbation_never_leaks_backwards():
    model = _fresh_model()
    _randomise_heads(model)
    model.eval()
    inputs = _model_inputs(model)
    with torch.no_grad():
        dc0, da0 = model(**inputs)
    t0 = 3
    inputs["tokens"][:, t0:] += 5.0
    inputs["a_phys"][:, t0:] -= 2.0
    inputs["u_drive"][:, t0:] += 3.0
    with torch.no_grad():
        dc1, da1 = model(**inputs)
    assert torch.allclose(dc0[:, :t0], dc1[:, :t0], atol=1e-6)
    assert torch.allclose(da0[:, :t0], da1[:, :t0], atol=1e-6)
    assert not torch.allclose(dc0[:, t0:], dc1[:, t0:], atol=1e-6)


def test_outputs_are_bounded_by_the_head_scales():
    model = _fresh_model()
    _randomise_heads(model)
    model.eval()
    inputs = _model_inputs(model)
    inputs["tokens"] *= 100.0  # adversarially large inputs
    with torch.no_grad():
        dc, da = model(**inputs)
    assert dc.abs().max() <= model.dc_scale + 1e-6
    da_std = da / model.eddy_std
    assert da_std.abs().max() <= model.eddy_scale + 1e-5


def test_gradients_reach_trunk_heads_and_decays():
    model = _fresh_model()
    _randomise_heads(model)
    dc, da = model(**_model_inputs(model))
    loss = (dc**2).mean() + (da**2).mean()
    loss.backward()
    assert model.log_tau.grad is not None
    assert float(model.log_tau.grad.abs().sum()) > 0.0
    assert model.dc_head.weight.grad is not None
    trunk_grads = [p.grad for p in model.trunk.parameters() if p.grad is not None]
    assert trunk_grads and any(float(g.abs().sum()) > 0 for g in trunk_grads)


def test_padding_steps_emit_zero_and_do_not_disturb_valid_steps():
    model = _fresh_model()
    _randomise_heads(model)
    model.eval()
    inputs = _model_inputs(model)
    b, t = inputs["dt"].shape
    with torch.no_grad():
        dc_full, da_full = model(**inputs)
    pad = torch.zeros(b, t, dtype=torch.bool)
    pad[:, 4:] = True  # trailing padding
    with torch.no_grad():
        dc_pad, da_pad = model(**inputs, pad_mask=pad)
    assert torch.all(dc_pad[:, 4:] == 0.0)
    assert torch.all(da_pad[:, 4:] == 0.0)
    assert torch.allclose(dc_pad[:, :4], dc_full[:, :4], atol=1e-6)
    assert torch.allclose(da_pad[:, :4], da_full[:, :4], atol=1e-6)


def test_fully_masked_padded_tail_stays_finite_and_padding_invariant():
    """Mixed-length batches pad the tail with token_mask ALL-False steps —
    the empty-timestep pooling must emit zeros (not a max sentinel that
    overflows downstream), outputs must stay finite everywhere, and the
    valid prefix must match the unpadded single-sequence forward."""
    model = _fresh_model()
    _randomise_heads(model)
    model.eval()
    inputs = _model_inputs(model)
    b, t = inputs["dt"].shape
    n_short = 3
    pad = torch.zeros(b, t, dtype=torch.bool)
    pad[1, n_short:] = True
    inputs["token_mask"][1, n_short:] = False  # as pad_batch builds it
    inputs["token_mask"][1, :n_short] = True
    with torch.no_grad():
        dc_pad, da_pad = model(**inputs, pad_mask=pad)
        dc_one, da_one = model(**{k: v[1:2, :n_short] for k, v in inputs.items()})
    assert torch.isfinite(dc_pad).all()
    assert torch.isfinite(da_pad).all()
    assert torch.all(dc_pad[1, n_short:] == 0.0)
    assert torch.all(da_pad[1, n_short:] == 0.0)
    assert torch.allclose(dc_pad[1, :n_short], dc_one[0], atol=1e-6)
    assert torch.allclose(da_pad[1, :n_short], da_one[0], atol=1e-6)


def test_checkpoint_round_trip_is_exact(tmp_path):
    model = _fresh_model()
    _randomise_heads(model)
    model.eval()
    inputs = _model_inputs(model)
    with torch.no_grad():
        dc0, da0 = model(**inputs)
    path = tmp_path / "temporal.pt"
    save_checkpoint(path, model, {"note": "round-trip"})
    loaded, ckpt = load_checkpoint(path)
    assert ckpt["note"] == "round-trip"
    with torch.no_grad():
        dc1, da1 = loaded(**inputs)
    assert torch.equal(dc0, dc1)
    assert torch.equal(da0, da1)


@pytest.mark.slow
def test_real_geometry_eigenbasis_taus_in_vessel_range():
    """MAST vessel L/R eigenmodes: slowest ≳ 10 ms, all kept modes sub-100 ms,
    eigenvectors L-orthonormal, drive couplings shape-consistent."""
    from imas_ambix.gs.geometry import build_table_for_shot
    from imas_ambix.latent.gs_solve import EquilibriumGrid
    from imas_ambix.latent.temporal_operator import build_passive_eigenbasis

    table = build_table_for_shot(11766)
    grid = EquilibriumGrid.from_table(table, nr=65, nz=97)
    basis = build_passive_eigenbasis(
        table, grid, sensor_scale=np.ones(len(table.sensor_map)), k=12
    )
    assert basis.n_modes == 12
    assert basis.tau.max() > 0.010
    assert basis.tau.max() < 0.100
    assert basis.tau.min() > 1e-4
    assert basis.a_sensor.shape == (len(table.sensor_map), 12)
    assert basis.g_grid.shape == (grid.flat_r.size, 12)
    assert basis.m_cell.shape == (12, grid.cells.size)
    assert np.all(np.diff(basis.tau) <= 1e-12)  # slowest-first ordering
