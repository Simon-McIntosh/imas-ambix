"""Pins for the plasma-driven vessel-eddy drive and the one-matrix solve.

All on an analytic single-ring fixture (no shot data):

1. **Lenz sign** — a rising positive plasma current induces an anti-parallel
   (negative) vessel-ring current whose vertical field at the plasma axis is
   CONFINING (same sign as the equilibrium requirement for +Ip), and the
   quasi-steady ramp amplitude matches the closed form ``−M·İp/R``.
2. **Exact transient decay** — into a current hold the eddy decays exactly
   ``e^{−Δt/τ}`` (the exact-ZOH contract): ``e^{−1} ≈ 37%`` of the ramp peak
   one time-constant in, below 10% only past ``2.303·τ`` — the drive is a
   ramp transient, gone on the vessel L/R times.
3. **Solved loop voltage** — the one-matrix pinned-current solve reads the
   plasma row's balance out as the applied voltage: with negligible vessel
   coupling and constant geometry it reduces to ``u = L_p·İp + R_p·Ip``.
4. **dL/dt is a flux-balance term** — at CONSTANT plasma current a growing
   minor radius changes the solved voltage by exactly ``Ip·dL_p/dt``: the
   evolving shape is as much a drive as the moving centroid.
5. **Moving centroid drives the vessel** — at constant Ip a centroid drifting
   toward the ring raises the linked flux and induces a Lenz-anti-parallel
   ring current, with no coil swing anywhere.
6. **Self-inductance pin** — ``L_p = μ0·R·(ln(8R/(a√κ)) − 2 + li/2)`` on a
   hand case, with the κ correction exactly ``−μ0·R·½ln κ``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from imas_ambix.gs import force_balance as fb
from imas_ambix.gs import geometry as gsg
from imas_ambix.gs import operator as op
from imas_ambix.latent.plasma_screening import (
    plasma_self_inductance,
    solve_pinned_plasma_circuit,
)
from imas_ambix.latent.temporal_operator import (
    PassiveCircuitSystem,
    predict_vessel_currents,
)

AXIS = (0.9, 0.0)
RING_R, RING_Z = 2.0, 0.0
L_RING = 1.2e-5  # ring self-inductance [H]
TAU_RING = 0.02  # ring L/R time [s]
R_RING = L_RING / TAU_RING


def _ring_table(ring_r: float = RING_R, ring_z: float = RING_Z) -> gsg.GeometryTable:
    """One passive toroidal ring; no coils, no sensors."""
    pf = [
        gsg.PFFilament(
            r=ring_r, z=ring_z, turns=1.0, width=0.04, height=0.04, circuit=2, xmult=1.0
        )
    ]
    sig = gsg.SetupSignature(
        n_bprobe=0,
        n_fluxloop=0,
        n_pf_filament=1,
        n_limiter=4,
        digest="deadbeef00000000",
    )
    return gsg.GeometryTable(
        signature=sig,
        shots=[1],
        b_probes=[],
        flux_loops=[],
        pf_filaments=pf,
        limiter_r=[0.3, 1.6, 1.6, 0.3],
        limiter_z=[-1.0, -1.0, 1.0, 1.0],
        sensor_map=[],
        passive_structures=[],
        amc_current_channels=[],
        unmatched_amb=[],
    )


def _ring_system() -> PassiveCircuitSystem:
    """The ring as a hand-built single-circuit L/R system (exact values)."""
    return PassiveCircuitSystem(
        circuits=np.array([2], dtype=np.int64),
        centroid_r=np.array([RING_R]),
        centroid_z=np.array([RING_Z]),
        lmat=np.array([[L_RING]]),
        r_diag=np.array([R_RING]),
        a_circ=np.zeros((0, 1)),
        g_circ=np.zeros((1, 1)),
        m_coil_circ=np.zeros((1, 0)),
        coil_channels=[],
        case_channel_row={},
        resistivity=7.2e-7,
    )


def _ramp_hold(dt: float = 5.0e-4, t_ramp: float = 0.2, t_hold: float = 0.3):
    """Linear Ip ramp 0 → 500 kA over ``t_ramp``, then hold."""
    times = np.arange(0.0, t_ramp + t_hold + dt, dt)
    ip = np.where(times < t_ramp, times / t_ramp, 1.0) * 5.0e5
    return times, ip, t_ramp


def _drive(table, vsys, times, ip):
    axis_rz = np.tile(np.asarray(AXIS), (times.size, 1))
    return predict_vessel_currents(
        table,
        vsys,
        np.zeros((times.size, 0)),
        [],
        times,
        ip_amperes=ip,
        axis_rz=axis_rz,
    )


def test_plasma_ramp_induces_lenz_confining_vessel_current():
    table = _ring_table()
    vsys = _ring_system()
    times, ip, t_ramp = _ramp_hold()
    i_coil, i_full = _drive(table, vsys, times, ip)
    assert np.all(i_coil == 0.0)  # no coil drive → coil-only state is zero

    ramp = (times > 0.01) & (times < t_ramp)
    assert np.all(i_full[ramp, 0] < 0.0)  # anti-parallel image current

    # quasi-steady amplitude: L di/dt + R i = −M·İp  →  i → −M·İp/R
    m = float(op.greens_psi(np.array([RING_R]), np.array([RING_Z]), *AXIS)[0])
    ip_dot = 5.0e5 / t_ramp
    k_end = int(np.searchsorted(times, t_ramp))
    expected = -m * ip_dot / R_RING * (1.0 - np.exp(-times[k_end] / TAU_RING))
    assert i_full[k_end, 0] == pytest.approx(expected, rel=1e-6)

    # the eddy field at the axis is CONFINING: same sign as the requirement
    cols = fb.passive_circuit_bz(
        table, vsys.circuits, np.array([AXIS[0]]), np.array([AXIS[1]])
    )
    bz_eddy = float(cols[0] @ i_full[k_end])
    assert bz_eddy < 0.0  # negative B_z confines a positive plasma current


def test_vessel_eddy_hold_decay_is_exact_single_mode():
    table = _ring_table()
    vsys = _ring_system()
    times, ip, t_ramp = _ramp_hold()
    _ic, i_full = _drive(table, vsys, times, ip)

    k0 = int(np.searchsorted(times, t_ramp))
    peak = float(np.abs(i_full[: k0 + 1, 0]).max())
    i0 = float(i_full[k0, 0])
    for delta in (0.005, 0.02, 0.04):
        k = int(np.searchsorted(times, t_ramp + delta))
        assert i_full[k, 0] == pytest.approx(
            i0 * np.exp(-(times[k] - times[k0]) / TAU_RING), rel=1e-9
        )
    # one τ into the hold the transient is e⁻¹ ≈ 37% of its ramp peak — an
    # L/R system can NOT be below 10% at one τ; the 10% line is crossed at
    # 2.303·τ (both pinned, so the transient claim is exact, not hopeful)
    k_tau = int(np.searchsorted(times, t_ramp + TAU_RING))
    assert abs(i_full[k_tau, 0]) / peak == pytest.approx(np.exp(-1.0), abs=5e-3)
    k_10 = int(np.searchsorted(times, t_ramp + 2.4 * TAU_RING))
    assert abs(i_full[k_10, 0]) / peak < 0.10


def _pinned_solve(table, vsys, times, ip, axis_rz, a_minor, r_p=3.0e-6):
    return solve_pinned_plasma_circuit(
        table,
        vsys,
        np.zeros((times.size, 0)),
        [],
        times,
        ip_amperes=ip,
        axis_rz=axis_rz,
        minor_radius=a_minor,
        elongation=1.0,
        internal_inductance=0.8,
        plasma_resistance_ohm=r_p,
    )


def test_solved_loop_voltage_reduces_to_inductive_plus_resistive():
    # ring far enough that the plasma↔vessel coupling is negligible: on a
    # linear ramp with constant geometry, u = L_p·İp + R_p·Ip exactly
    table = _ring_table(ring_r=30.0)
    vsys = _ring_system()
    times, ip, t_ramp = _ramp_hold(dt=1.0e-3)
    axis_rz = np.tile(np.asarray(AXIS), (times.size, 1))
    r_p = 3.0e-6
    sol = _pinned_solve(table, vsys, times, ip, axis_rz, 0.55, r_p=r_p)
    l_p = plasma_self_inductance(AXIS[0], 0.55, 1.0, 0.8)
    ip_dot = 5.0e5 / t_ramp
    ramp_interior = (times > 0.02) & (times < t_ramp - 0.02)
    expected = l_p * ip_dot + r_p * ip[ramp_interior]
    # rtol floor: the distant ring's back-reaction is finite (~7e-5 of u
    # while its eddy transient builds) — the point is the L·İp + R·Ip shape
    assert np.allclose(sol.u_loop[ramp_interior], expected, rtol=3e-4)
    hold_interior = times > t_ramp + 0.1
    assert np.allclose(sol.u_loop[hold_interior], r_p * ip[hold_interior], rtol=3e-4)


def test_dl_dt_enters_the_solved_voltage():
    # CONSTANT current, growing minor radius: the entire non-resistive
    # voltage is Ip·dL_p/dt = −μ0·R·(ȧ/a)·Ip — the shape change is a drive
    table = _ring_table(ring_r=30.0)
    vsys = _ring_system()
    dt = 1.0e-3
    times = np.arange(0.0, 0.2 + dt, dt)
    ip = np.full(times.size, 5.0e5)
    axis_rz = np.tile(np.asarray(AXIS), (times.size, 1))
    a_t = 0.3 + (0.6 - 0.3) * times / times[-1]
    r_p = 3.0e-6
    sol = _pinned_solve(table, vsys, times, ip, axis_rz, a_t, r_p=r_p)
    a_dot = (0.6 - 0.3) / times[-1]
    mu0 = 4.0e-7 * np.pi
    expected = -mu0 * AXIS[0] * (a_dot / a_t) * ip
    interior = slice(5, -5)
    assert np.allclose(
        sol.u_loop[interior] - r_p * ip[interior], expected[interior], rtol=1e-2
    )


def test_moving_centroid_drives_lenz_vessel_current():
    # constant Ip, centroid drifting toward the ring: the ring's linked flux
    # rises and Lenz demands an anti-parallel induced current — no coil
    # swing anywhere in the problem
    table = _ring_table()
    vsys = _ring_system()
    dt = 5.0e-4
    times = np.arange(0.0, 0.2 + dt, dt)
    ip = np.full(times.size, 5.0e5)
    ip[0] = 0.0  # quiescent start
    r_trace = 0.85 + (0.95 - 0.85) * times / times[-1]
    axis_rz = np.column_stack([r_trace, np.zeros_like(times)])
    sol = _pinned_solve(table, vsys, times, ip, axis_rz, 0.4)
    late = times > 0.05
    assert np.all(sol.i_vessel[late, 0] < 0.0)
    assert np.all(np.isfinite(sol.u_loop[1:]))


def test_plasma_self_inductance_hand_case_and_kappa_shift():
    """R = 0.9, a = 0.55, κ = 1, li = 0.8:
    L = μ0·0.9·(ln(8·0.9/0.55) − 2 + 0.4)."""
    mu0 = 4.0e-7 * np.pi
    hand = mu0 * 0.9 * (math.log(8.0 * 0.9 / 0.55) - 2.0 + 0.4)
    got = float(plasma_self_inductance(0.9, 0.55, 1.0, 0.8))
    assert got == pytest.approx(hand, rel=1e-12)
    # κ enters as −½·ln κ on the log term
    got_k = float(plasma_self_inductance(0.9, 0.55, 1.8, 0.8))
    assert got - got_k == pytest.approx(mu0 * 0.9 * 0.5 * math.log(1.8), rel=1e-10)
