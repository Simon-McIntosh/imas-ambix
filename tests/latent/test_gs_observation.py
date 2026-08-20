"""Tests for the differentiable (torch) GS observation operator + ψ-grid.

The GS observation operator wraps the geometry-only Green's-function
:class:`imas_ambix.gs.operator.ForwardOperator` into a torch module so the
hybrid latent's plasma-current amplitudes (θ, on the dimensionless polynomial
basis) map — differentiably — to predicted magnetics at the freely-known
sensor locations AND to the reconstructed poloidal-flux field ψ(R,Z) on an
arbitrary grid.  Two invariants pin it:

* the sensor prediction must agree bit-for-bit with the numpy
  :meth:`ForwardOperator.predict` (same Green's physics, torch backend);
* the ψ-field must be the superposition of the plasma-current basis
  Green's flux + the known-PF Green's flux, and must be differentiable
  w.r.t. the latent amplitudes θ (autograd), because topology is read
  from the *solved* ψ and the GS residual back-propagates through it.

No mirror / network needed — a synthetic single-coil campaign table is enough.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from imas_ambix.gs import operator as op
from imas_ambix.gs.machine_geometry import GeometryIdentity, OperatorGeometry
from imas_ambix.gs.residual import plasma_poly_basis
from imas_ambix.latent.gs_observation import GSObservation


def _synthetic_table() -> OperatorGeometry:
    """Minimal campaign: 1 vertical + 1 radial probe, 1 flux loop, 1 KNOWN PF
    coil (P4U-like) + 1 passive circuit."""
    bp_v = SimpleNamespace(index=0, r=1.5, z=0.0, angle_deg=-90.0, length=0.025)
    bp_r = SimpleNamespace(index=1, r=1.5, z=0.0, angle_deg=0.0, length=0.025)
    fl = SimpleNamespace(index=0, r=1.3, z=0.5)
    pf_known = [
        SimpleNamespace(
            r=1.50, z=1.10, turns=1.0, width=0.01, height=0.01, circuit=1, xmult=0.5
        ),
        SimpleNamespace(
            r=1.50, z=1.10, turns=1.0, width=0.01, height=0.01, circuit=1, xmult=0.5
        ),
    ]
    pf_passive = [
        SimpleNamespace(
            r=2.0, z=0.0, turns=1.0, width=0.01, height=0.01, circuit=2, xmult=1.0
        ),
    ]
    identity = GeometryIdentity(
        representation_key="mp2-fl1-fc3-lim4-deadbeef00000000",
        representation_digest="deadbeef00000000",
        derivation_id="synthetic-observation",
        physical_digest="",
        registry_digest="",
    )
    sensor_map = [
        SimpleNamespace(
            amb_channel="obv01",
            kind="b_probe",
            efm_index=0,
            r=1.5,
            z=0.0,
            angle_deg=-90.0,
            residual_m=0.001,
            flag="",
        ),
        SimpleNamespace(
            amb_channel="obr01",
            kind="b_probe",
            efm_index=1,
            r=1.5,
            z=0.0,
            angle_deg=0.0,
            residual_m=0.001,
            flag="",
        ),
        SimpleNamespace(
            amb_channel="fl_p4u_1",
            kind="flux_loop",
            efm_index=0,
            r=1.3,
            z=0.5,
            angle_deg=None,
            residual_m=0.001,
            flag="",
        ),
    ]
    return OperatorGeometry(
        identity=identity,
        probes=(bp_v, bp_r),
        loops=(fl,),
        conductors=tuple(pf_known + pf_passive),
        passives=(SimpleNamespace(name="wall_a", r=2.0, z=0.0, obsolete=False),),
        limiter_r=(0.3, 1.6, 1.6, 0.3),
        limiter_z=(-1.0, -1.0, 1.0, 1.0),
        polygon_sections=(),
        drive_map=(),
        sensor_map=tuple(sensor_map),
        unmatched_channels=(),
        active_circuits=(),
        available_current_channels=("p4u_coil_current", "plasma_current"),
        r0=0.85,
        minor_radius=0.65,
        unresolved_turns={},
        coil_channels=(),
        coil_column_matrix=np.zeros((len(sensor_map), 0), dtype=np.float64),
    )


def test_sensor_prediction_matches_numpy_forward_operator():
    """Torch GS observation must reproduce ForwardOperator.predict exactly."""
    table = _synthetic_table()
    fwd = op.build_operator(table)
    order = 1
    basis = plasma_poly_basis(fwd.plasma_rz, order, fwd.r0, fwd.minor_radius)
    n_dof = basis.shape[1]

    obs = GSObservation.from_table(table, grid_nr=8, grid_nz=10, profile_order=order)

    theta = np.linspace(-2.0, 3.0, n_dof)
    i_pf = np.array([1234.0])  # one KNOWN coil, amperes
    c_plasma = basis @ theta

    want = fwd.predict(i_pf, c_plasma=c_plasma)  # numpy, (n_sensor,)
    got = obs(
        torch.tensor(theta, dtype=torch.float64).unsqueeze(0),
        torch.tensor(i_pf, dtype=torch.float64).unsqueeze(0),
    )
    assert got.shape == (1, len(fwd.sensor_channels))
    np.testing.assert_allclose(got.squeeze(0).numpy(), want, rtol=1e-9, atol=1e-12)


def test_psi_field_is_greens_superposition_of_plasma_and_pf():
    """ψ on the grid = Σ_node greens_psi·c_plasma + Σ_coil greens_psi·i_pf."""
    table = _synthetic_table()
    fwd = op.build_operator(table)
    order = 1
    basis = plasma_poly_basis(fwd.plasma_rz, order, fwd.r0, fwd.minor_radius)
    n_dof = basis.shape[1]
    obs = GSObservation.from_table(table, grid_nr=6, grid_nz=7, profile_order=order)

    theta = np.linspace(0.5, -1.5, n_dof)
    i_pf = np.array([2000.0])
    c_plasma = basis @ theta

    # reference ψ on the module's own grid, computed independently from greens_psi
    gr = obs.grid_r.numpy()
    gz = obs.grid_z.numpy()
    ref = np.zeros(gr.shape, dtype=np.float64)
    for (nr, nz), c in zip(fwd.plasma_rz, c_plasma, strict=True):
        ref += c * op.greens_psi(gr, gz, float(nr), float(nz))
    # PF contribution: sum over the coil's filaments weighted by xmult, × i_pf
    for f in table.conductors:
        if f.circuit == 1:  # the KNOWN P4U coil
            ref += i_pf[0] * f.xmult * op.greens_psi(gr, gz, f.r, f.z)

    got = obs.psi_field(
        torch.tensor(theta, dtype=torch.float64).unsqueeze(0),
        torch.tensor(i_pf, dtype=torch.float64).unsqueeze(0),
    )
    assert got.shape == (1, gr.shape[0])
    np.testing.assert_allclose(got.squeeze(0).numpy(), ref, rtol=1e-9, atol=1e-12)


def test_psi_field_is_differentiable_wrt_theta():
    """Autograd must flow through the ψ readout back to the latent amplitudes."""
    table = _synthetic_table()
    fwd = op.build_operator(table)
    order = 1
    n_dof = plasma_poly_basis(fwd.plasma_rz, order, fwd.r0, fwd.minor_radius).shape[1]
    obs = GSObservation.from_table(table, grid_nr=5, grid_nz=5, profile_order=order)

    theta = torch.zeros(1, n_dof, dtype=torch.float64, requires_grad=True)
    i_pf = torch.tensor([[500.0]], dtype=torch.float64)
    psi = obs.psi_field(theta, i_pf)
    loss = (psi**2).sum()
    loss.backward()
    assert theta.grad is not None
    assert torch.isfinite(theta.grad).all()
    assert theta.grad.abs().sum() > 0  # ψ genuinely depends on θ


def test_reconstruction_grid_avoids_source_singularities():
    """A grid point coincident with a plasma/coil source must be nudged away.

    A field/current-carrying element's own Green's function diverges at zero
    distance (it is a point-filament model); a reconstruction grid built by a
    naive linspace can land arbitrarily close to a real coil or plasma node
    (confirmed on real MAST geometry: central-solenoid coils sit within 1-4 cm
    of a modest-resolution grid), producing a spurious near-singular ψ spike
    that corrupts anything computed from the reconstructed field (the
    transport prior's midplane profile, topology). The grid must keep at least
    ``min_source_distance`` from every plasma-basis node and PF filament.
    """
    table = _synthetic_table()
    # a synthetic table where a grid point WOULD exactly coincide with the
    # PF coil filament (r=1.50, z=1.10) at some (grid_nr, grid_nz) — force a
    # small grid so a coincidence is easy to construct deliberately: request a
    # grid whose node lattice includes (1.50, 1.10) by construction.
    min_d = 0.03
    obs = GSObservation.from_table(
        table, grid_nr=9, grid_nz=13, profile_order=1, min_source_distance=min_d
    )
    gr = obs.grid_r.numpy()
    gz = obs.grid_z.numpy()
    coil_r, coil_z = 1.50, 1.10
    dist_to_coil = np.hypot(gr - coil_r, gz - coil_z)
    assert dist_to_coil.min() >= min_d - 1e-9

    # and the reconstructed psi must be finite everywhere (no singularity blew up)
    theta = torch.zeros(1, obs.n_dof, dtype=torch.float64)
    i_pf = torch.tensor([[5000.0]], dtype=torch.float64)
    psi = obs.psi_field(theta, i_pf)
    assert torch.isfinite(psi).all()
