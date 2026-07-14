"""Unit tests for the source-free toroidal-harmonic boundary gate.

These pin the two behaviours that make the harmonic gate DIFFER from the
current-moment gate -- the supplied-axis flux reference and the vacuum-annulus
consistency cross-check -- plus the scored default of the CLI, all on tiny
synthetic fields (no data / zarr, no mpmath-heavy grids).
"""

from __future__ import annotations

import numpy as np

from imas_ambix.latent.gs_solve import EquilibriumGrid
from imas_ambix.latent.topology import _bilerp
from scripts.boundary_harmonic_gate_eval import (
    annulus_consistency_rms,
    build_parser,
    consistency_rms,
    hybrid_target_harmonic,
)


def _circular_grid(r0: float = 1.0, rb: float = 0.4, nr: int = 41, nz: int = 41):
    """A small circular-limiter grid stub with no coils (source-free)."""
    rg = np.linspace(r0 - rb - 0.2, r0 + rb + 0.2, nr)
    zg = np.linspace(-(rb + 0.2), rb + 0.2, nz)
    theta = np.linspace(0.0, 2.0 * np.pi, 96, endpoint=False)
    limiter_r = r0 + rb * np.cos(theta)
    limiter_z = rb * np.sin(theta)
    return EquilibriumGrid(
        rg=rg,
        zg=zg,
        limiter_r=limiter_r,
        limiter_z=limiter_z,
        coil_psi_columns=np.zeros((rg.size * zg.size, 0)),
        r0=r0,
    )


# --- hybrid_target_harmonic reads axis_psi at the SUPPLIED axis -------------


def test_hybrid_target_reads_axis_psi_at_supplied_axis_not_field_extremum():
    """axis_psi must be the TOTAL psi bilinearly interpolated at the supplied
    carrier axis, NOT the field's own numerical extremum (the key difference
    from the moment gate).  Build psi_tot whose O-point (max) is OFFSET from
    the supplied carrier axis and assert axis_psi == bilerp at the carrier."""
    r0, rb = 1.0, 0.4
    grid = _circular_grid(r0=r0, rb=rb)
    rr, zz = grid.mesh_r, grid.mesh_z
    # O-point (field maximum) deliberately OFFSET from the carrier axis
    o_r, o_z = r0 + 0.12, 0.06
    psi_tot = -(((rr - o_r) ** 2) + (zz - o_z) ** 2)  # max at (o_r, o_z)

    carrier_axis = np.array([r0, 0.0])  # NOT the field maximum
    # mask_radius=0 / exclude_radius=0 -> no interior masking, isolating the
    # axis_psi-at-supplied-axis behaviour under test.
    _, axis_psi, _, _ = hybrid_target_harmonic(
        psi_tot, grid, carrier_axis, (r0, 0.0), 0.0, 0.0
    )

    expected = _bilerp(psi_tot, grid.rg, grid.zg, r0, 0.0)
    assert np.isclose(axis_psi, expected)
    # and it is NOT the field's own extremum (that would be ~0 at the O-point)
    assert axis_psi < psi_tot.max() - 1e-3
    # the returned axis slots echo the supplied carrier axis
    target, _, _, _ = hybrid_target_harmonic(
        psi_tot, grid, carrier_axis, (r0, 0.0), 0.0, 0.0
    )
    assert np.isclose(target[0], r0) and np.isclose(target[1], 0.0)


# --- annulus consistency RMS ------------------------------------------------


def test_consistency_rms_zero_for_constant_offset():
    """Two fields differing by only a constant offset agree perfectly after
    the mean-difference removal -> RMS ~ 0."""
    rng = np.random.default_rng(0)
    carrier = rng.normal(size=(20, 20))
    harmonic = carrier + 3.14159  # pure constant offset
    mask = np.ones_like(carrier, dtype=bool)
    val = consistency_rms(carrier, harmonic, mask, dyn_range=1.0)
    assert val is not None and val < 1e-12


def test_consistency_rms_grows_when_fields_differ():
    """A non-constant discrepancy survives the offset removal and grows with
    its amplitude."""
    rng = np.random.default_rng(1)
    carrier = rng.normal(size=(20, 20))
    mask = np.ones_like(carrier, dtype=bool)
    small = consistency_rms(
        carrier, carrier + 0.1 * rng.normal(size=carrier.shape), mask, 1.0
    )
    big = consistency_rms(
        carrier, carrier + 1.0 * rng.normal(size=carrier.shape), mask, 1.0
    )
    assert small is not None and big is not None
    assert big > small > 1e-6


def test_consistency_rms_none_on_empty_mask_or_degenerate_range():
    carrier = np.ones((5, 5))
    harmonic = np.ones((5, 5))
    empty = np.zeros((5, 5), dtype=bool)
    assert consistency_rms(carrier, harmonic, empty, 1.0) is None
    full = np.ones((5, 5), dtype=bool)
    assert consistency_rms(carrier, harmonic, full, 0.0) is None


def test_annulus_consistency_rms_zero_for_offset_fields_over_annulus():
    """End-to-end over a real grid annulus: carrier == harmonic + offset -> ~0."""
    r0, rb = 1.0, 0.4
    grid = _circular_grid(r0=r0, rb=rb)
    rr, zz = grid.mesh_r, grid.mesh_z
    carrier = -(((rr - r0) ** 2) + zz**2)  # bowl, max on axis
    harmonic = carrier + 2.0
    axis_psi = float(carrier.max())
    boundary_psi = axis_psi - 0.5 * rb**2  # some mid-level contour
    val = annulus_consistency_rms(carrier, harmonic, grid, axis_psi, boundary_psi)
    assert val is not None and val < 1e-9


# --- CLI default ------------------------------------------------------------


def test_axis_source_default_is_patch():
    """The scored default must be 'patch' (origin-controlled); 'harmonic'
    remains available as the ablation only."""
    args = build_parser().parse_args([])
    assert args.axis_source == "patch"
