"""Analytic pins for the force-balance diagnostics.

These pins gate the measured-state diagnosis script: the Shafranov
vertical-field requirement is pinned on a hand-computed
filament-in-uniform-field balance, and the decay index is pinned on exact
power-law fields and on the far-field (dipole) limit of a symmetric coil
pair evaluated through the module's own filament kernel.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from imas_ambix.gs import force_balance as fb
from imas_ambix.gs import operator as op
from imas_ambix.gs.machine_geometry import GeometryIdentity, OperatorGeometry


def _filament(r, z, turns, width, height, circuit, xmult):
    return SimpleNamespace(
        r=r, z=z, turns=turns, width=width, height=height, circuit=circuit, xmult=xmult
    )


def _geometry(conductors):
    return OperatorGeometry(
        identity=GeometryIdentity("synthetic", "synthetic", "fixture", "", ""),
        probes=(),
        loops=(),
        conductors=tuple(conductors),
        passives=(),
        limiter_r=(0.3, 1.6, 1.6, 0.3),
        limiter_z=(-1.0, -1.0, 1.0, 1.0),
        polygon_sections=(),
        drive_map=(),
        sensor_map=(),
        unmatched_channels=(),
        active_circuits=(),
        available_current_channels=("p4u_coil_current", "plasma_current"),
        r0=0.85,
        minor_radius=0.65,
        unresolved_turns={},
        coil_channels=(),
        coil_column_matrix=np.zeros((0, 0)),
    )


# --- Shafranov vertical-field requirement ------------------------------


def test_shafranov_field_pins_hand_computed_balance():
    """Filament-in-uniform-field balance, hand-computed.

    Ip = 1 MA ring at R = 1.7 m, a = 0.5 m, βp + li/2 = 1.25:
    Λ = ln(8·1.7/0.5) + 1.25 − 1.5 = ln(27.2) − 0.25 = 3.0532171,
    |B_v| = (μ0/4π)·Ip/R·Λ = 1e-7·1e6/1.7·3.0532171 = 0.17960101 T,
    signed NEGATIVE for the positive current.
    """
    got = fb.shafranov_vertical_field(1.0e6, 1.7, 0.5, 1.25)
    hand = -1.0e-7 * 1.0e6 / 1.7 * (math.log(27.2) - 0.25)
    assert got == pytest.approx(hand, rel=1e-12)
    assert got == pytest.approx(-0.17960101, rel=1e-6)


def test_shafranov_field_sign_follows_current():
    """The ring force is Ip·B_z·r̂ — balance flips sign with the current."""
    pos = fb.shafranov_vertical_field(6.0e5, 0.9, 0.55, 1.0)
    neg = fb.shafranov_vertical_field(-6.0e5, 0.9, 0.55, 1.0)
    assert pos < 0.0
    assert neg == pytest.approx(-pos, rel=1e-12)


def test_shafranov_field_scales_with_pressure_term():
    """Higher βp + li/2 → deeper (more negative) requirement, linearly."""
    b1 = fb.shafranov_vertical_field(6.0e5, 0.9, 0.55, 1.0)
    b2 = fb.shafranov_vertical_field(6.0e5, 0.9, 0.55, 2.0)
    assert b2 - b1 == pytest.approx(-op.MU0 * 6.0e5 / (4.0 * np.pi * 0.9), rel=1e-12)


def test_shafranov_field_rejects_degenerate_geometry():
    assert math.isnan(fb.shafranov_vertical_field(6.0e5, 0.0, 0.5, 1.0))
    assert math.isnan(fb.shafranov_vertical_field(6.0e5, 0.9, 0.0, 1.0))


# --- elongation-corrected requirement ------------------------------------


def test_elongated_shafranov_hand_case():
    """ln(8R/(a√κ)) form, hand-computed.

    Ip = 1 MA at R = 1.7 m, a = 0.5 m, κ = 1.69 (√κ = 1.3), βp + li/2 = 1.25:
    Λ = ln(8·1.7/0.65) + 1.25 − 1.5 = ln(20.923…) − 0.25,
    |B_v| = 1e-7·1e6/1.7·Λ, signed negative.
    """
    got = fb.shafranov_vertical_field_elongated(1.0e6, 1.7, 0.5, 1.69, 1.25)
    hand = -1.0e-7 * 1.0e6 / 1.7 * (math.log(8.0 * 1.7 / (0.5 * 1.3)) - 0.25)
    assert got == pytest.approx(hand, rel=1e-12)


def test_elongated_shafranov_reduces_to_circular_at_unit_kappa():
    base = fb.shafranov_vertical_field(6.0e5, 0.9, 0.55, 1.0)
    elong = fb.shafranov_vertical_field_elongated(6.0e5, 0.9, 0.55, 1.0, 1.0)
    assert elong == pytest.approx(base, rel=1e-14)


def test_elongated_shafranov_softens_the_requirement():
    """κ > 1 → larger effective minor radius → shallower |B_v| requirement."""
    base = fb.shafranov_vertical_field(6.0e5, 0.9, 0.55, 1.0)
    elong = fb.shafranov_vertical_field_elongated(6.0e5, 0.9, 0.55, 1.8, 1.0)
    assert abs(elong) < abs(base)
    # the softening is exactly 0.5·ln κ on Λ
    lam_shift = 0.5 * math.log(1.8)
    assert base - elong == pytest.approx(
        -op.MU0 * 6.0e5 / (4.0 * np.pi * 0.9) * lam_shift, rel=1e-10
    )


def test_elongated_shafranov_rejects_degenerate_kappa():
    assert math.isnan(fb.shafranov_vertical_field_elongated(6e5, 0.9, 0.5, 0.0, 1.0))
    assert math.isnan(
        fb.shafranov_vertical_field_elongated(6e5, 0.9, 0.5, float("nan"), 1.0)
    )


# --- decay index --------------------------------------------------------


def test_decay_index_exact_on_power_law_fields():
    """B_z ∝ R^{−n0} has decay index exactly n0; uniform field exactly 0."""
    r = np.linspace(0.4, 1.4, 2001)
    for n0 in (0.0, 1.0, 1.4, 3.0):
        bz = -0.3 * r**-n0
        n = fb.decay_index(r, bz)
        inner = slice(10, -10)  # gradient edge stencils are first-order
        assert np.allclose(n[inner], n0, atol=5e-3), f"n0={n0}"


def test_decay_index_nan_at_field_reversal():
    """A sign-reversing field has no meaningful index at the null."""
    r = np.linspace(0.4, 1.4, 101)
    bz = r - 0.9  # crosses zero at R = 0.9
    n = fb.decay_index(r, bz)
    assert np.isnan(n[np.argmin(np.abs(bz))])


def test_decay_index_coil_pair_dipole_limit():
    """Far from a symmetric coil pair the midplane field is dipolar: n → 3.

    Two point loops (a = 0.1 m, z = ±0.05 m, 1 A each) evaluated through the
    module's own filament kernel at R = 10–20 a must show the equatorial
    dipole fall-off B_z ∝ R⁻³ to within the O((a/R)²) correction.
    """
    pair = [
        _filament(
            r=0.1, z=+0.05, turns=1.0, width=0.0, height=0.0, circuit=1, xmult=1.0
        ),
        _filament(
            r=0.1, z=-0.05, turns=1.0, width=0.0, height=0.0, circuit=1, xmult=1.0
        ),
    ]
    r = np.linspace(1.0, 2.0, 401)
    z = np.zeros_like(r)
    bz = fb.filament_bz(r, z, pair)
    # equatorial dipole field is ANTIPARALLEL to the moment: negative outside
    assert np.all(bz < 0.0)
    n = fb.decay_index(r, bz)
    assert np.allclose(n[20:-20], 3.0, atol=0.05)


# --- filament kernel against the textbook loop --------------------------


def test_filament_bz_matches_textbook_loop_centre():
    """A unit point loop's central field is μ0/(2a) per ampere."""
    loop = [
        _filament(r=0.5, z=0.0, turns=1.0, width=0.0, height=0.0, circuit=1, xmult=1.0)
    ]
    bz = fb.filament_bz(np.array([1e-8]), np.array([0.0]), loop)
    assert bz[0] == pytest.approx(op.MU0 / (2.0 * 0.5), rel=1e-6)


def test_filament_bz_respects_xmult_weighting():
    loop = [
        _filament(r=0.5, z=0.0, turns=1.0, width=0.0, height=0.0, circuit=1, xmult=0.25)
    ]
    bz = fb.filament_bz(np.array([1e-8]), np.array([0.0]), loop)
    assert bz[0] == pytest.approx(0.25 * op.MU0 / (2.0 * 0.5), rel=1e-6)


# --- known-coil column assembly (mirror of the operator merge) -----------


def _synthetic_table() -> OperatorGeometry:
    """One KNOWN P4U-like coil represented by two redundant circuits + one
    passive circuit — the merge/average case the operator resolves."""
    pf = [
        # circuit 1 and circuit 3 both represent the SAME physical coil,
        # each normalised to the full coil current (Σxmult = 1)
        _filament(
            r=1.50, z=1.10, turns=1.0, width=0.02, height=0.02, circuit=1, xmult=1.0
        ),
        _filament(
            r=1.50, z=1.10, turns=1.0, width=0.02, height=0.02, circuit=3, xmult=1.0
        ),
        # far structural conductor → INFERRED passive
        _filament(
            r=2.0, z=0.0, turns=1.0, width=0.01, height=0.01, circuit=2, xmult=1.0
        ),
    ]
    return _geometry(pf)


def test_known_coil_bz_merges_redundant_circuits_once():
    """Two redundant full-coil circuits must AVERAGE into one column — the
    same no-double-count rule the sensor operator applies."""
    table = _synthetic_table()
    geometry = table
    r = np.array([0.9])
    z = np.array([0.0])
    channels, cols = fb.known_coil_bz(geometry, r, z)
    assert channels == ["p4u_coil_current"]
    assert cols.shape == (1, 1)
    single = fb.filament_bz(r, z, [table.conductors[0]])
    assert cols[0, 0] == pytest.approx(single[0], rel=1e-9)


def test_known_coil_channel_order_matches_operator():
    """Column order must equal ForwardOperator.pf_amc_channels so measured
    i_pf vectors apply without permutation."""
    geometry = _synthetic_table()
    fwd = op.build_operator(geometry)
    channels, _cols = fb.known_coil_bz(geometry, np.array([0.9]), np.array([0.0]))
    assert channels == fwd.pf_amc_channels


def test_known_coil_psi_matches_operator_merge_and_order():
    """The ψ columns share the B_z assembly: same channels, same merge."""
    table = _synthetic_table()
    geometry = table
    r = np.array([0.9])
    z = np.array([0.0])
    channels, cols = fb.known_coil_psi(geometry, r, z)
    assert channels == ["p4u_coil_current"]
    single = fb.filament_psi(r, z, [table.conductors[0]])
    assert cols[0, 0] == pytest.approx(single[0], rel=1e-9)


def test_passive_circuit_bz_column_per_circuit():
    table = _synthetic_table()
    geometry = table
    cols = fb.passive_circuit_bz(
        geometry, np.array([2]), np.array([0.9]), np.array([0.0])
    )
    assert cols.shape == (1, 1)
    expected = fb.filament_bz(np.array([0.9]), np.array([0.0]), [table.conductors[2]])
    assert cols[0, 0] == pytest.approx(expected[0], rel=1e-12)


# --- waterfall grouping ---------------------------------------------------


def test_coil_group_names():
    assert fb.coil_group("sol_current") == "sol"
    assert fb.coil_group("p4u_coil_current") == "p4"
    assert fb.coil_group("p5l_coil_current") == "p5"
    assert fb.coil_group("p4u_case_current") == "case"
    assert fb.coil_group("p6u_coil_current") == "p6"
