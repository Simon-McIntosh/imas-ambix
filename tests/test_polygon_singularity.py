"""Regression coverage for the polygon kernel's symmetry-axis limit."""

from __future__ import annotations

import numpy as np

from imas_ambix.gs.polygon import polygon_greens

POLYGON = np.array([(0.85, 0.00), (0.97, 0.00), (1.05, 0.20), (0.93, 0.20)])


def test_symmetry_axis_uses_finite_physical_limit() -> None:
    """The direct kernel is finite where its cylindrical curl becomes 0/0."""
    target_r = np.zeros(3)
    target_z = np.array([-0.20, 0.00, 0.10])

    psi, br, bz = polygon_greens(target_r, target_z, POLYGON)

    values = np.stack((psi, br, bz), axis=-1)
    assert np.isfinite(values).all()
    np.testing.assert_array_equal(psi, np.zeros_like(psi))
    np.testing.assert_array_equal(br, np.zeros_like(br))
    np.testing.assert_allclose(
        bz,
        [
            5.741848350393802e-07,
            6.491858392112787e-07,
            6.589776815824526e-07,
        ],
        rtol=1e-14,
        atol=0.0,
    )


def test_symmetry_axis_limit_is_continuous() -> None:
    """The analytic axis value agrees with the off-axis kernel as R tends to zero."""
    target_z = np.array([-0.20, 0.00, 0.10])
    _, _, axis_bz = polygon_greens(np.zeros(3), target_z, POLYGON)
    _, near_br, near_bz = polygon_greens(np.full(3, 1.0e-4), target_z, POLYGON)

    np.testing.assert_allclose(near_bz, axis_bz, rtol=1e-8, atol=0.0)
    assert np.max(np.abs(near_br)) < 3.0e-11


def test_non_singular_outputs_match_numerical_reference() -> None:
    """Far, near, and on-edge paths satisfy their numerical references."""
    target_r = np.array([1.30, 0.70, 1.01])
    target_z = np.array([0.30, -0.20, 0.10])
    expected = np.array(
        [
            [
                1.6334156210630272e-06,
                1.8349239365879545e-07,
                -2.0087174900042578e-07,
            ],
            [
                9.5216398908694316e-07,
                -3.8768280026164401e-07,
                6.2212739183031727e-07,
            ],
            [
                3.2424998541903469e-06,
                -5.2648721253916966e-07,
                -1.5778225883976969e-06,
            ],
        ]
    )

    values = np.stack(polygon_greens(target_r, target_z, POLYGON), axis=-1)

    np.testing.assert_allclose(values, expected, rtol=1e-12, atol=0.0)
