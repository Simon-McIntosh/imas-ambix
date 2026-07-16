"""Staged-disc boundary read — pure-logic units.

The read's data-heavy path is validated against the firewalled referee in the
gate scripts; here we pin the self-contained pieces: the limiter radial extent,
the boundary-shift gate metric, and the sensor-array convention.
"""

from __future__ import annotations

import numpy as np

from imas_ambix.latent.boundary_disc import (
    DiscReadConfig,
    limiter_radial_extent_at_z,
    ring_shift_rms,
    sensor_signature_arrays,
)


def _circle(r0, z0, radius, n=64):
    th = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.column_stack([r0 + radius * np.cos(th), z0 + radius * np.sin(th)])


def test_limiter_extent_rectangular_vessel():
    lr = np.array([0.2, 1.8, 1.8, 0.2, 0.2])
    lz = np.array([-1.5, -1.5, 1.5, 1.5, -1.5])
    rh, rl = limiter_radial_extent_at_z(lr, lz, 0.0)
    assert abs(rh - 0.2) < 1e-9
    assert abs(rl - 1.8) < 1e-9
    # above the vessel: falls back to the polygon bounding radii
    rh2, rl2 = limiter_radial_extent_at_z(lr, lz, 5.0)
    assert rh2 <= rl2


def test_ring_shift_zero_for_identical_rings():
    ring = _circle(0.9, 0.0, 0.5)
    assert ring_shift_rms(ring, ring.copy(), (0.9, 0.0)) < 1e-12


def test_ring_shift_measures_radial_expansion():
    a = _circle(0.9, 0.0, 0.50)
    b = _circle(0.9, 0.0, 0.60)  # uniformly 10 cm larger
    shift = ring_shift_rms(a, b, (0.9, 0.0))
    assert abs(shift - 0.10) < 5e-3


def test_ring_shift_infinite_when_missing():
    ring = _circle(0.9, 0.0, 0.5)
    assert ring_shift_rms(ring, None, (0.9, 0.0)) == float("inf")
    assert ring_shift_rms(None, ring, (0.9, 0.0)) == float("inf")


def test_gate_rejects_large_shift():
    """The over-fit gate: a stage that moves the boundary by more than
    gate_shift_frac of the disc radius must be rejected."""
    cfg = DiscReadConfig()
    radius = 0.6
    small = ring_shift_rms(_circle(0.9, 0, 0.50), _circle(0.9, 0, 0.54), (0.9, 0))
    large = ring_shift_rms(_circle(0.9, 0, 0.50), _circle(0.9, 0, 0.65), (0.9, 0))
    assert small / radius < cfg.gate_shift_frac  # physical quadrupole: accepted
    assert large / radius > cfg.gate_shift_frac  # over-fit swing: rejected


def test_sensor_arrays_follow_sensor_map_order():
    class _M:
        def __init__(self, r, z, ang, kind):
            self.r, self.z, self.angle_deg, self.kind = r, z, ang, kind

    class _T:
        sensor_map = [
            _M(0.3, 0.1, 45.0, "b_probe"),
            _M(1.4, -0.2, None, "flux_loop"),
        ]

    sr, sz, sang, is_flux = sensor_signature_arrays(_T())
    assert sr.tolist() == [0.3, 1.4]
    assert sz.tolist() == [0.1, -0.2]
    assert sang.tolist() == [45.0, 0.0]  # None angle -> 0.0 (flux rows ignore it)
    assert is_flux.tolist() == [False, True]
