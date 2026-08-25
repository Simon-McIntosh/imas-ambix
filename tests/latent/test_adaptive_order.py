"""Adaptive harmonic order via held-out-sensor CV (the overfit guard).

The source-free harmonic fit overfits small / weakly-constrained plasmas: in-fit
misfit falls with order while the reconstruction grows high-field-side ripple.
Held-out-sensor CV catches it (CV misfit blows up on the overfit order).  These
tests pin that select_order_cv keeps resolution when the sensors support it and
steps down when they don't.
"""

from __future__ import annotations

import numpy as np

from imas_ambix.latent.boundary_harmonic import (
    HarmonicFitConfig,
    harmonic_labels,
    harmonic_sensor_matrix,
    select_harmonic_terms_cv,
    select_order_cv,
)


def _sensors(n=60):
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    # a ring of sensors around the plasma
    sr = 0.9 + 0.7 * np.cos(ang)
    sz = 0.7 * np.sin(ang)
    sang = np.full(n, 90.0)
    is_flux = np.zeros(n, dtype=bool)
    is_flux[::5] = True  # a few flux loops
    return sr, sz, sang, is_flux


def test_cv_steps_down_when_high_order_cannot_generalize():
    """Selector logic: when the order-3 columns fit the kept sensors but blow up
    on held-out ones (the overfit signature seen on real small-plasma slices,
    where near-pole P-function columns have extreme dynamic range), CV steps the
    order DOWN.  Built deterministically: the order-3 columns are near-zero
    everywhere except a spike on a few sensors, so holding those out leaves the
    order-3 coefficients unconstrained → CV misfit explodes; order-2 is smooth
    and CV-stable."""
    rng = np.random.default_rng(0)
    n = 40
    x = np.linspace(-1, 1, n)
    # order 0..2 columns (5): smooth, well-conditioned polynomials
    smooth = np.vstack([np.ones(n), x, x**2, np.cos(2 * x), np.sin(2 * x)]).T
    # order-3 columns (2): spikes on a few sensors, ~0 elsewhere — ungeneralizable
    spike = np.zeros((n, 2))
    spike[[5, 17, 29], 0] = 30.0
    spike[[9, 21, 33], 1] = 30.0
    a_max = np.hstack([smooth, spike])  # 7 cols = order 3
    true = np.zeros(7)
    true[:5] = rng.standard_normal(5)  # signal is order-2 only
    scale = 0.01 * np.abs(smooth @ true[:5]).std() * np.ones(n) + 1e-6
    measured = smooth @ true[:5] + rng.standard_normal(n) * scale
    order = select_order_cv(
        a_max, measured, np.zeros(n), np.ones(n, bool), scale, orders=(1, 2, 3)
    )
    assert order <= 2  # argmin-CV rejects the ungeneralizable order-3


def test_termwise_keeps_symmetric_drops_asymmetric():
    """Symmetry-aware: a field with a real cos(2θ) (elongation) term but a
    noise-only sin(2θ) term keeps h2c and drops h2s — the point a scalar order
    cutoff (which drops both together) cannot express."""
    rng = np.random.default_rng(4)
    n = 80
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    sr = 0.9 + 0.7 * np.cos(ang)
    sz = 0.7 * np.sin(ang)
    sang = np.full(n, 90.0)
    is_flux = np.zeros(n, dtype=bool)
    cfg = HarmonicFitConfig(pole_r=0.5, order=2)
    a_max = harmonic_sensor_matrix(sr, sz, sang, is_flux, cfg)
    labels = harmonic_labels(2)  # ['h0','h1c','h1s','h2c','h2s']
    ic, isn = labels.index("h2c"), labels.index("h2s")
    true = np.zeros(a_max.shape[1])
    true[0] = 1.0
    true[ic] = 2.0  # strong genuine elongation (cos 2θ)
    clean = a_max @ true
    scale = 5e-3 * np.abs(clean).std() * np.ones(n) + 1e-9
    measured = clean + rng.standard_normal(n) * scale
    sel = select_harmonic_terms_cv(
        a_max, measured, np.zeros(n), np.ones(n, bool), scale
    )
    assert sel[ic]  # elongation kept
    assert not sel[isn]  # noise-only asymmetric mode dropped


def test_termwise_always_keeps_position_dipole_terms():
    """The n≤1 terms (h0, h1c, h1s) set the plasma's radial + VERTICAL position
    and must NEVER be dropped, even when the sensors weakly constrain the
    up-down-asymmetric h1s — otherwise the boundary mis-places vertically."""
    rng = np.random.default_rng(7)
    n = 60
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    sr = 0.9 + 0.7 * np.cos(ang)
    sz = 0.7 * np.sin(ang)
    cfg = HarmonicFitConfig(pole_r=0.5, order=3)
    a_max = harmonic_sensor_matrix(sr, sz, np.full(n, 90.0), np.zeros(n, bool), cfg)
    labels = harmonic_labels(3)
    # signal has NO vertical asymmetry (h1s truly ~0) — CV would want to drop it
    true = np.zeros(a_max.shape[1])
    true[0] = 1.0
    true[labels.index("h1c")] = 1.5
    clean = a_max @ true
    scale = 0.02 * np.abs(clean).std() * np.ones(n) + 1e-9
    measured = clean + rng.standard_normal(n) * scale
    sel = select_harmonic_terms_cv(
        a_max, measured, np.zeros(n), np.ones(n, bool), scale
    )
    for name in ("h0", "h1c", "h1s"):
        assert sel[labels.index(name)], f"{name} must be retained (position term)"


def test_cv_keeps_full_order_when_well_constrained():
    """A clean order-3 signal with low noise and many sensors is CV-stable at 3."""
    sr, sz, sang, is_flux = _sensors(n=80)
    cfg_max = HarmonicFitConfig(pole_r=0.5, order=3)
    a_max = harmonic_sensor_matrix(sr, sz, sang, is_flux, cfg_max)
    rng = np.random.default_rng(2)
    true = rng.standard_normal(a_max.shape[1])  # genuine order-3 content
    clean = a_max @ true
    scale = 1e-3 * np.abs(clean).std() * np.ones(sr.size) + 1e-9
    measured = clean + rng.standard_normal(sr.size) * scale
    vacuum = np.zeros(sr.size)
    mask = np.ones(sr.size, dtype=bool)
    order = select_order_cv(
        a_max, measured, vacuum, mask, scale, orders=(1, 2, 3), ratio_cap=4.0
    )
    assert order == 3


def test_cv_falls_back_with_too_few_sensors():
    sr, sz, sang, is_flux = _sensors(n=60)
    a_max = harmonic_sensor_matrix(sr, sz, sang, is_flux, HarmonicFitConfig(order=3))
    mask = np.zeros(sr.size, dtype=bool)
    mask[:2] = True  # fewer than cv_folds
    order = select_order_cv(
        a_max, np.ones(sr.size), np.zeros(sr.size), mask, np.ones(sr.size),
        orders=(1, 2, 3),
    )
    assert order == 1
