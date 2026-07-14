"""Analytic tests for the source-free toroidal-harmonic annulus boundary read.

No corpus / zarr is needed: the physics claims are convention-INDEPENDENT
properties of the basis, so a recall error in the literature harmonic formula
fails a test rather than shipping.

Correctness ladder (plan §4):
  T1  each basis column solves the HOMOGENEOUS Grad-Shafranov operator
      (Delta* psi = psi_RR - psi_R/R + psi_ZZ = 0) to finite-difference floor.
  T2  a field that IS a low-order harmonic combination is recovered to
      numerical precision from its noise-free sensor signature.
  T3  the exact exterior flux of current filaments placed INSIDE the pole is
      fit to sub-cm-equivalent agreement in the annulus (physical validation);
      and the exterior-regular P set BEATS the Q set on a far-reaching domain
      (pins the decaying-set choice).
  T4  coordinate round-trip is exact and the focal-ring / axis limits hold
      (pins the cosh-vs-coth transform mistake).
  T5  the full fit API (SlicePayload -> HarmonicInversion -> psi_on_grid) wires
      up correctly and honours the sensor mask.
"""

from __future__ import annotations

import numpy as np

from imas_ambix.gs.operator import greens_psi
from imas_ambix.latent.boundary_harmonic import (
    HarmonicFitConfig,
    fit_harmonic,
    gs_operator,
    harmonic_columns,
    harmonic_labels,
    harmonic_sensor_matrix,
    mask_invalid_interior,
    toroidal_coords,
)
from imas_ambix.latent.patch_inverse import SlicePayload

POLE_R = 0.9
POLE_Z = 0.0


# --- T4: coordinate transform ----------------------------------------------


def test_toroidal_roundtrip_exact():
    """Forward (eta, theta) -> (R, Z) then the module inverse returns the same
    coordinates to machine precision — catches the cosh-vs-coth confusion."""
    eta = np.array([0.35, 0.8, 1.5, 2.5])
    theta = np.array([0.3, 1.1, 2.0, -1.4])
    a = POLE_R
    denom = np.cosh(eta) - np.cos(theta)
    r = a * np.sinh(eta) / denom
    z = POLE_Z + a * np.sin(theta) / denom
    cosh_eta, _, _, theta_b, cmc = toroidal_coords(r, z, a, POLE_Z)
    np.testing.assert_allclose(cosh_eta, np.cosh(eta), atol=1e-12)
    # theta compared modulo 2 pi
    dtheta = np.angle(np.exp(1j * (theta_b - theta)))
    np.testing.assert_allclose(dtheta, 0.0, atol=1e-12)
    np.testing.assert_allclose(cmc, np.cosh(eta) - np.cos(theta), atol=1e-12)


def test_focal_ring_and_axis_limits():
    """cosh eta -> 1 on the symmetry axis (R=0) and grows toward the focal ring."""
    # symmetry axis: R -> 0
    ce_axis, *_ = toroidal_coords(np.array([1e-6]), np.array([0.3]), POLE_R, POLE_Z)
    assert abs(float(ce_axis[0]) - 1.0) < 1e-3
    # close to the focal ring (R=pole_r, Z=pole_z): cosh eta large
    ce_ring, *_ = toroidal_coords(
        np.array([POLE_R + 1e-3]), np.array([POLE_Z]), POLE_R, POLE_Z
    )
    assert float(ce_ring[0]) > 50.0


# --- T1: homogeneous GS ----------------------------------------------------


def test_columns_solve_homogeneous_gs():
    """Every harmonic column satisfies Delta* psi = 0 to the FD truncation floor
    on annulus points away from the pole — independent of the formula's origin."""
    cfg = HarmonicFitConfig(pole_r=POLE_R, pole_z=POLE_Z, order=3)
    rg = np.linspace(0.3, 1.9, 80)
    zg = np.linspace(-1.2, 1.2, 90)
    rr, zz = np.meshgrid(rg, zg)
    cols, _ = harmonic_columns(rr.ravel(), zz.ravel(), cfg)
    far = (np.hypot(rr.ravel() - POLE_R, zz.ravel() - POLE_Z) > 0.35).reshape(zz.shape)
    for k in range(cols.shape[1]):
        psi = cols[:, k].reshape(zz.shape)
        lap = gs_operator(psi, rg, zg)
        interior = far[1:-1, 1:-1] & np.isfinite(lap[1:-1, 1:-1])
        # scale the residual by the field's own second-derivative magnitude so a
        # flat column is not spuriously "passing"; require << 1.
        scale = np.nanmax(np.abs(psi)) / (rg[1] - rg[0]) ** 2 + 1e-30
        rel = np.nanmax(np.abs(lap[1:-1, 1:-1][interior])) / scale
        assert rel < 1e-3, f"column {k}: Delta* rel residual {rel:.2e}"


# --- T2 / T5: source-free exactness + full fit API -------------------------


def _synthetic_sensors(n=48, seed=0):
    """A ring of flux loops + B-probes surrounding the plasma (in the annulus)."""
    rng = np.random.default_rng(seed)
    ang = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    rad = 0.7
    sr = POLE_R + rad * np.cos(ang)
    sz = POLE_Z + rad * np.sin(ang)
    # half flux loops (angle irrelevant), half B-probes with mixed orientation
    is_flux = np.zeros(n, dtype=bool)
    is_flux[::2] = True
    sang = np.where(is_flux, 0.0, rng.uniform(0.0, 180.0, size=n))
    return sr, sz, sang, is_flux


def test_recovers_source_free_field_exactly():
    """A field that IS a harmonic combination is recovered to numerical precision
    from its noise-free sensor signature (well-posed, K < S)."""
    cfg = HarmonicFitConfig(pole_r=POLE_R, pole_z=POLE_Z, order=3)
    sr, sz, sang, is_flux = _synthetic_sensors()
    a_sens = harmonic_sensor_matrix(sr, sz, sang, is_flux, cfg)
    rng = np.random.default_rng(1)
    c_true = rng.standard_normal(a_sens.shape[1])
    measured = a_sens @ c_true  # noise-free plasma signature (vacuum = 0)
    payload = SlicePayload(
        measured=measured,
        vacuum=np.zeros_like(measured),
        mask=np.ones(measured.size, dtype=bool),
        scale=np.full(measured.size, 1.0),
        i_pf=np.zeros(3),
        ip_amperes=1.0e5,
    )
    inv = fit_harmonic((sr, sz, sang, is_flux), payload, cfg)
    assert inv.misfit < 1e-12
    # coefficients recovered up to the fit's column normalisation
    pred = a_sens @ inv.coeffs
    np.testing.assert_allclose(pred, measured, rtol=1e-6, atol=1e-9)
    assert inv.labels == harmonic_labels(3)


def test_fit_ignores_masked_rows_and_grids():
    """Corrupting masked-out rows must not move the fit; psi_on_grid has the
    (nz, nr) shape the topology read expects."""
    cfg = HarmonicFitConfig(pole_r=POLE_R, pole_z=POLE_Z, order=2)
    sr, sz, sang, is_flux = _synthetic_sensors(seed=2)
    a_sens = harmonic_sensor_matrix(sr, sz, sang, is_flux, cfg)
    rng = np.random.default_rng(3)
    c_true = rng.standard_normal(a_sens.shape[1])
    measured = a_sens @ c_true
    S = measured.size
    mask = np.ones(S, dtype=bool)
    mask[::4] = False

    def run(meas):
        p = SlicePayload(
            measured=meas,
            vacuum=np.zeros(S),
            mask=mask,
            scale=np.full(S, 1.0),
            i_pf=np.zeros(3),
            ip_amperes=1.0e5,
        )
        return fit_harmonic((sr, sz, sang, is_flux), p, cfg)

    clean = run(measured.copy())
    dirty_meas = measured.copy()
    dirty_meas[~mask] += 1e3  # garbage on untrusted rows only
    dirty = run(dirty_meas)
    np.testing.assert_allclose(clean.coeffs, dirty.coeffs, rtol=1e-6, atol=1e-9)

    rg = np.linspace(0.2, 1.9, 33)
    zg = np.linspace(-1.0, 1.0, 41)
    psi = clean.psi_on_grid(rg, zg)
    assert psi.shape == (zg.size, rg.size)
    assert np.all(np.isfinite(psi))


# --- T3: filament recovery + P beats Q -------------------------------------


def _filament_flux(r, z, fils):
    out = np.zeros_like(np.asarray(r, dtype=np.float64))
    for fr, fz, w in fils:
        out = out + w * greens_psi(
            np.asarray(r, dtype=np.float64), np.asarray(z, dtype=np.float64), fr, fz
        )
    return out


def test_filament_flux_recovered_in_annulus():
    """Current filaments INSIDE the pole have an exact exterior flux; the P-set
    harmonic fit reproduces it at OUT-OF-FIT annulus points to sub-cm-equivalent
    accuracy, and the exterior-regular P set beats Q on the far domain."""
    fils = [(0.85, 0.05, 1.0), (0.95, -0.1, 0.7), (0.80, 0.15, 0.5), (1.0, 0.0, 0.9)]
    ang = np.linspace(0.0, 2.0 * np.pi, 60, endpoint=False)
    # fit on one annulus ring, test on a DIFFERENT (further-out) ring
    r_fit = POLE_R + 0.55 * np.cos(ang)
    z_fit = POLE_Z + 0.55 * np.sin(ang)
    r_te = POLE_R + 0.7 * np.cos(ang)
    z_te = POLE_Z + 0.7 * np.sin(ang)
    b_fit = _filament_flux(r_fit, z_fit, fils)
    b_te = _filament_flux(r_te, z_te, fils)
    scale = np.sqrt(np.mean(b_te**2))

    def fit_and_test(kind, order):
        cfg = HarmonicFitConfig(pole_r=POLE_R, pole_z=POLE_Z, order=order, kind=kind)
        a_fit, _ = harmonic_columns(r_fit, z_fit, cfg)
        cn = np.linalg.norm(a_fit, axis=0)
        cn = np.where(cn > 0, cn, 1.0)
        c = np.linalg.lstsq(a_fit / cn, b_fit, rcond=None)[0] / cn
        a_te, _ = harmonic_columns(r_te, z_te, cfg)
        return np.sqrt(np.mean((a_te @ c - b_te) ** 2)) / scale

    p_err = fit_and_test("P", 5)
    q_err = fit_and_test("Q", 5)
    # P generalises to the held-out ring; Q (divergent at infinity) cannot.
    assert p_err < 1e-3, f"P-set out-of-fit error {p_err:.2e} too large"
    assert q_err > 10.0 * p_err, f"P must beat Q (P={p_err:.2e}, Q={q_err:.2e})"


# --- annulus-only validity: mask the invalid interior ----------------------


def test_mask_invalid_interior_fills_confined_plateau():
    """The near-pole disk (where the ring functions diverge) is replaced by a
    single confined-side value; the annulus is untouched.  This is what keeps
    the boundary read in the valid annulus (the ray-cast finds no crossing in
    the masked interior)."""
    rg = np.linspace(0.2, 1.9, 40)
    zg = np.linspace(-1.2, 1.2, 50)
    rr, zz = np.meshgrid(rg, zg)
    pole = (POLE_R, 0.0)
    dpole = np.hypot(rr - pole[0], zz - pole[1])
    psi = 1.0 / (dpole + 0.05)  # blows up (positive) toward the pole
    axis = (0.96, 0.0)  # near the pole, on the high (confined) side
    radius = 0.25
    masked = mask_invalid_interior(psi, rg, zg, pole[0], pole[1], radius, axis_rz=axis)
    inside = dpole < radius
    # annulus untouched
    np.testing.assert_allclose(masked[~inside], psi[~inside])
    # interior is a single value, on the confined (high) side, past the annulus
    assert np.unique(np.round(masked[inside], 6)).size == 1
    ann = psi[~inside]
    assert masked[inside].flat[0] > np.median(ann) + 3.0 * np.std(ann)
