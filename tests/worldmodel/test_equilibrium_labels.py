"""Tests for the equilibrium-label extractor + the geometry probe.

The label extractor is exercised on SYNTHETIC equilibrium arrays (no Zarr
store) so the angle-resampling, time-interpolation, NaN-masking, X-point
sentinel handling, units and shapes are pinned exactly.  The probe is exercised
for forward + checkpoint-IO shape on a CPU dummy.
"""

from __future__ import annotations

import numpy as np
import pytest

from imas_ambix.worldmodel.equilibrium_labels import (
    LCFS_ANGLES,
    N_LCFS_ANGLES,
    TARGET_DIM,
    TARGET_NAMES,
    XPOINT_SENTINEL,
    build_geometry_from_arrays,
    resample_lcfs_radii,
    select_primary_xpoint,
)

# ---------------------------------------------------------------------------
# Angle-resampling of the LCFS about the axis
# ---------------------------------------------------------------------------


def test_resample_lcfs_radii_circle_is_constant():
    """A circular boundary about the axis -> constant radius at every angle."""
    axis_r, axis_z = 0.9, 0.0
    radius = 0.4
    phi = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    r = axis_r + radius * np.cos(phi)
    z = axis_z + radius * np.sin(phi)
    out = resample_lcfs_radii(r, z, axis_r, axis_z, LCFS_ANGLES)
    assert out.shape == (N_LCFS_ANGLES,)
    assert np.all(np.isfinite(out))
    np.testing.assert_allclose(out, radius, atol=1e-3)


def test_resample_lcfs_radii_ellipse_axis_aligned():
    """An axis-aligned ellipse -> a/b at θ=0/π and b at θ=π/2,3π/2."""
    axis_r, axis_z = 1.0, 0.1
    a, b = 0.5, 0.3  # R-semi, Z-semi
    phi = np.linspace(0, 2 * np.pi, 256, endpoint=False)
    r = axis_r + a * np.cos(phi)
    z = axis_z + b * np.sin(phi)
    out = resample_lcfs_radii(r, z, axis_r, axis_z, LCFS_ANGLES)
    # angles are 2π k / 8: index 0 -> 0 (outboard, radius a),
    # index 2 -> π/2 (top, radius b), index 4 -> π (inboard, a), index 6 -> 3π/2 (b)
    assert out[0] == pytest.approx(a, abs=1e-2)
    assert out[2] == pytest.approx(b, abs=1e-2)
    assert out[4] == pytest.approx(a, abs=1e-2)
    assert out[6] == pytest.approx(b, abs=1e-2)


def test_resample_lcfs_radii_undefined_axis_is_nan():
    out = resample_lcfs_radii(
        np.array([1.0, 1.1, 0.9]), np.array([0.0, 0.1, -0.1]), np.nan, 0.0
    )
    assert out.shape == (N_LCFS_ANGLES,)
    assert np.all(np.isnan(out))


def test_resample_lcfs_radii_too_few_points_is_nan():
    out = resample_lcfs_radii(
        np.array([1.0, np.nan]), np.array([0.0, np.nan]), 0.9, 0.0
    )
    assert np.all(np.isnan(out))


# ---------------------------------------------------------------------------
# Primary X-point selection (lower null, sentinel handling)
# ---------------------------------------------------------------------------


def test_select_primary_xpoint_lower_null():
    """Picks the most-negative-Z real null; sentinels/NaN -> NaN."""
    # 3 slices: both real (pick lower), one sentinel, one all-NaN.
    xr = np.array(
        [
            [0.50, XPOINT_SENTINEL, np.nan],  # row 0 (upper at slice 0)
            [0.52, 0.55, np.nan],  # row 1 (lower at slice 0)
        ]
    )
    xz = np.array(
        [
            [1.0, XPOINT_SENTINEL, np.nan],  # +Z upper
            [-1.0, -0.9, np.nan],  # -Z lower
        ]
    )
    r, z = select_primary_xpoint(xr, xz)
    assert r.shape == (3,)
    # slice 0: lower null is row 1 (z=-1.0)
    assert r[0] == pytest.approx(0.52)
    assert z[0] == pytest.approx(-1.0)
    # slice 1: only row 1 real
    assert r[1] == pytest.approx(0.55)
    assert z[1] == pytest.approx(-0.9)
    # slice 2: no real null
    assert np.isnan(r[2]) and np.isnan(z[2])


# ---------------------------------------------------------------------------
# Full label build: interpolation onto frame times + masking + units
# ---------------------------------------------------------------------------


def _synthetic_equilibrium(nt=10, n_bdy=40):
    """A synthetic shot: plasma defined on slices [3, 7), NaN elsewhere.

    Axis at (0.9, 0.05); circular LCFS radius 0.4; lower X-point at
    (0.55, -1.0).  Time base 200 Hz starting at 0.0 s.
    """
    t_eq = 0.005 * np.arange(nt)  # 200 Hz
    axis_r = np.full(nt, np.nan)
    axis_z = np.full(nt, np.nan)
    lcfs_r = np.full((n_bdy, nt), np.nan)
    lcfs_z = np.full((n_bdy, nt), np.nan)
    xr = np.full((2, nt), XPOINT_SENTINEL)
    xz = np.full((2, nt), XPOINT_SENTINEL)

    defined = range(3, 7)
    phi = np.linspace(0, 2 * np.pi, n_bdy, endpoint=False)
    for i in defined:
        axis_r[i] = 0.9
        axis_z[i] = 0.05
        lcfs_r[:, i] = 0.9 + 0.4 * np.cos(phi)
        lcfs_z[:, i] = 0.05 + 0.4 * np.sin(phi)
        xr[0, i], xz[0, i] = 0.50, 1.0  # upper null
        xr[1, i], xz[1, i] = 0.55, -1.0  # lower null (primary)
    return t_eq, axis_r, axis_z, xr, xz, lcfs_r, lcfs_z


def test_build_geometry_shapes_units_and_names():
    t_eq, axis_r, axis_z, xr, xz, lcfs_r, lcfs_z = _synthetic_equilibrium()
    frame_times = np.array([0.02, 0.025, 0.03])  # inside the defined window
    geo = build_geometry_from_arrays(
        shot_id=12345,
        frame_times=frame_times,
        t_eq=t_eq,
        axis_r=axis_r,
        axis_z=axis_z,
        x_point_r=xr,
        x_point_z=xz,
        lcfs_r=lcfs_r,
        lcfs_z=lcfs_z,
    )
    assert geo.target.shape == (3, TARGET_DIM)
    assert geo.finite_mask.shape == (3, TARGET_DIM)
    assert geo.names == TARGET_NAMES
    assert geo.units == "m"
    assert geo.target.dtype == np.float32
    assert geo.finite_mask.dtype == np.bool_


def test_build_geometry_values_in_defined_window():
    t_eq, axis_r, axis_z, xr, xz, lcfs_r, lcfs_z = _synthetic_equilibrium()
    # frame at t=0.025 s is exactly equilibrium slice 5 (defined).
    geo = build_geometry_from_arrays(
        shot_id=1,
        frame_times=np.array([0.025]),
        t_eq=t_eq,
        axis_r=axis_r,
        axis_z=axis_z,
        x_point_r=xr,
        x_point_z=xz,
        lcfs_r=lcfs_r,
        lcfs_z=lcfs_z,
    )
    assert np.all(geo.finite_mask[0])  # everything defined here
    tgt = geo.target[0]
    assert tgt[0] == pytest.approx(0.9, abs=1e-3)  # axis_R
    assert tgt[1] == pytest.approx(0.05, abs=1e-3)  # axis_Z
    assert tgt[2] == pytest.approx(0.55, abs=1e-3)  # xpt_R (lower null)
    assert tgt[3] == pytest.approx(-1.0, abs=1e-3)  # xpt_Z
    # 8 LCFS radii all ~0.4 (circle)
    np.testing.assert_allclose(tgt[4:], 0.4, atol=1e-2)


def test_build_geometry_masks_plasma_off_frames():
    """A frame BEFORE the defined window has NO finite labels (masked)."""
    t_eq, axis_r, axis_z, xr, xz, lcfs_r, lcfs_z = _synthetic_equilibrium()
    # t=0.0 is slice 0 (undefined) — outside the [3,7) defined window.
    geo = build_geometry_from_arrays(
        shot_id=1,
        frame_times=np.array([0.0]),
        t_eq=t_eq,
        axis_r=axis_r,
        axis_z=axis_z,
        x_point_r=xr,
        x_point_z=xz,
        lcfs_r=lcfs_r,
        lcfs_z=lcfs_z,
    )
    assert not geo.finite_mask[0].any()
    assert np.all(np.isnan(geo.target[0]))


def test_build_geometry_out_of_range_frame_is_masked():
    """A frame time beyond the equilibrium time base is fully masked."""
    t_eq, axis_r, axis_z, xr, xz, lcfs_r, lcfs_z = _synthetic_equilibrium()
    geo = build_geometry_from_arrays(
        shot_id=1,
        frame_times=np.array([10.0]),  # far past t_eq[-1]
        t_eq=t_eq,
        axis_r=axis_r,
        axis_z=axis_z,
        x_point_r=xr,
        x_point_z=xz,
        lcfs_r=lcfs_r,
        lcfs_z=lcfs_z,
    )
    assert not geo.finite_mask[0].any()


def test_build_geometry_interpolates_axis_between_slices():
    """Axis_R interpolates linearly between two defined slices."""
    t_eq = 0.005 * np.arange(10)
    nt, n_bdy = 10, 40
    axis_r = np.full(nt, np.nan)
    axis_z = np.full(nt, np.nan)
    lcfs_r = np.full((n_bdy, nt), np.nan)
    lcfs_z = np.full((n_bdy, nt), np.nan)
    xr = np.full((2, nt), XPOINT_SENTINEL)
    xz = np.full((2, nt), XPOINT_SENTINEL)
    phi = np.linspace(0, 2 * np.pi, n_bdy, endpoint=False)
    # axis_R ramps 0.8 -> 1.0 over slices 4 and 6 (t=0.020, 0.030).
    axis_r[4], axis_r[6] = 0.8, 1.0
    axis_r[5] = 0.9  # make slice 5 defined too so interp is well-posed
    for i in (4, 5, 6):
        axis_z[i] = 0.0
        lcfs_r[:, i] = axis_r[i] + 0.3 * np.cos(phi)
        lcfs_z[:, i] = 0.3 * np.sin(phi)
    geo = build_geometry_from_arrays(
        shot_id=1,
        frame_times=np.array([0.025]),  # midway between slice 4 and 6
        t_eq=t_eq,
        axis_r=axis_r,
        axis_z=axis_z,
        x_point_r=xr,
        x_point_z=xz,
        lcfs_r=lcfs_r,
        lcfs_z=lcfs_z,
    )
    assert geo.finite_mask[0, 0]
    assert geo.target[0, 0] == pytest.approx(0.9, abs=1e-2)


# ---------------------------------------------------------------------------
# Probe forward + checkpoint IO
# ---------------------------------------------------------------------------


def test_probe_forward_shapes_and_param_band():
    import torch

    from imas_ambix.worldmodel.equilibrium_probe import EquilibriumProbe, ProbeConfig

    cfg = ProbeConfig(in_frames=4, image_size=256, target_dim=TARGET_DIM)
    model = EquilibriumProbe(cfg)
    n = model.n_parameters()
    # ~2-5M parameter band (small CNN).
    assert 1_000_000 < n < 8_000_000, f"param count {n} out of band"

    x = torch.randn(2, 4, 256, 256)
    mean, log_sigma = model(x)
    assert mean.shape == (2, TARGET_DIM)
    assert log_sigma.shape == (2, TARGET_DIM)
    assert torch.isfinite(mean).all() and torch.isfinite(log_sigma).all()


def test_probe_nll_masking_and_finite():
    import torch

    from imas_ambix.worldmodel.equilibrium_probe import (
        EquilibriumProbe,
        ProbeConfig,
        gaussian_nll,
    )

    model = EquilibriumProbe(ProbeConfig(in_frames=3, target_dim=TARGET_DIM))
    x = torch.randn(2, 3, 256, 256)
    mean, log_sigma = model(x)
    target = torch.randn(2, TARGET_DIM)
    mask = torch.ones(2, TARGET_DIM)
    loss = gaussian_nll(mean, log_sigma, target, mask)
    assert torch.isfinite(loss)

    # fully-masked batch -> zero loss (no gradient contribution).
    zmask = torch.zeros(2, TARGET_DIM)
    zloss = gaussian_nll(mean, log_sigma, target, zmask)
    assert float(zloss.detach()) == pytest.approx(0.0)


def test_probe_checkpoint_roundtrip(tmp_path):
    import torch

    from imas_ambix.worldmodel.equilibrium_probe import (
        EquilibriumProbe,
        ProbeConfig,
        load_probe,
        save_probe,
    )

    model = EquilibriumProbe(ProbeConfig(in_frames=4, target_dim=TARGET_DIM))
    tmean = np.linspace(0.0, 1.0, TARGET_DIM)
    tstd = np.linspace(0.1, 0.5, TARGET_DIM)
    path = tmp_path / "probe.pt"
    save_probe(path, model, target_mean=tmean, target_std=tstd, extra={"epochs": 3})

    model2, m2, s2, extra = load_probe(path)
    np.testing.assert_allclose(m2, tmean)
    np.testing.assert_allclose(s2, tstd)
    assert extra["epochs"] == 3

    x = torch.randn(1, 4, 256, 256)
    model.eval()
    model2.eval()
    with torch.no_grad():
        a, _ = model(x)
        b, _ = model2(x)
    torch.testing.assert_close(a, b)


def test_predict_metres_destandardises():
    import torch

    from imas_ambix.worldmodel.equilibrium_probe import EquilibriumProbe, ProbeConfig

    model = EquilibriumProbe(ProbeConfig(in_frames=4, target_dim=TARGET_DIM))
    tmean = np.full(TARGET_DIM, 0.5)
    tstd = np.full(TARGET_DIM, 2.0)
    x = torch.randn(3, 4, 256, 256)
    mean_m, sigma_m = model.predict_metres(x, tmean, tstd)
    assert mean_m.shape == (3, TARGET_DIM)
    assert sigma_m.shape == (3, TARGET_DIM)
    assert np.all(sigma_m > 0)
