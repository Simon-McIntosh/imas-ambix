"""Tests for the equilibrium-label extractor + the geometry probe.

The label extractor is exercised on SYNTHETIC equilibrium arrays (no Zarr
store) so the angle-resampling, time-interpolation, NaN-masking, X-point
sentinel handling, units and shapes are pinned exactly.  The probe is exercised
for forward + checkpoint-IO shape on a CPU dummy.
"""

from __future__ import annotations

import numpy as np
import pytest

from imas_ambix.worldmodel import equilibrium_labels as equilibrium_labels_module
from imas_ambix.worldmodel.equilibrium_labels import (
    LCFS_ANGLES,
    N_LCFS_ANGLES,
    TARGET_DIM,
    TARGET_NAMES,
    XPOINT_SENTINEL,
    build_geometry_from_arrays,
    load_equilibrium_geometry,
    resample_lcfs_radii,
    xpoint_null_set,
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
# X-point null-set extraction
# ---------------------------------------------------------------------------


def test_xpoint_null_set_keeps_up_to_two_real_nulls():
    """Both real nulls survive while sentinels and invalid coordinates are masked."""
    xr = np.array(
        [
            [0.50, XPOINT_SENTINEL, np.nan, 2.50],
            [0.52, 0.55, np.nan, 0.60],
        ]
    )
    xz = np.array(
        [
            [1.0, XPOINT_SENTINEL, np.nan, 0.20],
            [-1.0, -0.9, np.nan, -0.80],
        ]
    )
    set_r, set_z = xpoint_null_set(xr, xz)
    assert set_r.shape == (2, 4)
    assert set_z.shape == (2, 4)
    np.testing.assert_allclose(set_r[:, 0], [0.50, 0.52])
    np.testing.assert_allclose(set_z[:, 0], [1.0, -1.0])
    np.testing.assert_allclose(set_r[:, 1], [0.55, np.nan], equal_nan=True)
    np.testing.assert_allclose(set_z[:, 1], [-0.9, np.nan], equal_nan=True)
    assert np.isnan(set_r[:, 2]).all()
    assert np.isnan(set_z[:, 2]).all()
    # The first candidate is outside the coarse vessel bounds, so the second
    # candidate is packed into the first available unordered slot.
    np.testing.assert_allclose(set_r[:, 3], [0.60, np.nan], equal_nan=True)
    np.testing.assert_allclose(set_z[:, 3], [-0.80, np.nan], equal_nan=True)


def test_xpoint_null_set_is_invariant_as_an_unordered_set():
    xr = np.array([[0.50], [0.52]])
    xz = np.array([[1.0], [-1.0]])
    set_r, set_z = xpoint_null_set(xr, xz)
    swapped_r, swapped_z = xpoint_null_set(xr[::-1], xz[::-1])

    original = {
        (float(r), float(z)) for r, z in zip(set_r[:, 0], set_z[:, 0], strict=True)
    }
    swapped = {
        (float(r), float(z))
        for r, z in zip(swapped_r[:, 0], swapped_z[:, 0], strict=True)
    }
    assert original == swapped


# ---------------------------------------------------------------------------
# Full label build: interpolation onto frame times + masking + units
# ---------------------------------------------------------------------------


def _synthetic_equilibrium(nt=10, n_bdy=40):
    """A synthetic shot: plasma defined on slices [3, 7), NaN elsewhere.

    Axis at (0.9, 0.05); circular LCFS radius 0.4; two nulls at
    (0.50, 1.0) and (0.55, -1.0). Time base 200 Hz starting at 0.0 s.
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
        xr[0, i], xz[0, i] = 0.50, 1.0
        xr[1, i], xz[1, i] = 0.55, -1.0
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
    np.testing.assert_allclose(
        tgt[2:6].reshape(2, 2),
        [[0.50, 1.0], [0.55, -1.0]],
        atol=1e-3,
    )
    # 8 LCFS radii all ~0.4 (circle)
    np.testing.assert_allclose(tgt[6:14], 0.4, atol=1e-2)


def test_build_geometry_null_slots_are_swap_invariant_as_a_set():
    t_eq, axis_r, axis_z, xr, xz, lcfs_r, lcfs_z = _synthetic_equilibrium()
    kwargs = {
        "shot_id": 1,
        "frame_times": np.array([0.025]),
        "t_eq": t_eq,
        "axis_r": axis_r,
        "axis_z": axis_z,
        "lcfs_r": lcfs_r,
        "lcfs_z": lcfs_z,
    }
    geo = build_geometry_from_arrays(x_point_r=xr, x_point_z=xz, **kwargs)
    swapped = build_geometry_from_arrays(
        x_point_r=xr[::-1], x_point_z=xz[::-1], **kwargs
    )

    np.testing.assert_allclose(geo.target[:, :2], swapped.target[:, :2])
    np.testing.assert_allclose(geo.target[:, 6:14], swapped.target[:, 6:14])
    original_set = {tuple(point) for point in geo.target[0, 2:6].reshape(2, 2)}
    swapped_set = {tuple(point) for point in swapped.target[0, 2:6].reshape(2, 2)}
    assert original_set == swapped_set


def test_build_geometry_absent_null_slot_is_masked():
    t_eq, axis_r, axis_z, xr, xz, lcfs_r, lcfs_z = _synthetic_equilibrium()
    xr[0, 5] = XPOINT_SENTINEL
    xz[0, 5] = XPOINT_SENTINEL
    geo = build_geometry_from_arrays(
        shot_id=1,
        frame_times=np.array([t_eq[5]]),
        t_eq=t_eq,
        axis_r=axis_r,
        axis_z=axis_z,
        x_point_r=xr,
        x_point_z=xz,
        lcfs_r=lcfs_r,
        lcfs_z=lcfs_z,
    )

    np.testing.assert_allclose(geo.target[0, 2:4], [0.55, -1.0])
    assert geo.finite_mask[0, 2:4].all()
    assert np.isnan(geo.target[0, 4:6]).all()
    assert not geo.finite_mask[0, 4:6].any()
    assert geo.finite_mask[0, :2].all()
    assert geo.finite_mask[0, 6:14].all()


def test_load_geometry_masks_omitted_null_arrays(monkeypatch):
    t_eq, axis_r, axis_z, _, _, lcfs_r, lcfs_z = _synthetic_equilibrium()
    equilibrium_group = {
        "time": t_eq,
        "magnetic_axis_r": axis_r,
        "magnetic_axis_z": axis_z,
        "lcfs_r": lcfs_r,
        "lcfs_z": lcfs_z,
    }
    monkeypatch.setattr(
        equilibrium_labels_module,
        "_read_equilibrium_group",
        lambda _shot_id, _level2_root: equilibrium_group,
    )

    geometry = load_equilibrium_geometry(1, np.array([t_eq[5]]))

    assert geometry.finite_mask[0, :2].all()
    assert np.isfinite(geometry.target[0, :2]).all()
    assert not geometry.finite_mask[0, 2:6].any()
    assert np.isnan(geometry.target[0, 2:6]).all()
    assert geometry.finite_mask[0, 6:14].all()
    assert np.isfinite(geometry.target[0, 6:14]).all()


def test_build_geometry_uses_nearest_native_null_set_without_interpolation():
    t_eq, axis_r, axis_z, xr, xz, lcfs_r, lcfs_z = _synthetic_equilibrium()
    xr[:, 4], xz[:, 4] = [0.40, 0.45], [0.80, -0.80]
    xr[:, 5], xz[:, 5] = [0.60, 0.65], [1.20, -1.20]
    geo = build_geometry_from_arrays(
        shot_id=1,
        frame_times=np.array([0.023]),
        t_eq=t_eq,
        axis_r=axis_r,
        axis_z=axis_z,
        x_point_r=xr,
        x_point_z=xz,
        lcfs_r=lcfs_r,
        lcfs_z=lcfs_z,
    )

    # 0.023 s is nearer the 0.025 s native slice than the 0.020 s slice.
    np.testing.assert_allclose(
        geo.target[0, 2:6].reshape(2, 2),
        [[0.60, 1.20], [0.65, -1.20]],
        atol=1e-3,
    )


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
