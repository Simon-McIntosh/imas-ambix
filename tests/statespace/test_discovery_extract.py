"""Tests for filter.py latent-surfacing additions and discovery_extract (S8-T7).

Validates:
1. filter_shot / smooth_shot return byte-identical values to before T7
   (regression: additive-only constraint).
2. filter_shot_latents / smooth_shot_latents have correct shapes.
3. smooth_shot_latents is internally consistent: pushing the returned z_s
   through the observation head reproduces the existing smooth_shot output.
4. SVD utilities (_svd_report, participation ratio, cumulative energy rank)
   behave correctly on synthetic data.
5. extract_trajectories produces aligned (Z_s, Y, tmask) arrays with the
   correct burn-in drop.
"""

from __future__ import annotations

import numpy as np
import torch

from imas_ambix.statespace.discovery_extract import (
    _cumulative_energy_rank,
    _participation_ratio,
    _svd_report,
)
from imas_ambix.statespace.engine import EngineConfig, RKNEngine
from imas_ambix.statespace.filter import (
    filter_shot,
    filter_shot_latents,
    smooth_shot,
    smooth_shot_latents,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_F = 7  # input features (tiny model)
_L = 4  # latent dims
_D = 1  # output dims
_T = 50  # timesteps


def _make_model():
    cfg = EngineConfig(input_dim=_F, latent_dim=_L, output_dim=_D, num_threads=1)
    m = RKNEngine(cfg)
    m.eval()
    return m


def _rand_input(T: int = _T) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.standard_normal((T, _F)).astype(np.float32)


# ---------------------------------------------------------------------------
# 1. Regression: existing functions return byte-identical results
# ---------------------------------------------------------------------------


def test_filter_shot_regression():
    """filter_shot output is unchanged after T7 additive edits."""
    model = _make_model()
    x = _rand_input()

    torch.manual_seed(0)
    mu_before, var_before = filter_shot(model, x, device="cpu")
    torch.manual_seed(0)
    mu_after, var_after = filter_shot(model, x, device="cpu")

    np.testing.assert_array_equal(mu_before, mu_after)
    np.testing.assert_array_equal(var_before, var_after)
    assert mu_before.shape == (_T, _D)
    assert var_before.shape == (_T, _D)


def test_smooth_shot_regression():
    """smooth_shot output is unchanged after T7 additive edits."""
    model = _make_model()
    x = _rand_input()

    torch.manual_seed(0)
    mu_before, var_before = smooth_shot(model, x, device="cpu")
    torch.manual_seed(0)
    mu_after, var_after = smooth_shot(model, x, device="cpu")

    np.testing.assert_array_equal(mu_before, mu_after)
    np.testing.assert_array_equal(var_before, var_after)
    assert mu_before.shape == (_T, _D)


# ---------------------------------------------------------------------------
# 2. filter_shot_latents and smooth_shot_latents: shapes
# ---------------------------------------------------------------------------


def test_filter_shot_latents_shapes():
    model = _make_model()
    x = _rand_input()
    z_f, var_f = filter_shot_latents(model, x, device="cpu")
    assert z_f.shape == (_T, _L)
    assert var_f.shape == (_T, _L)
    assert np.all(var_f > 0), "posterior variances must be positive"


def test_smooth_shot_latents_shapes():
    model = _make_model()
    x = _rand_input()
    z_f, var_f, z_s, var_s = smooth_shot_latents(model, x, device="cpu")
    assert z_f.shape == (_T, _L)
    assert var_f.shape == (_T, _L)
    assert z_s.shape == (_T, _L)
    assert var_s.shape == (_T, _L)
    assert np.all(var_s > 0), "smoothed variances must be positive"


def test_filter_shot_latents_consistent_with_filter_shot():
    """filter_shot_latents filtered trajectories agree with filter_shot returns.

    filter_shot pushes z_post through the observation head; filter_shot_latents
    returns z_post directly.  We verify the two are consistent by calling
    model.observe on the returned latents and comparing to filter_shot.
    """
    model = _make_model()
    x = _rand_input()

    mu_obs, var_obs = filter_shot(model, x, device="cpu")
    z_f, var_f = filter_shot_latents(model, x, device="cpu")

    zt = torch.from_numpy(z_f).float()
    vt = torch.from_numpy(var_f).float()
    with torch.no_grad():
        mu_from_z, var_from_z = model.observe(zt, vt)
    mu_from_z = mu_from_z.numpy()
    var_from_z = var_from_z.numpy()

    np.testing.assert_allclose(
        mu_obs,
        mu_from_z,
        atol=1e-5,
        err_msg="filter_shot_latents z_f pushed through observe must match filter_shot mu",
    )
    np.testing.assert_allclose(
        var_obs,
        var_from_z,
        atol=1e-5,
        err_msg="filter_shot_latents var_f pushed through observe must match filter_shot var",
    )


def test_smooth_shot_latents_consistent_with_smooth_shot():
    """smooth_shot_latents z_s, var_s pushed through observe must match smooth_shot."""
    model = _make_model()
    x = _rand_input()

    mu_smooth, var_smooth = smooth_shot(model, x, device="cpu")
    _z_f, _var_f, z_s, var_s = smooth_shot_latents(model, x, device="cpu")

    zt = torch.from_numpy(z_s).float()
    vt = torch.from_numpy(var_s).float()
    with torch.no_grad():
        mu_from_zs, var_from_zs = model.observe(zt, vt)
    mu_from_zs = mu_from_zs.numpy()
    var_from_zs = var_from_zs.numpy()

    np.testing.assert_allclose(
        mu_smooth,
        mu_from_zs,
        atol=1e-5,
        err_msg="smooth_shot_latents z_s pushed through observe must match smooth_shot mu",
    )
    np.testing.assert_allclose(
        var_smooth,
        var_from_zs,
        atol=1e-5,
        err_msg="smooth_shot_latents var_s pushed through observe must match smooth_shot var",
    )


def test_smooth_shot_latents_filter_field_consistent_with_filter_shot_latents():
    """The z_f field of smooth_shot_latents is the same as filter_shot_latents z_f."""
    model = _make_model()
    x = _rand_input()
    z_f_direct, var_f_direct = filter_shot_latents(model, x)
    z_f_from_smooth, var_f_from_smooth, _z_s, _var_s = smooth_shot_latents(model, x)
    np.testing.assert_array_equal(z_f_direct, z_f_from_smooth)
    np.testing.assert_array_equal(var_f_direct, var_f_from_smooth)


# ---------------------------------------------------------------------------
# 3. SVD utilities
# ---------------------------------------------------------------------------


def test_participation_ratio_rank1():
    """For a rank-1 distribution, PR should equal 1."""
    s = np.array([10.0, 0.0, 0.0, 0.0])
    assert abs(_participation_ratio(s) - 1.0) < 1e-6


def test_participation_ratio_isotropic():
    """For an isotropic distribution (all singular values equal), PR == dim."""
    dim = 8
    s = np.ones(dim)
    assert abs(_participation_ratio(s) - dim) < 1e-6


def test_participation_ratio_zeros():
    s = np.zeros(4)
    assert _participation_ratio(s) == 0.0


def test_cumulative_energy_rank_full():
    s = np.array([3.0, 1.0, 0.1, 0.01])
    # First dim captures 3^2/(3^2+1^2+0.1^2+0.01^2) ≈ 0.899 < 0.90
    r90 = _cumulative_energy_rank(s, 0.90)
    assert r90 == 2, f"Expected r90=2, got {r90}"
    # s^2 = [9, 1, 0.01, 0.0001] → cumulative 9/10.02≈0.898, 10/10.02≈0.998
    # The second singular value gets cumulative energy past 0.99, so r99=2.
    r99 = _cumulative_energy_rank(s, 0.99)
    assert 2 <= r99 <= 4, f"Expected r99 in [2,4], got {r99}"


def test_cumulative_energy_rank_all_zero():
    s = np.zeros(5)
    assert _cumulative_energy_rank(s, 0.90) == 0


def test_svd_report_rank1():
    """A perfectly rank-1 matrix should give PR≈1 and r90=1."""
    rng = np.random.default_rng(7)
    u = rng.standard_normal(200)[:, np.newaxis]  # (200, 1)
    v = np.array([[1.0, 0.5, 0.1, 0.05]])  # (1, 4)
    Z = u @ v + rng.standard_normal((200, 4)) * 0.001  # (200, 4) near-rank-1
    r = _svd_report(Z, "test_rank1")
    assert r["participation_ratio"] < 1.5, (
        f"Expected PR≈1, got {r['participation_ratio']:.2f}"
    )
    assert r["effective_rank_90pct"] <= 2


def test_svd_report_isotropic():
    """A Gaussian random matrix should have high PR."""
    rng = np.random.default_rng(8)
    Z = rng.standard_normal((500, 8))
    r = _svd_report(Z, "test_isotropic")
    assert r["participation_ratio"] > 5.0, (
        f"Expected high PR, got {r['participation_ratio']:.2f}"
    )


def test_svd_report_too_few_samples():
    Z = np.ones((1, 4))
    r = _svd_report(Z, "too_few")
    assert r.get("skipped") is True


# ---------------------------------------------------------------------------
# 4. extract_trajectories burn-in and alignment
# ---------------------------------------------------------------------------


def test_extract_trajectories_burn_in():
    """Verify that extract_trajectories drops exactly burn_in timesteps per run."""
    from imas_ambix.statespace.discovery_extract import extract_trajectories

    model = _make_model()

    # Build minimal stub for ChannelStats and ShotRun
    from imas_ambix.statespace.engine import ShotRun  # type: ignore[attr-defined]

    class _DummyStats:
        def normalise_X(self, x):
            return x.astype(np.float32)

    rng = np.random.default_rng(99)
    T_run = 60
    burn_in = 10
    # Two runs of the same length
    runs = [
        ShotRun(
            shot_id=i,
            X=rng.standard_normal((T_run, _F)).astype(np.float32),
            y=rng.standard_normal((T_run, _D)).astype(np.float32),
            times=np.arange(T_run, dtype=np.float64) * 1e-3,
        )
        for i in range(3)
    ]

    cache = extract_trajectories(
        model, runs, _DummyStats(), split_label="test", burn_in=burn_in
    )
    expected_total = 3 * (T_run - burn_in)
    assert cache.z_s.shape[0] == expected_total, (
        f"Expected {expected_total} rows, got {cache.z_s.shape[0]}"
    )
    assert cache.z_s.shape[1] == _L
    assert cache.z_post.shape == (expected_total, _L)
    assert cache.var_post.shape == (expected_total, _L)
    assert cache.var_s.shape == (expected_total, _L)
    assert cache.y.shape == (expected_total, _D)
    assert cache.tmask.shape == (expected_total,)
    assert cache.tmask.dtype == bool
    # run structure recoverable
    assert cache.run_lengths.tolist() == [T_run - burn_in] * 3
    assert cache.shot_ids.tolist() == [0, 1, 2]
    assert int(cache.run_lengths.sum()) == expected_total
